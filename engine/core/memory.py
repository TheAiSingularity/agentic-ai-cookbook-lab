"""engine.core.memory — trajectory logging + relevance retrieval.

Phase 2 of the research-engine master plan. Every query produces a trajectory
record (question, domain, evidence, intermediate steps, final answer, CoVe
verdicts, token + latency totals). Subsequent queries retrieve semantically
similar prior trajectories and inject 1-line summaries into the pipeline as
additional context.

Storage: SQLite at `~/.agentic-research/memory.db` by default (configurable
via `MEMORY_DB_PATH`). Three tables:

    trajectories — one row per query (full JSON blob)
    claims       — flattened verified claims for quick grounding lookups
    embeddings   — 1 row per trajectory with a float-array blob; indexed only
                   linearly since the stores we'd care about are ≤ 10 k rows

Privacy: everything is local to the machine. No network egress. Reset with
`engine.core.memory.reset(db_path=...)` or the `engine reset-memory` CLI.

Retrieval uses the same embedder as `core.rag` (defaults to
`nomic-embed-text` on Ollama; honors OPENAI_BASE_URL + EMBED_MODEL). Semantic
similarity = cosine.
"""

from __future__ import annotations

import json
import os
import sqlite3
import struct
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from core.rag.python.rag import _openai_embedder, _cosine

ENV = os.environ.get

DEFAULT_DB_PATH = Path(ENV("MEMORY_DB_PATH", str(Path.home() / ".agentic-research" / "memory.db")))
DEFAULT_TOP_K = int(ENV("MEMORY_TOP_K", "3"))
DEFAULT_MIN_SCORE = float(ENV("MEMORY_MIN_SCORE", "0.55"))

# `memory_mode` options — one of:
#   "off"         — never record, never retrieve
#   "session"     — record to an in-process list only (cleared on exit)
#   "persistent"  — SQLite at DEFAULT_DB_PATH (the default when flag set)
VALID_MODES = {"off", "session", "persistent"}


# ── Trajectory record shape ──────────────────────────────────────────

@dataclass
class Trajectory:
    """One full research query's audit trail."""

    query_id: str                      # timestamp-ish unique id
    timestamp: float
    question: str
    domain: str                        # free-text; "general" if unset
    final_answer: str
    verified_claims: list[dict] = field(default_factory=list)
    unverified_claims: list[str] = field(default_factory=list)
    evidence_urls: list[str] = field(default_factory=list)
    iterations: int = 0
    question_class: str = ""
    tokens_est: int = 0
    latency_s: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_state(cls, state: dict, *, domain: str = "general") -> "Trajectory":
        """Build a Trajectory from a finalized pipeline State dict."""
        tokens = sum(int(e.get("tokens_est", 0) or 0) for e in state.get("trace", []))
        latency = sum(float(e.get("latency_s", 0) or 0) for e in state.get("trace", []))
        verified = [c for c in state.get("claims") or [] if c.get("verified")]
        unverified = state.get("unverified") or []
        return cls(
            query_id=f"{int(time.time() * 1000):d}",
            timestamp=time.time(),
            question=state.get("question", ""),
            domain=domain,
            final_answer=state.get("answer", ""),
            verified_claims=verified,
            unverified_claims=list(unverified),
            evidence_urls=[e.get("url", "") for e in state.get("evidence") or []],
            iterations=int(state.get("iterations", 0) or 0),
            question_class=str(state.get("question_class", "") or ""),
            tokens_est=tokens,
            latency_s=round(latency, 3),
        )


# ── Embedding serialization ──────────────────────────────────────────

def _pack_embedding(vec: Iterable[float]) -> bytes:
    vec = list(vec)
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack_embedding(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# ── MemoryStore ──────────────────────────────────────────────────────

class MemoryStore:
    """SQLite-backed trajectory store with semantic retrieval.

    Usage:
        store = MemoryStore.open("persistent")
        store.record(trajectory, question_embedding=None)  # embeds on demand
        hits = store.retrieve("similar question", k=3)
        # each hit = (Trajectory, cosine_score)
        store.reset()  # wipe all trajectories
    """

    def __init__(self, path: Path | None, embedder: Callable | None = None):
        self.path = path
        self.embedder = embedder or _openai_embedder
        self._session: list[tuple[Trajectory, list[float]]] = []
        self._conn: sqlite3.Connection | None = None
        if path is not None:
            self._ensure_db()

    # ── factory ────────────────────────────────────────────────

    @classmethod
    def open(cls, mode: str, *, path: Path | None = None, embedder=None) -> "MemoryStore":
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        if mode == "off":
            return _NullStore()
        if mode == "session":
            return cls(path=None, embedder=embedder)
        # persistent
        p = path or DEFAULT_DB_PATH
        p.parent.mkdir(parents=True, exist_ok=True)
        return cls(path=p, embedder=embedder)

    # ── schema ─────────────────────────────────────────────────

    def _ensure_db(self) -> None:
        assert self.path is not None
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS trajectories (
                query_id        TEXT PRIMARY KEY,
                timestamp       REAL NOT NULL,
                question        TEXT NOT NULL,
                domain          TEXT NOT NULL,
                payload         TEXT NOT NULL   -- JSON of the full Trajectory
            );
            CREATE INDEX IF NOT EXISTS traj_domain ON trajectories(domain);
            CREATE INDEX IF NOT EXISTS traj_ts ON trajectories(timestamp DESC);

            CREATE TABLE IF NOT EXISTS embeddings (
                query_id        TEXT PRIMARY KEY REFERENCES trajectories(query_id) ON DELETE CASCADE,
                vec             BLOB NOT NULL
            );
        """)
        self._conn.commit()

    # ── record ────────────────────────────────────────────────

    def record(self, traj: Trajectory, question_embedding: list[float] | None = None) -> None:
        """Persist a trajectory; embed the question on demand if not supplied."""
        vec = question_embedding or self._embed(traj.question)
        if self._conn is not None:
            self._conn.execute(
                "INSERT OR REPLACE INTO trajectories(query_id,timestamp,question,domain,payload) "
                "VALUES (?,?,?,?,?)",
                (traj.query_id, traj.timestamp, traj.question, traj.domain, traj.to_json()),
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings(query_id,vec) VALUES (?,?)",
                (traj.query_id, _pack_embedding(vec)),
            )
            self._conn.commit()
        else:
            self._session.append((traj, vec))

    # ── retrieve ──────────────────────────────────────────────

    def retrieve(
        self,
        question: str,
        k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
        domain: str | None = None,
    ) -> list[tuple[Trajectory, float]]:
        """Return up to k prior trajectories whose questions are semantically
        similar (cosine ≥ min_score) to `question`, sorted by score desc.

        When `domain` is given, restrict retrieval to that domain.
        """
        if k <= 0:
            return []
        q_vec = self._embed(question)
        candidates: list[tuple[Trajectory, list[float]]] = []
        if self._conn is not None:
            rows = self._conn.execute(
                """SELECT t.payload, e.vec
                   FROM trajectories t JOIN embeddings e USING(query_id)
                   WHERE (? IS NULL OR t.domain = ?)""",
                (domain, domain),
            ).fetchall()
            candidates = [(Trajectory(**json.loads(p)), _unpack_embedding(v)) for p, v in rows]
        else:
            candidates = [(t, v) for t, v in self._session if (domain is None or t.domain == domain)]

        scored = [(t, _cosine(q_vec, v)) for t, v in candidates]
        scored = [(t, s) for t, s in scored if s >= min_score]
        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:k]

    # ── admin ─────────────────────────────────────────────────

    def reset(self) -> None:
        """Wipe all trajectories. Respects session-vs-persistent mode."""
        if self._conn is not None:
            self._conn.executescript("DELETE FROM embeddings; DELETE FROM trajectories;")
            self._conn.commit()
        self._session.clear()

    def count(self) -> int:
        if self._conn is not None:
            return self._conn.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
        return len(self._session)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ── helpers ───────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        return self.embedder([text])[0]


class _NullStore(MemoryStore):
    """No-op store used when `memory_mode == 'off'`."""

    def __init__(self):
        self._session = []
        self._conn = None
        self.path = None
        self.embedder = None

    def record(self, *_args, **_kwargs) -> None:  # noqa: D401
        return None

    def retrieve(self, *_args, **_kwargs) -> list[tuple[Trajectory, float]]:  # noqa: D401
        return []

    def reset(self) -> None:
        return None

    def count(self) -> int:
        return 0

    def close(self) -> None:
        return None


# ── Pipeline integration helpers ─────────────────────────────────────

def summarize_hits(hits: list[tuple[Trajectory, float]], *, max_chars: int = 200) -> str:
    """Render memory hits as a short, citable context block for injection."""
    if not hits:
        return ""
    lines = []
    for t, score in hits:
        preview = (t.final_answer or "").replace("\n", " ")
        if len(preview) > max_chars:
            preview = preview[:max_chars] + "…"
        lines.append(f"- ({score:.2f}) {t.question}\n  → {preview}")
    return "Relevant prior research (for context — verify independently):\n" + "\n".join(lines)


__all__ = [
    "DEFAULT_DB_PATH",
    "DEFAULT_TOP_K",
    "DEFAULT_MIN_SCORE",
    "VALID_MODES",
    "Trajectory",
    "MemoryStore",
    "summarize_hits",
]
