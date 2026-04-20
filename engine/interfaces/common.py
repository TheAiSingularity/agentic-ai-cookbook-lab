"""engine.interfaces.common — shared rendering + run helpers.

All three interfaces (CLI / TUI / Web) surface the same data: the answer,
cited sources, verified vs unverified claims, per-node trace, and memory
hits. This module owns the rendering + orchestration so the three
interface front-ends stay thin.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from engine.core import build_graph
from engine.core.memory import MemoryStore, Trajectory, summarize_hits


@dataclass
class RunResult:
    """Everything an interface needs to render a completed query."""

    question: str
    domain: str
    answer: str
    verified_claims: list[dict] = field(default_factory=list)
    unverified_claims: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)    # [{url, title, text, fetched}]
    trace: list[dict] = field(default_factory=list)
    memory_hits: list[dict] = field(default_factory=list)  # [{question, answer, score, domain, timestamp}]
    question_class: str = ""
    iterations: int = 0
    total_latency_s: float = 0.0
    total_tokens_est: int = 0


def _trace_totals(trace: list[dict]) -> tuple[float, int]:
    lat = sum(float(e.get("latency_s", 0) or 0) for e in trace)
    tok = sum(int(e.get("tokens_est", 0) or 0) for e in trace)
    return round(lat, 3), tok


def run_query(
    question: str,
    *,
    domain: str = "general",
    memory: MemoryStore | None = None,
    extra_context: str = "",
) -> RunResult:
    """Execute the engine pipeline end-to-end and package the result.

    If `memory` is provided, prior-trajectory hits are pulled first and
    their summaries optionally stitched into the question as the pipeline
    enters `classify`. The trajectory itself is recorded on completion.
    """
    memory_hits_payload: list[dict] = []
    injected_question = question

    if memory is not None:
        hits = memory.retrieve(question)
        if hits:
            memory_hits_payload = [
                {
                    "question": t.question,
                    "answer": t.final_answer,
                    "score": round(float(score), 4),
                    "domain": t.domain,
                    "timestamp": t.timestamp,
                    "query_id": t.query_id,
                }
                for t, score in hits
            ]
            injected_question = f"{question}\n\n(Context from prior related research:\n{summarize_hits(hits)}\n)"

    t0 = time.monotonic()
    graph = build_graph()
    state_in: dict[str, Any] = {
        "question": injected_question,
        "iterations": 0,
        "plan_rejects": 0,
        "trace": [],
    }
    if extra_context:
        state_in["question"] = f"{state_in['question']}\n\n{extra_context}"
    state = graph.invoke(state_in)
    total_wall = round(time.monotonic() - t0, 3)

    trace = state.get("trace", []) or []
    total_lat, total_tok = _trace_totals(trace)

    result = RunResult(
        question=question,
        domain=domain,
        answer=state.get("answer", "") or "",
        verified_claims=[c for c in (state.get("claims") or []) if c.get("verified")],
        unverified_claims=list(state.get("unverified") or []),
        sources=list(state.get("evidence_compressed") or state.get("evidence") or []),
        trace=trace,
        memory_hits=memory_hits_payload,
        question_class=state.get("question_class", "") or "",
        iterations=int(state.get("iterations", 0) or 0),
        total_latency_s=max(total_lat, total_wall),
        total_tokens_est=total_tok,
    )

    if memory is not None:
        traj = Trajectory.from_state({**state, "question": question}, domain=domain)
        memory.record(traj)

    return result


# ── Reusable formatting helpers (used by CLI + TUI + Web) ────────────

def format_verified_summary(result: RunResult) -> str:
    """One-line verified/unverified summary."""
    v = len(result.verified_claims)
    u = len(result.unverified_claims)
    total = v + u
    if total == 0:
        return "(no CoVe claims emitted)"
    return f"{v}/{total} claims verified" + (f" · {u} unverified" if u else "")


def format_sources(result: RunResult, max_chars: int = 140) -> list[dict]:
    """Return [{idx, url, title, preview, fetched}] for display."""
    rows = []
    for i, s in enumerate(result.sources, 1):
        text = s.get("text", "") or ""
        preview = text[:max_chars].replace("\n", " ")
        if len(text) > max_chars:
            preview += "…"
        rows.append({
            "idx": i,
            "url": s.get("url", ""),
            "title": s.get("title", s.get("url", "")),
            "preview": preview,
            "fetched": bool(s.get("fetched", False)),
        })
    return rows


def format_trace_per_node(result: RunResult) -> list[dict]:
    """Per-node aggregate for display."""
    by_node: dict[str, dict] = {}
    for e in result.trace:
        node = e.get("node", "?")
        b = by_node.setdefault(node, {"calls": 0, "latency_s": 0.0, "tokens_est": 0})
        b["calls"] += 1
        b["latency_s"] += float(e.get("latency_s", 0) or 0)
        b["tokens_est"] += int(e.get("tokens_est", 0) or 0)
    return [{"node": n, **b} for n, b in sorted(by_node.items(), key=lambda kv: -kv[1]["latency_s"])]


__all__ = [
    "RunResult",
    "run_query",
    "format_verified_summary",
    "format_sources",
    "format_trace_per_node",
]
