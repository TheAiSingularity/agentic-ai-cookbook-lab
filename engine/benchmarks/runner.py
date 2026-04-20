"""engine/benchmarks/runner.py — reproducible mini-benchmark harness.

Reads a JSONL fixture (one question per line with `{id, domain, question,
gold}`), runs each through the pipeline, scores against the `gold`
object's `must_contain` / `must_not_contain` string lists, and writes
aggregate + per-question metrics to disk.

Usage:
    python engine/benchmarks/runner.py simpleqa_mini.jsonl
    python engine/benchmarks/runner.py browsecomp_mini.jsonl --model gemma3:4b
    python engine/benchmarks/runner.py simpleqa_mini.jsonl --ablate rerank

Environment:
    Honors the same env vars as engine.core.pipeline. Use OPENAI_BASE_URL
    to route to any backend; defaults assume Mac local (Ollama + SearXNG).

Output:
    engine/benchmarks/results/<fixture_stem>/<timestamp>_summary.json
    engine/benchmarks/results/<fixture_stem>/<timestamp>_detail.jsonl

Ablations:
    --ablate rerank     ENABLE_RERANK=1 (vs default 0)
    --ablate no-fetch   ENABLE_FETCH=0  (vs default 1)
    --ablate no-compress ENABLE_COMPRESS=0
    --ablate no-verify  ENABLE_VERIFY=0
    --ablate no-flare   ENABLE_ACTIVE_RETR=0
    --ablate no-router  ENABLE_ROUTER=0
    Combine with `--ablate rerank,no-fetch` to run multi-var.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engine.interfaces.common import run_query  # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent


# ── Data shapes ──────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    id: str
    domain: str
    question: str
    answer: str
    wall_s: float
    verified: int
    total_claims: int
    iterations: int
    tokens_est: int
    must_contain_missing: list[str]
    must_not_contain_hits: list[str]
    passed: bool


@dataclass
class BenchmarkSummary:
    fixture: str
    timestamp: str
    n_questions: int
    n_passed: int
    pass_rate: float
    mean_wall_s: float
    mean_tokens_est: float
    verified_claims_total: int
    total_claims: int
    verified_ratio: float
    ablations: dict[str, str] = field(default_factory=dict)
    per_question: list[dict] = field(default_factory=list)


# ── Scoring ──────────────────────────────────────────────────────────

def _score(answer: str, gold: dict) -> tuple[list[str], list[str], bool]:
    """Return (missing_must_contain, hit_must_not_contain, passed)."""
    answer_lower = (answer or "").lower()
    missing = [t for t in (gold.get("must_contain") or []) if t.lower() not in answer_lower]
    hits = [t for t in (gold.get("must_not_contain") or []) if t.lower() in answer_lower]
    passed = len(missing) == 0 and len(hits) == 0
    return missing, hits, passed


# ── Runner ───────────────────────────────────────────────────────────

def _apply_ablations(ablations: list[str]) -> dict[str, str]:
    """Translate CLI ablation flags into env-var overrides and return them
    (for recording in the summary)."""
    applied: dict[str, str] = {}
    for flag in ablations:
        flag = flag.strip().lower()
        if not flag:
            continue
        if flag == "rerank":
            os.environ["ENABLE_RERANK"] = "1"
            applied["ENABLE_RERANK"] = "1"
        elif flag == "no-fetch":
            os.environ["ENABLE_FETCH"] = "0"
            applied["ENABLE_FETCH"] = "0"
        elif flag == "no-compress":
            os.environ["ENABLE_COMPRESS"] = "0"
            applied["ENABLE_COMPRESS"] = "0"
        elif flag == "no-verify":
            os.environ["ENABLE_VERIFY"] = "0"
            applied["ENABLE_VERIFY"] = "0"
        elif flag == "no-flare":
            os.environ["ENABLE_ACTIVE_RETR"] = "0"
            applied["ENABLE_ACTIVE_RETR"] = "0"
        elif flag == "no-router":
            os.environ["ENABLE_ROUTER"] = "0"
            applied["ENABLE_ROUTER"] = "0"
        else:
            print(f"[bench] unknown ablation flag: {flag!r}", file=sys.stderr)
    return applied


def run_benchmark(fixture_path: Path, *, model: str | None = None,
                  ablations: list[str] | None = None,
                  out_dir: Path | None = None) -> BenchmarkSummary:
    """Run one fixture through the pipeline, score every row, return summary."""
    fixture_path = Path(fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"fixture not found: {fixture_path}")

    applied = _apply_ablations(ablations or [])

    if model:
        for k in ("MODEL_PLANNER", "MODEL_SEARCHER", "MODEL_SYNTHESIZER",
                  "MODEL_VERIFIER", "MODEL_CRITIC", "MODEL_ROUTER", "MODEL_COMPRESSOR"):
            os.environ[k] = model
        applied["MODEL"] = model

    rows: list[dict[str, Any]] = []
    with fixture_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    per_q: list[QuestionResult] = []

    print(f"[bench] {fixture_path.name} · {len(rows)} questions · model={model or '(default)'}")
    for row in rows:
        t0 = time.monotonic()
        try:
            result = run_query(row["question"], domain=row.get("domain", "general"))
            wall = round(time.monotonic() - t0, 2)
            missing, hits, passed = _score(result.answer, row.get("gold") or {})
            qr = QuestionResult(
                id=row["id"],
                domain=row.get("domain", "general"),
                question=row["question"],
                answer=result.answer,
                wall_s=wall,
                verified=len(result.verified_claims),
                total_claims=len(result.verified_claims) + len(result.unverified_claims),
                iterations=result.iterations,
                tokens_est=result.total_tokens_est,
                must_contain_missing=missing,
                must_not_contain_hits=hits,
                passed=passed,
            )
        except Exception as exc:  # noqa: BLE001
            qr = QuestionResult(
                id=row["id"],
                domain=row.get("domain", "general"),
                question=row["question"],
                answer=f"[bench error] {exc}",
                wall_s=round(time.monotonic() - t0, 2),
                verified=0,
                total_claims=0,
                iterations=0,
                tokens_est=0,
                must_contain_missing=list(row.get("gold", {}).get("must_contain") or []),
                must_not_contain_hits=[],
                passed=False,
            )
        per_q.append(qr)
        mark = "✓" if qr.passed else "✗"
        print(f"  {mark} {qr.id}  {qr.wall_s:5.1f}s  verified={qr.verified}/{qr.total_claims}"
              f"  missing={len(qr.must_contain_missing)}")

    n_passed = sum(1 for q in per_q if q.passed)
    total_claims = sum(q.total_claims for q in per_q)
    verified_total = sum(q.verified for q in per_q)

    summary = BenchmarkSummary(
        fixture=fixture_path.name,
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_questions=len(per_q),
        n_passed=n_passed,
        pass_rate=round(n_passed / max(1, len(per_q)), 4),
        mean_wall_s=round(sum(q.wall_s for q in per_q) / max(1, len(per_q)), 2),
        mean_tokens_est=round(sum(q.tokens_est for q in per_q) / max(1, len(per_q)), 1),
        verified_claims_total=verified_total,
        total_claims=total_claims,
        verified_ratio=round(verified_total / max(1, total_claims), 4),
        ablations=applied,
        per_question=[asdict(q) for q in per_q],
    )

    out_dir = out_dir or (BENCH_DIR / "results" / fixture_path.stem)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = summary.timestamp.replace(":", "").replace("+0000", "Z")
    (out_dir / f"{ts}_summary.json").write_text(json.dumps(asdict(summary), indent=2))
    detail_path = out_dir / f"{ts}_detail.jsonl"
    with detail_path.open("w") as f:
        for q in per_q:
            f.write(json.dumps(asdict(q)) + "\n")

    print(f"\n[bench] passed {n_passed}/{len(per_q)}  "
          f"({summary.pass_rate * 100:.1f}%)  "
          f"mean wall {summary.mean_wall_s:.1f}s  "
          f"verified {verified_total}/{total_claims} "
          f"({summary.verified_ratio * 100:.1f}%)")
    print(f"[bench] results: {out_dir}")
    return summary


# ── CLI ──────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a mini-benchmark fixture.")
    parser.add_argument("fixture", help="Path to a JSONL fixture (relative to benchmarks/ ok).")
    parser.add_argument("--model", default=None,
                        help="Override all MODEL_* env vars with this single value.")
    parser.add_argument("--ablate", default="",
                        help="Comma-separated ablation flags: rerank, no-fetch, no-compress, "
                             "no-verify, no-flare, no-router.")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    fixture_path = Path(args.fixture)
    if not fixture_path.exists() and (BENCH_DIR / fixture_path).exists():
        fixture_path = BENCH_DIR / fixture_path

    ablations = [f for f in args.ablate.split(",") if f.strip()]
    summary = run_benchmark(
        fixture_path,
        model=args.model,
        ablations=ablations,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
    return 0 if summary.pass_rate >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
