"""Pareto analysis + plot for ablation results.

Consumes the `results.jsonl` produced by ablation.py, applies the scorer's
LLM-as-judge + citation metrics, and produces:

  1. An aggregate table per (config, dataset) with:
       factuality_mean, citation_accuracy, latency_p50, latency_p95, tokens_mean
  2. A Pareto scatter plot (PNG): accuracy vs compute (tokens or latency).

Usage:
    python pareto.py                             # score results.jsonl, emit table + plot.png
    python pareto.py --results results.jsonl --out-plot pareto.png
    python pareto.py --no-judge                  # skip LLM-as-judge (uses exact-match only)

Requires OPENAI_API_KEY for the judge unless `--no-judge`.
matplotlib is imported lazily so dry-run analysis works in minimal envs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, quantiles

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from scorer import _citation_accuracy, _citation_precision, _estimate_tokens, _judge_factuality  # noqa: E402


def _score_row(row: dict, judge_client, no_judge: bool) -> dict:
    """Compute per-row metrics. Returns a dict merged into the row."""
    ans = row.get("answer", "")
    ev: list[dict] = []  # scoring doesn't need evidence for factuality/precision,
    # and citation_accuracy treats [N] refs relative to implicit evidence count.
    # For a cleaner score, future runs should persist evidence alongside the answer.
    if no_judge:
        fact = 1.0 if row["gold_answer"].lower()[:40] in ans.lower() else 0.0
    else:
        fact = _judge_factuality(judge_client, row["question"], row["gold_answer"], ans)
    return {
        "factuality": fact,
        "citation_precision": _citation_precision(ans, row.get("must_cite_any", [])),
        "citation_accuracy": _citation_accuracy(ans, [{"url": ""} for _ in range(row.get("tokens_est", 0) // 100 + 1)]),
        # Fallback evidence count proxy — users' real runs should record evidence list
    }


def aggregate(rows: list[dict]) -> dict[tuple[str, str], dict]:
    """Aggregate per (source, config) — mean factuality, percentile latency, mean tokens."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r.get("source", ""), r["config"])].append(r)
    out: dict[tuple[str, str], dict] = {}
    for key, rs in groups.items():
        latencies = sorted(r.get("latency_s", 0.0) for r in rs)
        p50 = latencies[len(latencies) // 2] if latencies else 0.0
        p95 = quantiles(latencies, n=20)[18] if len(latencies) >= 20 else latencies[-1]
        out[key] = {
            "n": len(rs),
            "factuality_mean": round(mean(r["factuality"] for r in rs), 3) if rs else 0.0,
            "citation_precision_mean": round(mean(r["citation_precision"] for r in rs), 3) if rs else 0.0,
            "citation_accuracy_mean": round(mean(r["citation_accuracy"] for r in rs), 3) if rs else 0.0,
            "latency_p50_s": round(p50, 2),
            "latency_p95_s": round(p95, 2),
            "tokens_mean": round(mean(r.get("tokens_est", 0) for r in rs), 0) if rs else 0,
        }
    return out


def print_table(agg: dict[tuple[str, str], dict]) -> None:
    """Human-readable table grouped by dataset."""
    by_source: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for (source, config), row in agg.items():
        by_source[source].append((config, row))
    for source, entries in sorted(by_source.items()):
        entries.sort(key=lambda t: t[0])
        print(f"\n=== {source or '(unknown)'} ===")
        header = f"{'cfg':<4} {'n':>4} {'fact':>6} {'cprec':>7} {'cacc':>6} {'p50s':>7} {'p95s':>7} {'tok':>6}"
        print(header)
        print("-" * len(header))
        for cfg, row in entries:
            print(f"{cfg:<4} {row['n']:>4} {row['factuality_mean']:>6.3f} "
                  f"{row['citation_precision_mean']:>7.3f} {row['citation_accuracy_mean']:>6.3f} "
                  f"{row['latency_p50_s']:>7.2f} {row['latency_p95_s']:>7.2f} "
                  f"{int(row['tokens_mean']):>6}")


def pareto_plot(agg: dict[tuple[str, str], dict], out_path: Path) -> None:
    """Scatter plot: x=p50 latency, y=factuality. One panel per dataset."""
    try:
        import matplotlib.pyplot as plt  # noqa: E402
    except ImportError:
        print("matplotlib not installed — skipping plot. `pip install matplotlib` to enable.", file=sys.stderr)
        return
    by_source: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for (source, config), row in agg.items():
        by_source[source].append((config, row))
    n = len(by_source)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)
    for ax, (source, entries) in zip(axes[0], sorted(by_source.items())):
        for cfg, row in entries:
            ax.scatter(row["latency_p50_s"], row["factuality_mean"], s=80)
            ax.annotate(cfg, (row["latency_p50_s"], row["factuality_mean"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=9)
        ax.set_xlabel("p50 latency (s)")
        ax.set_ylabel("factuality (LLM-judge)")
        ax.set_title(f"Pareto — {source}")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nPareto plot → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results.jsonl")
    ap.add_argument("--out-plot", type=Path, default=HERE / "pareto.png")
    ap.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge factuality (exact-match fallback).")
    args = ap.parse_args()

    if not args.results.exists():
        print(f"No results file at {args.results}. Run ablation.py first.", file=sys.stderr)
        sys.exit(1)

    rows = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    judge = None
    if not args.no_judge:
        from openai import OpenAI
        judge = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
                       base_url=os.environ.get("OPENAI_BASE_URL"))

    scored = [{**r, **_score_row(r, judge, args.no_judge)} for r in rows]
    agg = aggregate(scored)
    print_table(agg)
    pareto_plot(agg, args.out_plot)


if __name__ == "__main__":
    main()
