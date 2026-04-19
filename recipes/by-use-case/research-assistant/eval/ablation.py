"""Ablation runner — executes the research agent across a matrix of configs
and records per-question metrics to `results.jsonl` for Pareto analysis.

Usage:
    python ablation.py --dataset datasets/simpleqa_seed.jsonl
    python ablation.py --dataset datasets/browsecomp_plus_seed.jsonl \
                       --configs A1,A2,A3,B1,B2,B3,B4
    python ablation.py --dataset datasets/simpleqa_100.jsonl --seeds 3

Configs are symbolic labels — each one sets the env vars the production
main.py reads. See `CONFIGS` below.

Design notes:
  - Every config launches a fresh subprocess per question so env-var changes
    actually take effect (main.py reads env at import time).
  - Output is streamed to results.jsonl, one JSON object per (config, question,
    seed) triple. Resumable: existing entries are skipped on re-run.
  - MiroThinker-1.7 baseline can be added as an external config that invokes
    its own inference script; we treat it as a black-box numbers source
    (see docs/paper-draft.md for methodology).

Requires OPENAI_API_KEY for the judge. The agent backend is whatever the
current env selects (OpenAI, Ollama, vLLM, SGLang).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
BEGINNER = REPO_ROOT / "recipes/by-use-case/research-assistant/beginner/main.py"
PRODUCTION = REPO_ROOT / "recipes/by-use-case/research-assistant/production/main.py"


@dataclass
class Config:
    """A symbolic ablation config — label + env var overrides + which main to run."""

    label: str
    description: str
    main: Path
    env: dict[str, str] = field(default_factory=dict)


# Default ablation matrix: 7 configs spanning Tier 1 + Tier 2.
CONFIGS: dict[str, Config] = {
    "A1": Config("A1", "Beginner + v0 RAG (naive cosine)", BEGINNER, {"RAG_VERSION": "v0"}),
    "A2": Config("A2", "Beginner + v1 RAG (hybrid BM25+dense+RRF)", BEGINNER, {"RAG_VERSION": "v1"}),
    "A3": Config("A3", "Beginner + v1 + contextual chunking", BEGINNER, {"RAG_VERSION": "v1", "ENABLE_CONTEXTUAL": "1"}),
    "B1": Config("B1", "Production, verify only (CoVe)", PRODUCTION,
                 {"ENABLE_HYDE": "0", "ENABLE_VERIFY": "1", "MAX_ITERATIONS": "0"}),
    "B2": Config("B2", "Production, HyDE + verify", PRODUCTION,
                 {"ENABLE_HYDE": "1", "ENABLE_VERIFY": "1", "MAX_ITERATIONS": "0"}),
    "B3": Config("B3", "Production, HyDE + verify + iterate (×1)", PRODUCTION,
                 {"ENABLE_HYDE": "1", "ENABLE_VERIFY": "1", "MAX_ITERATIONS": "1"}),
    "B4": Config("B4", "Production, all techniques + self-consistency N=3", PRODUCTION,
                 {"ENABLE_HYDE": "1", "ENABLE_VERIFY": "1", "MAX_ITERATIONS": "1",
                  "ENABLE_CONSISTENCY": "1", "CONSISTENCY_SAMPLES": "3"}),
}


def _run_one(config: Config, question: str) -> tuple[str, float, int]:
    """Invoke main.py as subprocess with config's env; return (answer, latency_s, exit_code)."""
    env = {**os.environ, **config.env}
    env.setdefault("OPENAI_API_KEY", "ollama")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, str(config.main), question],
        env=env, capture_output=True, text=True, timeout=600,
    )
    dt = time.time() - t0
    # main.py prints "Q: ...\nA: <answer>..." — grab everything after first "A: ".
    out = proc.stdout
    if "A: " in out:
        answer = out.split("A: ", 1)[1].strip()
    else:
        answer = out.strip()
    return answer, dt, proc.returncode


def _load_existing(path: Path) -> set[tuple[str, str, int]]:
    """Return (question_id, config_label, seed) tuples already recorded."""
    if not path.exists():
        return set()
    seen: set[tuple[str, str, int]] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            seen.add((row["question_id"], row["config"], row["seed"]))
        except (ValueError, KeyError):
            continue
    return seen


def _load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_ablation(dataset_path: Path, configs: list[Config], seeds: int, out_path: Path) -> None:
    dataset = _load_dataset(dataset_path)
    done = _load_existing(out_path)
    total = len(dataset) * len(configs) * seeds
    completed = len(done)
    print(f"Dataset: {dataset_path.name} ({len(dataset)} questions)")
    print(f"Configs: {', '.join(c.label for c in configs)}")
    print(f"Seeds: {seeds}  →  {total} runs total ({completed} already done)")
    print()
    with out_path.open("a") as fp:
        for q in dataset:
            for config in configs:
                for seed in range(seeds):
                    key = (q["id"], config.label, seed)
                    if key in done:
                        continue
                    print(f"[{q['id']}/{config.label}/s{seed}] ", end="", flush=True)
                    try:
                        answer, latency, rc = _run_one(config, q["question"])
                    except subprocess.TimeoutExpired:
                        answer, latency, rc = "", 600.0, -1
                    row = {
                        "question_id": q["id"],
                        "config": config.label,
                        "seed": seed,
                        "question": q["question"],
                        "gold_answer": q["gold_answer"],
                        "must_cite_any": q.get("must_cite_any", []),
                        "source": q.get("source", ""),
                        "answer": answer,
                        "latency_s": round(latency, 3),
                        "exit_code": rc,
                        "tokens_est": max(1, len(answer) // 4),
                    }
                    fp.write(json.dumps(row) + "\n")
                    fp.flush()
                    completed += 1
                    print(f"{latency:.1f}s  (rc={rc})  [{completed}/{total}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--configs", default="A1,A2,A3,B1,B2,B3,B4")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", type=Path, default=HERE / "results.jsonl")
    args = ap.parse_args()
    configs = [CONFIGS[label.strip()] for label in args.configs.split(",")]
    run_ablation(args.dataset, configs, args.seeds, args.out)
    print(f"\nDone. Results → {args.out}")


if __name__ == "__main__":
    main()
