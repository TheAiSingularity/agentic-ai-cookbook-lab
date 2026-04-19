"""Scorer for research-assistant — measures factuality and citation precision.

Usage: python scorer.py [dataset.jsonl]

Requires OPENAI_API_KEY (for LLM-as-judge) plus the full recipe stack
(EXA + GOOGLE + OPENAI keys) to run the agent over each question.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent / "beginner"))

from main import build_graph  # noqa: E402
from openai import OpenAI  # noqa: E402

JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-5-mini")
URL_RE = re.compile(r"https?://[^\s)]+")


def _judge_factuality(client: OpenAI, question: str, gold: str, answer: str) -> float:
    """LLM-as-judge. Returns 0.0 / 0.5 / 1.0 for incorrect / partial / correct."""
    prompt = (
        f"Rate the candidate answer against the gold answer for factual accuracy. "
        f"Reply with ONLY a number: 0 (incorrect), 0.5 (partially correct), or 1 (correct).\n\n"
        f"Question: {question}\nGold: {gold}\nCandidate: {answer}"
    )
    resp = client.chat.completions.create(
        model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}]
    )
    text = (resp.choices[0].message.content or "0").strip()
    try:
        return max(0.0, min(1.0, float(text.split()[0])))
    except (ValueError, IndexError):
        return 0.0


def _citation_precision(answer: str, must_cite_any: list[str]) -> float:
    """Proportion of `must_cite_any` domains that appear in the answer's URLs."""
    if not must_cite_any:
        return 1.0
    urls = URL_RE.findall(answer)
    cited_domains = {u for u in urls}
    hits = sum(1 for needle in must_cite_any if any(needle in u for u in cited_domains))
    return hits / max(1, len(must_cite_any))


def score_dataset(path: Path) -> dict:
    graph = build_graph()
    judge = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    factuality: list[float] = []
    citations: list[float] = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        result = graph.invoke(
            {"question": row["question"], "subqueries": [], "evidence": [], "answer": ""}
        )
        f = _judge_factuality(judge, row["question"], row["gold_answer"], result["answer"])
        c = _citation_precision(result["answer"], row.get("must_cite_any", []))
        factuality.append(f)
        citations.append(c)
        print(f"  {row['id']}: factuality={f:.2f}  citations={c:.2f}")
    return {
        "n": len(factuality),
        "factuality_mean": sum(factuality) / len(factuality) if factuality else 0.0,
        "citation_precision_mean": sum(citations) / len(citations) if citations else 0.0,
    }


if __name__ == "__main__":
    dataset = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "dataset.jsonl"
    print(f"Scoring {dataset} …")
    scores = score_dataset(dataset)
    print(f"\nAggregate over {scores['n']} examples:")
    print(f"  factuality_mean           = {scores['factuality_mean']:.3f}")
    print(f"  citation_precision_mean   = {scores['citation_precision_mean']:.3f}")
