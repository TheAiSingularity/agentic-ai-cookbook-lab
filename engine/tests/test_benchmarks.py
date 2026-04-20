"""Mocked tests for engine/benchmarks/runner.py — no real pipeline runs.

Uses the same run_query monkeypatch pattern as test_interfaces.py, plus
temp fixtures on disk to exercise the full runner flow end-to-end.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engine.benchmarks import runner  # noqa: E402
from engine.interfaces.common import RunResult  # noqa: E402


def _fake_result(q: str, *, answer: str, verified: int = 1, unverified: int = 0) -> RunResult:
    return RunResult(
        question=q, domain="general",
        answer=answer,
        verified_claims=[{"text": f"c{i}", "verified": True} for i in range(verified)],
        unverified_claims=[f"u{i}" for i in range(unverified)],
        sources=[], trace=[], memory_hits=[],
        question_class="factoid",
        iterations=1,
        total_latency_s=1.5,
        total_tokens_est=200,
    )


@pytest.fixture
def stubbed_pipeline(monkeypatch):
    """Replace runner's run_query with an answer-from-question stub."""
    def fake_run_query(q, **kwargs):
        # Echo the question so scoring behavior is predictable.
        return _fake_result(q, answer=f"Answer: {q}", verified=2, unverified=1)
    monkeypatch.setattr(runner, "run_query", fake_run_query)
    return fake_run_query


@pytest.fixture
def tiny_fixture(tmp_path):
    rows = [
        {"id": "t-01", "domain": "general", "question": "What is RAG?",
         "gold": {"must_contain": ["RAG"], "must_not_contain": ["nonsense"]}},
        {"id": "t-02", "domain": "papers", "question": "About BM25",
         "gold": {"must_contain": ["BM25"]}},
        {"id": "t-03", "domain": "general", "question": "A forbidden topic",
         "gold": {"must_contain": [], "must_not_contain": ["forbidden"]}},
    ]
    p = tmp_path / "tiny.jsonl"
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


# ── _score ──────────────────────────────────────────────────────────

def test_score_passes_when_all_must_contain_present():
    missing, hits, passed = runner._score("hello BM25 and rag", {"must_contain": ["BM25", "rag"]})
    assert missing == []
    assert hits == []
    assert passed is True


def test_score_fails_when_must_contain_missing():
    missing, hits, passed = runner._score("only BM25", {"must_contain": ["BM25", "RRF"]})
    assert missing == ["RRF"]
    assert passed is False


def test_score_fails_when_must_not_contain_matches():
    missing, hits, passed = runner._score(
        "definitely nonsense here",
        {"must_contain": [], "must_not_contain": ["nonsense"]},
    )
    assert hits == ["nonsense"]
    assert passed is False


def test_score_is_case_insensitive():
    missing, hits, passed = runner._score("Hello BM25", {"must_contain": ["bm25"]})
    assert missing == []
    assert passed is True


# ── ablation flag translation ───────────────────────────────────────

def test_apply_ablations_translates_known_flags(monkeypatch):
    # Clear env first
    for k in ("ENABLE_RERANK", "ENABLE_FETCH", "ENABLE_COMPRESS",
              "ENABLE_VERIFY", "ENABLE_ACTIVE_RETR", "ENABLE_ROUTER"):
        monkeypatch.delenv(k, raising=False)

    applied = runner._apply_ablations([
        "rerank", "no-fetch", "no-compress", "no-verify", "no-flare", "no-router"
    ])
    assert applied == {
        "ENABLE_RERANK": "1",
        "ENABLE_FETCH": "0",
        "ENABLE_COMPRESS": "0",
        "ENABLE_VERIFY": "0",
        "ENABLE_ACTIVE_RETR": "0",
        "ENABLE_ROUTER": "0",
    }
    import os
    for k, v in applied.items():
        assert os.environ[k] == v


def test_apply_ablations_ignores_unknown_flag(monkeypatch, capsys):
    applied = runner._apply_ablations(["bogus"])
    assert applied == {}
    err = capsys.readouterr().err
    assert "unknown ablation" in err


# ── run_benchmark ────────────────────────────────────────────────────

def test_run_benchmark_writes_summary_and_detail(stubbed_pipeline, tiny_fixture, tmp_path):
    out = tmp_path / "out"
    summary = runner.run_benchmark(tiny_fixture, out_dir=out)

    assert summary.n_questions == 3
    # Each answer is "Answer: {question}", scoring must_contain / must_not_contain accordingly.
    assert summary.n_passed >= 2  # t-01 and t-02 at least should pass (their keywords are in the echoed question)

    files = list(out.glob("*"))
    assert any(f.name.endswith("_summary.json") for f in files)
    assert any(f.name.endswith("_detail.jsonl") for f in files)

    # Summary JSON is valid.
    summary_json = next(f for f in files if f.name.endswith("_summary.json"))
    data = json.loads(summary_json.read_text())
    assert data["n_questions"] == 3
    assert "per_question" in data
    assert len(data["per_question"]) == 3


def test_run_benchmark_records_ablations_in_summary(stubbed_pipeline, tiny_fixture, tmp_path, monkeypatch):
    for k in ("ENABLE_RERANK", "ENABLE_FETCH"):
        monkeypatch.delenv(k, raising=False)
    summary = runner.run_benchmark(
        tiny_fixture, out_dir=tmp_path / "out", ablations=["rerank", "no-fetch"]
    )
    assert summary.ablations["ENABLE_RERANK"] == "1"
    assert summary.ablations["ENABLE_FETCH"] == "0"


def test_run_benchmark_handles_runner_exception(stubbed_pipeline, tiny_fixture, tmp_path, monkeypatch):
    def boom(q, **kwargs):
        raise RuntimeError("pipeline blew up")

    monkeypatch.setattr(runner, "run_query", boom)
    summary = runner.run_benchmark(tiny_fixture, out_dir=tmp_path / "out")
    # All 3 questions should count as failures with [bench error] answers.
    assert summary.n_passed == 0
    for q in summary.per_question:
        assert "[bench error]" in q["answer"]


def test_run_benchmark_model_override_propagates(stubbed_pipeline, tiny_fixture, tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_SYNTHESIZER", raising=False)
    _ = runner.run_benchmark(tiny_fixture, model="gemma3:4b", out_dir=tmp_path / "out")
    import os
    assert os.environ["MODEL_SYNTHESIZER"] == "gemma3:4b"


def test_run_benchmark_missing_fixture_errors():
    with pytest.raises(FileNotFoundError):
        runner.run_benchmark(Path("/nonexistent.jsonl"))


# ── Fixture invariants (shipped files) ──────────────────────────────

def test_shipped_simpleqa_fixture_is_valid_jsonl():
    p = Path(runner.BENCH_DIR) / "simpleqa_mini.jsonl"
    assert p.exists()
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert len(rows) == 20
    for r in rows:
        assert "id" in r and "question" in r and "gold" in r
        assert isinstance(r["gold"].get("must_contain", []), list)


def test_shipped_browsecomp_fixture_is_valid_jsonl():
    p = Path(runner.BENCH_DIR) / "browsecomp_mini.jsonl"
    assert p.exists()
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    assert len(rows) == 10
    for r in rows:
        assert r["id"].startswith("bc-")
        assert r.get("domain") in {"general", "medical", "papers", "financial", "stock_trading", "personal_docs"}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
