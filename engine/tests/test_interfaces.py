"""Mocked tests for engine.interfaces.{common,cli} — no real LLM calls.

The pipeline itself is mocked at the `build_graph` level: we monkeypatch
`engine.interfaces.common.build_graph` to return a fake graph whose
`.invoke(state)` returns a pre-canned final state. That way we exercise
the interface orchestration (memory injection, trajectory recording, CLI
output rendering) without any network or LLM work.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engine.core.memory import MemoryStore  # noqa: E402
from engine.interfaces import cli as cli_mod  # noqa: E402
from engine.interfaces import common as common_mod  # noqa: E402


# ── Pipeline stub ───────────────────────────────────────────────────

def _fake_final_state(*, question: str, answer: str = "Final answer [1].",
                      extra_trace: list[dict] | None = None) -> dict:
    trace = extra_trace or [
        {"node": "classify", "model": "test-m", "latency_s": 0.1, "tokens_est": 30},
        {"node": "synthesize", "model": "test-m", "latency_s": 1.5, "tokens_est": 400},
    ]
    return {
        "question": question,
        "question_class": "factoid",
        "answer": answer,
        "evidence": [{"url": "https://example/1", "title": "t1", "text": "evidence 1", "fetched": True}],
        "evidence_compressed": [{"url": "https://example/1", "title": "t1", "text": "compressed 1", "fetched": True}],
        "claims": [{"text": "claim1", "verified": True}, {"text": "claim2", "verified": False}],
        "unverified": ["claim2"],
        "iterations": 1,
        "trace": trace,
    }


@pytest.fixture
def patched_graph(monkeypatch):
    """Replace build_graph in engine.interfaces.common with a fake."""
    calls = {"n": 0, "last_state": None}

    fake_graph = mock.MagicMock()

    def fake_invoke(state):
        calls["n"] += 1
        calls["last_state"] = state
        return _fake_final_state(question=state.get("question", ""))

    fake_graph.invoke = fake_invoke
    monkeypatch.setattr(common_mod, "build_graph", lambda: fake_graph)
    return calls


# ── common.run_query ────────────────────────────────────────────────

def test_run_query_without_memory_returns_result(patched_graph):
    r = common_mod.run_query("what is rag", domain="general")
    assert r.question == "what is rag"
    assert r.answer == "Final answer [1]."
    assert r.question_class == "factoid"
    assert len(r.verified_claims) == 1
    assert r.unverified_claims == ["claim2"]
    assert r.total_latency_s > 0
    assert r.total_tokens_est >= 400
    assert r.iterations == 1
    assert r.memory_hits == []


def test_run_query_with_session_memory_records_trajectory(patched_graph):
    def fake_embed(batch):
        return [[float(len(s)), 0.0, 0.0] for s in batch]

    store = MemoryStore.open("session", embedder=fake_embed)
    _ = common_mod.run_query("first q", memory=store)
    assert store.count() == 1

    _ = common_mod.run_query("second q", memory=store)
    assert store.count() == 2


def test_run_query_with_memory_injects_hits_into_prompt(patched_graph):
    def fake_embed(batch):
        # All questions map to the same 2-d embedding so cosine = 1.
        return [[1.0, 0.0] for _ in batch]

    store = MemoryStore.open("session", embedder=fake_embed)
    # Prime memory with a related trajectory first.
    _ = common_mod.run_query("seed question", memory=store)

    # Second query: memory retrieval should fire; hits are in result.memory_hits.
    r = common_mod.run_query("follow up question", memory=store)
    assert len(r.memory_hits) >= 1

    # The pipeline's invoke saw the injected-question prompt (includes "Context from prior").
    invoked_state = patched_graph["last_state"]
    assert "Context from prior related research" in invoked_state["question"]


def test_format_verified_summary_various_cases(patched_graph):
    r1 = common_mod.run_query("q", memory=None)
    text = common_mod.format_verified_summary(r1)
    assert "1/2 claims verified" in text
    assert "1 unverified" in text


def test_format_sources_rows(patched_graph):
    r = common_mod.run_query("q", memory=None)
    rows = common_mod.format_sources(r)
    assert len(rows) == 1
    row = rows[0]
    assert row["idx"] == 1
    assert row["fetched"] is True
    assert row["url"].startswith("https://")


def test_format_trace_per_node(patched_graph):
    r = common_mod.run_query("q", memory=None)
    rows = common_mod.format_trace_per_node(r)
    assert {row["node"] for row in rows} == {"classify", "synthesize"}
    # Rows sorted by latency desc.
    assert rows[0]["latency_s"] >= rows[-1]["latency_s"]


# ── CLI parser + commands ────────────────────────────────────────────

def test_cli_build_parser_supports_bare_question(patched_graph, capsys):
    # When the first arg isn't a known subcommand, we rewrite to `ask <question>`.
    exit_code = cli_mod.main(["what is rag"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "what is rag" in out
    assert "Final answer [1]" in out


def test_cli_ask_json_output(patched_graph, capsys):
    exit_code = cli_mod.main(["ask", "what is rag", "--output", "json"])
    assert exit_code == 0
    raw = capsys.readouterr().out
    payload = json.loads(raw)
    assert payload["question"] == "what is rag"
    assert payload["answer"] == "Final answer [1]."
    assert payload["question_class"] == "factoid"
    assert "trace_per_node" in payload


def test_cli_ask_markdown_output_shows_sections(patched_graph, capsys):
    exit_code = cli_mod.main(["ask", "q", "--domain", "medical"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "sources" in out
    assert "hallucination check" in out
    assert "trace (per-node totals)" in out


def test_cli_reset_memory_wipes_store(tmp_path, patched_graph, capsys):
    db = tmp_path / "mem.db"
    # Record something first via run_query with persistent mode.
    def fake_embed(batch):
        return [[1.0, 0.0] for _ in batch]

    store = MemoryStore.open("persistent", path=db, embedder=fake_embed)
    _ = common_mod.run_query("seed", memory=store)
    store.close()
    assert db.exists()

    exit_code = cli_mod.main(["reset-memory", "--db-path", str(db)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "reset:" in out
    assert "1 trajectories wiped" in out


def test_cli_memory_count_reports_zero_on_fresh_db(tmp_path, patched_graph, capsys):
    db = tmp_path / "fresh.db"
    exit_code = cli_mod.main(["memory-count", "--db-path", str(db)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "0 trajectories" in out


def test_cli_version_prints(patched_graph, capsys):
    exit_code = cli_mod.main(["version"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert out.startswith("engine v")


def test_cli_domains_list_handles_absent_presets(patched_graph, capsys, monkeypatch):
    # Point the lookup at a temp dir with no YAML files.
    fake_root = Path("/nonexistent/path")
    monkeypatch.setattr(cli_mod, "_REPO_ROOT", fake_root)
    exit_code = cli_mod.main(["domains", "list"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "(no domain presets installed yet" in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
