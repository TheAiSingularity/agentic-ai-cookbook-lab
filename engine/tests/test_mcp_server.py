"""Mocked tests for engine.mcp.server — verify the `research` tool shape
without running the MCP wire protocol. The pipeline is stubbed at
`run_query` so these tests are fast and deterministic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


# Import the server module — this will fail with a friendly message if
# the `mcp` SDK isn't installed, which we handle via skipif.
mcp_available = True
try:
    from engine.mcp import server as mcp_server  # noqa: E402
except SystemExit:  # pragma: no cover
    mcp_available = False


from engine.interfaces.common import RunResult  # noqa: E402


pytestmark = pytest.mark.skipif(
    not mcp_available,
    reason="mcp SDK not installed; `pip install -r engine/requirements.txt` to enable",
)


def _fake_result(question: str) -> RunResult:
    return RunResult(
        question=question,
        domain="general",
        question_class="factoid",
        answer="Final answer [1].",
        verified_claims=[{"text": "claim1", "verified": True}],
        unverified_claims=["claim2"],
        sources=[{"url": "https://example/1", "title": "t1", "text": "evidence", "fetched": True}],
        trace=[{"node": "synthesize", "model": "m", "latency_s": 1.0, "tokens_est": 200}],
        memory_hits=[],
        iterations=1,
        total_latency_s=1.0,
        total_tokens_est=200,
    )


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Stub `run_query` so the MCP tool returns predictable structure."""
    monkeypatch.setattr(mcp_server, "run_query", lambda q, **kw: _fake_result(q))
    yield


def test_research_tool_returns_expected_shape(fake_pipeline):
    out = mcp_server.research("what is rag", domain="general", memory="session")
    assert out["question"] == "what is rag"
    assert out["question_class"] == "factoid"
    assert out["answer"] == "Final answer [1]."
    assert out["verified_summary"].startswith("1/2 claims verified")
    assert isinstance(out["verified_claims"], list) and len(out["verified_claims"]) == 1
    assert out["unverified_claims"] == ["claim2"]
    assert out["sources"][0]["url"] == "https://example/1"
    assert out["sources"][0]["fetched"] is True
    assert out["totals"]["wall_s"] == 1.0
    assert out["totals"]["tokens_est"] == 200


def test_research_tool_coerces_invalid_memory_mode(fake_pipeline):
    out = mcp_server.research("q", memory="bogus")
    # Falls back to "session" silently.
    assert "answer" in out


def test_research_tool_output_is_json_serializable(fake_pipeline):
    out = mcp_server.research("q")
    # Serializes cleanly — MCP expects JSON-safe payloads over the wire.
    s = json.dumps(out, default=str)
    assert "Final answer" in s


def test_reset_memory_tool_returns_count(monkeypatch, tmp_path):
    from engine.core.memory import MemoryStore

    # Seed a persistent store so reset has something to wipe.
    def fake_embed(batch):
        return [[1.0] for _ in batch]

    store = MemoryStore.open("persistent", path=tmp_path / "mem.db", embedder=fake_embed)
    from engine.core.memory import Trajectory
    import time
    store.record(Trajectory(
        query_id="q-1", timestamp=time.time(), question="x", domain="g",
        final_answer="y",
    ))
    store.close()

    # Route the module's MemoryStore.open to our tmp_path.
    import engine.core.memory as memory_mod
    real_open = memory_mod.MemoryStore.open

    def redirected_open(mode, *, path=None, embedder=None):
        return real_open(mode, path=tmp_path / "mem.db", embedder=fake_embed)

    monkeypatch.setattr(memory_mod.MemoryStore, "open", classmethod(
        lambda cls, mode, *, path=None, embedder=None: real_open(
            mode, path=tmp_path / "mem.db", embedder=fake_embed,
        )
    ))

    out = mcp_server.reset_memory()
    assert out["reset"] == 1


def test_memory_count_tool_zero_on_fresh(monkeypatch, tmp_path):
    import engine.core.memory as memory_mod

    def fake_embed(batch):
        return [[1.0] for _ in batch]

    real_open = memory_mod.MemoryStore.open
    monkeypatch.setattr(memory_mod.MemoryStore, "open", classmethod(
        lambda cls, mode, *, path=None, embedder=None: real_open(
            mode, path=tmp_path / "fresh.db", embedder=fake_embed,
        )
    ))

    out = mcp_server.memory_count()
    assert out["count"] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
