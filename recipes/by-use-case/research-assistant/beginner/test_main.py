"""Mocked tests for research-assistant/beginner (OpenAI-only variant).

No API keys required — all OpenAI client calls are patched. Verifies
the graph wiring, node contracts, and state shape. For a real live
check, use `make smoke` with OPENAI_API_KEY set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

os.environ.setdefault("OPENAI_API_KEY", "test")


def _chat_resp(text: str) -> object:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _responses_resp(blocks: list[tuple[str, list[str]]]) -> object:
    """blocks = [(answer_text, [cited_urls]), ...]"""
    message_items = []
    for text, urls in blocks:
        anns = [SimpleNamespace(type="url_citation", url=u) for u in urls]
        block = SimpleNamespace(type="output_text", text=text, annotations=anns)
        message_items.append(SimpleNamespace(type="message", content=[block]))
    return SimpleNamespace(output=message_items)


@pytest.fixture
def patched_openai(monkeypatch):
    """Patch OpenAI client with chat (plan/synthesize) and responses (search) stubs."""
    import main

    def chat_router(*args, **kwargs):
        prompt = kwargs.get("messages", [{}])[0].get("content", "")
        if "Break this research question" in prompt:
            return _chat_resp("sub one\nsub two\nsub three")
        return _chat_resp("Final answer with [1][2] citations.")

    client = mock.MagicMock()
    client.chat.completions.create.side_effect = chat_router
    client.responses.create.return_value = _responses_resp(
        [("snippet from web search", ["https://a.example/1", "https://b.example/2"])]
    )
    monkeypatch.setattr(main, "OpenAI", mock.MagicMock(return_value=client))

    # Stub core/rag embedder so it doesn't need a real API.
    from core.rag import Retriever

    def fake_embed(batch):
        return [[float(len(s)), float(len(s.split()))] for s in batch]

    original_init = Retriever.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.embedder = fake_embed

    monkeypatch.setattr(Retriever, "__init__", patched_init)
    return client


def test_plan_parses_subqueries(patched_openai):
    import main

    result = main._plan({"question": "q", "subqueries": [], "evidence": [], "answer": ""})
    assert result["subqueries"] == ["sub one", "sub two", "sub three"]


def test_search_collects_url_citations(patched_openai):
    import main

    state = {"question": "q", "subqueries": ["s1", "s2"], "evidence": [], "answer": ""}
    result = main._search(state)
    # 2 sub-queries × 1 message × 2 URLs each = 4 evidence items.
    assert len(result["evidence"]) == 4
    assert all({"url", "title", "text"} <= e.keys() for e in result["evidence"])
    assert all(e["url"].startswith("https://") for e in result["evidence"])


def test_search_falls_back_when_no_citations(patched_openai, monkeypatch):
    import main

    no_citations = _responses_resp([("answer with no citations", [])])
    client = main.OpenAI()
    client.responses.create.return_value = no_citations
    state = {"question": "q", "subqueries": ["only"], "evidence": [], "answer": ""}
    result = main._search(state)
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["url"] == ""
    assert result["evidence"][0]["text"] == "answer with no citations"


def test_retrieve_passes_through_when_few_evidence(patched_openai):
    import main

    evidence = [{"url": "u", "title": "t", "text": "short"}]
    state = {"question": "q", "subqueries": [], "evidence": evidence, "answer": ""}
    assert main._retrieve(state)["evidence"] == evidence


def test_retrieve_narrows_when_many_evidence(patched_openai, monkeypatch):
    import main

    monkeypatch.setattr(main, "TOP_K_EVIDENCE", 3)
    evidence = [{"url": f"u{i}", "title": f"t{i}", "text": f"text {i}"} for i in range(10)]
    state = {"question": "query", "subqueries": [], "evidence": evidence, "answer": ""}
    assert len(main._retrieve(state)["evidence"]) == 3


def test_synthesize_returns_answer_string(patched_openai):
    import main

    state = {"question": "q", "subqueries": [], "evidence": [{"url": "u", "title": "t", "text": "snippet"}], "answer": ""}
    assert "Final answer" in main._synthesize(state)["answer"]


def test_full_graph_end_to_end(patched_openai):
    import main

    result = main.build_graph().invoke({"question": "test", "subqueries": [], "evidence": [], "answer": ""})
    assert result["subqueries"] == ["sub one", "sub two", "sub three"]
    assert len(result["evidence"]) > 0
    assert "Final answer" in result["answer"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
