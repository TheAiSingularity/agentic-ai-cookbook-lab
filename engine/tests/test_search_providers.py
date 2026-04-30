"""Mocked tests for the web-search provider dispatch (`_searxng` vs `_exa`).

No network, no real API keys. The Exa SDK is patched in via
`monkeypatch.setattr(pipeline, "_get_exa_client", ...)` so each test
controls exactly what the SDK returns. The SearXNG path is exercised
through `requests.get` patching, matching the recipe-level test fixture
pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engine.core import pipeline  # noqa: E402


# ── Fakes ───────────────────────────────────────────────────────────


def _exa_response(items: list[dict]) -> SimpleNamespace:
    """Build an object that walks like Exa's SearchResponse."""
    results = []
    for it in items:
        results.append(SimpleNamespace(
            url=it.get("url", ""),
            title=it.get("title", ""),
            text=it.get("text"),
            highlights=it.get("highlights"),
            summary=it.get("summary"),
        ))
    return SimpleNamespace(results=results)


def _stub_exa_client(captured: dict, response: SimpleNamespace):
    """Return a fake Exa client whose .search_and_contents records kwargs."""
    headers: dict = {}

    def search_and_contents(query, **kwargs):
        captured["query"] = query
        captured["kwargs"] = kwargs
        return response

    return SimpleNamespace(headers=headers, search_and_contents=search_and_contents)


# ── _exa: response shape ────────────────────────────────────────────


def test_exa_returns_url_title_snippet_using_summary(monkeypatch):
    captured: dict = {}
    response = _exa_response([
        {"url": "https://a", "title": "A title", "summary": "A summary",
         "highlights": ["ignored when summary present"], "text": "ignored too"},
    ])
    client = _stub_exa_client(captured, response)
    monkeypatch.setattr(pipeline, "_get_exa_client", lambda: client)

    hits = pipeline._exa("anything", n=3)

    assert hits == [{"url": "https://a", "title": "A title", "snippet": "A summary"}]
    assert captured["query"] == "anything"
    assert captured["kwargs"]["num_results"] == 3
    assert captured["kwargs"]["type"] == "auto"
    assert "summary" in captured["kwargs"]
    assert "highlights" in captured["kwargs"]


def test_exa_falls_back_to_highlights_when_summary_missing(monkeypatch):
    response = _exa_response([
        {"url": "https://b", "title": "B", "highlights": ["one", "", "two"]},
    ])
    monkeypatch.setattr(pipeline, "_get_exa_client",
                        lambda: _stub_exa_client({}, response))
    hits = pipeline._exa("q")
    assert hits[0]["snippet"] == "one … two"


def test_exa_falls_back_to_text_when_no_summary_or_highlights(monkeypatch):
    long_text = "x" * 5000
    response = _exa_response([{"url": "https://c", "title": "C", "text": long_text}])
    monkeypatch.setattr(pipeline, "_get_exa_client",
                        lambda: _stub_exa_client({}, response))
    monkeypatch.setattr(pipeline, "EXA_HIGHLIGHTS_CHARS", 200)
    hits = pipeline._exa("q")
    assert len(hits[0]["snippet"]) == 200


def test_exa_returns_empty_snippet_when_no_content(monkeypatch):
    response = _exa_response([{"url": "https://d", "title": "D"}])
    monkeypatch.setattr(pipeline, "_get_exa_client",
                        lambda: _stub_exa_client({}, response))
    hits = pipeline._exa("q")
    assert hits[0] == {"url": "https://d", "title": "D", "snippet": ""}


def test_exa_returns_empty_list_when_response_has_no_results(monkeypatch):
    monkeypatch.setattr(pipeline, "_get_exa_client",
                        lambda: _stub_exa_client({}, SimpleNamespace(results=None)))
    assert pipeline._exa("q") == []


# ── _exa: kwargs forwarded to SDK ───────────────────────────────────


def test_exa_forwards_filter_env_vars_to_sdk(monkeypatch):
    captured: dict = {}
    response = _exa_response([])
    monkeypatch.setattr(pipeline, "_get_exa_client",
                        lambda: _stub_exa_client(captured, response))
    monkeypatch.setattr(pipeline, "EXA_SEARCH_TYPE", "neural")
    monkeypatch.setattr(pipeline, "EXA_CATEGORY", "research paper")
    monkeypatch.setattr(pipeline, "EXA_INCLUDE_DOMAINS", "arxiv.org, nature.com")
    monkeypatch.setattr(pipeline, "EXA_EXCLUDE_DOMAINS", "spam.example")
    monkeypatch.setattr(pipeline, "EXA_START_PUBLISHED_DATE", "2025-01-01")
    monkeypatch.setattr(pipeline, "EXA_END_PUBLISHED_DATE", "2026-01-01")

    pipeline._exa("q", n=7)

    kw = captured["kwargs"]
    assert kw["num_results"] == 7
    assert kw["type"] == "neural"
    assert kw["category"] == "research paper"
    assert kw["include_domains"] == ["arxiv.org", "nature.com"]
    assert kw["exclude_domains"] == ["spam.example"]
    assert kw["start_published_date"] == "2025-01-01"
    assert kw["end_published_date"] == "2026-01-01"


# ── _get_exa_client: lazy + tracking header ─────────────────────────


def test_get_exa_client_raises_without_api_key(monkeypatch):
    monkeypatch.setattr(pipeline, "EXA_API_KEY", "")
    monkeypatch.setattr(pipeline, "_EXA_CLIENT", None)
    with pytest.raises(RuntimeError, match="EXA_API_KEY"):
        pipeline._get_exa_client()


def test_get_exa_client_sets_integration_header(monkeypatch):
    monkeypatch.setattr(pipeline, "EXA_API_KEY", "fake-key")
    monkeypatch.setattr(pipeline, "_EXA_CLIENT", None)

    fake_client = SimpleNamespace(headers={"x-api-key": "fake-key"})

    class FakeExa:
        def __init__(self, key):
            assert key == "fake-key"
            self.__dict__.update(fake_client.__dict__)

    fake_module = SimpleNamespace(Exa=FakeExa)
    monkeypatch.setitem(sys.modules, "exa_py", fake_module)

    client = pipeline._get_exa_client()
    assert client.headers.get("x-exa-integration") == "agentic-research-engine-oss"


# ── _web_search: dispatch ───────────────────────────────────────────


def test_web_search_defaults_to_searxng(monkeypatch):
    monkeypatch.setattr(pipeline, "SEARCH_PROVIDER", "searxng")
    called = {"n": 0}

    def fake_searxng(query, n=5):
        called["n"] += 1
        assert query == "q"
        return [{"url": "u", "title": "t", "snippet": "s"}]

    monkeypatch.setattr(pipeline, "_searxng", fake_searxng)
    monkeypatch.setattr(pipeline, "_exa", lambda *a, **k: pytest.fail("exa called"))

    hits = pipeline._web_search("q")
    assert called["n"] == 1
    assert hits[0]["url"] == "u"


def test_web_search_dispatches_to_exa_when_configured(monkeypatch):
    monkeypatch.setattr(pipeline, "SEARCH_PROVIDER", "exa")
    called = {"n": 0}

    def fake_exa(query, n=5):
        called["n"] += 1
        assert query == "q"
        return [{"url": "ex", "title": "et", "snippet": "es"}]

    monkeypatch.setattr(pipeline, "_exa", fake_exa)
    monkeypatch.setattr(pipeline, "_searxng",
                        lambda *a, **k: pytest.fail("searxng called"))

    hits = pipeline._web_search("q")
    assert called["n"] == 1
    assert hits[0]["url"] == "ex"


# ── End-to-end through _search_one ──────────────────────────────────


def test_search_one_uses_dispatcher(monkeypatch):
    """_search_one should pull from _web_search, not call _searxng directly."""
    monkeypatch.setattr(pipeline, "_web_search",
                        lambda q, n=5: [{"url": "https://x", "title": "X", "snippet": "snip"}])
    monkeypatch.setattr(pipeline, "_chat", lambda model, prompt: "summary [1].")

    out = pipeline._search_one("subq")
    assert out == [{"url": "https://x", "title": "X", "text": "summary [1]."}]
