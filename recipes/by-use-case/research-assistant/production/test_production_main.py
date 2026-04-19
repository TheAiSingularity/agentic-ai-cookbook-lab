"""Mocked tests for research-assistant/production (Wave 2 Tiers 2 + 4).

Covers: HyDE gating, CoVe parsing, conditional iteration, self-consistency,
step-level critic (T4.1), FLARE active retrieval (T4.2), question classifier
router (T4.3), evidence compression (T4.4), plan refinement (T4.5).

No network, no API key, no model downloads.
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

os.environ.setdefault("OPENAI_API_KEY", "test")

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("production_main", Path(__file__).parent / "main.py")
main = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(main)


def _chat_resp(text: str) -> object:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _searxng_json(hits: list[tuple[str, str, str]]) -> dict:
    return {"results": [{"url": u, "title": t, "content": s} for u, t, s in hits]}


@pytest.fixture
def patched(monkeypatch):
    """Patch OpenAI (prompt-routed), SearXNG HTTP, and core/rag embedder."""

    def chat_router(*args, **kwargs):
        p = kwargs.get("messages", [{}])[0].get("content", "")
        # T4.3 — classify
        if "Classify this research question" in p:
            return _chat_resp("multihop")
        # T4.1 — step critic: return accept by default so pipelines flow through.
        if "step-level verifier" in p:
            return _chat_resp("VERDICT: accept\nFEEDBACK: ")
        # T4.4 — evidence compression
        if "Compress each numbered chunk" in p:
            return _chat_resp("[1] compressed A\n\n[2] compressed B")
        # HyDE
        if "concise factual paragraph" in p:
            return _chat_resp("Hypothetical answer text about the topic.")
        # Planner decomposition (also catches the refinement variant)
        if "Break this research question" in p:
            return _chat_resp("sub one\nsub two\nsub three")
        # Search summary
        if "Summarize these sources" in p:
            return _chat_resp("Search summary with [1] and [2] citations.")
        # CoVe verification
        if "List each standalone factual claim" in p:
            return _chat_resp("CLAIM: fact one\nVERIFIED: yes\nCLAIM: fact two\nVERIFIED: no\nCLAIM: fact three\nVERIFIED: yes")
        # Synthesizer
        if "Answer using the evidence" in p:
            return _chat_resp("Final answer [1] with citations [2].")
        return _chat_resp("unexpected prompt")

    client = mock.MagicMock()
    client.chat.completions.create.side_effect = chat_router
    monkeypatch.setattr(main, "OpenAI", mock.MagicMock(return_value=client))

    call_i = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        call_i["n"] += 1
        i = call_i["n"]
        r = mock.MagicMock()
        r.status_code = 200
        r.raise_for_status = mock.MagicMock()
        r.json = lambda: _searxng_json([
            (f"https://a.example/{i}-1", f"A{i}", f"snip A{i}"),
            (f"https://b.example/{i}-2", f"B{i}", f"snip B{i}"),
        ])
        return r

    monkeypatch.setattr(main.requests, "get", fake_get)

    from core.rag import HybridRetriever, Retriever

    def fake_embed(batch):
        return [[float(len(s)), float(len(s.split()))] for s in batch]

    for cls in (Retriever, HybridRetriever):
        original = cls.__init__

        def make_patched(orig):
            def patched_init(self, *args, **kwargs):
                orig(self, *args, **kwargs)
                self.embedder = fake_embed

            return patched_init

        monkeypatch.setattr(cls, "__init__", make_patched(original))

    return client


# ── Tier 2 coverage (updated for new state shape) ─────────────────────

def test_plan_parses_subqueries_and_skips_hyde_on_numeric(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_HYDE", True)
    state = {"question": "How many parameters does Gemma 4 have in 2026?", "iterations": 0, "question_class": "factoid"}
    result = main._plan(state)
    # Numeric + factoid → no HyDE.
    assert not any("Hypothetical" in s for s in result["subqueries"])


def test_plan_applies_hyde_on_conceptual_query(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_HYDE", True)
    state = {"question": "Why does contextual retrieval improve recall?", "iterations": 0, "question_class": "multihop"}
    result = main._plan(state)
    assert all("Hypothetical answer" in s for s in result["subqueries"])


def test_plan_skips_hyde_when_disabled(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_HYDE", False)
    state = {"question": "Why does contextual retrieval improve recall?", "iterations": 0, "question_class": "multihop"}
    result = main._plan(state)
    assert not any("Hypothetical" in s for s in result["subqueries"])


def test_verify_parses_cove_and_flags_unverified(patched):
    state = {
        "question": "q",
        "answer": "Final answer",
        "evidence": [{"url": "u1", "text": "E1"}, {"url": "u2", "text": "E2"}],
        "iterations": 0,
    }
    result = main._verify(state)
    assert len(result["claims"]) == 3
    assert sum(1 for c in result["claims"] if c["verified"]) == 2
    assert result["unverified"] == ["fact two"]


def test_verify_skipped_when_disabled(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_VERIFY", False)
    result = main._verify({"question": "q", "answer": "a", "evidence": [], "iterations": 0})
    assert result == {"claims": [], "unverified": []}


def test_after_verify_iterates_when_unverified_and_budget_remaining(patched):
    assert main._after_verify({"unverified": ["claim"], "iterations": 1}) == "search"


def test_after_verify_ends_when_budget_exhausted(patched, monkeypatch):
    monkeypatch.setattr(main, "MAX_ITERATIONS", 2)
    assert main._after_verify({"unverified": ["claim"], "iterations": 2}) is main.END


def test_after_verify_ends_when_all_verified(patched):
    assert main._after_verify({"unverified": [], "iterations": 0}) is main.END


def test_search_appends_on_iteration_without_duplicating(patched):
    state = {
        "question": "q",
        "subqueries": ["original sub"],
        "unverified": ["follow-up claim"],
        "evidence": [{"url": "https://a.example/1-1", "title": "A1", "text": "old"}],
        "iterations": 1,
    }
    result = main._search(state)
    assert any(e["text"] == "old" for e in result["evidence"])
    assert len(result["evidence"]) >= 2


def test_grounding_score_counts_valid_refs():
    ev = [{"url": "a"}, {"url": "b"}, {"url": "c"}]
    assert main._grounding_score("claim [1] and [2]", ev) == pytest.approx(2 ** 0.5)
    assert main._grounding_score("claim [1] and [7]", ev) == pytest.approx(0.5)
    assert main._grounding_score("no citations", ev) == 0.0
    assert main._grounding_score("a [1][2][3]", ev) > main._grounding_score("b [1]", ev)


def test_synthesize_consistency_picks_best_grounded(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_CONSISTENCY", True)
    monkeypatch.setattr(main, "CONSISTENCY_SAMPLES", 3)
    # Disable FLARE so _flare_augment doesn't interfere in the pick.
    monkeypatch.setattr(main, "ENABLE_ACTIVE_RETR", False)
    candidates = iter(["weak", "ok [1]", "best [1][2][3]"])
    monkeypatch.setattr(main, "_synthesize_once", lambda state: next(candidates))
    result = main._synthesize({"question": "q", "evidence": [{"url": "u1"}, {"url": "u2"}, {"url": "u3"}]})
    assert result["answer"] == "best [1][2][3]"


# ── T4.1 — step-level critic ──────────────────────────────────────────

def test_critic_accepts_when_disabled(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_STEP_VERIFY", False)
    accept, fb = main._critic("plan", "anything", "ctx")
    assert accept is True and fb == ""


def test_critic_parses_accept_verdict(patched):
    accept, fb = main._critic("plan", "sub one\nsub two\nsub three", "a question")
    assert accept is True


def test_critic_parses_redo_verdict(patched, monkeypatch):
    # Override router to return a redo verdict for this single test.
    def redo_router(*args, **kwargs):
        return _chat_resp("VERDICT: redo\nFEEDBACK: too vague")

    client = mock.MagicMock()
    client.chat.completions.create.side_effect = redo_router
    monkeypatch.setattr(main, "OpenAI", mock.MagicMock(return_value=client))
    accept, fb = main._critic("plan", "fuzzy", "ctx")
    assert accept is False
    assert "too vague" in fb


# ── T4.2 — FLARE active retrieval ─────────────────────────────────────

def test_flare_no_op_when_no_hedge_in_draft(patched):
    state = {"question": "q", "evidence": [{"url": "u1", "text": "E1"}]}
    out = main._flare_augment(state, "A confident answer [1].")
    assert out == "A confident answer [1]."


def test_flare_augments_on_hedged_draft(patched, monkeypatch):
    state = {
        "question": "q",
        "evidence": [{"url": "https://seen.example/1", "text": "E"}],
    }
    draft = "Answer [1]. The evidence does not specify the exact number."
    # When FLARE triggers it calls _search_one, which hits our faked SearXNG.
    out = main._flare_augment(state, draft)
    # On re-generation the mocked chat_router returns "Final answer..."
    assert "Final answer" in out


def test_flare_disabled_returns_draft_unchanged(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_ACTIVE_RETR", False)
    draft = "hedged answer — the evidence does not specify."
    out = main._flare_augment({"question": "q", "evidence": []}, draft)
    assert out == draft


# ── T4.3 — question classifier ────────────────────────────────────────

def test_classify_returns_label(patched):
    result = main._classify({"question": "Why is the sky blue?"})
    assert result["question_class"] == "multihop"


def test_classify_pass_through_when_disabled(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_ROUTER", False)
    result = main._classify({"question": "anything"})
    assert result["question_class"] == "multihop"


def test_classify_handles_garbled_label(patched, monkeypatch):
    # Router returns nonsense → falls back to multihop.
    def garbage_router(*args, **kwargs):
        p = kwargs.get("messages", [{}])[0].get("content", "")
        if "Classify this research question" in p:
            return _chat_resp("bananas and frogs")
        return _chat_resp("other")

    client = mock.MagicMock()
    client.chat.completions.create.side_effect = garbage_router
    monkeypatch.setattr(main, "OpenAI", mock.MagicMock(return_value=client))
    result = main._classify({"question": "q"})
    assert result["question_class"] == "multihop"


# ── T4.4 — evidence compression ───────────────────────────────────────

def test_compress_produces_compressed_view(patched):
    state = {
        "question": "q",
        "evidence": [
            {"url": "u1", "text": "long original A"},
            {"url": "u2", "text": "long original B"},
        ],
    }
    result = main._compress(state)
    comp = result["evidence_compressed"]
    assert len(comp) == 2
    assert comp[0]["text"] == "compressed A"
    assert comp[0]["url"] == "u1"  # URL preserved for citations


def test_compress_pass_through_when_disabled(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_COMPRESS", False)
    state = {"question": "q", "evidence": [{"url": "u", "text": "original"}]}
    result = main._compress(state)
    assert result["evidence_compressed"][0]["text"] == "original"


def test_compress_empty_evidence_returns_empty(patched):
    result = main._compress({"question": "q", "evidence": []})
    assert result["evidence_compressed"] == []


# ── T4.5 — plan refinement ────────────────────────────────────────────

def test_plan_refinement_triggers_once_on_reject(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_PLAN_REFINE", True)
    # Critic rejects on first call, accepts on second.
    critic_calls = {"n": 0}

    def mixed_critic(step, payload, ctx):
        critic_calls["n"] += 1
        return (critic_calls["n"] > 1, "too vague")

    monkeypatch.setattr(main, "_critic", mixed_critic)
    state = {"question": "Why does contextual retrieval improve recall?", "iterations": 0, "question_class": "multihop"}
    result = main._plan(state)
    assert result["plan_rejects"] == 1


def test_plan_refinement_skipped_when_disabled(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_PLAN_REFINE", False)
    monkeypatch.setattr(main, "_critic", lambda step, payload, ctx: (False, "fuzzy"))
    state = {"question": "q", "iterations": 0, "question_class": "multihop"}
    result = main._plan(state)
    assert result["plan_rejects"] == 0


# ── Full-graph integration ────────────────────────────────────────────

def test_full_graph_with_iteration(patched, monkeypatch):
    """Classify → plan → search → retrieve → compress → synthesize → verify."""
    monkeypatch.setattr(main, "MAX_ITERATIONS", 1)
    monkeypatch.setattr(main, "ENABLE_CONSISTENCY", False)
    monkeypatch.setattr(main, "ENABLE_ACTIVE_RETR", False)  # keep the graph linear for this test
    graph = main.build_graph()
    result = graph.invoke({"question": "Why does contextual retrieval improve recall?", "iterations": 0, "plan_rejects": 0})
    assert "Final answer" in result["answer"]
    assert result["question_class"] == "multihop"
    assert result.get("iterations", 0) >= 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
