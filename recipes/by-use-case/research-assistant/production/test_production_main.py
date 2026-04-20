"""Mocked tests for research-assistant/production (Wave 2 Tiers 2+4 + Wave 4).

Covers: HyDE gating, CoVe parsing, conditional iteration, self-consistency,
step-level critic (T4.1), FLARE active retrieval (T4.2), question classifier
router (T4.3), evidence compression (T4.4), plan refinement (T4.5), plus the
Wave 4 local-first engine enhancements: cross-encoder rerank wiring (W4.1),
trafilatura fetch_url node (W4.2), and observability trace (W4.3).

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
        # Synthesizer (W6 anti-hallucination clause)
        if "Answer the question using ONLY the evidence" in p:
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

    # W4.2 — disable network-touching fetch by default; individual tests that
    # want to exercise the node flip ENABLE_FETCH back on and stub _fetch_one.
    monkeypatch.setattr(main, "ENABLE_FETCH", False)
    # W4.3 — keep the trace buffer clean between tests.
    main._TRACE_BUFFER.clear()

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
    assert result["claims"] == [] and result["unverified"] == []
    assert "trace" in result


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


# ── W4.1 — cross-encoder rerank wiring ────────────────────────────────

def _ten_evidence() -> list[dict]:
    return [{"url": f"u{i}", "text": f"passage {i} about topic"} for i in range(10)]


def test_retrieve_passthrough_when_evidence_below_topk(patched):
    ev = [{"url": f"u{i}", "text": f"t{i}"} for i in range(3)]
    result = main._retrieve({"question": "q", "evidence": ev})
    assert result["evidence"] == ev


def test_retrieve_uses_hybrid_only_when_rerank_disabled(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_RERANK", False)
    # Flag if the reranker gets constructed (it must NOT when disabled).
    sentinel = {"called": False}

    def boom(*a, **k):
        sentinel["called"] = True
        raise AssertionError("reranker should not be instantiated when ENABLE_RERANK=0")

    monkeypatch.setattr(main, "_get_reranker", boom)
    result = main._retrieve({"question": "topic", "evidence": _ten_evidence()})
    assert sentinel["called"] is False
    assert len(result["evidence"]) == main.TOP_K_EVIDENCE


def test_retrieve_reranks_when_enabled(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_RERANK", True)

    class FakeReranker:
        def rerank(self, query, candidates, k):
            # Return the last k passages, scored in reverse order — different
            # from hybrid's ordering so we can assert rerank actually ran.
            passages = [c[0] if isinstance(c, tuple) else c for c in candidates]
            tail = passages[-k:]
            return [(p, float(len(tail) - i)) for i, p in enumerate(tail)]

    monkeypatch.setattr(main, "_get_reranker", lambda: FakeReranker())
    ev = _ten_evidence()
    result = main._retrieve({"question": "topic", "evidence": ev})
    kept_texts = [e["text"] for e in result["evidence"]]
    # With TOP_K_EVIDENCE=8 and our fake returning the last 8 of 10 candidates,
    # passages 0 and 1 should have been filtered out by the reranker.
    assert "passage 0 about topic" not in kept_texts
    assert len(kept_texts) == main.TOP_K_EVIDENCE


def test_retrieve_falls_back_when_reranker_raises(patched, monkeypatch, capsys):
    monkeypatch.setattr(main, "ENABLE_RERANK", True)

    class BrokenReranker:
        def rerank(self, *a, **k):
            raise RuntimeError("model download failed")

    monkeypatch.setattr(main, "_get_reranker", lambda: BrokenReranker())
    ev = _ten_evidence()
    result = main._retrieve({"question": "topic", "evidence": ev})
    assert len(result["evidence"]) == main.TOP_K_EVIDENCE
    assert "falling back to hybrid-only" in capsys.readouterr().err


# ── W4.2 — fetch_url / trafilatura ────────────────────────────────────

def test_fetch_url_passthrough_when_disabled(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_FETCH", False)
    ev = [{"url": "https://a.example/1", "text": "snippet"}]
    result = main._fetch_url({"question": "q", "evidence": ev})
    # Disabled node is a no-op on evidence; only trace is emitted.
    assert "evidence" not in result
    assert "trace" in result


def test_fetch_url_replaces_text_on_successful_fetch(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_FETCH", True)
    monkeypatch.setattr(main, "_fetch_one", lambda url: f"FULL TEXT FOR {url}")
    ev = [
        {"url": "https://a.example/1", "text": "snippet A"},
        {"url": "https://b.example/2", "text": "snippet B"},
    ]
    result = main._fetch_url({"question": "q", "evidence": ev})
    assert result["evidence"][0]["text"] == "FULL TEXT FOR https://a.example/1"
    assert result["evidence"][0]["fetched"] is True
    assert result["evidence"][1]["fetched"] is True


def test_fetch_url_keeps_snippet_when_fetch_fails(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_FETCH", True)
    monkeypatch.setattr(main, "_fetch_one", lambda url: None)
    ev = [{"url": "https://dead.example/1", "text": "original snippet"}]
    result = main._fetch_url({"question": "q", "evidence": ev})
    assert result["evidence"][0]["text"] == "original snippet"
    assert result["evidence"][0]["fetched"] is False


def test_fetch_url_respects_max_urls_cap(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_FETCH", True)
    monkeypatch.setattr(main, "FETCH_MAX_URLS", 2)
    called: list[str] = []

    def track(url):
        called.append(url)
        return f"full({url})"

    monkeypatch.setattr(main, "_fetch_one", track)
    ev = [{"url": f"u{i}", "text": f"s{i}"} for i in range(5)]
    result = main._fetch_url({"question": "q", "evidence": ev})
    assert len(called) == 2  # only first 2 fetched
    assert result["evidence"][0]["fetched"] is True
    assert result["evidence"][1]["fetched"] is True
    # Remaining 3 are preserved as-is (beyond the cap).
    assert len(result["evidence"]) == 5
    assert "fetched" not in result["evidence"][2]


def test_fetch_url_empty_evidence_returns_trace_only(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_FETCH", True)
    result = main._fetch_url({"question": "q", "evidence": []})
    assert "evidence" not in result
    assert isinstance(result["trace"], list)


# ── W4.3 — observability trace ────────────────────────────────────────

def test_chat_appends_trace_entry_when_enabled(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_TRACE", True)
    main._TRACE_BUFFER.clear()
    _ = main._chat("test-model", "hello world")
    assert len(main._TRACE_BUFFER) == 1
    entry = main._TRACE_BUFFER[0]
    assert entry["model"] == "test-model"
    assert entry["prompt_chars"] == len("hello world")
    assert entry["response_chars"] > 0
    assert entry["latency_s"] >= 0.0
    assert entry["tokens_est"] > 0


def test_chat_skips_trace_when_disabled(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_TRACE", False)
    main._TRACE_BUFFER.clear()
    _ = main._chat("test-model", "hi")
    assert main._TRACE_BUFFER == []


def test_drain_trace_tags_node_and_clears_buffer(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_TRACE", True)
    main._TRACE_BUFFER.clear()
    main._chat("m1", "p1")
    main._chat("m2", "p2")
    drained = main._drain_trace("plan")
    assert len(drained) == 2
    assert all(e["node"] == "plan" for e in drained)
    assert main._TRACE_BUFFER == []


def test_merge_trace_appends_extras(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_TRACE", True)
    main._TRACE_BUFFER.clear()
    main._chat("m1", "p1")
    merged = main._merge_trace(
        {"trace": [{"node": "prior", "model": "x"}]},
        "retrieve",
        extras=[{"model": "hybrid", "latency_s": 0.1, "tokens_est": 0}],
    )
    assert merged[0]["node"] == "prior"              # existing preserved
    assert merged[1]["node"] == "retrieve"           # drained LLM call
    assert merged[2]["node"] == "retrieve"           # extras tagged too
    assert merged[2]["model"] == "hybrid"


def test_classify_contributes_trace(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_TRACE", True)
    result = main._classify({"question": "q", "trace": []})
    assert any(e["node"] == "classify" for e in result["trace"])


def test_print_trace_summary_handles_empty_and_populated(capsys):
    main._print_trace_summary([])
    out = capsys.readouterr().out
    assert out == ""  # empty trace → no output
    main._print_trace_summary([
        {"node": "plan", "model": "m1", "latency_s": 0.5, "tokens_est": 100},
        {"node": "synthesize", "model": "m2", "latency_s": 1.2, "tokens_est": 500},
    ])
    out = capsys.readouterr().out
    assert "trace summary" in out
    assert "plan" in out and "synthesize" in out
    assert "m1" in out and "m2" in out


# ── W6 — small-model hardening ───────────────────────────────────────

def test_small_model_heuristic_matches_ollama_patterns():
    assert main._SMALL_MODEL_RE.search("gemma4:e2b")
    assert main._SMALL_MODEL_RE.search("qwen2.5:3b")
    assert main._SMALL_MODEL_RE.search("tinyllama:1b")
    assert main._SMALL_MODEL_RE.search("gpt-5-nano")
    assert main._SMALL_MODEL_RE.search("llama-3b-instruct")


def test_small_model_heuristic_does_not_match_capable_models():
    assert not main._SMALL_MODEL_RE.search("gpt-5-mini")
    assert not main._SMALL_MODEL_RE.search("gpt-4o-mini")
    assert not main._SMALL_MODEL_RE.search("Qwen/Qwen3.6-35B-A3B")
    assert not main._SMALL_MODEL_RE.search("claude-4.6-opus")
    assert not main._SMALL_MODEL_RE.search("gpt-5")


def test_default_top_k_respects_explicit_override():
    # Explicit env wins regardless of model name.
    assert main._default_top_k("gemma4:e2b", "10") == 10
    assert main._default_top_k("gpt-5-mini", "3") == 3


def test_default_top_k_shrinks_for_small_models():
    # Small model + no explicit → SMALL_MODEL_TOPK (default 5).
    assert main._default_top_k("gemma4:e2b", None) == 5
    assert main._default_top_k("gpt-5-nano", None) == 5


def test_default_top_k_keeps_8_for_capable_models():
    assert main._default_top_k("gpt-5-mini", None) == 8
    assert main._default_top_k("Qwen/Qwen3.6-35B-A3B", None) == 8


def test_compress_applies_per_chunk_cap_after_compression(patched, monkeypatch):
    monkeypatch.setattr(main, "PER_CHUNK_CHAR_CAP", 20)
    # Make compressor return long chunks exceeding the cap.
    def long_compressor(*args, **kwargs):
        return _chat_resp("[1] " + "A" * 200 + "\n[2] " + "B" * 50)

    client = mock.MagicMock()
    client.chat.completions.create.side_effect = long_compressor
    monkeypatch.setattr(main, "OpenAI", mock.MagicMock(return_value=client))

    state = {"question": "q", "evidence": [
        {"url": "u1", "text": "orig1"},
        {"url": "u2", "text": "orig2"},
    ]}
    result = main._compress(state)
    # Both chunks capped at 20 chars.
    assert all(len(c["text"]) <= 20 for c in result["evidence_compressed"])


def test_compress_caps_passthrough_chunks_when_disabled(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_COMPRESS", False)
    monkeypatch.setattr(main, "PER_CHUNK_CHAR_CAP", 15)
    state = {"question": "q", "evidence": [
        {"url": "u1", "text": "A" * 100},       # over cap → truncated
        {"url": "u2", "text": "short"},         # under cap → passthrough
    ]}
    result = main._compress(state)
    comp = result["evidence_compressed"]
    assert len(comp[0]["text"]) == 15
    assert comp[1]["text"] == "short"


def test_synthesize_prompt_includes_anti_hallucination_clause(patched, monkeypatch):
    captured: dict[str, str] = {}

    def capture(*args, **kwargs):
        captured["prompt"] = kwargs.get("messages", [{}])[0].get("content", "")
        return _chat_resp("Final answer [1].")

    client = mock.MagicMock()
    client.chat.completions.create.side_effect = capture
    monkeypatch.setattr(main, "OpenAI", mock.MagicMock(return_value=client))

    state = {"question": "q", "evidence_compressed": [{"url": "u", "text": "ev"}]}
    main._synthesize_once(state)
    p = captured["prompt"]
    # W6 refined clause covers three cases: full / partial / unrelated.
    assert "FULLY answers" in p
    assert "partially answers" in p
    assert "UNRELATED" in p
    assert "Never invent facts" in p
    assert "Never substitute a related topic" in p


# ── W5.1 — local corpus augmentation ─────────────────────────────────

def _make_fake_corpus(monkeypatch, chunks: list[dict]) -> None:
    """Install a fake corpus singleton that returns the given chunks on query()."""
    from core.rag import CorpusChunk

    class FakeIndex:
        def __init__(self, items):
            self.chunks = [
                CorpusChunk(text=c["text"], source=c["source"],
                            page=c.get("page"), chunk_idx=c.get("chunk_idx", 0))
                for c in items
            ]

        def query(self, q, k=5):
            return [(c, 1.0) for c in self.chunks[:k]]

    monkeypatch.setattr(main, "LOCAL_CORPUS_PATH", "/fake/corpus")
    monkeypatch.setattr(main, "_CORPUS", FakeIndex(chunks))
    monkeypatch.setattr(main, "_CORPUS_LOAD_FAILED", False)


def test_corpus_hits_returns_empty_when_path_unset(patched, monkeypatch):
    monkeypatch.setattr(main, "LOCAL_CORPUS_PATH", "")
    monkeypatch.setattr(main, "_CORPUS", None)
    assert main._corpus_hits("anything") == []


def test_corpus_hits_shapes_urls_with_page_and_chunk(patched, monkeypatch):
    _make_fake_corpus(monkeypatch, [
        {"text": "content A", "source": "paper.pdf", "page": 2, "chunk_idx": 7},
        {"text": "content B", "source": "notes.md", "page": None, "chunk_idx": 0},
    ])
    hits = main._corpus_hits("q", k=5)
    assert hits[0]["url"] == "corpus://paper.pdf#p2#c7"
    assert hits[0]["title"] == "paper.pdf (p2)"
    assert hits[0]["text"] == "content A"
    assert hits[1]["url"] == "corpus://notes.md#c0"
    assert hits[1]["title"] == "notes.md"


def test_corpus_hits_handles_query_failure(patched, monkeypatch, capsys):
    class BrokenIndex:
        chunks = []
        def query(self, *a, **k):
            raise RuntimeError("index corrupted")

    monkeypatch.setattr(main, "LOCAL_CORPUS_PATH", "/x")
    monkeypatch.setattr(main, "_CORPUS", BrokenIndex())
    monkeypatch.setattr(main, "_CORPUS_LOAD_FAILED", False)
    assert main._corpus_hits("q") == []
    assert "query failed" in capsys.readouterr().err


def test_get_corpus_caches_load_failure_and_falls_back(patched, monkeypatch, capsys):
    from core.rag import CorpusIndex

    monkeypatch.setattr(main, "LOCAL_CORPUS_PATH", "/nonexistent")
    monkeypatch.setattr(main, "_CORPUS", None)
    monkeypatch.setattr(main, "_CORPUS_LOAD_FAILED", False)

    call_count = {"n": 0}

    def failing_load(path, embedder=None):
        call_count["n"] += 1
        raise FileNotFoundError(str(path))

    monkeypatch.setattr(CorpusIndex, "load", classmethod(lambda cls, p, embedder=None: failing_load(p)))
    assert main._get_corpus() is None
    assert main._get_corpus() is None  # still None, but load not retried
    assert call_count["n"] == 1
    assert "load failed" in capsys.readouterr().err


def test_search_merges_corpus_hits_into_evidence(patched, monkeypatch):
    _make_fake_corpus(monkeypatch, [
        {"text": "local corpus fact", "source": "doc.md"},
    ])
    state = {"question": "q", "subqueries": ["sub one", "sub two"], "iterations": 0, "plan_rejects": 0}
    result = main._search(state)
    urls = [e["url"] for e in result["evidence"]]
    # Web hits come first (from the patched SearXNG), corpus hits appended.
    assert any(u.startswith("corpus://doc.md") for u in urls)
    # Trace records the corpus augmentation.
    assert any(e.get("model") == "corpus" for e in result["trace"])


def test_search_no_corpus_augmentation_when_unset(patched, monkeypatch):
    monkeypatch.setattr(main, "LOCAL_CORPUS_PATH", "")
    monkeypatch.setattr(main, "_CORPUS", None)
    state = {"question": "q", "subqueries": ["only sub"], "iterations": 0, "plan_rejects": 0}
    result = main._search(state)
    urls = [e["url"] for e in result["evidence"]]
    assert not any(u.startswith("corpus://") for u in urls)
    # No "corpus" entry in trace.
    assert not any(e.get("model") == "corpus" for e in result["trace"])


def test_fetch_one_skips_corpus_urls(patched):
    # Corpus URLs aren't fetchable; _fetch_one returns None so _fetch_url keeps
    # the existing text and flags fetched=False.
    assert main._fetch_one("corpus://paper.pdf#p0#c3") is None


def test_fetch_url_preserves_corpus_text(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_FETCH", True)
    ev = [
        {"url": "https://a.example/1", "text": "web snippet"},
        {"url": "corpus://paper.pdf#c0", "text": "full corpus chunk text"},
    ]
    # Make web fetches succeed; corpus fetches naturally return None via real _fetch_one.
    def fake_fetch(url):
        if url.startswith("corpus://"):
            return None
        return f"FULL({url})"
    monkeypatch.setattr(main, "_fetch_one", fake_fetch)
    result = main._fetch_url({"question": "q", "evidence": ev, "trace": []})
    by_url = {e["url"]: e for e in result["evidence"]}
    assert by_url["https://a.example/1"]["fetched"] is True
    assert by_url["corpus://paper.pdf#c0"]["fetched"] is False
    assert by_url["corpus://paper.pdf#c0"]["text"] == "full corpus chunk text"


# ── Full-graph integration ────────────────────────────────────────────

def test_full_graph_with_iteration(patched, monkeypatch):
    """Classify → plan → search → retrieve → fetch_url → compress → synthesize → verify."""
    monkeypatch.setattr(main, "MAX_ITERATIONS", 1)
    monkeypatch.setattr(main, "ENABLE_CONSISTENCY", False)
    monkeypatch.setattr(main, "ENABLE_ACTIVE_RETR", False)  # keep the graph linear for this test
    graph = main.build_graph()
    result = graph.invoke({"question": "Why does contextual retrieval improve recall?", "iterations": 0, "plan_rejects": 0, "trace": []})
    assert "Final answer" in result["answer"]
    assert result["question_class"] == "multihop"
    assert result.get("iterations", 0) >= 1


def test_full_graph_records_trace_across_nodes(patched, monkeypatch):
    monkeypatch.setattr(main, "ENABLE_TRACE", True)
    monkeypatch.setattr(main, "ENABLE_ACTIVE_RETR", False)
    graph = main.build_graph()
    result = graph.invoke({"question": "Why does contextual retrieval improve recall?", "iterations": 0, "plan_rejects": 0, "trace": []})
    nodes_in_trace = {e["node"] for e in result.get("trace", [])}
    assert {"classify", "plan", "search", "synthesize"}.issubset(nodes_in_trace)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
