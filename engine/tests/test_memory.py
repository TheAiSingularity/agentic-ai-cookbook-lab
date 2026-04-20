"""Mocked tests for engine.core.memory — no network, no API keys.

Uses a deterministic fake embedder so cosine similarity is predictable.
Covers: trajectory serialization, session-vs-persistent storage, reset,
retrieval ordering, domain filtering, and the `summarize_hits` helper.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engine.core.memory import (  # noqa: E402
    MemoryStore,
    Trajectory,
    summarize_hits,
    _NullStore,
)


def fake_embedder(batch):
    """Length + word count + vowel count — deterministic for tests."""
    return [
        [float(len(s)), float(len(s.split())),
         float(sum(1 for c in s.lower() if c in "aeiou"))]
        for s in batch
    ]


def _mk(question: str, *, answer: str = "A", domain: str = "general", verified: int = 1):
    return Trajectory(
        query_id=f"q-{question}",
        timestamp=time.time(),
        question=question,
        domain=domain,
        final_answer=answer,
        verified_claims=[{"text": "c", "verified": True}] * verified,
        unverified_claims=[],
        evidence_urls=["https://a.example/"],
        iterations=1,
        question_class="factoid",
        tokens_est=100,
        latency_s=1.2,
    )


# ── Trajectory serialization ─────────────────────────────────────────

def test_trajectory_to_json_roundtrips():
    import json
    t = _mk("hello")
    decoded = json.loads(t.to_json())
    assert decoded["question"] == "hello"
    assert decoded["domain"] == "general"
    assert decoded["iterations"] == 1


def test_trajectory_from_state_computes_tokens_and_latency():
    state = {
        "question": "q",
        "answer": "A",
        "claims": [{"text": "c1", "verified": True}, {"text": "c2", "verified": False}],
        "unverified": ["c2"],
        "evidence": [{"url": "u1"}, {"url": "u2"}],
        "iterations": 2,
        "question_class": "multihop",
        "trace": [
            {"model": "m", "latency_s": 1.5, "tokens_est": 200},
            {"model": "m", "latency_s": 0.5, "tokens_est": 100},
        ],
    }
    t = Trajectory.from_state(state, domain="medical")
    assert t.domain == "medical"
    assert t.tokens_est == 300
    assert t.latency_s == 2.0
    assert len(t.verified_claims) == 1
    assert t.unverified_claims == ["c2"]


# ── Factory modes ────────────────────────────────────────────────────

def test_open_off_returns_null_store():
    store = MemoryStore.open("off")
    assert isinstance(store, _NullStore)
    store.record(_mk("x"))
    assert store.retrieve("x") == []
    assert store.count() == 0


def test_open_session_is_in_memory_only():
    store = MemoryStore.open("session", embedder=fake_embedder)
    assert store._conn is None
    store.record(_mk("hello"))
    assert store.count() == 1


def test_open_persistent_creates_db(tmp_path):
    path = tmp_path / "mem.db"
    store = MemoryStore.open("persistent", path=path, embedder=fake_embedder)
    assert path.exists()
    store.close()


def test_open_rejects_invalid_mode():
    with pytest.raises(ValueError):
        MemoryStore.open("bogus")


# ── Record + retrieve ────────────────────────────────────────────────

def test_session_record_then_retrieve_returns_hits(tmp_path):
    store = MemoryStore.open("session", embedder=fake_embedder)
    store.record(_mk("what is rag"))
    store.record(_mk("what is bm25"))
    hits = store.retrieve("what is rag indexing", k=2, min_score=0.0)
    assert len(hits) <= 2
    # Cosine on fake_embedder's 3-d vectors: similar questions score high.
    assert hits[0][1] >= hits[-1][1]


def test_persistent_record_then_retrieve_survives_close_reopen(tmp_path):
    path = tmp_path / "mem.db"
    store1 = MemoryStore.open("persistent", path=path, embedder=fake_embedder)
    store1.record(_mk("first question"))
    store1.record(_mk("second question"))
    assert store1.count() == 2
    store1.close()

    store2 = MemoryStore.open("persistent", path=path, embedder=fake_embedder)
    assert store2.count() == 2
    hits = store2.retrieve("first question", k=3, min_score=0.0)
    assert hits[0][0].question == "first question"


def test_retrieve_respects_min_score(tmp_path):
    store = MemoryStore.open("session", embedder=fake_embedder)
    store.record(_mk("apples and oranges"))
    # A wildly different query with min_score=0.99 should return no hits.
    hits = store.retrieve("quantum chromodynamics", k=5, min_score=0.999)
    assert hits == []


def test_retrieve_filters_by_domain(tmp_path):
    store = MemoryStore.open("session", embedder=fake_embedder)
    store.record(_mk("medical q", domain="medical"))
    store.record(_mk("financial q", domain="financial"))
    hits_med = store.retrieve("some question", k=5, min_score=0.0, domain="medical")
    assert all(t.domain == "medical" for t, _ in hits_med)
    assert len(hits_med) == 1


def test_retrieve_with_k_zero_returns_empty():
    store = MemoryStore.open("session", embedder=fake_embedder)
    store.record(_mk("x"))
    assert store.retrieve("x", k=0) == []


# ── Reset ────────────────────────────────────────────────────────────

def test_reset_wipes_persistent_store(tmp_path):
    path = tmp_path / "mem.db"
    store = MemoryStore.open("persistent", path=path, embedder=fake_embedder)
    store.record(_mk("a"))
    store.record(_mk("b"))
    assert store.count() == 2
    store.reset()
    assert store.count() == 0
    store.close()


def test_reset_wipes_session_store():
    store = MemoryStore.open("session", embedder=fake_embedder)
    store.record(_mk("a"))
    store.reset()
    assert store.count() == 0


# ── Summaries ───────────────────────────────────────────────────────

def test_summarize_hits_empty_returns_empty_string():
    assert summarize_hits([]) == ""


def test_summarize_hits_caps_preview_length():
    t = _mk("long question", answer="x" * 500)
    rendered = summarize_hits([(t, 0.9)], max_chars=50)
    assert "long question" in rendered
    # Preview truncated with an ellipsis.
    assert "…" in rendered


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
