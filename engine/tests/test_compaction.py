"""Mocked tests for engine.core.compaction — no network, no API keys.

The summarizer callable is a fake function that returns deterministic
summaries so compaction outputs are inspectable without invoking any LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from engine.core.compaction import (  # noqa: E402
    compact,
    evidence_char_total,
    should_compact,
)


def _fake_summarizer(prompt: str) -> str:
    """Emit `[N] summary N` per chunk seen in the prompt."""
    lines = []
    n = 1
    while f"[{n}]" in prompt:
        lines.append(f"[{n}] summary {n}")
        n += 1
    return "\n".join(lines)


def _mk(url: str, text: str) -> dict:
    return {"url": url, "title": url, "text": text}


# ── evidence_char_total + should_compact ─────────────────────────────

def test_evidence_char_total_sums_text_lengths():
    assert evidence_char_total([]) == 0
    assert evidence_char_total([_mk("u", "12345")]) == 5
    assert evidence_char_total([_mk("u1", "aaa"), _mk("u2", "bbbbb")]) == 8


def test_should_compact_trips_at_threshold():
    e = [_mk(f"u{i}", "x" * 1000) for i in range(10)]  # 10_000 chars
    assert should_compact(e, limit_chars=9999) is True
    assert should_compact(e, limit_chars=10001) is False


def test_should_compact_respects_env_default():
    # Small evidence: never compacts under the default 24k limit.
    assert should_compact([_mk("u", "x" * 500)]) is False


# ── compact() — no-op paths ──────────────────────────────────────────

def test_compact_empty_evidence_returns_empty_stats():
    out, stats = compact([], "q", summarizer=_fake_summarizer)
    assert out == []
    assert stats["n_in"] == 0 and stats["n_compacted"] == 0


def test_compact_keeps_recent_items_intact():
    # 10 items, keep_recent=3 → items 7,8,9 untouched.
    ev = [_mk(f"u{i}", f"text-of-item-{i}") for i in range(10)]
    out, stats = compact(ev, "q", summarizer=_fake_summarizer, keep_recent=3, summary_chars=50)
    assert len(out) == 10
    for i in (7, 8, 9):
        assert out[i]["text"] == f"text-of-item-{i}"
    # Items 0-6 got compacted to "summary N" strings.
    for i in range(7):
        assert "summary" in out[i]["text"]
        assert out[i].get("compacted") is True


def test_compact_preserves_load_bearing_urls():
    ev = [
        _mk("https://verified.example/1", "original A"),
        _mk("https://other.example/2", "original B"),
        _mk("https://other.example/3", "original C"),
    ]
    claims = [
        {"text": "Per https://verified.example/1 the answer is X.", "verified": True},
        {"text": "Unverified claim on https://other.example/2.", "verified": False},
    ]
    # keep_recent=0 so only load-bearing preservation applies.
    out, stats = compact(ev, "q", summarizer=_fake_summarizer, claims=claims, keep_recent=0)
    assert out[0]["text"] == "original A"  # kept (verified URL)
    assert "summary" in out[1]["text"]     # compacted (only unverified cite)
    assert "summary" in out[2]["text"]     # compacted


def test_compact_urls_preserved_on_compacted_items():
    ev = [_mk(f"u{i}", f"text {i}") for i in range(5)]
    out, _ = compact(ev, "q", summarizer=_fake_summarizer, keep_recent=0)
    assert [o["url"] for o in out] == [f"u{i}" for i in range(5)]


def test_compact_stats_report_shrinkage():
    ev = [_mk(f"u{i}", "x" * 1000) for i in range(5)]
    out, stats = compact(ev, "q", summarizer=_fake_summarizer, keep_recent=0, summary_chars=50)
    assert stats["n_in"] == 5
    assert stats["n_compacted"] == 5
    assert stats["n_kept"] == 0
    assert stats["chars_before"] > stats["chars_after"]


# ── compact() — summarizer parsing ────────────────────────────────────

def test_compact_handles_malformed_summary_lines():
    def bad_summarizer(_prompt):
        return "[1] good one\ngarbled line\n[2]\n[3] good three"

    ev = [_mk(f"u{i}", f"long text {i}" * 100) for i in range(3)]
    out, _ = compact(ev, "q", summarizer=bad_summarizer, keep_recent=0, summary_chars=50)
    # [1] and [3] parsed fine; [2] had no summary → falls back to truncated original.
    assert "good one" in out[0]["text"]
    assert out[1]["text"] != "" and len(out[1]["text"]) <= 50
    assert "good three" in out[2]["text"]


def test_compact_uses_summarizer_only_once():
    calls = []

    def counting_summarizer(prompt):
        calls.append(prompt)
        return "\n".join(f"[{i+1}] summary" for i in range(10))

    ev = [_mk(f"u{i}", f"text {i}") for i in range(10)]
    _, _ = compact(ev, "q", summarizer=counting_summarizer, keep_recent=0)
    assert len(calls) == 1  # one batched call regardless of chunk count


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
