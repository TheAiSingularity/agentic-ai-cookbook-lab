"""engine.core.compaction — context-window compaction.

Phase 2 of the master plan. When evidence accumulates beyond a configured
threshold (default ~24 k chars for 4 B models), this module collapses older
chunks into 1-sentence summaries while preserving URLs and any claims that
were verified by CoVe in the current pipeline run.

The goal is to give small-context models (Gemma 3 4 B has an 8 k effective
reliable context) a chance to stay on-topic without losing cited sources.
The compactor is opt-in via `ENABLE_COMPACTION=1`; when off, behavior is
unchanged.

Design decisions:
- Operate on `state["evidence_compressed"]` after T4.4 compress ran. If
  that list is absent, fall back to `state["evidence"]`.
- Preserve items whose URLs appear in any verified claim ("load-bearing").
- Compact the remaining items using a cheap summarizer LLM call.
- Always keep the most recent `ENABLE_COMPACTION_KEEP_RECENT` items
  intact.
- Emit a trace entry so the user can see what was dropped.

Kept deliberately small (~130 LOC) so it composes cleanly with W4 + W6 +
W7 rather than duplicating their logic.
"""

from __future__ import annotations

import os
from typing import Callable

ENV = os.environ.get

ENABLE_COMPACTION = ENV("ENABLE_COMPACTION", "1") == "1"
CONTEXT_LIMIT_CHARS = int(ENV("CONTEXT_LIMIT_CHARS", "24000"))
KEEP_RECENT = int(ENV("COMPACTION_KEEP_RECENT", "3"))
COMPACTION_SUMMARY_CHARS = int(ENV("COMPACTION_SUMMARY_CHARS", "200"))


def evidence_char_total(evidence: list[dict]) -> int:
    """Total character count across evidence text fields."""
    return sum(len(e.get("text", "")) for e in (evidence or []))


def should_compact(evidence: list[dict], limit_chars: int | None = None) -> bool:
    """Return True when the accumulated evidence char total exceeds the limit."""
    limit = CONTEXT_LIMIT_CHARS if limit_chars is None else limit_chars
    return evidence_char_total(evidence) > limit


def _load_bearing_urls(claims: list[dict] | None) -> set[str]:
    """URLs of evidence items that appear in CoVe-verified claims' citation lists.

    For v1 we use a simple heuristic: any verified claim whose text
    contains a `corpus://` or http URL marks that URL as load-bearing.
    """
    if not claims:
        return set()
    urls: set[str] = set()
    for c in claims:
        if not c.get("verified"):
            continue
        text = c.get("text", "")
        for token in text.split():
            tok = token.strip("(),.;:'\"")
            if tok.startswith("http://") or tok.startswith("https://") or tok.startswith("corpus://"):
                urls.add(tok)
    return urls


def compact(
    evidence: list[dict],
    question: str,
    *,
    summarizer: Callable[[str], str],
    claims: list[dict] | None = None,
    keep_recent: int = KEEP_RECENT,
    summary_chars: int = COMPACTION_SUMMARY_CHARS,
) -> tuple[list[dict], dict]:
    """Return (compacted_evidence, trace_stats).

    `summarizer(prompt_text) -> summary_text` is an LLM callable; most
    callers will pass a closure wrapping `engine.core.models._chat`.

    Strategy:
    - Keep every item whose URL is load-bearing (cited by a verified claim).
    - Keep the most recent `keep_recent` items intact.
    - For the rest, run ONE summarizer call over the concatenated block to
      produce per-item one-liners; attach summaries as `text` while keeping
      URLs and titles. Bounded by `summary_chars` to guarantee shrinkage.
    """
    if not evidence:
        return [], {"n_in": 0, "n_compacted": 0, "n_kept": 0, "chars_before": 0, "chars_after": 0}

    load_bearing = _load_bearing_urls(claims)
    n = len(evidence)
    recent_cutoff = max(0, n - keep_recent)

    to_compact: list[tuple[int, dict]] = []
    kept: list[tuple[int, dict]] = []

    for i, e in enumerate(evidence):
        if i >= recent_cutoff or e.get("url") in load_bearing:
            kept.append((i, e))
        else:
            to_compact.append((i, e))

    chars_before = evidence_char_total(evidence)

    if not to_compact:
        return list(evidence), {
            "n_in": n, "n_compacted": 0, "n_kept": n,
            "chars_before": chars_before, "chars_after": chars_before,
        }

    # One summarizer call for the whole block — emit "[pos] <summary>" per line.
    # `pos` is the 1-based position within the to_compact list so the parser
    # below can look up summaries by enumerate-index regardless of where each
    # chunk sits in the full evidence list.
    bullets = "\n\n".join(
        f"[{pos}] (url: {e.get('url','')}) {e.get('text','')}"
        for pos, (_orig_i, e) in enumerate(to_compact, start=1)
    )
    prompt = (
        f"Summarize each of the numbered evidence chunks below in ONE short "
        f"sentence (max {summary_chars} chars) focused on the question. "
        f"Preserve the bracket indices. Output each as `[N] <summary>` on its own line.\n\n"
        f"Question: {question}\n\nChunks:\n{bullets}"
    )
    raw = summarizer(prompt)

    # Parse `[N] text` lines.
    summary_by_idx: dict[int, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("["):
            continue
        try:
            rbr = line.index("]")
            idx = int(line[1:rbr])
            summary_by_idx[idx] = line[rbr + 1:].strip()[:summary_chars]
        except (ValueError, IndexError):
            continue

    out: list[dict] = [None] * n  # type: ignore[list-item]
    for i, e in kept:
        out[i] = e
    for k, (i, e) in enumerate(to_compact, start=1):
        summary = summary_by_idx.get(k) or e.get("text", "")[:summary_chars]
        out[i] = {**e, "text": summary, "compacted": True}

    assert all(x is not None for x in out)
    chars_after = evidence_char_total(out)  # type: ignore[arg-type]
    return out, {  # type: ignore[return-value]
        "n_in": n,
        "n_compacted": len(to_compact),
        "n_kept": len(kept),
        "chars_before": chars_before,
        "chars_after": chars_after,
    }


__all__ = [
    "ENABLE_COMPACTION",
    "CONTEXT_LIMIT_CHARS",
    "KEEP_RECENT",
    "COMPACTION_SUMMARY_CHARS",
    "evidence_char_total",
    "should_compact",
    "compact",
]
