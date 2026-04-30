"""engine.core.pipeline — the 8-node research pipeline.

Extracted verbatim (behavior-preserving) from the legacy monolith at
`recipes/by-use-case/research-assistant/production/main.py`. All env gates
preserved; all three 2026 technique families (Tier 2, Tier 4, Wave 4–7)
wired. Phase 2 will layer memory + compaction on top of this module
without touching node internals.

Design note (monkey-patch safety): LLM calls go through
`_chat(...)` (namespace lookup) rather than a directly-imported
`_chat`. That way tests can `monkeypatch.setattr(engine.core.models,
"_chat", fake)` and the pipeline picks it up. The same discipline applies
to `models.OpenAI` which tests mock in the `patched` fixture.

Graph:

    [T4.3 classify] → plan → [T4.1 critic] → search → [T4.1 critic] → retrieve (+W4.1 rerank)
                                                                             │
                                                            [W4.2 fetch_url] ┘
                                                                             │
                                                                             ▼
                                       [T4.4 compress + W6.2 cap] ◀──────────┘
                                              │
                                              ▼
                                        synthesize ◀── [T4.2 FLARE]
                                              │
                                              ▼
                                        verify (CoVe)
                                              │
                               verified? ──yes──▶ END
                                              │
                                              no
                                              │
                               iterate (re-search failed claims) ──▶ search
"""

from __future__ import annotations

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

import requests

# Ensure repo root on sys.path so `core.rag` imports cleanly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.rag import CorpusIndex, CrossEncoderReranker, HybridRetriever  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402

# Namespace-lookup imports so monkeypatching engine.core.models.* propagates.
from engine.core import models, trace  # noqa: E402

# Re-export model-name symbols for callers that still read them from here.
from engine.core.models import (  # noqa: F401,E402
    MODEL_PLANNER, MODEL_SEARCHER, MODEL_SYNTHESIZER, MODEL_VERIFIER,
    MODEL_CRITIC, MODEL_ROUTER, MODEL_COMPRESSOR, ENABLE_STREAM,
    _SMALL_MODEL_RE, _default_top_k, _chat, _chat_stream, _llm, OpenAI,
)
from engine.core.trace import (  # noqa: F401,E402
    ENABLE_TRACE, _TRACE_BUFFER, _drain_trace, _merge_trace, _print_trace_summary,
)

ENV = os.environ.get

# ── Pipeline-level env ──────────────────────────────────────────────

NUM_SUBQUERIES = int(ENV("NUM_SUBQUERIES", "3"))
NUM_RESULTS_PER_QUERY = int(ENV("NUM_RESULTS_PER_QUERY", "5"))
SEARXNG_URL = ENV("SEARXNG_URL", "http://localhost:8888")

# Web-search provider: "searxng" (default, local-first) or "exa" (cloud, set EXA_API_KEY).
SEARCH_PROVIDER = ENV("SEARCH_PROVIDER", "searxng").lower()
EXA_API_KEY = ENV("EXA_API_KEY", "")
EXA_SEARCH_TYPE = ENV("EXA_SEARCH_TYPE", "auto")
EXA_CATEGORY = ENV("EXA_CATEGORY", "")
EXA_INCLUDE_DOMAINS = ENV("EXA_INCLUDE_DOMAINS", "")
EXA_EXCLUDE_DOMAINS = ENV("EXA_EXCLUDE_DOMAINS", "")
EXA_START_PUBLISHED_DATE = ENV("EXA_START_PUBLISHED_DATE", "")
EXA_END_PUBLISHED_DATE = ENV("EXA_END_PUBLISHED_DATE", "")
EXA_HIGHLIGHTS_CHARS = int(ENV("EXA_HIGHLIGHTS_CHARS", "500"))

TOP_K_EVIDENCE = _default_top_k(ENV("MODEL_SYNTHESIZER", ""), os.environ.get("TOP_K_EVIDENCE"))
PER_CHUNK_CHAR_CAP = int(ENV("PER_CHUNK_CHAR_CAP", "1200"))

ENABLE_HYDE = ENV("ENABLE_HYDE", "1") == "1"
ENABLE_VERIFY = ENV("ENABLE_VERIFY", "1") == "1"
MAX_ITERATIONS = int(ENV("MAX_ITERATIONS", "2"))
ENABLE_CONSISTENCY = ENV("ENABLE_CONSISTENCY", "0") == "1"
CONSISTENCY_SAMPLES = int(ENV("CONSISTENCY_SAMPLES", "3"))

ENABLE_ROUTER = ENV("ENABLE_ROUTER", "1") == "1"
ENABLE_STEP_VERIFY = ENV("ENABLE_STEP_VERIFY", "1") == "1"
ENABLE_ACTIVE_RETR = ENV("ENABLE_ACTIVE_RETR", "1") == "1"
ENABLE_COMPRESS = ENV("ENABLE_COMPRESS", "1") == "1"
ENABLE_PLAN_REFINE = ENV("ENABLE_PLAN_REFINE", "0") == "1"

ENABLE_RERANK = ENV("ENABLE_RERANK", "0") == "1"
RERANK_CANDIDATES = int(ENV("RERANK_CANDIDATES", "50"))
ENABLE_FETCH = ENV("ENABLE_FETCH", "1") == "1"
FETCH_TIMEOUT_SEC = int(ENV("FETCH_TIMEOUT_SEC", "10"))
FETCH_MAX_CHARS = int(ENV("FETCH_MAX_CHARS", "8000"))
FETCH_MAX_URLS = int(ENV("FETCH_MAX_URLS", "8"))

LOCAL_CORPUS_PATH = ENV("LOCAL_CORPUS_PATH", "")
LOCAL_CORPUS_TOP_K = int(ENV("LOCAL_CORPUS_TOP_K", "5"))

_NUMERIC_RE = re.compile(r"\b\d[\d,\.]*\b|\bhow many\b|\bwhen (was|did)\b|\bwhich year\b", re.IGNORECASE)
_CITE_RE = re.compile(r"\[(\d+)\]")
_HEDGE_RE = re.compile(
    r"(does not specify|is unclear|unclear from the evidence|i (don'?t|do not) know|not certain|"
    r"unknown|cannot determine|no information|not mentioned)",
    re.IGNORECASE,
)


class State(TypedDict, total=False):
    question: str
    question_class: str
    subqueries: list[str]
    evidence: list[dict]
    evidence_compressed: list[dict]
    answer: str
    claims: list[dict]
    unverified: list[str]
    iterations: int
    plan_rejects: int
    trace: list[dict]


# ── Search ──────────────────────────────────────────────────────────

def _searxng(query: str, n: int = NUM_RESULTS_PER_QUERY) -> list[dict]:
    r = requests.get(f"{SEARXNG_URL}/search", params={"q": query, "format": "json"}, timeout=20)
    r.raise_for_status()
    return [{"url": h.get("url", ""), "title": h.get("title", ""), "snippet": h.get("content", "")}
            for h in (r.json().get("results") or [])[:n]]


# Exa AI-powered search. Cloud-only: requires EXA_API_KEY. The client is
# constructed lazily so test environments without the key (or without
# `exa-py` installed) keep importing this module cleanly.
_EXA_CLIENT = None


def _get_exa_client():
    global _EXA_CLIENT
    if _EXA_CLIENT is not None:
        return _EXA_CLIENT
    if not EXA_API_KEY:
        raise RuntimeError(
            "SEARCH_PROVIDER=exa requires EXA_API_KEY to be set. "
            "Get one at https://exa.ai/."
        )
    from exa_py import Exa
    client = Exa(EXA_API_KEY)
    client.headers["x-exa-integration"] = "agentic-research-engine-oss"
    _EXA_CLIENT = client
    return client


def _exa_snippet(item) -> str:
    """Cascade through Exa content fields: summary → joined highlights → text head."""
    summary = getattr(item, "summary", None)
    if summary:
        return str(summary).strip()
    highlights = getattr(item, "highlights", None) or []
    joined = " … ".join(h for h in highlights if h)
    if joined:
        return joined.strip()
    text = getattr(item, "text", None)
    if text:
        return str(text)[:EXA_HIGHLIGHTS_CHARS].strip()
    return ""


def _exa(query: str, n: int = NUM_RESULTS_PER_QUERY) -> list[dict]:
    client = _get_exa_client()
    kwargs: dict = {
        "num_results": n,
        "type": EXA_SEARCH_TYPE,
        "highlights": {"max_characters": EXA_HIGHLIGHTS_CHARS},
        "summary": True,
    }
    if EXA_CATEGORY:
        kwargs["category"] = EXA_CATEGORY
    if EXA_INCLUDE_DOMAINS:
        kwargs["include_domains"] = [d.strip() for d in EXA_INCLUDE_DOMAINS.split(",") if d.strip()]
    if EXA_EXCLUDE_DOMAINS:
        kwargs["exclude_domains"] = [d.strip() for d in EXA_EXCLUDE_DOMAINS.split(",") if d.strip()]
    if EXA_START_PUBLISHED_DATE:
        kwargs["start_published_date"] = EXA_START_PUBLISHED_DATE
    if EXA_END_PUBLISHED_DATE:
        kwargs["end_published_date"] = EXA_END_PUBLISHED_DATE

    response = client.search_and_contents(query, **kwargs)
    return [
        {
            "url": getattr(item, "url", "") or "",
            "title": getattr(item, "title", "") or "",
            "snippet": _exa_snippet(item),
        }
        for item in (getattr(response, "results", None) or [])
    ]


def _web_search(query: str, n: int = NUM_RESULTS_PER_QUERY) -> list[dict]:
    """Dispatch to the configured web-search provider. Returns [{url,title,snippet}]."""
    if SEARCH_PROVIDER == "exa":
        return _exa(query, n)
    return _searxng(query, n)


# ── Shared helpers ─────────────────────────────────────────────────

def _grounding_score(answer: str, evidence: list[dict]) -> float:
    refs = {int(m) for m in _CITE_RE.findall(answer)}
    if not refs:
        return 0.0
    valid = sum(1 for r in refs if 1 <= r <= len(evidence))
    return (valid / len(refs)) * (valid ** 0.5)


def _critic(step: str, payload: str, context: str) -> tuple[bool, str]:
    if not ENABLE_STEP_VERIFY:
        return True, ""
    prompt = (
        f"You are a step-level verifier for a research agent pipeline. Judge the step's "
        f"output given the context. Respond on exactly two lines:\n"
        f"  VERDICT: accept | redo\n"
        f"  FEEDBACK: <one short sentence if redo, else empty>\n\n"
        f"Step: {step}\nContext: {context}\nOutput to judge:\n{payload}"
    )
    raw = _chat(MODEL_CRITIC, prompt)
    verdict_line = next((l for l in raw.splitlines() if l.strip().upper().startswith("VERDICT:")), "")
    accept = "accept" in verdict_line.lower() or "redo" not in verdict_line.lower()
    feedback = next((l.split(":", 1)[1].strip() for l in raw.splitlines() if l.strip().upper().startswith("FEEDBACK:")), "")
    return accept, feedback


# ── T4.3 · Classifier ──────────────────────────────────────────────

def _classify(state: State) -> dict:
    if not ENABLE_ROUTER:
        return {"question_class": "multihop", "trace": _merge_trace(state, "classify")}
    prompt = (
        "Classify this research question as exactly ONE of: factoid, multihop, synthesis.\n"
        "  factoid    = single short-answer fact (e.g. capital, year, name)\n"
        "  multihop   = needs to combine facts from multiple sources\n"
        "  synthesis  = open-ended comparison / explanation / analysis\n"
        "Reply with ONLY the single word.\n\n"
        f"Question: {state['question']}"
    )
    raw = _chat(MODEL_ROUTER, prompt).strip().lower()
    label = raw.split()[0] if raw else "multihop"
    if label not in {"factoid", "multihop", "synthesis"}:
        label = "multihop"
    return {"question_class": label, "trace": _merge_trace(state, "classify")}


# ── Plan (+ HyDE + critic + T4.5 refinement) ───────────────────────

def _hyde_expand(sub: str) -> str:
    hyde = _chat(MODEL_PLANNER,
                        f"Write one concise factual paragraph answering: {sub}\n"
                        f"Respond with ONLY the paragraph, no preamble.")
    return f"{sub}\n\n{hyde.strip()}"


def _plan(state: State) -> dict:
    n_subs = NUM_SUBQUERIES if state.get("question_class") != "factoid" else max(1, NUM_SUBQUERIES - 1)
    prompt = (f"Break this research question into exactly {n_subs} focused sub-queries. "
              f"One per line, no numbering.\n\nQuestion: {state['question']}")
    subs = [l.strip(" -•*") for l in _chat(MODEL_PLANNER, prompt).splitlines() if l.strip()][:n_subs]

    use_hyde = ENABLE_HYDE and not _NUMERIC_RE.search(state["question"]) \
        and state.get("question_class") != "factoid"
    if use_hyde:
        subs = [_hyde_expand(s) for s in subs]

    accept, _ = _critic("plan", "\n".join(subs), state["question"])
    rejects = state.get("plan_rejects", 0)
    if not accept and ENABLE_PLAN_REFINE and rejects == 0:
        prompt2 = prompt + "\n\nThe previous decomposition was rejected as too vague. Be more specific."
        subs = [l.strip(" -•*") for l in _chat(MODEL_PLANNER, prompt2).splitlines() if l.strip()][:n_subs]
        rejects = 1

    return {
        "subqueries": subs,
        "iterations": state.get("iterations", 0),
        "plan_rejects": rejects,
        "trace": _merge_trace(state, "plan"),
    }


# ── Search (web + W5 corpus augmentation) ───────────────────────────

def _search_one(sub: str) -> list[dict]:
    hits = _web_search(sub)
    if not hits:
        return []
    sources = "\n".join(f"[{i+1}] {h['title']} — {h['snippet']}  (url: {h['url']})" for i, h in enumerate(hits))
    summary = _chat(MODEL_SEARCHER,
                           f"Summarize these sources factually in 3-5 sentences with inline [1], [2] citations. "
                           f"Only use the information provided.\n\nSub-query: {sub}\n\nSources:\n{sources}")
    return [{"url": h["url"], "title": h["title"], "text": summary} for h in hits]


# W5.1 — local corpus singleton.
_CORPUS: CorpusIndex | None = None
_CORPUS_LOAD_FAILED = False


def _get_corpus() -> CorpusIndex | None:
    """Load the LOCAL_CORPUS_PATH index once per process; None if not configured.

    LOCAL_CORPUS_PATH read at call time (not module load) so domain presets
    that set it via env-var overrides take effect. Errors:
      - FileNotFoundError: treated as "not set yet" — we silently skip corpus
        augmentation. Common in misconfigured setups.
      - Any other Exception: treated as "corpus broken" — louder warning on
        stderr + cache the failure so we don't retry on every query.
    """
    global _CORPUS, _CORPUS_LOAD_FAILED
    path = os.environ.get("LOCAL_CORPUS_PATH", "") or LOCAL_CORPUS_PATH
    if _CORPUS is not None or _CORPUS_LOAD_FAILED or not path:
        return _CORPUS
    try:
        _CORPUS = CorpusIndex.load(path)
        print(f"[corpus] loaded {len(_CORPUS.chunks)} chunks from {path}",
              file=sys.stderr)
    except FileNotFoundError:
        # The index directory / file doesn't exist. Quiet skip (user likely
        # hasn't built the index yet); the warning would be noise.
        _CORPUS_LOAD_FAILED = True
        return None
    except Exception as exc:  # noqa: BLE001 — real breakage deserves a visible warning
        print(f"[corpus] LOAD BROKEN at {path}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        print(f"[corpus]   falling back to web-only search for the rest of this run",
              file=sys.stderr)
        _CORPUS_LOAD_FAILED = True
        return None
    return _CORPUS


def _corpus_hits(query: str, k: int = LOCAL_CORPUS_TOP_K) -> list[dict]:
    idx = _get_corpus()
    if not idx:
        return []
    try:
        hits = idx.query(query, k=k)
    except Exception as exc:  # noqa: BLE001
        print(f"[corpus] query failed: {exc}", file=sys.stderr)
        return []
    out: list[dict] = []
    for chunk, _ in hits:
        loc = f"corpus://{chunk.source}"
        if chunk.page is not None:
            loc += f"#p{chunk.page}"
        loc += f"#c{chunk.chunk_idx}"
        title = chunk.source + (f" (p{chunk.page})" if chunk.page is not None else "")
        out.append({"url": loc, "title": title, "text": chunk.text})
    return out


def _search(state: State) -> dict:
    subs = state.get("unverified") or state["subqueries"]
    with ThreadPoolExecutor(max_workers=max(len(subs), 1)) as pool:
        new_items = [e for batch in pool.map(_search_one, subs) for e in batch]

    corpus_count = 0
    if _get_corpus() is not None:
        for sub in subs:
            hits = _corpus_hits(sub)
            new_items.extend(hits)
            corpus_count += len(hits)

    existing = state.get("evidence", [])
    seen = {e["url"] for e in existing}
    evidence = existing + [e for e in new_items if e["url"] and e["url"] not in seen and not seen.add(e["url"])]

    if ENABLE_STEP_VERIFY and not state.get("unverified"):
        preview = "\n".join(f"[{i+1}] {e['title']}" for i, e in enumerate(evidence[:12]))
        _critic("search", preview, state["question"])

    extras: list[dict] | None = None
    if corpus_count:
        extras = [{"model": "corpus", "latency_s": 0.0, "tokens_est": 0,
                   "n_hits": corpus_count, "n_subqueries": len(subs)}]
    return {"evidence": evidence, "trace": _merge_trace(state, "search", extras)}


# ── W4.1 · Retrieve (hybrid + optional rerank) ─────────────────────

_RERANKER: CrossEncoderReranker | None = None


def _get_reranker() -> CrossEncoderReranker:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoderReranker()
    return _RERANKER


def _retrieve(state: State) -> dict:
    ev = state["evidence"]
    if len(ev) <= TOP_K_EVIDENCE:
        return {"evidence": ev, "trace": _merge_trace(state, "retrieve")}

    t0 = time.monotonic()
    r = HybridRetriever()
    r.add([e["text"] for e in ev])

    reranked_flag = False
    if ENABLE_RERANK:
        stage1_k = min(RERANK_CANDIDATES, len(ev))
        top = r.retrieve(state["question"], k=stage1_k)
        try:
            reranked = _get_reranker().rerank(state["question"], top, k=TOP_K_EVIDENCE)
            picked = [text for text, _ in reranked]
            reranked_flag = True
        except Exception as exc:  # noqa: BLE001
            print(f"[rerank] falling back to hybrid-only: {exc}", file=sys.stderr)
            picked = [text for text, _ in top[:TOP_K_EVIDENCE]]
    else:
        picked = [text for text, _ in r.retrieve(state["question"], k=TOP_K_EVIDENCE)]

    by_text = {e["text"]: e for e in ev}
    kept = [by_text[t] for t in picked if t in by_text]

    extras = [{
        "model": "hybrid+rerank" if reranked_flag else "hybrid",
        "latency_s": round(time.monotonic() - t0, 3),
        "tokens_est": 0,
        "n_in": len(ev),
        "n_out": len(kept),
    }]
    return {"evidence": kept, "trace": _merge_trace(state, "retrieve", extras)}


# ── W4.2 · Fetch ────────────────────────────────────────────────────

def _fetch_one(url: str) -> str | None:
    """Download `url`, return first FETCH_MAX_CHARS of clean text. Skips corpus://."""
    if url.startswith("corpus://"):
        return None
    try:
        import trafilatura  # type: ignore
    except ImportError:
        return None
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None
        text = trafilatura.extract(
            downloaded, favor_recall=False, include_comments=False, include_tables=False
        )
        if not text:
            return None
        return text[:FETCH_MAX_CHARS]
    except Exception:  # noqa: BLE001
        return None


def _fetch_url(state: State) -> dict:
    if not ENABLE_FETCH:
        return {"trace": _merge_trace(state, "fetch_url")}
    ev = state.get("evidence") or []
    if not ev:
        return {"trace": _merge_trace(state, "fetch_url")}

    targets = ev[:FETCH_MAX_URLS]
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(len(targets), 1)) as pool:
        fulls = list(pool.map(lambda e: _fetch_one(e["url"]), targets))

    enriched: list[dict] = []
    n_fetched = 0
    for e, full in zip(targets, fulls):
        if full:
            enriched.append({**e, "text": full, "fetched": True})
            n_fetched += 1
        else:
            enriched.append({**e, "fetched": False})
    enriched.extend(ev[FETCH_MAX_URLS:])

    extras = [{
        "model": "trafilatura",
        "latency_s": round(time.monotonic() - t0, 3),
        "tokens_est": 0,
        "n_fetched": n_fetched,
        "n_attempted": len(targets),
    }]
    return {"evidence": enriched, "trace": _merge_trace(state, "fetch_url", extras)}


# ── T4.4 · Compress (+ W6.2 per-chunk cap) ──────────────────────────

def _compress(state: State) -> dict:
    if not state.get("evidence"):
        return {
            "evidence_compressed": state.get("evidence", []),
            "trace": _merge_trace(state, "compress"),
        }

    compressed: list[dict] = list(state["evidence"])

    if ENABLE_COMPRESS:
        bullets = "\n\n".join(f"[{i+1}] {e['text']}" for i, e in enumerate(state["evidence"]))
        prompt = (
            f"Compress each numbered chunk below to 2-3 short sentences that keep ONLY what "
            f"is relevant to the question. Preserve the bracket indices exactly. Output each "
            f"compressed chunk as `[N] <compressed text>` on its own paragraph.\n\n"
            f"Question: {state['question']}\n\nChunks:\n{bullets}"
        )
        raw = _chat(MODEL_COMPRESSOR, prompt)
        for line in raw.splitlines():
            m = re.match(r"\[(\d+)\]\s*(.+)", line.strip())
            if not m:
                continue
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(compressed):
                compressed[idx] = {**compressed[idx], "text": m.group(2).strip()}

    compressed = [
        {**c, "text": c["text"][:PER_CHUNK_CHAR_CAP]} if len(c.get("text", "")) > PER_CHUNK_CHAR_CAP else c
        for c in compressed
    ]
    return {"evidence_compressed": compressed, "trace": _merge_trace(state, "compress")}


# ── Synthesize (+ FLARE) ────────────────────────────────────────────

def _synthesize_once(state: State) -> str:
    ev = state.get("evidence_compressed") or state["evidence"]
    bullets = "\n".join(f"[{i+1}] {e['text']}  (src: {e['url']})" for i, e in enumerate(ev))
    prompt = (
        f"Answer the question using ONLY the evidence. Rules:\n"
        f"  1. Cite every factual claim inline as [1], [2], etc.\n"
        f"  2. If the evidence FULLY answers the question: answer concisely.\n"
        f"  3. If the evidence partially answers the question: answer what the "
        f"evidence supports, then name the specific aspects that are NOT "
        f"supported by the evidence. Cite the supported parts.\n"
        f"  4. If the evidence is UNRELATED to the question (covers a different "
        f"topic entirely), reply with exactly: \"The provided evidence does "
        f"not answer this question.\" — nothing else.\n"
        f"  5. Never invent facts, definitions, or examples not present in the "
        f"evidence. Never substitute a related topic the evidence covers for "
        f"the actual question.\n\n"
        f"Question: {state['question']}\n\nEvidence:\n{bullets}"
    )
    if ENABLE_STREAM and not ENABLE_CONSISTENCY:
        return _chat_stream(MODEL_SYNTHESIZER, prompt)
    return _chat(MODEL_SYNTHESIZER, prompt)


def _flare_augment(state: State, draft: str) -> str:
    if not ENABLE_ACTIVE_RETR or not _HEDGE_RE.search(draft):
        return draft
    hedge_match = _HEDGE_RE.search(draft)
    start = max(0, draft.rfind(".", 0, hedge_match.start()))
    end = draft.find(".", hedge_match.end())
    focus = draft[start:end + 1 if end != -1 else len(draft)].strip(". ")
    targeted_query = f"{state['question']} — specifically: {focus}"
    new_hits = _search_one(targeted_query)
    seen = {e["url"] for e in state.get("evidence", [])}
    fresh = [e for e in new_hits if e["url"] and e["url"] not in seen]
    if not fresh:
        return draft
    state_aug = {**state, "evidence": (state.get("evidence_compressed") or state["evidence"]) + fresh}
    return _synthesize_once(state_aug)


def _synthesize(state: State) -> dict:
    if not ENABLE_CONSISTENCY:
        draft = _synthesize_once(state)
        return {"answer": _flare_augment(state, draft), "trace": _merge_trace(state, "synthesize")}
    candidates = [_flare_augment(state, _synthesize_once(state)) for _ in range(CONSISTENCY_SAMPLES)]
    best = max(candidates, key=lambda a: _grounding_score(a, state.get("evidence_compressed") or state["evidence"]))
    return {"answer": best, "trace": _merge_trace(state, "synthesize")}


# ── Verify + iteration (CoVe) ───────────────────────────────────────

def _verify(state: State) -> dict:
    if not ENABLE_VERIFY:
        return {"claims": [], "unverified": [], "trace": _merge_trace(state, "verify")}
    ev = state.get("evidence_compressed") or state["evidence"]
    bullets = "\n".join(f"[{i+1}] {e['text']}" for i, e in enumerate(ev))
    prompt = (f"You are verifying a candidate answer. List each standalone factual claim on its own line "
              f"as `CLAIM: <text>`. Then for each claim, output `VERIFIED: yes` or `VERIFIED: no` based "
              f"STRICTLY on whether the evidence below supports it.\n\n"
              f"Answer:\n{state['answer']}\n\nEvidence:\n{bullets}")
    raw = _chat(MODEL_VERIFIER, prompt)
    claims: list[dict] = []
    current: dict | None = None
    for line in raw.splitlines():
        s = line.strip()
        if s.upper().startswith("CLAIM:"):
            current = {"text": s.split(":", 1)[1].strip(), "verified": False}
            claims.append(current)
        elif s.upper().startswith("VERIFIED:") and current is not None:
            current["verified"] = "yes" in s.lower()
            current = None
    unverified = [c["text"] for c in claims if not c["verified"]]
    return {
        "claims": claims,
        "unverified": unverified,
        "iterations": state.get("iterations", 0) + 1,
        "trace": _merge_trace(state, "verify"),
    }


def _after_verify(state: State) -> str:
    if ENABLE_VERIFY and state.get("unverified") and state.get("iterations", 0) < MAX_ITERATIONS:
        return "search"
    return END


# ── Graph ───────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(State)
    for n, f in [("classify", _classify), ("plan", _plan), ("search", _search),
                 ("retrieve", _retrieve), ("fetch_url", _fetch_url), ("compress", _compress),
                 ("synthesize", _synthesize), ("verify", _verify)]:
        g.add_node(n, f)
    g.set_entry_point("classify")
    for a, b in [("classify", "plan"), ("plan", "search"), ("search", "retrieve"),
                 ("retrieve", "fetch_url"), ("fetch_url", "compress"),
                 ("compress", "synthesize"), ("synthesize", "verify")]:
        g.add_edge(a, b)
    g.add_conditional_edges("verify", _after_verify, {"search": "search", END: END})
    return g.compile()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is Anthropic's contextual retrieval and why does it reduce retrieval failures?"
    print(f"Q: {q}")
    result = build_graph().invoke({"question": q, "iterations": 0, "plan_rejects": 0, "trace": []})
    print(f"\n[class: {result.get('question_class', '?')}]")
    if not ENABLE_STREAM:
        print(f"\nA: {result['answer']}")
    if result.get("claims"):
        v = sum(1 for c in result["claims"] if c["verified"])
        print(f"\nVerified: {v}/{len(result['claims'])} claims  (iterations: {result.get('iterations', 0)})")
    if ENABLE_TRACE:
        _print_trace_summary(result.get("trace", []))
