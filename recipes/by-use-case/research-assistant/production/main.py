"""Research assistant — production tier (Wave 2 Tiers 2+4 + Wave 4 local-first).

Wave 4 layers three local-first enhancements on top of Wave 2/4 adaptive
verification. All three are env-gated and ship disabled or graceful-degrading
so the existing pipeline keeps working on first clone.

Pipeline:

  [T4.3 classify] → plan → [T4.1 critic] → search → [T4.1 critic] → retrieve (+W4.1 rerank)
                                                                        │
                                                       [W4.2 fetch_url] ◀┘
                                                              │
                                                              ▼
                                    [T4.4 compress] ◀─────────┘
                                          │
                                          ▼
                                    synthesize ◀── [T4.2 FLARE active retrieve]
                                          │
                                          ▼
                                       verify (CoVe)
                                          │
                           verified? ──yes──▶ (consistency) ──▶ END
                                          │
                                          no
                                          │
                           iterate (re-search failed claims) ──▶ search

References:
  - HyDE — Gao et al. 2023.
  - CoVe — Dhuliawala et al. 2023; MiroThinker-H1 2026.
  - ITER-RETGEN — Shao et al. 2023.
  - Self-consistency — Wang et al. 2022.
  - Process Reward / step-level critique (ThinkPRM pattern) — Khalifa et al. 2026.
  - FLARE — Forward-Looking Active REtrieval (Jiang et al. 2023; +62% on 2Wiki).
  - LongLLMLingua — prompt compression (Jiang et al. 2024; +17-21% on NaturalQuestions).
  - Compute-optimal test-time scaling — Snell et al. 2024; agent adaptation 2026.
  - bge-reranker-v2-m3 — BAAI cross-encoder reranking. Two-stage retrieval
    (hybrid → cross-encoder) lifts MRR 5-10 pts over hybrid-only on MTEB reranking.
  - Trafilatura — Barbaresi 2021 (ACL). Beats Readability/Goose3 on TREC-HTML F1.

Env vars (all defaults shown; most gates are 1=on):
  # Tier 2
  ENABLE_HYDE            1    skipped automatically on numeric queries
  ENABLE_VERIFY          1    CoVe after synthesize
  MAX_ITERATIONS         2    bound on verify→search iterations
  ENABLE_CONSISTENCY     0    opt-in; N× synthesize cost

  # Tier 4
  ENABLE_ROUTER          1    T4.3 — classify question, route compute
  ENABLE_STEP_VERIFY     1    T4.1 — critic after plan/search/retrieve
  ENABLE_ACTIVE_RETR     1    T4.2 — FLARE re-search on low-confidence claims
  ENABLE_COMPRESS        1    T4.4 — LLM-based evidence compression
  ENABLE_PLAN_REFINE     0    T4.5 — opt-in; replan when step critic rejects plan
  CONSISTENCY_SAMPLES    3

  # Wave 4 — local-first engine enhancements
  ENABLE_RERANK          0    W4.1 — two-stage: hybrid top-N → cross-encoder top-K.
                              Requires sentence-transformers + bge-reranker-v2-m3
                              (~560MB download on first run). Gracefully degrades
                              to hybrid-only if the model can't load.
  RERANK_CANDIDATES     50    first-stage pool size when ENABLE_RERANK=1
  ENABLE_FETCH           1    W4.2 — pull full page text with trafilatura.
                              SearXNG returns snippets; this reads the article.
                              Gracefully degrades to snippets if trafilatura
                              isn't installed or a URL fails.
  FETCH_TIMEOUT_SEC     10
  FETCH_MAX_CHARS     8000    truncate per page after clean-text extraction
  FETCH_MAX_URLS         8    cap concurrent fetches per retrieve cycle
  ENABLE_TRACE           1    W4.3 — record {node, model, latency_s, tokens_est}
                              per LLM call. Printed as a summary at CLI end.

  # Wave 5 — local corpus augmentation
  LOCAL_CORPUS_PATH      ""   W5.1 — directory produced by scripts/index_corpus.py.
                              When set, `_search` augments SearXNG hits with the
                              top-K local matches per sub-query. Unset = web-only.
  LOCAL_CORPUS_TOP_K      5   per-subquery hits pulled from the local corpus.
"""

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

import requests  # noqa: E402
from core.rag import CorpusIndex, CrossEncoderReranker, HybridRetriever  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402
from openai import OpenAI  # noqa: E402

ENV = os.environ.get
MODEL_PLANNER = ENV("MODEL_PLANNER", "gpt-5-nano")
MODEL_SEARCHER = ENV("MODEL_SEARCHER", "gpt-5-mini")
MODEL_SYNTHESIZER = ENV("MODEL_SYNTHESIZER", "gpt-5-mini")
MODEL_VERIFIER = ENV("MODEL_VERIFIER", MODEL_PLANNER)
MODEL_CRITIC = ENV("MODEL_CRITIC", MODEL_PLANNER)
MODEL_ROUTER = ENV("MODEL_ROUTER", MODEL_PLANNER)
MODEL_COMPRESSOR = ENV("MODEL_COMPRESSOR", MODEL_PLANNER)

NUM_SUBQUERIES = int(ENV("NUM_SUBQUERIES", "3"))
NUM_RESULTS_PER_QUERY = int(ENV("NUM_RESULTS_PER_QUERY", "5"))
TOP_K_EVIDENCE = int(ENV("TOP_K_EVIDENCE", "8"))
SEARXNG_URL = ENV("SEARXNG_URL", "http://localhost:8888")

ENABLE_HYDE = ENV("ENABLE_HYDE", "1") == "1"
ENABLE_VERIFY = ENV("ENABLE_VERIFY", "1") == "1"
MAX_ITERATIONS = int(ENV("MAX_ITERATIONS", "2"))
ENABLE_CONSISTENCY = ENV("ENABLE_CONSISTENCY", "0") == "1"
CONSISTENCY_SAMPLES = int(ENV("CONSISTENCY_SAMPLES", "3"))

ENABLE_ROUTER = ENV("ENABLE_ROUTER", "1") == "1"  # T4.3
ENABLE_STEP_VERIFY = ENV("ENABLE_STEP_VERIFY", "1") == "1"  # T4.1
ENABLE_ACTIVE_RETR = ENV("ENABLE_ACTIVE_RETR", "1") == "1"  # T4.2
ENABLE_COMPRESS = ENV("ENABLE_COMPRESS", "1") == "1"  # T4.4
ENABLE_PLAN_REFINE = ENV("ENABLE_PLAN_REFINE", "0") == "1"  # T4.5

# Wave 4 — local-first engine enhancements.
ENABLE_RERANK = ENV("ENABLE_RERANK", "0") == "1"  # W4.1
RERANK_CANDIDATES = int(ENV("RERANK_CANDIDATES", "50"))
ENABLE_FETCH = ENV("ENABLE_FETCH", "1") == "1"  # W4.2
FETCH_TIMEOUT_SEC = int(ENV("FETCH_TIMEOUT_SEC", "10"))
FETCH_MAX_CHARS = int(ENV("FETCH_MAX_CHARS", "8000"))
FETCH_MAX_URLS = int(ENV("FETCH_MAX_URLS", "8"))
ENABLE_TRACE = ENV("ENABLE_TRACE", "1") == "1"  # W4.3

# Wave 5 — local corpus augmentation.
LOCAL_CORPUS_PATH = ENV("LOCAL_CORPUS_PATH", "")
LOCAL_CORPUS_TOP_K = int(ENV("LOCAL_CORPUS_TOP_K", "5"))

_NUMERIC_RE = re.compile(r"\b\d[\d,\.]*\b|\bhow many\b|\bwhen (was|did)\b|\bwhich year\b", re.IGNORECASE)
_CITE_RE = re.compile(r"\[(\d+)\]")
# FLARE: hedges that signal low-confidence claims worth re-searching.
_HEDGE_RE = re.compile(
    r"(does not specify|is unclear|unclear from the evidence|i (don'?t|do not) know|not certain|"
    r"unknown|cannot determine|no information|not mentioned)",
    re.IGNORECASE,
)


class State(TypedDict, total=False):
    question: str
    question_class: str              # T4.3: "factoid" | "multihop" | "synthesis"
    subqueries: list[str]
    evidence: list[dict]
    evidence_compressed: list[dict]  # T4.4 — compressed view used by synthesize
    answer: str
    claims: list[dict]
    unverified: list[str]
    iterations: int
    plan_rejects: int                # T4.5 — bound on plan-refinement loops
    trace: list[dict]                # W4.3 — per-call observability


# ── W4.3 · Trace plumbing ─────────────────────────────────────────────
# Module-level buffer captures every LLM call; nodes drain it into state.trace
# so the CLI can print a per-node, per-model summary after the graph runs.

_TRACE_BUFFER: list[dict] = []


def _drain_trace(node: str) -> list[dict]:
    """Tag buffered LLM-call entries with `node` and return them; clears the buffer."""
    if not _TRACE_BUFFER:
        return []
    entries = [{"node": node, **e} for e in _TRACE_BUFFER]
    _TRACE_BUFFER.clear()
    return entries


def _merge_trace(state: State, node: str, extras: list[dict] | None = None) -> list[dict]:
    """Return the state's full trace list with this node's entries appended."""
    delta = _drain_trace(node)
    if extras:
        delta.extend({"node": node, **e} for e in extras)
    return state.get("trace", []) + delta


# ── LLM plumbing ──────────────────────────────────────────────────────

def _llm() -> OpenAI:
    return OpenAI(api_key=ENV("OPENAI_API_KEY", "ollama"), base_url=ENV("OPENAI_BASE_URL"))


def _chat(model: str, prompt: str, temperature: float = 0.0) -> str:
    t0 = time.monotonic()
    resp = _llm().chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], temperature=temperature
    )
    content = resp.choices[0].message.content or ""
    if ENABLE_TRACE:
        _TRACE_BUFFER.append({
            "model": model,
            "latency_s": round(time.monotonic() - t0, 3),
            "prompt_chars": len(prompt),
            "response_chars": len(content),
            "tokens_est": (len(prompt) + len(content)) // 4,
        })
    return content


def _searxng(query: str, n: int = NUM_RESULTS_PER_QUERY) -> list[dict]:
    r = requests.get(f"{SEARXNG_URL}/search", params={"q": query, "format": "json"}, timeout=20)
    r.raise_for_status()
    return [{"url": h.get("url", ""), "title": h.get("title", ""), "snippet": h.get("content", "")}
            for h in (r.json().get("results") or [])[:n]]


# ── Shared helpers ────────────────────────────────────────────────────

def _grounding_score(answer: str, evidence: list[dict]) -> float:
    """Blended validity-ratio × sqrt(coverage) — the ranking metric for self-consistency."""
    refs = {int(m) for m in _CITE_RE.findall(answer)}
    if not refs:
        return 0.0
    valid = sum(1 for r in refs if 1 <= r <= len(evidence))
    return (valid / len(refs)) * (valid ** 0.5)


def _critic(step: str, payload: str, context: str) -> tuple[bool, str]:
    """T4.1 — ThinkPRM-style step-level critic.

    Returns (accept, feedback). A short LLM call that judges whether `payload`
    is a valid output for `step` given `context`. Deterministic protocol:
    the critic replies `VERDICT: accept|redo` on the first line and optional
    `FEEDBACK: ...` on the second. Any parse failure = accept (fail open).
    """
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


# ── T4.3 · Router ─────────────────────────────────────────────────────

def _classify(state: State) -> dict:
    """T4.3 — classify question into {factoid, multihop, synthesis}; downstream nodes use this."""
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


# ── Plan (+ HyDE + optional critic) ───────────────────────────────────

def _hyde_expand(sub: str) -> str:
    hyde = _chat(MODEL_PLANNER,
                 f"Write one concise factual paragraph answering: {sub}\n"
                 f"Respond with ONLY the paragraph, no preamble.")
    return f"{sub}\n\n{hyde.strip()}"


def _plan(state: State) -> dict:
    """Decompose question into sub-queries; HyDE-expand unless numeric; critic-verify."""
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
        # T4.5 — one-shot replan: regenerate with a tightening instruction.
        prompt2 = prompt + "\n\nThe previous decomposition was rejected as too vague. Be more specific."
        subs = [l.strip(" -•*") for l in _chat(MODEL_PLANNER, prompt2).splitlines() if l.strip()][:n_subs]
        rejects = 1

    return {
        "subqueries": subs,
        "iterations": state.get("iterations", 0),
        "plan_rejects": rejects,
        "trace": _merge_trace(state, "plan"),
    }


# ── Search (+ optional critic on coverage) ────────────────────────────

def _search_one(sub: str) -> list[dict]:
    hits = _searxng(sub)
    if not hits:
        return []
    sources = "\n".join(f"[{i+1}] {h['title']} — {h['snippet']}  (url: {h['url']})" for i, h in enumerate(hits))
    summary = _chat(MODEL_SEARCHER,
                    f"Summarize these sources factually in 3-5 sentences with inline [1], [2] citations. "
                    f"Only use the information provided.\n\nSub-query: {sub}\n\nSources:\n{sources}")
    return [{"url": h["url"], "title": h["title"], "text": summary} for h in hits]


# W5.1 — local corpus singleton, loaded lazily so process startup stays fast.
_CORPUS: CorpusIndex | None = None
_CORPUS_LOAD_FAILED = False


def _get_corpus() -> CorpusIndex | None:
    """Return the on-disk corpus if `LOCAL_CORPUS_PATH` is configured, else None.

    Graceful: logs a warning and returns None on any load failure so the
    pipeline keeps working with web-only search.
    """
    global _CORPUS, _CORPUS_LOAD_FAILED
    if _CORPUS is not None or _CORPUS_LOAD_FAILED or not LOCAL_CORPUS_PATH:
        return _CORPUS
    try:
        _CORPUS = CorpusIndex.load(LOCAL_CORPUS_PATH)
        print(f"[corpus] loaded {len(_CORPUS.chunks)} chunks from {LOCAL_CORPUS_PATH}",
              file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"[corpus] load failed, falling back to web-only: {exc}", file=sys.stderr)
        _CORPUS_LOAD_FAILED = True
        return None
    return _CORPUS


def _corpus_hits(query: str, k: int = LOCAL_CORPUS_TOP_K) -> list[dict]:
    """W5.1 — fetch top-k local-corpus hits for `query`, shaped like SearXNG hits.

    Returns evidence items with `corpus://` URLs so fetch_url skips them and
    citations remain traceable to their source file.
    """
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
    """Parallel search; dedupe by URL; append on iteration.

    When `LOCAL_CORPUS_PATH` is set, web hits are augmented with top-K local
    matches per sub-query (W5.1). Local hits carry `corpus://` URLs so the
    fetch_url node skips them — their text is already the full chunk.
    """
    subs = state.get("unverified") or state["subqueries"]
    with ThreadPoolExecutor(max_workers=max(len(subs), 1)) as pool:
        new_items = [e for batch in pool.map(_search_one, subs) for e in batch]
    # W5.1 — augment with local corpus results (if configured).
    corpus_count = 0
    if _get_corpus() is not None:
        for sub in subs:
            hits = _corpus_hits(sub)
            new_items.extend(hits)
            corpus_count += len(hits)
    existing = state.get("evidence", [])
    seen = {e["url"] for e in existing}
    evidence = existing + [e for e in new_items if e["url"] and e["url"] not in seen and not seen.add(e["url"])]

    # T4.1 — critic on coverage; if reject, the agent still proceeds but flags concern.
    if ENABLE_STEP_VERIFY and not state.get("unverified"):
        preview = "\n".join(f"[{i+1}] {e['title']}" for i, e in enumerate(evidence[:12]))
        _critic("search", preview, state["question"])  # side-effect only: logs concern

    extras: list[dict] | None = None
    if corpus_count:
        extras = [{"model": "corpus", "latency_s": 0.0, "tokens_est": 0,
                   "n_hits": corpus_count, "n_subqueries": len(subs)}]
    return {"evidence": evidence, "trace": _merge_trace(state, "search", extras)}


# ── W4.1 · Retrieve with optional cross-encoder rerank ────────────────

# Module-level singleton so the cross-encoder model only loads once per process.
_RERANKER: CrossEncoderReranker | None = None


def _get_reranker() -> CrossEncoderReranker:
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = CrossEncoderReranker()
    return _RERANKER


def _retrieve(state: State) -> dict:
    """Hybrid retrieval, optionally followed by cross-encoder rerank (W4.1).

    - No rerank: hybrid top-K passes straight through (existing behavior).
    - Rerank on: hybrid retrieves RERANK_CANDIDATES, then cross-encoder picks top-K.
    - If the cross-encoder fails to load (no sentence-transformers, no model
      download), we log and fall back to hybrid-only — never crash the pipeline.
    """
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
        except Exception as exc:  # noqa: BLE001 — fall back on any failure
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


# ── W4.2 · Fetch full page text ───────────────────────────────────────

def _fetch_one(url: str) -> str | None:
    """Download `url`, return clean article text (first FETCH_MAX_CHARS), or None.

    Local corpus URLs (`corpus://…`) are skipped — their text is already the
    full chunk, and they're not fetchable over HTTP anyway. Returning None
    makes `_fetch_url` keep the existing text and mark `fetched: False`.
    """
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
    except Exception:  # noqa: BLE001 — any network / parse error → fall back to snippet
        return None


def _fetch_url(state: State) -> dict:
    """W4.2 — replace per-source snippets with full article text where possible.

    SearXNG gives us search-result snippets (~200 chars). Multi-hop questions
    need the actual article — contextual retrieval, numerical citations, etc.
    This node fetches each URL (bounded concurrency), extracts clean text via
    trafilatura, and overwrites `e["text"]`. URLs that fail fetching keep
    their snippet-derived text.
    """
    if not ENABLE_FETCH:
        return {"trace": _merge_trace(state, "fetch_url")}
    ev = state.get("evidence") or []
    if not ev:
        return {"trace": _merge_trace(state, "fetch_url")}

    targets = ev[:FETCH_MAX_URLS]  # cap network load
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
    enriched.extend(ev[FETCH_MAX_URLS:])  # preserve any evidence beyond cap

    extras = [{
        "model": "trafilatura",
        "latency_s": round(time.monotonic() - t0, 3),
        "tokens_est": 0,
        "n_fetched": n_fetched,
        "n_attempted": len(targets),
    }]
    return {"evidence": enriched, "trace": _merge_trace(state, "fetch_url", extras)}


# ── T4.4 · Compress ───────────────────────────────────────────────────

def _compress(state: State) -> dict:
    """T4.4 — LLM-based evidence compression (portable alternative to LongLLMLingua).

    Goal: cut evidence text ~3× while preserving claims that answer the question.
    Implementation: one compressor call asks for a 2-3 sentence distilled version of
    each evidence chunk, focused on the question. Keeps URLs intact so citations still work.
    """
    if not ENABLE_COMPRESS or not state.get("evidence"):
        return {
            "evidence_compressed": state.get("evidence", []),
            "trace": _merge_trace(state, "compress"),
        }
    bullets = "\n\n".join(f"[{i+1}] {e['text']}" for i, e in enumerate(state["evidence"]))
    prompt = (
        f"Compress each numbered chunk below to 2-3 short sentences that keep ONLY what "
        f"is relevant to the question. Preserve the bracket indices exactly. Output each "
        f"compressed chunk as `[N] <compressed text>` on its own paragraph.\n\n"
        f"Question: {state['question']}\n\nChunks:\n{bullets}"
    )
    raw = _chat(MODEL_COMPRESSOR, prompt)
    compressed: list[dict] = list(state["evidence"])  # default: pass-through
    for line in raw.splitlines():
        m = re.match(r"\[(\d+)\]\s*(.+)", line.strip())
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(compressed):
            compressed[idx] = {**compressed[idx], "text": m.group(2).strip()}
    return {"evidence_compressed": compressed, "trace": _merge_trace(state, "compress")}


# ── Synthesize (+ FLARE active retrieval on hedged claims) ───────────

def _synthesize_once(state: State) -> str:
    ev = state.get("evidence_compressed") or state["evidence"]
    bullets = "\n".join(f"[{i+1}] {e['text']}  (src: {e['url']})" for i, e in enumerate(ev))
    prompt = (f"Answer using the evidence. Cite sources inline as [1], [2]. Be concise and factual. "
              f"If an aspect is not supported by the evidence, say so explicitly.\n\n"
              f"Question: {state['question']}\n\nEvidence:\n{bullets}")
    return _chat(MODEL_SYNTHESIZER, prompt)


def _flare_augment(state: State, draft: str) -> str:
    """T4.2 — if the draft hedges on any claim, re-search for that claim and regenerate once."""
    if not ENABLE_ACTIVE_RETR or not _HEDGE_RE.search(draft):
        return draft
    hedge_match = _HEDGE_RE.search(draft)
    # Pull one sentence around the hedge as the re-search target.
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
    """Synthesize once (or N for self-consistency); optionally FLARE-augment any hedged output."""
    if not ENABLE_CONSISTENCY:
        draft = _synthesize_once(state)
        return {"answer": _flare_augment(state, draft), "trace": _merge_trace(state, "synthesize")}
    candidates = [_flare_augment(state, _synthesize_once(state)) for _ in range(CONSISTENCY_SAMPLES)]
    best = max(candidates, key=lambda a: _grounding_score(a, state.get("evidence_compressed") or state["evidence"]))
    return {"answer": best, "trace": _merge_trace(state, "synthesize")}


# ── Verify + iteration ────────────────────────────────────────────────

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


# ── Graph ─────────────────────────────────────────────────────────────

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


# ── CLI trace summary ─────────────────────────────────────────────────

def _print_trace_summary(trace: list[dict]) -> None:
    """One-pass summary: per-node and per-model totals."""
    if not trace:
        return
    by_node: dict[str, dict] = {}
    by_model: dict[str, dict] = {}
    total_latency = 0.0
    total_tokens = 0
    for e in trace:
        node = e.get("node", "?")
        model = e.get("model", "?")
        latency = float(e.get("latency_s", 0) or 0)
        tokens = int(e.get("tokens_est", 0) or 0)
        total_latency += latency
        total_tokens += tokens
        for bucket, key in ((by_node, node), (by_model, model)):
            b = bucket.setdefault(key, {"calls": 0, "latency_s": 0.0, "tokens_est": 0})
            b["calls"] += 1
            b["latency_s"] += latency
            b["tokens_est"] += tokens

    print(f"\n── trace summary ({len(trace)} entries, {total_latency:.2f}s total, ~{total_tokens} tokens) ──")
    print("  by node:")
    for node, b in sorted(by_node.items(), key=lambda kv: -kv[1]["latency_s"]):
        print(f"    {node:12s}  calls={b['calls']:2d}  latency={b['latency_s']:6.2f}s  tokens~{b['tokens_est']}")
    print("  by model:")
    for model, b in sorted(by_model.items(), key=lambda kv: -kv[1]["latency_s"]):
        print(f"    {model:22s}  calls={b['calls']:2d}  latency={b['latency_s']:6.2f}s  tokens~{b['tokens_est']}")


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is Anthropic's contextual retrieval and why does it reduce retrieval failures?"
    print(f"Q: {q}")
    result = build_graph().invoke({"question": q, "iterations": 0, "plan_rejects": 0, "trace": []})
    print(f"\n[class: {result.get('question_class', '?')}]")
    print(f"\nA: {result['answer']}")
    if result.get("claims"):
        v = sum(1 for c in result["claims"] if c["verified"])
        print(f"\nVerified: {v}/{len(result['claims'])} claims  (iterations: {result.get('iterations', 0)})")
    if ENABLE_TRACE:
        _print_trace_summary(result.get("trace", []))
