"""Research assistant — SOTA agentic recipe (portable across providers).

LangGraph plan→search→retrieve→synthesize. Talks to any OpenAI-compatible
LLM endpoint (OpenAI, Ollama, vLLM, Groq…) via OPENAI_BASE_URL, and uses
self-hosted SearXNG for web search. See techniques.md for SOTA citations.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # let core.rag resolve

import requests  # noqa: E402
from core.rag import HybridRetriever, Retriever  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402
from openai import OpenAI  # noqa: E402

ENV = os.environ.get
MODEL_PLANNER, MODEL_SEARCHER, MODEL_SYNTHESIZER = (ENV("MODEL_PLANNER", "gpt-5-nano"), ENV("MODEL_SEARCHER", "gpt-5-mini"), ENV("MODEL_SYNTHESIZER", "gpt-5-mini"))
NUM_SUBQUERIES, NUM_RESULTS_PER_QUERY, TOP_K_EVIDENCE, SEARXNG_URL = int(ENV("NUM_SUBQUERIES", "3")), int(ENV("NUM_RESULTS_PER_QUERY", "5")), int(ENV("TOP_K_EVIDENCE", "8")), ENV("SEARXNG_URL", "http://localhost:8888")

State = TypedDict("State", {"question": str, "subqueries": list[str], "evidence": list[dict], "answer": str})


def _llm() -> OpenAI:  # honors OPENAI_BASE_URL → works with Ollama / vLLM / OpenAI
    return OpenAI(api_key=ENV("OPENAI_API_KEY", "ollama"), base_url=ENV("OPENAI_BASE_URL"))


def _plan(state: State) -> dict:
    """Break the question into N focused sub-queries (cheap planner model)."""
    prompt = (f"Break this research question into exactly {NUM_SUBQUERIES} focused sub-queries. "
              f"Return one per line, no numbering, no prose.\n\nQuestion: {state['question']}")
    resp = _llm().chat.completions.create(model=MODEL_PLANNER, messages=[{"role": "user", "content": prompt}])
    subs = [l.strip(" -•*") for l in (resp.choices[0].message.content or "").splitlines() if l.strip()][:NUM_SUBQUERIES]
    return {"subqueries": subs}


def _searxng(query: str, n: int = NUM_RESULTS_PER_QUERY) -> list[dict]:  # {url,title,snippet} via SearXNG JSON API
    r = requests.get(f"{SEARXNG_URL}/search", params={"q": query, "format": "json"}, timeout=20)
    r.raise_for_status()
    return [{"url": h.get("url", ""), "title": h.get("title", ""), "snippet": h.get("content", "")}
            for h in (r.json().get("results") or [])[:n]]


def _search_one(sub: str) -> list[dict]:
    """SearXNG search + LLM summarize-with-citations for a single sub-query."""
    hits = _searxng(sub)
    if not hits:
        return []
    sources = "\n".join(f"[{i+1}] {h['title']} — {h['snippet']}  (url: {h['url']})" for i, h in enumerate(hits))
    prompt = (f"Summarize these sources factually in 3-5 sentences, citing inline as [1], [2], etc. "
              f"Only use the information provided.\n\nSub-query: {sub}\n\nSources:\n{sources}")
    resp = _llm().chat.completions.create(model=MODEL_SEARCHER, messages=[{"role": "user", "content": prompt}])
    return [{"url": h["url"], "title": h["title"], "text": resp.choices[0].message.content or ""} for h in hits]


def _search(state: State) -> dict:
    """Fan-out: parallel search across sub-queries; dedupe evidence by URL."""
    with ThreadPoolExecutor(max_workers=NUM_SUBQUERIES) as pool:
        items = [e for batch in pool.map(_search_one, state["subqueries"]) for e in batch]
    seen: set[str] = set()
    return {"evidence": [e for e in items if e["url"] not in seen and not seen.add(e["url"])]}


def _retrieve(state: State) -> dict:
    """Retrieve top-k evidence via core/rag — RAG_VERSION=v0 (cosine) or v1 (hybrid, default)."""
    ev = state["evidence"]
    if len(ev) <= TOP_K_EVIDENCE:
        return {"evidence": ev}
    r = Retriever() if ENV("RAG_VERSION", "v1") == "v0" else HybridRetriever()
    r.add([e["text"] for e in ev])
    top = r.retrieve(state["question"], k=TOP_K_EVIDENCE)
    by_text = {e["text"]: e for e in ev}
    return {"evidence": [by_text[text] for text, _ in top if text in by_text]}


def _synthesize(state: State) -> dict:
    """Produce final cited answer using the synthesizer model."""
    bullets = "\n".join(f"[{i+1}] {e['text']}  (src: {e['url']})" for i, e in enumerate(state["evidence"]))
    prompt = (f"Answer using the evidence below. Cite sources inline as [1], [2], etc. Be concise and "
              f"factual.\n\nQuestion: {state['question']}\n\nEvidence:\n{bullets}")
    resp = _llm().chat.completions.create(model=MODEL_SYNTHESIZER, messages=[{"role": "user", "content": prompt}])
    return {"answer": resp.choices[0].message.content or ""}


def build_graph():
    """Wire the 4-node LangGraph: plan → search → retrieve → synthesize."""
    g = StateGraph(State)
    [g.add_node(n, f) for n, f in [("plan", _plan), ("search", _search), ("retrieve", _retrieve), ("synthesize", _synthesize)]]
    g.set_entry_point("plan")
    [g.add_edge(a, b) for a, b in [("plan", "search"), ("search", "retrieve"), ("retrieve", "synthesize"), ("synthesize", END)]]
    return g.compile()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What is Anthropic's contextual retrieval and why does it reduce retrieval failures?"
    print(f"Q: {q}\nA: {build_graph().invoke({'question': q, 'subqueries': [], 'evidence': [], 'answer': ''})['answer']}")
