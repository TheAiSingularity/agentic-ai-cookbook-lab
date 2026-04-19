"""Research assistant — SOTA agentic recipe (OpenAI-only).

LangGraph plan→search→retrieve→synthesize. Single OpenAI key covers every
step via model routing + the web_search tool. Override models via
MODEL_PLANNER / MODEL_SEARCHER / MODEL_SYNTHESIZER env vars.
See techniques.md for the SOTA choices with primary-source citations.
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # let core.rag resolve

from core.rag import Retriever  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402
from openai import OpenAI  # noqa: E402

MODEL_PLANNER = os.getenv("MODEL_PLANNER", "gpt-5-nano")
MODEL_SEARCHER = os.getenv("MODEL_SEARCHER", "gpt-5-mini")
MODEL_SYNTHESIZER = os.getenv("MODEL_SYNTHESIZER", "gpt-5-mini")
NUM_SUBQUERIES = int(os.getenv("NUM_SUBQUERIES", "3"))
TOP_K_EVIDENCE = int(os.getenv("TOP_K_EVIDENCE", "8"))

State = TypedDict("State", {"question": str, "subqueries": list[str], "evidence": list[dict], "answer": str})


def _plan(state: State) -> dict:
    """Break the question into N focused sub-queries (cheap planner model)."""
    prompt = (f"Break this research question into exactly {NUM_SUBQUERIES} focused sub-queries. "
              f"Return one per line, no numbering, no prose.\n\nQuestion: {state['question']}")
    resp = OpenAI().chat.completions.create(model=MODEL_PLANNER, messages=[{"role": "user", "content": prompt}])
    subs = [l.strip(" -•*") for l in (resp.choices[0].message.content or "").splitlines() if l.strip()][:NUM_SUBQUERIES]
    return {"subqueries": subs}


def _search_one(sub: str) -> list[dict]:
    """One Responses + web_search call for a single sub-query; returns evidence items."""
    resp = OpenAI().responses.create(model=MODEL_SEARCHER, tools=[{"type": "web_search"}],
        input=f"Research this and return a concise, factual summary with source URLs: {sub}")
    items: list[dict] = []
    for item in resp.output:
        for block in getattr(item, "content", []) or []:
            if getattr(block, "type", None) != "output_text":
                continue
            urls = [a.url for a in (block.annotations or []) if getattr(a, "type", None) == "url_citation"]
            items.extend({"url": u, "title": u, "text": block.text} for u in urls or [""])
    return items


def _search(state: State) -> dict:
    """Fan-out: search each sub-query in parallel; collect evidence across all."""
    with ThreadPoolExecutor(max_workers=NUM_SUBQUERIES) as pool:
        results = list(pool.map(_search_one, state["subqueries"]))
    return {"evidence": [e for items in results for e in items]}


def _retrieve(state: State) -> dict:
    """Use core/rag to pick the top-k most relevant evidence for synthesis."""
    ev = state["evidence"]
    if len(ev) <= TOP_K_EVIDENCE:
        return {"evidence": ev}
    r = Retriever()
    r.add([e["text"] for e in ev])
    top = r.retrieve(state["question"], k=TOP_K_EVIDENCE)
    by_text = {e["text"]: e for e in ev}
    return {"evidence": [by_text[text] for text, _ in top]}


def _synthesize(state: State) -> dict:
    """Produce final cited answer using the synthesizer model."""
    bullets = "\n".join(f"[{i+1}] {e['text']}  (src: {e['url']})" for i, e in enumerate(state["evidence"]))
    prompt = (f"Answer using the evidence below. Cite sources inline as [1], [2], etc. Be concise "
              f"and factual.\n\nQuestion: {state['question']}\n\nEvidence:\n{bullets}")
    resp = OpenAI().chat.completions.create(model=MODEL_SYNTHESIZER, messages=[{"role": "user", "content": prompt}])
    return {"answer": resp.choices[0].message.content or ""}


def build_graph():
    """Wire the 4-node LangGraph: plan → search → retrieve → synthesize."""
    g = StateGraph(State)
    for name, fn in [("plan", _plan), ("search", _search), ("retrieve", _retrieve), ("synthesize", _synthesize)]:
        g.add_node(name, fn)
    g.set_entry_point("plan")
    for a, b in [("plan", "search"), ("search", "retrieve"), ("retrieve", "synthesize"), ("synthesize", END)]:
        g.add_edge(a, b)
    return g.compile()


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What is Anthropic's contextual retrieval and why does it reduce retrieval failures?"
    print(f"Q: {question}\n")
    result = build_graph().invoke({"question": question, "subqueries": [], "evidence": [], "answer": ""})
    print(f"A: {result['answer']}")
