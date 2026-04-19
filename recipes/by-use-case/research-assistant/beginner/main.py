"""Research assistant — SOTA agentic recipe.

LangGraph pipeline: plan → search → retrieve → synthesize.
Stack: Gemini 2.5 Flash (plan) · Exa highlights (search) · core/rag
(retrieve) · GPT-5 mini (synthesize). ~$0.01–$0.03 per query.
Override model names via MODEL_PLANNER / MODEL_SYNTHESIZER env vars.
See techniques.md for the SOTA choices with primary-source citations.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TypedDict

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))  # let core.rag resolve

from core.rag import Retriever  # noqa: E402
from exa_py import Exa  # noqa: E402
from google import genai  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402
from openai import OpenAI  # noqa: E402

MODEL_PLANNER = os.getenv("MODEL_PLANNER", "gemini-2.5-flash")
MODEL_SYNTHESIZER = os.getenv("MODEL_SYNTHESIZER", "gpt-5-mini")
SEARCH_RESULTS_PER_QUERY = int(os.getenv("SEARCH_RESULTS_PER_QUERY", "3"))
TOP_K_HIGHLIGHTS = int(os.getenv("TOP_K_HIGHLIGHTS", "8"))


State = TypedDict("State", {"question": str, "subqueries": list[str], "evidence": list[dict], "answer": str})


def _plan(state: State) -> dict:
    """Break the question into 3 focused sub-queries (cheap planner model)."""
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    prompt = (
        f"Break this research question into exactly 3 focused sub-queries that, "
        f"together, would produce a well-cited answer. Return one sub-query per "
        f"line, no numbering, no prose.\n\nQuestion: {state['question']}"
    )
    resp = client.models.generate_content(model=MODEL_PLANNER, contents=prompt)
    subs = [line.strip(" -•*") for line in resp.text.splitlines() if line.strip()][:3]
    return {"subqueries": subs}


def _search(state: State) -> dict:
    """Run each sub-query through Exa; collect highlight snippets as evidence."""
    exa = Exa(api_key=os.environ["EXA_API_KEY"])
    evidence: list[dict] = []
    for sub in state["subqueries"]:
        resp = exa.search_and_contents(
            sub, num_results=SEARCH_RESULTS_PER_QUERY, highlights=True, type="auto"
        )
        for r in resp.results:
            for hl in (r.highlights or []):
                evidence.append({"url": r.url, "title": r.title or r.url, "text": hl})
    return {"evidence": evidence}


def _retrieve(state: State) -> dict:
    """Use core/rag to pick the top-k most relevant highlights for synthesis."""
    ev = state["evidence"]
    if len(ev) <= TOP_K_HIGHLIGHTS:
        return {"evidence": ev}
    r = Retriever()
    r.add([e["text"] for e in ev])
    top = r.retrieve(state["question"], k=TOP_K_HIGHLIGHTS)
    by_text = {e["text"]: e for e in ev}
    return {"evidence": [by_text[text] for text, _ in top]}


def _synthesize(state: State) -> dict:
    """Produce final cited answer using the stronger reasoning model."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    bullets = "\n".join(f"[{i+1}] {e['text']}  (src: {e['url']})" for i, e in enumerate(state["evidence"]))
    prompt = (
        f"Answer using the evidence below. Cite sources inline as [1], [2], etc. "
        f"Be concise and factual.\n\nQuestion: {state['question']}\n\nEvidence:\n{bullets}"
    )
    resp = client.chat.completions.create(model=MODEL_SYNTHESIZER, messages=[{"role": "user", "content": prompt}])
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
