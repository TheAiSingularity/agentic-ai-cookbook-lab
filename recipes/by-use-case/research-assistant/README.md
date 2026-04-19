# research-assistant

**Levels:** beginner ⬛ · production ⬛ · rust ⬛ · **all pending — ships in Wave 1**

## What it does
Answers research questions by combining web search, page-fetching, and an LLM reasoning loop. Given a question like "what are the tradeoffs between ColBERT and BGE rerankers in 2026?", it searches, reads, synthesizes, and produces a cited answer.

## Who it's for
Anyone who needs a production-grade research agent that's cheap to run, accurate, and easy to extend. Clone it, swap the eval set for your domain, and you have a custom research tool. This is also the canonical reference implementation for LangGraph + contextual-RAG + Exa — the 2026 SOTA research stack.

## Why you'd use it
- **Cheap and accurate:** Gemini 2.5 Flash + Exa highlights keeps per-query cost at $0.01–$0.03
- **Reproducible quality:** ships with a 10-question eval set and a factuality + citation-precision scorer
- **Production path included:** graduate to `production/` for HermesClaw-sandboxed execution with observability

## SOTA stack (April 2026)

| Component | Choice | Rationale |
|---|---|---|
| **Orchestration** | LangGraph | Lowest token overhead for stateful search-and-synthesize loops. Production-standard Python framework. |
| **Web search** | Exa | 81% vs Tavily 71% on complex retrieval; 2–3× faster; sends 50–75% fewer tokens to the LLM via query-dependent highlights |
| **Retrieval** | `core/rag/` (contextual retrieval + BM25 + dense hybrid + cross-encoder rerank) | Anthropic contextual retrieval reduces retrieval failures by 67%; two-stage pipeline hits Recall@5 0.816 |
| **LLM (default)** | Gemini 2.5 Flash | $0.25/$1.50 per M tokens, 1M context, 363 tok/s. Independently rated "best cheap model for high-volume" March 2026. |
| **LLM (hard steps)** | GPT-5 mini | Highest agentic-task accuracy (OSWorld-Verified 72.2) — used only when reasoning complexity demands it |

Pattern: plan → Exa search → Exa get-contents (highlights) → synthesize → iterate → cited answer.

See [`beginner/techniques.md`](beginner/techniques.md) for primary-source citations on every choice. *(Lands Wave 1.)*

## Eval

10 research questions × gold answers. Scorer measures:
- **Factuality** — answer vs gold, via LLM-as-judge with citations
- **Citation precision** — proportion of cited sources actually supporting claims

`make eval` reproduces the score.

## Expected cost per query
$0.01–$0.03 at defaults (Gemini Flash-Lite + Exa + one rerank pass).

## See also
- [`../../../core/rag/`](../../../core/rag/) — the retrieval module this recipe pulls from
- [`../../../foundations/what-is-hermes-agent.md`](../../../foundations/what-is-hermes-agent.md) — context on the agent runtime
- [`../../../comparisons/rag-sota-2026.md`](../../../comparisons/) — landscape page explaining why this retrieval stack won
