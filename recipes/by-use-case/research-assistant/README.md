# research-assistant

**Levels:** beginner ✅ · production ⬛ · rust ⬛ · **beginner tier shipped**

## What it does
Answers research questions by combining web search, evidence retrieval, and an LLM reasoning loop. Given a question like "what are the tradeoffs between ColBERT and BGE rerankers in 2026?", it searches, reads, synthesizes, and produces a cited answer.

## Who it's for
Anyone who needs a production-grade research agent that's cheap to run, accurate, and easy to extend. Clone it, swap the eval set for your domain, and you have a custom research tool. Canonical reference implementation for LangGraph + `core/rag` + OpenAI's `web_search` tool.

## Why you'd use it
- **One key:** just `OPENAI_API_KEY` — no Exa / Tavily / Gemini account needed
- **Parallel fan-out:** 3 sub-queries run concurrently → ~60–90s wall-clock for a typical query
- **Reproducible quality:** ships with an eval harness (gold answers + factuality + citation-precision scorer)
- **Production path included:** graduate to `production/` for HermesClaw-sandboxed execution with observability

## SOTA stack (OpenAI-only)

| Component | Choice | Rationale |
|---|---|---|
| **Orchestration** | LangGraph | Lowest token overhead for stateful workflows |
| **Planner** | `gpt-5-nano` | Cheapest tier — sub-query generation doesn't need reasoning muscle |
| **Searcher** | `gpt-5-mini` + `web_search` tool (Responses API) | Built-in multi-step web search; no second API key |
| **Retrieval** | `core/rag` v0 (OpenAI embeddings + cosine) | Narrows many highlights to top-k most relevant |
| **Synthesizer** | `gpt-5-mini` | Strong reasoning where it matters most — the final cited answer |

Pattern: plan → parallel `web_search` across sub-queries → retrieve → synthesize with citations.

See [`beginner/techniques.md`](beginner/techniques.md) for primary-source citations on every choice.

## Eval

Seeded with 3 research questions; full 10-question eval lands alongside closing [THE-89](https://linear.app/theaisingularity/issue/THE-89). Scorer measures:
- **Factuality** — LLM-as-judge rating of candidate vs gold answer
- **Citation precision** — proportion of required source domains actually cited

`make eval` reproduces the scores.

## Expected cost per query
**~$0.05–$0.25** — dominated by OpenAI's `web_search` tool per-call fee. Nano planner + cosine retrieval + a single synthesize call are pennies; the 3 parallel `web_search` invocations carry the bulk. Cheaper per-token routes exist (e.g., Exa + Gemini), but they require two more keys. The cookbook's default prioritizes one-key simplicity — see `techniques.md` for a note on when to swap back.

## See also
- [`../../../core/rag/`](../../../core/rag/) — the retrieval module this recipe pulls from
- [`../../../foundations/what-is-hermes-agent.md`](../../../foundations/what-is-hermes-agent.md) — context on the agent runtime
- [`../../../comparisons/rag-sota-2026.md`](../../../comparisons/) — landscape page explaining why this retrieval stack won (lands Wave 2)
