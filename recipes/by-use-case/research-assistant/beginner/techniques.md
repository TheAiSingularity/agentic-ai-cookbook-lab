# Techniques — research-assistant/beginner

Every SOTA choice in this recipe, with a primary-source link. When a newer
benchmark changes the answer, update this file and the recipe together.

---

## Framework: LangGraph

**Why:** Lowest token overhead among 2026 Python agent frameworks for
stateful workflows. Graph nodes with direct state transitions avoid the
repeated chat-history passing that inflates token costs in loop-oriented
frameworks. Clean fit for `plan → search → retrieve → synthesize`.

- [2026 AI Agent Framework Decision Guide (dev.to)](https://dev.to/linou518/the-2026-ai-agent-framework-decision-guide-langgraph-vs-crewai-vs-pydantic-ai-b2h)
  — LangGraph achieves the lowest latency and token usage in head-to-head
  benchmarks.

## Web search: OpenAI's built-in `web_search` tool (Responses API)

**Why:** Removes the need for a second API key (Exa, Tavily, Brave, etc.).
The tool is invoked from the model's own reasoning loop — the searcher
model (gpt-5-mini) decides when to search, issues multiple queries as
needed, and returns a synthesized answer with URL citations in the
response's `annotations`.

Trade-off: `web_search` is bundled into the LLM call, so you pay both
the model tokens *and* a per-search fee. Typical runs do 2–4 internal
searches per call. Cheaper than piecing together Exa + Gemini if you value
one-key simplicity; possibly more expensive for extremely high volume.

- [OpenAI Responses API + tools (docs)](https://platform.openai.com/docs/guides/responses)
- [OpenAI web_search tool reference](https://platform.openai.com/docs/guides/tools-web-search)

## Retrieval ranking: `core/rag` v0 (cosine), upgrading to hybrid + rerank

**Why v0 now:** Establishes the API surface recipes will depend on. Naive
cosine is fine when your evidence set is already query-focused (which
happens here — each sub-query's `web_search` response is already scoped).

**Why v1 next:** SOTA retrieval in 2026 is a two-stage pipeline: hybrid
(BM25 + dense) with reciprocal-rank-fusion followed by cross-encoder
reranking on top-50. Benchmarked at Recall@5 0.816 and MRR@3 0.605,
outperforming all single-stage methods. Contextual retrieval at indexing
time reduces retrieval failures by up to 67%.

- [RAG review 2025–2026 (RAGFlow)](https://ragflow.io/blog/rag-review-2025-from-rag-to-context)
- [Benchmarking retrieval for financial docs (arXiv)](https://arxiv.org/html/2604.01733)
- [Advanced RAG patterns 2026 (dev.to)](https://dev.to/young_gao/rag-is-not-dead-advanced-retrieval-patterns-that-actually-work-in-2026-2gbo)

## LLM routing: gpt-5-nano for plan, gpt-5-mini for search + synthesize

**Why routing:** Sending simple tasks to the cheapest tier and harder ones
to a more capable model reduces cost 50–80% while maintaining quality.
The planner just needs to list sub-queries — that's a trivial LLM call,
and the nano tier handles it fine. Search and synthesis need real
reasoning, so we route them to mini.

**Tier availability:** Model names (`gpt-5-nano`, `gpt-5-mini`) are what
exists on a standard OpenAI account today. Override via
`MODEL_PLANNER` / `MODEL_SEARCHER` / `MODEL_SYNTHESIZER` env vars when
newer tiers ship or when you want to dial quality up (e.g., `gpt-5` for
synthesize on high-stakes questions).

- [Artificial Analysis model leaderboard](https://artificialanalysis.ai/leaderboards/models)
- [LM Council benchmarks](https://lmcouncil.ai/benchmarks)
- [BenchLM: Best Budget LLMs 2026](https://benchlm.ai/blog/posts/best-budget-llms-2026)

## Parallel fan-out

**Why:** The search step runs one call per sub-query, each doing its own
multi-step web search. Serial fan-out of 3 sub-queries takes ~120–150s.
We parallelize with `ThreadPoolExecutor` (not async) because the
`openai` SDK's sync client is thread-safe and threading is the minimum
complexity increment for IO-bound parallelism.

Parallel fan-out of 3 sub-queries typically runs in 45–70s — roughly
matching a single search call's own latency (since they overlap).

## What nobody tells you

**Getting rid of Exa trades per-token cost for per-search fees.**
OpenAI's `web_search` tool has a per-call charge on top of token costs.
For a hobby agent doing dozens of queries a day, that's trivial. For a
production system serving thousands of queries, the economics may flip —
at which point swapping back in Exa + Gemini (two cheap APIs) can win on
marginal cost. The single-key OpenAI-only design is optimized for
developer experience first, not raw cost ceiling. When you hit scale,
benchmark both.
