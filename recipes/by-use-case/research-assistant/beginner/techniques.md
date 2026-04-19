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
  — LangGraph achieves the lowest latency and token usage in head-to-head benchmarks.

## Inference: OpenAI-compatible endpoint (portable)

**Why:** Every modern LLM server — OpenAI's own API, Ollama, vLLM, Groq,
Together — exposes the same `/v1/chat/completions` shape. The recipe
depends only on that interface (via the `openai` Python SDK), so
switching between paid cloud, on-prem GPU, and Apple-Silicon local
inference is one env var: `OPENAI_BASE_URL`.

The local paths use:
- **Ollama** (macOS / dev) — simplest to run on Apple Silicon; good at mid-size open-weight models
- **vLLM** (Linux GPU) — highest-throughput open-source inference server; tensor-parallel across multiple GPUs

- [Ollama OpenAI-compat docs](https://github.com/ollama/ollama/blob/main/docs/openai.md)
- [vLLM OpenAI-compat server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

## Web search: SearXNG (self-hosted meta-search)

**Why:** No paid API key. No rate limits you didn't set yourself.
SearXNG queries DuckDuckGo, Bing, Wikipedia, arXiv, and ~80 other
engines in parallel and aggregates results. JSON API makes it a drop-in
replacement for Exa / Tavily / Brave. Running it in Docker on `:8888`
adds one `docker compose up` to setup.

The recipe's `search` node fetches SearXNG results and then asks the
searcher LLM to summarize them with inline `[N]` citations. That pattern
is deterministic, eval-friendly (you can hold search fixed and vary the
LLM), and works identically across every backend.

- [SearXNG project](https://github.com/searxng/searxng)
- [SearXNG JSON API docs](https://docs.searxng.org/user/search_api.html)

## Retrieval ranking: `core/rag` v1 — hybrid BM25 + dense + RRF

**Why hybrid:** Sparse (BM25) catches exact tokens, rare strings, and
numbers that dense embeddings miss. Dense catches paraphrase that BM25
misses. Reciprocal Rank Fusion (k=60) combines the two rank lists
without needing score normalization — the SOTA fusion choice in 2026.

**Why two-stage (future, production tier):** Beginner uses hybrid only.
Production tier layers a cross-encoder reranker on top-50 candidates
(BAAI/bge-reranker-v2-m3). Benchmarked: Recall@5 0.816, MRR@3 0.605 —
outperforms every single-stage method. ~25% token reduction on downstream
synthesis because top-8 is tighter.

**Contextual retrieval (future, at-index-time):** Prepend a 1-2 sentence
LLM-generated context to each chunk before indexing. Anthropic: **−35%
retrieval failures from contextual embeddings alone, −49% combined with
contextual BM25, −67% when also reranked.** Adds one cheap LLM call per
chunk at ingest; zero cost at query time. Available as
`core.rag.contextualize_chunks`.

- [core/rag README](../../../../core/rag/README.md) — public API and usage
- [core/rag BENCHMARKS](../../../../core/rag/BENCHMARKS.md) — live numbers
- [RAG review 2025–2026 (RAGFlow)](https://ragflow.io/blog/rag-review-2025-from-rag-to-context)
- [Benchmarking retrieval for financial docs (arXiv)](https://arxiv.org/html/2604.01733)
- [Anthropic contextual retrieval](https://www.anthropic.com/news/contextual-retrieval)

## LLM routing (three-tier)

**Why routing:** Sending simple tasks to a cheap tier and harder tasks
to a more capable tier reduces cost 50–80% without measurable quality
loss on the overall workflow.

- **Plan** is one short LLM call producing a list of sub-queries. A
  nano / 2B-class model handles it fine.
- **Search** summarizes 5 results per sub-query with citations. Needs
  reasoning and citation discipline — mid-tier model is the sweet spot.
- **Synthesize** is the final answer over all kept evidence. Needs the
  strongest reasoning available — mid-tier or above.

Pick the exact model names via `MODEL_PLANNER` / `MODEL_SEARCHER` /
`MODEL_SYNTHESIZER`. Defaults match OpenAI SKUs that exist today.

- [Artificial Analysis model leaderboard](https://artificialanalysis.ai/leaderboards/models)
- [LM Council benchmarks](https://lmcouncil.ai/benchmarks)

## Parallel fan-out

**Why:** The search step runs one call per sub-query, each doing a
SearXNG query + LLM summarization. Serial fan-out of 3 sub-queries is
~3× slower than necessary. We parallelize with `ThreadPoolExecutor` (not
async) because the `openai` SDK's sync client is thread-safe and
threading is the minimum complexity increment for IO-bound parallelism.

## Evidence dedup by URL

**Why:** Fan-out across N sub-queries frequently surfaces the same URL
multiple times (different sub-queries, same canonical source). The
search node dedupes by URL before retrieval runs, so top-k isn't
wasted on duplicates. Pure post-processing, zero LLM cost.

## Going further — production tier

The beginner tier stops at one-shot synthesis. The [production
tier](../production/README.md) adds two stacked layers:

**Tier 2 — adaptive verification:**
- **HyDE** query rewriting (gated against numeric queries)
- **Chain-of-Verification** — claim-by-claim vs evidence
- **Iterative retrieval** — re-search unverified claims
- **Self-consistency voting** (opt-in)

**Tier 4 — 2026 SOTA layered on top:**
- **T4.1 step-level critic** — ThinkPRM-style judgment after each node
- **T4.2 FLARE active retrieval** — re-search mid-synthesis when a claim is hedged (+62% on multi-hop 2Wiki in literature)
- **T4.3 classifier router** — compute scales with question difficulty
- **T4.4 evidence compression** — LongLLMLingua-style, +17-21% accuracy with 4× fewer tokens
- **T4.5 plan refinement** — replan when critic rejects decomposition (opt-in)

Target: close the gap to MiroThinker-H1 (88.2 BrowseComp) using
commodity open-weight LLMs without any model-specific training. See
`eval/ablation.py` for the 12-config ablation matrix.

## What nobody tells you

**The cheapest agent in 2026 is one that runs on your own GPU.**
OpenAI's `web_search` tool, Tavily, Exa — all priced per call. At
hobby volume they're trivial. At production scale, every per-query fee
compounds. A vLLM + SearXNG rig on a workstation pays zero marginal
cost per query after setup. That's the reason this recipe's defaults
are local-first: we optimize for *zero-marginal-cost* research at the
edge where agents actually get used — iterative experimentation, eval
loops, long-running research jobs.
