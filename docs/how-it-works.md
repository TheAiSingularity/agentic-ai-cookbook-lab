# How the research tool works

Three explanations of the same thing, at three depths. Pick the one that
fits your audience.

---

## 30-second pitch

It's a deep-research agent that answers hard questions with cited sources.
Decomposes your question, searches the web through a self-hosted
meta-search, reads the results, verifies each claim against the evidence,
and re-searches anything it's unsure about. Runs on any LLM — OpenAI,
local Ollama, or a GPU box with vLLM — no lock-in. When self-hosted,
**zero dollars per query.**

## 2-minute pitch

It's a **LangGraph pipeline with eight nodes**:

```
classify → plan → search → retrieve → fetch_url → compress → synthesize → verify
```

Every node is env-toggleable, so we can leave-one-out ablate every technique.

**Retrieval is 2026 SOTA:** BM25 + dense hybrid with Reciprocal Rank Fusion,
cross-encoder reranking (BAAI/bge-reranker-v2-m3), optional contextual
chunking from Anthropic. Search is **self-hosted SearXNG** meta-searching
DuckDuckGo / Bing / Wikipedia / arXiv — no paid API. Each result URL is
then fetched and cleaned with **trafilatura** so the agent reads the full
article, not just a snippet.

**Verification is the wedge.** After synthesis, a Chain-of-Verification
(CoVe) step decomposes the answer into discrete claims and checks each
against the evidence. If any fail, it re-searches for those specific
claims and regenerates. A **step-level critic** judges every major node's
output along the way — ThinkPRM-style. When the synthesizer hedges ("the
evidence does not specify..."), **FLARE** triggers a targeted re-search.

**Reproducible:** 12-config ablation matrix, 114 unit tests, runs on
an Apple M4 Mac or a 4×RTX 6000 Pro workstation with one command each.

**Fully observable.** Every LLM call records its model, latency, and token
cost into a per-query trace, printed at the end of the run. No SaaS
telemetry hook — the trace stays on your machine.

## Technical depth

The architecture composes five layers.

**Tier 1 — retrieval stack:**
BM25 + dense + RRF, optional cross-encoder rerank on top-50, Anthropic-style
contextual retrieval (up to −67% retrieval failures).

**Tier 2 — adaptive verification:**
- HyDE query rewriting, auto-gated on numeric queries
- Chain-of-Verification after synthesis
- Iterative retrieval bounded by `MAX_ITERATIONS`
- Self-consistency voting (opt-in)

**Tier 4 — 2026 SOTA techniques layered on top:**
- **T4.1** Step-level critic (ThinkPRM pattern) — judges each node's output
- **T4.2** FLARE active retrieval — re-search on hedged claims (+62% on 2Wiki in literature)
- **T4.3** Question classifier router — routes compute by `{factoid|multihop|synthesis}`
- **T4.4** LLM-based evidence compression (LongLLMLingua-lite, +17-21% literature gain)
- **T4.5** Plan refinement — regenerate decomposition when critic rejects (opt-in)

**Wave 4 — local-first engine enhancements:**
- **W4.1** Cross-encoder rerank — `HybridRetriever` top-50 → `bge-reranker-v2-m3` top-K. Apache-2.0, self-hosted, ~560MB one-time download.
- **W4.2** Full-page fetch — `trafilatura` pulls each result URL and extracts clean article text (beats Readability/Goose3 on TREC-HTML F1). Bounded concurrency; snippet fallback on failure.
- **W4.3** Observability trace — per-call `{node, model, latency_s, tokens_est}` recorded to `state["trace"]` and summarized at CLI end. No external telemetry.

**Tier 3 — evaluation infrastructure:**
12-config ablation runner (A1-A3 · B1-B4 · C1-C5), Pareto plotter, 4-metric
scorer (factuality via LLM-judge, citation-accuracy, citation-precision,
latency).

Everything talks to any OpenAI-compatible endpoint via `OPENAI_BASE_URL`.

**Thesis:** *adaptive verification composed on commodity open-weight LLMs
substitutes for trajectory-distillation training.*

---

## Comparison to SOTA (April 2026, honest)

| Capability | **GPT-5.4 Pro** (closed SOTA) | **MiroThinker-H1** (open) | **OpenResearcher-30B-A3B** | **Ours** |
|---|---|---|---|---|
| BrowseComp | **89.3** | 88.2 | 54.8 (BCP) | **TBD** — VM run pending |
| Open weights | ❌ | ✅ | ✅ | **✅** (uses any) |
| Zero $ / query | ❌ | ❌ hosted | ❌ hosted | **✅** when self-hosted |
| One-key setup | ❌ | ❌ | ❌ | **✅** (Ollama on Mac) |
| Step-level verification | ? | ✅ (H1's wedge) | ✅ | **✅** T4.1 |
| Adaptive compute routing | ? | ✅ | ✅ | **✅** T4.3 |
| FLARE-class active retrieval | ? | ? | partial | **✅** T4.2 |
| Evidence compression | ? | ? | ? | **✅** T4.4 |
| Multi-hop iterative retrieval | ✅ | ✅ | ✅ | **✅** Tier 2 |
| Self-consistency voting | ? | ? | ? | **✅** (opt-in) |
| Trajectory-distillation training | ✅ implied | ✅ | ✅ (96k × 100-turn) | ❌ **no training** |
| Public ablation matrix | ❌ | partial | partial | **✅** 12 configs |
| Portable (OpenAI/Ollama/vLLM/SGLang) | N/A | ❌ | ❌ | **✅** via OPENAI_BASE_URL |
| Full-page fetch (not just snippets) | ✅ | ✅ | ✅ | **✅** W4.2 trafilatura |
| Per-call observability without SaaS | ❌ | ? | ? | **✅** W4.3 local trace |
| Rust MCP tool case-study | N/A | N/A | N/A | **✅** |

### What SOTA has that we don't

- **Trajectory-trained weights.** OpenResearcher's jump from 20.8% → 54.8%
  on BrowseComp-Plus came from training on 96k research trajectories. We
  do none of that — we compose techniques on off-the-shelf models. That's
  the real gap.
- **GPT-5.4 Pro's raw scale.** At 89.3 BrowseComp, probably out of reach
  without training.

### What we have that SOTA doesn't

- **Full portability.** No one else ships a deep-research agent that runs
  identically on OpenAI, Ollama, vLLM, and SGLang via a single env var.
- **Explicit leave-one-out ablation.** The 12-config matrix (A1–C5) shows
  which techniques actually move the needle on a fixed model.
- **Zero-marginal-cost operation.** Queries are free once the GPU box
  is up.
- **A complete open-source stack in one repo.** Retrieval, verification,
  evaluation, paper draft, Rust MCP tool, reproducibility kit — all MIT.

### The realistic ceiling for this approach

Closing to **~80–85 on BrowseComp-Plus** with a commodity open 35B model
(`Qwen/Qwen3.6-35B-A3B`) and **zero model-specific training**. If we hit
that, it's a publishable result because it means *architectural
composition can substitute for specialized training* — the paper's thesis.

---

## TL;DR analogies

- **vs OpenAI Deep Research:** same idea, but you own the weights, the
  search, and the verification — no metered API, no vendor lock-in.
- **vs LangChain off-the-shelf RAG agents:** we've composed 9 specific
  2026 SOTA techniques on top, each independently benchmarkable.
- **vs MiroThinker-H1:** same architectural playbook (verification-integrated
  reasoning), but portable, reproducible, and running on commodity models
  rather than their specialized training.

---

## Where to look in the code

| What | Where |
|---|---|
| The core pipeline | `recipes/by-use-case/research-assistant/production/main.py` |
| Simple version | `recipes/by-use-case/research-assistant/beginner/main.py` (100 LOC) |
| Retrieval primitives | `core/rag/python/{hybrid,rerank,contextual,rag}.py` |
| Self-hosted search | `scripts/searxng/docker-compose.yml` |
| Ablation runner | `recipes/by-use-case/research-assistant/eval/ablation.py` |
| Pareto plot | `recipes/by-use-case/research-assistant/eval/pareto.py` |
| Paper draft | `docs/paper-draft.md` |
| Mac setup | `scripts/setup-local-mac.sh` |
| GPU VM setup | `scripts/setup-vm-gpu.sh` |
| Rust MCP case study | `recipes/by-pattern/rust-mcp-search-tool/` |
