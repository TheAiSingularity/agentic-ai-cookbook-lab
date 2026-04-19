<p align="center">
  <strong>agentic-ai-cookbook-lab</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/status-wave%202%20tier%204-brightgreen.svg" alt="Status">
  <img src="https://img.shields.io/badge/recipes-2-green.svg" alt="Recipes">
  <img src="https://img.shields.io/badge/tests-68%2F68-brightgreen.svg" alt="Tests">
  <img src="https://img.shields.io/badge/languages-python%20%2B%20rust-green.svg" alt="Languages">
</p>

**SOTA research agents that verify their own answers, runnable on any LLM backend, $0 per query when self-hosted.**

The OpenAI / Anthropic Cookbook model, applied to agentic research:
one opinionated, benchmarked implementation per task. No vendor lock-in —
everything talks to any OpenAI-compatible endpoint via one env var
(`OPENAI_BASE_URL`), so the same code runs on OpenAI, **Ollama** (Mac),
**vLLM** or **SGLang** (Linux GPU). Search is **self-hosted SearXNG** —
no paid search API.

For a fuller explanation at three depths (30-second / 2-minute /
technical) and an honest comparison vs GPT-5.4 Pro, MiroThinker-H1, and
OpenResearcher-30B-A3B, see [`docs/how-it-works.md`](docs/how-it-works.md).

---

## Recipes

| Recipe | Levels | What it does |
|---|---|---|
| [**research-assistant**](recipes/by-use-case/research-assistant/) | `beginner` (100 LOC) · `production` (384 LOC w/ 5 Tier 4 techniques) | Answers hard research questions with cited sources; decomposes → searches → verifies → iterates |
| [**trading-copilot**](recipes/by-use-case/trading-copilot/) | pending | Market research + alerts (research only — NOT auto-execution) |

Plus one Rust case-study recipe under `by-pattern/`:

- [**rust-mcp-search-tool**](recipes/by-pattern/rust-mcp-search-tool/) — ~5 MB static MCP server wrapping SearXNG; demonstrates where Rust genuinely earns its place in the agent stack (and where it doesn't).

## The research-assistant stack — what's actually shipped

Four tiers of techniques, each env-toggleable so leave-one-out ablations
are trivial. **68/68 unit tests green**, all mocked (no network or API
key required for `pytest`).

**Tier 1 — retrieval** (lives in `core/rag/`)
- BM25 + dense embeddings + Reciprocal Rank Fusion (`HybridRetriever`)
- Cross-encoder reranker (lazy-loaded `BAAI/bge-reranker-v2-m3`)
- Anthropic-style contextual chunking

**Tier 2 — adaptive verification** (production tier)
- HyDE query rewriting, auto-gated on numeric queries
- Chain-of-Verification after synthesis
- Iterative retrieval bounded by `MAX_ITERATIONS`
- Self-consistency voting (opt-in)

**Tier 3 — reproducibility** (eval harness)
- 12-config ablation runner (`eval/ablation.py`)
- Pareto plotter (`eval/pareto.py`)
- Four metrics: factuality (LLM-judge), citation-accuracy, citation-precision, latency

**Tier 4 — 2026 SOTA layered on top**
- T4.1 ThinkPRM-style step-level critic after plan + search
- T4.2 FLARE active retrieval on hedged claims
- T4.3 Question classifier router → compute scales with difficulty
- T4.4 LLM-based evidence compression before synthesize
- T4.5 Plan refinement when critic rejects decomposition (opt-in)

## Three ways to run it

```bash
git clone https://github.com/TheAiSingularity/agentic-ai-cookbook-lab
cd agentic-ai-cookbook-lab
```

**Mac, fully local (free):**
```bash
bash scripts/setup-local-mac.sh     # installs Ollama + gemma4:e2b + nomic-embed + SearXNG
cd recipes/by-use-case/research-assistant/beginner
make install
# env exported by setup script; or set manually per beginner/README.md
make smoke
```
On an Apple M4 Pro this runs end-to-end in ~40 s / query. Zero dollars.

**GPU VM (4× RTX 6000 Pro, self-hosted):**
```bash
bash scripts/setup-vm-gpu.sh --engine sglang --spec-dec \
  --model Qwen/Qwen3.6-35B-A3B
# exports env; then
cd recipes/by-use-case/research-assistant/production
make smoke
```
Uses SGLang's RadixAttention prefix caching + EAGLE speculative
decoding for near-frontier throughput on a commodity rig.

**OpenAI (pay-per-query):**
```bash
cd scripts/searxng && docker compose up -d
export OPENAI_API_KEY=sk-...
cd recipes/by-use-case/research-assistant/beginner && make install && make smoke
```
Same code. Different `OPENAI_BASE_URL`.

## Repo layout

```
recipes/
  by-use-case/
    research-assistant/      # beginner + production tiers, eval harness, 12-config ablation
    trading-copilot/         # pending
  by-pattern/
    rust-mcp-search-tool/    # Rust case study

core/
  rag/                       # HybridRetriever · CrossEncoderReranker · contextualize_chunks

scripts/
  searxng/                   # docker-compose for self-hosted meta-search
  setup-local-mac.sh         # Mac dev stack (Ollama)
  setup-vm-gpu.sh            # Linux GPU stack (vLLM or SGLang, + optional EAGLE)

docs/
  how-it-works.md            # elevator pitches + SOTA comparison (read this first)
  paper-draft.md             # arXiv tech report skeleton, methodology, ablation matrix

foundations/                 # OpenClaw / OpenShell / NemoClaw / Hermes Agent explainers
```

## Status

- **Wave 0** skeleton · **Wave 0.5** SOTA-per-task pivot · **Wave 1** beginner recipe live · **Wave 2** tiers 1/2/3/4 — **all shipped.**
- Pending work is on the user's end: run the full 12-config ablation on the GPU VM with SimpleQA-100 + BrowseComp-Plus-50 to produce the paper's numbers.

## Related

- [HermesClaw](https://github.com/TheAiSingularity/hermesclaw) — the secure runtime these recipes can run inside
- [NVIDIA/OpenShell](https://github.com/NVIDIA/OpenShell) — kernel-level agent sandbox
- [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) — self-improving agent

MIT licensed.
