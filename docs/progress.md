# Progress — `agentic-ai-cookbook-lab`

One-page summary of what's been built, with links into the details.
For the elevator pitch and SOTA comparison, see
[`how-it-works.md`](how-it-works.md). For the research-paper skeleton,
see [`paper-draft.md`](paper-draft.md).

---

## Current state (Wave 4 shipped)

| | |
|---|---|
| Recipes live | **research-assistant** (beginner + production + eval) · **trading-copilot** (beginner + production + eval) |
| Case-study recipe | **rust-mcp-search-tool** |
| Core shared library | `core/rag/` v1 — `HybridRetriever` · `CrossEncoderReranker` · `contextualize_chunks` |
| Tests | **114/114 green**, all mocked (no network / no API keys) |
| Portable stack | OpenAI · Ollama · vLLM · SGLang — one env var (`OPENAI_BASE_URL`) |
| Search | Self-hosted **SearXNG** (Docker) — no paid API |
| Full-page fetch | **trafilatura** — self-hosted clean-text article extraction |
| Reranking | **BAAI/bge-reranker-v2-m3** via `sentence-transformers` — Apache-2.0, local |
| Observability | Per-call trace (node, model, latency, tokens) — no SaaS |
| Repo visibility | Private |
| License | MIT |

---

## Wave-by-wave history

### Wave 0 — skeleton (initial scaffold)
Directory layout, issue templates, CI stubs, LICENSE, CONTRIBUTING, scripts/searxng/, initial setup-local-mac.sh / setup-vm-gpu.sh.

### Wave 0.5 — SOTA-per-task pivot (DEC-006)
Dropped the 4-framework-comparison matrix. Each recipe is now one opinionated SOTA implementation. This is the OpenAI Cookbook / Anthropic Cookbook model.

### Wave 1 — research-assistant beginner
First runnable recipe. `plan → search → retrieve → synthesize`. 100-LOC single file. Tested live end-to-end with OpenAI.

### Wave 2 — research-assistant full SOTA stack

**Tier 1** — `core/rag/` v1: BM25 + dense + RRF hybrid retrieval, lazy cross-encoder reranker, Anthropic-style contextual chunking.

**Tier 2** — production tier: HyDE · CoVe · iterative retrieval · self-consistency.

**Tier 3** — ablation harness: 12-config runner, Pareto plotter, SimpleQA/BrowseComp-Plus fixtures (seeds), backtest scorer with 4 metrics.

**Tier 4** — 2026 SOTA layered on top: ThinkPRM-style step critic · FLARE active retrieval · question-classifier router · LongLLMLingua-style evidence compression · plan refinement. Each env-gated.

[Wave 2 details + citations](paper-draft.md)

### Wave 3 — trading-copilot recipe end-to-end

Second flagship recipe in one session. Beginner + production + eval harness + backtest.

- Beginner: `load_config → gather → analyze → skeptic → alert_router`, yfinance + `ta` + SearXNG news + webhook adapters (Slack/Telegram/Discord) with stdout fallback.
- Production: step critic (T4.1) · self-consistency skeptic · CoVe-style `verify_alerts` (claims checked against raw data) · optional PRAW social layer.
- Eval: pandas-only backtest scorer with signal precision/recall, sample_window.yaml (6 months × 3 tickers).
- Safety: build-time forbidden-symbols tests fail if anyone adds execution semantics (`place_order` / `alpaca` / `ib_insync` / etc.).
- [DEC-009](../.project/decisions.md) documents techniques we deliberately skipped from research-assistant (HyDE, FLARE, compression, plan refinement, classifier router) because they don't transfer to structured-data monitoring.

### Wave 4 — research-assistant local-first engine enhancements

Three shippable upgrades to the production research pipeline, all running
on fully open-source / self-hostable stacks. No paid API is required for
any of them.

| Enhancement | What it does | Env gate | Default |
|---|---|---|---|
| **W4.1 · Cross-encoder rerank** | `HybridRetriever` was shipped in Wave 2 Tier 1 but its two-stage partner `CrossEncoderReranker` (BAAI/bge-reranker-v2-m3) was never wired in. Now `_retrieve` runs hybrid → top-50 → cross-encoder → top-K. Graceful fallback to hybrid-only if the model can't load. | `ENABLE_RERANK` | `0` (opt-in — first run downloads ~560MB) |
| **W4.2 · Full-page fetch** | New `_fetch_url` node between `retrieve` and `compress`. Uses `trafilatura` to download + clean-text-extract each evidence URL, replacing SearXNG's 200-char snippets with full articles. Bounded concurrency; per-URL failures fall back to the snippet. | `ENABLE_FETCH` | `1` (on) |
| **W4.3 · Observability trace** | Every `_chat` call records `{node, model, latency_s, tokens_est, prompt_chars, response_chars}` into `state["trace"]`. CLI prints per-node and per-model totals at the end. Makes local debugging + ablation work actually tractable. | `ENABLE_TRACE` | `1` (on) |

16 new mocked tests cover reranker wiring (passthrough / hybrid-only /
rerank-on / fallback-on-failure), fetch_url (disabled / success / failure
/ max-URLs cap / empty-evidence), and trace (chat instrumentation /
node-tagged drain / extras merge / full-graph recording / summary
rendering). Brings the repo-wide total from 98 → **114 green**.

---

## Live verification (Mac / Ollama / gemma4:e2b / SearXNG)

Every recipe has been run end-to-end on a Mac M4 Pro with local models and real data sources. No paid APIs. All smokes pass.

| Recipe | Wall clock | Note |
|---|---|---|
| research-assistant beginner | ~40 s | Full plan → search → retrieve → synthesize cycle |
| research-assistant production (Wave 2 T4) | ~116 s | + classify + critic + FLARE + compress, 4/4 claims verified |
| trading-copilot beginner | 44 s | 3 tickers scanned, 0 false alerts (correct — rules are strict) |
| trading-copilot production | 46 s | Full stack + critic notes captured |
| trading-copilot backtest | 21 s | 3 scan dates on AAPL Feb 2024, reproducible metrics |

---

## How someone new would land and use this

1. **Read [`how-it-works.md`](how-it-works.md) first** — 30-second / 2-minute / technical pitches plus the honest comparison vs GPT-5.4 Pro / MiroThinker-H1 / OpenResearcher.
2. **Pick a backend path:**
   - Mac local → `bash scripts/setup-local-mac.sh` (Ollama + SearXNG + gemma4:e2b)
   - GPU VM → `bash scripts/setup-vm-gpu.sh --engine sglang --spec-dec --model Qwen/Qwen3.6-35B-A3B`
   - OpenAI cloud → just set `OPENAI_API_KEY`, no stack needed (+ `docker compose up -d` in `scripts/searxng/` for search)
3. **Run a recipe:**
   - `cd recipes/by-use-case/research-assistant/beginner && make smoke`
   - `cd recipes/by-use-case/trading-copilot/beginner && make smoke`
4. **Read [`paper-draft.md`](paper-draft.md)** if you're interested in the research-paper angle.

---

## Open work (what's NOT yet shipped)

These are the known next steps, in priority order:

1. **GPU VM ablation run** for the research paper — the harness is shipped; user needs to download SimpleQA-100 + BrowseComp-Plus-50 and run `make ablate` with Qwen3.6-35B-A3B to get publishable numbers. Wave 4's rerank + fetch + trace will all be part of the ablation matrix.

2. **Local corpus indexing** — a script that takes a directory of PDFs/markdown and builds a `core/rag`-indexed offline corpus that research-assistant can query alongside SearXNG.

3. **Streaming synthesis** — tokens streamed to stdout as they generate; unlocks reactive FLARE and better UX for interactive local use.

4. **Cross-recipe shared lib** — `_llm()` / `_chat()` / `_critic()` are duplicated across both recipes; refactor into `core/llm/` when a third recipe arrives.

5. **Rust MCP search-tool `BENCHMARKS.md`** — recipe ships with measurement scaffolding but no filled-in numbers yet (requires a `cargo build --release` run).

---

## Design principles (applied consistently)

- **Portable by default.** Every recipe talks to any OpenAI-compatible endpoint via `OPENAI_BASE_URL`. The same code runs on Mac / VM / cloud with env-var swaps.
- **Env-gated techniques.** Every technique independently toggleable for leave-one-out ablation.
- **Test-first.** Every recipe has a mocked test suite that runs with no network and no API keys. Live smokes only verify the integration after the mocked suite is green.
- **Safety rails at build time** where relevant (trading-copilot's forbidden-symbols tests).
- **Single session = one coherent commit.** Commits are semantically complete units — no half-shipped features.
- **Honest docs.** When we haven't run the VM ablation, `paper-draft.md` says `TBD`. When a technique doesn't transfer (trading-copilot skipping HyDE), the techniques.md says so explicitly.

---

## File map (at a glance)

```
agentic-ai-cookbook-lab/
├── README.md                          # repo hero, portable-stack story
├── docs/
│   ├── progress.md                    # ← YOU ARE HERE
│   ├── how-it-works.md                # elevator pitches + SOTA comparison
│   └── paper-draft.md                 # arXiv tech report skeleton
├── core/
│   └── rag/                           # HybridRetriever · Reranker · contextualize_chunks
├── recipes/
│   ├── by-use-case/
│   │   ├── research-assistant/        # beginner + production + eval + ablation
│   │   └── trading-copilot/           # beginner + production + eval + backtest
│   └── by-pattern/
│       └── rust-mcp-search-tool/      # case study: where Rust earns its place
├── scripts/
│   ├── searxng/                       # self-hosted meta-search (docker compose)
│   ├── setup-local-mac.sh             # Ollama + Docker + SearXNG + model pull
│   └── setup-vm-gpu.sh                # vLLM or SGLang + EAGLE spec-dec
└── .project/                          # decisions · journal · architecture (Linear-first)
```

---

## Recent commits (tip-down)

- `(pending)` Wave 4: local-first engine enhancements (rerank + fetch_url + trace)
- `2d28e74` Wave 3: trading-copilot recipe end-to-end (beginner + production + eval)
- `8ac5ad6` Drop youtube-analyzer; rewrite stale READMEs to match Wave 2 Tier 4
- `4260cf9` docs: add how-it-works elevator pitches + SOTA comparison
- `36ade99` Wave 2 Tier 4: classifier + step critic + FLARE + compress + plan refine
- `9825fa6` Wave 2 Tier 3: benchmark harness, ablation matrix, paper draft, Rust MCP case study
- `eecf5dc` Wave 2 Tier 1: core/rag v1 (hybrid+rerank+contextual), scorer metrics, SGLang+EAGLE setup
- `e1fe976` Portable local-inference stack: SearXNG + Ollama/vLLM (OpenAI-compatible)
- `1e5f123` research-assistant: pivot to OpenAI-only with web_search tool
- `b32faca` Align model names with real OpenAI/Gemini SKUs + verify real API path
- `e65ed8b` Wave 1: ship research-assistant/beginner end-to-end
- `846246f` Wave 0.5 — pivot from framework comparison to SOTA-per-task
- `791be65` Initial skeleton — Wave 0
