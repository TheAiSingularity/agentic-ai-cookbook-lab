# Progress — `agentic-ai-cookbook-lab`

One-page summary of what's been built, with links into the details.
For the elevator pitch and SOTA comparison, see
[`how-it-works.md`](how-it-works.md). For the research-paper skeleton,
see [`paper-draft.md`](paper-draft.md).

---

## Current state (through Engine Master Plan Phase 8)

After Wave 8's cookbook pivot, the repo transformed into an
**engine-centric local research platform**. The master plan (see
[`research-engine-master-plan.md`](../.project/plans/research-engine-master-plan.md))
tracks Phases 0-9; Phases 0-8 are shipped, Phase 9 (rebrand + public
launch) is **prepared but intentionally held** until user greenlight.

| | |
|---|---|
| Flagship | `engine/` — 8-node LangGraph pipeline + memory + compaction + 3 interfaces + MCP + plugin loader |
| Recipes (archived, still work) | research-assistant, trading-copilot, document-qa, rust-mcp-search-tool |
| Core shared library | `core/rag/` v1 — `HybridRetriever` · `CrossEncoderReranker` · `contextualize_chunks` · `CorpusIndex` |
| Tests | **228+ green**, all mocked (no network / no API keys) |
| Default Mac model | `gemma3:4b` (3.3 GB via Ollama) — measured 40% faster + more nuanced vs `gemma4:e2b` baseline |
| Portable stack | OpenAI · Ollama · vLLM · SGLang — one env var (`OPENAI_BASE_URL`) |
| Search | Self-hosted **SearXNG** (Docker) — no paid API |
| Interfaces | CLI · Textual TUI · FastAPI + HTMX Web GUI (all three shipped) |
| Memory | SQLite trajectory log at `~/.agentic-research/memory.db` + semantic retrieval (off/session/persistent) |
| Compaction | Context-window compactor preserving CoVe-verified URLs |
| Domain presets | 6 shipped (general/medical/papers/financial/stock_trading/personal_docs) |
| Plugin/skill loader | Claude plugins + Hermes skills; gh / file / url sources; forbidden-symbols scan |
| MCP | Python FastMCP server (`research`, `reset_memory`, `memory_count`) + Claude plugin bundle (4 skills) |
| Benchmark harness | `engine/benchmarks/runner.py` + `simpleqa_mini.jsonl` (20) + `browsecomp_mini.jsonl` (10) |
| Observability | Per-call trace (node, model, latency, tokens) — no SaaS |
| Repo visibility | **Private** (held for Phase 9 go-live) |
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

### Engine Master Plan — Phases 0 → 8 (2026-04-20 / 04-21)

Strategic pivot of the whole repo from "SOTA cookbook of recipes" to
"flagship local research engine." Master plan in
[`.project/plans/research-engine-master-plan.md`](../.project/plans/research-engine-master-plan.md).

| phase | delivered |
|---|---|
| **0 — Hygiene** | Flipped repo private, baseline 159/159 tests confirmed, plan copied into repo, established "no Co-Authored-By" commit convention |
| **1 — Engine core extraction** | Split 826-LOC production/main.py into `engine/core/{pipeline,models,trace}.py`; `production/main.py` became a thin shim. Live Gemma 3 4B characterization: **44s wall on Scenario A vs 78s on gemma4:e2b** (40% faster, more nuanced answer). Tests: 159 → 184. |
| **2 — Memory + compaction** | `engine/core/memory.py` (SQLite trajectory log + semantic retrieval, 3 modes) + `engine/core/compaction.py` (preserves CoVe-verified URLs). 25 new tests. |
| **3 — Three interfaces** | CLI + Textual TUI + FastAPI + HTMX Web GUI, all sharing `engine/interfaces/common.py`. 13 new tests. |
| **4 — MCP + Claude plugin** | Python FastMCP server with `research` / `reset_memory` / `memory_count`; submittable Claude plugin bundle with 4 skills. 5 new tests. |
| **5 — Plugin/skill loader** | Disk-backed registry under `~/.agentic-research/plugins/`, supports `gh:`, `file:`, `https://` sources; Claude plugin + Hermes skill formats; safety scan on install. 27 new tests. |
| **6 — Domain presets + 5 examples** | Hand-rolled YAML parser, `DomainPreset` loader, 6 shipped presets (general / medical / papers / financial / stock_trading / personal_docs), 5 worked examples with expected outputs. 16 new tests. |
| **7 — Docs + contributor guide** | Rewritten `CONTRIBUTING.md`, new `docs/architecture.md`, `docs/plugins-skills.md`, `docs/domains.md`, `docs/self-learning.md`. Issue + PR templates. `CODE_OF_CONDUCT.md`. Root README pivoted to engine-centric. |
| **8 — Benchmark harness** | `engine/benchmarks/runner.py` with `must_contain` / `must_not_contain` scoring + ablation flags (rerank / no-fetch / no-compress / no-verify / no-flare / no-router). Shipped fixtures: 20-question SimpleQA-mini + 10-question BrowseComp-mini. 13 new tests. |
| **9 — Rebrand + public launch** | **HELD**. Launch copy drafted (`docs/launch-copy.md`), go-live checklist written (`docs/launch-checklist.md`). Repo stays private until user greenlights the flip. |

Total net impact: repo-wide tests **159 → 228+** green; engine gained
~5000 LOC across core + interfaces + MCP + benchmarks; 8 new docs
pages; full 3-interface + MCP + plugin story end-to-end.

### Wave 8 — document-qa recipe (third flagship, corpus-only)

Third flagship recipe, bringing the total to two flagship + one flagship + one Rust case study. Validates that `core.rag` — specifically `CorpusIndex` and `HybridRetriever` — composes cleanly without any of the web-reach machinery that pays off for open-web research.

- 4-node LangGraph: `load_corpus → retrieve → synthesize (streaming) → verify (CoVe)`. Deliberately no router, no iteration, no compressor — on a bounded corpus those don't earn their complexity.
- Drop documents at `DOCS_DIR` (PDFs / markdown / text / HTML), or point `CORPUS_PATH` at a prebuilt index from `scripts/index_corpus.py`. Everything else is optional env tweaks.
- Build-time safety rail: AST-based test rejects any appearance of `searxng`, `requests.get`, `trafilatura.fetch_url`, or `webhook` in executable code. The rail walks the AST (not raw text) so documentation mentioning those terms doesn't false-positive.
- 10 mocked tests covering node contracts (load from dir, load from prebuilt, error when neither set, URL shaping, synthesize with/without hits, verify counts, verify disabled), full-graph integration, and the safety-rail.
- Live smoke: built a 23-chunk fixture corpus in 1.1 s via Ollama `nomic-embed-text`; answered "what retrieval techniques does the cookbook ship" in 34.8 s with an 8/8-verified structured answer citing `corpus://retrieval.md#c0`/`#c2`/`#c8` plus `corpus://pipeline.md#c7`. Streaming UX worked — tokens arrived live.

Supports the `core/rag` graduation bar from DEC-004: if the library is genuinely reusable, a bring-your-own-documents recipe should be ~200 LOC and reuse everything without duplication. It is (169 LOC main, 175 LOC tests) and it does.

### Wave 7 — streaming synthesis

Single-measure wave focused purely on UX. On a 95-175 s production query, users previously waited in stdout silence. Now they see the answer type out live.

- `_chat_stream()` helper calling OpenAI-compatible `stream=True`, accumulating `delta.content` tokens into a full answer while writing each to a sink (default stdout + flush). Falls back to `_chat()` if the backend rejects streaming. Trace entry shape matches `_chat` plus `streamed: True`.
- `_synthesize_once` uses streaming when `ENABLE_STREAM=1` and `ENABLE_CONSISTENCY=0`. Self-consistency mode batches — streaming N interleaved candidates would be UX soup.
- 6 new tests (sink capture, batched fallback on backend error, streamed flag in trace, synthesize uses stream path, synthesize batches when stream off, synthesize batches when consistency on). The `patched` fixture disables streaming by default since its mock returns `SimpleNamespace` not iterables.
- Live verified against `gemma4:e2b` on Ollama: 173-char answer streamed in 9.52 s; trace entry flagged `streamed=True`.

### Wave 6 — small-model hardening (anti-hallucination for gemma4:e2b-class inference)

End-to-end testing on Mac with the 2 B local model exposed three hallucination failure modes when the synthesizer got large evidence contexts: off-topic essays, topic-substitution, and plausible-sounding partial answers that missed the actual question. Wave 6 addresses all three with prompting + caps + an auto-TopK heuristic, each env-gated.

| Measure | What it does | Default |
|---|---|---|
| **W6.1 · Refined synthesize prompt** | Three-case rule: `FULL answer` / `partial answer + named gaps` / `UNRELATED → refuse exactly`. Explicit "never invent" + "never substitute a related topic". Refined through two iterations — first binary version caused false refusals on partial-evidence questions. | always-on |
| **W6.2 · Per-chunk char cap** (`PER_CHUNK_CHAR_CAP`) | After compress, every evidence chunk is hard-truncated regardless of whether compress ran. Bounds the synthesize prompt even when the compressor fails or is disabled. | 1200 chars |
| **W6.3 · Auto-TopK heuristic** (`SMALL_MODEL_TOPK`) | When `MODEL_SYNTHESIZER` matches `:e2b` / `:2b` / `:3b` / `-2b` / `nano`, reduce `TOP_K_EVIDENCE` from 8 → 5. Explicit `TOP_K_EVIDENCE` overrides. Does NOT match `mini` (gpt-5-mini / gpt-4o-mini are cloud-hosted and fine). | auto-detect |

**Live re-run results on Mac gemma4:e2b (same SearXNG + trafilatura + trace as Wave 4):**

| Scenario | Baseline (Wave 4) | After Wave 6 |
|---|---|---|
| A · factoid (Anthropic CR year + %) | partial: year unspecified + 49%/35% cited · 94.7 s | **"September 2024 [5] … 67% reduction [1][2][3]" · 78 s** |
| B · multi-hop (Anthropic CR vs bge-reranker) | rambling essay on reranker side only · 152 s | **Clean refusal: "evidence does not answer this question" · 229 s** |
| C · synthesis (MiroThinker + OpenResearcher) | generic essay · 140 s | **Partial answer on techniques with [1][3] citations + explicit "evidence does not provide information on verification" · 176 s** |
| B + rerank | off-topic logarithm essay · 208 s | **Clean refusal · 306 s** |

No hallucinations across any of the four scenarios. Factoid gives direct answers. Partial-evidence questions give structured partial answers that name the gaps. Unrelated evidence triggers clean refusals. Tests: 135 → **143 green** (8 new W6 cases).

### Wave 5 — local corpus indexing + rerank verified live

Close the other half of "local-first": the agent can now search your own
documents alongside the web, and the rerank stage was verified against the
real `bge-reranker-v2-m3` model end-to-end.

| | |
|---|---|
| `core/rag/python/corpus.py` | `CorpusIndex` — persistable, source-tracked, built on `HybridRetriever`. Readers: `.md`, `.markdown`, `.txt` (raw), `.pdf` (via `pypdf`), `.html`, `.htm` (via trafilatura). Paragraph-aware character-window chunking with overlap. Disk format: `manifest.json` + `index.pkl`. BM25 state rebuilt at load time. |
| `scripts/index_corpus.py` | CLI: `build`, `info`, `query`. Honors `OPENAI_BASE_URL` + `EMBED_MODEL` so embeddings flow through the same portable stack (Ollama nomic-embed-text on Mac; OpenAI on cloud). |
| Production integration | `LOCAL_CORPUS_PATH` env var loads the index lazily on first `_search` call. Each sub-query pulls `LOCAL_CORPUS_TOP_K` matches (default 5), shaped as evidence items with `corpus://<source>#p<page>#c<chunk>` URLs. `_fetch_url` skips those URLs automatically (their text is already the full chunk). Trace records the augmentation. Graceful fallback to web-only on load failure — never blocks the pipeline. |
| Tests | 13 new `test_corpus.py` (chunking, indexing, persistence round-trip, broken-file skipping, stable chunk IDs); 8 new pipeline tests (corpus URL shaping, load-failure caching, `_search` merge, `_fetch_url` skip). 114 → **135 green**. |
| Live Mac smoke | Built a 5-file / 239-chunk corpus from the repo's own docs via Ollama `nomic-embed-text` in ~1 s. CLI query `"cross-encoder reranker"` returned the right chunks from `techniques.md`, `progress.md`, `paper-draft.md`. Production pipeline with `LOCAL_CORPUS_PATH` set: 5 web hits + 3 corpus hits merged into evidence; `_fetch_url` skipped all 3 corpus URLs (corpus_fetched=0, web_fetched=5). |
| Rerank verified live | Scenario B re-run with `ENABLE_RERANK=1`. Cross-encoder model downloaded (~560 MB, one-time) and loaded in ~20 s; rerank inference then added ~1-2 s per call. Total wall-clock 208 s on Mac vs 152 s without rerank — delta is the one-time download + loading. On subsequent runs the cold-load is cached. |

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

1. **GPU VM ablation run** for the research paper — the harness is shipped; user needs to download SimpleQA-100 + BrowseComp-Plus-50 and run `make ablate` with Qwen3.6-35B-A3B to get publishable numbers. Wave 4's rerank + fetch + trace and Wave 5's corpus augmentation are all part of the ablation matrix now.

2. **Streaming synthesis** — tokens streamed to stdout as they generate; unlocks reactive FLARE and better UX for interactive local use.

3. **Corpus contextualization opt-in** — wire `contextualize_chunks` into `CorpusIndex.build` behind a `CONTEXTUALIZE=1` flag. Costs one LLM call per chunk at index time; Anthropic reports −35 to −67% retrieval failures. Deliberately deferred until a bigger local model makes the 239-chunk case cheap (a 30B on GPU VM would do it in under a minute).

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

- `037871b` Phase 8: mini-benchmark harness + fixtures
- `7d4185c` Phase 7: docs + contributor guide rewrite
- `a4e8641` Phase 6: domain presets + 5 worked examples
- `1e371e6` Phase 5: plugin + skill loader (Claude + Hermes formats)
- `292b85f` Phase 4: MCP server + Claude plugin bundle
- `13f38c6` Phase 3: CLI + TUI + Web GUI (three interfaces in parallel)
- `15ffc16` Phase 2: memory persistence + context compaction
- `a67cff5` Phase 1: engine.core extracted; Gemma 3 4B beats e2b baseline
- `37a4108` Phase 0: master plan for SOTA local research engine
- `cdc3d4c` Wave 8: document-qa recipe (3rd flagship, corpus-only 4-node pipeline)
- `ace5e2e` Wave 7: streaming synthesis (tokens to stdout as they generate)
- `563e862` Wave 6: small-model hardening (anti-hallucination + caps + auto-topK)
- `af0a09b` docs: root README refreshed for Wave 5; defensive gitignore patterns
- `d8af86a` Wave 5: local corpus indexing (`CorpusIndex` + `scripts/index_corpus.py` + `LOCAL_CORPUS_PATH` integration)
- `4f27c03` Wave 4: local-first engine enhancements (rerank + fetch_url + trace)
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
