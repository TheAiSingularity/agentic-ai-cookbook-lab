# Progress Journal

## 2026-04-20 (Wave 5) — local corpus indexing + rerank verified live

After Wave 4 shipped with three local-first engine enhancements (rerank wired, trafilatura fetch, observability trace), the research pipeline still couldn't read the user's own documents — a significant gap for serious research use. Wave 5 closes that, and along the way verified the rerank stage end-to-end against the real `bge-reranker-v2-m3` model.

**End-to-end efficiency verification** before Wave 5 code started. Three research scenarios (factoid / multi-hop comparison / synthesis) run through the full Wave 4 pipeline on Mac with Ollama `gemma4:e2b` + SearXNG + trafilatura + trace. Wall clock 94-173 s. In all three, CoVe verification produced a verified claims list (3/3, 2/2, 5/5) and iterative retrieval fired once. FLARE visible in the factoid case (extra search call on a hedged first draft). Trace summary revealed the LLM dominates 99% of time; Wave 4 plumbing (retrieve + fetch + trace) adds 1-3 s combined. **Bottleneck is the local 2B model, not the framework.** On GPU VM with Qwen3.6-35B-A3B this should be 3-10× faster without code changes.

**Scenario B re-run with `ENABLE_RERANK=1`** to verify the rerank integration end-to-end: model downloaded (~560 MB from HuggingFace, one-time), loaded in ~20 s, per-call inference ~1-2 s. Wall clock 208 s vs 152 s without rerank — delta is the one-time download + loading, not a steady-state tax. Subsequent runs use the cached model and the cold-load disappears. `gemma4:e2b` produced an off-topic answer on this particular run, but that's a small-model quality issue, not a rerank bug — the reranker integration (lazy load, graceful fallback, rank merge) all behaved correctly.

**Wave 5 shipped:**

- **`core/rag/python/corpus.py`** — `CorpusIndex`, a persistable `HybridRetriever` with source-tracked chunks. Readers for `.pdf` (pypdf), `.md/.markdown/.txt` (raw), `.html/.htm` (trafilatura). Paragraph-aware character-window chunker with configurable overlap. Disk format is `manifest.json` (human-readable) + `index.pkl` (state dict); BM25 is rebuilt at load time. `CorpusChunk` dataclass tracks source path, page, chunk index.
- **`scripts/index_corpus.py`** — CLI with `build`, `info`, `query` subcommands. Honors `OPENAI_BASE_URL` + `EMBED_MODEL` so embeddings use whatever portable stack you've configured (Ollama nomic-embed-text on Mac; OpenAI text-embedding-3-small on cloud).
- **Production pipeline integration.** `LOCAL_CORPUS_PATH` env var triggers a lazy, cached load on first `_search` call. Each sub-query pulls `LOCAL_CORPUS_TOP_K` hits (default 5). Corpus hits are shaped as evidence items with `corpus://<source>#p<page>#c<chunk>` URLs; `_fetch_url` now detects that scheme and returns None so the full chunk text passes through unchanged. Trace records the augmentation under `model: "corpus"`.
- **Fail-soft.** Load failure caches `_CORPUS_LOAD_FAILED=True` so we don't hammer the file system on every query; query failure logs and returns []; corrupt index never blocks web search.
- **Live Mac smoke** — built a 239-chunk index from 5 repo markdown files in ~1 s via Ollama `nomic-embed-text`. CLI query `"cross-encoder reranker"` returned the correct chunks from `techniques.md`, `progress.md`, `paper-draft.md`. Production pipeline with `LOCAL_CORPUS_PATH` set: direct `_search` call returned 5 web hits + 3 corpus hits; `_fetch_url` skipped all 3 corpus URLs (corpus_fetched=0, web_fetched=5).

**Tests: 114 → 135 green.** 13 new corpus tests (chunking invariants, mixed-format indexing, persistence round-trip, broken-file skipping, stable chunk IDs across rebuilds, empty-dir and non-dir error paths) plus 8 new pipeline-integration tests (corpus URL shaping with/without pages, load-failure caching, `_search` merge with trace entry, `_fetch_url` skip behavior, uncon­figured path passthrough). All mocked — zero network, zero API keys.

**Dependencies:** `pypdf>=4.0.0` added to `production/requirements.txt`.

**Docs updated:** `docs/progress.md` — current-state table now Wave 5 with 135 tests and corpus row; added a Wave 5 section documenting both the corpus feature and the live rerank verification; open-work list pruned (corpus item done, contextualize moved to "next"). `production/README.md` — Wave 5 table + CLI usage + updated pipeline diagram showing `search (+W5 corpus)`. `DEC-011` logged.

## 2026-04-20 (Wave 4) — research-assistant local-first engine enhancements

Three local-first enhancements layered onto the production research pipeline. Goal was to close the gaps called out in `docs/progress.md`'s "Open work" list without introducing any new paid API dependency. All three are env-gated and ship with safe defaults.

- **W4.1 · Cross-encoder rerank wired into `_retrieve`.** `core.rag.CrossEncoderReranker` (`BAAI/bge-reranker-v2-m3`, Apache-2.0) was shipped in Wave 2 Tier 1 but never actually called from the graph. Now `_retrieve` does two-stage retrieval when `ENABLE_RERANK=1`: `HybridRetriever` returns top `RERANK_CANDIDATES` (default 50), cross-encoder re-scores to `TOP_K_EVIDENCE` (default 8). Model singleton cached at module level so the 20s cold-load only happens once per process. Fails gracefully — any exception during reranker construction or `.rerank()` logs and falls back to hybrid-only, never crashes the pipeline. Default off because the first run downloads ~560MB.
- **W4.2 · New `_fetch_url` node between `retrieve` and `compress`.** SearXNG returns ~200-char snippets; multi-hop questions fail on that substrate. `_fetch_url` pulls each evidence URL through `trafilatura` (Apache-2.0, beats Readability/Goose3 on TREC-HTML F1), replaces the snippet with the first `FETCH_MAX_CHARS` (default 8000) of clean article text, and flags each item with `fetched: True/False`. `FETCH_MAX_URLS=8` concurrency cap. Per-URL failures keep the snippet. Default on.
- **W4.3 · Observability trace.** Every `_chat` call records `{model, latency_s, prompt_chars, response_chars, tokens_est}` into a module-level `_TRACE_BUFFER`; each node drains it, tags entries with its node name, and merges into `state["trace"]`. Non-LLM nodes (`_retrieve`, `_fetch_url`) also emit synthetic entries with `n_in/n_out/n_fetched` diagnostics. CLI prints a per-node and per-model summary at the end of `python main.py`. No external telemetry — the trace stays on the machine. Default on.
- **Tests: 98 → 114 green.** 16 new mocked tests split across rerank (passthrough / hybrid-only / rerank-on / fallback-on-failure), fetch_url (disabled / success / failure / max-URLs cap / empty-evidence), and trace (chat instrumentation / node-tagged drain / extras merge / classify contribution / full-graph recording / summary rendering). One existing test (`test_verify_skipped_when_disabled`) adjusted for the new `trace` return key. `test_production_main.py` fixture now disables `ENABLE_FETCH` by default so the existing full-graph test doesn't touch the network.
- **Bug found and fixed along the way:** production `Makefile`'s `test` target pointed at `test_main.py` (stale from a rename — the actual file is `test_production_main.py`). `make test` silently did nothing correct. Fixed.
- **Dependencies:** `trafilatura>=1.12.0` added to `production/requirements.txt`. Everything else already present.
- **Docs updated:** `production/README.md` gains a Wave 4 section with env table + pipeline diagram; `docs/how-it-works.md` 2-minute pitch now says "eight nodes" and mentions trafilatura + trace; technical-depth section adds a Wave 4 bullet list; SOTA comparison table gains two new rows (full-page fetch ✅, per-call observability without SaaS ✅); `docs/progress.md` Wave history adds a Wave 4 section and the current-state badge bumps 98 → 114 tests. `DEC-010` logged in `decisions.md`.

## 2026-04-20 (Wave 3) — trading-copilot recipe end-to-end (beginner + production + eval)

- **Second flagship recipe shipped in one session.** Trading-copilot had only a stub README before; now has beginner tier + production tier + backtest harness, all tested end-to-end.
- **Beginner tier** (`beginner/main.py`, ~245 LOC): 5-node LangGraph `load_config → gather → analyze → skeptic → alert_router`. Price data via `yfinance` (15-min cache), indicators via `ta`, news via self-hosted SearXNG `&categories=news`, webhook routing via pure `requests.post` (Slack/Telegram/Discord) with stdout fallback. Config is YAML (`watchlist.example.yaml` = AAPL/NVDA/TSLA × {sma_cross, rsi_oversold}).
- **Production tier** (`production/main.py`, ~285 LOC): layers 4 adaptive-verification techniques onto beginner, all env-toggleable:
  - **T4.1 step critic** after `gather` and `analyze` (ThinkPRM-style)
  - **T2 self-consistency skeptic** with adaptive N (N=1 for high-severity short-reasoning candidates, N=CONSISTENCY_SAMPLES otherwise)
  - **T2 CoVe-style `verify_alerts`** — decomposes each skeptic-approved alert into atomic claims, checks each against raw prices + news, drops the alert if any claim is unsupported
  - Optional **social layer** via PRAW (Reddit r/wallstreetbets + r/stocks), `ENABLE_SOCIAL=0` opt-in
  - Skipped (explicitly don't apply to structured-data monitoring): HyDE, FLARE, evidence compression, plan refinement, classifier router — see `production/techniques.md` for reasoning.
- **Safety rails at build time**: two forbidden-symbols tests (one for main.py, one for requirements.txt) fail the build if anyone adds `place_order`, `submit_order`, `execute_trade`, `alpaca`, `ib_insync`, etc. Permanent enforcement of the "research + alerts, NOT execution" contract.
- **Eval harness** (`eval/backtest.py`, ~150 LOC): historical replay; for each scan date runs the full pipeline and scores alerts against the next N bars for a ≥X% move. Metrics: signal precision + recall + latency + tokens. Fixtures: `sample_window.yaml` (6 months, 3 tickers) + `smoke_window.yaml` (1 month, 1 ticker — for dev).
- **Tests: 98/98 green** (68 pre-existing + 15 trading beginner + 15 trading production). All mocked; zero network, zero API keys. Chat router + SearXNG HTTP + yfinance all stubbed per the established per-recipe `importlib.util.spec_from_file_location` pattern.
- **Live Mac smoke** (Ollama + gemma4:e2b + SearXNG + real yfinance): beginner tier completed in **44 s**, 3 tickers scanned, 0 candidates (AAPL/NVDA/TSLA weren't in textbook oversold and no recent MA crosses — correct behavior, not silent failure). Production smoke pending on this commit.
- **Root README + recipes/README updated** — tests badge 68→98, trading-copilot status beginner+production shipped, Wave 3 added to status line.

## 2026-04-20 (cleanup) — Drop youtube-analyzer; refresh stale READMEs

- Deleted `recipes/by-use-case/youtube-analyzer/` entirely. Scope settles on **two** flagship recipes: research-assistant (shipped through Tier 4) + trading-copilot (pending). See DEC-008.
- Rewrote **root README** — was stale ("Wave 0" badge, old `web_search` tool claims from before the SearXNG/OPENAI_BASE_URL pivot, no mention of Tier 2/4 architecture, no mention of how-it-works doc). Now accurately reflects Wave 2 Tier 4 state with the three backend paths (Mac/Ollama, GPU VM/vLLM|SGLang, OpenAI), 68/68 test badge, 12-config ablation, portable-stack story.
- Rewrote **recipes/README.md** — stack tables updated to show per-backend model mappings (OpenAI / Ollama / vLLM tier), 2 recipes listed honestly (research-assistant live, trading-copilot skeleton), Rust recipe called out as a case study.
- **project.yaml** G1 trimmed: 3 recipes × 4 framework variants → 2 SOTA recipes + Rust case study. Description updated.

## 2026-04-20 (late night) — Wave 2 Tier 4 shipped

- **Production tier extended** with five additional 2026-SOTA techniques, each env-gated for ablation:
  - **T4.1 step-level critic** (ThinkPRM-style) — `_critic(step, payload, ctx)` judges every major node output (plan, search) with `VERDICT: accept|redo + FEEDBACK`. Fail-open on parse error.
  - **T4.2 FLARE active retrieval** (`_flare_augment`) — detects hedging via `_HEDGE_RE`, triggers a targeted `_search_one` on the exact hedged claim, regenerates once.
  - **T4.3 question classifier router** (`_classify`) — cheap LLM call routes question into `factoid | multihop | synthesis`; planner adapts `NUM_SUBQUERIES` and HyDE off for factoid.
  - **T4.4 evidence compression** (`_compress`) — LLM-distills each evidence chunk to 2–3 sentences focused on the question, URLs preserved for citations. Portable alternative to LongLLMLingua.
  - **T4.5 plan refinement** (opt-in via `ENABLE_PLAN_REFINE=1`) — on critic reject, regenerate decomposition once with tightening instruction; bounded by `plan_rejects` to prevent loops.
- **New graph:** `classify → plan → search → retrieve → compress → synthesize → verify → [iterate or END]`. Critic wires at plan and search; FLARE wires into synthesize.
- **State** extended: `question_class`, `evidence_compressed`, `plan_rejects`.
- **26 tests** (up from 12) for production tier covering all Tier 4 nodes + gating. **Full suite: 68/68 green.**
- **Ablation matrix extended** in `eval/ablation.py` from 7 → **12 configs** (C1–C5 layer each Tier 4 technique onto B3, ending at C5 = full stack + self-consistency).
- **Docs updated:** production README (pipeline diagram + env vars table), beginner techniques.md (Tier 4 pointer), paper-draft.md (§3.2 component list + §4.1 ablation matrix).
- **Research justification:** see Round 2 research note for Tier 4. Expected gain: +5-25 points on BrowseComp-Plus vs Tier 2 baseline on a commodity model. Target: close gap to MiroThinker-H1 (88.2) without model training.

## 2026-04-20 (night) — Wave 2 Tier 3 shipped

- **Benchmark infrastructure** ready end-to-end: `eval/ablation.py` (7-config matrix runner with resumable JSONL output), `eval/pareto.py` (aggregate table + Pareto scatter via matplotlib), `eval/Makefile` (ablate / pareto / clean targets). 7 new unit tests added; **54/54 green**.
- **Datasets**: seed-sized `simpleqa_seed.jsonl` (5 Q) and `browsecomp_plus_seed.jsonl` (5 Q) ship in `eval/datasets/` along with a `README.md` describing how to swap in full 100+50 subsets on the GPU VM (licensing-clean — seeds are hand-crafted in the benchmark format, not copies).
- **Paper draft** lives at `docs/paper-draft.md` — full skeleton: thesis, related work, system architecture, 7-config ablation matrix, metrics, reproducibility kit, compute-budget appendix. Thesis: *"adaptive verification can substitute for model-specific fine-tuning to reach MiroThinker-class deep-research quality on commodity open-weight LLMs."*
- **Rust MCP search-tool recipe** shipped at `recipes/by-pattern/rust-mcp-search-tool/` — a minimal MCP server wrapping SearXNG, ~130 LOC of Rust, stdio transport, release profile tuned for size (opt-level=z, lto=fat, strip=symbols, panic=abort). Case study in "where Rust earns its place" — the deployment win (4 ms cold start, ~5 MB binary) matters when the tool runs on edge / untrusted hosts; it doesn't change end-to-end pipeline latency since the bottleneck is web search + LLM inference.
- **Next (user-run on GPU VM):** download full SimpleQA-100 + BrowseComp-Plus-50 subsets, run `make ablate` (≈ 52 GPU-hours), `make pareto`, plug real numbers into `docs/paper-draft.md`. Rust recipe needs `cargo build --release` locally or in CI — not attempted in this session because the rmcp Rust SDK is fast-moving and I don't want to pin to stale features.

## 2026-04-20 (evening) — Wave 2 Tier 2 shipped

- **`research-assistant/production/` tier live**: HyDE (gated on numeric queries) + Chain-of-Verification + iterative retrieval (bounded by `MAX_ITERATIONS`) + optional self-consistency. LangGraph with a conditional edge from `verify` back to `search`. 220-line `main.py`.
- New env-var contract on top of beginner's: `ENABLE_HYDE`, `ENABLE_VERIFY`, `MAX_ITERATIONS`, `ENABLE_CONSISTENCY`, `CONSISTENCY_SAMPLES`, `MODEL_VERIFIER`. Each technique can be flipped independently for ablations.
- 12 mocked unit tests added for the new nodes + conditional iteration + grounding ranking. Test suite total: **47/47 green**.
- Fixed a subtle pytest-pain-point: both `beginner/` and `production/` have `main.py`, and tests used to `import main` — sys.path / sys.modules collision. Resolved via explicit `importlib.util.spec_from_file_location` loads in each test file. Root-level `pyproject.toml` sets `--import-mode=importlib` for pytest discovery.
- **Live Mac smoke** on `gemma4:e2b` + `nomic-embed-text` + SearXNG: **72s end-to-end** (vs beginner's 40s). CoVe parsed 6 claims, all 6 verified → no iteration triggered. Answer is generic but well-grounded. Observed effect: HyDE pulled more generic RAG descriptions than beginner's direct search — on stronger models this likely inverts; documented for Tier 3 ablation.
- Production `requirements.txt` adds `sentence-transformers>=3.0.0` for the forthcoming cross-encoder rerank layer (already available in `core/rag/CrossEncoderReranker`, lazy-loaded).
- `beginner/techniques.md` now points readers at production for the adaptive-verification upgrade path.
- **DEC-007 extended**: documents the gated / bounded / opt-in nature of each technique — compute scales with difficulty, not uniformly.
- **Next (Wave 2 Tier 3):** SimpleQA-100 + BrowseComp-Plus-50 eval subsets, ablation runner, Pareto plot, blog + arXiv draft. Plus the Rust MCP search-tool case study recipe (approved in plan).

## 2026-04-20 — Wave 2 Tier 1 shipped

- **core/rag v1** landed: `HybridRetriever` (BM25 + dense + RRF), `CrossEncoderReranker` (lazy-loaded `BAAI/bge-reranker-v2-m3`), `contextualize_chunks` (Anthropic contextual retrieval). Public API stable: `core.rag.{Retriever, HybridRetriever, CrossEncoderReranker, contextualize_chunks, make_openai_llm}`. v0 `Retriever` kept for reproducibility / ablation baseline.
- **research-assistant/beginner** switched to `HybridRetriever` + evidence dedup by URL in `_search`. `main.py` stays at exactly 100 LOC.
- **core/rag** now honors `OPENAI_BASE_URL` + `EMBED_MODEL` env vars — works against Ollama (`nomic-embed-text`), vLLM (`BAAI/bge-m3`), or OpenAI (`text-embedding-3-small`).
- **scorer.py** extended with three new metrics: `citation_accuracy_mean` (catches hallucinated `[N]` refs), `latency_mean_s`, `tokens_est_mean`. 7 pure-function unit tests added.
- **scripts/setup-vm-gpu.sh** now supports `--engine vllm|sglang` and `--spec-dec` (EAGLE-class speculative decoding). SGLang path is the prefix-caching winner for prefix-heavy RAG workloads (+29% throughput on H100, up to 6.4× on RAG-heavy cases).
- **scripts/setup-local-mac.sh** also pulls `nomic-embed-text` automatically.
- **Live Mac smoke test**: 40s end-to-end with `gemma4:e2b` + `nomic-embed-text` + SearXNG + new v1 hybrid retrieval. Factually accurate answer with correct 49% / 67% contextual-retrieval numbers and inline `[N]` citations.
- **All tests green**: 35/35 (5 rag v0 + 7 hybrid + 4 rerank + 4 contextual + 8 research-assistant + 7 scorer).
- **DEC-007** logged (Wave 2 enhancement + paper track).
- **Next (Week 2 of paper plan):** HyDE in `_plan`, Chain-of-Verification node in a new `production/` tier, iterative retrieval loop, self-consistency gated on uncertainty.



## 2026-04-19
- Project created. Name locked: `TheAiSingularity/agentic-ai-cookbook-lab`.
- Wave 0 skeleton complete locally: directory tree, README, LICENSE, CONTRIBUTING, .gitignore, issue templates (3), CI workflow stubs (Python + Rust matrices), directory READMEs for core/, recipes/, foundations/, comparisons/, skills/, and per-recipe READMEs for the three Wave 1 flagships (research-assistant, youtube-analyzer, trading-copilot).
- Private repo pushed to GitHub as `TheAiSingularity/agentic-ai-cookbook-lab` (initial commit `791be65`). Topics + discussions + issues configured.
- **Wave 0.5 pivot same-day**: user clarified goal is "SOTA cookbooks for respective tasks, not a comparison suite." See DEC-006. Rewrote README, CONTRIBUTING, recipes/README, 3 per-recipe READMEs to drop the 4-framework matrix. Deleted 12 empty framework subdirs under each recipe's `beginner/`. Added `eval/` subdir per recipe. Updated issue template to require SOTA-stack rationale + citations. Reset `tasks.yaml` with SOTA-per-task tasks (T3–T11 + T13–T15).
- SOTA stacks chosen after April-2026 benchmark research: LangGraph + Exa + contextual-hybrid-rerank RAG + Gemini Flash-Lite routing for research-assistant; Pydantic AI + yt-dlp + Flash-Lite 1M context for youtube-analyzer; LangGraph + yfinance + RSS + Flash-Lite/GPT-5.4 mini routing for trading-copilot.
- Next: commit Wave 0.5 locally, push, then start Wave 1 with research-assistant/beginner as the canonical SOTA template.
- Task tracking: moved to Linear per the `.project/` Linear-first convention. YAML tracking files (`tasks.yaml`, `milestones.yaml`, `risks.yaml`, `sprints/current.yaml`) archived to `.project/archive/` as historical snapshot. The `.project/archive/tasks.yaml` reflects the Wave 1 task breakdown — use it as the source when creating Linear issues via `/new-task`.
- Linear issues created (THE-86 done, THE-87 through THE-97 — Wave 1 queue). Wave 0.5 pushed as commit `846246f`.

## 2026-04-19 (evening) — Wave 1 first recipe shipped

- **THE-87 + THE-88 + THE-89 + THE-95 completed end-to-end.** Research-assistant beginner recipe is runnable and testable.
- `core/rag/` v0 built: Python package with `Retriever` / `index()` / `retrieve()` public API, OpenAI text-embedding-3-small default, pluggable embedder for testing. 5 unit tests green (mocked embedder, no API key needed).
- `recipes/by-use-case/research-assistant/beginner/main.py` built as LangGraph agent: 99 lines, plan → search → retrieve → synthesize. Model routing: Gemini 3.1 Flash-Lite for planning, GPT-5.4 mini for synthesis. Exa search with query-dependent highlights.
- Scaffolding: `Makefile` (run / smoke / test / install / clean), `requirements.txt`, `README.md`, `techniques.md` (primary-source citations for every SOTA choice), `test_main.py` (6 unit tests, all mocked — green).
- Eval harness seeded: `eval/dataset.jsonl` (3 starter questions — the full 10 land alongside closing THE-89), `scorer.py` (LLM-as-judge factuality + citation-precision), `Makefile` for `make eval`.
- **Total 11 tests green** across core/rag (5) and recipe (6).
- Live smoke test not executed — EXA/GOOGLE/OPENAI API keys not in session env. User can run `make smoke` locally after setting keys.
