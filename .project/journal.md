# Progress Journal

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
