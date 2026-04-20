# Decisions

## DEC-012 — Wave 6: small-model hardening (anti-hallucination + caps + auto-topK)
**Date:** 2026-04-20
**Context:** Live end-to-end testing on Mac with `gemma4:e2b` exposed three hallucination failure modes we hadn't seen in mocked tests: (1) Scenario B with rerank on produced an off-topic logarithm-power-rule essay; (2) Scenario D with corpus attached produced a veterinary-management-system essay; (3) partial-evidence questions got plausible-sounding essays that were technically cited but missed the point. Root cause analysis from the trace: `gemma4:e2b` gets ~19k tokens of compressed evidence and loses the question thread. The pipeline worked correctly — the prompting was too permissive for small-model inference.
**Decision:** Ship three independent W6 hardening measures, each with its own env knob so capable models can opt out:
- **W6.1 — Anti-hallucination synthesize prompt.** Refined through two iterations in the same session. First attempt was too binary ("FULL answer or refuse") which regressed Scenario A's valid partial answer. Final three-case rule: if evidence fully answers → answer; if partially → answer the supported parts + name the gaps; only refuse entirely if evidence is UNRELATED. Explicit "never invent" and "never substitute a related topic" clauses. Result on re-run: Scenario A produced a better answer than baseline ("September 2024 + 67%" vs baseline's "year unspecified + 49%/35%") in less wall-clock time (78s vs 94.7s).
- **W6.2 — Per-chunk char cap after compress (`PER_CHUNK_CHAR_CAP=1200`).** Compressor is still called normally, but every chunk — whether compressed or pass-through — is hard-truncated before synthesize. Bounds the synthesize prompt regardless of compressor quality. When `ENABLE_COMPRESS=0`, still applies (catches the case where raw evidence is too long).
- **W6.3 — Small-model TOP_K heuristic.** If `MODEL_SYNTHESIZER` matches `_SMALL_MODEL_RE` (`:e2b`, `:2b`, `:3b`, `-2b`, `nano`) and `TOP_K_EVIDENCE` isn't set explicitly, reduce to `SMALL_MODEL_TOPK=5` (from the default 8). Deliberately does NOT match `mini` — `gpt-5-mini`/`gpt-4o-mini` are cloud-hosted and handle context well. Users who set `TOP_K_EVIDENCE` explicitly keep their value.
**Consequences:** `test_production_main.py` gains 8 W6 tests (regex matching, explicit-override respect, small-model detection, capable-model non-match, per-chunk cap both with/without compress, prompt content assertions). Existing tests adjusted: the fixture's chat_router keys off the new prompt's opening phrase. 135 → **143 tests green**. Production `main.py` grows ~30 LOC. Synthesize prompt grows ~400 chars (negligible cost on any model). Live re-runs across 4 scenarios confirm hallucinations eliminated; partial-answer quality preserved or improved; wall-clock roughly flat (cleaner refusals sometimes faster, structured partial answers sometimes a bit slower due to CoVe finding more claims to verify).

## DEC-011 — Wave 5: local corpus indexing as a first-class pipeline input
**Date:** 2026-04-20
**Context:** After Wave 4, the research pipeline could read the live web fully (SearXNG → trafilatura) but couldn't read the user's own documents. This is the single biggest capability gap for "serious research" use: a user with a folder of PDFs, internal markdown notes, or downloaded papers wants the agent to reason over those alongside the web. Two design choices to make: (1) where does ingestion happen — at pipeline runtime, or offline via a build step? (2) how do corpus hits flow through the existing graph — a new node, or augment an existing one?
**Decision:** Ship two artifacts:
- **`core/rag/python/corpus.py`** — `CorpusIndex`, a persistable `HybridRetriever` with source-tracked `CorpusChunk`s. Readers for `.pdf` (pypdf), `.md/.markdown/.txt` (raw), `.html/.htm` (trafilatura). Paragraph-aware character-window chunking with overlap. Disk format is `manifest.json` (human-readable) + `index.pkl` (state); BM25 is rebuilt at load time since it's fast and not worth pickling. Shipped with 13 mocked tests covering chunking invariants, format mix, persistence round-trip, broken-file graceful skip, and stable chunk IDs.
- **`scripts/index_corpus.py`** — CLI with `build / info / query` subcommands. Honors `OPENAI_BASE_URL` + `EMBED_MODEL` so embedding flows through the same portable stack users already have set up.
- **Pipeline integration is a `_search` augmentation, not a new node.** `LOCAL_CORPUS_PATH` env var triggers lazy load on first query. Each sub-query pulls `LOCAL_CORPUS_TOP_K` hits (default 5); they're shaped as evidence items with `corpus://<source>#p<page>#c<chunk>` URLs. `_fetch_url` detects the scheme and returns None, so the full-chunk text is preserved without a wasted network round-trip. Trace logs the augmentation with `model: "corpus"`, `n_hits`, `n_subqueries`.
- **Fail soft everywhere.** Load failure caches `_CORPUS_LOAD_FAILED` so we don't retry on every query; query failure logs and returns []; corrupt index file doesn't block web search.
**Consequences:** `core.rag` exports two new public symbols (`CorpusIndex`, `CorpusChunk`). Tests jump 114 → **135 green** (13 corpus + 8 pipeline integration). `pypdf>=4.0` added to production requirements. Production `main.py` grows +80 LOC for the integration. No existing behavior changes when `LOCAL_CORPUS_PATH` is unset — pipeline stays web-only by default. Future work noted in `progress.md`: `CONTEXTUALIZE=1` flag to run `contextualize_chunks` at build time (−35 to −67% retrieval failures per Anthropic); deferred until a larger local model makes the per-chunk LLM cost cheap.

## DEC-010 — Wave 4: local-first engine enhancements (rerank + fetch + trace), all env-gated
**Date:** 2026-04-20
**Context:** After Wave 2 Tier 4 shipped, the production pipeline had three holes. (1) `CrossEncoderReranker` existed in `core/rag` but was never wired in — production used hybrid-only retrieval. (2) SearXNG returns 200-char snippets, which fail on multi-hop factual questions that need full-article context. (3) Debugging locally was opaque — no per-call timing/token visibility, and shipping a SaaS telemetry hook would violate the "no vendor lock-in, zero-$/query" positioning.
**Decision:** Ship three enhancements, all local-first / self-hostable / env-gated with safe defaults:
- **W4.1** · Cross-encoder rerank wired into `_retrieve`. Two-stage (hybrid top-N → `BAAI/bge-reranker-v2-m3` top-K). `ENABLE_RERANK=0` default because first run downloads ~560MB; turning it on is a one-line env flip. Graceful fallback to hybrid-only if model loading fails.
- **W4.2** · New `_fetch_url` node between `retrieve` and `compress`. Uses `trafilatura` (Apache-2.0) to clean-text-extract each evidence URL. `ENABLE_FETCH=1` default, `FETCH_MAX_URLS=8` concurrency cap, per-URL failures silently fall back to snippet.
- **W4.3** · Observability trace. `_chat` records `{node, model, latency_s, tokens_est, prompt_chars, response_chars}` via a module-level `_TRACE_BUFFER`; each node drains + tags it into `state["trace"]`; CLI prints per-node / per-model summary at end. `ENABLE_TRACE=1` default. No network egress.
**Consequences:** Production main.py grows from 384 → ~485 LOC. Test count jumps 98 → 114 (16 new Wave 4 tests, 1 existing adjusted for the new `trace` return key). `trafilatura>=1.12.0` added to requirements. The graph gains one node: `retrieve → fetch_url → compress`. `docs/how-it-works.md` SOTA table gains two rows (full-page fetch ✅, per-call observability without SaaS ✅). Nothing in Tier 2 or Tier 4 was modified — these are purely additive.

## DEC-009 — trading-copilot skips research-assistant's HyDE/FLARE/compression/plan-refinement/router
**Date:** 2026-04-20
**Context:** Research-assistant's production tier has 9 techniques across Tiers 2 + 4. Transferring all of them to trading-copilot would add complexity without benefit — many are domain-specific to open-web research.
**Decision:** Production trading-copilot inherits only the techniques that transfer to structured-data monitoring: **step critic** (T4.1), **self-consistency skeptic** (T2), **CoVe-style alert verification** (T2 equivalent — same principle, different data substrate). HyDE, FLARE, evidence compression, plan refinement, and the classifier router are explicitly skipped with documented reasoning in `production/techniques.md`.
**Consequences:** Production trading-copilot tier stays ~285 LOC (vs research-assistant's 384). Same "adaptive verification" story, but honestly scoped. Future: revisit compression if `ENABLE_SOCIAL=1` makes news/social feeds balloon.

## DEC-008 — Drop youtube-analyzer; focus on research-assistant + trading-copilot
**Date:** 2026-04-20
**Context:** Three recipes were scoped in Wave 1 (research-assistant, youtube-analyzer, trading-copilot). After Wave 2's deep investment in research-assistant — four tiers of techniques, 68 unit tests, 12-config ablation matrix, full paper draft — youtube-analyzer never got any implementation and added scope confusion. The trading-copilot story (market research + alerts, not execution) is differentiated and has a natural place for the adaptive-verification stack; youtube-analyzer overlapped too much with generic "summarize long video" tooling.
**Decision:** Delete `recipes/by-use-case/youtube-analyzer/`. Keep two flagship recipes: research-assistant (shipped through Tier 4) and trading-copilot (pending). Rewrite root README and recipes/README.md to reflect current scope accurately.
**Consequences:** README was significantly stale ("Wave 0" status badge, mentioned `web_search` tool even though we pivoted to SearXNG + OPENAI_BASE_URL portability months ago). Rewrite fixes that. G1 goal in project.yaml trimmed to 2 recipes. No code touched beyond youtube-analyzer/ deletion and README surfaces.

## DEC-007 — Wave 2: SOTA enhancement track + research paper
**Date:** 2026-04-20
**Context:** The research-assistant recipe works end-to-end locally and on cloud, but isn't SOTA. User wants to push to world-class on speed+accuracy+compute axes and publish. Reference target: MiroThinker-H1 at 88.2 BrowseComp (surpasses Gemini-3.1-Pro and Claude-4.6-Opus). User's rig: 4× RTX 6000 Pro Blackwell, fully self-hosted.
**Decision:** Ship Wave 2 enhancements in three tiers:
- **Tier 1 (quality at zero/negative compute cost):** `core/rag` v1 (BM25 + dense + RRF hybrid), cross-encoder reranker (lazy-loaded), contextual chunking helper, evidence dedup in `_search`, citation-grounding validation in scorer, SGLang alternative + EAGLE speculative decoding flags.
- **Tier 2 (adaptive compute for accuracy):** HyDE planner rewriting, Chain-of-Verification after synthesize, iterative retrieval (ITER-RETGEN), self-consistency gated on uncertainty. Lives in a future `production/` tier.
- **Tier 3 (infra + paper):** SimpleQA-100 + BrowseComp-Plus-50 harness, Pareto ablation plot, blog post + arXiv tech report.
**Rust:** keep main pipeline Python (vLLM/SGLang + HF tokenizers are already Rust under the hood). One dedicated Rust artifact — `recipes/by-pattern/rust-mcp-search-tool/` — showcasing 4 ms cold start, 5 MB binary as a case study.
**Benchmarks:** SimpleQA-100 + BrowseComp-Plus-50.
**Paper:** Blog post + arXiv TR (no conference deadline).
**Consequences:** Wave 1 ships the baseline; Wave 2 is where we compete with MiroThinker. Every technique leave-one-out ablated. `core/rag` v0 stays for reproducibility; v1 is the new default via `HybridRetriever`.



## DEC-001 — Format: runnable recipe gallery, not tutorial series
**Date:** 2026-04-19
**Context:** Initial framing was "tutorial series / handbook" explaining OpenClaw, OpenShell, Hermes Agent, NemoClaw, plus the broader ecosystem. Tested against real GitHub data: pure tutorial / link-list repos plateau at 300–1.3K stars. Runnable-recipe repos (`Shubhamsaboo/awesome-llm-apps`: 106K⭐ / 15% fork-to-star ratio) dominate interaction in this category.
**Decision:** Pivot to clone-and-run recipe gallery. Foundations (4 pages) and comparisons (6 pages) stay as SEO support — not the headline.
**Consequences:** Higher interaction ceiling. Every recipe must `make run` in ≤60s from a fresh clone or it doesn't ship. CI load grows with recipe count.

## DEC-002 — Three flagship recipes for Wave 1, not fifty
**Date:** 2026-04-19
**Context:** Original scope was 50+ recipes. User pushed back: ship 3 first, prove the format, then scale.
**Decision:** Wave 1 ships **three recipes**: research-assistant, youtube-analyzer, trading-copilot. Each shipped in **four framework implementations** (plus vanilla baseline) to showcase side-by-side comparison. Twelve runnable impls total.
**Consequences:** Faster to first real ship. Framework comparison becomes a Wave 1 deliverable rather than a nice-to-have. Recipe count grows iteratively.

## DEC-003 — Python + Rust, strategic not parallel
**Date:** 2026-04-19
**Context:** Original plan was Python + TypeScript. User clarified: Rust, not TS.
**Decision:** Python is universal across all recipes. Rust is clustered in categories where it genuinely wins: MCP servers, inference runtimes, tool binaries, edge agents, sandbox-escape research. Target ~50 Python recipes / ~10–15 Rust recipes over the full roadmap — not 1:1 parity.
**Consequences:** Differentiated positioning (no serious agent-recipe repo exists in Rust). Lower maintenance cost than naive parity. TypeScript is deliberately out of scope.

## DEC-004 — Single repo with core/ graduation path, not two repos
**Date:** 2026-04-19
**Context:** User asked whether RAG should be its own repo. Analysis: two repos up front = 2× setup, halved network effect, premature abstraction.
**Decision:** Shared `core/` library inside this repo (rag / memory / tools / sandbox). Recipes import from `core/` rather than duplicating. Graduate `core/rag/` into a standalone `TheAiSingularity/agentic-rag` repo when it hits traction criteria (≥1K stars on this repo, ≥20 recipes using it, stable API across 2 waves).
**Consequences:** Cross-promotion between recipes and core library. Cleaner API via real-usage pressure before spin-out. Later fork is a 2-hour job, not an architectural migration.

## DEC-006 — SOTA-per-task, not framework comparison
**Date:** 2026-04-19
**Context:** Wave 0 shipped with recipes advertising 4 framework implementations each for side-by-side comparison. User clarified the real goal: "SOTA cookbooks for respective tasks, not a comparison suite." The framework-comparison angle was my invention, not theirs. The correct model is OpenAI Cookbook / Anthropic Cookbook — one opinionated, state-of-the-art implementation per task.
**Decision:** Every recipe ships exactly **one** implementation — the SOTA stack for that task, chosen for cheapest-yet-most-accurate. Framework comparison suites are dropped from inside recipes. `comparisons/` keeps its role as standalone landscape pages (SEO-valuable). Every recipe adds `techniques.md` (citations for SOTA choices) and `eval/` (reproducible scorer against gold answers) so the SOTA claim is verifiable.
**SOTA stacks chosen (April 2026, benchmark-backed):**
- research-assistant: LangGraph + Exa + `core/rag/` (contextual + hybrid + rerank) + Gemini 3.1 Flash-Lite (→ GPT-5.4 mini for hard reasoning). Expected cost: $0.01–$0.03/query.
- youtube-analyzer: Pydantic AI + yt-dlp (+ Groq Whisper fallback) + Gemini 3.1 Flash-Lite (1M context). Expected cost: $0.001–$0.02/video.
- trading-copilot: LangGraph + yfinance + RSS + model routing (Flash-Lite analyst → GPT-5.4 mini skeptic). Expected cost: $0.005–$0.02/cycle.
**Consequences:** 12 empty framework subdirs deleted from Wave 0 scaffold. Per-recipe README, CONTRIBUTING, root README, recipes/README, and recipe-request issue template all rewritten. Tasks.yaml reset: T3/T4/T5/T6/T7/T8 now scope "SOTA impl + techniques.md + eval harness" per recipe instead of 4-framework port. Comparison pages still ship in Wave 2 — but in `comparisons/`, not inside recipes.

## DEC-005 — Name: agentic-ai-cookbook-lab
**Date:** 2026-04-19
**Context:** Iterated through multiple candidate names. User wanted descriptive, okay-with-collisions, lab vibe. Target form: "agentic AI recipe/cookbook + lab."
**Decision:** `TheAiSingularity/agentic-ai-cookbook-lab`. Zero star-weighted GitHub collisions, descriptive (every keyword earns its keep), matches user's phrasing.
**Consequences:** URL is 23 characters — long but clean. "Agentic" keyword pulls in high-intent 2026 search traffic. "Cookbook" signals the content format. "Lab" signals experimentation / comparison.
