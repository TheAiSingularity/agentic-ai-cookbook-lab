# Changelog

All notable changes to this project are documented here, following
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning
follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- toc -->

- [Unreleased](#unreleased)
- [0.1.2 — MCP registry ownership proof](#012--mcp-registry-ownership-proof-2026-04-21)
- [0.1.1 — packaging + marketplaces](#011--packaging--marketplaces-2026-04-21)
- [0.1.0 — public alpha](#010--public-alpha-2026-04-21)
- [Pre-0.1 wave history](#pre-01-wave-history)

<!-- /toc -->

---

## [Unreleased]

Work-in-progress on `main` between releases. Nothing here yet.

---

## [0.1.2] — MCP registry ownership proof — 2026-04-21

Hotfix for the `mcp-publisher publish` validation. The registry
requires an **ownership marker** in the PyPI package's README to
prove the PyPI package and the MCP-registry entry are controlled by
the same owner.

### Fixed

- **Added the `mcp-name: io.github.TheAiSingularity/agentic-research`
  marker** to `README.md` in a new "MCP registry ownership" footer
  section. This is the literal string the registry validator scans
  for in the PyPI-hosted README.
- **Bumped all version strings from 0.1.1 → 0.1.2** (pyproject.toml,
  engine/__init__.py, server.json ×2, plugin.json) since PyPI does
  not permit re-uploading the same version even after a yank.

### Why this is a version bump

PyPI is append-only per version. A yank hides a version but doesn't
free the filename. The README change has to land inside a new version's
sdist/wheel, which means a new version number. The actual pipeline
and feature surface are identical to 0.1.1.

### Install

```bash
pip install --upgrade agentic-research-engine   # → 0.1.2
```

---

## [0.1.1] — packaging + marketplaces — 2026-04-21

PyPI-ready. Both official marketplace submissions prepared. No new
user-facing features; naming consistency fix and install UX.

### Added

- **Full `pyproject.toml`** — package metadata, runtime deps mirroring
  `engine/requirements.txt`, `dev` + `rerank` optional extras, and two
  console-script entry points:
  - `agentic-research` → `engine.interfaces.cli:main`
  - `agentic-research-mcp` → `engine.mcp.server:main`
  - Setuptools `package-data` rules to ship domain YAMLs, web
    templates/CSS, Claude plugin manifest + skills, and benchmark
    fixtures inside the wheel.
- **`MANIFEST.in`** — explicit sdist inclusion rules for README,
  CHANGELOG, LICENSE, engine assets, docs. Prunes venvs, build
  artifacts, tests.
- **`server.json`** at repo root — MCP registry manifest.
  `name: io.github.TheAiSingularity/agentic-research`, points at the
  PyPI package, declares stdio transport + 9 environment variables.
- **`marketplace.json`** at repo root — Anthropic Claude plugin
  marketplace discovery manifest. Single plugin entry pointing at
  `engine/mcp/claude_plugin/`.
- **`docs/submit-claude-plugin.md`** — step-by-step submission guide
  including slash-command smoke test, community-aggregator listings,
  rollback notes.
- **`docs/submit-mcp-registry.md`** — step-by-step PyPI upload +
  `mcp-publisher` flow including pitfalls and version-bump playbook.

### Changed

- **MCP server name unified** — `FastMCP("engine-research")` renamed to
  `FastMCP("agentic-research")` so the package name, plugin name, MCP
  server name, and registry identifier all align. Users searching
  "agentic research" on any marketplace now find the same thing.
- **`engine.__version__`** bumped `0.1.0` → `0.1.1`.
- **Plugin manifest version** bumped `0.1.0` → `0.1.1`.

### Install (new options in this release)

From PyPI:

```bash
pip install agentic-research-engine
agentic-research version                   # → "engine v0.1.1"
agentic-research ask "what is hybrid retrieval?" --domain papers
```

From source (unchanged):

```bash
git clone https://github.com/TheAiSingularity/agentic-research-engine-oss
cd agentic-research-engine-oss/engine && make install
```

### Distribution surfaces ready

| surface | status |
|---|---|
| PyPI (`agentic-research-engine`) | artifacts built + smoke-tested in fresh venv; awaiting `twine upload` |
| MCP registry (`io.github.TheAiSingularity/agentic-research`) | `server.json` ready; submit via `mcp-publisher publish` |
| Anthropic Claude marketplace | `marketplace.json` at repo root; `/plugin marketplace add <repo-url>` installs |
| claudemarketplaces.com (community) | follow-up PR per [`docs/submit-claude-plugin.md`](docs/submit-claude-plugin.md) |
| mcp.so (community) | picks up automatically from the MCP registry once published |

---

## [0.1.0] — public alpha — 2026-04-21

Initial public release of the **engine** — a local-first research agent
with three interfaces, MCP distribution, plugin loader, memory, and a
benchmark harness. Repo renamed from `agentic-ai-cookbook-lab` (its
original cookbook-era name) to `agentic-research-engine-oss` on the
same day to match the engine-centric positioning.

### Added

**Engine core** (`engine/core/`):
- `pipeline.py` — 8-node LangGraph research pipeline (classify → plan →
  search → retrieve → fetch_url → compress → synthesize → verify) with
  2026-SOTA composition (HyDE, CoVe, iterative retrieval, FLARE,
  classifier router, step critic, LongLLMLingua-lite compression,
  cross-encoder rerank, Anthropic contextual chunking). 30+ env-toggle
  flags for leave-one-out ablation.
- `models.py` — OpenAI-compatible LLM plumbing, `_chat` + `_chat_stream`,
  small-model heuristic regex (`_SMALL_MODEL_RE`) for Ollama-class models.
- `trace.py` — W4.3 observability, per-call trace buffer, CLI summary printer.
- `memory.py` — SQLite trajectory log + semantic retrieval.
  Three modes: `off` / `session` / `persistent`. Store at
  `~/.agentic-research/memory.db` by default.
- `compaction.py` — context-window compactor that preserves CoVe-verified
  URLs and recent items, collapses older evidence via a single LLM call.
- `domains.py` — YAML preset loader + hand-rolled parser (no PyYAML dep).
  Six shipped presets: general, medical, papers, financial, stock_trading,
  personal_docs.
- `plugins.py` — disk-backed registry for Claude plugins and Hermes skills.
  Sources: `gh:owner/repo[@ref]`, `file:/path`, `https://marketplace.json`.
  Safety scan for forbidden symbols before any install.

**Three interfaces** (`engine/interfaces/`):
- `cli.py` — rich stdout CLI with subcommands (`ask`, `reset-memory`,
  `memory-count`, `domains`, `version`). Flags for `--domain`, `--memory`,
  `--api-key`, `--model`, `--output {markdown,json}`. Bare-question form
  auto-routes to `ask`.
- `tui.py` — Textual keyboard-driven TUI. Panes: sources, answer +
  hallucination flags, trace + memory hits. Works over SSH.
- `web/app.py` — FastAPI + HTMX + Jinja2. Local-only on
  `http://127.0.0.1:8080`. Dark CSS. SSE-capable streaming. Input
  validation for `domain` + `memory` against known values.

**MCP + Claude plugin distribution** (`engine/mcp/`):
- `server.py` — Python MCP server via `FastMCP`. Tools: `research`,
  `reset_memory`, `memory_count`. Stdio transport (MCP default).
- `claude_plugin/` — submittable Claude plugin bundle with
  `.claude-plugin/plugin.json` + four skills
  (`/research`, `/cite-sources`, `/verify-claim`, `/set-domain`).

**Benchmark harness** (`engine/benchmarks/`):
- `runner.py` — JSONL fixture reader with `must_contain` / `must_not_contain`
  scoring + ablation flag translation (rerank, no-fetch, no-compress,
  no-verify, no-flare, no-router).
- `simpleqa_mini.jsonl` — 20 self-referential factoid questions.
- `browsecomp_mini.jsonl` — 10 multi-hop + synthesis questions across
  the six domains.
- `RESULTS.md` — Phase 1 Gemma 3 4B characterization numbers.

**Domain presets** (`engine/domains/*.yaml`):
- `general.yaml` — broad default, no biases.
- `medical.yaml` — PubMed/Cochrane/NEJM bias; `min_verified_ratio: 0.75`;
  no prescriptive medical advice.
- `papers.yaml` — arXiv/Semantic-Scholar/OpenReview bias; per-paper
  structured output.
- `financial.yaml` — SEC/Reuters/Bloomberg bias; every numeric claim
  requires a date; `min_verified_ratio: 0.80`.
- `stock_trading.yaml` — news category, **hard rule: never recommends
  buy/sell/hold** (shares forbidden-symbols safety rail with
  trading-copilot recipe).
- `personal_docs.yaml` — air-gapped; only `corpus://` URLs allowed.

**Worked examples** (`engine/examples/*.md`):
- `01_medical_covid_treatment.md` — Paxlovid / molnupiravir evidence.
- `02_paper_contextual_retrieval.md` — reproducible from Phase 1.
- `03_financial_nvda_earnings.md` — NVDA 10-Q / 10-K summary.
- `04_technical_rerank_comparison.md` — reranker vs contextual retrieval.
- `05_personal_pdf_summary.md` — bring-your-own-PDFs.

**Google Colab tutorials** (`tutorials/`):
- `01_engine_api_quickstart.ipynb` — mocked end-to-end walk through the
  engine API with no local setup required.
- `02_groq_cloud_inference.ipynb` — live inference via a free-tier Groq
  API key for users who don't have a local GPU.
- `03_build_your_own_corpus.ipynb` — upload PDFs, index them, query.
- `04_mcp_server_from_python.ipynb` — call the MCP server's `research`
  tool from a Python client.
- `05_domain_presets_showcase.ipynb` — compare medical, papers,
  financial domain presets side-by-side.

**Docs** (`docs/`):
- `architecture.md` — deep technical spec (pipeline, memory, compaction,
  plugins, MCP, trace, env vars, threading, failure modes).
- `plugins-skills.md` — clear terminology + writing + installing plugins.
- `domains.md` — preset schema + writing a new one.
- `self-learning.md` — trajectory model + memory retrieval.
- `launch-checklist.md` — go-live sequence.
- `launch-copy.md` — drafted HN / Reddit / Twitter / Discord copy.

**Contribution infrastructure**:
- `CONTRIBUTING.md` — rewritten; three principles (honesty, privacy,
  runs-on-a-laptop), 8 good-first-issue candidates, plugin + domain
  submission lanes, no Co-Authored-By policy.
- `CODE_OF_CONDUCT.md` — Contributor Covenant v2.1.
- `.github/ISSUE_TEMPLATE/` — bug, feature, plugin templates.
- `.github/PULL_REQUEST_TEMPLATE.md`.
- `.github/workflows/engine-tests.yml` — CI matrix running the mocked
  suite on every PR affecting engine / core / recipes.

### Fixed during the 0.1 push (gap-analysis round)

- **Domain presets are now actually applied**. `run_query()` was
  accepting `domain=…` but the 6 YAML presets were dead code. Now
  `_apply_domain_preset()` loads the preset, injects env-var overrides
  (LOCAL_CORPUS_PATH, TOP_K_EVIDENCE), and appends the preset's
  synthesize_prompt_extra to the question so the synthesize node
  honors the domain rules. Unknown domain falls back to `general`
  with a stderr warning.
- **Plugin frontmatter parser now unquotes list items**. Previously
  `triggers: ["run-now"]` became `['"run-now"']` — broke skill trigger
  matching. Double and single quotes both handled.
- **Web app validates `domain` + `memory`**. Malformed POSTs
  previously crashed with opaque internal errors; now return 400
  with the list of valid values.
- **Corpus load distinguishes "not set" from "broken"**. FileNotFoundError
  → quiet skip (user hasn't built the index yet). Any other Exception
  → loud stderr warning with `LOAD BROKEN` + the exception type/message
  + the fallback behavior.
- **Stream fallback logs instead of swallowing errors**. When the LLM
  backend rejects `stream=True`, `_chat_stream()` falls back to batched
  `_chat()` but now prints the exception type on stderr (only when
  `ENABLE_TRACE=1`, to keep scripted runs clean).
- **Removed unused imports** in `cli.py` (`_print_trace_summary`) and
  `tui.py` (`ListItem`, `ListView`, `Label`).
- **`general.yaml` no longer has an empty `top_k_evidence:` key** —
  commented out with a note about the pipeline's own default.

### Known limitations

- Gemma 3 4B is 15–25 % below 30 B+ open models on complex multi-hop
  reasoning. We don't claim to beat GPT-5.4 Pro or MiroThinker-H1 —
  we claim to be the best `$0` research agent that runs on a laptop.
- `tools_enabled:` field in domain presets is declared but not yet
  wired to specialist tools. Planned for 0.2.
- No LoRA fine-tuning loop in v1 (deferred until GPU access +
  trajectory-data volume justify it).
- Hosted SaaS not shipped — local-first only for v1.
- Full ablation-matrix numbers on SimpleQA-Mini + BrowseComp-Mini land
  in the 0.2 README once a fresh run completes post-rename.

---

## Pre-0.1 wave history

The Engine Master Plan (Phases 0–8) followed eight "waves" of feature
work while the repo was still branded as `agentic-research-engine-oss`.
Brief summary; full per-wave prose in [`docs/progress.md`](docs/progress.md).

### Wave 8 — `document-qa` recipe (third flagship, corpus-only pipeline)
Shipped a 4-node corpus-only research recipe as a standalone example of
`core/rag`'s `CorpusIndex` in action. 170 LOC main + 10 mocked tests.

### Wave 7 — streaming synthesis
Tokens stream to stdout as the synthesizer generates. Graceful fallback
to batched on backends that reject `stream=True`. 6 new tests.

### Wave 6 — small-model hardening
Three-case synthesize prompt (FULL / partial with gap flagging /
UNRELATED → refuse) + per-chunk char cap + small-model TopK heuristic.
Killed the "9 hallucination failure mode we observed on gemma4:e2b.
8 new tests.

### Wave 5 — local corpus indexing
`core/rag/python/corpus.py` + `scripts/index_corpus.py` CLI. PDFs / md /
txt / HTML. Paragraph-aware chunker with overlap. 13 + 8 new tests.

### Wave 4 — local-first engine enhancements
Cross-encoder reranker wired into `_retrieve` (was shipped as an
unused library in Wave 2). `_fetch_url` node via trafilatura. W4.3
observability trace. 16 new tests.

### Wave 3 — `trading-copilot` recipe end-to-end
Market research + alerts, no auto-execution. Build-time forbidden-
symbols tests. 30 new tests.

### Wave 2 — full SOTA stack on research-assistant production
Tier 2 adaptive verification (HyDE / CoVe / iteration / self-consistency)
+ Tier 3 evaluation harness + Tier 4 2026 SOTA techniques (step critic,
FLARE, classifier router, compression, plan refinement).

### Wave 1 — research-assistant beginner
First runnable recipe. Plan → search → retrieve → synthesize. 100 LOC
single file.

### Wave 0 / 0.5 — skeleton + SOTA-per-task pivot
Repo scaffold, initial recipes, pivot away from 4-framework-comparison
matrix to one opinionated implementation per task.

---

[Unreleased]: https://github.com/TheAiSingularity/agentic-research-engine-oss/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/TheAiSingularity/agentic-research-engine-oss/releases/tag/v0.1.0
