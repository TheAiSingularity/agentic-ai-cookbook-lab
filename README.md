<p align="center">
  <strong>agentic-research-engine-oss</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/pypi/v/agentic-research-engine?color=blue&label=pypi" alt="PyPI">
  <img src="https://img.shields.io/badge/version-0.1.3--alpha-orange.svg" alt="Version">
  <img src="https://img.shields.io/badge/default-gemma%203%204B%20local-green.svg" alt="Default">
  <img src="https://img.shields.io/badge/tests-137%2F137-brightgreen.svg" alt="Tests">
  <img src="https://img.shields.io/badge/interfaces-CLI%20%7C%20TUI%20%7C%20web-blue.svg" alt="Interfaces">
  <img src="https://img.shields.io/badge/MCP-python%20server%20%2B%20claude%20plugin-6aa3ff.svg" alt="MCP">
</p>

**The best $0 research agent that runs on a laptop.** Open-source
end-to-end, reproducible, privacy-preserving. No cloud dependency by
default; no telemetry; every LLM call, every source, and every
verification decision is visible.

---

## Table of contents

- [TL;DR](#tldr)
- [Why use this instead of…](#why-use-this-instead-of)
- [Quickstart — Mac local](#quickstart--mac-local)
- [Quickstart — no install (Google Colab)](#quickstart--no-install-google-colab)
- [Three ways to drive it](#three-ways-to-drive-it)
- [What ships](#what-ships)
- [Domain presets](#domain-presets)
- [Bring your own documents](#bring-your-own-documents)
- [MCP + Claude plugin](#mcp--claude-plugin)
- [Plugin / skill loader](#plugin--skill-loader)
- [Architecture at a glance](#architecture-at-a-glance)
- [Repo layout](#repo-layout)
- [Configuration (env vars)](#configuration-env-vars)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Honest limits](#honest-limits)
- [Status + roadmap](#status--roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## TL;DR

**Local-first research agent that verifies its own answers.** Runs on
Gemma 3 4B + Ollama (3.3 GB on disk) for `$0/query`; swaps to any
OpenAI-compatible endpoint with one env var.

```bash
pip install agentic-research-engine
agentic-research ask "what is Anthropic's contextual retrieval?" --domain papers
```

| | |
|---|---|
| **Interfaces** | CLI · Textual TUI · FastAPI web GUI · MCP server (Claude Desktop / Cursor / Continue) |
| **Pipeline** | 8-node LangGraph (`classify → plan → search → retrieve → fetch → compress → synthesize → verify`); every node env-toggleable for ablation |
| **Retrieval** | SearXNG meta-search + trafilatura fetch + hybrid BM25 / dense / RRF; opt-in `bge-reranker-v2-m3` cross-encoder |
| **Reasoning** | HyDE query expansion · FLARE active retrieval · Chain-of-Verification (Dhuliawala et al 2023) · ThinkPRM step critic |
| **Domains** | 6 presets (`general` · `medical` · `papers` · `financial` · `stock_trading` · `personal_docs`) — write your own in 10 lines of YAML |
| **Plugins** | load Claude plugins or `agentskills.io` skills from GitHub or local paths |
| **Memory** | opt-in local SQLite trajectory log with semantic retrieval; wipe anytime; no telemetry |
| **Providers** | OpenAI · Groq · vLLM · SGLang · Together · Ollama — any OpenAI-compatible endpoint via `OPENAI_BASE_URL` |
| **Quality** | 137 mocked tests, zero-network · honest live benchmarks published in [`RESULTS.md`](engine/benchmarks/RESULTS.md) · MIT end-to-end |

---

## Why use this instead of…

| you currently use | we give you |
|---|---|
| **Perplexity / ChatGPT Deep Research / Kagi Assistant** | the same reasoning-with-citations flow, **local and free**, with your data never leaving the machine |
| **Perplexica self-hosted** | the UX Perplexica has plus a CoVe verifier, FLARE active retrieval, adaptive compute router, and Claude-plugin packaging |
| **Khoj** | stronger research-specific reasoning (we're not personal-knowledge-focused), six domain presets, and an MCP server for other agents to call |
| **gpt-researcher** | newer pipeline architecture, better small-model handling, observable trace, plugin ecosystem |
| **MiroThinker-H1 / OpenResearcher-30B** | they're stronger on BrowseComp; we run on a laptop with no GPU and cost $0 |
| **Writing your own LangGraph research agent** | save 2-3 months; reuse our 8-node pipeline + 30+ tested env gates + 137 tests |

**Honest read:** on complex multi-hop reasoning benchmarks, Gemma 3 4B
sits 15–25% below 30 B+ open models. We don't claim to beat GPT-5.4
Pro. We claim to be the best **$0, runs-on-your-laptop, fully-open**
research agent in April 2026.

---

## Quickstart — Mac local

### Option A — PyPI (fastest)

```bash
# 1) Local inference (Ollama + Gemma 3 4B + embedding model — 3.6 GB combined)
brew install ollama
ollama pull gemma3:4b nomic-embed-text

# 2) Self-hosted meta-search (Docker; optional but recommended)
docker run -d --name searxng -p 8888:8080 searxng/searxng

# 3) The engine itself
pip install agentic-research-engine

# 4) Go
export OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama
export MODEL_SYNTHESIZER=gemma3:4b EMBED_MODEL=nomic-embed-text
export SEARXNG_URL=http://localhost:8888
agentic-research ask "what is Anthropic's contextual retrieval?" --domain papers
```

### Option B — from source

```bash
# 1) Same local-inference prereqs as Option A (ollama pull + docker run)

# 2) Clone + install (gives you the CLI, TUI, Web GUI, MCP server, benchmarks, tutorials)
git clone https://github.com/TheAiSingularity/agentic-research-engine-oss
cd agentic-research-engine-oss
(cd scripts/searxng && docker compose up -d)
cd engine && make install
make smoke    # end-to-end run on the canonical "what is contextual retrieval" question
```

Expected wall-clock on an M-series Mac: **~45 s** for a factoid,
~90 s for multi-hop synthesis. Zero dollars per query.

### Higher honesty — cloud-model mode

Gemma 3 4B is surprisingly good at **structure** (plan, route, verify,
compress) but confabulates **specific factoids** when SearXNG doesn't
surface a source containing the right token. Live SimpleQA-mini run on
2026-04-21 (see [`engine/benchmarks/RESULTS.md`](engine/benchmarks/RESULTS.md))
showed `gemma3:4b` emitting "2023" for *"year Anthropic published
Contextual Retrieval"* (gold: 2024) and "LayoutLMv3" for *"which
cross-encoder for reranking"* (gold: bge-reranker-v2-m3).

The fix you probably want isn't a smarter synthesizer — it's a
**more honest** one. A 5-question head-to-head on the same retrieval
output showed `gpt-5-nano` + `gpt-5-mini` refuse to confabulate when
evidence was missing (*"The provided evidence does not answer this
question"*), where `gemma3:4b` confidently guessed. Per-claim
faithfulness went from 82.9 % → 100 %. Pass rate barely moved (1/5
vs 0/5) because **retrieval is the real bottleneck** — if SearXNG
didn't return a source with the gold token, neither model can
produce it.

Swap the whole stack to a cloud endpoint:

```bash
# drop the Ollama base URL (fall back to OpenAI cloud)
unset OPENAI_BASE_URL
export OPENAI_API_KEY=sk-...
# defaults are already cloud-sized: gpt-5-nano for plan/verify, gpt-5-mini for synth.
# Explicit override if you want to pin them:
export MODEL_PLANNER=gpt-5-nano
export MODEL_SYNTHESIZER=gpt-5-mini        # or gpt-5, claude-sonnet-4-5, etc.
agentic-research ask "…" --domain papers
```

Cost is dominated by synthesizer tokens (~5–15 k per query). Full
cloud mode with `gpt-5-nano` + `gpt-5-mini` runs roughly
**$0.02–0.05 per research query** and is ~2-3× slower than Gemma
local (measured: 127 s vs 52 s mean wall on the 5-question subset).
Works with any OpenAI-compatible endpoint — Groq, Together, Mistral,
DeepSeek, local vLLM — so you can pick a cheap fast model
(`llama-3.3-70b` on Groq ≈ $0.003/query) or a frontier one. Per-node
base-URL routing (run gemma3:4b locally for plan/verify AND gpt-5-mini
on cloud for synth in the same query) is tracked for 0.2; today the
pipeline uses one global `OPENAI_BASE_URL`.

**The bigger accuracy lever is retrieval.** Point
`LOCAL_CORPUS_PATH` at an indexed corpus containing your answer and
either model will be correct.

---

## Quickstart — no install (Google Colab)

Five runnable notebooks in [`tutorials/`](tutorials/):

1. [**01 — Engine API quickstart** (mocked, no key)](tutorials/01_engine_api_quickstart.ipynb) — see how the pipeline works without running inference.
2. [**02 — Groq cloud inference** (free tier)](tutorials/02_groq_cloud_inference.ipynb) — real LLM, no local GPU.
3. [**03 — Build your own corpus**](tutorials/03_build_your_own_corpus.ipynb) — upload PDFs, index them, query.
4. [**04 — MCP server from Python**](tutorials/04_mcp_server_from_python.ipynb) — drive the engine as a tool from another agent.
5. [**05 — Domain presets showcase**](tutorials/05_domain_presets_showcase.ipynb) — compare presets on the same question.

Each notebook is self-contained, runs end-to-end on Colab free tier, no
credit card required.

---

## Three ways to drive it

### CLI

```bash
engine ask "what is hybrid retrieval?" --domain papers --memory session
engine reset-memory
engine domains list
engine version
```

### TUI (Textual — keyboard-driven, SSH-safe)

```bash
make tui
```

Three panes: sources · answer + hallucination flags · trace + memory hits.
Press <kbd>Enter</kbd> to ask, <kbd>Ctrl-M</kbd> to cycle memory mode,
<kbd>Ctrl-L</kbd> to clear, <kbd>Ctrl-Q</kbd> to quit.

### Web GUI (FastAPI + HTMX on `localhost:8080`)

```bash
make gui
# open http://127.0.0.1:8080 in your browser
```

No auth. No cloud. No analytics. Dark theme. Streams tokens in place.

---

## What ships

### `engine/` — the flagship

8-node LangGraph pipeline with 2026-SOTA composition:
`classify → plan → search → retrieve → fetch_url → compress → synthesize → verify`

Every stage is env-toggleable for leave-one-out ablation. Techniques
folded in: HyDE, CoVe verification, iterative retrieval, FLARE active
retrieval, question classifier router, step critic (ThinkPRM pattern),
LongLLMLingua-lite compression, cross-encoder rerank
(`BAAI/bge-reranker-v2-m3`), Anthropic contextual chunking, W6 small-
model hardening (three-case synthesize prompt + per-chunk char cap).

### `core/rag/` — reusable retrieval primitives (v1 stable)

`HybridRetriever` (BM25 + dense + RRF) · `CrossEncoderReranker` ·
`contextualize_chunks` (Anthropic pattern) · `CorpusIndex` (bring-
your-own-PDFs). 5 exports, used by the engine and the archived
recipes.

### `archive/recipes/` — pre-engine reference recipes

`research-assistant`, `trading-copilot`, `document-qa`,
`rust-mcp-search-tool`. All still work; all tests still pass. The
`research-assistant/production/main.py` is a thin shim over
`engine.core.pipeline` so the cookbook framing is preserved.

---

## Domain presets

Six YAML files in `engine/domains/`:

| preset | when to use |
|---|---|
| `general` | default; anything |
| `medical` | disease / treatment / drug / trial (PubMed / Cochrane / NEJM bias; no prescriptive advice) |
| `papers` | academic CS / ML / physics / biology (arXiv + Semantic Scholar + OpenReview) |
| `financial` | SEC filings, earnings, company fundamentals (dates on every number) |
| `stock_trading` | technical + news per ticker — **hard rule: never recommends buy/sell/hold** |
| `personal_docs` | Q&A over your own corpus, air-gapped (only `corpus://` URLs allowed) |

Write your own in ~10 lines of YAML — see [`docs/domains.md`](docs/domains.md).

---

## Bring your own documents

```bash
python scripts/index_corpus.py build ~/papers --out ~/papers.idx
export LOCAL_CORPUS_PATH=~/papers.idx
engine ask "what do my papers say about contextual retrieval?" --domain personal_docs
```

Supported formats: PDF (via pypdf), Markdown, plain text, HTML (via
trafilatura). The index persists as a directory with a human-readable
`manifest.json` + a pickled `index.pkl`. Rebuild anytime the docs change.

Details: [`docs/self-learning.md`](docs/self-learning.md) covers the
trajectory + memory model; [`docs/plugins-skills.md`](docs/plugins-skills.md)
covers external plugins.

---

## MCP + Claude plugin

`engine/mcp/server.py` is a Python MCP server exposing:
- `research(question, domain?, memory?)` → structured `{answer, verified_claims, unverified_claims, sources, trace, totals, memory_hits}`
- `reset_memory()`
- `memory_count()`

Bundled Claude plugin at `engine/mcp/claude_plugin/` — four skills
(`/research`, `/cite-sources`, `/verify-claim`, `/set-domain`), ready to
submit to the Anthropic marketplace.

Register in Claude Desktop:

```jsonc
// ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "engine": {
      "command": "python",
      "args": ["-m", "engine.mcp.server"],
      "env": {
        "OPENAI_BASE_URL": "http://localhost:11434/v1",
        "OPENAI_API_KEY":  "ollama",
        "MODEL_SYNTHESIZER": "gemma3:4b",
        "SEARXNG_URL":    "http://localhost:8888"
      }
    }
  }
}
```

---

## Plugin / skill loader

Install third-party Claude plugins or Hermes (`agentskills.io`) skills:

```bash
engine plugins install gh:owner/some-research-plugin@v1
engine plugins install file:./my-local-plugin
engine plugins install https://example.com/marketplace.json
engine plugins list
engine plugins uninstall some-plugin
```

Safety: every install runs a forbidden-symbols scan
(`eval(`, `exec(`, `os.system(`, …) — rejects plugins that would
execute arbitrary code. Registry lives at
`~/.agentic-research/plugins/`, fully inspectable, wipable.

Full docs: [`docs/plugins-skills.md`](docs/plugins-skills.md).

---

## Architecture at a glance

```
                ┌─────────────┐
                │   question  │
                └──────┬──────┘
                       ▼
           ┌─────────────────────────┐   T4.3 router  — route by question type
           │  classify               │
           └──────────┬──────────────┘
                      ▼
           ┌─────────────────────────┐   T1 decompose · T2 HyDE · T4.1 critic
           │  plan                   │   T4.5 refine-on-reject
           └──────────┬──────────────┘
                      ▼
           ┌─────────────────────────┐   SearXNG parallel × N
           │  search                 │   + W5 local corpus (optional)
           │  (+ T4.1 critic)        │   + T4.1 coverage critic
           └──────────┬──────────────┘
                      ▼
           ┌─────────────────────────┐   T1 hybrid BM25 + dense + RRF
           │  retrieve               │   W4.1 cross-encoder rerank (opt-in)
           │  (+ W4.1 rerank)        │
           └──────────┬──────────────┘
                      ▼
           ┌─────────────────────────┐   W4.2 trafilatura clean-text
           │  fetch_url              │   skips corpus:// URLs
           └──────────┬──────────────┘
                      ▼
           ┌─────────────────────────┐   T4.4 LLM distillation
           │  compress               │   + W6.2 per-chunk char cap
           │  (+ W6.2 cap)           │
           └──────────┬──────────────┘
                      ▼
           ┌─────────────────────────┐   T2 synth · T4.2 FLARE on hedges
           │  synthesize             │   W6.1 three-case anti-hallucinate
           │  (+ FLARE + stream)     │   W7 streaming
           └──────────┬──────────────┘
                      ▼
           ┌─────────────────────────┐   T2 CoVe — decompose + verify
           │  verify                 │
           └────────┬────────────────┘
                    │
              verified? ── yes ──▶ END
                    │
                    no
                    │
           ◀────── re-search unverified claims ──── loop (bounded by MAX_ITERATIONS)
```

Every stage has an `ENABLE_*` flag so you can leave-one-out ablate.
Deep spec: [`docs/architecture.md`](docs/architecture.md).

---

## Repo layout

```
agentic-research-engine-oss/
├── engine/                        the flagship research engine
│   ├── core/                      pipeline · models · trace · memory
│   │   ├── pipeline.py              · compaction · domains · plugins
│   │   ├── models.py
│   │   ├── trace.py
│   │   ├── memory.py
│   │   ├── compaction.py
│   │   ├── domains.py
│   │   └── plugins.py
│   ├── interfaces/
│   │   ├── cli.py                 rich stdout CLI with subcommands
│   │   ├── tui.py                 Textual TUI
│   │   └── web/                   FastAPI + HTMX localhost GUI
│   ├── mcp/
│   │   ├── server.py              Python FastMCP server
│   │   └── claude_plugin/         submittable Claude plugin bundle
│   ├── domains/                   6 YAML presets
│   ├── examples/                  5 worked research examples
│   ├── benchmarks/                mini SimpleQA + BrowseComp fixtures + runner
│   └── tests/                     pytest suite (all mocked, zero-network)
├── core/rag/                      shared retrieval primitives (stable v1)
├── archive/                       pre-engine recipes (kept for reference)
├── tutorials/                     5 Google Colab notebooks
│   ├── 01_engine_api_quickstart.ipynb
│   ├── 02_groq_cloud_inference.ipynb
│   ├── 03_build_your_own_corpus.ipynb
│   ├── 04_mcp_server_from_python.ipynb
│   └── 05_domain_presets_showcase.ipynb
├── scripts/
│   ├── searxng/                   self-hosted meta-search (docker-compose)
│   ├── setup-local-mac.sh         Ollama + Docker + SearXNG one-liner
│   ├── setup-vm-gpu.sh            Linux + vLLM/SGLang setup
│   └── index_corpus.py            build a CorpusIndex from PDFs/md/txt
├── docs/
│   ├── architecture.md            deep technical spec
│   ├── plugins-skills.md          write + install plugins
│   ├── domains.md                 write a new preset
│   ├── self-learning.md           trajectory logging + memory
│   ├── progress.md                wave-by-wave build log
│   ├── how-it-works.md            elevator pitches + SOTA comparison
│   ├── launch-checklist.md        go-live sequence
│   └── launch-copy.md             drafted HN / Reddit / Twitter copy
├── .github/
│   ├── workflows/
│   │   └── engine-tests.yml       CI: mocked suite on every PR
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── LICENSE                        MIT
└── README.md                      you're reading it
```

---

## Configuration (env vars)

Full list in `engine/core/pipeline.py` header. Most-common knobs:

| var | default | purpose |
|---|---|---|
| `OPENAI_BASE_URL` | unset (cloud OpenAI) | route to Ollama / vLLM / Groq / etc. |
| `OPENAI_API_KEY` | `ollama` | sentinel for local; real key for cloud |
| `MODEL_SYNTHESIZER` | `gpt-5-mini` (cloud) or `gemma3:4b` (Mac-local path) | final-answer model. Swap to `gpt-5`, `claude-sonnet-4-5`, `llama-3.3-70b` on Groq, etc., for higher factoid accuracy while keeping the rest of the pipeline local. |
| `TOP_K_EVIDENCE` | auto (5 for small, 8 for large models) | retrieval budget |
| `ENABLE_RERANK` | `0` | opt-in; first run downloads bge-reranker-v2-m3 (~560 MB) |
| `ENABLE_FETCH` | `1` | trafilatura full-page fetch |
| `ENABLE_STREAM` | `1` | stream synthesis tokens to stdout |
| `ENABLE_TRACE` | `1` | per-call observability + summary at CLI end |
| `LOCAL_CORPUS_PATH` | unset | set to an index dir to augment search with your docs |
| `MEMORY_DB_PATH` | `~/.agentic-research/memory.db` | SQLite trajectory store |
| `SEARCH_PROVIDER` | `searxng` | swap the web-search backend; set to `exa` to use Exa AI-powered search (requires `pip install exa-py` + `EXA_API_KEY`) |
| `EXA_API_KEY` | unset | API key for the Exa provider; only read when `SEARCH_PROVIDER=exa` |

Full list: [`docs/architecture.md`](docs/architecture.md) env-vars section.

---

## Testing

```bash
cd engine && make test     # 120+ mocked tests in engine/tests/
# or repo-wide:
PYTHONPATH=$(pwd) .venv/bin/python -m pytest core/rag recipes engine/tests -q
```

All tests are mocked — no network, no API key, no model downloads. Live
integration smokes are separate (`make smoke`).

CI runs on every push / PR touching engine / core / recipes — see
[`.github/workflows/engine-tests.yml`](.github/workflows/engine-tests.yml).

---

## Troubleshooting

| symptom | likely cause | fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'engine'` | `PYTHONPATH` missing the repo root | `export PYTHONPATH=$(pwd)` from the repo root |
| CLI answer is empty + fast | Ollama not running | `ollama serve` in another terminal, or `ollama list` to check |
| `Connection refused on :8888` | SearXNG not up | `cd scripts/searxng && docker compose up -d` |
| `Connection refused on :11434` | Ollama not running | `ollama serve`, or let the system service start it |
| First `make smoke` hangs ~20 s before output | Model warming up on first request | normal; subsequent queries are faster |
| `ENABLE_RERANK=1` stalls on first run | 560 MB bge-reranker download | wait it out once; cached after |
| `[corpus] LOAD BROKEN` | corrupt or wrong-version index | delete + rebuild via `scripts/index_corpus.py` |
| TUI shows gibberish over SSH | terminal too narrow | resize to ≥ 100 cols; Textual needs space for the 3-pane layout |
| Web GUI shows `Invalid memory mode` | malformed POST | use the form UI; values validated against `off/session/persistent` |
| Streaming cuts off mid-answer | flaky backend | re-run; batched fallback kicks in on next attempt. Set `ENABLE_STREAM=0` if it persists |
| `zsh: command not found: twine` (or similar) after `uv pip install <pkg>` | uv's venv isn't auto-activated by your shell | use `.venv/bin/<cmd> …`, `uv run <cmd> …`, or `source .venv/bin/activate` before running |
| `bad interpreter: .../python3: no such file or directory` after moving or renaming the repo dir | venv shebangs are absolute paths tied to the dir the venv was created in | recreate: `rm -rf .venv && uv venv && uv pip install -e .` (or re-install whatever you had) |
| `make test` says 0 tests collected | wrong CWD | run from the `engine/` dir or set `PYTHONPATH` |
| Claude Desktop doesn't see the plugin | plugin.json in wrong path | `/plugin marketplace add <absolute-path-to>/engine/mcp/claude_plugin` |

Still stuck? Open an issue with the [`bug_report`](.github/ISSUE_TEMPLATE/bug_report.md)
template — include `ollama list`, `engine version`, and the error.

---

## Honest limits

- **Gemma 4B ≠ GPT-5.4 Pro.** 15–25 % below 30 B+ open models on hard
  multi-hop. We position as "best $0 local", not "SOTA."
- **Gemma 3 4B confabulates specific factoids** when SearXNG doesn't
  return a source that contains the right token. Measured on
  SimpleQA-mini: 0/20 strict pass rate (see
  [`engine/benchmarks/RESULTS.md`](engine/benchmarks/RESULTS.md) —
  `verified_ratio` 85.5 %, zero `must_not_contain` hits; the model
  isn't emitting *banned* strings, it's picking wrong ones). Mitigations:
  (a) swap the whole stack to a cloud endpoint (see "Higher factoid
  accuracy" above — `$0.02–0.05/query` with `gpt-5-nano` + `gpt-5-mini`),
  (b) give the engine a `LOCAL_CORPUS_PATH` so your own docs become
  retrieval targets, (c) set `ENABLE_RERANK=1` to bias retrieval
  toward the right sources.
- **CoVe confirms internal consistency, not ground truth.** Every
  synthesized claim is checked against retrieved evidence; claims
  don't get verified *by the world*. If retrieval misses, CoVe will
  still happily verify a confidently-wrong answer. The engine will
  never fabricate citations, but it can confidently repeat wrong
  information that was in its evidence pool.
- **No LoRA fine-tuning in v1.** Trajectory data is collected; actual
  model training deferred until GPU access + data volume.
- **No hosted SaaS.** Local-first is the entire v1 positioning.
- **Team / multi-user features.** Out of scope for v1.
- **General web crawler / own search index.** Not shipping. SearXNG
  stays. A curated research-focused index may land in v2.
- **Mobile.** Not in scope.

---

## Status + roadmap

- **0.1.3 — public alpha** (current). Features listed above; on PyPI +
  the official MCP registry + the Anthropic plugin marketplace. See
  [`CHANGELOG.md`](CHANGELOG.md).
- **0.2** — specialist tool wiring (`tools_enabled` field in presets finally activates), first LoRA run if GPU arrives, plugin catalog in `docs/`.
- **0.3** — team-collab features (shared memory, PR-driven domain presets), desktop app packaging via Tauri.
- **0.4+** — open-work tracked in [GitHub Issues](https://github.com/TheAiSingularity/agentic-research-engine-oss/issues).

---

## Contributing

Good first issues: [`CONTRIBUTING.md`](CONTRIBUTING.md). RFCs for
anything pipeline-scope. Plugin + domain-preset submissions welcome.

No Co-Authored-By trailers; author-as-written-by.

---

## License

MIT. See [`LICENSE`](LICENSE).

### Related (sibling projects)

- [HermesClaw](https://github.com/TheAiSingularity/hermesclaw) — the secure runtime these recipes can run inside
- [NVIDIA/OpenShell](https://github.com/NVIDIA/OpenShell) — kernel-level agent sandbox
- [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) — self-improving agent (whose `agentskills.io` skill format we interoperate with)

---

### MCP registry ownership

This PyPI package is the official source of the MCP server registered at
<https://registry.modelcontextprotocol.io>. The line below is the
ownership marker the registry validates — **do not remove** when
editing this README.

mcp-name: io.github.TheAiSingularity/agentic-research
