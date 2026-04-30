# Architecture

The engine is a LangGraph pipeline over a shared retrieval stack, with
three parallel interface layers, MCP distribution, and a plugin loader
that respects both the Claude plugin spec and the `agentskills.io`
Hermes skill format. This document is the deep dive — the README has the
pitch; this has the details.

---

## The pipeline (8 nodes)

```
         ┌─────────┐
         │question │
         └────┬────┘
              ▼
   ┌──────────────────────────┐
   │ classify                 │  T4.3 · route by question type
   │ {factoid|multihop|       │
   │  synthesis}              │
   └────────────┬─────────────┘
                ▼
   ┌──────────────────────────┐
   │ plan (+ HyDE + critic    │  T1 decompose · T4.1 step verify
   │      + T4.5 refine)      │  T2 HyDE · T4.5 re-plan on reject
   └────────────┬─────────────┘
                ▼
   ┌──────────────────────────┐
   │ search (parallel)        │  SearXNG × n + W5 corpus hits
   │   + W5 local corpus      │
   │   + T4.1 coverage critic │
   └────────────┬─────────────┘
                ▼
   ┌──────────────────────────┐
   │ retrieve                 │  T1 hybrid BM25+dense+RRF
   │   + W4.1 rerank (opt)    │  W4.1 cross-encoder BAAI/bge-reranker-v2-m3
   └────────────┬─────────────┘
                ▼
   ┌──────────────────────────┐
   │ fetch_url                │  W4.2 trafilatura clean-text
   │   (skips corpus:// URLs) │
   └────────────┬─────────────┘
                ▼
   ┌──────────────────────────┐
   │ compress (+ W6.2 cap)    │  T4.4 LLM distillation +
   │                          │  W6.2 per-chunk hard cap
   └────────────┬─────────────┘
                ▼
   ┌──────────────────────────┐
   │ synthesize (+ FLARE)     │  T2 synth · T4.2 FLARE on hedges
   │   W7 streaming           │  W6.1 three-case anti-hallucinate
   └────────────┬─────────────┘
                ▼
   ┌──────────────────────────┐
   │ verify (CoVe)            │  T2 claim decomposition + check
   └──┬───────────────────────┘
      │
      ▼
  ┌───────────────────┐
  │ verified?         │    if unverified && iter < MAX: back to search
  └─┬─────────────────┘                           else: END
    │
    ▼
  [END]
```

Every node is individually env-toggleable for leave-one-out ablation
(see `engine/core/pipeline.py` header for the full list of `ENABLE_*`
flags). That lets contributors add a new node and measure its
contribution without mutating existing tests.

### Concrete state shape (`TypedDict, total=False`)

```python
class State(TypedDict, total=False):
    question:             str
    question_class:       str                         # "factoid" | "multihop" | "synthesis"
    subqueries:           list[str]
    evidence:             list[dict]                  # [{url, title, text, fetched}]
    evidence_compressed:  list[dict]
    answer:               str
    claims:               list[dict]                  # [{text, verified}]
    unverified:           list[str]
    iterations:           int
    plan_rejects:         int
    trace:                list[dict]                  # [{node, model, latency_s, tokens_est, …}]
```

Nodes mutate by returning partial dicts that LangGraph merges into the
state — no mutable in-place updates.

---

## Module layout (where the code actually lives)

```
engine/
├── core/                        pipeline + shared primitives
│   ├── pipeline.py              the 8 nodes + build_graph
│   ├── models.py                LLM plumbing + model routing + small-model regex
│   ├── trace.py                 W4.3 observability
│   ├── memory.py                trajectory log + semantic retrieval
│   ├── compaction.py            context-limit compactor
│   ├── domains.py               preset loader + YAML parser
│   └── plugins.py               Claude-plugin + Hermes-skill installer
├── interfaces/
│   ├── common.py                run_query + shared rendering helpers
│   ├── cli.py                   CLI
│   ├── tui.py                   Textual TUI
│   └── web/                     FastAPI + HTMX + Jinja2 GUI
├── mcp/
│   ├── server.py                Python MCP server (stdio, tools: research, reset_memory, memory_count)
│   └── claude_plugin/           submittable Claude plugin bundle
│       ├── .claude-plugin/plugin.json
│       └── skills/{research,cite-sources,verify-claim,set-domain}.md
├── domains/                     6 preset YAMLs
├── examples/                    5 worked examples (integration-test fixtures)
├── benchmarks/                  mini fixtures + runner + RESULTS.md
└── tests/                       pytest for everything in engine/
```

`core/rag/` (the retrieval primitives) lives outside `engine/` so it can
graduate into a standalone library when the criteria in DEC-004 are met.
`recipes/` under `archive/` stays untouched — the research-assistant
production/main.py is a re-export shim over `engine.core.pipeline`
(Phase 1 refactor).

---

## Memory model

`engine/core/memory.py` defines a three-state memory knob:

| mode           | store                                  | persists across runs? |
|---             |---                                     |---                    |
| `off`          | `_NullStore` (no-op)                   | no                    |
| `session`      | in-process list                        | no                    |
| `persistent`   | SQLite at `~/.agentic-research/memory.db` | yes                |

Tables (persistent only):

- `trajectories(query_id, timestamp, question, domain, payload_json)`
- `embeddings(query_id, vec_blob)`

Every query writes one trajectory row + one embeddings row. On subsequent
queries (if memory on), the pipeline retrieves top-K trajectories whose
question-embedding cosine is ≥ `MEMORY_MIN_SCORE` (default 0.55) and
injects their summaries as extra prompt context.

Wipe the store with `engine reset-memory` (CLI), the `reset_memory()`
MCP tool, or by deleting `~/.agentic-research/memory.db`.

---

## Compaction model

When total evidence chars exceed `CONTEXT_LIMIT_CHARS` (default 24 000
for 4 B-class synthesizers), `engine/core/compaction.py` triggers. It:

1. Identifies **load-bearing URLs** — any URL cited in a CoVe-verified
   claim on this query.
2. Preserves those items intact.
3. Preserves the most-recent `COMPACTION_KEEP_RECENT` items (default 3).
4. Runs ONE compactor LLM call over the remaining chunks, collapsing each
   to a one-sentence summary bounded by `COMPACTION_SUMMARY_CHARS` (200).
5. Returns the rebuilt evidence list + stats (`n_in`, `n_compacted`,
   `n_kept`, `chars_before`, `chars_after`).

This is separate from T4.4 `_compress` — compress always runs (on evidence
right after fetch), compaction runs as a capacity safety net before
synthesize when the context pressure is real.

---

## Plugin + skill model

`engine/core/plugins.py` is a disk-backed registry at
`~/.agentic-research/plugins/` with:

- `index.json` — list of installed plugins + source URIs + versions
- `<plugin-name>/` — plugin contents (manifest + skills directory)

Supported sources:

- `gh:<owner>/<repo>[@<ref>]` — git clone (`--depth 1`)
- `file:<absolute-path>` — local directory or single `.md` skill
- `https://.../marketplace.json` — remote manifest (Claude plugin spec)

Supported formats:

- **Claude plugin** (`.claude-plugin/plugin.json` + `skills/*.md`)
- **Hermes / agentskills.io skill** (single `.md` with YAML frontmatter)

Safety:

- Default-deny on unknown schemes.
- Pre-install scan for forbidden symbols (`eval(`, `exec(`, `os.system(`,
  …). Any match rejects the install.
- Plugins do NOT auto-execute on install; they become entries in the
  registry and are surfaced via the plugin manager UI in the
  CLI/TUI/Web GUI.

---

## MCP + Claude plugin distribution

`engine/mcp/server.py` exposes three tools over stdio via the FastMCP SDK:

| tool                | args                                 | returns |
|---                  |---                                   |---      |
| `research`          | `question, domain?, memory?`         | `{answer, verified_claims, unverified_claims, sources, trace, totals, memory_hits}` |
| `reset_memory`      | (none)                               | `{reset: int}` |
| `memory_count`      | (none)                               | `{count: int}` |

The Claude plugin bundle in `engine/mcp/claude_plugin/` is ready for
submission:

- `.claude-plugin/plugin.json` declares name, version, skills, and the
  `engine` MCP server with Ollama defaults pre-wired.
- `skills/research.md`, `skills/cite-sources.md`, `skills/verify-claim.md`,
  `skills/set-domain.md` are the user-facing entry points.

---

## Trace

Every LLM call (including streamed) appends to a module-level
`_TRACE_BUFFER`. Each node drains the buffer and folds the entries into
`state["trace"]` with the node name attached. At CLI/MCP completion, the
aggregate is rendered per-node and per-model.

Structure:

```python
{
    "node":            str,          # classify / plan / search / …
    "model":           str,           # e.g. "gemma3:4b", "hybrid", "trafilatura"
    "latency_s":       float,
    "prompt_chars":    int,
    "response_chars":  int,
    "tokens_est":      int,           # (prompt + response) // 4
    "streamed":        bool            # optional
}
```

**No data leaves the machine.** Trace is local-only; tests enforce this
(see `test_production_main.py::test_full_graph_records_trace_across_nodes`).

---

## Configuration (env vars)

The pipeline has ~35 env gates. Exhaustive list lives in
`engine/core/pipeline.py` (header docstring) and
`engine/core/models.py`. Grouped:

| group              | examples |
|---                 |---                                     |
| endpoint + model   | `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `MODEL_*`, `EMBED_MODEL` |
| retrieval          | `NUM_SUBQUERIES`, `TOP_K_EVIDENCE`, `ENABLE_RERANK`, `RERANK_CANDIDATES` |
| fetch              | `ENABLE_FETCH`, `FETCH_MAX_CHARS`, `FETCH_MAX_URLS`, `FETCH_TIMEOUT_SEC` |
| compress / synth   | `ENABLE_COMPRESS`, `PER_CHUNK_CHAR_CAP`, `ENABLE_STREAM`, `ENABLE_CONSISTENCY` |
| verify             | `ENABLE_VERIFY`, `MAX_ITERATIONS`, `ENABLE_ACTIVE_RETR` |
| router + critic    | `ENABLE_ROUTER`, `ENABLE_STEP_VERIFY`, `ENABLE_PLAN_REFINE` |
| small-model        | `SMALL_MODEL_TOPK`, `PER_CHUNK_CHAR_CAP` |
| memory             | `MEMORY_DB_PATH`, `MEMORY_TOP_K`, `MEMORY_MIN_SCORE` |
| compaction         | `ENABLE_COMPACTION`, `CONTEXT_LIMIT_CHARS`, `COMPACTION_KEEP_RECENT` |
| corpus             | `LOCAL_CORPUS_PATH`, `LOCAL_CORPUS_TOP_K`, `EMBED_MODEL` |
| web search         | `SEARCH_PROVIDER` (`searxng` default, or `exa`), `SEARXNG_URL`, `EXA_API_KEY`, `EXA_SEARCH_TYPE`, `EXA_CATEGORY`, `EXA_INCLUDE_DOMAINS`, `EXA_EXCLUDE_DOMAINS`, `EXA_START_PUBLISHED_DATE`, `EXA_END_PUBLISHED_DATE`, `EXA_HIGHLIGHTS_CHARS` |
| trace              | `ENABLE_TRACE` |

---

## Threading model

- `_search` fans out sub-queries in parallel via `ThreadPoolExecutor`.
- `_fetch_url` fans out URL fetches in parallel via `ThreadPoolExecutor`.
- Every other node is sequential.
- `_TRACE_BUFFER` is a plain list; CPython's GIL makes `.append`
  thread-safe. Entry ordering may be non-deterministic across parallel
  threads but each entry is intact.

---

## Failure modes and degradation

| component           | on failure | observable |
|---                  |---         |---         |
| Ollama down         | OpenAI API error exception | pipeline aborts with clear error |
| SearXNG down        | empty evidence list | `_retrieve` passes through; synthesize has nothing to cite → refusal |
| trafilatura import  | `_fetch_one` returns None | snippet used; `fetched=False` flag in source |
| cross-encoder model | graceful fallback to hybrid-only | stderr note `[rerank] falling back …` |
| CorpusIndex load    | `_CORPUS_LOAD_FAILED` cached | next query uses web-only; warning on stderr |
| Plugin safety scan  | install rejected with RuntimeError | no partial state on disk |
| Streaming unsupported | `_chat_stream` falls back to `_chat` | no user-visible change |

---

## Dependencies

Runtime (from `engine/requirements.txt`):

- `langgraph` — orchestration
- `openai` — OpenAI-compatible client
- `rank_bm25` — sparse retrieval
- `sentence-transformers` — optional cross-encoder reranker
- `trafilatura` — HTML → clean text
- `pypdf` — PDF ingestion for CorpusIndex
- `textual` + `rich` — TUI
- `fastapi` + `uvicorn` + `jinja2` + `sse-starlette` — web GUI
- `mcp` — MCP server SDK
- `pyyaml` — listed but not currently used (domain preset parser is
  hand-rolled to avoid the dep)

All MIT or Apache-2.0.

External services (all self-hostable, zero-cost):

- **Ollama** — local inference
- **SearXNG** — local meta-search (Docker)
- **Hugging Face** — one-time download of reranker model if enabled
