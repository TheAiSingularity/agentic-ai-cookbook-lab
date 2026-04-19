# research-assistant/beginner

The canonical beginner-tier research assistant. LangGraph wires a 4-node
pipeline that answers research questions with citations — using a single
OpenAI key and no Exa / Tavily / Gemini keys.

## What it does

Give it a research question — it plans sub-queries, searches the web via
OpenAI's built-in `web_search` tool, narrows evidence with `core/rag`, and
synthesizes a cited answer.

```
plan ──▶ search (parallel) ──▶ retrieve ──▶ synthesize
(nano)   (mini + web_search)    (core/rag)   (mini)
```

## Stack

| Step | Tool | Why |
|---|---|---|
| plan | `gpt-5-nano` | Cheapest OpenAI tier — sub-query generation doesn't need reasoning muscle |
| search | `gpt-5-mini` + `web_search` tool via Responses API | Model does multi-step web search internally; returns URL-cited snippets. Run in parallel across sub-queries. |
| retrieve | `core/rag` v0 (OpenAI embeddings + cosine) | Narrows many highlights to top-k most relevant |
| synthesize | `gpt-5-mini` | Strong reasoning where it matters most — the final cited answer |

See [`techniques.md`](techniques.md) for primary-source citations.

## Run

```bash
# Single API key covers every step
export OPENAI_API_KEY=sk-...

# Install and run
make install
make run Q="your research question here"

# Or a canned smoke test
make smoke
```

Expected wall-clock: ~45–90s for a typical query (3 parallel web-search
calls, each doing multi-step reasoning, then one synthesis). Expected
cost: ~$0.05–$0.25 per query — dominated by `web_search` tool fees.

## Test (no API key needed)

```bash
make test
```

Uses fully mocked clients — verifies the graph wiring, node contracts,
parallel fan-out, and state shape without touching any network.

## Override the defaults

```bash
export MODEL_PLANNER=gpt-5-mini        # more capable planner
export MODEL_SEARCHER=gpt-5             # deeper research searcher
export MODEL_SYNTHESIZER=gpt-5          # better final synthesis
export NUM_SUBQUERIES=5                 # broader fan-out
export TOP_K_EVIDENCE=12                # more evidence into synthesis
```

## Files

```
beginner/
├── main.py            # The LangGraph agent (~96 lines, commented)
├── requirements.txt
├── Makefile           # run · smoke · test · install · clean
├── README.md          # you're reading it
├── techniques.md      # primary-source citations for every choice
└── test_main.py       # mocked unit tests (12, all green)
```
