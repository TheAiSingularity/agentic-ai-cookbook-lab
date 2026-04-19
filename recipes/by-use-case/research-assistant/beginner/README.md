# research-assistant/beginner

The canonical beginner-tier research assistant. LangGraph wires a 4-node
pipeline that answers research questions with citations.

## What it does

Give it a research question — it plans sub-queries, searches the web via Exa,
narrows evidence with `core/rag`, and synthesizes a cited answer.

```
plan ──▶ search ──▶ retrieve ──▶ synthesize
(Gemini) (Exa)      (core/rag)   (GPT-5 mini)
```

## Stack

| Step | Tool | Why |
|---|---|---|
| plan | Gemini 2.5 Flash | Cheap, 1M ctx — sub-query generation doesn't need reasoning muscle |
| search | Exa `search_and_contents` with highlights | Query-dependent highlights send 50–75% fewer tokens to the LLM |
| retrieve | `core/rag` v0 (cosine) | Narrows many highlights to top-k most relevant |
| synthesize | GPT-5 mini | Highest 2026 agentic-task accuracy where it matters most — the final answer |

See [`techniques.md`](techniques.md) for primary-source citations.

## Run

```bash
# Prerequisites
export EXA_API_KEY=...
export GOOGLE_API_KEY=...    # Gemini
export OPENAI_API_KEY=...    # GPT-5 mini + embeddings (for core/rag)

# Install and run
make install
make run Q="your research question here"

# Or a canned smoke test
make smoke
```

## Test (no API keys needed)

```bash
make test
```

Uses fully mocked clients — verifies the graph wiring, node contracts, and
state shape without touching any network.

## Expected cost

$0.01–$0.03 per query at defaults (3 sub-queries × 3 Exa results × 1 synthesis
call). Swap `MODEL_PLANNER` / `MODEL_SYNTHESIZER` env vars if you want to
route differently.

## Files

```
beginner/
├── main.py            # The LangGraph agent (~100 lines, commented)
├── requirements.txt
├── Makefile           # run · smoke · test · install · clean
├── README.md          # you're reading it
├── techniques.md      # primary-source citations for every choice
└── test_main.py       # mocked unit tests
```
