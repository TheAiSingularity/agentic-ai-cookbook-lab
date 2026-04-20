# Agentic Research — Claude plugin

Local, verifiable, open-source research agent. Runs fully on your machine
via Ollama + Gemma 3 4B + SearXNG + trafilatura. Every source, every LLM
call, every verification decision visible. No cloud dependency. No
telemetry. Apache-2.0 / MIT all the way down.

## What this plugin adds to Claude

Four Claude skills that wrap one MCP server tool:

| skill | what it does |
|---|---|
| `/research` | full research query with cited answer + hallucination check |
| `/cite-sources` | show the sources behind the last research answer |
| `/verify-claim` | run a targeted CoVe verification on a single claim |
| `/set-domain` | route into medical / papers / financial / stock / personal_docs preset |

Under the hood: the `engine` MCP server exposes a single `research()`
tool that returns structured JSON (answer, sources, verified claims,
trace, memory hits). Claude invokes it; the skills format the result.

## Installation

**Prerequisites** (one-time setup on your machine):

```bash
# 1. Ollama + Gemma 3 4B + embeddings model
brew install ollama                 # or: https://ollama.com/download
ollama pull gemma3:4b               # 3.3 GB
ollama pull nomic-embed-text        # 274 MB

# 2. Self-hosted SearXNG (Docker)
cd agentic-ai-cookbook-lab/scripts/searxng && docker compose up -d

# 3. Engine install
cd agentic-ai-cookbook-lab/engine && make install
```

**Plugin install** (in Claude Desktop or Claude Code):

1. Clone this repo: `git clone https://github.com/TheAiSingularity/agentic-ai-cookbook-lab`
2. Register as a marketplace: `/plugin marketplace add <path-to-clone>/engine/mcp/claude_plugin`
3. Install: `/plugin install agentic-research`
4. Verify: `/plugin list` should show `agentic-research` enabled.
5. Try it: `/research What is Anthropic's contextual retrieval?`

## Demo

```
/research Who introduced contextual retrieval and what percentage reduction
           did they report?

A: Contextual Retrieval was introduced by Anthropic [1]. It reduced the
   number of failed retrievals by 49% and, when combined with reranking,
   by 67% [1].

verified_summary: 3/3 claims verified
sources:
  [1] ● https://anthropic.com/news/contextual-retrieval
  [2] ● https://rag-benchmark.example
  ...
```

## Privacy posture

- **No outbound network except to your SearXNG + your Ollama + the URLs
  SearXNG returns to trafilatura.**
- **No analytics, no crash reporting, no model-usage telemetry.** The
  only data that leaves your machine is the HTTP request you explicitly
  send via SearXNG to reach the open web.
- **Memory is local.** `~/.agentic-research/memory.db` (SQLite).
  Inspectable with `sqlite3`. Wipe anytime with `/research reset-memory`
  or `engine reset-memory`.
- **Source code is MIT/Apache-2.0.** `engine/`, `core/rag/`,
  `BAAI/bge-reranker-v2-m3`, `trafilatura`, `sentence-transformers`,
  `rank_bm25`, `searxng`, `ollama`. No proprietary components.

## License

MIT. See the repo root.
