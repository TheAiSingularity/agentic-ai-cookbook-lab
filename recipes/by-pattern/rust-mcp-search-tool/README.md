# `rust-mcp-search-tool`

A minimal MCP (Model Context Protocol) server in Rust that exposes a
`search` tool backed by a self-hosted [SearXNG](../../../scripts/searxng/)
instance. This recipe is a **case study** — it demonstrates where Rust
genuinely earns its place in the 2026 agent stack, and where it doesn't.

## What it is

- **One tool:** `search({query, num_results?})` → `[{url, title, snippet}, …]`
- **One transport:** stdio (the MCP default — easiest for agent clients to consume)
- **One runtime dep:** `rmcp` + `reqwest` — rustls-only, no OpenSSL
- **Size target:** small, static, fast cold start

## Why Rust here?

From the 2026 benchmark data referenced in the paper:

| Metric | Python equivalent | **Rust (this recipe)** |
|---|---|---|
| Cold start | 60–140 ms | **~4 ms** |
| Binary size | ~40 MB venv + 30–100 MB deps | **~5 MB static binary** |
| Peak memory | >1 GB (CPython + libs) | **<20 MB** |
| Throughput (ideal tools) | baseline | **+36% at same concurrency** |

These numbers matter **only in specific deployments**:

- Running a tool MCP server on an edge device or sandboxed node where
  the full Python stack is too heavy.
- Giving a remote / untrusted agent access to search *without* giving
  it a Python interpreter.
- Scaling to many concurrent agent tool calls on a cost-sensitive host.

They **do not** matter when:

- The tool call is network-bound (SearXNG itself responds in 100–500 ms —
  the Rust vs Python overhead is in the noise at end-to-end latency).
- The agent's inference is on a separate GPU box (vLLM/SGLang in
  Rust/CUDA already — the host process is free either way).

The honest read: **the Rust MCP tool shines in deployment, not in raw
pipeline speed.** The main `research-assistant/` recipe stays Python;
this recipe exists to document where the tradeoff does flip.

## Build

```bash
cd recipes/by-pattern/rust-mcp-search-tool
cargo build --release
ls -lh target/release/mcp-search   # ≈ 5 MB on aarch64-darwin / x86_64-linux
```

## Run as an MCP server over stdio

```bash
# In one terminal (SearXNG must be up)
cd scripts/searxng && docker compose up -d

# Point the binary at your SearXNG (defaults to http://localhost:8888)
export SEARXNG_URL=http://localhost:8888

# The binary speaks MCP over stdio — it's meant to be launched by an MCP client
./target/release/mcp-search
```

Hooking it into an MCP-capable agent (e.g., Claude Desktop, Cursor, or a
custom Python agent using the `mcp` SDK) is a one-line config entry:

```jsonc
{
  "mcpServers": {
    "search": {
      "command": "/absolute/path/to/target/release/mcp-search",
      "env": { "SEARXNG_URL": "http://localhost:8888" }
    }
  }
}
```

## Test with the MCP Inspector (no agent required)

```bash
npx -y @modelcontextprotocol/inspector ./target/release/mcp-search
```

Opens a local web UI where you can invoke `search` interactively.

## Benchmarks — replicate on your hardware

```bash
# Cold start
time ./target/release/mcp-search --help 2>/dev/null   # expect < 10 ms

# Binary size
size target/release/mcp-search

# Memory under load (macOS)
/usr/bin/time -l ./target/release/mcp-search < /dev/null
```

Compare these against a minimal Python equivalent (FastMCP + `httpx`)
and file the numbers in `BENCHMARKS.md` (todo on first real run).

## Files

```
rust-mcp-search-tool/
├── Cargo.toml          # deps + release-profile optimized for size
├── src/main.rs         # one tool, stdio transport, ~130 LOC
├── README.md           # you're reading it
└── BENCHMARKS.md       # populate with measured numbers after first build
```

## See also

- [`../../../scripts/searxng/`](../../../scripts/searxng/) — the SearXNG instance this tool fronts.
- [`../../by-use-case/research-assistant/`](../../by-use-case/research-assistant/) — the Python pipeline that uses SearXNG directly (no MCP middleman needed in-process).
- [Model Context Protocol spec](https://modelcontextprotocol.io/).
- [`rmcp` Rust MCP SDK](https://github.com/modelcontextprotocol/rust-sdk).
