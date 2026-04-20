"""engine.mcp.server — Python MCP server exposing the research pipeline.

One tool: `research({question, domain?, memory?}) → structured JSON`.

Stdio transport (the MCP default). Clients configure via their
`mcp.json` / `claude_desktop_config.json`:

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

The tool returns a structured payload compatible with MCP's JSON output
mode so downstream agents can reason about the verified/unverified
claim lists without parsing free-form prose.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover — handled at runtime
    print(
        "[engine.mcp] mcp SDK is not installed. "
        "Run `pip install -r engine/requirements.txt` or `pip install mcp`.",
        file=sys.stderr,
    )
    raise SystemExit(1)

from engine.core.memory import MemoryStore, VALID_MODES  # noqa: E402
from engine.interfaces.common import (  # noqa: E402
    format_sources,
    format_trace_per_node,
    format_verified_summary,
    run_query,
)


mcp = FastMCP("engine-research")


@mcp.tool()
def research(question: str, domain: str = "general", memory: str = "session") -> dict:
    """Run a full research query through the local engine.

    Args:
        question: The research question.
        domain:   Preset name: general / medical / papers / financial / stock_trading / personal_docs.
        memory:   off | session | persistent.

    Returns:
        {
            "question":          original query
            "domain":            resolved domain
            "question_class":    router output (factoid / multihop / synthesis)
            "answer":            synthesizer output with inline [N] citations
            "verified_summary":  e.g. "3/5 claims verified · 2 unverified"
            "verified_claims":   [{text, verified}]
            "unverified_claims": [str]
            "sources":           [{idx, url, title, preview, fetched}]
            "trace":             [{node, calls, latency_s, tokens_est}]
            "totals":            {wall_s, tokens_est, iterations}
            "memory_hits":       [{question, answer, score, domain, query_id, timestamp}]
        }
    """
    if memory not in VALID_MODES:
        memory = "session"

    store = MemoryStore.open(memory)
    try:
        result = run_query(question, domain=domain, memory=store)
    finally:
        store.close()

    return {
        "question": result.question,
        "domain": result.domain,
        "question_class": result.question_class,
        "answer": result.answer,
        "verified_summary": format_verified_summary(result),
        "verified_claims": result.verified_claims,
        "unverified_claims": result.unverified_claims,
        "sources": format_sources(result),
        "trace": format_trace_per_node(result),
        "totals": {
            "wall_s": result.total_latency_s,
            "tokens_est": result.total_tokens_est,
            "iterations": result.iterations,
        },
        "memory_hits": result.memory_hits,
    }


@mcp.tool()
def reset_memory() -> dict:
    """Wipe the persistent memory store. Returns the count of trajectories wiped."""
    store = MemoryStore.open("persistent")
    n = store.count()
    store.reset()
    store.close()
    return {"reset": n}


@mcp.tool()
def memory_count() -> dict:
    """Return the count of trajectories in the persistent memory store."""
    store = MemoryStore.open("persistent")
    n = store.count()
    store.close()
    return {"count": n}


def main() -> None:
    """Run the MCP server over stdio (the default transport)."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
