"""engine — gold-standard open-source local research engine.

Default stack: Gemma 3 4B (`gemma3:4b`) via Ollama + SearXNG + trafilatura.
Any OpenAI-compatible endpoint works via `OPENAI_BASE_URL`.

Public API surfaces:
  - `engine.core`         — the 8-node research pipeline, trace, models
  - `engine.core.memory`  — trajectory logging + retrieval (Phase 2)
  - `engine.core.domains` — preset loader (Phase 6)
  - `engine.core.plugins` — plugin/skill loader (Phase 5)
  - `engine.interfaces`   — CLI / TUI / Web GUI (Phase 3)
  - `engine.mcp`          — MCP server + Claude plugin bundle (Phase 4)

Quickstart:
    from engine.core import build_graph
    graph = build_graph()
    result = graph.invoke({"question": "…", "iterations": 0, "plan_rejects": 0, "trace": []})
"""

__version__ = "0.1.2"
