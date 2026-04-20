"""engine.interfaces — three ways to drive the engine.

- `cli`  — rich stdout CLI (the default; wraps the pipeline with flags)
- `tui`  — Textual TUI (interactive: panes for trace, sources, hallucination flags)
- `web`  — FastAPI + HTMX GUI on localhost:8080

All three call into `engine.core.build_graph` and share the same memory /
compaction / trace infrastructure. The engine never pushes to any cloud
telemetry; everything users see is also everything that exists.
"""
