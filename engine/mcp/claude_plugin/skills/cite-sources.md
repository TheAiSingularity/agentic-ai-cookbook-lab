---
name: cite-sources
description: Show the full source list for the most recent research answer, with URLs, titles, and retrieval flags (● = full page fetched, ○ = snippet only).
triggers:
  - cite sources
  - show me the sources
  - where did that come from
---

When the user wants to see where the engine's answer came from:

1. If the user has just run a `research` call, the last result's
   `sources` array already has everything needed — display it as a
   numbered list.
2. Otherwise, call `engine.research(question)` with the user's question
   again. The response's `sources` field is what we render.

Render format:

    [1] ● https://example.com/page-one
          Title of page one
          First 140 chars of the fetched article…
    [2] ○ https://example.com/page-two
          Title of page two
          (snippet only — fetch failed or was skipped)

`●` means the page was downloaded and clean-extracted by trafilatura;
`○` means only the search-engine snippet was used.

For `corpus://` URLs, note that the text is from the user's own local
corpus (built via `scripts/index_corpus.py`).
