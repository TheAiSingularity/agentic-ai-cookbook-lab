---
name: research
description: Run a full local research query with source citations, CoVe-verified claims, and per-node trace. Uses the `engine` MCP server's `research` tool under the hood.
triggers:
  - research this
  - look this up
  - find sources for
  - verify
---

Use this skill when the user asks for cited, verifiable research on any
topic that benefits from web search + retrieval + verification. Default
domain is `general`; switch domains via the `set-domain` skill when the
user specifies medical, academic-paper, financial, or stock research.

Procedure:

1. Call `engine.research(question, domain=<inferred>, memory="session")`.
2. Surface the `answer` to the user verbatim — it already carries `[N]`
   citations.
3. If `verified_claims` is non-empty, emphasize the verified summary
   (e.g. "3/5 claims verified · 2 unverified"). If any unverified
   claims exist, warn the user and list them.
4. When user asks "where did that come from?", call the `cite-sources`
   skill with the most-recent question.
5. For deeper verification of a specific claim, call the `verify-claim`
   skill.

Do not editorialize beyond what the sources support. Never invent
information the pipeline did not surface. If the engine refused the
query ("The provided evidence does not answer this question."), relay
that refusal honestly.
