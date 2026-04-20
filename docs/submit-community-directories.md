# Community directory submissions

How to get listed on the four community-run directories that track
Claude plugins, Claude skills, and MCP servers. Unlike the Anthropic
marketplace + official MCP registry (see
[`submit-claude-plugin.md`](submit-claude-plugin.md) and
[`submit-mcp-registry.md`](submit-mcp-registry.md)), these are
third-party, community-operated, and use different discovery patterns.

<!-- toc -->

- [claudemarketplaces.com — auto-discovery](#claudemarketplacescom--auto-discovery)
- [glama.ai/mcp/servers — auto-pulled from MCP registry](#glamaaimcpservers--auto-pulled-from-mcp-registry)
- [claudeskills.info — manual skill submission](#claudeskillsinfo--manual-skill-submission)
- [pulsemcp.com / mcp.so — auto-discovery](#pulsemcpcom--mcpso--auto-discovery)
- [What's live where](#whats-live-where)

<!-- /toc -->

---

## claudemarketplaces.com — auto-discovery

**What it is:** Community aggregator (run by [@mertbuilds](https://github.com/mertbuilds); not Anthropic-operated) that crawls GitHub repos for `.claude-plugin/marketplace.json` files once per day.

**Our submission path:** ZERO explicit submission. The crawler looks for a file at the exact path `.claude-plugin/marketplace.json` in public repos. Ours lives at [`.claude-plugin/marketplace.json`](../.claude-plugin/marketplace.json) — the crawler should pick it up within ~24 hours of the repo going public.

**If we don't appear after 48 hours:**

1. Check their GitHub repo ([`mertbuilds/claudemarketplaces.com`](https://github.com/mertbuilds/claudemarketplaces.com)) for a submission-issue template.
2. Open an issue with this body:

```markdown
Title: Submit: TheAiSingularity/agentic-research

Hi — submitting a marketplace for indexing. The marketplace.json is at
the standard `.claude-plugin/marketplace.json` path; the crawler
appears to have missed it.

  repo:     https://github.com/TheAiSingularity/agentic-research-engine-oss
  manifest: https://raw.githubusercontent.com/TheAiSingularity/agentic-research-engine-oss/main/.claude-plugin/marketplace.json
  plugin:   agentic-research
  category: research · rag · local-llm · verification

Plugin description: Local-first research agent that verifies its own
answers. Runs on Gemma 3 4B + Ollama, $0/query. 8-node LangGraph
pipeline with CoVe verification, cross-encoder rerank, contextual
chunking, 6 domain presets.

Thanks!
```

**Why this route is worth it:** ~5 min of effort. Low-traffic surface but pure SEO. Someone Googling "claude plugin research" may click through.

---

## glama.ai/mcp/servers — auto-pulled from MCP registry

**What it is:** Editorially curated MCP server directory (21 k+ listings as of April 2026). Auto-pulls from `registry.modelcontextprotocol.io`.

**Our submission path:** ZERO. Our server is already registered at `io.github.TheAiSingularity/agentic-research` on the official MCP registry (published 2026-04-21 via `mcp-publisher publish`). Glama continuously re-indexes the registry; expect the listing within 24 hours.

**Where to check:**

- <https://glama.ai/mcp/servers?q=agentic-research>
- <https://glama.ai/mcp/servers/io.github.TheAiSingularity/agentic-research> (if they use our namespace as the slug)

**If not indexed after 48 hours:**

1. Check [glama.ai/mcp/servers/submit](https://glama.ai/mcp/servers/submit) for a manual re-index form.
2. If there's a GitHub tracker for their directory, open an issue referencing our registry entry and asking them to re-poll.

**Why this route matters:** Glama's listings include quality scores + tool counts. Strong placement here puts us in front of the agent-builder community.

---

## claudeskills.info — SKIPPED

**Status (2026-04-21):** skipped. The submission form rejected our repo
with a generic "Failed to create submission" error — most likely because
their validator expects a **dedicated single-skill repo** (placeholder
URL convention `username/skill-name` singular) rather than a large repo
with skills nested inside. Fixing would require either (a) spinning up
a second standalone repo mirroring just the skill, or (b) waiting for
their submission endpoint to surface a clearer error.

Neither was worth the return given that the same standalone skill is
already discoverable via:

- GitHub search on our main repo (`skills/verify-answer/SKILL.md`)
- The organic traffic from the full-engine listings on mcp.so, glama.ai,
  and the official MCP registry

If a future contributor wants to re-attempt, Plan B is:

1. `gh repo create TheAiSingularity/verify-answer-skill --public`
2. Copy `skills/verify-answer/SKILL.md` + `LICENSE` to the new repo's root
3. Submit that repo URL at <https://claudeskills.info/submit>

## claudeskills.info (legacy submission notes)

**What it is:** Community directory for Claude skills (single-purpose `SKILL.md` files, not full plugins). ~650 skills indexed.

**Our submission path:** Manual, one skill per submission. **Only the standalone prompt-only skills qualify** — the four skills bundled with our MCP server require our engine to run and don't fit this site's model.

### What we submit

**Skill 1: `verify-answer`**

- **GitHub URL:** https://github.com/TheAiSingularity/agentic-research-engine-oss/blob/main/skills/verify-answer/SKILL.md
- **Raw URL:** https://raw.githubusercontent.com/TheAiSingularity/agentic-research-engine-oss/main/skills/verify-answer/SKILL.md
- **Name:** `verify-answer`
- **One-line description:** *Stress-test any answer against its evidence using Chain-of-Verification. Returns per-claim verdicts: VERIFIED / UNVERIFIED / CONTRADICTED.*
- **Tags:** `verification`, `research`, `fact-check`, `hallucination`, `cove`, `prompt-only`
- **Triggers:** `verify this`, `fact check`, `is this accurate`, `check the claims`
- **License:** MIT
- **Author:** TheAiSingularity

### How to submit

1. Visit [claudeskills.info/submit](https://claudeskills.info/submit) (or look for a "Submit" button on the homepage).
2. Paste the URLs + metadata above.
3. Provide a 1-3 paragraph description drawing from the first half of [`skills/verify-answer/SKILL.md`](../skills/verify-answer/SKILL.md).
4. If the form asks for an example invocation, paste the **example** section from the skill.

**If submission is via GitHub PR instead of a form:**

Open a PR on their directory repo adding an entry:

```markdown
---
name: verify-answer
repo: https://github.com/TheAiSingularity/agentic-research-engine-oss
path: skills/verify-answer/SKILL.md
description: Stress-test any answer against its evidence using Chain-of-Verification. Per-claim verdicts: VERIFIED / UNVERIFIED / CONTRADICTED.
tags: [verification, research, fact-check, hallucination, cove, prompt-only]
license: MIT
author: TheAiSingularity
---
```

### Why ship a standalone skill at all

Four reasons:

1. **Discovery surface.** Users browsing claudeskills.info never encounter our MCP server if we only ship tool-wired skills. The standalone skill gets them in the door.
2. **Backlink economics.** The skill's README mentions the full engine twice. Users who like the skill click through to `pip install agentic-research-engine`.
3. **Methodology showcase.** CoVe is genuinely useful even without our pipeline. Making it one-command available is a real contribution back to the Claude ecosystem.
4. **Low cost to us.** Prompt-only skill = no maintenance burden beyond keeping the Markdown up to date.

---

## pulsemcp.com / mcp.so — auto-discovery

Both pull from the official MCP registry on a schedule. Same situation as Glama: our entry is already there, no action needed. Expected indexing: within 24-48 hours of `mcp-publisher publish`.

---

## What's live where

| surface | status | action needed |
|---|---|---|
| [GitHub repo](https://github.com/TheAiSingularity/agentic-research-engine-oss) | ✅ public, v0.1.2 | — |
| [PyPI package](https://pypi.org/project/agentic-research-engine/) | ✅ live | — |
| [MCP registry](https://registry.modelcontextprotocol.io/v0/servers?search=agentic-research) | ✅ live | — |
| Anthropic plugin marketplace | ✅ distributable via `/plugin marketplace add <repo-url>` | — |
| claudemarketplaces.com | ⏳ awaiting crawler pickup | wait 24h; escalate via GitHub issue if needed |
| glama.ai/mcp/servers | ⏳ awaiting registry sync | wait 24h |
| mcp.so | ⏳ awaiting registry sync | wait 24h |
| pulsemcp.com | ⏳ awaiting registry sync | wait 24h |
| claudeskills.info | ⏭️ skipped (their validator rejected our multi-purpose repo; Plan B notes in section above) | — |
