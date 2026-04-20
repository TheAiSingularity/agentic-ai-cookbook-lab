# Public launch checklist

The repo is currently **private**. Before flipping public, walk through
this list. Nothing here is done for you — each step is a deliberate
decision.

---

## Decisions locked before launch day

- [ ] **Final project name** picked. Candidates from the master plan:
      `agentic-research`, `open-research-engine`, `researchlab`,
      `agentic-research-intelligence`, `freelance`, `reflective`.
      Default if nothing chosen: `agentic-research`.
- [ ] **Renaming decision**:
      - [ ] Rename the repo on GitHub (`gh repo rename <new-name>`)
      - [ ] OR keep `agentic-ai-cookbook-lab` and just reposition the README.
- [ ] **First full benchmark run** complete — `engine/benchmarks/RESULTS.md`
      has real numbers from the shipped `simpleqa_mini` + `browsecomp_mini`
      fixtures. Today the file documents the harness; Phase 9 fills in
      the pass-rate, mean wall, verified-ratio.
- [ ] **Security posture scan** — run:
      ```bash
      git log --all -p | grep -iE "sk-(proj|live|ant)|api[_-]?key.*['\"][a-zA-Z0-9_-]{30,}"
      ```
      Should return nothing. Also verify `.gitignore` covers every
      path you've touched recently.
- [ ] **Repo description** on GitHub updated (it currently says
      "SOTA agent recipes…" — accurate but dated; the engine-centric
      version is better).

---

## Go-live sequence (run these in order, ~30 min)

```bash
# 1. Rename (optional — only if you picked a new name)
gh repo rename TheAiSingularity/agentic-ai-cookbook-lab <new-name>

# 2. Flip visibility
gh repo edit TheAiSingularity/<name> \
  --visibility public \
  --accept-visibility-change-consequences

# 3. Update description + homepage + topics
gh repo edit TheAiSingularity/<name> \
  --description "The best \$0 research agent that runs on a laptop. Local Gemma 3 4B + SearXNG + trafilatura. CLI · TUI · Web GUI · MCP · Claude plugin." \
  --add-topic research-agent \
  --add-topic rag \
  --add-topic local-first \
  --add-topic ollama \
  --add-topic claude-plugin \
  --add-topic mcp-server

# 4. Tag the launch commit
git tag -a v0.1.0 -m "v0.1.0 — public launch"
git push --tags

# 5. Verify visibility flipped
gh repo view TheAiSingularity/<name> --json visibility,url
```

---

## Submissions (after public)

### Anthropic Claude plugin marketplace

```bash
# Test the plugin manifest locally first
cd engine/mcp/claude_plugin
# Inside Claude Desktop or Claude Code:
/plugin marketplace add $(pwd)
/plugin install agentic-research
/research What is Anthropic's contextual retrieval?
```

If all three pass, submit:

1. Go to https://platform.claude.com/plugins/submit (or
   https://claude.ai/settings/plugins/submit).
2. Paste the GitHub repo URL (pointing at `engine/mcp/claude_plugin/`).
3. Fill in the plugin metadata (comes from `plugin.json`).
4. Submit. Anthropic reviews; `Anthropic Verified` badge requires
   additional manual review.

### MCP registry

```bash
# Assumes you've published the engine as a Python package to PyPI first,
# OR that the MCP entry uses a command that resolves on users' systems.
pip install mcp-publisher          # or npm i -g @modelcontextprotocol/publisher
mcp-publisher login                # GitHub device flow
mcp-publisher publish              # registers metadata at registry.modelcontextprotocol.io
```

### Community aggregators (third-party, discovery)

- **claudemarketplaces.com** — open an issue / PR on their GitHub with a
  link to our plugin repo.
- **mcp.so** — similar, their indexing script picks up new servers via
  the official registry anyway.
- **claudeskills.info** — manual submission.
- **HermesHub** — if we also publish a Hermes-skill-format variant,
  open a PR on `amanning3390/hermeshub`.

---

## Launch-day comms (drafts in `docs/launch-copy.md`)

- [ ] **Hacker News** `Show HN` post (timing: Tuesday or Wednesday, 8-10am PT).
- [ ] **r/LocalLLaMA** post (rich formatting; screenshots of TUI welcome).
- [ ] **Twitter / X** thread (5-7 tweets with GIFs if possible).
- [ ] **LinkedIn** (optional, longer-form).
- [ ] **Anthropic Discord #community-plugins** announcement.
- [ ] **Personal network blast** (friends who might be early users).

Draft copy in `docs/launch-copy.md`.

---

## Day-after / week-after

- [ ] Respond to every issue within 24 h for the first week.
- [ ] Label good-first-issues aggressively; new contributors find them
      fast when the repo is on HN's front page.
- [ ] Add `v0.2.0` milestone with the first round of feature-request
      triage.
- [ ] Watch `engine/benchmarks/results/` for any live runs users
      contribute — offer to include anonymized numbers in RESULTS.md.
- [ ] If the repo crosses 100 stars: consider the hosted SaaS path
      (deferred in the original plan); if it crosses 1 k stars,
      schedule a v0.2 roadmap post.

---

## Not doing at launch (by design)

- **Hosted SaaS**. Everything stays local-first in v0.1.
- **LoRA fine-tuning**. Deferred per the master plan.
- **Team / multi-user workspaces**. Scope for v0.2+.
- **Mobile app**. Out of scope.
- **Desktop app packaging** (Tauri / Electron). The web GUI at
  `localhost:8080` covers the "GUI that isn't a terminal" need; native
  packaging lands later when there's clear demand.

---

## Status as of now (pre-launch)

- **Visibility**: PRIVATE
- **Master plan**: 8 of 9 phases complete; Phase 9 = this file + launch
  copy + bench run + the public flip (when you're ready).
- **Tests**: 228+ green, all mocked, across engine + core/rag + archived
  recipes + trading-copilot.
- **Last commit**: see `git log --oneline -5`.
