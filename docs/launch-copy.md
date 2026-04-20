# Launch-day copy (drafts)

Paste these into the appropriate surfaces when Phase 9 flips public.
Numbers + screenshots slot in at the `[INSERT-…]` placeholders.

---

## Show HN — headline + body

**Title**: `Show HN: agentic-research — a local research agent that verifies its own answers`

> After a few months building this: an open-source research agent that
> runs fully on your laptop (Mac M-series with Ollama + Gemma 3 4B),
> cites every source, verifies every factual claim against the sources
> it cited, and shows you the full trace of what happened.
>
> The design wedge: nobody ships "offline-first local agent + great UX
> + verified reasoning" in one package. Perplexica (local UX but weak
> reasoning), Khoj (personal knowledge but not research-optimized),
> MiroThinker-H1 (strong reasoning but hosted), OpenResearcher-30B
> (strong but no UX) — each does one of the three well.
>
> What's shipped:
> • 8-node LangGraph pipeline: classify → plan → search → retrieve →
>   fetch_url → compress → synthesize → verify. All stages env-toggleable.
> • Retrieval: BM25 + dense + RRF + optional cross-encoder rerank
>   (BAAI/bge-reranker-v2-m3), Anthropic-style contextual chunking.
> • Three parallel interfaces — CLI, Textual TUI, FastAPI web GUI.
> • MCP server + Claude plugin bundle — one `research` tool with
>   structured JSON output including verified/unverified claim lists.
> • Local memory — SQLite trajectory log at `~/.agentic-research/memory.db`
>   with semantic retrieval across prior queries. Opt-in,
>   wipe-able, no cloud.
> • Six domain presets (general/medical/papers/financial/stock_trading/
>   personal_docs) plus a plugin loader for third-party Claude plugins
>   and Hermes/agentskills.io skills.
> • 228+ mocked tests, all running with zero network + zero API key.
>
> Honest limits: Gemma 4B is 15-25% below 30B+ open models on complex
> multi-hop reasoning. We don't claim to beat GPT-5.4 Pro. We claim to
> be the best $0 local research agent. Numbers in
> `engine/benchmarks/RESULTS.md`.
>
> Quickstart (Mac):
>
>     brew install ollama
>     ollama pull gemma3:4b nomic-embed-text
>     (cd scripts/searxng && docker compose up -d)
>     git clone https://github.com/TheAiSingularity/<repo>
>     cd <repo>/engine && make install && make smoke
>
> Then `make tui` for the Textual terminal UI or `make gui` for
> `localhost:8080`.
>
> Feedback wanted on: (1) the three-interface decision — does anyone
> actually use all three, or should we pick one to polish? (2) the
> memory model — is SQLite-on-disk the right primitive, or do people
> want encrypted export? (3) which domain preset is missing that would
> be obviously useful.
>
> [INSERT-GITHUB-LINK]

---

## r/LocalLLaMA — markdown

```markdown
# I built a local research agent that verifies its own answers

After a few months of tinkering, sharing an open-source project I've
been working on: a research agent that runs fully on-device via Ollama
+ Gemma 3 4B, cites every source, and runs a Chain-of-Verification step
against the sources it cited.

**Why this vs Perplexica / Khoj / gpt-researcher**: those are great but
each covers one axis. Perplexica has the UX but thin reasoning. Khoj is
personal-knowledge-focused, not research-optimized. gpt-researcher's
architecture is older. None of them have all three of {local-first,
good UX, verified reasoning} in the same package.

**The wedge**: 8-node LangGraph pipeline with explicit stages for
classify → plan → search → retrieve → fetch_url → compress → synthesize
→ verify. Every stage is env-toggleable for leave-one-out ablation.

**What runs locally:**
- Ollama + Gemma 3 4B (the default; 3.3 GB)
- SearXNG (self-hosted meta-search, Docker)
- trafilatura for full-page fetch
- `BAAI/bge-reranker-v2-m3` for optional cross-encoder rerank
- nomic-embed-text for semantic retrieval

**What you see:**
- Three interfaces — CLI, Textual TUI, FastAPI web GUI — all pulling
  from the same engine core.
- Full trace: every LLM call, per-node latency, token estimates.
- Hallucination check pane: every CoVe claim labeled verified / unverified.
- Memory hits: if you've asked something similar before, the prior
  answer shows up as context (optional).

**Honest about limits:**
Gemma 4B is noticeably weaker than 30B+ open models on hard multi-hop
questions. I'm not claiming to beat MiroThinker-H1 (88.2 BrowseComp)
or GPT-5.4 Pro. I'm claiming to be the best $0 local research agent —
which is a distinct and unserved niche.

**Quickstart:**
```bash
brew install ollama
ollama pull gemma3:4b nomic-embed-text
cd scripts/searxng && docker compose up -d
cd ../../engine && make install && make smoke
```

Then `make tui` for the terminal UI.

**MCP + Claude plugin**: there's a submittable Claude plugin bundle in
`engine/mcp/claude_plugin/` with four skills (`/research`,
`/cite-sources`, `/verify-claim`, `/set-domain`). Once approved on the
marketplace, `/plugin install agentic-research` in Claude Desktop gets
you the full engine as a tool.

**Bring your own docs:**
```bash
python scripts/index_corpus.py build ~/papers --out ~/papers.idx
export LOCAL_CORPUS_PATH=~/papers.idx
engine ask "what do my papers say about X?" --domain personal_docs
```

GitHub: [INSERT-GITHUB-LINK]

Any feedback very welcome, especially from people running 4B-class
models for real work.
```

---

## Twitter / X — 7-tweet thread

**1/** Just shipped `agentic-research` — an open-source research agent
that runs fully on your laptop and verifies its own answers.

No cloud. No API key required. No telemetry. $0 per query when you're
running on local Ollama + Gemma 3 4B.

[SCREENSHOT-OF-TUI]

**2/** The design wedge: nobody shipped "local-first + great UX +
verified reasoning" in one package.

- Perplexica: UX good, reasoning thin
- Khoj: personal-knowledge, not research
- MiroThinker-H1: strong reasoning, hosted-only
- OpenResearcher-30B: strong, no UX

**3/** What's inside: 8-node LangGraph pipeline —

```
classify → plan → search → retrieve → fetch_url →
compress → synthesize → verify
```

Every node is env-toggleable. Every LLM call is traced. Every factual
claim in the answer is CoVe-verified against its sources.

**4/** Three interfaces in parallel —

• CLI (the default; rich stdout)
• Textual TUI (keyboard-driven, works over SSH)
• FastAPI + HTMX Web GUI (localhost:8080)

All three pull from the same engine core. Pick whatever fits.

[SCREENSHOTS — CLI, TUI, Web]

**5/** Memory is opt-in and local: SQLite at
`~/.agentic-research/memory.db` with semantic retrieval. Prior queries
show up as context for related new ones. Wipe anytime with
`engine reset-memory`.

**6/** MCP server + Claude plugin bundle both ship. Submitting to
the marketplace now. Once approved:

```
/plugin install agentic-research
/research What is Anthropic's contextual retrieval?
```

One structured-JSON tool. Four skills that wrap it.

**7/** Honest about what it is: the best $0 local research agent.
NOT a GPT-5.4 Pro killer. Gemma 4B is 15-25% below 30B+ on hard
multi-hop; numbers in RESULTS.md.

Star if you try it. PRs welcome.

[INSERT-GITHUB-LINK]

---

## Anthropic Discord #community-plugins

> Hi everyone — sharing a Claude plugin I just submitted: **agentic-research**.
>
> It's a local-first research agent packaged as a Claude plugin. The
> MCP server runs on the user's machine (Python + mcp SDK) and exposes
> a single `research(question, domain, memory)` tool that returns a
> structured payload — answer + verified_claims + unverified_claims +
> sources + trace.
>
> Defaults are Ollama + Gemma 3 4B + SearXNG (all self-hostable, $0).
> Users who want cloud route to any OpenAI-compatible endpoint via
> `--api-key`.
>
> Four bundled skills: `/research`, `/cite-sources`, `/verify-claim`,
> `/set-domain` (medical / papers / financial / stock_trading /
> personal_docs presets).
>
> Code + plugin bundle: [INSERT-GITHUB-LINK]
> Plugin manifest: `engine/mcp/claude_plugin/.claude-plugin/plugin.json`
>
> Feedback welcome before I submit to the official marketplace.

---

## Personal-network email / DM

> Hey — quick heads up in case it's useful to you.
>
> I've been building a local-first research agent — runs fully on a Mac
> with Gemma 3 4B via Ollama, no cloud, no API keys. It plugs into
> Claude Desktop via MCP so you can `/research <question>` and get a
> cited answer with every claim CoVe-verified against the sources.
>
> The specific thing I think you'd care about: [CUSTOMIZE — e.g.
> "medical domain preset with PubMed bias" / "bring-your-own-PDFs
> flow" / "Textual TUI that works over SSH"]
>
> Still private for now, flipping public [DATE]. Happy to share early
> access if you want to kick the tires.

---

## Notes

- Keep the HN post under 250 words. The link + the "what's the gap
  we're filling" paragraph is what gets clicks.
- r/LocalLLaMA rewards specificity — exact model names, size in GB,
  wall-clock seconds. Round numbers look made up.
- Twitter thread should have one media attachment (GIF or screenshot)
  per 2-3 tweets; text-only threads under-perform.
- Don't announce on multiple surfaces in the same hour. Spread over
  48-72 h.
