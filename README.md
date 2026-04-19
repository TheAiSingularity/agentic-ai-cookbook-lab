<p align="center">
  <!-- banner.png goes here in Wave 1 -->
  <strong>agentic-ai-cookbook-lab</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/status-wave%200-orange.svg" alt="Status">
  <img src="https://img.shields.io/badge/recipes-3-green.svg" alt="Recipes">
  <img src="https://img.shields.io/badge/languages-python%20%2B%20rust-green.svg" alt="Languages">
</p>

**SOTA cookbooks for agent tasks — one opinionated, cheap-yet-accurate implementation per recipe.**

This is the OpenAI Cookbook / Anthropic Cookbook model, focused on agent tasks. Each recipe ships one state-of-the-art implementation — researched, benchmarked, and priced — not a menu of alternatives. The goal: the authoritative answer to "how do I build X in 2026."

Every recipe works end-to-end in under a minute from a fresh clone. Every choice (framework, retrieval stack, LLM, data source) is justified in that recipe's `techniques.md` with primary-source citations. Every recipe has an `eval/` harness so the accuracy claim is reproducible.

The default sandboxed runtime for production-tier recipes is [HermesClaw](https://github.com/TheAiSingularity/hermesclaw) — Hermes Agent inside NVIDIA OpenShell, kernel-enforced.

---

## The three flagship recipes (Wave 1)

| Recipe | What it does | SOTA stack | Cost per run |
|---|---|---|---|
| [research-assistant](recipes/by-use-case/research-assistant/) | Answers research questions with web search + RAG + synthesis, fully cited | **OpenAI-only**: LangGraph · `web_search` tool · `core/rag/` · `gpt-5-nano` + `gpt-5-mini` | ~$0.05–$0.25 |
| [youtube-analyzer](recipes/by-use-case/youtube-analyzer/) | Transcript → chapters → summary → titles, with typed schemas | Pydantic AI · yt-dlp (+ Groq Whisper fallback) · Gemini 2.5 Flash (1M context) | $0.001–$0.02 |
| [trading-copilot](recipes/by-use-case/trading-copilot/) | Market research + alerts (NOT auto-execution) | LangGraph · yfinance + RSS · Flash-Lite → GPT-5 mini routing for skeptic critique | $0.005–$0.02 |

## Repo layout

```
recipes/       # SOTA recipes, organized by use-case and by pattern
core/          # Shared primitives: rag · memory · tools · sandbox
foundations/   # Plain-English explainers: OpenClaw · OpenShell · NemoClaw · Hermes Agent
comparisons/   # Landscape pages (framework/protocol/sandbox landscape)
skills/        # Reusable agent skills
```

## Levels

Every recipe declares its available levels:

- **`beginner/`** — ≤100 lines, heavily commented, `make run` in ≤60 seconds. **Every recipe has this.**
- **`production/`** — real tests, observability, HermesClaw compose, SLO/cost numbers. **Flagship recipes only.**
- **`rust/`** — for categories where Rust genuinely wins (MCP servers, inference runtimes, tool binaries).

## Running a recipe

```bash
git clone https://github.com/TheAiSingularity/agentic-ai-cookbook-lab
cd agentic-ai-cookbook-lab/recipes/by-use-case/research-assistant/beginner
make run
```

## Status

This is **Wave 0** — skeleton complete. Wave 1 (3 SOTA recipes + 4 foundations pages + `core/rag/` v0) ships next.

Every recipe ships with:
- `techniques.md` — SOTA techniques used, with primary-source citations
- `eval/` — fixed eval set + reproducible scorer

---

## Related

- [HermesClaw](https://github.com/TheAiSingularity/hermesclaw) — the secure runtime these recipes run inside
- [NVIDIA/OpenShell](https://github.com/NVIDIA/OpenShell) — the sandbox underneath HermesClaw
- [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) — the agent

MIT licensed.
