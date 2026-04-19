# `recipes/`

Opinionated SOTA agent recipes. One implementation per task — the best framework, retrieval stack, LLM routing, and eval for *that* task.

Two indexes:
- [`by-use-case/`](by-use-case/) — organized by outcome (research / analysis / trading / ...)
- [`by-pattern/`](by-pattern/) — organized by technique (deferred past Wave 1)

## Wave 1 recipes — SOTA stacks

| Recipe | Framework | LLM | Key components | Cost per run |
|---|---|---|---|---|
| [research-assistant/](by-use-case/research-assistant/) | LangGraph | Gemini 2.5 Flash | Exa search · `core/rag/` (contextual + hybrid + rerank) | $0.01–$0.03 |
| [youtube-analyzer/](by-use-case/youtube-analyzer/) | Pydantic AI | Gemini 2.5 Flash (1M ctx) | yt-dlp · Groq Whisper Large v3 Turbo fallback | $0.001–$0.02 |
| [trading-copilot/](by-use-case/trading-copilot/) | LangGraph | Flash-Lite + GPT-5 mini (routing) | yfinance · RSS news · Slack/Telegram alerts | $0.005–$0.02 |

Every stack choice is backed by April 2026 benchmarks and pricing — see each recipe's `techniques.md` for citations.

## Recipe levels

Every recipe declares its available levels with a badge in its README:

- **beginner** ✅ — always present. One file, ≤100 lines, `make run` in ≤60s.
- **production** ⬛ — opt-in. Real tests, observability, HermesClaw sandbox, SLO/cost numbers.
- **rust** ⬛ — opt-in. Only where Rust genuinely wins.

## Why SOTA-per-task, not framework comparison

Earlier framings of this repo considered shipping 4 framework implementations per recipe for comparison. We deliberately chose the cookbook model instead:

- **Readers want the answer, not the menu.** "How should I build X" has one best-2026 answer for most tasks.
- **Opinions are the value.** Unopinionated recipes read like bland docs; opinionated recipes ship with their rationale.
- **Framework comparisons still have a home** — they live in [`../comparisons/`](../comparisons/) as landscape pages, decoupled from recipe structure.
- **Evals keep us honest.** Every recipe ships with `eval/` — reproducible scorer against a fixed eval set. If a better stack shows up, the eval tells us, and we swap.

## Adding a recipe

See the root [CONTRIBUTING.md](../CONTRIBUTING.md). TL;DR: open a **recipe-request** issue first with the proposed SOTA stack + rationale, use the existing structure as template, `make run` must work in ≤60s, `make eval` must produce a reproducible score.
