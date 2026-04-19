# trading-copilot

**Levels:** beginner ⬛ · production ⬛ · rust ⬛ · **all pending — ships in Wave 1**

> **This is a research and alerting tool — NOT an auto-execution bot.** No orders are placed. No broker integration. Educational use only.

## What it does
Given a watchlist and a research question, fetches market data, pulls news, applies multi-role LLM reasoning (analyst + skeptic pattern), and emits structured alerts to Slack or Telegram when pre-defined conditions are met.

## Who it's for
- Retail investors / quants building research tooling
- Anyone learning how to do streaming-data agents, stateful monitoring, and routed alerts
- Engineers who want a canonical example of LLM routing — cheap default model for summarization, smarter model only for the critique step

## Why you'd use it
- **Public data sources only** — yfinance + RSS. No broker API, no real money, no regulatory headaches.
- **Cost-aware by design** — model routing sends routine analysis to Gemini Flash-Lite and escalates only the skeptic-critique step to GPT-5 mini
- **Alerts, not execution** — the human stays in the loop
- **Per-cycle cost: $0.005–$0.02**

## SOTA stack (April 2026)

| Component | Choice | Rationale |
|---|---|---|
| **Orchestration** | LangGraph | Stateful graph fits long-running monitoring with human-in-loop alerts. Lowest token overhead per 2026 framework benchmarks. |
| **LLM (analyst step)** | Gemini 2.5 Flash | Cheap summarization of data fetches ($0.25/$1.50 per M tokens) |
| **LLM (skeptic/critique)** | GPT-5 mini | Highest agentic-task accuracy (OSWorld-Verified 72.2) — escalation pays for itself in reduced false positives |
| **Price data** | `yfinance` | Free, reliable |
| **News** | RSS feeds (primary) + optional Polygon/NewsAPI | RSS covers most public-market signals at zero cost |
| **Alert routing** | Slack / Telegram webhooks | Structured payload schema, one handler per channel |

Pattern: data-gather → analyst → **skeptic critique** → alert router. Multi-role catches reasoning errors at low marginal cost (critique runs only on triggered signals).

See [`beginner/techniques.md`](beginner/techniques.md) for primary-source citations. *(Lands Wave 1.)*

## Eval

Historical backtest on a fixed rule set (e.g., "alert when 50/200 MA cross + negative news sentiment"). Scorer measures:
- **Signal precision** — of alerts issued, how many matched a meaningful market move?
- **Signal recall** — of meaningful moves in the backtest window, how many did the agent flag?

`make eval` reproduces the score on a held-out window.

## Expected cost per polling cycle
$0.005–$0.02 (one Flash-Lite analysis + one GPT-5 mini critique on triggered signals only).

## Disclaimer

This recipe ships as an **educational example of agent patterns applied to market research**. It is not financial advice, it is not an execution engine, and it does not guarantee profits or losses. Use at your own risk.

## See also
- [`../../../foundations/what-is-hermes-agent.md`](../../../foundations/what-is-hermes-agent.md)
