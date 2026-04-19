# Contributing

Thanks for your interest. This repo is a **SOTA cookbook for agent tasks** — opinionated, benchmarked, cheap-yet-accurate recipes. Contributions should follow that format so the collection stays coherent.

## The rules that don't bend

1. **Every beginner recipe runs in ≤90s from a fresh clone via `make run`.** Target ≤60s for simple recipes; research-oriented recipes that fan out web searches get a relaxed ≤90s budget (parallelized). Longer than that = doesn't ship.
2. **Every beginner recipe is ≤100 lines and heavily commented** so a newcomer can read the whole thing in one sitting.
3. **Every recipe README answers three questions, in this order:** What does it do? Who is it for? Why would you use it?
4. **Every recipe is opinionated.** Pick the SOTA stack for the task, say why, don't hedge. We pick one framework, one retrieval stack, one LLM routing strategy — the *best* one, backed by current benchmarks.
5. **Every recipe ships with `techniques.md`** that cites the SOTA techniques used, with primary-source links.
6. **Every recipe ships with `eval/`** — fixed eval set + reproducible scorer. Accuracy claims must be verifiable.
7. **No AI-generated fluff** in foundations or comparison pages. Primary sources, cited links, real opinions.

## Adding a new recipe

Use the existing `research-assistant/` as your template. Every recipe directory needs:

```
recipes/by-use-case/<recipe-name>/
├── README.md                 # What / Who / Why, SOTA stack + rationale, cost-per-run estimate
├── recipe.yaml               # Metadata (used by the gallery regenerator)
├── beginner/
│   ├── main.py               # ≤100 lines, heavy comments
│   ├── requirements.txt
│   ├── Makefile              # Must have a `run` target
│   └── techniques.md         # SOTA techniques used, with citations
├── eval/
│   ├── dataset.jsonl         # Gold answers or reference outputs
│   ├── scorer.py             # Reproducible scorer
│   └── Makefile              # `make eval` target
└── production/               # Optional — flagship recipes only
    ├── app/ · tests/ · observability/
    ├── compose.yml           # HermesClaw sandbox preset wired in
    └── runbook.md
```

Open an issue with the **recipe-request** template first so the scope and chosen SOTA stack are agreed before you write code.

## Adding a production tier

Only graduate a recipe to the `production/` tier when:
- The beginner version is stable and used as a reference
- The eval scores are reproducible
- You add at least three of: `pytest` tests, observability (Logfire / OpenTelemetry), HermesClaw compose file, `.env.example` with real-life secrets, a runbook, SLO/cost/latency numbers in the README
- The production tier imports from `core/` rather than duplicating shared code

## Adding a foundation or comparison page

Both follow the three-question opener (What / Who / Why) and include one **"What nobody tells you"** callout — the contrarian or non-obvious observation that separates the page from docs-style competitors.

## `core/` contributions

`core/rag/`, `core/memory/`, `core/tools/`, and `core/sandbox/` are the shared library. Changes here must:
- Keep a clean public API (we're building toward a potential standalone spin-out for `core/rag/`)
- Ship with tests in `core/<module>/tests/`
- Not break any recipe that depends on them (CI runs every recipe)

## Code style

- Python: f-strings, type hints, pathlib, 4-space indentation. Minimal comments — well-named identifiers first.
- Rust: standard rustfmt. Clippy clean.
- No comments explaining WHAT the code does. Only WHY, and only when the why is non-obvious.

## The PR checklist

- [ ] `make run` works from a fresh clone in ≤60s (beginner)
- [ ] `make eval` produces a reproducible score
- [ ] `pytest` green (production)
- [ ] README answers What / Who / Why + declares SOTA stack
- [ ] `techniques.md` cites primary sources for every technique used
- [ ] Recipe metadata (`recipe.yaml`) updated
- [ ] Links check with `markdown-link-check`
