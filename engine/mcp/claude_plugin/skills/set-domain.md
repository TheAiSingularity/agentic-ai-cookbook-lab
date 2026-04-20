---
name: set-domain
description: Route a research query into a domain preset (medical, papers, financial, stock_trading, personal_docs, general). Domain presets tune search sources, prompts, and verification strictness.
triggers:
  - medical research
  - paper research
  - stock research
  - financial research
  - search my docs
---

Domain presets (shipped in engine/domains/*.yaml as of Phase 6):

| preset          | typical question                           | search sources tuned for        |
|---              |---                                         |---                              |
| `general`       | anything; default                          | broad SearXNG meta-search       |
| `medical`       | disease / treatment / drug / trial         | PubMed-biased queries, stricter verify |
| `papers`        | academic CS/ML / physics / biology paper   | arXiv + preprints + Semantic Scholar |
| `financial`     | company fundamentals, market commentary    | Bloomberg/Reuters/FT-biased     |
| `stock_trading` | technical indicators + current news per ticker | yfinance + SearXNG news       |
| `personal_docs` | your own PDF / markdown corpus             | no web — `LOCAL_CORPUS_PATH` only |

When a user says "do medical research on X," call
`engine.research(question=X, domain="medical")`. Do not hand-tune the
prompt; the domain preset handles that.

`personal_docs` requires the user to have built an index first:

    python scripts/index_corpus.py build ~/my-docs --out ~/my-docs.idx
    export LOCAL_CORPUS_PATH=~/my-docs.idx

If the user hasn't done that, surface the instruction before running the
query.
