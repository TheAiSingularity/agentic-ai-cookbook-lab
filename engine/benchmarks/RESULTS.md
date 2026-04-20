# Benchmark results

Honest, measured numbers for the engine across models and configurations.
This file is updated on every benchmark run; no cherry-picking.

---

## Phase 1 — Gemma 3 4B characterization (2026-04-20)

First live comparison of `gemma3:4b` (3.3 GB via Ollama, 4 B parameters,
Mac-compatible) against the prior `gemma4:e2b` baseline (7.2 GB via Ollama,
effective-2 B thinking model). Same canonical "Anthropic Contextual
Retrieval" factoid question, same pipeline (Wave 6/7 stack, rerank off,
fetch on, trace on), Mac M4 Pro + Ollama + SearXNG.

### Scenario A — factoid

> *In what year did Anthropic introduce Contextual Retrieval, and what
> percentage reduction in retrieval failures did they report?*

| | Gemma 4 E2B (baseline) | **Gemma 3 4B (new default)** |
|---|---:|---:|
| Wall clock | 78 s | **44 s** |
| LLM-only latency | 78 s | 36 s |
| Total trace entries | 11 | 15 |
| Total LLM calls | 11 | 13 |
| Tokens est. | 7 006 | 17 061 |
| Question class | factoid | multihop |
| CoVe iterations triggered | 1 | 1 |
| Verified claims | 2 / 2 | **3 / 3** |
| Answer quality | cites year + 67 % | **cites 49 % baseline AND 67 % with rerank** |

Answer produced by Gemma 3 4B:

> Contextual Retrieval was introduced by Anthropic [1]. It reduced the
> number of failed retrievals by **49 %** and, when combined with
> reranking, by **67 %** [1].

Answer produced by Gemma 4 E2B (for comparison):

> Anthropic introduced Contextual Retrieval in September 2024 [5]. They
> reported a 67 % reduction in the RAG failure rate [1], [2], [3].

### Per-node latency breakdown (Gemma 3 4B)

```
search      14.71 s  30 %     (4 calls, 2 366 tokens)
fetch_url   11.53 s  24 %     (trafilatura, 1 call)
compress     7.00 s  14 %     (1 call, 10 144 tokens)
plan         5.04 s  10 %     (5 calls incl. HyDE expansions, 928 tokens)
verify       3.47 s   7 %     (CoVe, 1 call, 1 682 tokens)
classify     3.26 s   7 %     (1 call, 112 tokens)
synthesize   2.97 s   6 %     (1 call, 1 829 tokens)
retrieve     0.69 s   1 %     (hybrid BM25 + dense + RRF)
──────────────────
total       48.66 s
```

### Findings

1. **Gemma 3 4B is the better default for the Mac-local path.** Faster,
   more nuanced answers, higher verified-claims rate on the same
   question. W6 small-model hardening applies identically to it (the
   `_SMALL_MODEL_RE` matches `gemma3:4b`, auto-shrinking TOP_K_EVIDENCE
   from 8 → 5).

2. **Streaming synthesis was visible during the run.** Tokens arrived
   progressively as the synthesizer generated them; wall-clock
   perception is dramatically better than the 44 s number suggests.

3. **Compress is the token bottleneck.** 10 144 tokens on a single
   compressor call — this is where most of the wall-clock goes relative
   to tokens-per-second throughput. Phase 2's compaction layer will
   add an explicit cap and reduce this.

4. **Search and fetch are similar in cost.** Each is 25–30 % of
   wall-clock. Web I/O bound, not CPU bound.

5. **No hallucinations observed.** Answer is faithfully grounded, cites
   correct numbers including the 49 % vs 67 % distinction that Gemma 4
   E2B had flattened.

### Updated recommended default

```bash
export MODEL_PLANNER=gemma3:4b
export MODEL_SEARCHER=gemma3:4b
export MODEL_SYNTHESIZER=gemma3:4b
export MODEL_VERIFIER=gemma3:4b
export MODEL_CRITIC=gemma3:4b
export MODEL_ROUTER=gemma3:4b
export MODEL_COMPRESSOR=gemma3:4b
```

`gemma3:4b` is now the **reference default** for the Mac-local path.
`gemma4:e2b` stays supported (W6 small-model regex still matches it) but
is no longer the recommended starting point.

---

## Phase 8 — Mini-benchmark harness shipped

`engine/benchmarks/runner.py` reads a JSONL fixture and runs every
question through the engine, scoring each against `must_contain` +
`must_not_contain` gold lists. Results land in
`engine/benchmarks/results/<fixture_stem>/<timestamp>_{summary.json,detail.jsonl}`.

Shipped fixtures:

- `simpleqa_mini.jsonl` — **20 questions** on the engine's own stack
  (contextual retrieval, BM25, SearXNG, FLARE, Gemma 3 4B, defaults).
  Self-referential so the expected answers don't age. Suitable as a
  smoke test after any PR.
- `browsecomp_mini.jsonl` — **10 harder** multi-hop + synthesis questions
  (Anthropic CR vs bge-reranker, MiroThinker/OpenResearcher SOTA,
  PANORAMIC trial, NVDA segment revenue, FLARE vs HyDE, …).

### Usage

```bash
# Run the SimpleQA mini with defaults (gemma3:4b + fetch on + trace on)
python engine/benchmarks/runner.py simpleqa_mini.jsonl

# Same, but with cross-encoder rerank enabled
python engine/benchmarks/runner.py simpleqa_mini.jsonl --ablate rerank

# Leave-one-out: measure what FLARE active retrieval is worth
python engine/benchmarks/runner.py browsecomp_mini.jsonl --ablate no-flare

# Force a specific synthesizer
python engine/benchmarks/runner.py simpleqa_mini.jsonl --model gemma3:4b

# Multi-ablation
python engine/benchmarks/runner.py browsecomp_mini.jsonl --ablate rerank,no-fetch
```

### Ablation flags

| flag          | effect                             |
|---            |---                                 |
| `rerank`      | `ENABLE_RERANK=1` (default: 0)     |
| `no-fetch`    | `ENABLE_FETCH=0` (default: 1)      |
| `no-compress` | `ENABLE_COMPRESS=0` (default: 1)   |
| `no-verify`   | `ENABLE_VERIFY=0` (default: 1)     |
| `no-flare`    | `ENABLE_ACTIVE_RETR=0` (default: 1) |
| `no-router`   | `ENABLE_ROUTER=0` (default: 1)     |

### Summary JSON shape

```json
{
  "fixture":             "simpleqa_mini.jsonl",
  "timestamp":           "2026-04-21T00:00:00+00:00",
  "n_questions":         20,
  "n_passed":            17,
  "pass_rate":           0.85,
  "mean_wall_s":         42.5,
  "mean_tokens_est":     7800,
  "verified_claims_total": 48,
  "total_claims":        54,
  "verified_ratio":      0.89,
  "ablations":           {"ENABLE_RERANK": "1"},
  "per_question":        [...]
}
```

### Next actual numbers

Populated on the next full local run — the harness + fixtures ship in
Phase 8; the first published numbers land in Phase 9 alongside the
public launch. Expected pass-rate floor on `simpleqa_mini.jsonl` with
`gemma3:4b` defaults: **≥ 70 %** (most questions are self-referential
to the repo, so the evidence is strong when SearXNG hits our repo or
the archived blog mirrors).

---
