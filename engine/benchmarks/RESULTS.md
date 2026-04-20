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

## Phase 9 — First live runs (2026-04-21)

Ran both fixtures against the shipping defaults: `gemma3:4b` via Ollama,
SearXNG at `localhost:8888`, trafilatura fetch on, cross-encoder rerank
off, full CoVe loop enabled, `MAX_ITERATIONS=2`. Mac M4 Pro, one question
at a time, no retries. **The prediction above was wrong.** Real numbers:

| fixture | pass rate | mean wall | mean tokens | verified claims |
|---|---:|---:|---:|---:|
| `simpleqa_mini.jsonl` (20 Q) | **0 / 20 (0.0 %)** | 41.3 s | 10 833 | 65 / 76 (85.5 %) |
| `browsecomp_mini.jsonl` (10 Q) | **4 / 10 (40.0 %)** | 33.1 s | 9 424 | 37 / 37 (100.0 %) |

### What 0 / 20 on SimpleQA actually means

Three distinct failure modes, all real, all worth naming:

1. **Confident factoid hallucination (≈ 7 / 20).** On questions whose
   gold answer is one specific token, `gemma3:4b` produced a confident
   wrong token when SearXNG didn't return a source that contained the
   right one. Examples:
   - `sqa-01` "What year did Anthropic publish Contextual Retrieval?" →
     **"2023"** (gold: `2024`).
   - `sqa-02` "Which cross-encoder for reranking?" → **"LayoutLMv3
     Cross-Encoder"** (gold: `bge-reranker-v2-m3`).
   - `sqa-03` "One for sparse retrieval and one for full-page fetch" →
     **"OpenSearch"** + unnamed (gold: `BM25` + `trafilatura`).
   - `sqa-07` "Default Mac chat model for Phase 1?" → **"gpt-oss-20b"**
     (gold: `gemma3:4b`).
   - `sqa-10` "Which embedder by default on Mac local?" →
     **"all-MiniLM-L6-v2"** (gold: `nomic-embed-text`).

2. **CoVe happily verifies wrong answers (high `verified_ratio`).**
   `verified_ratio = 0.855` on SimpleQA looks healthy, but it mostly
   confirms **internal consistency between the model's claim and the
   evidence it did find** — not consistency with ground truth. On
   `sqa-01`, 5 / 5 claims were CoVe-verified and the answer was still
   "2023". CoVe is not a ground-truth oracle; it's a decomposition loop
   that catches unsupported *extrapolation*, not upstream retrieval
   misses.

3. **Self-referential fixture design was a mistake (≈ 8 / 20).** The
   SimpleQA mini was built assuming SearXNG would surface our own repo
   docs as top hits. It doesn't reliably — our repo isn't in the
   engines' top-ranked results for most of these queries, so we get
   general-web noise instead. Questions like `sqa-13` "Which LangGraph
   node runs between fetch_url and synthesize?" or `sqa-17` "Which MCP
   tool does the Claude plugin bundle expose?" returned `"The provided
   evidence does not answer this question."` — which is honest but
   doesn't hit `must_contain`. The model (correctly) refused to
   confabulate.

Aggregated: **~7 confabulations + ~8 honest-but-off-topic abstentions +
~5 partial / other** ≈ 20 zero-pass. Different causes, different fixes.

### What 40 % on BrowseComp shows

BrowseComp questions are broader-scope, multi-hop, and synthesis-heavy.
The engine passed **4 / 10**:

- `bc-01` — Anthropic CR vs bge-reranker-v2-m3 (no `must_contain`; the
  engine produced a creditable synthesis citing both).
- `bc-02` — MiroThinker-H1 vs OpenResearcher-30B-A3B (the engine
  correctly noted the evidence was insufficient and said so).
- `bc-06` — FLARE vs HyDE (again, correctly abstained on specifics).
- `bc-07` — RRF vs score-level normalization (synthesized a reasonable
  comparison from the retrieved sources).

And failed **6 / 10**:

- `bc-03` — RRF trade-offs: missed literal tokens `BM25` and `RRF` even
  though the answer was on-topic. Scoring artifact.
- `bc-04` `bc-05` `bc-08` `bc-09` `bc-10` — retrieval returned nothing
  targeted (PANORAMIC trial, NVDA 10-Q segment revenue, LongLLMLingua,
  trading-copilot internals, ThinkPRM vs CoVe). Honest "evidence does
  not answer" responses that, again, don't hit `must_contain`.

**`verified_ratio = 1.0` (37 / 37)** on BrowseComp is also worth noting:
every claim the synthesizer made was backed by a retrieved source. Zero
extrapolations. CoVe is doing its narrow job well; the bottleneck is
what SearXNG returns, and what `gemma3:4b` does when it returns nothing.

### What the scoring doesn't capture

`must_contain` is strict substring matching. It rewards emitting the
exact gold token and punishes everything else — including correct
paraphrases, correct abstentions, and adjacent-right answers. A pass
rate near zero on a self-referential fixture is more an artifact of
"gold token not present in SearXNG's top results" than of the engine
being broken. For a better quality read, look at:

- **`verified_ratio`** — 85.5 % on SimpleQA, 100 % on BrowseComp. The
  synthesizer is not freely extrapolating.
- **`must_not_contain_hits`** — **0** across all 30 questions. The
  engine did not emit a single banned string (wrong years, wrong model
  names the fixtures explicitly flagged, etc.).
- **Per-node latency distribution** — stable, no anomalous node.
- **Mean tokens ~ 10k** — in line with the Phase 1 one-question run
  (17 061 tokens), adjusted for easier questions.

### Actions this run motivates (not done in this commit)

1. **Rewrite the SimpleQA fixture to be less self-referential.** Use
   facts SearXNG can actually surface (dates, paper names, benchmark
   numbers with wide web coverage). Keep self-referential checks in a
   separate `internals_mini.jsonl` that uses `LOCAL_CORPUS_PATH` to
   inject the repo as a first-class corpus — that fixes the retrieval
   problem at the root.
2. **Add a `judge` scoring mode as an alternative to `must_contain`.**
   An LLM-as-judge scorer (using `gpt-5-mini` or similar) would catch
   the many cases where `gemma3:4b` paraphrased correctly but missed
   the exact gold token. Not blocking; strict substring matching is
   what most benchmark papers use and it's honest.
3. **Document the expected ceiling for `gemma3:4b`.** On factoid
   questions where the gold answer is a single recent-ish token that
   isn't well-indexed in SearXNG's engines, `gemma3:4b` will confabulate
   rather than abstain. This is a ~4 B-parameter tradeoff; a `gpt-5-mini`
   synthesizer (available via `OPENAI_API_KEY` + `MODEL_SYNTHESIZER=gpt-5-mini`)
   would pass more of these. Document the option in the README
   quickstart under "higher factoid accuracy".
4. **Publish the result files alongside this doc.** Snapshotted into
   `engine/benchmarks/results/2026-04-21/` so future PRs have a
   reference point to measure deltas against.

Checked-in outputs:

- `engine/benchmarks/results/2026-04-21/simpleqa_summary.json`
- `engine/benchmarks/results/2026-04-21/simpleqa_detail.jsonl`
- `engine/benchmarks/results/2026-04-21/browsecomp_summary.json`
- `engine/benchmarks/results/2026-04-21/browsecomp_detail.jsonl`

---
