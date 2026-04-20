# research-assistant/production

Same pipeline as `beginner/`, plus adaptive-verification and local-first
engine enhancements that target the hardest questions. This tier exists
because single-pass answering fails on multi-hop / ambiguous queries, and
because SearXNG snippets alone are too thin for serious research. Keep
`beginner/` for teachability; ship `production/` when quality matters.

## What's added on top of beginner

### Tier 2 (adaptive verification)

| Node / technique | Effect | Gated by |
|---|---|---|
| **HyDE** in `_plan` | Hypothetical-answer embedding for retrieval | `ENABLE_HYDE=1`; auto-skipped on numeric/factoid |
| **CoVe** in `_verify` | Claim-by-claim verification vs evidence | `ENABLE_VERIFY=1` |
| **Iterative retrieval** | Re-search unverified claims, regenerate | `MAX_ITERATIONS` (default 2) |
| **Self-consistency** | Sample N candidates, pick best by grounding | `ENABLE_CONSISTENCY` (opt-in) |

### Tier 4 (2026 SOTA techniques layered on top)

| Node / technique | Effect | Gated by |
|---|---|---|
| **T4.3 · Classifier router** (`_classify`) | Classifies question into `factoid / multihop / synthesis`; downstream nodes adapt compute | `ENABLE_ROUTER=1` |
| **T4.1 · Step-level critic** (`_critic`) | ThinkPRM-style judge after each major step; rejects bad plans/search/etc. | `ENABLE_STEP_VERIFY=1` |
| **T4.4 · Evidence compression** (`_compress`) | LLM-distills evidence 2-3× before synthesize (LongLLMLingua-lite) | `ENABLE_COMPRESS=1` |
| **T4.2 · FLARE active retrieval** (`_flare_augment`) | Detects hedged claims, re-searches for that exact claim, regenerates | `ENABLE_ACTIVE_RETR=1` |
| **T4.5 · Plan refinement** | One-shot replan when the critic rejects the decomposition | `ENABLE_PLAN_REFINE=0` (opt-in, has loop risk) |

### Wave 4 (local-first engine enhancements)

Local-first means no paid APIs on the hot path. These three plug into the
same portable stack (Ollama / vLLM / SGLang via `OPENAI_BASE_URL`, SearXNG
for search, `core/rag` for retrieval).

| Enhancement | Effect | Gated by |
|---|---|---|
| **W4.1 · Cross-encoder rerank** (`_retrieve`) | Two-stage: `HybridRetriever` returns top-N (default 50), `BAAI/bge-reranker-v2-m3` re-scores to top-K. Falls back to hybrid-only if the model can't load. | `ENABLE_RERANK=0` (opt-in; ~560MB first-run download) |
| **W4.2 · Full-page fetch** (`_fetch_url`) | Pulls each SearXNG result URL, extracts clean article text with `trafilatura`. Concurrency-bounded. Per-URL failures keep the snippet as fallback. | `ENABLE_FETCH=1` (default on) |
| **W4.3 · Observability trace** (`_chat` + CLI) | Records `{node, model, latency_s, tokens_est, prompt/response chars}` for every LLM call. Printed as a per-node / per-model summary at CLI end. | `ENABLE_TRACE=1` (default on) |

Everything above is open-source and self-hostable. `BAAI/bge-reranker-v2-m3`
is Apache-2.0, `trafilatura` is Apache-2.0, `sentence-transformers` is
Apache-2.0, SearXNG is AGPL, Ollama is MIT. No key required anywhere.

### Wave 5 (local corpus augmentation)

| Enhancement | Effect | Gated by |
|---|---|---|
| **W5.1 · Local corpus** (`_search`) | `scripts/index_corpus.py` builds a `CorpusIndex` from a directory of PDFs / markdown / text / HTML. When `LOCAL_CORPUS_PATH` is set, each sub-query pulls `LOCAL_CORPUS_TOP_K` hits (default 5) from the index and merges them into evidence with `corpus://<source>#p<page>#c<chunk>` URLs. `_fetch_url` skips those URLs — their text is already the full chunk. | `LOCAL_CORPUS_PATH=""` (unset = web-only) |

Build a corpus once, query it from CLI or attach it to the pipeline:

```bash
# build
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
EMBED_MODEL=nomic-embed-text \
python scripts/index_corpus.py build ~/papers --out ~/papers.idx

# query from CLI
OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=ollama \
EMBED_MODEL=nomic-embed-text \
python scripts/index_corpus.py query ~/papers.idx "attention mechanism" --k 5

# attach to production pipeline
export LOCAL_CORPUS_PATH=~/papers.idx
export LOCAL_CORPUS_TOP_K=5
make run Q="your question"
```

### Pipeline (with Tier 4 + Wave 4 + Wave 5)

```
classify → plan (+HyDE, +critic) → search (+W5 corpus, +critic) → retrieve (+W4.1 rerank)
                                                                         │
                                                          W4.2 fetch_url ┘
                                                                         │
                                                                         ▼
                                                                      compress
                                                                         │
                                                                         ▼
                                                                synthesize (+FLARE)
                                                                         │
                                                                         ▼
                                                                  verify (CoVe)
                                                                         │
                                                     verified ──yes──▶ END
                                                                         │
                                                                         no
                                                                         │
                                               iterate (re-search failed claims) ─▶ search
```

Compute scales with question difficulty: factoid questions exit the
classifier with shallower budgets; multihop/synthesis get the full
stack. See `eval/ablation.py` for the 12-config ablation matrix.

Easy questions exit in one pass (~same latency as beginner). Hard
questions get 2–3 extra LLM calls (verify + regenerate) and occasionally
one iteration of re-search for unverified claims.

## Run

Same env contract as `beginner/` plus the four knobs below. With your
local Ollama stack:

```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
export MODEL_PLANNER=gemma4:e2b
export MODEL_SEARCHER=gemma4:e2b
export MODEL_SYNTHESIZER=gemma4:e2b
export MODEL_VERIFIER=gemma4:e2b       # CoVe — cheap model is fine
export EMBED_MODEL=nomic-embed-text
export SEARXNG_URL=http://localhost:8888

# Tier 2 toggles (defaults shown)
export ENABLE_HYDE=1
export ENABLE_VERIFY=1
export MAX_ITERATIONS=2
export ENABLE_CONSISTENCY=0
export CONSISTENCY_SAMPLES=3

# Tier 4 toggles (defaults shown — all on except plan refine)
export ENABLE_ROUTER=1          # T4.3 question classifier → adaptive compute
export ENABLE_STEP_VERIFY=1     # T4.1 step-level critic (ThinkPRM pattern)
export ENABLE_ACTIVE_RETR=1     # T4.2 FLARE re-search on hedged claims
export ENABLE_COMPRESS=1        # T4.4 evidence compression before synthesize
export ENABLE_PLAN_REFINE=0     # T4.5 replan when critic rejects (opt-in)

# Wave 4 toggles (local-first engine enhancements)
export ENABLE_RERANK=0          # W4.1 opt-in; first run downloads bge-reranker-v2-m3 (~560MB)
export RERANK_CANDIDATES=50     # first-stage hybrid pool size
export ENABLE_FETCH=1           # W4.2 trafilatura clean-text fetch (snippets → articles)
export FETCH_TIMEOUT_SEC=10
export FETCH_MAX_CHARS=8000     # truncate per page after extraction
export FETCH_MAX_URLS=8         # cap concurrent fetches per cycle
export ENABLE_TRACE=1           # W4.3 per-call observability, summary printed at CLI end

# Wave 5 toggles (local corpus augmentation)
export LOCAL_CORPUS_PATH=""     # W5.1 set to a directory built by scripts/index_corpus.py
export LOCAL_CORPUS_TOP_K=5     # per-subquery hits pulled from the local corpus

make install
make run Q="your hard multi-hop research question"
```

Turning on rerank adds a `sentence-transformers` model load on first call
(cold ≈ 20s, warm < 1s). Leave it off for quick CLI iterations, turn it
on for the eval harness and real research runs.

On a GPU VM with vLLM/SGLang, point `OPENAI_BASE_URL` at `:8000/v1` and
set the model names to real tags (`Qwen/Qwen3.6-35B-A3B`, etc.).

## Test

```bash
make test   # mocked; no API key or network needed
```

## Sandboxed execution (HermesClaw)

Once `core/sandbox` lands, `compose.yml` here will boot the full
pipeline inside a HermesClaw sandbox so network egress and filesystem
access are policy-enforced. Placeholder for now.

## Expected cost / latency

| Question type | Extra calls vs beginner | Added latency |
|---|---|---|
| Easy, fully-verified first pass | +1 (verify) | +10–30s |
| Unverified → one iteration | +1 search + +1 synth + +1 verify | +30–90s |
| Hard + self-consistency enabled | N× synth + N× verify | 2–3× baseline |

All still $0 on a fully-local rig. On paid OpenAI: roughly 2× beginner's
per-query cost when iteration triggers.

## See also

- [`beginner/`](../beginner/) — the lean reference implementation (100 LOC)
- [`eval/`](../eval/) — benchmark harness (SimpleQA-100 + BrowseComp-Plus-50
  land with Tier 3)
- [`../../../core/rag/`](../../../core/rag/) — retrieval primitives used here
