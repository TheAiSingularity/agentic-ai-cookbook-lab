# research-assistant/production

Same pipeline as `beginner/`, plus four adaptive-verification techniques
that target the hardest questions. This tier exists because single-pass
answering fails on multi-hop / ambiguous queries. Keep `beginner/` for
teachability; ship `production/` when quality matters.

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

### Pipeline (with Tier 4)

```
classify → plan (+HyDE, +critic) → search (+critic) → retrieve → compress
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

make install
make run Q="your hard multi-hop research question"
```

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
