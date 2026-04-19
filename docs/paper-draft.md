# Adaptive Verification Substitutes Fine-Tuning for SOTA Deep Research

*Working draft — numbers are placeholders until the ablation runs on the GPU VM.*

**Authors:** TBD (TheAiSingularity)
**Target venue:** Technical blog post + arXiv tech report (no formal peer-review deadline)
**Draft date:** 2026-04-20
**Repository:** https://github.com/TheAiSingularity/agentic-ai-cookbook-lab (commit hash locked at submission)

---

## Abstract (one paragraph, last thing we write)

*Placeholder. Final version after numbers: "We show that combining hybrid
retrieval + cross-encoder reranking + contextual chunking + Chain-of-
Verification + iterative retrieval on commodity open-weight LLMs
(`Qwen3.6-35B-A3B`, `Gemma-4-31B`) on a single 4×RTX 6000 Pro
workstation matches the BrowseComp-Plus performance of MiroThinker-1.7
(a specialized research-trained model) without any model-specific
training. We quantify each technique's marginal contribution via
leave-one-out ablation on SimpleQA-100 and BrowseComp-Plus-50. All code,
configs, and datasets are released for reproducibility."*

---

## 1. Introduction

Open-source deep-research agents made major jumps in early 2026:
MiroThinker-1.7 hit **74.0 on BrowseComp** with open weights; MiroThinker-H1,
which integrates verification into the inference loop, pushed this to
**88.2 on BrowseComp** — surpassing Gemini-3.1-Pro (85.9) and
Claude-4.6-Opus (84.0). The H1 result is striking because its technique
is architectural (verification at local + global levels), not scale-bound.

This paper asks a simple question: **can that architectural trick be
composed on top of any off-the-shelf open-weight LLM, without retraining,
to reach competitive numbers?** If yes, deep-research SOTA becomes a
recipe problem rather than a training problem — democratizing the
capability for any lab with a multi-GPU workstation.

**Thesis:** A research agent built from (a) hybrid retrieval, (b)
contextual chunking, (c) Chain-of-Verification, (d) iterative retrieval,
and (e) optional self-consistency, running on commodity open-weight
LLMs, reaches within 5 points of MiroThinker-1.7's BrowseComp-Plus score
while using a fraction of the inference compute.

**Secondary contribution:** the full stack is portable — we demonstrate
it runs identically on OpenAI, Ollama, and vLLM/SGLang via the
OpenAI-compatible API. Every component is MIT-licensed. Every dataset is
open. Every number is reproducible from a single `make ablate` on a
~$15k workstation.

---

## 2. Related work

### 2.1 Deep-research agents (2024–2026)

- **OpenAI Deep Research** (closed) — introduced the "plan-execute-synthesize" pattern for web research.
- **MiroThinker-1.7 / H1** (open) — MiroMind, 2026. H1's wedge is
  verification-integrated reasoning.
- **MiroFlow** (orchestration) — 3-tier (Foundation / Agent / Control)
  hierarchical framework with MCP-based tools.
- **Browser Use / Agent-Sandbox / agent-infra** (tool substrate) —
  parallel work on sandboxed tool calls.

### 2.2 Retrieval techniques we build on

- **Hybrid retrieval** — BM25 + dense + RRF (classical; RAGFlow 2026
  review; arXiv 2604.01733).
- **Cross-encoder reranking** — BAAI/bge-reranker-v2-m3 (2024).
- **Contextual retrieval** — Anthropic 2024; -35 to -67% retrieval failures.
- **HyDE** — Gao et al. 2023; gated for precision queries per our findings.

### 2.3 Verification and consistency

- **Self-consistency** — Wang et al. 2022.
- **Chain-of-Verification (CoVe)** — Dhuliawala et al. 2023.
- **ITER-RETGEN** — Shao et al. 2023.
- **MiroThinker-H1** — 2026 application of verification to agent inference.

### 2.4 Inference infrastructure

- **vLLM / SGLang** — 2026 production-standard OSS inference engines.
  SGLang's RadixAttention gives +29% throughput on H100 and up to 6.4×
  on prefix-heavy RAG (particula.tech / morphllm.com 2026).
- **EAGLE speculative decoding** — 2–3× decode speedup at zero quality
  loss; built into vLLM and SGLang.

---

## 3. Method

### 3.1 System architecture

```
plan (+HyDE) → search → retrieve (hybrid + rerank) → synthesize
        │                                              │
        │                                              ▼
        │                                         verify (CoVe)
        │                                              │
        │                                              ▼
        │                              verified? ──yes──▶ (consistency) ──▶ END
        │                                              │
        │                                              no
        │                                              │
        └──── iterate (re-search failed claims, ≤K times) ┘
```

Implemented as a LangGraph `StateGraph` with a conditional edge from
`verify` back to `search`. The conditional predicate checks `unverified
claims ≠ ∅` and `iterations < MAX_ITERATIONS`.

### 3.2 Components

- **Classify (Tier 4.3):** A cheap classifier call routes the question
  to one of `{factoid, multihop, synthesis}`; downstream nodes use this
  label to adapt compute budgets (fewer sub-queries for factoid, etc.).
- **Planner:** N sub-queries generated by the cheapest-tier model. HyDE
  expansion appended to each sub-query, gated on a regex over the
  original question (numeric / factoid queries skip HyDE).
- **Plan refinement (Tier 4.5):** If the step critic rejects the
  decomposition, regenerate once with a tightening instruction.
- **Search:** N parallel SearXNG queries. Each sub-query's top-k
  snippets are LLM-summarized with inline citations.
- **Step critic (Tier 4.1):** After plan and after search, a
  ThinkPRM-style judge scores the step with `VERDICT: accept|redo` +
  feedback. Logs per-step; triggers refinement in plan, records concern
  in search.
- **Retrieve:** Hybrid BM25 + dense + RRF (k=60); optional contextual
  chunking; optional cross-encoder rerank on top-50 → top-K.
- **Compress (Tier 4.4):** LLM-based evidence distillation; each
  chunk is condensed to 2-3 sentences focused on the question, URLs
  preserved for citations. Portable alternative to LongLLMLingua.
- **Synthesize:** Final answer produced with inline `[N]` citations.
- **FLARE active retrieval (Tier 4.2):** A regex scan of the draft
  answer detects hedging phrases ("the evidence does not specify",
  etc.); triggers a targeted re-search for just that claim, then
  regenerates once with the fresh evidence.
- **Self-consistency:** N samples ranked by coverage-weighted grounding
  score `(valid_refs / total_refs) × sqrt(valid_refs)`.
- **Verify (CoVe):** Verifier model parses the answer into discrete
  claims and labels each `VERIFIED: yes|no` against evidence.
- **Iterate:** Unverified claims become new sub-queries; their search
  results append to evidence; synthesize re-runs. Bounded by
  `MAX_ITERATIONS` (default 2).

### 3.3 Implementation details

- **LLM backends:** OpenAI, Ollama, vLLM, SGLang — all via the
  OpenAI-compatible `/v1/chat/completions` API. Portability preserved
  via env vars (`OPENAI_BASE_URL`, `MODEL_*`, `EMBED_MODEL`).
- **Inference engine:** vLLM or SGLang depending on `--engine` flag.
  SGLang recommended for the prefix-cache gain on our prompt-heavy
  pipeline.
- **Speculative decoding:** EAGLE via `--spec-dec` — orthogonal to all
  other techniques; enabled for all timed runs.
- **Tokenizers:** HuggingFace `tokenizers` (Rust-backed under the
  Python import) — we get Rust's speedup without changing language.

---

## 4. Experimental setup

- **Hardware:** 4× NVIDIA RTX 6000 Pro Blackwell (384 GB VRAM), Ubuntu 24.04, CUDA 12.x.
- **Inference:** vLLM (baseline) and SGLang (prefix-cache comparison);
  EAGLE speculative decoding enabled by default.
- **Primary LLM:** `Qwen/Qwen3.6-35B-A3B` (MoE, 35B total / 3B active).
  **Secondary:** `google/gemma-4-31b-it` for a model-comparison axis.
- **Embeddings:** `BAAI/bge-m3` (self-hosted).
- **Reranker:** `BAAI/bge-reranker-v2-m3` (self-hosted, lazy-loaded).
- **Search:** SearXNG self-hosted, meta-search over DDG/Bing/Wikipedia/arXiv.
- **Judge:** `gpt-5-mini` via OpenAI API (only the evaluation uses paid
  API — the agent itself is fully self-hosted).
- **Benchmarks:**
  - **SimpleQA-100** — 100-question subset of OpenAI's SimpleQA (random
    seed 42 over full 4,326-question dataset).
  - **BrowseComp-Plus-50** — 50-question subset of BrowseComp-Plus
    (ACL 2026; `texttron/BrowseComp-Plus`; random seed 42).

### 4.1 Ablation matrix (12 configs)

**Tier 1 (retrieval):**
| Label | Retriever | HyDE | Verify | Iterate | Consistency |
|---|---|---|---|---|---|
| **A1** | v0 naive cosine | ✗ | ✗ | 0 | ✗ |
| **A2** | v1 hybrid (BM25+dense+RRF) | ✗ | ✗ | 0 | ✗ |
| **A3** | v1 + contextual + rerank | ✗ | ✗ | 0 | ✗ |

**Tier 2 (verification):**
| Label | Retriever | HyDE | Verify | Iterate | Consistency |
|---|---|---|---|---|---|
| **B1** | v1 + rerank | ✗ | ✓ | 0 | ✗ |
| **B2** | v1 + rerank | ✓ | ✓ | 0 | ✗ |
| **B3** | v1 + rerank | ✓ | ✓ | ≤1 | ✗ |
| **B4** | v1 + rerank | ✓ | ✓ | ≤1 | ✓ (N=3) |

**Tier 4 (SOTA techniques layered onto B3):**
| Label | Step critic | FLARE | Compress | Router | Plan refine | Consistency |
|---|---|---|---|---|---|---|
| **C1** | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| **C2** | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| **C3** | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| **C4** | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| **C5** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (N=3) |

Each config runs every question 3 times (seeds 0–2) and we report means.

### 4.2 Metrics

- **Factuality** — LLM-as-judge (`gpt-5-mini`, temperature 0) returns
  {0, 0.5, 1} for the candidate answer vs gold.
- **Citation accuracy** — proportion of `[N]` refs in the answer that
  point at a real evidence row (catches hallucinated indices).
- **Citation precision** — proportion of `must_cite_any` domains present
  in answer URLs.
- **Latency p50, p95** — wall-clock seconds per question.
- **Tokens (est.)** — rough cost proxy; ~1 token per 4 chars.

### 4.3 Reference baseline

`MiroThinker-1.7` served on the same rig via the instructions in its
repo. We treat it as a black-box numbers source — we don't retrain or
modify weights.

---

## 5. Results

*(placeholder tables; replace when ablation run completes)*

### 5.1 Top-line on BrowseComp-Plus-50

| Config | Factuality | Cite Accuracy | p50 (s) | Tokens/query |
|---|---|---|---|---|
| A1 baseline | **TBD** | TBD | TBD | TBD |
| A3 (Tier 1 max) | TBD | TBD | TBD | TBD |
| B3 (Tier 2 default) | TBD | TBD | TBD | TBD |
| B4 (Tier 2 full) | TBD | TBD | TBD | TBD |
| **MiroThinker-1.7** reference | TBD | — | TBD | TBD |

### 5.2 Pareto frontier

[placeholder for `eval/pareto.png`]

### 5.3 Leave-one-out ablation

*Table of ΔFactuality when each technique is individually disabled.*

### 5.4 Error analysis

*Categorize failures: retrieval failure, reasoning failure, verification
false-positive, verification false-negative. Count per config.*

---

## 6. Discussion

### 6.1 When adaptive verification helps most

*To write from data: we expect biggest gains on multi-hop and citation-
heavy questions; smallest gains on single-fact lookups.*

### 6.2 When HyDE hurts

*Quantify the numeric-query regression; compare gated vs ungated.*

### 6.3 Cost breakdown per config

*Per-query cost in LLM-calls, GPU-seconds, $0 on our rig. Table.*

### 6.4 Where Rust earns its place

`recipes/by-pattern/rust-mcp-search-tool/` — Rust MCP server wrapping
SearXNG. **4 ms** cold start vs ~60–140 ms for a Python equivalent;
~5 MB binary vs ~40 MB Python venv. The SearXNG client itself is
HTTP-bound so end-to-end latency gains are negligible — but the Rust
artifact is useful in deployments where a remote agent needs a small,
self-contained search tool without the full Python stack.

---

## 7. Conclusion

*Thesis-level summary: SOTA deep-research is composable.*

---

## 8. Reproducibility

All runs reproducible from the commit hash at submission. To rebuild all
numbers on the same hardware:

```bash
git clone https://github.com/TheAiSingularity/agentic-ai-cookbook-lab
cd agentic-ai-cookbook-lab
bash scripts/setup-vm-gpu.sh --engine sglang --spec-dec \
  --model Qwen/Qwen3.6-35B-A3B
cd recipes/by-use-case/research-assistant/eval
# Replace seed datasets with the full-100 / full-50 downloads per datasets/README.md
make ablate CONFIGS=A1,A2,A3,B1,B2,B3,B4
make pareto
```

Seeds, configs, model tags, and GPU ordering all recorded in the repo
state at submission time.

---

## Appendix A. Compute budget

- **Ablation run:** 7 configs × (100 SimpleQA + 50 BrowseComp-Plus) × 3 seeds
  = 3,150 runs × ~60 s/run ≈ **~52 GPU-hours** on the rig (tensor-parallel=4).
- **Paid API (judge only):** 3,150 questions × 2 LLM-judge calls each
  × ~800 tokens/call ≈ 5M tokens of `gpt-5-mini`. At April 2026 pricing
  that's roughly $1–2 of judge cost for the full replication.

## Appendix B. Negative findings to include

*Fill in from ablation — things that we expected to help but didn't, or
that helped less than literature suggested. Honest reporting strengthens
the paper.*
