# Benchmark datasets

Two benchmark subsets used by the ablation runner. Small seed samples ship
in this repo; **pull the full datasets before running the ablation on your
GPU VM** — the seeds exist so `ablation.py` works end-to-end in tests and
on a Mac without the full downloads.

## Format

One JSON object per line in each `.jsonl` file:

```json
{
  "id": "sq_001",
  "question": "Which chemist won the Nobel Prize in 2024?",
  "gold_answer": "Demis Hassabis, John Jumper, and David Baker won the 2024 Nobel Prize in Chemistry.",
  "must_cite_any": ["nobelprize.org", "wikipedia.org"],
  "source": "SimpleQA"
}
```

- `id`: stable identifier for per-question accounting.
- `question`: the prompt passed to the agent.
- `gold_answer`: reference answer used by the LLM-as-judge factuality scorer.
- `must_cite_any`: substrings at least one of which should appear in the
  agent's cited URLs (proxy for "did it find the canonical source").
- `source`: `SimpleQA` or `BrowseComp-Plus` — lets the ablation runner
  bucket results by benchmark.

## Datasets

### `simpleqa_seed.jsonl` (5 Q seeds, representative)

SimpleQA is OpenAI's factuality benchmark — 4,326 short-answer questions
designed to test whether models hallucinate. Full dataset at:
https://github.com/openai/simple-evals (MIT).

On your GPU VM, replace this seed file with a curated 100-question
subset (diverse domains: history, science, pop culture, geography).
Script to do that:

```bash
# From the OpenAI simple-evals repo
python scripts/select_simpleqa_subset.py --n 100 --seed 42 \
  > datasets/simpleqa_100.jsonl
```

(Script not shipped here — drop it in when your VM has the full data.)

### `browsecomp_plus_seed.jsonl` (5 Q seeds, representative)

BrowseComp-Plus (ACL 2026 Main, `texttron/BrowseComp-Plus` on GitHub) is
the fair-and-transparent evaluation benchmark for deep-research agents —
successor to OpenAI's BrowseComp with gold URLs for reproducible
retrieval evaluation.

On your GPU VM, replace this seed with a 50-question subset:

```bash
git clone https://github.com/texttron/BrowseComp-Plus /tmp/bcp
python scripts/convert_browsecomp_plus.py --n 50 --seed 42 \
  --in /tmp/bcp/data.jsonl > datasets/browsecomp_plus_50.jsonl
```

## Why seeds first

Shipping real data copies is an attribution / licensing headache. The
seeds are hand-crafted by us in the same format — they're labeled with
`source` set to the benchmark they emulate, but they're not *from* those
benchmarks. Real benchmark files overwrite these at eval time.
