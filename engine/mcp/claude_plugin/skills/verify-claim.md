---
name: verify-claim
description: Run a targeted verification of a single factual claim against fresh evidence. Uses the engine's CoVe-grade verifier under the hood.
triggers:
  - verify this claim
  - is this true
  - fact check
---

When the user wants a single specific claim checked:

1. Extract the claim into a concise factual question
   (e.g. "Did Anthropic publish Contextual Retrieval in September 2024?"
   rather than "Is it true that they published it?").
2. Call `engine.research(question=<claim-as-question>, memory="off")`.
3. Look at `verified_claims` + `unverified_claims`. The engine's CoVe
   verifier decomposes the answer into atomic factual claims and tags
   each with `VERIFIED: yes|no` against the evidence.
4. Report back:
   - **If every claim returned is verified**: say "Supported by the
     evidence" and cite the top 1-2 sources.
   - **If any are unverified**: say "Partially supported — the engine
     could not verify: <list>" and show the verified subset plus the
     sources behind them.
   - **If the answer is the refusal phrase** ("The provided evidence
     does not answer this question."): say "No evidence found in the
     sources the engine could reach" and list where it looked.

Never assert something is true that the engine's CoVe step did not
verify.
