# Completed-answer evaluation, v1

## Gate 1: scope and implementation plan

Freeze this protocol, task manifest and parser before new model observations.
Reuse the pinned checkpoint, native cache-free generation and existing
`completed_final_integer_v1` parser; do not change historical outputs or scoring.
Add one small shared record validator/scorer, boundary tests, and a baseline smoke
runner. Publish its raw evidence in the existing Pages disclosure, without redesign.
Only after code, real-model smoke, documentation and public deployment pass may
intervention comparisons begin. No controller training is part of this gate.

## Inputs and partitions

`configs/completion_v1_tasks.json` fixes twelve development arithmetic questions
(generator validation templates, seed 20261101) and sixty reserved test questions
(generator test templates, seed 20261102). The partitions have disjoint exact
prompts and template IDs. Existing generator templates are not newly discovered,
nor guaranteed absent from pretraining or earlier research. "Reserved" means not
used for this campaign's development or executed by this smoke runner. Seeds of
the same problem are not independent questions.

The smoke executes **only completion_dev_000**, selected by position before new
outcomes. The remaining development questions belong to the later timing/utility
pilot, not this gate. A single completed smoke cannot establish task difficulty,
baseline accuracy or a suitable held-out sample size. Do not alter the held-out
set after inspecting outcomes. No efficacy hypothesis test is performed here.

## Shared generation and scoring contract

- Model: ByteDance/Ouro-1.4B-Thinking, pinned revision
  `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`, verified local weights.
- Pinned tokenizer chat template; one user message containing the manifest's
  prompt, `add_generation_prompt=True`, `enable_thinking=False`. This template
  option does not prohibit generated reasoning. No answer is in generation input.
- Four recurrent depths, forced last-step readout, BF16, eager attention,
  `use_cache=False`, greedy argmax, no beam search or repetition penalties.
- Maximum 2048 new tokens. Stop **only** at the first tokenizer EOS or budget.
  No newline, apparent answer, intermediate integer, or correctness-based stop.
- Preserve prompt/generated token IDs, decoded text with special tokens, exact
  full-prefix SHA-256/length at every token, physical recurrent-boundary call
  counts and elapsed time. Duplicate full prefixes can be reconstructed exactly;
  do not store a quadratic second copy of token IDs in the new public report.
- EOS means completed; budget exhaustion means unfinished even if the text
  happens to contain the correct answer. Retain every attempted problem.
- Use existing `extract_completed_integer` unchanged, only on EOS-completed
  output and outside the completed thought region. Bare integers, a unique
  terminal integer box, or a terminal Answer/Result integer label are accepted.
  Unrecognized output is invalid-format, not silently rescued by first-integer
  matching or gold-aware parsing. Compare extracted integers exactly to gold
  **after generation**; no teacher-forced gold scoring or selection.
- Exclusive output/journal creation, pre-run source/input/checkpoint hashes,
  post-run identity checks, and independent boundary-hook/counter reconciliation.
  An exception preserves the partial journal; it never becomes a successful run.

## Required outcomes and comparison boundary

Report `correct`, `incorrect`, `invalid_format`, and `budget_exhausted` separately.
Accuracy uses **all attempted questions**, with the latter three counted as not
correct. Also report completion/valid-format counts, output tokens, calls and
prefix-length-weighted recurrent token positions. Calls alone are not FLOPs:
full-prefix work grows with sequence length. Wall time is hardware-specific.

Gate 2 must separately freeze intervention timing, scope, donor construction and
strength, compare ordinary recurrence, direct mixing, matched random controls
and excursion-and-return, and charge donor/reference/selection work. Same output
token limits do not imply equal compute. Best-of-N selection must not access gold.
Existing Gaussian donor branches are perturbed states, not proven alternative
semantic reasoning strategies. No inherited historical efficacy result clears
this gate. Controller evaluation remains conditional on useful intervention.

## Reproduction

```bash
uv run pytest -q tests/test_completed_answer_protocol.py
uv run python scripts/verify_completed_answer.py --output /tmp/completed-answer-new.json
```

Requires the already downloaded exact checkpoint. A new output and sibling JSONL
path are mandatory. Synthetic/mini-model tests are labeled and never counted as
real checkpoint observations. See `reports/completed_answer_verification.md` for
the eventual execution/deployment status; this frozen document is not that proof.
