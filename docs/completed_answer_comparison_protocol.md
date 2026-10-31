# Gate 2: fixed-method completed-answer comparison smoke

Prerequisite: gate 1 completed, including public verification at `d74a61f` and
documentation closeout `7de1851` (CI 41245742299 passed). No subagents.

## Frozen scope before observations

This is an exploratory **technical comparison smoke on completion_dev_000**, not
an efficacy pilot or held-out benchmark. Reuse its immutable native S0 record from
gate 1; do not rerun or relabel it. Verify eight greedy S0 adapter tokens against
that recorded native prefix before new method completions, charging these parity
calls separately. A sampled check does not prove all future-token/device parity.

Generate five new completions, in fixed order: B_only, M0, M1, E0, E1. Same pinned
checkpoint/tokenizer, BF16, MPS, full-prefix uncached greedy generation, four depths,
2048-token ceiling and EOS-only stop as gate 1. Same unchanged prompt and integer
parser. Gold is only read by offline scoring, not generation, selection or stopping.
Retain unfinished, invalid and wrong outputs; no favorable-arm filtering.

- Fork after depth one; B receives normalized Gaussian perturbation, relative
  norm 0.25, CPU-seeded by the existing branch helper. Advance within the same
  prefix; branches do not persist across output tokens. Seed 20261101 + 131*k.
- Fixed intervention into depth three (state depth two), at **every generated
  token**, alpha=beta=0.65, full scope, trust radius 0.1, no gating/projection.
  This reuses gate C's larger-radius setting, not a new outcome-tuned search.
- B_only: no transfer, emit the scheduler's B branch. Both A/B computations are
  charged (seven calls/token); not an optimized independent-sampling baseline or
  best-of-N selection. No oracle uses gold to pick a branch or completion.
- M0: direct state mixing; M1: direct update mixing; each seven calls/token.
- E0: excursion-and-return along a random direction matching donor distance;
  E1: excursion-and-return toward B; each eight calls/token.
- E0 matches input direction distance, **not returned output displacement**.
  Gate C's output-matched controls remain mechanism evidence, not completed-answer
  utility evidence. A later efficacy campaign still needs output-matched ablations
  and an independently generated full-answer selection baseline.

Use the existing scheduler/operators rather than another latent transition engine.
Do not reuse the old short-answer runner's per-prefix full-vocabulary logit cache
for long generation. The new loop releases logits/states each token and performs
no teacher-forced gold scoring. Record actual prefix hashes, selected hidden-state
hashes, seeds, telemetry and physical/scheduler/adapter call reconciliation.

## Cost and interpretation

E0/E1 have matched recurrent call counts **at the same prefix**, not necessarily
equal total work after their generated prefixes diverge. Count actual tokens,
calls, recurrent token positions and wall seconds for each whole completion. The
native S0 record's 1,524 calls and 541.415 seconds remain its recorded cost, not
zero because reused. Diagnostic parity work is additional campaign overhead.
Recurrent token positions are not FLOPs; attention cost is not linear in length.
No compute-matched accuracy claim, statistical significance or controller utility
can follow from this single already-correct development question.

Create exclusive append-only journal and per-method files in a new directory.
Journal each token; on failure keep partial evidence and exit, never silently
restart or overwrite. Freeze source/input/checkpoint hashes before observation
and verify afterward. Run methods serially to avoid exhausting 16 GB memory.
No test questions or controller training/evaluation are allowed in this gate.
Gate 3 starts only after this technical comparison, docs and Pages are verified.
