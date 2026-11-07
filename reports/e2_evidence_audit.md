# E2 evidence audit — 2026-11-08

## Correction and scope

E2 has already been evaluated. The later one-question E1 smoke does not replace
or refute those adaptive-E2 results. This audit recounts stored outcomes in the
original 32-job report committed at `bfe4f046d4759df76bda96a085f701afa1b66c45`
and the expanded 76-job report committed at `70cadbf`. It does not rerun models
or independently validate every answer. Reproduce with:

```sh
python scripts/audit_e2_evidence.py
pytest -q tests/test_e2_evidence_audit.py
```

| Stored report | S0 correct | E2 correct | E2 rescues / regressions | S0 / E2 tokens | S0 / E2 recurrent calls | Records missing raw token IDs |
|---|---:|---:|---:|---:|---:|---:|
| [32 jobs, 16 tasks](colab_l4_deceptive_traps_3000_32jobs.json) | 11/16 | 12/16 | 2 / 1 | 22,725 / 20,667 | 90,964 / 89,141 | 0/32 |
| [76 jobs, 38 tasks](colab_l4_deceptive_traps_3000_76jobs.json) | 26/38 | 25/38 | 4 / 5 | 62,972 / 58,873 | 252,040 / 251,963 | 32/76 |

The expanded report's stored accuracy is 68.4% for S0 and 65.8% for E2
(-2.6 percentage points). Tokens decrease 6.5%, but recurrent calls decrease only
0.031% (77 calls). Fewer output tokens are not equivalent to proportionately less
compute. Calls are not FLOPs; concurrent-worker wall times are not isolated
latency measurements. These selected, overlapping development collections do not
support population-level significance claims, and must not be pooled as 54
independent tasks.

The original `trap_bat_ball_glove` pair was stored as `invalid_format`; the expanded
report reclassifies both as `correct` without changing their raw generation.
Therefore even overlapping stored scores use different extraction behavior.

## Wins and losses, not just highlights

Expanded-report E2 rescues (all have raw token IDs for both methods):
- `trap_hotel_empty_rooms`
- `trap_potted_tree_soil`
- `trap_math_parentheses_addition`
- `trap_rope_burn_timing`

Expanded-report E2 regressions:
- `trap_non_hanoi_disk_moves` — raw token IDs missing
- `trap_aime_empty_intersection_sum` — raw token IDs missing
- `trap_aime_empty_grid_chips` — raw token IDs missing
- `trap_monty_hall_4doors` — raw token IDs present
- `trap_simplebench_frying_pan_ice` — raw token IDs missing

Missing raw tokens mean these are stored claims, not independently replayable
observations. Do not silently drop the missing records to manufacture a favorable
subset, and do not infer that missing evidence proves the underlying result false.
Keep the originals unchanged; recover source artifacts or label them unresolved.

## What changed since the cited commit?

- **Data:** 22 tasks were added after `bfe4f04`, but they are already included in
  the 76-job report. No further deceptive-task expansion occurred between that
  report and this audit. Dataset-source documentation is not new benchmark data.
- **E2 mechanism:** no new donor/update-space algorithm was introduced afterward.
  The token-window state machine was made explicit. No-op/unstable interventions
  now do not count or start a cooldown; this can affect trajectories when those
  cases occur. The donor-cache selection fix concerns P0, not an E2 breakthrough.
- **Execution validation:** later FP32 cached/native gates passed, while BF16
  gates failed their fixed tolerances. The historical parallel runner selects
  BF16 on CUDA. That does not by itself negate a cached S0/E2 comparison, but its
  numerical regime must not be labeled FP32-parity-validated.
- **Provenance:** historical summaries do not bind model revision, executed source
  hashes or prompt tokens. The findings document names revision `846cff8...`, but
  the runner imports `PINNED_OURO_THINKING_REV`, which was `3aaa222...` at the cited
  commit. The summaries cannot resolve which weights actually ran. New canonical
  checkpoints record substantially stronger identities.
- **Runner safety:** `run_deceptive_parallel.py` resumes by task ID/method without
  checking prompts, budgets, model or code; unknown task IDs are silently skipped.
  Its blocking result queue can hang after worker initialization failure. Do not
  use that legacy resume path for a new controlled campaign. Integrate this task
  family into the checkpointed single-worker runner before rerunning.

## Recommended next work, in order

1. Recover missing raw generations if they exist, freeze one offline scorer and
   rescore both historical reports consistently. Separate unrecoverable rows from
   raw-backed evidence without excluding their stored adverse outcomes.
2. Integrate deceptive tasks into `run_colab_l4.py` with the historical E2 settings
   (including perturbation relative norm 0.14, alpha/beta 0.65, window 1), rather
   than silently using its different suite defaults. Preserve exact manifests,
   model/source identities and per-generation checkpoints.
3. A **bounded E2 replication is justified by the precision/provenance changes**,
   not by pretending E2 was never evaluated. First replicate all four reported
   rescues and five regressions at frozen FP32 settings with paired S0. This is an
   outcome-selected diagnostic set, not an unbiased accuracy estimate. Do not
   extend budgets after seeing outcomes. Follow with a frozen full-set comparison
   only if execution checks pass.
4. Improve E2 only after diagnosing the losses. Candidate hypothesis: uncertainty
   is not sufficient evidence that an excursion will help. Add decision telemetry
   and test an acceptance/abstention rule against unchanged E2 with matched-cost
   controls. Confidence gain alone is not proof of correctness. Train or tune only
   on development data, never on the evaluation answers.

Do not create another E3 variant merely to name a polishing pass. Learned E3
variants already exist in the scheduler, but the hardened cached-generation API
currently does not support them; they require their own execution and controller
validation before comparison. No new GPU run was launched for this audit.

Historical narrative claims about attention-head fixation, specific token-direction
causality or attractor escape are hypotheses, not measurements in these records.
The headline finding is **mixed E2 evidence with some promising rescues**, not a
verified overall accuracy improvement or a measured causal mechanism.
