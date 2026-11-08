# E2 direction comparison — original 16-task cohort

Approved change: replace E2's weighted mixture of top competing output-head rows
with the single highest-logit competitor. Keep its existing 85% tangent / 15% raw
direction treatment and relative perturbation norm 0.14. No new geometric detector,
normalization/retraction rule, acceptance test, controller or candidate lookahead.
Exact ties use the lowest token ID. Missing output heads fail closed for the new
option; they do not invoke random perturbations.

## Frozen comparison

`configs/e2_direction_original16.json` binds all 16 task IDs, prompt/answer hashes,
the original report hash and budget. These are precisely the tasks behind the
stored S0 11/16 versus E2 12/16 result at `bfe4f046...`, not the later 38-task set
or a cherry-picked rescue subset. All selected prompts and answers were compared
to that commit and are unchanged.

- Arms: S0, E2 with its existing mixture, E2 with the single competitor (48 runs).
- Same pinned Ouro-1.4B-Thinking revision, FP32/SDPA, L4, seed 42, 3,000 output tokens.
- E2 alpha/beta 0.65, visit step 2, last-token scope, pre-norm updates, trust bound
  0.35, excursion window 1; unchanged policy thresholds and cooldown behavior.
- One worker, interleaved arms per task, complete-generation checkpoints downloaded
  as they finish. No early stopping based on outcomes or budget extension.
- Same current scorer across all new arms. EOS completion required; budget
  exhaustion is unsuccessful. Report rescues AND regressions and generated tokens,
  recurrent calls and recorded wall time. Recurrent calls are not total FLOPs.

The adaptive timing **policy** stays fixed, not necessarily the realized attempt
positions: changing a donor can change later outputs and decisions. This measures
the end-to-end effect of donor selection under E2, not a strict fixed-schedule
mechanistic ablation. Synthetic fixed-input checks isolate direction construction.

Historical runs used BF16 and weaker provenance; the scorer has also been repaired
since then. The original scores remain historical reference, not interchangeable
FP32 controls. The whole cohort is already development data: no held-out or broad
model-portability claim follows even if the new option wins here.

## Execution

Use the existing runner; both E2 arms remain method `E2` and are distinguished by
`perturbation_method` in rows and checkpoint contracts, not new method names:

```sh
python scripts/run_colab_l4.py --suite deceptive --methods S0 E2 \
  --e2-directions mixture single --max-new-tokens 3000 \
  --dtype float32 --attn sdpa --seed 42 \
  --task-ids <all task_ids from configs/e2_direction_original16.json> \
  --checkpoint-dir <persistent-directory> --output <new-summary.json>
```

No tuning or winner selection until all 16 task triples are available. Report paired
counts and uncertainty cautiously: these deliberately selected, previously viewed
problems are not a random sample of reasoning tasks.
