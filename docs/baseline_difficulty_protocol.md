# Development baseline difficulty pilot

Prerequisite: measurement v2 completed and publicly verified (`b315d22`,
`2f9f96b`, `d9c440e`; CI 41274007747 passed). No subagents.

## Freeze before new observations

Evaluate **all twelve development questions** in the unchanged
`configs/completion_v1_tasks.json`, in ID order. Reuse the immutable native S0
completion for `completion_dev_000`; run only `completion_dev_001` through `011`
as eleven new native S0 completions. No outcome-dependent selection, retries or
early campaign stopping. Never execute the sixty reserved test questions.

Use the identical pinned checkpoint/tokenizer, BF16 MPS, four depths, forced
last-step readout, eager attention, cache-free native greedy generation, unchanged
chat prompt and 2048-token ceiling from gate 1. Stop at first EOS or budget only.
Generation receives only the question, not gold/metadata/candidates. No donor,
mixing, excursion, learned controller, teacher forcing or gold-based stopping.

Validate actual generated token IDs and full-prefix/call ledgers using the
existing native validator; apply `completed_final_integer_v2` offline. Preserve
v1 parsing fields separately on new records so version changes remain explicit.
Keep incorrect, malformed and unfinished attempts in the all-attempt denominator.
The reused S0's output/cost remains unchanged; only its separately named v2
assessment is new. It was already observed and is not an independent fresh run.

Independently observe every native forward pass: exact prefix hash/length, greedy
readout proposal and physical call count. After completion, require proposals to
equal actual generated IDs and all counters/ledgers to agree. Partial proposals
are diagnostic events, not finished answers. Remove hooks on success/failure.

## Evidence and cost

Create a new exclusive directory, per-question JSON and append-only journal. Hash
source/task/reference/checkpoint files before and after. Journal each forward pass
and every completed attempt. Exceptions preserve partial evidence and abort;
there is no silent restart, overwrite, favorable filtering or skipping failures.
Run serially; this can take hours on the 16 GB local machine. No additional GPU or
cloud resource is provisioned. Interruptions are infrastructure failures, not
automatically completed or incorrect task outcomes.

Report twelve questions, three existing development templates and all outcome
counts; do not count tokens as independent samples. Include tokens, real recurrent
calls, recurrent token positions and wall seconds per question, plus newly run
and reused costs separately. The forward/journal diagnostics add overhead: wall
time is instrumented pipeline runtime, not optimized inference throughput or FLOPs.

## Decision rule

This convenience development pilot diagnoses task headroom, not general accuracy,
statistical superiority, power or held-out performance. Report by template and
retain every raw response. A completed recognized incorrect integer provides a
candidate baseline failure, not proof an intervention can rescue it. Format errors
or budget exhaustion do not automatically establish wrong arithmetic.

- If all twelve recognized answers are correct, report **no observed rescue
  opportunity on this development pool**. Do not spend more on interventions or
  train a controller on this pool; prepare harder development tasks and an
  independently frozen corresponding test set instead. Keep existing test data
  untouched; never modify a test set after observing its outcomes.
- If recognized incorrect answers occur, inspect the raw failure types and freeze
  a paired controlled utility protocol before further model work. Do not evaluate
  only failures and then claim general improvement; measure degradations too.
- If formatting/truncation prevents interpretation, report headroom as unresolved
  and address the measurement/generation issue in a new protocol, not by repairing
  current scores mid-run.

No intervention pilot may begin until this baseline run, raw evidence, docs,
tests and public Pages verification are complete. Useful intervention evidence
is still required before any new controller evaluation.
