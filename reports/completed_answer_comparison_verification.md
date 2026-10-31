# Gate 2: fixed-method completion comparison

Status: implementation validated; real comparison pending. Gate 1 was completed and publicly
verified first. **No new comparison result is available yet. Gate 3 is not started.**

[Frozen protocol](../docs/completed_answer_comparison_protocol.md).
One already-correct development question is reserved for technical validation of
B_only, M0, M1, E0 and E1 against the recorded native S0 output. This is not an
efficacy pilot. No held-out questions or controllers are executed.

The new decoder reuses the existing scheduler/operators, shared chat prompt and
final-answer parser. It accepts no gold argument, performs no teacher-forced
scoring, and releases states/full-vocabulary logits between generated tokens.
Physical hooks must agree with scheduler and adapter counters. Every token is
journaled; exclusive per-method outputs preserve unfinished and failed evidence.

B_only emits the donor branch without transfer and charges both branches; it is
not optimized best-of-N sampling. E0 randomizes excursion direction at matched
input distance, not output norm. E0/E1 each cost eight calls at the same prefix;
different generated lengths still mean different whole-answer costs. The recorded
native S0 run is referenced with its real cost, not counted as free or rerun.

Ten tests pass for method call counts, seeds, prefix hashes, EOS/budget handling,
gold-free generation, failure-hook cleanup and refusing to overwrite partial
evidence. The decoder tests use synthetic mini-model
fixtures, **not verification of the new methods on the actual checkpoint**.

Validation before observation: **177 tests passed**, including the cached real
Ouro integration checks for the existing adapter. Scoped Ruff, JavaScript syntax,
TypeScript and export checks passed. The local browser matrix passed all eight
recordings, three demos, keyboard/retry/download checks and 320–1440 px layouts,
with no console/network failures. This does not verify new real completions.

After real evidence exists, verify the raw outputs, journal reconciliation, tests,
documentation and Pages before closing this gate or starting the utility pilot.
