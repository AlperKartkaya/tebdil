# Passive reasoning recorder verification

Status: **implementation verified; frozen two-case recording and public evidence
not complete**. No intervention or controller experiment has begun.

[Protocol](../docs/passive_reasoning_protocol.md) fixes the first level-0 development
CNF/planning pair before observations. True-mode native generation stays unchanged;
actual last-position input/raw/normalized states are aligned within each prefix.
Exact-token repetition is recorded as a heuristic candidate flag, not a semantic
stall label or progress score. No task/verifier/gold enters the observer.

Before observations: **312 full tests passed**, including five new observer/CLI
checks, existing real Ouro integration and an additional actual BF16 checkpoint
passive/unobserved three-token parity test (12 recurrent calls in each arm).
Mini-native parity also verifies unchanged weights; actual tensor snapshots are
reconstructed and hashed. Nonfinite inputs and failed journal callbacks clean up
all hooks. This proves bounded recorder behavior, not complete-answer parity,
model difficulty or intervention utility.

Scoped Ruff, JS syntax, TypeScript, frozen-task regeneration, static export checks
and diff checks passed. Full local browser matrix passed on the preserved original
page: eight recordings, three demonstrations, five widths, keyboard and retry/
checksum controls, no browser errors or failed/external requests.

The recorder must now be committed before running the actual pair. Preserve all
outputs, source/checkpoint hashes, per-token state diagnostics and append-only
journals. Do not start controlled continuations before actual evidence audit,
process review and public release verification. No semantic stall classifier is
validated by these implementation tests.
