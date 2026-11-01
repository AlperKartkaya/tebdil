# Reasoning task/verifier gate

Status: **complete: implementation, local checks and public release verified**.
No new model observations. All six development and twelve reserved reasoning
instances remain unexecuted; the old arithmetic observations/reserved set stay intact.

The [frozen protocol](../docs/reasoning_tasks_protocol.md) defines structural
selection, exact prompts/partitions, True-mode generation/stopping, final-only JSON
grammar, alternate valid answers and separate outcome classes. Its public download
is a task specification, not a recording or model result. No intervention or stall
monitor is implemented by this gate.

- **37 focused tests**: exhaustive Boolean cross-checks over all assignments in all
  18 instances; any satisfying assignment/optimal route; key persistence and
  revisits; alternative shortest paths; format errors, unfinished/malformed CoT;
  deterministic pool/partition/prompt drift; no overwrite; dev-only public payload.
- **307 full tests passed**, including the existing actual cached Ouro integration
  and prior real evidence audits. This is not new execution on the reasoning pool.
- Scoped Ruff, JS syntax, TypeScript, frozen-task byte regeneration, export
  checksums and diff checks pass. CI now runs task-byte regeneration explicitly.
- Full local browser matrix passes: all eight historical recordings, three recorded
  demonstrations, five desktop/mobile widths, keyboard and failure/retry/checksum
  controls. No browser errors, failed or external requests.
- Original inline CSS, sections and historical runs/evidence metadata are unchanged.
  The only manifest addition is the separately checksummed development-prompt file.
- XERJ's old project search was marked stale; direct source inspection supplied
  current facts. Refreshed task-module index: **8/8 Python files**, 47 records,
  no skipped files. Its data map and new solver/verifier search were inspected.
  This limited refresh is not claimed to cover the entire repository.

Limits: shared generators/wording, small instance-held-out pool, no isomorphism/OOD
certification, unknown model difficulty, no monitor calibration, efficacy, uncertainty
estimate or controller result. Classical solver success is not model reasoning.

Release `3a989b6`: [CI and Pages deployment passed](https://github.com/AlperKartkaya/tebdil/actions/runs/41329465971).
The full public browser matrix passed, including the new development-only download,
with no browser errors or failed/external requests. Deployed HTML, app script,
manifest and prompt JSON match repository bytes. Protocol, verification and full-pool
GitHub links returned HTTP 200. Point 2 is closed; passive recording is next, not
completed by these checks.
