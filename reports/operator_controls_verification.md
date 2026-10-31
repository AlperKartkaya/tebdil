# Gate C: operator controls verification

Gate C is complete, committed and publicly deployed.

## What changed

- Corrected R0 descriptions and its public formula to **F(A + input noise)**.
  Historical R0 numerics, API, artifacts and scores are unchanged. Its pre-cast
  noise norm is beta times the clipped visit offset, not E1's output correction.
- Added an analysis-only seeded random output-displacement control with actual
  reference-minus-base target norm, finite/context/scope checks and isolated state.
  It refreshes history/gates but makes no recurrent calls. Reference cost is charged.
- Fixed M1 retaining the normal A gate after changing its hidden state. Regression
  reproduced the stale gate before repair. Last-step hidden outputs are unchanged.
- Locked affine equivalence with clipping, nonlinear residual decomposition,
  deterministic controls, scope/isolation and readout history with synthetic tests.
  Corrected effective-strength captions without changing page structure or styling.

## Real-checkpoint observation

[Raw report](ouro_operator_controls.json), [append-only journal](ouro_operator_controls.jsonl),
[frozen design](../docs/operator_controls_protocol.md). Checkpoint, input and source
SHA-256 identities were frozen before observations and are included in the report.
The same three pilot chat prefixes from gate B were used, not new held-out examples.

Apple MPS, pinned Ouro-1.4B-Thinking, FP32/BF16, full scope, depth four, intervention
at depth two, no cache, eager attention and last-step readout. Twelve dependent
conditions (three prefixes × two radii × two precisions), fourteen readouts each.
**306 actual recurrent calls**, independently reconciled with the native recurrent
boundary norm hook; 57.661 seconds. Construction/reference calls are diagnostic
overhead, not per-method compute-matched costs. All formula, donor isolation,
input-state isolation and no-fallback checks passed.

| Check | FP32, six conditions | BF16, six conditions |
|---|---:|---:|
| Pre-cast norm of E1 − M1 − beta C | 0.0000116–0.0000156 | 0.0000113–0.0000154 |
| Same error after casting outputs | 0.0000116–0.0000156 | 0.597–0.900 |
| Visit casting offset from endpoint interpolation | 0.000000829–0.00000395 | 0.449–0.621 |
| Maximum realized random-output norm relative error | 0.0000113% | 0.8441% |

These are Frobenius norms over all token/hidden coordinates, not per-token errors.
They demonstrate why the exact-arithmetic identity must not be claimed as literal
BF16 equality. Noise targets use actual cast reference displacements; matching is
FP32 before casting, with realized error retained for every control.

## No efficacy conclusion

The residual is nonzero, but its first-answer-token log-probability effect relative
to M1 has mixed signs and is not consistently above the three matched random
residual directions. This observable is **not answer correctness**: the prompts are
chat-formatted and the model can produce reasoning before an answer. No generation,
rescue rate, p-value, held-out accuracy, learned-controller comparison or statistical
support is inferred. Scale sweeps {0, 0.5, 1, 2} and all random directions remain in
the raw report; no favorable subset was selected for publication.

The design was fixed before these observations, but it is an exploratory follow-up
to previous pilot work, not a preregistered confirmatory study. Real normalization
and precision effects now warrant the next ordered gate. Controller evaluation
still requires a separately demonstrated useful intervention and frozen evaluation.

## Release verification

- Fresh `uv run pytest -q`: **135 passed**, including real Ouro integration.
- Export checksums/pairing, JS syntax, TypeScript and `git diff --check` pass.
- Targeted Ruff E4/E7/E9/F passes, excluding the pre-existing unused `Optional`
  import in `controls.py` (F401); no new lint violation is suppressed.
- Full local browser pass: eight runs, three demo cases, keyboard, download,
  retry/checksum failures, MathJax; widths 320/390/768/1024/1440. No page errors,
  failed normal requests, external runtime requests or body overflow. Expanded
  evidence screenshot inspected. Original CSS/IDs/order regression tests pass.
- Manifest comparison proves append-only evidence: every old run, metric and file
  identity is unchanged. Seven pre-existing dirty research artifacts are untouched.
- The frozen protocol's initial status reflects its pre-run snapshot; this record
  carries current completion status.
- Commit `65cd09b`: [CI and Pages deployment passed](https://github.com/AlperKartkaya/tebdil/actions/runs/41241237400).
- Full public browser pass repeats the local matrix, including new evidence retrieval
  and corrected R0 replay. Public index, app, manifest and operator-evidence SHA-256
  match the committed files. No browser errors or normal-request failures.

Only after this deployment and verification was gate C closed. Gate D may now begin;
no controller-utility or publication gate is implicitly cleared.
