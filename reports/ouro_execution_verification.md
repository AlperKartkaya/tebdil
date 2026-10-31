# Point A — Ouro execution-contract verification

2026-10-31. Sequential research pass; no task campaign or historical artifact was regenerated.

## Changes

- Added explicit position IDs and rejected unsupported cache/depth contexts.
- Fixed `native_gate` to respect the native configured forced exit step.
- Fixed `validate-adapter` to compare the configured readout, rather than always
  using native forced-final logits.
- Added reusable per-depth parity validation and an exclusive-output real-model
  verification script with checkpoint/source SHA-256 identities.
- Added a separate expandable Pages evidence entry without changing the original
  CSS, section order, charts, historical metrics or raw experiment shards. This
  independent execution evidence remains available if trace loading fails.

## Evidence

- `uv run pytest -q`: **93 passed**, including the real-checkpoint integration.
- Real verification script: **all_passed=true** on Apple MPS; FP32 and BF16,
  three short fixtures per dtype, four depths each. Hidden states, exit gates,
  forced logits and native-gate logits matched exactly within each dtype.
  Thresholds 1.0 and 0.5 cover 48 native-gate comparisons.
- The script rehashed every checkpoint file listed in committed model provenance;
  all matched. Cross-precision differences are recorded rather than hidden.
- Export hashes/pairing, JavaScript syntax, TypeScript and scoped Ruff
  (`E4,E7,E9,F`, excluding the pre-existing vendored-import `E402` boundary) pass.
- Chromium at `/tebdil/`: all eight datasets, three demo cases, downloads,
  checksum/failure/retry behavior and keyboard controls pass. New execution
  evidence expands by keyboard and its real JSON loads. No page errors,
  external runtime requests, failed normal requests or document overflow at
  widths 320/390/768/1024/1440. Desktop stage screenshot inspected.
- XERJ: the previous index could not reconcile its durable generation. A separate
  `tebdilcontract` index was built successfully: 153 canonical files, all 58 code
  files indexed. Vendored sources are excluded by its policy and were inspected
  directly. No old index was deleted or treated as current.

The new [JSON record](ouro_execution_contract.json) is also exported unchanged in
Pages. CI checks its file identities and export checksums; CI does not download or
execute the real checkpoint.

## Boundaries

This is an execution-fidelity gate on short fixed prefixes, not an efficacy,
long-context, cached-generation, CUDA or precision-robustness claim. The existing
seven modified research artifacts were left untouched and were not republished.
Historical cost-ledger failures remain disclosed. Later operator and task studies
must supply separate evidence.
