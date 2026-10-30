# MVP Scientific Verification

Verification date: 2026-10-30 (Europe/Istanbul)
Source revision inspected: `fd074b80bf0943ea14d73d9a09b8a0dc012bb18c`

## Fresh checks performed

### Real Ouro replay

Command:

```bash
.venv/bin/python -m tebdil.audit \
  --run-id phase7_8_fresh_test \
  --replay-problem-id arith_fresh_test_001 \
  --require-real-replay
```

The replay used the locally cached `ByteDance/Ouro-1.4B-Thinking` checkpoint at pinned revision `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d` on Apple MPS. CUDA was unavailable. The complete 2.87 GB `model.safetensors` file matched SHA-256 `6730581fd366f6d1689728c61637520228385cdfc17c0b24f19aa840b00565fb`; the configuration, tokenizer, and vendored model-code hashes also matched the provenance record.

All seven selected real-model replay cases matched their recorded answer, gold log-probability, recurrent-call count, and action type. The cases covered S0/CONTINUE, E1/EXCURSION, EXIT_2/EXIT, E3_mlp/EXCURSION, E3_ridge/EXCURSION, a natural learned-policy CONTINUE, and a natural learned-policy EXIT.

The final replay was repeated **after** the fail-closed patch. The [current audit JSON](mvp_audit_fresh_test.json) records `real_model_replay_verified=true`, `unit_checks_passed=true`, `artifact_consistency_passed=false`, and `all_passed=false`. Real replay passes without hiding the historical cost-provenance failures. It reports 901/972 arithmetic-consistent ledgers and flags every historical P0/P1 record for incomplete total-selection provenance.

### Fail-closed artifact audit

Command after the patch:

```bash
.venv/bin/python -m tebdil.audit --run-id phase7_8_fresh_test
```

Expected result: exit code `1`, `all_passed=false`. All 972 records had at least two recurrent calls, but only 901/972 historical token ledgers matched their displayed aggregate call count. The 71 mismatches are historical P0/P1 derived-selection rows. The historical artifacts were not changed or regenerated. Future P0/P1 records preserve selected-output provenance and list every consulted A/B/C source. Their B-run ledger is labeled only as an idealized reference parallel budget; it is not accepted as verified executed selection cost.

### Tests

```text
.venv/bin/pytest -q
68 passed
```

The integrated suite passes, including the real-Ouro integration and static release contracts. The separate historical artifact audit still fails as disclosed above; unit tests passing does not imply all recorded ledgers are valid.

## Scientific status

- **Model execution:** verified against the pinned real Ouro checkpoint, not a toy adapter alone.
- **Excursion semantics:** the implementation computes a recurrent transition at the visited state, derives its update, and applies the scoped returned update to receiving trajectory A. Identity/no-op and branch-isolation tests pass.
- **Recorded hidden states:** the audited run contains nonzero real hidden-state snapshots; replayed records match fresh forward execution.
- **Controller classification:** E3 uses a learned action scorer with hand-written feasibility gates when `use_heuristic_gates=true`. It is therefore a hybrid learned scorer, not a fully learned end-to-end policy.
- **Comparison scope:** non-parallel methods retain measured call ledgers. Historical and future P0/P1 selection costs remain unverified; no comparison should be described as strictly compute-matched on this basis.
- **Projection integrity:** future E0 random-direction records no longer publish an invented donor-interpolation `VISIT` coordinate. Historical projections were not recomputed.

## Remaining limits

- No full experiment campaign was rerun. Existing accuracy, rescue, and uncertainty estimates remain historical evidence.
- Historical P0/P1 aggregate compute costs cannot be repaired from their selected-output token ledgers alone. Future records expose an idealized reference budget and consulted source identities, but the audit will not verify executed selection cost until every consulted source is accounted for.
- Existing curvature telemetry measures the full nonlinear curvature before token-scope or coordinate projection. It must not be labeled as the applied intervention norm.
- Apple MPS was available; CUDA-specific behavior was not tested.

Parallel selection caveat: all historical P0/P1 total-selection costs are unverified, including rows whose selected-output ledger sums happen to match. The 71/81 counts describe arithmetic mismatches, not the full scope of uncertain parallel execution.
