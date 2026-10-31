# Recorded-demo clarity pass

2026-10-31. Incremental changes on restored original page `77e9ecf`; not another redesign.

## Changed

- Both method selectors now share one handler and preserve recording, problem, seed, token position and recurrent step. Missing snapshots remain selected and explicitly unavailable; no clamping or invented default steps. Unavailable contexts withhold the diagram/PCA highlight and disable playback, not the recorded final answer.
- The original narrative box explains the exact selected outgoing action, recorded donor/strength, visited update norm and returned shift. Expandable details expose raw decision reason, non-probabilistic action score, token donor/visit compute, and captured state hashes. Controls and skipped interventions are not described as applied excursions. The whole-token schematic still draws only its first visit, separately from selected-step telemetry.
- Three existing cases are reachable directly beside the replay: recovery where mixing also succeeds, recovery where mixing fails, and degradation. The terminal states paired S0 correctness/cost and qualifies non-exclusive recovery.
- Corrected recording-dependent branch counts/strengths, general visited-state wording (not necessarily midpoint), and stale sample-size copy. Parallel-selection costs remain unverified; no efficiency advantage was added.

## Three-minute demonstration script

This is a rehearsed interaction sequence, not a timed human-usability study.

1. **0:00–1:00 — Recovery, not exclusivity.** Select “Recovery · mixing also works.” Controller recording, `arith_test_003`, seed 42, token 1, step 2: E3 MLP chooses donor C, α_eff=β=0.65; visited update norm 72.143; returned shift 42.232; measured residual 26.291 is a different quantity. Open decision/compute details: 11 token calls, including 6 donor and 1 visit. Final 48 is correct versus S0's 42. Whole-answer costs: E3 22, S0 8, M1 14, E1 16 calls. Mixing and fixed excursion also recover; this example is not compute savings or exclusive benefit. Change method through select/chips without losing token/step.
2. **1:00–2:00 — Distinct mechanism.** Select “Recovery · mixing fails”: phase3 `graph_validation_002`, seed 42. E1 yields 17 versus S0/M1's 13. Compare C0's visited-state anchor with E1's return to A. These legacy records have first-token snapshots but no token ledger; donor ID, hashes, returned-shift norm and total compute are not inferred. Open provenance and download the selected trace or manifest (two actions).
3. **2:00–3:00 — Negative evidence and limits.** Select “Degradation”: phase3 `arith_validation_003`, seed 42. S0 gives correct 40; E1/M1 give incorrect 36. Show the raw trace. Return to controller `arith_test_000`, step 4 on S0, then E3: its EXIT at step 2 means step 4 is unavailable, not silently replaced. Results remain exploratory; seeds are repeated measurements, not independent problems.

## Fresh local verification

- `uv run pytest -q -m 'not real_model'`: **73 passed, 1 deselected**. The context-preservation regression was observed failing before the fix.
- Export checksum/pairing check: all eight recordings pass. Scientific trace shards, manifests, evidence, results and stylesheet were not regenerated or changed.
- JavaScript syntax, TypeScript 5.9.3, scoped Ruff (`E4,E7,E9,F`) and first-party diff checks pass. Original inline CSS SHA-256, 12-section order and 55 original IDs remain regression-locked.
- Headless Chromium at `/tebdil/`: all eight datasets, exact raw selection, control semantics, step 5, token projection withholding, playback, filtering, heatmap, failure/retry/checksum rejection and keyboard skip link pass. Added checks exercise all three walkthroughs, downloaded JSON equality, recording/manifest URLs, select/chip context preservation, early-exit missing depth, missing-token deep link and keyboard method activation.
- Viewports 320/390/768/1024/1440: no document overflow. Desktop stage and mobile controls/narrative screenshots visually inspected. Zero page errors, failed normal requests or external runtime requests; MathJax renders without errors.
- Independent review found no remaining P0/P1 in this bounded pass after corrections.

## Boundaries

No new model inference or real-model integration rerun: this pass changes presentation/comparison logic only. Frontend verification does not establish real-model execution, benchmark independence, causal benefit or compute-matched superiority. Existing provenance and parallel-cost limitations remain. Seven pre-existing dirty research artifact/report files were left untouched and excluded from the commit.

No dependencies, CSS/theme changes, new pages, research campaign or controller features. Guidance used: installed Ponytail, accessibility and webapp-testing skills; historical reference/skill provenance remains in `reports/references_and_skills.md`. Production CI and deployed browser verification are reported separately after deployment.
