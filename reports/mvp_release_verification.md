# MVP release verification — 2026-10-30

## Delivered

A single static five-step observatory replaces the repeated landing/dashboard. Shared run, problem/seed, method, generated-token and recurrent-step selection links the controller, paired outputs and evidence. Method-specific schematics distinguish state mixing, update mixing, visit-without-return, noise, ablations and genuine excursion-and-return. No recorded replay is labeled live inference.

The exporter preserves 2,866 original rows across eight runs, recomputes paired summaries and emits checksum-bound lazy recordings plus downloadable provenance. The startup manifest is 48,801 bytes instead of the former 34,364,207-byte bundle. No frontend framework or runtime dependency was added. Historical results were not modified to improve their appearance.

## Fresh verification

- `pytest -q`: **68 passed**, including the actual cached pinned Ouro checkpoint integration, native-forward equivalence, identity cases, branch isolation and real excursion controls.
- Current-code real audit replay: **7/7 cases match on Apple MPS**. [Raw audit](mvp_audit_fresh_test.json) and [scientific scope](mvp_scientific_verification.md).
- Historical artifact audit: **expected exit 1**, not a release-green research certificate. Its real replay passes; historical parallel total-selection costs remain unverified, with 71 arithmetic mismatches among 972 fresh records.
- `python3 scripts/build_landing_data.py --check`: eight exported runs pass checksum/pairing/count/disclosure checks without importing the research environment.
- A fresh `git archive` of the committed project, without ignored `results/`, passes export checks and **67 non-real-model tests**, with one real-model test deliberately deselected.
- `tsc -p tsconfig.json`: passes using TypeScript 5.9.3; typed manifest/trace and DOM contracts, no emitted build dependency.
- `node --check site/static/app.js` and Python `compileall`: pass.
- Changed Python files pass Ruff's correctness rules (`--isolated --select E4,E7,E9,F`). The inherited broader modernization/style policy was inspected but is not claimed clean; unrelated typing/style modernization was not performed.
- Relevant README/methodology/findings/report links resolve to local files. Repository source links and exported evidence URLs are regression-tested.

## Browser and visual checks

Real browser at `/tebdil/` subpath, desktop and 390px mobile:

- All five tabs, next/previous navigation and three mechanism comparisons work.
- Run/case/method/token/step selections change actual recorded data. Evidence method selection works before downloading a trace; table selection opens its corresponding experiment.
- CONTINUE/EXCURSION/EXIT readouts are token/step-specific; fixed/heuristic/learned-hybrid labels are distinct.
- M0/M1/C0/R0 do not claim excursion-return or display placeholder zero residuals as measurements. Missing per-token data has an explicit disabled fallback label.
- Forced fetch failure exposes retry and recovers. With the manifest blocked, conceptual navigation/diagrams still work, while evidence explicitly remains unavailable.
- Keyboard Arrow/Home/End tab navigation, visible focus, reduced-motion styles and local links checked. No page overflow at 390px or fresh application console errors in successful flows.
- Desktop problem/mechanism/decision/experiment/evidence and mobile experiment screenshots inspected. Palette contrast checks and limitations are in [references and skills](references_and_skills.md).

## Boundaries

No push or production deployment occurred. The Pages workflow was locally checked, not run remotely; repository Pages source/environment settings still govern deployment. No CUDA test, full replacement experiment campaign, independent replication, screen-reader certification or full WCAG audit was performed. Small exploratory generated-task results do not establish a general TEBDIL accuracy advantage.

Pre-existing modified research artifacts and `reports/final_results.json` were preserved outside these commits. Public copies are explicit checksum-bound snapshots, not a claim that every historical source generation agrees. Rebuilding from raw results still requires the local research inputs; a clean clone serves and verifies the committed export without them.
