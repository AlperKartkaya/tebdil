# Actual original-page restoration

2026-10-31. Baseline: `fd074b8:site/index.html`, not the rejected five-panel replacement or its later recoloring.

## Restored and preserved

- Original 12 sections in order, all 55 original IDs, floating contents rail, publication hero, four metric cards, four-column fork/visit/return stage, adjacent PCA/telemetry, dark output terminal, research tables, parameter grid, three case cards, explorer, and citation.
- Exact original inline CSS, SHA-256 locked in `tests/test_site_release.py`. Separate small stylesheet repairs focus, responsive overflow, and loading states; it does not reskin the page.
- Original fonts, Bulma, icon family, and MathJax are local, licensed assets. Sources/hashes: `reports/original_asset_sources.md`.

## Corrections retained on that page

- Eight real recording sets, checksum-verified lazy loading, token and exact recurrent-step selectors, meaningful method/case/explorer controls, downloads and execution manifests.
- No invented visit coordinates or borrowing another method/token's PCA. The four-column mechanism is labeled schematic; later depths/multiple visits are inspected in telemetry rather than fabricated on the template.
- Control-specific explanations distinguish direct mixing, no-return switching, noise, independent selection, and excursion-and-return. Missing telemetry stays unavailable. Decisions are not borrowed from another step.
- Costs fail closed for parallel-selection reference budgets and mismatched ledgers. Tables show paired rescue/degradation and problem-cluster uncertainty.
- Normalized legacy heuristic decision fields without borrowing steps, distinguished ungated learned policies, and clear stale run summaries on failure.
- Replaced stale hand-entered accuracy/call claims with manifest-backed summaries; corrected the visit-offset clipping description and actual Ouro normalization names. Corrected case prompts/targets and acknowledged oracle/mixing recoveries and degradation.
- The grid and scope evidence added here exactly match those artifacts embedded in the old `fd074b8` bundle. Existing exported trace shards and results were not regenerated from the unrelated dirty research files.

## Fresh verification

- `uv run pytest -q -m 'not real_model'`: **70 passed, 1 deselected**.
- `uv run python scripts/build_landing_data.py --check`: eight runs verified, including hashes and pairing.
- `node --check site/static/app.js`, `node --check scripts/verify_site_browser.cjs`, `tsc -p tsconfig.json`: pass.
- `uvx ruff check scripts/build_landing_data.py tests/test_site_release.py --select E4,E7,E9,F`: pass. `git diff --check` on first-party release files: pass. The verbatim upstream Inter OFL license retains one trailing space; its recorded hash is preserved.
- Headless Chromium browser verification at `/tebdil/`: all eight recording selections (40–972 rows), matching selected raw traces, method controls, step 5, later-token projection withholding, playback, heatmap, scope table, case-card navigation, family filtering, skip-link keyboard focus, failed-load retry, checksum rejection, and stale-output hiding: pass.
- Viewports 320, 390, 768, 1024, 1440 px: no document overflow. Diagrams, formulas, and tables may scroll within their own containers. Screenshots inspected for hero, original stage, research/results, and dark case section, compared with locally served `fd074b8`.
- Browser log: zero page errors, zero failed normal asset requests, zero external runtime requests; over 100 rendered math containers, zero MathJax error elements.

Re-run browser checks with an existing Playwright installation (no site dependency added):

```sh
# Serve this checkout under the intended Pages subpath first.
SITE_URL=http://127.0.0.1:8096/tebdil/ \
TEBDIL_PLAYWRIGHT_MODULE="$(npm root -g)/agent-browser/node_modules/playwright-core" \
node scripts/verify_site_browser.cjs
```

The module override is optional when `playwright` is already available to Node. Browser evidence is written to `/tmp/tebdil-original-restoration-browser.json`; screenshots use `/tmp/tebdil-final-*.png`.

## Evidence boundaries

This is a presentation/runtime restoration, not a new model experiment. Real-model integration was **not rerun** in this pass; frontend/browser checks do not prove model execution, benchmark independence, or a general reasoning benefit. Recorded data retain their existing provenance limitations. No live GPU backend is claimed. Larger independent, compute-matched evaluation remains research work.

Pre-existing modified controller/counterfactual/audit/final-results artifacts were left untouched and excluded from this release.
