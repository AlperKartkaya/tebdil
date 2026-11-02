# TEBDİL — research computation hold

The user explicitly suspended new research computation and experiments on
2026-11-02 until they understand and approve the end-to-end computational process.
Only an explicit user decision to resume computation lifts this hold; a generic
"continue" on documentation or code review does not.

- Do not load model checkpoints, execute model forwards/generation, run mini/toy
  model experiments, train controllers, benchmark, sweep parameters or start a backend.
- Do not run the full pytest suite or merely exclude `real_model`: unmarked tests
  also execute mini-model forwards. During the hold, the reviewed artifact/static
  release suite is `uv run pytest -q tests/test_site_release.py -m "not real_model"`.
- The sole grandfathered run is the already-running fixed-cap baseline follow-up
  frozen at `2309ea0`, PID 58098, output `.research/reasoning-budget-20261102`.
  Let that existing process finish and preserve its artifacts. Do not relaunch,
  resume, retry or extend it. This exception expires when the process exits.
- Allowed: source/documentation inspection, explanation, existing-file checksum
  and ledger audits, static frontend checks, and browser tests of recorded data
  with no inference backend. These do not authorize new scientific observations.
- Keep frozen scientific sources and historical results unchanged. Preserve the
  original page design. Never regenerate historical exports from dirty artifacts;
  use `scripts/build_landing_data.py --check` only.

Start with [the computation walkthrough](docs/computation_walkthrough.md) and
[the sequential research status](docs/research_execution_pass.md).
