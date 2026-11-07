# Pipeline review — 2026-11-07

## Aim and boundaries

TEBDIL tests whether scoped latent visit/return updates improve recurrent reasoning without replacing the receiver with a donor. Recovery, accuracy and efficiency are different endpoints. This is not retrieval augmentation, rationale imitation, or evidence that every change in output is an improvement.

Current execution entry point: `scripts/run_colab_l4.py`. Historical protocol-named files remain evidence dependencies; deleting or renaming them merely to remove version suffixes would break reproducibility. Do not run historical controllers as if they were the current protocol.

## Verified fixes

- Pages failure run `42370779614` reports a missing `site/static/evidence/real_tensor_bundle.pt`. The local file passes manifest checks but was excluded by `*.pt`. A narrow Git exception includes the exact existing evidence, without weakening checks or regenerating results.
- The supplied bridge used a shared fixed credential, a fixed port, broad `pkill`, and advertised a URL before testing service readiness. The replacement uses private random credentials, OS-selected ports, owned process handles, bounded startup attempts and authenticated public health.
- Stall comparison used `zip`, silently ignoring an unmatched continuation suffix. Comparisons now include unequal-length suffixes. This metric remains positional token difference, not edit distance or correctness.
- Runner model and tokenizer were hard-coded. Both now receive the same explicit model ID and immutable revision; run summaries record model, revision and seed.
- README incorrectly said computation remained prohibited. It now follows the November 5 resumption instruction.

## Method issues requiring further work

1. The cached scheduler commits receiver depth states before P0 selects a potentially different output branch. Selected-donor output/cache consistency needs a dedicated real-cache regression test before P0 is used as a persistent comparison. The follow-up now commits the selected branch history; a forced-donor regression fails before the fix and passes afterward.
2. Cached generation has implicit thinking/window/cooldown states and length-dependent hard-coded cooldowns. The follow-up extracts these into `ExcursionWindowState`, preserving existing cooldown values and fixed-method bypass semantics, with transition tests.
3. Stall checkpoints are hand-selected development prefixes. Results are diagnostic, not unbiased estimates of intervention utility. Freeze trigger selection without labels before reserved testing.
4. Locally adapted literature examples are not official benchmark scores. Some source licenses remain unresolved. The dataset registry exposes this rather than guessing permissions.
5. The runner now checkpoints each completed generation atomically across suites. Keys bind code hashes, hardware, model revision, prompt IDs and generation settings. Corrupt records fail closed; failed generations are never cached. Use mounted persistent storage for `--checkpoint-dir` and a fresh summary path on resume. Prior generation costs remain attached to reused records, marked `checkpoint_reused`; they are not new execution costs.

## Multi-model evaluation

The upstream [Ouro release](https://ouro-llm.github.io/) includes 1.4B and 2.6B Thinking models. The runner accepts `--model ByteDance/Ouro-2.6B-Thinking --revision <full-commit-sha>`. Both checkpoints have since passed the bounded FP32 L4 parity gates described below. This does not establish long-generation accuracy or intervention benefit. Use one worker and measure peak memory for each task budget.

Freeze identical task IDs, prompts, stopping/parser rules and paired seeds. Report parameters, recurrent calls, tokens, wall time and peak memory. Use model-specific colors only after verified results exist; include labels/markers so color is not the sole identifier. Do not manufacture comparative graph points from advertised model sizes or unrelated upstream scores.

## Initial review boundary (before Colab follow-up)

The baseline suite passed 340 tests with four real-model tests excluded. Public tunnel startup, Google runtime recovery, 2.6B parity and GPU outcomes were not tested in this review. No Colab connection was supplied. Site styling and historical scores remain unchanged.

## Follow-up execution evidence

Authenticated connection to the replacement bridge succeeded. An isolated Colab checkout passed six targeted tests. A real Ouro-1.4B L4 smoke generated two tokens each for S0, P0 and E2; all stopped on budget, E2 made no excursion. Peak allocated CUDA memory was 2,889,218,048 bytes. See `reports/pipeline_gpu_smoke.json`. This is not accuracy, native parity, donor-win parity, or 2.6B validation. The existing runtime report was preserved locally; the original checkout was not modified.

## Precision and recovery pass

See [cached parity findings](../reports/cached_parity_findings.md) for eight real L4 conditions across 1.4B and 2.6B. FP32 passes the bounded gate; BF16 does not. The runner now defaults to FP32 and exposes BF16 explicitly rather than silently treating these regimes as interchangeable.

Checkpoint publication now renames a completed temporary directory instead of relying on hard links, which mounted Drive may not support. Records remain checksum-checked, and source identity includes vendored model code and Transformers/CUDA versions. Earlier cache layouts remain untouched and are not reused under the changed source identity. Filesystem publication does not guarantee remote-drive durability: retain downloaded copies, and test the specific mount before long runs.

The unused literature-only partial-summary writer was removed; all suites use the same generation checkpoint path. Invalid budgets, duplicate methods, unknown selected task IDs, empty/batched prompts and unsupported cached depth/readout settings now fail explicitly. Missing comparison baselines produce null metrics, not fabricated zero differences. CI now runs the whole non-real-model test suite.

Publisher license declarations are pinned in `docs/dataset_license_metadata.json`. Apple's GSM-Symbolic card declares CC-BY-NC-ND-4.0; distinguish local inspiration from copied/modified data before using or redistributing more items. Unresolved source rights remain admission gates.

A fresh-process checkpoint test verifies reuse without recomputation. Interrupted temporary publication and corrupt payloads are also tested. The live Colab checkpoint-restart trial could not start after the tunnel hostname stopped resolving; Google runtime replacement and mounted-Drive durability remain unverified. All eight parity records were downloaded before disconnection.

No-op excursions (`alpha=0` or `beta=0`) previously inflated `excursion_count` despite matching baseline tokens. The counter now excludes identity/unstable attempts, and these cases have multi-token cache regressions. Cached generation accepts only the currently supported baseline/control/excursion methods; historical scheduler names are not silently interpreted as fixed excursions.

## Fresh-runtime follow-up

A replacement Colab VM was observed with the old `/content` checkout absent. Both models passed FP32/SDPA checks at 128/512-token prefixes. A real four-token generation checkpoint was exported locally, uploaded into a new directory, and reused by a fresh process with identical records. See [recovery findings](../reports/runtime_recovery_findings.md). Mounted-Drive durability and automatic VM restart remain outside the verified claim.
