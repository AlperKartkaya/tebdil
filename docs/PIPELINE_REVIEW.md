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

The upstream [Ouro release](https://ouro-llm.github.io/) includes 1.4B and 2.6B Thinking models. The runner accepts `--model ByteDance/Ouro-2.6B-Thinking --revision <full-commit-sha>`. This is plumbing, not validated model compatibility or an established L4 memory fit. Run native/cached parity and one short smoke first. Use one worker initially and measure peak memory.

Freeze identical task IDs, prompts, stopping/parser rules and paired seeds. Report parameters, recurrent calls, tokens, wall time and peak memory. Use model-specific colors only after verified results exist; include labels/markers so color is not the sole identifier. Do not manufacture comparative graph points from advertised model sizes or unrelated upstream scores.

## Validation boundary

The baseline suite passed 340 tests with four real-model tests excluded. Public tunnel startup, Google runtime recovery, 2.6B parity and GPU outcomes were not tested in this review. No Colab connection was supplied. Site styling and historical scores remain unchanged.

## Follow-up execution evidence

Authenticated connection to the replacement bridge succeeded. An isolated Colab checkout passed six targeted tests. A real Ouro-1.4B L4 smoke generated two tokens each for S0, P0 and E2; all stopped on budget, E2 made no excursion. Peak allocated CUDA memory was 2,889,218,048 bytes. See `reports/pipeline_gpu_smoke.json`. This is not accuracy, native parity, donor-win parity, or 2.6B validation. The existing runtime report was preserved locally; the original checkout was not modified.
