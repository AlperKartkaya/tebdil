# Cached-generation parity gate — 2026-11-07

## Design

Two pinned Ouro Thinking checkpoints, each tested on NVIDIA L4 across BF16/FP32 and eager/SDPA attention. Each condition uses token-ID prefixes `[1]` and `[1,2,3,4,5]`, followed by three teacher-forced native argmax steps each. Every cached prediction is compared with a fresh full-prefix native final-depth prediction on the same prefix. This is six diagnostic steps, not six reasoning tasks.

The gate requires finite logits, equal cache lengths, exact top-1 agreement, and maximum absolute logit error no greater than 0.25 (BF16) or 0.0001 (FP32). Thresholds were set before the corresponding runs; failed records remain failed.

| Model | Precision | Attention | Maximum logit error | Top-1 agreement | Gate | Raw record |
|---|---|---|---:|---:|---|---|
| Ouro-1.4B-Thinking | bfloat16 | sdpa | 0.296875 | 6/6 | FAIL | [cached_parity_14b.json](cached_parity_14b.json) |
| Ouro-1.4B-Thinking | bfloat16 | eager | 0.359375 | 6/6 | FAIL | [cached_parity_14b_bfloat16_eager.json](cached_parity_14b_bfloat16_eager.json) |
| Ouro-1.4B-Thinking | float32 | sdpa | 2.0980835e-05 | 6/6 | PASS | [cached_parity_14b_float32_sdpa.json](cached_parity_14b_float32_sdpa.json) |
| Ouro-1.4B-Thinking | float32 | eager | 1.7166138e-05 | 6/6 | PASS | [cached_parity_14b_fp32.json](cached_parity_14b_fp32.json) |
| Ouro-2.6B-Thinking | bfloat16 | sdpa | 0.390625 | 5/6 | FAIL | [cached_parity_26b.json](cached_parity_26b.json) |
| Ouro-2.6B-Thinking | bfloat16 | eager | 0.3125 | 6/6 | FAIL | [cached_parity_26b_bfloat16_eager.json](cached_parity_26b_bfloat16_eager.json) |
| Ouro-2.6B-Thinking | float32 | sdpa | 1.7404556e-05 | 6/6 | PASS | [cached_parity_26b_float32_sdpa.json](cached_parity_26b_float32_sdpa.json) |
| Ouro-2.6B-Thinking | float32 | eager | 1.6212463e-05 | 6/6 | PASS | [cached_parity_26b_fp32.json](cached_parity_26b_fp32.json) |

## Interpretation and next-run contract

Both models fit the L4 for these short prefixes. Peak FP32 allocations were about 5.76 GB (1.4B) and 10.71 GB (2.6B); these are not long-context memory guarantees. FP32 passes with both attention implementations. BF16 fails the numerical gate with both implementations; 2.6B SDPA also changes one top-1 token.

The precision dependence is evidence consistent with numerical drift, not a proof that all cache/intervention paths are correct. Use `--dtype float32 --attn sdpa` for the next controlled development study. Do not relabel historical BF16 results as parity-validated. Keep cached S0 as the paired baseline for cached interventions; do not mix full-prefix and cached baselines unnoticed.

No task-accuracy gain, long-generation equivalence, or cross-model intervention benefit follows. P0 donor-cache storage is covered separately by a real miniature-cache regression; these GPU gates exercise S0 only.

Reproduce with `scripts/check_cached_parity.py`, supplying the model and full revision from each raw record plus `--dtype`, `--attention` and a new `--output`. Every record resolves to its exact hash-named script snapshot in `parity_sources/`; the current CLI adds precision and attention selection to the initial BF16/SDPA diagnostic. Short-prefix synthetic diagnostics must be supplemented with chat-format and longer-prefix checks before a paper claim.
