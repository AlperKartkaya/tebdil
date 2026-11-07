# Fresh-runtime recovery and extended parity — 2026-11-07

## Observed runtime replacement

The replacement bridge authenticated successfully and exposed an idle NVIDIA L4. The previous checkout and all prior `/content` results were absent. Previously downloaded/committed records survived on the local machine and Git. No Google Drive mount was present. We restored checkout `9163149` rather than pretending interrupted processes resumed.

## Environment boundary

This fresh Colab image shipped Python 3.13.16, Torch 2.14.1+cu130 and Transformers 5.22.1; TEBDIL requires Transformers 4.55.0. An initial global dependency install exposed conflicts with notebook Gradio/diffusers packages. TEBDIL was moved into `/content/tebdil-env`, reusing system CUDA Torch while isolating Transformers 4.55.0, tokenizers 0.21.4 and huggingface-hub 0.36.2. The notebook's Transformers 5.22.1 and compatible hub range were restored; its final dependency check still reported missing `jedi` for IPython. We did not establish whether that notebook issue predated this session. TEBDIL imports and GPU execution succeeded.

`venv` failed at `ensurepip`; `venv --without-pip --system-site-packages` plus the existing pip's `--python` target succeeded. The bridge skill now documents this bootstrap boundary. Do not copy the initial global-install approach.

## Extended FP32/SDPA parity

Each model was evaluated at synthetic prefix lengths 128 and 512, with three teacher-forced native argmax continuations per prefix. The existing absolute tolerance remained 0.0001, with exact top-1 and cache-length agreement required. Both passed. These checks do not measure reasoning accuracy or guarantee arbitrary long-generation parity.

| Model | Maximum absolute logit difference | Peak allocated CUDA bytes | Top-1 agreement | Record |
|---|---:|---:|---:|---|
| ByteDance/Ouro-1.4B-Thinking | 2.8610229e-05 | 6,921,975,296 | 6/6 | [extended_cache_parity_14b.json](extended_cache_parity_14b.json) |
| ByteDance/Ouro-2.6B-Thinking | 2.0027161e-05 | 12,967,712,256 | 6/6 | [extended_cache_parity_26b.json](extended_cache_parity_26b.json) |

The exact diagnostic source is archived by SHA-256 in `parity_sources/`. Reproduce with `scripts/check_cached_parity.py --dtype float32 --attention sdpa --prefix-lengths 128 512 --tokens 3`, plus the pinned model revision and a new output path.

## Checkpoint export/restore

1. Run one development completion with S0, FP32/SDPA and a four-token budget.
2. Download its generation checkpoint and summary to the local machine; verify SHA-256.
3. Upload the exact checkpoint bytes into a new runtime directory, preserving the contract-derived key.
4. Launch a fresh runner process with the same execution identity and a new summary path.
5. Verify `checkpoint_reused=true` and exact equality of the original/restored record apart from that reuse flag.

[Transfer digest](checkpoint_recovery/transfer.json), [checkpoint](checkpoint_recovery/generation.json), [original summary](checkpoint_recovery/original_summary.json), and [restored summary](checkpoint_recovery/restored_summary.json) are committed. Automated tests recompute the key/payload hashes and compare both records.

The four-token completion was budget-exhausted, not a correct or incorrect completed answer. Historical generation costs remain in the reused record; they are not costs of new inference. This verifies exported-checkpoint restoration across directories/processes. We did not delete the VM during this trial, test mounted-Drive durability, or demonstrate automatic Google runtime restart.

Reference boundary: “native” here means the repository’s vendored Ouro full-prefix forward at the final recurrent depth. These gates compare that forward with the custom cached scheduler, not an independently loaded upstream implementation. Upstream-code equivalence and long completed reasoning trajectories require separate checks.

## Pinned 2.6B source audit

A subsequent [static source comparison](upstream_26b_source_audit.json) fetched the pinned upstream modeling/configuration files without executing them. Modeling ASTs match outside the intentionally modified cache class. Configuration ASTs match after excluding the upstream `pad_token_id=2` assignment absent from the vendor copy. This narrows the architecture-compatibility uncertainty but does not replace an independent execution check or establish padded-batch equivalence.
