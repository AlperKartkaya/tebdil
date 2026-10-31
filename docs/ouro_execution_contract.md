# Ouro execution contract

## Supported computation

TEBDİL executes the frozen `ByteDance/Ouro-1.4B-Thinking` checkpoint at revision
`3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`, using the vendored model definitions.
One recurrent step is **all 24 shared decoder layers, the native boundary RMSNorm,
and the exit gate**. The state is the normalized stack output; the adapter does not
remove or add a normalization at this boundary.

Execution is full-prefix, eager attention, with **KV caching disabled**. Each
branch retains its own state and readout histories. Token IDs, effective attention
mask, position IDs and recurrent depth identify its transition context. Only
states with compatible contexts may interact. A transition rejects enabled-cache
contexts and inconsistent depth metadata.

`initialize` accepts `[batch, tokens]` IDs, an optional same-shaped padding mask,
and optional nonnegative int64 position IDs of shape `[1 or batch, tokens]`.
Default positions are `0..L-1`, including padding, exactly as in an explicit native
forward without supplied positions. Position IDs are not silently inferred from
the padding mask. Custom position IDs are copied into the stored context.

## Readout is not computation savings

- **`last_step`** projects the latest state. Its reference is native forward with
  `exit_at_step=depth-1` (the native argument is zero-indexed).
- **`native_gate`** honors a valid configured `early_exit_step` first; otherwise
  it applies the configured cumulative gate threshold to the available history.
  Without a configured step or threshold it uses the final state, as native does.
- A native forward computes all configured recurrent steps before selecting its
  readout. Gate-selected earlier states do not by themselves prove saved compute.
- The recorded task configurations use `last_step`. They must not be described as
  evaluations of Ouro's default generation policy or cached generation.

## Fresh verification

The [machine-readable execution record](../reports/ouro_execution_contract.json)
contains checkpoint-file and source-file SHA-256 identities, device, framework,
input fixtures, all depth checks and cross-precision measurements.

On Apple MPS, **both FP32 and BF16 matched native execution exactly** for three
short token fixtures: one token, a left/right-padded batch, and an eight-token
prefix with shifted positions. At all four depths, hidden states, gates and forced
readout logits matched. Native-gate logits also matched at thresholds `1.0` and
`0.5`: 24 depth checks and 48 native-gate comparisons in total.

FP32 and BF16 are **not equivalent to each other**. Final-state maximum absolute
differences ranged from `0.0641` to `1.3615`; last-valid-token logit differences
ranged from `0.1862` to `0.4448`. The four compared last-valid-token argmax outputs
agreed. These are diagnostics on short token fixtures, not precision-robustness
evidence for excursion residuals, task accuracy or long contexts.

## Reproduce

With the exact checkpoint already in the Hugging Face cache:

```bash
uv run pytest -q tests/test_adapter_equivalence.py tests/test_adapter_cli.py
uv run python scripts/verify_ouro_execution.py --output /tmp/ouro-execution-new.json
uv run pytest -q -m real_model
```

The script refuses to overwrite evidence and verifies every cached checkpoint
file against the committed provenance hashes. Missing weights or a failed check
are not replaced with mini-model results. Randomly initialized mini models are
used only for unit tests, including readout overrides and drift detection.

## Limits

This establishes bounded adapter fidelity, **not** intervention usefulness,
benchmark integrity, controller quality, long-context behavior, CUDA parity or
cached-generation compatibility. Historical task results and known cost-ledger
failures remain unchanged. Normalization and precision effects on the excursion
operator require their own later controlled experiments.
