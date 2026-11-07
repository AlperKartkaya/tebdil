# Two-model completion smoke

Frozen before execution: one existing development question (`completion_dev_000`),
S0 and E1, seed 42, 256 generated-token cap, FP32/SDPA, on the same Colab L4.
Run Ouro-1.4B-Thinking and Ouro-2.6B-Thinking sequentially at the pinned revisions
already used for parity. Use the completion suite's unchanged excursion/policy
settings. No reserved-test questions, prompt edits, or budget extensions based on
observed answers. Keep every attempted outcome, including budget exhaustion.

Purpose: verify chat-formatted generation and record actual termination/costs on
both supported model sizes. One arithmetic question is not a benchmark, evidence
of general reasoning improvement, or a compute-matched comparison. S0 and E1 have
the same output-token cap, not necessarily the same recurrent compute. A zero
intervention count cannot support an excursion-efficacy claim.

Commands (each with a unique output and checkpoint directory):

```sh
python scripts/run_colab_l4.py --suite completion --max-tasks 1 \
  --methods S0 E1 --max-new-tokens 256 --seed 42 --dtype float32 --attn sdpa \
  --output <summary.json> --checkpoint-dir <checkpoints>
# For 2.6B add:
# --model ByteDance/Ouro-2.6B-Thinking \
# --revision f1edd81e7ac41355db670500ceaf204e0f73af68
```

## Provenance and configuration boundary

Executed source: Git commit `c336a859449bc07eff43f5cf0cd9895773f1ce8d`.
The runner source is archived alongside the raw summaries and generation
checkpoints. Checkpoints bind prompt token IDs, effective settings, model revision,
runtime versions and source hashes. `transfer.json` records remote byte hashes
verified after download. No runtime credentials are stored here.

This trial predates the summary-configuration fix: its top-level
`excursion_config`/`policy_config` are **base defaults**, not necessarily effective
suite settings. Use each checkpoint's `contract.parameters` as the authoritative
settings. In particular, the completion suite uses a one-token excursion window,
not the four-token base default. New runner summaries name these top-level fields
`base_excursion_config`/`base_policy_config` and attach effective settings to
completion/literature records. Historical raw artifacts remain unmodified.

## Observed outcomes

| Model | Method | Outcome | Generated tokens | Actual excursions | Recurrent block calls | Generation seconds |
|---|---|---|---:|---:|---:|---:|
| Ouro-1.4B-Thinking | S0 | Budget exhausted | 256 | 0 | 1,028 | 32.7533 |
| Ouro-1.4B-Thinking | E1 | Budget exhausted | 256 | 256 | 1,796 | 55.6949 |
| Ouro-2.6B-Thinking | S0 | Correct, final answer 29 | 185 | 0 | 744 | 46.3805 |
| Ouro-2.6B-Thinking | E1 | Budget exhausted | 256 | 256 | 1,796 | 109.8898 |

All four attempts are retained; none reused checkpoints. The S0 call count
includes four prompt-prefill calls plus four calls per output token. These call
counts are not FLOPs: prefill sequence length, branch work and model size matter.
Generation times are single observations, not latency distributions.

E1 is the fixed/unconditional intervention control, not the adaptive E2 policy.
It intervened on every generated token here. This single development smoke shows
no excursion benefit and a worse termination outcome for 2.6B E1 at the frozen
budget. It does not establish general degradation, compare adaptive methods, or
justify tuning on reserved tests. No budget was extended after seeing outcomes.
