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
