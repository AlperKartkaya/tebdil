# Geometry shadow verification — 2026-11-08

This is an engineering parity check, **not an efficacy result**. No geometric
intervention ran and no E2/E3 policy changed.

## Execution

- Code: `4764e1514ceec0bbdaa4dd60283783316153b784`.
- Model: pinned `ByteDance/Ouro-1.4B-Thinking`, revision
  `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`.
- Device/precision/attention: Colab NVIDIA L4, FP32, SDPA.
- One fixed arithmetic prompt; 64 generated tokens per arm, both budget-exhausted.
- Baseline then shadow, same prompt and checkpoint, no task answers supplied to
  the observer. Four native passes per token plus four prefill calls.
- Reproduction: `python scripts/verify_geometry_shadow.py --real-model --tokens 64
  --output <new-report.json>` in the project's supported environment.

## Observed results

| Check | Result |
| --- | --- |
| Generated token IDs | All 64 identical |
| Final logits | All 64 full-logit SHA-256 hashes identical |
| Committed K/V | Every slot's SHA-256 identical |
| Recurrent calls | 260 in each arm |
| Extra vocabulary projections | 64 in shadow; not free computation |
| Geometric interventions | Zero |
| Shadow diagnostics | 8 warm-up/insufficient-history; 56 progress |
| Eligible / selected | 0 / 0 |
| Tensor-only retraction vs rotating raw residual and applying native RMSNorm | Passed; maximum absolute error 9.5367431640625e-07 |
| Zero radius / inactive tokens / input preservation | Exact |

The separate retraction check made three ordinary reference transitions and
never fed the retracted candidate into a model transition. It does not validate
visit/return cache integration. This real prompt did not exercise the warning or
selection path; synthetic tests cover those mechanics, not their semantic value.

Raw timings (13.3897 seconds baseline, 8.7172 seconds shadow) include diagnostic
hashing, initialization/warm-up and an unbalanced run order. **Do not interpret
them as a speedup or an overhead estimate.** The readout projection count is the
explicit additional-work measure available from this probe.

## Evidence and boundaries

`ouro_1_4b_fp32.json` contains raw token IDs, both logit/cache hash lists, all shadow
records, the prompt, package versions, model/code revisions and 54 source-file
SHA-256 digests. These source digests were matched to the local checkout after
transfer. `transfer.json` stores the remote file's SHA-256; the downloaded bytes
were verified before recording this summary. No runtime URL/token is stored.

Remaining gates: selected-visit construction and return-operator reconciliation;
branch/cache commit and rollback parity; then controlled, paired, completed-answer
experiments. Nothing here demonstrates better answers, detector accuracy, model
portability, or throughput gains. The website's efficacy figures are unchanged.
