# Passive reasoning recording v1

Frozen before observations. This is a **mechanical recording pilot**, not a
validated semantic stall detector, difficulty benchmark, intervention or controller.
The first two development tasks in the frozen task manifest are selected now:
level-0 CNF and level-0 key/door planning. No outcome-driven task selection or retry.
The other four development and twelve reserved instances are not executed by this
pilot; the old 12/60 arithmetic data remain untouched.

## Unchanged generation

Reuse `run_reasoning_completion` and native generation, not another decoder.
Exact pinned Ouro-1.4B-Thinking/checkpoint/tokenizer, BF16 Apple MPS, eager attention,
True-mode input opening, greedy, four depths, forced last-step, no KV cache.
EOS or 2,048 newly generated tokens including EOS; task v1 final-only scoring
is **offline after generation**. Wrong, invalid and budget-limited attempts stay.
Question string only enters the generator. No answer, instance, solution search,
future text, task verifier, manual annotation or eventual outcome enters the observer.

## Real aligned states, not invented trajectories

For the next output-token index k, native input is prompt + emitted tokens [0:k].
Observe the **last actual input position** L−1 at each recurrent depth t=0…3:
first-layer input, full recurrent block output before shared RMSNorm, and actual
normalized output. The normalized output must exactly match the next depth's input.
All four observations share the same prefix length/hash and actual positions
0…L−1. Unpadded attention/position creation is unchanged native generation; no new
mask or cache code. No growing-prefix tensor subtraction across different k.

Record dtype-aware tensor hashes, L2 norms, ||normalized−input||, relative update
and cosine in FP32 CPU arithmetic. The t=0 input is a token embedding; later inputs
are previous normalized states. These quantities include normalization effects
and **do not measure semantic progress, convergence, correctness or confidence**.
Zero denominators produce null, not invented zero similarity. Nonfinite values or
misaligned boundaries abort, preserving partial evidence and removing hooks.

At fixed k = 0,127,255,511,1023,2047, also save actual last-position input/raw/
normalized vectors. Other steps retain exact hashes/scalars, not full tensors.
No PCA, latent coordinates, extra recurrent calls or added vocabulary readouts.
Snapshots are local states from one prefix, not persistent agents across tokens.
Mini-native regression verifies identical tokens, calls and weights with/without
observation, reconstructed snapshot metrics/hashes, and failure-hook cleanup.

## Causal candidate repeat screen

Evaluate only the already-emitted token prefix, **before appending k's proposed
next token**, never the future completion. Decode this prefix to identify the
open reasoning section. A closing `</think>` makes it final; repeated closing or
generated opening tags mark malformed. Partial tag fragments remain open until
they decode as a complete tag. The input's opening is not regenerated.

Candidate flag: at least 128 emitted tokens, phase reasoning, and the last eight
IDs match at least two earlier eight-ID windows inside the most recent 128 IDs.
Overlaps count. Save the prior-match count at every step, including non-flags and
final/malformed phases. This exact-repeat rule is **fixed and heuristic, not learned
or a proof of a stall**; repetitive tokens can be productive checking. Latent norms
are passive diagnostics and do not trigger the flag. A wrong answer can lack repeats.
Flags are recorded after the native forward and cannot claim saved work. Any later
prefix-fork experiment must charge/reconstruct that work and all donor/monitor cost.

## Evidence and stop conditions

Hash all generator, observer, verifier, test, protocol and data inputs plus cached
checkpoint files before/after. Preserve returned IDs, exact raw CoT/final text,
independent native proposals, costs, prefix hashes, actual depth states, per-token
features, source identities and append-only case/run journals. Existing output
directories are refused; partial failures are not automatically resumed or retried.
Physical calls and recurrent token positions count the actual full-prefix work;
wall time includes CPU copies, journal I/O and validation, not optimized throughput.

After recording: reconstruct every prefix/position, state chain and snapshot hash;
recompute repeat flags and final scores; retain all outcomes. Review actual CoT,
not just numbers. Offline process rubric: distinguish new constraint/path/search
information, productive verification or correction, and repeated material with no
new task-relevant inference. State uncertain/no-label explicitly. Exact-repeat
flags alone cannot assign these semantic labels or establish causal rescue.
Report false alarms and missed failures if defensible annotations are available;
one analyst/two cases cannot validate a general stall detector.

Do not proceed to a recovery/controller claim based on mechanical checks. A
controlled-continuation protocol needs separately frozen selection, interventions,
compute-matched controls, endpoints and uncertainty, and an eligible causal prefix.
If this pilot has no eligible/defensible stall examples, report that limit rather
than forcing a flag, changing thresholds after outcomes or using an oracle time.

```bash
uv run pytest -q tests/test_passive_reasoning.py
uv run python scripts/record_passive_reasoning.py --output-dir /tmp/passive-reasoning-new
```

This may take hours on 16 GB MPS. No GPU inference server is required or exposed
by GitHub Pages. Public replay is explicitly recorded, never live model execution.
