# Reasoning baseline budget follow-up v1

Freeze this protocol, runner and tests before new observations. This is an
adaptive development diagnostic after two unfinished 2048-token attempts, not a
preregistered benchmark, stall detector, prompt comparison or intervention study.

## Selection and stopping

Retain both original level-0 development cases, in original manifest order (CNF,
then key/door planning). Resume each original prompt plus all 2048 actual emitted
tokens, unchanged. Generate at most **512 additional tokens** or until native EOS,
whichever comes first: total output cap **2560**. The extension is a resource cap,
not evidence that 2560 is adequate. Do not extend again, retry, choose a different
case, coach the model, close its thinking tag or insert a final-answer cue based
on outcomes. No reserved-set inference, new prompt, donor, controller or TEBDİL arm.

Reuse pinned Ouro-1.4B-Thinking, BF16 MPS, four depths, forced last-step, eager,
cache disabled, native greedy generation and all-position logits. Preserve
`enable_thinking=True` from the original prompt. Cache-free full-prefix generation
has no hidden/KV state to restore across tokens. Before resuming each case, replay
the last original proposal (k=2047) and require exact token and last-depth normalized
state-hash parity. Charge the replay calls separately. Abort and preserve partial
evidence on mismatch, changed sources/checkpoint, invalid ledger or nonfinite logits.

## Recording and scoring

Use the existing native generator, independent emitted-token observation and
physical RMSNorm boundary accounting. Journal every new proposal with its token
ID, absolute output index, actual full-prefix hash/length and physical calls.
Persist compact suffix ledgers: full processed IDs are checked in memory against
the original prompt plus old/new emitted IDs, reconstructible without storing
the same prefix hundreds of times. Preserve the original recording by reference
and checksum; do not relabel its 8192 calls per case as newly executed.

Decode the actual combined old/new output with special tokens retained. Use the
existing reasoning-boundary splitter and unchanged task verifier offline. Score
only a native EOS-terminated, closed-reasoning final answer. Keep budget exhaustion,
malformed boundaries, invalid final grammar and recognized wrong answers distinct.
Do not score integers/routes mentioned inside CoT. Retain failures as well as successes.

Report per-case new tokens, total tokens, stop/boundary/verification outcome,
new generation calls and recurrent token positions, parity-check calls, reused
baseline calls and measured instrumented runtime. Runtime excludes loading and
final provenance checks; token positions are not FLOPs or a latency benchmark.
The two cases are reused observations, not independent new samples. No significance
test, population accuracy, causal rescue or “stuck” label follows from this check.

## Interpretation and release

Completion after token 2048 would show that this original budget truncated that
particular trajectory, not that TEBDİL is useful or the prompt optimal. Continued
truncation would leave budget adequacy and task/prompt suitability unresolved;
it would not justify automatically making problems harder or infinite extensions.
A wrong completed answer establishes a bounded baseline failure, not causal stall
timing. Causal selection and intervention utility still require a separate freeze.

Publish immutable raw report/journals and actual suffix text with a clear **ordinary
baseline, no TEBDİL** label. Keep the previous 2048-token results and sensitivity
probe unchanged. Validate source/input/checkpoint hashes before and after the run;
preserve an interrupted run rather than overwrite it.
