# Point B — trajectory semantics and baseline calibration

2026-10-31. **Complete: code, documentation, deployment and public browser checks passed.**
Implementation: `22cc752`; [CI and Pages deployment](https://github.com/AlperKartkaya/tebdil/actions/runs/41239947652).
Point A remains separately verified in `e0728b3` and [deployment 41219360287](https://github.com/AlperKartkaya/tebdil/actions/runs/41219360287).

## Implementation

- Per-token records now identify actual prefix IDs/hashes/lengths, branch seeds,
  encoded prompts, generated IDs, prompt format, branch lifetime and stop reason.
  Branches re-fork inside each generated token's depth unroll; hidden states do
  not persist across output positions. Subsequent influence is through emitted tokens.
- EOS stops immediately even when decoded text is empty. Diagnostic chat permits
  reasoning newlines; the historical raw stopping policy remains explicitly labeled.
- FLOP estimates now sum each token's actual prefix length. These remain estimates,
  not measured FLOPs or evidence of compute matching.
- Full teacher-forced gold-sequence scoring no longer truncates at the generation
  budget. Its regression failed first, then passed with all target tokens and
  separately charged scoring calls. Two different gold targets leave the generated
  token unchanged. Scoring occurs after generation, not inside controller decisions.
- Chat scoring requires EOS and an explicit final integer outside thought. It
  never scores the first integer in an unfinished continuation. The conservative
  `completed_final_integer_v1` parser was developed after inspecting two pilot
  outputs; raw extraction remains the explicitly labeled legacy policy.

## Real-checkpoint evidence

The initial [calibration](ouro_baseline_calibration.json) preserves nine evaluations
on the same three pilot problems: seed 20261031, BF16, forced depth 4, MPS.
Raw/2 and raw/64 each scored 1/3; chat/128 always truncated thinking.

The new [native completion](ouro_baseline_completion.json) follows up **all three**
chat cases, with the same prefixes, greedy forced-depth-4 execution, disabled
caching and a 2048-token ceiling. Checkpoint files were rehashed before loading;
source identities were captured before generation. Its [JSONL journal](ouro_baseline_completion.jsonl)
preserves each finished case immediately. Hooks count actual normalization-boundary
executions and record every actual forward-input prefix, not inferred savings.

| Family | Generated tokens | Actual recurrent calls | Stop | First 128 vs adapter |
|---|---:|---:|---|---|
| Arithmetic | 437 | 1,748 | EOS | Exact agreement |
| Logic | 483 | 1,932 | EOS | Exact agreement |
| Graph | 1,514 | 6,056 | EOS | Exact agreement |

All three final responses contain correct mathematics by inspection, but fail the
original whole-response integer format because they contain prose. A separate
[post-hoc annotation](ouro_completion_scoring.json) recognizes 50 and 0; it rejects
the graph's prose conclusion of 15. That rejection is retained, not tuned away.
These are parser/protocol diagnostics, **not a 2/3 mathematical accuracy estimate**,
a held-out benchmark, new independent problems, or a TEBDİL comparison.
Original outputs and strict scores remain unchanged.

Measured case wall times were 689.223, 854.096 and 7,480.436 seconds. This expensive
cache-free path is not a practical-caching or campaign-throughput benchmark.
The initial interrupted attempt is excluded because its raw output was not saved.
A mini cached-generation probe failed on the pinned Transformers cache API; no
real cached baseline ran. No upstream cache implementation was patched.

## Verification

- `uv run pytest -q`: **122 passed**, including real Ouro integration on MPS.
- Regression checks reconstruct every native prefix and calibration branch seed,
  reconcile inference/scoring calls, match the completion journal to final JSON,
  verify source and input identities, and reproduce post-hoc parsing independently.
- Scoped Ruff (`E4,E7,E9,F`), JavaScript syntax, TypeScript and export hashes/pairing pass.
- Full Chromium verification locally and on the public GitHub Pages `/tebdil/` URL: all eight recorded datasets, three demo
  cases, downloads, evidence links, keyboard, failure/retry/checksum handling pass.
  New native and post-hoc JSON downloads are checked separately. No page errors,
  failed normal requests, external runtime dependencies or overflow at widths
  320/390/768/1024/1440. Expanded evidence screenshot inspected.
- Original CSS, section order, IDs, experiment shards and historical scores are
  preserved. Seven pre-existing modified research artifacts were not regenerated,
  staged or republished. Only new evidence was appended to the export manifest.

## Boundaries and next gate

Three pilot problems do not establish baseline competence, optimal prompting,
precision robustness, statistical improvement or intervention efficacy. The parser
is deliberately conservative and post-hoc; future held-out comparisons must freeze
one common prompt/completion/scoring protocol and disclose format validity separately.
Historical raw-prompt interventions cannot be fairly compared with this longer
native generation as an efficacy result. CUDA, native caching, branch-safe caching,
and a complete new held-out campaign are unverified.

Pages deployment and the full public browser checks passed. Point B is closed
for this bounded semantics/pilot-calibration scope; point C may now begin.
A comparison against `6fe919e` also confirms that the export manifest is append-only:
every existing run metric, evidence entry and artifact identity is unchanged.
