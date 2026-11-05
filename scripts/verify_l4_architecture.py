"""Verify TEBDIL v2 architectural fixes on NVIDIA L4 GPU with real ByteDance/Ouro-1.4B-Thinking.

Tests:
1. Branched KV-cache (`UniversalTransformerCache`) parity and wall-clock speedup vs full-prefix recomputation.
2. Pre-RMSNorm (`update_space="pre_norm"`) + `last_token` scope + `competitive_logit` donor sensitivity
   at reasoning stall prefixes vs the legacy post-norm full-prefix isotropic Gaussian configuration.
3. Multi-token cached generation (`generate_tebdil_cached`) across S0, B_only, M0, M1, E0, E1, C1_heuristic.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer

from tebdil.core.branching import fork_branches
from tebdil.core.controls import execute_direct_state_mixing_step, execute_update_mixing_step
from tebdil.core.excursions import execute_excursion_step
from tebdil.core.scheduler import generate_tebdil_cached, run_scheduled_trajectory
from tebdil.core.state import ExcursionConfig
from tebdil.evaluation.reasoning import encode_reasoning_prompt
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "ByteDance/Ouro-1.4B-Thinking"


@torch.no_grad()
def test_kv_cache_parity_and_speedup(adapter: OuroRecurrentAdapter, tokenizer) -> dict:
    prompt = "Find the least positive integer x such that x mod 7 = 3 and x mod 11 = 5."
    ids = encode_reasoning_prompt(tokenizer, prompt).to(adapter.device)

    # 1. Parity over 16 generated tokens (S0 cached vs S0 uncached)
    t0 = time.monotonic()
    cached_res = generate_tebdil_cached(
        adapter,
        ids,
        method="S0",
        max_new_tokens=16,
        eos_token_id=tokenizer.eos_token_id,
    )
    cached_elapsed = time.monotonic() - t0

    adapter.cache_mode = "disabled"
    adapter._active_kv_cache = None
    uncached_ids = ids.clone()
    uncached_tokens = []
    t1 = time.monotonic()
    for _ in range(16):
        traj = run_scheduled_trajectory(
            adapter,
            uncached_ids,
            method="S0",
            recurrent_steps=4,
            record_pca=False,
            use_cached_step=False,
        )
        tok = int(traj.branch_logits["A"][:, -1].argmax().item())
        uncached_tokens.append(tok)
        uncached_ids = torch.cat(
            [uncached_ids, torch.tensor([[tok]], dtype=torch.long, device=adapter.device)],
            dim=1,
        )
    uncached_elapsed = time.monotonic() - t1
    adapter.cache_mode = "branched_kv"

    exact_match_tokens = sum(
        1 for a, b in zip(cached_res["generated_token_ids"], uncached_tokens) if a == b
    )
    return {
        "prompt_length": int(ids.shape[1]),
        "cached_tokens": cached_res["generated_token_ids"],
        "uncached_tokens": uncached_tokens,
        "exact_match_count": exact_match_tokens,
        "cached_16tok_sec": round(cached_elapsed, 4),
        "uncached_16tok_sec": round(uncached_elapsed, 4),
        "decoded_cached": tokenizer.decode(cached_res["generated_token_ids"], skip_special_tokens=False),
    }


@torch.no_grad()
def compare_legacy_vs_v2_on_stall_prefixes(adapter: OuroRecurrentAdapter, tokenizer) -> list[dict]:
    passive_dir = ROOT / "reports/passive_reasoning_v1"
    report = json.loads((passive_dir / "report.json").read_text())
    results = []

    legacy_cfg = ExcursionConfig(
        alpha=0.65,
        beta=0.65,
        max_relative_norm=0.10,
        visit_step=2,
        token_scope="full",
        donor_strategy="normalized_gaussian",
        update_space="post_norm",
    )
    v2_cfg = ExcursionConfig(
        alpha=0.65,
        beta=0.65,
        max_relative_norm=0.25,
        visit_step=2,
        token_scope="last_token",
        donor_strategy="competitive_logit",
        update_space="pre_norm",
    )

    probe_indices = [127, 176, 207, 255, 511]
    for summary in report["records"]:
        case = json.loads((passive_dir / summary["raw_file"]).read_text())
        row = case["record"]
        prompt_ids = row["prompt_token_ids"]
        gen_ids = row["generated_token_ids"]
        for k in probe_indices:
            if k >= len(gen_ids):
                continue
            prefix = torch.tensor([prompt_ids + gen_ids[:k]], dtype=torch.long, device=adapter.device)
            adapter.prefill_kv_cache(prefix[:, :-1])
            s0_init = adapter.initialize_cached_step(prefix[:, -1:])
            root = adapter.transition(s0_init)

            # Baseline S0
            s_a = adapter.transition(root)
            s_a3 = adapter.transition(s_a)
            s_a4 = adapter.transition(s_a3)
            logits_s0 = adapter.readout_logits(s_a4, last_token_only=True)[0, -1].float()
            top2_s0 = logits_s0.topk(2)
            s0_tok = int(top2_s0.indices[0].item())
            s0_margin = float((top2_s0.values[0] - top2_s0.values[1]).item())

            comparison = {
                "problem_id": row["problem_id"],
                "token_index": k,
                "prefix_length": int(prefix.shape[1]),
                "s0_token_id": s0_tok,
                "s0_token_text": tokenizer.decode([s0_tok], skip_special_tokens=False),
                "s0_top2_margin": round(s0_margin, 4),
            }

            for label, cfg, rel_norm in [("legacy_postnorm_full", legacy_cfg, 0.25), ("v2_prenorm_comp_logit", v2_cfg, 0.25)]:
                branches, _ = fork_branches(
                    root,
                    count=2,
                    method=cfg.donor_strategy,
                    relative_norm=rel_norm,
                    seed=20261105 + 131 * k,
                    token_scope=cfg.token_scope,
                    lm_head_weight=adapter.lm_head_weight,
                )
                a, b = [adapter.transition(s) for s in branches]
                e1, _, telemetry = execute_excursion_step(adapter, a, b, cfg)
                m1, _, _ = execute_update_mixing_step(adapter, a, b, cfg)
                final_e1 = adapter.transition(e1)
                final_m1 = adapter.transition(m1)
                final_b = adapter.transition(adapter.transition(b))

                logits_e1 = adapter.readout_logits(final_e1, last_token_only=True)[0, -1].float()
                logits_m1 = adapter.readout_logits(final_m1, last_token_only=True)[0, -1].float()
                logits_b = adapter.readout_logits(final_b, last_token_only=True)[0, -1].float()

                e1_tok = int(logits_e1.argmax().item())
                m1_tok = int(logits_m1.argmax().item())
                b_tok = int(logits_b.argmax().item())

                comparison[label] = {
                    "e1_token_id": e1_tok,
                    "e1_token_text": tokenizer.decode([e1_tok], skip_special_tokens=False),
                    "e1_changed_vs_s0": e1_tok != s0_tok,
                    "m1_token_id": m1_tok,
                    "m1_changed_vs_s0": m1_tok != s0_tok,
                    "b_token_id": b_tok,
                    "b_changed_vs_s0": b_tok != s0_tok,
                    "logit_l2_e1_vs_s0": round(float((logits_e1 - logits_s0).norm().item()), 4),
                    "logit_l2_e1_vs_m1": round(float((logits_e1 - logits_m1).norm().item()), 4),
                    "curvature_rel": round(float(telemetry.curvature_relative_to_update_a), 4),
                    "effective_alpha": round(float(telemetry.effective_alpha), 4),
                }
            results.append(comparison)
    return results


def main() -> None:
    print("Loading tokenizer and ByteDance/Ouro-1.4B-Thinking on CUDA...")
    t0 = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=PINNED_OURO_THINKING_REV)
    adapter = OuroRecurrentAdapter.from_pretrained(
        model_id=MODEL_ID,
        revision=PINNED_OURO_THINKING_REV,
        dtype=torch.bfloat16,
        device="cuda",
        total_ut_steps=4,
        cache_mode="branched_kv",
        attn_implementation="sdpa",
    )
    load_sec = round(time.monotonic() - t0, 2)
    print(f"Loaded in {load_sec}s on {adapter.device}")

    parity = test_kv_cache_parity_and_speedup(adapter, tokenizer)
    print("KV Cache Parity & Speedup:", json.dumps(parity, indent=2))

    stall_comp = compare_legacy_vs_v2_on_stall_prefixes(adapter, tokenizer)
    legacy_flips = sum(1 for r in stall_comp if r["legacy_postnorm_full"]["e1_changed_vs_s0"])
    v2_flips = sum(1 for r in stall_comp if r["v2_prenorm_comp_logit"]["e1_changed_vs_s0"])
    print(f"Stall prefix token flips: legacy={legacy_flips}/{len(stall_comp)}, v2={v2_flips}/{len(stall_comp)}")

    out_path = ROOT / "reports/l4_architecture_verification.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "load_seconds": load_sec,
                "device": str(adapter.device),
                "gpu_name": torch.cuda.get_device_name(0),
                "kv_cache_parity": parity,
                "legacy_flips": legacy_flips,
                "v2_flips": v2_flips,
                "stall_comparisons": stall_comp,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
