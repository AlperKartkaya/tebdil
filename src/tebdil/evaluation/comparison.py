"""Gold-free fixed-method completion using the existing recurrent scheduler."""

from dataclasses import asdict

import torch

from tebdil.core.scheduler import run_scheduled_trajectory
from tebdil.core.state import hash_tensor
from tebdil.evaluation.completion import encode_completion_prompt
from tebdil.tasks.completion import extract_completed_integer

METHOD_CALLS = {"S0": 4, "B_only": 7, "M0": 7, "M1": 7, "E0": 8, "E1": 8}


@torch.no_grad()
def generate_fixed_completion(adapter, tokenizer, prompt, method, config, *, seed, budget, on_token=None):
    """Re-fork inside each token; never retain logits from previous token positions."""
    if method not in METHOD_CALLS or type(budget) is not int or budget <= 0:
        raise ValueError("Unsupported fixed method or token budget")
    if (adapter.total_steps != 4 or adapter.readout_mode != "last_step"
            or config.visit_step != 2 or config.token_scope != "full"
            or config.gate_modulated or config.subspace_fraction != 1.0
            or config.coordinate_topk_fraction != 1.0):
        raise ValueError("Only the frozen full-scope four-depth comparison is supported")
    ids = encode_completion_prompt(tokenizer, prompt).to(adapter.device)
    prompt_ids = ids.tolist()[0]
    tokens, ledger = [], []
    physical_calls = 0
    calls_before = adapter.recurrent_block_calls

    def boundary(module, inputs, output):
        nonlocal physical_calls
        physical_calls += 1

    handle = adapter.model.model.norm.register_forward_hook(boundary)
    try:
        for k in range(budget):
            before = physical_calls
            result = run_scheduled_trajectory(
                adapter, ids, method="P1" if method == "B_only" else method,
                recurrent_steps=4, fork_after_step=1, branch_count=2,
                perturbation_method="normalized_gaussian", perturbation_relative_norm=0.25,
                excursion_config=config, seed=seed + 131 * k, record_pca=False,
            )
            calls = physical_calls - before
            if (calls != result.recurrent_block_calls or calls != METHOD_CALLS[method]
                    or result.controller_calls or result.exited_early_step is not None):
                raise ValueError("Physical/scheduler call counts or fixed-method contract disagree")
            telemetry = result.excursion_telemetry
            if telemetry and (telemetry.skipped_unstable or not telemetry.post_finite):
                raise ValueError("Intervention instability; preserve partial journal, do not substitute baseline")
            selected = "B" if method == "B_only" else "A"
            logits = result.branch_logits[selected]
            if not torch.isfinite(logits).all():
                raise ValueError("Nonfinite readout")
            token = int(logits[:, -1].argmax().item())
            tokens.append(token)
            entry = {
                "token_index": k, "token_id": token, "prefix_length": ids.shape[1],
                "prefix_token_ids_sha256": hash_tensor(ids, 64),
                "calls_before": before, "recurrent_block_calls": calls,
                "branch_seed": seed + 131 * k, "selected_branch": selected,
                "selected_hidden_hash": hash_tensor(result.branch_states[selected].hidden, 64),
                "intervention": asdict(telemetry) if telemetry else None,
            }
            ledger.append(entry)
            if on_token:
                on_token(entry)
            # Do not keep full-vocabulary readouts or recurrent states across output tokens.
            del logits, result
            if token == tokenizer.eos_token_id:
                break
            ids = torch.cat([ids, torch.tensor([[token]], dtype=torch.long, device=adapter.device)], dim=1)
    finally:
        handle.remove()
    if physical_calls != adapter.recurrent_block_calls - calls_before:
        raise ValueError("Adapter and independent physical counters disagree")
    completed = tokens[-1] == tokenizer.eos_token_id
    raw = tokenizer.decode(tokens, skip_special_tokens=False)
    answer = extract_completed_integer(raw, tokenizer.eos_token, completed=completed)
    return {
        "method": method, "prompt_token_ids": prompt_ids, "generated_token_ids": tokens,
        "decoded_text_with_special_tokens": raw, "token_ledger": ledger,
        "completed": completed, "stop_reason": "eos_token" if completed else "max_output_tokens",
        "format_valid": answer is not None, "extracted_answer": answer,
        "answer_extraction": "completed_final_integer_v1",
        "recurrent_block_calls": physical_calls,
        "recurrent_token_positions": sum(e["prefix_length"] * e["recurrent_block_calls"] for e in ledger),
    }
