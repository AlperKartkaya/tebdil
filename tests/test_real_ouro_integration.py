"""Integration test using the real ByteDance/Ouro-1.4B-Thinking checkpoint."""

from dataclasses import asdict
import json
import os
from pathlib import Path
import pytest
import torch
from transformers import AutoTokenizer

from tebdil.core.branching import fork_branches
from tebdil.core.controls import (
    execute_norm_matched_random_perturbation_step,
    execute_random_direction_excursion_step,
)
from tebdil.core.excursions import execute_excursion_step
from tebdil.core.state import ExcursionConfig, hash_tensor
from tebdil.evaluation.metrics import compare_logits
from tebdil.models.ouro_adapter import OuroRecurrentAdapter, PINNED_OURO_THINKING_REV
from tebdil.models.validation import validate_ouro_execution


@pytest.mark.real_model
def test_real_ouro_checkpoint_excursion_and_negative_controls(tmp_path: Path):
    """Verify Phase 1 requirements on the actual ByteDance/Ouro-1.4B-Thinking checkpoint."""
    repo_id = "ByteDance/Ouro-1.4B-Thinking"
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, revision=PINNED_OURO_THINKING_REV, local_files_only=True
    )
    adapter = OuroRecurrentAdapter.from_pretrained(
        repo_id=repo_id,
        revision=PINNED_OURO_THINKING_REV,
        dtype=torch.bfloat16,
        readout_mode="last_step",
        total_ut_steps=4,
        local_files_only=True,
    )

    prompt = "Directed graph edges: 0->1 (cost 3), 1->3 (cost 9), 0->2 (cost 4), 2->3 (cost 6), 1->2 (cost 2). What is the minimum path cost from node 0 to node 3? Output only the integer cost.\nAnswer: "
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(adapter.device)

    execution_contract = validate_ouro_execution(adapter, input_ids)
    assert execution_contract["all_passed"], execution_contract

    # 1. Verify exact equivalence with native OuroForCausalLM.forward
    with torch.no_grad():
        native_out = adapter.model(input_ids=input_ids, use_cache=False, exit_at_step=3)
    adapter.reset_call_counter()
    s0 = adapter.initialize(input_ids)
    st_base = s0.clone("A")
    for _ in range(4):
        st_base = adapter.transition(st_base)
    assert adapter.recurrent_block_calls == 4
    base_logits = adapter.readout_logits(st_base)
    cmp_native = compare_logits(native_out.logits, base_logits)
    assert cmp_native.max_abs_diff == 0.0
    assert cmp_native.top1_match is True

    # 2. Fork at step 1, advance to step 2
    s1 = adapter.transition(adapter.initialize(input_ids))
    branches, pert_reports = fork_branches(
        s1, count=2, perturbation_method="normalized_gaussian", relative_norm=0.25, seed=42
    )
    a2 = adapter.transition(branches[0])
    b2 = adapter.transition(branches[1])

    # Verify independent storage and shared weights
    assert a2.hidden.data_ptr() != b2.hidden.data_ptr()
    assert not torch.equal(a2.hidden, b2.hidden)

    # Normal transition from step 2 -> 3
    ref_a3 = adapter.transition(a2.clone())

    # 3. Negative Control 1: alpha = 0
    na_a0, _, tel_a0 = execute_excursion_step(
        adapter, a2.clone(), b2.clone(), ExcursionConfig(alpha=0.0, beta=0.65, max_relative_norm=0.35, visit_step=2)
    )
    assert torch.equal(na_a0.hidden, ref_a3.hidden)

    # 4. Negative Control 2: beta = 0
    na_b0, _, tel_b0 = execute_excursion_step(
        adapter, a2.clone(), b2.clone(), ExcursionConfig(alpha=0.65, beta=0.0, max_relative_norm=0.35, visit_step=2)
    )
    assert torch.equal(na_b0.hidden, ref_a3.hidden)

    # 5. Negative Control 3: Identical receiving and donor states (A == B)
    na_eq, _, tel_eq = execute_excursion_step(
        adapter, a2.clone(), a2.clone("B"), ExcursionConfig(alpha=0.65, beta=0.65, max_relative_norm=0.35, visit_step=2)
    )
    assert torch.equal(na_eq.hidden, ref_a3.hidden)

    # 6. Negative Control 4: Controlled random perturbation (R0 and E0)
    cfg_real = ExcursionConfig(alpha=0.65, beta=0.65, max_relative_norm=0.35, visit_step=2)
    na_r0, _, tel_r0 = execute_norm_matched_random_perturbation_step(
        adapter, a2.clone(), b2.clone(), cfg_real, seed=42
    )
    na_e0, _, tel_e0 = execute_random_direction_excursion_step(
        adapter, a2.clone(), b2.clone(), cfg_real, seed=42
    )

    # 7. Real Cooperative Latent Excursion (E1) at step 2 -> 3, then continue step 3 -> 4
    na_e1_step3, nb_e1_step3, tel_e1 = execute_excursion_step(
        adapter, a2.clone(), b2.clone(), cfg_real
    )
    # Donor B must match normal transition of B
    ref_b3 = adapter.transition(b2.clone())
    assert torch.equal(nb_e1_step3.hidden, ref_b3.hidden)
    assert tel_e1.curvature_frob > 0.0
    assert not torch.equal(na_e1_step3.hidden, ref_a3.hidden)

    # Continue A' from step 3 -> 4
    na_e1_step4 = adapter.transition(na_e1_step3)
    e1_logits = adapter.readout_logits(na_e1_step4)
    cmp_e1_vs_base = compare_logits(base_logits, e1_logits)

    # Write Phase 1 verification evidence artifact (to tmp_path unless canonical write requested)
    if os.environ.get("TEBDIL_WRITE_CANONICAL_EVIDENCE") == "1":
        repo_root = Path(__file__).resolve().parents[1]
        ev_path = repo_root / "artifacts" / "evidence" / "phase1_excursion_verification.json"
    else:
        ev_path = tmp_path / "phase1_excursion_verification.json"
    ev_path.parent.mkdir(parents=True, exist_ok=True)
    ev_payload = {
        "model_id": repo_id,
        "model_revision": PINNED_OURO_THINKING_REV,
        "device": str(adapter.device),
        "execution_contract": execution_contract,
        "native_forward_equivalence": asdict(cmp_native),
        "negative_controls": {
            "alpha_zero_exact_match": bool(torch.equal(na_a0.hidden, ref_a3.hidden)),
            "beta_zero_exact_match": bool(torch.equal(na_b0.hidden, ref_a3.hidden)),
            "identical_states_exact_match": bool(torch.equal(na_eq.hidden, ref_a3.hidden)),
            "disabled_intervention_s0_hash": hash_tensor(st_base.hidden),
            "random_perturbation_r0_post_hash": hash_tensor(na_r0.hidden),
            "random_direction_e0_curvature_frob": round(tel_e0.curvature_frob, 6),
        },
        "real_excursion_step2_trace": asdict(tel_e1),
        "donor_b_undisturbed_exact_match": bool(torch.equal(nb_e1_step3.hidden, ref_b3.hidden)),
        "step4_continuation_vs_baseline": asdict(cmp_e1_vs_base),
        "baseline_top1_decoded": tokenizer.decode([int(base_logits[:, -1, :].argmax().item())]),
        "intervened_top1_decoded": tokenizer.decode([int(e1_logits[:, -1, :].argmax().item())]),
    }
    ev_path.write_text(json.dumps(ev_payload, indent=2), encoding="utf-8")
