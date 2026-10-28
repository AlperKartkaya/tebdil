"""Unit tests for the self-directed TEBDIL controller action space, features, training, and scheduling."""

import json
from pathlib import Path
import torch

from tebdil.controller import (
    FEATURE_SCHEMA,
    LearnedControllerCheckpoint,
    SelfDirectedController,
    build_discrete_action_space,
    collect_counterfactual_dataset,
    extract_state_features,
    featurize_state_action,
    train_learned_controller,
)
from tebdil.core.branching import fork_branches
from tebdil.core.scheduler import run_scheduled_trajectory
from tebdil.core.state import ExcursionConfig
from tebdil.evaluation.runner import ByteLevelToyTokenizer
from tebdil.models.ouro_adapter import OuroRecurrentAdapter


def test_action_space_and_state_action_featurization():
    adapter = OuroRecurrentAdapter.from_mini_config(total_ut_steps=4, seed=42)
    s0 = adapter.initialize(torch.tensor([[1, 2, 3, 4, 5]]))
    s1 = adapter.transition(s0)
    branches, _ = fork_branches(s1, count=3, relative_norm=0.10, seed=42)
    branches = [adapter.transition(b) for b in branches]

    sf = extract_state_features(
        adapter=adapter,
        state_a=branches[0],
        donors=branches[1:],
        total_steps=4,
        budget_remaining=4,
    )
    assert len(sf.donors) == 2
    actions = build_discrete_action_space(num_donors=2, allow_exit=True)
    # 1 CONTINUE + 1 EXIT + 2 donors * 3 (alpha, beta) pairs = 8 actions
    assert len(actions) == 8

    for act in actions:
        vec = featurize_state_action(sf, act)
        assert len(vec) == len(FEATURE_SCHEMA)
        assert all(isinstance(x, float) for x in vec)


def test_counterfactual_training_and_state_sensitivity(tmp_path: Path):
    adapter = OuroRecurrentAdapter.from_mini_config(vocab_size=128, total_ut_steps=4, seed=77)
    tokenizer = ByteLevelToyTokenizer(vocab_size=128)

    cf_data = collect_counterfactual_dataset(
        adapter=adapter,
        tokenizer=tokenizer,
        split="train",
        task_family="all",
        num_problems=6,
        seeds=(42,),
        recurrent_steps=4,
        fork_after_step=1,
        visit_step=2,
        branch_count=2,
        output_path=tmp_path / "cf_train.json",
    )
    assert len(cf_data["records"]) == 6

    ridge_ckpt = train_learned_controller(
        train_records=cf_data["records"],
        model_type="ridge",
        controller_id="test_ridge",
    )
    mlp_ckpt = train_learned_controller(
        train_records=cf_data["records"],
        model_type="mlp",
        controller_id="test_mlp",
        mlp_epochs=40,
    )

    ckpt_path = tmp_path / "ridge.json"
    ridge_ckpt.save(ckpt_path)
    loaded = LearnedControllerCheckpoint.load(ckpt_path)
    assert loaded.controller_id == "test_ridge"

    # Verify scheduler integration for EXIT_2, H_budget, E3_ridge, E3_mlp
    input_ids = torch.tensor([[10, 20, 30, 40]])
    res_exit = run_scheduled_trajectory(
        adapter=adapter,
        input_ids=input_ids,
        method="EXIT_2",
        recurrent_steps=4,
        fork_after_step=1,
        branch_count=2,
        excursion_config=ExcursionConfig(visit_step=2),
    )
    # EXIT at step 2 uses 1 shared call (step 1) + 2 branch calls (step 2) = 3 calls
    assert res_exit.recurrent_block_calls == 3
    assert res_exit.exited_early_step == 2

    res_ridge = run_scheduled_trajectory(
        adapter=adapter,
        input_ids=input_ids,
        method="E3_ridge",
        recurrent_steps=4,
        fork_after_step=1,
        branch_count=2,
        excursion_config=ExcursionConfig(visit_step=2),
        controller_checkpoint=loaded,
    )
    assert res_ridge.recurrent_block_calls in (3, 7, 8)
    assert res_ridge.controller_calls == 1

    res_mlp = run_scheduled_trajectory(
        adapter=adapter,
        input_ids=input_ids,
        method="E3_mlp",
        recurrent_steps=4,
        fork_after_step=1,
        branch_count=2,
        excursion_config=ExcursionConfig(visit_step=2),
        controller_checkpoint=mlp_ckpt,
    )
    assert res_mlp.recurrent_block_calls in (3, 7, 8)
