"""Change only rival selection; preserve E2's scope, radius and return inputs."""
import pytest
import torch
from tebdil.core.perturbations import apply_perturbation


def test_single_competitor_formula_scope_norm_and_rng():
    hidden = torch.tensor([[[7., 8., 9.], [1., 0., 0.]]])
    weights = torch.tensor([[3., 0., 0.], [2., 1., 0.], [1., 0., 10.]])
    before = torch.random.get_rng_state().clone()
    results = []
    for seed in (42, 777):
        result, report = apply_perturbation(hidden, method='competitive_logit_single',
                                          relative_norm=.14, lm_head_weight=weights, seed=seed)
        delta = torch.tensor([-.15, 1., 0.]); delta *= .14 / delta.norm()
        torch.testing.assert_close(result[0, -1], hidden[0, -1] + delta)
        assert torch.equal(result[:, :-1], hidden[:, :-1])
        assert report.actual_relative_norm == pytest.approx(.14)
        results.append(result)
    assert torch.equal(results[0], results[1])
    assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.equal(hidden, torch.tensor([[[7., 8., 9.], [1., 0., 0.]]]))


def test_single_competitor_exact_ties_use_lowest_token_ids():
    h = torch.tensor([[[1., 0., 0.]]])
    w = torch.tensor([[1., 0., 0.], [1., 1., 0.], [1., 0., 1.]])
    result, _ = apply_perturbation(h, method='competitive_logit_single', relative_norm=.1, lm_head_weight=w)
    torch.testing.assert_close(result, torch.tensor([[[1., .1, 0.]]]))


def test_original_mixture_is_unchanged():
    gen = torch.Generator().manual_seed(73)
    hidden = torch.randn(1, 2, 8, generator=gen)
    weights = torch.randn(16, 8, generator=gen)
    values, indices = torch.topk(hidden[:, -1] @ weights.T, k=8, dim=-1)
    alternative = (values[:, 1:5].softmax(-1).unsqueeze(-1) * weights[indices[:, 1:5]]).sum(1)
    d = alternative - weights[indices[:, 0]]
    h = hidden[:, -1]
    radial = ((d * h).sum(-1, keepdim=True) / h.square().sum(-1, keepdim=True)) * h
    d = .85 * (d - radial) + .15 * d
    mask = torch.zeros(1, 2, 1); mask[:, -1] = 1
    noise = d.unsqueeze(1).expand_as(hidden) * mask
    noise = noise * (.14 * float(torch.linalg.vector_norm(hidden * mask)) / float(torch.linalg.vector_norm(noise)))
    expected = hidden + noise
    actual, _ = apply_perturbation(hidden, method='competitive_logit', relative_norm=.14, lm_head_weight=weights)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize('weight,branch', [(None, 1), (torch.ones(1, 3), 1),
                                           (torch.full((3, 3), float('nan')), 1), (torch.eye(3), 2)])
def test_single_competitor_never_falls_back_to_noise(weight, branch):
    with pytest.raises(ValueError):
        apply_perturbation(torch.ones(1, 1, 3), method='competitive_logit_single',
                          lm_head_weight=weight, branch_index=branch)


def test_original_gain_cohort_is_complete_and_unchanged():
    import hashlib
    import json
    from pathlib import Path
    root = Path(__file__).parents[1]
    manifest = json.loads((root / 'configs/e2_direction_original16.json').read_text())
    source = (root / manifest['source_report']).read_bytes()
    assert hashlib.sha256(source).hexdigest() == manifest['source_report_sha256']
    rows = json.loads(source)['deceptive_traps_records']
    assert set(manifest['task_ids']) == {r['problem_id'] for r in rows}
    assert len(manifest['task_ids']) == 16 and len(rows) == 32
    assert manifest['max_new_tokens'] == 3000
    tasks = {t['problem_id']: t for t in json.loads((root / 'configs/deceptive_traps_v1_tasks.json').read_text())['tasks']}
    for pid in manifest['task_ids']:
        content = {k: tasks[pid][k] for k in ('prompt', 'gold_answer')}
        assert hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest() == manifest['task_sha256'][pid]


def test_single_competitor_reaches_cached_excursion_operator():
    from tebdil.core.scheduler import generate_tebdil_cached
    from tebdil.core.state import ExcursionConfig
    from tebdil.models.ouro_adapter import OuroRecurrentAdapter
    adapter = OuroRecurrentAdapter.from_mini_config(vocab_size=32, hidden_size=16,
                                                   intermediate_size=32, num_hidden_layers=1)
    result = generate_tebdil_cached(adapter, None, torch.tensor([[1, 2]]), method='E1',
                                   max_new_tokens=3, perturbation_method='competitive_logit_single',
                                   perturbation_relative_norm=.14,
                                   excursion_config=ExcursionConfig(alpha=.65, beta=.65))
    # Four prefill calls plus seven per token; unused donors stop after the visit.
    assert result['excursion_count'] == 3 and result['recurrent_block_calls'] == 25
    assert all(torch.isfinite(k).all() for k in adapter._active_kv_cache.key_cache)
