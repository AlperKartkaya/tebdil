"""Validate transferred parity evidence, not efficacy or detector accuracy."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[1] / 'reports/geometry_shadow'


def test_transferred_geometry_shadow_evidence_is_consistent():
    manifest = json.loads((ROOT / 'transfer.json').read_text())
    path = ROOT / 'ouro_1_4b_fp32.json'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['sha256'][path.name]
    report = json.loads(path.read_text())
    assert report['all_passed'] and all(report['checks'].values())
    assert report['model_id'] == 'ByteDance/Ouro-1.4B-Thinking'
    assert report['model_revision'] == '3aaa2224253a92ca45cf2e3d427c360e1ef9c93d'
    assert report['git_revision'] == '4764e1514ceec0bbdaa4dd60283783316153b784'
    assert report['dtype'] == 'torch.float32' and report['attention'] == 'sdpa'
    assert report['gpu'] == 'NVIDIA L4'
    assert len(report['source_sha256']) == 54
    assert all(len(digest) == 64 and int(digest, 16) >= 0 for digest in report['source_sha256'].values())
    native, shadow = report['runs']
    assert native['generated_token_ids'] == shadow['generated_token_ids']
    assert len(native['generated_token_ids']) == report['budget'] == 64
    assert report['final_logits_sha256'][0] == report['final_logits_sha256'][1]
    assert len(report['final_logits_sha256'][0]) == 64
    assert report['committed_kv_sha256'][0] == report['committed_kv_sha256'][1]
    assert len(report['committed_kv_sha256'][0]) > 0
    assert native['recurrent_block_calls'] == shadow['recurrent_block_calls'] == 260
    for run in (native, shadow):
        assert run['excursion_count'] == 0 and not run['excursion_token_indices']
        assert run['stop_reason'] == 'max_output_tokens' and not run['completed']
    rows = shadow['geometry_shadow']['records']
    assert len(rows) == shadow['geometry_shadow']['extra_readout_projections'] == 64
    start = len(report['prompt_ids'][0]) - 1
    for i, row in enumerate(rows):
        assert row['token_position'] == start + i and row['boundary'] == 3
        assert row['reference_count'] == min(i, 32)
        assert row['bank_positions'] == list(range(start + max(0, i - 64), start + i))
        assert row['selected'] is None and not row['eligible'] and not row['attempted']
        assert row['motion']['reason'] == ('insufficient_or_invalid_geometry' if i < 8 else 'progress')
    assert report['retraction_max_abs_error'] < 2e-5
    assert report['retraction_reference_calls'] == 3
