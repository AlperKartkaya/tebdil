import torch
from tebdil.core import scheduler
from tebdil.models.ouro_adapter import OuroRecurrentAdapter


def test_p0_commits_selected_donor_history(monkeypatch):
    adapter = OuroRecurrentAdapter.from_mini_config(vocab_size=128)
    adapter.initialize_cached_step = adapter.initialize
    committed = []
    adapter.commit_cached_step = lambda states: committed.extend(states)
    def readout(adapter, state):
        logits = torch.zeros(1, 1, 128)
        logits[..., 1] = 1 if state.branch_id == 'A' else 10
        return logits
    monkeypatch.setattr(scheduler, '_fast_readout', readout)
    result = scheduler.run_scheduled_trajectory(adapter, torch.tensor([[1, 2]]), method='P0', recurrent_steps=4, fork_after_step=1, branch_count=2, use_cached_step=True, record_snapshots=False, record_pca=False)
    assert result.primary_state.branch_id != 'A'
    assert len(committed) == 4
    assert all(s.branch_id == result.primary_state.branch_id for s in committed[1:])
    assert committed[-1] is result.primary_state


def test_window_cooldown_and_final_phase():
    state = scheduler.ExcursionWindowState(cooldown_remaining=1)
    assert state.select('E2', False) == ('S0', False)
    assert state.select('E2', False) == ('E2', False)
    state.intervene(False, 3)
    assert state.select('E2', False) == ('E1', True)
    state.intervene(True, 3)
    assert state.select('E2', False) == ('E1', True)
    assert state.select('E2', False) == ('S0', False)
    state.close_thinking()
    assert state.select('E2', False) == ('S0', False)
    assert state.select('E1', True) == ('E1', False)
