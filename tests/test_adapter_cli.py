"""Validation must compare the readout actually selected by the experiment."""

import argparse
import json

import torch

from tebdil import cli
from tebdil.models.ouro_adapter import OuroRecurrentAdapter


def test_cli_validation_uses_native_gate_when_configured(tmp_path, monkeypatch, capsys):
    adapter = OuroRecurrentAdapter.from_mini_config(readout_mode="native_gate")
    adapter.model.early_exit_threshold = 0.5

    class Tokenizer:
        def encode(self, *args, **kwargs):
            return torch.tensor([[1, 4, 9]])

    monkeypatch.setattr(cli, "load_adapter_and_tokenizer", lambda cfg: (
        adapter, Tokenizer(), adapter.model_id, adapter.model_revision,
    ))
    config = tmp_path / "validation.yaml"
    config.write_text("evaluation:\n  num_problems: 1\n")
    assert cli.cmd_validate_adapter(argparse.Namespace(config=str(config))) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["readout_mode"] == "native_gate"
    assert report["all_passed"]
