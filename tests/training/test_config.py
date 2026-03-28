"""Tests for TrainingConfig."""

import dataclasses
from pathlib import Path

import pytest

from bob.config import TrainingConfig


def test_training_config_from_yaml(tmp_path: Path) -> None:
    yaml_content = """
model:
  vocab_size: 26
  d_model: 48
  n_heads: 3
  n_layers: 3
  d_ff: 192
  max_seq_len: 10

training:
  data_path: data/input.txt
  train_split: 0.9
  batch_size: 32
  max_steps: 5000
  warmup_steps: 100
  learning_rate: 3.0e-4
  min_lr: 3.0e-5
  weight_decay: 0.1
  grad_clip: 1.0
  eval_interval: 500
  eval_steps: 50
  checkpoint_interval: 1000
  checkpoint_dir: checkpoints
"""
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)

    config = TrainingConfig.from_yaml(config_file)

    assert config.data_path == "data/input.txt"
    assert config.train_split == 0.9
    assert config.batch_size == 32
    assert config.max_steps == 5000
    assert config.warmup_steps == 100
    assert config.learning_rate == pytest.approx(3e-4)
    assert config.min_lr == pytest.approx(3e-5)
    assert config.weight_decay == pytest.approx(0.1)
    assert config.grad_clip == pytest.approx(1.0)
    assert config.eval_interval == 500
    assert config.eval_steps == 50
    assert config.checkpoint_interval == 1000
    assert config.checkpoint_dir == "checkpoints"


def test_training_config_is_frozen(tmp_path: Path) -> None:
    yaml_content = """
model:
  vocab_size: 26
  d_model: 48
  n_heads: 3
  n_layers: 3
  d_ff: 192
  max_seq_len: 10

training:
  data_path: data/input.txt
  train_split: 0.9
  batch_size: 32
  max_steps: 5000
  warmup_steps: 100
  learning_rate: 3.0e-4
  min_lr: 3.0e-5
  weight_decay: 0.1
  grad_clip: 1.0
  eval_interval: 500
  eval_steps: 50
  checkpoint_interval: 1000
  checkpoint_dir: checkpoints
"""
    config_file = tmp_path / "test.yaml"
    config_file.write_text(yaml_content)
    config = TrainingConfig.from_yaml(config_file)

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.batch_size = 64  # type: ignore[misc]
