"""Tests for checkpoint save/load and vocab persistence."""

from pathlib import Path

import torch

from bob.training.checkpoint import (
    load_best_checkpoint,
    load_vocab,
    save_best_checkpoint,
    save_vocab,
)


def _make_model_and_optimizer() -> tuple[torch.nn.Linear, torch.optim.Optimizer]:
    model = torch.nn.Linear(4, 4, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    return model, optimizer


def test_save_best_checkpoint_creates_file(tmp_path: Path) -> None:
    model, optimizer = _make_model_and_optimizer()
    save_best_checkpoint(1.5, 42, model, optimizer, str(tmp_path))
    assert (tmp_path / "best.pt").exists()


def test_load_best_checkpoint_restores_step(tmp_path: Path) -> None:
    model, optimizer = _make_model_and_optimizer()
    save_best_checkpoint(1.5, 42, model, optimizer, str(tmp_path))

    model2, optimizer2 = _make_model_and_optimizer()
    step, _ = load_best_checkpoint(model2, optimizer2, str(tmp_path))
    assert step == 42


def test_load_best_checkpoint_restores_validation_loss(tmp_path: Path) -> None:
    model, optimizer = _make_model_and_optimizer()
    save_best_checkpoint(2.75, 10, model, optimizer, str(tmp_path))

    model2, optimizer2 = _make_model_and_optimizer()
    _, validation_loss = load_best_checkpoint(model2, optimizer2, str(tmp_path))
    assert validation_loss == 2.75


def test_load_best_checkpoint_restores_weights(tmp_path: Path) -> None:
    model, optimizer = _make_model_and_optimizer()
    torch.nn.init.constant_(model.weight, 1.23)
    save_best_checkpoint(0.9, 10, model, optimizer, str(tmp_path))

    model2, optimizer2 = _make_model_and_optimizer()
    torch.nn.init.constant_(model2.weight, 0.0)
    load_best_checkpoint(model2, optimizer2, str(tmp_path))




def test_save_best_checkpoint_overwrites(tmp_path: Path) -> None:
    model, optimizer = _make_model_and_optimizer()
    save_best_checkpoint(1.5, 100, model, optimizer, str(tmp_path))
    save_best_checkpoint(1.2, 200, model, optimizer, str(tmp_path))

    model2, optimizer2 = _make_model_and_optimizer()
    step, validation_loss = load_best_checkpoint(model2, optimizer2, str(tmp_path))
    assert step == 200
    assert validation_loss == 1.2


def test_load_best_checkpoint_no_checkpoint(tmp_path: Path) -> None:
    model, optimizer = _make_model_and_optimizer()
    step, validation_loss = load_best_checkpoint(model, optimizer, str(tmp_path))
    assert step == 0
    assert validation_loss == float("inf")


def test_load_best_checkpoint_dir_missing(tmp_path: Path) -> None:
    model, optimizer = _make_model_and_optimizer()
    missing_dir = str(tmp_path / "nonexistent")
    step, validation_loss = load_best_checkpoint(model, optimizer, missing_dir)
    assert step == 0
    assert validation_loss == float("inf")


def test_save_and_load_vocab(tmp_path: Path) -> None:
    chars = ["!", " ", "a", "b", "c"]
    save_vocab(chars, str(tmp_path))
    assert (tmp_path / "vocab.json").exists()

    loaded = load_vocab(str(tmp_path))
    assert loaded == chars
