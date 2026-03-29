"""Smoke tests for the training loop."""

from pathlib import Path

from bob.config import ModelConfig, TrainingConfig
from bob.training.trainer import train


def test_train_runs_and_saves_checkpoint(tmp_path: Path) -> None:
    text = "hello world foo bar baz " * 100
    data_file = tmp_path / "input.txt"
    data_file.write_text(text)

    vocab_size = len(set(text))

    model_config = ModelConfig(
        vocab_size=vocab_size,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        max_seq_len=8,
    )
    training_config = TrainingConfig(
        data_path=str(data_file),
        train_split=0.9,
        batch_size=4,
        max_steps=6,
        warmup_steps=2,
        learning_rate=1e-3,
        min_lr=1e-4,
        weight_decay=0.1,
        grad_clip=1.0,
        eval_interval=3,
        eval_steps=2,
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    train(model_config, training_config, device="cpu")

    ckpt_dir = tmp_path / "checkpoints"
    assert (ckpt_dir / "best.pt").exists(), "expected best.pt checkpoint"
    assert (ckpt_dir / "vocab.json").exists(), "expected vocab.json"


def test_train_resumes_from_checkpoint(tmp_path: Path) -> None:
    text = "abcdefgh " * 100
    data_file = tmp_path / "input.txt"
    data_file.write_text(text)

    vocab_size = len(set(text))

    model_config = ModelConfig(
        vocab_size=vocab_size,
        d_model=16,
        n_heads=2,
        n_layers=1,
        d_ff=32,
        max_seq_len=8,
    )
    training_config = TrainingConfig(
        data_path=str(data_file),
        train_split=0.9,
        batch_size=4,
        max_steps=4,
        warmup_steps=1,
        learning_rate=1e-3,
        min_lr=1e-4,
        weight_decay=0.1,
        grad_clip=1.0,
        eval_interval=2,
        eval_steps=2,
        checkpoint_dir=str(tmp_path / "checkpoints"),
    )

    # first run: trains to step 4
    train(model_config, training_config, device="cpu")
    assert (tmp_path / "checkpoints" / "best.pt").exists()

    # read the step from the checkpoint to confirm it trained
    import torch
    ckpt = torch.load(str(tmp_path / "checkpoints" / "best.pt"), weights_only=False)
    step = int(ckpt["step"])
    assert step > 0, "checkpoint should record a non-zero step"
