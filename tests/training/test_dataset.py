"""Tests for CharDataset and build_dataloaders."""

from pathlib import Path

import torch

from bob.training.dataset import CharDataset, build_dataloaders


def test_char_dataset_len() -> None:
    ids = list(range(10))
    ds = CharDataset(ids, seq_len=3)
    assert len(ds) == 7  # 10 - 3


def test_char_dataset_getitem_x_y_shift() -> None:
    ids = [0, 1, 2, 3, 4]
    ds = CharDataset(ids, seq_len=3)
    x, y = ds[0]
    assert x.tolist() == [0, 1, 2]
    assert y.tolist() == [1, 2, 3]


def test_char_dataset_getitem_last() -> None:
    ids = [0, 1, 2, 3, 4]
    ds = CharDataset(ids, seq_len=3)
    x, y = ds[1]
    assert x.tolist() == [1, 2, 3]
    assert y.tolist() == [2, 3, 4]


def test_char_dataset_tensor_dtype() -> None:
    ds = CharDataset([0, 1, 2, 3], seq_len=2)
    x, y = ds[0]
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_build_dataloaders_returns_tokenizer(tmp_path: Path) -> None:
    text = "hello world " * 100
    data_file = tmp_path / "input.txt"
    data_file.write_text(text)

    train_loader, validation_loader, tokenizer = build_dataloaders(
        str(data_file), train_split=0.9, seq_len=8, batch_size=4
    )

    assert tokenizer.vocab_size == len(set(text))


def test_build_dataloaders_split_sizes(tmp_path: Path) -> None:
    # 1000 chars → 900 train tokens, 100 validation tokens
    text = "abcde" * 200  # 1000 chars, vocab size 5
    data_file = tmp_path / "input.txt"
    data_file.write_text(text)

    train_loader, validation_loader, tokenizer = build_dataloaders(
        str(data_file), train_split=0.9, seq_len=4, batch_size=8
    )

    # train split has 900 tokens → 896 samples (900 - 4)
    assert len(train_loader.dataset) == 896  # type: ignore[arg-type]
    # validation split has 100 tokens → 96 samples (100 - 4)
    assert len(validation_loader.dataset) == 96  # type: ignore[arg-type]


def test_build_dataloaders_batch_shape(tmp_path: Path) -> None:
    text = "abcdefgh" * 50
    data_file = tmp_path / "input.txt"
    data_file.write_text(text)

    train_loader, _, _ = build_dataloaders(
        str(data_file), train_split=0.9, seq_len=8, batch_size=4
    )

    x, y = next(iter(train_loader))
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)


def test_tokenizer_chars_roundtrip() -> None:
    from bob.tokenizer.tokenizer import Tokenizer

    tokenizer = Tokenizer.from_text("hello")
    assert tokenizer.chars == sorted(set("hello"))
