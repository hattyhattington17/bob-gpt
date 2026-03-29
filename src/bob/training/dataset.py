"""Character-level dataset and dataloader construction."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from bob.tokenizer.tokenizer import Tokenizer


class CharDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset of (x, y) pairs for next-token prediction.

    Each item is a sequence x of length seq_len and target y = x shifted right by one.
    """

    def __init__(self, token_ids: list[int], seq_len: int) -> None:
        """Initialize with token IDs and sequence length."""
        self._ids = torch.tensor(token_ids, dtype=torch.long)
        self._seq_len = seq_len

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self._ids) - self._seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (x, y) pair at index idx."""
        x = self._ids[idx : idx + self._seq_len]
        y = self._ids[idx + 1 : idx + self._seq_len + 1]
        return x, y


def build_dataloaders(
    data_path: str,
    train_split: float,
    seq_len: int,
    batch_size: int,
) -> tuple[
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    DataLoader[tuple[torch.Tensor, torch.Tensor]],
    Tokenizer,
]:
    """Read text, tokenize, split into train/validation sets, return dataloaders and tokenizer.

    Args:
        data_path: Path to text file.
        train_split: Fraction of tokens used for training (e.g. 0.9).
        seq_len: Sequence length for each sample.
        batch_size: Batch size for both loaders.

    Returns:
        Tuple of (train_loader, validation_loader, tokenizer).
    """
    text = Path(data_path).read_text()
    tokenizer = Tokenizer.from_text(text)
    ids = tokenizer.encode(text)

    split = int(len(ids) * train_split)
    train_ds = CharDataset(ids[:split], seq_len)
    validation_ds = CharDataset(ids[split:], seq_len)

    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    validation_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        validation_ds, batch_size=batch_size, shuffle=False
    )
    return train_loader, validation_loader, tokenizer
