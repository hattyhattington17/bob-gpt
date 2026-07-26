"""Character-level dataset and dataloader construction."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from bob.tokenizer.tokenizer import Tokenizer


class CharDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset of (x, y) training pairs for next-token prediction.

    x and y have shape (seq_len), y is x shifted left by one.
        ex: if seq_len=5 and the text is "hello world", then one sample would be:
            x = [h, e, l, l, o]
            y = [e, l, l, o,  ]
    """

    def __init__(self, token_ids: list[int], seq_len: int) -> None:
        """Init with token ids of training or eval split of the corpus and batch sequence length."""
        self._ids = torch.tensor(token_ids, dtype=torch.long)
        self._seq_len = seq_len

    def __len__(self) -> int:
        """Return number of training pairs in the dataset."""
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
    """Read text, tokenize, split into train/eval sets, return dataloaders and tokenizer.

    Args:
        data_path: Path to text file.
        train_split: Fraction of tokens used for training (e.g. 0.9).
        seq_len: Sequence length for each sample.
        batch_size: Batch size (number of sequences per batch) for both loaders.

    Returns:
        Tuple of (train_loader, eval_loader, tokenizer).
    """
    # tokenize the text and convert to token ids
    text = Path(data_path).read_text()
    tokenizer = Tokenizer.from_text(text)
    ids = tokenizer.encode(text)

    # split into training and eval sets
    split = int(len(ids) * train_split)
    train_ds = CharDataset(ids[:split], seq_len)
    eval_ds = CharDataset(ids[split:], seq_len)

    # create dataloaders for training and evaluation, shuffle the training set
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    eval_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(
        eval_ds, batch_size=batch_size, shuffle=False
    )
    return train_loader, eval_loader, tokenizer
