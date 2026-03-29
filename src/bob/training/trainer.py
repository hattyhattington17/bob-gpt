"""Training loop for Bob."""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bob.config import ModelConfig, TrainingConfig
from bob.inference.generate import generate
from bob.model.transformer import Bob
from bob.tokenizer.tokenizer import Tokenizer
from bob.training.checkpoint import load_best_checkpoint, save_best_checkpoint, save_vocab
from bob.training.dataset import build_dataloaders
from bob.training.schedule import get_lr


def train(model_config: ModelConfig, config: TrainingConfig, device: str) -> None:
    """Run the training loop.

    Builds dataloaders, saves vocab, constructs model + optimizer, resumes from
    the best checkpoint if one exists, then trains for config.max_steps steps.
    Logs train/validation loss and a generated sample every eval_interval steps.
    Saves a new best checkpoint whenever validation loss improves.

    Args:
        model_config: Model architecture config.
        config: Training hyperparameters.
        device: Torch device string, e.g. "cpu" or "mps".
    """
    train_loader, validation_loader, tokenizer = build_dataloaders(
        config.data_path, config.train_split, model_config.max_seq_len, config.batch_size
    )
    save_vocab(tokenizer.chars, config.checkpoint_dir)

    assert model_config.vocab_size == tokenizer.vocab_size, (
        f"vocab_size in config ({model_config.vocab_size}) != tokenizer vocab_size "
        f"({tokenizer.vocab_size}). Update configs/nano.yaml vocab_size to match."
    )

    model = Bob(model_config).to(device)

    decay_params = [
        p
        for n, p in model.named_parameters()
        if p.ndim >= 2 and "embed" not in n and "norm" not in n
    ]
    # weight decay on embeddings reduces expressiveness of token representations
    no_decay_params = [
        p
        for n, p in model.named_parameters()
        if not (p.ndim >= 2 and "embed" not in n and "norm" not in n)
    ]
    # tell optimizer which parameters to apply weight decay to
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )

    start_step, best_validation_loss = load_best_checkpoint(model, optimizer, config.checkpoint_dir)
    train_iter = itertools.cycle(train_loader)

    for step in range(start_step, config.max_steps):
        model.train()
        # load a training pair to the gpu
        x, y = next(train_iter)
        x, y = x.to(device), y.to(device)  # (B, T), (B, T)

        logits = model(x)  # (B, T, vocab_size)
        loss = F.cross_entropy(logits.view(-1, model_config.vocab_size), y.view(-1))  # (B*T,)

        # clear gradients from the previous step out of the optimizer
        optimizer.zero_grad()
        # compute gradients and write them into the parameters' .grad attributes 
        # for optimizer.step() to read
        loss.backward()  # type: ignore[no-untyped-call]
        # scale the gradients down if their norm exceeds config.grad_clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        # compute the learning rate for this step and set it on the optimizer
        lr = get_lr(
            step, config.warmup_steps, config.max_steps, config.learning_rate, config.min_lr
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # read the gradients and take an optimization step
        optimizer.step()

        # check if it's time to evaluate on the validation set and log progress
        if (step + 1) % config.eval_interval == 0:
            validation_loss = _eval(
                model, validation_loader, config.eval_steps, model_config.vocab_size, device
            )
            sample = _sample(model, tokenizer, validation_loader, model_config, device)
            print(
                f"step {step + 1:5d} | train_loss {loss.item():.4f} | "
                f"validation_loss {validation_loss:.4f} | lr {lr:.2e} | {sample!r}"
            )
            # store a new checkpoint if validation loss improved
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                save_best_checkpoint(
                    validation_loss, step + 1, model, optimizer, config.checkpoint_dir
                )


def _eval(
    model: Bob,
    validation_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    eval_steps: int,
    vocab_size: int,
    device: str,
) -> float:
    """Evaluate model on validation_loader for up to eval_steps batches.

    Args:
        model: Model to evaluate.
        validation_loader: Validation data loader.
        eval_steps: Maximum number of batches to evaluate.
        vocab_size: Vocabulary size for cross-entropy.
        device: Torch device string.

    Returns:
        Mean cross-entropy loss over evaluated batches.
    """
    # run model in eval mode, disables dropout and other training-specific behavior
    model.eval()
    total_loss = 0.0
    count = 0
    # average the loss across eval_steps batches
    with torch.inference_mode():
        for x, y in validation_loader:
            if count >= eval_steps:
                break
            x, y = x.to(device), y.to(device)  # (B, T), (B, T)
            logits = model(x)  # (B, T, vocab_size)
            # compute batch cross entropy loss on validation batch and accumulate it into total_loss
            total_loss += F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item()  # (B*T,)
            count += 1
    return total_loss / count if count > 0 else 0.0


def _sample(
    model: Bob,
    tokenizer: Tokenizer,
    validation_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    model_config: ModelConfig,
    device: str,
) -> str:
    """Generate a short sample from the first validation batch for logging.

    Args:
        model: Model to sample from.
        tokenizer: Tokenizer for decoding.
        validation_loader: Used to get a prompt.
        model_config: Model config for max_seq_len.
        device: Torch device string.

    Returns:
        Decoded generated string.
    """
    model.eval()
    x, _ = next(iter(validation_loader))  # x: (B, T)
    prompt_ids = x[0, :3].tolist()  # (3,) — first 3 tokens of first sequence
    output_ids = generate(
        model, prompt_ids, max_new_tokens=40, max_seq_len=model_config.max_seq_len, device=device
    )
    return tokenizer.decode(output_ids)
