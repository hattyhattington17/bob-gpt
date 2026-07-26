"""Training loop for Bob."""

import logging

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

logger = logging.getLogger(__name__)


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
    # construct dataloaders and tokenizer from the training text, train on max_seq_len tokens
    train_loader, validation_loader, tokenizer = build_dataloaders(
        config.data_path, config.train_split, model_config.max_seq_len, config.batch_size
    )
    # save the tokenizer vocab to the checkpoint directory for later inference
    save_vocab(tokenizer.chars, config.checkpoint_dir)
    # ensure embedding table and LM head are the same size as the tokenizer vocab
    # model must have been trained with the same vocab size as the tokenizer
    if model_config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            f"vocab_size in config ({model_config.vocab_size}) != tokenizer vocab_size "
            f"({tokenizer.vocab_size}). Update configs/nano.yaml vocab_size to match."
        )
    # move model (all trainable parameters) to the device
    model = Bob(model_config).to(device)

    # weight decay applied to all params except embedding and norms
    # norm neutral value is 1
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if p.ndim >= 2 and "embed" not in n and "norm" not in n:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    # attach parameters to AdamW optimizer with weight decay
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
    )

    # resume from best checkpoint if one exists
    start_step, best_validation_loss = load_best_checkpoint(model, optimizer, config.checkpoint_dir)

    # create iterator over training batches
    train_iter = iter(train_loader)

    # training loop
    for step in range(start_step, config.max_steps):
        model.train()
        # at end of dataset create new iterator to start with newly shuffled training batches
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)  # (B, T), (B, T)

        # compute logits over all positions in the input sequence x
        logits = model(x, padding_mask=torch.ones_like(x, dtype=torch.long))  # (B, T, vocab_size)
        logits = logits.flatten(0, 1)  # (B*T, vocab_size)
        # compute mean cross entropy loss between predicted logits and true labels y
        loss = F.cross_entropy(logits, y.flatten(), reduction="mean")  # scalar

        # clear gradients from previous step, otherwise they accumulate
        optimizer.zero_grad()
        # compute gradients for this step, write to parameter .grad attributes
        loss.backward()  # type: ignore[no-untyped-call]
        # scale the gradients down if their norm exceeds config.grad_clip
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        # compute the learning rate for this step and set it on the optimizer
        lr = get_lr(
            step, config.warmup_steps, config.max_steps, config.learning_rate, config.min_lr
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # read each parameter's .grad attribute and update the parameter in place
        optimizer.step()

        # every eval_interval steps, evaluate the model on the eval set and log
        # training and eval loss, learning rate, and a generated sample from the model
        if (step + 1) % config.eval_interval == 0:
            validation_loss = _eval(
                model, validation_loader, config.eval_steps, model_config.vocab_size, device
            )
            sample = _sample(model, tokenizer, validation_loader, device)
            logger.info(
                "step %5d | train_loss %.4f | validation_loss %.4f | lr %.2e | %r",
                step + 1,
                loss.item(),
                validation_loss,
                lr,
                sample,
            )

            # update best checkpoint if validation loss improved
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
    """Evaluate model on the eval set for eval_steps batches and return mean cross-entropy loss.

    Takes mean over all tokens instead of all batches to account for last batch being shorter.

    Args:
        model: Model to evaluate.
        validation_loader: Validation data loader.
        eval_steps: Maximum number of batches to evaluate.
        vocab_size: Vocabulary size for cross-entropy.
        device: Torch device string.

    Returns:
        Mean cross-entropy loss over all eval_steps*B*T tokens.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    # average the loss across eval_steps batches
    with torch.inference_mode():
        for step, (x, y) in enumerate(validation_loader):
            if step >= eval_steps:
                break
            x, y = x.to(device), y.to(device)  # (B, T), (B, T)
            logits = model(
                x, padding_mask=torch.ones_like(x, dtype=torch.long)
            )  # (B, T, vocab_size)
            # compute batch cross entropy loss on validation batch and accumulate it into total_loss
            total_loss += (
                F.cross_entropy(logits.view(-1, vocab_size), y.view(-1)).item() * y.numel()
            )
            total_tokens += y.numel()  # B*T

    return total_loss / total_tokens if total_tokens > 0 else 0.0


def _sample(
    model: Bob,
    tokenizer: Tokenizer,
    validation_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: str,
) -> str:
    """Generate a short sample from the first validation batch for logging.

    Args:
        model: Model to sample from.
        tokenizer: Tokenizer for decoding.
        validation_loader: Used to get a prompt.
        device: Torch device string.

    Returns:
        Decoded generated string.
    """
    model.eval()
    x, _ = next(iter(validation_loader))  # x: (B, T)
    prompt_ids = x[0, :3].unsqueeze(0).to(device)  # (1, 3) — first 3 tokens of first sequence
    output_ids = generate(model, prompt_ids, max_new_tokens=5)
    return tokenizer.decode(output_ids[0].tolist())  # decode first sequence in batch
