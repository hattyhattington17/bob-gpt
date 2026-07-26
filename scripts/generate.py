"""Main script to run the model on a prompt and generate max_new_tokens tokens."""

import argparse
from pathlib import Path

import torch

from bob.config import ModelConfig
from bob.inference.generate import generate
from bob.model.transformer import Bob
from bob.tokenizer.tokenizer import Tokenizer
from bob.training.checkpoint import load_vocab

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "nano.yaml"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"
MAX_NEW_TOKENS = 200


def main() -> None:
    """Run the model on a prompt and generate max_new_tokens tokens."""
    # load args
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    config = ModelConfig.from_yaml(CONFIG_PATH)

    # Load the tokenizer from the saved vocab
    chars = load_vocab(CHECKPOINT_DIR)
    tokenizer = Tokenizer(chars)

    # Load the model and checkpoint
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Bob(config).to(device)
    ckpt = torch.load(CHECKPOINT_DIR / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()  # eval mode - sets self.training = False on all submodules

    # model input length T=len(input_token_ids)
    input_token_ids = tokenizer.encode(args.prompt)
    # batch size B=1
    # generate max_new_tokens tokens after the input sequence
    output_token_ids = generate(
        model, torch.tensor([input_token_ids], dtype=torch.long, device=device), args.max_new_tokens
    )  # (B, T + max_new_tokens)
    print(tokenizer.decode(output_token_ids[0].tolist()))


if __name__ == "__main__":
    main()
