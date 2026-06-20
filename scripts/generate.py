"""Main script to run the model and generate text from a prompt."""

import argparse
from pathlib import Path

import torch

from bob.config import ModelConfig
from bob.inference.generate import generate
from bob.model.transformer import Bob
from bob.tokenizer.tokenizer import Tokenizer
from bob.training.checkpoint import load_vocab

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "nano.yaml"
MAX_NEW_TOKENS = 200


def main() -> None:
    """Main function to run the script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    config = ModelConfig.from_yaml(CONFIG_PATH)

    chars = load_vocab(args.checkpoint_dir)
    tokenizer = Tokenizer(chars)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Bob(config).to(device)

    ckpt = torch.load(
        Path(args.checkpoint_dir) / "best.pt", weights_only=False, map_location=device
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    token_ids = tokenizer.encode(args.prompt)
    output_ids = generate(model, token_ids, MAX_NEW_TOKENS, config.max_seq_len, device)
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
