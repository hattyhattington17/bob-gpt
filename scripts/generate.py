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
    # load args
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    args = parser.parse_args()

    config = ModelConfig.from_yaml(CONFIG_PATH)

    # Load the tokenizer from the saved vocab
    chars = load_vocab("checkpoints")
    tokenizer = Tokenizer(chars)
    input_token_ids = tokenizer.encode(args.prompt)

    # Load the model and checkpoint
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Bob(config).to(device)
    ckpt = torch.load(Path("checkpoints") / "best.pt", weights_only=False, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()  # eval mode - disable dropout

    # generate tokens after the input prompt - B=1
    output_token_ids = generate(
        model, torch.tensor([input_token_ids], dtype=torch.long).to(device), args.max_new_tokens
    )  # (B, T)
    print(tokenizer.decode(output_token_ids[0].tolist()))


if __name__ == "__main__":
    main()
