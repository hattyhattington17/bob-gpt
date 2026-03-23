import argparse
from pathlib import Path

import torch

from bob.config import ModelConfig
from bob.inference.generate import generate
from bob.model.transformer import Bob
from bob.tokenizer.tokenizer import Tokenizer

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "nano.yaml"
MAX_NEW_TOKENS = 10


def main() -> None:
    """Main function to run the script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=False)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_text("abcdefgh")
    print("initialized tokenizer with vocabulary size:", tokenizer.vocab_size)

    config = ModelConfig.from_yaml(CONFIG_PATH)
    assert config.vocab_size == tokenizer.vocab_size, (
        "vocab size in config must match tokenizer vocab size"
    )

    # move the model to the appropriate device (GPU if available, otherwise CPU)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Bob(config).to(device)

    # run the model in eval mode (disables dropout, etc)
    model.eval()

    # tokenize the prompt
    token_ids = tokenizer.encode(args.prompt or "abc")
    print("input token ids:", token_ids)

    output_ids = generate(model, token_ids, MAX_NEW_TOKENS, config.max_seq_len, device)
    print(tokenizer.decode(output_ids))


if __name__ == "__main__":
    main()
