
import torch

from bob.config import ModelConfig
from bob.model.transformer import Bob
from bob.tokenizer.tokenizer import Tokenizer

tokenizer = Tokenizer.from_text("abcddefg")

print("vocab_size", tokenizer.vocab_size)

config = ModelConfig.from_yaml("configs/nano.yaml")
assert config.vocab_size == tokenizer.vocab_size, "vocab size in config must match tokenizer vocab size"
print("d_head", config.d_head)


input_text = "abcabcab"
token_ids = tokenizer.encode(input_text)
token_ids = torch.tensor([token_ids])  
model = Bob(config)
model.forward(input)