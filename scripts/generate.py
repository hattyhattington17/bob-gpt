
from bob.config import ModelConfig
from bob.tokenizer.tokenizer import Tokenizer

tok = Tokenizer.from_text("abcd")

print(tok.encode("abc")) # [0, 1, 2]
print(tok.vocab_size)
print(tok.decode([0, 1, 2]))  # "abc"

config = ModelConfig.from_yaml("configs/nano.yaml")
print(config.d_head)