
from bob.tokenizer.tokenizer import Tokenizer

tok = Tokenizer.from_text("abcd")

print(tok.encode("abc")) # [0, 1, 2]
print(tok.vocab_size)
print(tok.decode([0, 1, 2]))  # "abc"