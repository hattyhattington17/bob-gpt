"""
Character-level tokenizer mapping characters to IDs.
"""

class Tokenizer:
    
    def __init__(self, chars: list[str]) -> None:
        """initialize with list of all characters in the vocabulary"""
        self._char_to_id = {ch: i for i, ch in enumerate(chars)}
        self._id_to_char = {i: ch for i, ch in enumerate(chars)}

    @classmethod
    def from_text(cls, text: str) -> "Tokenizer":
        """initialize the tokenizer with a vocab of all unique characters in the string"""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self._char_to_id)

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token IDs."""
        return [self._char_to_id[ch] for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        """Decode list of token IDs to text."""
        return "".join(self._id_to_char[i] for i in token_ids)
