"""Character-level tokenizer mapping characters to IDs."""


class Tokenizer:
    """Character-level tokenizer mapping characters to integer IDs."""
    
    def __init__(self, chars: list[str]) -> None:
        """Initialize with list of all characters in the vocabulary."""
        self._char_to_id = {ch: i for i, ch in enumerate(chars)}
        self._id_to_char = list(chars)  # can index directly by ID to get char

    @classmethod
    def from_text(cls, text: str) -> "Tokenizer":
        """Initialize the tokenizer with a vocab of all unique characters in the string."""
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self._char_to_id)

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token IDs."""
        for ch in text:
            if ch not in self._char_to_id:
                raise ValueError(f"Character '{ch}' not in tokenizer vocabulary")
        return [self._char_to_id[ch] for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        """Decode list of token IDs to text."""
        return "".join(self._id_to_char[i] for i in token_ids)
