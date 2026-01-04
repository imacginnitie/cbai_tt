"""
Simple character-level tokenizer for addition task.
Vocabulary: 0-9 (digits), +, =, space (13 tokens total)
"""

import os
from typing import List, Optional

from transformers import PreTrainedTokenizer


class CharTokenizer(PreTrainedTokenizer):
    """Simple character-level tokenizer for addition problems."""

    def __init__(self, **kwargs):
        # Define vocabulary: digits 0-9, operators +, =, and space
        vocab = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "=", " "]

        # Create mappings
        self.char_to_id = {char: idx for idx, char in enumerate(vocab)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}

        # Initialize parent with special tokens
        # Only set tokens that aren't already in kwargs (for from_pretrained compatibility)
        init_kwargs = {
            "pad_token": " ",
            "eos_token": "=",
            "bos_token": None,
            "unk_token": None,
            "model_max_length": 512,
        }
        # Update with kwargs, but don't override if already set
        for key, value in init_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        super().__init__(**kwargs)

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into characters."""
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Convert a token (character) to its ID."""
        return self.char_to_id.get(token, self.char_to_id[" "])

    def _convert_id_to_token(self, index: int) -> str:
        """Convert an ID to its token (character)."""
        return self.id_to_char.get(index, "")

    def get_vocab(self):
        """Return vocabulary as dict."""
        return self.char_to_id.copy()

    @property
    def vocab_size(self):
        """Return vocabulary size."""
        return len(self.char_to_id)

    def _decode(
        self, token_ids: List[int], skip_special_tokens: bool = False, **kwargs
    ) -> str:
        """Decode token IDs to text."""
        return "".join([self._convert_id_to_token(id) for id in token_ids])

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ):
        """Save vocabulary to file."""
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.txt",
        )
        vocab = list(self.char_to_id.keys())
        with open(vocab_file, "w", encoding="utf-8") as f:
            f.write("\n".join(vocab))
        return (vocab_file,)
