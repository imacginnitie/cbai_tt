"""
Simple character-level tokenizer for addition task.
Vocabulary: 0-9 (digits), +, =, space (13 tokens total)
"""

import json
import os
from typing import List, Optional, Union


class CharTokenizer:
    """Simple character-level tokenizer for addition problems."""

    def __init__(self):
        # Define vocabulary: digits 0-9, operators +, =, and space
        self.vocab = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "=", " "]
        # Create char to id mapping
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        self.vocab_size = len(self.vocab)

        # Add special tokens for compatibility (not used but needed for interface)
        self.pad_token = " "
        self.pad_token_id = self.char_to_id[self.pad_token]
        self.eos_token = "="
        self.eos_token_id = self.char_to_id[self.eos_token]
        self.bos_token = None
        self.bos_token_id = None

    def __len__(self):
        return self.vocab_size

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], List[List[int]]]:
        """Encode text to token IDs."""
        token_ids = [self.char_to_id.get(char, self.pad_token_id) for char in text]

        if return_tensors == "pt":
            import torch

            return torch.tensor([token_ids])
        elif return_tensors == "np":
            import numpy as np

            return np.array([token_ids])

        return token_ids

    def decode(
        self,
        token_ids: Union[List[int], List[List[int]]],
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode token IDs to text."""
        if isinstance(token_ids[0], list):
            # Handle batch
            return [self.decode(ids, skip_special_tokens) for ids in token_ids]

        text = "".join([self.id_to_char.get(id, "") for id in token_ids])
        return text

    def __call__(self, text: Union[str, List[str]], **kwargs):
        """Make tokenizer callable like HuggingFace tokenizers."""
        if isinstance(text, list):
            return self.batch_encode_plus(text, **kwargs)
        else:
            return self.encode_plus(text, **kwargs)

    def encode_plus(self, text: str, return_tensors: Optional[str] = None, **kwargs):
        """Encode text and return dict with input_ids."""
        token_ids = self.encode(text, return_tensors=return_tensors)
        return {"input_ids": token_ids}

    def batch_encode_plus(
        self, texts: List[str], return_tensors: Optional[str] = None, **kwargs
    ):
        """Encode batch of texts."""
        input_ids = [self.encode(text) for text in texts]

        if return_tensors == "pt":
            import torch

            # Pad to same length
            max_len = max(len(ids) for ids in input_ids)
            padded = []
            for ids in input_ids:
                padded_ids = ids + [self.pad_token_id] * (max_len - len(ids))
                padded.append(padded_ids)
            return {"input_ids": torch.tensor(padded)}
        elif return_tensors == "np":
            import numpy as np

            max_len = max(len(ids) for ids in input_ids)
            padded = []
            for ids in input_ids:
                padded_ids = ids + [self.pad_token_id] * (max_len - len(ids))
                padded.append(padded_ids)
            return {"input_ids": np.array(padded)}

        return {"input_ids": input_ids}

    def save_pretrained(self, save_directory: str):
        """Save tokenizer to directory."""
        os.makedirs(save_directory, exist_ok=True)
        tokenizer_config = {
            "vocab": self.vocab,
            "vocab_size": self.vocab_size,
            "pad_token": self.pad_token,
            "eos_token": self.eos_token,
        }
        with open(os.path.join(save_directory, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer_config, f, indent=2)

    @classmethod
    def from_pretrained(cls, save_directory: str):
        """Load tokenizer from directory."""
        tokenizer = cls()
        config_path = os.path.join(save_directory, "tokenizer_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            # Verify vocab matches
            if config["vocab"] != tokenizer.vocab:
                raise ValueError("Vocabulary mismatch when loading tokenizer")
        return tokenizer
