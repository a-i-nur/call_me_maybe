"""Public SDK access and lossless ByteLevel token bytes."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from src.io import InputError

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


def load_model() -> Small_LLM_Model:
    """Create one default Qwen model, keeping SDK imports off the empty path."""
    from llm_sdk import Small_LLM_Model

    try:
        return Small_LLM_Model()
    except Exception as error:
        raise InputError(f"cannot load SDK model: {error}") from error


def byte_alphabet() -> dict[str, int]:
    """Reconstruct the reversible alphabet used by ByteLevel tokenizers."""
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    byte_values = visible + [value for value in range(256) if value not in visible]
    symbols = visible + list(range(256, 256 + 256 - len(visible)))
    return {chr(symbol): value for value, symbol in zip(byte_values, symbols)}


def token_texts(model: Small_LLM_Model) -> dict[int, str | bytes]:
    """Read public tokenizer data and recover bytes, including UTF-8 fragments."""
    try:
        with open(model.get_path_to_tokenizer_file(), encoding="utf-8") as source:
            tokenizer = json.load(source)
        if tokenizer.get("decoder", {}).get("type") != "ByteLevel":
            raise ValueError("expected a ByteLevel tokenizer decoder")
        vocabulary = tokenizer["model"]["vocab"]
        if not isinstance(vocabulary, dict) or any(
            type(token_id) is not int or token_id < 0
            for token_id in vocabulary.values()
        ):
            raise ValueError("expected a token-to-integer vocabulary")
        alphabet = byte_alphabet()
        texts: dict[int, str | bytes] = {}
        for token, token_id in vocabulary.items():
            fragment = bytes(alphabet[character] for character in token)
            try:
                texts[token_id] = fragment.decode("utf-8")
            except UnicodeDecodeError:
                texts[token_id] = fragment
        if not texts:
            raise ValueError("empty tokenizer vocabulary")
        return texts
    except Exception as error:
        raise InputError(f"cannot read SDK token vocabulary: {error}") from error
