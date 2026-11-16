"""Public SDK access and a conservative token-text vocabulary."""

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


def token_texts(model: Small_LLM_Model) -> dict[int, str]:
    """Decode base vocabulary IDs through the public SDK, excluding lossy text.

    Partial UTF-8 tokens decode to replacement characters in isolation. Until
    byte-aware transitions are implemented, exclude these rather than corrupt
    text. Complete Unicode tokens and JSON Unicode escapes remain available.
    Special tokens are absent from the base vocabulary or decode to empty text.
    """
    try:
        with open(model.get_path_to_vocab_file(), encoding="utf-8") as source:
            vocabulary = json.load(source)
        if not isinstance(vocabulary, dict) or any(
            type(token_id) is not int or token_id < 0
            for token_id in vocabulary.values()
        ):
            raise ValueError("expected a token-to-integer vocabulary")
        texts: dict[int, str] = {}
        for token_id in sorted(set(vocabulary.values())):
            text = model.decode([token_id])
            if text and "\ufffd" not in text:
                texts[token_id] = text
        if not texts:
            raise ValueError("no losslessly decoded tokens")
        return texts
    except Exception as error:
        raise InputError(f"cannot read SDK token vocabulary: {error}") from error
