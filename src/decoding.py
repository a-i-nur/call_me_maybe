"""Greedy generation with a full-token grammar mask applied before argmax."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.grammar import Grammar, advance
from src.io import InputError

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


MAX_GENERATED_TOKENS = 256


def allowed_tokens(
    grammar: Grammar, state: int, vocabulary: dict[int, str]
) -> dict[int, int]:
    """Map each fully legal token to the state reached by all its characters."""
    allowed: dict[int, int] = {}
    for token_id, text in vocabulary.items():
        target = advance(grammar, state, text)
        if target is not None:
            allowed[token_id] = target
    return allowed


def masked_argmax(logits: list[float], allowed: dict[int, int]) -> int:
    """Mask prohibited IDs to minus infinity before deterministic selection."""
    scores = np.asarray(logits, dtype=float)
    if scores.ndim != 1 or np.isnan(scores).any() or np.isposinf(scores).any():
        raise InputError("model returned invalid logits")
    if any(token_id < 0 or token_id >= len(scores) for token_id in allowed):
        raise InputError("SDK vocabulary does not match logits")
    masked = np.full(scores.shape, -np.inf)
    indices = list(allowed)
    masked[indices] = scores[indices]
    if not np.isfinite(masked).any():
        raise InputError("empty token mask: no legal token with a finite logit")
    return int(np.argmax(masked))


def generate(
    model: Small_LLM_Model,
    prompt: str,
    grammar: Grammar,
    vocabulary: dict[int, str],
    max_tokens: int = MAX_GENERATED_TOKENS,
) -> str:
    """Generate one complete grammar value or fail without returning a prefix."""
    try:
        context: list[int] = model.encode(prompt).tolist()[0]
        generated: list[int] = []
        text = ""
        state = 0
        masks: dict[int, dict[int, int]] = {}
        for _ in range(max_tokens):
            if state not in masks:
                masks[state] = allowed_tokens(grammar, state, vocabulary)
            allowed = masks[state]
            if not allowed:
                raise InputError("empty token mask: no grammar continuation")
            logits = model.get_logits_from_input_ids(context + generated)
            token_id = masked_argmax(logits, allowed)
            generated.append(token_id)
            text += vocabulary[token_id]
            state = allowed[token_id]
            # Detect tokenizer context effects instead of silently changing text.
            if model.decode(generated) != text:
                raise InputError("SDK token text is not compositional")
            if state in grammar.accepting:
                return text
        raise InputError(f"generation exceeded the {max_tokens}-token limit")
    except InputError:
        raise
    except Exception as error:
        raise InputError(f"SDK generation failed: {error}") from error
