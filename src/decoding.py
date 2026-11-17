"""Greedy generation with a full-token grammar mask applied before argmax."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Mapping

import numpy as np

from src.grammar import Grammar, advance, advance_bytes
from src.io import InputError

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


MAX_GENERATED_TOKENS = 256
DecodeState = tuple[int, bytes]


def allowed_tokens(
    grammar: Grammar, state: DecodeState, vocabulary: Mapping[int, str | bytes],
    candidates: Iterable[int] | None = None,
) -> dict[int, DecodeState]:
    """Map each fully legal token to the state reached by all its characters."""
    allowed: dict[int, DecodeState] = {}
    token_ids = vocabulary if candidates is None else candidates
    for token_id in token_ids:
        fragment = vocabulary[token_id]
        if isinstance(fragment, str) and not state[1]:
            character_state = advance(grammar, state[0], fragment)
            target = ((character_state, b"") if character_state is not None
                      else None)
        else:
            token_bytes = (fragment.encode("utf-8") if isinstance(fragment, str)
                           else fragment)
            target = advance_bytes(grammar, state, token_bytes)
        if target is not None:
            allowed[token_id] = target
    return allowed


def masked_argmax(logits: list[float], allowed: dict[int, DecodeState]) -> int:
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
    vocabulary: Mapping[int, str | bytes],
    max_tokens: int = MAX_GENERATED_TOKENS,
) -> str:
    """Generate one complete grammar value or fail without returning a prefix."""
    try:
        context: list[int] = model.encode(prompt).tolist()[0]
        generated: list[int] = []
        output = bytearray()
        state: DecodeState = (0, b"")
        masks: dict[DecodeState, dict[int, DecodeState]] = {}
        first_char: dict[str, list[int]] = {}
        for token_id, fragment in vocabulary.items():
            if not fragment:
                continue
            if isinstance(fragment, str):
                initial = fragment[0]
            else:
                initial = chr(fragment[0]) if fragment[0] < 128 else "\0"
            first_char.setdefault(initial, []).append(token_id)
        for _ in range(max_tokens):
            if state not in masks:
                candidates = None
                if not state[1] and state[0] not in grammar.string_states:
                    next_characters = grammar.transitions[state[0]]
                    candidates = [token_id for character in next_characters
                                  for token_id in first_char.get(character, [])]
                    if any(ord(character) >= 128 for character in next_characters):
                        candidates.extend(first_char.get("\0", []))
                masks[state] = allowed_tokens(grammar, state, vocabulary, candidates)
            allowed = masks[state]
            if not allowed:
                raise InputError("empty token mask: no grammar continuation")
            logits = model.get_logits_from_input_ids(context + generated)
            token_id = masked_argmax(logits, allowed)
            generated.append(token_id)
            fragment = vocabulary[token_id]
            output.extend(
                fragment.encode("utf-8") if isinstance(fragment, str) else fragment
            )
            state = allowed[token_id]
            if not state[1]:
                text = output.decode("utf-8")
                if model.decode(generated) != text:
                    raise InputError("SDK token text is not compositional")
                if state[0] in grammar.accepting:
                    return text
        raise InputError(f"generation exceeded the {max_tokens}-token limit")
    except InputError:
        raise
    except Exception as error:
        raise InputError(f"SDK generation failed: {error}") from error
