"""Finite-state grammars for names and ordered string/number arguments."""

from __future__ import annotations

import json
import string

from pydantic import BaseModel, Field

from src.models import FunctionDefinition


class Grammar(BaseModel):
    """Store character transitions, string-body states and accepting states."""

    transitions: list[dict[str, int]] = Field(
        default_factory=lambda: [dict[str, int]()]
    )
    string_states: set[int] = Field(default_factory=set)
    accepting: set[int] = Field(default_factory=set)


def new_state(grammar: Grammar) -> int:
    """Append an empty state and return its index."""
    grammar.transitions.append({})
    return len(grammar.transitions) - 1


def literal(grammar: Grammar, starts: list[int], text: str) -> list[int]:
    """Connect all starts to one path spelling the complete literal."""
    for character in text:
        target = new_state(grammar)
        for start in starts:
            grammar.transitions[start][character] = target
        starts = [target]
    return starts


def json_string(grammar: Grammar, starts: list[int]) -> list[int]:
    """Append a JSON string, including escapes and four-digit Unicode escapes."""
    body = literal(grammar, starts, '"')[0]
    grammar.string_states.add(body)
    end = literal(grammar, [body], '"')
    escape = literal(grammar, [body], "\\")[0]
    for character in '"\\/bfnrt':
        grammar.transitions[escape][character] = body
    current = literal(grammar, [escape], "u")[0]
    for index in range(4):
        target = body if index == 3 else new_state(grammar)
        grammar.transitions[current] = dict.fromkeys(string.hexdigits, target)
        current = target
    return end


def json_number(grammar: Grammar, starts: list[int]) -> list[int]:
    """Append the JSON number DFA; return only its complete-number states."""
    sign, zero, integer, dot, fraction, exponent, exp_sign, exp_digits = (
        new_state(grammar) for _ in range(8)
    )
    for start in starts:
        grammar.transitions[start]["-"] = sign
    for start in [*starts, sign]:
        grammar.transitions[start]["0"] = zero
        grammar.transitions[start].update(dict.fromkeys("123456789", integer))
    grammar.transitions[integer].update(dict.fromkeys(string.digits, integer))
    for start in [zero, integer]:
        grammar.transitions[start]["."] = dot
    for start in [dot, fraction]:
        grammar.transitions[start].update(dict.fromkeys(string.digits, fraction))
    for start in [zero, integer, fraction]:
        grammar.transitions[start].update(dict.fromkeys("eE", exponent))
    grammar.transitions[exponent].update(dict.fromkeys("+-", exp_sign))
    for start in [exponent, exp_sign, exp_digits]:
        grammar.transitions[start].update(dict.fromkeys(string.digits, exp_digits))
    return [zero, integer, fraction, exp_digits]


def arguments_grammar(definition: FunctionDefinition) -> Grammar:
    """Compile exact ordered keys and supported values into an object grammar."""
    grammar = Grammar()
    starts = literal(grammar, [0], "{")
    for index, (name, parameter) in enumerate(definition.parameters.items()):
        if parameter.type not in {"string", "number"} or parameter.enum is not None:
            raise ValueError("stage 2 supports string/number without enum only")
        prefix = (", " if index else "") + json.dumps(name) + ": "
        starts = literal(grammar, starts, prefix)
        if parameter.type == "string":
            starts = json_string(grammar, starts)
        else:
            starts = json_number(grammar, starts)
    grammar.accepting.update(literal(grammar, starts, "}"))
    return grammar


def choices_grammar(choices: list[str]) -> Grammar:
    """Compile complete alternatives into a trie without prefix ambiguity."""
    grammar = Grammar()
    for choice in choices:
        state = 0
        for character in choice:
            target = grammar.transitions[state].get(character)
            if target is None:
                target = new_state(grammar)
                grammar.transitions[state][character] = target
            state = target
        grammar.accepting.add(state)
    return grammar


def advance(grammar: Grammar, state: int, text: str) -> int | None:
    """Consume every character; reject any invalid suffix or empty token."""
    if not text:
        return None
    for character in text:
        target = grammar.transitions[state].get(character)
        if target is not None:
            state = target
        elif (
            state in grammar.string_states
            and ord(character) >= 0x20
            and not 0xD800 <= ord(character) <= 0xDFFF
        ):
            continue
        else:
            return None
    return state
