"""Finite-state grammars for names and ordered, typed JSON arguments."""

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
    """Append a JSON string with valid Unicode scalar escapes."""
    body = literal(grammar, starts, '"')[0]
    grammar.string_states.add(body)
    end = literal(grammar, [body], '"')
    escape = literal(grammar, [body], "\\")[0]
    for character in '"\\/bfnrt':
        grammar.transitions[escape][character] = body
    unicode_start = literal(grammar, [escape], "u")[0]
    hex_digits = string.hexdigits

    def suffix(start: int, count: int, end_state: int) -> None:
        """Connect a fixed number of hexadecimal digits to one state."""
        current = start
        for index in range(count):
            target = end_state if index == count - 1 else new_state(grammar)
            grammar.transitions[current].update(dict.fromkeys(hex_digits, target))
            current = target

    ordinary = new_state(grammar)
    grammar.transitions[unicode_start].update(
        dict.fromkeys("0123456789abcefABCEF", ordinary)
    )
    suffix(ordinary, 3, body)
    d_prefix = new_state(grammar)
    grammar.transitions[unicode_start].update(dict.fromkeys("dD", d_prefix))
    ordinary_d = new_state(grammar)
    grammar.transitions[d_prefix].update(dict.fromkeys("01234567", ordinary_d))
    suffix(ordinary_d, 2, body)
    high = new_state(grammar)
    grammar.transitions[d_prefix].update(dict.fromkeys("89abAB", high))
    after_high = new_state(grammar)
    suffix(high, 2, after_high)
    low_start = literal(grammar, [after_high], "\\u")[0]
    low_d = literal(grammar, [low_start], "d")[0]
    grammar.transitions[low_start]["D"] = low_d
    low_second = new_state(grammar)
    grammar.transitions[low_d].update(dict.fromkeys("cdefCDEF", low_second))
    suffix(low_second, 2, body)
    return end


def json_number(grammar: Grammar, starts: list[int], integer_only: bool) -> list[int]:
    """Append JSON numeric syntax without excluding finite magnitudes."""
    sign, zero, integer = (new_state(grammar) for _ in range(3))
    for start in starts:
        grammar.transitions[start]["-"] = sign
    for start in [*starts, sign]:
        grammar.transitions[start]["0"] = zero
        grammar.transitions[start].update(dict.fromkeys("123456789", integer))
    grammar.transitions[integer].update(dict.fromkeys(string.digits, integer))
    completed = [zero, integer]
    if integer_only:
        return completed
    dot, fraction = new_state(grammar), new_state(grammar)
    for start in completed:
        grammar.transitions[start]["."] = dot
    grammar.transitions[dot].update(dict.fromkeys(string.digits, fraction))
    grammar.transitions[fraction].update(dict.fromkeys(string.digits, fraction))
    completed.append(fraction)
    exponent = new_state(grammar)
    for start in completed:
        grammar.transitions[start].update(dict.fromkeys("eE", exponent))
    exp_sign = new_state(grammar)
    grammar.transitions[exponent].update(dict.fromkeys("+-", exp_sign))
    exp_digits = new_state(grammar)
    for start in [exponent, exp_sign, exp_digits]:
        grammar.transitions[start].update(dict.fromkeys(string.digits, exp_digits))
    return [*completed, exp_digits]


def alternatives(grammar: Grammar, starts: list[int], values: list[str]) -> list[int]:
    """Connect exact JSON literals with shared prefixes from each start."""
    ends = []
    for value in values:
        for start in starts:
            state = start
            for character in value:
                target = grammar.transitions[state].get(character)
                if target is None:
                    target = new_state(grammar)
                    grammar.transitions[state][character] = target
                state = target
            ends.append(state)
    return ends


def arguments_grammar(definition: FunctionDefinition) -> Grammar:
    """Compile exact ordered keys and supported values into an object grammar."""
    grammar = Grammar()
    starts = literal(grammar, [0], "{")
    for index, (name, parameter) in enumerate(definition.parameters.items()):
        prefix = (", " if index else "") + json.dumps(name) + ": "
        starts = literal(grammar, starts, prefix)
        if parameter.enum is not None:
            starts = alternatives(
                grammar, starts,
                [json.dumps(value, ensure_ascii=False) for value in parameter.enum],
            )
        elif parameter.type == "string":
            starts = json_string(grammar, starts)
        elif parameter.type == "boolean":
            starts = alternatives(grammar, starts, ["true", "false"])
        else:
            starts = json_number(grammar, starts, parameter.type == "integer")
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


def advance_bytes(
    grammar: Grammar, state: tuple[int, bytes], fragment: bytes
) -> tuple[int, bytes] | None:
    """Advance on bytes, keeping incomplete UTF-8 until the next token."""
    if not fragment:
        return None
    character_state, pending = state
    for byte in fragment:
        pending += bytes((byte,))
        try:
            character = pending.decode("utf-8")
        except UnicodeDecodeError as error:
            if error.reason == "unexpected end of data" and len(pending) < 4:
                if (character_state not in grammar.string_states
                        and not any(
                            character.encode("utf-8").startswith(pending)
                            for character in grammar.transitions[character_state]
                            if ord(character) >= 128
                        )):
                    return None
                continue
            return None
        target = advance(grammar, character_state, character)
        if target is None:
            return None
        character_state, pending = target, b""
    return character_state, pending
