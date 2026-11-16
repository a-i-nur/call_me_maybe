"""Argument extraction constrained by the selected function definition."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from src.decoding import generate
from src.grammar import arguments_grammar
from src.io import reject_constant, unique_object
from src.models import FunctionDefinition
from src.selector import chat_prompt

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


def extract_arguments(
    model: Small_LLM_Model,
    definition: FunctionDefinition,
    request: str,
    vocabulary: dict[int, str],
) -> dict[str, Any]:
    """Generate the required argument object without evaluating the function."""
    parameters = ", ".join(
        f"{name}: {parameter.type}"
        for name, parameter in definition.parameters.items()
    )
    prompt = chat_prompt(
        f"Function: {definition.name}({parameters})\n"
        f"Purpose: {definition.description}\n"
        "Return a JSON object containing only the inputs needed to call this "
        "function for the user. Extract the operands from the request; the "
        "request itself is not an operand. Do not execute the function. "
        "Decode JSON string literals and preserve their values.",
        request,
    )
    text = generate(model, prompt, arguments_grammar(definition), vocabulary)
    result: dict[str, Any] = json.loads(
        text, object_pairs_hook=unique_object, parse_constant=reject_constant
    )
    return result
