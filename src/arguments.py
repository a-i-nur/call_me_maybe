"""Argument extraction constrained by the selected function definition."""

from __future__ import annotations

import json
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Mapping

from src.decoding import generate
from src.grammar import arguments_grammar
from src.io import InputError, reject_constant, unique_object
from src.models import FunctionDefinition
from src.selector import chat_prompt

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


def extract_arguments(
    model: Small_LLM_Model,
    definition: FunctionDefinition,
    request: str,
    vocabulary: Mapping[int, str | bytes],
) -> dict[str, Any]:
    """Generate the required argument object without evaluating the function."""
    parameters = ", ".join(
        f"{name}: {parameter.type}"
        + (f" allowed {json.dumps(parameter.enum, ensure_ascii=False)}"
           if parameter.enum is not None else "")
        for name, parameter in definition.parameters.items()
    )
    prompt = chat_prompt(
        f"Function: {definition.name}({parameters})\n"
        f"Purpose: {definition.description}\n"
        "Return only a JSON object of inputs for this function. Extract values "
        "from the request; do not execute the function or use the whole request "
        "as an input. Preserve quoted text, escape sequences, and literal "
        "replacement words or symbols. If an input is a regular expression, "
        "translate the described match into regex syntax, including character "
        "classes, quantifiers, or boundaries when needed. Copy numeric literals "
        "exactly, including all digits and exponents.",
        request,
    )
    text = generate(model, prompt, arguments_grammar(definition), vocabulary)
    try:
        result: dict[str, Any] = json.loads(
            text,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
            parse_float=Decimal,
        )
    except (ValueError, InvalidOperation) as error:
        raise InputError(f"cannot parse generated arguments: {error}") from error
    return result
