"""Independent validation and explicit numeric output normalization."""

from __future__ import annotations

from decimal import Decimal
import math
from typing import Any

from src.io import InputError
from src.models import FunctionCall, FunctionDefinition
from src.models import enum_value_matches


def finite_number(value: Any, name: str) -> float:
    """Normalize a generated JSON number or report overflow and underflow."""
    if type(value) not in (int, float, Decimal):
        raise InputError(f"generated parameter {name!r} is not a number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise InputError(f"generated parameter {name!r} overflows float") from error
    if not math.isfinite(number):
        raise InputError(f"generated parameter {name!r} overflows finite float")
    if value != 0 and number == 0:
        raise InputError(f"generated parameter {name!r} underflows float")
    return number


def validate_call(
    prompt: str, definition: FunctionDefinition, parameters: dict[str, Any]
) -> FunctionCall:
    """Check exact keys/types, then serialize number values as finite floats."""
    if not isinstance(parameters, dict):
        raise InputError("generated parameters must be a JSON object")
    if parameters.keys() != definition.parameters.keys():
        raise InputError("generated parameters do not match the required keys")
    normalized: dict[str, Any] = {}
    for name, parameter in definition.parameters.items():
        value = parameters[name]
        if parameter.type == "number":
            value = finite_number(value, name)
        elif not enum_value_matches(parameter.type, value):
            raise InputError(f"generated parameter {name!r} has an invalid type/value")
        if parameter.enum is not None and value not in parameter.enum:
            raise InputError(f"generated parameter {name!r} is outside its enum")
        normalized[name] = value
    return FunctionCall(prompt=prompt, name=definition.name, parameters=normalized)
