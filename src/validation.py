"""Independent validation and explicit numeric output normalization."""

from __future__ import annotations

from typing import Any

from src.io import InputError
from src.models import FunctionCall, FunctionCatalog, FunctionDefinition
from src.models import enum_value_matches


def check_stage_two(catalog: FunctionCatalog) -> None:
    """Reject unsupported generation schemas before loading the model."""
    for definition in catalog.root:
        for name, parameter in definition.parameters.items():
            if parameter.type not in {"string", "number"} or parameter.enum is not None:
                raise InputError(
                    f"stage 2 cannot generate {definition.name}.{name}: "
                    "only string/number without enum are implemented"
                )


def validate_call(
    prompt: str, definition: FunctionDefinition, parameters: dict[str, Any]
) -> FunctionCall:
    """Check exact keys/types, then serialize number values as finite floats."""
    if parameters.keys() != definition.parameters.keys():
        raise InputError("generated parameters do not match the required keys")
    normalized: dict[str, Any] = {}
    for name, parameter in definition.parameters.items():
        value = parameters[name]
        if not enum_value_matches(parameter.type, value):
            raise InputError(f"generated parameter {name!r} has an invalid type/value")
        if parameter.enum is not None and value not in parameter.enum:
            raise InputError(f"generated parameter {name!r} is outside its enum")
        if parameter.type == "string":
            if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
                raise InputError(f"generated parameter {name!r} has a lone surrogate")
        normalized[name] = float(value) if parameter.type == "number" else value
    return FunctionCall(prompt=prompt, name=definition.name, parameters=normalized)
