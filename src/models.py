"""Validated models for the project input formats."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
    model_validator,
)


ValueType = Literal["string", "number", "integer", "boolean"]


class ParameterDefinition(BaseModel):
    """Describe one required function parameter in the supported dialect."""

    model_config = ConfigDict(extra="forbid", strict=True)

    type: ValueType
    enum: list[Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def reject_null_enum(cls, data: Any) -> Any:
        """Reject explicit null while allowing the enum field to be omitted."""
        if isinstance(data, dict) and "enum" in data and data["enum"] is None:
            raise ValueError("enum must be omitted instead of null")
        return data

    @model_validator(mode="after")
    def validate_enum(self) -> "ParameterDefinition":
        """Require a non-empty enum whose values match the declared type."""
        if self.enum is None:
            return self
        if not self.enum:
            raise ValueError("enum must not be empty")
        for value in self.enum:
            if not enum_value_matches(self.type, value):
                raise ValueError(
                    f"enum value {value!r} does not match type {self.type!r}"
                )
        return self


class ReturnDefinition(BaseModel):
    """Describe the supported return metadata from a function definition."""

    model_config = ConfigDict(extra="forbid", strict=True)

    type: ValueType


class FunctionDefinition(BaseModel):
    """Describe a callable function and all of its required parameters."""

    model_config = ConfigDict(extra="forbid", strict=True)

    name: str = Field(min_length=1)
    description: str
    parameters: dict[str, ParameterDefinition]
    returns: ReturnDefinition

    @field_validator("parameters")
    @classmethod
    def validate_parameter_names(
        cls, parameters: dict[str, ParameterDefinition]
    ) -> dict[str, ParameterDefinition]:
        """Reject empty parameter names."""
        if any(not name for name in parameters):
            raise ValueError("parameter names must not be empty")
        return parameters


class FunctionCatalog(RootModel[list[FunctionDefinition]]):
    """Hold a non-empty catalog with unique function names."""

    @model_validator(mode="after")
    def validate_catalog(self) -> "FunctionCatalog":
        """Reject missing definitions and duplicate function names."""
        if not self.root:
            raise ValueError("at least one function definition is required")
        names = [definition.name for definition in self.root]
        if len(names) != len(set(names)):
            raise ValueError("function names must be unique")
        return self


class InputRequest(BaseModel):
    """Represent one prompt from the input batch."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str


class InputBatch(RootModel[list[InputRequest]]):
    """Represent an ordered batch of prompts."""


class FunctionCall(BaseModel):
    """Hold the exact output envelope after definition-specific validation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    prompt: str
    name: str
    parameters: dict[str, Any]


def enum_value_matches(value_type: ValueType, value: Any) -> bool:
    """Return whether one JSON enum value matches the supported declared type."""
    if value_type == "string":
        return type(value) is str
    if value_type == "boolean":
        return type(value) is bool
    if value_type == "integer":
        return type(value) is int
    if type(value) not in (int, float):
        return False
    return math.isfinite(value)
