"""Strict JSON input and atomic output helpers."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.models import FunctionCatalog, InputBatch


class InputError(Exception):
    """Report an expected input, processing, or output boundary failure."""


def reject_constant(value: str) -> None:
    """Reject non-standard JSON numeric constants such as NaN and Infinity."""
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build an object while rejecting duplicate JSON keys."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    """Read strict JSON from a regular file and return its decoded value."""
    try:
        with path.open("r", encoding="utf-8") as source:
            return json.load(
                source,
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError,
            RecursionError) as error:
        raise InputError(f"cannot read valid JSON from {path}: {error}") from error


def load_catalog(path: Path) -> FunctionCatalog:
    """Load and validate the complete function catalog."""
    try:
        return FunctionCatalog.model_validate(load_json(path), strict=True)
    except ValidationError as error:
        raise InputError(f"invalid function definitions in {path}: {error}") from error


def load_requests(path: Path) -> InputBatch:
    """Load and validate the ordered request batch."""
    try:
        return InputBatch.model_validate(load_json(path), strict=True)
    except ValidationError as error:
        raise InputError(f"invalid request batch in {path}: {error}") from error


def write_json_atomic(path: Path, value: Any) -> None:
    """Replace an output only after its complete JSON has reached disk."""
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except (OSError, TypeError, ValueError) as error:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise InputError(
            f"cannot safely write output {path}: {error}; no new result was published"
        ) from error
