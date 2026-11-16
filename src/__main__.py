"""Command-line entry point for Call Me Maybe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from src.io import InputError, load_catalog, load_requests, write_json_atomic


DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calls.json")


def build_parser() -> argparse.ArgumentParser:
    """Build the project command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate schema-valid function calls from prompts."
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS,
        help="path to function definitions JSON",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="path to prompt batch JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="path for generated function calls JSON",
    )
    return parser


def run(arguments: argparse.Namespace) -> None:
    """Validate inputs and process the stage-one empty-batch case."""
    load_catalog(arguments.functions_definition)
    requests = load_requests(arguments.input)
    if requests.root:
        raise InputError(
            "non-empty input cannot be processed yet: selector and decoder "
            "are not implemented; no new result was written"
        )
    write_json_atomic(arguments.output, [])


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and translate expected failures into exit status 1."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        run(arguments)
    except (InputError, ValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
