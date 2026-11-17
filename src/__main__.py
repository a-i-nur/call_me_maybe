"""Command-line entry point for Call Me Maybe."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError

from src.io import InputError, load_catalog, load_requests, write_json_atomic
from src.arguments import extract_arguments
from src.selector import select_function
from src.sdk import load_model, token_texts
from src.validation import validate_call


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
    """Validate inputs, generate a complete batch, and publish it atomically."""
    catalog = load_catalog(arguments.functions_definition)
    requests = load_requests(arguments.input)
    if not requests.root:
        write_json_atomic(arguments.output, [])
        return
    model = load_model()
    vocabulary = token_texts(model)
    results = []
    for request in requests.root:
        definition = select_function(model, catalog, request.prompt, vocabulary)
        parameters = extract_arguments(model, definition, request.prompt, vocabulary)
        result = validate_call(request.prompt, definition, parameters)
        results.append(result.model_dump())
    write_json_atomic(arguments.output, results)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and translate expected failures into exit status 1."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        run(arguments)
    except (InputError, ValidationError, ValueError, OverflowError) as error:
        print(f"error: {error}; no new result was published", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
