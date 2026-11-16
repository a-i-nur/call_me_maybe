.PHONY: install run debug clean lint

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	find src -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f data/output/function_calls.json

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
