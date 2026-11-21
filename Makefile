.PHONY: install install-school run debug clean clean-school lint lint-strict

SCHOOL_DIR ?= /goinfre/$(USER)/call-me-maybe

ifneq ($(wildcard $(SCHOOL_DIR)/venv/pyvenv.cfg),)
export UV_PROJECT_ENVIRONMENT := $(SCHOOL_DIR)/venv
export UV_CACHE_DIR := $(SCHOOL_DIR)/uv-cache
export UV_PYTHON_INSTALL_DIR := $(SCHOOL_DIR)/python
export HF_HOME := $(SCHOOL_DIR)/huggingface
endif

install:
	uv sync

install-school:
	mkdir -p "$(SCHOOL_DIR)"
	UV_PROJECT_ENVIRONMENT="$(SCHOOL_DIR)/venv" \
	UV_CACHE_DIR="$(SCHOOL_DIR)/uv-cache" \
	UV_PYTHON_INSTALL_DIR="$(SCHOOL_DIR)/python" uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	find src -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -f data/output/function_calls.json

clean-school:
	rm -rf -- "$(SCHOOL_DIR)/venv" "$(SCHOOL_DIR)/uv-cache" \
		"$(SCHOOL_DIR)/python" "$(SCHOOL_DIR)/huggingface"

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
