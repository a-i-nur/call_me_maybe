"""LLM selection among the current function names and an internal refusal."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Mapping

from src.decoding import generate
from src.grammar import choices_grammar
from src.io import InputError
from src.models import FunctionCatalog, FunctionDefinition

if TYPE_CHECKING:
    from llm_sdk import Small_LLM_Model


def chat_prompt(instruction: str, content: str) -> str:
    """Format the default Qwen chat turn with thinking disabled explicitly."""
    return (
        f"<|im_start|>system\n{instruction}<|im_end|>\n"
        f"<|im_start|>user\n{content}\n/no_think<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def select_function(
    model: Small_LLM_Model,
    catalog: FunctionCatalog,
    request: str,
    vocabulary: Mapping[int, str | bytes],
) -> FunctionDefinition:
    """Let constrained model logits choose a JSON name or null for no match."""
    choices = [json.dumps(item.name) for item in catalog.root] + ["null"]
    content = json.dumps(
        {"functions": catalog.model_dump(exclude_none=True), "request": request},
        ensure_ascii=False,
    )
    prompt = chat_prompt(
        "Identify which listed function best fulfills the request. "
        "If none is applicable, return null. Return only one exact function "
        "name as a JSON string, or null.",
        content,
    )
    name = json.loads(generate(model, prompt, choices_grammar(choices), vocabulary))
    if name is None:
        raise InputError(f"no matching function for request {request!r}")
    for definition in catalog.root:
        if definition.name == name:
            return definition
    raise InputError("selector produced an unknown function name")
