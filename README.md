*This project has been created as part of the 42 curriculum by aakhmeto.*

# Call Me Maybe

## Description

This Python CLI translates a batch of natural-language prompts into structured
function calls. Given a JSON catalog of available functions and a JSON list of
user prompts, it uses the supplied `llm_sdk` wrapper and the default
`Qwen/Qwen3-0.6B` model with constrained decoding to select the appropriate
function and extract typed arguments. It does not execute the functions. Each
entry in the output JSON array contains strictly `prompt`, `name`, and
`parameters`.

## Instructions

Requirements: Python 3.10 or newer and
[uv](https://docs.astral.sh/uv/getting-started/installation/).

Install dependencies from the repository root:

```sh
uv sync
```

The first execution downloads the `Qwen/Qwen3-0.6B` weights and tokenizer files
to the local Hugging Face cache; subsequent runs reuse the cached files.

Available `Makefile` targets:

- `make install` — install project dependencies via `uv sync`
- `make install-school` — install into `/goinfre/$USER/call-me-maybe`
- `make run` — run the program with default input and output paths
- `make debug` — run the program under Python's built-in `pdb` debugger
- `make clean` — remove Python/linter caches and generated output
- `make clean-school` — remove the school environment and uv/Python/Hugging Face downloads from goinfre
- `make lint` — run `flake8 .` and `mypy .` with the required subject flags
- `make lint-strict` — run `flake8 .` and the optional `mypy . --strict`

## Example usage

Run with the default paths (`data/input/functions_definition.json`,
`data/input/function_calling_tests.json`, and
`data/output/function_calls.json`):

```sh
uv run python -m src
```

Run with custom file paths:

```sh
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Example output entry in `data/output/function_calls.json`:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  }
]
```

## Algorithm explanation

1. **Input validation:** Both input JSON files are parsed strictly (rejecting
   duplicate keys and non-finite constants such as `NaN` or `Infinity`) and
   validated with Pydantic models (`strict=True`) before loading the LLM.
2. **Function selection:** The model receives the catalog of function names,
   descriptions, parameter names, and the user prompt. Its output is
   constrained by a trie grammar (`choices_grammar`) permitting only the exact
   JSON-encoded function names from the catalog or `null` if no function
   applies. Selection is driven purely by the model's logits without keyword
   heuristics.
3. **Schema-driven grammar compilation:** For the chosen function,
   `arguments_grammar` builds a deterministic finite automaton (DFA) that
   enforces the exact parameter keys in order and restricts each parameter value
   to its declared type (`string`, `number`, `integer`, `boolean`, or allowed
   `enum` literals).
4. **Token-by-token constrained decoding:** At each generation step,
   `Small_LLM_Model.get_logits_from_input_ids` returns logits over the full
   vocabulary. Every candidate token is tested character-by-character (and
   byte-by-byte for split UTF-8 sequences) against the current automaton state.
   Disallowed tokens have their logits set to `-inf` **before** greedy selection
   (`np.argmax`). Generation terminates only when the automaton reaches an
   accepting state after the closing brace `}`.
5. **Post-validation and atomic write:** The generated JSON object is parsed
   and validated against the function definition. Numeric parameters of type
   `number` are normalized to finite Python `float` values (e.g., `2.0`), while
   `integer` parameters remain integers. Once the entire batch succeeds, the
   result is written to a temporary file, synced to disk, and atomically moved
   to the target output path.

## Design decisions

- **Two-stage constrained generation:** Separating function selection from
  argument extraction keeps prompts concise and allows compiling a small,
  deterministic schema automaton tailored to the selected function.
- **Byte-level UTF-8 state tracking:** Because BPE tokens in `Qwen/Qwen3-0.6B`
  can split multibyte UTF-8 characters across adjacent tokens, the decoder
  state pairs the grammar state index with a pending byte buffer
  (`tuple[int, bytes]`), reconstructing exact token bytes from the SDK
  tokenizer file.
- **Public SDK interface only:** The implementation interacts with
  `Small_LLM_Model` exclusively through its public methods (`encode`, `decode`,
  `get_logits_from_input_ids`, and `get_path_to_tokenizer_file`) without
  importing forbidden external ML frameworks in project code.
- **Fail-fast and atomic output:** Invalid schemas, malformed inputs,
  unmatched prompts, or generation errors exit cleanly with status `1` and a
  descriptive message to `stderr`, leaving any pre-existing output file intact.

## Performance analysis

- **JSON and schema validity:** 100% of generated outputs are valid, parseable
  JSON objects that strictly conform to the required output keys and parameter
  types.
- **Accuracy:** Consistently achieves >90% accuracy on function selection and
  argument extraction (100% on the default 11-prompt test suite).
- **Execution speed:** Processes the full batch of test prompts in ~35–45
  seconds on a laptop GPU (RTX 4060). At 42 Heilbronn, the i7-13700 CPU
  (8 threads) processed all 11 public prompts in 118 seconds, including the
  initial model download, staying within the 5-minute threshold. Per-state mask
  caching and first-character token indexing avoid rescanning the full 151,936-token
  vocabulary on every step.

## Challenges faced

- **Full-sequence forward pass overhead:** Because `get_logits_from_input_ids`
  recomputes logits over the entire token sequence at every step without a
  public KV-cache API, keeping prompts compact and caching legal token masks
  per automaton state were essential to meet the runtime budget on CPU.
- **Partial UTF-8 tokens:** Validating decoded strings alone would either
  reject valid multi-token Unicode characters or admit invalid byte prefixes.
  Tracking incomplete UTF-8 sequences in `advance_bytes` solved this cleanly.
- **Numeric range limits:** Standard JSON syntax permits numbers larger than a
  64-bit IEEE-754 float. Parsing numbers via `Decimal` and checking for
  overflow and underflow in `finite_number` prevents silent rounding to
  `Infinity` or `0.0`.

## Testing strategy

Local unit and integration tests (kept outside the submission repository as
required by the subject) verify:

- DFA transitions and logit masking before `argmax` for strings, escape
  sequences, Unicode surrogate pairs, integers, floats, booleans, and enums;
- multibyte UTF-8 characters split across token boundaries;
- error handling for missing files, malformed or deeply nested JSON, duplicate
  keys, unsupported schemas, unmatched prompts, and numeric overflow;
- end-to-end execution with `Qwen/Qwen3-0.6B`, confirming output structure,
  prompt order preservation, and atomic rollback on failure.

## Resources

- [RFC 8259 — The JavaScript Object Notation (JSON) Data Interchange Format](https://datatracker.ietf.org/doc/html/rfc8259)
- [uv Documentation — Managing Projects and Virtual Environments](https://docs.astral.sh/uv/concepts/projects/sync/)
- [Pydantic Documentation — Models and Strict Mode Validation](https://docs.pydantic.dev/latest/concepts/models/)
- [Python `json` Module Documentation](https://docs.python.org/3/library/json.html)
- [Qwen/Qwen3-0.6B Model Card on Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B)

### AI usage disclosure

AI tools were used during development to analyze subject requirements, discuss
finite-state grammar design for constrained decoding, review edge cases in
UTF-8 byte handling and JSON validation, and draft documentation and unit
tests. All code and tests were reviewed, verified, and understood by the
author, who takes full responsibility for the final implementation.
