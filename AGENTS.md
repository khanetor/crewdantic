# crewdantic contributor guide

## Project

Crewdantic is a minimal sequential crew abstraction over Pydantic AI. Agent and task definitions live in CrewAI-style YAML; Python supplies Pydantic model classes and the typed `Crew[OutputT]` boundary.

## Layout

- `src/crewdantic/crew.py`: YAML validation, agent construction, task execution, and output typing.
- `tests/test_crew.py`: behavioral tests using Pydantic AI `FunctionModel`; no network access.
- `examples/`: runnable examples. `research_crew.py` is the live structured-output smoke test using Gemini models.

## Commands

```console
uv sync
uv run python -m unittest discover -s tests
uv run ty check src tests examples
uv run ruff check src tests examples
uv run ruff format src tests examples
```

Run the live example only with a key supplied through the environment:

```console
GEMINI_API_KEY=... uv run python examples/research_crew.py
```

Never store API keys in source, YAML, examples, tests, or `.env` files committed to Git.

## Design invariants

- Keep the implementation small and in existing files unless a separate module removes real complexity.
- Keep library code async-runtime-neutral and never import `asyncio`. Runnable examples and tests use AnyIO; do not add AnyIO as a core dependency unless `src` requires it.
- Resolve relative YAML paths beside the module declaring the `Crew` subclass.
- Every task must reference a declared agent.
- Every agent must declare `llm`; `output_model` remains optional and defaults to `str`.
- Resolve `output_model` by Pydantic class name from the constructor's `pydantic_models` list.
- `kickoff()` runs tasks in YAML order and forwards each result to the next task.
- `Crew[OutputT]` types the final result. For parameterized crews, construction must reject a final task whose agent output does not exactly match `OutputT`, before any agent runs.
- Unparameterized `Crew` subclasses remain supported without final-output matching.
- Keep LLM calls explicit in the task sequence; do not add hidden model calls for parsing or finalization.

## Tests

- Test observable constructor and kickoff behavior, not implementation details.
- Use `FunctionModel` for deterministic model responses.
- Cover both structured success and configuration failure before execution.
- Run the full unit suite, `ty`, and Ruff after behavioral changes.
