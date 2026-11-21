# crewdantic contributor guide

## Project

Crewdantic is a minimal sequential crew abstraction and typed workflow layer over Pydantic AI. Agent and task definitions live in CrewAI-style YAML; Python supplies Pydantic model classes and a typed `Crew(...)` constructor, and `Workflow` subclasses declare `@step` methods forming a checked dependency graph.

## Layout

- `src/crewdantic/crew.py`: YAML validation, agent construction, task execution, and output typing.
- `src/crewdantic/workflow.py`: `@step` graph collection, construction-time validation, and execution.
- `src/crewdantic/_visualization.py`: Mermaid and HTML rendering shared by crews and workflows.
- `tests/test_crew.py`: crew behavioral tests using Pydantic AI `FunctionModel`; no network access.
- `tests/test_workflow.py`: workflow graph and execution tests; no network access.
- `tests/agents.yaml`, `tests/tasks.yaml`: fixtures resolved beside `tests/test_crew.py`.
- `examples/basic/`: live structured-output smoke test using Gemini models.
- `examples/shared_agents/`: two crews sharing agents across different task sets.
- `examples/tools/`: keyless example using an SDK `Tool` and a Python function tool.
- `examples/capability/`: keyless example using a reusable capability.
- `examples/workflow/`: keyless example using a typed workflow.

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
GEMINI_API_KEY=... uv run python examples/basic/main.py
```

Never store API keys in source, YAML, examples, tests, or `.env` files committed to Git.

## Design invariants

- Keep the implementation small and in existing files unless a separate module removes real complexity.
- Keep library code async-runtime-neutral and never import `asyncio`. Runnable examples and tests use AnyIO, which lives in the `dev` dependency group; do not add AnyIO as a core dependency unless `src` requires it.
- Accept agent and task YAML paths explicitly through the `Crew` constructor. Resolve relative strings and `Path` values beside the module constructing the crew.
- Every task must reference a declared agent.
- Every agent must declare `llm`; `output_model` remains optional and defaults to `str`.
- Resolve `output_model` by Pydantic class name from `output_type` or the constructor's additional `pydantic_models`.
- `kickoff()` runs tasks in YAML order, forwards each result to the next task, and supplies its context to every agent as Pydantic AI dependencies.
- Constructor `output_type` and `deps_type` arguments type the final result and each agent's `RunContext`, defaulting to `str` and `object`. Construction must reject a missing task sequence or a final task whose agent output does not exactly match `output_type`, before any agent runs.
- Keep LLM calls explicit in the task sequence; do not add hidden model calls for parsing or finalization.
- `kickoff()` treats a missing `usage_limits` as no request limit (`UsageLimits(request_limit=None)`); Pydantic AI would otherwise apply its own 50-request default.
- YAML duplicate keys are rejected instead of silently keeping the last one.
- `Workflow` steps must be decorated with `@step`, including overrides; construction rejects plain overrides, undefined dependencies, type mismatches, cycles, and steps that do not lead to `final_step`.

## Tests

- Test observable constructor and kickoff behavior, not implementation details.
- Use `FunctionModel` for deterministic model responses.
- Cover both structured success and configuration failure before execution.
- Run the full unit suite, `ty`, and Ruff after behavioral changes.
