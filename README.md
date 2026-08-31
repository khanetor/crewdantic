# crewdantic

A minimal crew abstraction for Pydantic AI, configured with CrewAI-style YAML.

## Installation

```console
pip install crewdantic
```

## Usage

Place the definitions beside the Python module that declares the crew:

```python
# research_crew.py
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import RunContext, Tool

from crewdantic import Crew

class ResearchOutput(BaseModel):
    summary: str


@dataclass
class ResearchContext:
    documents: list[str]

def search(ctx: RunContext[ResearchContext], query: str) -> list[str]:
    return [
        document
        for document in ctx.deps.documents
        if query.casefold() in document.casefold()
    ]

```


```yaml
# agents.yaml
researcher:
  role: "{expertise} Researcher"
  goal: Apply {expertise} expertise
  backstory: An experienced {expertise} researcher
  llm: openai:gpt-5.2
  tools:
    - search
composer:
  role: Report composer
  goal: Turn research into a concise report
  backstory: An experienced technical writer
  llm: openai:gpt-5.2
  output_model: ResearchOutput
```

```yaml
# tasks.yaml
research:
  description: Research {topic}
  expected_output: A concise report about {topic}
  agent: researcher
compose:
  description: Compose the final report from the research
  expected_output: A concise structured report
  agent: composer
```

`Crew(...)` turns each agent declaration into a Pydantic AI `Agent`.
Every agent must declare a Pydantic AI model name as `llm`; invalid names fail
during crew construction. Agents return text unless they declare
`output_model`, matching a registered Pydantic model class name. `kickoff()`
runs tasks in YAML order, passes each output to the next task, and returns the
final task's agent output.

`output_type` defaults to `str`, and `deps_type` defaults to `object`. Supplying
them gives static type checkers the final output and shared Pydantic AI context
types. Construction verifies that the final task's agent uses the declared or
default output type before any agent runs:

```python
crew = Crew(
    agent_definition="agents.yaml",
    task_definition="tasks.yaml",
    output_type=ResearchOutput,
    deps_type=ResearchContext,
    agent_variables={"expertise": "AI"},
    tools=[Tool(search)],
)
report = await crew.kickoff(
    {"topic": "AI agents"},
    context=ResearchContext(documents=["AI agents use tools."]),
)
```

Relative strings and `Path` values resolve beside the Python module that
constructs the crew; absolute paths are used unchanged.

`output_type` is automatically available to YAML `output_model` references.
Pass `pydantic_models` only for additional structured output types used by
intermediate agents.

Pydantic AI context-aware tool callables, `Tool` instances, and capabilities
are supplied once through the constructor and assigned to agents by name in
`agents.yaml`.

Callable names, `Tool.name`, and capability IDs or class names are used for
YAML lookup. Function tools take `RunContext[ResearchContext]`; the context
type is applied to every generated Pydantic AI agent, and `kickoff()` forwards
the context instance to every task. Tools can also be registered on a
constructed agent with Pydantic AI's decorators:

```python
@crew.agents["researcher"].tool
def word_count(_ctx: RunContext[ResearchContext], text: str) -> int:
    return len(text.split())
```

Agent roles, goals, and backstories interpolate `agent_variables` during crew
construction. Task descriptions and expected outputs interpolate `kickoff()`
inputs independently.
Template values may be any Python object supported by string formatting,
including lists, mappings, and Pydantic models.

## Examples

The repository includes deterministic examples that require no API keys:

```console
uv run python examples/tools/main.py
uv run python examples/capability/main.py
```

The tools example registers both a Pydantic AI `Tool` and a Python function.
The capability example registers a custom reusable capability. Both select
Pydantic AI's built-in `test` model in YAML, so application code needs no
model override and the examples make no network calls.

## Roadmap

- Pydantic AI toolsets and MCP integration
- Workflow orchestration
