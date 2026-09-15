# crewdantic

A minimal crew abstraction for Pydantic AI, configured with CrewAI-style YAML.

## Installation

```console
pip install --index-url https://codeberg.org/api/packages/khanetor/pypi/simple/ crewdantic
```

With [uv](https://docs.astral.sh/uv/):

```console
uv add --index https://codeberg.org/api/packages/khanetor/pypi/simple/ crewdantic
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
Every agent must declare a Pydantic AI model name as `llm`; model names are
parsed at crew construction, so unknown providers fail there (model-name
validity beyond the provider is checked when a call is made). Agents return
text unless they declare `output_model`, matching a registered Pydantic model
class name.

`kickoff()` runs tasks in YAML order and returns the final task's agent output.
Each task receives all previous task outputs by default in both sequential and
hierarchical crews. Set `context` to a list of earlier task names to select only
those outputs; `context: []` or `context: null` supplies none. Without a
manager, every task must name its agent.

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
    inputs={"topic": "AI agents"},
    context=ResearchContext(documents=["AI agents use tools."]),
)
```

Multimodal files ride alongside `inputs` as labeled `file_inputs`. A `{label}`
placeholder in a task description or expected output injects the file part at
that exact position; referencing every label is required. Import Pydantic AI
content parts from `pydantic_ai` directly:

```python
from pathlib import Path
from pydantic_ai import BinaryContent, ImageUrl

report = await crew.kickoff(
    inputs={"topic": "AI agents"},
    file_inputs={
        "chart": BinaryContent(
            data=Path("chart.png").read_bytes(), media_type="image/png"
        ),
        "diagram": ImageUrl(url="https://example.com/diagram.png"),
    },
)
```

```yaml
review:
  description: "Review {chart} against the reference {diagram}."
  expected_output: Findings list
  agent: reviewer
```

A single agent with `manager: true` makes the crew hierarchical:

```yaml
manager:
  manager: true
  llm: openai:gpt-5.2
  output_model: ResearchOutput
researcher:
  role: Researcher
  goal: Find authoritative evidence
  backstory: An experienced researcher
  llm: openai:gpt-5.2
writer:
  role: Writer
  goal: Explain the evidence clearly
  backstory: An experienced technical writer
  llm: openai:gpt-5.2
```

Manager role, goal, and backstory have defaults and can be overridden. Only one
manager is allowed. The manager runs tasks in YAML order and receives
`delegate_work` and `ask_question` tools. A task with an `agent` restricts those
tools to that worker; an unassigned task allows every non-manager agent. The
manager's `output_model` determines the final crew output type, while
intermediate manager outputs are text. `usage_limits` passed to `kickoff()`
apply across manager and worker model calls.

Relative strings and `Path` values resolve beside the Python module that
constructs the crew; absolute paths are used unchanged.

`to_mermaid_chart()` returns the declared task order as Mermaid source.
`render()` synchronously writes that chart to a `.mermaid` or `.html` file:

```python
mermaid = crew.to_mermaid_chart()
crew.render("research_crew.mermaid")
crew.render("research_crew.html")
```

HTML output loads Mermaid from its CDN when opened in a browser.

`output_type` is automatically available to YAML `output_model` references.
Pass `pydantic_models` only for additional structured output types used by
intermediate agents.

Pydantic AI context-aware tool callables, `Tool` instances, toolsets, and
capabilities are supplied once through the constructor and assigned to agents
by name in `agents.yaml`.

Callable names, `Tool.name`, toolset mapping keys, and capability IDs or class
names are used for YAML lookup. Function tools take
`RunContext[ResearchContext]`; the context type is applied to every generated
Pydantic AI agent, and `kickoff()` forwards the context instance to every task.
Tools can also be registered on a constructed agent with Pydantic AI's
decorators:

```python
@crew.agents["researcher"].tool
def word_count(_ctx: RunContext[ResearchContext], text: str) -> int:
    return len(text.split())
```

For MCP, prefer Pydantic AI's `MCP` capability. Register it in Python and
assign its `id` in YAML:

```yaml
researcher:
  role: Researcher
  goal: Use the filesystem
  backstory: Careful
  llm: openai:gpt-5.2
  capabilities:
    - filesystem
```

```python
from pydantic_ai.capabilities import MCP

crew = Crew(
    agent_definition="agents.yaml",
    task_definition="tasks.yaml",
    capabilities=[
        MCP("https://mcp.example.com/api", id="filesystem"),
    ],
)
```

Use a direct `MCPToolset` when its lower-level client configuration or shared
instance is needed:

```yaml
researcher:
  role: Researcher
  goal: Use the filesystem
  backstory: Careful
  llm: openai:gpt-5.2
  toolsets:
    - filesystem
```

```python
from pydantic_ai.mcp import MCPToolset

crew = Crew(
    agent_definition="agents.yaml",
    task_definition="tasks.yaml",
    toolsets={
        "filesystem": MCPToolset("https://mcp.example.com/api"),
    },
)
```

Any Pydantic AI `AgentToolset` can be a mapping value. Undefined capability
and toolset references fail during crew construction. URL-based configurations
require an MCP server already running at that address.

Agent roles, goals, and backstories interpolate `agent_variables` during crew
construction. Task descriptions and expected outputs interpolate `kickoff()`
inputs independently.
Template values may be any Python object supported by string formatting,
including lists, mappings, and Pydantic models.

## Workflows

`Workflow[InputT, OutputT]` builds a typed DAG from `@step` method
parameters. Pass the immutable workflow input through the reserved
`workflow_input` parameter; other parameter names refer to upstream steps.
Every workflow ends at `final_step`, whose return annotation must match
`OutputT`. Steps may be synchronous or asynchronous:

```python
from dataclasses import dataclass

from crewdantic import Workflow, step


@dataclass(frozen=True)
class ReportInput:
    topic: str


@dataclass(frozen=True)
class Report:
    body: str


class ReportWorkflow(Workflow[ReportInput, Report]):
    @step
    async def research(self, workflow_input: ReportInput) -> str:
        return f"Research about {workflow_input.topic}"

    @step
    def final_step(self, research: str) -> Report:
        return Report(body=research)


report = await ReportWorkflow().kickoff(ReportInput(topic="workflows"))
```

`to_mermaid_chart()` returns the declared step dependency graph as Mermaid
source, including branches, joins, and a marked final step. `render()` writes
that graph to `.mermaid` or Mermaid-backed `.html` output:

```python
workflow = ReportWorkflow()
mermaid = workflow.to_mermaid_chart()
workflow.render("report_workflow.html")
```

HTML output loads Mermaid from its CDN when opened and requires network access.

Construction rejects undefined or mismatched dependencies, cycles, steps that
do not lead to `final_step`, and plain (non-`@step`) overrides of
`to_mermaid_chart`/`render`. Independent steps are resolved deterministically
today.

Subclasses inherit steps and may add or override them by name; the merged
graph is re-validated at class definition, most-derived definition wins:

```python
class TitledReportWorkflow(ReportWorkflow):
    @step
    def title(self, research: str) -> str:
        return research.title()

    @step
    def final_step(self, research: str, title: str) -> Report:
        return Report(body=f"{title}\n\n{research}")
```

Overriding `to_mermaid_chart` or `render` is rejected, whether or not the
method is marked with `@step`.

## Examples

Three examples require no API keys:

```console
uv run python examples/workflow/main.py
uv run python examples/tools/main.py
uv run python examples/capability/main.py
```

The workflow example runs a typed DAG using only Python dataclasses. The tools
example registers both a Pydantic AI `Tool` and a Python function. The
capability example registers a custom reusable capability. The `basic` and
`shared_agents` examples call Gemini and need `GOOGLE_API_KEY` set.

## Development

Type checking uses [ty](https://docs.astral.sh/ty/) with every rule enabled;
run `uv run ty check --error all src tests examples` locally so CI matches.
