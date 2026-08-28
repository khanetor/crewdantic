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
from pydantic import BaseModel

from crewdantic import Crew

class ResearchOutput(BaseModel):
    summary: str


class ResearchCrew(Crew[ResearchOutput]):
    agent_definition = "agents.yaml"
    task_definition = "tasks.yaml"
```

```yaml
# agents.yaml
researcher:
  role: "{expertise} Researcher"
  goal: Apply {expertise} expertise
  backstory: An experienced {expertise} researcher
  llm: openai:gpt-5.2
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

`ResearchCrew` turns each agent declaration into a Pydantic AI `Agent`.
Every agent must declare a Pydantic AI model name as `llm`; invalid names fail
during crew construction. Agents return text unless they declare
`output_model`, matching the class name of one of the Pydantic models passed
to the crew. `kickoff()` runs tasks in YAML order, passes each output to the
next task, and returns the final task's agent output. `Crew[ResearchOutput]`
makes that return type available to static type checkers. Crew construction
also verifies that the final task's agent uses the same output type, before
any agent runs:

```python
crew = ResearchCrew(
    agent_variables={"expertise": "AI"},
    pydantic_models=[ResearchOutput],
)
report = await crew.kickoff({"topic": "AI agents"})
```

Agent roles, goals, and backstories interpolate `agent_variables` during crew
construction. Task descriptions and expected outputs interpolate `kickoff()`
inputs independently.
Template values may be any Python object supported by string formatting,
including lists, mappings, and Pydantic models.