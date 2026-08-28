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
from crewdantic import Crew


class ResearchCrew(Crew):
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
```

```yaml
# tasks.yaml
research:
  description: Research {topic}
  expected_output: A concise report about {topic}
  agent: researcher
```

`ResearchCrew` turns each agent declaration into a Pydantic AI `Agent`.
Every agent must declare a Pydantic AI model name as `llm`; invalid names fail
during crew construction. `kickoff()` runs tasks in YAML order and passes each
output to the next task:

```python
crew = ResearchCrew(agent_variables={"expertise": "AI"})
report = await crew.kickoff({"topic": "AI agents"})
```

Agent roles, goals, and backstories interpolate `agent_variables` during crew
construction. Task descriptions and expected outputs interpolate `kickoff()`
inputs independently.
Template values may be any Python object supported by string formatting,
including lists, mappings, and Pydantic models.