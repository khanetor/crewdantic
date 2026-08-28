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
  role: Researcher
  goal: Research the requested topic
  backstory: An experienced researcher
```

```yaml
# tasks.yaml
research:
  description: Research the requested topic
  expected_output: A concise report
  agent: researcher
```

Instantiating `ResearchCrew` validates both files and exposes
`agent_definitions` and `task_definitions` mappings.