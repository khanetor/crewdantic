from __future__ import annotations

import anyio
from pydantic import BaseModel

from crewdantic import Crew


class ResearchReport(BaseModel):
    topic: str
    summary: str
    key_points: list[str]


class ResearchCrew(Crew[ResearchReport]):
    agent_definition = "agents.yaml"
    task_definition = "tasks.yaml"


async def main() -> None:
    topic = "honeybee communication"
    crew = ResearchCrew(
        agent_variables={"topic": topic},
        pydantic_models=[ResearchReport],
    )
    report = await crew.kickoff({"topic": topic})

    assert isinstance(report, ResearchReport)
    assert report.key_points
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    anyio.run(main)
