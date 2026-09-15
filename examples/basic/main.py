import anyio
from pydantic import BaseModel

from crewdantic import Crew


class ResearchReport(BaseModel):
    topic: str
    summary: str
    key_points: list[str]


async def main() -> None:
    topic = "honeybee communication"
    crew = Crew(
        agent_definition="agents.yaml",
        task_definition="tasks.yaml",
        output_type=ResearchReport,
        agent_variables={"topic": topic},
    )
    report = await crew.kickoff(inputs={"topic": topic})

    assert isinstance(report, ResearchReport)
    assert report.key_points
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    anyio.run(main)
