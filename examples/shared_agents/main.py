import anyio
from pydantic import BaseModel

from crewdantic import Crew


class ResearchReport(BaseModel):
    topic: str
    summary: str
    key_points: list[str]


async def main() -> None:
    topic = "honeybee communication"
    crews = {
        "full": Crew(
            agent_definition="agents.yaml",
            task_definition="research_tasks.yaml",
            output_type=ResearchReport,
            agent_variables={"topic": topic},
        ),
        "brief": Crew(
            agent_definition="agents.yaml",
            task_definition="brief_tasks.yaml",
            output_type=ResearchReport,
            agent_variables={"topic": topic},
        ),
    }
    report = await crews["brief"].kickoff({"topic": topic})

    assert isinstance(report, ResearchReport)
    assert report.key_points
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    anyio.run(main)
