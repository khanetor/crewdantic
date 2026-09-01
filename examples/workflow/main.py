from dataclasses import dataclass

import anyio

from crewdantic import Workflow, step


@dataclass(frozen=True, slots=True)
class ReportInput:
    topic: str


@dataclass(frozen=True, slots=True)
class Research:
    fact: str


@dataclass(frozen=True, slots=True)
class Audience:
    name: str


@dataclass(frozen=True, slots=True)
class Report:
    topic: str
    body: str


class ReportWorkflow(Workflow[ReportInput, Report]):
    @step
    async def research(self, workflow_input: ReportInput) -> Research:
        return Research(f"{workflow_input.topic} can be expressed as a typed DAG.")

    @step
    async def audience(self, workflow_input: ReportInput) -> Audience:
        return Audience(f"developers learning about {workflow_input.topic}")

    @step
    def draft(self, research: Research, audience: Audience) -> str:
        return f"For {audience.name}: {research.fact}"

    @step
    def final_step(
        self,
        workflow_input: ReportInput,
        draft: str,
    ) -> Report:
        return Report(topic=workflow_input.topic, body=draft)


async def main() -> None:
    report = await ReportWorkflow().kickoff(ReportInput("workflow orchestration"))
    assert report.topic == "workflow orchestration"
    assert "typed DAG" in report.body
    print(report)


if __name__ == "__main__":
    anyio.run(main)
