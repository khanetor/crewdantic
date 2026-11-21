from typing import Annotated, Literal

import anyio
from pydantic import BaseModel, Field

from crewdantic import Crew

Severity = (
    Annotated[Literal[0], Field(description="Cosmetic or trivial")]
    | Annotated[Literal[1], Field(description="Blocks one user, workaround exists")]
    | Annotated[Literal[2], Field(description="Blocks many users")]
)


class TriageDecision(BaseModel):
    """Route a support message."""

    urgent: bool = Field(description="Does someone need to act now?")
    team: Literal["billing", "engineering", "sales"] = Field(
        description="Which team owns this?"
    )
    severity: Severity = Field(description="How bad is it?")


async def main() -> None:
    crew = Crew(
        agent_definition="agents.yaml",
        task_definition="tasks.yaml",
        output_type=TriageDecision,
    )
    decision = await crew.kickoff(
        inputs={"message": "Delivery was late but the product quality was excellent."}
    )
    print(decision.model_dump_json(indent=2))


if __name__ == "__main__":
    anyio.run(main)
