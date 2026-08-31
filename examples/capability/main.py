import anyio
from pydantic_ai.capabilities import AbstractCapability

from crewdantic import Crew


class ConciseCapability(AbstractCapability[object]):
    def get_instructions(self) -> str:
        return "Answer in one short sentence."


async def main() -> None:
    crew = Crew(
        agent_definition="agents.yaml",
        task_definition="tasks.yaml",
        capabilities=[ConciseCapability()],
    )
    result = await crew.kickoff()
    print(result)


if __name__ == "__main__":
    anyio.run(main)
