from dataclasses import dataclass

import anyio
from pydantic_ai import RunContext, Tool

from crewdantic import Crew


@dataclass
class CalculatorDeps:
    offset: int


def add_offset(ctx: RunContext[CalculatorDeps], value: int) -> int:
    return value + ctx.deps.offset


def multiply(_ctx: RunContext[CalculatorDeps], left: int, right: int) -> int:
    return left * right


async def main() -> None:
    crew = Crew(
        agent_definition="agents.yaml",
        task_definition="tasks.yaml",
        deps_type=CalculatorDeps,
        tools=[
            Tool(add_offset),
            multiply,
        ],
    )
    result = await crew.kickoff(context=CalculatorDeps(offset=10))
    print(result)


if __name__ == "__main__":
    anyio.run(main)
