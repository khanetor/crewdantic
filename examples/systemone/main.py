import anyio

from crewdantic import Crew, ScoreDecision


async def main() -> None:
    crew = Crew(
        agent_definition="agents.yaml",
        task_definition="tasks.yaml",
        output_type=ScoreDecision,
        systemone_model="d3:4b",
    )
    decision = await crew.kickoff(
        inputs={"message": "Delivery was late but the product quality was excellent."}
    )
    print(decision.model_dump_json(indent=2))


if __name__ == "__main__":
    anyio.run(main)
