import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path

import anyio
from pydantic import ValidationError
from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    TextPart,
    UserError,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from crewdantic import Crew


class ExampleCrew(Crew):
    agent_definition = "agents.yaml"
    task_definition = "tasks.yaml"


class CrewTests(unittest.TestCase):
    def test_loads_definitions_relative_to_crew_module(self) -> None:
        crew = ExampleCrew()

        self.assertEqual(crew.agent_definitions["researcher"].role, "Researcher")
        self.assertEqual(crew.task_definitions["research"].agent, "researcher")

    def test_builds_agents_and_runs_tasks_sequentially(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agents = root / "agents.yaml"
            tasks = root / "tasks.yaml"
            agents.write_text(
                "researcher:\n"
                "  role: '{topic} Researcher'\n"
                "  goal: Find facts about {topic}\n"
                "  backstory: Curious about {topic}\n"
                "  llm: test\n"
                "writer:\n"
                "  role: Writer\n"
                "  goal: Explain facts\n"
                "  backstory: Clear\n"
                "  llm: test\n"
            )
            tasks.write_text(
                "research:\n"
                "  description: Research {topic}\n"
                "  expected_output: Facts about {topic}\n"
                "  agent: researcher\n"
                "write:\n"
                "  description: Write the report\n"
                "  expected_output: A concise report\n"
                "  agent: writer\n"
            )
            requests: list[tuple[str, str]] = []

            def respond(messages, info):
                instructions = info.instructions
                assert instructions is not None
                request = next(
                    message for message in reversed(messages)
                    if isinstance(message, ModelRequest)
                )
                prompt = next(
                    part.content for part in request.parts
                    if isinstance(part, UserPromptPart)
                )
                assert isinstance(prompt, str)
                requests.append((instructions, prompt))
                return ModelResponse(
                    parts=[TextPart(content=f"output {len(requests)}")]
                )

            class SequentialCrew(Crew):
                agent_definition = agents
                task_definition = tasks

            crew = SequentialCrew(
                agent_variables={"topic": [{"name": "bees"}]}
            )
            with ExitStack() as overrides:
                model = FunctionModel(respond)
                for agent in crew.agents.values():
                    overrides.enter_context(agent.override(model=model))
                output = anyio.run(
                    crew.kickoff, {"topic": [{"name": "wasps"}]}
                )

            self.assertEqual(set(crew.agents), {"researcher", "writer"})
            self.assertEqual(crew.agents["researcher"].name, "researcher")
            self.assertEqual(output, "output 2")
            self.assertIn("Role: [{'name': 'bees'}] Researcher", requests[0][0])
            self.assertIn(
                "Goal: Find facts about [{'name': 'bees'}]", requests[0][0]
            )
            self.assertIn(
                "Backstory: Curious about [{'name': 'bees'}]", requests[0][0]
            )
            self.assertIn("Research [{'name': 'wasps'}]", requests[0][1])
            self.assertIn(
                "Expected output:\nFacts about [{'name': 'wasps'}]",
                requests[0][1],
            )
            self.assertIn("Role: Writer", requests[1][0])
            self.assertIn("Previous task output:\noutput 1", requests[1][1])

    def test_requires_a_valid_llm_for_each_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agents = root / "agents.yaml"
            tasks = root / "tasks.yaml"
            tasks.write_text("{}\n")

            class InvalidCrew(Crew):
                agent_definition = agents
                task_definition = tasks

            agents.write_text(
                "researcher:\n"
                "  role: Researcher\n"
                "  goal: Research\n"
                "  backstory: Curious\n"
            )
            with self.assertRaises(ValidationError):
                InvalidCrew()

            agents.write_text(
                "researcher:\n"
                "  role: Researcher\n"
                "  goal: Research\n"
                "  backstory: Curious\n"
                "  llm: not-a-model\n"
            )
            with self.assertRaisesRegex(UserError, "Unknown model: not-a-model"):
                InvalidCrew()

    def test_requires_definition_overrides(self) -> None:
        class MissingBoth(Crew):
            pass

        with self.assertRaisesRegex(
            TypeError,
            "MissingBoth must define agent_definition, task_definition",
        ):
            MissingBoth()

        class MissingAgent(Crew):
            task_definition = "tasks.yaml"

        with self.assertRaisesRegex(
            TypeError, "MissingAgent must define agent_definition"
        ):
            MissingAgent()

        class MissingTask(Crew):
            agent_definition = "agents.yaml"

        with self.assertRaisesRegex(
            TypeError, "MissingTask must define task_definition"
        ):
            MissingTask()

    def test_reports_each_missing_definition_file(self) -> None:
        class MissingAgentCrew(Crew):
            agent_definition = "missing-agents.yaml"
            task_definition = "tasks.yaml"

        with self.assertRaisesRegex(
            FileNotFoundError, "agent_definition file not found"
        ):
            MissingAgentCrew()

        class MissingTaskCrew(Crew):
            agent_definition = "agents.yaml"
            task_definition = "missing-tasks.yaml"

        with self.assertRaisesRegex(
            FileNotFoundError, "task_definition file not found"
        ):
            MissingTaskCrew()

    def test_requires_an_agent_for_each_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agents = root / "agents.yaml"
            tasks = root / "tasks.yaml"
            agents.write_text(
                "researcher:\n"
                "  role: Researcher\n"
                "  goal: Research\n"
                "  backstory: Curious\n"
                "  llm: test\n"
            )
            tasks.write_text(
                "research:\n"
                "  description: Research the topic\n"
                "  expected_output: A report\n"
            )

            class InvalidCrew(Crew):
                agent_definition = agents
                task_definition = tasks

            with self.assertRaises(ValidationError):
                InvalidCrew()

    def test_rejects_unknown_task_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agents = root / "agents.yaml"
            tasks = root / "tasks.yaml"
            agents.write_text(
                "researcher:\n"
                "  role: Researcher\n"
                "  goal: Research\n"
                "  backstory: Curious\n"
                "  llm: test\n"
            )
            tasks.write_text(
                "research:\n"
                "  description: Research the topic\n"
                "  expected_output: A report\n"
                "  agent: writer\n"
            )

            class InvalidCrew(Crew):
                agent_definition = agents
                task_definition = tasks

            with self.assertRaisesRegex(ValueError, "undefined agent 'writer'"):
                InvalidCrew()


if __name__ == "__main__":
    unittest.main()
