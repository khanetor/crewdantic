import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from crewdantic import Crew


class ExampleCrew(Crew):
    agent_definition = "agents.yaml"
    task_definition = "tasks.yaml"


class CrewTests(unittest.TestCase):
    def test_loads_definitions_relative_to_crew_module(self) -> None:
        crew = ExampleCrew()

        self.assertEqual(crew.agent_definitions["researcher"].role, "Researcher")
        self.assertEqual(crew.task_definitions["research"].agent, "researcher")

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
