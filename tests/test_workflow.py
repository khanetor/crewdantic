import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import assert_type

import anyio
from typing_extensions import override

from crewdantic import Workflow, step


@dataclass(frozen=True, slots=True)
class ReportInput:
    topic: str


@dataclass(frozen=True, slots=True)
class Research:
    finding: str


@dataclass(frozen=True, slots=True)
class Market:
    audience: str


@dataclass(frozen=True, slots=True)
class Report:
    topic: str
    finding: str
    audience: str


class ReportWorkflow(Workflow[ReportInput, Report]):
    def __init__(self) -> None:
        self.calls: list[str] = []

    @step
    async def research(self, workflow_input: ReportInput) -> Research:
        self.calls.append("research")
        return Research(f"Finding about {workflow_input.topic}")

    @step
    def market(self, workflow_input: ReportInput) -> Market:
        self.calls.append("market")
        return Market("developers")

    @step
    def final_step(
        self,
        workflow_input: ReportInput,
        research: Research,
        market: Market,
    ) -> Report:
        self.calls.append("final_step")
        return Report(workflow_input.topic, research.finding, market.audience)


class WorkflowTests(unittest.TestCase):
    def test_resolves_step_outputs_and_returns_final_output(self) -> None:
        workflow = ReportWorkflow()
        assert_type(workflow, ReportWorkflow)

        report = anyio.run(
            workflow.kickoff,
            ReportInput(topic="typed workflows"),
        )

        assert_type(report, Report)
        self.assertEqual(
            report,
            Report(
                topic="typed workflows",
                finding="Finding about typed workflows",
                audience="developers",
            ),
        )
        self.assertCountEqual(workflow.calls[:-1], ["research", "market"])
        self.assertEqual(workflow.calls[-1], "final_step")

    def test_visualizes_branching_dependencies_and_writes_outputs(self) -> None:
        class BranchingWorkflow(Workflow[ReportInput, Report]):
            @step
            def source(self, workflow_input: ReportInput) -> str:
                return workflow_input.topic

            @step
            def research(self, source: str) -> Research:
                return Research(source)

            @step
            def market(self, source: str) -> Market:
                return Market(source)

            @step
            def final_step(self, research: Research, market: Market) -> Report:
                return Report("", research.finding, market.audience)

        workflow = BranchingWorkflow()
        expected = (
            "flowchart LR\n"
            '  step_0["source"]\n'
            '  step_1["research"]\n'
            '  step_2["market"]\n'
            '  step_3["final_step<br/>final"]\n'
            "  step_0 --> step_1\n"
            "  step_0 --> step_2\n"
            "  step_1 --> step_3\n"
            "  step_2 --> step_3"
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mermaid = root / "workflow.mermaid"
            html = root / "workflow.html"

            self.assertEqual(workflow.to_mermaid_chart(), expected)
            self.assertIsNone(workflow.render(mermaid))
            self.assertEqual(mermaid.read_text(), expected)
            self.assertIsNone(workflow.render(html))
            html_content = html.read_text()
            self.assertIn("Crewdantic visualization", html_content)
            self.assertIn("mermaid@11", html_content)

            with self.assertRaises(ValueError):
                workflow.render(root / "workflow.txt")

    def test_rejects_visualization_method_names_as_steps(self) -> None:
        with self.assertRaises(TypeError):

            class ChartStep(Workflow[ReportInput, Report]):
                @step
                @override
                def to_mermaid_chart(self) -> str:
                    return "chart"

                @step
                def final_step(self, to_mermaid_chart: str) -> Report:
                    return Report(to_mermaid_chart, "", "")

        with self.assertRaises(TypeError):

            class RenderStep(Workflow[ReportInput, Report]):
                @step
                def filename(self) -> str | Path:
                    return "workflow.html"

                @step
                @override
                def render(self, filename: str | Path) -> None:
                    _ = filename

                @step
                def final_step(self, render: None) -> Report:
                    return Report("", "", "")

    def test_requires_final_step(self) -> None:
        with self.assertRaises(TypeError):

            class MissingFinal(Workflow[ReportInput, Report]):
                @step
                async def research(self, workflow_input: ReportInput) -> Research:
                    return Research(workflow_input.topic)

    def test_rejects_undefined_upstream_step(self) -> None:
        with self.assertRaises(ValueError):

            class MissingUpstream(Workflow[ReportInput, Report]):
                @step
                async def final_step(self, research: Research) -> Report:
                    return Report("", research.finding, "")

    def test_rejects_invalid_dependency_types(self) -> None:
        with self.assertRaises(TypeError):

            class InvalidDependency(Workflow[ReportInput, Report]):
                @step
                async def research(self, workflow_input: ReportInput) -> Research:
                    return Research(workflow_input.topic)

                @step
                async def final_step(self, research: str) -> Report:
                    return Report("", research, "")

    def test_rejects_invalid_workflow_boundary_types(self) -> None:
        with self.assertRaises(TypeError):

            class InvalidInput(Workflow[ReportInput, Report]):
                @step
                async def final_step(self, workflow_input: str) -> Report:
                    return Report(workflow_input, "", "")

        with self.assertRaises(TypeError):

            class InvalidOutput(Workflow[ReportInput, Report]):
                @step
                async def final_step(self, workflow_input: ReportInput) -> str:
                    return workflow_input.topic

    def test_rejects_variadic_step_parameters(self) -> None:
        with self.assertRaises(TypeError):

            class VariadicWorkflow(Workflow[ReportInput, Report]):
                @step
                async def final_step(self, *workflow_input: ReportInput) -> Report:
                    return Report(workflow_input[0].topic, "", "")

    def test_rejects_cycles_and_orphaned_steps(self) -> None:
        with self.assertRaises(ValueError):

            class CyclicWorkflow(Workflow[ReportInput, Report]):
                @step
                async def first(self, second: int) -> int:
                    return second

                @step
                async def second(self, first: int) -> int:
                    return first

                @step
                async def final_step(self, first: int) -> Report:
                    return Report("", str(first), "")

        with self.assertRaises(ValueError):

            class OrphanedWorkflow(Workflow[ReportInput, Report]):
                @step
                async def unused(self) -> str:
                    return "unused"

                @step
                async def final_step(self, workflow_input: ReportInput) -> Report:
                    return Report(workflow_input.topic, "", "")

    def test_rejects_steps_downstream_of_final_step(self) -> None:
        with self.assertRaises(ValueError):

            class DownstreamOfFinal(Workflow[ReportInput, Report]):
                @step
                async def final_step(self, workflow_input: ReportInput) -> Report:
                    return Report(workflow_input.topic, "", "")

                @step
                async def after(self, final_step: Report) -> str:
                    return final_step.topic

    def test_inherits_and_overrides_steps(self) -> None:
        class BaseReportWorkflow(Workflow[ReportInput, Report]):
            @step
            async def research(self, workflow_input: ReportInput) -> Research:
                return Research(f"Finding about {workflow_input.topic}")

            @step
            def final_step(
                self,
                workflow_input: ReportInput,
                research: Research,
            ) -> Report:
                return Report(workflow_input.topic, research.finding, "")

        class ExtendedReportWorkflow(BaseReportWorkflow):
            @step
            def market(self, workflow_input: ReportInput) -> Market:
                return Market("developers")

            @step
            @override
            def final_step(  # ty: ignore[incompatible-method-override]
                self,
                workflow_input: ReportInput,
                research: Research,
                market: Market,
            ) -> Report:
                return Report(workflow_input.topic, research.finding, market.audience)

        report = anyio.run(
            ExtendedReportWorkflow().kickoff,
            ReportInput(topic="inherited steps"),
        )

        self.assertEqual(
            report,
            Report("inherited steps", "Finding about inherited steps", "developers"),
        )

    def test_rejects_non_step_visualization_overrides(self) -> None:
        with self.assertRaises(TypeError):

            class ChartOverride(Workflow[ReportInput, Report]):
                @override
                def to_mermaid_chart(self) -> str:
                    return "custom chart"

                @step
                def final_step(self, workflow_input: ReportInput) -> Report:
                    return Report(workflow_input.topic, "", "")

        with self.assertRaises(TypeError):

            class RenderOverride(Workflow[ReportInput, Report]):
                @override
                def render(self, filename: str | Path) -> None:
                    _ = filename

                @step
                def final_step(self, workflow_input: ReportInput) -> Report:
                    return Report(workflow_input.topic, "", "")


if __name__ == "__main__":
    unittest.main()
