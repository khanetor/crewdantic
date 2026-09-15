import tempfile
import unittest
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import assert_type

import anyio
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ValidationError
from pydantic_ai import (
    BinaryContent,
    ImageUrl,
    ModelRequest,
    ModelResponse,
    RunContext,
    TextPart,
    Tool,
    ToolCallPart,
    ToolReturnPart,
    UsageLimitExceeded,
    UsageLimits,
    UserError,
    UserPromptPart,
)
from pydantic_ai.capabilities import MCP, AbstractCapability
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.function import FunctionModel
from typing_extensions import override

from crewdantic import Crew


@contextmanager
def definition_paths() -> Iterator[tuple[Path, Path]]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        yield root / "agents.yaml", root / "tasks.yaml"


class ResearchOutput(BaseModel):
    finding: str


@dataclass
class ToolContext:
    offset: int


class CrewTests(unittest.TestCase):
    def test_loads_relative_string_definition_paths_and_type_defaults(self) -> None:
        crew = Crew(
            agent_definition="agents.yaml",
            task_definition="tasks.yaml",
        )
        assert_type(crew, Crew[str, object])

        self.assertEqual(crew.agent_definitions["researcher"].role, "Researcher")
        self.assertEqual(crew.task_definitions["research"].agent, "researcher")

    def test_returns_mermaid_and_optionally_writes_mermaid_or_html(self) -> None:
        with definition_paths() as (_, tasks):
            tasks.write_text(
                '"`research":\n'
                "  description: Research\n"
                "  expected_output: Facts\n"
                "  agent: researcher\n"
                "write:\n"
                "  description: Write\n"
                "  expected_output: Report\n"
                "  agent: researcher\n"
            )
            crew = Crew(
                agent_definition=Path(__file__).with_name("agents.yaml"),
                task_definition=tasks,
            )
            expected = (
                "flowchart LR\n"
                '  task_0["#96;research<br/>agent: researcher"]\n'
                '  task_1["write<br/>agent: researcher"]\n'
                "  task_0 --> task_1"
            )

            mermaid = tasks.with_name("crew.mermaid")
            html = tasks.with_name("crew.html")

            self.assertEqual(crew.to_mermaid_chart(), expected)
            self.assertIsNone(crew.render(mermaid))
            self.assertEqual(mermaid.read_text(), expected)
            self.assertIsNone(crew.render(html))
            html_content = html.read_text()
            self.assertIn("flowchart LR", html_content)
            self.assertIn("mermaid@11", html_content)

            with self.assertRaises(ValueError):
                crew.render(tasks.with_name("crew.txt"))

    def test_builds_agents_and_runs_tasks_sequentially(self) -> None:
        with definition_paths() as (agents, tasks):
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
                    message
                    for message in reversed(messages)
                    if isinstance(message, ModelRequest)
                )
                prompt = next(
                    part.content
                    for part in request.parts
                    if isinstance(part, UserPromptPart)
                )
                assert isinstance(prompt, str)
                requests.append((instructions, prompt))
                return ModelResponse(
                    parts=[TextPart(content=f"output {len(requests)}")]
                )

            crew = Crew(
                agent_definition=agents,
                task_definition=tasks,
                agent_variables={"topic": [{"name": "bees"}]},
            )
            with ExitStack() as overrides:
                model = FunctionModel(respond)
                for agent in crew.agents.values():
                    overrides.enter_context(agent.override(model=model))
                output = anyio.run(
                    partial(crew.kickoff, inputs={"topic": [{"name": "wasps"}]})
                )

            self.assertEqual(set(crew.agents), {"researcher", "writer"})
            self.assertEqual(crew.agents["researcher"].name, "researcher")
            self.assertEqual(output, "output 2")
            self.assertIn("Role: [{'name': 'bees'}] Researcher", requests[0][0])
            self.assertIn("Goal: Find facts about [{'name': 'bees'}]", requests[0][0])
            self.assertIn("Backstory: Curious about [{'name': 'bees'}]", requests[0][0])
            self.assertIn("Research [{'name': 'wasps'}]", requests[0][1])
            self.assertIn(
                "Expected output:\nFacts about [{'name': 'wasps'}]",
                requests[0][1],
            )
            self.assertIn("Role: Writer", requests[1][0])
            self.assertIn("Previous task outputs:\noutput 1", requests[1][1])

    def test_uses_all_or_selected_previous_task_outputs_as_context(self) -> None:
        with definition_paths() as (agents, tasks):
            agents.write_text(
                "worker:\n"
                "  role: Worker\n"
                "  goal: Complete tasks\n"
                "  backstory: Reliable\n"
                "  llm: test\n"
            )
            tasks.write_text(
                "first:\n"
                "  description: First\n"
                "  expected_output: First output\n"
                "  agent: worker\n"
                "second:\n"
                "  description: Second\n"
                "  expected_output: Second output\n"
                "  agent: worker\n"
                "third:\n"
                "  description: Third\n"
                "  expected_output: Third output\n"
                "  agent: worker\n"
                "selected:\n"
                "  description: Selected\n"
                "  expected_output: Selected output\n"
                "  agent: worker\n"
                "  context:\n"
                "    - first\n"
                "    - third\n"
                "empty:\n"
                "  description: Empty\n"
                "  expected_output: Empty output\n"
                "  agent: worker\n"
                "  context: []\n"
                '"null":\n'
                "  description: No context\n"
                "  expected_output: Null output\n"
                "  agent: worker\n"
                "  context: null\n"
            )
            prompts: list[str] = []

            def respond(messages, _info):
                prompt = next(
                    part.content
                    for message in reversed(messages)
                    if isinstance(message, ModelRequest)
                    for part in message.parts
                    if isinstance(part, UserPromptPart)
                )
                assert isinstance(prompt, str)
                prompts.append(prompt)
                return ModelResponse(parts=[TextPart(content=f"output {len(prompts)}")])

            crew = Crew(agent_definition=agents, task_definition=tasks)
            with crew.agents["worker"].override(model=FunctionModel(respond)):
                output = anyio.run(crew.kickoff)

            self.assertEqual(output, "output 6")
            self.assertNotIn("Previous task output", prompts[0])
            self.assertTrue(
                prompts[1].endswith("Previous task outputs:\noutput 1"),
                prompts[1],
            )
            self.assertTrue(
                prompts[2].endswith("Previous task outputs:\noutput 1\n\noutput 2"),
                prompts[2],
            )
            self.assertTrue(
                prompts[3].endswith("Previous task outputs:\noutput 1\n\noutput 3"),
                prompts[3],
            )
            self.assertNotIn("output 2", prompts[3])
            self.assertNotIn("Previous task output", prompts[4])
            self.assertNotIn("Previous task output", prompts[5])

    def test_rejects_context_references_to_non_previous_tasks(self) -> None:
        with definition_paths() as (agents, tasks):
            agents.write_text(
                "worker:\n"
                "  role: Worker\n"
                "  goal: Complete tasks\n"
                "  backstory: Reliable\n"
                "  llm: test\n"
            )
            tasks.write_text(
                "first:\n"
                "  description: First\n"
                "  expected_output: First output\n"
                "  agent: worker\n"
                "  context: [later]\n"
                "later:\n"
                "  description: Later\n"
                "  expected_output: Later output\n"
                "  agent: worker\n"
            )

            with self.assertRaises(ValueError):
                Crew(agent_definition=agents, task_definition=tasks)

    def test_runs_hierarchical_tasks_through_manager(self) -> None:
        with definition_paths() as (agents, tasks):
            agents.write_text(
                "manager:\n"
                "  manager: true\n"
                "  llm: test\n"
                "  output_model: ResearchOutput\n"
                "researcher:\n"
                "  role: Researcher\n"
                "  goal: Find facts\n"
                "  backstory: Curious\n"
                "  llm: test\n"
                "writer:\n"
                "  role: Writer\n"
                "  goal: Explain facts\n"
                "  backstory: Clear\n"
                "  llm: test\n"
            )
            tasks.write_text(
                "research:\n"
                "  description: Research bees\n"
                "  expected_output: Verified facts\n"
                "  agent: researcher\n"
                "outline:\n"
                "  description: Outline the report\n"
                "  expected_output: A clear outline\n"
                "compose:\n"
                "  description: Compose the report\n"
                "  expected_output: One structured finding\n"
            )
            worker_calls: list[tuple[str, str]] = []

            def prompt_from(messages) -> str:
                prompt = next(
                    part.content
                    for message in messages
                    for part in message.parts
                    if isinstance(part, UserPromptPart)
                )
                assert isinstance(prompt, str)
                return prompt

            def manage(messages, info):
                prompt = prompt_from(messages)
                tool_returns = [
                    part
                    for message in messages
                    if isinstance(message, ModelRequest)
                    for part in message.parts
                    if isinstance(part, ToolReturnPart)
                ]
                self.assertEqual(
                    {tool.name for tool in info.function_tools},
                    {"ask_question", "delegate_work"},
                )

                if prompt.startswith("Research"):
                    if not tool_returns:
                        return ModelResponse(
                            parts=[
                                ToolCallPart(
                                    tool_name="delegate_work",
                                    args={
                                        "task": "Find one bee fact",
                                        "context": "The report is about bees",
                                        "coworker": "researcher",
                                    },
                                    tool_call_id="research",
                                ),
                                ToolCallPart(
                                    tool_name="ask_question",
                                    args={
                                        "question": "Can you write this?",
                                        "context": "The report is about bees",
                                        "coworker": "writer",
                                    },
                                    tool_call_id="disallowed",
                                ),
                            ]
                        )
                    self.assertEqual(
                        {part.tool_call_id: part.content for part in tool_returns},
                        {
                            "research": "Worker findings",
                            "disallowed": (
                                "Unknown coworker 'writer'. "
                                "Available coworkers: 'researcher'"
                            ),
                        },
                    )
                    return ModelResponse(parts=[TextPart(content="Managed research")])

                if prompt.startswith("Outline"):
                    self.assertIn("Managed research", prompt)
                    if not tool_returns:
                        return ModelResponse(
                            parts=[
                                ToolCallPart(
                                    tool_name="ask_question",
                                    args={
                                        "question": "Suggest an outline",
                                        "context": "Worker findings",
                                        "coworker": "writer",
                                    },
                                    tool_call_id="outline",
                                )
                            ]
                        )
                    self.assertEqual(tool_returns[0].content, "Worker outline")
                    return ModelResponse(parts=[TextPart(content="Managed outline")])

                self.assertIn("Managed research", prompt)
                self.assertIn("Managed outline", prompt)
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=info.output_tools[0].name,
                            args={"finding": "Bees communicate by dancing."},
                        )
                    ]
                )

            def work(messages, info):
                prompt = prompt_from(messages)
                worker_calls.append((info.instructions or "", prompt))
                if prompt.startswith("Find"):
                    return ModelResponse(parts=[TextPart(content="Worker findings")])
                return ModelResponse(parts=[TextPart(content="Worker outline")])

            crew = Crew(
                agent_definition=agents,
                task_definition=tasks,
                output_type=ResearchOutput,
            )
            self.assertTrue(crew.agent_definitions["manager"].manager)
            self.assertFalse(crew.agent_definitions["researcher"].manager)
            self.assertEqual(
                crew.agent_definitions["manager"].role,
                "Crew Manager",
            )

            with (
                crew.agents["manager"].override(model=FunctionModel(manage)),
                crew.agents["researcher"].override(model=FunctionModel(work)),
                crew.agents["writer"].override(model=FunctionModel(work)),
            ):
                output = anyio.run(crew.kickoff)

            self.assertEqual(
                output,
                ResearchOutput(finding="Bees communicate by dancing."),
            )
            self.assertEqual(len(worker_calls), 2)
            self.assertIn("Role: Researcher", worker_calls[0][0])
            self.assertIn("Find one bee fact", worker_calls[0][1])
            self.assertIn("Role: Writer", worker_calls[1][0])
            self.assertIn("Suggest an outline", worker_calls[1][1])
            self.assertIn("manager: manager", crew.to_mermaid_chart())
            self.assertIn("worker: any", crew.to_mermaid_chart())

            with (
                crew.agents["manager"].override(model=FunctionModel(manage)),
                crew.agents["researcher"].override(model=FunctionModel(work)),
                crew.agents["writer"].override(model=FunctionModel(work)),
                self.assertRaises(UsageLimitExceeded),
            ):
                anyio.run(
                    partial(crew.kickoff, usage_limits=UsageLimits(request_limit=1))
                )

    def test_manager_delegation_forwards_file_inputs_to_workers(self) -> None:
        with definition_paths() as (agents, tasks):
            agents.write_text(
                "manager:\n"
                "  manager: true\n"
                "  llm: test\n"
                "  output_model: ResearchOutput\n"
                "researcher:\n"
                "  role: Researcher\n"
                "  goal: Find facts\n"
                "  backstory: Curious\n"
                "  llm: test\n"
            )
            tasks.write_text(
                "research:\n"
                "  description: Research bees using {chart}\n"
                "  expected_output: Verified facts\n"
            )
            image = BinaryContent(data=b"png-bytes", media_type="image/png")
            worker_prompts: list[object] = []

            def manage(messages, info):
                request = next(
                    message
                    for message in reversed(messages)
                    if isinstance(message, ModelRequest)
                )
                if not any(isinstance(part, ToolReturnPart) for part in request.parts):
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                tool_name="delegate_work",
                                args={
                                    "task": "Find one bee fact",
                                    "context": "The report is about bees",
                                    "coworker": "researcher",
                                },
                                tool_call_id="research",
                            )
                        ]
                    )
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=info.output_tools[0].name,
                            args={"finding": "Bees communicate by dancing."},
                        )
                    ]
                )

            def work(messages, info):
                request = next(
                    message for message in messages if isinstance(message, ModelRequest)
                )
                prompt = next(
                    part.content
                    for part in request.parts
                    if isinstance(part, UserPromptPart)
                )
                worker_prompts.append(prompt)
                return ModelResponse(parts=[TextPart(content="Worker findings")])

            crew = Crew(
                agent_definition=agents,
                task_definition=tasks,
                output_type=ResearchOutput,
            )
            with (
                crew.agents["manager"].override(model=FunctionModel(manage)),
                crew.agents["researcher"].override(model=FunctionModel(work)),
            ):
                output = anyio.run(
                    partial(
                        crew.kickoff,
                        file_inputs={"chart": image},
                    )
                )

            self.assertEqual(
                output,
                ResearchOutput(finding="Bees communicate by dancing."),
            )
            self.assertEqual(len(worker_prompts), 1)
            parts = worker_prompts[0]
            assert isinstance(parts, list)
            assert isinstance(parts[0], str)
            self.assertIn("Find one bee fact", parts[0])
            self.assertEqual(parts[1], image)

    def test_rejects_multiple_manager_agents(self) -> None:
        with definition_paths() as (agents, tasks):
            agents.write_text(
                "first:\n"
                "  manager: true\n"
                "  llm: test\n"
                "second:\n"
                "  manager: true\n"
                "  llm: test\n"
                "worker:\n"
                "  role: Worker\n"
                "  goal: Work\n"
                "  backstory: Capable\n"
                "  llm: test\n"
            )
            tasks.write_text("work:\n  description: Work\n  expected_output: Result\n")

            with self.assertRaises(ValueError):
                Crew(agent_definition=agents, task_definition=tasks)

    def test_rejects_invalid_hierarchical_configurations(self) -> None:
        with definition_paths() as (agents, tasks):
            manager = "manager:\n  manager: true\n  llm: test\n"
            worker = (
                "worker:\n"
                "  role: Worker\n"
                "  goal: Work\n"
                "  backstory: Capable\n"
                "  llm: test\n"
            )
            task = "work:\n  description: Work\n  expected_output: Result\n"
            cases = (
                ("manager without worker", manager, task, ValueError),
                (
                    "task assigned to manager",
                    worker + manager,
                    task + "  agent: manager\n",
                    ValueError,
                ),
                (
                    "unknown worker",
                    manager + worker,
                    task + "  agent: missing\n",
                    ValueError,
                ),
                ("empty tasks", manager + worker, "{}\n", TypeError),
            )

            for name, agent_yaml, task_yaml, error_type in cases:
                with self.subTest(name):
                    agents.write_text(agent_yaml)
                    tasks.write_text(task_yaml)
                    with self.assertRaises(error_type):
                        Crew(agent_definition=agents, task_definition=tasks)

            agents.write_text(
                "manager:\n"
                "  manager: true\n"
                "  llm: test\n"
                "  output_model: ResearchOutput\n" + worker
            )
            tasks.write_text(task)
            with self.assertRaises(TypeError):
                Crew(
                    agent_definition=agents,
                    task_definition=tasks,
                    pydantic_models=[ResearchOutput],
                )

    def test_rejects_manager_delegation_tool_name_collisions(self) -> None:
        def delegate_work(_ctx: RunContext[object]) -> str:
            return "custom"

        def ask_question(_ctx: RunContext[object]) -> str:
            return "custom"

        with definition_paths() as (agents, tasks):
            tasks.write_text("work:\n  description: Work\n  expected_output: Result\n")

            for tool in (delegate_work, ask_question):
                with self.subTest(tool.__name__):
                    agents.write_text(
                        "manager:\n"
                        "  manager: true\n"
                        "  llm: test\n"
                        "  tools:\n"
                        f"    - {tool.__name__}\n"
                        "worker:\n"
                        "  role: Worker\n"
                        "  goal: Work\n"
                        "  backstory: Capable\n"
                        "  llm: test\n"
                    )
                    with self.assertRaises(ValueError):
                        Crew(
                            agent_definition=agents,
                            task_definition=tasks,
                            tools=[tool],
                        )

    def test_supports_pydantic_ai_tools_capabilities_and_decorators(self) -> None:
        with definition_paths() as (agents, tasks):
            agents.write_text(
                "researcher:\n"
                "  role: Researcher\n"
                "  goal: Use tools\n"
                "  backstory: Capable\n"
                "  llm: test\n"
                "  tools:\n"
                "    - increment\n"
                "    - double\n"
                "  capabilities:\n"
                "    - ExtraInstructions\n"
            )
            tasks.write_text(
                "research:\n"
                "  description: Calculate\n"
                "  expected_output: A result\n"
                "  agent: researcher\n"
            )

            def increment(ctx: RunContext[ToolContext], value: int) -> int:
                return value + 1 + ctx.deps.offset

            def double(ctx: RunContext[ToolContext], value: int) -> int:
                return value * 2 + ctx.deps.offset

            def unused(ctx: RunContext[ToolContext], value: int) -> int:
                return value + ctx.deps.offset

            class ExtraInstructions(AbstractCapability[ToolContext]):
                @override
                def get_instructions(self) -> str:
                    return "Capability enabled."

            class UnusedInstructions(AbstractCapability[ToolContext]):
                @override
                def get_instructions(self) -> str:
                    return "Unused capability."

            crew = Crew(
                agent_definition=agents,
                task_definition=tasks,
                deps_type=ToolContext,
                tools=[Tool(increment, name="increment"), double, unused],
                capabilities=[ExtraInstructions(), UnusedInstructions()],
            )

            @crew.agents["researcher"].tool
            def decrement(ctx: RunContext[ToolContext], value: int) -> int:
                return value - 1 + ctx.deps.offset

            def respond(messages, info):
                tool_returns = [
                    part
                    for message in messages
                    if isinstance(message, ModelRequest)
                    for part in message.parts
                    if isinstance(part, ToolReturnPart)
                ]
                if not tool_returns:
                    self.assertEqual(
                        {tool.name for tool in info.function_tools},
                        {"increment", "double", "decrement"},
                    )
                    self.assertIn("Capability enabled.", info.instructions or "")
                    self.assertNotIn("Unused capability.", info.instructions or "")
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                tool_name="increment",
                                args={"value": 1},
                                tool_call_id="increment",
                            ),
                            ToolCallPart(
                                tool_name="double",
                                args={"value": 2},
                                tool_call_id="double",
                            ),
                            ToolCallPart(
                                tool_name="decrement",
                                args={"value": 4},
                                tool_call_id="decrement",
                            ),
                        ]
                    )

                self.assertEqual(
                    {part.tool_name: part.content for part in tool_returns},
                    {"increment": 12, "double": 14, "decrement": 13},
                )
                return ModelResponse(parts=[TextPart(content="Tools worked")])

            self.assertIs(crew.agents["researcher"].deps_type, ToolContext)
            with crew.agents["researcher"].override(model=FunctionModel(respond)):
                output = anyio.run(
                    partial(crew.kickoff, context=ToolContext(offset=10))
                )

            self.assertEqual(output, "Tools worked")
            agents.write_text(agents.read_text().replace("increment", "missing"))
            with self.assertRaises(ValueError):
                Crew(
                    agent_definition=agents,
                    task_definition=tasks,
                    deps_type=ToolContext,
                    tools=[Tool(increment, name="increment"), double],
                )

            agents.write_text(
                agents.read_text()
                .replace("missing", "increment")
                .replace("ExtraInstructions", "MissingCapability")
            )
            with self.assertRaises(ValueError):
                Crew(
                    agent_definition=agents,
                    task_definition=tasks,
                    deps_type=ToolContext,
                    tools=[Tool(increment, name="increment"), double],
                    capabilities=[ExtraInstructions()],
                )

    def test_supports_mcp_capabilities_and_toolsets(self) -> None:
        def exercise(yaml_key: str) -> None:
            with definition_paths() as (agents, tasks):
                agents.write_text(
                    "researcher:\n"
                    "  role: Researcher\n"
                    "  goal: Use MCP tools\n"
                    "  backstory: Connected\n"
                    "  llm: test\n"
                    f"  {yaml_key}:\n"
                    "    - math\n"
                )
                tasks.write_text(
                    "research:\n"
                    "  description: Calculate\n"
                    "  expected_output: A result\n"
                    "  agent: researcher\n"
                )
                calls: list[int] = []

                server = FastMCP("math")

                @server.tool()
                def increment(value: int) -> int:
                    calls.append(value)
                    return value + 1

                def build_crew():
                    if yaml_key == "capabilities":
                        return Crew(
                            agent_definition=agents,
                            task_definition=tasks,
                            capabilities=[MCP(local=server, id="math")],
                        )
                    return Crew(
                        agent_definition=agents,
                        task_definition=tasks,
                        toolsets={"math": MCPToolset(server, id="math")},
                    )

                crew = build_crew()

                def respond(_messages, info):
                    if not calls:
                        self.assertEqual(
                            {tool.name for tool in info.function_tools}, {"increment"}
                        )
                        return ModelResponse(
                            parts=[
                                ToolCallPart(
                                    tool_name="increment",
                                    args={"value": 2},
                                    tool_call_id="increment",
                                )
                            ]
                        )
                    return ModelResponse(parts=[TextPart(content="MCP worked")])

                with crew.agents["researcher"].override(model=FunctionModel(respond)):
                    output = anyio.run(crew.kickoff)

                self.assertEqual(output, "MCP worked")
                self.assertEqual(calls, [2])

                agents.write_text(agents.read_text().replace("math", "missing"))
                with self.assertRaises(ValueError):
                    build_crew()

        for yaml_key in ("capabilities", "toolsets"):
            with self.subTest(yaml_key=yaml_key):
                exercise(yaml_key)

    def test_uses_final_agent_output_model(self) -> None:
        with definition_paths() as (agents, tasks):
            agents.write_text(
                "researcher:\n"
                "  role: Researcher\n"
                "  goal: Find facts\n"
                "  backstory: Curious\n"
                "  llm: test\n"
                "composer:\n"
                "  role: Composer\n"
                "  goal: Write clearly\n"
                "  backstory: Clear\n"
                "  llm: test\n"
                "  output_model: ResearchOutput\n"
            )
            tasks.write_text(
                "research:\n"
                "  description: Research bees\n"
                "  expected_output: Raw findings\n"
                "  agent: researcher\n"
                "compose:\n"
                "  description: Compose the final report\n"
                "  expected_output: One structured finding\n"
                "  agent: composer\n"
            )

            crew = Crew(
                agent_definition=agents,
                task_definition=tasks,
                output_type=ResearchOutput,
            )

            def respond(messages, info):
                if not info.output_tools:
                    return ModelResponse(parts=[TextPart(content="Raw findings")])
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name=info.output_tools[0].name,
                            args={"finding": "Bees communicate by dancing."},
                        )
                    ]
                )

            with ExitStack() as overrides:
                model = FunctionModel(respond)
                for agent in crew.agents.values():
                    overrides.enter_context(agent.override(model=model))
                output = anyio.run(crew.kickoff)

            assert_type(output, ResearchOutput)
            self.assertEqual(
                output,
                ResearchOutput(finding="Bees communicate by dancing."),
            )
            self.assertIs(crew.agents["researcher"].output_type, str)
            self.assertIs(crew.agents["composer"].output_type, ResearchOutput)

    def test_rejects_final_agent_output_mismatch_at_construction(self) -> None:
        with definition_paths() as (agents, tasks):
            agents.write_text(
                "writer:\n"
                "  role: Writer\n"
                "  goal: Write clearly\n"
                "  backstory: Clear\n"
                "  llm: test\n"
            )
            tasks.write_text(
                "write:\n"
                "  description: Write the report\n"
                "  expected_output: A report\n"
                "  agent: writer\n"
            )

            with self.assertRaises(TypeError):
                Crew(
                    agent_definition=agents,
                    task_definition=tasks,
                    output_type=ResearchOutput,
                )

    def test_rejects_unknown_agent_output_model(self) -> None:
        with definition_paths() as (agents, tasks):
            agents.write_text(
                "researcher:\n"
                "  role: Researcher\n"
                "  goal: Find facts\n"
                "  backstory: Curious\n"
                "  llm: test\n"
                "  output_model: MissingOutput\n"
            )
            tasks.write_text(
                "research:\n"
                "  description: Research\n"
                "  expected_output: Findings\n"
                "  agent: researcher\n"
            )

            with self.assertRaises(ValueError):
                Crew(
                    agent_definition=agents,
                    task_definition=tasks,
                    pydantic_models=[ResearchOutput],
                )

    def test_requires_a_valid_llm_for_each_agent(self) -> None:
        with definition_paths() as (agents, tasks):
            tasks.write_text(
                "research:\n"
                "  description: Research\n"
                "  expected_output: Findings\n"
                "  agent: researcher\n"
            )
            agents.write_text(
                "researcher:\n"
                "  role: Researcher\n"
                "  goal: Research\n"
                "  backstory: Curious\n"
            )
            with self.assertRaises(ValidationError):
                Crew(
                    agent_definition=agents,
                    task_definition=tasks,
                )

            agents.write_text(
                "researcher:\n"
                "  role: Researcher\n"
                "  goal: Research\n"
                "  backstory: Curious\n"
                "  llm: not-a-model\n"
            )
            with self.assertRaises(UserError):
                Crew(
                    agent_definition=agents,
                    task_definition=tasks,
                )

    def test_reports_each_missing_definition_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            Crew(
                agent_definition="missing-agents.yaml",
                task_definition="tasks.yaml",
            )

        with self.assertRaises(FileNotFoundError):
            Crew(
                agent_definition="agents.yaml",
                task_definition="missing-tasks.yaml",
            )

    def test_requires_an_agent_for_each_task(self) -> None:
        with definition_paths() as (agents, tasks):
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

            with self.assertRaises(ValueError):
                Crew(
                    agent_definition=agents,
                    task_definition=tasks,
                )

    def test_rejects_unknown_task_agent(self) -> None:
        with definition_paths() as (agents, tasks):
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

            with self.assertRaises(ValueError):
                Crew(
                    agent_definition=agents,
                    task_definition=tasks,
                )

    def test_rejects_unknown_definition_keys(self) -> None:
        with definition_paths() as (agents, tasks):
            agents.write_text(
                "researcher:\n"
                "  role: Researcher\n"
                "  goal: Research\n"
                "  backstory: Curious\n"
                "  llm: test\n"
                "  tool: typo\n"
            )
            tasks.write_text(
                "research:\n"
                "  description: Research the topic\n"
                "  expected_output: A report\n"
                "  agent: researcher\n"
            )

            with self.assertRaises(ValidationError):
                Crew(
                    agent_definition=agents,
                    task_definition=tasks,
                )

    def test_rejects_undefined_agent_variable_placeholder(self) -> None:
        with definition_paths() as (agents, tasks):
            agents.write_text(
                "researcher:\n"
                "  role: '{missing_var} Researcher'\n"
                "  goal: Research\n"
                "  backstory: Curious\n"
                "  llm: test\n"
            )
            tasks.write_text(
                "research:\n"
                "  description: Research the topic\n"
                "  expected_output: A report\n"
                "  agent: researcher\n"
            )

            with self.assertRaisesRegex(ValueError, "missing_var"):
                Crew(
                    agent_definition=agents,
                    task_definition=tasks,
                )

    def test_rejects_undefined_task_input_placeholder(self) -> None:
        with definition_paths() as (agents, tasks):
            agents.write_text(
                "researcher:\n"
                "  role: Researcher\n"
                "  goal: Research\n"
                "  backstory: Curious\n"
                "  llm: test\n"
            )
            tasks.write_text(
                "research:\n"
                "  description: Research {undefined_var}\n"
                "  expected_output: A report\n"
                "  agent: researcher\n"
            )
            crew = Crew(
                agent_definition=agents,
                task_definition=tasks,
            )

            with self.assertRaisesRegex(
                ValueError, "Task 'research': Undefined placeholder"
            ):
                anyio.run(crew.kickoff)

    def test_kickoff_injects_file_inputs_at_labeled_placeholders(self) -> None:
        with definition_paths() as (agents, tasks):
            agents.write_text(
                "researcher:\n"
                "  role: Researcher\n"
                "  goal: Research\n"
                "  backstory: Curious\n"
                "  llm: test\n"
            )
            tasks.write_text(
                "research:\n"
                "  description: Review {chart} against {report}\n"
                "  expected_output: A report\n"
                "  agent: researcher\n"
            )
            contents: list[object] = []

            def respond(messages, info):
                request = next(
                    message
                    for message in reversed(messages)
                    if isinstance(message, ModelRequest)
                )
                part = next(
                    part for part in request.parts if isinstance(part, UserPromptPart)
                )
                contents.append(part.content)
                return ModelResponse(parts=[TextPart(content="done")])

            image = BinaryContent(data=b"png-bytes", media_type="image/png")
            url = ImageUrl(url="https://example.com/chart.png")
            crew = Crew(agent_definition=agents, task_definition=tasks)
            with crew.agents["researcher"].override(model=FunctionModel(respond)):
                output = anyio.run(
                    partial(
                        crew.kickoff,
                        inputs={"topic": "sales"},
                        file_inputs={"chart": image, "report": url},
                    )
                )

            self.assertEqual(output, "done")
            parts = contents[0]
            assert isinstance(parts, list)
            self.assertEqual(
                parts,
                [
                    "Review ",
                    image,
                    " against ",
                    url,
                    "\n\nExpected output:\nA report",
                ],
            )

    def test_kickoff_rejects_unreferenced_file_inputs(self) -> None:
        with definition_paths() as (agents, tasks):
            agents.write_text(
                "researcher:\n"
                "  role: Researcher\n"
                "  goal: Research\n"
                "  backstory: Curious\n"
                "  llm: test\n"
            )
            tasks.write_text(
                "research:\n"
                "  description: Analyze nothing\n"
                "  expected_output: A report\n"
                "  agent: researcher\n"
            )
            crew = Crew(agent_definition=agents, task_definition=tasks)
            with self.assertRaisesRegex(
                ValueError,
                "Task 'research' does not reference file input\\(s\\): 'chart'",
            ):
                anyio.run(
                    partial(
                        crew.kickoff,
                        file_inputs={
                            "chart": BinaryContent(data=b"x", media_type="image/png")
                        },
                    )
                )


if __name__ == "__main__":
    unittest.main()
