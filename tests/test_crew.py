import tempfile
import unittest
from collections.abc import AsyncIterator
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import assert_type

import anyio
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ValidationError
from pydantic_ai import (
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

from crewdantic import Crew


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
        self.assertFalse(hasattr(crew.agent_definitions["researcher"], "verbose"))
        self.assertFalse(hasattr(crew.task_definitions["research"], "markdown"))

    def test_returns_mermaid_and_optionally_writes_mermaid_or_html(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = root / "tasks.yaml"
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

            mermaid = root / "crew.mermaid"
            html = root / "crew.html"

            self.assertEqual(crew.to_mermaid_chart(), expected)
            self.assertIsNone(anyio.run(crew.render, mermaid))
            self.assertEqual(mermaid.read_text(), expected)
            self.assertIsNone(anyio.run(crew.render, html))
            html_content = html.read_text()
            self.assertIn("flowchart LR", html_content)
            self.assertIn("mermaid@11", html_content)

            with self.assertRaisesRegex(ValueError, ".mermaid or .html"):
                anyio.run(crew.render, root / "crew.txt")

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
                output = anyio.run(crew.kickoff, {"topic": [{"name": "wasps"}]})

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
            self.assertIn("Previous task output:\noutput 1", requests[1][1])

    def test_runs_hierarchical_tasks_through_manager(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agents = root / "agents.yaml"
            tasks = root / "tasks.yaml"
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
                    crew.kickoff,
                    None,
                    None,
                    UsageLimits(request_limit=1),
                )

    def test_rejects_multiple_manager_agents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agents = root / "agents.yaml"
            tasks = root / "tasks.yaml"
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

            with self.assertRaisesRegex(
                ValueError,
                "multiple manager agents: 'first', 'second'",
            ):
                Crew(agent_definition=agents, task_definition=tasks)

    def test_rejects_invalid_hierarchical_configurations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agents = root / "agents.yaml"
            tasks = root / "tasks.yaml"
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
                (
                    "manager without worker",
                    manager,
                    task,
                    ValueError,
                    "at least one worker agent",
                ),
                (
                    "task assigned to manager",
                    worker + manager,
                    task + "  agent: manager\n",
                    ValueError,
                    "cannot assign manager 'manager' as a worker",
                ),
                (
                    "unknown worker",
                    manager + worker,
                    task + "  agent: missing\n",
                    ValueError,
                    "references undefined agent 'missing'",
                ),
                (
                    "empty tasks",
                    manager + worker,
                    "{}\n",
                    TypeError,
                    "has no tasks",
                ),
            )

            for name, agent_yaml, task_yaml, error_type, message in cases:
                with self.subTest(name):
                    agents.write_text(agent_yaml)
                    tasks.write_text(task_yaml)
                    with self.assertRaisesRegex(error_type, message):
                        Crew(agent_definition=agents, task_definition=tasks)

            agents.write_text(
                "manager:\n"
                "  manager: true\n"
                "  llm: test\n"
                "  output_model: ResearchOutput\n" + worker
            )
            tasks.write_text(task)
            with self.assertRaisesRegex(
                TypeError,
                "manager 'manager' with output ResearchOutput",
            ):
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

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agents = root / "agents.yaml"
            tasks = root / "tasks.yaml"
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
                    with self.assertRaisesRegex(
                        ValueError,
                        f"reserved delegation tools: '{tool.__name__}'",
                    ):
                        Crew(
                            agent_definition=agents,
                            task_definition=tasks,
                            tools=[tool],
                        )

    def test_supports_pydantic_ai_tools_capabilities_and_decorators(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agents = root / "agents.yaml"
            tasks = root / "tasks.yaml"
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
                def get_instructions(self) -> str:
                    return "Capability enabled."

            class UnusedInstructions(AbstractCapability[ToolContext]):
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
                output = anyio.run(crew.kickoff, None, ToolContext(offset=10))

            self.assertEqual(output, "Tools worked")
            agents.write_text(agents.read_text().replace("increment", "missing"))
            with self.assertRaisesRegex(
                ValueError,
                "Agent 'researcher' references undefined tools: 'missing'",
            ):
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
            with self.assertRaisesRegex(
                ValueError,
                "Agent 'researcher' references undefined capabilities: "
                "'MissingCapability'",
            ):
                Crew(
                    agent_definition=agents,
                    task_definition=tasks,
                    deps_type=ToolContext,
                    tools=[Tool(increment, name="increment"), double],
                    capabilities=[ExtraInstructions()],
                )

    def test_supports_mcp_capabilities_and_toolsets(self) -> None:
        def exercise(yaml_key: str) -> None:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                agents = root / "agents.yaml"
                tasks = root / "tasks.yaml"
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
                lifecycle: list[str] = []
                calls: list[int] = []

                @asynccontextmanager
                async def lifespan(_: FastMCP[None]) -> AsyncIterator[None]:
                    lifecycle.append("start")
                    try:
                        yield None
                    finally:
                        lifecycle.append("stop")

                server = FastMCP("math", lifespan=lifespan)

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
                self.assertEqual(lifecycle, ["start", "stop"])

                def fail(_messages, _info):
                    raise RuntimeError("model failed")

                with (
                    crew.agents["researcher"].override(model=FunctionModel(fail)),
                    self.assertRaisesRegex(RuntimeError, "model failed"),
                ):
                    anyio.run(crew.kickoff)

                self.assertEqual(lifecycle, ["start", "stop", "start", "stop"])

                agents.write_text(agents.read_text().replace("math", "missing"))
                with self.assertRaisesRegex(
                    ValueError,
                    f"Agent 'researcher' references undefined {yaml_key}: 'missing'",
                ):
                    build_crew()

        for yaml_key in ("capabilities", "toolsets"):
            with self.subTest(yaml_key=yaml_key):
                exercise(yaml_key)

    def test_uses_final_agent_output_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agents = root / "agents.yaml"
            tasks = root / "tasks.yaml"
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
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agents = root / "agents.yaml"
            tasks = root / "tasks.yaml"
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

            with self.assertRaisesRegex(
                TypeError,
                "Crew declares output ResearchOutput, but final task 'write' "
                "uses agent 'writer' with output str",
            ):
                Crew(
                    agent_definition=agents,
                    task_definition=tasks,
                    output_type=ResearchOutput,
                )

    def test_rejects_unknown_agent_output_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agents = root / "agents.yaml"
            tasks = root / "tasks.yaml"
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

            with self.assertRaisesRegex(
                ValueError,
                "Agent 'researcher' references undefined Pydantic model "
                "'MissingOutput'",
            ):
                Crew(
                    agent_definition=agents,
                    task_definition=tasks,
                    pydantic_models=[ResearchOutput],
                )

    def test_requires_a_valid_llm_for_each_agent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            agents = root / "agents.yaml"
            tasks = root / "tasks.yaml"
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
            with self.assertRaisesRegex(UserError, "Unknown model: not-a-model"):
                Crew(
                    agent_definition=agents,
                    task_definition=tasks,
                )

    def test_reports_each_missing_definition_file(self) -> None:
        with self.assertRaisesRegex(
            FileNotFoundError, "agent_definition file not found"
        ):
            Crew(
                agent_definition="missing-agents.yaml",
                task_definition="tasks.yaml",
            )

        with self.assertRaisesRegex(
            FileNotFoundError, "task_definition file not found"
        ):
            Crew(
                agent_definition="agents.yaml",
                task_definition="missing-tasks.yaml",
            )

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

            with self.assertRaisesRegex(ValueError, "sequential crew"):
                Crew(
                    agent_definition=agents,
                    task_definition=tasks,
                )

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

            with self.assertRaisesRegex(ValueError, "undefined agent 'writer'"):
                Crew(
                    agent_definition=agents,
                    task_definition=tasks,
                )


if __name__ == "__main__":
    unittest.main()
