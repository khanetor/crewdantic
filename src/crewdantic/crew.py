import inspect
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Annotated, Any, Generic, cast

import yaml  # type: ignore[import-untyped]
from anyio import Path as AsyncPath
from pydantic import BaseModel, Field, TypeAdapter
from pydantic_ai import Agent, AgentCapability, Tool
from pydantic_ai.tools import AgentDepsT, ToolFuncEither
from pydantic_ai.toolsets import AgentToolset
from typing_extensions import TypeVar

NonEmptyString = Annotated[str, Field(min_length=1)]


class AgentDefinition(BaseModel):
    role: NonEmptyString
    goal: NonEmptyString
    backstory: NonEmptyString
    llm: NonEmptyString
    output_model: Annotated[NonEmptyString | None, Field(default=None)]
    tools: Annotated[list[NonEmptyString], Field(default_factory=list)]
    toolsets: Annotated[list[NonEmptyString], Field(default_factory=list)]
    capabilities: Annotated[list[NonEmptyString], Field(default_factory=list)]


class TaskDefinition(BaseModel):
    description: NonEmptyString
    expected_output: NonEmptyString
    agent: NonEmptyString


Definition = TypeVar("Definition")
CrewOutputT = TypeVar("CrewOutputT", bound=BaseModel | str, default=str)
_AGENT_DEFINITIONS = TypeAdapter(dict[str, AgentDefinition])
_TASK_DEFINITIONS = TypeAdapter(dict[str, TaskDefinition])


def _load_yaml(path: Path, adapter: TypeAdapter[Definition]) -> Definition:
    with path.open(encoding="utf-8") as definition_file:
        return adapter.validate_python(yaml.safe_load(definition_file))


class Crew(Generic[CrewOutputT, AgentDepsT]):
    def __init__(
        self,
        *,
        agent_definition: str | Path,
        task_definition: str | Path,
        output_type: type[CrewOutputT] | None = None,
        deps_type: type[AgentDepsT] | None = None,
        agent_variables: Mapping[str, object] | None = None,
        pydantic_models: Sequence[type[BaseModel]] = (),
        tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
        toolsets: Mapping[str, AgentToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] = (),
    ) -> None:
        output_type = cast(type[CrewOutputT], output_type or str)
        deps_type = cast(type[AgentDepsT], deps_type or object)
        frame = inspect.currentframe()
        assert frame is not None and frame.f_back is not None
        definition_root = Path(frame.f_back.f_code.co_filename).resolve().parent
        del frame
        self.agent_definitions = _load_yaml(
            self._definition_path(
                "agent_definition", agent_definition, definition_root
            ),
            _AGENT_DEFINITIONS,
        )
        self.task_definitions = _load_yaml(
            self._definition_path("task_definition", task_definition, definition_root),
            _TASK_DEFINITIONS,
        )

        for task_name, task in self.task_definitions.items():
            if task.agent not in self.agent_definitions:
                raise ValueError(
                    f"Task {task_name!r} references undefined agent {task.agent!r}"
                )

        registered_tools = {
            tool.name if isinstance(tool, Tool) else cast(Any, tool).__name__: tool
            for tool in tools
        }
        if len(registered_tools) != len(tools):
            raise ValueError("Tool names must be unique")
        registered_toolsets = toolsets or {}

        registered_capabilities = {
            (
                getattr(capability, "id", None)
                or getattr(capability, "__name__", type(capability).__name__)
            ): capability
            for capability in capabilities
        }
        if len(registered_capabilities) != len(capabilities):
            raise ValueError("Capability names must be unique")

        for agent_name, definition in self.agent_definitions.items():
            for kind, assigned, registered in (
                ("tools", definition.tools, registered_tools),
                ("toolsets", definition.toolsets, registered_toolsets),
                ("capabilities", definition.capabilities, registered_capabilities),
            ):
                undefined = set(assigned) - registered.keys()
                if undefined:
                    raise ValueError(
                        f"Agent {agent_name!r} references undefined {kind}: "
                        + ", ".join(repr(name) for name in sorted(undefined))
                    )

        variables = agent_variables or {}
        output_models = {model.__name__: model for model in pydantic_models}
        if output_type is not str:
            output_models[output_type.__name__] = cast(type[BaseModel], output_type)
        for agent_name, definition in self.agent_definitions.items():
            if (
                definition.output_model is not None
                and definition.output_model not in output_models
            ):
                raise ValueError(
                    f"Agent {agent_name!r} references undefined Pydantic model "
                    f"{definition.output_model!r}"
                )
        if not self.task_definitions:
            raise TypeError(
                f"Crew declares output {output_type.__name__} but has no tasks"
            )
        final_task_name = next(reversed(self.task_definitions))
        final_task = self.task_definitions[final_task_name]
        final_definition = self.agent_definitions[final_task.agent]
        final_output_type = (
            output_models[final_definition.output_model]
            if final_definition.output_model is not None
            else str
        )
        if final_output_type is not output_type:
            raise TypeError(
                f"Crew declares output {output_type.__name__}, but final task "
                f"{final_task_name!r} uses agent {final_task.agent!r} "
                f"with output {final_output_type.__name__}"
            )

        self.agents: dict[str, Agent[AgentDepsT, BaseModel | str]] = {
            name: Agent(
                model=definition.llm,
                name=name,
                deps_type=deps_type,
                output_type=(
                    output_models[definition.output_model]
                    if definition.output_model is not None
                    else str
                ),
                instructions=(
                    f"Role: {definition.role.format_map(variables)}\n"
                    f"Goal: {definition.goal.format_map(variables)}\n"
                    f"Backstory: {definition.backstory.format_map(variables)}"
                ),
                tools=[registered_tools[tool] for tool in definition.tools],
                toolsets=[
                    registered_toolsets[toolset] for toolset in definition.toolsets
                ],
                capabilities=[
                    registered_capabilities[capability]
                    for capability in definition.capabilities
                ],
            )
            for name, definition in self.agent_definitions.items()
        }

    def to_mermaid_chart(self) -> str:
        def label(value: str) -> str:
            return escape(" ".join(value.split())).replace("`", "#96;")

        nodes = [
            f'  task_{index}["{label(name)}<br/>agent: {label(task.agent)}"]'
            for index, (name, task) in enumerate(self.task_definitions.items())
        ]
        edges = [
            f"  task_{index} --> task_{index + 1}" for index in range(len(nodes) - 1)
        ]
        return "\n".join(("flowchart LR", *nodes, *edges))

    async def render(self, filename: str | Path) -> None:
        diagram = self.to_mermaid_chart()
        output = Path(filename)
        match output.suffix.casefold():
            case ".mermaid":
                content = diagram
            case ".html":
                content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Crew visualization</title>
  <style>
    body {{ margin: 0; font-family: system-ui, sans-serif; }}
    main {{ min-height: 100vh; display: grid; place-items: center; }}
    .mermaid {{ max-width: 100%; overflow: auto; }}
  </style>
</head>
<body>
  <main>
    <pre class="mermaid" aria-label="Sequential crew task graph">{escape(diagram)}</pre>
  </main>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs";
  </script>
</body>
</html>
"""
            case suffix:
                raise ValueError(
                    f"Visualization output must end in .mermaid or .html, not {suffix!r}"
                )

        await AsyncPath(output).write_text(content, encoding="utf-8")

    async def kickoff(
        self,
        inputs: Mapping[str, object] | None = None,
        context: AgentDepsT | None = None,
    ) -> CrewOutputT:
        values = inputs or {}
        previous_output: BaseModel | str | None = None

        for task in self.task_definitions.values():
            prompt = (
                f"{task.description.format_map(values)}\n\n"
                f"Expected output:\n{task.expected_output.format_map(values)}"
            )
            if previous_output is not None:
                prompt += f"\n\nPrevious task output:\n{previous_output}"

            result = await self.agents[task.agent].run(
                prompt, deps=cast(AgentDepsT, context)
            )
            previous_output = result.output

        return cast(CrewOutputT, previous_output if previous_output is not None else "")

    @staticmethod
    def _definition_path(attribute: str, definition: str | Path, root: Path) -> Path:
        path = Path(definition)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise FileNotFoundError(f"{attribute} file not found: {path}")
        return path
