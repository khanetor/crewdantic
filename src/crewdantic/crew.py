import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Generic, cast

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, TypeAdapter
from pydantic_ai import Agent, AgentCapability, Tool
from pydantic_ai.tools import AgentDepsT, ToolFuncEither
from typing_extensions import TypeVar

NonEmptyString = Annotated[str, Field(min_length=1)]


class AgentDefinition(BaseModel):
    role: NonEmptyString
    goal: NonEmptyString
    backstory: NonEmptyString
    llm: NonEmptyString
    output_model: Annotated[NonEmptyString | None, Field(default=None)]
    tools: Annotated[list[NonEmptyString], Field(default_factory=list)]
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
                capabilities=[
                    registered_capabilities[capability]
                    for capability in definition.capabilities
                ],
            )
            for name, definition in self.agent_definitions.items()
        }

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
