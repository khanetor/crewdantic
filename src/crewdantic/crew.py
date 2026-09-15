import inspect
import re
from collections.abc import Mapping, Sequence
from html import escape
from pathlib import Path
from typing import Annotated, Any, Generic, cast

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from pydantic_ai import (
    Agent,
    AgentCapability,
    RunContext,
    RunUsage,
    Tool,
    UsageLimitExceeded,
    UsageLimits,
    UserContent,
)
from pydantic_ai.tools import AgentDepsT, ToolFuncEither
from pydantic_ai.toolsets import AgentToolset, FunctionToolset
from typing_extensions import TypeVar

from ._visualization import write_visualization

NonEmptyString = Annotated[str, Field(min_length=1)]


class AgentDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    manager: Annotated[bool, Field(default=False)]
    role: NonEmptyString
    goal: NonEmptyString
    backstory: NonEmptyString
    llm: NonEmptyString
    output_model: Annotated[NonEmptyString | None, Field(default=None)]
    tools: Annotated[list[NonEmptyString], Field(default_factory=list)]
    toolsets: Annotated[list[NonEmptyString], Field(default_factory=list)]
    capabilities: Annotated[list[NonEmptyString], Field(default_factory=list)]

    @model_validator(mode="before")
    @classmethod
    def add_manager_defaults(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and value.get("manager") is True:
            return {
                "role": "Crew Manager",
                "goal": "Coordinate workers and produce the requested output.",
                "backstory": (
                    "You delegate work to specialists, evaluate their responses, "
                    "and synthesize the final result."
                ),
                **value,
            }
        return value


class TaskDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: NonEmptyString
    expected_output: NonEmptyString
    agent: Annotated[NonEmptyString | None, Field(default=None)]
    context: Annotated[list[NonEmptyString] | None, Field(default=None)]


_Definition = TypeVar("_Definition")
CrewOutputT = TypeVar("CrewOutputT", bound=BaseModel | str, default=str)
_DelegationDepsT = TypeVar("_DelegationDepsT")
_AGENT_DEFINITIONS = TypeAdapter(dict[str, AgentDefinition])
_TASK_DEFINITIONS = TypeAdapter(dict[str, TaskDefinition])


def _definition_path(attribute: str, definition: str | Path, root: Path) -> Path:
    path = Path(definition)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"{attribute} file not found: {path}")
    return path


def _load_yaml(path: Path, adapter: TypeAdapter[_Definition]) -> _Definition:
    with path.open(encoding="utf-8") as definition_file:
        return adapter.validate_python(yaml.safe_load(definition_file))


def _safe_format(
    template: str,
    variables: Mapping[str, object],
    file_labels: Mapping[str, UserContent] | None = None,
) -> str:
    labels = set(file_labels or ())
    placeholders = re.findall(r"\{(\w+)\}", template)
    unknown = [
        name for name in placeholders if name not in variables and name not in labels
    ]
    if unknown:
        raise ValueError(f"Undefined placeholder {unknown[0]!r}")
    formatted = template
    for name in labels:
        formatted = formatted.replace(f"{{{name}}}", f"\0{name}\1")
    try:
        formatted = formatted.format_map(variables)
    except KeyError as error:
        raise ValueError(f"Undefined placeholder {error.args[0]!r}") from error
    return formatted.replace("\0", "{").replace("\1", "}")


def _inject_files(
    prompt: str, files: Mapping[str, UserContent], task_name: str
) -> str | list[str | UserContent]:
    """Splice {label} file parts into the prompt; unreferenced labels error."""
    if not files:
        return prompt
    parts: list[str | UserContent] = []
    used: set[str] = set()
    for segment in re.split(r"(\{\w+\})", prompt):
        name = segment[1:-1] if segment.startswith("{") else ""
        if name in files:
            parts.append(files[name])
            used.add(name)
        else:
            parts.append(segment)
    unused = files.keys() - used
    if unused:
        raise ValueError(
            f"Task {task_name!r} does not reference file input(s): "
            + ", ".join(repr(name) for name in sorted(unused))
        )
    return parts if any(not isinstance(part, str) for part in parts) else prompt


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
            _definition_path("agent_definition", agent_definition, definition_root),
            _AGENT_DEFINITIONS,
        )
        self.task_definitions = _load_yaml(
            _definition_path("task_definition", task_definition, definition_root),
            _TASK_DEFINITIONS,
        )

        self.manager_name = self._find_manager_name()
        self._validate_task_assignments()

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

        if self.manager_name is not None:
            conflicting_tools = {"ask_question", "delegate_work"}.intersection(
                self.agent_definitions[self.manager_name].tools
            )
            if conflicting_tools:
                raise ValueError(
                    f"Manager {self.manager_name!r} cannot define reserved "
                    "delegation tools: "
                    + ", ".join(repr(name) for name in sorted(conflicting_tools))
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
        self._validate_output_type(output_type, output_models)
        self._output_type = output_type

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
                instructions=_safe_format(
                    f"Role: {definition.role}\n"
                    f"Goal: {definition.goal}\n"
                    f"Backstory: {definition.backstory}",
                    variables,
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

    async def kickoff(
        self,
        context: AgentDepsT | None = None,
        usage_limits: UsageLimits | None = None,
        *,
        inputs: Mapping[str, object] | None = None,
        file_inputs: Mapping[str, UserContent] | None = None,
    ) -> CrewOutputT:
        values = inputs or {}
        files = file_inputs or {}
        usage = RunUsage()
        limits = usage_limits or UsageLimits()
        task_outputs: dict[str, BaseModel | str] = {}

        for index, (task_name, task) in enumerate(self.task_definitions.items()):
            try:
                description = _safe_format(task.description, values, files)
                expected_output = _safe_format(task.expected_output, values, files)
            except ValueError as error:
                raise ValueError(f"Task {task_name!r}: {error}") from error
            prompt = f"{description}\n\nExpected output:\n{expected_output}"
            context_outputs = (
                list(task_outputs.values())
                if "context" not in task.model_fields_set
                else [task_outputs[name] for name in task.context or []]
            )
            if context_outputs:
                prompt += "\n\nPrevious task outputs:\n" + "\n\n".join(
                    str(output) for output in context_outputs
                )
            prompt_parts: str | list[str | UserContent] = _inject_files(
                prompt, files, task_name
            )

            if self.manager_name is None:
                assert task.agent is not None
                result = await self.agents[task.agent].run(
                    prompt_parts,
                    deps=cast(AgentDepsT, context),
                    usage=usage,
                    usage_limits=limits,
                )
            else:
                coworkers = {
                    name: agent
                    for name, agent in self.agents.items()
                    if name != self.manager_name
                    and (task.agent is None or name == task.agent)
                }
                result = await self.agents[self.manager_name].run(
                    prompt_parts,
                    output_type=(
                        self._output_type
                        if index == len(self.task_definitions) - 1
                        else str
                    ),
                    deps=cast(AgentDepsT, context),
                    usage=usage,
                    usage_limits=limits,
                    toolsets=[self._delegation_toolset(coworkers, files)],
                )
            task_outputs[task_name] = result.output

        return cast(CrewOutputT, next(reversed(task_outputs.values())))

    def to_mermaid_chart(self) -> str:
        def label(value: str) -> str:
            return escape(" ".join(value.split())).replace("`", "#96;")

        nodes = []
        for index, (name, task) in enumerate(self.task_definitions.items()):
            if self.manager_name is None:
                assignment = f"agent: {label(task.agent or '')}"
            else:
                assignment = (
                    f"manager: {label(self.manager_name)}<br/>"
                    f"worker: {label(task.agent or 'any')}"
                )
            nodes.append(f'  task_{index}["{label(name)}<br/>{assignment}"]')
        edges = [
            f"  task_{index} --> task_{index + 1}" for index in range(len(nodes) - 1)
        ]
        return "\n".join(("flowchart LR", *nodes, *edges))

    def render(self, filename: str | Path) -> None:
        write_visualization(self.to_mermaid_chart(), filename)

    def _find_manager_name(self) -> str | None:
        manager_names = [
            name
            for name, definition in self.agent_definitions.items()
            if definition.manager
        ]
        if len(manager_names) > 1:
            raise ValueError(
                "Crew defines multiple manager agents: "
                + ", ".join(repr(name) for name in manager_names)
            )
        manager_name = manager_names[0] if manager_names else None
        if manager_name is not None and len(self.agent_definitions) == 1:
            raise ValueError("A hierarchical crew requires at least one worker agent")
        return manager_name

    def _validate_task_assignments(self) -> None:
        previous_tasks: set[str] = set()
        for task_name, task in self.task_definitions.items():
            for context_task in task.context or []:
                if context_task not in previous_tasks:
                    raise ValueError(
                        f"Task {task_name!r} context references non-previous task "
                        f"{context_task!r}"
                    )
            if task.agent is None:
                if self.manager_name is None:
                    raise ValueError(
                        f"Task {task_name!r} must declare an agent in a sequential crew"
                    )
            elif task.agent not in self.agent_definitions:
                raise ValueError(
                    f"Task {task_name!r} references undefined agent {task.agent!r}"
                )
            elif task.agent == self.manager_name:
                raise ValueError(
                    f"Task {task_name!r} cannot assign manager {task.agent!r} as a worker"
                )
            previous_tasks.add(task_name)

    def _validate_output_type(
        self,
        output_type: type[CrewOutputT],
        output_models: Mapping[str, type[BaseModel]],
    ) -> None:
        if not self.task_definitions:
            raise TypeError(
                f"Crew declares output {output_type.__name__} but has no tasks"
            )
        final_task_name = next(reversed(self.task_definitions))
        final_agent_name = (
            self.manager_name or self.task_definitions[final_task_name].agent
        )
        assert final_agent_name is not None
        final_definition = self.agent_definitions[final_agent_name]
        final_output_type = (
            output_models[final_definition.output_model]
            if final_definition.output_model is not None
            else str
        )
        if final_output_type is output_type:
            return
        if self.manager_name is not None:
            source = f"manager {self.manager_name!r}"
        else:
            source = f"final task {final_task_name!r} uses agent {final_agent_name!r}"
        raise TypeError(
            f"Crew declares output {output_type.__name__}, but {source} "
            f"with output {final_output_type.__name__}"
        )

    @staticmethod
    def _delegation_toolset(
        coworkers: Mapping[str, Agent[_DelegationDepsT, BaseModel | str]],
        file_parts: Mapping[str, UserContent],
    ) -> FunctionToolset[_DelegationDepsT]:
        available = ", ".join(repr(name) for name in coworkers)

        async def run_coworker(
            ctx: RunContext[_DelegationDepsT],
            coworker: str,
            request: str,
            context: str,
        ) -> BaseModel | str:
            agent = coworkers.get(coworker)
            if agent is None:
                return (
                    f"Unknown coworker {coworker!r}. Available coworkers: {available}"
                )
            prompt = f"{request}\n\nContext:\n{context}"
            for name in file_parts:
                prompt += f"\n\n{{{name}}}"
            if file_parts:
                prompt = _inject_files(
                    prompt, file_parts, f"delegation to {coworker!r}"
                )
            try:
                result = await agent.run(
                    prompt,
                    deps=ctx.deps,
                    usage=ctx.usage,
                    usage_limits=ctx.usage_limits,
                )
            except UsageLimitExceeded:
                raise
            except Exception as error:  # noqa: BLE001
                return f"Coworker {coworker!r} failed: {error}"
            return result.output

        async def delegate_work(
            ctx: RunContext[_DelegationDepsT],
            task: str,
            context: str,
            coworker: str,
        ) -> BaseModel | str:
            """Delegate a task to a coworker."""
            return await run_coworker(ctx, coworker, task, context)

        async def ask_question(
            ctx: RunContext[_DelegationDepsT],
            question: str,
            context: str,
            coworker: str,
        ) -> BaseModel | str:
            """Ask a coworker a question."""
            return await run_coworker(ctx, coworker, question, context)

        return FunctionToolset(
            [delegate_work, ask_question],
            instructions=f"Available coworkers: {available}",
        )
