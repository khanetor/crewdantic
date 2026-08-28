from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, ClassVar, Generic, TypeVar, cast, get_args, get_origin

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic_ai import Agent

NonEmptyString = Annotated[str, Field(min_length=1)]


class AgentDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: NonEmptyString
    goal: NonEmptyString
    backstory: NonEmptyString
    llm: NonEmptyString
    output_model: NonEmptyString | None = None


class TaskDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    description: NonEmptyString
    expected_output: NonEmptyString
    agent: NonEmptyString


Definition = TypeVar("Definition")
CrewOutputT = TypeVar("CrewOutputT", bound=BaseModel | str)
_AGENT_DEFINITIONS = TypeAdapter(dict[str, AgentDefinition])
_TASK_DEFINITIONS = TypeAdapter(dict[str, TaskDefinition])


def _load_yaml(path: Path, adapter: TypeAdapter[Definition]) -> Definition:
    with path.open(encoding="utf-8") as definition_file:
        return adapter.validate_python(yaml.safe_load(definition_file))


class Crew(Generic[CrewOutputT]):
    agent_definition: ClassVar[str | Path | None] = None
    task_definition: ClassVar[str | Path | None] = None

    def __init__(
        self,
        agent_variables: Mapping[str, object] | None = None,
        pydantic_models: list[type[BaseModel]] | None = None,
    ) -> None:
        missing_definitions = [
            attribute
            for attribute in ("agent_definition", "task_definition")
            if getattr(type(self), attribute) is None
        ]
        if missing_definitions:
            raise TypeError(
                f"{type(self).__name__} must define {', '.join(missing_definitions)}"
            )

        self.agent_definitions = _load_yaml(
            self._definition_path("agent_definition"), _AGENT_DEFINITIONS
        )
        self.task_definitions = _load_yaml(
            self._definition_path("task_definition"), _TASK_DEFINITIONS
        )

        for task_name, task in self.task_definitions.items():
            if task.agent not in self.agent_definitions:
                raise ValueError(
                    f"Task {task_name!r} references undefined agent {task.agent!r}"
                )

        variables = agent_variables or {}
        output_models = {model.__name__: model for model in pydantic_models or []}
        for agent_name, definition in self.agent_definitions.items():
            if (
                definition.output_model is not None
                and definition.output_model not in output_models
            ):
                raise ValueError(
                    f"Agent {agent_name!r} references undefined Pydantic model "
                    f"{definition.output_model!r}"
                )
        declared_output_type = self._declared_output_type()
        if declared_output_type is not None:
            if not self.task_definitions:
                raise TypeError(
                    f"{type(self).__name__} declares crew output "
                    f"{declared_output_type.__name__} but has no tasks"
                )
            final_task_name = next(reversed(self.task_definitions))
            final_task = self.task_definitions[final_task_name]
            final_definition = self.agent_definitions[final_task.agent]
            final_output_type = (
                output_models[final_definition.output_model]
                if final_definition.output_model is not None
                else str
            )
            if final_output_type is not declared_output_type:
                raise TypeError(
                    f"{type(self).__name__} declares crew output "
                    f"{declared_output_type.__name__}, but final task "
                    f"{final_task_name!r} uses agent {final_task.agent!r} "
                    f"with output {final_output_type.__name__}"
                )

        self.agents = {
            name: Agent(
                model=definition.llm,
                name=name,
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
            )
            for name, definition in self.agent_definitions.items()
        }

    async def kickoff(self, inputs: Mapping[str, object] | None = None) -> CrewOutputT:
        values = inputs or {}
        previous_output: BaseModel | str | None = None

        for task in self.task_definitions.values():
            prompt = (
                f"{task.description.format_map(values)}\n\n"
                f"Expected output:\n{task.expected_output.format_map(values)}"
            )
            if previous_output is not None:
                prompt += f"\n\nPrevious task output:\n{previous_output}"

            result = await self.agents[task.agent].run(prompt)
            previous_output = result.output

        return cast(CrewOutputT, previous_output if previous_output is not None else "")

    @classmethod
    def _declared_output_type(
        cls,
    ) -> type[BaseModel | str] | None:
        for base in getattr(cls, "__orig_bases__", ()):
            if get_origin(base) is Crew:
                output_type = get_args(base)[0]
                return output_type if isinstance(output_type, type) else None
        return None

    @classmethod
    def _definition_path(cls, attribute: str) -> Path:
        definition = getattr(cls, attribute)
        assert definition is not None

        path = Path(definition)
        if not path.is_absolute():
            path = Path(inspect.getfile(cls)).resolve().parent / path
        if not path.is_file():
            raise FileNotFoundError(f"{attribute} file not found: {path}")
        return path
