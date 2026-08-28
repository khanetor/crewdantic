from __future__ import annotations

import inspect
from pathlib import Path
from typing import Annotated, ClassVar, TypeVar

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

NonEmptyString = Annotated[str, Field(min_length=1)]


class AgentDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: NonEmptyString
    goal: NonEmptyString
    backstory: NonEmptyString


class TaskDefinition(BaseModel):
    model_config = ConfigDict(extra="allow")

    description: NonEmptyString
    expected_output: NonEmptyString
    agent: NonEmptyString


Definition = TypeVar("Definition")
_AGENT_DEFINITIONS = TypeAdapter(dict[str, AgentDefinition])
_TASK_DEFINITIONS = TypeAdapter(dict[str, TaskDefinition])


def _load_yaml(path: Path, adapter: TypeAdapter[Definition]) -> Definition:
    with path.open(encoding="utf-8") as definition_file:
        return adapter.validate_python(yaml.safe_load(definition_file))


class Crew:
    agent_definition: ClassVar[str | Path | None] = None
    task_definition: ClassVar[str | Path | None] = None

    def __init__(self) -> None:
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
