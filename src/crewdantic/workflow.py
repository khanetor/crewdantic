import inspect
from collections.abc import Callable
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import (
    ClassVar,
    Generic,
    TypeVar,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

WorkflowInputT = TypeVar("WorkflowInputT")
WorkflowOutputT = TypeVar("WorkflowOutputT")
StepFunctionT = TypeVar("StepFunctionT", bound=Callable[..., object])
_STEP_MARKER = "__crewdantic_step__"
_WORKFLOW_INPUT = "workflow_input"
_FINAL_STEP = "final_step"


@dataclass(frozen=True)
class _StepDefinition:
    parameters: tuple[str, ...]
    output_type: object


def step(function: StepFunctionT) -> StepFunctionT:
    setattr(function, _STEP_MARKER, True)
    return function


class Workflow(Generic[WorkflowInputT, WorkflowOutputT]):
    _step_definitions: ClassVar[dict[str, _StepDefinition]]
    _step_order: ClassVar[tuple[str, ...]]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        try:
            input_type, output_type = next(
                get_args(base)
                for base in getattr(cls, "__orig_bases__", ())
                if get_origin(base) is Workflow
            )
        except StopIteration:
            raise TypeError("Workflow must declare input and output types") from None
        functions = {
            name: value
            for name, value in cls.__dict__.items()
            if getattr(value, _STEP_MARKER, False)
        }
        if _FINAL_STEP not in functions:
            raise TypeError("Workflow must declare a @step method named 'final_step'")
        if _WORKFLOW_INPUT in functions:
            raise TypeError(f"{_WORKFLOW_INPUT!r} is reserved for the workflow input")

        definitions: dict[str, _StepDefinition] = {}
        annotations: dict[str, dict[str, object]] = {}
        for name, function in functions.items():
            signature = inspect.signature(function)
            parameters = tuple(signature.parameters.values())
            if not parameters or parameters[0].name != "self":
                raise TypeError(f"Workflow step {name!r} must be an instance method")

            hints = get_type_hints(function)
            if "return" not in hints:
                raise TypeError(f"Workflow step {name!r} must declare a return type")
            for parameter in parameters[1:]:
                if parameter.kind not in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ):
                    raise TypeError(
                        f"Workflow step {name!r} parameter {parameter.name!r} "
                        "must accept a keyword argument"
                    )
                if parameter.name not in hints:
                    raise TypeError(
                        f"Workflow step {name!r} parameter {parameter.name!r} "
                        "must declare a type"
                    )

            definitions[name] = _StepDefinition(
                tuple(parameter.name for parameter in parameters[1:]),
                hints["return"],
            )
            annotations[name] = hints

        for name, definition in definitions.items():
            for parameter in definition.parameters:
                parameter_type = annotations[name][parameter]
                if parameter == _WORKFLOW_INPUT:
                    if parameter_type != input_type:
                        raise TypeError(
                            f"Workflow step {name!r} input type {parameter_type!r} "
                            f"does not match workflow input {input_type!r}"
                        )
                    continue
                if parameter == _FINAL_STEP:
                    raise ValueError(
                        f"Workflow step {name!r} cannot depend on final_step"
                    )
                if parameter not in definitions:
                    raise ValueError(
                        f"Workflow step {name!r} depends on undefined step {parameter!r}"
                    )
                dependency_type = definitions[parameter].output_type
                if parameter_type != dependency_type:
                    raise TypeError(
                        f"Workflow step {name!r} parameter {parameter!r} has type "
                        f"{parameter_type!r}, but step {parameter!r} returns "
                        f"{dependency_type!r}"
                    )

        final_output_type = definitions[_FINAL_STEP].output_type
        if final_output_type != output_type:
            raise TypeError(
                f"Workflow declares output {output_type!r}, but final_step returns "
                f"{final_output_type!r}"
            )

        dependencies = {
            name: set(definition.parameters) - {_WORKFLOW_INPUT}
            for name, definition in definitions.items()
        }
        try:
            cls._step_order = tuple(TopologicalSorter(dependencies).static_order())
        except CycleError as error:
            cycle = " -> ".join(map(str, error.args[1]))
            raise ValueError(f"Workflow contains a cycle: {cycle}") from error

        upstream_steps = {
            dependency
            for step_dependencies in dependencies.values()
            for dependency in step_dependencies
        }
        orphaned = definitions.keys() - upstream_steps - {_FINAL_STEP}
        if orphaned:
            raise ValueError(
                "Workflow steps do not lead to final_step: "
                + ", ".join(repr(name) for name in sorted(orphaned))
            )

        cls._step_definitions = definitions

    async def kickoff(self, workflow_input: WorkflowInputT) -> WorkflowOutputT:
        outputs: dict[str, object] = {}
        for name in self._step_order:
            arguments = {
                parameter: (
                    workflow_input
                    if parameter == _WORKFLOW_INPUT
                    else outputs[parameter]
                )
                for parameter in self._step_definitions[name].parameters
            }
            method = cast(Callable[..., object], getattr(self, name))
            output = method(**arguments)
            if inspect.isawaitable(output):
                output = await output
            outputs[name] = output

        return cast(WorkflowOutputT, outputs[_FINAL_STEP])
