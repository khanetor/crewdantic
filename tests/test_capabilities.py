import json
import os
import tempfile
import unittest
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import anyio
import httpx
from pydantic_ai import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel

from crewdantic import (
    ChoiceDecision,
    Crew,
    NoulDecision,
    ScoreDecision,
    SystemOne,
)

CHOICE = {
    "type": "choice",
    "choice": "mixed",
    "probabilities": {"positive": 0.2, "negative": 0.2, "mixed": 0.6},
    "confidence": 0.4,
}
SCORE = {
    "type": "score",
    "score": 1.5,
    "legend": {"0": "low", "1": "medium", "2": "high"},
    "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
    "confidence": 0.3,
}
NOUL = {"type": "noul", "noul": 0.75}


def fake_systemone(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.content)
    answers = {
        "choice": CHOICE,
        "score": SCORE,
        "noul": NOUL,
    }
    question = payload["questions"]["decision"]
    return httpx.Response(
        200,
        json={
            "model": payload.get("model", "fake"),
            "answers": {"decision": answers[question["type"]]},
            "usage": {"input_tokens": 3, "output_tokens": 0},
        },
    )


@contextmanager
def definition_paths() -> Generator[tuple[Path, Path]]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        yield root / "agents.yaml", root / "tasks.yaml"


class SystemOneCapabilityTests(unittest.TestCase):
    def capability(self, requests: list[httpx.Request]) -> SystemOne:
        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return fake_systemone(request)

        return SystemOne(
            base_url="https://systemone.test",
            model="jev-test",
            api_key="secret",
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        )

    def test_typed_decision_helpers(self) -> None:
        requests: list[httpx.Request] = []
        capability = self.capability(requests)

        async def exercise() -> None:
            noul = await capability.noul("It is sunny", "Is it sunny?")
            choice = await capability.choice(
                "Delivery was late but quality is good",
                "What is the sentiment?",
                {"positive": None, "mixed": "Both"},
            )
            score = await capability.score(
                "Delivery was late", "Rate satisfaction", ["low", "high"]
            )

            self.assertEqual(noul, NoulDecision(probability=0.75))
            self.assertIsInstance(choice, ChoiceDecision)
            self.assertEqual(choice.choice, "mixed")
            self.assertAlmostEqual(choice.confidence, 0.4)
            self.assertIsInstance(score, ScoreDecision)
            self.assertEqual(score.legend["2"], "high")
            self.assertAlmostEqual(score.score, 1.5)

        anyio.run(exercise)

        self.assertEqual(len(requests), 3)
        request = requests[0]
        self.assertEqual(str(request.url), "https://systemone.test/v1/systemone")
        self.assertEqual(request.headers["Authorization"], "Bearer secret")
        payload: dict[str, Any] = json.loads(request.content)
        self.assertEqual(payload["model"], "jev-test")
        self.assertEqual(payload["state"], "It is sunny")
        self.assertEqual(payload["questions"]["decision"]["type"], "noul")

    def test_reads_environment_and_requires_base_url(self) -> None:
        with patch.dict(
            os.environ,
            {
                "SYSTEMONE_BASE_URL": "https://env.test",
                "SYSTEMONE_MODEL": "env-model",
                "SYSTEMONE_API_KEY": "env-key",
            },
        ):
            capability = SystemOne()
            self.assertEqual(capability.base_url, "https://env.test")
            self.assertEqual(capability.model, "env-model")
            self.assertEqual(capability.api_key, "env-key")

        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            SystemOne()

    def test_crew_uses_capability_from_yaml(self) -> None:
        with definition_paths() as (agents, tasks):
            agents.write_text(
                "researcher:\n"
                "  role: Researcher\n"
                "  goal: Make decisions\n"
                "  backstory: Decisive\n"
                "  llm: test\n"
                "  capabilities:\n"
                "    - SystemOne\n"
            )
            tasks.write_text(
                "decide:\n"
                "  description: Decide about the report\n"
                "  expected_output: A decision\n"
                "  agent: researcher\n"
            )
            requests: list[httpx.Request] = []
            crew = Crew(
                agent_definition=agents,
                task_definition=tasks,
                capabilities=[self.capability(requests)],
            )

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
                        {"noul", "choice", "score"},
                    )
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                tool_name="noul",
                                args={
                                    "state": "The report has a reproduction.",
                                    "instructions": "Does it include steps?",
                                },
                                tool_call_id="noul",
                            )
                        ]
                    )
                self.assertIsInstance(tool_returns[0].content, NoulDecision)
                return ModelResponse(parts=[TextPart(content="Decided")])

            with crew.agents["researcher"].override(model=FunctionModel(respond)):
                output = anyio.run(crew.kickoff)

            self.assertEqual(output, "Decided")
            self.assertEqual(len(requests), 1)

            agents.write_text(agents.read_text().replace("SystemOne", "Missing"))
            with self.assertRaises(ValueError):
                Crew(
                    agent_definition=agents,
                    task_definition=tasks,
                    capabilities=[self.capability([])],
                )
