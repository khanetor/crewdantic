"""Reusable capabilities for Crew agents."""

import os
from dataclasses import dataclass
from typing import Annotated, Any

import httpx
from pydantic import BaseModel, Field
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import FunctionToolset
from typing_extensions import override

Probability = Annotated[float, Field(ge=0, le=1)]


class NoulDecision(BaseModel):
    """Probability that a statement is true."""

    probability: Probability


class ChoiceDecision(BaseModel):
    """Selected option and the calibrated probability of every option."""

    choice: str
    probabilities: dict[str, float]
    confidence: Probability


class ScoreDecision(BaseModel):
    """Probability-weighted position on an ordered scale."""

    score: float
    legend: dict[str, Any]
    probabilities: dict[str, float]
    confidence: Probability


@dataclass(init=False)
class SystemOne(AbstractCapability[AgentDepsT]):
    """Capability exposing System One decision tools to an agent.

    The agent receives `noul`, `choice`, and `score` tools backed by a
    `/v1/systemone` endpoint. Values fall back to `SYSTEMONE_BASE_URL`,
    `SYSTEMONE_MODEL`, and `SYSTEMONE_API_KEY`; an explicit `client` replaces
    the per-call ``httpx.AsyncClient`` (useful for tests).
    """

    base_url: str
    model: str | None
    api_key: str | None
    client: httpx.AsyncClient | None

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        id: str | None = None,
        description: str | None = None,
        defer_loading: bool = False,
    ) -> None:
        base_url = base_url or os.environ.get("SYSTEMONE_BASE_URL")
        if base_url is None or base_url == "":
            raise ValueError("SystemOne requires a base_url or SYSTEMONE_BASE_URL")
        self.base_url = base_url
        self.model = model or os.environ.get("SYSTEMONE_MODEL")
        self.api_key = api_key or os.environ.get("SYSTEMONE_API_KEY")
        self.client = client
        self.id = id
        self.description = description
        self.defer_loading = defer_loading

    async def noul(
        self,
        state: str,
        instructions: str,
        criteria: dict[str, str] | None = None,
    ) -> NoulDecision:
        """Return the probability that a statement about `state` is true."""
        question: dict[str, Any] = {"type": "noul", "instructions": instructions}
        if criteria is not None:
            question["criteria"] = criteria
        answer = await self._ask(state, question)
        return NoulDecision(probability=answer["noul"])

    async def choice(
        self,
        state: str,
        instructions: str,
        criteria: dict[str, str | None],
    ) -> ChoiceDecision:
        """Choose one option about `state` from `criteria` (option -> description)."""
        answer = await self._ask(
            state,
            {"type": "choice", "instructions": instructions, "criteria": criteria},
        )
        return ChoiceDecision(
            choice=answer["choice"],
            probabilities=answer["probabilities"],
            confidence=answer["confidence"],
        )

    async def score(
        self,
        state: str,
        instructions: str,
        criteria: list[str],
    ) -> ScoreDecision:
        """Rate `state` across ordered `criteria` levels, lowest to highest."""
        answer = await self._ask(
            state, {"type": "score", "instructions": instructions, "criteria": criteria}
        )
        return ScoreDecision(
            score=answer["score"],
            legend=answer["legend"],
            probabilities=answer["probabilities"],
            confidence=answer["confidence"],
        )

    @override
    def get_toolset(self) -> FunctionToolset[AgentDepsT]:
        return FunctionToolset([self.noul, self.choice, self.score])

    async def _ask(self, state: str, question: dict[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {"state": state, "questions": {"decision": question}}
        if self.model is not None:
            payload["model"] = self.model
        headers = (
            {"Authorization": f"Bearer {self.api_key}"}
            if self.api_key is not None
            else {}
        )
        url = f"{self.base_url.rstrip('/')}/v1/systemone"
        if self.client is not None:
            response = await self.client.post(url, json=payload, headers=headers)
        else:
            # shortcut: a fresh client per call; pass a shared client to reuse connections
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        answer: dict[str, Any] = response.json()["answers"]["decision"]
        return answer
