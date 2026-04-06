"""Async orchestration for the LLM-driven retail trader swarm."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

from core.models import AgentDecision, MarketSnapshot, Order
from swarm.client import GroqAsyncClient
from swarm.models import LLMOrderResponse, RetailPersona
from swarm.personas import generate_personas
from swarm.prompts import build_system_prompt, build_user_prompt


class SwarmManager:
    """Coordinates concurrent LLM requests across many retail personas."""

    def __init__(
        self,
        client: GroqAsyncClient,
        personas: Iterable[RetailPersona] | None = None,
        agent_count: int = 50,
    ) -> None:
        self.client = client
        self.personas = list(personas) if personas is not None else generate_personas(agent_count)
        self.last_errors: list[str] = []

    async def collect_decisions(
        self,
        snapshot: MarketSnapshot,
        market_news: str,
    ) -> list[AgentDecision]:
        """Gather validated decisions for the whole swarm in parallel."""
        tasks = [
            self._request_decision(persona=persona, snapshot=snapshot, market_news=market_news)
            for persona in self.personas
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        decisions: list[AgentDecision] = []
        self.last_errors = []
        for result in results:
            if isinstance(result, Exception):
                self.last_errors.append(str(result))
                continue
            decisions.append(result)
        return decisions

    async def generate_orders(
        self,
        snapshot: MarketSnapshot,
        market_news: str,
    ) -> list[Order]:
        """Convert swarm decisions into normalized orders for the fast lane."""
        decisions = await self.collect_decisions(snapshot=snapshot, market_news=market_news)
        orders: list[Order] = []
        for decision in decisions:
            order = decision.to_order()
            if order is not None:
                orders.append(order)
        return orders

    async def _request_decision(
        self,
        persona: RetailPersona,
        snapshot: MarketSnapshot,
        market_news: str,
    ) -> AgentDecision:
        system_prompt = build_system_prompt(persona)
        user_prompt = build_user_prompt(snapshot, market_news)
        payload = await self.client.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
        response = LLMOrderResponse.model_validate(payload)
        return response.to_agent_decision(persona)
