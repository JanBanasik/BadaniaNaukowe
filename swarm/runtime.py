"""Runtime helpers for selecting and configuring swarm providers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

import numpy as np

from swarm.client import (
    AsyncLLMClient,
    GroqClientConfig,
    OllamaClientConfig,
    OpenAICompatibleClientConfig,
    create_llm_client,
)

ResolvedProvider = Literal["mock", "groq", "ollama", "lmstudio"]


class MockSwarmClient:
    """Deterministic fallback swarm client for local end-to-end testing."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)

    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        del system_prompt, user_prompt
        action = self.rng.choice(["buy", "sell", "hold"], p=[0.4, 0.4, 0.2]).item()
        if action == "hold":
            return {
                "action": "hold",
                "price_type": "market",
                "volume": 0.0,
                "limit_price": None,
                "confidence": 0.5,
                "rationale": "Mock client elected to wait.",
            }
        return {
            "action": action,
            "price_type": "market",
            "volume": float(self.rng.integers(1, 6)),
            "limit_price": None,
            "confidence": float(self.rng.uniform(0.4, 0.9)),
            "rationale": "Mock client generated a deterministic placeholder action.",
        }

    async def close(self) -> None:
        """No-op close for the mock client."""


@dataclass(slots=True)
class ResolvedSwarmClient:
    """Resolved swarm client plus metadata for logging."""

    provider: ResolvedProvider
    client: AsyncLLMClient
    model_name: str
    config_payload: dict[str, object]


def build_swarm_client(
    provider: ResolvedProvider,
    seed: int,
) -> ResolvedSwarmClient:
    """Instantiate one swarm client from environment-backed defaults."""
    if provider == "mock":
        return ResolvedSwarmClient(
            provider="mock",
            client=MockSwarmClient(seed=seed),
            model_name="mock-swarm",
            config_payload={"seed": seed},
        )
    if provider == "ollama":
        config = OllamaClientConfig(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api/chat"),
            model_name=os.getenv("OLLAMA_MODEL", "llama3:8b"),
            timeout_seconds=float(os.getenv("OLLAMA_TIMEOUT", "60")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        )
        return ResolvedSwarmClient(
            provider="ollama",
            client=create_llm_client("ollama", config),
            model_name=config.model_name,
            config_payload=config.model_dump(mode="json"),
        )
    if provider == "lmstudio":
        config = OpenAICompatibleClientConfig(
            base_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1/chat/completions"),
            model_name=os.getenv("LMSTUDIO_MODEL", "local-model"),
            api_key=os.getenv("OPENAI_COMPATIBLE_API_KEY"),
            timeout_seconds=float(os.getenv("LMSTUDIO_TIMEOUT", "60")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
        )
        return ResolvedSwarmClient(
            provider="lmstudio",
            client=create_llm_client("openai_compatible", config),
            model_name=config.model_name,
            config_payload=config.model_dump(mode="json"),
        )

    config = GroqClientConfig(
        base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions"),
        model_name=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
        api_key=os.getenv("GROQ_API_KEY"),
        timeout_seconds=float(os.getenv("GROQ_TIMEOUT", "60")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
    )
    return ResolvedSwarmClient(
        provider="groq",
        client=create_llm_client("groq", config),
        model_name=config.model_name,
        config_payload=config.model_dump(mode="json"),
    )
