"""Asynchronous Groq client wrapper used by the swarm manager."""

from __future__ import annotations

import json
import os
from typing import Any

import aiohttp
from pydantic import BaseModel, ConfigDict, Field


class GroqClientConfig(BaseModel):
    """Runtime settings for the Groq API client."""

    model_config = ConfigDict(extra="forbid")

    api_key: str | None = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    model_name: str = "llama3-8b-8192"
    base_url: str = "https://api.groq.com/openai/v1/chat/completions"
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class GroqAsyncClient:
    """Thin async client that requests JSON-only completions from Groq."""

    def __init__(
        self,
        config: GroqClientConfig | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.config = config or GroqClientConfig()
        self._session = session
        self._owns_session = session is None

    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Send a chat completion request and parse the returned JSON content."""
        if not self.config.api_key:
            raise RuntimeError("GROQ_API_KEY is required to call the Groq API.")

        session = await self._get_session()
        payload = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        async with session.post(
            self.config.base_url,
            json=payload,
            headers=headers,
        ) as response:
            response.raise_for_status()
            data = await response.json()

        content = data["choices"][0]["message"]["content"]
        return json.loads(content)

    async def close(self) -> None:
        """Close the internally-owned session if one was created."""
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
