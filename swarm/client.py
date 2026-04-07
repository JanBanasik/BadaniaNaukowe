"""Provider-agnostic asynchronous LLM clients used by the swarm manager."""

from __future__ import annotations

import json
import os
from typing import Any, Literal, Protocol

import aiohttp
from pydantic import BaseModel, ConfigDict, Field

ProviderName = Literal["groq", "ollama", "openai_compatible"]


class AsyncLLMClient(Protocol):
    """Small contract implemented by all swarm LLM backends."""

    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Return a JSON payload for one prompted persona."""

    async def close(self) -> None:
        """Release any network resources held by the client."""


class BaseClientConfig(BaseModel):
    """Common runtime settings shared by all HTTP-backed clients."""

    model_config = ConfigDict(extra="forbid")

    model_name: str
    base_url: str
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)


class GroqClientConfig(BaseClientConfig):
    """Runtime settings for the Groq API client."""

    api_key: str | None = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    base_url: str = "https://api.groq.com/openai/v1/chat/completions"
    model_name: str = "llama3-8b-8192"


class OpenAICompatibleClientConfig(BaseClientConfig):
    """Runtime settings for a local OpenAI-compatible endpoint."""

    api_key: str | None = Field(default_factory=lambda: os.getenv("OPENAI_COMPATIBLE_API_KEY"))
    base_url: str = "http://localhost:1234/v1/chat/completions"
    model_name: str = "local-llama3"


class OllamaClientConfig(BaseClientConfig):
    """Runtime settings for an Ollama backend."""

    base_url: str = "http://localhost:11434/api/chat"
    model_name: str = "llama3:8b"


class _BaseHTTPAsyncClient:
    """Shared session management for HTTP-backed LLM clients."""

    def __init__(
        self,
        timeout_seconds: float,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self._session = session
        self._owns_session = session is None
        self._timeout_seconds = timeout_seconds

    async def close(self) -> None:
        """Close the internally-owned session if one was created."""
        if self._owns_session and self._session is not None:
            await self._session.close()
            self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def _post_json(
        self,
        *,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        session = await self._get_session()
        async with session.post(url, json=payload, headers=headers) as response:
            response.raise_for_status()
            return await response.json()


class GroqAsyncClient(_BaseHTTPAsyncClient):
    """Thin async client that requests JSON-only completions from Groq."""

    def __init__(
        self,
        config: GroqClientConfig | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.config = config or GroqClientConfig()
        super().__init__(timeout_seconds=self.config.timeout_seconds, session=session)

    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Send a chat completion request and parse the returned JSON content."""
        if not self.config.api_key:
            raise RuntimeError("GROQ_API_KEY is required to call the Groq API.")

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
        data = await self._post_json(url=self.config.base_url, payload=payload, headers=headers)
        return _parse_json_content(data["choices"][0]["message"]["content"])


class OpenAICompatibleAsyncClient(_BaseHTTPAsyncClient):
    """Client for local servers exposing the OpenAI chat-completions API."""

    def __init__(
        self,
        config: OpenAICompatibleClientConfig | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.config = config or OpenAICompatibleClientConfig()
        super().__init__(timeout_seconds=self.config.timeout_seconds, session=session)

    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        data = await self._post_json(url=self.config.base_url, payload=payload, headers=headers)
        return _parse_json_content(data["choices"][0]["message"]["content"])


class OllamaAsyncClient(_BaseHTTPAsyncClient):
    """Client for Ollama's local chat endpoint."""

    def __init__(
        self,
        config: OllamaClientConfig | None = None,
        session: aiohttp.ClientSession | None = None,
    ) -> None:
        self.config = config or OllamaClientConfig()
        super().__init__(timeout_seconds=self.config.timeout_seconds, session=session)

    async def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        payload = {
            "model": self.config.model_name,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.config.temperature},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = await self._post_json(url=self.config.base_url, payload=payload, headers=None)
        message = data.get("message", {})
        if "content" in message:
            return _parse_json_content(message["content"])
        if "response" in data:
            return _parse_json_content(data["response"])
        raise ValueError("Ollama response did not contain a JSON content field.")


def create_llm_client(
    provider: ProviderName,
    config: GroqClientConfig | OpenAICompatibleClientConfig | OllamaClientConfig | None = None,
    session: aiohttp.ClientSession | None = None,
) -> AsyncLLMClient:
    """Create an async LLM client for the selected provider."""
    if provider == "groq":
        if config is not None and not isinstance(config, GroqClientConfig):
            raise TypeError("groq provider requires GroqClientConfig.")
        return GroqAsyncClient(config=config, session=session)
    if provider == "openai_compatible":
        if config is not None and not isinstance(config, OpenAICompatibleClientConfig):
            raise TypeError("openai_compatible provider requires OpenAICompatibleClientConfig.")
        return OpenAICompatibleAsyncClient(config=config, session=session)
    if provider == "ollama":
        if config is not None and not isinstance(config, OllamaClientConfig):
            raise TypeError("ollama provider requires OllamaClientConfig.")
        return OllamaAsyncClient(config=config, session=session)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def _parse_json_content(content: str) -> dict[str, Any]:
    """Parse JSON content returned by the remote model."""
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM response was not valid JSON: {content}") from exc
