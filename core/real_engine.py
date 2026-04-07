"""Adapter layer for plugging a concrete matching engine into the simulator."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from core.models import MarketSnapshot, Order


class RealEngineAdapterConfig(BaseModel):
    """Configuration for dynamically instantiating a real matching engine backend."""

    model_config = ConfigDict(extra="forbid")

    factory_path: str
    factory_kwargs: dict[str, Any] = Field(default_factory=dict)
    reset_method: str = "reset"
    advance_method: str = "advance"
    submit_order_method: str = "submit_order"
    snapshot_method: str = "get_snapshot"
    submit_as_dict: bool = False


class RealMatchingEngineAdapter:
    """Thin adapter around a concrete engine implementation."""

    def __init__(
        self,
        backend: Any | None = None,
        config: RealEngineAdapterConfig | None = None,
    ) -> None:
        if backend is None and config is None:
            raise ValueError("Either backend or config must be provided.")

        self.config = config
        self.backend = backend if backend is not None else self._instantiate_backend(config)

    def reset(self) -> None:
        """Reset the wrapped matching engine."""
        getattr(self.backend, self._config().reset_method)()

    def advance(self, ticks: int = 1) -> None:
        """Advance the wrapped matching engine."""
        getattr(self.backend, self._config().advance_method)(ticks)

    def submit_order(self, order: Order) -> None:
        """Submit one order to the wrapped engine."""
        payload: Order | dict[str, Any]
        if self._config().submit_as_dict:
            payload = order.model_dump(mode="json")
        else:
            payload = order
        getattr(self.backend, self._config().submit_order_method)(payload)

    def get_snapshot(self, depth_levels: int = 5) -> MarketSnapshot | Mapping[str, Any]:
        """Fetch a snapshot from the wrapped engine."""
        snapshot = getattr(self.backend, self._config().snapshot_method)(depth_levels)
        if isinstance(snapshot, MarketSnapshot):
            return snapshot
        if isinstance(snapshot, Mapping):
            return snapshot
        if hasattr(snapshot, "model_dump"):
            return snapshot.model_dump(mode="python")
        if hasattr(snapshot, "__dict__"):
            return vars(snapshot)
        raise TypeError(f"Unsupported snapshot payload type: {type(snapshot)!r}")

    def _config(self) -> RealEngineAdapterConfig:
        if self.config is None:
            return RealEngineAdapterConfig(factory_path="unused")
        return self.config

    @staticmethod
    def _instantiate_backend(config: RealEngineAdapterConfig | None) -> Any:
        if config is None:
            raise ValueError("config is required when backend is not provided.")
        factory = load_object_from_path(config.factory_path)
        return factory(**config.factory_kwargs)


def load_object_from_path(path: str) -> Any:
    """Load a Python object from a `module:attribute` import path."""
    if ":" not in path:
        raise ValueError("factory_path must use the format 'module:attribute'.")
    module_name, attribute_name = path.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute_name)
    except AttributeError as exc:
        raise AttributeError(f"Object '{attribute_name}' not found in module '{module_name}'.") from exc
