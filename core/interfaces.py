"""Core interfaces that decouple the simulator from concrete LOB code."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from core.models import MarketSnapshot, Order


class MatchingEngineProtocol(Protocol):
    """Minimal contract expected from the supervisor-provided matching engine."""

    def reset(self) -> None:
        """Reset the market state to the start of an episode."""

    def advance(self, ticks: int = 1) -> None:
        """Advance the fast lane by the requested number of ticks."""

    def submit_order(self, order: Order) -> None:
        """Inject an order into the order book or matching engine."""

    def get_snapshot(self, depth_levels: int = 5) -> MarketSnapshot | Mapping[str, Any]:
        """Return a market snapshot in typed or dictionary form."""


def coerce_snapshot(snapshot: MarketSnapshot | Mapping[str, Any]) -> MarketSnapshot:
    """Normalize snapshot payloads from concrete engines into one schema."""
    if isinstance(snapshot, MarketSnapshot):
        return snapshot
    return MarketSnapshot.model_validate(snapshot)
