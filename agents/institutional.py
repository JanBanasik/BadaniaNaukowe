"""Institutional fast-lane agents for deterministic market participation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque

import numpy as np

from core.models import MarketSnapshot, Order


class BaseInstitutionalAgent(ABC):
    """Abstract interface for deterministic fast-lane institutional agents."""

    @abstractmethod
    def generate_orders(self, snapshot: MarketSnapshot) -> list[Order]:
        """Produce one or more orders based on the latest market state."""

    def reset(self) -> None:
        """Reset any internal state before a new episode."""


class MeanReversionMarketMaker(BaseInstitutionalAgent):
    """Mean-reversion market maker that provides liquidity or fades extremes."""

    def __init__(
        self,
        window: int = 8,
        deviation_threshold: float = 0.05,
        passive_volume: float = 3.0,
        aggressive_volume: float = 8.0,
        min_quote_distance: float = 0.01,
        agent_id: str = "mean_reversion_market_maker",
    ) -> None:
        self.window = window
        self.deviation_threshold = deviation_threshold
        self.passive_volume = passive_volume
        self.aggressive_volume = aggressive_volume
        self.min_quote_distance = min_quote_distance
        self.agent_id = agent_id
        self.mid_prices: Deque[float] = deque(maxlen=window)

    def reset(self) -> None:
        """Clear moving-average state between episodes."""
        self.mid_prices.clear()

    def generate_orders(self, snapshot: MarketSnapshot) -> list[Order]:
        """Either fade strong deviations or quote both sides in calm markets."""
        self.mid_prices.append(snapshot.mid_price)
        if len(self.mid_prices) < self.window:
            return self._passive_quotes(snapshot)

        sma = float(np.mean(self.mid_prices))
        deviation = snapshot.mid_price - sma
        if abs(deviation) >= self.deviation_threshold:
            return [self._aggressive_counter_order(deviation)]
        return self._passive_quotes(snapshot)

    def _passive_quotes(self, snapshot: MarketSnapshot) -> list[Order]:
        half_spread = max(snapshot.spread * 0.5, self.min_quote_distance)
        bid_price = max(self.min_quote_distance, snapshot.mid_price - half_spread)
        ask_price = snapshot.mid_price + half_spread
        return [
            Order(
                action="buy",
                price_type="limit",
                volume=self.passive_volume,
                limit_price=bid_price,
                agent_id=self.agent_id,
                source="system",
            ),
            Order(
                action="sell",
                price_type="limit",
                volume=self.passive_volume,
                limit_price=ask_price,
                agent_id=self.agent_id,
                source="system",
            ),
        ]

    def _aggressive_counter_order(self, deviation: float) -> Order:
        action = "sell" if deviation > 0 else "buy"
        return Order(
            action=action,
            price_type="market",
            volume=self.aggressive_volume,
            agent_id=self.agent_id,
            source="system",
        )
