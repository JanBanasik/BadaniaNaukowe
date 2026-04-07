"""Lightweight mock matching engine for local environment smoke tests."""

from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np

from core.models import Level, MarketSnapshot, Order


class MockMatchingEngine:
    """Minimal matching-engine stand-in with stable synthetic book snapshots."""

    def __init__(
        self,
        initial_mid_price: float = 100.0,
        base_spread: float = 0.10,
        tick_size: float = 0.01,
        impact_coefficient: float = 0.0005,
        max_impact_per_order: float = 0.05,
        seed: int | None = 7,
    ) -> None:
        self.initial_mid_price = initial_mid_price
        self.base_spread = base_spread
        self.tick_size = tick_size
        self.impact_coefficient = impact_coefficient
        self.max_impact_per_order = max_impact_per_order
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.tick = 0
        self.mid_price = initial_mid_price
        self.last_price = initial_mid_price
        self.spread = base_spread
        self.net_flow = 0.0
        self.submitted_orders: Deque[Order] = deque(maxlen=256)

    def reset(self) -> None:
        """Reset the engine to a deterministic baseline state."""
        self.rng = np.random.default_rng(self.seed)
        self.tick = 0
        self.mid_price = self.initial_mid_price
        self.last_price = self.initial_mid_price
        self.spread = self.base_spread
        self.net_flow = 0.0
        self.submitted_orders.clear()

    def advance(self, ticks: int = 1) -> None:
        """Advance the mock clock and add a small amount of background drift."""
        if ticks <= 0:
            return

        self.tick += ticks
        self.net_flow *= 0.85**ticks
        drift = float(self.rng.normal(loc=0.0, scale=0.002 * np.sqrt(ticks)))
        self.mid_price = max(self.tick_size, self.mid_price + drift + 0.001 * self.net_flow)
        self.spread = max(self.tick_size, self.base_spread + min(abs(self.net_flow) * 0.0005, 0.05))
        self.last_price = self.mid_price

    def submit_order(self, order: Order) -> None:
        """Accept orders and nudge the dummy mid-price via signed order flow."""
        self.submitted_orders.append(order)

        signed_volume = order.volume if order.action == "buy" else -order.volume
        impact_scale = 1.0 if order.price_type == "market" else 0.5
        impact = np.clip(
            signed_volume * self.impact_coefficient * impact_scale,
            -self.max_impact_per_order,
            self.max_impact_per_order,
        )

        self.net_flow += signed_volume
        self.mid_price = max(self.tick_size, self.mid_price + float(impact))
        self.last_price = self.mid_price

    def get_snapshot(self, depth_levels: int = 5) -> MarketSnapshot:
        """Return a coherent synthetic book around the current mid-price."""
        effective_spread = max(self.tick_size, self.spread)
        half_spread = effective_spread * 0.5

        bid_prices = self.mid_price - half_spread - self.tick_size * np.arange(depth_levels)
        ask_prices = self.mid_price + half_spread + self.tick_size * np.arange(depth_levels)

        base_bid_volume = 60.0 + max(self.net_flow, 0.0) * 0.1
        base_ask_volume = 60.0 + max(-self.net_flow, 0.0) * 0.1
        bid_volumes = np.maximum(
            1.0,
            base_bid_volume - 3.0 * np.arange(depth_levels) + self.rng.uniform(-5.0, 5.0, size=depth_levels),
        )
        ask_volumes = np.maximum(
            1.0,
            base_ask_volume - 3.0 * np.arange(depth_levels) + self.rng.uniform(-5.0, 5.0, size=depth_levels),
        )

        bids = [
            Level(price=float(max(self.tick_size, bid_prices[index])), volume=float(bid_volumes[index]))
            for index in range(depth_levels)
        ]
        asks = [
            Level(price=float(max(self.tick_size, ask_prices[index])), volume=float(ask_volumes[index]))
            for index in range(depth_levels)
        ]

        bid_total = float(np.sum(bid_volumes))
        ask_total = float(np.sum(ask_volumes))
        total_volume = bid_total + ask_total
        imbalance = 0.0 if total_volume == 0.0 else (bid_total - ask_total) / total_volume

        return MarketSnapshot(
            bids=bids,
            asks=asks,
            last_price=float(self.last_price),
            mid_price=float(self.mid_price),
            imbalance=float(np.clip(imbalance, -1.0, 1.0)),
            spread=float(effective_spread),
            tick=self.tick,
        )
