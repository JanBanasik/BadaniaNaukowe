"""Test backend used to validate the real-engine adapter path."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FakeRealBackend:
    tick: int = 0
    mid_price: float = 100.0
    spread: float = 0.1
    submitted_orders: list[dict] = field(default_factory=list)

    def reset(self) -> None:
        self.tick = 0
        self.mid_price = 100.0
        self.spread = 0.1
        self.submitted_orders.clear()

    def advance(self, ticks: int = 1) -> None:
        self.tick += ticks
        self.mid_price += 0.01 * ticks

    def submit_order(self, order: dict) -> None:
        self.submitted_orders.append(order)
        signed_volume = order["volume"] if order["action"] == "buy" else -order["volume"]
        self.mid_price += 0.001 * signed_volume

    def get_snapshot(self, depth_levels: int = 5) -> dict:
        bids = [
            {"price": self.mid_price - self.spread * 0.5 - 0.01 * index, "volume": 100.0 - index}
            for index in range(depth_levels)
        ]
        asks = [
            {"price": self.mid_price + self.spread * 0.5 + 0.01 * index, "volume": 100.0 - index}
            for index in range(depth_levels)
        ]
        return {
            "bids": bids,
            "asks": asks,
            "last_price": self.mid_price,
            "mid_price": self.mid_price,
            "imbalance": 0.0,
            "spread": self.spread,
            "tick": self.tick,
        }


def create_backend() -> FakeRealBackend:
    return FakeRealBackend()
