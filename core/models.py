"""Shared data models for the hybrid market simulator."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

OrderAction = Literal["buy", "sell"]
DecisionAction = Literal["buy", "sell", "hold"]
PriceType = Literal["market", "limit"]
OrderSource = Literal["rl", "swarm", "system"]


class Level(BaseModel):
    """Represents one price level in the order book."""

    model_config = ConfigDict(extra="forbid")

    price: float = Field(ge=0.0)
    volume: float = Field(ge=0.0)


class Order(BaseModel):
    """Normalized order schema used by the matching engine."""

    model_config = ConfigDict(extra="forbid")

    action: OrderAction
    price_type: PriceType = "market"
    volume: float = Field(gt=0.0)
    limit_price: float | None = Field(default=None, ge=0.0)
    agent_id: str | None = None
    source: OrderSource = "system"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_limit_fields(self) -> "Order":
        if self.price_type == "limit" and self.limit_price is None:
            raise ValueError("limit_price is required when price_type='limit'.")
        if self.price_type == "market" and self.limit_price is not None:
            raise ValueError("limit_price must be omitted when price_type='market'.")
        return self


class AgentDecision(BaseModel):
    """Validated action proposal emitted by the RL or swarm agent."""

    model_config = ConfigDict(extra="forbid")

    action: DecisionAction
    price_type: PriceType = "market"
    volume: float = Field(default=0.0, ge=0.0)
    limit_price: float | None = Field(default=None, ge=0.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str | None = None
    agent_id: str | None = None
    source: OrderSource = "system"

    @model_validator(mode="after")
    def validate_consistency(self) -> "AgentDecision":
        if self.action == "hold":
            if self.volume != 0.0:
                raise ValueError("Hold decisions must have volume=0.")
            if self.limit_price is not None:
                raise ValueError("Hold decisions cannot set limit_price.")
            return self

        if self.volume <= 0.0:
            raise ValueError("Trade decisions must have positive volume.")
        if self.price_type == "limit" and self.limit_price is None:
            raise ValueError("Limit decisions must include limit_price.")
        if self.price_type == "market" and self.limit_price is not None:
            raise ValueError("Market decisions cannot include limit_price.")
        return self

    def to_order(self) -> Order | None:
        """Convert actionable decisions into normalized orders."""
        if self.action == "hold":
            return None

        return Order(
            action=self.action,
            price_type=self.price_type,
            volume=self.volume,
            limit_price=self.limit_price,
            agent_id=self.agent_id,
            source=self.source,
        )


class MarketSnapshot(BaseModel):
    """Serializable snapshot of the market state used by both lanes."""

    model_config = ConfigDict(extra="forbid")

    bids: list[Level] = Field(default_factory=list)
    asks: list[Level] = Field(default_factory=list)
    last_price: float = Field(ge=0.0)
    mid_price: float = Field(ge=0.0)
    imbalance: float = Field(ge=-1.0, le=1.0)
    spread: float = Field(ge=0.0)
    tick: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_observation_vector(self, depth_levels: int) -> np.ndarray:
        """Flatten the order book and summary metrics into a fixed-size vector."""
        bids = self.bids[:depth_levels]
        asks = self.asks[:depth_levels]

        features: list[float] = []
        for levels in (bids, asks):
            padded_levels = list(levels) + [Level(price=0.0, volume=0.0)] * (depth_levels - len(levels))
            for level in padded_levels:
                features.extend((level.price, level.volume))

        features.extend((self.imbalance, self.last_price, self.mid_price, self.spread))
        return np.asarray(features, dtype=np.float32)


class SimulationConfig(BaseModel):
    """Shared configuration for environment and simulation timing."""

    model_config = ConfigDict(extra="forbid")

    depth_levels: int = Field(default=5, gt=0)
    fast_ticks_per_cycle: int = Field(default=25, gt=0)
    max_episode_steps: int = Field(default=1_000, gt=0)
    rl_order_size: float = Field(default=1.0, gt=0.0)
    swarm_agent_count: int = Field(default=50, ge=1, le=100)
    default_market_news: str = "No exogenous market news."
