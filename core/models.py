"""Shared data models for the hybrid market simulator."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

OrderAction = Literal["buy", "sell"]
DecisionAction = Literal["buy", "sell", "hold"]
PriceType = Literal["market", "limit", "stop_market", "stop_limit"]
OrderSource = Literal["rl", "swarm", "system", "institutional", "accumulation", "noise"]
TimeInForce = Literal["GTC", "IOC", "FOK", "GTD"]
MarketPhase = Literal["pre_open", "opening_auction", "continuous", "closing_auction", "closed"]


def _validate_price_fields(
    price_type: PriceType,
    limit_price: float | None,
    stop_price: float | None,
) -> None:
    if price_type == "market":
        if limit_price is not None:
            raise ValueError("limit_price must be omitted when price_type='market'.")
        if stop_price is not None:
            raise ValueError("stop_price must be omitted when price_type='market'.")
    elif price_type == "limit":
        if limit_price is None:
            raise ValueError("limit_price is required when price_type='limit'.")
        if stop_price is not None:
            raise ValueError("stop_price must be omitted when price_type='limit'.")
    elif price_type == "stop_market":
        if stop_price is None:
            raise ValueError("stop_price is required when price_type='stop_market'.")
        if limit_price is not None:
            raise ValueError("limit_price must be omitted when price_type='stop_market'.")
    elif price_type == "stop_limit":
        if stop_price is None:
            raise ValueError("stop_price is required when price_type='stop_limit'.")
        if limit_price is None:
            raise ValueError("limit_price is required when price_type='stop_limit'.")


def _validate_time_in_force(time_in_force: TimeInForce, expiry_tick: int | None) -> None:
    if time_in_force == "GTD" and expiry_tick is None:
        raise ValueError("expiry_tick is required when time_in_force='GTD'.")
    if time_in_force != "GTD" and expiry_tick is not None:
        raise ValueError("expiry_tick must be omitted unless time_in_force='GTD'.")


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
    stop_price: float | None = Field(default=None, ge=0.0)
    time_in_force: TimeInForce = "GTC"
    expiry_tick: int | None = Field(default=None, ge=0)
    agent_id: str | None = None
    source: OrderSource = "system"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def validate_price_fields(self) -> "Order":
        _validate_price_fields(self.price_type, self.limit_price, self.stop_price)
        _validate_time_in_force(self.time_in_force, self.expiry_tick)
        return self


class AgentDecision(BaseModel):
    """Validated action proposal emitted by the RL or swarm agent."""

    model_config = ConfigDict(extra="forbid")

    action: DecisionAction
    price_type: PriceType = "market"
    volume: float = Field(default=0.0, ge=0.0)
    limit_price: float | None = Field(default=None, ge=0.0)
    stop_price: float | None = Field(default=None, ge=0.0)
    time_in_force: TimeInForce = "GTC"
    expiry_tick: int | None = Field(default=None, ge=0)
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
            if self.stop_price is not None:
                raise ValueError("Hold decisions cannot set stop_price.")
            if self.expiry_tick is not None:
                raise ValueError("Hold decisions cannot set expiry_tick.")
            return self

        if self.volume <= 0.0:
            raise ValueError("Trade decisions must have positive volume.")
        _validate_price_fields(self.price_type, self.limit_price, self.stop_price)
        _validate_time_in_force(self.time_in_force, self.expiry_tick)
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
            stop_price=self.stop_price,
            time_in_force=self.time_in_force,
            expiry_tick=self.expiry_tick,
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
    phase: MarketPhase = "continuous"
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
    noise_bid_lambda: float = Field(default=3.0, ge=0.0)
    noise_ask_lambda: float = Field(default=3.0, ge=0.0)
    noise_price_scale: float = Field(default=0.5, gt=0.0)
    noise_min_distance: float = Field(default=0.01, ge=0.0)
    noise_min_volume: float = Field(default=1.0, gt=0.0)
    noise_max_volume: float = Field(default=100.0, gt=0.0)
    noise_warmup_steps: int = Field(default=10, ge=0)
    reward_inventory_penalty: float = Field(default=0.001, ge=0.0)
    transaction_cost_bps: float = Field(default=0.0, ge=0.0)
    commission_bps: float = Field(default=0.0, ge=0.0)
    commission_min_per_trade: float = Field(default=0.0, ge=0.0)
    ticks_per_session: int = Field(default=1000, gt=0)
    sessions_per_year: int = Field(default=252, gt=0)
    cycles_per_session: int = Field(default=40, gt=0)
    opening_auction_ticks: int = Field(default=20, ge=0)
    closing_auction_ticks: int = Field(default=20, ge=0)
    accumulation_days: int = Field(default=5, gt=0)
    accumulation_volume: float = Field(default=1.0, gt=0.0)
    session_schedule: list[tuple[str, int]] | None = None
    random_seed: int | None = None

    @model_validator(mode="after")
    def validate_auction_windows(self) -> "SimulationConfig":
        total_auction = self.opening_auction_ticks + self.closing_auction_ticks
        if total_auction >= self.ticks_per_session:
            raise ValueError(
                "opening_auction_ticks + closing_auction_ticks must be < ticks_per_session."
            )
        return self
