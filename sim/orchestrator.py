"""Two-speed simulation middleware for fast LOB and slow swarm lanes."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from core.interfaces import MatchingEngineProtocol, coerce_snapshot
from core.models import MarketSnapshot, Order, SimulationConfig
from swarm.manager import SwarmManager


class SimulationOrchestrator:
    """Synchronize fast exchange ticks with slower LLM swarm interventions."""

    def __init__(
        self,
        matching_engine: MatchingEngineProtocol,
        swarm_manager: SwarmManager,
        config: SimulationConfig | None = None,
    ) -> None:
        self.matching_engine = matching_engine
        self.swarm_manager = swarm_manager
        self.config = config or SimulationConfig()
        self.records: list[dict[str, Any]] = []
        self.cycle_index = 0

    def run_fast_lane(self, ticks: int | None = None) -> MarketSnapshot:
        """Advance the LOB for a batch of high-frequency ticks."""
        self.matching_engine.advance(ticks or self.config.fast_ticks_per_cycle)
        snapshot = self.matching_engine.get_snapshot(depth_levels=self.config.depth_levels)
        return coerce_snapshot(snapshot)

    async def run_cycle(self, market_news: str | None = None) -> dict[str, Any]:
        """Execute one fast-lane batch followed by one slow-lane swarm pulse."""
        news = market_news or self.config.default_market_news
        snapshot = self.run_fast_lane()
        swarm_orders = await self.swarm_manager.generate_orders(snapshot=snapshot, market_news=news)
        injected_orders = self.inject_swarm_orders(swarm_orders)

        record = {
            "cycle": self.cycle_index,
            "tick": snapshot.tick,
            "market_news": news,
            "last_price": snapshot.last_price,
            "mid_price": snapshot.mid_price,
            "spread": snapshot.spread,
            "imbalance": snapshot.imbalance,
            "swarm_orders": len(swarm_orders),
            "injected_orders": len(injected_orders),
            "swarm_volume": sum(order.volume for order in injected_orders),
            "swarm_errors": len(self.swarm_manager.last_errors),
        }
        self.records.append(record)
        self.cycle_index += 1
        return record

    async def run(
        self,
        cycles: int,
        news_feed: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Run multiple cycles and return a metrics dataframe."""
        for index in range(cycles):
            if news_feed and index < len(news_feed):
                news = news_feed[index]
            else:
                news = self.config.default_market_news
            await self.run_cycle(market_news=news)
        return self.to_dataframe()

    def inject_swarm_orders(self, swarm_orders: list[Order]) -> list[Order]:
        """Inject the swarm output into the LOB as market orders."""
        injected_orders: list[Order] = []
        for order in swarm_orders:
            market_order = order.model_copy(
                update={
                    "price_type": "market",
                    "limit_price": None,
                }
            )
            self.matching_engine.submit_order(market_order)
            injected_orders.append(market_order)
        return injected_orders

    def snapshot_market(self) -> MarketSnapshot:
        """Fetch a standalone market snapshot without advancing time."""
        snapshot = self.matching_engine.get_snapshot(depth_levels=self.config.depth_levels)
        return coerce_snapshot(snapshot)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert recorded cycle metrics into a dataframe."""
        return pd.DataFrame(self.records)

    def export_metrics(self, path: str | Path) -> pd.DataFrame:
        """Persist orchestration metrics to disk."""
        dataframe = self.to_dataframe()
        dataframe.to_csv(path, index=False)
        return dataframe
