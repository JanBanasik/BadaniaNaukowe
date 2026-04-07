"""Two-speed simulation middleware for fast LOB and slow swarm lanes."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import time
from typing import Any

import pandas as pd

from core.interfaces import MatchingEngineProtocol, coerce_snapshot
from core.models import MarketSnapshot, Order, SimulationConfig
from sim.experiment_logger import ExperimentLogger
from swarm.manager import SwarmManager


class SimulationOrchestrator:
    """Synchronize fast exchange ticks with slower LLM swarm interventions."""

    def __init__(
        self,
        matching_engine: MatchingEngineProtocol,
        swarm_manager: SwarmManager,
        config: SimulationConfig | None = None,
        logger: ExperimentLogger | None = None,
    ) -> None:
        self.matching_engine = matching_engine
        self.swarm_manager = swarm_manager
        self.config = config or SimulationConfig()
        self.logger = logger
        self.records: list[dict[str, Any]] = []
        self.cycle_index = 0
        self.last_snapshot: MarketSnapshot | None = None
        self.last_injected_orders: list[Order] = []

        if self.logger is not None:
            self.logger.write_metadata(
                {
                    "component": "simulation_orchestrator",
                    "config": self.config.model_dump(mode="json"),
                }
            )

    def run_fast_lane(self, ticks: int | None = None) -> MarketSnapshot:
        """Advance the LOB for a batch of high-frequency ticks."""
        self.matching_engine.advance(ticks or self.config.fast_ticks_per_cycle)
        snapshot = self.matching_engine.get_snapshot(depth_levels=self.config.depth_levels)
        return coerce_snapshot(snapshot)

    async def run_cycle(self, market_news: str | None = None) -> dict[str, Any]:
        """Execute one fast-lane batch followed by one slow-lane swarm pulse."""
        snapshot = self.run_fast_lane()
        return await self.process_snapshot(snapshot=snapshot, market_news=market_news)

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

    async def process_snapshot(
        self,
        snapshot: MarketSnapshot,
        market_news: str | None = None,
    ) -> dict[str, Any]:
        """Run only the slow lane against an already-produced market snapshot."""
        news = market_news or self.config.default_market_news
        start_time = time.perf_counter()
        swarm_orders = await self.swarm_manager.generate_orders(snapshot=snapshot, market_news=news)
        swarm_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
        injected_orders = self.inject_swarm_orders(swarm_orders)
        self.last_snapshot = snapshot
        self.last_injected_orders = injected_orders

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
            "swarm_latency_ms": swarm_latency_ms,
        }
        self.records.append(record)
        self._log_cycle(snapshot=snapshot, swarm_orders=swarm_orders, injected_orders=injected_orders, record=record)
        self.cycle_index += 1
        return record

    def flush_logs(self) -> Path | None:
        """Persist accumulated orchestrator metrics into the configured run folder."""
        if self.logger is None:
            return None

        dataframe = self.to_dataframe()
        csv_path = self.logger.run_dir / "orchestrator_metrics.csv"
        dataframe.to_csv(csv_path, index=False)
        self.logger.write_json(
            "orchestrator_summary.json",
            {
                "component": "simulation_orchestrator",
                "cycle_count": len(self.records),
                "latest_tick": self.records[-1]["tick"] if self.records else None,
                "latest_mid_price": self.records[-1]["mid_price"] if self.records else None,
                "total_swarm_orders": sum(record["swarm_orders"] for record in self.records),
                "total_injected_orders": sum(record["injected_orders"] for record in self.records),
                "total_swarm_volume": sum(record["swarm_volume"] for record in self.records),
                "total_swarm_errors": sum(record["swarm_errors"] for record in self.records),
            }
        )
        return csv_path

    def _log_cycle(
        self,
        *,
        snapshot: MarketSnapshot,
        swarm_orders: list[Order],
        injected_orders: list[Order],
        record: dict[str, Any],
    ) -> None:
        if self.logger is None:
            return

        self.logger.write_snapshot(snapshot)
        self.logger.append_record_jsonl("orchestrator_cycles.jsonl", record)
        self.logger.write_records_jsonl(
            "latest_swarm_orders.jsonl",
            [order.model_dump(mode="json") for order in swarm_orders],
        )
        self.logger.write_records_jsonl(
            "latest_injected_orders.jsonl",
            [order.model_dump(mode="json") for order in injected_orders],
        )
