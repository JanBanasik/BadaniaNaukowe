"""Run a small real-provider swarm demo against a sample market snapshot."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.models import Level, MarketSnapshot
from sim import ExperimentLogger
from swarm import SwarmManager, generate_personas
from swarm.runtime import build_swarm_client

RunProvider = Literal["groq", "ollama", "lmstudio"]


def build_sample_snapshot() -> MarketSnapshot:
    """Construct a representative market snapshot for ad-hoc swarm runs."""
    return MarketSnapshot(
        bids=[
            Level(price=99.98, volume=160.0),
            Level(price=99.97, volume=150.0),
            Level(price=99.96, volume=142.0),
            Level(price=99.95, volume=131.0),
            Level(price=99.94, volume=120.0),
        ],
        asks=[
            Level(price=100.02, volume=158.0),
            Level(price=100.03, volume=149.0),
            Level(price=100.04, volume=140.0),
            Level(price=100.05, volume=130.0),
            Level(price=100.06, volume=122.0),
        ],
        last_price=100.00,
        mid_price=100.00,
        imbalance=0.01,
        spread=0.04,
        tick=250,
    )


async def main(provider: RunProvider, agent_count: int, market_news: str) -> None:
    """Run a small swarm collection cycle and print readable diagnostics."""
    resolved_client = build_swarm_client(provider, seed=0)
    personas = generate_personas(agent_count)
    manager = SwarmManager(client=resolved_client.client, personas=personas)
    snapshot = build_sample_snapshot()
    logger = ExperimentLogger(provider=provider, model_name=resolved_client.model_name)

    try:
        print(f"Provider: {provider}")
        print(f"Agents: {len(personas)}")
        print(f"Snapshot: mid={snapshot.mid_price:.2f}, spread={snapshot.spread:.2f}, tick={snapshot.tick}")
        print(f"News: {market_news}")
        print(f"Run directory: {logger.run_dir}")

        logger.write_metadata(
            {
                "agent_count": len(personas),
                "market_news": market_news,
                "client_config": resolved_client.config_payload,
                "personas": [persona.model_dump(mode="json") for persona in personas],
            }
        )
        logger.write_snapshot(snapshot)

        start_time = time.perf_counter()
        decisions = await manager.collect_decisions(snapshot=snapshot, market_news=market_news)
        elapsed_ms = round((time.perf_counter() - start_time) * 1000.0, 3)
        orders = [decision.to_order() for decision in decisions]
        actionable_orders = [order for order in orders if order is not None]

        decision_records = [decision.model_dump(mode="json") for decision in decisions]
        order_records = [order.model_dump(mode="json") for order in actionable_orders]
        logger.write_records_jsonl("decisions.jsonl", decision_records)
        logger.write_records_jsonl("orders.jsonl", order_records)
        logger.write_text_lines("errors.log", manager.last_errors)

        summary = {
            "provider": provider,
            "model_name": resolved_client.model_name,
            "agent_count": len(personas),
            "decision_count": len(decisions),
            "actionable_order_count": len(actionable_orders),
            "error_count": len(manager.last_errors),
            "duration_ms": elapsed_ms,
            "market_news": market_news,
            "snapshot_mid_price": snapshot.mid_price,
            "snapshot_spread": snapshot.spread,
            "snapshot_tick": snapshot.tick,
        }
        logger.write_summary(summary)

        print(f"Decisions received: {len(decisions)} / {len(personas)}")
        print(f"Actionable orders: {len(actionable_orders)}")
        print(f"Duration: {elapsed_ms} ms")
        if manager.last_errors:
            print(f"Errors: {len(manager.last_errors)}")
            for error in manager.last_errors[:5]:
                print(f"  - {error}")
        else:
            print("Errors: 0")

        print("Sample decisions:")
        for decision in decisions[: min(5, len(decisions))]:
            print(
                f"  - {decision.agent_id}: action={decision.action}, "
                f"price_type={decision.price_type}, volume={decision.volume}, "
                f"confidence={decision.confidence}, rationale={decision.rationale}"
            )

        print("Artifacts written:")
        print(f"  - {logger.run_dir / 'metadata.json'}")
        print(f"  - {logger.run_dir / 'snapshot.json'}")
        print(f"  - {logger.run_dir / 'summary.json'}")
        print(f"  - {logger.run_dir / 'decisions.jsonl'}")
        print(f"  - {logger.run_dir / 'orders.jsonl'}")
        print(f"  - {logger.run_dir / 'errors.log'}")
        print("Run complete.")
    finally:
        await resolved_client.client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a provider-backed swarm demo.")
    parser.add_argument("--provider", choices=["groq", "ollama", "lmstudio"], required=True)
    parser.add_argument("--agent-count", type=int, default=5)
    parser.add_argument(
        "--news",
        default="Inflation data slightly beats expectations while equity futures stay range-bound.",
    )
    args = parser.parse_args()
    asyncio.run(main(provider=args.provider, agent_count=args.agent_count, market_news=args.news))
