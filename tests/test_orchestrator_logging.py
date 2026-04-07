"""Integration checks for orchestrator logging artifacts."""

from __future__ import annotations

import asyncio
import tempfile
import unittest

from core import MockMatchingEngine, SimulationConfig
from sim import ExperimentLogger, SimulationOrchestrator
from swarm.manager import SwarmManager
from swarm.runtime import MockSwarmClient


class OrchestratorLoggingTests(unittest.TestCase):
    def test_orchestrator_flush_logs_writes_cycle_artifacts(self) -> None:
        async def exercise() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                logger = ExperimentLogger(
                    provider="test",
                    model_name="mock-swarm",
                    base_dir=temp_dir,
                    run_id="orchestrator_case",
                )
                manager = SwarmManager(client=MockSwarmClient(seed=7), agent_count=2)
                orchestrator = SimulationOrchestrator(
                    matching_engine=MockMatchingEngine(seed=7),
                    swarm_manager=manager,
                    config=SimulationConfig(),
                    logger=logger,
                )
                await orchestrator.run_cycle("orchestrator logging test")
                csv_path = orchestrator.flush_logs()

                self.assertTrue((logger.run_dir / "orchestrator_cycles.jsonl").exists())
                self.assertTrue((logger.run_dir / "latest_swarm_orders.jsonl").exists())
                self.assertTrue((logger.run_dir / "latest_injected_orders.jsonl").exists())
                self.assertTrue((logger.run_dir / "orchestrator_summary.json").exists())
                self.assertEqual(csv_path, logger.run_dir / "orchestrator_metrics.csv")

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
