"""Deterministic integration checks for the unified experiment runner."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from sim.run_experiment import ExperimentConfig, run_experiment


class RunExperimentSmokeTests(unittest.TestCase):
    def test_run_experiment_creates_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                run_id="artifact_check",
                output_base_dir=temp_dir,
                num_cycles=4,
                swarm_update_freq=2,
                swarm_provider="mock",
                matching_engine_backend="mock",
                agent_count=3,
            )
            run_dir = asyncio.run(run_experiment(config))

            expected_paths = [
                "config.json",
                "metrics.csv",
                "summary.json",
                "orchestrator_cycles.jsonl",
                "orchestrator_metrics.csv",
                "orchestrator_summary.json",
                "checkpoints/ppo_initial.zip",
                "checkpoints/ppo_final.zip",
            ]
            for relative_path in expected_paths:
                self.assertTrue((run_dir / relative_path).exists(), relative_path)

    def test_run_experiment_is_deterministic_for_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_kwargs = {
                "output_base_dir": temp_dir,
                "num_cycles": 4,
                "swarm_update_freq": 2,
                "swarm_provider": "mock",
                "matching_engine_backend": "mock",
                "agent_count": 3,
                "seed": 123,
            }
            first_run = asyncio.run(run_experiment(ExperimentConfig(run_id="run_a", **base_kwargs)))
            second_run = asyncio.run(run_experiment(ExperimentConfig(run_id="run_b", **base_kwargs)))

            first_metrics = pd.read_csv(first_run / "metrics.csv")
            second_metrics = pd.read_csv(second_run / "metrics.csv")
            deterministic_columns = [column for column in first_metrics.columns if column != "swarm_latency_ms"]
            pd.testing.assert_frame_equal(
                first_metrics[deterministic_columns],
                second_metrics[deterministic_columns],
            )

    def test_real_backend_adapter_path_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = ExperimentConfig(
                run_id="real_backend_check",
                output_base_dir=temp_dir,
                num_cycles=3,
                swarm_update_freq=2,
                enable_swarm=False,
                agent_count=0,
                matching_engine_backend="real",
                real_engine_factory_path="tests.fake_real_backend:create_backend",
                real_engine_submit_as_dict=True,
            )
            run_dir = asyncio.run(run_experiment(config))
            metrics = pd.read_csv(run_dir / "metrics.csv")
            self.assertEqual(len(metrics), 3)


if __name__ == "__main__":
    unittest.main()
