"""Canonical entrypoint for unified GABM-RL experiment runs."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from stable_baselines3 import PPO

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected via SB3
    torch = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.institutional import MeanReversionMarketMaker
from core import MarketEnv, MockMatchingEngine, RealEngineAdapterConfig, RealMatchingEngineAdapter, SimulationConfig
from core.interfaces import MatchingEngineProtocol
from sim import ExperimentLogger, SimulationOrchestrator
from sim.scenarios import list_scenarios, load_scenario
from swarm.manager import SwarmManager
from swarm.runtime import ResolvedProvider, build_swarm_client

MatchingEngineBackend = Literal["mock", "real"]


class ExperimentConfig(BaseModel):
    """Top-level configuration for one experiment run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str | None = None
    scenario_name: str | None = None
    output_base_dir: str = "runs"
    seed: int = 42
    num_cycles: int = Field(default=20, gt=0)
    swarm_update_freq: int = Field(default=5, gt=0)
    agent_count: int = Field(default=5, ge=0)
    market_news: str = "Inflation data slightly beats expectations while equity futures stay range-bound."
    enable_swarm: bool = True
    enable_institutional_agent: bool = True
    swarm_provider: ResolvedProvider = "mock"
    matching_engine_backend: MatchingEngineBackend = "mock"
    real_engine_factory_path: str | None = None
    real_engine_factory_kwargs: dict[str, Any] = Field(default_factory=dict)
    real_engine_reset_method: str = "reset"
    real_engine_advance_method: str = "advance"
    real_engine_submit_order_method: str = "submit_order"
    real_engine_snapshot_method: str = "get_snapshot"
    real_engine_submit_as_dict: bool = False
    pretrain_timesteps: int = Field(default=0, ge=0)
    deterministic_policy: bool = True
    ppo_policy: str = "MlpPolicy"
    ppo_learning_rate: float = Field(default=3e-4, gt=0.0)
    ppo_gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    ppo_n_steps: int = Field(default=64, gt=0)
    ppo_batch_size: int = Field(default=32, gt=0)
    institution_window: int = Field(default=5, gt=1)
    institution_threshold: float = Field(default=0.05, gt=0.0)
    institution_order_size: float = Field(default=5.0, gt=0.0)
    env: SimulationConfig = Field(default_factory=SimulationConfig)


def set_seed(seed: int, env: MarketEnv | None = None) -> None:
    """Seed Python, NumPy, torch, and optionally the Gymnasium environment."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if env is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)


def create_run_id(explicit_run_id: str | None = None) -> str:
    """Generate a timestamp-based run identifier."""
    if explicit_run_id:
        return explicit_run_id
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def build_matching_engine(config: ExperimentConfig) -> MatchingEngineProtocol:
    """Instantiate the selected matching engine backend."""
    if config.matching_engine_backend == "mock":
        return MockMatchingEngine(seed=config.seed)

    if not config.real_engine_factory_path:
        raise ValueError(
            "matching_engine_backend='real' requires real_engine_factory_path in ExperimentConfig or CLI."
        )

    adapter_config = RealEngineAdapterConfig(
        factory_path=config.real_engine_factory_path,
        factory_kwargs=config.real_engine_factory_kwargs,
        reset_method=config.real_engine_reset_method,
        advance_method=config.real_engine_advance_method,
        submit_order_method=config.real_engine_submit_order_method,
        snapshot_method=config.real_engine_snapshot_method,
        submit_as_dict=config.real_engine_submit_as_dict,
    )
    return RealMatchingEngineAdapter(config=adapter_config)


def save_config(run_dir: Path, config: ExperimentConfig) -> Path:
    """Persist the experiment configuration to config.json."""
    path = run_dir / "config.json"
    path.write_text(json.dumps(config.model_dump(mode="json"), indent=2, ensure_ascii=True), encoding="utf-8")
    return path


async def run_experiment(config: ExperimentConfig) -> Path:
    """Run the unified experiment loop and persist all outputs."""
    run_id = create_run_id(config.run_id)
    resolved_client = build_swarm_client(config.swarm_provider, seed=config.seed) if config.enable_swarm else build_swarm_client("mock", seed=config.seed)
    logger = ExperimentLogger(
        provider=resolved_client.provider,
        model_name=resolved_client.model_name,
        base_dir=config.output_base_dir,
        run_id=run_id,
    )
    run_dir = logger.run_dir
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    save_config(run_dir, config)

    matching_engine = build_matching_engine(config)
    institutional_agents = []
    if config.enable_institutional_agent:
        institutional_agents.append(
            MeanReversionMarketMaker(
                window=config.institution_window,
                deviation_threshold=config.institution_threshold,
                passive_volume=max(1.0, config.institution_order_size * 0.5),
                aggressive_volume=config.institution_order_size,
            )
        )

    env = MarketEnv(
        matching_engine=matching_engine,
        config=config.env,
        institutional_agents=institutional_agents,
    )
    set_seed(config.seed)
    env.reset(seed=config.seed)
    set_seed(config.seed, env=env)

    model = PPO(
        policy=config.ppo_policy,
        env=env,
        learning_rate=config.ppo_learning_rate,
        gamma=config.ppo_gamma,
        n_steps=config.ppo_n_steps,
        batch_size=config.ppo_batch_size,
        verbose=0,
        tensorboard_log=str(run_dir / "tensorboard"),
        seed=config.seed,
        device="cpu",
    )
    model.save(str(checkpoints_dir / "ppo_initial"))

    if config.pretrain_timesteps > 0:
        model.learn(total_timesteps=config.pretrain_timesteps, progress_bar=False)
        model.save(str(checkpoints_dir / "ppo_pretrained"))

    swarm_manager = SwarmManager(client=resolved_client.client, agent_count=config.agent_count)
    orchestrator = SimulationOrchestrator(
        matching_engine=matching_engine,
        swarm_manager=swarm_manager,
        config=config.env,
        logger=logger,
    )
    logger.write_metadata(
        {
            "component": "experiment_runner",
            "scenario_name": config.scenario_name,
            "swarm_provider": resolved_client.provider,
            "model_name": resolved_client.model_name,
            "swarm_client_config": resolved_client.config_payload,
        }
    )

    observation, _ = env.reset(seed=config.seed)
    metrics: list[dict[str, object]] = []

    try:
        for cycle in range(config.num_cycles):
            action, _ = model.predict(observation, deterministic=config.deterministic_policy)
            observation, reward, terminated, truncated, step_info = env.step(int(action))

            post_fast_lane_snapshot = orchestrator.snapshot_market()
            swarm_latency_ms = 0.0
            swarm_orders = 0
            injected_orders = 0
            provider_error_count = 0
            if config.enable_swarm and config.agent_count > 0 and (cycle + 1) % config.swarm_update_freq == 0:
                swarm_record = await orchestrator.process_snapshot(
                    snapshot=post_fast_lane_snapshot,
                    market_news=config.market_news,
                )
                swarm_latency_ms = float(swarm_record["swarm_latency_ms"])
                swarm_orders = int(swarm_record["swarm_orders"])
                injected_orders = int(swarm_record["injected_orders"])
                provider_error_count = int(swarm_record["swarm_errors"])

            final_snapshot = orchestrator.snapshot_market()
            metrics.append(
                {
                    "cycle": cycle,
                    "tick": final_snapshot.tick,
                    "mid_price": final_snapshot.mid_price,
                    "spread": final_snapshot.spread,
                    "imbalance": final_snapshot.imbalance,
                    "rl_step_reward": float(reward),
                    "rl_inventory": step_info["inventory"],
                    "rl_cash": step_info["rl_cash"],
                    "rl_realized_pnl": step_info["rl_realized_pnl"],
                    "rl_unrealized_pnl": step_info["rl_unrealized_pnl"],
                    "rl_total_equity": step_info["rl_total_equity"],
                    "swarm_latency_ms": swarm_latency_ms,
                    "swarm_orders": swarm_orders,
                    "injected_orders": injected_orders,
                    "provider_error_count": provider_error_count,
                    "institutional_order_count": step_info["institutional_order_count"],
                    "institutional_order_actions": ",".join(step_info["institutional_order_actions"]),
                    "env_terminated": terminated,
                    "env_truncated": truncated,
                }
            )

            if terminated or truncated:
                observation, _ = env.reset(seed=config.seed + cycle + 1)

        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(run_dir / "metrics.csv", index=False)
        orchestrator.flush_logs()
        model.save(str(checkpoints_dir / "ppo_final"))
        logger.write_summary(
            {
                "component": "experiment_runner",
                "run_id": run_id,
                "scenario_name": config.scenario_name,
                "num_cycles": config.num_cycles,
                "swarm_provider": resolved_client.provider,
                "matching_engine_backend": config.matching_engine_backend,
                "final_mid_price": metrics[-1]["mid_price"] if metrics else None,
                "final_spread": metrics[-1]["spread"] if metrics else None,
                "final_rl_total_equity": metrics[-1]["rl_total_equity"] if metrics else None,
                "final_rl_realized_pnl": metrics[-1]["rl_realized_pnl"] if metrics else None,
                "final_rl_unrealized_pnl": metrics[-1]["rl_unrealized_pnl"] if metrics else None,
                "total_institution_orders": sum(int(row["institutional_order_count"]) for row in metrics),
                "total_swarm_updates": sum(1 for row in metrics if row["swarm_latency_ms"] > 0.0),
                "total_provider_errors": sum(int(row["provider_error_count"]) for row in metrics),
            }
        )
        return run_dir
    finally:
        await resolved_client.client.close()
        env.close()


def parse_args() -> ExperimentConfig:
    """Parse CLI flags into an ExperimentConfig."""
    parser = argparse.ArgumentParser(description="Run a unified GABM-RL experiment.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-cycles", type=int, default=None)
    parser.add_argument("--swarm-update-freq", type=int, default=None)
    parser.add_argument("--agent-count", type=int, default=None)
    parser.add_argument("--swarm-provider", choices=["mock", "groq", "ollama", "lmstudio"], default=None)
    parser.add_argument("--matching-engine-backend", choices=["mock", "real"], default=None)
    parser.add_argument("--real-engine-factory-path", default=None)
    parser.add_argument("--pretrain-timesteps", type=int, default=None)
    parser.add_argument("--market-news", default=None)
    args = parser.parse_args()

    if args.list_scenarios:
        print("\n".join(list_scenarios()))
        raise SystemExit(0)

    config_data: dict[str, Any] = {
        "env": SimulationConfig(
            random_seed=42,
            noise_warmup_steps=2,
            fast_ticks_per_cycle=25,
        ).model_dump(mode="python")
    }

    if args.scenario:
        scenario_data = load_scenario(args.scenario)
        config_data = _deep_merge(config_data, scenario_data)
        config_data["scenario_name"] = args.scenario

    cli_overrides = {
        "run_id": args.run_id,
        "seed": args.seed,
        "num_cycles": args.num_cycles,
        "swarm_update_freq": args.swarm_update_freq,
        "agent_count": args.agent_count,
        "swarm_provider": args.swarm_provider,
        "matching_engine_backend": args.matching_engine_backend,
        "real_engine_factory_path": args.real_engine_factory_path,
        "pretrain_timesteps": args.pretrain_timesteps,
        "market_news": args.market_news,
    }
    config_data.update({key: value for key, value in cli_overrides.items() if value is not None})

    if "seed" in config_data:
        config_data.setdefault("env", {})
        config_data["env"]["random_seed"] = config_data["seed"]

    config_data["env"] = SimulationConfig.model_validate(config_data.get("env", {}))
    return ExperimentConfig.model_validate(config_data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


if __name__ == "__main__":
    experiment_config = parse_args()
    output_dir = asyncio.run(run_experiment(experiment_config))
    print(f"Experiment completed. Artifacts saved to: {output_dir}")
