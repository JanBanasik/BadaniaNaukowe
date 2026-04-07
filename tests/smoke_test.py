"""Runnable smoke test for the mock-backed market environment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.config import PPOTrainingConfig
from agents.train_ppo import build_model
from core import MarketEnv, MockMatchingEngine, SimulationConfig


def run_environment_smoke(step_count: int = 15) -> None:
    """Reset the environment and drive it with random actions."""
    config = SimulationConfig(
        depth_levels=5,
        max_episode_steps=32,
        noise_warmup_steps=2,
        noise_bid_lambda=2.0,
        noise_ask_lambda=2.0,
        random_seed=123,
    )
    engine = MockMatchingEngine(seed=123)
    env = MarketEnv(matching_engine=engine, config=config)

    observation, info = env.reset(seed=123)
    print("Environment smoke test")
    print(f"  initial observation shape: {observation.shape}")
    print(f"  initial tick: {info['tick']}")
    print(f"  initial mid price: {info['mid_price']:.4f}")

    rng = np.random.default_rng(123)
    last_reward = 0.0
    truncated = False
    terminated = False
    last_info = info

    for step_index in range(step_count):
        action = int(rng.integers(0, env.action_space.n))
        observation, reward, terminated, truncated, last_info = env.step(action)
        last_reward = reward
        assert observation.shape == env.observation_space.shape
        if terminated or truncated:
            print(f"  rollout stopped early at step {step_index + 1}")
            break

    print(f"  final tick: {last_info['tick']}")
    print(f"  final mid price: {last_info['mid_price']:.4f}")
    print(f"  last reward: {last_reward:.6f}")
    print(f"  terminated: {terminated}, truncated: {truncated}")

    env.close()


def run_ppo_initialization_smoke() -> None:
    """Construct PPO against the mock environment without training."""
    env_config = SimulationConfig(
        max_episode_steps=32,
        noise_warmup_steps=0,
        noise_bid_lambda=1.0,
        noise_ask_lambda=1.0,
        random_seed=321,
    )
    training_config = PPOTrainingConfig(
        total_timesteps=16,
        n_steps=16,
        batch_size=16,
        verbose=0,
        tensorboard_log=None,
        env=env_config,
    )

    engine = MockMatchingEngine(seed=321)
    env, model = build_model(matching_engine=engine, config=training_config)
    print("PPO smoke test")
    print(f"  environment type: {type(env).__name__}")
    print(f"  policy class: {model.policy.__class__.__name__}")
    print(f"  action space size: {env.action_space.n}")
    env.close()


def main() -> None:
    run_environment_smoke()
    run_ppo_initialization_smoke()
    print("Smoke test completed successfully.")


if __name__ == "__main__":
    main()
