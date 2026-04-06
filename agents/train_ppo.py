"""Training entrypoints for the PPO agent baseline."""

from __future__ import annotations

from stable_baselines3 import PPO

from agents.config import PPOTrainingConfig
from core.env import MarketEnv
from core.interfaces import MatchingEngineProtocol


def build_environment(
    matching_engine: MatchingEngineProtocol,
    config: PPOTrainingConfig,
) -> MarketEnv:
    """Create the Gymnasium environment around a concrete matching engine."""
    return MarketEnv(matching_engine=matching_engine, config=config.env)


def build_model(
    matching_engine: MatchingEngineProtocol,
    config: PPOTrainingConfig,
) -> tuple[MarketEnv, PPO]:
    """Create the PPO model and its wrapped market environment."""
    env = build_environment(matching_engine=matching_engine, config=config)
    model = PPO(
        policy=config.policy,
        env=env,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        verbose=config.verbose,
        tensorboard_log=config.tensorboard_log,
    )
    return env, model


def train(
    matching_engine: MatchingEngineProtocol,
    config: PPOTrainingConfig | None = None,
) -> PPO:
    """Run PPO training with the provided matching engine implementation."""
    training_config = config or PPOTrainingConfig()
    _, model = build_model(matching_engine=matching_engine, config=training_config)
    model.learn(total_timesteps=training_config.total_timesteps)
    return model


def main() -> None:
    """Guide users toward the real integration point."""
    print("Use agents.train_ppo.train(matching_engine=...) after wiring a concrete LOB engine.")


if __name__ == "__main__":
    main()
