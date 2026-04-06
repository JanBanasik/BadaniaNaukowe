"""Training configuration for the PPO trading agent."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from core.models import SimulationConfig


class PPOTrainingConfig(BaseModel):
    """Centralized PPO hyperparameters for baseline experiments."""

    model_config = ConfigDict(extra="forbid")

    policy: str = "MlpPolicy"
    total_timesteps: int = Field(default=100_000, gt=0)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    gamma: float = Field(default=0.99, gt=0.0, le=1.0)
    n_steps: int = Field(default=2_048, gt=0)
    batch_size: int = Field(default=64, gt=0)
    verbose: int = Field(default=1, ge=0)
    tensorboard_log: str | None = "data/tensorboard"
    env: SimulationConfig = Field(default_factory=SimulationConfig)
