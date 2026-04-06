"""Gymnasium wrapper around the fast limit order book lane."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.interfaces import MatchingEngineProtocol, coerce_snapshot
from core.models import AgentDecision, MarketSnapshot, SimulationConfig


class MarketEnv(gym.Env):
    """Minimal Gymnasium environment for a market making or execution agent."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        matching_engine: MatchingEngineProtocol,
        config: SimulationConfig | None = None,
    ) -> None:
        super().__init__()
        self.matching_engine = matching_engine
        self.config = config or SimulationConfig()
        self.current_step = 0
        self.inventory = 0.0
        self.previous_snapshot: MarketSnapshot | None = None

        observation_size = self.config.depth_levels * 4 + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options

        self.matching_engine.reset()
        self.current_step = 0
        self.inventory = 0.0
        snapshot = self._get_snapshot()
        self.previous_snapshot = snapshot
        return snapshot.to_observation_vector(self.config.depth_levels), self._build_info(snapshot)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        decision = self._decode_action(action)
        if decision.action != "hold":
            order = decision.to_order()
            if order is not None:
                self.matching_engine.submit_order(order)
                self._update_inventory(decision)

        self.matching_engine.advance(1)
        self.current_step += 1

        snapshot = self._get_snapshot()
        reward = self._compute_reward(snapshot)
        self.previous_snapshot = snapshot

        terminated = False
        truncated = self.current_step >= self.config.max_episode_steps
        observation = snapshot.to_observation_vector(self.config.depth_levels)
        return observation, reward, terminated, truncated, self._build_info(snapshot)

    def render(self) -> dict[str, Any]:
        snapshot = self.previous_snapshot or self._get_snapshot()
        return snapshot.model_dump(mode="json")

    def close(self) -> None:
        """Nothing to clean up in the scaffolded environment."""

    def _get_snapshot(self) -> MarketSnapshot:
        snapshot = self.matching_engine.get_snapshot(depth_levels=self.config.depth_levels)
        return coerce_snapshot(snapshot)

    def _decode_action(self, action: int) -> AgentDecision:
        action_map = {
            0: AgentDecision(action="hold", volume=0.0, source="rl"),
            1: AgentDecision(action="buy", volume=self.config.rl_order_size, source="rl"),
            2: AgentDecision(action="sell", volume=self.config.rl_order_size, source="rl"),
        }
        if action not in action_map:
            raise ValueError(f"Unsupported discrete action: {action}")
        return action_map[action]

    def _update_inventory(self, decision: AgentDecision) -> None:
        signed_volume = decision.volume if decision.action == "buy" else -decision.volume
        self.inventory += signed_volume

    def _compute_reward(self, snapshot: MarketSnapshot) -> float:
        if self.previous_snapshot is None:
            return 0.0

        price_delta = snapshot.last_price - self.previous_snapshot.last_price
        return float(self.inventory * price_delta)

    def _build_info(self, snapshot: MarketSnapshot) -> dict[str, Any]:
        return {
            "tick": snapshot.tick,
            "last_price": snapshot.last_price,
            "mid_price": snapshot.mid_price,
            "spread": snapshot.spread,
            "imbalance": snapshot.imbalance,
            "inventory": self.inventory,
        }
