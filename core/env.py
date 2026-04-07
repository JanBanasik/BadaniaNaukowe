"""Gymnasium wrapper and synthetic baseline flow for the fast LOB lane."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.interfaces import MatchingEngineProtocol, coerce_snapshot
from core.models import AgentDecision, MarketSnapshot, Order, SimulationConfig

if TYPE_CHECKING:
    from agents.institutional import BaseInstitutionalAgent


class PoissonNoiseGenerator:
    """Inject zero-intelligence limit-order flow around the current mid-price."""

    def __init__(
        self,
        bid_lambda: float,
        ask_lambda: float,
        price_scale: float,
        min_distance: float,
        min_volume: float,
        max_volume: float,
        seed: int | None = None,
    ) -> None:
        self.bid_lambda = bid_lambda
        self.ask_lambda = ask_lambda
        self.price_scale = price_scale
        self.min_distance = min_distance
        self.min_volume = min_volume
        self.max_volume = max_volume
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        """Re-seed the generator when the environment resets."""
        self.rng = np.random.default_rng(seed)

    def generate_orders(self, snapshot: MarketSnapshot) -> list[Order]:
        """Sample bid and ask limit orders using vectorized NumPy draws."""
        bid_count = int(self.rng.poisson(self.bid_lambda))
        ask_count = int(self.rng.poisson(self.ask_lambda))
        if bid_count == 0 and ask_count == 0:
            return []

        reference_mid = snapshot.mid_price if snapshot.mid_price > 0.0 else max(snapshot.last_price, 1.0)
        half_spread = max(snapshot.spread * 0.5, self.min_distance)
        timestamp = datetime.now(timezone.utc)

        orders: list[Order] = []
        orders.extend(
            self._build_orders(
                action="buy",
                count=bid_count,
                reference_mid=reference_mid,
                half_spread=half_spread,
                timestamp=timestamp,
            )
        )
        orders.extend(
            self._build_orders(
                action="sell",
                count=ask_count,
                reference_mid=reference_mid,
                half_spread=half_spread,
                timestamp=timestamp,
            )
        )
        return orders

    def _build_orders(
        self,
        action: str,
        count: int,
        reference_mid: float,
        half_spread: float,
        timestamp: datetime,
    ) -> list[Order]:
        if count <= 0:
            return []

        distances = np.abs(
            self.rng.normal(
                loc=half_spread,
                scale=self.price_scale,
                size=count,
            )
        )
        distances = np.maximum(distances, self.min_distance)
        volumes = self.rng.uniform(self.min_volume, self.max_volume, size=count)

        if action == "buy":
            prices = np.maximum(reference_mid - distances, self.min_distance)
        else:
            prices = reference_mid + distances

        return [
            Order.model_construct(
                action=action,
                price_type="limit",
                volume=float(volumes[index]),
                limit_price=float(prices[index]),
                agent_id=f"noise_{action}_{index}",
                source="system",
                timestamp=timestamp,
            )
            for index in range(count)
        ]


class MarketEnv(gym.Env):
    """Minimal Gymnasium environment for a market making or execution agent."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        matching_engine: MatchingEngineProtocol,
        config: SimulationConfig | None = None,
        institutional_agents: Iterable["BaseInstitutionalAgent"] | None = None,
    ) -> None:
        super().__init__()
        self.matching_engine = matching_engine
        self.config = config or SimulationConfig()
        self.institutional_agents = list(institutional_agents or [])
        self.current_step = 0
        self.inventory = 0.0
        self.cash = 0.0
        self.average_entry_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_equity = 0.0
        self.previous_total_equity = 0.0
        self.previous_snapshot: MarketSnapshot | None = None
        self.last_institutional_orders: list[Order] = []
        self.noise_generator = PoissonNoiseGenerator(
            bid_lambda=self.config.noise_bid_lambda,
            ask_lambda=self.config.noise_ask_lambda,
            price_scale=self.config.noise_price_scale,
            min_distance=self.config.noise_min_distance,
            min_volume=self.config.noise_min_volume,
            max_volume=self.config.noise_max_volume,
            seed=self.config.random_seed,
        )

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
        self.noise_generator.reset(seed=seed if seed is not None else self.config.random_seed)
        self.current_step = 0
        self.inventory = 0.0
        self.cash = 0.0
        self.average_entry_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_equity = 0.0
        self.previous_total_equity = 0.0
        self.last_institutional_orders = []

        for agent in self.institutional_agents:
            agent.reset()

        for _ in range(self.config.noise_warmup_steps):
            snapshot = self._get_snapshot()
            self._submit_orders(self.noise_generator.generate_orders(snapshot))
            self.matching_engine.advance(1)

        snapshot = self._get_snapshot()
        self._mark_to_market(snapshot.last_price)
        self.previous_snapshot = snapshot
        self.previous_total_equity = self.total_equity
        return snapshot.to_observation_vector(self.config.depth_levels), self._build_info(snapshot)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        current_snapshot = self.previous_snapshot or self._get_snapshot()
        decision = self._decode_action(action)
        if decision.action != "hold":
            order = decision.to_order()
            if order is not None:
                execution_price = max(current_snapshot.last_price, current_snapshot.mid_price, self.config.noise_min_distance)
                self._apply_rl_fill(decision, execution_price)
                self._submit_orders([order])

        institutional_snapshot = self._get_snapshot()
        institutional_orders = self._generate_institutional_orders(institutional_snapshot)
        self.last_institutional_orders = institutional_orders
        self._submit_orders(institutional_orders)

        noise_snapshot = self._get_snapshot()
        self._submit_orders(self.noise_generator.generate_orders(noise_snapshot))
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
        self._mark_to_market(snapshot.last_price)
        equity_delta = self.total_equity - self.previous_total_equity
        inventory_penalty = self.config.reward_inventory_penalty * abs(self.inventory)
        reward = float(equity_delta - inventory_penalty)
        self.previous_total_equity = self.total_equity
        return reward

    def _build_info(self, snapshot: MarketSnapshot) -> dict[str, Any]:
        return {
            "tick": snapshot.tick,
            "last_price": snapshot.last_price,
            "mid_price": snapshot.mid_price,
            "spread": snapshot.spread,
            "imbalance": snapshot.imbalance,
            "inventory": self.inventory,
            "average_entry_price": self.average_entry_price,
            "rl_cash": self.cash,
            "rl_realized_pnl": self.realized_pnl,
            "rl_unrealized_pnl": self.unrealized_pnl,
            "rl_total_equity": self.total_equity,
            "institutional_order_count": len(self.last_institutional_orders),
            "institutional_order_actions": [order.action for order in self.last_institutional_orders],
        }

    def _submit_orders(self, orders: list[Order]) -> None:
        for order in orders:
            self.matching_engine.submit_order(order)

    def _generate_institutional_orders(self, snapshot: MarketSnapshot) -> list[Order]:
        orders: list[Order] = []
        for agent in self.institutional_agents:
            orders.extend(agent.generate_orders(snapshot))
        return orders

    def _apply_rl_fill(self, decision: AgentDecision, execution_price: float) -> None:
        """Approximate market-order fills for RL PnL accounting."""
        volume = decision.volume
        price = execution_price
        transaction_cost = price * volume * (self.config.transaction_cost_bps / 10_000.0)

        if decision.action == "buy":
            self.cash -= price * volume + transaction_cost
            self._apply_position_change(side=1, volume=volume, price=price)
        else:
            self.cash += price * volume - transaction_cost
            self._apply_position_change(side=-1, volume=volume, price=price)

    def _apply_position_change(self, side: int, volume: float, price: float) -> None:
        if side not in (-1, 1):
            raise ValueError("side must be -1 or 1")

        if side == 1:
            if self.inventory >= 0:
                new_inventory = self.inventory + volume
                if new_inventory > 0:
                    self.average_entry_price = (
                        (self.average_entry_price * self.inventory) + (price * volume)
                    ) / new_inventory
                self.inventory = new_inventory
                return

            cover_volume = min(volume, abs(self.inventory))
            self.realized_pnl += (self.average_entry_price - price) * cover_volume
            self.inventory += cover_volume
            remaining_volume = volume - cover_volume
            if self.inventory == 0:
                self.average_entry_price = 0.0
            if remaining_volume > 0:
                self.average_entry_price = price
                self.inventory = remaining_volume
            return

        if self.inventory <= 0:
            new_inventory = abs(self.inventory) + volume
            if new_inventory > 0:
                self.average_entry_price = (
                    (self.average_entry_price * abs(self.inventory)) + (price * volume)
                ) / new_inventory
            self.inventory -= volume
            return

        close_volume = min(volume, self.inventory)
        self.realized_pnl += (price - self.average_entry_price) * close_volume
        self.inventory -= close_volume
        remaining_volume = volume - close_volume
        if self.inventory == 0:
            self.average_entry_price = 0.0
        if remaining_volume > 0:
            self.average_entry_price = price
            self.inventory = -remaining_volume

    def _mark_to_market(self, mark_price: float) -> None:
        if self.inventory > 0:
            self.unrealized_pnl = (mark_price - self.average_entry_price) * self.inventory
        elif self.inventory < 0:
            self.unrealized_pnl = (self.average_entry_price - mark_price) * abs(self.inventory)
        else:
            self.unrealized_pnl = 0.0
        self.total_equity = self.cash + (self.inventory * mark_price)
