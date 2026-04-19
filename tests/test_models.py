"""Tests for extended Order, AgentDecision, MarketSnapshot, SimulationConfig."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from core.models import AgentDecision, MarketSnapshot, Order, SimulationConfig


class OrderPriceTypeTests(unittest.TestCase):
    def test_market_defaults(self) -> None:
        order = Order(action="buy", volume=1.0)
        self.assertEqual(order.price_type, "market")
        self.assertIsNone(order.limit_price)
        self.assertIsNone(order.stop_price)
        self.assertEqual(order.time_in_force, "GTC")
        self.assertIsNone(order.expiry_tick)

    def test_limit_requires_limit_price(self) -> None:
        Order(action="buy", price_type="limit", volume=1.0, limit_price=100.0)
        with self.assertRaisesRegex(ValidationError, "limit_price is required"):
            Order(action="buy", price_type="limit", volume=1.0)

    def test_market_rejects_limit_price(self) -> None:
        with self.assertRaisesRegex(ValidationError, "limit_price must be omitted"):
            Order(action="buy", price_type="market", volume=1.0, limit_price=100.0)

    def test_market_rejects_stop_price(self) -> None:
        with self.assertRaisesRegex(ValidationError, "stop_price must be omitted"):
            Order(action="buy", price_type="market", volume=1.0, stop_price=95.0)

    def test_stop_market_requires_stop_price(self) -> None:
        Order(action="buy", price_type="stop_market", volume=1.0, stop_price=105.0)
        with self.assertRaisesRegex(ValidationError, "stop_price is required"):
            Order(action="buy", price_type="stop_market", volume=1.0)

    def test_stop_market_rejects_limit_price(self) -> None:
        with self.assertRaisesRegex(ValidationError, "limit_price must be omitted"):
            Order(
                action="buy",
                price_type="stop_market",
                volume=1.0,
                stop_price=105.0,
                limit_price=110.0,
            )

    def test_stop_limit_requires_both_prices(self) -> None:
        Order(
            action="sell",
            price_type="stop_limit",
            volume=1.0,
            stop_price=95.0,
            limit_price=94.0,
        )
        with self.assertRaisesRegex(ValidationError, "limit_price is required"):
            Order(action="sell", price_type="stop_limit", volume=1.0, stop_price=95.0)
        with self.assertRaisesRegex(ValidationError, "stop_price is required"):
            Order(action="sell", price_type="stop_limit", volume=1.0, limit_price=94.0)


class OrderTimeInForceTests(unittest.TestCase):
    def test_gtd_requires_expiry_tick(self) -> None:
        Order(action="buy", volume=1.0, time_in_force="GTD", expiry_tick=500)
        with self.assertRaisesRegex(ValidationError, "expiry_tick is required"):
            Order(action="buy", volume=1.0, time_in_force="GTD")

    def test_non_gtd_rejects_expiry_tick(self) -> None:
        for tif in ("GTC", "IOC", "FOK"):
            with self.assertRaisesRegex(ValidationError, "expiry_tick must be omitted"):
                Order(action="buy", volume=1.0, time_in_force=tif, expiry_tick=500)


class AgentDecisionTests(unittest.TestCase):
    def test_hold_rejects_extras(self) -> None:
        AgentDecision(action="hold", volume=0.0)
        with self.assertRaisesRegex(ValidationError, "Hold decisions cannot set stop_price"):
            AgentDecision(action="hold", volume=0.0, stop_price=100.0)
        with self.assertRaisesRegex(ValidationError, "Hold decisions cannot set expiry_tick"):
            AgentDecision(action="hold", volume=0.0, expiry_tick=500)

    def test_to_order_propagates_new_fields(self) -> None:
        decision = AgentDecision(
            action="buy",
            price_type="stop_limit",
            volume=2.5,
            stop_price=105.0,
            limit_price=106.0,
            time_in_force="GTD",
            expiry_tick=1000,
            source="rl",
        )
        order = decision.to_order()
        self.assertIsNotNone(order)
        assert order is not None
        self.assertEqual(order.price_type, "stop_limit")
        self.assertEqual(order.stop_price, 105.0)
        self.assertEqual(order.limit_price, 106.0)
        self.assertEqual(order.time_in_force, "GTD")
        self.assertEqual(order.expiry_tick, 1000)

    def test_to_order_returns_none_for_hold(self) -> None:
        self.assertIsNone(AgentDecision(action="hold", volume=0.0).to_order())

    def test_trade_decision_applies_price_rules(self) -> None:
        with self.assertRaisesRegex(ValidationError, "stop_price is required"):
            AgentDecision(action="buy", price_type="stop_market", volume=1.0)


class MarketSnapshotTests(unittest.TestCase):
    def test_phase_defaults_to_continuous(self) -> None:
        snapshot = MarketSnapshot(
            last_price=100.0, mid_price=100.0, imbalance=0.0, spread=0.2
        )
        self.assertEqual(snapshot.phase, "continuous")

    def test_phase_accepts_all_literals(self) -> None:
        for phase in ("pre_open", "opening_auction", "continuous", "closing_auction", "closed"):
            snapshot = MarketSnapshot(
                last_price=100.0,
                mid_price=100.0,
                imbalance=0.0,
                spread=0.2,
                phase=phase,
            )
            self.assertEqual(snapshot.phase, phase)

    def test_invalid_phase_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            MarketSnapshot(
                last_price=100.0,
                mid_price=100.0,
                imbalance=0.0,
                spread=0.2,
                phase="halted",
            )


class SimulationConfigTests(unittest.TestCase):
    def test_default_clock_fields(self) -> None:
        config = SimulationConfig()
        self.assertEqual(config.ticks_per_session, 1000)
        self.assertEqual(config.sessions_per_year, 252)
        self.assertEqual(config.cycles_per_session, 40)
        self.assertEqual(config.opening_auction_ticks, 20)
        self.assertEqual(config.closing_auction_ticks, 20)
        self.assertEqual(config.accumulation_days, 5)
        self.assertEqual(config.accumulation_volume, 1.0)
        self.assertEqual(config.commission_bps, 0.0)
        self.assertEqual(config.commission_min_per_trade, 0.0)
        self.assertIsNone(config.session_schedule)

    def test_auction_windows_must_fit_session(self) -> None:
        with self.assertRaisesRegex(ValidationError, "must be <"):
            SimulationConfig(
                ticks_per_session=100,
                opening_auction_ticks=60,
                closing_auction_ticks=60,
            )

    def test_custom_session_schedule(self) -> None:
        config = SimulationConfig(
            session_schedule=[("opening_auction", 10), ("continuous", 80), ("closing_auction", 10)]
        )
        self.assertEqual(
            config.session_schedule,
            [("opening_auction", 10), ("continuous", 80), ("closing_auction", 10)],
        )


if __name__ == "__main__":
    unittest.main()
