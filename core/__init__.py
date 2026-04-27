"""Core interfaces and market environment for the simulator."""

from core.env import MarketEnv, PoissonNoiseGenerator
from core.interfaces import MatchingEngineProtocol, coerce_snapshot
from core.mock_engine import MockMatchingEngine
from core.models import (
    AgentDecision,
    Level,
    MarketPhase,
    MarketSnapshot,
    Order,
    PriceType,
    SimulationConfig,
    TimeInForce,
)
from core.real_engine import RealEngineAdapterConfig, RealMatchingEngineAdapter

__all__ = [
    "AgentDecision",
    "Level",
    "MarketEnv",
    "MarketPhase",
    "MarketSnapshot",
    "MatchingEngineProtocol",
    "MockMatchingEngine",
    "Order",
    "PoissonNoiseGenerator",
    "PriceType",
    "RealEngineAdapterConfig",
    "RealMatchingEngineAdapter",
    "SimulationConfig",
    "TimeInForce",
    "coerce_snapshot",
]
