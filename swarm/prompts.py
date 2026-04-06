"""Prompt builders for retail-trader swarm interactions."""

from __future__ import annotations

import json

from core.models import MarketSnapshot
from swarm.models import RetailPersona


def build_system_prompt(persona: RetailPersona) -> str:
    """Describe one retail persona and the required JSON response schema."""
    notes = "\n".join(f"- {note}" for note in persona.style_notes)
    return f"""
You are simulating one retail trader in a financial market.

Persona ID: {persona.persona_id}
Archetype: {persona.archetype}
Risk appetite: {persona.risk_appetite:.2f}
Holding horizon: {persona.holding_horizon}
Capital limit: {persona.capital_limit:.2f}
Behavior notes:
{notes}

Respond with JSON only using this schema:
{{
  "action": "buy" | "sell" | "hold",
  "price_type": "market" | "limit",
  "volume": float,
  "limit_price": float | null,
  "confidence": float | null,
  "rationale": "short explanation"
}}

Rules:
- Use volume=0 and limit_price=null for hold.
- Use positive volume for buy and sell.
- Use limit_price only if price_type is limit.
""".strip()


def build_user_prompt(snapshot: MarketSnapshot, market_news: str) -> str:
    """Format the market context for one LLM call."""
    payload = {
        "market_news": market_news,
        "snapshot": snapshot.model_dump(mode="json"),
    }
    return "Decide what to do next based on the following market context:\n" + json.dumps(
        payload,
        indent=2,
    )
