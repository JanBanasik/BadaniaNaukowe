"""Persona generators for retail-trader swarm members."""

from __future__ import annotations

from itertools import cycle

from swarm.models import RetailPersona

BASE_PERSONAS = [
    RetailPersona(
        persona_id="template_fomo",
        archetype="fomo_driven",
        risk_appetite=0.85,
        holding_horizon="intraday",
        capital_limit=7_500.0,
        style_notes=[
            "Chases momentum and sharp breakouts.",
            "Overweights recent price action.",
        ],
    ),
    RetailPersona(
        persona_id="template_panicker",
        archetype="panicker",
        risk_appetite=0.25,
        holding_horizon="intraday",
        capital_limit=5_000.0,
        style_notes=[
            "Quickly de-risks during volatility spikes.",
            "Sensitive to negative market news.",
        ],
    ),
    RetailPersona(
        persona_id="template_value",
        archetype="value_investor",
        risk_appetite=0.45,
        holding_horizon="swing",
        capital_limit=12_500.0,
        style_notes=[
            "Prefers mean reversion and perceived discounts.",
            "Less reactive to short-term noise.",
        ],
    ),
]


def generate_personas(count: int) -> list[RetailPersona]:
    """Create a deterministic mix of retail personas for the swarm."""
    if count <= 0:
        return []

    personas: list[RetailPersona] = []
    for index, template in zip(range(count), cycle(BASE_PERSONAS)):
        personas.append(
            template.model_copy(
                update={
                    "persona_id": f"{template.archetype}_{index:03d}",
                }
            )
        )
    return personas
