"""Pydantic models used by the slow-lane LLM swarm."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.models import AgentDecision, MarketSnapshot

PersonaArchetype = Literal["fomo_driven", "panicker", "value_investor"]
HoldingHorizon = Literal["intraday", "swing", "position"]


class RetailPersona(BaseModel):
    """Describes the behavioral style of one LLM retail trader."""

    model_config = ConfigDict(extra="forbid")

    persona_id: str
    archetype: PersonaArchetype
    risk_appetite: float = Field(ge=0.0, le=1.0)
    holding_horizon: HoldingHorizon = "intraday"
    capital_limit: float = Field(default=10_000.0, gt=0.0)
    style_notes: list[str] = Field(default_factory=list)


class SwarmRequest(BaseModel):
    """Inputs sent to each LLM persona."""

    model_config = ConfigDict(extra="forbid")

    persona: RetailPersona
    snapshot: MarketSnapshot
    market_news: str


class LLMOrderResponse(BaseModel):
    """Strict JSON schema expected back from the LLM layer."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["buy", "sell", "hold"]
    price_type: Literal["market", "limit"] = "market"
    volume: float = Field(default=0.0, ge=0.0)
    limit_price: float | None = Field(default=None, ge=0.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    rationale: str | None = None

    @model_validator(mode="after")
    def validate_consistency(self) -> "LLMOrderResponse":
        if self.action == "hold":
            if self.volume != 0.0:
                raise ValueError("Hold responses must use volume=0.")
            if self.limit_price is not None:
                raise ValueError("Hold responses cannot include limit_price.")
            return self

        if self.volume <= 0.0:
            raise ValueError("Trade responses must have positive volume.")
        if self.price_type == "limit" and self.limit_price is None:
            raise ValueError("Limit responses must include limit_price.")
        if self.price_type == "market" and self.limit_price is not None:
            raise ValueError("Market responses cannot include limit_price.")
        return self

    def to_agent_decision(self, persona: RetailPersona) -> AgentDecision:
        """Attach persona metadata to the validated response."""
        return AgentDecision(
            action=self.action,
            price_type=self.price_type,
            volume=self.volume,
            limit_price=self.limit_price,
            confidence=self.confidence,
            rationale=self.rationale,
            agent_id=persona.persona_id,
            source="swarm",
        )
