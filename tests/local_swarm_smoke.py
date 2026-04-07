"""Runnable smoke test for local swarm backends."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Literal

from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.models import Level, MarketSnapshot
from swarm import (
    OllamaClientConfig,
    OpenAICompatibleClientConfig,
    SwarmManager,
    create_llm_client,
    generate_personas,
)

LocalProvider = Literal["openai_compatible", "ollama", "both"]


def build_sample_snapshot() -> MarketSnapshot:
    """Construct a deterministic market snapshot for local swarm testing."""
    return MarketSnapshot(
        bids=[
            Level(price=99.95, volume=120.0),
            Level(price=99.94, volume=110.0),
            Level(price=99.93, volume=105.0),
        ],
        asks=[
            Level(price=100.05, volume=118.0),
            Level(price=100.06, volume=108.0),
            Level(price=100.07, volume=101.0),
        ],
        last_price=100.00,
        mid_price=100.00,
        imbalance=0.02,
        spread=0.10,
        tick=42,
    )


async def openai_handler(request: web.Request) -> web.Response:
    """Serve an OpenAI-compatible JSON completion payload."""
    state = request.app["state"]
    state["openai_calls"] += 1
    call_index = state["openai_calls"]
    payload = {
        "action": "buy" if call_index % 2 else "sell",
        "price_type": "market",
        "volume": 5.0 + call_index,
        "limit_price": None,
        "confidence": 0.75,
        "rationale": f"mock-openai-call-{call_index}",
    }
    return web.json_response({"choices": [{"message": {"content": json.dumps(payload)}}]})


async def ollama_handler(request: web.Request) -> web.Response:
    """Serve an Ollama-style chat payload with JSON content."""
    state = request.app["state"]
    state["ollama_calls"] += 1
    call_index = state["ollama_calls"]
    payload = {
        "action": "sell" if call_index % 2 else "buy",
        "price_type": "market",
        "volume": 3.0 + call_index,
        "limit_price": None,
        "confidence": 0.65,
        "rationale": f"mock-ollama-call-{call_index}",
    }
    return web.json_response({"message": {"content": json.dumps(payload)}})


async def start_mock_server() -> tuple[web.AppRunner, int]:
    """Start a temporary local server implementing both local provider APIs."""
    app = web.Application()
    app["state"] = {"openai_calls": 0, "ollama_calls": 0}
    app.router.add_post("/v1/chat/completions", openai_handler)
    app.router.add_post("/api/chat", ollama_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()
    sockets = getattr(site, "_server").sockets
    port = int(sockets[0].getsockname()[1])
    return runner, port


async def run_provider_smoke(provider: Literal["openai_compatible", "ollama"], port: int) -> None:
    """Exercise one local provider path through SwarmManager."""
    if provider == "openai_compatible":
        config = OpenAICompatibleClientConfig(
            base_url=f"http://127.0.0.1:{port}/v1/chat/completions",
            model_name="mock-openai-compatible",
            timeout_seconds=5.0,
        )
    else:
        config = OllamaClientConfig(
            base_url=f"http://127.0.0.1:{port}/api/chat",
            model_name="mock-ollama",
            timeout_seconds=5.0,
        )

    client = create_llm_client(provider=provider, config=config)
    personas = generate_personas(3)
    manager = SwarmManager(client=client, personas=personas)
    snapshot = build_sample_snapshot()
    orders = await manager.generate_orders(snapshot=snapshot, market_news="Retail traders react to a CPI surprise.")

    assert len(orders) == len(personas)
    assert all(order.price_type == "market" for order in orders)
    assert all(order.agent_id is not None for order in orders)

    print(f"{provider} smoke")
    print(f"  decisions/orders: {len(orders)}")
    print(f"  sample order: {orders[0].model_dump(mode='json')}")

    await client.close()


async def main(provider: LocalProvider) -> None:
    runner, port = await start_mock_server()
    try:
        if provider == "both":
            await run_provider_smoke("openai_compatible", port)
            await run_provider_smoke("ollama", port)
        else:
            await run_provider_smoke(provider, port)
        print("Local swarm smoke test completed successfully.")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke test for local swarm providers.")
    parser.add_argument(
        "--provider",
        choices=["openai_compatible", "ollama", "both"],
        default="both",
        help="Which local provider shape to validate.",
    )
    args = parser.parse_args()
    asyncio.run(main(provider=args.provider))
