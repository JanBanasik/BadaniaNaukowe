# Hybrid GABM-RL Financial Market Simulator

Baseline scaffold for a hybrid market simulator that combines:

- a fast limit order book lane wrapped as a `gymnasium` environment,
- a slow asynchronous LLM swarm lane for retail trader behavior,
- and an event-driven orchestrator that synchronizes both clocks.

## Layout

- `core/`: shared market models, matching-engine interfaces, and Gymnasium wrapper.
- `agents/`: PPO configuration and training entrypoints.
- `swarm/`: Groq client, persona templates, prompts, and async swarm manager.
- `sim/`: two-speed orchestrator and simulation control flow.
- `data/`: output location for logs and performance metrics.
- `docs/`: architecture and baseline documentation.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python main.py
```

The scaffold is intentionally interface-first. You still need to plug in a concrete matching engine implementation from the supervisor-provided LOB code before running training or full simulations.
