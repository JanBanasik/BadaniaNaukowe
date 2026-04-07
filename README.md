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

## Smoke Tests

Environment and PPO smoke test:

```bash
uv run python tests/smoke_test.py
```

Local swarm provider contract smoke:

```bash
uv run python tests/local_swarm_smoke.py --provider both
```

This second script does not require a real model server. It launches an in-process mock HTTP server that emulates both the OpenAI-compatible and Ollama response shapes so you can verify the client layer and `SwarmManager` parsing path.

Unified experiment-runner integration tests:

```bash
make test
```

## Local Swarm Backends

The swarm layer now supports three providers through the same interface:

- `groq`
- `ollama`
- `openai_compatible`

Recommended local development flow:

1. Use `ollama` or an OpenAI-compatible local server while you iterate on prompts and JSON validation.
2. Keep the same `SwarmManager` code and switch only the client configuration.
3. Move to `groq` later for larger evaluation runs.

### Ollama

Typical defaults:

- endpoint: `http://localhost:11434/api/chat`
- model: `llama3:8b`

Example setup:

```bash
ollama pull llama3:8b
ollama serve
```

### OpenAI-Compatible Local Servers

Examples include LM Studio, vLLM, and `llama.cpp` server mode.

Typical default:

- endpoint: `http://localhost:1234/v1/chat/completions`

If your server exposes a different route, pass the full chat-completions URL in the client config.

### Groq

Set:

```bash
export GROQ_API_KEY=your_key_here
```

The hosted Groq client remains available for the same prompt and validation flow.

## Make Targets

Convenience commands:

```bash
make install
make smoke
make swarm-smoke
make experiment
make test
make lmstudio
make ollama
make groq
```

Useful overrides:

```bash
make lmstudio AGENT_COUNT=8 LMSTUDIO_MODEL="meta-llama-3-8b-instruct"
make ollama AGENT_COUNT=6 OLLAMA_MODEL="llama3:8b"
make groq AGENT_COUNT=10 GROQ_API_KEY=...
make experiment SWARM_PROVIDER=mock NUM_CYCLES=50 SWARM_UPDATE_FREQ=5
make experiment SCENARIO=full_hybrid_mock
```

`make lmstudio`, `make ollama`, and `make groq` run a small real-provider swarm demo against a sample market snapshot and print readable console diagnostics for decisions, actionable orders, and any returned errors.

Each real-provider run also writes structured artifacts to `data/runs/<timestamp_provider>/`, including:

- `metadata.json`
- `snapshot.json`
- `summary.json`
- `decisions.jsonl`
- `orders.jsonl`
- `errors.log`

`make experiment` runs the unified experiment harness and writes full-run artifacts to `runs/<run_id>/`, including:

- `config.json`
- `metrics.csv`
- `summary.json`
- `orchestrator_cycles.jsonl`
- `orchestrator_metrics.csv`
- `orchestrator_summary.json`
- `checkpoints/ppo_initial.zip`
- `checkpoints/ppo_final.zip`

The runner now supports:

- named scenarios from `configs/scenarios/`
- a dedicated institutional module under `agents/institutional.py`
- a richer RL accounting model with cash, realized PnL, unrealized PnL, and total equity
- a real-engine adapter path via `matching_engine_backend="real"` plus `--real-engine-factory-path module:factory`

Example:

```bash
make experiment SCENARIO=rl_noise_only
make experiment SCENARIO=rl_institutional_noise
make experiment SCENARIO=full_hybrid_mock
make experiment SWARM_PROVIDER=mock MATCHING_ENGINE_BACKEND=real REAL_ENGINE_FACTORY_PATH="tests.fake_real_backend:create_backend"
```
