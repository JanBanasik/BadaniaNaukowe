# Current State And Next Steps

## Short Answer

The project is now a solid **mock-backed research harness**.

It is ready for:

- system testing,
- provider testing with `ollama`, `lmstudio`, and `groq`,
- fast-lane ablations with RL, institutional agents, and Poisson noise,
- and structured experiment logging.

It is **not yet** a real-market simulator because the real supervisor LOB adapter is still the main missing dependency.

## What `make ollama` Does

`make ollama` does **not** run a long experiment loop.

It runs:

- one provider-backed swarm demo,
- on one sample `MarketSnapshot`,
- across `AGENT_COUNT` personas,
- then logs the returned decisions and actionable orders.

By default:

- `AGENT_COUNT=5`

So by default `make ollama` makes **5 LLM calls once** and prints/logs the result.

If you want more swarm agents in that demo:

```bash
make ollama AGENT_COUNT=20
```

If you want a real experiment loop instead, use:

```bash
make experiment
```

or for a preset:

```bash
make experiment SCENARIO=full_hybrid_mock
```

## Current Architecture

### Fast Lane

The fast lane currently contains:

- the matching engine interface,
- `MarketEnv`,
- the RL agent,
- the institutional market maker,
- and the Poisson noise process.

Per fast-lane tick, the current execution order is:

1. RL agent action
2. institutional agent action(s)
3. Poisson noise order generation
4. matching engine advance
5. snapshot / reward / logging

### Slow Lane

The slow lane contains:

- retail personas,
- prompt generation,
- LLM providers (`mock`, `ollama`, `lmstudio`, `groq`),
- `SwarmManager`,
- and response validation into normalized `Order` objects.

The swarm does **not** directly receive a stream of raw RL or institutional orders.

Instead, it receives:

- a `MarketSnapshot`,
- plus a `market_news` string,
- then returns retail orders that are injected back into the market.

### Swarm Agent Interaction And Scale

**Do swarm agents talk to each other?**

**No.** In the current implementation, each persona is called **independently**. Every agent gets:

- its own **system prompt** (persona archetype, risk, notes),
- and a **user prompt** built from the **same** `MarketSnapshot` plus the **same** `market_news` string.

There is **no** peer-to-peer channel: one agent does not see another agent’s reply, order, or “sentiment.” So effects like “FOMO trader sees others buying and piles in” are **not** modeled unless you **add** that information explicitly—for example:

- an aggregate summary in the prompt (“last pulse: X% buy / Y% sell”),
- a second round where agents see anonymized counts,
- or a graph/neighborhood model (each agent only sees k neighbors).

That is intentional for now: it keeps the slow lane simple, reproducible, and easy to validate. You can decide later whether to add contagion or social signals.

**Large swarms (e.g. 300+ agents) and asynchronicity**

The swarm layer uses **`asyncio.gather`** so many LLM HTTP calls can run **concurrently** (non-blocking I/O). That is “asynchronous” in the usual asyncio sense.

Scaling to **hundreds of agents** is **not** automatic:

- providers enforce **rate limits** and **timeouts**,
- memory and wall-clock time grow with N,
- unbounded concurrency can overwhelm local Ollama or remote APIs.

For large N you would typically add **bounded concurrency** (e.g. a semaphore), batching, or a local inference stack. None of that is required for the current baseline; the code is fine as-is for small-to-moderate swarm sizes while you decide next steps.

### Experiment Layer

The unified runner in `sim/run_experiment.py` now supports:

- deterministic seeds,
- named scenarios,
- structured run folders,
- PPO checkpoints,
- metrics collection,
- orchestrator logs,
- mock or real engine selection,
- and optional live swarm providers.

## What Is Ready Right Now

### Ready

- mock-backed experiments
- RL environment smoke testing
- PPO initialization and short experiment runs
- local and hosted swarm-provider testing
- institutional-agent integration
- structured logs in `runs/` and `data/runs/`
- scenario-based experiment execution
- integration tests for runner and orchestrator

### Not Ready Yet

- real supervisor LOB integration
- realistic queue/matching microstructure behavior
- publication-grade execution metrics
- large benchmark suites over many seeds and scenarios
- dashboard / UI exploration layer

## Logging Status

The project now logs well enough for research prototyping.

### Provider Demo Logs

Stored in `data/runs/<run_id>/`

These contain:

- provider metadata,
- input snapshot,
- decisions,
- normalized orders,
- and provider errors.

### Experiment Logs

Stored in `runs/<run_id>/`

These contain:

- `config.json`
- `metrics.csv`
- `summary.json`
- `orchestrator_cycles.jsonl`
- `orchestrator_metrics.csv`
- `orchestrator_summary.json`
- checkpoints

## Best Next Steps

### 1. Real Matching Engine Adapter

Highest priority.

Replace the fake backend path with the real supervisor LOB using:

- `core/real_engine.py`
- `matching_engine_backend="real"`
- `real_engine_factory_path`

This is the step that turns the system from a realistic prototype into a real market simulator.

### 2. Better Research Metrics

Expand experiment outputs with:

- spread capture,
- execution quality,
- slippage,
- market impact,
- inventory risk,
- provider latency distributions,
- per-persona action mix.

### 3. More Institutional Diversity

Add at least one more fast-lane deterministic agent, for example:

- passive inventory-aware market maker,
- execution fund,
- trend-following participant.

That will make the fast lane less dependent on one institutional style.

### 4. Scenario Sweeps

Run controlled sweeps over:

- `swarm_update_freq`
- `AGENT_COUNT`
- provider choice
- institutional thresholds
- Poisson noise intensity
- news variations

This is where the first real research results will start to appear.

### 5. UI Later

A Streamlit UI makes sense later, but only after:

- the real engine is wired in,
- metrics are richer,
- and the experiment outputs are stable enough to visualize.

### 6. Optional Later: Inter-Agent / Contagion

Only if your research needs it: extend prompts or add a second phase so agents can react to **aggregate** or **neighbor** behavior (see **Swarm Agent Interaction And Scale** above). The baseline deliberately avoids that until you choose a design.

## Recommended Research Outcomes

The most realistic near-term outcomes are:

1. Compare `RL + noise` vs `RL + noise + institutional` vs `RL + noise + institutional + swarm`.
2. Measure how swarm update frequency changes spread, mid-price drift, and RL equity.
3. Compare `mock`, `ollama`, `lmstudio`, and `groq` by latency, order mix, and downstream market effect.
4. Test whether the institutional market maker dampens or amplifies swarm-driven shocks.
5. Compare different reward designs for the RL agent once the real engine is connected.

## Practical Recommendation

For now:

- use the Poisson/noise + mock engine setup for development and feature testing,
- use `make ollama` / `make lmstudio` for swarm sanity checks,
- use `make experiment` for structured prototype experiments,
- and treat the real LOB adapter as the next critical milestone.
