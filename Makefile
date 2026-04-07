.PHONY: help install smoke swarm-smoke experiment analyze test lmstudio ollama groq

# Optional: make analyze RUN_ID=<id> OR make analyze <run_id> (see help)
RUN_ID ?=

UV ?= uv
PYTHON ?= python

AGENT_COUNT ?= 5
NEWS ?= Inflation data slightly beats expectations while equity futures stay range-bound.
NUM_CYCLES ?= 20
SWARM_UPDATE_FREQ ?= 5
SWARM_PROVIDER ?= mock
MATCHING_ENGINE_BACKEND ?= mock
SCENARIO ?=
REAL_ENGINE_FACTORY_PATH ?=

OLLAMA_BASE_URL ?= http://localhost:11434/api/chat
OLLAMA_MODEL ?= llama3:8b

LMSTUDIO_BASE_URL ?= http://localhost:1234/v1/chat/completions
LMSTUDIO_MODEL ?= local-model

GROQ_BASE_URL ?= https://api.groq.com/openai/v1/chat/completions
GROQ_MODEL ?= llama3-8b-8192

help:
	@echo "Available targets:"
	@echo "  make install        - create uv environment and install project"
	@echo "  make smoke          - run MarketEnv + PPO smoke test"
	@echo "  make swarm-smoke    - run local provider contract smoke"
	@echo "  make experiment     - run the unified experiment harness"
	@echo "  make analyze        - summarize runs/<run_id>/ (RUN_ID=... or make analyze <run_id>)"
	@echo "  make test           - run integration tests"
	@echo "  make lmstudio       - run real swarm demo against LM Studio/OpenAI-compatible server"
	@echo "  make ollama         - run real swarm demo against Ollama"
	@echo "  make groq           - run real swarm demo against Groq"
	@echo ""
	@echo "Optional variables:"
	@echo "  AGENT_COUNT=<n>"
	@echo "  NUM_CYCLES=<n>"
	@echo "  SWARM_UPDATE_FREQ=<n>"
	@echo "  SWARM_PROVIDER=mock|lmstudio|ollama|groq"
	@echo "  MATCHING_ENGINE_BACKEND=mock|real"
	@echo "  SCENARIO=<scenario_name>"
	@echo "  REAL_ENGINE_FACTORY_PATH=module:factory"
	@echo "  RUN_ID=<run_id>     (for make analyze)"
	@echo "  NEWS='custom headline'"
	@echo "  OLLAMA_BASE_URL / OLLAMA_MODEL"
	@echo "  LMSTUDIO_BASE_URL / LMSTUDIO_MODEL"
	@echo "  GROQ_BASE_URL / GROQ_MODEL / GROQ_API_KEY"

install:
	$(UV) venv
	$(UV) pip install -e .

smoke:
	$(UV) run $(PYTHON) tests/smoke_test.py

swarm-smoke:
	$(UV) run $(PYTHON) tests/local_swarm_smoke.py --provider both

experiment:
	$(UV) run $(PYTHON) sim/run_experiment.py $(if $(SCENARIO),--scenario "$(SCENARIO)",) --matching-engine-backend "$(MATCHING_ENGINE_BACKEND)" --swarm-provider "$(SWARM_PROVIDER)" --num-cycles "$(NUM_CYCLES)" --swarm-update-freq "$(SWARM_UPDATE_FREQ)" --agent-count "$(AGENT_COUNT)" --market-news "$(NEWS)" $(if $(REAL_ENGINE_FACTORY_PATH),--real-engine-factory-path "$(REAL_ENGINE_FACTORY_PATH)",)

# Positional id: `make analyze 20260407_123456` (requires catch-all `%` below).
analyze:
	@set -e; \
	if [ -n "$(RUN_ID)" ]; then \
		$(UV) run $(PYTHON) scripts/analyze_run.py "$(RUN_ID)"; \
	elif [ -n "$(word 2,$(MAKECMDGOALS))" ]; then \
		$(UV) run $(PYTHON) scripts/analyze_run.py "$(word 2,$(MAKECMDGOALS))"; \
	else \
		echo "Usage: make analyze RUN_ID=<run_id>  OR  make analyze <run_id>"; \
		exit 1; \
	fi

%:
	@:

test:
	$(UV) run $(PYTHON) -m unittest tests.test_run_experiment_smoke tests.test_orchestrator_logging

lmstudio:
	LMSTUDIO_BASE_URL="$(LMSTUDIO_BASE_URL)" LMSTUDIO_MODEL="$(LMSTUDIO_MODEL)" \
	$(UV) run $(PYTHON) tests/live_swarm_demo.py --provider lmstudio --agent-count "$(AGENT_COUNT)" --news "$(NEWS)"

ollama:
	OLLAMA_BASE_URL="$(OLLAMA_BASE_URL)" OLLAMA_MODEL="$(OLLAMA_MODEL)" \
	$(UV) run $(PYTHON) tests/live_swarm_demo.py --provider ollama --agent-count "$(AGENT_COUNT)" --news "$(NEWS)"

groq:
	GROQ_BASE_URL="$(GROQ_BASE_URL)" GROQ_MODEL="$(GROQ_MODEL)" \
	$(UV) run $(PYTHON) tests/live_swarm_demo.py --provider groq --agent-count "$(AGENT_COUNT)" --news "$(NEWS)"
