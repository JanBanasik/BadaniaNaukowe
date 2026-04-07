"""Scenario loading helpers for named experiment presets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_SCENARIO_DIR = Path("configs/scenarios")


def list_scenarios(base_dir: str | Path = DEFAULT_SCENARIO_DIR) -> list[str]:
    """List available scenario names from the configuration directory."""
    directory = Path(base_dir)
    if not directory.exists():
        return []
    return sorted(path.stem for path in directory.glob("*.json"))


def load_scenario(name: str, base_dir: str | Path = DEFAULT_SCENARIO_DIR) -> dict[str, Any]:
    """Load one scenario preset from JSON."""
    path = Path(base_dir) / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Scenario '{name}' was not found at {path}.")
    return json.loads(path.read_text(encoding="utf-8"))
