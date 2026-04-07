"""Lightweight structured logging for experiment and demo runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Persist structured run artifacts under the data directory."""

    def __init__(
        self,
        provider: str,
        model_name: str,
        base_dir: str | Path = "data/runs",
        run_label: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.base_dir = Path(base_dir)
        if run_id is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            label = _slugify(run_label or provider)
            run_id = f"{timestamp}_{label}"
        self.run_id = run_id
        self.run_dir = self.base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_metadata(self, metadata: dict[str, Any]) -> Path:
        """Write run metadata to disk."""
        payload = {
            "run_id": self.run_id,
            "provider": self.provider,
            "model_name": self.model_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **metadata,
        }
        return self._write_json("metadata.json", payload)

    def write_snapshot(self, snapshot: Any) -> Path:
        """Persist a snapshot payload."""
        payload = snapshot.model_dump(mode="json") if hasattr(snapshot, "model_dump") else snapshot
        return self._write_json("snapshot.json", payload)

    def write_summary(self, summary: dict[str, Any]) -> Path:
        """Persist a single run summary."""
        return self._write_json("summary.json", summary)

    def write_json(self, filename: str, payload: dict[str, Any]) -> Path:
        """Persist an arbitrary JSON payload under the run directory."""
        return self._write_json(filename, payload)

    def write_records_jsonl(self, filename: str, records: list[dict[str, Any]]) -> Path:
        """Write a list of dictionaries as JSONL."""
        path = self.run_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True))
                handle.write("\n")
        return path

    def append_record_jsonl(self, filename: str, record: dict[str, Any]) -> Path:
        """Append one dictionary to a JSONL file."""
        path = self.run_dir / filename
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")
        return path

    def write_text_lines(self, filename: str, lines: list[str]) -> Path:
        """Write newline-delimited text output."""
        path = self.run_dir / filename
        content = "\n".join(lines)
        if content:
            content += "\n"
        path.write_text(content, encoding="utf-8")
        return path

    def write_csv_text(self, filename: str, content: str) -> Path:
        """Write pre-rendered CSV content to disk."""
        path = self.run_dir / filename
        path.write_text(content, encoding="utf-8")
        return path

    def _write_json(self, filename: str, payload: dict[str, Any]) -> Path:
        path = self.run_dir / filename
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return path


def _slugify(value: str) -> str:
    """Convert free-form labels into stable directory-safe strings."""
    cleaned = "".join(char.lower() if char.isalnum() else "_" for char in value)
    collapsed = "_".join(part for part in cleaned.split("_") if part)
    return collapsed or "run"
