#!/usr/bin/env python3
"""Summarize a single experiment run under runs/<run_id>/."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def analyze_run(run_dir: Path, write_report: bool) -> int:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.is_file():
        print(f"error: missing {metrics_path}", file=sys.stderr)
        return 1

    import pandas as pd

    df = pd.read_csv(metrics_path)
    config = _load_json(run_dir / "config.json")
    summary = _load_json(run_dir / "summary.json")

    lines: list[str] = []
    lines.append(f"Run directory: {run_dir.resolve()}")
    lines.append(f"Rows (cycles): {len(df)}")
    if config:
        lines.append(
            f"Config: scenario={config.get('scenario_name')!r} "
            f"provider={config.get('swarm_provider')!r} "
            f"backend={config.get('matching_engine_backend')!r}"
        )
    if summary:
        lines.append(f"Summary keys: {', '.join(sorted(summary.keys()))}")

    numeric_cols = [
        c
        for c in df.columns
        if c not in ("institutional_order_actions",)
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    lines.append("")
    lines.append("Per-column summary (numeric):")
    desc = df[numeric_cols].describe().T
    lines.append(desc.to_string())

    if "cycle" in df.columns:
        lines.append("")
        lines.append("First / last cycle snapshot:")
        first = df.iloc[0]
        last = df.iloc[-1]

        def _pick(series: object, *names: str) -> object:
            s = series  # pandas Series
            for n in names:
                if n in s.index:
                    val = s[n]
                    if val == val:  # not NaN
                        return val
            return None

        for label, row in (("first", first), ("last", last)):
            chunks = [
                f"  {label}: mid={_pick(row, 'mid_price')} spread={_pick(row, 'spread')}",
                f"equity={_pick(row, 'rl_total_equity', 'rl_agent_pnl')}",
            ]
            if "rl_realized_pnl" in df.columns:
                chunks.append(f"realized_pnl={_pick(row, 'rl_realized_pnl')}")
            if "rl_unrealized_pnl" in df.columns:
                chunks.append(f"unrealized_pnl={_pick(row, 'rl_unrealized_pnl')}")
            lines.append(" ".join(chunks))

    text = "\n".join(lines) + "\n"
    print(text, end="")

    if write_report:
        out = run_dir / "analysis_report.txt"
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out}", file=sys.stderr)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze runs/<run_id>/metrics.csv and related artifacts.")
    parser.add_argument(
        "run_id",
        nargs="?",
        default=None,
        help="Run id (directory name under runs/)",
    )
    parser.add_argument(
        "--run-id",
        dest="run_id_flag",
        default=None,
        help="Alternative to positional run_id",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("runs"),
        help="Parent directory containing run folders (default: runs)",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Write analysis_report.txt inside the run directory",
    )
    args = parser.parse_args()
    run_id = args.run_id_flag or args.run_id
    if not run_id:
        parser.error("run_id is required (positional or --run-id)")

    run_dir = args.runs_dir / run_id
    if not run_dir.is_dir():
        print(f"error: not a directory: {run_dir}", file=sys.stderr)
        return 1

    return analyze_run(run_dir, write_report=args.write_report)


if __name__ == "__main__":
    raise SystemExit(main())
