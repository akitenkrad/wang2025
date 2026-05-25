#!/usr/bin/env python3
"""show_experiment_settings.py — print a results directory's settings.

Pretty-prints `config.json` / `sweep_config.json` / `run_metadata.json` from a
results directory (default `results/latest`) so the experiment parameters and
LLM provenance (provider / model / temperature / seed / cache-hit) are visible.

Usage:
    uv run culture-llm-tools show-experiment-settings
    uv run culture-llm-tools show-experiment-settings --results-dir results/latest
"""

from __future__ import annotations

import argparse
import json
import os


def _show(results_dir: str, name: str) -> bool:
    path = os.path.join(results_dir, name)
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"=== {name} ===")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    print()
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="culture-llm-tools show-experiment-settings")
    parser.add_argument("--results-dir", default="results/latest")
    args = parser.parse_args(argv)

    found = False
    for name in ("config.json", "sweep_config.json", "run_metadata.json"):
        found |= _show(args.results_dir, name)
    if not found:
        print(f"[show-experiment-settings] no settings files in {args.results_dir}")


if __name__ == "__main__":
    main()
