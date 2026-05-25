#!/usr/bin/env python3
"""show_experiment_settings.py — print a results directory's settings.

Pretty-prints `config.json` (run) or `sweep_config.json` (sweep) plus
`run_metadata.json` from a results directory (default `results/latest`) so the
experiment parameters and LLM provenance (provider / model / temperature / seed /
cache-hit) are visible. `results/latest` is resolved.

Usage:
    uv run culture-llm-tools show-experiment-settings
    uv run culture-llm-tools show-experiment-settings --results-dir results/latest
    uv run culture-llm-tools show-experiment-settings --results-dir results/latest --json

I/O, the run-config table and the LLM-metadata block are delegated to the shared
helper `socsim_tools` (byte-equivalent output). The sweep-config table (with its
nested F×q compound lines) and the `--json` `kind` field are repo-specific and
stay in this module.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from socsim_tools.io import load_run_metadata, resolve_results_dir
from socsim_tools.settings import render_run_config, render_run_metadata

# config key → display label (padded so the colon column aligns).
# render_run_config formats each as f"{label}: {value}", so labels exclude the
# trailing ": " and are space-padded to match the run renderer's alignment.
FIELD_LABELS = {
    "provider": "provider          ",
    "width": "width             ",
    "height": "height            ",
    "features": "features F        ",
    "traits": "traits q          ",
    "events_per_step": "events/step       ",
    "rounds": "rounds            ",
    "snapshot_interval": "snapshot_interval ",
    "seed": "seed (core)       ",
    "llm_temperature": "LLM temperature   ",
    "llm_seed": "LLM seed          ",
    "output_dir": "output_dir        ",
}


def _find_config_file(results_dir: Path) -> tuple[Path, str]:
    """Find config.json (run) or sweep_config.json (sweep)."""
    run_cfg = results_dir / "config.json"
    sweep_cfg = results_dir / "sweep_config.json"
    if run_cfg.exists():
        return run_cfg, "run"
    if sweep_cfg.exists():
        return sweep_cfg, "sweep"
    raise FileNotFoundError(
        f"no settings file in: {results_dir}\n"
        f"  expected: config.json (run) or sweep_config.json (sweep)"
    )


def render_sweep_config(cfg: dict, source: Path) -> str:
    """Render the sweep-config table (repo-specific; nested F×q compound lines)."""
    feats = cfg.get("features", {})
    traits = cfg.get("traits", {})
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("experiment settings (sweep)")
    lines.append("=" * 70)
    lines.append(f"settings file: {source}")
    lines.append("-" * 70)
    lines.append(f"provider          : {cfg.get('provider', '-')}")
    lines.append(f"width             : {cfg.get('width', '-')}")
    lines.append(f"height            : {cfg.get('height', '-')}")
    lines.append(
        f"features F        : {feats.get('min', '-')}..{feats.get('max', '-')} "
        f"step {feats.get('step', '-')}"
    )
    lines.append(
        f"traits q          : {traits.get('min', '-')}..{traits.get('max', '-')} "
        f"step {traits.get('step', '-')}"
    )
    lines.append(f"runs              : {cfg.get('runs', '-')}")
    lines.append(f"rounds            : {cfg.get('rounds', '-')}")
    lines.append(f"events/step       : {cfg.get('events_per_step', '-')}")
    lines.append(f"snapshot_interval : {cfg.get('snapshot_interval', '-')}")
    lines.append(f"seed (base)       : {cfg.get('seed', '-')}")
    lines.append(f"LLM temperature   : {cfg.get('llm_temperature', '-')}")
    lines.append(f"LLM seed          : {cfg.get('llm_seed', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="culture-llm-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        "--results_dir",
        default="results/latest",
        help="results directory (default: results/latest)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of a table.",
    )
    args = parser.parse_args(argv)

    results_dir = resolve_results_dir(args.results_dir)
    if not results_dir.exists():
        print(f"error: directory does not exist: {results_dir}", file=sys.stderr)
        return 1

    try:
        cfg_path, kind = _find_config_file(results_dir)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    with cfg_path.open(encoding="utf-8") as f:
        cfg = json.load(f)
    meta = load_run_metadata(results_dir)

    if args.json:
        payload = {"source": str(cfg_path), "kind": kind, "config": cfg, "run_metadata": meta}
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        if kind == "run":
            print(render_run_config(cfg, cfg_path, FIELD_LABELS))
        else:
            print(render_sweep_config(cfg, cfg_path))
        if meta is not None:
            print(render_run_metadata(meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
