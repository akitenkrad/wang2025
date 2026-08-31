#!/usr/bin/env python3
"""show_experiment_settings.py — print a run directory's settings.

Reads a runvault run's `config.json` (an envelope; the conditions sit under
`parameters`) and prints the parameters that were actually used. Which subcommand
the run was is answered by `run.json`, which also carries the LLM provenance (the
`llm` block: provider / model / temperature); the call counts and the cache-hit
rate are run-scope metrics in `metrics.csv`, and a run that made no LLM calls has
no such rows at all rather than zeroes.

A pre-runvault flat `config.json` / `sweep_config.json` (plus its
`run_metadata.json`) is still read as it stands, including a `results/latest`
symlink.

Run directories can be located with:
    runvault path --experiment culture-llm --latest --subcommand run --standalone
    runvault path --experiment culture-llm --latest --subcommand sweep

Usage:
    culture-llm-tools show-experiment-settings
    culture-llm-tools show-experiment-settings --results-dir "$(runvault path --experiment culture-llm --latest --subcommand run --standalone)"
    culture-llm-tools show-experiment-settings --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from runvault.read import config_parameters, load_run_meta, run_scope_metrics, runvault_path
from socsim_tools.io import load_run_metadata, resolve_results_dir
from socsim_tools.settings import render_run_config, render_run_metadata

EXPERIMENT = "culture-llm"

# config key → display label (padded so the colon column aligns).
# render_run_config formats each as f"{label}: {value}", so labels exclude the
# trailing ": " and are space-padded to match the run renderer's alignment.
FIELD_LABELS = {
    "provider": "provider          ",
    "side": "side              ",
    "condition_id": "condition         ",
    "mock": "mock              ",
    "width": "width             ",
    "height": "height            ",
    "features": "features F        ",
    "traits": "traits q          ",
    "events_per_step": "events/step       ",
    "rounds": "rounds            ",
    "runs": "runs              ",
    "snapshot_interval": "snapshot_interval ",
    "seed": "seed (core)       ",
    "base_seed": "seed (base)       ",
    "llm_temperature": "LLM temperature   ",
    "llm_seed": "LLM seed          ",
    "llm_cache_path": "LLM cache         ",
    "output_dir": "output_dir        ",
}


def render_sweep_config(cfg: dict, source: Path) -> str:
    """Render the sweep-parent table (repo-specific; the F×q grid definition).

    A legacy `sweep_config.json` nested the ranges under `features` / `traits`;
    runvault's conditions are flat (`features_min` and friends). Both are read.
    """
    feats = cfg.get("features", {})
    traits = cfg.get("traits", {})
    f_min = cfg.get("features_min", feats.get("min", "-") if isinstance(feats, dict) else "-")
    f_max = cfg.get("features_max", feats.get("max", "-") if isinstance(feats, dict) else "-")
    f_step = cfg.get("features_step", feats.get("step", "-") if isinstance(feats, dict) else "-")
    t_min = cfg.get("traits_min", traits.get("min", "-") if isinstance(traits, dict) else "-")
    t_max = cfg.get("traits_max", traits.get("max", "-") if isinstance(traits, dict) else "-")
    t_step = cfg.get("traits_step", traits.get("step", "-") if isinstance(traits, dict) else "-")
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("experiment settings (sweep)")
    lines.append("=" * 70)
    lines.append(f"settings file: {source}")
    lines.append("-" * 70)
    lines.append(f"provider          : {cfg.get('provider', '-')}")
    lines.append(f"width             : {cfg.get('width', '-')}")
    lines.append(f"height            : {cfg.get('height', '-')}")
    lines.append(f"features F        : {f_min}..{f_max} step {f_step}")
    lines.append(f"traits q          : {t_min}..{t_max} step {t_step}")
    lines.append(f"runs              : {cfg.get('runs', '-')}")
    lines.append(f"rounds            : {cfg.get('rounds', '-')}")
    lines.append(f"events/step       : {cfg.get('events_per_step', '-')}")
    lines.append(f"snapshot_interval : {cfg.get('snapshot_interval', '-')}")
    lines.append(f"seed (base)       : {cfg.get('base_seed', cfg.get('seed', '-'))}")
    lines.append(f"LLM temperature   : {cfg.get('llm_temperature', '-')}")
    lines.append(f"LLM seed          : {cfg.get('llm_seed', '-')}")
    lines.append("=" * 70)
    return "\n".join(lines)


def render_llm_block(meta: dict, scoped: dict) -> str:
    """The run's LLM provenance: the `llm` block plus the call-count metrics."""
    llm = meta.get("llm")
    if llm is None:
        return (
            "-" * 70
            + "\nLLM: この run は LLM を使っていません (呼び出し 0)．\n"
            + "-" * 70
        )
    lines = ["-" * 70, "LLM provenance (run.json の llm ブロック)", "-" * 70]
    lines.append(f"provider          : {llm.get('provider', '-')}")
    lines.append(f"model             : {llm.get('model_snapshot', '-')}")
    lines.append(f"temperature       : {llm.get('temperature', '-')}")
    if "llm_calls" in scoped:
        lines.append(f"calls             : {scoped['llm_calls']:.0f}")
        lines.append(f"cache hits        : {scoped['llm_cache_hits']:.0f}")
        lines.append(f"cache-hit rate    : {scoped['llm_cache_hit_rate'] * 100:.1f}%")
    else:
        # 率は分母が 0 のとき «0» ではなく «定義できない» ので，行そのものが無い．
        lines.append("calls             : 0 (cache-hit rate は定義されない)")
    lines.append("=" * 70)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="culture-llm-tools show-experiment-settings",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", "--results_dir", default=None,
        help="run ディレクトリ (未指定時は runvault に最新の run を聞く)",
    )
    parser.add_argument("--results-root", "--results_root", default="results")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table.")
    args = parser.parse_args(argv)

    if args.results_dir is None:
        results_dir = Path(
            runvault_path(EXPERIMENT, args.results_root, subcommand="run", standalone=True)
        )
    else:
        results_dir = resolve_results_dir(args.results_dir)
    if not results_dir.exists():
        print(f"error: directory does not exist: {results_dir}", file=sys.stderr)
        return 1

    cfg = config_parameters(results_dir, required=False)
    meta = load_run_meta(results_dir, required=False)
    source = results_dir / "config.json"

    if cfg is None:
        # legacy sweep: the conditions lived in sweep_config.json
        sweep_cfg = results_dir / "sweep_config.json"
        if not sweep_cfg.exists():
            print(
                f"error: no settings file in: {results_dir}\n"
                f"  expected: config.json (runvault envelope / legacy flat) "
                f"or sweep_config.json (legacy sweep)",
                file=sys.stderr,
            )
            return 1
        with sweep_cfg.open(encoding="utf-8") as f:
            cfg = json.load(f)
        kind, source = "sweep", sweep_cfg
    elif meta is not None:
        kind = str(meta.get("subcommand", "run"))
    else:
        # legacy flat config.json carried "command"
        kind = "sweep" if cfg.get("command") == "sweep" else "run"

    if args.json:
        payload = {"source": str(source), "kind": kind, "config": cfg}
        if meta is not None:
            payload["run"] = {
                "run_uid": meta.get("run_uid"),
                "subcommand": meta.get("subcommand"),
                "llm": meta.get("llm"),
                "rng": meta.get("rng"),
                "lineage": meta.get("lineage"),
            }
            payload["llm_metrics"] = {
                k: v for k, v in run_scope_metrics(results_dir).items() if k.startswith("llm_")
            }
        else:
            payload["run_metadata"] = load_run_metadata(results_dir)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if kind == "sweep":
        print(render_sweep_config(cfg, source))
    else:
        # 条件は run / sweep-point / reproduce-condition / compare-side / mock-smoke で
        # 少しずつ違う．持っていない欄を「-」で並べても読みにくいだけなので，実際に
        # 記録されている欄だけを出す (`llm_cache_path` は古典経路では null)．
        labels = {
            key: label
            for key, label in FIELD_LABELS.items()
            if cfg.get(key) is not None
        }
        print(render_run_config(cfg, source, labels))
    if meta is not None:
        print(render_llm_block(meta, run_scope_metrics(results_dir)))
    else:
        legacy_meta = load_run_metadata(results_dir)
        if legacy_meta is not None:
            print(render_run_metadata(legacy_meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
