#!/usr/bin/env python3
"""compare_report.py — classical vs LLM quantitative comparison figure.

Reads a runvault `compare` parent (written by `culture-llm compare`) and renders
a grouped-bar comparison of the headline metrics (n_stable_regions, LC, GP/N,
final_round), plus prints the deltas table.

The two sides are two child runs (`subcommand=compare-side`): each is a whole
simulation of the same board with the same seed, differing only in the
interaction mechanism, so each has its own `metrics.csv` and its own `llm` block.
The differences between them are a cross-condition aggregate and live once, on
the parent, as `scope=sweep` metrics — they are not repeated here.

The classical side is the deterministic Axelrod baseline (0 LLM calls); the LLM
side is the YuLan-OneSim variant (mock or live).

Usage:
    uv run culture-llm-tools compare-report
    uv run culture-llm-tools compare-report --compare-dir "$(runvault path --experiment culture-llm --latest --subcommand compare)"
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from runvault.read import (
    config_parameters,
    events_table,
    figures_dir,
    run_scope_metrics,
    runvault_path,
    sweep_children,
)

EXPERIMENT = "culture-llm"

COLOR_BG = "#FAFAF8"
COLOR_CLASSICAL = "#4c97c9"
COLOR_LLM = "#F44336"

METRICS = ["n_stable_regions", "lc", "gp_per_agent", "final_round"]
LABELS = ["n_stable_regions", "LC", "GP/N", "final_round"]


def load_report(compare_dir: str) -> dict | None:
    """`{matched_config, sides: {label: {...}}, deltas: {...}}` from the parent.

    A pre-runvault `compare_report.json` is still read as it stands.
    """
    legacy = os.path.join(compare_dir, "compare_report.json")
    if os.path.exists(legacy):
        report = json.load(open(legacy, encoding="utf-8"))
        return {
            "matched_config": report["matched_config"],
            "mock": report["mock"],
            "sides": {
                report["classical"]["label"]: report["classical"],
                report["llm"]["label"]: report["llm"],
            },
            "deltas": report["deltas"],
        }

    params = config_parameters(compare_dir)
    sides: dict[str, dict] = {}
    for child in sweep_children(compare_dir):
        p = config_parameters(child)
        scoped = run_scope_metrics(child)
        terminal = events_table(child, kind="terminal").iloc[0]
        sides[p["side"]] = {
            "label": p["side"],
            "provider": p["provider"],
            "n_stable_regions": float(terminal["n_stable_regions"]),
            "max_region_size": float(terminal["max_region_size"]),
            "n_distinct_cultures": float(terminal["n_distinct_cultures"]),
            "lc": float(terminal["lc"]),
            "gp": float(terminal["gp"]),
            "gp_per_agent": float(terminal["gp_per_agent"]),
            "final_round": scoped["final_round"],
            "converged": bool(scoped["converged"]),
            # 0 呼び出しの run は率の行そのものを書かない (欠測を 0 で埋めない)．
            "total_llm_calls": scoped.get("llm_calls", 0.0),
            "cache_hit_rate": scoped.get("llm_cache_hit_rate"),
        }
    deltas = run_scope_metrics(compare_dir)
    return {
        "matched_config": {k: params[k] for k in
                           ("width", "height", "features", "traits", "rounds", "seed")},
        "mock": params["mock"],
        "sides": sides,
        "deltas": {m: deltas[f"delta_{m}"] for m in
                   ("n_stable_regions", "lc", "gp", "gp_per_agent", "final_round")},
    }


def order_sides(sides: dict[str, dict]) -> tuple[dict, dict]:
    """`(classical, llm)` — the LLM side is whichever is not the classical one."""
    classical = sides["classical"]
    llm = next(v for k, v in sides.items() if k != "classical")
    return classical, llm


def render(report: dict, output_dir: str) -> str:
    classical, llm = order_sides(report["sides"])
    c_vals = [classical[m] for m in METRICS]
    l_vals = [llm[m] for m in METRICS]

    w = 0.38
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.2 * len(METRICS), 4))
    fig.patch.set_facecolor(COLOR_BG)
    for k, lab in enumerate(LABELS):
        ax = axes[k]
        ax.set_facecolor(COLOR_BG)
        ax.bar([0], [c_vals[k]], w, color=COLOR_CLASSICAL, label="classical")
        ax.bar([w * 1.2], [l_vals[k]], w, color=COLOR_LLM, label=llm["label"])
        ax.set_xticks([0, w * 1.2])
        ax.set_xticklabels(["classical", llm["label"]], rotation=20, fontsize=8)
        ax.set_title(lab, fontsize=10)
    cfg = report["matched_config"]
    fig.suptitle(
        f"Classical vs LLM (matched: {cfg['width']}×{cfg['height']}, F={cfg['features']}, "
        f"q={cfg['traits']}, rounds={cfg['rounds']}, seed={cfg['seed']}, mock={report['mock']})"
    )
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(output_dir, "compare_report.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[compare-report] wrote {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="culture-llm-tools compare-report", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--compare-dir", "--compare_dir", "--results-dir", "--results_dir",
                        default=None,
                        help="比較の親 run (未指定時は最新の compare)")
    parser.add_argument("--results-root", "--results_root", default="results")
    parser.add_argument("--output-dir", "--output_dir", default=None,
                        help="図の保存先 (default: results/culture-llm/figures/{run_slug})")
    args = parser.parse_args(argv)

    compare_dir = args.compare_dir
    if compare_dir is None:
        compare_dir = runvault_path(EXPERIMENT, args.results_root, subcommand="compare")
    output_dir = args.output_dir or figures_dir(compare_dir)
    os.makedirs(output_dir, exist_ok=True)

    report = load_report(compare_dir)
    if report is None:
        print(f"[compare-report] no comparison in {compare_dir}")
        return 1

    classical, llm = order_sides(report["sides"])
    deltas = report["deltas"]
    print("=== classical vs LLM (matched config) ===")
    print(f"{'metric':18} {'classical':>12}  {llm['label']:>12}  {'delta (LLM-classical)':>22}")
    for m, lab in [("n_stable_regions", "n_stable_regions"), ("lc", "LC"),
                   ("gp", "GP=|C|/N²"), ("gp_per_agent", "GP/N"), ("final_round", "final_round")]:
        d = deltas.get(m, float("nan"))
        print(f"{lab:18} {classical[m]:>12.4f}  {llm[m]:>12.4f}  {d:>22.4f}")
    hit = llm["cache_hit_rate"]
    hit_text = "n/a" if hit is None else f"{hit * 100:.1f}%"
    print(f"\nclassical LLM calls: {classical['total_llm_calls']:.0f}  |  "
          f"{llm['label']} LLM calls: {llm['total_llm_calls']:.0f} (cache-hit {hit_text})")

    render(report, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
