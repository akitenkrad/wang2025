#!/usr/bin/env python3
"""compare_report.py — classical vs LLM quantitative comparison figure.

Reads `compare_report.json` (written by `culture-llm compare`; matched-config
classical vs LLM run) and renders a grouped-bar comparison of the headline
metrics (n_stable_regions, LC, GP/N, final_round), plus prints the deltas table.

The classical side is the deterministic Axelrod baseline (0 LLM calls); the LLM
side is the YuLan-OneSim variant (mock or live). See `note` in the JSON.

Usage:
    uv run culture-llm-tools compare-report
    uv run culture-llm-tools compare-report --results-dir results/latest
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

COLOR_BG = "#FAFAF8"
COLOR_CLASSICAL = "#4c97c9"
COLOR_LLM = "#F44336"


def load_report(results_dir: str) -> dict | None:
    path = os.path.join(results_dir, "compare_report.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def render(report: dict, output_dir: str) -> str:
    classical = report["classical"]
    llm = report["llm"]
    metrics = ["n_stable_regions", "lc", "gp_per_agent", "final_round"]
    labels = ["n_stable_regions", "LC", "GP/N", "final_round"]
    c_vals = [classical[m] for m in metrics]
    l_vals = [llm[m] for m in metrics]

    x = np.arange(len(metrics))
    w = 0.38
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 4))
    fig.patch.set_facecolor(COLOR_BG)
    for k, (m, lab) in enumerate(zip(metrics, labels)):
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
    parser.add_argument("--results-dir", "--results_dir", default="results/latest")
    parser.add_argument("--output-dir", "--output_dir", default=None)
    args = parser.parse_args(argv)

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)

    report = load_report(results_dir)
    if report is None:
        print(f"[compare-report] no compare_report.json in {results_dir}")
        return 1

    classical, llm, deltas = report["classical"], report["llm"], report["deltas"]
    print("=== classical vs LLM (matched config) ===")
    print(f"{'metric':18} {'classical':>12} {classical['label']:>0}", end="")
    print(f"  {llm['label']:>12}  {'delta (LLM-classical)':>22}")
    for m, lab in [("n_stable_regions", "n_stable_regions"), ("lc", "LC"),
                   ("gp", "GP=|C|/N²"), ("gp_per_agent", "GP/N"), ("final_round", "final_round")]:
        d = deltas.get(m, float("nan"))
        print(f"{lab:18} {classical[m]:>12.4f} {'':>0}  {llm[m]:>12.4f}  {d:>22.4f}")
    print(f"\nclassical LLM calls: {classical['total_llm_calls']}  |  "
          f"{llm['label']} LLM calls: {llm['total_llm_calls']} "
          f"(cache-hit {llm['cache_hit_rate'] * 100:.1f}%)")

    render(report, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
