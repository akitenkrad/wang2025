#!/usr/bin/env python3
"""visualize.py — single-run visualization for the culture-dissemination model.

Reads a `run` results directory (default `results/latest`) and produces:
  - culture_map.png   : the final culture grid coloured by distinct culture profile
  - lc_gp_timeseries.png : LC and GP (and the auxiliary GP/N) per round

Usage:
    uv run culture-llm-tools visualize
    uv run culture-llm-tools visualize --results-dir results/latest --output-dir out
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_BG = "#FAFAF8"
COLOR_LC = "#4c97c9"
COLOR_GP = "#F44336"
COLOR_GPN = "#9C27B0"

# Appendix F targets (LLM variant): LC exceeds 0.50 by ~round 60; GP ~0.35-0.40.
LC_TARGET = 0.50
GP_BAND = (0.35, 0.40)


def load_config(results_dir: str) -> dict | None:
    path = os.path.join(results_dir, "config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def plot_culture_map(results_dir: str, output_dir: str, cfg: dict | None) -> None:
    path = os.path.join(results_dir, "culture_grid_final.csv")
    if not os.path.exists(path):
        print(f"[visualize] no culture grid at {path}; skipping culture map")
        return
    df = pd.read_csv(path, dtype={"culture": str})
    rows = int(df["row"].max()) + 1
    cols = int(df["col"].max()) + 1

    # Map each distinct culture string to an integer colour id.
    uniq = {c: i for i, c in enumerate(sorted(df["culture"].unique()))}
    grid = np.zeros((rows, cols), dtype=int)
    for _, r in df.iterrows():
        grid[int(r["row"]), int(r["col"])] = uniq[r["culture"]]

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(COLOR_BG)
    ax.imshow(grid, cmap="tab20", interpolation="nearest")
    title = "Final culture map"
    if cfg:
        title += f"  (F={cfg.get('features')}, q={cfg.get('traits')}, provider={cfg.get('provider')})"
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out = os.path.join(output_dir, "culture_map.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize] wrote {out}  ({len(uniq)} distinct cultures)")


def plot_timeseries(results_dir: str, output_dir: str) -> None:
    path = os.path.join(results_dir, "metrics.csv")
    if not os.path.exists(path):
        print(f"[visualize] no metrics at {path}; skipping time series")
        return
    df = pd.read_csv(path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(COLOR_BG)

    ax1.plot(df["round"], df["lc"], color=COLOR_LC, label="LC")
    ax1.axhline(LC_TARGET, color="gray", ls="--", lw=1, label=f"LC target {LC_TARGET}")
    ax1.set_xlabel("round")
    ax1.set_ylabel("local convergence (LC)")
    ax1.set_title("Local convergence over rounds")
    ax1.set_facecolor(COLOR_BG)
    ax1.legend()

    ax2.plot(df["round"], df["gp"], color=COLOR_GP, label="GP = |C|/N² (documented)")
    if "gp_per_agent" in df.columns:
        ax2.plot(df["round"], df["gp_per_agent"], color=COLOR_GPN, label="GP/N = |C|/N (auxiliary)")
    ax2.axhspan(GP_BAND[0], GP_BAND[1], color="orange", alpha=0.15, label="Appendix F band 0.35-0.40")
    ax2.set_xlabel("round")
    ax2.set_ylabel("global polarization (GP)")
    ax2.set_title("Global polarization over rounds")
    ax2.set_facecolor(COLOR_BG)
    ax2.legend()

    fig.tight_layout()
    out = os.path.join(output_dir, "lc_gp_timeseries.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize] wrote {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="culture-llm-tools visualize")
    parser.add_argument("--results-dir", default="results/latest")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)

    cfg = load_config(results_dir)
    plot_culture_map(results_dir, output_dir, cfg)
    plot_timeseries(results_dir, output_dir)


if __name__ == "__main__":
    main()
