#!/usr/bin/env python3
"""visualize.py — single-run visualization for the culture-dissemination model.

Reads a runvault `run` directory and produces:
  - culture_map.png      : the final culture grid coloured by distinct culture profile
  - lc_gp_timeseries.png : LC and GP (and the auxiliary GP/N) per round

Which run is asked of runvault when `--results-dir` is omitted
(`runvault path --experiment culture-llm --latest --subcommand run --standalone`);
`results/` is never scanned for a directory that looks recent.

The figures go *beside* the run (`results/culture-llm/figures/<run_slug>/`):
`manifest.csv` is settled by `finish()`, so anything added to the run afterwards
would carry no hash.

Usage:
    uv run culture-llm-tools visualize
    uv run culture-llm-tools visualize --results-dir "$(runvault path --experiment culture-llm --latest --subcommand run --standalone)"
    uv run culture-llm-tools visualize --output-dir out
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from runvault.read import (
    artifacts_dir,
    config_parameters,
    figures_dir,
    metrics_wide,
    runvault_path,
)

EXPERIMENT = "culture-llm"

COLOR_BG = "#FAFAF8"
COLOR_LC = "#4c97c9"
COLOR_GP = "#F44336"
COLOR_GPN = "#9C27B0"

# Appendix F targets (LLM variant): LC exceeds 0.50 by ~round 60; GP ~0.35-0.40.
# Read off Figure 9 rather than printed as a table, which is why they annotate the
# plot instead of sitting in the run's `reference.csv`.
LC_TARGET = 0.50
GP_BAND = (0.35, 0.40)


def load_metrics(run_dir: str) -> pd.DataFrame:
    """Per-round metrics as one row per round.

    runvault's `metrics.csv` is long, so `metrics_wide` turns it back. The time
    axis is `step` there and `round` in this model's own vocabulary; a legacy wide
    `metrics.csv` already has a `round` column and is left alone.
    """
    df = metrics_wide(os.path.join(run_dir, "metrics.csv"))
    if "step" in df.columns and "round" not in df.columns:
        df = df.rename(columns={"step": "round"})
    return df


def plot_culture_map(run_dir: str, output_dir: str, cfg: dict | None) -> None:
    path = os.path.join(artifacts_dir(run_dir), "culture_grid_final.csv")
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


def plot_timeseries(run_dir: str, output_dir: str) -> None:
    if not os.path.exists(os.path.join(run_dir, "metrics.csv")):
        print(f"[visualize] no metrics in {run_dir}; skipping time series")
        return
    df = load_metrics(run_dir)

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
    parser.add_argument(
        "--results-dir", "--results_dir", default=None,
        help=(
            "runvault の run ディレクトリ．未指定時は runvault に最新の run を聞く "
            "(--experiment culture-llm --subcommand run --standalone)．"
        ),
    )
    parser.add_argument(
        "--results-root", "--results_root", default="results",
        help="--results-dir 未指定時に runvault が探す results ルート (default: results)",
    )
    parser.add_argument(
        "--output-dir", "--output_dir", default=None,
        help="図の保存先 (default: results/culture-llm/figures/{run_slug})",
    )
    args = parser.parse_args(argv)

    run_dir = args.results_dir
    if run_dir is None:
        run_dir = runvault_path(
            EXPERIMENT, args.results_root, subcommand="run", standalone=True
        )
    output_dir = args.output_dir or figures_dir(run_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[visualize] run: {run_dir}")
    cfg = config_parameters(run_dir, required=False)
    plot_culture_map(run_dir, output_dir, cfg)
    plot_timeseries(run_dir, output_dir)


if __name__ == "__main__":
    main()
