#!/usr/bin/env python3
"""visualize_sweep.py — F×q sweep visualization for the culture-dissemination model.

Reads a runvault `sweep` parent and produces:
  - sweep_heatmap.png : F×q heatmap of mean n_stable_regions
  - sweep_lc_gp.png   : F×q heatmaps of mean LC and mean GP

The one-row-per-trial table is rebuilt from the sweep parent's children
(`subcommand=sweep-point`): runvault keeps no `sweep_summary.csv` on disk, and a
heatmap of *means over trials* needs the individual trials, which live in each
child's `events.jsonl`. A pre-runvault `sweep_summary.csv` is still read as it
stands.

Usage:
    uv run culture-llm-tools visualize-sweep
    uv run culture-llm-tools visualize-sweep --sweep-dir "$(runvault path --experiment culture-llm --latest --subcommand sweep)"
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from runvault.read import figures_dir, runvault_path, sweep_events_table

EXPERIMENT = "culture-llm"

COLOR_BG = "#FAFAF8"


def load_summary(sweep_dir: str) -> pd.DataFrame:
    """One row per trial (`provider`, `features`, `traits`, final metrics).

    The condition columns come from each child's `parameters`; the trial columns
    from its terminal events. Only `provider` / `features` / `traits` are asked
    for as parameter columns — `sweep_events_table` overwrites an event column
    with a same-named parameter column, so the terminal's own `seed` must not be
    shadowed (the conditions call the base seed `base_seed` for exactly that
    reason).
    """
    legacy = os.path.join(sweep_dir, "sweep_summary.csv")
    if os.path.exists(legacy):
        return pd.read_csv(legacy)

    df = sweep_events_table(sweep_dir, ["provider", "features", "traits"], kind="terminal")
    df["run"] = df["unit_id"].str.removeprefix("trial-").astype(int)
    df["converged"] = ~df["censored"]
    df["final_round"] = df["t"]
    return df


def _pivot(df: pd.DataFrame, value: str) -> tuple[np.ndarray, list[int], list[int]]:
    g = df.groupby(["features", "traits"])[value].mean().reset_index()
    feats = sorted(g["features"].unique())
    traits = sorted(g["traits"].unique())
    mat = np.full((len(feats), len(traits)), np.nan)
    fi = {f: i for i, f in enumerate(feats)}
    ti = {t: i for i, t in enumerate(traits)}
    for _, r in g.iterrows():
        mat[fi[int(r["features"])], ti[int(r["traits"])]] = r[value]
    return mat, feats, traits


def _heatmap(ax, mat, feats, traits, title, cmap="viridis") -> None:
    im = ax.imshow(mat, cmap=cmap, aspect="auto", origin="lower")
    ax.set_xticks(range(len(traits)))
    ax.set_xticklabels(traits)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats)
    ax.set_xlabel("traits q")
    ax.set_ylabel("features F")
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="culture-llm-tools visualize-sweep")
    parser.add_argument(
        "--sweep-dir", "--sweep_dir", "--results-dir", "--results_dir", default=None,
        help=(
            "掃引親 run のディレクトリ．未指定時は runvault に最新の掃引を聞く "
            "(--experiment culture-llm --subcommand sweep)．"
        ),
    )
    parser.add_argument(
        "--results-root", "--results_root", default="results",
        help="--sweep-dir 未指定時に runvault が探す results ルート (default: results)",
    )
    parser.add_argument(
        "--output-dir", "--output_dir", default=None,
        help="図の保存先 (default: results/culture-llm/figures/{run_slug})",
    )
    args = parser.parse_args(argv)

    sweep_dir = args.sweep_dir
    if sweep_dir is None:
        sweep_dir = runvault_path(EXPERIMENT, args.results_root, subcommand="sweep")
    output_dir = args.output_dir or figures_dir(sweep_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[visualize-sweep] sweep: {sweep_dir}")
    df = load_summary(sweep_dir)
    providers = sorted(df["provider"].unique()) if "provider" in df.columns else ["none"]

    # n_stable_regions heatmap (per provider if multiple).
    fig, axes = plt.subplots(1, len(providers), figsize=(6 * len(providers), 5), squeeze=False)
    fig.patch.set_facecolor(COLOR_BG)
    for k, prov in enumerate(providers):
        sub = df[df["provider"] == prov] if "provider" in df.columns else df
        mat, feats, traits = _pivot(sub, "n_stable_regions")
        _heatmap(axes[0][k], mat, feats, traits, f"mean n_stable_regions ({prov})")
    fig.tight_layout()
    out = os.path.join(output_dir, "sweep_heatmap.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize-sweep] wrote {out}")

    # LC + GP heatmaps for the first provider (or combined classical).
    sub = df[df["provider"] == providers[0]] if "provider" in df.columns else df
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(COLOR_BG)
    if "lc" in sub.columns:
        mat, feats, traits = _pivot(sub, "lc")
        _heatmap(ax1, mat, feats, traits, f"mean LC ({providers[0]})", cmap="plasma")
    if "gp" in sub.columns:
        mat, feats, traits = _pivot(sub, "gp")
        _heatmap(ax2, mat, feats, traits, f"mean GP ({providers[0]})", cmap="magma")
    fig.tight_layout()
    out = os.path.join(output_dir, "sweep_lc_gp.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[visualize-sweep] wrote {out}")


if __name__ == "__main__":
    main()
