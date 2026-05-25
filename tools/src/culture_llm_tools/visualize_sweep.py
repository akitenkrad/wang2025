#!/usr/bin/env python3
"""visualize_sweep.py — F×q sweep visualization for the culture-dissemination model.

Reads a `sweep` results directory (default `results/latest`) and produces:
  - sweep_heatmap.png    : F×q heatmap of mean n_stable_regions
  - sweep_lc_gp.png      : F×q heatmaps of mean LC and mean GP
If multiple providers are present in sweep_summary.csv, a classical-vs-LLM
comparison panel is added.

Usage:
    uv run culture-llm-tools visualize-sweep
    uv run culture-llm-tools visualize-sweep --results-dir results/latest --output-dir out
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_BG = "#FAFAF8"


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
    parser.add_argument("--results-dir", default="results/latest")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(results_dir, "sweep_summary.csv")
    if not os.path.exists(path):
        print(f"[visualize-sweep] no sweep_summary.csv at {path}")
        return
    df = pd.read_csv(path)
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
