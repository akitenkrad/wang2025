#!/usr/bin/env python3
"""animate.py — intermediate culture-map animation / montage.

Reads the per-round culture-grid snapshots written by `culture-llm run
--snapshot-interval N` (under `<results-dir>/snapshots/culture_grid_round_*.csv`,
indexed by `snapshots/index.json`) and renders:

  - culture_animation.gif : an animated culture map over rounds (if Pillow is
                            available; one frame per snapshot).
  - culture_montage.png   : a static grid-of-panels montage (always produced).

A stable colour id is assigned to each distinct culture string **across all
snapshots** so a region keeps its colour through the animation.

Usage:
    uv run culture-llm-tools animate
    uv run culture-llm-tools animate --results-dir results/latest --fps 4
"""

from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_BG = "#FAFAF8"


def load_index(results_dir: str) -> list[int] | None:
    path = os.path.join(results_dir, "snapshots", "index.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("rounds", [])


def load_snapshot(results_dir: str, round_: int) -> pd.DataFrame:
    name = f"culture_grid_round_{round_:06}.csv"
    path = os.path.join(results_dir, "snapshots", name)
    return pd.read_csv(path, dtype={"culture": str})


def build_global_colormap(results_dir: str, rounds: list[int]) -> dict[str, int]:
    """Assign each distinct culture string a stable colour id across all frames."""
    cultures: set[str] = set()
    for r in rounds:
        df = load_snapshot(results_dir, r)
        cultures.update(df["culture"].unique())
    return {c: i for i, c in enumerate(sorted(cultures))}


def to_grid(df: pd.DataFrame, colormap: dict[str, int]) -> np.ndarray:
    rows = int(df["row"].max()) + 1
    cols = int(df["col"].max()) + 1
    grid = np.zeros((rows, cols), dtype=int)
    for _, r in df.iterrows():
        grid[int(r["row"]), int(r["col"])] = colormap[r["culture"]]
    return grid


def render_montage(results_dir: str, output_dir: str, rounds: list[int],
                   colormap: dict[str, int]) -> str:
    n = len(rounds)
    ncols = min(n, 5)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows), squeeze=False)
    fig.patch.set_facecolor(COLOR_BG)
    vmax = max(len(colormap) - 1, 1)
    for k, r in enumerate(rounds):
        ax = axes[k // ncols][k % ncols]
        grid = to_grid(load_snapshot(results_dir, r), colormap)
        ax.imshow(grid, cmap="tab20", interpolation="nearest", vmin=0, vmax=vmax)
        ax.set_title(f"round {r}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.suptitle(f"Culture-map snapshots ({len(colormap)} distinct cultures)")
    fig.tight_layout()
    out = os.path.join(output_dir, "culture_montage.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[animate] wrote {out}  ({n} frames)")
    return out


def render_gif(results_dir: str, output_dir: str, rounds: list[int],
               colormap: dict[str, int], fps: int) -> str | None:
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        print("[animate] Pillow not available; skipping GIF (montage still produced)")
        return None

    vmax = max(len(colormap) - 1, 1)
    frames = []
    for r in rounds:
        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor(COLOR_BG)
        grid = to_grid(load_snapshot(results_dir, r), colormap)
        ax.imshow(grid, cmap="tab20", interpolation="nearest", vmin=0, vmax=vmax)
        ax.set_title(f"round {r}")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.canvas.draw()
        # RGBA buffer → RGB PIL image (version-robust).
        buf = np.asarray(fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(buf[:, :, :3].copy()))
        plt.close(fig)

    out = os.path.join(output_dir, "culture_animation.gif")
    duration = max(int(1000 / max(fps, 1)), 1)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    print(f"[animate] wrote {out}  ({len(frames)} frames, {fps} fps)")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="culture-llm-tools animate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", "--results_dir", default="results/latest")
    parser.add_argument("--output-dir", "--output_dir", default=None)
    parser.add_argument("--fps", type=int, default=4, help="animation frames per second")
    parser.add_argument("--no-gif", action="store_true", help="montage only (skip GIF)")
    args = parser.parse_args(argv)

    results_dir = args.results_dir
    output_dir = args.output_dir or results_dir
    os.makedirs(output_dir, exist_ok=True)

    rounds = load_index(results_dir)
    if not rounds:
        print(
            f"[animate] no snapshots in {results_dir}/snapshots/.\n"
            f"  Re-run with: culture-llm run ... --snapshot-interval N",
        )
        return 1

    colormap = build_global_colormap(results_dir, rounds)
    render_montage(results_dir, output_dir, rounds, colormap)
    if not args.no_gif:
        render_gif(results_dir, output_dir, rounds, colormap, args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
