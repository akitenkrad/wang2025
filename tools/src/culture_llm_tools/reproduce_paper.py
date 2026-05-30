#!/usr/bin/env python3
"""reproduce_paper.py — Appendix F / Axelrod Table 7-2 batch reproduction.

Drives the Rust `culture-llm reproduce` subcommand (classical `--provider none`,
offline-verifiable, 0 LLM calls), then renders the observed-vs-paper figures from
its `reproduce_detail.csv` + `reproduce_summary.json` into
`results/reproduce_<ts>/figures/`:

  - regions_vs_paper.png : observed mean n_stable_regions vs Axelrod Table 7-2
                           targets (per condition, with tolerance bands).
  - lc_gp_by_condition.png : mean LC and mean GP/N per condition.

The Rust side owns the simulation + the PASS/off verdicts (written into
`reproduce_summary.json`); this Python step adds the figures and prints the
observed-vs-paper table. Mirrors the sibling `hegselmann2002` reproduce driver.

Usage:
    uv run culture-llm-tools reproduce               # full (runs=30)
    uv run culture-llm-tools reproduce --quick        # fast smoke (runs=5)
    uv run culture-llm-tools reproduce --skip-build    # reuse a prior build
    uv run culture-llm-tools reproduce --results-dir results/reproduce_<ts>  # figures only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLOR_BG = "#FAFAF8"
COLOR_OBS = "#4c97c9"
COLOR_TGT = "#F44336"
COLOR_BAND = "#FFB74D"
COLOR_LC = "#4c97c9"
COLOR_GPN = "#9C27B0"

# This module lives at tools/src/culture_llm_tools/reproduce_paper.py;
# parents[3] is the workspace (cargo) root. Override via CULTURE_LLM_PROJECT_ROOT.
_env_root = os.environ.get("CULTURE_LLM_PROJECT_ROOT")
PROJECT_ROOT = Path(_env_root).resolve() if _env_root else Path(__file__).resolve().parents[3]


def ensure_build() -> None:
    print("=== cargo build --release ===")
    subprocess.run(["cargo", "build", "--release"], cwd=PROJECT_ROOT, check=True)


def run_reproduce(output_dir: Path, runs: int, rounds: int, seed: int, quick: bool) -> Path:
    """Invoke `culture-llm reproduce`, returning the produced reproduce_<ts> dir."""
    args = [
        "cargo", "run", "--release", "--quiet", "--",
        "reproduce",
        "--provider", "none",
        "--runs", str(runs),
        "--rounds", str(rounds),
        "--seed", str(seed),
        "--output-dir", str(output_dir),
    ]
    if quick:
        args.append("--quick")
    before = {p.name for p in output_dir.iterdir()} if output_dir.exists() else set()
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)
    after = {p.name for p in output_dir.iterdir()}
    new = sorted(n for n in (after - before) if n.startswith("reproduce_"))
    if new:
        return output_dir / new[-1]
    # Fallback: latest reproduce_* by mtime.
    candidates = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("reproduce_")]
    if not candidates:
        raise RuntimeError(f"no reproduce_<ts> directory produced under {output_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def render_regions_vs_paper(summary: dict, figures_dir: Path) -> Path:
    conds = summary["conditions"]
    ids = [c["id"] for c in conds]
    obs = [c["observed_mean_regions"] for c in conds]
    tgt = [c["target_regions"] for c in conds]
    x = np.arange(len(ids))
    w = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.bar(x - w / 2, obs, w, color=COLOR_OBS, label="observed (this impl)")
    ax.bar(x + w / 2, tgt, w, color=COLOR_TGT, alpha=0.7, label="Axelrod Table 7-2 target")
    for i, c in enumerate(conds):
        verdict = "PASS" if c["within_tolerance"] else "off"
        ax.text(x[i], max(obs[i], tgt[i]) + 0.4, verdict, ha="center", fontsize=9,
                color="#2E7D32" if c["within_tolerance"] else "#B71C1C")
    ax.set_xticks(x)
    ax.set_xticklabels(ids)
    ax.set_ylabel("mean n_stable_regions")
    ax.set_title(
        f"Appendix F / Table 7-2 reproduction "
        f"(provider={summary['provider']}, runs={summary['runs']}): "
        f"{summary['n_pass']}/{summary['n_total']} within tolerance"
    )
    ax.legend()
    fig.tight_layout()
    out = figures_dir / "regions_vs_paper.png"
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def render_lc_gp_by_condition(summary: dict, figures_dir: Path) -> Path:
    conds = summary["conditions"]
    ids = [c["id"] for c in conds]
    lc = [c["mean_lc"] for c in conds]
    gpn = [c["mean_gp_per_agent"] for c in conds]
    x = np.arange(len(ids))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor(COLOR_BG)
    ax1.set_facecolor(COLOR_BG)
    ax1.bar(x, lc, color=COLOR_LC)
    ax1.axhline(0.50, color="gray", ls="--", lw=1, label="LC ref 0.50")
    ax1.set_xticks(x)
    ax1.set_xticklabels(ids)
    ax1.set_ylabel("mean LC")
    ax1.set_title("Mean local convergence by condition")
    ax1.legend()

    ax2.set_facecolor(COLOR_BG)
    ax2.bar(x, gpn, color=COLOR_GPN)
    ax2.axhspan(0.35, 0.40, color=COLOR_BAND, alpha=0.2, label="Appendix F band 0.35-0.40")
    ax2.set_xticks(x)
    ax2.set_xticklabels(ids)
    ax2.set_ylabel("mean GP/N = |C|/N (auxiliary)")
    ax2.set_title("Mean GP/N by condition (documented GP=|C|/N² is tiny)")
    ax2.legend()

    fig.tight_layout()
    out = figures_dir / "lc_gp_by_condition.png"
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def render_figures(repro_dir: Path) -> dict:
    summary_path = repro_dir / "reproduce_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"no reproduce_summary.json at {summary_path}")
    with summary_path.open(encoding="utf-8") as f:
        summary = json.load(f)
    figures_dir = repro_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=== observed vs paper (Axelrod Table 7-2 / Appendix F) ===")
    print(f"{'id':8} {'F':>3} {'q':>3} {'observed':>9} {'target':>7} {'LC':>6} {'GP/N':>6}  verdict")
    for c in summary["conditions"]:
        print(
            f"{c['id']:8} {c['features']:>3} {c['traits']:>3} "
            f"{c['observed_mean_regions']:>9.2f} {c['target_regions']:>7.1f} "
            f"{c['mean_lc']:>6.3f} {c['mean_gp_per_agent']:>6.3f}  "
            f"{'PASS' if c['within_tolerance'] else 'off'}"
        )
    print(f"-> {summary['n_pass']}/{summary['n_total']} within tolerance")

    render_regions_vs_paper(summary, figures_dir)
    render_lc_gp_by_condition(summary, figures_dir)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="culture-llm-tools reproduce",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", "--output_dir", default="results",
                        help="results root (cargo writes reproduce_<ts>/ here)")
    parser.add_argument("--results-dir", "--results_dir", default=None,
                        help="render figures from an existing reproduce_<ts> dir (skips cargo)")
    parser.add_argument("--runs", type=int, default=30, help="runs per condition (full mode)")
    parser.add_argument("--rounds", type=int, default=20000, help="max engine ticks per run")
    parser.add_argument("--seed", type=int, default=42, help="seed base")
    parser.add_argument("--quick", action="store_true",
                        help="fast smoke (runs=5, rounds<=5000); not for validation")
    parser.add_argument("--skip-build", action="store_true", help="skip cargo build --release")
    parser.add_argument("--workspace-root", "--workspace_root", default=None,
                        help="cargo workspace root (default: inferred)")
    args = parser.parse_args(argv)

    global PROJECT_ROOT
    if args.workspace_root:
        PROJECT_ROOT = Path(args.workspace_root).resolve()

    # Figures-only mode: render from an existing reproduce_<ts> dir.
    if args.results_dir is not None:
        repro_dir = Path(args.results_dir)
        if not repro_dir.is_absolute():
            repro_dir = PROJECT_ROOT / repro_dir
        try:
            summary = render_figures(repro_dir)
        except Exception as e:  # noqa: BLE001
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0 if summary["n_pass"] == summary["n_total"] else 1

    if shutil.which("cargo") is None:
        print("error: cargo not found; install the Rust toolchain.", file=sys.stderr)
        return 2

    output_root = Path(args.output_dir)
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root

    runs = 5 if args.quick else args.runs
    rounds = min(args.rounds, 5000) if args.quick else args.rounds

    if not args.skip_build:
        ensure_build()

    repro_dir = run_reproduce(output_root, runs, rounds, args.seed, args.quick)
    print(f"reproduce dir: {repro_dir}")
    summary = render_figures(repro_dir)
    print(f"figures -> {repro_dir / 'figures'}")
    return 0 if summary["n_pass"] == summary["n_total"] else 1


if __name__ == "__main__":
    sys.exit(main())
