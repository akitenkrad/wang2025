#!/usr/bin/env python3
"""reproduce_paper.py — Appendix F / Axelrod Table 7-2 batch reproduction.

Drives the Rust `culture-llm reproduce` subcommand (classical `--provider none`,
offline-verifiable, 0 LLM calls), then renders the observed-vs-paper figures from
the runvault runs it produced:

  - regions_vs_paper.png   : observed mean n_stable_regions vs the Axelrod
                             Table 7-2 targets, per condition.
  - lc_gp_by_condition.png : mean LC and mean GP/N per condition.

Where each number comes from:

  - **observed**  — the condition child's `metrics.csv` (`mean_n_stable_regions`
    and friends, `scope=run`).
  - **published** — the same child's `reference.csv`, written with its source
    (`Axelrod (1997) Table 7-2 …`).
  - **tolerance / PASS / off** — the parent's
    `artifacts/reproduce_verdicts.csv`. The band is ours, not the paper's, and it
    is declared once, in Rust; this module reads it rather than restating it, so
    the two cannot drift apart.

Usage:
    uv run culture-llm-tools reproduce               # full (runs=30)
    uv run culture-llm-tools reproduce --quick        # fast smoke (runs=5)
    uv run culture-llm-tools reproduce --skip-build    # reuse a prior build
    uv run culture-llm-tools reproduce --results-dir "$(runvault path --experiment culture-llm --latest --subcommand reproduce)"
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from runvault.read import (
    artifacts_dir,
    config_parameters,
    figures_dir,
    run_scope_metrics,
    runvault_path,
    sweep_children,
)

EXPERIMENT = "culture-llm"

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


def run_reproduce(results_root: Path, runs: int, rounds: int, seed: int, quick: bool) -> str:
    """Invoke `culture-llm reproduce`, returning the parent run directory.

    The directory is resolved by asking runvault for the latest finished
    `reproduce` run rather than by diffing the results root before and after: the
    run_slug carries a random suffix, so two runs in the same second no longer
    collide and there is nothing to disambiguate by mtime.
    """
    args = [
        "cargo", "run", "--release", "--quiet", "--",
        "reproduce",
        "--provider", "none",
        "--runs", str(runs),
        "--rounds", str(rounds),
        "--seed", str(seed),
        "--output-dir", str(results_root),
    ]
    if quick:
        args.append("--quick")
    subprocess.run(args, cwd=PROJECT_ROOT, check=True)
    return runvault_path(EXPERIMENT, str(results_root), subcommand="reproduce")


def read_reference(child_dir: str) -> dict[str, tuple[float, str]]:
    """`{metric: (published value, where it was read)}` from a child's reference.csv."""
    path = os.path.join(child_dir, "reference.csv")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return {r["name"]: (float(r["value"]), r["source"]) for r in csv.DictReader(f)}


def read_verdicts(parent_dir: str) -> dict[str, dict]:
    """`{condition id: verdict row}` from the parent's `artifacts/`."""
    path = os.path.join(artifacts_dir(parent_dir), "reproduce_verdicts.csv")
    if not os.path.exists(path):
        raise RuntimeError(f"no reproduce_verdicts.csv at {path}")
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def collect(parent_dir: str) -> list[dict]:
    """One row per condition: the child's observation + its published target."""
    verdicts = read_verdicts(parent_dir)
    rows: list[dict] = []
    for child in sweep_children(parent_dir):
        params = config_parameters(child)
        scoped = run_scope_metrics(child)
        reference = read_reference(child)
        cid = params["condition_id"]
        target, source = reference.get("mean_n_stable_regions", (float("nan"), ""))
        verdict = verdicts[cid]
        rows.append({
            "id": cid,
            "features": params["features"],
            "traits": params["traits"],
            "runs": int(scoped["n_units"]),
            "observed": scoped["mean_n_stable_regions"],
            "target": target,
            "source": source,
            "mean_lc": scoped["mean_lc"],
            "mean_gp_per_agent": scoped["mean_gp_per_agent"],
            "tolerance": float(verdict["tolerance"]),
            "abs_error": float(verdict["abs_error"]),
            "within_tolerance": verdict["within_tolerance"] == "true",
        })
    order = list(verdicts)
    rows.sort(key=lambda r: order.index(r["id"]))
    return rows


def render_regions_vs_paper(rows: list[dict], provider: str, figures_out: Path) -> Path:
    ids = [r["id"] for r in rows]
    obs = [r["observed"] for r in rows]
    tgt = [r["target"] for r in rows]
    x = np.arange(len(ids))
    w = 0.38

    n_pass = sum(1 for r in rows if r["within_tolerance"])
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)
    ax.bar(x - w / 2, obs, w, color=COLOR_OBS, label="observed (this impl)")
    ax.bar(x + w / 2, tgt, w, color=COLOR_TGT, alpha=0.7, label="Axelrod Table 7-2 target")
    for i, r in enumerate(rows):
        verdict = "PASS" if r["within_tolerance"] else "off"
        ax.text(x[i], max(obs[i], tgt[i]) + 0.4, verdict, ha="center", fontsize=9,
                color="#2E7D32" if r["within_tolerance"] else "#B71C1C")
    ax.set_xticks(x)
    ax.set_xticklabels(ids)
    ax.set_ylabel("mean n_stable_regions")
    ax.set_title(
        f"Appendix F / Table 7-2 reproduction "
        f"(provider={provider}, runs={rows[0]['runs']}): "
        f"{n_pass}/{len(rows)} within tolerance"
    )
    ax.legend()
    fig.tight_layout()
    out = figures_out / "regions_vs_paper.png"
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def render_lc_gp_by_condition(rows: list[dict], figures_out: Path) -> Path:
    ids = [r["id"] for r in rows]
    lc = [r["mean_lc"] for r in rows]
    gpn = [r["mean_gp_per_agent"] for r in rows]
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
    out = figures_out / "lc_gp_by_condition.png"
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def render_figures(parent_dir: str) -> list[dict]:
    rows = collect(parent_dir)
    provider = config_parameters(parent_dir)["provider"]
    figures_out = Path(figures_dir(parent_dir))
    figures_out.mkdir(parents=True, exist_ok=True)

    print("=== observed vs paper (Axelrod Table 7-2 / Appendix F) ===")
    print(f"{'id':8} {'F':>3} {'q':>3} {'observed':>9} {'target':>7} {'LC':>6} {'GP/N':>6}  verdict")
    for r in rows:
        print(
            f"{r['id']:8} {r['features']:>3} {r['traits']:>3} "
            f"{r['observed']:>9.2f} {r['target']:>7.1f} "
            f"{r['mean_lc']:>6.3f} {r['mean_gp_per_agent']:>6.3f}  "
            f"{'PASS' if r['within_tolerance'] else 'off'}"
        )
    n_pass = sum(1 for r in rows if r["within_tolerance"])
    print(f"-> {n_pass}/{len(rows)} within tolerance")
    print("   targets are each condition child's reference.csv (with its source);")
    print("   the tolerance band is ours and lives in the parent's reproduce_verdicts.csv")

    render_regions_vs_paper(rows, provider, figures_out)
    render_lc_gp_by_condition(rows, figures_out)
    print(f"figures -> {figures_out}")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="culture-llm-tools reproduce",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", "--output_dir", default="results",
                        help="results root (runvault writes <root>/culture-llm/ here)")
    parser.add_argument("--results-dir", "--results_dir", default=None,
                        help="render figures from an existing reproduce parent (skips cargo)")
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

    # Figures-only mode: render from an existing reproduce parent.
    if args.results_dir is not None:
        parent = Path(args.results_dir)
        if not parent.is_absolute():
            parent = PROJECT_ROOT / parent
        try:
            rows = render_figures(str(parent))
        except Exception as e:  # noqa: BLE001
            print(f"error: {e}", file=sys.stderr)
            return 1
        return 0 if all(r["within_tolerance"] for r in rows) else 1

    if shutil.which("cargo") is None:
        print("error: cargo not found; install the Rust toolchain.", file=sys.stderr)
        return 2

    results_root = Path(args.output_dir)
    if not results_root.is_absolute():
        results_root = PROJECT_ROOT / results_root

    runs = 5 if args.quick else args.runs
    rounds = min(args.rounds, 5000) if args.quick else args.rounds

    if not args.skip_build:
        ensure_build()

    parent = run_reproduce(results_root, runs, rounds, args.seed, args.quick)
    print(f"reproduce parent: {parent}")
    rows = render_figures(parent)
    return 0 if all(r["within_tolerance"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
