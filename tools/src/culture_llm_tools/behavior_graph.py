#!/usr/bin/env python3
"""behavior_graph.py — render the behaviour-graph / ODD concept export.

Reads `artifacts/behavior_graph.json` (written by `culture-llm run`; a structured
ODD outline + a node/edge behaviour graph derived from the fixed model) and
renders:

  - behavior_graph.png : a node-coloured-by-kind diagram of the behaviour graph
                         (agent / state / event / metric), with labelled edges.

This is the visualization side of the *concept demo* of YuLan-OneSim's
ODD/behaviour-graph auto-construction: the artefact is a faithful structured
description of the fixed Axelrod scenario, NOT an LLM-synthesised construction
(see `provenance` in the JSON and the architecture docs).

Usage:
    uv run culture-llm-tools behavior-graph
    uv run culture-llm-tools behavior-graph --results-dir "$(runvault path --experiment culture-llm --latest --subcommand run --standalone)"
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
from runvault.read import artifacts_dir, figures_dir, runvault_path

EXPERIMENT = "culture-llm"

COLOR_BG = "#FAFAF8"
KIND_COLOR = {
    "agent": "#4c97c9",
    "state": "#9C27B0",
    "event": "#FF9800",
    "metric": "#2E7D32",
}
# Deterministic layout columns by kind (left → right pipeline).
KIND_COL = {"agent": 0, "state": 1, "event": 2, "metric": 3}


def load_graph(run_dir: str) -> dict | None:
    path = os.path.join(artifacts_dir(run_dir), "behavior_graph.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _layout(nodes: list[dict]) -> dict[str, tuple[float, float]]:
    """Deterministic columnar layout: x by kind, y stacked within a kind."""
    by_kind: dict[str, list[dict]] = {}
    for n in nodes:
        by_kind.setdefault(n["kind"], []).append(n)
    pos: dict[str, tuple[float, float]] = {}
    for kind, group in by_kind.items():
        x = KIND_COL.get(kind, 4)
        for i, n in enumerate(group):
            y = -(i - (len(group) - 1) / 2.0)
            pos[n["id"]] = (float(x), float(y))
    return pos


def render(graph: dict, output_dir: str) -> str:
    nodes = graph["nodes"]
    edges = graph["edges"]
    pos = _layout(nodes)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(COLOR_BG)
    ax.set_facecolor(COLOR_BG)

    # edges first (under nodes).
    for e in edges:
        if e["from"] not in pos or e["to"] not in pos:
            continue
        x0, y0 = pos[e["from"]]
        x1, y1 = pos[e["to"]]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="#999999", lw=1.1,
                            connectionstyle="arc3,rad=0.08"),
        )
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, e["relation"], fontsize=7,
                color="#666666", ha="center", va="center")

    # nodes.
    for n in nodes:
        x, y = pos[n["id"]]
        color = KIND_COLOR.get(n["kind"], "#607D8B")
        ax.scatter([x], [y], s=2200, color=color, alpha=0.85, zorder=3, edgecolors="white")
        ax.text(x, y, n["label"], fontsize=7.5, color="white", ha="center", va="center",
                zorder=4, wrap=True)

    # legend by kind.
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markersize=12, label=k) for k, c in KIND_COLOR.items()]
    ax.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
              bbox_to_anchor=(0.5, 1.08))

    ax.set_title(f"Behaviour graph — {graph.get('variant', '')}", fontsize=12, pad=24)
    ax.axis("off")
    fig.tight_layout()
    out = os.path.join(output_dir, "behavior_graph.png")
    fig.savefig(out, dpi=150, facecolor=COLOR_BG)
    plt.close(fig)
    print(f"[behavior-graph] wrote {out}  ({len(nodes)} nodes, {len(edges)} edges)")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="culture-llm-tools behavior-graph", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", "--results_dir", default=None,
                        help="runvault の run ディレクトリ (未指定時は最新の run)")
    parser.add_argument("--results-root", "--results_root", default="results")
    parser.add_argument("--output-dir", "--output_dir", default=None,
                        help="図の保存先 (default: results/culture-llm/figures/{run_slug})")
    parser.add_argument("--print-odd", action="store_true", help="also print the ODD protocol")
    args = parser.parse_args(argv)

    run_dir = args.results_dir
    if run_dir is None:
        run_dir = runvault_path(
            EXPERIMENT, args.results_root, subcommand="run", standalone=True
        )
    output_dir = args.output_dir or figures_dir(run_dir)
    os.makedirs(output_dir, exist_ok=True)

    graph = load_graph(run_dir)
    if graph is None:
        print(f"[behavior-graph] no behavior_graph.json in {artifacts_dir(run_dir)}")
        return 1

    if args.print_odd:
        odd = graph.get("odd", {})
        print(f"=== ODD protocol — {graph.get('variant', '')} ===")
        print(f"provenance: {graph.get('provenance', '')}")
        for key in ("purpose", "entities_state_variables_scales", "process_overview_scheduling",
                    "design_concepts", "initialization", "input_data", "submodels"):
            print(f"\n[{key}]\n{odd.get(key, '')}")
        print()

    render(graph, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
