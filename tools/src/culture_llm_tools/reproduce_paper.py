#!/usr/bin/env python3
"""reproduce_paper.py — Appendix F (LC/GP) reproduction helper (Phase 3 stub).

Given an LLM-variant `run` results directory, reports whether the LC/GP time
series matches the paper's Appendix F targets:
  - LC exceeds 0.50 by round ~60
  - GP settles in the 0.35-0.40 band (see the documented GP inconsistency: the
    auxiliary gp_per_agent = |C|/N is also reported, as |C|/N² cannot reach that
    band; see docs/architecture.md).

The full multi-condition batch reproduction (classical-vs-LLM quantitative
comparison report) is a Phase 3 feature; this stub checks a single run.

Usage:
    uv run culture-llm-tools reproduce
    uv run culture-llm-tools reproduce --results-dir results/latest
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

LC_TARGET = 0.50
LC_TARGET_ROUND = 60
GP_BAND = (0.35, 0.40)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="culture-llm-tools reproduce")
    parser.add_argument("--results-dir", default="results/latest")
    args = parser.parse_args(argv)

    path = os.path.join(args.results_dir, "metrics.csv")
    if not os.path.exists(path):
        print(f"[reproduce] no metrics.csv at {path}")
        return
    df = pd.read_csv(path)

    print("=== Appendix F reproduction check (Phase 3 stub) ===")
    print(f"results: {args.results_dir}  ({len(df)} rounds)")

    # LC by round 60 (or final if shorter).
    by_round = df[df["round"] <= LC_TARGET_ROUND]
    lc_at = by_round["lc"].iloc[-1] if len(by_round) else df["lc"].iloc[-1]
    lc_ok = lc_at > LC_TARGET
    print(
        f"LC at round <= {LC_TARGET_ROUND}: {lc_at:.3f}  "
        f"(target > {LC_TARGET}) -> {'PASS' if lc_ok else 'below target'}"
    )

    gp_final = df["gp"].iloc[-1]
    gp_in_band = GP_BAND[0] <= gp_final <= GP_BAND[1]
    print(f"GP (documented |C|/N²) final: {gp_final:.5f}  (band {GP_BAND}) -> {'in band' if gp_in_band else 'OUT of band'}")
    if "gp_per_agent" in df.columns:
        gpn = df["gp_per_agent"].iloc[-1]
        gpn_band = GP_BAND[0] <= gpn <= GP_BAND[1]
        print(
            f"GP/N (auxiliary |C|/N) final: {gpn:.5f}  (band {GP_BAND}) -> "
            f"{'in band' if gpn_band else 'out of band'}"
        )
        print(
            "NOTE: the documented GP=|C|/N² cannot reach 0.35-0.40 for reasonable N; "
            "the paper's target most plausibly uses |C|/N (the auxiliary column). "
            "See docs/architecture.md for the inconsistency note."
        )


if __name__ == "__main__":
    main()
