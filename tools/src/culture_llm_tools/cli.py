"""culture-llm-tools — unified CLI dispatcher.

Usage:
    culture-llm-tools visualize [...]
    culture-llm-tools visualize-sweep [...]
    culture-llm-tools show-experiment-settings [...]
    culture-llm-tools reproduce [...]

Arguments after the subcommand are passed verbatim to that subcommand's argparse.
Add `--help` after a subcommand for its own help.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="culture-llm-tools",
        description="YuLan-OneSim (Wang et al. 2025) culture-dissemination visualization & analysis",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "visualize",
        help="single-run visualization (culture map + LC/GP time series)",
        add_help=False,
    )
    subparsers.add_parser(
        "visualize-sweep",
        help="sweep visualization (F×q heatmap of n_stable_regions; classical vs LLM)",
        add_help=False,
    )
    subparsers.add_parser(
        "show-experiment-settings",
        help="print a results directory's settings (config / sweep_config / run_metadata)",
        add_help=False,
    )
    subparsers.add_parser(
        "reproduce",
        help="Appendix F LC/GP reproduction helper (Phase 3 stub)",
        add_help=False,
    )

    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        parser.parse_args(argv)
        return

    command = argv[0]
    rest = argv[1:]
    if command == "visualize":
        from culture_llm_tools.visualize import main as run_main

        run_main(rest)
    elif command == "visualize-sweep":
        from culture_llm_tools.visualize_sweep import main as run_main

        run_main(rest)
    elif command == "show-experiment-settings":
        from culture_llm_tools.show_experiment_settings import main as run_main

        run_main(rest)
    elif command == "reproduce":
        from culture_llm_tools.reproduce_paper import main as run_main

        run_main(rest)
    else:
        parser.parse_args(argv)


if __name__ == "__main__":
    main()
