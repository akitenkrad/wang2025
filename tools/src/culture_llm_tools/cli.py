"""culture-llm-tools — unified CLI dispatcher.

Usage:
    culture-llm-tools visualize [...]
    culture-llm-tools visualize-sweep [...]
    culture-llm-tools show-experiment-settings [...]
    culture-llm-tools reproduce [...]
    culture-llm-tools animate [...]
    culture-llm-tools behavior-graph [...]
    culture-llm-tools compare-report [...]

Arguments after the subcommand are passed verbatim to that subcommand's argparse.
Add `--help` after a subcommand for its own help.

The dispatcher assembly is delegated to the shared helper
`socsim_tools.cli.build_dispatcher` (prog name, subcommands, help text and argv
routing are unchanged). The visualization / settings implementations
(visualize / visualize_sweep / show_experiment_settings / reproduce_paper) stay
repo-local.
"""

from __future__ import annotations

from socsim_tools.cli import build_dispatcher

main = build_dispatcher(
    prog="culture-llm-tools",
    description="YuLan-OneSim (Wang et al. 2025) culture-dissemination visualization & analysis",
    subcommands={
        "visualize": (
            "single-run visualization (culture map + LC/GP time series)",
            "culture_llm_tools.visualize:main",
        ),
        "visualize-sweep": (
            "sweep visualization (F×q heatmap of n_stable_regions; classical vs LLM)",
            "culture_llm_tools.visualize_sweep:main",
        ),
        "show-experiment-settings": (
            "print a results directory's settings (config / sweep_config / run_metadata)",
            "culture_llm_tools.show_experiment_settings:main",
        ),
        "reproduce": (
            "Appendix F / Table 7-2 batch reproduction (observed-vs-paper LC/GP + figures)",
            "culture_llm_tools.reproduce_paper:main",
        ),
        "animate": (
            "intermediate culture-map animation / montage from --snapshot-interval grids",
            "culture_llm_tools.animate:main",
        ),
        "behavior-graph": (
            "render the behaviour-graph / ODD concept export (behavior_graph.json)",
            "culture_llm_tools.behavior_graph:main",
        ),
        "compare-report": (
            "classical vs LLM quantitative comparison figure (compare_report.json)",
            "culture_llm_tools.compare_report:main",
        ),
    },
)


if __name__ == "__main__":
    main()
