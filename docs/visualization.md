[English](visualization.md) | [日本語](visualization.ja.md)

# Visualization

The Python package `culture-llm-tools` (module `culture_llm_tools`) reads the Rust simulation outputs and renders figures with matplotlib / pandas / numpy / Pillow. Install at the workspace root with `uv sync`.

Which run a tool reads is asked of runvault: pass `--results-dir` (or `--sweep-dir` / `--compare-dir`) to name one, or leave it out and the tool calls `runvault path --experiment culture-llm --latest --subcommand …`. `results/` is never scanned for a directory that looks recent. The `runvault` binary is found on PATH or through the `RUNVAULT` environment variable.

Figures are written **beside** the run, in `<results-root>/culture-llm/figures/<run_slug>/`: `manifest.csv` is settled when the run finishes, so anything added to the run afterwards would carry no hash. Pre-runvault `results/<timestamp>/` directories are still read when passed explicitly.

## `visualize`

Single-run visualization (reads a `run` run directory; by default the latest standalone `run`).

```bash
uv run culture-llm-tools visualize
uv run culture-llm-tools visualize --results-dir "$(runvault path --experiment culture-llm --latest --subcommand run --standalone)" --output-dir out
```

Outputs:

- `culture_map.png` — the final culture grid (`artifacts/culture_grid_final.csv`), coloured by distinct culture profile.
- `lc_gp_timeseries.png` — LC and GP (and the auxiliary GP/N) per round, with the Appendix F LC target line (0.50) and the GP band (0.35–0.40).

## `visualize-sweep`

Sweep visualization. The one-row-per-trial table is rebuilt from the sweep parent's children (`subcommand=sweep-point`): a heatmap of *means over trials* needs the individual trials, which live in each child's `events.jsonl` rather than in the aggregate metrics. A pre-runvault `sweep_summary.csv` is read as it stands.

```bash
uv run culture-llm-tools visualize-sweep
```

Outputs:

- `sweep_heatmap.png` — F×q heatmap of mean `n_stable_regions` (one panel per provider; a classical-vs-LLM comparison when both are present).
- `sweep_lc_gp.png` — F×q heatmaps of mean LC and mean GP.

## `show-experiment-settings`

Pretty-prints the run's conditions (`config.json`'s `parameters`) plus its LLM provenance: the `llm` block from `run.json` and the `llm_calls` / `llm_cache_hits` / `llm_cache_hit_rate` metrics. A run that made no LLM calls has neither, and says so rather than printing zeroes. Which subcommand the run was is answered by `run.json`. Legacy flat `config.json` / `sweep_config.json` / `run_metadata.json` are still read.

```bash
uv run culture-llm-tools show-experiment-settings
uv run culture-llm-tools show-experiment-settings --results-dir "$(runvault path --experiment culture-llm --latest --subcommand sweep)"
```

## `reproduce`

Appendix F / Table 7-2 batch reproduction driver. Invokes the Rust `culture-llm reproduce` subcommand (classical, offline), then reads what it recorded: the observed means from each condition child's `metrics.csv`, the published targets from that child's `reference.csv`, and the tolerance band and PASS/off verdict from the parent's `artifacts/reproduce_verdicts.csv`. The band is declared once, in Rust; this tool reads it rather than restating it, so the two cannot drift apart.

```bash
uv run culture-llm-tools reproduce                                 # full (cargo + figures)
uv run culture-llm-tools reproduce --quick                          # fast smoke
uv run culture-llm-tools reproduce --results-dir "$(runvault path --experiment culture-llm --latest --subcommand reproduce)"  # figures only
```

Outputs (beside the parent run, under `figures/<run_slug>/`):

- `regions_vs_paper.png` — observed mean `n_stable_regions` vs the Axelrod Table 7-2 targets, per condition, with the PASS/off verdict.
- `lc_gp_by_condition.png` — mean LC and mean GP/N per condition, with the LC reference line and the Appendix F GP band (0.35–0.40).

## `animate`

Intermediate culture-map animation / montage from the `--snapshot-interval` grids (`artifacts/snapshots/`).

```bash
uv run culture-llm-tools animate --fps 4
```

Outputs:

- `culture_montage.png` — a grid of culture-map panels, one per snapshot round (always produced).
- `culture_animation.gif` — an animated culture map over rounds (Pillow; skipped with `--no-gif` or if Pillow is unavailable). A stable colour is assigned to each distinct culture across all frames.

## `behavior-graph`

Renders the behaviour-graph / ODD concept export (`artifacts/behavior_graph.json`, written by `run`). This is the visualization side of the **concept demo** of YuLan-OneSim's ODD / behaviour-graph auto-construction: the artefact is a faithful structured description of the fixed Axelrod scenario, derived from the model configuration — **not** an LLM-synthesised construction.

```bash
uv run culture-llm-tools behavior-graph --print-odd
```

Outputs:

- `behavior_graph.png` — nodes coloured by kind (agent / state / event / metric) with labelled edges; the LLM variant adds persona / memory / `llm_decision` nodes. `--print-odd` also prints the seven ODD-protocol sections.

## `compare-report`

Renders the classical-vs-LLM comparison figure from a `compare` parent run: each side's final metrics come from its child run (`subcommand=compare-side`), the differences from the parent's `scope=sweep` metrics. A pre-runvault `compare_report.json` is read as it stands.

```bash
uv run culture-llm-tools compare-report
uv run culture-llm-tools compare-report --compare-dir "$(runvault path --experiment culture-llm --latest --subcommand compare)"
```

Outputs:

- `compare_report.png` — grouped bars for `n_stable_regions`, LC, GP/N and `final_round`, classical vs LLM; the deltas table is printed to stdout.
