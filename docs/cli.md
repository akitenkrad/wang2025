[English](cli.md) | [日本語](cli.ja.md)

# CLI reference

Binary: `culture-llm`. Subcommands: `run`, `sweep`, `reproduce`, `compare`.

## `run`

Run a single configuration (classical or LLM interaction).

| Flag | Default | Meaning |
|------|---------|---------|
| `--provider` | `none` | `none` (classical, no LLM) / `ollama` / `openai` |
| `--width` | `10` | grid width (columns) |
| `--height` | `10` | grid height (rows) |
| `--features`, `-f` | `5` | number of features `F` |
| `--traits`, `-q` | `10` | number of traits `q` |
| `--runs` | `1` | independent runs (classical: averaged; LLM: typically 1) |
| `--rounds` | `20000` | maximum engine ticks |
| `--events-per-step` | `0` | micro-events per tick (`0` = n_sites) |
| `--snapshot-interval` | `0` | intermediate culture-grid snapshot interval in rounds (`0` = final only) |
| `--seed` | (random) | random seed (governs the socsim core layer) |
| `--temperature` | `0.0` | LLM generation temperature |
| `--llm-seed` | `0` | LLM generation seed (backend) |
| `--cache-path` | `.llm_cache/cache.json` | prompt→response cache (LLM path) |
| `--output-dir` | `results` | results root (runvault writes `<root>/culture-llm/<run_slug>/`) |

Examples:

```bash
# classical Table 7-2 baseline (no LLM, 0 LLM calls)
cargo run --release -- run --provider none --width 10 --height 10 --features 5 --traits 10 --runs 30 --seed 42

# LLM variant, small grid, Ollama first
OLLAMA_MODEL=llama3.2:latest cargo run --release -- run --provider ollama --width 5 --height 2 --features 5 --traits 5 --rounds 100 --seed 42

# intermediate culture-grid snapshots (for the animation tool)
cargo run --release -- run --provider none --width 10 --height 10 --features 5 --traits 8 --snapshot-interval 50 --seed 42
```

Every `run` also writes `artifacts/behavior_graph.json` (the behaviour-graph / ODD concept export, derived from the model). With `--snapshot-interval N > 0` it writes intermediate culture grids to `artifacts/snapshots/culture_grid_round_<NNNNNN>.csv` (indexed by `artifacts/snapshots/index.json`), always including round 0 and the final round.

`--runs N` runs N simulations and records the **last** one in detail, exactly as before. That makes `runs` part of the conditions (it decides which trial is kept), so it is recorded in `parameters`; `master_seed` is the seed that actually governed that trial and `replicate_index` is `N-1`, while the root seed given on the command line stays in `/parameters.seed`. Omitting `--seed` no longer loses the seed: one is drawn, recorded, and used.

## `sweep`

Sweep features `F` × traits `q`. One sweep parent run (`subcommand=sweep`) holding the grid definition, plus **one child run per `(F, q)` cell** (`subcommand=sweep-point`). Each cell's `runs` trials are one `terminal` line each in the child's `events.jsonl`, and the cell's aggregates (`n_units`, `mean_n_stable_regions`, `mean_lc`, …) are its run-scope metrics. There is no `sweep_summary.csv`: the one-row-per-trial table is rebuilt from the children by `culture-llm-tools visualize-sweep`.

| Flag | Default | Meaning |
|------|---------|---------|
| `--provider` | `none` | classical / LLM |
| `--width` / `--height` | `10` / `10` | grid size |
| `--features-min/max/step` | `5` / `15` / `5` | feature range (inclusive) |
| `--traits-min/max/step` | `5` / `15` / `5` | trait range (inclusive) |
| `--runs` | `30` | runs per `(F, q)` |
| `--rounds` | `20000` | maximum engine ticks |
| `--events-per-step` | `0` | micro-events per tick (`0` = n_sites) |
| `--snapshot-interval` | `10` | recorded in sweep_config |
| `--seed` | `42` | seed base (each run derives an independent seed) |
| `--temperature` / `--llm-seed` / `--cache-path` | as `run` | LLM settings |
| `--output-dir` | `results` | results root (runvault writes `<root>/culture-llm/<run_slug>/`) |

```bash
cargo run --release -- sweep --provider none \
    --features-min 5 --features-max 15 --features-step 5 \
    --traits-min   5 --traits-max   15 --traits-step   5 \
    --runs 30 --seed 42
```

## `reproduce`

Appendix F / Axelrod Table 7-2 batch reproduction. Runs the four conditions (F5q10, F5q15, F10q10, F15q15) on a 10×10 grid for `--runs` runs each (classical `--provider none`, offline, 0 LLM calls), as a parent run plus **one child run per condition** (`subcommand=reproduce-condition`). Each child keeps its trials as `terminal` lines, its condition means as run-scope metrics (`mean_n_stable_regions`, `mean_lc`, …), and Axelrod's published target with its source in `reference.csv`. The tolerance band and the PASS/off verdict are ours rather than the paper's, so they go to the parent's `artifacts/reproduce_verdicts.csv` and the console. The Python `culture-llm-tools reproduce` renders the observed-vs-paper figures beside the parent.

| Flag | Default | Meaning |
|------|---------|---------|
| `--provider` | `none` | classical baseline (offline-verifiable) |
| `--runs` | `30` | runs per condition (averaged) |
| `--rounds` | `20000` | maximum engine ticks per run |
| `--seed` | `42` | seed base (each run derives an independent seed) |
| `--quick` | off | fast smoke (`runs=5`, `rounds ≤ 5000`) — not for validation |
| `--output-dir` | `results` | results root |

```bash
cargo run --release -- reproduce --runs 30 --seed 42
cargo run --release -- reproduce --quick          # fast end-to-end smoke
```

## `compare`

Classical (`--provider none`) vs LLM quantitative comparison on a **matched** config (same grid / seed / rounds). A parent run (`subcommand=compare`) plus **one child run per side** (`subcommand=compare-side`): each side is a whole simulation of the same board differing only in mechanism, so each carries its own per-round `metrics.csv`, its own `terminal` line and — on the LLM side — its own `llm` block. The differences (`delta_n_stable_regions`, `delta_lc`, `delta_gp`, `delta_gp_per_agent`, `delta_final_round`) sit once, on the parent, as `scope=sweep` metrics. Pass `--mock` to run the LLM side with a deterministic scripted client (no network), so the comparison runs end-to-end offline; without `--mock` the LLM side is the live env-built client.

| Flag | Default | Meaning |
|------|---------|---------|
| `--llm-provider` | `ollama` | LLM provider to compare against the classical baseline |
| `--mock` | off | deterministic scripted LLM client (offline; CI / sandbox) |
| `--width` / `--height` | `5` / `4` | matched grid size |
| `--features` / `--traits` | `5` / `5` | matched `F` / `q` |
| `--rounds` | `100` | maximum engine ticks |
| `--seed` | `42` | shared seed (both sides) |
| `--temperature` / `--llm-seed` / `--cache-path` | as `run` | LLM settings (live path) |
| `--output-dir` | `results` | results root |

```bash
# offline (scripted mock LLM): classical live + LLM structural
cargo run --release -- compare --mock --features 5 --traits 5 --rounds 100 --seed 42

# live LLM (requires a reachable Ollama / OpenAI backend)
OLLAMA_MODEL=llama3.2:latest cargo run --release -- compare --llm-provider ollama --rounds 100 --seed 42
```
