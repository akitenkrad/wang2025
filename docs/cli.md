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
| `--output-dir` | `results` | output base directory |

Examples:

```bash
# classical Table 7-2 baseline (no LLM, 0 LLM calls)
cargo run --release -- run --provider none --width 10 --height 10 --features 5 --traits 10 --runs 30 --seed 42

# LLM variant, small grid, Ollama first
OLLAMA_MODEL=llama3.2:latest cargo run --release -- run --provider ollama --width 5 --height 2 --features 5 --traits 5 --rounds 100 --seed 42

# intermediate culture-grid snapshots (for the animation tool)
cargo run --release -- run --provider none --width 10 --height 10 --features 5 --traits 8 --snapshot-interval 50 --seed 42
```

Every `run` also writes `behavior_graph.json` (the behaviour-graph / ODD concept export, derived from the model). With `--snapshot-interval N > 0` it writes intermediate culture grids to `snapshots/culture_grid_round_<NNNNNN>.csv` (indexed by `snapshots/index.json`), always including round 0 and the final round.

## `sweep`

Sweep features `F` × traits `q` and aggregate `n_stable_regions` / LC / GP into `sweep_summary.csv`.

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
| `--output-dir` | `results` | output base directory |

```bash
cargo run --release -- sweep --provider none \
    --features-min 5 --features-max 15 --features-step 5 \
    --traits-min   5 --traits-max   15 --traits-step   5 \
    --runs 30 --seed 42
```

## `reproduce`

Appendix F / Axelrod Table 7-2 batch reproduction. Runs the four conditions (F5q10, F5q15, F10q10, F15q15) on a 10×10 grid for `--runs` runs each (classical `--provider none`, offline, 0 LLM calls), and writes `reproduce_summary.json` (observed mean `n_stable_regions` vs the published target, with a PASS/off verdict per condition + mean LC / GP / GP-per-agent) plus `reproduce_detail.csv` (per-condition per-run rows). The Python `culture-llm-tools reproduce` renders the observed-vs-paper figures into `figures/`.

| Flag | Default | Meaning |
|------|---------|---------|
| `--provider` | `none` | classical baseline (offline-verifiable) |
| `--runs` | `30` | runs per condition (averaged) |
| `--rounds` | `20000` | maximum engine ticks per run |
| `--seed` | `42` | seed base (each run derives an independent seed) |
| `--quick` | off | fast smoke (`runs=5`, `rounds ≤ 5000`) — not for validation |
| `--output-dir` | `results` | writes `reproduce_<ts>/` here |

```bash
cargo run --release -- reproduce --runs 30 --seed 42
cargo run --release -- reproduce --quick          # fast end-to-end smoke
```

## `compare`

Classical (`--provider none`) vs LLM quantitative comparison on a **matched** config (same grid / seed / rounds). Writes `compare_report.json` with both sides' headline metrics (`n_stable_regions`, LC, GP, GP-per-agent, convergence, LLM-call / cache-hit counts) and their deltas. Pass `--mock` to run the LLM side with a deterministic scripted client (no network), so the comparison runs end-to-end offline; without `--mock` the LLM side is the live env-built client.

| Flag | Default | Meaning |
|------|---------|---------|
| `--llm-provider` | `ollama` | LLM provider to compare against the classical baseline |
| `--mock` | off | deterministic scripted LLM client (offline; CI / sandbox) |
| `--width` / `--height` | `5` / `4` | matched grid size |
| `--features` / `--traits` | `5` / `5` | matched `F` / `q` |
| `--rounds` | `100` | maximum engine ticks |
| `--seed` | `42` | shared seed (both sides) |
| `--temperature` / `--llm-seed` / `--cache-path` | as `run` | LLM settings (live path) |
| `--output-dir` | `results` | writes `compare_<ts>/` here |

```bash
# offline (scripted mock LLM): classical live + LLM structural
cargo run --release -- compare --mock --features 5 --traits 5 --rounds 100 --seed 42

# live LLM (requires a reachable Ollama / OpenAI backend)
OLLAMA_MODEL=llama3.2:latest cargo run --release -- compare --llm-provider ollama --rounds 100 --seed 42
```

---
*This file was generated by Claude Code.*
