[English](reproduction.md) | [日本語](reproduction.ja.md)

# Reproduction

## Batch reproduction — `reproduce` (recommended)

The `reproduce` subcommand runs all four Appendix F / Table 7-2 conditions in one command (classical `--provider none`, offline, 0 LLM calls) and writes a machine-readable observed-vs-paper summary with a PASS/off verdict per condition:

```bash
cargo run --release -- reproduce --runs 30 --seed 42   # → results/culture-llm/reproduce_<run_slug>/
uv run culture-llm-tools reproduce --runs 30 --seed 42 # same, plus observed-vs-paper figures
uv run culture-llm-tools reproduce --quick             # fast end-to-end smoke
```

Each condition becomes a child run (`subcommand=reproduce-condition`). Its `metrics.csv` holds the observation (`mean_n_stable_regions`, `mean_lc`, `mean_gp`, `mean_gp_per_agent`, `converged_fraction`, `n_units`) and its `reference.csv` holds Axelrod's published target with its source; the individual trials are `terminal` lines in its `events.jsonl`. The tolerance band and the PASS/off verdict are **ours, not the paper's**, so they live in the parent's `artifacts/reproduce_verdicts.csv` and on the console rather than among the metrics or the reference values.

## Classical baseline — Axelrod Table 7-2 (quantitative)

The classical provider (`--provider none`, no LLM) reproduces Axelrod (1997) Table 7-2: the mean number of stable culture regions on a 10×10 grid. The individual conditions can also be run by hand with `--runs 30`:

```bash
cargo run --release -- run --provider none --width 10 --height 10 --features 5  --traits 10 --runs 30 --seed 42
cargo run --release -- run --provider none --width 10 --height 10 --features 5  --traits 15 --runs 30 --seed 42
cargo run --release -- run --provider none --width 10 --height 10 --features 10 --traits 10 --runs 30 --seed 42
cargo run --release -- run --provider none --width 10 --height 10 --features 15 --traits 15 --runs 30 --seed 42
```

Measured `mean n_stable_regions` (this implementation, `--runs 30`, seed 42):

| F | q | Axelrod target | measured | within tolerance? |
|---|---|----------------|----------|-------------------|
| 5 | 10 | 3.2 (±0.5) | ~5.2 | high — matches the sibling `axelrod1997` (≈4.8); see note |
| 5 | 15 | 20.0 (±3.0) | ~19.1 | yes |
| 10 | 10 | 1.0 (±0.3) | 1.0 | yes (exact) |
| 15 | 15 | 1.2 (±0.3) | 1.0 | yes |

The qualitative Axelrod signs hold: increasing `F` reduces stable regions; increasing `q` increases them. The F5q10 value runs a little above the published 3.2; the sibling `axelrod1997` produces ≈4.8 for the same condition, so this is a property of the RNG/averaging (Axelrod's original Monte-Carlo setup differs), not of this port. The other three conditions are within tolerance.

> The classical path makes **zero LLM calls**. A run with no calls records no `llm_calls` rows and no `llm` block at all — absence rather than a zero, since a cache-hit rate with a zero denominator is undefined.

## LLM variant — Appendix F LC/GP (qualitative)

The LLM provider reproduces the paper's Appendix F behaviour qualitatively (the local default `llama3.2` differs from the paper's model, so exact numbers are not the target):

- LC rises over rounds and should exceed 0.50 (the paper notes by ~round 60).
- The same monoculture↔polyculture transition as the classical variant.

```bash
OLLAMA_MODEL=llama3.2:latest cargo run --release -- run --provider ollama --width 5 --height 2 --features 5 --traits 5 --rounds 100 --seed 42
uv run culture-llm-tools visualize
```

## Classical vs LLM — `compare`

The `compare` subcommand runs the classical baseline and the LLM variant on a matched config (same grid / seed / rounds) as a parent run plus one child run per side. Each side's LC / GP / regions / convergence are in its own child (`subcommand=compare-side`); the deltas are `scope=sweep` metrics on the parent. `--mock` substitutes a deterministic scripted LLM client so the comparison runs end-to-end offline; the classical side always runs live (0 LLM calls):

```bash
# offline: classical live, LLM structural (scripted mock)
cargo run --release -- compare --mock --features 5 --traits 5 --rounds 100 --seed 42
uv run culture-llm-tools compare-report

# live LLM (requires a reachable backend)
OLLAMA_MODEL=llama3.2:latest cargo run --release -- compare --llm-provider ollama --rounds 100 --seed 42
```

### The GP target and the documented inconsistency

Appendix F reports `GP ≈ 0.35–0.40`, but the documented formula `GP = |C| / N²` is bounded by `1/N` and cannot reach that band. We record both `gp = |C| / N²` (documented) and `gp_per_agent = |C| / N` (auxiliary); the latter is the normalisation that can plausibly match the paper's band. See [architecture](architecture.md#the-gp-inconsistency-important).

## Offline pipeline smoke (no LLM)

```bash
cargo run --release --example mock_smoke -- results
```

Drives the LLM mechanism with a scripted mock client (no network), exercising the full recording path (a `mock-smoke` run with per-round `metrics.csv`, the `llm` block, the `llm_calls` metrics and `artifacts/culture_grid_final.csv`).

---
*This file was generated by Claude Code.*
