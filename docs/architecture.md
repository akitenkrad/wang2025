[English](architecture.md) | [日本語](architecture.ja.md)

# Architecture

## World state — `CultureWorld`

A fixed-site model: every cell is occupied, no agent moves, only the culture vector mutates. We therefore use the `socsim-grid::CellGrid<Culture>` + precomputed `Adjacency` pattern (identical to `axelrod1997`), not the occupancy-tracking `GridIndex`.

- `cells: CellGrid<Culture>` — `Culture = Vec<usize>`; cell value = culture vector, flat index `idx = r*cols + c`. The single source of truth.
- `adjacency: Adjacency` — CSR Von Neumann (4-neighbour) table, built once from `cells.grid()`.
- `n_features` (`F`), `n_traits` (`q`), `width`, `height`.
- LLM layer: `personas: BTreeMap<AgentId, String>`, `memory: BTreeMap<AgentId, Vec<String>>` (empty for the classical variant).
- `lc_history`, `gp_history` — per-round buffers pushed by the convergence mechanism.

`WorldState::agent_ids()` returns `0..width*height` as sorted `AgentId`s.

## Mechanisms × phases

| Mechanism | Phase | Role |
|-----------|-------|------|
| `ClassicalInteractionMechanism` | `Interaction` | Deterministic Axelrod baseline. 1 tick = `events_per_step` micro-events: pick site `s` + random neighbour `nb`, compute `sim`, with probability `sim` copy one differing feature from `nb`. No LLM. |
| `LLMInteractionMechanism` | `Interaction` | YuLan-OneSim variant. Same event-driven framing; the adoption decision (whether / which differing feature) is delegated to the LLM, given persona + own culture + neighbour culture. All LLM calls are confined here; site memory is updated. |
| `ConvergenceMechanism` | `PostStep` | Each step computes LC / GP and pushes them to the world history; detects the absorbing state (every adjacent pair `sim ∈ {0,1}`) and calls `request_stop`. |

`ClassicalInteractionMechanism` and `LLMInteractionMechanism` are **mutually exclusive**: the driver adds exactly one based on `config.provider`. This makes the two directly comparable on the same world and metrics.

## Update semantics

Event-driven (the standard form for Axelrod / voter models), matching YuLan-OneSim's asynchronous event-bus paradigm. One engine tick batches `events_per_step` micro-events (default = `n_sites`). Site selection uses `ctx.rng`, so results are scheduler-independent; the scheduler is `RandomActivationScheduler` by convention.

## RNG streams (determinism)

```text
RNG_WORLD_INIT = 0   // initial culture placement + persona assignment
RNG_ENGINE     = 1   // scheduler / engine / event site & neighbour draws
```

`init_rng = SimRng::from_seed(derive_seed(root, &[RNG_WORLD_INIT]))`; the engine seed is `derive_seed(root, &[RNG_ENGINE])`. Same convention as `axelrod1997` / `schelling1971`. The `run` subcommand derives a per-run seed `derive_seed(base, &[F, q, run])` so multi-run averages are reproducible.

## Two-layer determinism

- **Lower (deterministic socsim core):** culture init, site/neighbour draws, scheduling, metrics. Bit-reproducible given a seed.
- **Upper (non-deterministic LLM):** confined to `LLMInteractionMechanism` via `llm.rs`'s `CachingClient<Box<dyn LlmClient>>`. Production wires `FallbackClient<OllamaClient, OpenAiClient>` (Ollama first → OpenAI fallback); tests inject a `socsim_llm::mock::ScriptedClient`. `temperature=0` + fixed seed + the prompt→response cache replay identical responses on re-run. The run's `run.json` records provider / model / temperature in its `llm` block; the call count and cache-hit rate are run-scope metrics (`llm_calls` / `llm_cache_hits` / `llm_cache_hit_rate`). A run that made no LLM calls has no such rows and no `llm` block — a rate with a zero denominator is undefined, not zero.

## Equations

Culture vector: `c_s = (c_s^1, …, c_s^F)`, `c_s^i ∈ {0,…,q-1}`.

Similarity and classical interaction probability:

```text
sim(s, nb) = |{ i : c_s^i = c_nb^i }| / F
P(interact | s, nb) = sim(s, nb)
```

Absorbing condition: stable ⇔ for all adjacent `(s, nb)`, `sim(s, nb) ∈ {0, 1}`.

Appendix F validation metrics:

```text
LC = (1/|E|) Σ_{(i,j)∈E} |F_i ∩ F_j| / |F|       (local convergence; mean adjacent-pair similarity)
GP = |C| / N²                                     (global polarization, as documented)
```

where `E` is the set of adjacent agent pairs, `|C|` the number of distinct culture regions (connected clusters of identical culture), and `N` the total number of agents.

## The GP inconsistency (important)

The design doc documents `GP = |C| / N²`, but flags a known inconsistency: Appendix F reports `GP ≈ 0.35–0.40` at steady state, which **`|C| / N²` cannot reach** for any reasonable `N`. Since `|C| ≤ N`, `|C| / N²` is bounded by `1/N` (e.g. `N = 100` → at most `0.01`). The paper's `0.35–0.40` target therefore almost certainly uses a **different normalisation** — most plausibly `|C| / N` (regions per agent), which *can* sit in that band when many small regions persist.

We do **not** silently "fix" the formula. The simulation:

- implements `GP = |C| / N²` literally as the documented `gp` field/column, **and**
- additionally records `gp_per_agent = |C| / N` as an auxiliary column,

and the reproduction helper and visualizer compare both against the `0.35–0.40` band. See `metrics.rs::global_polarization` for the in-code comment.

## Intermediate snapshots

When `run --snapshot-interval N` is given with `N > 0`, the run driver clones the culture grid at round 0, every `N` rounds, and the final round into `SimulationResult::snapshots`, written to the run's `artifacts/snapshots/culture_grid_round_<NNNNNN>.csv` (+ `artifacts/snapshots/index.json` listing the rounds). The Python `animate` tool renders these into a culture-map montage + GIF. With `N = 0` only the final grid is written (the default).

## Behaviour-graph / ODD concept export

YuLan-OneSim constructs, from a natural-language scenario, both an ODD-protocol document and an internal behaviour graph (agents → events → state updates) that the engine runs. Reproducing that *LLM-driven construction pipeline* is out of scope here. As a clearly-scoped **concept demo**, `run` instead runs that map in reverse: `odd::build_behavior_graph` derives — **deterministically, from the fixed `Config` + wired mechanisms, with no LLM** — a structured `behavior_graph.json` containing the seven ODD sections and a node/edge behaviour graph. The graph is variant-aware: the LLM variant adds persona / memory / `llm_decision` nodes mirroring exactly which mechanism the driver wires. The `behavior_graph.json` `provenance` field states honestly that this is a structured description of a fixed model, not an LLM-synthesised artefact. The Python `behavior-graph` tool renders it to a diagram.

## Reproduction & comparison harnesses

- `reproduce` runs the four Appendix F / Table 7-2 conditions (F5q10, F5q15, F10q10, F15q15) on a 10×10 grid (classical, offline, 0 LLM calls) as a parent run plus **one child run per condition** (`subcommand=reproduce-condition`). Each child holds its trials as terminal events, its condition means as run-scope metrics (`mean_n_stable_regions` and friends), and Axelrod's published target — with its source — in its `reference.csv`. The tolerance band is *ours*, not the paper's, so it and the PASS/off verdict live in the parent's `artifacts/reproduce_verdicts.csv` instead. The Python `reproduce` tool renders the observed-vs-paper figures.
- `compare` runs the classical baseline and the LLM variant on a **matched** config (same grid / seed / rounds) as a parent plus **one child run per side** (`subcommand=compare-side`). Each side is a whole simulation of the same board differing only in mechanism, so each gets its own `metrics.csv` and its own `llm` block; putting both in one run would collide on the metric primary key and force one run to name two models. The differences between the sides are a cross-condition aggregate and sit once, on the parent, as `scope=sweep` metrics (`delta_lc` and friends). `--mock` runs the LLM side with a deterministic scripted client so the whole comparison runs offline; live LLM numbers are pseudo-determinised by the prompt cache.

## Outputs

Where the results go is [runvault](https://github.com/akitenkrad/rs-runvault)'s business: one execution is one run directory, `<results-root>/culture-llm/<run_slug>/`, named by `Run::start`. Nothing here makes a timestamped directory or a `latest` symlink of its own. Locate a run with `runvault path --experiment culture-llm --latest --subcommand <sub>`.

- `config.json` — an envelope; the conditions sit under `parameters`. `llm_cache_path` is a *place* rather than a condition, so it is excluded from `config_hash`.
- `run.json` — run identity: subcommand, seeds, lineage, the `llm` block, the replication metadata.
- `metrics.csv` — long form (`run_uid, step, step_unit, scope, name, value`). Per round (`step_unit=round`, `scope=run`): `lc`, `gp`, `gp_per_agent`, `n_stable_regions`, `max_region_size`, `n_distinct_cultures`. Without a step: `converged`, `final_round`, and on the LLM path `llm_calls` / `llm_cache_hits` / `llm_cache_hit_rate`.
- `events.jsonl` — one `terminal` line per simulation (`outcome` / `censored` / `budget` / `seed` + the final metrics) plus the `observation` lines saying when it was watched.
- `reference.csv` — the values the paper printed, with their source. Only the `reproduce` children have one.
- `artifacts/culture_grid_final.csv` — final culture grid (`row, col, culture`) for the culture-map visualization. A spatial snapshot with no time axis whose `culture` is a label, so a table rather than a metric.
- `artifacts/behavior_graph.json` — the behaviour-graph / ODD concept export (every `run`).
- `artifacts/snapshots/culture_grid_round_<NNNNNN>.csv` + `artifacts/snapshots/index.json` — intermediate culture grids (only with `--snapshot-interval N > 0`).
- `artifacts/reproduce_verdicts.csv` — the tolerance band and the PASS/off verdict (`reproduce` parent only). Neither a metric nor a reported value.
- `manifest.csv` / `status.json` — settled by `finish()`. Figures drawn afterwards go *beside* the run, in `<results-root>/culture-llm/figures/<run_slug>/`, so they cannot disagree with the manifest.

The subcommands map onto runs as: `run` → one run; `sweep` → a parent (`sweep`) + one child per `(F, q)` cell (`sweep-point`); `reproduce` → a parent + one child per condition (`reproduce-condition`); `compare` → a parent + one child per side (`compare-side`); `examples/mock_smoke` → one run (`mock-smoke`). A child never reuses the `run` subcommand name, so `runvault path --subcommand run` is unambiguous.

Result directories written before the migration to runvault (`results/<timestamp>/`) are not rewritten; pass one to a tool's `--results-dir` and it is read as it stands.
