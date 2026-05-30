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
- **Upper (non-deterministic LLM):** confined to `LLMInteractionMechanism` via `llm.rs`'s `CachingClient<Box<dyn LlmClient>>`. Production wires `FallbackClient<OllamaClient, OpenAiClient>` (Ollama first → OpenAI fallback); tests inject a `socsim_llm::mock::ScriptedClient`. `temperature=0` + fixed seed + the prompt→response cache replay identical responses on re-run. `run_metadata.json` records provider / model / endpoint / temperature / seed / cache-hit.

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

When `run --snapshot-interval N` is given with `N > 0`, the run driver clones the culture grid at round 0, every `N` rounds, and the final round into `SimulationResult::snapshots`, written to `snapshots/culture_grid_round_<NNNNNN>.csv` (+ `snapshots/index.json` listing the rounds). The Python `animate` tool renders these into a culture-map montage + GIF. With `N = 0` only the final grid is written (the default).

## Behaviour-graph / ODD concept export

YuLan-OneSim constructs, from a natural-language scenario, both an ODD-protocol document and an internal behaviour graph (agents → events → state updates) that the engine runs. Reproducing that *LLM-driven construction pipeline* is out of scope here. As a clearly-scoped **concept demo**, `run` instead runs that map in reverse: `odd::build_behavior_graph` derives — **deterministically, from the fixed `Config` + wired mechanisms, with no LLM** — a structured `behavior_graph.json` containing the seven ODD sections and a node/edge behaviour graph. The graph is variant-aware: the LLM variant adds persona / memory / `llm_decision` nodes mirroring exactly which mechanism the driver wires. The `behavior_graph.json` `provenance` field states honestly that this is a structured description of a fixed model, not an LLM-synthesised artefact. The Python `behavior-graph` tool renders it to a diagram.

## Reproduction & comparison harnesses

- `reproduce` runs the four Appendix F / Table 7-2 conditions (F5q10, F5q15, F10q10, F15q15) on a 10×10 grid (classical, offline, 0 LLM calls) and writes `reproduce_<ts>/reproduce_summary.json` (observed mean `n_stable_regions` vs the published target + a PASS/off verdict per condition + mean LC / GP / GP-per-agent) and `reproduce_detail.csv`. The Python `reproduce` tool renders the observed-vs-paper figures.
- `compare` runs the classical baseline and the LLM variant on a **matched** config (same grid / seed / rounds) and writes `compare_<ts>/compare_report.json` (both sides' metrics + deltas). `--mock` runs the LLM side with a deterministic scripted client so the whole comparison runs offline; live LLM numbers are pseudo-determinised by the prompt cache.

## Outputs

`results/{timestamp}/` (+ a `latest` symlink):

- `config.json` — run parameters.
- `metrics.csv` — long-format, one row per round: `round, lc, gp, gp_per_agent, n_stable_regions, max_region_size, n_distinct_cultures`.
- `culture_grid_final.csv` — final culture grid (`row, col, culture`) for the culture-map visualization.
- `run_metadata.json` — provider / model / endpoint / temperature / seed / cache-hit / converged / final_round.
- `behavior_graph.json` — the behaviour-graph / ODD concept export (every `run`).
- `snapshots/culture_grid_round_<NNNNNN>.csv` + `snapshots/index.json` — intermediate culture grids (only with `--snapshot-interval N > 0`).
- `sweep_summary.csv` + `sweep_config.json` — for the `sweep` subcommand.
- `reproduce_<ts>/{reproduce_summary.json, reproduce_detail.csv, figures/}` — for `reproduce`.
- `compare_<ts>/compare_report.json` — for `compare`.

---
*This file was generated by Claude Code.*
