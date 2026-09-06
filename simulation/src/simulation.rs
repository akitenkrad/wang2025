//! Initialisation and run driver (SimulationBuilder wiring + two-layer LLM).
//!
//! Wires the two-layer determinism contract:
//! - **lower (deterministic socsim core)**: `derive_seed(root, &[0])` seeds the
//!   init RNG (random culture placement + persona assignment), `derive_seed(root,
//!   &[1])` seeds the engine RNG (= site/neighbour draws). Bit-reproducible.
//! - **upper (non-deterministic LLM)**: confined to [`crate::llm`]'s cached
//!   Ollama→OpenAI fallback client, pseudo-determinised via `temperature=0` /
//!   fixed `seed` + the prompt→response cache. Model / provider / temperature go
//!   into the run's `llm` block and the call counts into its run-scope metrics
//!   (see [`crate::record`]); this module only carries them out of the run.
//!
//! The driver picks **exactly one** interaction mechanism based on
//! `config.provider`: `none` → [`socsim_mechanisms::AxelrodMechanism`] (no
//! LLM), `ollama` / `openai` → [`LLMInteractionMechanism`].

use std::cell::RefCell;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::rc::Rc;

use csv::Writer;

use socsim_core::{derive_seed, SimRng};
use socsim_engine::{RandomActivationScheduler, SimulationBuilder};
use socsim_llm::{LlmClient, MetadataCollector};
use socsim_mechanisms::AxelrodMechanism;

use crate::config::{Config, Provider};
use crate::llm::CultureClient;
use crate::mechanisms::{
    no_observer, ConvergenceMechanism, EventObserver, LLMInteractionMechanism, SharedClient,
    SharedMetadata,
};
use crate::metrics::{compute_metrics, RunMetrics};
use crate::world::CultureWorld;

/// RNG label for world initialisation (culture placement + persona assignment).
const RNG_WORLD_INIT: u64 = 0;
/// RNG label for the socsim engine (= site/neighbour draws in the mechanism).
const RNG_ENGINE: u64 = 1;

/// Persona templates (round-robin assignment; deterministic). LLM variant only.
pub const PERSONAS: [&str; 6] = [
    "a cautious traditionalist who values their heritage",
    "an open-minded cosmopolitan who enjoys new customs",
    "a pragmatic neighbour who adapts to those around them",
    "a proud individualist who resists changing their ways",
    "a sociable community-minded person who seeks common ground",
    "a curious learner forming their cultural identity",
];

/// Per-round metrics record (long-format CSV row material).
#[derive(Debug, Clone)]
pub struct RoundMetrics {
    /// Round index (engine tick; `t=0` is the initial state).
    pub round: usize,
    /// Local convergence LC.
    pub lc: f64,
    /// Global polarization GP (documented `|C| / N²`).
    pub gp: f64,
    /// Auxiliary `|C| / N`.
    pub gp_per_agent: f64,
    /// Stable region count at this round.
    pub n_stable_regions: usize,
    /// Largest region size.
    pub max_region_size: usize,
    /// Distinct culture count.
    pub n_distinct_cultures: usize,
}

/// One intermediate culture-grid snapshot (round + the board at that round).
///
/// Collected during the run when `config.snapshot_interval > 0` so the Python
/// tools can render an intermediate culture-map animation / montage instead of
/// only the final grid. The initial state (round 0) and the final round are
/// always included.
#[derive(Clone)]
pub struct GridSnapshot {
    /// Engine round (tick) at which the snapshot was taken (`0` = initial state).
    pub round: usize,
    /// The world (culture grid) at that round.
    pub world: CultureWorld,
}

/// Result of a single run.
pub struct SimulationResult {
    /// Whether the absorbing (stable) state was reached.
    pub converged: bool,
    /// Final round number.
    pub final_round: usize,
    /// Per-round metrics history (including the initial state at round 0).
    pub round_history: Vec<RoundMetrics>,
    /// Final aggregate metrics.
    pub final_metrics: RunMetrics,
    /// Final board state.
    pub world: CultureWorld,
    /// Intermediate culture-grid snapshots (empty unless `snapshot_interval > 0`).
    /// Always includes round 0 (initial) and the final round.
    pub snapshots: Vec<GridSnapshot>,
    /// LLM call metadata (cache-hit rate etc.).
    pub metadata: MetadataCollector,
    /// LLM model name (goes into the run's `llm` block).
    pub llm_model: String,
    /// LLM endpoint (primary; classifies the provider in the `llm` block).
    pub llm_endpoint: String,
}

/// Initialise the world (random culture placement + persona assignment for the
/// LLM variant). All draws use the supplied init RNG (deterministic core layer).
pub fn init_world(cfg: &Config, rng: &mut SimRng) -> CultureWorld {
    let mut world = CultureWorld::random_init(
        cfg.width,
        cfg.height,
        cfg.features,
        cfg.traits,
        rng,
        cfg.rounds as u64,
    );
    if cfg.provider.is_llm() {
        world.assign_personas(&PERSONAS);
    }
    world
}

/// Run the simulation with an optional pre-built [`CultureClient`].
///
/// `client = None` runs the classical (no-LLM) path. `client = Some(_)` runs the
/// LLM path with the supplied client (production via
/// [`crate::llm::build_live_client`], tests via [`crate::llm::wrap_client`] over
/// a `ScriptedClient`).
///
/// There is deliberately **no** entry point that builds the client itself: the
/// model name and the endpoint are known only to whoever built it, and they are
/// what fills `run.json`'s `llm` block. An entry point that hid the construction
/// would let a run be recorded with that block empty.
pub fn run_with_client(
    cfg: &Config,
    client: Option<CultureClient>,
) -> Result<SimulationResult, String> {
    run_with_shared_client(cfg, client.map(|c| Rc::new(RefCell::new(c))))
}

/// The same, with a client that outlives one run.
///
/// A driver that runs several trials in a row builds the client once and hands
/// the same handle to each: the prompt cache then stays warm across trials
/// exactly as it did when every trial rebuilt the client from the same
/// file-backed cache.
pub fn run_with_shared_client(
    cfg: &Config,
    client: Option<SharedClient>,
) -> Result<SimulationResult, String> {
    run_with_shared_client_observed(cfg, client, |_| {}, no_observer())
}

/// The same, calling `on_round` once per engine tick and `on_event` once per
/// adoption decision the LLM is asked for.
///
/// Only a caller that reports progress uses this one. The two are separate
/// because the cost sits in a different place on each path: the classical rule
/// is the round, the LLM rule is the single decision inside it. Neither
/// callback can be one closure — `on_round` is called from this function and
/// may borrow the caller's `Stage`, while `on_event` is called from inside a
/// mechanism, which enters the engine as a `'static` box and can only be
/// handed a shared one.
pub fn run_with_shared_client_observed(
    cfg: &Config,
    client: Option<SharedClient>,
    mut on_round: impl FnMut(usize),
    on_event: EventObserver,
) -> Result<SimulationResult, String> {
    let root = cfg.seed.unwrap_or_else(rand::random);
    let events_per_step = cfg.effective_events_per_step();

    // Lower layer: deterministic init RNG.
    let mut init_rng = SimRng::from_seed(derive_seed(root, &[RNG_WORLD_INIT]));
    let world = init_world(cfg, &mut init_rng);

    // Metadata sharing (LLM path).
    let shared_meta: SharedMetadata = Rc::new(RefCell::new(MetadataCollector::new()));
    let (llm_model, llm_endpoint, shared_client): (String, String, Option<SharedClient>) =
        match client {
            Some(c) => {
                let (model, endpoint) = {
                    let borrowed = c.borrow();
                    (
                        borrowed.inner().model().to_string(),
                        borrowed.inner().endpoint().to_string(),
                    )
                };
                (model, endpoint, Some(c))
            }
            None => ("none".to_string(), "none".to_string(), None),
        };

    let mut builder = SimulationBuilder::new(world)
        .scheduler(Box::new(RandomActivationScheduler))
        .seed(derive_seed(root, &[RNG_ENGINE]));

    // Pick exactly one interaction mechanism by provider.
    match (cfg.provider, &shared_client) {
        (Provider::None, _) => {
            // Classical (deterministic Axelrod) path now uses the reusable pack
            // mechanism (ported from this repo's ClassicalInteractionMechanism);
            // CultureWorld supplies the CultureVectors + Neighbors capabilities.
            builder = builder.add_mechanism(Box::new(AxelrodMechanism::new(events_per_step)));
        }
        (_, Some(sc)) => {
            builder = builder.add_mechanism(Box::new(LLMInteractionMechanism::new(
                Rc::clone(sc),
                Rc::clone(&shared_meta),
                cfg.llm.clone(),
                events_per_step,
                Rc::clone(&on_event),
            )));
        }
        (_, None) => {
            return Err("LLM provider selected but no client supplied".to_string());
        }
    }
    builder = builder.add_mechanism(Box::new(ConvergenceMechanism));

    let mut sim = builder.build();

    // Snapshot cadence: 0 disables intermediate snapshots (final grid only).
    let snapshot_interval = cfg.snapshot_interval;
    let mut snapshots: Vec<GridSnapshot> = Vec::new();

    // Initial-state metrics (round 0).
    let mut round_history: Vec<RoundMetrics> = Vec::new();
    {
        let m = compute_metrics(sim.world());
        round_history.push(RoundMetrics {
            round: 0,
            lc: m.local_convergence,
            gp: m.global_polarization,
            gp_per_agent: m.gp_per_agent,
            n_stable_regions: m.n_stable_regions,
            max_region_size: m.max_region_size,
            n_distinct_cultures: m.n_distinct_cultures,
        });
        // Round 0 is always captured when snapshots are enabled.
        if snapshot_interval > 0 {
            snapshots.push(GridSnapshot {
                round: 0,
                world: sim.world().clone(),
            });
        }
    }

    let mut converged = false;
    let mut final_round = 0usize;

    sim.run_observed(|report| {
        let t = report.t as usize;
        let m = compute_metrics(report.world);
        round_history.push(RoundMetrics {
            round: t,
            lc: m.local_convergence,
            gp: m.global_polarization,
            gp_per_agent: m.gp_per_agent,
            n_stable_regions: m.n_stable_regions,
            max_region_size: m.max_region_size,
            n_distinct_cultures: m.n_distinct_cultures,
        });
        // Intermediate snapshot at every `snapshot_interval` rounds.
        if snapshot_interval > 0 && t.is_multiple_of(snapshot_interval) {
            snapshots.push(GridSnapshot {
                round: t,
                world: report.world.clone(),
            });
        }
        converged = *report.scratch.get::<bool>("converged").unwrap_or(&false);
        final_round = t;
        on_round(t);
    })
    .map_err(|e| format!("simulation run failed: {e}"))?;

    // Persist cache (LLM path, file-backed only).
    if let Some(sc) = &shared_client {
        if cfg.llm.cache_path.is_some() {
            sc.borrow()
                .cache()
                .save()
                .map_err(|e| format!("cache save failed: {e}"))?;
        }
    }

    let final_world = sim.world().clone();
    let final_metrics = compute_metrics(&final_world);
    let metadata = shared_meta.borrow().clone();

    // Always include the final round as the last snapshot (deduped if the final
    // round already fell on a snapshot boundary).
    if snapshot_interval > 0 && snapshots.last().map(|s| s.round) != Some(final_round) {
        snapshots.push(GridSnapshot {
            round: final_round,
            world: final_world.clone(),
        });
    }

    Ok(SimulationResult {
        converged,
        final_round,
        round_history,
        final_metrics,
        world: final_world,
        snapshots,
        metadata,
        llm_model,
        llm_endpoint,
    })
}

// --------------------------------------------------------------------------- //
// Output writers
// --------------------------------------------------------------------------- //

/// Create the output directory (the run's `artifacts/`).
pub fn ensure_output_dir(output_dir: &str) {
    std::fs::create_dir_all(output_dir).expect("failed to create output directory");
}

/// Write a serializable value as pretty-printed JSON.
pub fn write_json<T: serde::Serialize + ?Sized>(value: &T, path: &Path) {
    let file =
        File::create(path).unwrap_or_else(|e| panic!("failed to create {}: {e}", path.display()));
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)
        .unwrap_or_else(|e| panic!("failed to write {}: {e}", path.display()));
    use std::io::Write;
    writer
        .flush()
        .unwrap_or_else(|e| panic!("failed to flush {}: {e}", path.display()));
}

/// Save the culture grid as CSV (one row per site: row, col, culture vector
/// joined by `-`). Mirrors the `axelrod1997` grid output for the culture-map viz.
///
/// This is a **spatial snapshot**: one row per site, keyed by `(row, col)` with no
/// time axis, and `culture` is a label (`"0-1-2"`), not a number. It is therefore
/// a table rather than a metric, and lives under the run's `artifacts/`.
pub fn save_culture_grid(world: &CultureWorld, output_dir: &str, name: &str) {
    let path = format!("{output_dir}/{name}");
    let file = File::create(&path).expect("failed to create culture grid CSV");
    let mut wtr = Writer::from_writer(BufWriter::new(file));
    wtr.write_record(["row", "col", "culture"])
        .expect("header write failed");
    let cols = world.width();
    for idx in 0..world.n_sites() {
        let r = idx / cols;
        let c = idx % cols;
        let culture = world
            .culture(idx)
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join("-");
        wtr.write_record(&[r.to_string(), c.to_string(), culture])
            .expect("record write failed");
    }
    wtr.flush().expect("flush failed");
}

/// Save the intermediate culture-grid snapshots as
/// `snapshots/culture_grid_round_<NNNNNN>.csv` (zero-padded round), plus a
/// `snapshots/index.json` listing the rounds. No-op when there are no snapshots
/// (`snapshot_interval == 0`). The Python `animate` tool renders these into a
/// culture-map montage / GIF.
///
/// `output_dir` is the run's `artifacts/`, so the snapshots sit under
/// `artifacts/snapshots/` and are hashed into `manifest.csv` by `finish()`.
pub fn save_snapshots(result: &SimulationResult, output_dir: &str) -> Vec<usize> {
    if result.snapshots.is_empty() {
        return Vec::new();
    }
    let snap_dir = format!("{output_dir}/snapshots");
    ensure_output_dir(&snap_dir);
    let mut rounds = Vec::with_capacity(result.snapshots.len());
    for snap in &result.snapshots {
        let name = format!("culture_grid_round_{:06}.csv", snap.round);
        save_culture_grid(&snap.world, &snap_dir, &name);
        rounds.push(snap.round);
    }
    let index = serde_json::json!({
        "rounds": rounds,
        "n_snapshots": rounds.len(),
        "pattern": "culture_grid_round_{round:06}.csv",
    });
    write_json(&index, Path::new(&format!("{snap_dir}/index.json")));
    rounds
}

/// How the LLM layer is pinned down, for the docs and the CLI banner.
///
/// Not a number and not a condition, so it is neither a metric nor a parameter;
/// the numbers it talks about live in the run-scope `llm_calls` /
/// `llm_cache_hits` / `llm_cache_hit_rate` metrics and in `run.json`'s `llm`
/// block.
pub const DETERMINISM_NOTE: &str =
    "LLM output is outside socsim bit-reproducibility; the prompt->response cache (with \
     temperature=0 and fixed seed) is the reproducibility mechanism. The socsim core (culture \
     init, site/neighbour draws, scheduling, metrics) is deterministic given the seed. The \
     classical provider makes zero LLM calls.";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Provider;

    fn classical_cfg() -> Config {
        Config {
            width: 6,
            height: 6,
            features: 5,
            traits: 8,
            rounds: 5_000,
            provider: Provider::None,
            seed: Some(42),
            ..Config::default()
        }
    }

    #[test]
    fn classical_run_is_deterministic() {
        let cfg = classical_cfg();
        let a = run_with_client(&cfg, None).unwrap();
        let b = run_with_client(&cfg, None).unwrap();
        assert_eq!(a.final_round, b.final_round);
        assert_eq!(
            a.world.cells.cells(),
            b.world.cells.cells(),
            "same seed must reproduce the board exactly"
        );
        assert_eq!(a.metadata.total(), 0, "classical path makes 0 LLM calls");
    }
}
