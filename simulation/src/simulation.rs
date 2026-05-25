//! Initialisation and run driver (SimulationBuilder wiring + two-layer LLM).
//!
//! Wires the two-layer determinism contract:
//! - **lower (deterministic socsim core)**: `derive_seed(root, &[0])` seeds the
//!   init RNG (random culture placement + persona assignment), `derive_seed(root,
//!   &[1])` seeds the engine RNG (= site/neighbour draws). Bit-reproducible.
//! - **upper (non-deterministic LLM)**: confined to [`crate::llm`]'s cached
//!   Ollama→OpenAI fallback client, pseudo-determinised via `temperature=0` /
//!   fixed `seed` + the prompt→response cache. Model / endpoint / temperature /
//!   seed / cache-hit are recorded into `run_metadata.json`.
//!
//! The driver picks **exactly one** interaction mechanism based on
//! `config.provider`: `none` → [`socsim_mechanisms::AxelrodMechanism`] (no
//! LLM), `ollama` / `openai` → [`LLMInteractionMechanism`].

use std::cell::RefCell;
use std::fs::File;
use std::io::BufWriter;
use std::rc::Rc;

use csv::Writer;
use serde::Serialize;

use socsim_core::{derive_seed, SimRng};
use socsim_engine::{RandomActivationScheduler, SimulationBuilder};
use socsim_llm::{LlmClient, MetadataCollector};
use socsim_mechanisms::AxelrodMechanism;

use crate::config::{Config, Provider};
use crate::llm::{build_live_client, CultureClient};
use crate::mechanisms::{
    ConvergenceMechanism, LLMInteractionMechanism, SharedClient, SharedMetadata,
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
    /// LLM call metadata (cache-hit rate etc.).
    pub metadata: MetadataCollector,
    /// LLM model name (run_metadata).
    pub llm_model: String,
    /// LLM endpoint (run_metadata; primary).
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

/// Run the simulation, building the production LLM client from environment
/// variables (only used when `provider` is an LLM provider).
pub fn run(cfg: &Config) -> Result<SimulationResult, String> {
    if cfg.provider.is_llm() {
        let client =
            build_live_client(&cfg.llm).map_err(|e| format!("LLM client build failed: {e}"))?;
        run_with_client(cfg, Some(client))
    } else {
        run_with_client(cfg, None)
    }
}

/// Run the simulation with an optional pre-built [`CultureClient`].
///
/// `client = None` runs the classical (no-LLM) path. `client = Some(_)` runs the
/// LLM path with the supplied client (production via [`build_live_client`], tests
/// via [`crate::llm::wrap_client`] over a `ScriptedClient`).
pub fn run_with_client(
    cfg: &Config,
    client: Option<CultureClient>,
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
                let model = c.inner().model().to_string();
                let endpoint = c.inner().endpoint().to_string();
                (model, endpoint, Some(Rc::new(RefCell::new(c))))
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
            )));
        }
        (_, None) => {
            return Err("LLM provider selected but no client supplied".to_string());
        }
    }
    builder = builder.add_mechanism(Box::new(ConvergenceMechanism));

    let mut sim = builder.build();

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
        converged = *report.scratch.get::<bool>("converged").unwrap_or(&false);
        final_round = t;
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

    Ok(SimulationResult {
        converged,
        final_round,
        round_history,
        final_metrics,
        world: final_world,
        metadata,
        llm_model,
        llm_endpoint,
    })
}

// --------------------------------------------------------------------------- //
// Output writers
// --------------------------------------------------------------------------- //

/// Create the output directory.
pub fn ensure_output_dir(output_dir: &str) {
    socsim_results::ensure_dir(output_dir).expect("failed to create output directory");
}

/// `metrics.csv` row (long-format, one row per round).
#[derive(Serialize)]
struct MetricsRow {
    round: usize,
    lc: f64,
    gp: f64,
    gp_per_agent: f64,
    n_stable_regions: usize,
    max_region_size: usize,
    n_distinct_cultures: usize,
}

/// Save the per-round metrics history as long-format CSV.
///
/// The write mechanism is delegated to `socsim_results::write_csv` (each row is
/// `serialize`d with a header on the first row — the standard csv-crate behaviour,
/// byte-equivalent to the former hand-rolled writer). The row struct
/// [`MetricsRow`] (including the `gp` / `gp_per_agent` columns) stays repo-local.
pub fn save_metrics(result: &SimulationResult, output_dir: &str) {
    let rows: Vec<MetricsRow> = result
        .round_history
        .iter()
        .map(|r| MetricsRow {
            round: r.round,
            lc: r.lc,
            gp: r.gp,
            gp_per_agent: r.gp_per_agent,
            n_stable_regions: r.n_stable_regions,
            max_region_size: r.max_region_size,
            n_distinct_cultures: r.n_distinct_cultures,
        })
        .collect();
    let path = format!("{output_dir}/metrics.csv");
    socsim_results::write_csv(&rows, &path).expect("failed to write metrics.csv");
}

/// Save the culture grid as CSV (one row per site: row, col, culture vector
/// joined by `-`). Mirrors the `axelrod1997` grid output for the culture-map viz.
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

/// `run_metadata.json` (LLM model / endpoint / temperature / seed / cache stats).
#[derive(Serialize)]
pub struct RunMetadataJson {
    pub provider: String,
    pub llm_model: String,
    pub llm_endpoint: String,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub total_calls: usize,
    pub cache_hits: usize,
    pub cache_hit_rate: f64,
    pub converged: bool,
    pub final_round: usize,
    pub determinism_note: &'static str,
}

/// Save `run_metadata.json`.
pub fn save_run_metadata(result: &SimulationResult, cfg: &Config, output_dir: &str) {
    let meta = RunMetadataJson {
        provider: cfg.provider.label().to_string(),
        llm_model: result.llm_model.clone(),
        llm_endpoint: result.llm_endpoint.clone(),
        llm_temperature: cfg.llm.temperature,
        llm_seed: cfg.llm.seed,
        total_calls: result.metadata.total(),
        cache_hits: result.metadata.cache_hits(),
        cache_hit_rate: result.metadata.cache_hit_rate(),
        converged: result.converged,
        final_round: result.final_round,
        determinism_note: "LLM output is outside socsim bit-reproducibility; the prompt->response \
                           cache (with temperature=0 and fixed seed) is the reproducibility \
                           mechanism. The socsim core (culture init, site/neighbour draws, \
                           scheduling, metrics) is deterministic given the seed. The classical \
                           provider makes zero LLM calls.",
    };
    // pretty-print JSON is delegated to `socsim_results::write_json`
    // (serde_json::to_writer_pretty + flush internally; byte-equivalent to the
    // former writer). The value sources (model / endpoint / temperature / seed
    // from result / cfg) and the `RunMetadataJson` shape are kept as-is —
    // `MetadataCollector::summary()` is deliberately NOT used because it can
    // change bytes on cache-hit / zero-call runs (classical makes 0 LLM calls).
    let path = format!("{output_dir}/run_metadata.json");
    socsim_results::write_json(&meta, &path).expect("failed to write run_metadata.json");
}

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
