//! Mock-driven smoke run (no live LLM).
//!
//! Exercises the **LLM interaction path** (`LLMInteractionMechanism`) using a
//! `socsim_llm::mock::ScriptedClient`, so the LLM pipeline + the recording layer
//! can be validated in a network-free sandbox (localhost:11434 blocked). The
//! scripted "model" decides adoption by echoing the first candidate feature index
//! it is offered, which drives cultures toward local convergence just like the
//! real homophily decision.
//!
//! The results go where runvault puts them: `Run::start` names the run directory
//! under the results root given as the first argument.
//!
//! ```bash
//! cargo run --release --example mock_smoke -- results
//! ```

use std::env;

use runvault::{Run, RunOptions};
use serde::Serialize;

use culture_llm::config::{Config, LlmSettings, Provider};
use culture_llm::llm::wrap_client;
use culture_llm::record::{self, DOMAIN, EXPERIMENT, REPO_ID};
use culture_llm::simulation::{ensure_output_dir, run_with_client, save_culture_grid};
use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

/// The smoke's conditions. Same shape as the `run` subcommand's, minus the
/// options it does not take (there is one trial, and the mock client's cache is
/// in memory so there is no cache path to record).
#[derive(Serialize)]
struct SmokeParameters {
    provider: &'static str,
    width: usize,
    height: usize,
    features: usize,
    traits: usize,
    events_per_step: usize,
    rounds: usize,
    snapshot_interval: usize,
    runs: usize,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
}

fn main() {
    let results_root = env::args().nth(1).unwrap_or_else(|| "results".to_string());
    let seed = 42u64;

    let cfg = Config {
        width: 5,
        height: 4,
        features: 5,
        traits: 5,
        events_per_step: 0, // = n_sites
        rounds: 30,
        snapshot_interval: 0,
        provider: Provider::Ollama, // requested primary (mock client substitutes)
        seed: Some(seed),
        llm: LlmSettings {
            temperature: 0.0,
            seed: 0,
            cache_path: None, // in-memory for the smoke
        },
    };

    // Scripted "model": adopt the FIRST differing feature index offered in the
    // prompt's "[a, b, c]" candidate list. The prompt lists candidates as
    // "differ on these feature indices: [i, j, ...]". We parse the first integer
    // inside the first bracket pair after that phrase.
    let backend = ScriptedClient::new("mock-culture-llm", |prompt: &str| {
        if let Some(pos) = prompt.find("differ on these feature indices: [") {
            let tail = &prompt[pos..];
            if let Some(open) = tail.find('[') {
                let after = &tail[open + 1..];
                let first: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
                if !first.is_empty() {
                    return first; // adopt that feature
                }
            }
        }
        "-1".to_string() // adopt nothing
    });
    let client = wrap_client(backend, PromptCache::in_memory());
    // The client is built before the run starts: the model name and the endpoint
    // are what fill `run.json`'s `llm` block, and only the builder knows them.
    let model = client.inner().model().to_string();
    let endpoint = client.inner().endpoint().to_string();

    let parameters = SmokeParameters {
        provider: cfg.provider.label(),
        width: cfg.width,
        height: cfg.height,
        features: cfg.features,
        traits: cfg.traits,
        events_per_step: cfg.effective_events_per_step(),
        rounds: cfg.rounds,
        snapshot_interval: cfg.snapshot_interval,
        runs: 1,
        seed,
        llm_temperature: cfg.llm.temperature,
        llm_seed: cfg.llm.seed,
    };

    let mut rv = Run::start(
        RunOptions::new(EXPERIMENT, "mock-smoke")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&results_root)
            .parameters(&parameters)
            .expect("runvault: parameters の組み立てに失敗")
            .seed_pointers(["/seed"])
            .master_seed(seed)
            .llm(record::llm_block(&model, &endpoint, cfg.llm.temperature))
            .replication(record::replication()),
    )
    .expect("runvault: mock smoke の開始に失敗");

    let artifacts = rv.dir().join("artifacts").to_string_lossy().into_owned();
    ensure_output_dir(&artifacts);

    let result = run_with_client(&cfg, Some(client)).expect("mock run failed");

    record::log_round_metrics(&mut rv, &result.round_history);
    record::log_run_scope(&mut rv, &result);
    record::log_llm_metrics(&mut rv, &result.metadata);
    let observed: Vec<u64> = result
        .round_history
        .iter()
        .map(|r| r.round as u64)
        .collect();
    record::log_terminal(&mut rv, "run", seed, cfg.rounds, observed, &result);
    save_culture_grid(&result.world, &artifacts, "culture_grid_final.csv");

    println!(
        "final LC={:.3} GP={:.5} n_stable_regions={} converged={} rounds={} LLM_calls={}",
        result.final_metrics.local_convergence,
        result.final_metrics.global_polarization,
        result.final_metrics.n_stable_regions,
        result.converged,
        result.final_round,
        result.metadata.total(),
    );
    let dir = rv.finish().expect("runvault: mock smoke の完了に失敗");
    println!("mock smoke wrote: {}", dir.display());
}
