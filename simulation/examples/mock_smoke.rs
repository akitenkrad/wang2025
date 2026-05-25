//! Mock-driven smoke run (no live LLM).
//!
//! Exercises the **LLM interaction path** (`LLMInteractionMechanism`) using a
//! `socsim_llm::mock::ScriptedClient`, so the LLM pipeline + output writers can
//! be validated in a network-free sandbox (localhost:11434 blocked). The scripted
//! "model" decides adoption by echoing the first candidate feature index it is
//! offered, which drives cultures toward local convergence just like the real
//! homophily decision.
//!
//! ```bash
//! cargo run --release --example mock_smoke -- results
//! ```

use std::env;

use socsim_results::{refresh_latest_symlink, timestamp, write_json};

use culture_llm::config::{Config, LlmSettings, Provider};
use culture_llm::llm::wrap_client;
use culture_llm::simulation::{
    ensure_output_dir, run_with_client, save_culture_grid, save_metrics, save_run_metadata,
};
use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

fn main() {
    let base = env::args().nth(1).unwrap_or_else(|| "results".to_string());
    let timestamp = timestamp();
    let output_dir = format!("{base}/{timestamp}");

    let cfg = Config {
        width: 5,
        height: 4,
        features: 5,
        traits: 5,
        events_per_step: 0, // = n_sites
        rounds: 30,
        snapshot_interval: 0,
        provider: Provider::Ollama, // requested primary (mock client substitutes)
        seed: Some(42),
        llm: LlmSettings {
            temperature: 0.0,
            seed: 0,
            cache_path: None, // in-memory for the smoke
        },
        output_dir: output_dir.clone(),
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

    ensure_output_dir(&cfg.output_dir);
    let result = run_with_client(&cfg, Some(client)).expect("mock run failed");
    save_metrics(&result, &cfg.output_dir);
    save_culture_grid(&result.world, &cfg.output_dir, "culture_grid_final.csv");
    save_run_metadata(&result, &cfg, &cfg.output_dir);

    // config.json (delegated to socsim_results::write_json).
    let cfg_path = format!("{}/config.json", cfg.output_dir);
    write_json(&cfg.to_run_config_json(), &cfg_path).unwrap();

    // latest symlink (delegated to socsim_results).
    let _ = refresh_latest_symlink(&base, &timestamp);

    println!("mock smoke wrote: {output_dir}");
    println!(
        "final LC={:.3} GP={:.5} n_stable_regions={} converged={} rounds={} LLM_calls={}",
        result.final_metrics.local_convergence,
        result.final_metrics.global_polarization,
        result.final_metrics.n_stable_regions,
        result.converged,
        result.final_round,
        result.metadata.total(),
    );
}
