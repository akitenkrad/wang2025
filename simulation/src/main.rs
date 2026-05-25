//! YuLan-OneSim (Wang et al. 2025) — Axelrod culture-dissemination CLI.
//!
//! `run`       : single configuration; classical (`--provider none`) or LLM
//!               (`--provider ollama|openai`) interaction on the same world.
//! `sweep`     : sweep features `F` × traits `q` (classical or LLM), aggregating
//!               `n_stable_regions` and LC/GP into `sweep_summary.csv`.
//! `reproduce` : Appendix F LC/GP reproduction helper (Phase 3 stub — see below).

use std::fs::{self, File};
use std::io::BufWriter;
use std::path::Path;

use chrono::Local;
use clap::{Parser, Subcommand};
use csv::Writer;

use culture_llm::config::{parse_provider, Config, LlmSettings};
use culture_llm::simulation::{
    ensure_output_dir, run, save_culture_grid, save_metrics, save_run_metadata, SimulationResult,
};

use socsim_core::derive_seed;

// --------------------------------------------------------------------------- //
// CLI
// --------------------------------------------------------------------------- //

#[derive(Parser, Debug)]
#[command(
    name = "culture-llm",
    about = "YuLan-OneSim (Wang et al. 2025) — Axelrod culture dissemination (classical vs LLM)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run a single configuration (classical or LLM interaction).
    Run(RunArgs),
    /// Sweep features F × traits q and aggregate stable-region / LC / GP metrics.
    Sweep(SweepArgs),
    /// Appendix F LC/GP reproduction helper (Phase 3 stub).
    Reproduce(ReproduceArgs),
}

#[derive(Parser, Debug)]
struct RunArgs {
    /// Interaction provider (none / ollama / openai).
    #[arg(long, default_value = "none")]
    provider: String,
    /// Grid width.
    #[arg(long, default_value_t = 10)]
    width: usize,
    /// Grid height.
    #[arg(long, default_value_t = 10)]
    height: usize,
    /// Number of features F.
    #[arg(long, short = 'f', default_value_t = 5)]
    features: usize,
    /// Number of traits q.
    #[arg(long, short = 'q', default_value_t = 10)]
    traits: usize,
    /// Number of independent runs (classical: averaged; LLM: typically 1).
    #[arg(long, default_value_t = 1)]
    runs: usize,
    /// Maximum number of engine ticks (rounds).
    #[arg(long, default_value_t = 20_000)]
    rounds: usize,
    /// Micro-events per engine tick (0 = n_sites).
    #[arg(long, default_value_t = 0)]
    events_per_step: usize,
    /// Snapshot interval in rounds for intermediate culture grids (0 = final only).
    #[arg(long, default_value_t = 0)]
    snapshot_interval: usize,
    /// Random seed (governs the socsim core layer only).
    #[arg(long)]
    seed: Option<u64>,
    /// LLM generation temperature (default 0.0).
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    /// LLM generation seed (passed to the backend).
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,
    /// Prompt → response cache path (LLM path; default .llm_cache/cache.json).
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,
    /// Output base directory.
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct SweepArgs {
    /// Interaction provider (none / ollama / openai).
    #[arg(long, default_value = "none")]
    provider: String,
    /// Grid width.
    #[arg(long, default_value_t = 10)]
    width: usize,
    /// Grid height.
    #[arg(long, default_value_t = 10)]
    height: usize,
    /// Features F: start.
    #[arg(long, default_value_t = 5)]
    features_min: usize,
    /// Features F: end (inclusive).
    #[arg(long, default_value_t = 15)]
    features_max: usize,
    /// Features F: step.
    #[arg(long, default_value_t = 5)]
    features_step: usize,
    /// Traits q: start.
    #[arg(long, default_value_t = 5)]
    traits_min: usize,
    /// Traits q: end (inclusive).
    #[arg(long, default_value_t = 15)]
    traits_max: usize,
    /// Traits q: step.
    #[arg(long, default_value_t = 5)]
    traits_step: usize,
    /// Runs per (F, q) combination.
    #[arg(long, default_value_t = 30)]
    runs: usize,
    /// Maximum number of engine ticks (rounds).
    #[arg(long, default_value_t = 20_000)]
    rounds: usize,
    /// Micro-events per engine tick (0 = n_sites).
    #[arg(long, default_value_t = 0)]
    events_per_step: usize,
    /// Snapshot interval in rounds (passthrough; recorded in config).
    #[arg(long, default_value_t = 10)]
    snapshot_interval: usize,
    /// Random seed base.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// LLM generation temperature.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    /// LLM generation seed.
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,
    /// Prompt → response cache path (LLM path; shared across the sweep).
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,
    /// Output base directory.
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct ReproduceArgs {
    /// Output base directory.
    #[arg(long, default_value = "results")]
    output_dir: String,
}

// --------------------------------------------------------------------------- //
// CSV rows
// --------------------------------------------------------------------------- //

#[derive(serde::Serialize)]
struct SweepRow {
    provider: String,
    features: usize,
    traits: usize,
    run: usize,
    width: usize,
    height: usize,
    seed: u64,
    converged: bool,
    final_round: usize,
    n_stable_regions: usize,
    max_region_size: usize,
    n_distinct_cultures: usize,
    lc: f64,
    gp: f64,
    gp_per_agent: f64,
}

// --------------------------------------------------------------------------- //
// helpers
// --------------------------------------------------------------------------- //

/// Refresh the `latest` symlink (Unix only).
fn refresh_latest(output_dir: &str, target: &str) {
    let symlink_path = Path::new(output_dir).join("latest");
    if symlink_path.is_symlink() || symlink_path.exists() {
        let _ = fs::remove_file(&symlink_path);
    }
    #[cfg(unix)]
    {
        let _ = std::os::unix::fs::symlink(target, &symlink_path);
    }
}

/// Enumerate an inclusive range with a step (min..=max).
fn enumerate_range(min: usize, max: usize, step: usize) -> Vec<usize> {
    let mut v = Vec::new();
    let mut x = min;
    while x <= max {
        v.push(x);
        x += step.max(1);
    }
    v
}

fn make_llm_settings(temperature: f32, llm_seed: u64, cache_path: &str) -> LlmSettings {
    LlmSettings {
        temperature,
        seed: llm_seed,
        cache_path: Some(cache_path.to_string()),
    }
}

// --------------------------------------------------------------------------- //
// run
// --------------------------------------------------------------------------- //

fn cmd_run(args: RunArgs) {
    let provider = parse_provider(&args.provider).unwrap_or_else(|e| panic!("{e}"));
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let output_dir = format!("{}/{}", args.output_dir, timestamp);
    ensure_output_dir(&output_dir);
    if provider.is_llm() {
        if let Some(parent) = Path::new(&args.cache_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
    }

    let base_cfg = Config {
        width: args.width,
        height: args.height,
        features: args.features,
        traits: args.traits,
        events_per_step: args.events_per_step,
        rounds: args.rounds,
        snapshot_interval: args.snapshot_interval,
        provider,
        seed: args.seed,
        llm: make_llm_settings(args.temperature, args.llm_seed, &args.cache_path),
        output_dir: output_dir.clone(),
    };

    println!("=== YuLan-OneSim (Wang et al. 2025) — Axelrod culture dissemination ===");
    println!(
        "provider: {} | grid: {}×{} | F={} | q={} | runs={} | rounds={}",
        provider.label(),
        base_cfg.width,
        base_cfg.height,
        base_cfg.features,
        base_cfg.traits,
        args.runs,
        base_cfg.rounds,
    );
    println!("seed (base): {:?} | output: {}", base_cfg.seed, output_dir);
    println!("----------------------------------------------------------------------");

    // config.json
    {
        let path = format!("{output_dir}/config.json");
        let file = File::create(&path).expect("failed to create config.json");
        serde_json::to_writer_pretty(BufWriter::new(file), &base_cfg.to_run_config_json())
            .expect("failed to write config.json");
    }

    let mut sum_regions = 0.0f64;
    let mut sum_lc = 0.0f64;
    let mut sum_gp = 0.0f64;
    let mut n_converged = 0usize;
    let mut last_result: Option<SimulationResult> = None;

    for run_idx in 0..args.runs.max(1) {
        // Derive a per-run seed (deterministic) from the base seed.
        let seed = match base_cfg.seed {
            Some(s) => Some(derive_seed(
                s,
                &[
                    base_cfg.features as u64,
                    base_cfg.traits as u64,
                    run_idx as u64,
                ],
            )),
            None => None,
        };
        let cfg = Config {
            seed,
            ..base_cfg.clone()
        };
        let result = run(&cfg).unwrap_or_else(|e| panic!("run failed: {e}"));
        if result.converged {
            n_converged += 1;
        }
        sum_regions += result.final_metrics.n_stable_regions as f64;
        sum_lc += result.final_metrics.local_convergence;
        sum_gp += result.final_metrics.global_polarization;

        println!(
            "[{}/{}] seed={:?} converged={} round={} regions={} LC={:.3} GP={:.5} GP/N={:.3}",
            run_idx + 1,
            args.runs.max(1),
            seed,
            result.converged,
            result.final_round,
            result.final_metrics.n_stable_regions,
            result.final_metrics.local_convergence,
            result.final_metrics.global_polarization,
            result.final_metrics.gp_per_agent,
        );
        last_result = Some(result);
    }

    // Write per-round metrics + final grid + run_metadata for the last run.
    let result = last_result.expect("at least one run");
    save_metrics(&result, &output_dir);
    save_culture_grid(&result.world, &output_dir, "culture_grid_final.csv");
    save_run_metadata(&result, &base_cfg, &output_dir);

    refresh_latest(&args.output_dir, &timestamp);

    let runs = args.runs.max(1) as f64;
    println!("----------------------------------------------------------------------");
    println!(
        "done: {}/{} converged | mean n_stable_regions = {:.2} | mean LC = {:.3} | mean GP = {:.5}",
        n_converged,
        args.runs.max(1),
        sum_regions / runs,
        sum_lc / runs,
        sum_gp / runs,
    );
    println!(
        "LLM calls: {} | cache-hit: {} ({:.1}%) | model: {}",
        result.metadata.total(),
        result.metadata.cache_hits(),
        result.metadata.cache_hit_rate() * 100.0,
        result.llm_model,
    );
    println!("metrics  → {output_dir}/metrics.csv");
    println!("grid     → {output_dir}/culture_grid_final.csv");
    println!("metadata → {output_dir}/run_metadata.json");
    println!("config   → {output_dir}/config.json");
}

// --------------------------------------------------------------------------- //
// sweep
// --------------------------------------------------------------------------- //

fn cmd_sweep(args: SweepArgs) {
    let provider = parse_provider(&args.provider).unwrap_or_else(|e| panic!("{e}"));
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let dir_name = format!("{timestamp}_sweep");
    let sweep_dir = format!("{}/{}", args.output_dir, dir_name);
    fs::create_dir_all(&sweep_dir).expect("failed to create sweep dir");
    if provider.is_llm() {
        if let Some(parent) = Path::new(&args.cache_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
    }

    let feature_vals = enumerate_range(args.features_min, args.features_max, args.features_step);
    let traits_vals = enumerate_range(args.traits_min, args.traits_max, args.traits_step);
    let n_combos = feature_vals.len() * traits_vals.len();
    let n_total = n_combos * args.runs;

    println!("=== YuLan-OneSim — culture dissemination parameter sweep ===");
    println!(
        "provider: {} | grid: {}×{} | F={:?} | q={:?} | runs={} | total {} runs",
        provider.label(),
        args.width,
        args.height,
        feature_vals,
        traits_vals,
        args.runs,
        n_total,
    );
    println!("output: {sweep_dir}");
    println!("------------------------------------------------------------");

    // sweep_config.json
    {
        let config_json = serde_json::json!({
            "command": "sweep",
            "provider": provider.label(),
            "width": args.width,
            "height": args.height,
            "features": { "min": args.features_min, "max": args.features_max, "step": args.features_step },
            "traits": { "min": args.traits_min, "max": args.traits_max, "step": args.traits_step },
            "runs": args.runs,
            "rounds": args.rounds,
            "events_per_step": args.events_per_step,
            "snapshot_interval": args.snapshot_interval,
            "seed": args.seed,
            "llm_temperature": args.temperature,
            "llm_seed": args.llm_seed,
        });
        let path = format!("{sweep_dir}/sweep_config.json");
        let file = File::create(&path).expect("failed to create sweep_config.json");
        serde_json::to_writer_pretty(BufWriter::new(file), &config_json)
            .expect("failed to write sweep_config.json");
    }

    let path = format!("{sweep_dir}/sweep_summary.csv");
    let file = File::create(&path).expect("failed to create sweep_summary.csv");
    let mut wtr = Writer::from_writer(BufWriter::new(file));

    let mut idx = 0usize;
    for &features in &feature_vals {
        for &traits in &traits_vals {
            let mut sum_regions = 0.0f64;
            let mut n_converged = 0usize;

            for run_idx in 0..args.runs {
                idx += 1;
                let seed =
                    derive_seed(args.seed, &[features as u64, traits as u64, run_idx as u64]);
                let cfg = Config {
                    width: args.width,
                    height: args.height,
                    features,
                    traits,
                    events_per_step: args.events_per_step,
                    rounds: args.rounds,
                    snapshot_interval: args.snapshot_interval,
                    provider,
                    seed: Some(seed),
                    llm: make_llm_settings(args.temperature, args.llm_seed, &args.cache_path),
                    output_dir: sweep_dir.clone(),
                };
                let result = run(&cfg).unwrap_or_else(|e| panic!("run failed: {e}"));
                if result.converged {
                    n_converged += 1;
                }
                sum_regions += result.final_metrics.n_stable_regions as f64;

                wtr.serialize(SweepRow {
                    provider: provider.label().to_string(),
                    features,
                    traits,
                    run: run_idx,
                    width: args.width,
                    height: args.height,
                    seed,
                    converged: result.converged,
                    final_round: result.final_round,
                    n_stable_regions: result.final_metrics.n_stable_regions,
                    max_region_size: result.final_metrics.max_region_size,
                    n_distinct_cultures: result.final_metrics.n_distinct_cultures,
                    lc: result.final_metrics.local_convergence,
                    gp: result.final_metrics.global_polarization,
                    gp_per_agent: result.final_metrics.gp_per_agent,
                })
                .expect("failed to write sweep row");
            }
            let mean = sum_regions / args.runs.max(1) as f64;
            println!(
                "[{}/{}] F={:<3} q={:<3} → converged={}/{} mean_regions={:.2}",
                idx, n_total, features, traits, n_converged, args.runs, mean
            );
        }
    }
    wtr.flush().expect("flush failed");

    refresh_latest(&args.output_dir, &dir_name);
    println!("------------------------------------------------------------");
    println!("sweep done.");
    println!("summary → {sweep_dir}/sweep_summary.csv");
    println!("config  → {sweep_dir}/sweep_config.json");
}

// --------------------------------------------------------------------------- //
// reproduce (Phase 3 stub)
// --------------------------------------------------------------------------- //

fn cmd_reproduce(_args: ReproduceArgs) {
    println!("`reproduce` (Appendix F LC/GP batch reproduction) is a Phase 3 feature.");
    println!("For now, run the classical Table 7-2 reproductions and the LLM LC/GP run directly:");
    println!("  culture-llm run --provider none --width 10 --height 10 --features 5 --traits 10 --runs 30 --seed 42");
    println!("  culture-llm run --provider ollama --width 5 --height 2 --features 5 --traits 5 --rounds 100 --seed 42");
    println!("then use: culture-llm-tools reproduce  (Python-side LC/GP aggregation, also a Phase 3 stub).");
}

// --------------------------------------------------------------------------- //
// main
// --------------------------------------------------------------------------- //

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Reproduce(args) => cmd_reproduce(args),
    }
}
