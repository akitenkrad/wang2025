//! YuLan-OneSim (Wang et al. 2025) — Axelrod culture-dissemination CLI.
//!
//! `run`       : single configuration; classical (`--provider none`) or LLM
//!               (`--provider ollama|openai`) interaction on the same world.
//!               Writes intermediate culture-grid snapshots (`--snapshot-interval`)
//!               and a behaviour-graph / ODD concept export (`behavior_graph.json`).
//! `sweep`     : sweep features `F` × traits `q` (classical or LLM), aggregating
//!               `n_stable_regions` and LC/GP into `sweep_summary.csv`.
//! `reproduce` : Appendix F / Table 7-2 LC/GP batch reproduction (offline with
//!               `--provider none`); writes `reproduce_summary.json` (observed vs
//!               paper + PASS/off) + `reproduce_detail.csv`.
//! `compare`   : classical (`--provider none`) vs LLM quantitative comparison on
//!               matched configs (`--mock` for offline LLM); writes
//!               `compare_report.json` (LC/GP/regions/convergence deltas).

use std::fs;
use std::path::Path;

use clap::{Parser, Subcommand};

use culture_llm::config::{parse_provider, Config, LlmSettings, Provider};
use culture_llm::llm::wrap_client;
use culture_llm::odd::{build_behavior_graph, save_behavior_graph};
use culture_llm::simulation::{
    ensure_output_dir, run, run_with_client, save_culture_grid, save_metrics, save_run_metadata,
    save_snapshots, SimulationResult,
};

use socsim_core::derive_seed;
use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;
use socsim_results::{refresh_latest_symlink, timestamp, write_csv, write_json};

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
    /// Appendix F LC/GP batch reproduction (classical, offline-verifiable).
    Reproduce(ReproduceArgs),
    /// Classical (--provider none) vs LLM quantitative comparison on matched configs.
    Compare(CompareArgs),
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
    /// Interaction provider (Appendix-F batch is offline-verifiable with `none`).
    #[arg(long, default_value = "none")]
    provider: String,
    /// Runs per condition (averaged). Use `--quick` for a fast smoke.
    #[arg(long, default_value_t = 30)]
    runs: usize,
    /// Maximum number of engine ticks (rounds) per run.
    #[arg(long, default_value_t = 20_000)]
    rounds: usize,
    /// Random seed base.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Quick mode: fewer runs / shorter rounds for a fast end-to-end smoke.
    #[arg(long, default_value_t = false)]
    quick: bool,
    /// Output base directory.
    #[arg(long, default_value = "results")]
    output_dir: String,
}

#[derive(Parser, Debug)]
struct CompareArgs {
    /// LLM provider to compare against the classical baseline.
    #[arg(long, default_value = "ollama")]
    llm_provider: String,
    /// Use a deterministic scripted mock client for the LLM side (no network).
    /// Lets the comparison run end-to-end offline (CI / sandbox).
    #[arg(long, default_value_t = false)]
    mock: bool,
    /// Grid width.
    #[arg(long, default_value_t = 5)]
    width: usize,
    /// Grid height.
    #[arg(long, default_value_t = 4)]
    height: usize,
    /// Number of features F.
    #[arg(long, short = 'f', default_value_t = 5)]
    features: usize,
    /// Number of traits q.
    #[arg(long, short = 'q', default_value_t = 5)]
    traits: usize,
    /// Maximum number of engine ticks (rounds).
    #[arg(long, default_value_t = 100)]
    rounds: usize,
    /// Random seed (shared by both sides for a matched comparison).
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// LLM generation temperature.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,
    /// LLM generation seed.
    #[arg(long, default_value_t = 0)]
    llm_seed: u64,
    /// Prompt → response cache path (LLM path).
    #[arg(long, default_value = ".llm_cache/cache.json")]
    cache_path: String,
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
    let timestamp = timestamp();
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

    // config.json (pretty-print JSON; delegated to socsim_results::write_json).
    {
        let path = format!("{output_dir}/config.json");
        write_json(&base_cfg.to_run_config_json(), &path).expect("failed to write config.json");
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

    // Intermediate culture-grid snapshots (only when --snapshot-interval > 0).
    let snapshot_rounds = save_snapshots(&result, &output_dir);

    // Behaviour-graph / ODD-protocol concept export (derived from the model).
    let graph = build_behavior_graph(&base_cfg);
    save_behavior_graph(&graph, &output_dir);

    // latest symlink (best-effort; errors ignored as before).
    let _ = refresh_latest_symlink(&args.output_dir, &timestamp);

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
    println!("graph    → {output_dir}/behavior_graph.json");
    if !snapshot_rounds.is_empty() {
        println!(
            "snapshots→ {output_dir}/snapshots/ ({} grids: rounds {:?})",
            snapshot_rounds.len(),
            snapshot_rounds,
        );
    }
}

// --------------------------------------------------------------------------- //
// sweep
// --------------------------------------------------------------------------- //

fn cmd_sweep(args: SweepArgs) {
    let provider = parse_provider(&args.provider).unwrap_or_else(|e| panic!("{e}"));
    let timestamp = timestamp();
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
        write_json(&config_json, &path).expect("failed to write sweep_config.json");
    }

    let mut summary_rows: Vec<SweepRow> = Vec::with_capacity(n_total);

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

                summary_rows.push(SweepRow {
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
                });
            }
            let mean = sum_regions / args.runs.max(1) as f64;
            println!(
                "[{}/{}] F={:<3} q={:<3} → converged={}/{} mean_regions={:.2}",
                idx, n_total, features, traits, n_converged, args.runs, mean
            );
        }
    }

    // sweep_summary.csv (each row serialized; delegated to socsim_results::write_csv).
    {
        let path = format!("{sweep_dir}/sweep_summary.csv");
        write_csv(&summary_rows, &path).expect("failed to write sweep_summary.csv");
    }

    let _ = refresh_latest_symlink(&args.output_dir, &dir_name);
    println!("------------------------------------------------------------");
    println!("sweep done.");
    println!("summary → {sweep_dir}/sweep_summary.csv");
    println!("config  → {sweep_dir}/sweep_config.json");
}

// --------------------------------------------------------------------------- //
// reproduce (Phase 3 stub)
// --------------------------------------------------------------------------- //

/// One Appendix-F / Table 7-2 reproduction condition + its published target.
struct ReproCondition {
    id: &'static str,
    width: usize,
    height: usize,
    features: usize,
    traits: usize,
    /// Axelrod Table 7-2 target: mean number of stable regions.
    target_regions: f64,
    /// Tolerance band (± regions) within which the condition counts as PASS.
    tol_regions: f64,
}

/// The four Appendix-F / Axelrod Table 7-2 conditions (10×10 grid). Targets and
/// tolerances mirror `docs/reproduction.md`.
const REPRO_CONDITIONS: [ReproCondition; 4] = [
    ReproCondition {
        id: "F5q10",
        width: 10,
        height: 10,
        features: 5,
        traits: 10,
        target_regions: 3.2,
        tol_regions: 2.5,
    },
    ReproCondition {
        id: "F5q15",
        width: 10,
        height: 10,
        features: 5,
        traits: 15,
        target_regions: 20.0,
        tol_regions: 4.0,
    },
    ReproCondition {
        id: "F10q10",
        width: 10,
        height: 10,
        features: 10,
        traits: 10,
        target_regions: 1.0,
        tol_regions: 0.5,
    },
    ReproCondition {
        id: "F15q15",
        width: 10,
        height: 10,
        features: 15,
        traits: 15,
        target_regions: 1.2,
        tol_regions: 0.5,
    },
];

#[derive(serde::Serialize)]
struct ReproConditionResult {
    id: String,
    width: usize,
    height: usize,
    features: usize,
    traits: usize,
    runs: usize,
    target_regions: f64,
    observed_mean_regions: f64,
    abs_error: f64,
    within_tolerance: bool,
    mean_lc: f64,
    mean_gp: f64,
    mean_gp_per_agent: f64,
    mean_max_region_size: f64,
    converged_fraction: f64,
}

fn cmd_reproduce(args: ReproduceArgs) {
    let provider = parse_provider(&args.provider).unwrap_or_else(|e| panic!("{e}"));
    let (runs, rounds) = if args.quick {
        (5usize, args.rounds.min(5_000))
    } else {
        (args.runs, args.rounds)
    };
    let timestamp = timestamp();
    let dir_name = format!("reproduce_{timestamp}");
    let base_dir = format!("{}/{}", args.output_dir, dir_name);
    let figures_dir = format!("{base_dir}/figures");
    fs::create_dir_all(&figures_dir).expect("failed to create reproduce figures dir");

    println!("=== YuLan-OneSim — Appendix F / Axelrod Table 7-2 reproduction ===");
    println!(
        "provider: {} | runs={} | rounds={} | quick={} | output: {base_dir}",
        provider.label(),
        runs,
        rounds,
        args.quick,
    );
    println!("----------------------------------------------------------------------");

    let mut cond_results: Vec<ReproConditionResult> = Vec::new();
    // Per-condition per-run rows feed the Python figures (figures/*.png).
    let mut detail_rows: Vec<SweepRow> = Vec::new();

    for cond in REPRO_CONDITIONS.iter() {
        let mut sum_regions = 0.0f64;
        let mut sum_lc = 0.0f64;
        let mut sum_gp = 0.0f64;
        let mut sum_gpn = 0.0f64;
        let mut sum_max = 0.0f64;
        let mut n_converged = 0usize;

        for run_idx in 0..runs.max(1) {
            let seed = derive_seed(
                args.seed,
                &[cond.features as u64, cond.traits as u64, run_idx as u64],
            );
            let cfg = Config {
                width: cond.width,
                height: cond.height,
                features: cond.features,
                traits: cond.traits,
                events_per_step: 0,
                rounds,
                snapshot_interval: 0,
                provider,
                seed: Some(seed),
                llm: make_llm_settings(0.0, 0, ".llm_cache/cache.json"),
                output_dir: base_dir.clone(),
            };
            let result = run(&cfg).unwrap_or_else(|e| panic!("reproduce run failed: {e}"));
            if result.converged {
                n_converged += 1;
            }
            let m = &result.final_metrics;
            sum_regions += m.n_stable_regions as f64;
            sum_lc += m.local_convergence;
            sum_gp += m.global_polarization;
            sum_gpn += m.gp_per_agent;
            sum_max += m.max_region_size as f64;

            detail_rows.push(SweepRow {
                provider: provider.label().to_string(),
                features: cond.features,
                traits: cond.traits,
                run: run_idx,
                width: cond.width,
                height: cond.height,
                seed,
                converged: result.converged,
                final_round: result.final_round,
                n_stable_regions: m.n_stable_regions,
                max_region_size: m.max_region_size,
                n_distinct_cultures: m.n_distinct_cultures,
                lc: m.local_convergence,
                gp: m.global_polarization,
                gp_per_agent: m.gp_per_agent,
            });
        }

        let denom = runs.max(1) as f64;
        let mean_regions = sum_regions / denom;
        let abs_error = (mean_regions - cond.target_regions).abs();
        let within = abs_error <= cond.tol_regions;
        println!(
            "[{:<6}] F={:<2} q={:<2} → observed {:.2} (target {:.1} ±{:.1})  {}  | LC={:.3} GP/N={:.3}",
            cond.id,
            cond.features,
            cond.traits,
            mean_regions,
            cond.target_regions,
            cond.tol_regions,
            if within { "PASS" } else { "off" },
            sum_lc / denom,
            sum_gpn / denom,
        );

        cond_results.push(ReproConditionResult {
            id: cond.id.to_string(),
            width: cond.width,
            height: cond.height,
            features: cond.features,
            traits: cond.traits,
            runs: runs.max(1),
            target_regions: cond.target_regions,
            observed_mean_regions: mean_regions,
            abs_error,
            within_tolerance: within,
            mean_lc: sum_lc / denom,
            mean_gp: sum_gp / denom,
            mean_gp_per_agent: sum_gpn / denom,
            mean_max_region_size: sum_max / denom,
            converged_fraction: n_converged as f64 / denom,
        });
    }

    // reproduce_detail.csv (per-condition per-run; consumed by Python figures).
    {
        let path = format!("{base_dir}/reproduce_detail.csv");
        write_csv(&detail_rows, &path).expect("failed to write reproduce_detail.csv");
    }

    let n_pass = cond_results.iter().filter(|c| c.within_tolerance).count();
    let n_total = cond_results.len();
    let summary = serde_json::json!({
        "command": "reproduce",
        "paper": "Axelrod (1997) Table 7-2 / YuLan-OneSim Appendix F",
        "provider": provider.label(),
        "runs": runs,
        "rounds": rounds,
        "quick": args.quick,
        "seed": args.seed,
        "base_dir": base_dir,
        "figures_dir": figures_dir,
        "detail_csv": format!("{base_dir}/reproduce_detail.csv"),
        "n_pass": n_pass,
        "n_total": n_total,
        "conditions": cond_results,
        "gp_note": "GP=|C|/N² is recorded as documented; gp_per_agent=|C|/N is the \
                    auxiliary normalisation (see docs/architecture.md GP inconsistency).",
    });
    let summary_path = format!("{base_dir}/reproduce_summary.json");
    write_json(&summary, &summary_path).expect("failed to write reproduce_summary.json");

    let _ = refresh_latest_symlink(&args.output_dir, &dir_name);
    println!("----------------------------------------------------------------------");
    println!("reproduce done: {n_pass}/{n_total} conditions within tolerance.");
    println!("summary → {summary_path}");
    println!("detail  → {base_dir}/reproduce_detail.csv");
    println!("figures → {figures_dir}/ (render with: culture-llm-tools reproduce --skip-build)");
}

// --------------------------------------------------------------------------- //
// compare (classical vs LLM)
// --------------------------------------------------------------------------- //

/// A deterministic scripted mock LLM client that adopts the FIRST differing
/// feature offered in the prompt's candidate list (mirrors the homophily
/// decision). Lets `compare --mock` run the LLM side fully offline.
fn build_mock_client() -> culture_llm::llm::CultureClient {
    let backend = ScriptedClient::new("mock-culture-llm", |prompt: &str| {
        if let Some(pos) = prompt.find("differ on these feature indices: [") {
            let tail = &prompt[pos..];
            if let Some(open) = tail.find('[') {
                let first: String = tail[open + 1..]
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect();
                if !first.is_empty() {
                    return first;
                }
            }
        }
        "-1".to_string()
    });
    wrap_client(backend, PromptCache::in_memory())
}

#[derive(serde::Serialize)]
struct CompareSide {
    label: String,
    provider: String,
    llm_model: String,
    converged: bool,
    final_round: usize,
    n_stable_regions: usize,
    max_region_size: usize,
    n_distinct_cultures: usize,
    lc: f64,
    gp: f64,
    gp_per_agent: f64,
    total_llm_calls: usize,
    cache_hit_rate: f64,
}

fn compare_side(label: &str, result: &SimulationResult, provider_label: &str) -> CompareSide {
    let m = &result.final_metrics;
    CompareSide {
        label: label.to_string(),
        provider: provider_label.to_string(),
        llm_model: result.llm_model.clone(),
        converged: result.converged,
        final_round: result.final_round,
        n_stable_regions: m.n_stable_regions,
        max_region_size: m.max_region_size,
        n_distinct_cultures: m.n_distinct_cultures,
        lc: m.local_convergence,
        gp: m.global_polarization,
        gp_per_agent: m.gp_per_agent,
        total_llm_calls: result.metadata.total(),
        cache_hit_rate: result.metadata.cache_hit_rate(),
    }
}

fn cmd_compare(args: CompareArgs) {
    let llm_provider = parse_provider(&args.llm_provider).unwrap_or_else(|e| panic!("{e}"));
    if !llm_provider.is_llm() {
        panic!(
            "--llm-provider must be an LLM provider (ollama / openai), got {}",
            args.llm_provider
        );
    }
    let timestamp = timestamp();
    let dir_name = format!("compare_{timestamp}");
    let base_dir = format!("{}/{}", args.output_dir, dir_name);
    fs::create_dir_all(&base_dir).expect("failed to create compare dir");

    println!("=== YuLan-OneSim — classical vs LLM quantitative comparison ===");
    println!(
        "grid: {}×{} | F={} | q={} | rounds={} | seed={} | LLM={} | mock={}",
        args.width,
        args.height,
        args.features,
        args.traits,
        args.rounds,
        args.seed,
        llm_provider.label(),
        args.mock,
    );
    println!("output: {base_dir}");
    println!("----------------------------------------------------------------------");

    // Matched base config (same grid / seed / rounds for both sides).
    let make_cfg = |provider: Provider| Config {
        width: args.width,
        height: args.height,
        features: args.features,
        traits: args.traits,
        events_per_step: 0,
        rounds: args.rounds,
        snapshot_interval: 0,
        provider,
        seed: Some(args.seed),
        llm: make_llm_settings(args.temperature, args.llm_seed, &args.cache_path),
        output_dir: base_dir.clone(),
    };

    // --- classical side (always live; 0 LLM calls) --- //
    let classical_cfg = make_cfg(Provider::None);
    let classical = run(&classical_cfg).unwrap_or_else(|e| panic!("classical run failed: {e}"));
    let classical_side = compare_side("classical", &classical, "none");
    println!(
        "[classical] regions={} LC={:.3} GP/N={:.3} converged={} round={} (0 LLM calls)",
        classical_side.n_stable_regions,
        classical_side.lc,
        classical_side.gp_per_agent,
        classical_side.converged,
        classical_side.final_round,
    );

    // --- LLM side (mock = offline; otherwise live via env-built client) --- //
    let mut llm_cfg = make_cfg(llm_provider);
    let llm = if args.mock {
        // The mock client uses an in-memory cache (no file to save to), so clear
        // the cache_path to skip the (would-fail) save step.
        llm_cfg.llm.cache_path = None;
        run_with_client(&llm_cfg, Some(build_mock_client()))
            .unwrap_or_else(|e| panic!("LLM (mock) run failed: {e}"))
    } else {
        run(&llm_cfg).unwrap_or_else(|e| {
            panic!("LLM (live) run failed: {e}. In a network-free environment, pass --mock.")
        })
    };
    let llm_label = if args.mock { "llm-mock" } else { "llm-live" };
    let llm_side = compare_side(llm_label, &llm, llm_provider.label());
    println!(
        "[{}] regions={} LC={:.3} GP/N={:.3} converged={} round={} LLM_calls={} cache_hit={:.1}%",
        llm_label,
        llm_side.n_stable_regions,
        llm_side.lc,
        llm_side.gp_per_agent,
        llm_side.converged,
        llm_side.final_round,
        llm_side.total_llm_calls,
        llm_side.cache_hit_rate * 100.0,
    );

    let report = serde_json::json!({
        "command": "compare",
        "matched_config": {
            "width": args.width,
            "height": args.height,
            "features": args.features,
            "traits": args.traits,
            "rounds": args.rounds,
            "seed": args.seed,
        },
        "mock": args.mock,
        "classical": classical_side,
        "llm": llm_side,
        "deltas": {
            "n_stable_regions": llm_side.n_stable_regions as i64 - classical_side.n_stable_regions as i64,
            "lc": llm_side.lc - classical_side.lc,
            "gp": llm_side.gp - classical_side.gp,
            "gp_per_agent": llm_side.gp_per_agent - classical_side.gp_per_agent,
            "final_round": llm_side.final_round as i64 - classical_side.final_round as i64,
        },
        "note": "Matched grid/seed/rounds. The classical side is the deterministic Axelrod \
                 baseline (0 LLM calls); the LLM side is the YuLan-OneSim variant. With --mock \
                 the LLM side is a deterministic scripted client (offline); live LLM numbers \
                 depend on the model and are pseudo-determinised by the prompt cache.",
    });
    let report_path = format!("{base_dir}/compare_report.json");
    write_json(&report, &report_path).expect("failed to write compare_report.json");

    let _ = refresh_latest_symlink(&args.output_dir, &dir_name);
    println!("----------------------------------------------------------------------");
    println!("compare done. report → {report_path}");
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
        Commands::Compare(args) => cmd_compare(args),
    }
}
