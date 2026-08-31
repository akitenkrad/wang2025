//! YuLan-OneSim (Wang et al. 2025) — Axelrod culture-dissemination CLI.
//!
//! `run`       : single configuration; classical (`--provider none`) or LLM
//!               (`--provider ollama|openai`) interaction on the same world.
//!               Writes intermediate culture-grid snapshots (`--snapshot-interval`)
//!               and a behaviour-graph / ODD concept export (`behavior_graph.json`).
//! `sweep`     : sweep features `F` × traits `q` (classical or LLM). A sweep
//!               parent plus one child run per `(F, q)` cell.
//! `reproduce` : Appendix F / Table 7-2 LC/GP batch reproduction (offline with
//!               `--provider none`). A parent plus one child run per condition,
//!               with Axelrod's published targets in each child's `reference.csv`.
//! `compare`   : classical (`--provider none`) vs LLM quantitative comparison on
//!               matched configs (`--mock` for offline LLM). A parent plus one
//!               child run per side.
//!
//! Where the results go is runvault's business. There is no timestamped
//! directory and no `latest` symlink of our own: `Run::start` names and creates
//! the run directory, per-round numbers go to its `metrics.csv`, each trial's
//! final state to its `events.jsonl`, and the tables that are neither (the
//! culture grid, the snapshots, the ODD export) under its `artifacts/`.

use std::cell::RefCell;
use std::fs;
use std::path::Path;
use std::rc::Rc;

use clap::{Parser, Subcommand};
use runvault::{Lineage, Run, RunOptions};
use serde::Serialize;

use culture_llm::config::{parse_provider, Config, LlmSettings, Provider};
use culture_llm::llm::{build_live_client, wrap_client, CultureClient};
use culture_llm::mechanisms::SharedClient;
use culture_llm::odd::{build_behavior_graph, save_behavior_graph};
use culture_llm::record::{self, TrialOutcome, DOMAIN, EXPERIMENT, REPO_ID};
use culture_llm::simulation::{
    ensure_output_dir, run_with_shared_client, save_culture_grid, save_snapshots, SimulationResult,
};

use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

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
    /// Ollama 接続先 URL（指定時は環境変数 OLLAMA_HOST を上書きする）．
    #[arg(long, global = true)]
    ollama_host: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run a single configuration (classical or LLM interaction).
    Run(RunArgs),
    /// Sweep features F × traits q; a sweep parent plus one child per cell.
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
    /// Random seed (governs the socsim core layer only; omitted = drawn once and recorded).
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
    /// Results root (runvault writes `<root>/culture-llm/<run_slug>/`).
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
    /// Snapshot interval in rounds (passthrough; recorded in the conditions).
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
    /// Results root (runvault writes `<root>/culture-llm/<run_slug>/`).
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
    /// Results root (runvault writes `<root>/culture-llm/<run_slug>/`).
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
    /// Results root (runvault writes `<root>/culture-llm/<run_slug>/`).
    #[arg(long, default_value = "results")]
    output_dir: String,
}

// --------------------------------------------------------------------------- //
// 実験条件 (runvault の `config.json` の `parameters`)
// --------------------------------------------------------------------------- //

/// `run` 1 本の条件．
///
/// 旧 `config.json` に対して 2 つ足してある:
///
/// - `runs` — `--runs N` は N 本回して **最後の 1 本**の詳細を残す．どの試行が
///   記録されるかを決めるので条件の一部だが，旧 `config.json` には無く，
///   `config_hash` がその違いに盲目だった．
/// - `llm_cache_path` — LLM 経路の再生元．結果を決めるのは «そこに何が入っているか»
///   であって path 自体ではないので «置き場» であり，`hash_exclude` で
///   `config_hash` から外す．古典経路は触らないので `null`．
///
/// `output_dir` は落とした — run ディレクトリが出力先そのものである．
#[derive(Serialize)]
struct RunParameters {
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
    llm_cache_path: Option<String>,
}

/// 掃引親 run の条件 (F × q のグリッド定義そのもの)．
///
/// シードは `base_seed` と名乗る．終端イベントの自由欄 `seed` は試行ごとに違う
/// 派生シードで，`sweep_events_table` がパラメータ列でイベント列を上書きする以上，
/// 同名にすると条件側の値が試行側を黙って潰しうる．
#[derive(Serialize)]
struct SweepParameters {
    provider: &'static str,
    width: usize,
    height: usize,
    features_min: usize,
    features_max: usize,
    features_step: usize,
    traits_min: usize,
    traits_max: usize,
    traits_step: usize,
    runs: usize,
    rounds: usize,
    events_per_step: usize,
    snapshot_interval: usize,
    base_seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
    llm_cache_path: Option<String>,
}

/// 掃引の子 run (`(F, q)` 1 セル × `runs` 試行) の条件．
///
/// `run` の条件に `runs` が付いた形だが，サブコマンド名は `run` ではなく
/// `sweep-point` にしてある．`run` は 1 本のシミュレーション，子は同一セルの
/// `runs` 本で，中身の違う 2 つを同じ名前に同居させると
/// `runvault path --subcommand run` がどちらを返すか分からなくなる．
#[derive(Serialize)]
struct SweepPointParameters {
    provider: &'static str,
    width: usize,
    height: usize,
    features: usize,
    traits: usize,
    runs: usize,
    rounds: usize,
    events_per_step: usize,
    snapshot_interval: usize,
    base_seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
    llm_cache_path: Option<String>,
}

/// 再現の親 run の条件 (4 条件のバッチ定義)．
#[derive(Serialize)]
struct ReproduceParameters {
    provider: &'static str,
    conditions: Vec<&'static str>,
    runs: usize,
    rounds: usize,
    quick: bool,
    base_seed: u64,
}

/// 再現の子 run (条件 1 つ × `runs` 試行) の条件．
#[derive(Serialize)]
struct ReproduceConditionParameters {
    condition_id: &'static str,
    provider: &'static str,
    width: usize,
    height: usize,
    features: usize,
    traits: usize,
    runs: usize,
    rounds: usize,
    events_per_step: usize,
    snapshot_interval: usize,
    base_seed: u64,
}

/// 比較の親 run の条件 (両側に共通の一致 config)．
#[derive(Serialize)]
struct CompareParameters {
    sides: Vec<&'static str>,
    llm_provider: &'static str,
    mock: bool,
    width: usize,
    height: usize,
    features: usize,
    traits: usize,
    rounds: usize,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
    llm_cache_path: Option<String>,
}

/// 比較の子 run (片側 1 本) の条件．
#[derive(Serialize)]
struct CompareSideParameters {
    side: &'static str,
    provider: &'static str,
    mock: bool,
    width: usize,
    height: usize,
    features: usize,
    traits: usize,
    events_per_step: usize,
    rounds: usize,
    snapshot_interval: usize,
    seed: u64,
    llm_temperature: f32,
    llm_seed: u64,
    llm_cache_path: Option<String>,
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

/// LLM クライアントを共有ハンドルにする．
///
/// **`Run::start` より前に組む** — モデル名と endpoint を知っているのは
/// クライアントを組んだ側だけで，それが `run.json` の `llm` ブロックの中身だから
/// である．掃引のように何本も回す場合も 1 度だけ組んで同じハンドルを渡す
/// (プロンプトキャッシュは移行前も file 経由で試行をまたいで温まっていた)．
fn share(client: CultureClient) -> SharedClient {
    Rc::new(RefCell::new(client))
}

/// 共有ハンドルが名乗るモデル名と endpoint．
fn client_identity(client: &SharedClient) -> (String, String) {
    let borrowed = client.borrow();
    (
        borrowed.inner().model().to_string(),
        borrowed.inner().endpoint().to_string(),
    )
}

/// 本番の LLM クライアント (Ollama 第一候補 → OpenAI フォールバック + キャッシュ)．
fn build_client(settings: &LlmSettings) -> SharedClient {
    share(build_live_client(settings).unwrap_or_else(|e| panic!("LLM client build failed: {e}")))
}

/// A deterministic scripted mock LLM client that adopts the FIRST differing
/// feature offered in the prompt's candidate list (mirrors the homophily
/// decision). Lets `compare --mock` run the LLM side fully offline.
fn build_mock_client() -> CultureClient {
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

/// LLM 経路のときだけキャッシュの置き場を作る．
fn prepare_cache_dir(provider: Provider, cache_path: &str) {
    if provider.is_llm() {
        if let Some(parent) = Path::new(cache_path).parent() {
            let _ = fs::create_dir_all(parent);
        }
    }
}

/// LLM 経路のときだけ cache path を条件に載せる (古典経路は触らないので `null`)．
fn cache_path_of(provider: Provider, cache_path: &str) -> Option<String> {
    provider.is_llm().then(|| cache_path.to_string())
}

/// `hash_exclude` に渡す «置き場» のポインタ．
const CACHE_POINTER: [&str; 1] = ["/llm_cache_path"];

/// 掃引・再現の子で試行をまたいだ LLM 呼び出しを数える．
#[derive(Default)]
struct LlmTally {
    calls: usize,
    hits: usize,
}

impl LlmTally {
    fn add(&mut self, result: &SimulationResult) {
        self.calls += result.metadata.total();
        self.hits += result.metadata.cache_hits();
    }
}

// --------------------------------------------------------------------------- //
// run
// --------------------------------------------------------------------------- //

fn cmd_run(args: RunArgs) {
    let provider = parse_provider(&args.provider).unwrap_or_else(|e| panic!("{e}"));
    prepare_cache_dir(provider, &args.cache_path);

    // シードを実体化してから記録する．--seed 省略時にシミュレーション側で
    // rand::random に落とすと，実際に使われたシードがどこにも残らない．
    let base_seed = args.seed.unwrap_or_else(rand::random::<u64>);
    let runs = args.runs.max(1);

    let llm = make_llm_settings(args.temperature, args.llm_seed, &args.cache_path);
    let base_cfg = Config {
        width: args.width,
        height: args.height,
        features: args.features,
        traits: args.traits,
        events_per_step: args.events_per_step,
        rounds: args.rounds,
        snapshot_interval: args.snapshot_interval,
        provider,
        seed: None, // 試行ごとに派生させる
        llm,
    };

    // クライアントは Run::start の前に組む (llm ブロックのため)．
    let client = provider.is_llm().then(|| build_client(&base_cfg.llm));

    let parameters = RunParameters {
        provider: provider.label(),
        width: base_cfg.width,
        height: base_cfg.height,
        features: base_cfg.features,
        traits: base_cfg.traits,
        events_per_step: base_cfg.effective_events_per_step(),
        rounds: base_cfg.rounds,
        snapshot_interval: base_cfg.snapshot_interval,
        runs,
        seed: base_seed,
        llm_temperature: args.temperature,
        llm_seed: args.llm_seed,
        llm_cache_path: cache_path_of(provider, &args.cache_path),
    };

    // `--runs N` は掃引ではないので子には割らない．同じ条件を N 本回して最後の
    // 試行の詳細だけを残す既存の動きなので，master_seed には実際に世界を支配した
    // シードを書き，replicate_index を N-1 にする．根のシードは /parameters.seed．
    let recorded_seed = record::trial_seed(base_seed, args.features, args.traits, runs - 1);
    let mut options = RunOptions::new(EXPERIMENT, "run")
        .repo_id(REPO_ID)
        .domain(DOMAIN)
        .results_root(&args.output_dir)
        .parameters(&parameters)
        .expect("runvault: parameters の組み立てに失敗")
        .hash_exclude(CACHE_POINTER)
        .seed_pointers(["/seed"])
        .master_seed(recorded_seed)
        .replicate_index((runs - 1) as u64)
        .replication(record::replication());
    if let Some(c) = &client {
        let (model, endpoint) = client_identity(c);
        options = options.llm(record::llm_block(&model, &endpoint, args.temperature));
    }
    let mut rv = Run::start(options).expect("runvault: run の開始に失敗");

    // run ディレクトリが出力先そのもの．表と盤面は artifacts/ の下へ．
    let artifacts = rv.dir().join("artifacts").to_string_lossy().into_owned();
    ensure_output_dir(&artifacts);

    println!("=== YuLan-OneSim (Wang et al. 2025) — Axelrod culture dissemination ===");
    println!(
        "provider: {} | grid: {}×{} | F={} | q={} | runs={} | rounds={}",
        provider.label(),
        base_cfg.width,
        base_cfg.height,
        base_cfg.features,
        base_cfg.traits,
        runs,
        base_cfg.rounds,
    );
    println!(
        "seed (base): {} | output: {}",
        base_seed,
        rv.dir().display()
    );
    println!("----------------------------------------------------------------------");

    let mut sum_regions = 0.0f64;
    let mut sum_lc = 0.0f64;
    let mut sum_gp = 0.0f64;
    let mut n_converged = 0usize;
    let mut last: Option<(u64, SimulationResult)> = None;

    for run_idx in 0..runs {
        // Derive a per-run seed (deterministic) from the base seed.
        let seed = record::trial_seed(base_seed, base_cfg.features, base_cfg.traits, run_idx);
        let cfg = Config {
            seed: Some(seed),
            ..base_cfg.clone()
        };
        let result = run_with_shared_client(&cfg, client.clone())
            .unwrap_or_else(|e| panic!("run failed: {e}"));
        if result.converged {
            n_converged += 1;
        }
        sum_regions += result.final_metrics.n_stable_regions as f64;
        sum_lc += result.final_metrics.local_convergence;
        sum_gp += result.final_metrics.global_polarization;

        println!(
            "[{}/{}] seed={} converged={} round={} regions={} LC={:.3} GP={:.5} GP/N={:.3}",
            run_idx + 1,
            runs,
            seed,
            result.converged,
            result.final_round,
            result.final_metrics.n_stable_regions,
            result.final_metrics.local_convergence,
            result.final_metrics.global_polarization,
            result.final_metrics.gp_per_agent,
        );
        last = Some((seed, result));
    }

    // Record the last run (the one master_seed / replicate_index name).
    let (seed, result) = last.expect("at least one run");
    record::log_round_metrics(&mut rv, &result.round_history);
    record::log_run_scope(&mut rv, &result);
    record::log_llm_metrics(&mut rv, &result.metadata);
    // run は全ラウンドを観測して metrics.csv に残しているので，観測時刻も全ラウンド．
    let observed: Vec<u64> = result
        .round_history
        .iter()
        .map(|r| r.round as u64)
        .collect();
    record::log_terminal(&mut rv, "run", seed, base_cfg.rounds, observed, &result);

    save_culture_grid(&result.world, &artifacts, "culture_grid_final.csv");
    let snapshot_rounds = save_snapshots(&result, &artifacts);

    // Behaviour-graph / ODD-protocol concept export (derived from the model).
    let graph = build_behavior_graph(&base_cfg);
    save_behavior_graph(&graph, &artifacts);

    println!("----------------------------------------------------------------------");
    let denom = runs as f64;
    println!(
        "done: {}/{} converged | mean n_stable_regions = {:.2} | mean LC = {:.3} | mean GP = {:.5}",
        n_converged,
        runs,
        sum_regions / denom,
        sum_lc / denom,
        sum_gp / denom,
    );
    println!(
        "LLM calls: {} | cache-hit: {} ({:.1}%) | model: {}",
        result.metadata.total(),
        result.metadata.cache_hits(),
        result.metadata.cache_hit_rate() * 100.0,
        result.llm_model,
    );

    let dir = rv.finish().expect("runvault: run の完了に失敗");
    println!("metrics  → {}/metrics.csv", dir.display());
    println!("terminal → {}/events.jsonl", dir.display());
    println!(
        "grid     → {}/artifacts/culture_grid_final.csv",
        dir.display()
    );
    println!("graph    → {}/artifacts/behavior_graph.json", dir.display());
    println!("config   → {}/config.json", dir.display());
    if !snapshot_rounds.is_empty() {
        println!(
            "snapshots→ {}/artifacts/snapshots/ ({} grids: rounds {:?})",
            dir.display(),
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
    prepare_cache_dir(provider, &args.cache_path);

    let feature_vals = enumerate_range(args.features_min, args.features_max, args.features_step);
    let traits_vals = enumerate_range(args.traits_min, args.traits_max, args.traits_step);
    let n_combos = feature_vals.len() * traits_vals.len();
    let n_total = n_combos * args.runs;

    let llm = make_llm_settings(args.temperature, args.llm_seed, &args.cache_path);
    let client = provider.is_llm().then(|| build_client(&llm));
    let llm_identity = client.as_ref().map(client_identity);

    let sweep_parameters = SweepParameters {
        provider: provider.label(),
        width: args.width,
        height: args.height,
        features_min: args.features_min,
        features_max: args.features_max,
        features_step: args.features_step,
        traits_min: args.traits_min,
        traits_max: args.traits_max,
        traits_step: args.traits_step,
        runs: args.runs,
        rounds: args.rounds,
        events_per_step: args.events_per_step,
        snapshot_interval: args.snapshot_interval,
        base_seed: args.seed,
        llm_temperature: args.temperature,
        llm_seed: args.llm_seed,
        llm_cache_path: cache_path_of(provider, &args.cache_path),
    };

    // 親 run: F × q のグリッド定義そのものを parameters に持つ．個別セルの指標は
    // 書かない．親は 1 本のシミュレーションではないので master_seed を名乗らず，
    // base seed は /parameters.base_seed と seed_pointers 経由で execution_hash に残る．
    let parent = Run::start(
        RunOptions::new(EXPERIMENT, "sweep")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&sweep_parameters)
            .expect("runvault: sweep の parameters の組み立てに失敗")
            .hash_exclude(CACHE_POINTER)
            .seed_pointers(["/base_seed"])
            .sweep_parent()
            .replication(record::replication()),
    )
    .expect("runvault: sweep 親 run の開始に失敗");

    let sweep_id = parent
        .sweep_id()
        .expect("runvault: sweep 親に sweep_id がありません")
        .to_string();
    let parent_run_uid = parent.run_uid().to_string();

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
    println!(
        "seed (base): {} | output: {}",
        args.seed,
        parent.dir().display()
    );
    println!("------------------------------------------------------------");

    let mut idx = 0usize;
    for &features in &feature_vals {
        for &traits in &traits_vals {
            let params = SweepPointParameters {
                provider: provider.label(),
                width: args.width,
                height: args.height,
                features,
                traits,
                runs: args.runs,
                rounds: args.rounds,
                events_per_step: args.events_per_step,
                snapshot_interval: args.snapshot_interval,
                base_seed: args.seed,
                llm_temperature: args.temperature,
                llm_seed: args.llm_seed,
                llm_cache_path: cache_path_of(provider, &args.cache_path),
            };

            // 子は「その (F, q) の試行群」そのもの．master_seed は親と同じ base で，
            // セルが違えば config_hash が違うので run としては別物になる．同じ条件の
            // 繰り返しは無いので replicate_index は 0．
            let mut child_options = RunOptions::new(EXPERIMENT, "sweep-point")
                .repo_id(REPO_ID)
                .domain(DOMAIN)
                .results_root(&args.output_dir)
                .parameters(&params)
                .expect("runvault: 子 run の parameters の組み立てに失敗")
                .hash_exclude(CACHE_POINTER)
                .seed_pointers(["/base_seed"])
                .master_seed(args.seed)
                .replicate_index(0)
                .lineage(Lineage {
                    sweep_id: Some(sweep_id.clone()),
                    parent_run_uid: Some(parent_run_uid.clone()),
                    ..Default::default()
                })
                .replication(record::replication());
            if let Some((model, endpoint)) = &llm_identity {
                child_options =
                    child_options.llm(record::llm_block(model, endpoint, args.temperature));
            }
            let mut child = Run::start(child_options).expect("runvault: 子 run の開始に失敗");

            let mut trials: Vec<TrialOutcome> = Vec::with_capacity(args.runs);
            let mut tally = LlmTally::default();
            for run_idx in 0..args.runs {
                idx += 1;
                let seed = record::trial_seed(args.seed, features, traits, run_idx);
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
                    llm: llm.clone(),
                };
                let result = run_with_shared_client(&cfg, client.clone())
                    .unwrap_or_else(|e| panic!("run failed: {e}"));
                // 掃引が見るのは各試行の最終ラウンドだけなので，観測時刻もそこ 1 点．
                record::log_terminal(
                    &mut child,
                    &format!("trial-{run_idx}"),
                    seed,
                    args.rounds,
                    [result.final_round as u64],
                    &result,
                );
                tally.add(&result);
                trials.push(TrialOutcome::from_result(&result));
            }
            record::log_condition_summary(&mut child, &trials);
            record::log_llm_totals(&mut child, tally.calls, tally.hits);

            let n_converged = trials.iter().filter(|t| t.converged).count();
            let mean = trials
                .iter()
                .map(|t| t.n_stable_regions as f64)
                .sum::<f64>()
                / trials.len() as f64;
            println!(
                "[{}/{}] F={:<3} q={:<3} → converged={}/{} mean_regions={:.2}",
                idx, n_total, features, traits, n_converged, args.runs, mean
            );

            child.finish().expect("runvault: 子 run の完了に失敗");
        }
    }

    let dir = parent
        .finish()
        .expect("runvault: sweep 親 run の完了に失敗");
    println!("------------------------------------------------------------");
    println!("sweep done: {n_total} runs across {n_combos} cells.");
    println!("掃引定義 → {}/config.json", dir.display());
    println!("各セルの試行は子 run (subcommand=sweep-point) の events.jsonl にあります");
}

// --------------------------------------------------------------------------- //
// reproduce
// --------------------------------------------------------------------------- //

/// One Appendix-F / Table 7-2 reproduction condition + its published target.
struct ReproCondition {
    id: &'static str,
    width: usize,
    height: usize,
    features: usize,
    traits: usize,
    /// Axelrod Table 7-2 target: mean number of stable regions. A number the
    /// paper printed, so it goes into the child's `reference.csv`.
    target_regions: f64,
    /// Tolerance band (± regions) within which the condition counts as PASS.
    /// **Ours, not the paper's** — neither a metric nor a reported value, so it
    /// stays in the parent's `artifacts/` verdict table and on the console.
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

/// One row of the parent's `artifacts/reproduce_verdicts.csv`.
///
/// The observed mean is a metric of the child and the target is that child's
/// `reference.csv` row; what only lives here is the band we chose and the
/// verdict it produces.
#[derive(Serialize)]
struct VerdictRow {
    id: &'static str,
    features: usize,
    traits: usize,
    runs: usize,
    target_regions: f64,
    observed_mean_regions: f64,
    abs_error: f64,
    tolerance: f64,
    within_tolerance: bool,
    child_run_slug: String,
}

fn cmd_reproduce(args: ReproduceArgs) {
    let provider = parse_provider(&args.provider).unwrap_or_else(|e| panic!("{e}"));
    let (runs, rounds) = if args.quick {
        (5usize, args.rounds.min(5_000))
    } else {
        (args.runs, args.rounds)
    };
    let runs = runs.max(1);
    // 再現バッチは LLM の設定を CLI で受け取らない (付録 F の照合は古典経路)．
    let cache_path = ".llm_cache/cache.json";
    let llm = make_llm_settings(0.0, 0, cache_path);
    prepare_cache_dir(provider, cache_path);
    let client = provider.is_llm().then(|| build_client(&llm));
    let llm_identity = client.as_ref().map(client_identity);

    let parent_parameters = ReproduceParameters {
        provider: provider.label(),
        conditions: REPRO_CONDITIONS.iter().map(|c| c.id).collect(),
        runs,
        rounds,
        quick: args.quick,
        base_seed: args.seed,
    };

    let parent = Run::start(
        RunOptions::new(EXPERIMENT, "reproduce")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&parent_parameters)
            .expect("runvault: reproduce の parameters の組み立てに失敗")
            .seed_pointers(["/base_seed"])
            .sweep_parent()
            .replication(record::replication()),
    )
    .expect("runvault: reproduce 親 run の開始に失敗");

    let sweep_id = parent
        .sweep_id()
        .expect("runvault: reproduce 親に sweep_id がありません")
        .to_string();
    let parent_run_uid = parent.run_uid().to_string();
    let parent_artifacts = parent
        .dir()
        .join("artifacts")
        .to_string_lossy()
        .into_owned();
    ensure_output_dir(&parent_artifacts);

    println!("=== YuLan-OneSim — Appendix F / Axelrod Table 7-2 reproduction ===");
    println!(
        "provider: {} | runs={} | rounds={} | quick={} | output: {}",
        provider.label(),
        runs,
        rounds,
        args.quick,
        parent.dir().display(),
    );
    println!("----------------------------------------------------------------------");

    let mut verdicts: Vec<VerdictRow> = Vec::with_capacity(REPRO_CONDITIONS.len());

    for cond in REPRO_CONDITIONS.iter() {
        let params = ReproduceConditionParameters {
            condition_id: cond.id,
            provider: provider.label(),
            width: cond.width,
            height: cond.height,
            features: cond.features,
            traits: cond.traits,
            runs,
            rounds,
            events_per_step: cond.width * cond.height,
            snapshot_interval: 0,
            base_seed: args.seed,
        };

        let mut child_options = RunOptions::new(EXPERIMENT, "reproduce-condition")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&params)
            .expect("runvault: 条件の parameters の組み立てに失敗")
            .seed_pointers(["/base_seed"])
            .master_seed(args.seed)
            .replicate_index(0)
            .lineage(Lineage {
                sweep_id: Some(sweep_id.clone()),
                parent_run_uid: Some(parent_run_uid.clone()),
                ..Default::default()
            })
            .replication(record::replication());
        if let Some((model, endpoint)) = &llm_identity {
            child_options = child_options.llm(record::llm_block(model, endpoint, 0.0));
        }
        let mut child = Run::start(child_options).expect("runvault: 条件の子 run の開始に失敗");

        // 論文の報告値は reference.csv の担当．観測値 (mean_n_stable_regions) と
        // 同じ名前で並ぶので，差は後から名前で突き合わせて取れる．こちらが置いた
        // 許容幅は入れない — 原著が印字した数ではないからである．
        child
            .log_reference("mean_n_stable_regions", cond.target_regions)
            .scope("run")
            .target("table7-2")
            .source(format!(
                "Axelrod (1997) Table 7-2 (F={}, q={}), via Wang et al. (2025) Appendix F",
                cond.features, cond.traits
            ))
            .send()
            .expect("論文の報告値の記録に失敗");

        let mut trials: Vec<TrialOutcome> = Vec::with_capacity(runs);
        let mut tally = LlmTally::default();
        for run_idx in 0..runs {
            let seed = record::trial_seed(args.seed, cond.features, cond.traits, run_idx);
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
                llm: llm.clone(),
            };
            let result = run_with_shared_client(&cfg, client.clone())
                .unwrap_or_else(|e| panic!("reproduce run failed: {e}"));
            record::log_terminal(
                &mut child,
                &format!("trial-{run_idx}"),
                seed,
                rounds,
                [result.final_round as u64],
                &result,
            );
            tally.add(&result);
            trials.push(TrialOutcome::from_result(&result));
        }
        record::log_condition_summary(&mut child, &trials);
        record::log_llm_totals(&mut child, tally.calls, tally.hits);

        let denom = trials.len() as f64;
        let mean_regions = trials
            .iter()
            .map(|t| t.n_stable_regions as f64)
            .sum::<f64>()
            / denom;
        let mean_lc = trials.iter().map(|t| t.lc).sum::<f64>() / denom;
        let mean_gpn = trials.iter().map(|t| t.gp_per_agent).sum::<f64>() / denom;
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
            mean_lc,
            mean_gpn,
        );

        verdicts.push(VerdictRow {
            id: cond.id,
            features: cond.features,
            traits: cond.traits,
            runs,
            target_regions: cond.target_regions,
            observed_mean_regions: mean_regions,
            abs_error,
            tolerance: cond.tol_regions,
            within_tolerance: within,
            child_run_slug: child.run_slug().to_string(),
        });

        child.finish().expect("runvault: 条件の子 run の完了に失敗");
    }

    // 許容幅つきの判定表は指標でも報告値でもないので artifacts/ に残す．
    {
        let path = format!("{parent_artifacts}/reproduce_verdicts.csv");
        let mut wtr = csv::Writer::from_path(&path).expect("failed to create verdict CSV");
        for row in &verdicts {
            wtr.serialize(row).expect("verdict row write failed");
        }
        wtr.flush().expect("verdict CSV flush failed");
    }

    let n_pass = verdicts.iter().filter(|v| v.within_tolerance).count();
    let n_total = verdicts.len();
    let dir = parent
        .finish()
        .expect("runvault: reproduce 親 run の完了に失敗");
    println!("----------------------------------------------------------------------");
    println!("reproduce done: {n_pass}/{n_total} conditions within tolerance.");
    println!(
        "判定表 → {}/artifacts/reproduce_verdicts.csv",
        dir.display()
    );
    println!("報告値 → 各条件の子 run (subcommand=reproduce-condition) の reference.csv");
    println!("観測値 → 同じ子 run の metrics.csv (mean_n_stable_regions ほか)");
}

// --------------------------------------------------------------------------- //
// compare (classical vs LLM)
// --------------------------------------------------------------------------- //

/// 比較の片側 1 本を子 run として回す．
///
/// 両側とも «同じ盤面・同じシード・同じラウンド数で回した 1 本のシミュレーション»
/// で，機構だけが違う．したがってこれは 1 回の実行の中の比較ではなく，模型の別々の
/// 実行が 2 つある — 子 run に割っても起きていない実行を主張することにはならない．
/// 逆に 1 本の run に押し込むと，`metrics.csv` の主キーが両側で衝突し，`llm`
/// ブロックが «モデル無し» と «llama3.2» の 2 つを同時に名乗ることになる．
fn run_compare_side(
    args: &CompareArgs,
    side: &'static str,
    provider: Provider,
    client: Option<SharedClient>,
    llm: &LlmSettings,
    sweep_id: &str,
    parent_run_uid: &str,
) -> SimulationResult {
    let params = CompareSideParameters {
        side,
        provider: provider.label(),
        mock: args.mock,
        width: args.width,
        height: args.height,
        features: args.features,
        traits: args.traits,
        events_per_step: args.width * args.height,
        rounds: args.rounds,
        snapshot_interval: 0,
        seed: args.seed,
        llm_temperature: args.temperature,
        llm_seed: args.llm_seed,
        llm_cache_path: cache_path_of(provider, &args.cache_path),
    };

    let mut options = RunOptions::new(EXPERIMENT, "compare-side")
        .repo_id(REPO_ID)
        .domain(DOMAIN)
        .results_root(&args.output_dir)
        .parameters(&params)
        .expect("runvault: 比較の子の parameters の組み立てに失敗")
        .hash_exclude(CACHE_POINTER)
        .seed_pointers(["/seed"])
        .master_seed(args.seed)
        .replicate_index(0)
        .lineage(Lineage {
            sweep_id: Some(sweep_id.to_string()),
            parent_run_uid: Some(parent_run_uid.to_string()),
            ..Default::default()
        })
        .replication(record::replication());
    if let Some(c) = &client {
        let (model, endpoint) = client_identity(c);
        options = options.llm(record::llm_block(&model, &endpoint, args.temperature));
    }
    let mut rv = Run::start(options).expect("runvault: 比較の子 run の開始に失敗");

    let mut cfg = Config {
        width: args.width,
        height: args.height,
        features: args.features,
        traits: args.traits,
        events_per_step: 0,
        rounds: args.rounds,
        snapshot_interval: 0,
        provider,
        seed: Some(args.seed),
        llm: llm.clone(),
    };
    if args.mock && provider.is_llm() {
        // The mock client uses an in-memory cache (no file to save to), so clear
        // the cache_path to skip the (would-fail) save step.
        cfg.llm.cache_path = None;
    }

    let result = run_with_shared_client(&cfg, client).unwrap_or_else(|e| {
        panic!("{side} run failed: {e}. In a network-free environment, pass --mock.")
    });

    record::log_round_metrics(&mut rv, &result.round_history);
    record::log_run_scope(&mut rv, &result);
    record::log_llm_metrics(&mut rv, &result.metadata);
    let observed: Vec<u64> = result
        .round_history
        .iter()
        .map(|r| r.round as u64)
        .collect();
    record::log_terminal(&mut rv, side, args.seed, args.rounds, observed, &result);

    let artifacts = rv.dir().join("artifacts").to_string_lossy().into_owned();
    ensure_output_dir(&artifacts);
    save_culture_grid(&result.world, &artifacts, "culture_grid_final.csv");

    rv.finish().expect("runvault: 比較の子 run の完了に失敗");
    result
}

fn cmd_compare(args: CompareArgs) {
    let llm_provider = parse_provider(&args.llm_provider).unwrap_or_else(|e| panic!("{e}"));
    if !llm_provider.is_llm() {
        panic!(
            "--llm-provider must be an LLM provider (ollama / openai), got {}",
            args.llm_provider
        );
    }
    prepare_cache_dir(llm_provider, &args.cache_path);
    let llm = make_llm_settings(args.temperature, args.llm_seed, &args.cache_path);
    let llm_label: &'static str = if args.mock { "llm-mock" } else { "llm-live" };

    let parent_parameters = CompareParameters {
        sides: vec!["classical", llm_label],
        llm_provider: llm_provider.label(),
        mock: args.mock,
        width: args.width,
        height: args.height,
        features: args.features,
        traits: args.traits,
        rounds: args.rounds,
        seed: args.seed,
        llm_temperature: args.temperature,
        llm_seed: args.llm_seed,
        llm_cache_path: cache_path_of(llm_provider, &args.cache_path),
    };

    // 親は 2 本のシミュレーションではないので master_seed を名乗らない．両側に
    // 共通のシードは /parameters.seed と seed_pointers 経由で execution_hash に残る．
    let mut parent = Run::start(
        RunOptions::new(EXPERIMENT, "compare")
            .repo_id(REPO_ID)
            .domain(DOMAIN)
            .results_root(&args.output_dir)
            .parameters(&parent_parameters)
            .expect("runvault: compare の parameters の組み立てに失敗")
            .hash_exclude(CACHE_POINTER)
            .seed_pointers(["/seed"])
            .sweep_parent()
            .replication(record::replication()),
    )
    .expect("runvault: compare 親 run の開始に失敗");

    let sweep_id = parent
        .sweep_id()
        .expect("runvault: compare 親に sweep_id がありません")
        .to_string();
    let parent_run_uid = parent.run_uid().to_string();

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
    println!("output: {}", parent.dir().display());
    println!("----------------------------------------------------------------------");

    // --- classical side (always live; 0 LLM calls) --- //
    let classical = run_compare_side(
        &args,
        "classical",
        Provider::None,
        None,
        &llm,
        &sweep_id,
        &parent_run_uid,
    );
    println!(
        "[classical] regions={} LC={:.3} GP/N={:.3} converged={} round={} (0 LLM calls)",
        classical.final_metrics.n_stable_regions,
        classical.final_metrics.local_convergence,
        classical.final_metrics.gp_per_agent,
        classical.converged,
        classical.final_round,
    );

    // --- LLM side (mock = offline; otherwise live via env-built client) --- //
    let client = if args.mock {
        share(build_mock_client())
    } else {
        build_client(&llm)
    };
    let llm_result = run_compare_side(
        &args,
        llm_label,
        llm_provider,
        Some(client),
        &llm,
        &sweep_id,
        &parent_run_uid,
    );
    println!(
        "[{}] regions={} LC={:.3} GP/N={:.3} converged={} round={} LLM_calls={} cache_hit={:.1}%",
        llm_label,
        llm_result.final_metrics.n_stable_regions,
        llm_result.final_metrics.local_convergence,
        llm_result.final_metrics.gp_per_agent,
        llm_result.converged,
        llm_result.final_round,
        llm_result.metadata.total(),
        llm_result.metadata.cache_hit_rate() * 100.0,
    );

    // 両側をまたいだ差は «掃引全体の集約» なので親の scope=sweep 指標にする．
    // 各側の値は子が持っているので，同じ数を 2 箇所には置かない．
    let c = &classical.final_metrics;
    let l = &llm_result.final_metrics;
    parent
        .log_metrics(
            record::SWEEP_SCOPE,
            &[
                (
                    "delta_n_stable_regions",
                    l.n_stable_regions as f64 - c.n_stable_regions as f64,
                ),
                ("delta_lc", l.local_convergence - c.local_convergence),
                ("delta_gp", l.global_polarization - c.global_polarization),
                ("delta_gp_per_agent", l.gp_per_agent - c.gp_per_agent),
                (
                    "delta_final_round",
                    llm_result.final_round as f64 - classical.final_round as f64,
                ),
            ],
        )
        .expect("比較の差の記録に失敗");

    let dir = parent
        .finish()
        .expect("runvault: compare 親 run の完了に失敗");
    println!("----------------------------------------------------------------------");
    println!("compare done.");
    println!(
        "差 (LLM − classical) → {}/metrics.csv (scope=sweep)",
        dir.display()
    );
    println!("各側 → 子 run (subcommand=compare-side)");
}

// --------------------------------------------------------------------------- //
// main
// --------------------------------------------------------------------------- //

fn main() {
    let cli = Cli::parse();
    if let Some(host) = cli.ollama_host.as_deref() {
        std::env::set_var("OLLAMA_HOST", host);
    }
    match cli.command {
        Commands::Run(args) => cmd_run(args),
        Commands::Sweep(args) => cmd_sweep(args),
        Commands::Reproduce(args) => cmd_reproduce(args),
        Commands::Compare(args) => cmd_compare(args),
    }
}
