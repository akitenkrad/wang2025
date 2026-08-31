//! runvault への記録の共通部分．
//!
//! 論文メタデータ (research) は `run` / `sweep` / `reproduce` / `compare` の
//! どれでも同一なので，ここ 1 箇所で組み立てる．ラウンドごとの指標，
//! シミュレーション 1 本の終端行，条件 1 点ぶんの集約もここに集める．

use runvault::{Llm, Replication, Run, Target, Work};
use serde::Serialize;

use socsim_llm::MetadataCollector;

use crate::simulation::{RoundMetrics, SimulationResult};

/// runvault 上の実験名．`runvault path --experiment` に渡す値でもある．
/// バイナリ名と揃えて，姉妹実装 `axelrod1997` と取り違えないようにする．
pub const EXPERIMENT: &str = "culture-llm";
/// リポジトリの安定 id．git remote の名前とは独立に固定する．
pub const REPO_ID: &str = "wang2025";
/// 分野．文化ベクトルの初期化・サイト/隣人の抽選・活性化順がいずれも乱数駆動で，
/// `master_seed` が要る．LLM が採用判断を担うが，測っているのはモデルの安全性では
/// なく Axelrod 文化伝播の創発なので `llm-safety` ではない．LLM 側の同一性は
/// `run.json` の `llm` ブロックが持つ．
pub const DOMAIN: &str = "simulation";

/// 時間軸の単位．
///
/// このモデルの 1 刻みは «エンジンの 1 tick» — `events_per_step` 個の micro-event
/// (サイト抽選 → 隣人抽選 → 採用判断) を回してから収束判定と LC/GP 集計をする 1 巡
/// である．旧 `metrics.csv` の列名も `round` だった．語彙の `round` を使う
/// (同じく LLM 駆動で 1 巡 = 1 round の zhao2024 / gao2023 と揃える)．
pub const T_UNIT: &str = "round";

/// ラウンドごとの指標と run 全体の指標の粒度．いずれも盤面全体の集約なので `run`．
const SCOPE: &str = "run";

/// 掃引・比較の親が持つ «条件をまたいだ集約» の粒度．
pub const SWEEP_SCOPE: &str = "sweep";

/// この再現実験が対象としている論文．
///
/// `run` も掃引の子も比較の子も同じ主張を対象とする — 掃引は F × q を変えて
/// 単文化⇄多文化の転移条件を見るためのもので，別の対象を持たない．
///
/// `Target::table` の `table7-2` は Axelrod (1997) Table 7-2 で，本論文の付録 F が
/// LLM 版の検証に使っている定量ベンチマークそのものである．付録 F の Figure 9
/// (LC が round 60 までに 0.50 超・GP が 0.35–0.40 で安定) は figure から読んだ帯で
/// あって印字された数ではないので，`reference.csv` には入れず docs に残す．
pub fn replication() -> Replication {
    let mut work = Work::arxiv("2505.07581")
        .title(
            "YuLan-OneSim: Towards the Next Generation of Social Simulator with Large Language \
             Models",
        )
        .year(2025)
        .source_version("preprint");
    // vault 側の同定にも使えるよう paper-id も残す (work_id は arXiv 側)．
    work.paper_id = Some("P00001802".to_string());
    Replication::new(work)
        .target(Target::table(
            "table7-2",
            "Axelrod (1997) Table 7-2 — mean number of stable culture regions",
        ))
        .target(Target::claim(
            "local-homogenisation-with-global-diversity",
            "Local interaction homogenises culture regionally while global cultural diversity \
             survives",
        ))
        .obsidian_note("研究/98_論文レポート/80-再現実験/実装完了/wang2025/設計書.md")
}

// --------------------------------------------------------------------------- //
// LLM ブロック
// --------------------------------------------------------------------------- //

/// 実際に応答したバックエンドを `run.json` の `llm` ブロックに落とす．
///
/// `model` / `endpoint` はクライアントが名乗った値をそのまま使う．`provider` は
/// runvault の語彙ではなく自由記述なので，endpoint から «どのゲートウェイが答えたか»
/// を決める．推測しているのは分類だけで，値そのものは記録から採る．
///
/// **LLM を 1 度も叩かない古典経路の run には呼ばない** — «モデル none» を名乗らせ
/// ないためである (呼び出し側が provider を見て分岐する)．
pub fn llm_block(model: &str, endpoint: &str, temperature: f32) -> Llm {
    let provider = if endpoint.starts_with("mock://") {
        "mock"
    } else if endpoint.contains("openai") {
        "openai"
    } else {
        "ollama"
    };
    Llm {
        provider: provider.to_string(),
        model_snapshot: model.to_string(),
        temperature: Some(temperature as f64),
        // 採用判断のプロンプトは相手の文化ベクトルごとに組み立てられ，固定の
        // system prompt を持たない．無いものを hash しない．
        system_prompt_hash: None,
    }
}

// --------------------------------------------------------------------------- //
// ラウンドごとの指標
// --------------------------------------------------------------------------- //

/// ラウンドごとの 6 指標を書く (`round` は時間軸なので値としては書かない)．
///
/// 旧 `metrics.csv` の列名 (`lc` / `gp` / `gp_per_agent` / `n_stable_regions` /
/// `max_region_size` / `n_distinct_cultures`) をそのまま指標名にする．いずれも
/// 盤面全体を 1 つの数に畳んだ量で，系列 (どのサイトか) を持たない．
pub fn log_round_metrics(run: &mut Run, history: &[RoundMetrics]) {
    for r in history {
        run.log_metrics_at(
            r.round as u64,
            T_UNIT,
            SCOPE,
            &[
                ("lc", r.lc),
                ("gp", r.gp),
                ("gp_per_agent", r.gp_per_agent),
                ("n_stable_regions", r.n_stable_regions as f64),
                ("max_region_size", r.max_region_size as f64),
                ("n_distinct_cultures", r.n_distinct_cultures as f64),
            ],
        )
        .unwrap_or_else(|e| panic!("round {} の指標の記録に失敗: {e}", r.round));
    }
}

/// run 全体を 1 つの値で表す量．
///
/// `converged` は «吸収状態に達したか» の 0/1 指標変数で，カテゴリに番号を振った
/// ものではない — 複数 run にわたる平均が収束率そのものになる．
pub fn log_run_scope(run: &mut Run, result: &SimulationResult) {
    run.log_metrics(
        SCOPE,
        &[
            ("converged", if result.converged { 1.0 } else { 0.0 }),
            ("final_round", result.final_round as f64),
        ],
    )
    .expect("run スコープの指標の記録に失敗");
}

/// LLM の呼び出し数と cache-hit．
///
/// 旧 `run_metadata.json` の `total_calls` / `cache_hits` / `cache_hit_rate` に
/// あたる．**呼び出しが 1 本も無ければ行そのものを書かない** — 率は分母が 0 の
/// とき «0» ではなく «定義できない» からで，欠測を 0 で埋めない．古典経路
/// (`--provider none`) は必ず 0 呼び出しなので，この関数は何も書かない．
pub fn log_llm_metrics(run: &mut Run, meta: &MetadataCollector) {
    log_llm_totals(run, meta.total(), meta.cache_hits());
}

/// 同じ 3 指標を，複数の試行にわたる合計から書く．
///
/// 掃引と再現の子は 1 本の run が `runs` 本の試行を持つので，収集器も試行ごとに
/// 分かれている．どれか 1 本だけを名乗らせず，その条件で実際に投げた総数を書く．
pub fn log_llm_totals(run: &mut Run, calls: usize, cache_hits: usize) {
    if calls == 0 {
        return;
    }
    run.log_metrics(
        SCOPE,
        &[
            ("llm_calls", calls as f64),
            ("llm_cache_hits", cache_hits as f64),
            ("llm_cache_hit_rate", cache_hits as f64 / calls as f64),
        ],
    )
    .expect("LLM 指標の記録に失敗");
}

// --------------------------------------------------------------------------- //
// 終端イベント
// --------------------------------------------------------------------------- //

/// `events.jsonl` に書く観測行．
///
/// 予約キーだけを持つ．数はここには書かない — ラウンドごとの値は `metrics.csv`
/// (run スコープ) が，試行の最終値は下の [`TerminalEvent`] が正本なので，同じ数を
/// 2 箇所に置くと食い違う余地ができる．この行が持つのは「その単位をいつ見たか」
/// という時間軸だけである．
///
/// `terminal` 行だけでも生存時間解析は組めるが (`schema/v1/event.json` の
/// terminal の注記)，`runvault verify --deep` は terminal の `unit_id` が
/// observation にも現れることを要求するので，観測した時刻を明示的に残す．
#[derive(Serialize)]
struct ObservationEvent<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
}

/// 観測 1 点を書く．
fn log_observation(run: &mut Run, unit_id: &str, t: u64) {
    run.log_event(
        "observation",
        &ObservationEvent {
            unit_id,
            t,
            t_unit: T_UNIT,
        },
    )
    .unwrap_or_else(|e| panic!("{unit_id} の t={t} の observation の記録に失敗: {e}"));
}

/// `events.jsonl` に書く終端行．
///
/// 先頭 6 フィールドは runvault の予約語 (`terminal` はこれを全部要求する)．
/// 残りは自由欄で，旧 `sweep_summary.csv` / `reproduce_detail.csv` の 1 行が
/// この 1 行に対応する．
///
/// 欄名は掃引パラメータと重ならないようにしてある — `sweep_events_table` は同名の
/// パラメータ列でイベント列を上書きするので，衝突すると黙って消える．試行ごとの
/// シードはここでは `seed`，掃引の条件側は `base_seed` と別名にした．
#[derive(Serialize)]
struct TerminalEvent<'a> {
    unit_id: &'a str,
    t: u64,
    t_unit: &'static str,
    outcome: &'static str,
    censored: bool,
    budget: u64,
    seed: u64,
    n_stable_regions: usize,
    max_region_size: usize,
    n_distinct_cultures: usize,
    lc: f64,
    gp: f64,
    gp_per_agent: f64,
}

/// シミュレーション 1 本を `terminal` イベントとして書く．
///
/// 打ち切り (`censored`) の行は `t == budget` でなければならない．ドライバは
/// `ConvergenceMechanism` が吸収状態を見つけたときだけ `request_stop` を掛け，
/// それ以外は `rounds` まで回すので，収束しなかった run は必ず上限に達している．
/// この不変条件は runvault が `log_event` の書き込み時に検査するので，ここでは
/// 二重に持たない．
///
/// `observed` はこの単位を観測した時刻の列で，終端の `t` を必ず含む．`run` は
/// 全ラウンドを観測して `metrics.csv` に残すので全ラウンドを，掃引と再現は各試行の
/// 最終ラウンドしか見ないのでその 1 点だけを渡す．
pub fn log_terminal(
    run: &mut Run,
    unit_id: &str,
    seed: u64,
    rounds: usize,
    observed: impl IntoIterator<Item = u64>,
    result: &SimulationResult,
) {
    for t in observed {
        log_observation(run, unit_id, t);
    }

    let m = &result.final_metrics;
    let event = TerminalEvent {
        unit_id,
        t: result.final_round as u64,
        t_unit: T_UNIT,
        outcome: if result.converged {
            "converged"
        } else {
            "unconverged"
        },
        censored: !result.converged,
        budget: rounds as u64,
        seed,
        n_stable_regions: m.n_stable_regions,
        max_region_size: m.max_region_size,
        n_distinct_cultures: m.n_distinct_cultures,
        lc: m.local_convergence,
        gp: m.global_polarization,
        gp_per_agent: m.gp_per_agent,
    };
    run.log_event("terminal", &event)
        .unwrap_or_else(|e| panic!("{unit_id} の terminal イベントの記録に失敗: {e}"));
}

// --------------------------------------------------------------------------- //
// 条件 1 点ぶんの集約 (掃引 / 再現の子 run)
// --------------------------------------------------------------------------- //

/// 1 つの条件で回した試行群の最終値．集約の材料になる．
pub struct TrialOutcome {
    /// 吸収状態に達したか．
    pub converged: bool,
    /// 収束 (または打ち切り) したラウンド．
    pub final_round: usize,
    /// 安定文化領域の数．
    pub n_stable_regions: usize,
    /// 最大領域の大きさ．
    pub max_region_size: usize,
    /// 盤面上の異なる文化の数．
    pub n_distinct_cultures: usize,
    /// 局所収束 LC．
    pub lc: f64,
    /// 大域分極 GP (文書どおりの `|C| / N²`)．
    pub gp: f64,
    /// 補助の `|C| / N`．
    pub gp_per_agent: f64,
}

impl TrialOutcome {
    /// [`SimulationResult`] の最終状態から取り出す．
    pub fn from_result(result: &SimulationResult) -> Self {
        let m = &result.final_metrics;
        TrialOutcome {
            converged: result.converged,
            final_round: result.final_round,
            n_stable_regions: m.n_stable_regions,
            max_region_size: m.max_region_size,
            n_distinct_cultures: m.n_distinct_cultures,
            lc: m.local_convergence,
            gp: m.global_polarization,
            gp_per_agent: m.gp_per_agent,
        }
    }
}

/// 1 条件を 1 つの値で表す指標．
///
/// 試行ごとの値は `events.jsonl` の担当なので，ここには集約しか書かない．試行
/// ごとの `n_stable_regions` を指標にすると (`run_uid`, `step`, `scope`, `name`)
/// が重複するので，散らばりが要る図は `events.jsonl` から組み直す．
///
/// `n_units` は予約指標名 (観測主体の数)．この条件で観測した試行の本数である．
pub fn log_condition_summary(run: &mut Run, trials: &[TrialOutcome]) {
    let n = trials.len();
    assert!(n > 0, "試行が 1 本もありません");
    let n_f = n as f64;

    let n_converged = trials.iter().filter(|t| t.converged).count();
    let mean = |f: &dyn Fn(&TrialOutcome) -> f64| trials.iter().map(f).sum::<f64>() / n_f;

    run.log_metrics(
        SCOPE,
        &[
            ("n_units", n_f),
            ("n_converged", n_converged as f64),
            ("converged_fraction", n_converged as f64 / n_f),
            (
                "mean_n_stable_regions",
                mean(&|t| t.n_stable_regions as f64),
            ),
            ("mean_max_region_size", mean(&|t| t.max_region_size as f64)),
            (
                "mean_n_distinct_cultures",
                mean(&|t| t.n_distinct_cultures as f64),
            ),
            ("mean_lc", mean(&|t| t.lc)),
            ("mean_gp", mean(&|t| t.gp)),
            ("mean_gp_per_agent", mean(&|t| t.gp_per_agent)),
            ("mean_final_round", mean(&|t| t.final_round as f64)),
        ],
    )
    .expect("条件の集約指標の記録に失敗");
}

// --------------------------------------------------------------------------- //
// シードの派生
// --------------------------------------------------------------------------- //

/// 試行 1 本のシードを base seed から決定的に派生させる．
///
/// `master_seed` として記録するのは実際に世界を支配したこの値の方で，CLI で与えた
/// 根のシードは `/parameters` 側にある．`(base, features, traits, index)` が同じ
/// なら常に同じ値を返し，どれか 1 つでも違えば別の値になる — この性質が壊れると，
/// 記録した `master_seed` から run を組み直せなくなる．
///
/// 引数の並びは移行前の `derive_seed(base, &[features, traits, run_idx])` と同一で，
/// 同じ条件は移行前と同じシードを引く．
pub fn trial_seed(base: u64, features: usize, traits: usize, index: usize) -> u64 {
    socsim_core::derive_seed(base, &[features as u64, traits as u64, index as u64])
}

#[cfg(test)]
mod tests {
    use super::trial_seed;

    #[test]
    fn same_inputs_give_the_same_seed() {
        assert_eq!(trial_seed(42, 5, 10, 3), trial_seed(42, 5, 10, 3));
        for index in 0..8 {
            assert_eq!(
                trial_seed(2026, 5, 15, index),
                trial_seed(2026, 5, 15, index),
                "index={index} で再現しなかった"
            );
        }
    }

    #[test]
    fn each_coordinate_changes_the_seed() {
        let base = trial_seed(42, 5, 10, 0);
        assert_ne!(base, trial_seed(43, 5, 10, 0), "base が効いていない");
        assert_ne!(base, trial_seed(42, 10, 10, 0), "features が効いていない");
        assert_ne!(base, trial_seed(42, 5, 15, 0), "traits が効いていない");
        assert_ne!(base, trial_seed(42, 5, 10, 1), "index が効いていない");
    }

    #[test]
    fn one_condition_gives_distinct_seeds_across_trials() {
        let seeds: std::collections::BTreeSet<u64> =
            (0..64).map(|i| trial_seed(42, 5, 10, i)).collect();
        assert_eq!(seeds.len(), 64, "同一条件の試行でシードが衝突した");
    }

    /// 具体値を固定する．
    ///
    /// 移行前の `sweep_summary.csv` が記録していたシードと同じ値であることを
    /// 押さえる．ここが変わるのは socsim の `derive_seed` が変わったときで，
    /// そのときは過去の run と結果を比較できなくなっている．
    #[test]
    fn golden_values_are_pinned() {
        assert_eq!(trial_seed(42, 5, 5, 0), 8_386_320_814_092_227_575);
        assert_eq!(trial_seed(42, 5, 5, 1), 8_386_319_714_580_599_364);
    }
}
