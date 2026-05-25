//! Integration tests for the YuLan-OneSim Axelrod culture-dissemination scenario.
//!
//! **No live LLM required.** The classical path needs no LLM at all; the LLM path
//! is driven by `socsim_llm::mock::ScriptedClient`. Tests cover: classical
//! determinism, similarity, stable-region BFS counting, absorbing-state
//! detection, the Axelrod F/q signs (F↑ → fewer regions, q↑ → more regions),
//! LC monotone-ish increase, mechanism selection by provider, and the LLM
//! mechanism via a scripted client.

use culture_llm::config::{Config, LlmSettings, Provider};
use culture_llm::llm::wrap_client;
use culture_llm::mechanisms::{adoption_prompt, parse_adoption};
use culture_llm::metrics::{
    compute_metrics, count_stable_regions, global_polarization, is_stable, similarity,
};
use culture_llm::simulation::{run_with_client, SimulationResult};
use culture_llm::world::CultureWorld;

use socsim_llm::mock::ScriptedClient;
use socsim_llm::PromptCache;

// --------------------------------------------------------------------------- //
// helpers
// --------------------------------------------------------------------------- //

fn classical_cfg(width: usize, height: usize, features: usize, traits: usize) -> Config {
    Config {
        width,
        height,
        features,
        traits,
        events_per_step: 0,
        rounds: 50_000,
        provider: Provider::None,
        seed: Some(42),
        ..Config::default()
    }
}

/// A scripted client that always adopts the FIRST candidate feature index offered.
fn always_adopt_first_client() -> culture_llm::llm::CultureClient {
    let backend = ScriptedClient::new("mock-model", |prompt: &str| {
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

fn mean_regions(width: usize, height: usize, features: usize, traits: usize, runs: usize) -> f64 {
    let mut sum = 0.0;
    for run_idx in 0..runs {
        let mut cfg = classical_cfg(width, height, features, traits);
        cfg.seed = Some(1000 + run_idx as u64);
        let result = run_with_client(&cfg, None).unwrap();
        sum += result.final_metrics.n_stable_regions as f64;
    }
    sum / runs as f64
}

// --------------------------------------------------------------------------- //
// similarity
// --------------------------------------------------------------------------- //

#[test]
fn similarity_basic() {
    assert_eq!(similarity(&[1, 2, 3], &[1, 2, 3]), 1.0);
    assert_eq!(similarity(&[1, 2, 3], &[4, 5, 6]), 0.0);
    assert!((similarity(&[1, 2, 3], &[1, 9, 9]) - (1.0 / 3.0)).abs() < 1e-12);
}

// --------------------------------------------------------------------------- //
// stable-region BFS counting
// --------------------------------------------------------------------------- //

#[test]
fn stable_region_counting() {
    // 2×2 all-identical → 1 region, size 4, 1 distinct culture.
    let cultures = vec![vec![1, 1], vec![1, 1], vec![1, 1], vec![1, 1]];
    let w = CultureWorld::new(2, 2, 2, 5, cultures, 1);
    let (n_regions, max_size, distinct) = count_stable_regions(&w);
    assert_eq!(n_regions, 1);
    assert_eq!(max_size, 4);
    assert_eq!(distinct, 1);

    // 2×2 all-distinct → 4 regions, size 1.
    let cultures = vec![vec![0, 0], vec![1, 1], vec![2, 2], vec![3, 3]];
    let w = CultureWorld::new(2, 2, 2, 5, cultures, 1);
    let (n_regions, max_size, distinct) = count_stable_regions(&w);
    assert_eq!(n_regions, 4);
    assert_eq!(max_size, 1);
    assert_eq!(distinct, 4);
}

// --------------------------------------------------------------------------- //
// absorbing-state detection
// --------------------------------------------------------------------------- //

#[test]
fn absorbing_state_detection() {
    // All identical → stable (every adjacent sim = 1).
    let cultures = vec![vec![1, 1], vec![1, 1], vec![1, 1], vec![1, 1]];
    let w = CultureWorld::new(2, 2, 2, 5, cultures, 1);
    assert!(is_stable(&w));

    // Fully disjoint neighbours → stable (every adjacent sim = 0).
    let cultures = vec![vec![0, 0], vec![1, 1], vec![2, 2], vec![3, 3]];
    let w = CultureWorld::new(2, 2, 2, 5, cultures, 1);
    assert!(is_stable(&w));

    // A partially-matching adjacent pair (sim = 0.5) → not stable.
    let cultures = vec![vec![0, 0], vec![0, 1], vec![2, 2], vec![3, 3]];
    let w = CultureWorld::new(2, 2, 2, 5, cultures, 1);
    assert!(!is_stable(&w));
}

// --------------------------------------------------------------------------- //
// GP documented formula + auxiliary
// --------------------------------------------------------------------------- //

#[test]
fn gp_documented_and_auxiliary() {
    // 100 agents, 10 regions: GP = 10/10000 = 0.001; GP/N = 10/100 = 0.1.
    let (gp, gp_per_agent) = global_polarization(10, 100);
    assert!((gp - 0.001).abs() < 1e-12);
    assert!((gp_per_agent - 0.1).abs() < 1e-12);
}

// --------------------------------------------------------------------------- //
// classical determinism
// --------------------------------------------------------------------------- //

#[test]
fn classical_run_deterministic() {
    let cfg = classical_cfg(6, 6, 5, 8);
    let a = run_with_client(&cfg, None).unwrap();
    let b = run_with_client(&cfg, None).unwrap();
    assert_eq!(a.final_round, b.final_round);
    assert_eq!(a.world.cells.cells(), b.world.cells.cells());
    assert_eq!(a.metadata.total(), 0, "classical path makes 0 LLM calls");
}

// --------------------------------------------------------------------------- //
// Axelrod F/q signs: F↑ → fewer regions, q↑ → more regions
// --------------------------------------------------------------------------- //

#[test]
fn axelrod_feature_sign_more_features_fewer_regions() {
    // q fixed; increasing F should not increase regions (Axelrod: F↑ → fewer).
    let r_f5 = mean_regions(10, 10, 5, 10, 8);
    let r_f15 = mean_regions(10, 10, 15, 10, 8);
    assert!(
        r_f15 <= r_f5 + 1e-9,
        "F↑ should give fewer/equal stable regions (F5={r_f5}, F15={r_f15})"
    );
}

#[test]
fn axelrod_trait_sign_more_traits_more_regions() {
    // F fixed; increasing q should increase regions (Axelrod: q↑ → more).
    let r_q5 = mean_regions(10, 10, 5, 5, 8);
    let r_q15 = mean_regions(10, 10, 5, 15, 8);
    assert!(
        r_q15 >= r_q5 - 1e-9,
        "q↑ should give more/equal stable regions (q5={r_q5}, q15={r_q15})"
    );
}

// --------------------------------------------------------------------------- //
// LC increases over the run (local convergence)
// --------------------------------------------------------------------------- //

#[test]
fn lc_increases_over_run() {
    let cfg = classical_cfg(10, 10, 5, 5);
    let result = run_with_client(&cfg, None).unwrap();
    let first = result.round_history.first().unwrap().lc;
    let last = result.round_history.last().unwrap().lc;
    assert!(
        last >= first,
        "LC should be monotone-ish increasing (first={first}, last={last})"
    );
}

// --------------------------------------------------------------------------- //
// mechanism selection by provider
// --------------------------------------------------------------------------- //

#[test]
fn provider_none_makes_no_llm_calls() {
    let cfg = classical_cfg(5, 5, 5, 5);
    let result = run_with_client(&cfg, None).unwrap();
    assert_eq!(result.metadata.total(), 0);
    assert_eq!(result.llm_model, "none");
}

#[test]
fn provider_llm_requires_client() {
    let mut cfg = classical_cfg(5, 5, 5, 5);
    cfg.provider = Provider::Ollama;
    // No client supplied for an LLM provider → error.
    assert!(run_with_client(&cfg, None).is_err());
}

// --------------------------------------------------------------------------- //
// LLM mechanism via ScriptedClient updates cultures + records calls
// --------------------------------------------------------------------------- //

#[test]
fn llm_mechanism_runs_and_updates() {
    let cfg = Config {
        width: 5,
        height: 4,
        features: 5,
        traits: 5,
        events_per_step: 0,
        rounds: 20,
        provider: Provider::Ollama,
        seed: Some(42),
        llm: LlmSettings::default(),
        ..Config::default()
    };
    let result: SimulationResult =
        run_with_client(&cfg, Some(always_adopt_first_client())).unwrap();
    // The scripted client adopts features, so at least one LLM call is recorded
    // and local convergence ends up >= its initial value.
    assert!(result.metadata.total() > 0, "LLM path should record calls");
    let first = result.round_history.first().unwrap().lc;
    let last = result.round_history.last().unwrap().lc;
    assert!(
        last >= first,
        "adopting features should not decrease LC (first={first}, last={last})"
    );
    let m = compute_metrics(&result.world);
    assert_eq!(
        m.n_distinct_cultures,
        result.final_metrics.n_distinct_cultures
    );
}

#[test]
fn llm_mechanism_deterministic_with_fixed_mock() {
    let cfg = Config {
        width: 5,
        height: 4,
        features: 5,
        traits: 5,
        events_per_step: 0,
        rounds: 15,
        provider: Provider::Ollama,
        seed: Some(7),
        llm: LlmSettings::default(),
        ..Config::default()
    };
    let a = run_with_client(&cfg, Some(always_adopt_first_client())).unwrap();
    let b = run_with_client(&cfg, Some(always_adopt_first_client())).unwrap();
    assert_eq!(
        a.world.cells.cells(),
        b.world.cells.cells(),
        "same seed + same scripted model → identical board"
    );
}

// --------------------------------------------------------------------------- //
// adoption prompt / parsing
// --------------------------------------------------------------------------- //

#[test]
fn adoption_parse_roundtrip() {
    let diffs = vec![1, 3, 4];
    assert_eq!(parse_adoption("3", &diffs), Some(3));
    assert_eq!(parse_adoption("adopt feature 4 please", &diffs), Some(4));
    assert_eq!(parse_adoption("-1", &diffs), None); // adopt nothing
    assert_eq!(parse_adoption("2", &diffs), None); // not a differing feature
    assert_eq!(parse_adoption("nonsense", &diffs), None);
}

#[test]
fn adoption_prompt_contains_candidates() {
    let p = adoption_prompt("a curious person", &[0, 1, 2], &[0, 9, 9], &[], &[1, 2]);
    assert!(p.contains("differ on these feature indices: [1, 2]"));
    assert!(p.contains("a curious person"));
}
