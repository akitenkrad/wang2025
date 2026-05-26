//! Simulation configuration.
//!
//! Holds the Axelrod culture-dissemination core parameters (grid size, number of
//! features `F`, number of traits `q`, event budget) and the LLM-layer settings,
//! plus the `Provider` switch that selects the classical vs LLM interaction
//! mechanism. The two interaction mechanisms are mutually exclusive.

use serde::Serialize;

// --------------------------------------------------------------------------- //
// Provider
// --------------------------------------------------------------------------- //

/// Interaction-mechanism provider.
///
/// Selects which (mutually exclusive) interaction mechanism the driver wires:
/// - `None`   → the reusable `socsim_mechanisms::AxelrodMechanism`
///   (deterministic Axelrod baseline, no LLM). This is the default and the
///   in-sandbox smoke path.
/// - `Ollama` / `OpenAi` → `LLMInteractionMechanism` (LLM decides whether / which
///   feature to adopt). The provider order at runtime is always «Ollama first →
///   OpenAI fallback»; this enum records which provider the run *requested* as
///   primary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    /// No LLM — the classical deterministic Axelrod rule.
    None,
    /// LLM-driven, Ollama as the requested primary backend.
    Ollama,
    /// LLM-driven, OpenAI as the requested primary backend.
    OpenAi,
}

impl Provider {
    /// Stable lowercase label (used in CSV / JSON / directory names).
    pub fn label(&self) -> &'static str {
        match self {
            Provider::None => "none",
            Provider::Ollama => "ollama",
            Provider::OpenAi => "openai",
        }
    }

    /// Whether this provider uses the LLM interaction mechanism.
    pub fn is_llm(&self) -> bool {
        !matches!(self, Provider::None)
    }
}

/// Parse a [`Provider`] from a CLI string.
pub fn parse_provider(s: &str) -> Result<Provider, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "none" | "classical" | "off" => Ok(Provider::None),
        "ollama" => Ok(Provider::Ollama),
        "openai" => Ok(Provider::OpenAi),
        _ => Err(format!(
            "invalid provider: \"{s}\" (none / ollama / openai)"
        )),
    }
}

// --------------------------------------------------------------------------- //
// LLM settings
// --------------------------------------------------------------------------- //

/// LLM レイヤの設定 (provider / model / temperature / seed / cache)．
///
/// 定義は `socsim-llm` に集約済み (各 replication で同一だった struct を統合)．
/// `crate::config::LlmSettings` パスは re-export で温存する．
pub use socsim_llm::LlmSettings;

// --------------------------------------------------------------------------- //
// Config
// --------------------------------------------------------------------------- //

/// Configuration for a single run.
#[derive(Debug, Clone)]
pub struct Config {
    /// Grid width (columns).
    pub width: usize,
    /// Grid height (rows).
    pub height: usize,
    /// Number of cultural features `F` (length of each culture vector).
    pub features: usize,
    /// Number of traits `q` (values each feature can take, 0..q).
    pub traits: usize,
    /// Number of micro-events per engine tick (default = n_sites).
    pub events_per_step: usize,
    /// Maximum number of engine ticks (rounds).
    pub rounds: usize,
    /// Snapshot interval (in rounds) for culture-grid snapshots (0 disables intermediate).
    pub snapshot_interval: usize,
    /// Interaction provider (none / ollama / openai).
    pub provider: Provider,
    /// Random seed (None = random; governs the socsim core layer only).
    pub seed: Option<u64>,
    /// LLM-layer settings.
    pub llm: LlmSettings,
    /// Output directory.
    pub output_dir: String,
}

impl Default for Config {
    /// Axelrod Table 7-2 baseline (10×10, F=5, q=10), classical provider.
    fn default() -> Self {
        Config {
            width: 10,
            height: 10,
            features: 5,
            traits: 10,
            events_per_step: 0, // 0 = derive n_sites at build time
            rounds: 20_000,
            snapshot_interval: 0,
            provider: Provider::None,
            seed: Some(42),
            llm: LlmSettings::default(),
            output_dir: "results".to_string(),
        }
    }
}

impl Config {
    /// Number of sites (= width * height), at least 1.
    pub fn n_sites(&self) -> usize {
        (self.width * self.height).max(1)
    }

    /// Effective events-per-step (defaults to n_sites when 0).
    pub fn effective_events_per_step(&self) -> usize {
        if self.events_per_step == 0 {
            self.n_sites()
        } else {
            self.events_per_step
        }
    }
}

/// JSON representation of a `run`'s `config.json`.
#[derive(Serialize)]
pub struct RunConfigJson {
    pub command: &'static str,
    pub width: usize,
    pub height: usize,
    pub features: usize,
    pub traits: usize,
    pub events_per_step: usize,
    pub rounds: usize,
    pub snapshot_interval: usize,
    pub provider: String,
    pub seed: Option<u64>,
    pub llm_temperature: f32,
    pub llm_seed: u64,
    pub output_dir: String,
}

impl Config {
    /// Build the `config.json` representation.
    pub fn to_run_config_json(&self) -> RunConfigJson {
        RunConfigJson {
            command: "run",
            width: self.width,
            height: self.height,
            features: self.features,
            traits: self.traits,
            events_per_step: self.effective_events_per_step(),
            rounds: self.rounds,
            snapshot_interval: self.snapshot_interval,
            provider: self.provider.label().to_string(),
            seed: self.seed,
            llm_temperature: self.llm.temperature,
            llm_seed: self.llm.seed,
            output_dir: self.output_dir.clone(),
        }
    }
}
