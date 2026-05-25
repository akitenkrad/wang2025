//! LLM client layer (Ollama first → OpenAI fallback + cache).
//!
//! This module is a thin builder over the `socsim-llm` composition API. It
//! confines the **upper layer (non-deterministic LLM)** of the two-layer
//! architecture; the lower deterministic socsim core only touches it through the
//! [`CultureClient`] type alias.
//!
//! # Composition (Ollama first → OpenAI fallback → cache)
//!
//! ```text
//! CachingClient< Box<dyn LlmClient> >
//!   └─ cache: PromptCache (prompt → response; the pseudo-determinism mechanism)
//!      └─ backend = FallbackClient< OllamaClient, OpenAiClient >
//!           ├─ primary:   OllamaClient   (OLLAMA_HOST / OLLAMA_MODEL)
//!           └─ secondary: OpenAiClient   (OPENAI_API_KEY / OPENAI_MODEL)
//! ```
//!
//! `FallbackClient` is provided by socsim-llm ("Ollama first → any error → OpenAI").
//! `CachingClient` layers a prompt→response cache on top, which together with
//! `temperature=0` / fixed `seed` pseudo-determinises re-runs. Tests inject a
//! `socsim_llm::mock::ScriptedClient` through the same [`CultureClient`]; the
//! `impl LlmClient for Box<dyn LlmClient>` forwarding (socsim-llm issue #26)
//! makes a dedicated newtype unnecessary.

use std::path::Path;

use socsim_llm::{CachingClient, LlmClient, LlmConfig, LlmError, PromptCache};

use crate::config::LlmSettings;

/// The caching client type used by this simulation.
///
/// The backend is type-erased into `Box<dyn LlmClient>`: production wires
/// `FallbackClient<OllamaClient, OpenAiClient>`, tests inject a `ScriptedClient`.
pub type CultureClient = CachingClient<Box<dyn LlmClient>>;

/// Build the production «Ollama first → OpenAI fallback + cache» client from
/// environment variables.
///
/// - Ollama: `OLLAMA_HOST` (default `http://localhost:11434`) / `OLLAMA_MODEL`
///   (default `llama3.2:latest`).
/// - OpenAI: `OPENAI_API_KEY` / `OPENAI_MODEL` (default `gpt-4o-mini`); if unset
///   an empty-key placeholder is installed (never called if Ollama succeeds; a
///   Config error only surfaces if both backends fail).
/// - Cache: a JSON file at `settings.cache_path` if present, else in-memory.
pub fn build_live_client(settings: &LlmSettings) -> Result<CultureClient, LlmError> {
    // The «Ollama first → OpenAI fallback → type-erase → cache» assembly is
    // delegated to socsim-llm's `build_live_client` (behaviour is identical to the
    // former hand-rolled implementation). This wrapper is the thin layer that
    // accepts the replication-specific `LlmSettings` (cache_path).
    socsim_llm::build_live_client(settings.cache_path.as_deref().map(Path::new))
}

/// Wrap any [`LlmClient`] (e.g. `mock::ScriptedClient`) in a cache to produce a
/// [`CultureClient`] (mainly for tests / the offline mock smoke).
pub fn wrap_client<C: LlmClient + 'static>(backend: C, cache: PromptCache) -> CultureClient {
    let boxed: Box<dyn LlmClient> = Box::new(backend);
    CachingClient::new(boxed, cache)
}

/// Build the socsim-llm [`LlmConfig`] from [`LlmSettings`].
pub fn llm_config(settings: &LlmSettings) -> LlmConfig {
    LlmConfig::deterministic()
        .with_temperature(settings.temperature)
        .with_seed(settings.seed)
}
