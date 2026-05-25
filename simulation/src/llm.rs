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

use socsim_llm::{
    CachingClient, FallbackClient, LlmClient, LlmConfig, LlmError, OllamaClient, OpenAiClient,
    PromptCache,
};

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
    let ollama = OllamaClient::from_env();
    let openai = OpenAiClient::from_env().unwrap_or_else(|_| {
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| "gpt-4o-mini".into());
        OpenAiClient::new("", model)
    });

    let fallback = FallbackClient::new(ollama, openai);
    let backend: Box<dyn LlmClient> = Box::new(fallback);

    let cache = match &settings.cache_path {
        Some(path) => PromptCache::open(path)?,
        None => PromptCache::in_memory(),
    };
    Ok(CachingClient::new(backend, cache))
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
