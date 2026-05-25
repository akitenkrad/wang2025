//! Interaction and convergence mechanisms for the culture-dissemination model.
//!
//! The **two-layer boundary** lives here. The lower (deterministic socsim core)
//! layer draws site + neighbour with `ctx.rng` (ChaCha20); the upper
//! (non-deterministic LLM) layer reaches the [`CultureClient`] (cached Ollama →
//! OpenAI fallback) only inside [`LLMInteractionMechanism`].
//!
//! Mechanisms (mutually exclusive interaction; one chosen by `config.provider`):
//! - **classical Axelrod baseline** (`Interaction`, `--provider none`): now the
//!   reusable [`socsim_mechanisms::AxelrodMechanism`], wired to
//!   [`CultureWorld`] via the `socsim-core` `CultureVectors` + `Neighbors`
//!   capability traits. 1 tick = `events_per_step` micro-events; each event picks
//!   a site + random neighbour, computes similarity, and with probability `sim`
//!   copies one differing feature from the neighbour. The pack mechanism was
//!   ported FROM this repo's former `ClassicalInteractionMechanism`, so its rule
//!   and RNG draw order match. (No longer defined here — see `simulation.rs`.)
//! - [`LLMInteractionMechanism`] (`Interaction`): the YuLan-OneSim variant. Same
//!   event-driven framing, but the LLM decides whether / which feature to adopt
//!   given persona + own culture + neighbour culture (JSON decision); the
//!   site's memory is updated. All LLM calls are confined to this mechanism.
//! - [`ConvergenceMechanism`] (`PostStep`): detects the absorbing state
//!   (`request_stop`) and pushes per-round LC / GP to the world history.

use std::cell::RefCell;
use std::rc::Rc;

use rand::Rng;

use socsim_core::{AgentId, Mechanism, Phase, Result, SocsimError, StepContext};
use socsim_llm::MetadataCollector;

use crate::config::LlmSettings;
use crate::llm::{llm_config, CultureClient};
use crate::metrics::{global_polarization, is_stable, local_convergence};
use crate::world::CultureWorld;

/// Shared LLM client (shared between the run driver and the mechanism).
pub type SharedClient = Rc<RefCell<CultureClient>>;
/// Shared metadata collector (cache-hit rate etc., aggregated after the run).
pub type SharedMetadata = Rc<RefCell<MetadataCollector>>;

// --------------------------------------------------------------------------- //
// LLM-driven interaction (YuLan-OneSim variant)
// --------------------------------------------------------------------------- //

/// Build the adoption prompt for the LLM variant.
///
/// Gives the site's persona, its own culture vector, and the neighbour's culture
/// vector, and asks for a JSON decision of which differing feature (if any) to
/// adopt. The prompt is built deterministically so a warm cache replays identical
/// responses (pseudo-determinism).
pub fn adoption_prompt(
    persona: &str,
    own: &[usize],
    neighbor: &[usize],
    memory: &[String],
    diffs: &[usize],
) -> String {
    let own_s = format_vec(own);
    let nb_s = format_vec(neighbor);
    let diffs_s = diffs
        .iter()
        .map(|f| f.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let mem = if memory.is_empty() {
        "(no prior memory)".to_string()
    } else {
        let start = memory.len().saturating_sub(3);
        memory[start..].join(" | ")
    };
    format!(
        "You are a person with this persona: {persona}.\n\
         Your cultural feature vector is: [{own_s}] (feature index : trait value).\n\
         A neighbour you are talking to has the vector: [{nb_s}].\n\
         You differ on these feature indices: [{diffs_s}].\n\
         Recent memory: {mem}\n\n\
         People tend to adopt traits from others they already share culture with \
         (homophily). Decide whether to adopt ONE of the differing features from \
         your neighbour. Answer with a SINGLE integer: the feature index to adopt \
         (must be one of [{diffs_s}]), or -1 to adopt nothing. Answer with only the integer."
    )
}

/// Format a culture vector as `idx:val` pairs for the prompt.
fn format_vec(v: &[usize]) -> String {
    v.iter()
        .enumerate()
        .map(|(i, &t)| format!("{i}:{t}"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Parse the LLM's adoption decision: the first integer in the text. Returns the
/// chosen feature index if it is a valid differing feature, else `None`
/// (adopt nothing / unparseable / out of set).
pub fn parse_adoption(text: &str, diffs: &[usize]) -> Option<usize> {
    // Extract the first (optionally leading-minus) integer token.
    let mut token = String::new();
    for ch in text.chars() {
        let is_minus = ch == '-' && token.is_empty();
        if is_minus || ch.is_ascii_digit() {
            token.push(ch);
        } else if !token.is_empty() {
            break;
        }
    }
    let val: i64 = token.parse().ok()?;
    if val < 0 {
        return None;
    }
    let feat = val as usize;
    if diffs.contains(&feat) {
        Some(feat)
    } else {
        None
    }
}

/// LLM-driven culture-adoption mechanism (`Interaction` phase).
///
/// Same event-driven framing as the classical mechanism (site + neighbour drawn
/// with `ctx.rng`, deterministic core), but the *adoption decision* is delegated
/// to the LLM. All LLM calls are confined to this mechanism. The client and
/// metadata collector are shared with the run driver via `Rc<RefCell<…>>`.
pub struct LLMInteractionMechanism {
    client: SharedClient,
    metadata: SharedMetadata,
    settings: LlmSettings,
    events_per_step: usize,
}

impl LLMInteractionMechanism {
    /// Build from a shared client, shared metadata, LLM settings, and the
    /// per-tick event count.
    pub fn new(
        client: SharedClient,
        metadata: SharedMetadata,
        settings: LlmSettings,
        events_per_step: usize,
    ) -> Self {
        LLMInteractionMechanism {
            client,
            metadata,
            settings,
            events_per_step: events_per_step.max(1),
        }
    }

    /// Process one LLM-driven adoption event.
    fn one_event(&self, ctx: &mut StepContext<'_, CultureWorld>) -> Result<()> {
        let n = ctx.world.n_sites();
        if n == 0 {
            return Ok(());
        }
        // --- site + neighbour drawn deterministically (socsim core) --- //
        let s = ctx.rng.gen_range(0..n);
        let neighbors = ctx.world.adjacency.neighbors(s);
        if neighbors.is_empty() {
            return Ok(());
        }
        let nb = neighbors[ctx.rng.gen_range(0..neighbors.len())];
        if s == nb {
            return Ok(());
        }

        let own = ctx.world.culture(s).clone();
        let neighbor = ctx.world.culture(nb).clone();
        let diffs: Vec<usize> = (0..ctx.world.n_features)
            .filter(|&f| own[f] != neighbor[f])
            .collect();
        // Identical (no diffs) or fully disjoint → like the classical absorbing
        // edge, nothing to adopt; skip the LLM call entirely.
        if diffs.is_empty() || diffs.len() == ctx.world.n_features {
            return Ok(());
        }

        let sid = AgentId(s as u64);
        let persona = ctx
            .world
            .personas
            .get(&sid)
            .cloned()
            .unwrap_or_else(|| "an ordinary person".to_string());
        let memory = ctx.world.memory.get(&sid).cloned().unwrap_or_default();

        let prompt = adoption_prompt(&persona, &own, &neighbor, &memory, &diffs);

        // --- LLM decision (cached; Ollama → OpenAI fallback) --- //
        let decision = {
            let mut client = self.client.borrow_mut();
            let resp = client
                .complete(&prompt, &llm_config(&self.settings))
                .map_err(|e| SocsimError::Mechanism(format!("LLM call failed: {e}")))?;
            self.metadata.borrow_mut().record(resp.metadata.clone());
            resp.text
        };

        // --- apply adoption + update memory --- //
        if let Some(feat) = parse_adoption(&decision, &diffs) {
            let new_val = neighbor[feat];
            ctx.world
                .cells
                .get_idx_mut(s)
                .expect("idx out of range (llm event write)")[feat] = new_val;
            if let Some(mem) = ctx.world.memory.get_mut(&sid) {
                mem.push(format!(
                    "adopted feature {feat} -> {new_val} from neighbour"
                ));
            }
        } else if let Some(mem) = ctx.world.memory.get_mut(&sid) {
            mem.push("kept my culture (adopted nothing)".to_string());
        }
        Ok(())
    }
}

impl Mechanism<CultureWorld> for LLMInteractionMechanism {
    fn name(&self) -> &str {
        "llm_interaction"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::Interaction]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, CultureWorld>) -> Result<()> {
        for _ in 0..self.events_per_step {
            self.one_event(ctx)?;
        }
        ctx.scratch.insert("delta_events", self.events_per_step);
        Ok(())
    }
}

// --------------------------------------------------------------------------- //
// Convergence + per-round LC / GP recording
// --------------------------------------------------------------------------- //

/// Convergence mechanism (`PostStep` phase).
///
/// Each step: compute LC / GP and push them to the world's history buffers, then
/// detect the absorbing state (every adjacent pair has `sim ∈ {0, 1}`) and
/// `request_stop` if reached. Also writes `converged` to `scratch` for the driver.
pub struct ConvergenceMechanism;

impl Mechanism<CultureWorld> for ConvergenceMechanism {
    fn name(&self) -> &str {
        "convergence"
    }

    fn phases(&self) -> &'static [Phase] {
        &[Phase::PostStep]
    }

    fn apply(&mut self, _phase: Phase, ctx: &mut StepContext<'_, CultureWorld>) -> Result<()> {
        let lc = local_convergence(ctx.world);
        // count_stable_regions returns (n_regions, max, n_distinct); GP uses n_regions.
        let (n_regions, _max, _distinct) = crate::metrics::count_stable_regions(ctx.world);
        let (gp, _gp_per_agent) = global_polarization(n_regions, ctx.world.n_sites());

        ctx.world.lc_history.push(lc);
        ctx.world.gp_history.push(gp);

        let stable = is_stable(ctx.world);
        ctx.scratch.insert("converged", stable);
        ctx.scratch.insert("lc", lc);
        ctx.scratch.insert("gp", gp);
        if stable {
            ctx.request_stop();
        }
        Ok(())
    }
}
