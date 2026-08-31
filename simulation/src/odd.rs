//! Behaviour-graph / ODD-protocol auto-construction (concept demo).
//!
//! YuLan-OneSim builds, from a no-code scenario description, both an **ODD
//! protocol** document and an internal **behaviour graph** (agents → events →
//! state updates) that the engine executes. Reproducing the *full* LLM-driven
//! scenario-construction pipeline is out of scope for this single-scenario
//! replication. This module instead provides a **clearly-scoped, deterministic
//! concept demo**: it derives a structured ODD outline and a behaviour graph
//! **from the already-fixed model** (the [`Config`] + the wired mechanisms),
//! rather than synthesising them with an LLM.
//!
//! In other words: the real YuLan-OneSim direction is *NL scenario → ODD +
//! behaviour graph → executable model*. Here the model is fixed (the Axelrod
//! culture-dissemination scenario), so we run that map **in reverse**, emitting
//! the ODD/graph artefacts the paper would have produced for this scenario. The
//! output is a single `behavior_graph.json` (ODD sections + a node/edge graph)
//! that the Python tools can render to a diagram. This is a faithful structured
//! description, not an LLM-generated one — see the README/architecture notes.

use serde::Serialize;

use crate::config::Config;

/// A node in the behaviour graph (an agent role, an event, or a state variable).
#[derive(Debug, Clone, Serialize)]
pub struct GraphNode {
    /// Stable node id (referenced by edges).
    pub id: String,
    /// Node kind: `agent` / `event` / `state` / `metric`.
    pub kind: String,
    /// Human-readable label.
    pub label: String,
}

/// A directed edge in the behaviour graph (`from` → `to`, with a relation verb).
#[derive(Debug, Clone, Serialize)]
pub struct GraphEdge {
    /// Source node id.
    pub from: String,
    /// Target node id.
    pub to: String,
    /// Relation verb (`triggers` / `reads` / `writes` / `produces`).
    pub relation: String,
}

/// The seven ODD-protocol sections (Grimm et al.) for this scenario.
#[derive(Debug, Clone, Serialize)]
pub struct OddProtocol {
    pub purpose: String,
    pub entities_state_variables_scales: String,
    pub process_overview_scheduling: String,
    pub design_concepts: String,
    pub initialization: String,
    pub input_data: String,
    pub submodels: String,
}

/// The full behaviour-graph / ODD export.
#[derive(Debug, Clone, Serialize)]
pub struct BehaviorGraph {
    /// Scenario name (fixed for this replication).
    pub scenario: String,
    /// Whether this is the classical (no-LLM) or LLM-driven variant.
    pub variant: String,
    /// How the artefact was produced (honesty marker — this is a structured
    /// description of a fixed model, NOT an LLM-synthesised construction).
    pub provenance: String,
    /// The ODD protocol sections.
    pub odd: OddProtocol,
    /// Behaviour-graph nodes.
    pub nodes: Vec<GraphNode>,
    /// Behaviour-graph edges.
    pub edges: Vec<GraphEdge>,
}

fn node(id: &str, kind: &str, label: &str) -> GraphNode {
    GraphNode {
        id: id.to_string(),
        kind: kind.to_string(),
        label: label.to_string(),
    }
}

fn edge(from: &str, to: &str, relation: &str) -> GraphEdge {
    GraphEdge {
        from: from.to_string(),
        to: to.to_string(),
        relation: relation.to_string(),
    }
}

/// Build the behaviour graph + ODD protocol from the fixed model [`Config`].
///
/// The variant (classical vs LLM) is read from `cfg.provider`; the interaction
/// node and its edges differ between the two (the LLM variant adds persona /
/// memory nodes and an `llm_decision` event), mirroring exactly which mechanism
/// the driver wires.
pub fn build_behavior_graph(cfg: &Config) -> BehaviorGraph {
    let is_llm = cfg.provider.is_llm();
    let variant = if is_llm {
        "llm-driven (YuLan-OneSim variant)"
    } else {
        "classical (deterministic Axelrod baseline)"
    };

    // --- ODD protocol sections (derived from the fixed model) --- //
    let odd = OddProtocol {
        purpose: "Reproduce Axelrod (1997) local convergence with global polarization on a \
                  grid of cultural agents, comparing a deterministic baseline against an \
                  LLM-driven culture-adoption variant (YuLan-OneSim Appendix F)."
            .to_string(),
        entities_state_variables_scales: format!(
            "Entities: grid sites (agents). State variables: a culture vector of F={} features, \
             each a trait in 0..q (q={}). Spatial scale: a {}x{} Von Neumann lattice (fixed \
             boundary). Temporal scale: engine ticks of {} micro-events each.",
            cfg.features,
            cfg.traits,
            cfg.width,
            cfg.height,
            cfg.effective_events_per_step(),
        ),
        process_overview_scheduling: "Per micro-event: draw a site and a random neighbour; \
            (classical) with probability = feature-similarity, copy one differing feature; \
            (LLM) the LLM decides whether/which differing feature to adopt. PostStep: compute \
            LC/GP and check the absorbing state."
            .to_string(),
        design_concepts: "Emergence: macro polarization from local homophilic copying. \
            Interaction: pairwise, neighbour-local. Stochasticity: site/neighbour draws \
            (deterministic given the seed). Observation: LC, GP, stable-region count."
            .to_string(),
        initialization: format!(
            "Each site is assigned F={} independent uniform traits in 0..q (q={}); the LLM \
             variant additionally assigns a round-robin persona per site.",
            cfg.features, cfg.traits
        ),
        input_data: "None (no external time series); all state is generated from the seed."
            .to_string(),
        submodels: if is_llm {
            "Similarity = matching features / F. Adoption = LLM JSON decision over the differing \
             feature indices, conditioned on persona + own culture + neighbour culture + memory. \
             Convergence = every adjacent pair has similarity in {0,1}."
                .to_string()
        } else {
            "Similarity = matching features / F. Adoption = with probability `similarity`, copy \
             one uniformly-chosen differing feature from the neighbour. Convergence = every \
             adjacent pair has similarity in {0,1}."
                .to_string()
        },
    };

    // --- behaviour-graph nodes --- //
    let mut nodes = vec![
        node("agent_site", "agent", "Grid site (cultural agent)"),
        node(
            "state_culture",
            "state",
            "Culture vector (F features × q traits)",
        ),
        node("event_interaction", "event", "Interaction micro-event"),
        node("event_convergence", "event", "Convergence check (PostStep)"),
        node("metric_lc", "metric", "Local convergence (LC)"),
        node("metric_gp", "metric", "Global polarization (GP)"),
        node("metric_regions", "metric", "Stable region count"),
    ];

    let mut edges = vec![
        edge("agent_site", "state_culture", "owns"),
        edge("agent_site", "event_interaction", "triggers"),
        edge("event_interaction", "state_culture", "writes"),
        edge("event_convergence", "metric_lc", "produces"),
        edge("event_convergence", "metric_gp", "produces"),
        edge("event_convergence", "metric_regions", "produces"),
        edge("state_culture", "event_convergence", "reads"),
    ];

    if is_llm {
        nodes.push(node("state_persona", "state", "Persona (NL attribute)"));
        nodes.push(node("state_memory", "state", "Short-term memory"));
        nodes.push(node("event_llm_decision", "event", "LLM adoption decision"));
        edges.push(edge("agent_site", "state_persona", "owns"));
        edges.push(edge("agent_site", "state_memory", "owns"));
        edges.push(edge("event_interaction", "event_llm_decision", "triggers"));
        edges.push(edge("state_persona", "event_llm_decision", "reads"));
        edges.push(edge("state_memory", "event_llm_decision", "reads"));
        edges.push(edge("event_llm_decision", "state_culture", "writes"));
        edges.push(edge("event_llm_decision", "state_memory", "writes"));
    }

    BehaviorGraph {
        scenario: "Axelrod culture dissemination (YuLan-OneSim Appendix F)".to_string(),
        variant: variant.to_string(),
        provenance: "Structured description derived deterministically from the fixed model \
                     (Config + wired mechanisms). This is a concept demo of YuLan-OneSim's \
                     ODD/behaviour-graph construction, NOT an LLM-synthesised artefact."
            .to_string(),
        odd,
        nodes,
        edges,
    }
}

/// Write the behaviour graph to `behavior_graph.json` under `output_dir`.
///
/// `output_dir` is the run's `artifacts/`: the export is a structured document
/// derived from the config, not a measurement, so it is a table-shaped artefact
/// rather than a metric.
pub fn save_behavior_graph(graph: &BehaviorGraph, output_dir: &str) {
    let path = std::path::PathBuf::from(format!("{output_dir}/behavior_graph.json"));
    crate::simulation::write_json(graph, &path);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, Provider};

    #[test]
    fn classical_graph_has_no_llm_nodes() {
        let cfg = Config {
            provider: Provider::None,
            ..Config::default()
        };
        let g = build_behavior_graph(&cfg);
        assert!(g.variant.contains("classical"));
        assert!(!g.nodes.iter().any(|n| n.id == "event_llm_decision"));
        assert!(!g.nodes.iter().any(|n| n.id == "state_persona"));
    }

    #[test]
    fn llm_graph_adds_persona_memory_decision() {
        let cfg = Config {
            provider: Provider::Ollama,
            ..Config::default()
        };
        let g = build_behavior_graph(&cfg);
        assert!(g.variant.contains("llm"));
        assert!(g.nodes.iter().any(|n| n.id == "event_llm_decision"));
        assert!(g.nodes.iter().any(|n| n.id == "state_persona"));
        assert!(g.nodes.iter().any(|n| n.id == "state_memory"));
        // Every edge references declared nodes.
        let ids: std::collections::HashSet<&str> = g.nodes.iter().map(|n| n.id.as_str()).collect();
        for e in &g.edges {
            assert!(ids.contains(e.from.as_str()), "unknown from: {}", e.from);
            assert!(ids.contains(e.to.as_str()), "unknown to: {}", e.to);
        }
    }
}
