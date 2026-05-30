//! YuLan-OneSim (Wang et al. 2025) — Axelrod culture-dissemination scenario.
//!
//! This crate reproduces **one concrete scenario** from the YuLan-OneSim paper
//! (Appendix F): the Axelrod (1997) culture-dissemination model, implemented on
//! the socsim framework as a direct extension of the `axelrod1997` sibling. It
//! offers a **two-layer comparison** on the same `WorldState`:
//!
//! - a **classical, deterministic** similarity-probability interaction rule (the
//!   Axelrod baseline; `--provider none`, no LLM), and
//! - an **LLM-driven** culture-adoption rule (the YuLan-OneSim variant;
//!   `--provider ollama|openai`).
//!
//! The two interaction mechanisms are mutually exclusive: the driver wires
//! exactly one based on `config.provider`. A `ConvergenceMechanism` (PostStep)
//! detects the absorbing state and records the per-round LC / GP history.
//!
//! Modules: world state (`world`), interaction + convergence mechanisms
//! (`mechanisms`), LLM client layer (`llm`), run/sweep driver (`simulation`),
//! aggregate metrics (`metrics`), and configuration (`config`).

pub mod config;
pub mod llm;
pub mod mechanisms;
pub mod metrics;
pub mod odd;
pub mod simulation;
pub mod world;
