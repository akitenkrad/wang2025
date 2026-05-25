//! World state for the Axelrod culture-dissemination model on socsim.
//!
//! `CultureWorld` implements socsim's [`WorldState`]. Each site holds a culture
//! vector (`Vec<usize>`); the board is fully occupied with no agent movement —
//! only the culture vectors mutate. We therefore use the **`CellGrid<Culture>` +
//! precomputed `Adjacency`** pattern (same as `axelrod1997`), not the
//! occupancy-tracking [`socsim_grid::GridIndex`].
//!
//! Cell state lives in the socsim-grid [`CellGrid<Culture>`] (cell value =
//! culture vector, flat index `idx = r*cols + c`), the single source of truth.
//! Von Neumann neighbours are precomputed once into a CSR [`Adjacency`] held on
//! the world (`adj.neighbors(idx)` gives an O(1) borrow). The LLM variant adds a
//! per-site **persona** and short-term **memory**, plus per-round LC / GP
//! history buffers used by the convergence mechanism.

use std::collections::BTreeMap;

use rand::Rng;

use socsim_core::{AgentId, CultureVectors, Neighbors, SimClock, SimRng, WorldState};
use socsim_grid::{Adjacency, Boundary, CellGrid, Grid, Neighborhood};

/// A single site's culture vector. `culture[i]` is the trait value (0..q) of
/// feature `i`.
pub type Culture = Vec<usize>;

/// World state for the Axelrod culture-dissemination model.
#[derive(Clone)]
pub struct CultureWorld {
    /// Simulation clock.
    pub clock: SimClock,
    /// Cell grid holding each site's culture vector (cell value = culture vector,
    /// flat index `idx = r*cols + c`). The single source of truth for state.
    pub cells: CellGrid<Culture>,
    /// CSR-precomputed Von Neumann neighbour table (flat idx → neighbour flat
    /// idx), built from `cells.grid()` for hot-loop speed.
    pub adjacency: Adjacency,
    /// Number of features `F`.
    pub n_features: usize,
    /// Number of traits `q`.
    pub n_traits: usize,
    /// Grid width (columns).
    pub width: usize,
    /// Grid height (rows).
    pub height: usize,
    // --- LLM layer (empty for the classical variant) --- //
    /// Per-site persona description (natural-language attribute injected into the
    /// LLM prompt).
    pub personas: BTreeMap<AgentId, String>,
    /// Per-site short-term memory (recent adoption decisions, etc.).
    pub memory: BTreeMap<AgentId, Vec<String>>,
    /// Per-round local-convergence (LC) history (for the Appendix F time series).
    pub lc_history: Vec<f64>,
    /// Per-round global-polarization (GP) history.
    pub gp_history: Vec<f64>,
}

impl CultureWorld {
    /// Build a world from grid dimensions and a vector of culture vectors
    /// (precomputes the neighbour table).
    pub fn new(
        width: usize,
        height: usize,
        n_features: usize,
        n_traits: usize,
        cultures: Vec<Culture>,
        t_max: u64,
    ) -> Self {
        debug_assert_eq!(cultures.len(), width * height);
        let grid = Grid::new(height, width, Boundary::Fixed);
        // Precompute Von Neumann neighbours into CSR (flat idx → neighbour flat idx).
        let adjacency = grid.adjacency(Neighborhood::VonNeumann);
        // Store the culture vectors in the CellGrid (consumed row-major, order preserved).
        let mut iter = cultures.into_iter();
        let cells = CellGrid::from_fn(grid, |_, _| {
            iter.next().expect("not enough cultures for width*height")
        });
        CultureWorld {
            clock: SimClock::new(t_max),
            cells,
            adjacency,
            n_features,
            n_traits,
            width,
            height,
            personas: BTreeMap::new(),
            memory: BTreeMap::new(),
            lc_history: Vec::new(),
            gp_history: Vec::new(),
        }
    }

    /// Random initialisation: assign each site `F` trait values drawn
    /// independently from `0..q`.
    pub fn random_init(
        width: usize,
        height: usize,
        n_features: usize,
        n_traits: usize,
        rng: &mut SimRng,
        t_max: u64,
    ) -> Self {
        let n = width * height;
        let cultures: Vec<Culture> = (0..n)
            .map(|_| {
                (0..n_features)
                    .map(|_| rng.gen_range(0..n_traits))
                    .collect()
            })
            .collect();
        Self::new(width, height, n_features, n_traits, cultures, t_max)
    }

    /// Number of sites (= width * height).
    #[inline]
    pub fn n_sites(&self) -> usize {
        self.width * self.height
    }

    /// Borrow the culture vector at flat index `idx`.
    #[inline]
    pub fn culture(&self, idx: usize) -> &Culture {
        self.cells.get_idx(idx).expect("idx out of range (culture)")
    }

    /// Grid width (columns).
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Grid height (rows).
    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    /// Attach a persona to every site (round-robin from the supplied templates).
    /// Also initialises an empty memory buffer per site. Used by the LLM variant.
    pub fn assign_personas(&mut self, personas: &[&str]) {
        if personas.is_empty() {
            return;
        }
        for i in 0..self.n_sites() {
            let id = AgentId(i as u64);
            self.personas
                .insert(id, personas[i % personas.len()].to_string());
            self.memory.entry(id).or_default();
        }
    }
}

impl WorldState for CultureWorld {
    fn agent_ids(&self) -> Vec<AgentId> {
        (0..self.n_sites()).map(|i| AgentId(i as u64)).collect()
    }

    fn clock(&self) -> &SimClock {
        &self.clock
    }

    fn clock_mut(&mut self) -> &mut SimClock {
        &mut self.clock
    }
}

// --------------------------------------------------------------------------- //
// socsim-core capability traits → enables the reusable AxelrodMechanism
// (socsim-mechanisms) to drive the classical path. The mappings below are
// exactly what the repo-local `classical_event` used: site = flat idx, feature
// access goes through the same `CellGrid<Culture>`, neighbours come from the
// same precomputed Von Neumann `Adjacency` — so the RNG draw order (site →
// neighbour → interact-prob → feature) and resulting state stay byte-identical.
// --------------------------------------------------------------------------- //

impl Neighbors for CultureWorld {
    /// Von Neumann neighbours of `id`, taken from the same precomputed CSR
    /// [`Adjacency`] the classical mechanism uses (`adjacency.neighbors(idx)`),
    /// in the same order — guaranteeing identical neighbour draws.
    fn neighbors_of(&self, id: AgentId) -> Vec<AgentId> {
        self.adjacency
            .neighbors(id.0 as usize)
            .iter()
            .map(|&nb| AgentId(nb as u64))
            .collect()
    }
}

impl CultureVectors for CultureWorld {
    fn n_features(&self) -> usize {
        self.n_features
    }

    /// Read feature `f` of site `id`. Repo culture values are `usize` in `0..q`
    /// (`q` small); the cast to `u32` is lossless.
    fn feature(&self, id: AgentId, f: usize) -> u32 {
        self.culture(id.0 as usize)[f] as u32
    }

    /// Overwrite feature `f` of site `id`. The `u32` value originates from a
    /// sibling site's `usize` trait (always `< q`), so the cast back is lossless.
    fn set_feature(&mut self, id: AgentId, f: usize, value: u32) {
        let idx = id.0 as usize;
        self.cells
            .get_idx_mut(idx)
            .expect("idx out of range (set_feature)")[f] = value as usize;
    }
}
