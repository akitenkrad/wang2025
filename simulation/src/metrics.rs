//! Aggregate metrics for the culture-dissemination model.
//!
//! - `n_stable_regions` / `max_region_size` / `n_distinct_cultures`: Axelrod's
//!   structural metrics (4-connected BFS over identical cultures), mirroring
//!   `axelrod1997`.
//! - `local_convergence` (LC): the mean over adjacent pairs of the fraction of
//!   matching features. Appendix F / Figure 9 target: exceeds 0.50 by round ~60.
//! - `global_polarization` (GP): the documented field `|C| / N²` (`|C|` = number
//!   of distinct culture regions, `N` = total agents) — see the GP inconsistency
//!   note below.

use std::collections::{HashSet, VecDeque};

use crate::world::CultureWorld;

/// Aggregate metrics for a single board state.
#[derive(Debug, Clone)]
pub struct RunMetrics {
    /// Number of stable culture regions (4-connected components of identical
    /// culture vectors). Axelrod's headline metric.
    pub n_stable_regions: usize,
    /// Size (sites) of the largest region.
    pub max_region_size: usize,
    /// Number of distinct culture profiles on the board.
    pub n_distinct_cultures: usize,
    /// Local convergence LC — mean adjacent-pair similarity (per round).
    pub local_convergence: f64,
    /// Global polarization GP as documented = `|C| / N²` (see inconsistency note).
    pub global_polarization: f64,
    /// Auxiliary `|C| / N` (regions-per-agent) — recorded alongside GP because
    /// the documented `|C| / N²` cannot reach the Appendix F 0.35–0.40 target
    /// for any reasonable `N` (see [`global_polarization`]).
    pub gp_per_agent: f64,
}

/// Similarity between two culture vectors (matching features / feature count).
/// Assumes `a.len() == b.len()`.
#[inline]
pub fn similarity(a: &[usize], b: &[usize]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let shared = a.iter().zip(b.iter()).filter(|(x, y)| x == y).count();
    shared as f64 / a.len() as f64
}

/// Count stable culture regions: 4-connected components of sites sharing the
/// **exact same** culture vector, via BFS over the precomputed Von Neumann
/// neighbour table. Returns region count, largest-region size, and distinct
/// culture count. (Mirrors `axelrod1997`.)
pub fn count_stable_regions(world: &CultureWorld) -> (usize, usize, usize) {
    let n = world.n_sites();
    let mut visited = vec![false; n];
    let mut n_regions = 0usize;
    let mut max_region_size = 0usize;

    for start in 0..n {
        if visited[start] {
            continue;
        }
        visited[start] = true;
        n_regions += 1;

        let mut size = 1usize;
        let mut queue: VecDeque<usize> = VecDeque::new();
        queue.push_back(start);

        while let Some(cur) = queue.pop_front() {
            for &nb in world.adjacency.neighbors(cur) {
                if !visited[nb] && world.culture(cur) == world.culture(nb) {
                    visited[nb] = true;
                    size += 1;
                    queue.push_back(nb);
                }
            }
        }

        if size > max_region_size {
            max_region_size = size;
        }
    }

    let mut set: HashSet<&Vec<usize>> = HashSet::new();
    for site in world.cells.cells() {
        set.insert(site);
    }

    (n_regions, max_region_size, set.len())
}

/// Local convergence LC: the mean over **adjacent (undirected) pairs** of the
/// fraction of matching features, `LC = (1/|E|) Σ_{(i,j)∈E} sim(i,j)`.
///
/// The CSR neighbour table lists each undirected edge twice (i→j and j→i); since
/// `sim` is symmetric, averaging over all directed neighbour entries equals the
/// undirected-pair average.
pub fn local_convergence(world: &CultureWorld) -> f64 {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for i in 0..world.n_sites() {
        for &j in world.adjacency.neighbors(i) {
            sum += similarity(world.culture(i), world.culture(j));
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

/// Global polarization GP.
///
/// **Documented formula (literal):** `GP = |C| / N²`, where `|C|` is the number
/// of distinct culture *regions* (connected clusters sharing the same culture)
/// and `N` is the total number of agents.
///
/// ## Known inconsistency (do NOT silently "fix")
/// The design doc (§4.3 / §4.8) flags an inconsistency: Appendix F reports
/// `GP ≈ 0.35–0.40` at steady state, but `|C| / N²` is bounded by `1/N` (since
/// `|C| ≤ N`), which for any reasonable `N` (e.g. `N = 100` → max `0.01`) cannot
/// reach `0.35–0.40`. The paper's target therefore almost certainly uses a
/// **different normalisation** (most plausibly `|C| / N`, i.e. regions per
/// agent, which *can* sit in that band when many small regions persist). We
/// implement `|C| / N²` literally as the documented field, but also record
/// `|C| / N` as the auxiliary [`RunMetrics::gp_per_agent`] column so the
/// reproduction can compare against the band without altering the documented
/// formula. The discrepancy is also noted in `docs/architecture.md`.
pub fn global_polarization(n_regions: usize, n_agents: usize) -> (f64, f64) {
    if n_agents == 0 {
        return (0.0, 0.0);
    }
    let n = n_agents as f64;
    let c = n_regions as f64;
    let gp = c / (n * n); // documented |C| / N²
    let gp_per_agent = c / n; // auxiliary |C| / N
    (gp, gp_per_agent)
}

/// Global stability check: every adjacent pair has `sim ∈ {0, 1}` (absorbing
/// state). Returns `false` as soon as a pair with `0 < sim < 1` is found.
pub fn is_stable(world: &CultureWorld) -> bool {
    for i in 0..world.n_sites() {
        for &j in world.adjacency.neighbors(i) {
            let sim = similarity(world.culture(i), world.culture(j));
            if sim > 0.0 && sim < 1.0 {
                return false;
            }
        }
    }
    true
}

/// Compute the full [`RunMetrics`] for the current board.
pub fn compute_metrics(world: &CultureWorld) -> RunMetrics {
    let (n_regions, max_region_size, n_distinct) = count_stable_regions(world);
    let lc = local_convergence(world);
    let (gp, gp_per_agent) = global_polarization(n_regions, world.n_sites());
    RunMetrics {
        n_stable_regions: n_regions,
        max_region_size,
        n_distinct_cultures: n_distinct,
        local_convergence: lc,
        global_polarization: gp,
        gp_per_agent,
    }
}
