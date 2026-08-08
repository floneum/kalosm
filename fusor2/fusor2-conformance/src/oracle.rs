//! The debug ILP extraction oracle. It implements the same [`Extractor`]
//! trait as the shipped local search and must agree with it on small graphs.
//! It is exponential by construction and refuses anything past its caps with
//! [`Error::Budget`] rather than running forever.
//!
//! Everything below the search is shared with the shipped extractor: the same
//! `realize`, `CostModel::total` on the same realized DAG, `lower_bound` for
//! pruning, and `derive_plan`. Where `LocalSearch` walks a move frontier, this
//! enumerates.

use std::sync::Arc;

use fixedbitset::FixedBitSet;
use rustc_hash::FxHashMap;

use fusor2_cost::{LocalSearch, lower_bound, moves, plan, realize, verify_plan};
use fusor2_ir::cost::{CostModel, Picoseconds};
use fusor2_ir::device::Caps;
use fusor2_ir::egraph::{ClassId, EGraph, Id};
use fusor2_ir::extract::{ExtractBudget, Extraction, Extractor, Plan};
use fusor2_ir::ir::level1::{SchedPoint, ScheduleDomain};
use fusor2_ir::ir::level2::ArenaPlanner;
use fusor2_ir::{Error, Result};

/// Exhaustive branch-and-bound over `(sigma, m, theta)`, pruned by the
/// admissible lower bound.
pub struct IlpExtractor {
    arena: Arc<dyn ArenaPlanner>,
}

impl IlpExtractor {
    /// Nodes past which the oracle refuses outright.
    pub const MAX_NODES: usize = 64;
    /// Union chains past which the oracle refuses. The smallest interesting
    /// chain has two members, and `2^18` already exceeds the product cap.
    pub const MAX_CHAINS: usize = 18;
    /// Product of `|members|` over all chains.
    pub const MAX_SIGMA_POINTS: usize = 4_096;
    /// Flippable nodes past which the `m` enumeration refuses; `2^12` is the
    /// same 4,096 ceiling.
    pub const MAX_FLIP_BITS: usize = 12;
    /// Product of schedule-domain sizes over the selected nodes.
    pub const MAX_THETA_POINTS: usize = 4_096;
    /// Total `(sigma, m, theta)` points costed before the oracle gives up.
    pub const MAX_EVALUATIONS: usize = 1_000_000;

    pub fn new() -> Self {
        Self {
            arena: fusor2_tile::Planner::shared(),
        }
    }

    pub fn with_arena(arena: Arc<dyn ArenaPlanner>) -> Self {
        Self { arena }
    }

    pub fn arena(&self) -> &Arc<dyn ArenaPlanner> {
        &self.arena
    }
}

impl Default for IlpExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for IlpExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("IlpExtractor")
    }
}

/// One class the oracle branches over, with its members in ascending id order
/// so the enumeration is reproducible.
struct Chain {
    class: ClassId,
    members: Vec<Id>,
}

/// Everything one search needs, gathered once.
struct Search<'a> {
    graph: &'a EGraph,
    roots: &'a [Id],
    cost: &'a dyn CostModel,
    arena: &'a dyn ArenaPlanner,
    lb: Vec<Picoseconds>,
    chains: Vec<Chain>,
    evaluations: usize,
    best: Option<(Picoseconds, Extraction)>,
}

impl IlpExtractor {
    fn search(
        &self,
        graph: &EGraph,
        roots: &[Id],
        cost: &dyn CostModel,
    ) -> Result<(Extraction, Picoseconds)> {
        if graph.len() > Self::MAX_NODES {
            return Err(Error::Budget(format!(
                "the ILP oracle refuses a {}-node graph (MAX_NODES = {})",
                graph.len(),
                Self::MAX_NODES
            )));
        }
        if roots.is_empty() {
            return Err(Error::Plan("the ILP oracle needs at least one root".into()));
        }

        let mut chains = Vec::new();
        let mut singletons: FxHashMap<ClassId, Id> = FxHashMap::default();
        let mut product: usize = 1;
        for class in realize::classes(graph) {
            let mut members = graph.members(class);
            members.sort_unstable();
            match members.len() {
                0 => return Err(Error::Plan(format!("class {} has no members", class.0))),
                1 => {
                    singletons.insert(class, members[0]);
                }
                n => {
                    product = product.saturating_mul(n);
                    chains.push(Chain {
                        class,
                        members: members.to_vec(),
                    });
                }
            }
        }
        if chains.len() > Self::MAX_CHAINS {
            return Err(Error::Budget(format!(
                "the ILP oracle refuses {} union chains (MAX_CHAINS = {})",
                chains.len(),
                Self::MAX_CHAINS
            )));
        }
        if product > Self::MAX_SIGMA_POINTS {
            return Err(Error::Budget(format!(
                "the ILP oracle refuses a {product}-point sigma product \
                 (MAX_SIGMA_POINTS = {})",
                Self::MAX_SIGMA_POINTS
            )));
        }
        // Deterministic branching order.
        chains.sort_by_key(|c| c.class);

        let lb = lower_bound::lower_bound(graph, cost);
        let mut search = Search {
            graph,
            roots,
            cost,
            arena: self.arena.as_ref(),
            lb,
            chains,
            evaluations: 0,
            best: None,
        };

        // Seed the incumbent from the shipped search so branch-and-bound has a
        // real bound from the first node. It is a bound, not an answer: any
        // point the enumeration finds that is strictly cheaper replaces it.
        if let Ok(seed) = LocalSearch::new(self.arena.clone(), cost.facts().caps.clone())
            .seed(graph, roots, &search.lb, cost)
            && let Ok(realized) = realize::realize(graph, roots, &seed, cost, search.arena)
        {
            let seed_cost = realize::exact_cost(&realized, &seed, cost);
            search.best = Some((seed_cost, seed));
        }

        let mut sigma = singletons;
        search.branch(0, &mut sigma, Picoseconds(0))?;

        search
            .best
            .take()
            .map(|(c, ex)| (ex, c))
            .ok_or_else(|| Error::Plan("the ILP oracle found no feasible extraction".into()))
    }
}

impl Search<'_> {
    /// A valid lower bound on any completion of a partial assignment.
    ///
    /// `lower_bound` sums each distinct child class once, so the root bound
    /// already contains `lb[class]` for every reachable class. Pinning a class
    /// to a member raises the total by at least `lb[member] - lb[class]`, so
    /// adding one copy of that non-negative delta still underestimates and
    /// pruning on it cannot discard the optimum.
    fn root_bound(&self, penalty: Picoseconds) -> Picoseconds {
        let base: u64 = self
            .roots
            .iter()
            .map(|r| self.lb[self.graph.class_of(*r).0.index()].0)
            .max()
            .unwrap_or(0);
        Picoseconds(base.saturating_add(penalty.0))
    }

    fn branch(
        &mut self,
        depth: usize,
        sigma: &mut FxHashMap<ClassId, Id>,
        penalty: Picoseconds,
    ) -> Result<()> {
        if let Some((incumbent, _)) = &self.best
            && self.root_bound(penalty) > *incumbent
        {
            // Every completion of this prefix is already worse.
            return Ok(());
        }
        if depth == self.chains.len() {
            return self.evaluate_sigma(sigma);
        }

        let class = self.chains[depth].class;
        let members = self.chains[depth].members.clone();
        let class_lb = self.lb[class.0.index()];
        for member in members {
            let delta = self.lb[member.index()].0.saturating_sub(class_lb.0);
            sigma.insert(class, member);
            self.branch(
                depth + 1,
                sigma,
                Picoseconds(penalty.0.saturating_add(delta)),
            )?;
        }
        sigma.remove(&class);
        Ok(())
    }

    /// Enumerate every `m` and every `theta` under one complete `sigma`.
    fn evaluate_sigma(&mut self, sigma: &FxHashMap<ClassId, Id>) -> Result<()> {
        let mut probe = Extraction {
            sigma: sigma.clone(),
            m: FixedBitSet::with_capacity(self.graph.len()),
            theta: FxHashMap::default(),
        };
        // Forced materializations: the roots, plus every node `Flip` refuses.
        // An `Effect::InPlace` node inlined into two consumers would apply its
        // atomics twice, so pinning is a precondition, not an enumeration
        // variable.
        for r in self.roots {
            let Ok(selected) = realize::select(self.graph, &probe, *r) else {
                return Ok(());
            };
            probe.m.insert(selected.index());
        }
        for i in 0..self.graph.len() {
            if moves::is_pinned(self.graph, self.roots, Id(i as u32)) {
                probe.m.insert(i);
            }
        }

        // One realization at the forced `m` gives the node set every point
        // under this sigma shares. An infeasible sigma (a selection cycle, an
        // unselected class) is skipped rather than failing the search: the
        // enumeration ranges over a superset of the feasible set.
        let Ok(shape) = realize::realize(self.graph, self.roots, &probe, self.cost, self.arena)
        else {
            return Ok(());
        };

        let flippable: Vec<Id> = shape
            .order
            .iter()
            .copied()
            .filter(|id| {
                realize::leaf_role(self.graph, *id) == realize::LeafRole::NotLeaf
                    && !probe.m.contains(id.index())
            })
            .collect();
        if flippable.len() > IlpExtractor::MAX_FLIP_BITS {
            return Err(Error::Budget(format!(
                "the ILP oracle refuses {} flippable nodes (MAX_FLIP_BITS = {})",
                flippable.len(),
                IlpExtractor::MAX_FLIP_BITS
            )));
        }

        let theta_axes = self.theta_axes(&shape)?;
        let forced_m = probe.m.clone();

        for bits in 0u32..(1u32 << flippable.len()) {
            let mut m = forced_m.clone();
            for (bit, id) in flippable.iter().enumerate() {
                if bits & (1 << bit) != 0 {
                    m.insert(id.index());
                }
            }
            self.evaluate_theta(sigma, &m, &theta_axes)?;
        }
        Ok(())
    }

    /// The `(node, domain points)` axes theta ranges over under one sigma.
    fn theta_axes(&self, shape: &realize::Realized) -> Result<Vec<(Id, Vec<SchedPoint>)>> {
        let mut axes: Vec<(Id, Vec<SchedPoint>)> = Vec::new();
        let mut product: usize = 1;
        for id in &shape.order {
            let Some(domain) = realize::domain_of(self.graph, *id) else {
                continue;
            };
            if matches!(domain, ScheduleDomain::Point) {
                continue;
            }
            if domain.is_empty() {
                return Err(Error::Legality(format!(
                    "{id} carries an empty schedule domain and is unselectable"
                )));
            }
            let points: Vec<SchedPoint> = domain.iter().collect();
            product = product.saturating_mul(points.len());
            axes.push((*id, points));
        }
        if product > IlpExtractor::MAX_THETA_POINTS {
            return Err(Error::Budget(format!(
                "the ILP oracle refuses a {product}-point theta product \
                 (MAX_THETA_POINTS = {})",
                IlpExtractor::MAX_THETA_POINTS
            )));
        }
        Ok(axes)
    }

    fn evaluate_theta(
        &mut self,
        sigma: &FxHashMap<ClassId, Id>,
        m: &FixedBitSet,
        axes: &[(Id, Vec<SchedPoint>)],
    ) -> Result<()> {
        let total: usize = axes.iter().map(|(_, p)| p.len()).product::<usize>().max(1);
        for point in 0..total {
            let mut theta = FxHashMap::default();
            let mut rest = point;
            for (id, points) in axes {
                theta.insert(*id, points[rest % points.len()]);
                rest /= points.len();
            }
            self.consider(Extraction {
                sigma: sigma.clone(),
                m: m.clone(),
                theta,
            })?;
        }
        Ok(())
    }

    /// Price one point on the realized DAG and keep it if it strictly wins.
    /// Ties keep the incumbent, so the answer is deterministic in the fixed
    /// enumeration order.
    fn consider(&mut self, extraction: Extraction) -> Result<()> {
        self.evaluations += 1;
        if self.evaluations > IlpExtractor::MAX_EVALUATIONS {
            return Err(Error::Budget(format!(
                "the ILP oracle exceeded {} evaluations",
                IlpExtractor::MAX_EVALUATIONS
            )));
        }
        let Ok(realized) =
            realize::realize(self.graph, self.roots, &extraction, self.cost, self.arena)
        else {
            return Ok(());
        };
        let cost = realize::exact_cost(&realized, &extraction, self.cost);
        let better = match &self.best {
            None => true,
            Some((incumbent, _)) => cost < *incumbent,
        };
        if better {
            self.best = Some((cost, extraction));
        }
        Ok(())
    }
}

impl Extractor for IlpExtractor {
    /// The same admissible bound the shipped search uses.
    fn lower_bound(&self, graph: &EGraph, cost: &dyn CostModel) -> Vec<Picoseconds> {
        lower_bound::lower_bound(graph, cost)
    }

    fn extract(
        &self,
        graph: &EGraph,
        roots: &[Id],
        cost: &dyn CostModel,
        budget: ExtractBudget,
    ) -> Result<Plan> {
        // The oracle is exhaustive; its own caps are the `MAX_*` constants.
        let _ = budget;
        let (extraction, cost_ps) = self.search(graph, roots, cost)?;
        let realized = realize::realize(graph, roots, &extraction, cost, self.arena.as_ref())?;
        plan::derive_plan(graph, &extraction, &realized, cost.facts(), cost_ps)
    }

    fn verify_plan(&self, graph: &EGraph, plan: &Plan) -> Result<()> {
        verify_plan::verify_plan(graph, plan)
    }
}

/// The shipped local search must reach the same plan and the same cost as the
/// oracle on a graph small enough to enumerate. `PlanHash` equality is the
/// strong half: two extractions with equal cost but different
/// `(sigma, m, theta)` are not the same decision, and the plan is the cache
/// key.
pub fn assert_oracle_agrees(
    graph: &EGraph,
    roots: &[Id],
    cost: &dyn CostModel,
    arena: Arc<dyn ArenaPlanner>,
    caps: Caps,
) -> std::result::Result<(), String> {
    let oracle = IlpExtractor::with_arena(arena.clone());
    let shipped = LocalSearch::new(arena, caps);

    let optimal = oracle
        .extract(graph, roots, cost, ExtractBudget::default())
        .map_err(|e| format!("oracle: {e}"))?;
    let found = shipped
        .extract(graph, roots, cost, ExtractBudget::default())
        .map_err(|e| format!("local search: {e}"))?;

    if found.cost != optimal.cost {
        return Err(format!(
            "local search cost {} ps, oracle optimum {} ps ({} launches vs {})",
            found.cost.0,
            optimal.cost.0,
            found.launches.len(),
            optimal.launches.len()
        ));
    }
    if found.hash != optimal.hash {
        return Err(format!(
            "local search and oracle tie on cost ({} ps) but chose different plans: \
             0x{:032x} vs 0x{:032x}. Equal cost is not the same decision — the plan \
             is the cache key.",
            found.cost.0, found.hash.0, optimal.hash.0
        ));
    }
    Ok(())
}

/// The twelve small graphs `assert_oracle_agrees` must hold on, by name. Each
/// is a shape where a greedy search can go wrong: rematerialization, the
/// epilogue-versus-merge conflict, split-K folding, the four scatter-add
/// lowerings, alias versus gather, and the quantized repack.
pub const ORACLE_GRAPHS: [&str; 12] = [
    "elementwise_chain",
    "two_consumer_producer_rematerializes",
    "matmul_plus_epilogue",
    "matmul_epilogue_vs_merged_wave",
    "split_k_fold",
    "fold_split_declined_without_reassoc",
    "scatter_add_four_way",
    "scatter_add_atomic_unavailable",
    "view_alias_vs_gather",
    "view_spine_sunk_into_contract",
    "quantized_repack_amortized",
    "quantized_repack_step_local",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_caps_are_ordered_so_the_product_ceilings_bind() {
        // 2^18 > 4096, so 18 two-member chains hit the product cap rather
        // than being silently enumerated.
        assert!(1usize << IlpExtractor::MAX_CHAINS > IlpExtractor::MAX_SIGMA_POINTS);
        assert_eq!(1usize << IlpExtractor::MAX_FLIP_BITS, 4_096);
        assert_eq!(IlpExtractor::MAX_SIGMA_POINTS, 4_096);
        assert_eq!(IlpExtractor::MAX_THETA_POINTS, 4_096);
        assert_eq!(IlpExtractor::MAX_NODES, 64);
    }

    #[test]
    fn the_twelve_named_graphs_are_distinct() {
        let mut names = ORACLE_GRAPHS.to_vec();
        names.sort_unstable();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "a graph name is repeated");
        assert_eq!(ORACLE_GRAPHS.len(), 12);
    }

    #[test]
    fn the_oracle_is_an_object_safe_extractor() {
        let oracle: Box<dyn Extractor> = Box::new(IlpExtractor::default());
        assert!(format!("{:?}", IlpExtractor::new()).contains("Ilp"));
        drop(oracle);
    }
}
