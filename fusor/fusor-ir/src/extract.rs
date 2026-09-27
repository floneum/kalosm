//! Node selection, materialization and schedule points, decided together
//! against the exact global cost.

use crate::cost::{CostModel, Picoseconds};
use crate::egraph::{ClassId, EGraph, Id};
use crate::error::Result;
use crate::ir::launch::SchedPoint;
use crate::shape::Dim;
use fixedbitset::FixedBitSet;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// The complete extraction state. Cost is not defined per e-class; it is
/// evaluated on the realized DAG under `(sigma, m, theta)`.
#[derive(Clone, Debug, Default)]
pub struct Extraction {
    /// E-class -> the selected member of that class.
    pub sigma: FxHashMap<ClassId, Id>,
    /// Buffer outputs derived from the selected DAG. A composite's final
    /// stage aliases its owner instead of allocating another output.
    pub m: FixedBitSet,
    /// Schedule point per selected node carrying a `ScheduleDomain`.
    pub theta: FxHashMap<Id, SchedPoint>,
}

impl Extraction {
    pub fn is_materialized(&self, id: Id) -> bool {
        self.m.contains(id.index())
    }
    pub fn selected(&self, class: ClassId) -> Option<Id> {
        self.sigma.get(&class).copied()
    }
}

/// Choices local search makes over valid graph variants and schedules.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Move {
    Reselect(ClassId),
    Reschedule(Id),
}

/// Extraction limits, deterministic and clock-free: the plan is a
/// cross-process cache key, so effort is bounded in realized node visits.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ExtractBudget {
    pub moves_per_chain: u32,
    /// Realized node visits the local search may spend.
    pub max_move_work: u64,
}

impl Default for ExtractBudget {
    /// `64 * |chains|` moves, 90k realized node visits.
    fn default() -> Self {
        // `FUSOR_MOVE_WORK` overrides the visit budget.
        let max_move_work = std::env::var("FUSOR_MOVE_WORK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(90_000);
        Self {
            moves_per_chain: 64,
            max_move_work,
        }
    }
}

impl ExtractBudget {
    /// The move ceiling on a graph of `nodes` nodes and `chains` classes.
    pub fn move_cap(&self, nodes: usize, chains: u32) -> u32 {
        let by_work = (self.max_move_work / (nodes.max(1) as u64)).min(u32::MAX as u64) as u32;
        let by_chain = self.moves_per_chain.saturating_mul(chains.max(1));
        by_work.min(by_chain)
    }
}

/// The extracted plan's identity: `hash(realized term + M + theta + device)`.
/// Symbols hash as symbols, so one plan serves a whole shape family.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct PlanHash(pub u128);

/// Whether a buffer is read, written or both by a launch.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BindKind {
    Read,
    Write,
    ReadWrite,
}

/// One storage binding of one launch, in binding-index order.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BindingPlan {
    pub binding: u32,
    pub value: Id,
    pub kind: BindKind,
    /// The value lives in the step arena; arena values share one binding.
    pub arena: bool,
}

/// One buffer the plan allocates, with the padded strides its geometry needs.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferPlan {
    pub value: Id,
    pub layout: crate::shape::Layout,
    pub elements: Dim,
    pub dtype: crate::dtype::Dtype,
    pub persistence: crate::dtype::Persistence,
    /// Byte offset in the step arena; `None` allocates its own buffer.
    pub arena: Option<u64>,
}

/// One dispatch in the extracted plan; `grid` is already folded to 3-D.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Dispatch {
    pub root: Id,
    pub members: SmallVec<[Id; 8]>,
    pub bindings: Vec<BindingPlan>,
    pub grid: [u32; 3],
    pub block: u32,
}

/// A complete, verified plan. `symbols` are the dims the uniform block must
/// carry, in binding order.
#[derive(Clone, Debug)]
pub struct Plan {
    pub extraction: Extraction,
    pub launches: Vec<Dispatch>,
    pub buffers: Vec<BufferPlan>,
    /// Bytes of the interval-colored step arena.
    pub arena_bytes: u64,
    pub symbols: Vec<crate::shape::SymId>,
    /// The `symbols` that are runtime scalars (`f32` words); the rest are
    /// `u32` extents, offsets or strides.
    pub scalar_symbols: Vec<crate::shape::SymId>,
    pub hash: PlanHash,
    pub cost: Picoseconds,
}

/// The extraction interface. Object-safe.
pub trait Extractor: Send + Sync {
    /// Per-node arithmetic floor, indexed by node id, for candidate ordering.
    fn lower_bound(&self, graph: &EGraph, cost: &dyn CostModel) -> Vec<Picoseconds>;

    /// Seed, realize, cost exactly, then local-search under `budget`.
    fn extract(
        &self,
        graph: &EGraph,
        roots: &[Id],
        cost: &dyn CostModel,
        budget: ExtractBudget,
    ) -> Result<Plan>;

    /// Extend a previous selection (a search hint) to the requested roots.
    fn extract_seeded(
        &self,
        graph: &EGraph,
        roots: &[Id],
        cost: &dyn CostModel,
        budget: ExtractBudget,
        seed: &Plan,
    ) -> Result<Plan> {
        let _ = seed;
        self.extract(graph, roots, cost, budget)
    }

    /// Hard conformance assert on the winner; a failure is an error, never a
    /// fallback.
    #[cfg(feature = "compiler-tests")]
    fn verify_plan(&self, graph: &EGraph, plan: &Plan) -> Result<()>;

    /// Test-only member sweep, independent of tuning budgets and caches.
    #[cfg(feature = "compiler-tests")]
    fn test_launch_variants(
        &self,
        graph: &EGraph,
        roots: &[Id],
        base: &Plan,
        launch_ix: usize,
        cost: &dyn CostModel,
    ) -> Vec<(String, Plan)>;

    /// Alternative plans for one launch of `base`: every `(class member,
    /// schedule point)` pair its class offers, each re-planned whole.
    /// Contractions below `min_macs` return nothing.
    fn launch_variants(
        &self,
        graph: &EGraph,
        roots: &[Id],
        base: &Plan,
        launch_ix: usize,
        cost: &dyn CostModel,
        min_macs: u64,
    ) -> Vec<(String, Plan)> {
        let _ = (graph, roots, base, launch_ix, cost, min_macs);
        Vec::new()
    }

    /// Candidate labels and their realized costs, cheapest first; the label
    /// space [`Self::replan_with_variants`] resolves against. An unrealizable
    /// label keeps its place at the maximum cost.
    fn launch_variant_labels(
        &self,
        graph: &EGraph,
        roots: &[Id],
        base: &Plan,
        launch_ix: usize,
        cost: &dyn CostModel,
        min_macs: u64,
    ) -> Vec<(String, Picoseconds)> {
        let _ = (graph, roots, base, launch_ix, cost, min_macs);
        Vec::new()
    }

    /// Construct the plan of a completed selection without searching.
    fn replan_extraction(
        &self,
        graph: &EGraph,
        roots: &[Id],
        ex: &mut Extraction,
        cost: &dyn CostModel,
    ) -> Result<Plan> {
        let _ = (graph, roots, ex, cost);
        Err(crate::error::Error::Plan(
            "this extractor cannot replan".into(),
        ))
    }

    /// Replan `base` with the named variant applied at each launch of
    /// `swaps`, verified once; `None` when nothing applied or the plan failed.
    fn replan_with_variants(
        &self,
        graph: &EGraph,
        roots: &[Id],
        base: &Plan,
        cost: &dyn CostModel,
        min_macs: u64,
        swaps: &[(usize, String)],
    ) -> Option<Plan> {
        let _ = (graph, roots, base, cost, min_macs, swaps);
        None
    }
}

/// The replay memo key: the root closure's term (symbols as symbols) and the
/// device.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReplayKey {
    pub l0_term: u64,
    pub device: u64,
}
