//! Extraction: node selection, materialization and schedule points, decided
//! together against the exact global cost.

use crate::cost::{CostModel, Picoseconds};
use crate::egraph::{ClassId, EGraph, Id};
use crate::error::Result;
use crate::ir::level1::SchedPoint;
use crate::shape::Dim;
use fixedbitset::FixedBitSet;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// The complete extraction state. Cost is evaluated on the realized DAG
/// under `(sigma, m, theta)`, never per e-class.
#[derive(Clone, Debug, Default)]
pub struct Extraction {
    /// E-class -> the selected member of that class.
    pub sigma: FxHashMap<ClassId, Id>,
    /// The materialized set. A node in `M` pays one write and each consumer
    /// pays one read; a node outside `M` is inlined into every consumer,
    /// paying its math once per consumer and no traffic. Rematerialization
    /// prices as `saved_write + saved_reads - recompute * (consumers - 1)`.
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

/// The three moves local search makes. An `Effect::InPlace` node is pinned in
/// `M` and `Flip` is refused on it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Move {
    Reselect(ClassId),
    Flip(Id),
    Reschedule(Id),
}

/// Extraction limits. Deterministic: ties break by node id, the full schedule
/// domain stays reachable, and the accept test is the exact global cost.
/// Search effort is bounded in realized node visits, not wall time, so the
/// resulting `PlanHash` does not depend on machine load.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ExtractBudget {
    pub moves_per_chain: u32,
    /// Realized node visits the local search may spend: it stops after
    /// `max_move_work / graph.len()` moves, whichever of that and
    /// `moves_per_chain * chains` is smaller.
    pub max_move_work: u64,
}

impl Default for ExtractBudget {
    /// `64 * |chains|` moves, capped at 90k realized node visits.
    fn default() -> Self {
        Self {
            moves_per_chain: 64,
            max_move_work: 90_000,
        }
    }
}

impl ExtractBudget {
    /// The move ceiling this budget implies on a graph of `nodes` nodes and
    /// `chains` classes. A pure function of the budget and the graph.
    pub fn move_cap(&self, nodes: usize, chains: u32) -> u32 {
        let by_work = (self.max_move_work / (nodes.max(1) as u64)).min(u32::MAX as u64) as u32;
        let by_chain = self.moves_per_chain.saturating_mul(chains.max(1));
        by_work.min(by_chain)
    }
}

/// The extracted plan's identity, and the cache key:
/// `hash(realized DAG term + M + theta + DeviceFacts::fingerprint)`.
/// `Dim::Sym` and `Leaf::Uniform` hash as symbols, not values, so one plan
/// serves a whole shape family.
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
}

/// One buffer the plan allocates. The layout carries the padded strides the
/// selected geometry needs, including split-K scratch slices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferPlan {
    pub value: Id,
    pub layout: crate::shape::Layout,
    pub elements: Dim,
    pub dtype: crate::dtype::Dtype,
    pub persistence: crate::dtype::Persistence,
}

/// One dispatch in the extracted plan. `grid` is after the 3-D fold against
/// `max_compute_workgroups_per_dimension`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Launch {
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
    pub launches: Vec<Launch>,
    pub buffers: Vec<BufferPlan>,
    pub symbols: Vec<crate::shape::SymId>,
    pub hash: PlanHash,
    pub cost: Picoseconds,
}

/// The extraction interface. Object-safe. The shipped implementation is a
/// local search; `fusor2-conformance` provides a debug ILP oracle behind the
/// same trait that must agree with it on small graphs.
pub trait Extractor: Send + Sync {
    /// Admissible lower bound, bottom-up, `O(nodes)`: `min over n in c of
    /// (math_ps(n) + sum over *distinct* child chains lb(child))` — zero
    /// traffic, free sharing, min over the schedule domain. Indexed by node
    /// id.
    fn lower_bound(&self, graph: &EGraph, cost: &dyn CostModel) -> Vec<Picoseconds>;

    /// Seed, realize, cost exactly, then local-search under `budget`.
    fn extract(
        &self,
        graph: &EGraph,
        roots: &[Id],
        cost: &dyn CostModel,
        budget: ExtractBudget,
    ) -> Result<Plan>;

    /// Hard conformance assert on the winner: every selected non-leaf is
    /// L1; every geometry legal against the exact `ArenaPlan`; every
    /// operand access satisfiable; every buffer stride derivable; no
    /// `InPlace` node inlined. A failure is an error, never a fallback.
    fn verify_plan(&self, graph: &EGraph, plan: &Plan) -> Result<()>;

    /// Alternative plans for one launch of `base`: every `(class member,
    /// schedule point)` pair the launch root's class offers, each re-planned
    /// whole, so family and geometry vary together. Contractions below
    /// `min_macs` return nothing. The default returns no alternatives.
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
}

/// The replay memo, keyed on the extraction inputs rather than a structural
/// fingerprint: the memo is valid only when all three inputs match.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReplayKey {
    pub l0_term: u64,
    pub device: u64,
    pub binding: u64,
}
