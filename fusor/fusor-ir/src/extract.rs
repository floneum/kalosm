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

/// Extraction limits. Deterministic: ties break by node id, the full
/// schedule domain stays reachable, and the accept test is always the exact
/// global cost.
///
/// No term is a clock: the plan is the cache key and the cache is
/// cross-process, so a deadline would produce a different `PlanHash` on a
/// loaded machine than on an idle one. Search effort is bounded in realized
/// node visits via [`Self::max_move_work`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ExtractBudget {
    pub moves_per_chain: u32,
    /// Realized node visits the local search may spend: it stops after
    /// `max_move_work / graph.len()` moves, whichever of that and
    /// `moves_per_chain * chains` is smaller.
    pub max_move_work: u64,
}

impl Default for ExtractBudget {
    /// `64 * |chains|` moves, 90k realized node visits.
    ///
    /// Limits additional local-search work after the complete seed sweep.
    fn default() -> Self {
        // `FUSOR_MOVE_WORK` overrides the visit budget, for measuring what
        // a longer search buys on a given graph.
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
    /// The move ceiling this budget implies on a graph of `nodes` nodes and
    /// `chains` classes. A pure function of the budget and the graph, so two
    /// runs of one graph search exactly as far as each other.
    pub fn move_cap(&self, nodes: usize, chains: u32) -> u32 {
        let by_work = (self.max_move_work / (nodes.max(1) as u64)).min(u32::MAX as u64) as u32;
        let by_chain = self.moves_per_chain.saturating_mul(chains.max(1));
        by_work.min(by_chain)
    }
}

/// The extracted plan's identity. The plan is the cache key:
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
    /// The value lives in the plan's step arena: all arena values in a
    /// launch share one binding, with typed views at their own offsets.
    pub arena: bool,
}

/// One buffer the plan allocates. Allocation is derived from the plan:
/// the layout carries the padded strides the selected geometry needs,
/// including split-K scratch slices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferPlan {
    pub value: Id,
    pub layout: crate::shape::Layout,
    pub elements: Dim,
    pub dtype: crate::dtype::Dtype,
    pub persistence: crate::dtype::Persistence,
    /// Byte offset in the step arena, for a step-local intermediate whose
    /// live range was packed there; `None` allocates its own buffer.
    pub arena: Option<u64>,
}

/// One dispatch in the extracted plan. `grid` is after the 3-D fold against
/// `max_compute_workgroups_per_dimension`.
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
    /// Bytes of the step arena: intermediates with disjoint live ranges
    /// share it, interval-colored at derivation.
    pub arena_bytes: u64,
    pub symbols: Vec<crate::shape::SymId>,
    /// The subset of `symbols` that are runtime scalars (`Leaf::Uniform`),
    /// carried as `f32` words. Every other symbol is an extent, offset or
    /// stride: a `u32` word, whether or not a buffer layout mentions it.
    pub scalar_symbols: Vec<crate::shape::SymId>,
    pub hash: PlanHash,
    pub cost: Picoseconds,
}

/// The extraction interface. Object-safe. One implementation is the shipped
/// local search; `fusor-conformance` ships a debug ILP oracle behind the
/// same trait that must agree with it on small graphs.
pub trait Extractor: Send + Sync {
    /// Per-node arithmetic floor, indexed by node id, for candidate ordering.
    /// It is not a dependency-DAG cost: selection compares complete realized
    /// plans so every shared producer and every distinct branch is counted.
    fn lower_bound(&self, graph: &EGraph, cost: &dyn CostModel) -> Vec<Picoseconds>;

    /// Seed, realize, cost exactly, then local-search under `budget`.
    fn extract(
        &self,
        graph: &EGraph,
        roots: &[Id],
        cost: &dyn CostModel,
        budget: ExtractBudget,
    ) -> Result<Plan>;

    /// Extend a previous selection from this graph to the requested roots.
    /// The seed is a search hint; its buffers and cost must be constructed again.
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

    /// Hard conformance assert on the winner: every selected non-leaf is
    /// Launch; every geometry legal against the exact `ArenaPlan`; every
    /// operand access satisfiable; every buffer stride derivable; no
    /// `InPlace` node inlined. A failure is an error, never a fallback.
    #[cfg(feature = "compiler-tests")]
    fn verify_plan(&self, graph: &EGraph, plan: &Plan) -> Result<()>;

    /// Test-only member sweep. Coverage is independent of tuning budgets,
    /// cost thresholds and the timing cache.
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
    /// schedule point)` pair the launch root's class offers, each re-planned
    /// whole. Family and geometry vary together — see the
    /// `candidate_schedules_for` doc in `fusor-tile`.
    ///
    /// Contractions below `min_macs` return nothing. The default is "no
    /// alternatives".
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

    /// Candidate labels and their complete realized costs, without deriving
    /// buffers, bindings or plan hashes. Lower costs come first; a label that
    /// cannot realize keeps its place in the offering with the maximum cost.
    ///
    /// The list is a superset of what `launch_variants` returns — a label
    /// here may still fail to realize — and is exactly the label space
    /// [`Self::replan_with_variants`] resolves against, so a name from here
    /// always names the same `(member, schedule point)` there.
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

    /// Replan `base` with the named variant applied at each launch of
    /// `swaps`, composed into one extraction and re-planned (and so
    /// verified) once. Labels are the strings `launch_variants` returns; a
    /// launch whose label matches no member point, or whose selection move
    /// is illegal, contributes nothing. `None` when nothing applied or the
    /// composed plan failed to build — the caller falls back to sequential
    /// adoption. The default is "no batch".
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

/// The replay memo, keyed on the extraction inputs. Validity is "the inputs
/// are identical": the root closure's term (symbols as symbols, so one plan
/// serves a whole shape family and the values reach the dispatch through the
/// uniform block) and the device.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReplayKey {
    pub l0_term: u64,
    pub device: u64,
}
