//! The one extraction: node selection, materialization and schedule points,
//! decided together against the exact global cost.

use crate::cost::{CostModel, Picoseconds};
use crate::egraph::{ClassId, EGraph, Id};
use crate::error::Result;
use crate::ir::level1::SchedPoint;
use crate::shape::Dim;
use fixedbitset::FixedBitSet;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// The complete extraction state. Cost is **not** defined per e-class: a
/// shared value would be counted once per path (or once globally, both
/// wrong), and materialization is not a property of any node. Cost is
/// evaluated on the realized DAG under `(sigma, m, theta)`.
#[derive(Clone, Debug, Default)]
pub struct Extraction {
    /// E-class -> the selected member of that class.
    pub sigma: FxHashMap<ClassId, Id>,
    /// The materialized set. A node in `M` pays one write and each consumer
    /// pays one read; a node outside `M` is inlined into every consumer,
    /// paying its math once per consumer and no traffic. Rematerialization
    /// is therefore priced exactly as
    /// `saved_write + saved_reads - recompute * (consumers - 1)`.
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

/// The three moves local search makes. `Flip` is **refused when the node is
/// pinned** — an `Effect::InPlace` node is pinned in `M`, because inlining
/// an atomic scatter into two consumers doubles the embedding gradient.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Move {
    Reselect(ClassId),
    Flip(Id),
    Reschedule(Id),
}

/// Extraction limits. Deterministic: ties break by node id. **The full
/// schedule domain stays reachable** and the accept test is always the exact
/// global cost.
///
/// **No term is a clock.** The plan is the cache key and the cache is
/// cross-process, so an extraction that stops when a deadline expires
/// produces a different `PlanHash` on a loaded machine than on an idle one.
/// Every move re-realizes the whole DAG, so the honest unit of search effort
/// is *realized node visits*, which is what [`Self::max_move_work`] bounds.
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
    /// 90k is the deterministic restatement of the 2 ms wall clock it
    /// replaces: at ~20 ns per realized node it is the same ~1.8 ms of
    /// search on the 100-600 node graphs the suite is made of, and on the
    /// 3,000-node step graph it admits the ~30 moves the clock did.
    ///
    /// # Raising it was measured and deliberately not landed
    ///
    /// **The measurement below is stale in its numbers and still correct in
    /// its finding.** It was taken before `fusor2_cost::extract`'s
    /// co-selection pass landed, which moved the same four shapes further than
    /// raising this constant ever did (7/7, 8/8, 7/6, 29/19 -> 5/5, 6/6, 5/5,
    /// 17/17) *at this budget*. The `1_000_000` figures below have not been
    /// re-taken on top of it and should not be quoted as current. What has
    /// been re-checked is the blocker: with co-selection landed,
    /// `attention_causal_plan_is_no_worse_than_dense` passes at 90k. Whether
    /// it still separates at convergence is an open question, and the
    /// experiment is cheap.
    ///
    /// Recorded here so the experiment is not repeated blind. Two corrections
    /// to the obvious reading of these numbers first:
    ///
    /// * **`max_move_work` is not the only binding term, and on the attention
    ///   graphs it stops being the binding one almost immediately.** Measured
    ///   on the square-attention graph `attention_causal_plan_is_no_worse_than_dense`
    ///   builds: 605 nodes but only *15 e-classes*, so `by_chain` is
    ///   `64 * 15 = 960`. At 90k `by_work` is `90_000 / 605 = 148` and binds;
    ///   at 1M it is 1652 and `by_chain`'s 960 binds instead. Anything above
    ///   ~600k therefore changes nothing on a graph this shape without also
    ///   raising [`Self::moves_per_chain`].
    /// * The four `launch_ceiling` shapes do improve. Measured cpu/gpu at
    ///   `max_move_work = 1_000_000`: `attention_forward` 7/7 -> 6/7,
    ///   `attention_with_lse` 8/8 -> 7/8, `attention_causal_forward` 7/6 ->
    ///   6/6, `attention_grads_all_three` 29/19 -> 19/19. Since each ceiling
    ///   is the larger of the two backends, only two ceilings actually move
    ///   (causal 7 -> 6, grads 29 -> 19); the GPU plans are unchanged.
    ///
    /// What blocks it is `attention_causal_plan_is_no_worse_than_dense`, and
    /// it is a real finding rather than a brittle assert. At 90k *both*
    /// searches are truncated onto the same 7-launch / 2240-byte plan, so the
    /// case passes because neither side got to search, not because the
    /// invariant holds. Let both run to convergence (`moves_per_chain = 4096`,
    /// `max_move_work = 10_000_000`, both `capped=false`) and they separate:
    /// dense reaches 6 launches / 1600 bytes and causal 6 launches / 1840,
    /// because dense finds a **40-element buffer where causal keeps a
    /// 100-element one** — two slots of one 20-element carrier replacing a
    /// whole `[B,H,Lq,Lk]` intermediate. The rules that fired are identical on
    /// the two graphs (`STRIP` fires on neither), so this is not a missing
    /// law: it is the causal graph's local optimum genuinely being worse. The
    /// roofline costs agree and are 0.002% apart — the extractor is minimising
    /// time and the case asserts bytes.
    ///
    /// So the budget is not the last blocker; the causal side's inability to
    /// reach the two-slot carrier is. Raising this constant before that is
    /// fixed trades one measured regression for two ceiling improvements, and
    /// `git log` would record it as a green round. It is left at 90k.
    fn default() -> Self {
        Self {
            moves_per_chain: 64,
            max_move_work: 90_000,
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

/// The extracted plan's identity. **The plan is the cache key**:
/// `hash(realized DAG term + M + theta + DeviceFacts::fingerprint)`.
/// `Dim::Sym` and `Leaf::Uniform` hash as symbols, not values, so one plan
/// serves a whole shape family. There is no `hash_kernel_fields`, no
/// `structural_kernel_key` and no golden byte files.
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

/// One buffer the plan allocates. **Allocation is derived from the plan**:
/// the layout carries the padded strides the selected geometry needs,
/// including split-K scratch slices, so the reference's exact-stride
/// equality test and silent generic-reduce fallback become an invariant.
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

/// The extraction interface. Object-safe. One implementation is the shipped
/// local search; `fusor2-conformance` ships a debug ILP oracle behind the
/// same trait that must agree with it on small graphs — a greedy search
/// compared only against itself cannot distinguish "found the optimum" from
/// "made the same mistake".
pub trait Extractor: Send + Sync {
    /// Admissible lower bound, bottom-up, `O(nodes)`: `min over n in c of
    /// (math_ps(n) + sum over *distinct* child chains lb(child))` — zero
    /// traffic, free sharing, min over the schedule domain. A genuine
    /// relaxation in both regimes, so it is a valid seed *and* a valid
    /// branch-and-bound prune. Indexed by node id.
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
    /// whole. Family **and** geometry vary together, which is the only way a
    /// single measurement can rank them — see the `candidate_geoms_for` doc
    /// in `fusor2-tile`.
    ///
    /// Contractions below `min_macs` return nothing, so a suite of 3x4x2048
    /// matmuls never pays for a measurement round. The default is "no
    /// alternatives": an extractor that cannot replan is simply not tuned.
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

/// The replay memo, keyed on the *extraction inputs* rather than a
/// structural fingerprint. Validity is "the inputs are identical", so a
/// training loop can no longer freeze step 1's decisions forever.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReplayKey {
    pub l0_term: u64,
    pub device: u64,
    pub binding: u64,
}
