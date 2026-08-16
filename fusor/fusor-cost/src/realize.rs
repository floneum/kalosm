//! Realizing an [`Extraction`] into a DAG, and cutting that DAG into launches.
//!
//! Launches are the connected components of the realized DAG cut at `M`
//! boundaries and at forced boundaries (index-space mismatch, fold-to-fold
//! dependency). Consumer counts come from the DAG, so rematerialization is
//! priced as `saved_write + saved_reads - recompute * (consumers - 1)`.

use fusor_ir::cost::{CostModel, LaunchPlan, Picoseconds};
use fusor_ir::device::Caps;
use fusor_ir::dtype::Dtype;
use fusor_ir::egraph::{ClassId, EGraph, Id};
use fusor_ir::error::{Error, Result};
use fusor_ir::extract::Extraction;
use fusor_ir::facts::{ValueFacts, Work};
use fusor_ir::ir::Op;
use fusor_ir::ir::kernel::{
    ArenaPlanner, MemoryLevel, ScalarElement, Tile, TileDecl, TileLayout, Tiles,
};
use fusor_ir::ir::launch::{Effect, FoldStrat, IndexSpace, Launch, SchedPoint, ScheduleDomain};
use fusor_ir::ir::logical::{LeafKind, Logical};
use fusor_ir::shape::Dim;
use smallvec::SmallVec;
use std::sync::Arc;

/// Extent a `Dim::Sym` prices at: a nominal value keeps the ranking total
/// without letting a concrete binding leak into the plan.
pub const SYM_NOMINAL: u64 = 1024;

/// What role a leaf plays in the realized DAG.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LeafRole {
    /// Not a leaf; an ordinary launch member.
    NotLeaf,
    /// A constant or a uniform scalar: no buffer, no traffic, no component.
    Free,
    /// An externally supplied buffer: read traffic, but never a write and
    /// never a `BufferPlan` — allocation derives only what the plan produces.
    External,
}

/// A dense map from [`Id`] to `T`. Ids are dense and monotone, so every
/// per-node table in the realized DAG is an array lookup rather than a hash.
#[derive(Clone, Debug, Default)]
pub struct IdMap<T> {
    slots: Vec<Option<T>>,
}

impl<T> IdMap<T> {
    pub fn with_len(n: usize) -> Self {
        Self {
            slots: (0..n).map(|_| None).collect(),
        }
    }

    #[inline]
    pub fn get(&self, id: Id) -> Option<&T> {
        self.slots.get(id.index())?.as_ref()
    }

    #[inline]
    pub fn contains(&self, id: Id) -> bool {
        self.get(id).is_some()
    }

    #[inline]
    pub fn insert(&mut self, id: Id, value: T) {
        if self.slots.len() <= id.index() {
            self.slots.resize_with(id.index() + 1, || None);
        }
        self.slots[id.index()] = Some(value);
    }

    #[inline]
    pub fn entry_or_default(&mut self, id: Id) -> &mut T
    where
        T: Default,
    {
        if self.slots.len() <= id.index() {
            self.slots.resize_with(id.index() + 1, || None);
        }
        self.slots[id.index()].get_or_insert_with(T::default)
    }

    pub fn iter(&self) -> impl Iterator<Item = (Id, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, v)| Some((Id(i as u32), v.as_ref()?)))
    }
}

impl<T: Copy> IdMap<T> {
    #[inline]
    pub fn copied(&self, id: Id) -> Option<T> {
        self.get(id).copied()
    }
}

/// Per-node values that do not change while the graph does not: [`Work`] is
/// a pure function of `(op, operand facts, own facts)`, so the search
/// computes it once instead of once per move.
#[derive(Default)]
pub struct NodeCache {
    work: Vec<Option<Work>>,
}

impl NodeCache {
    pub fn new(len: usize) -> Self {
        Self {
            work: (0..len).map(|_| None).collect(),
        }
    }

    fn work_of(&mut self, graph: &EGraph, id: Id) -> Work {
        if self.work.len() <= id.index() {
            self.work.resize_with(id.index() + 1, || None);
        }
        if let Some(w) = self.work[id.index()] {
            return w;
        }
        let node = graph.node(id);
        let ins: SmallVec<[ValueFacts; 4]> = node
            .children
            .iter()
            .map(|c| graph.facts(*c).clone())
            .collect();
        let w = graph.semantics().work(&node.op, &ins, graph.facts(id));
        self.work[id.index()] = Some(w);
        w
    }
}

/// One launch: a connected component of the realized DAG.
#[derive(Clone, Debug)]
pub struct Component {
    pub root: Id,
    pub members: Vec<Id>,
    /// `(bytes, reread)` per distinct external operand, in operand-id order.
    pub reads: Vec<(u64, u32)>,
    /// The producers those reads come from, same order.
    pub external: Vec<Id>,
    pub writes: u64,
    pub work: Work,
    pub resident_lanes: u64,
    pub wg_bytes: u64,
    pub grid: [u32; 3],
    pub block: u32,
}

/// The DAG one `(sigma, m, theta)` denotes.
///
/// `LaunchPlan` borrows its `members` and `reads` slices, so an owned launch
/// list would make this struct self-referential; launches are built on demand
/// by [`Realized::launches`] from the owned [`Component`]s.
#[derive(Clone, Debug, Default)]
pub struct Realized {
    /// Selected nodes in post-order, leaves included.
    pub order: Vec<Id>,
    /// Distinct realized consumers, plus one when the node is a root.
    pub consumers: IdMap<u32>,
    /// The consumers themselves, so a per-node query is O(consumers) rather
    /// than a scan of the whole operand map.
    pub consumer_nodes: IdMap<SmallVec<[Id; 4]>>,
    /// Component index per non-leaf selected node.
    pub launch_of: IdMap<u32>,
    pub components: Vec<Component>,
    /// Resolved children per selected node, in operand order.
    pub operands: IdMap<SmallVec<[Id; 4]>>,
    /// The roots after resolution through `sigma`.
    pub roots: Vec<Id>,
}

impl Realized {
    /// Borrowed launch views for the cost model. `extraction` supplies the
    /// `theta` map the plans point at, so no map is cloned per move.
    pub fn launches<'a>(&'a self, extraction: &'a Extraction) -> Vec<LaunchPlan<'a>> {
        self.components
            .iter()
            .map(|c| LaunchPlan {
                members: &c.members,
                root: c.root,
                theta: &extraction.theta,
                reads: &c.reads,
                writes: c.writes,
                work: c.work,
                resident_lanes: c.resident_lanes,
                wg_bytes: c.wg_bytes,
                grid: c.grid,
            })
            .collect()
    }

    pub fn is_root(&self, id: Id) -> bool {
        self.roots.contains(&id)
    }
}

/// Math a cooperative contraction's staging fill re-executes beyond what the
/// schedule-independent `work_of` row counts.
///
/// The A tile is re-staged once per n-tile of the grid and the B tile once
/// per m-tile, and each staging pass runs the side's `pre` per loaded
/// element. `work_of` prices one execution per element, so the extra is
/// `pre_work x (tiles - 1)` per side; an identity `pre` contributes zero.
fn staging_rework(graph: &EGraph, member: Id, theta: Option<SchedPoint>) -> Work {
    let Some(SchedPoint::Coop { geom, .. }) = theta else {
        return Work::default();
    };
    let Op::Launch(Launch::Contract {
        m,
        n,
        k,
        batch,
        a,
        b,
        ..
    }) = &graph.node(member).op
    else {
        return Work::default();
    };
    let priced = |d: &Dim| d.as_const().unwrap_or(1).max(1);
    let (m, n, k, batch) = (priced(m), priced(n), priced(k), priced(batch));
    let tiles_m = m.div_ceil(u64::from(geom.bm.max(1))).max(1);
    let tiles_n = n.div_ceil(u64::from(geom.bn.max(1))).max(1);
    let side_extra = |side: &fusor_ir::ir::launch::ContractSide, elems: u64, tiles: u64| {
        let mut w = fusor_ir::semantics::work::epilogue_work(&side.pre, elems);
        for o in &side.ops {
            let d = fusor_ir::semantics::work::decode_ops_of(graph.facts(o.src).dtype);
            w.index_ops = w.index_ops.saturating_add(elems.saturating_mul(d));
        }
        w.scale(tiles.saturating_sub(1))
    };
    let a_extra = side_extra(a, batch.saturating_mul(m).saturating_mul(k), tiles_n);
    let b_extra = side_extra(b, batch.saturating_mul(k).saturating_mul(n), tiles_m);
    a_extra.add(b_extra)
}

/// MACs a cooperative geometry issues on tile padding beyond the useful
/// `batch*m*n*k` that `work_of` prices. Padding is priced through
/// `Work::macs`, never vetoed.
fn coop_padding(graph: &EGraph, member: Id, theta: Option<SchedPoint>) -> Work {
    let Some(SchedPoint::Coop { geom, .. }) = theta else {
        return Work::default();
    };
    let Op::Launch(Launch::Contract { m, n, k, batch, .. }) = &graph.node(member).op else {
        return Work::default();
    };
    let priced = |d: &Dim| d.as_const().unwrap_or(1).max(1);
    let (m, n, k, batch) = (priced(m), priced(n), priced(k), priced(batch));
    let m_pad = m
        .div_ceil(u64::from(geom.bm.max(1)))
        .saturating_mul(u64::from(geom.bm.max(1)));
    let n_pad = n
        .div_ceil(u64::from(geom.bn.max(1)))
        .saturating_mul(u64::from(geom.bn.max(1)));
    let extra = m_pad
        .saturating_mul(n_pad)
        .saturating_sub(m.saturating_mul(n))
        .saturating_mul(k)
        .saturating_mul(batch);
    Work {
        macs: extra,
        ..Work::default()
    }
}

/// Realize `(sigma, m, theta)` from `roots` and cut it into launches.
///
/// A class with no `sigma` entry is [`Error::Plan`]; so is a selection whose
/// resolved edges form a cycle (a class member created *after* its own
/// consumer can be selected into one, and the search must be able to reject
/// that rather than loop).
pub fn realize(
    graph: &EGraph,
    roots: &[Id],
    extraction: &Extraction,
    cost: &dyn CostModel,
    arena: &dyn ArenaPlanner,
) -> Result<Realized> {
    let mut cache = NodeCache::new(graph.len());
    realize_with(graph, roots, extraction, cost, arena, &mut cache)
}

/// The same, reusing a [`NodeCache`] across the whole local search.
pub fn realize_with(
    graph: &EGraph,
    roots: &[Id],
    extraction: &Extraction,
    cost: &dyn CostModel,
    arena: &dyn ArenaPlanner,
    cache: &mut NodeCache,
) -> Result<Realized> {
    let caps = &cost.facts().caps;
    let resolved_roots = roots
        .iter()
        .map(|r| select(graph, extraction, *r))
        .collect::<Result<Vec<_>>>()?;

    let (order, operands) = walk(graph, extraction, &resolved_roots).map_err(Error::from)?;
    let (consumers, consumer_nodes) =
        count_consumers(graph.len(), &order, &operands, &resolved_roots);
    let (launch_of, groups) = cut(graph, extraction, &order, &operands, &resolved_roots);
    let components = groups
        .into_iter()
        .map(|members| {
            build_component(
                graph,
                extraction,
                &consumers,
                &launch_of,
                &resolved_roots,
                members,
                caps,
                arena,
                cache,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(Realized {
        order,
        consumers,
        consumer_nodes,
        launch_of,
        components,
        operands,
        roots: resolved_roots,
    })
}

/// `cost.total` over the realized launches. The accept test for every
/// local-search move is this number, never a local delta heuristic.
pub fn exact_cost(
    realized: &Realized,
    extraction: &Extraction,
    cost: &dyn CostModel,
) -> Picoseconds {
    let launches = realized.launches(extraction);
    cost.total(extraction, &launches)
}

/// True when an edge must be cut regardless of `M`: a leaf operand, an
/// index-space mismatch, a fold-to-fold dependency, a merged wave, an
/// in-place producer, or a producer that is itself a root.
pub fn forced_boundary(
    graph: &EGraph,
    extraction: &Extraction,
    roots: &[Id],
    producer: Id,
    consumer: Id,
) -> bool {
    if leaf_role(graph, producer) != LeafRole::NotLeaf {
        return true;
    }
    if extraction.is_materialized(producer) || roots.contains(&producer) {
        return true;
    }
    if graph.semantics().effect(&graph.node(producer).op) != Effect::Pure {
        return true;
    }
    structural_boundary(graph, producer, consumer)
}

/// The half of [`forced_boundary`] that `M` cannot argue with: a merged wave,
/// an index-space mismatch, or a chained reduction (the consumer's first
/// iteration needs the producer's whole axis to have landed).
///
/// Also the materialization obligation: an edge cut for one of these reasons
/// puts producer and consumer in different launches, so the producer has to
/// land in a buffer. `verify_plan`'s clause 3 is this statement.
pub fn structural_boundary(graph: &EGraph, producer: Id, consumer: Id) -> bool {
    if !index_space(graph, consumer).covers(&index_space(graph, producer)) {
        return true;
    }
    reduces(graph, producer) && reduces(graph, consumer)
}

/// True when `producer` must be in `M` for this edge to be runnable.
///
/// Either the edge is a [`structural_boundary`], so producer and consumer
/// land in different launches whatever `M` says. Or the consumer's own node
/// never absorbed the producer: a launch is lowered from one node, so a
/// producer can only share a kernel with its consumer where a rule already
/// folded it into one node whose operands are the producer's. Inlining any
/// other edge leaves the consumer's kernel reading an operand nothing ever
/// wrote.
///
/// A materialization obligation, not a cut rule: the cost model can still
/// price an inlined producer; the seed, the repair and the `FLIP` frontier
/// refuse to ship one.
pub fn needs_own_buffer(graph: &EGraph, producer: Id, consumer: Id) -> bool {
    structural_boundary(graph, producer, consumer) || !absorbs(graph, consumer, producer)
}

/// True when `consumer`'s own node already names `producer`'s class as a
/// member it computes, rather than as an operand it reads.
fn absorbs(graph: &EGraph, consumer: Id, producer: Id) -> bool {
    let class = graph.class_of(producer);
    match &graph.node(consumer).op {
        Op::Launch(Launch::Region { members, .. }) => {
            members.iter().any(|m| graph.class_of(*m) == class)
        }
        _ => false,
    }
}

/// The member `sigma` selected for `id`'s class.
pub fn select(graph: &EGraph, extraction: &Extraction, id: Id) -> Result<Id> {
    let class = graph.class_of(id);
    extraction
        .sigma
        .get(&class)
        .copied()
        .ok_or_else(|| Error::Plan(format!("class {} has no selected member", class.0)))
}

pub fn leaf_role(graph: &EGraph, id: Id) -> LeafRole {
    match &graph.node(id).op {
        Op::Logical(Logical::Leaf(LeafKind::Const { .. } | LeafKind::Uniform { .. })) => {
            LeafRole::Free
        }
        Op::Logical(Logical::Leaf(_)) => LeafRole::External,
        _ => LeafRole::NotLeaf,
    }
}

pub fn reduces(graph: &EGraph, id: Id) -> bool {
    matches!(
        graph.node(id).op,
        Op::Launch(Launch::Fold { .. } | Launch::Contract { .. })
            | Op::Logical(Logical::Fold { .. } | Logical::Contract { .. })
    )
}

/// The iteration domain of one node. Launch nodes carry it; everything else is
/// priced over its own shape.
pub fn index_space(graph: &EGraph, id: Id) -> IndexSpace {
    match &graph.node(id).op {
        Op::Launch(
            Launch::Map { space, .. }
            | Launch::Fold { space, .. }
            | Launch::Gather { space, .. }
            | Launch::Scatter { space, .. },
        ) => space.clone(),
        Op::Launch(Launch::Contract { batch, m, n, .. }) => IndexSpace::new([*batch, *m, *n]),
        _ => IndexSpace {
            dims: graph.facts(id).shape.clone(),
        },
    }
}

/// Extent a dim prices at.
pub const fn dim_extent(d: Dim) -> u64 {
    match d.as_const() {
        Some(v) => v,
        None => SYM_NOMINAL,
    }
}

pub fn elements_of(facts: &ValueFacts) -> u64 {
    facts
        .shape
        .iter()
        .map(|d| dim_extent(*d))
        .fold(1u64, |a, b| a.saturating_mul(b))
}

pub fn bytes_of(facts: &ValueFacts) -> u64 {
    let elems = elements_of(facts);
    match facts.dtype {
        Dtype::Q(fmt) => {
            let be = fmt.block_elements() as u64;
            elems.div_ceil(be) * fmt.block_bytes(fusor_ir::dtype::QLayout::Native) as u64
        }
        d => elems.saturating_mul(d.byte_size()),
    }
}

pub fn iterations_of(space: &IndexSpace) -> u64 {
    space
        .dims
        .iter()
        .map(|d| dim_extent(*d))
        .fold(1u64, |a, b| a.saturating_mul(b))
        .max(1)
}

/// Scalar element a dtype stages as.
pub const fn scalar_element(d: Dtype) -> ScalarElement {
    match d {
        Dtype::F32 | Dtype::Q(_) => ScalarElement::F32,
        Dtype::F16 => ScalarElement::F16,
        Dtype::BF16 => ScalarElement::BF16,
        Dtype::U32 => ScalarElement::U32,
        Dtype::I32 => ScalarElement::I32,
    }
}

/// The workgroup tiles a schedule point declares, fed straight into
/// [`ArenaPlanner::workgroup_bytes`]. This is the exact planner, never an
/// estimator. `lanes` is the fold carrier's accumulator lane count, `1` for
/// every other node: both emitters allocate one scratch tile of
/// [`fusor_ir::ir::launch::emitted_block`] elements per lane, so passing `1`
/// for a promoted carrier under-counts its scratch and lets `verify_plan`
/// admit a plan the GPU then refuses to lower.
pub fn tiles_for(
    theta: Option<SchedPoint>,
    elem: ScalarElement,
    fold_lanes: Option<u64>,
    caps: &Caps,
) -> Tiles {
    fn tile(name: &'static str, elem: ScalarElement, extents: &[u32]) -> Tile {
        Arc::new(TileDecl::new(
            elem.element(),
            TileLayout::contiguous(MemoryLevel::Workgroup, extents),
            name,
        ))
    }
    let mut decls: SmallVec<[Tile; 8]> = SmallVec::new();
    match theta {
        Some(SchedPoint::Coop { geom, staging, .. }) => {
            let passes = geom.n_passes.max(1);
            for _ in 0..staging.max(1) {
                decls.push(tile("coop_a", elem, &[geom.bm.max(1), geom.bk.max(1)]));
                decls.push(tile(
                    "coop_b",
                    elem,
                    &[geom.bk.max(1), (geom.bn / passes).max(1)],
                ));
            }
        }
        Some(SchedPoint::Sgemm(p)) => {
            let depth = if p.double_buffer { 2 } else { 1 };
            for _ in 0..depth {
                decls.push(tile("sgemm_a", elem, &[p.bm.max(1), p.bk.max(1)]));
                decls.push(tile("sgemm_b", elem, &[p.bk.max(1), p.bn.max(1)]));
            }
        }
        Some(SchedPoint::Sgemv(p)) => {
            // The subgroup-per-column structure (`cols > 1`) closes each
            // column inside one subgroup and stages nothing.
            if p.cols <= 1 {
                decls.push(tile("sgemv_partials", elem, &[p.subgroups.max(1)]));
            }
        }
        _ => {}
    }
    // A fold's cross-lane close is one scratch tile of `emitted_block`
    // elements per accumulator lane, at whatever block the point implies —
    // including a `Point`, which lowers at the default block. This arm is
    // keyed on the node being a fold, not on the point being a fold strategy.
    if let Some(lanes) = fold_lanes {
        let lane_group = fold_lane_group(theta, caps);
        // A one-lane group declares no tile: every invocation owns a whole
        // output row, so the cross-lane merge is an identity. Both emitters
        // skip the close at `lane_group == 1` and `fold_scratch_bytes`
        // reports 0 for the same strategy; all three statements of this
        // footprint have to agree.
        if lane_group > 1 {
            let block = fusor_ir::ir::launch::emitted_block(lane_group, caps);
            let extent = u32::try_from(lanes.max(1).saturating_mul(u64::from(block.max(1))))
                .unwrap_or(u32::MAX);
            decls.push(tile("fold_scratch", elem, &[extent]));
        }
    }
    Tiles { decls }
}

/// The lane group a fold lowers at under `theta`. A point that is not a fold
/// strategy — a `Point`, or a geometry inherited from a contraction domain —
/// takes the emitters' default, which is `emitted_block(1)`.
fn fold_lane_group(theta: Option<SchedPoint>, caps: &Caps) -> u32 {
    match theta {
        Some(SchedPoint::Fold(s)) => s.lane_group(caps.subgroup_width()),
        // The emitters' default is the full block, not 1: a `Point` fold
        // closes over the whole workgroup and stages
        // `lanes * block * acc_bytes`. Reporting 1 here would under-count its
        // footprint to zero and admit a plan the emitter cannot lay out.
        _ => fusor_ir::ir::launch::emitted_block(1, caps),
    }
}

/// Lanes per workgroup and workgroup count implied by one schedule point.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub block: u32,
    pub workgroups: u64,
}

pub fn geometry(theta: Option<SchedPoint>, space: &IndexSpace, caps: &Caps) -> Geometry {
    let width = caps.subgroup_width().max(1);
    let default_block = caps
        .limits
        .max_compute_invocations_per_workgroup
        .clamp(1, 256);
    let dims = &space.dims;
    let rank = dims.len();
    let m = if rank >= 2 {
        dim_extent(dims[rank - 2])
    } else {
        1
    };
    let n = if rank >= 1 {
        dim_extent(dims[rank - 1])
    } else {
        1
    };
    let batch: u64 = dims
        .iter()
        .take(rank.saturating_sub(2))
        .map(|d| dim_extent(*d))
        .fold(1u64, |a, b| a.saturating_mul(b));
    let total = iterations_of(space);

    match theta {
        Some(SchedPoint::Coop { geom, splits, .. }) => Geometry {
            block: geom.lanes(width).max(1),
            workgroups: m.div_ceil(geom.bm.max(1) as u64)
                * n.div_ceil(geom.bn.max(1) as u64)
                * batch
                * splits.max(1) as u64,
        },
        Some(SchedPoint::Sgemm(p)) => Geometry {
            block: ((p.bm / p.tm.max(1)) * (p.bn / p.tn.max(1))).max(1),
            workgroups: m.div_ceil(p.bm.max(1) as u64) * n.div_ceil(p.bn.max(1) as u64) * batch,
        },
        // The grid `lower_sgemv` actually launches: one workgroup per output
        // element at `cols == 1` (`batch * m * n`), one per `cols`-wide
        // column group at `cols > 1` (`batch * m * ceil(n / cols)`,
        // `lower_sgemv_subgroup_cols`).
        Some(SchedPoint::Sgemv(p)) => Geometry {
            block: (p.subgroups.max(1) * width).max(1),
            workgroups: m
                .saturating_mul(batch)
                .saturating_mul(n.div_ceil(u64::from(p.cols.max(1))))
                .max(1),
        },
        Some(SchedPoint::Fold(strat)) => {
            let lanes = match strat {
                FoldStrat::Subgroup => width,
                FoldStrat::WgTree { lane_group } | FoldStrat::LoopThenTree { lane_group, .. } => {
                    lane_group.max(1)
                }
            };
            Geometry {
                block: lanes.max(1),
                workgroups: (total / n.max(1)).max(1),
            }
        }
        Some(SchedPoint::Map(t)) => {
            let per = default_block as u64 * t.tm.max(1) as u64 * t.vector.max(1) as u64;
            Geometry {
                block: default_block,
                workgroups: total.div_ceil(per.max(1)).max(1),
            }
        }
        _ => Geometry {
            block: default_block,
            workgroups: total.div_ceil(default_block as u64).max(1),
        },
    }
}

/// The 3-D fold against `max_compute_workgroups_per_dimension`. **Slab count
/// first, then size x**: saturating x instead leaves the last slab nearly
/// empty and every extra group still runs the prologue.
pub fn distribute_workgroups(total: impl Into<u64>, max_per_dim: u32) -> [u32; 3] {
    let total = total.into();
    let max = u64::from(max_per_dim.max(1));
    if total <= max {
        return [total as u32, 1, 1];
    }
    let y = total.div_ceil(max).min(max);
    let x = total.div_ceil(y).min(max);
    let z = total
        .div_ceil(x.saturating_mul(y))
        .min(u64::from(u32::MAX))
        .max(1);
    [x as u32, y as u32, z as u32]
}

enum Frame {
    Enter(Id),
    Exit(Id),
}

type Operands = IdMap<SmallVec<[Id; 4]>>;

/// Why [`walk`] could not order the selected DAG.
///
/// `Cycle` is repairable: it names a class whose selected member closes a
/// loop, and [`crate::extract`] re-selects that one class.
enum WalkFail {
    Cycle(Id),
    Other(Error),
}

impl From<WalkFail> for Error {
    fn from(f: WalkFail) -> Self {
        match f {
            WalkFail::Cycle(v) => Error::Plan(format!(
                "selection is cyclic through {v}: a class member selected above its own consumer"
            )),
            WalkFail::Other(e) => e,
        }
    }
}

/// The node at which this selection closes a cycle, if it closes one.
///
/// The node graph is acyclic, but a selection over it need not be: [`select`]
/// replaces an operand id by its class's selected member, which may have a
/// larger id than the consumer that reached it. Two classes can form a cycle
/// in which neither member names its own class, so [`is_self_referential`]
/// (the depth-1 case) sees nothing.
pub fn selection_cycle(graph: &EGraph, extraction: &Extraction, roots: &[Id]) -> Option<Id> {
    let resolved = roots
        .iter()
        .map(|r| select(graph, extraction, *r))
        .collect::<Result<Vec<_>>>()
        .ok()?;
    match walk(graph, extraction, &resolved) {
        Err(WalkFail::Cycle(v)) => Some(v),
        _ => None,
    }
}

fn walk(
    graph: &EGraph,
    extraction: &Extraction,
    roots: &[Id],
) -> std::result::Result<(Vec<Id>, Operands), WalkFail> {
    const UNSEEN: u8 = 0;
    const OPEN: u8 = 1;
    const DONE: u8 = 2;

    let mut state = vec![UNSEEN; graph.len()];
    let mut order: Vec<Id> = Vec::new();
    let mut operands: Operands = IdMap::with_len(graph.len());
    let mut stack: Vec<Frame> = roots.iter().rev().map(|r| Frame::Enter(*r)).collect();

    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Enter(v) => match state[v.index()] {
                DONE => {}
                OPEN => return Err(WalkFail::Cycle(v)),
                _ => {
                    state[v.index()] = OPEN;
                    let kids: SmallVec<[Id; 4]> = graph
                        .node(v)
                        .children
                        .iter()
                        .map(|c| select(graph, extraction, *c))
                        .collect::<Result<_>>()
                        .map_err(WalkFail::Other)?;
                    stack.push(Frame::Exit(v));
                    for c in kids.iter().rev() {
                        stack.push(Frame::Enter(*c));
                    }
                    operands.insert(v, kids);
                }
            },
            Frame::Exit(v) => {
                state[v.index()] = DONE;
                order.push(v);
            }
        }
    }
    Ok((order, operands))
}

type Consumers = (IdMap<u32>, IdMap<SmallVec<[Id; 4]>>);

fn count_consumers(len: usize, order: &[Id], operands: &Operands, roots: &[Id]) -> Consumers {
    let mut seen: IdMap<SmallVec<[Id; 4]>> = IdMap::with_len(len);
    for v in order {
        for c in operands.get(*v).map(|o| o.as_slice()).unwrap_or(&[]) {
            let e = seen.entry_or_default(*c);
            if !e.contains(v) {
                e.push(*v);
            }
        }
    }
    let mut out: IdMap<u32> = IdMap::with_len(len);
    for v in order {
        let mut n = seen.get(*v).map_or(0, |c| c.len() as u32);
        if roots.contains(v) {
            n += 1;
        }
        out.insert(*v, n);
    }
    (out, seen)
}

/// Disjoint-set over positions in `order`.
struct Dsu(Vec<usize>);

impl Dsu {
    fn new(n: usize) -> Self {
        Self((0..n).collect())
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.0[x] != x {
            self.0[x] = self.0[self.0[x]];
            x = self.0[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            // Keep the *earlier* position as the representative so component
            // numbering follows `order` and is therefore deterministic.
            let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
            self.0[hi] = lo;
        }
    }
}

fn cut(
    graph: &EGraph,
    extraction: &Extraction,
    order: &[Id],
    operands: &Operands,
    roots: &[Id],
) -> (IdMap<u32>, Vec<Vec<Id>>) {
    let mut pos: IdMap<usize> = IdMap::with_len(graph.len());
    for (i, v) in order.iter().enumerate() {
        pos.insert(*v, i);
    }
    let mut dsu = Dsu::new(order.len());

    for (i, v) in order.iter().enumerate() {
        if leaf_role(graph, *v) != LeafRole::NotLeaf {
            continue;
        }
        for c in operands.get(*v).map(|o| o.as_slice()).unwrap_or(&[]) {
            if forced_boundary(graph, extraction, roots, *c, *v) {
                continue;
            }
            if let Some(j) = pos.get(*c) {
                dsu.union(i, *j);
            }
        }
    }

    let mut index_of: Vec<u32> = vec![u32::MAX; order.len()];
    let mut groups: Vec<Vec<Id>> = Vec::new();
    let mut launch_of: IdMap<u32> = IdMap::with_len(graph.len());
    for (i, v) in order.iter().enumerate() {
        if leaf_role(graph, *v) != LeafRole::NotLeaf {
            continue;
        }
        let r = dsu.find(i);
        if index_of[r] == u32::MAX {
            groups.push(Vec::new());
            index_of[r] = (groups.len() - 1) as u32;
        }
        let idx = index_of[r];
        groups[idx as usize].push(*v);
        launch_of.insert(*v, idx);
    }
    (launch_of, groups)
}

#[allow(clippy::too_many_arguments)]
fn build_component(
    graph: &EGraph,
    extraction: &Extraction,
    consumers: &IdMap<u32>,
    launch_of: &IdMap<u32>,
    roots: &[Id],
    members: Vec<Id>,
    caps: &Caps,
    arena: &dyn ArenaPlanner,
    cache: &mut NodeCache,
) -> Result<Component> {
    let own = members
        .first()
        .and_then(|m| launch_of.copied(*m))
        .unwrap_or(0);

    // The component's output is the last member that lands in a buffer.
    let root = members
        .iter()
        .rev()
        .find(|m| extraction.is_materialized(**m) || roots.contains(m))
        .copied()
        .or_else(|| members.last().copied())
        .ok_or_else(|| Error::Plan("empty launch component".into()))?;

    let mut writes = 0u64;
    let mut work = Work::default();
    for m in &members {
        let out = graph.facts(*m);
        let mut w = cache.work_of(graph, *m);
        let theta_m = extraction.theta.get(m).copied();
        w = w.add(staging_rework(graph, *m, theta_m));
        w = w.add(coop_padding(graph, *m, theta_m));
        let materialized = extraction.is_materialized(*m) || roots.contains(m);
        if materialized {
            writes = writes.saturating_add(bytes_of(out));
            work = work.add(w);
        } else {
            // Inlined into every consumer: pays its math once per consumer
            // and no traffic.
            work = work.add(w.scale(consumers.copied(*m).unwrap_or(1).max(1) as u64));
        }
    }

    // Distinct external operands, with the reread factor the consuming
    // iteration space implies.
    let mut ext: Vec<(Id, u64, u32)> = Vec::new();
    for m in &members {
        let iters = iterations_of(&index_space(graph, *m));
        for c in graph.node(*m).children.iter() {
            let c = select(graph, extraction, *c)?;
            if launch_of.copied(c) == Some(own) {
                continue;
            }
            if leaf_role(graph, c) == LeafRole::Free {
                continue;
            }
            let facts = graph.facts(c);
            let elems = elements_of(facts).max(1);
            let reread = iters.div_ceil(elems).max(1).min(u32::MAX as u64) as u32;
            match ext.iter_mut().find(|(id, _, _)| *id == c) {
                Some(slot) => slot.2 = slot.2.max(reread),
                None => ext.push((c, bytes_of(facts), reread)),
            }
        }
    }
    ext.sort_by_key(|(id, _, _)| *id);

    let theta = extraction.theta.get(&root).copied();
    let space = index_space(graph, root);
    let geom = geometry(theta, &space, caps);
    let lanes = fold_footprint(graph, root).map(|(l, _)| l);
    let tiles = tiles_for(theta, scalar_element(graph.facts(root).dtype), lanes, caps);
    let wg_bytes = arena.workgroup_bytes(&tiles, caps)? as u64;

    Ok(Component {
        root,
        members,
        reads: ext.iter().map(|(_, b, r)| (*b, *r)).collect(),
        external: ext.iter().map(|(id, _, _)| *id).collect(),
        writes,
        work,
        resident_lanes: geom.workgroups.saturating_mul(geom.block as u64),
        wg_bytes,
        grid: distribute_workgroups(
            geom.workgroups,
            caps.limits.max_compute_workgroups_per_dimension,
        ),
        block: geom.block,
    })
}

/// True when a class has exactly one member, in which case selection is
/// forced and no member vector need be built.
pub fn is_singleton(graph: &EGraph, class: ClassId) -> bool {
    !matches!(graph.node(class.0).op, Op::Union(..))
}

/// True when `id` is a node the plan may actually select: a `Leaf`, or
/// anything at `Level::Launch`.
///
/// This is clause 1 of `verify_plan`, and every decision that writes `sigma`
/// has to agree with it. It cannot be left to the cost model: a `Logical` node
/// and its lowered `Launch` twin report the same `work()`, so cost ties and a
/// tie broken by smaller `Id` returns the un-lowered original.
pub fn is_runnable(graph: &EGraph, id: Id) -> bool {
    if !matches!(graph.node(id).op, Op::Logical(Logical::Leaf(_)))
        && graph.level(id) != fusor_ir::ir::Level::Launch
    {
        return false;
    }
    !is_self_referential(graph, id)
}

/// True when `id` names its own e-class as an operand.
///
/// Such a member cannot be selected for that class: the selection would
/// denote "compute X by computing X". A rule bug must degrade the plan, never
/// make a class unextractable.
pub fn is_self_referential(graph: &EGraph, id: Id) -> bool {
    let class = graph.class_of(id);
    graph
        .node(id)
        .children
        .iter()
        .any(|c| graph.class_of(*c) == class)
}

/// A member's fold carrier footprint: accumulator lanes and accumulator bytes.
/// `None` for anything that is not a `Fold`, and for a `Fold` whose slot
/// extent is symbolic — an unallocatable carrier the fold domain generator
/// already declines to score.
pub fn fold_footprint(graph: &EGraph, id: Id) -> Option<(u64, u64)> {
    match &graph.node(id).op {
        Op::Launch(Launch::Fold { carrier, acc, .. }) => Some((carrier.lanes()?, acc.byte_size())),
        _ => None,
    }
}

/// Whether this device can actually run `id` at `theta`.
///
/// Only the fold clause is stated here: it is the only one whose footprint
/// depends on a node property the schedule domain was generated before
/// knowing. `PROMOTE` carries the pre-promotion domain over verbatim, but the
/// inherited strategies were admitted at one accumulator lane and the
/// promoted nest holds `lanes` of them.
pub fn point_is_legal(graph: &EGraph, id: Id, theta: SchedPoint, caps: &Caps) -> bool {
    let Some((lanes, acc_bytes)) = fold_footprint(graph, id) else {
        return true;
    };
    // The same default as [`fold_lane_group`]: a point that is not a fold
    // strategy lowers at the emitters' full block. The two defaults have to
    // be the same value or this predicate admits a node whose tiles the arena
    // then rejects.
    let strat = match theta {
        SchedPoint::Fold(s) => s,
        _ => FoldStrat::WgTree {
            lane_group: fold_lane_group(Some(theta), caps),
        },
    };
    fusor_ir::ir::launch::fold_scratch_bytes(&strat, lanes, acc_bytes, caps.subgroup_width(), caps)
        <= u64::from(caps.limits.max_compute_workgroup_storage_size)
}

/// Whether `id`'s schedule domain offers a point this device can run.
///
/// A node whose whole domain is illegal is unselectable, not merely
/// expensive: a lowering refusal is a hard assert, so selecting one mints a
/// crash rather than a slow plan.
pub fn has_legal_point(graph: &EGraph, id: Id, caps: &Caps) -> bool {
    let Some(domain) = domain_of(graph, id) else {
        return true;
    };
    domain.iter().any(|p| point_is_legal(graph, id, p, caps))
}

pub fn selectable(graph: &EGraph, class: ClassId, caps: &Caps) -> Vec<Id> {
    let members = graph.members(class);
    let acyclic: Vec<Id> = members
        .iter()
        .copied()
        .filter(|m| !is_self_referential(graph, *m))
        .collect();
    let pool = if acyclic.is_empty() { members } else { acyclic };
    let runnable: Vec<Id> = pool
        .iter()
        .copied()
        .filter(|m| is_runnable(graph, *m))
        .collect();
    let pool = if runnable.is_empty() { pool } else { runnable };
    // Schedulability filters last and falls back the same way the two filters
    // above do: a class whose every member is unschedulable is a missing
    // rule, and `verify_plan` names it precisely.
    let schedulable: Vec<Id> = pool
        .iter()
        .copied()
        .filter(|m| has_legal_point(graph, *m, caps))
        .collect();
    if schedulable.is_empty() {
        pool
    } else {
        schedulable
    }
}

/// The classes reachable from `roots`, ascending, plus a node mask covering
/// every id those classes hold — members and `Union` spines both.
///
/// Reachability is the children closure over every member, so it covers
/// everything selection, pricing or realization can touch while excluding the
/// ambient graph a long-lived session accumulates.
///
/// The mask is closed: every child of every masked node resolves to a masked
/// class whose ids are all masked, so a fixpoint over masked slots alone
/// (see `lower_bound_scoped`) equals the whole-graph
/// fixpoint restricted to the mask.
pub fn reachable(graph: &EGraph, roots: &[Id]) -> (Vec<ClassId>, fixedbitset::FixedBitSet) {
    let mut mask = fixedbitset::FixedBitSet::with_capacity(graph.len());
    let mut seen = fixedbitset::FixedBitSet::with_capacity(graph.len());
    let mut out: Vec<ClassId> = Vec::new();
    let mut work: Vec<ClassId> = Vec::new();
    let push = |class: ClassId, seen: &mut fixedbitset::FixedBitSet, work: &mut Vec<ClassId>| {
        if !seen.contains(class.0.index()) {
            seen.insert(class.0.index());
            work.push(class);
        }
    };
    for r in roots {
        push(graph.class_of(*r), &mut seen, &mut work);
    }
    let mut stack: Vec<Id> = Vec::new();
    while let Some(class) = work.pop() {
        out.push(class);
        // Walk the union spine and every member; ids of distinct classes are
        // disjoint, so the mask doubles as this walk's visited set.
        stack.push(class.0);
        while let Some(cur) = stack.pop() {
            if mask.contains(cur.index()) {
                continue;
            }
            mask.insert(cur.index());
            let node = graph.node(cur);
            match &node.op {
                Op::Union(a, b) => {
                    stack.push(*a);
                    stack.push(*b);
                }
                _ => {
                    for ch in node.children.iter() {
                        push(graph.class_of(*ch), &mut seen, &mut work);
                    }
                }
            }
        }
    }
    out.sort_unstable();
    (out, mask)
}

/// Every class in the graph, ascending. Iteration order of every decision
/// path is this, never a hash map's.
pub fn classes(graph: &EGraph) -> Vec<ClassId> {
    let mut out: Vec<ClassId> = Vec::new();
    let mut seen = fixedbitset::FixedBitSet::with_capacity(graph.len());
    for i in 0..graph.len() {
        let class = graph.class_of(Id(i as u32));
        if !seen.contains(class.0.index()) {
            seen.insert(class.0.index());
            out.push(class);
        }
    }
    out.sort_unstable();
    out
}

/// The schedule domain a launch node carries.
pub fn domain_of(graph: &EGraph, id: Id) -> Option<&ScheduleDomain> {
    match &graph.node(id).op {
        Op::Launch(l1) => l1.schedule(),
        _ => None,
    }
}
