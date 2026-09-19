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
use fusor_ir::ir::launch::{
    Effect, FoldStrat, IndexSpace, Launch, SchedPoint, ScheduleDomain, slab_lanes_per_row,
    slab_subgroup_width,
};
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
    pub line_bytes: u64,
    pub coop_steps: u64,
    pub lane_steps: u64,
    /// Slab members that live in workgroup memory: read by nothing outside
    /// the slab, and within the budget the device leaves after the fold
    /// scratch. No buffer, no binding, no traffic.
    pub private: Vec<Id>,
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
                line_bytes: c.line_bytes,
                coop_steps: c.coop_steps,
                lane_steps: c.lane_steps,
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
    let (launch_of, groups) =
        cut(graph, extraction, &order, &operands, &resolved_roots).map_err(Error::from)?;
    let components = groups
        .into_iter()
        .map(|members| {
            build_component(
                graph,
                extraction,
                &consumers,
                &consumer_nodes,
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
    // A slab's members run as its stages, materialized or not: that is the
    // one edge a buffer does not cut.
    if slab_stage(graph, consumer, producer).is_some() {
        return false;
    }
    if extraction.is_materialized(producer) || roots.contains(&producer) {
        return true;
    }
    if graph.semantics().effect(&graph.node(producer).op) != Effect::Pure {
        return true;
    }
    structural_boundary(graph, producer, consumer)
}

/// `Some(is_last)` when `producer` is a member of the slab `consumer`.
pub fn slab_stage(graph: &EGraph, consumer: Id, producer: Id) -> Option<bool> {
    let Op::Launch(Launch::Slab { members, .. } | Launch::Group { members, .. }) =
        &graph.node(consumer).op
    else {
        return None;
    };
    let pos = members.iter().position(|m| *m == producer)?;
    Some(pos + 1 == members.len())
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
    // Every stage but the last lands in its own buffer, where the stages
    // after it read it; the last stage lands in the slab's.
    if let Some(last) = slab_stage(graph, consumer, producer) {
        return !last;
    }
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
        Op::Launch(Launch::Contract { output, .. }) => output.clone(),
        Op::Launch(Launch::Slab { members, .. } | Launch::Group { members, .. }) => members
            .last()
            .map(|m| index_space(graph, *m))
            .unwrap_or_default(),
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
        Some(SchedPoint::Sgemv(p))
            // The subgroup-per-column structure (`cols > 1`) closes each
            // column inside one subgroup and stages nothing.
            if p.cols <= 1 => {
                decls.push(tile("sgemv_partials", elem, &[p.subgroups.max(1)]));
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

/// Bytes a cache line holds on every device this targets; the amplification
/// is a ratio against it, so the exact figure matters less than having one.
const LINE_BYTES: u64 = 128;

/// How many times its useful bytes a fold's operand read moves through the
/// memory pipe at `lane_group` lanes per row. A subgroup's lanes cover
/// `contig` consecutive elements per load — the lane group's share of a row
/// when the reduced axis is innermost, the adjacent rows the subgroup serves
/// otherwise — and each such run costs whole lines.
pub fn fold_line_amplification(
    dims: &[u64],
    axis: usize,
    lane_group: u32,
    caps: &Caps,
    elem_bytes: u64,
) -> u64 {
    let Some(&k) = dims.get(axis) else {
        return 1;
    };
    let inner: u64 = dims[axis + 1..].iter().product::<u64>().max(1);
    let sg = u64::from(caps.subgroup_width().max(1));
    let lg = u64::from(lane_group.max(1)).min(sg);
    let line_elems = (LINE_BYTES / elem_bytes.max(1)).max(1);
    let contig = if inner == 1 {
        lg.min(k.max(1))
    } else {
        (sg / lg).min(inner)
    }
    .max(1);
    let runs = sg / contig.min(sg);
    let lines = runs.max(1) * contig.div_ceil(line_elems);
    (lines * line_elems / sg).clamp(1, line_elems)
}

/// The longest dependent chain one workgroup of `root` runs at `theta`: for
/// a tiled contraction the k steps of one tile (split-K divides them), for
/// a fold the iterations of one lane over the reduced axis, for a slab the
/// sum over its stages. What no occupancy shortens.
pub fn serial_steps(
    graph: &EGraph,
    root: Id,
    theta: Option<SchedPoint>,
    block: u32,
    caps: &Caps,
) -> (u64, u64) {
    match &graph.node(root).op {
        Op::Launch(Launch::Slab { slabs, members, .. }) => {
            let slabs = u64::from((*slabs).max(1));
            let mut steps = 0u64;
            for m in members.iter() {
                match &graph.node(*m).op {
                    Op::Launch(Launch::Fold {
                        space,
                        axis,
                        carrier,
                        ..
                    }) => {
                        let Some(total) = space.iterations() else {
                            continue;
                        };
                        let k = space
                            .dims
                            .get(*axis as usize)
                            .and_then(|d| d.as_const())
                            .unwrap_or(1)
                            .max(1);
                        let rows = (total / k) / slabs;
                        let lpr = slab_subgroup_width(block, rows, k, carrier, caps)
                            .unwrap_or_else(|| slab_lanes_per_row(block, rows, k));
                        let groups = u64::from(block / lpr.max(1)).max(1);
                        steps += rows.div_ceil(groups).max(1) * k.div_ceil(u64::from(lpr));
                    }
                    Op::Launch(Launch::Map { space, .. }) => {
                        let Some(total) = space.iterations() else {
                            continue;
                        };
                        steps += (total / slabs).div_ceil(u64::from(block)).max(1);
                    }
                    _ => {}
                }
            }
            (0, steps)
        }
        op => node_serial_steps(op, theta, caps),
    }
}

/// [`serial_steps`] for one launch node from its own op: a contraction's k
/// steps at `theta`, a fold's iterations per lane.
pub fn node_serial_steps(op: &Op, theta: Option<SchedPoint>, caps: &Caps) -> (u64, u64) {
    match op {
        Op::Launch(Launch::Contract { k, .. }) => {
            let k = k.as_const().unwrap_or(1).max(1);
            // A step is one fragment depth of k, whatever `bk` stages at
            // once: a deeper tile runs its fragments back to back, so the
            // chain is as long either way.
            let depth = u64::from(fusor_ir::ir::launch::CoopGeom::COOP_DIM.max(1));
            // Every family walks k in dependent steps; a GEMV lane's dot
            // and a subgroup's fragment chain are the same length in
            // depths. The floor separates split from unsplit, and tiled
            // from scalar folds — not one family from another.
            match theta {
                // One subgroup multiplies its `(bm / rg) x (bn / cg)` block a
                // fragment at a time, every depth: that chain is the step
                // count. `16x16` on one subgroup and `32x32` on four are the
                // same chain; what separates them is traffic.
                // A depth step is a staged load and a barrier; `16x16` on
                // one subgroup (four multiplies per depth) measured the same
                // as on four (one each), so the multiplies are not the step.
                Some(SchedPoint::Coop { .. }) => (k.div_ceil(depth), 0),
                // A scalar-tiled lane walks every k with its register tile's
                // FMAs and staged loads: measured 8-10x a fragment chain on
                // the same shape (281 us against 33 us at 1024x96x96).
                Some(SchedPoint::Sgemm(_)) => (0, k.saturating_mul(4)),
                Some(SchedPoint::Sgemv(_)) => (k.div_ceil(depth) * 4, 0),
                _ => (0, k),
            }
        }
        Op::Launch(Launch::Fold { space, axis, .. }) => {
            let k = space
                .dims
                .get(*axis as usize)
                .and_then(|d| d.as_const())
                .unwrap_or(1);
            (
                0,
                k.div_ceil(u64::from(fold_lane_group(theta, caps).max(1))),
            )
        }
        // The dense scatter walks every update once per output lane, each
        // step a dependent index load.
        Op::Launch(Launch::Scatter { ops, .. }) => {
            let updates = ops
                .get(1)
                .map(|o| {
                    o.layout
                        .shape()
                        .iter()
                        .map(|d| d.as_const().unwrap_or(1))
                        .product::<u64>()
                })
                .unwrap_or(1);
            (0, updates.max(1))
        }
        _ => (0, 0),
    }
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
        Some(SchedPoint::Coop { geom, .. }) => Geometry {
            block: geom.lanes(width).max(1),
            workgroups: m.div_ceil(geom.bm.max(1) as u64)
                * n.div_ceil(geom.bn.max(1) as u64)
                * batch,
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
        // A fold workgroup has `emitted_block(lane_group)` lanes and computes
        // `block / lane_group` output rows.
        Some(SchedPoint::Fold(strat)) => {
            let (block, lane_group) = match strat {
                FoldStrat::Subgroup => (
                    width.min(caps.limits.max_compute_invocations_per_workgroup.max(1)),
                    width,
                ),
                FoldStrat::WgTree { lane_group } | FoldStrat::LoopThenTree { lane_group, .. } => {
                    let lg = lane_group.max(1);
                    (fusor_ir::ir::launch::emitted_block(lg, caps), lg)
                }
            };
            let rows = (total / n.max(1)).max(1);
            let per_group = u64::from((block / lane_group.max(1)).max(1));
            Geometry {
                block: block.max(1),
                workgroups: rows.div_ceil(per_group).max(1),
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

/// Why the selected nodes or their launch components cannot be ordered.
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
            WalkFail::Cycle(v) => Error::Plan(format!("selection is cyclic through {v}")),
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
/// (the depth-1 case) sees nothing. Composite ownership can also turn an
/// acyclic node order into cyclic launch dependencies.
pub fn selection_cycle(graph: &EGraph, extraction: &Extraction, roots: &[Id]) -> Option<Id> {
    let resolved = roots
        .iter()
        .map(|r| select(graph, extraction, *r))
        .collect::<Result<Vec<_>>>()
        .ok()?;
    let attempt = walk(graph, extraction, &resolved).and_then(|(order, operands)| {
        let mut seed = extraction.clone();
        seed.m = materializations(graph, &resolved, &order, &operands);
        cut(graph, &seed, &order, &operands, &resolved)
    });
    match attempt {
        Err(WalkFail::Cycle(v)) => Some(v),
        _ => None,
    }
}

pub(crate) fn seed_materializations(
    graph: &EGraph,
    extraction: &Extraction,
    roots: &[Id],
) -> Result<fixedbitset::FixedBitSet> {
    let roots = roots
        .iter()
        .map(|r| select(graph, extraction, *r))
        .collect::<Result<Vec<_>>>()?;
    let (order, operands) = walk(graph, extraction, &roots).map_err(Error::from)?;
    Ok(materializations(graph, &roots, &order, &operands))
}

pub(crate) fn selected_order(
    graph: &EGraph,
    extraction: &Extraction,
    roots: &[Id],
) -> Result<Vec<Id>> {
    let roots = roots
        .iter()
        .map(|r| select(graph, extraction, *r))
        .collect::<Result<Vec<_>>>()?;
    walk(graph, extraction, &roots)
        .map(|(order, _)| order)
        .map_err(Error::from)
}

fn materializations(
    graph: &EGraph,
    roots: &[Id],
    order: &[Id],
    operands: &Operands,
) -> fixedbitset::FixedBitSet {
    let (consumers, readers) = count_consumers(graph.len(), order, operands, roots);
    let mut materialized = fixedbitset::FixedBitSet::with_capacity(graph.len());
    for &id in order {
        if leaf_role(graph, id) == LeafRole::NotLeaf
            && (roots.contains(&id)
                || graph.semantics().effect(&graph.node(id).op) != Effect::Pure
                || consumers.copied(id).unwrap_or(0) > 1
                || readers
                    .get(id)
                    .is_some_and(|cs| cs.iter().any(|c| needs_own_buffer(graph, id, *c))))
        {
            materialized.insert(id.index());
        }
    }
    materialized
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
                    // A slab names its members by id — they are the spellings
                    // its lowering reads — and its last member shares its
                    // class, so selecting them would walk back into the slab.
                    let by_id = matches!(
                        graph.node(v).op,
                        Op::Launch(Launch::Slab { .. } | Launch::Group { .. })
                    );
                    let kids: SmallVec<[Id; 4]> = graph
                        .node(v)
                        .children
                        .iter()
                        .map(|c| {
                            if by_id {
                                Ok(*c)
                            } else {
                                select(graph, extraction, *c)
                            }
                        })
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
) -> std::result::Result<(IdMap<u32>, Vec<Vec<Id>>), WalkFail> {
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

    // Groups came out in the order their *first* node appears, which is not
    // a dependency order: a composite whose first member reads nothing may
    // have a later member that reads a launch appearing after that first
    // node. Order the groups as a DAG instead, earliest-first among the
    // ready ones so the order stays deterministic.
    let n = groups.len();
    let mut deps: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut indegree = vec![0usize; n];
    for (g, members) in groups.iter().enumerate() {
        let mut seen: SmallVec<[usize; 8]> = SmallVec::new();
        for v in members {
            for c in operands.get(*v).map(|o| o.as_slice()).unwrap_or(&[]) {
                let Some(d) = launch_of.copied(*c) else {
                    continue;
                };
                let d = d as usize;
                if d == g || seen.contains(&d) {
                    continue;
                }
                seen.push(d);
                deps[d].push(g);
                indegree[g] += 1;
            }
        }
    }
    let mut ready: std::collections::BinaryHeap<std::cmp::Reverse<usize>> = (0..n)
        .filter(|g| indegree[*g] == 0)
        .map(std::cmp::Reverse)
        .collect();
    let mut sorted: Vec<usize> = Vec::with_capacity(n);
    while let Some(std::cmp::Reverse(g)) = ready.pop() {
        sorted.push(g);
        for &h in &deps[g] {
            indegree[h] -= 1;
            if indegree[h] == 0 {
                ready.push(std::cmp::Reverse(h));
            }
        }
    }
    if sorted.len() != n {
        let mut cycle = Vec::new();
        let mut g = (0..n).find(|g| indegree[*g] > 0).unwrap();
        loop {
            if let Some(start) = cycle.iter().position(|previous| *previous == g) {
                for &g in &cycle[start..] {
                    for &id in &groups[g] {
                        if matches!(
                            graph.node(id).op,
                            Op::Launch(Launch::Slab { .. } | Launch::Group { .. })
                        ) && extraction.selected(graph.class_of(id)) == Some(id)
                        {
                            return Err(WalkFail::Cycle(id));
                        }
                    }
                }
                return Err(WalkFail::Other(Error::Plan(
                    "selected fusion creates cyclic launch dependencies".into(),
                )));
            }
            cycle.push(g);
            g = (0..n)
                .find(|d| indegree[*d] > 0 && deps[*d].contains(&g))
                .unwrap();
        }
    }
    let mut renumber = vec![0u32; n];
    for (new, old) in sorted.iter().enumerate() {
        renumber[*old] = new as u32;
    }
    let mut reordered: Vec<Vec<Id>> = Vec::with_capacity(n);
    for old in &sorted {
        reordered.push(std::mem::take(&mut groups[*old]));
    }
    for members in &reordered {
        for v in members {
            let old = launch_of.copied(*v).unwrap_or(0);
            launch_of.insert(*v, renumber[old as usize]);
        }
    }
    Ok((launch_of, reordered))
}

#[allow(clippy::too_many_arguments)]
fn build_component(
    graph: &EGraph,
    extraction: &Extraction,
    consumers: &IdMap<u32>,
    consumer_nodes: &IdMap<SmallVec<[Id; 4]>>,
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
    if std::env::var_os("FUSOR_SLAB_LOG").is_some()
        && let Some(slab) = members
            .iter()
            .find(|m| matches!(graph.node(**m).op, Op::Launch(Launch::Slab { .. })))
    {
        let Op::Launch(Launch::Slab { members: sm, .. }) = &graph.node(*slab).op else {
            unreachable!()
        };
        let extra: Vec<Id> = members
            .iter()
            .copied()
            .filter(|m| m != slab && !sm.contains(m))
            .collect();
        if !extra.is_empty() {
            let kinds: Vec<String> = extra
                .iter()
                .map(|e| {
                    format!(
                        "{e}:{:?}:mat={}",
                        graph.node(*e).op.tag(),
                        extraction.is_materialized(*e)
                    )
                })
                .collect();
            eprintln!(
                "COMPONENT slab {slab} has {} extra nodes: {kinds:?}",
                extra.len()
            );
        }
    }

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
        let by_id = matches!(
            graph.node(*m).op,
            Op::Launch(Launch::Slab { .. } | Launch::Group { .. })
        );
        for c in graph.node(*m).children.iter() {
            let c = if by_id {
                *c
            } else {
                select(graph, extraction, *c)?
            };
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
    // A group's members each take their own workgroups at their own block;
    // the dispatch is their sum at the widest block.
    let group_geoms: Vec<(Id, Geometry)> = match &graph.node(root).op {
        Op::Launch(Launch::Group { members: gm, .. }) => gm
            .iter()
            .map(|m| (*m, member_geometry(graph, extraction, *m, caps)))
            .collect(),
        _ => Vec::new(),
    };
    let geom = match &graph.node(root).op {
        Op::Launch(Launch::Group { .. }) => Geometry {
            block: group_geoms.iter().map(|(_, g)| g.block).max().unwrap_or(1),
            workgroups: group_geoms
                .iter()
                .map(|(_, g)| {
                    let d = distribute_workgroups(
                        g.workgroups,
                        caps.limits.max_compute_workgroups_per_dimension,
                    );
                    u64::from(d[0]) * u64::from(d[1]) * u64::from(d[2])
                })
                .sum::<u64>()
                .max(1),
        },
        // One workgroup per slab, at the block the widest stage's share of
        // one slab asks for — the emitter's own arithmetic.
        Op::Launch(Launch::Slab { slabs, members, .. }) => {
            let widest = members
                .iter()
                .filter_map(|m| index_space(graph, *m).iterations())
                .map(|n| n / u64::from((*slabs).max(1)))
                .max()
                .unwrap_or(1);
            Geometry {
                block: fusor_ir::ir::launch::slab_block(widest, caps),
                workgroups: u64::from(*slabs).max(1),
            }
        }
        _ => geometry(theta, &space, caps),
    };
    let lanes = fold_footprint(graph, root).map(|(l, _)| l);
    let tiles = tiles_for(theta, scalar_element(graph.facts(root).dtype), lanes, caps);
    let mut wg_bytes = arena.workgroup_bytes(&tiles, caps)? as u64;

    // Uncoalesced fold reads: every operand walked at the fold's iteration
    // space pays the line amplification of its lane group. A slab pays it
    // per fold stage at that stage's lanes per row.
    let mut line_bytes = 0u64;
    let amp_of = |m: Id, lane_group: u32| -> (u64, u64) {
        let Op::Launch(Launch::Fold { space, axis, .. }) = &graph.node(m).op else {
            return (1, 0);
        };
        let dims: Vec<u64> = space.dims.iter().filter_map(|d| d.as_const()).collect();
        if dims.len() != space.rank() {
            return (1, 0);
        }
        let elem = graph.facts(m).dtype.byte_size().max(1);
        (
            fold_line_amplification(&dims, *axis as usize, lane_group, caps, elem),
            dims.iter().product(),
        )
    };
    let stage_amp: Vec<(Id, u64, u64)> = match &graph.node(root).op {
        Op::Launch(Launch::Fold { .. }) => vec![{
            let (a, n) = amp_of(root, fold_lane_group(theta, caps));
            (root, a, n)
        }],
        Op::Launch(Launch::Slab {
            slabs, members: sm, ..
        }) => sm
            .iter()
            .filter_map(|m| {
                let Op::Launch(Launch::Fold {
                    space,
                    axis,
                    carrier,
                    ..
                }) = &graph.node(*m).op
                else {
                    return None;
                };
                let total = space.iterations()?;
                let k = space.dims.get(*axis as usize)?.as_const()?.max(1);
                let rows = (total / k) / u64::from((*slabs).max(1));
                let lpr = slab_subgroup_width(geom.block, rows, k, carrier, caps)
                    .unwrap_or_else(|| slab_lanes_per_row(geom.block, rows, k));
                let (a, n) = amp_of(*m, lpr);
                Some((*m, a, n))
            })
            .collect(),
        _ => Vec::new(),
    };
    // A tiled contraction pulls each operand once per tile on the other
    // side: `M*N*K*(1/bn + 1/bm)` elements through the memory pipe, of which
    // `M*K + K*N` are the operands themselves.
    if let Op::Launch(Launch::Contract { m, n, k, batch, .. }) = &graph.node(root).op
        && let Some((bm, bn)) = match theta {
            Some(SchedPoint::Coop { geom, .. }) => Some((u64::from(geom.bm), u64::from(geom.bn))),
            Some(SchedPoint::Sgemm(p)) => Some((u64::from(p.bm), u64::from(p.bn))),
            _ => None,
        }
    {
        let (m, n, k, batch) = (
            m.as_const().unwrap_or(1).max(1),
            n.as_const().unwrap_or(1).max(1),
            k.as_const().unwrap_or(1).max(1),
            batch.as_const().unwrap_or(1).max(1),
        );
        let elem = graph.facts(root).dtype.byte_size().max(1);
        let pulled = batch * k * (m * n.div_ceil(bn.max(1)) + n * m.div_ceil(bm.max(1)));
        let useful = batch * k * (m + n);
        line_bytes = line_bytes.saturating_add(pulled.saturating_sub(useful).saturating_mul(elem));
    }

    if !stage_amp.is_empty() {
        for m in &members {
            let by_id = matches!(
                graph.node(*m).op,
                Op::Launch(Launch::Slab { .. } | Launch::Group { .. })
            );
            // The fold whose iteration space walks this member's operands:
            // the member itself when it is a stage, else the root.
            let (amp, total) = stage_amp
                .iter()
                .find(|(s, _, _)| s == m)
                .or_else(|| stage_amp.first())
                .map_or((1, 0), |(_, a, n)| (*a, *n));
            if amp <= 1 {
                continue;
            }
            for c in graph.node(*m).children.iter() {
                let c = if by_id {
                    *c
                } else {
                    select(graph, extraction, *c)?
                };
                let facts = graph.facts(c);
                if launch_of.copied(c) == Some(own) || elements_of(facts) != total {
                    continue;
                }
                line_bytes = line_bytes.saturating_add(bytes_of(facts).saturating_mul(amp - 1));
            }
        }
    }

    let (coop_steps, lane_steps) = if group_geoms.is_empty() {
        serial_steps(graph, root, theta, geom.block, caps)
    } else {
        // Members run side by side: the chain is the longest of theirs.
        group_geoms
            .iter()
            .map(|(m, g)| serial_steps(graph, *m, extraction.theta.get(m).copied(), g.block, caps))
            .fold((0, 0), |a, b| (a.0.max(b.0), a.1.max(b.1)))
    };

    // A slab's middle members nothing outside reads live in workgroup memory
    // as far as it fits; the rest, and anything read outside, in buffers.
    let mut private: Vec<Id> = Vec::new();
    // A group's member slabs keep their own privates; the group itself has
    // none.
    let slab_roots: Vec<Id> = match &graph.node(root).op {
        Op::Launch(Launch::Slab { .. }) => vec![root],
        Op::Launch(Launch::Group { members: gm, .. }) => gm
            .iter()
            .copied()
            .filter(|m| matches!(graph.node(*m).op, Op::Launch(Launch::Slab { .. })))
            .collect(),
        _ => Vec::new(),
    };
    for slab in slab_roots {
        let Op::Launch(Launch::Slab { members: sm, .. }) = &graph.node(slab).op else {
            continue;
        };
        let inside: rustc_hash::FxHashSet<ClassId> =
            sm.iter().map(|m| graph.class_of(*m)).collect();
        let shared = |m: Id| {
            consumer_nodes.get(m).is_some_and(|cs| {
                cs.iter()
                    .any(|c| c != &slab && c != &root && !inside.contains(&graph.class_of(*c)))
            })
        };
        let (p, used) = slab_layout(graph, slab, caps, roots, &shared)?;
        for m in &p {
            writes = writes.saturating_sub(bytes_of(graph.facts(*m)));
        }
        wg_bytes = wg_bytes.max(used);
        private.extend(p);
    }

    Ok(Component {
        root,
        members,
        reads: ext.iter().map(|(_, b, r)| (*b, *r)).collect(),
        external: ext.iter().map(|(id, _, _)| *id).collect(),
        writes,
        work,
        resident_lanes: geom.workgroups.saturating_mul(geom.block as u64),
        wg_bytes,
        line_bytes,
        coop_steps,
        lane_steps,
        private,
        grid: distribute_workgroups(
            geom.workgroups,
            caps.limits.max_compute_workgroups_per_dimension,
        ),
        block: geom.block,
    })
}

/// The geometry one node launches at on its own: a slab's, or its schedule
/// point's over its index space.
fn member_geometry(graph: &EGraph, extraction: &Extraction, m: Id, caps: &Caps) -> Geometry {
    match &graph.node(m).op {
        Op::Launch(Launch::Slab { slabs, members, .. }) => {
            let widest = members
                .iter()
                .filter_map(|s| index_space(graph, *s).iterations())
                .map(|n| n / u64::from((*slabs).max(1)))
                .max()
                .unwrap_or(1);
            Geometry {
                block: fusor_ir::ir::launch::slab_block(widest, caps),
                workgroups: u64::from(*slabs).max(1),
            }
        }
        _ => geometry(
            extraction.theta.get(&m).copied(),
            &index_space(graph, m),
            caps,
        ),
    }
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
    // A slab's last member is in the slab's own class by construction, and is
    // read by id rather than selected; that is not a cycle.
    if matches!(
        graph.node(id).op,
        Op::Launch(Launch::Slab { .. } | Launch::Group { .. })
    ) {
        return false;
    }
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

/// Whether a composite can materialize its externally visible values in
/// this graph. This depends on consumers and buffer choices, not kernel
/// correctness or the contents of a schedule domain.
pub fn composite_bindings_fit(graph: &EGraph, id: Id, caps: &Caps) -> bool {
    if matches!(graph.node(id).op, Op::Launch(Launch::Slab { .. }))
        && !slab_bindings_fit(graph, id, caps)
    {
        return false;
    }
    if matches!(graph.node(id).op, Op::Launch(Launch::Group { .. }))
        && !group_bindings_fit(graph, id, caps)
    {
        return false;
    }
    true
}

/// Whether group `id` can bind: each member's own buffers and the distinct
/// outside inputs, a member slab's stages as its layout says. Memoized
/// like [`slab_bindings_fit`].
pub fn group_bindings_fit(graph: &EGraph, id: Id, caps: &Caps) -> bool {
    thread_local! {
        static MEMO: std::cell::RefCell<rustc_hash::FxHashMap<(u64, Id), (usize, bool)>> =
            std::cell::RefCell::new(rustc_hash::FxHashMap::default());
    }
    let key = (graph.arena_id(), id);
    if let Some((len, fit)) = MEMO.with(|m| m.borrow().get(&key).copied())
        && len == graph.len()
    {
        return fit;
    }
    let fit = group_bindings_fit_uncached(graph, id, caps);
    MEMO.with(|m| m.borrow_mut().insert(key, (graph.len(), fit)));
    fit
}

fn group_bindings_fit_uncached(graph: &EGraph, id: Id, caps: &Caps) -> bool {
    if let Op::Launch(Launch::Group { members, .. }) = &graph.node(id).op {
        let mut inputs: rustc_hash::FxHashSet<ClassId> = rustc_hash::FxHashSet::default();
        let mut outs = 0usize;
        for m in members.iter() {
            match &graph.node(*m).op {
                Op::Launch(Launch::Slab { members: sm, .. }) => {
                    if !slab_bindings_fit(graph, *m, caps) {
                        return false;
                    }
                    let own: rustc_hash::FxHashSet<ClassId> =
                        sm.iter().map(|s| graph.class_of(*s)).collect();
                    for s in sm.iter() {
                        for c in graph.node(*s).children.iter() {
                            let class = graph.class_of(*c);
                            if !own.contains(&class) {
                                inputs.insert(class);
                            }
                        }
                    }
                    let shared = |x: Id| {
                        graph.any_reader(graph.class_of(x), |r| !own.contains(&graph.class_of(r)))
                    };
                    let Ok((private, _)) = slab_layout(graph, *m, caps, graph.roots(), &shared)
                    else {
                        return false;
                    };
                    outs += sm.len() - private.len();
                }
                _ => {
                    outs += 1;
                    for c in graph.node(*m).children.iter() {
                        inputs.insert(graph.class_of(*c));
                    }
                }
            }
        }
        let own: rustc_hash::FxHashSet<ClassId> =
            members.iter().map(|m| graph.class_of(*m)).collect();
        let root_classes: rustc_hash::FxHashSet<ClassId> =
            graph.roots().iter().map(|r| graph.class_of(*r)).collect();
        let inputs = inputs
            .iter()
            .filter(|c| !own.contains(c) && own_buffer(graph, **c, &root_classes))
            .count();
        let _ = outs;
        let outs = members
            .iter()
            .filter(|m| root_classes.contains(&graph.class_of(**m)))
            .count();
        if 2 + outs + inputs > caps.limits.max_storage_buffers_per_shader_stage as usize {
            return false;
        }
    }
    true
}

/// Whether slab `id` can bind, with every middle member nothing outside the
/// slab reads kept in workgroup memory as far as it fits. The readers index
/// says what is read outside; `build_component` decides the same layout
/// from the realized consumers, which read no more than that.
pub fn slab_bindings_fit(graph: &EGraph, id: Id, caps: &Caps) -> bool {
    let Op::Launch(Launch::Slab { members, .. }) = &graph.node(id).op else {
        return true;
    };
    // Readers only grow with the graph, so the answer is a function of
    // `(graph, node, graph.len())`; the extractor asks per class per move.
    thread_local! {
        static MEMO: std::cell::RefCell<rustc_hash::FxHashMap<(u64, Id), (usize, bool)>> =
            std::cell::RefCell::new(rustc_hash::FxHashMap::default());
    }
    let key = (graph.arena_id(), id);
    if let Some((len, fit)) = MEMO.with(|m| m.borrow().get(&key).copied())
        && len == graph.len()
    {
        return fit;
    }
    let classes: rustc_hash::FxHashSet<ClassId> =
        members.iter().map(|m| graph.class_of(*m)).collect();
    let shared =
        |m: Id| graph.any_reader(graph.class_of(m), |r| !classes.contains(&graph.class_of(r)));
    // The graph's roots are every value a caller may read back: a root
    // member lands in a buffer whichever roots this extraction has.
    let fit = slab_layout(graph, id, caps, graph.roots(), &shared).is_ok();
    MEMO.with(|m| m.borrow_mut().insert(key, (graph.len(), fit)));
    fit
}

/// Whether a class binds its own storage buffer: an external leaf, or a
/// root the caller reads back. Every other value is packed into the step
/// arena, which a launch binds once.
pub fn own_buffer(graph: &EGraph, class: ClassId, roots: &rustc_hash::FxHashSet<ClassId>) -> bool {
    roots.contains(&class)
        || graph
            .members(class)
            .iter()
            .any(|m| leaf_role(graph, *m) == LeafRole::External)
}

/// A slab's workgroup memory and bindings: the widest fold stage's scratch,
/// then as many middle members as fit — smallest share first, of those
/// `shared` says nothing outside reads and that are not roots — and the
/// storage buffers left over: the uniform block, the output, every distinct
/// outside class read, and every middle member still in a buffer. `Err`
/// when those exceed the device's bindings.
pub fn slab_layout(
    graph: &EGraph,
    root: Id,
    caps: &Caps,
    roots: &[Id],
    shared: &dyn Fn(Id) -> bool,
) -> Result<(Vec<Id>, u64)> {
    let Op::Launch(Launch::Slab {
        slabs, members: sm, ..
    }) = &graph.node(root).op
    else {
        return Ok((Vec::new(), 0));
    };
    let slabs = u64::from((*slabs).max(1));
    let widest = sm
        .iter()
        .filter_map(|m| index_space(graph, *m).iterations())
        .map(|n| n / slabs)
        .max()
        .unwrap_or(1);
    let block = u64::from(fusor_ir::ir::launch::slab_block(widest, caps));
    let limit = u64::from(caps.limits.max_compute_workgroup_storage_size);
    let scratch = sm
        .iter()
        .filter_map(|m| fold_footprint(graph, *m))
        .map(|(lanes, acc)| lanes.saturating_mul(acc).saturating_mul(block))
        .max()
        .unwrap_or(0);
    let mut used = scratch;
    let middle = &sm[..sm.len().saturating_sub(1)];
    let root_classes: rustc_hash::FxHashSet<ClassId> =
        roots.iter().map(|r| graph.class_of(*r)).collect();
    let mut unshared: Vec<(u64, Id)> = middle
        .iter()
        .filter(|m| !root_classes.contains(&graph.class_of(**m)) && !shared(**m))
        .map(|m| (bytes_of(graph.facts(*m)) / slabs, *m))
        .collect();
    unshared.sort_unstable();
    let mut private: Vec<Id> = Vec::new();
    // `FUSOR_NO_PRIVATE`: every member in a buffer, for bisecting.
    if std::env::var_os("FUSOR_NO_PRIVATE").is_some() {
        unshared.clear();
    }
    for (share, m) in unshared {
        if used.saturating_add(share) > limit {
            continue;
        }
        used += share;
        private.push(m);
    }
    // Stage order, so the lowering declares tiles in the order it runs.
    private.sort_unstable_by_key(|m| sm.iter().position(|x| x == m));

    let classes: rustc_hash::FxHashSet<ClassId> = sm.iter().map(|m| graph.class_of(*m)).collect();
    let mut inputs: rustc_hash::FxHashSet<ClassId> = rustc_hash::FxHashSet::default();
    for m in sm.iter() {
        for c in graph.node(*m).children.iter() {
            let class = graph.class_of(*c);
            if !classes.contains(&class) && leaf_role(graph, *c) != LeafRole::Free {
                inputs.insert(class);
            }
        }
    }
    // The uniform block, the step arena, the output when the caller reads
    // it back, and every input or buffered middle member that is a leaf or
    // a root: everything else lives in the arena binding.
    let owns = |c: ClassId| own_buffer(graph, c, &root_classes);
    let bound = 2
        + inputs.iter().filter(|c| owns(**c)).count()
        + middle
            .iter()
            .filter(|m| !private.contains(m) && owns(graph.class_of(**m)))
            .count()
        + usize::from(owns(graph.class_of(root)));
    let limit_bufs = caps.limits.max_storage_buffers_per_shader_stage as usize;
    if bound > limit_bufs && std::env::var_os("FUSOR_SLAB_LOG").is_some() {
        eprintln!(
            "LAYOUT slab {root}: {bound} bindings > {limit_bufs}: {} inputs, {} middle, {} private, wg {used}/{limit}",
            inputs.len(),
            middle.len(),
            private.len()
        );
    }
    if bound > limit_bufs {
        return Err(Error::Plan(format!(
            "slab {root} binds {bound} storage buffers over the {limit_bufs}-buffer limit: \
             its middle members do not fit workgroup memory"
        )));
    }
    Ok((private, used))
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
    // Account for composites whose externally visible members need buffers.
    let schedulable: Vec<Id> = pool
        .iter()
        .copied()
        .filter(|m| composite_bindings_fit(graph, *m, caps))
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
