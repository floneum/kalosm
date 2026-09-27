//! Resolve selected nodes, derive their buffers, and order their dispatches.

use crate::nodes::{
    Mnkb, Tiling, composite_members, domain_of, fold_lane_group, fold_theta, is_composite,
    is_group, resolved_children, sgemv_block, sgemv_lanes,
};
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
    AccessPlan, FoldStrat, IndexSpace, Launch, SchedPoint, ScheduleDomain, slab_lanes_per_row,
    slab_subgroup_width,
};
use fusor_ir::ir::logical::{LeafKind, Logical};
use fusor_ir::shape::Dim;
use smallvec::SmallVec;
use std::sync::Arc;

/// Extent a `Dim::Sym` prices at, so no concrete binding leaks into a plan.
pub(crate) const SYM_NOMINAL: u64 = 1024;

/// What role a leaf plays in the realized DAG.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum LeafRole {
    /// Not a leaf; an ordinary launch member.
    NotLeaf,
    /// A constant or a uniform scalar: no buffer, no traffic, no component.
    Free,
    /// An externally supplied buffer: read traffic, never a `BufferPlan`.
    External,
}

/// Indexed node values, with slot storage recycled between candidate walks.
#[derive(Clone, Debug, Default)]
pub(crate) struct IdMap<T> {
    slots: Vec<u32>,
    values: Vec<(Id, T)>,
}

impl<T> IdMap<T> {
    fn with_len(n: usize, pool: &mut Vec<Vec<u32>>) -> Self {
        let mut slots = pool.pop().unwrap_or_default();
        if slots.len() < n {
            slots.resize(n, u32::MAX);
        }
        Self {
            slots,
            values: Vec::new(),
        }
    }

    #[inline]
    pub(crate) fn get(&self, id: Id) -> Option<&T> {
        self.values
            .get(*self.slots.get(id.index())? as usize)
            .map(|(_, value)| value)
    }

    #[inline]
    pub(crate) fn contains(&self, id: Id) -> bool {
        self.get(id).is_some()
    }

    #[inline]
    pub(crate) fn insert(&mut self, id: Id, value: T) {
        if self.slots.len() <= id.index() {
            self.slots.resize(id.index() + 1, u32::MAX);
        }
        let slot = &mut self.slots[id.index()];
        if *slot == u32::MAX {
            *slot = u32::try_from(self.values.len()).expect("node table exceeds Id capacity");
            assert_ne!(*slot, u32::MAX, "node table exceeds Id capacity");
            self.values.push((id, value));
        } else {
            self.values[*slot as usize].1 = value;
        }
    }

    #[inline]
    pub(crate) fn entry_or_default(&mut self, id: Id) -> &mut T
    where
        T: Default,
    {
        if !self.contains(id) {
            self.insert(id, T::default());
        }
        &mut self.values[self.slots[id.index()] as usize].1
    }

    fn recycle(mut self, pool: &mut Vec<Vec<u32>>) {
        for (id, _) in &self.values {
            self.slots[id.index()] = u32::MAX;
        }
        pool.push(self.slots);
    }
}

impl<T: Copy> IdMap<T> {
    #[inline]
    pub(crate) fn copied(&self, id: Id) -> Option<T> {
        self.get(id).copied()
    }
}

/// Per-node work, schedule seeds by normalized op and facts, and built
/// components, within one search with a fixed cost model.
#[derive(Default)]
pub struct NodeCache {
    graph: Option<u64>,
    slots: Vec<Vec<u32>>,
    work: Vec<Option<Work>>,
    schedules: rustc_hash::FxHashMap<crate::lower_bound::ShapeKey, SchedPoint>,
    context: Option<ComponentContext>,
    components: rustc_hash::FxHashMap<SmallVec<[MemberState; 1]>, Arc<Component>>,
}

#[derive(PartialEq, Eq)]
struct ComponentContext {
    nodes: usize,
    caps: Caps,
    roots: Vec<ClassId>,
}

#[derive(PartialEq, Eq, Hash)]
struct MemberState {
    id: Id,
    theta: Option<SchedPoint>,
    materialized: bool,
    inputs: SmallVec<[Id; 4]>,
    readers: SmallVec<[Id; 4]>,
}

impl NodeCache {
    pub fn new(len: usize) -> Self {
        Self {
            work: (0..len).map(|_| None).collect(),
            ..Self::default()
        }
    }

    pub(crate) fn clear_schedules(&mut self) {
        self.schedules.clear();
        self.clear_components();
    }

    fn clear_components(&mut self) {
        self.components.clear();
        self.context = None;
    }

    fn bind_graph(&mut self, graph: &EGraph) {
        if self.graph.is_some_and(|arena| arena != graph.arena_id()) {
            *self = Self::new(graph.len());
        }
        self.graph = Some(graph.arena_id());
    }

    fn component_context(&mut self, graph: &EGraph, roots: &[Id], caps: &Caps) {
        self.bind_graph(graph);
        let mut roots: Vec<_> = roots.iter().map(|r| graph.class_of(*r)).collect();
        roots.sort_unstable();
        roots.dedup();
        let context = ComponentContext {
            nodes: graph.len(),
            caps: caps.clone(),
            roots,
        };
        if self.context.as_ref() != Some(&context) {
            self.components.clear();
            self.context = Some(context);
        }
    }

    pub(crate) fn seed_schedule(
        &mut self,
        graph: &EGraph,
        id: Id,
        cost: &dyn CostModel,
    ) -> Option<SchedPoint> {
        self.bind_graph(graph);
        let domain = domain_of(graph, id)?;
        if matches!(domain, ScheduleDomain::Point) {
            return Some(SchedPoint::Point);
        }
        let key = crate::lower_bound::shape_key(graph, id);
        if let Some(theta) = self.schedules.get(&key) {
            return Some(*theta);
        }
        let theta =
            crate::lower_bound::cheapest_point(graph.node(id), &key.1, &key.2, domain, cost)?;
        self.schedules.insert(key, theta);
        Some(theta)
    }

    fn work_of(&mut self, graph: &EGraph, id: Id) -> Work {
        self.bind_graph(graph);
        if self.work.len() <= id.index() {
            self.work.resize_with(id.index() + 1, || None);
        }
        if let Some(w) = self.work[id.index()] {
            return w;
        }
        let w = work_at(graph, id, dim_extent);
        self.work[id.index()] = Some(w);
        w
    }
}

fn quantized_work(
    graph: &EGraph,
    id: Id,
    inputs: &[Id],
    theta: Option<SchedPoint>,
    caps: &Caps,
    work: &mut Work,
) -> Result<()> {
    let Some(SchedPoint::Sgemv(p)) = theta else {
        return Ok(());
    };
    let op = &graph.node(id).op;
    let (Op::Launch(Launch::Contract { a, b, .. }), Some(Mnkb { m, n, k, batch })) =
        (op, Mnkb::of(op, |d| d.as_const().unwrap_or(0)))
    else {
        return Ok(());
    };
    if caps.kind != fusor_ir::device::DeviceKind::Gpu
        || m == 0
        || n == 0
        || k == 0
        || batch == 0
        || batch.saturating_mul(k).saturating_mul(n) > u64::from(u32::MAX)
    {
        return Ok(());
    }
    for (operand, &src) in b.ops.iter().zip(inputs.iter().skip(a.len())) {
        let Some((fmt, layout)) = quantized_storage(graph, src) else {
            continue;
        };
        let shape = operand.layout.shape();
        let strides = operand.layout.strides();
        let mut row_elements = 1u64;
        let contiguous_rows = shape[..shape.len().saturating_sub(1)]
            .iter()
            .zip(strides)
            .rev()
            .all(|(dim, stride)| {
                let Some(dim) = dim.as_const() else {
                    return false;
                };
                if dim > 1 && *stride != Dim::Const(row_elements) {
                    return false;
                }
                row_elements = row_elements.saturating_mul(dim);
                true
            });
        if !contiguous_rows || row_elements != batch * k || shape.last() != Some(&Dim::Const(n)) {
            continue;
        }
        let Some(offset) = operand
            .layout
            .offset()
            .as_const()
            .and_then(|v| u32::try_from(v).ok())
        else {
            continue;
        };
        let Some(column_stride) = strides
            .last()
            .and_then(|s| s.as_const())
            .and_then(|v| u32::try_from(v).ok())
        else {
            continue;
        };
        let block = sgemv_block(p, caps);
        let key = crate::quantized::DecodeKey {
            fmt,
            layout,
            k: column_stride,
            reduction: k as u32,
            elements: (batch * k * n) as u32,
            m: u32::try_from(m).unwrap_or(u32::MAX),
            n: n as u32,
            offset,
            params: p,
            width: sgemv_lanes(p, caps),
        };
        let (instructions, _) = crate::quantized::decode_window(key)?;
        let generic = batch
            .saturating_mul(n)
            .saturating_mul(k)
            .saturating_mul(fusor_ir::semantics::work::quant_decode_ops(fmt));
        let lanes = batch
            .saturating_mul(m)
            .saturating_mul(n.div_ceil(u64::from(p.cols.max(1))))
            .saturating_mul(u64::from(block));
        work.index_ops = work
            .index_ops
            .saturating_sub(generic)
            .saturating_add(instructions.saturating_mul(lanes));
    }
    Ok(())
}

/// One launch: a connected component of the realized DAG.
#[derive(Clone, Debug, PartialEq, Eq)]
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
    /// Slab members kept in workgroup memory: no buffer, binding or traffic.
    pub private: Vec<Id>,
    pub grid: [u32; 3],
    pub block: u32,
}

impl Component {
    fn launch<'a>(&'a self, extraction: &'a Extraction) -> LaunchPlan<'a> {
        LaunchPlan {
            members: &self.members,
            root: self.root,
            theta: &extraction.theta,
            reads: &self.reads,
            writes: self.writes,
            work: self.work,
            resident_lanes: self.resident_lanes,
            wg_bytes: self.wg_bytes,
            line_bytes: self.line_bytes,
            coop_steps: self.coop_steps,
            lane_steps: self.lane_steps,
            grid: self.grid,
        }
    }
}

/// The DAG one `(sigma, m, theta)` denotes; launches borrow from it on demand.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Realized {
    /// Selected nodes in post-order, leaves included.
    pub order: Vec<Id>,
    pub components: Vec<Arc<Component>>,
    /// The roots after resolution through `sigma`.
    pub roots: Vec<Id>,
}

impl Realized {
    /// Borrowed launch views for the cost model.
    pub fn launches<'a>(&'a self, extraction: &'a Extraction) -> Vec<LaunchPlan<'a>> {
        self.components
            .iter()
            .map(|c| c.launch(extraction))
            .collect()
    }

    pub(crate) fn cost_replacing(
        &self,
        index: usize,
        component: &Component,
        extraction: &Extraction,
        cost: &dyn CostModel,
    ) -> Picoseconds {
        let launches: Vec<_> = self
            .components
            .iter()
            .enumerate()
            .map(|(i, old)| if i == index { component } else { old }.launch(extraction))
            .collect();
        cost.total(&launches)
    }

    pub fn is_root(&self, id: Id) -> bool {
        self.roots.contains(&id)
    }
}

/// A cooperative member's extra work: the staging fill re-running each side's
/// `pre` once per tile past the first, padded-tile MACs and operand staging.
/// Returns the padded output bytes its stores write.
fn coop_work(
    graph: &EGraph,
    member: Id,
    theta: Option<SchedPoint>,
    caps: &Caps,
    work: &mut Work,
) -> Option<u64> {
    let Some(SchedPoint::Coop { geom, staging }) = theta else {
        return None;
    };
    let op = &graph.node(member).op;
    let (Op::Launch(Launch::Contract { a, b, .. }), Some(Mnkb { m, n, k, batch })) =
        (op, Mnkb::priced(op))
    else {
        return None;
    };
    let tiling = Tiling::new(m, n, geom.bm, geom.bn);
    let (bm, bn) = (tiling.bm, tiling.bn);
    let passes = u64::from(geom.n_passes.max(1));
    let side_extra = |side: &fusor_ir::ir::launch::ContractSide, elems: u64, tiles: u64| {
        let mut w = fusor_ir::semantics::work::epilogue_work(&side.pre, elems);
        for o in &side.ops {
            let d = fusor_ir::semantics::work::decode_ops_of(graph.facts(o.src).dtype);
            w.index_ops = w.index_ops.saturating_add(elems.saturating_mul(d));
        }
        w.scale(tiles.saturating_sub(1))
    };
    let tiles_n = tiling.tiles_n.saturating_mul(passes);
    *work = work
        .add(side_extra(
            a,
            batch.saturating_mul(m).saturating_mul(k),
            tiles_n,
        ))
        .add(side_extra(
            b,
            batch.saturating_mul(k).saturating_mul(n),
            tiling.tiles_m.max(1),
        ));
    let groups = distribute_workgroups(
        tiling.groups(batch),
        caps.limits.max_compute_workgroups_per_dimension,
    )
    .into_iter()
    .map(u64::from)
    .product::<u64>();
    let k_step = u64::from(geom.bk.max(1)).saturating_mul(u64::from(staging.max(1)));
    let k_pad = k.div_ceil(k_step).saturating_mul(k_step);
    let output_elements = groups.saturating_mul(bm).saturating_mul(bn);
    // Every K step writes both operand tiles; each subgroup column reads A
    // and each subgroup row reads B.
    let staged = bm
        .saturating_mul(1 + u64::from(geom.cg.max(1)))
        .saturating_add((bn / passes).saturating_mul(1 + u64::from(geom.rg.max(1))));
    let operand_bytes = scalar_element(graph.facts(a.primary().src).dtype).byte_size();
    *work = work.add(Work {
        macs: output_elements
            .saturating_mul(k_pad)
            .saturating_sub(batch.saturating_mul(m).saturating_mul(n).saturating_mul(k)),
        wg_bytes: groups
            .saturating_mul(passes)
            .saturating_mul(k_pad)
            .saturating_mul(staged)
            .saturating_mul(operand_bytes),
        ..Work::default()
    });
    Some(output_elements.saturating_mul(graph.facts(member).dtype.byte_size()))
}

/// Realize `(sigma, m, theta)` from `roots` and cut it into launches. An
/// unselected class or a cyclic selection is [`Error::Plan`].
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

/// Realize with cached node work; the planner is fixed for the cache's life.
pub(crate) fn realize_with(
    graph: &EGraph,
    roots: &[Id],
    extraction: &Extraction,
    cost: &dyn CostModel,
    arena: &dyn ArenaPlanner,
    cache: &mut NodeCache,
) -> Result<Realized> {
    cache.clear_components();
    Selected::new(graph, extraction, roots, cache)?.realize(graph, extraction, cost, arena, cache)
}

pub(crate) struct Selected {
    pub order: Vec<Id>,
    operands: Operands,
    roots: Vec<Id>,
}

impl Selected {
    pub(crate) fn new(
        graph: &EGraph,
        extraction: &Extraction,
        roots: &[Id],
        cache: &mut NodeCache,
    ) -> Result<Self> {
        cache.bind_graph(graph);
        let roots = roots
            .iter()
            .map(|r| select(graph, extraction, *r))
            .collect::<Result<Vec<_>>>()?;
        let (order, operands) =
            walk(graph, extraction, &roots, &mut cache.slots).map_err(Error::from)?;
        Ok(Self {
            order,
            operands,
            roots,
        })
    }

    pub(crate) fn recycle(self, cache: &mut NodeCache) {
        self.operands.recycle(&mut cache.slots);
    }

    pub(crate) fn buffers(&self, graph: &EGraph) -> fixedbitset::FixedBitSet {
        let mut buffers = fixedbitset::FixedBitSet::with_capacity(graph.len());
        for &id in &self.order {
            if leaf_role(graph, id) == LeafRole::NotLeaf {
                buffers.insert(id.index());
            }
            if let Some(last) = composite_members(graph, id).and_then(|m| m.last()) {
                buffers.remove(last.index());
            }
        }
        buffers
    }

    pub(crate) fn realize(
        self,
        graph: &EGraph,
        extraction: &Extraction,
        cost: &dyn CostModel,
        arena: &dyn ArenaPlanner,
        cache: &mut NodeCache,
    ) -> Result<Realized> {
        let Self {
            order,
            operands,
            roots,
        } = self;
        let caps = &cost.facts().caps;
        cache.component_context(graph, &roots, caps);
        let mut readers = IdMap::<SmallVec<[Id; 4]>>::with_len(graph.len(), &mut cache.slots);
        for id in &order {
            for child in operands.get(*id).into_iter().flatten() {
                readers.entry_or_default(*child).push(*id);
            }
        }
        let result = cut(graph, extraction, &order, &operands, &mut cache.slots)
            .map_err(Error::from)
            .and_then(|(owners, groups)| {
                let components = groups
                    .into_iter()
                    .map(|members| {
                        let composite = members.len() > 1;
                        let key: SmallVec<[_; 1]> = members
                            .iter()
                            .map(|&id| MemberState {
                                id,
                                theta: extraction.theta.get(&id).copied(),
                                materialized: extraction.is_materialized(id) || roots.contains(&id),
                                inputs: operands.get(id).cloned().unwrap_or_default(),
                                readers: if composite {
                                    readers.get(id).cloned().unwrap_or_default()
                                } else {
                                    SmallVec::new()
                                },
                            })
                            .collect();
                        if let Some(component) = cache.components.get(&key) {
                            return Ok(component.clone());
                        }
                        let component = Arc::new(build_component(
                            graph,
                            extraction,
                            |id| operands.get(id).map_or(&[], SmallVec::as_slice),
                            |id| readers.get(id).map_or(&[], SmallVec::as_slice),
                            |id| owners.copied(id),
                            &roots,
                            members.into_vec(),
                            caps,
                            arena,
                            cache,
                        )?);
                        cache.components.insert(key, component.clone());
                        Ok(component)
                    })
                    .collect::<Result<Vec<_>>>();
                owners.recycle(&mut cache.slots);
                components
            });
        readers.recycle(&mut cache.slots);
        operands.recycle(&mut cache.slots);
        Ok(Realized {
            order,
            components: result?,
            roots,
        })
    }
}

/// `cost.total` over the realized launches: every move's accept test.
pub fn exact_cost(
    realized: &Realized,
    extraction: &Extraction,
    cost: &dyn CostModel,
) -> Picoseconds {
    let launches = realized.launches(extraction);
    cost.total(&launches)
}

/// Rebuild one ordinary launch over its selected inputs.
pub(crate) fn ordinary_component(
    graph: &EGraph,
    extraction: &Extraction,
    node: Id,
    cost: &dyn CostModel,
    arena: &dyn ArenaPlanner,
    cache: &mut NodeCache,
) -> Result<Component> {
    if !matches!(graph.node(node).op, Op::Launch(_)) || is_composite(graph, node) {
        return Err(Error::Plan("ordinary component requires one launch".into()));
    }
    let inputs = graph
        .node(node)
        .children
        .iter()
        .map(|id| select(graph, extraction, *id))
        .collect::<Result<SmallVec<[Id; 4]>>>()?;
    build_component(
        graph,
        extraction,
        |id| if id == node { &inputs } else { &[] },
        |_| &[],
        Some,
        &[node],
        vec![node],
        &cost.facts().caps,
        arena,
        cache,
    )
}

/// The member `sigma` selected for `id`'s class.
pub(crate) fn select(graph: &EGraph, extraction: &Extraction, id: Id) -> Result<Id> {
    extraction
        .selected(graph.class_of(id))
        .ok_or_else(|| unselected(graph, id))
}

fn unselected(graph: &EGraph, id: Id) -> Error {
    Error::Plan(format!(
        "class {} has no selected member",
        graph.class_of(id).0
    ))
}

pub(crate) fn leaf_role(graph: &EGraph, id: Id) -> LeafRole {
    match &graph.node(id).op {
        Op::Logical(Logical::Leaf(LeafKind::Const { .. } | LeafKind::Uniform { .. })) => {
            LeafRole::Free
        }
        Op::Logical(Logical::Leaf(_)) => LeafRole::External,
        _ => LeafRole::NotLeaf,
    }
}

/// The iteration domain of one node; a non-launch prices over its shape.
pub(crate) fn index_space(graph: &EGraph, id: Id) -> IndexSpace {
    match &graph.node(id).op {
        Op::Launch(Launch::StreamFold { fold, .. }) => match fold.as_ref() {
            Launch::Fold { space, .. } => space.clone(),
            _ => unreachable!("admitted streamed Fold"),
        },
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
pub(crate) fn dim_extent(d: Dim) -> u64 {
    d.evaluate(&mut |_| Some(SYM_NOMINAL))
        .unwrap_or(SYM_NOMINAL)
}

fn op_at(op: &Op, extent: &impl Fn(Dim) -> u64) -> Op {
    struct Extents<'a, F>(&'a F);
    impl<F: Fn(Dim) -> u64> fusor_ir::ir::visit::VisitMut for Extents<'_, F> {
        fn dim(&mut self, dim: &mut Dim) {
            *dim = Dim::Const((self.0)(*dim));
        }
    }
    let mut op = op.clone();
    op.visit_mut(&mut Extents(extent));
    op
}

pub(crate) fn work_at(graph: &EGraph, id: Id, extent: impl Fn(Dim) -> u64) -> Work {
    let facts = |id| {
        let mut facts = graph.facts(id).clone();
        for dim in &mut facts.shape {
            *dim = Dim::Const(extent(*dim));
        }
        facts
    };
    let node = graph.node(id);
    let ins: SmallVec<[ValueFacts; 4]> = node.children.iter().map(|id| facts(*id)).collect();
    graph
        .semantics()
        .work(&op_at(&node.op, &extent), &ins, &facts(id))
}

/// The product of `dims` at their priced extents.
fn extent_product(dims: &[Dim]) -> u64 {
    dims.iter()
        .map(|d| dim_extent(*d))
        .fold(1u64, |a, b| a.saturating_mul(b))
}

pub(crate) fn elements_of(facts: &ValueFacts) -> u64 {
    extent_product(&facts.shape)
}

pub(crate) fn bytes_of(facts: &ValueFacts) -> u64 {
    let elems = elements_of(facts);
    match facts.dtype {
        Dtype::Q(fmt) => {
            let be = fmt.block_elements() as u64;
            elems.div_ceil(be) * fmt.block_bytes(fusor_ir::dtype::QLayout::Native) as u64
        }
        d => elems.saturating_mul(d.byte_size()),
    }
}

/// The storage format and layout of a quantized value's leaf.
pub(crate) fn quantized_storage(
    graph: &EGraph,
    id: Id,
) -> Option<(fusor_ir::dtype::QFmt, fusor_ir::dtype::QLayout)> {
    if !matches!(graph.facts(id).dtype, Dtype::Q(_)) {
        return None;
    }
    graph
        .class_ids(graph.class_of(id))
        .into_iter()
        .find_map(|id| match graph.node(id).op {
            Op::Logical(Logical::Leaf(LeafKind::Quantized { fmt, layout, .. })) => {
                Some((fmt, layout))
            }
            _ => None,
        })
}

fn stored_bytes(graph: &EGraph, id: Id) -> u64 {
    let facts = graph.facts(id);
    quantized_storage(graph, id).map_or_else(
        || bytes_of(facts),
        |(fmt, layout)| {
            elements_of(facts).div_ceil(u64::from(fmt.block_elements()))
                * u64::from(fmt.block_bytes(layout))
        },
    )
}

pub(crate) fn iterations_of(space: &IndexSpace) -> u64 {
    extent_product(&space.dims).max(1)
}

/// Scalar element a dtype stages as.
pub(crate) const fn scalar_element(d: Dtype) -> ScalarElement {
    match d {
        Dtype::F32 | Dtype::Q(_) => ScalarElement::F32,
        Dtype::F16 => ScalarElement::F16,
        Dtype::BF16 => ScalarElement::BF16,
        Dtype::U32 => ScalarElement::U32,
        Dtype::I32 => ScalarElement::I32,
    }
}

/// Workgroup elements needed by a fold's accumulator lanes at its emitted schedule.
pub(crate) fn fold_scratch_elements(
    graph: &EGraph,
    id: Id,
    theta: Option<SchedPoint>,
    caps: &Caps,
) -> Option<u64> {
    let Op::Launch(op) = &graph.node(id).op else {
        return None;
    };
    let fold = match op {
        Launch::StreamFold { fold, .. } => fold.as_ref(),
        _ => op,
    };
    let Launch::Fold { carrier, .. } = fold else {
        return None;
    };
    let scratch = op.fold_schedule(theta, caps).map_or_else(
        || {
            let lanes = fold_lane_group(theta, caps);
            if lanes <= 1 {
                0
            } else {
                fusor_ir::ir::launch::emitted_block(lanes, caps)
            }
        },
        |s| s.scratch,
    );
    Some(carrier.lanes()?.saturating_mul(u64::from(scratch)))
}

/// Workgroup bytes of `id`'s staging and fold scratch at `theta`.
pub(crate) fn tile_bytes(
    graph: &EGraph,
    id: Id,
    theta: Option<SchedPoint>,
    caps: &Caps,
    arena: &dyn ArenaPlanner,
) -> Result<u32> {
    let scratch = fold_scratch_elements(graph, id, theta, caps);
    let tiles = tiles_for(theta, scalar_element(graph.facts(id).dtype), scratch);
    arena.workgroup_bytes(&tiles, caps)
}

/// Workgroup tiles for a schedule and its fold's exact scratch element count.
fn tiles_for(theta: Option<SchedPoint>, elem: ScalarElement, fold_scratch: Option<u64>) -> Tiles {
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
        // `cols > 1` closes each column in one subgroup and stages nothing.
        Some(SchedPoint::Sgemv(p)) if p.cols <= 1 => {
            decls.push(tile("sgemv_partials", elem, &[p.subgroups.max(1)]));
        }
        _ => {}
    }
    if let Some(elements) = fold_scratch.filter(|e| *e > 0) {
        let extent = u32::try_from(elements).unwrap_or(u32::MAX);
        decls.push(tile("fold_scratch", elem, &[extent]));
    }

    Tiles { decls }
}

/// Bytes per cache line; amplification is a ratio against it.
const LINE_BYTES: u64 = 128;

/// Element stride of an sgemv operand's k axis; `trailing` is the extent
/// after k: `1` for A, `n` for B.
fn sgemv_k_stride(layout: &fusor_ir::shape::Layout, trailing: u64) -> u64 {
    let shape = layout.shape();
    let mut after = 1u64;
    let mut axis = shape.len();
    while axis > 0 && after < trailing {
        axis -= 1;
        after = after.saturating_mul(dim_extent(shape[axis]).max(1));
    }
    if after != trailing || axis == 0 {
        return 1;
    }
    dim_extent(layout.strides()[axis - 1]).max(1)
}

/// [`operand_line_amplification`] of a dense row-major fold operand.
pub(crate) fn fold_line_amplification(
    dims: &[u64],
    axis: usize,
    lane_group: u32,
    caps: &Caps,
    elem_bytes: u64,
) -> u64 {
    let mut strides = vec![1u64; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1].saturating_mul(dims[i + 1].max(1));
    }
    operand_line_amplification(dims, &strides, axis, lane_group, caps, elem_bytes)
}

/// How many times its useful bytes an operand read at a fold's space moves
/// through memory at `lane_group` lanes per row: each distinct line one
/// subgroup load touches costs a whole line.
pub(crate) fn operand_line_amplification(
    dims: &[u64],
    strides: &[u64],
    axis: usize,
    lane_group: u32,
    caps: &Caps,
    elem_bytes: u64,
) -> u64 {
    let Some(&k) = dims.get(axis) else {
        return 1;
    };
    let elem = elem_bytes.max(1);
    let line_elems = (LINE_BYTES / elem).max(1);
    let sg = u64::from(caps.subgroup_width().max(1));
    let lg = u64::from(lane_group.max(1)).min(sg).min(k.max(1));
    let outputs: u64 = dims
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != axis)
        .map(|(_, d)| (*d).max(1))
        .product();
    let rows = (sg / lg).min(outputs).max(1);
    let mut lines: SmallVec<[u64; 64]> = SmallVec::new();
    for r in 0..rows {
        let mut rest = r;
        let mut base = 0u64;
        for i in (0..dims.len()).rev() {
            if i == axis {
                continue;
            }
            let d = dims[i].max(1);
            base = base.saturating_add((rest % d).saturating_mul(strides[i]));
            rest /= d;
        }
        for j in 0..lg {
            let addr = base.saturating_add(j.saturating_mul(strides[axis]));
            lines.push(addr.saturating_mul(elem) / LINE_BYTES);
        }
    }
    let reads = lines.len() as u64;
    lines.sort_unstable();
    lines.dedup();
    (lines.len() as u64 * line_elems / reads.max(1)).clamp(1, line_elems)
}

/// The `(coop, lane)` dependent chain one workgroup of `root` runs at
/// `theta`; a slab sums its stages. What no occupancy shortens.
pub(crate) fn serial_steps(
    graph: &EGraph,
    root: Id,
    theta: Option<SchedPoint>,
    block: u32,
    caps: &Caps,
) -> (u64, u64) {
    match &graph.node(root).op {
        Op::Launch(Launch::Slab { slabs, members, .. }) => {
            let mut steps = 0u64;
            for m in members.iter() {
                if let Some(SlabFold { rows, k, lanes }) =
                    slab_fold_stage(graph, *m, *slabs, block, caps)
                {
                    let groups = u64::from(block / lanes.max(1)).max(1);
                    steps += rows.div_ceil(groups).max(1) * k.div_ceil(u64::from(lanes));
                } else if let Op::Launch(Launch::Map { space, .. }) = &graph.node(*m).op {
                    let total = iterations_of(space);
                    steps += (total / u64::from((*slabs).max(1)))
                        .div_ceil(u64::from(block))
                        .max(1);
                }
            }
            (0, steps)
        }
        op => node_serial_steps(&op_at(op, &dim_extent), theta, caps),
    }
}

/// [`serial_steps`] of one launch node from its own op.
pub fn node_serial_steps(op: &Op, theta: Option<SchedPoint>, caps: &Caps) -> (u64, u64) {
    match op {
        Op::Launch(Launch::StreamFold {
            producer,
            fold,
            operand,
            ..
        }) => {
            let Launch::Fold { space, axis, .. } = producer.as_ref() else {
                unreachable!("admitted streamed Fold")
            };
            let k = space.dims[*axis as usize].as_const().unwrap_or(1);
            let copies = fusor_ir::semantics::work::stream_evaluations(fold, *operand)
                / fold.iter_space().iterations().unwrap_or(1).max(1);
            let (_, steps) = node_serial_steps(&Op::Launch(*fold.clone()), theta, caps);
            (0, steps.saturating_mul(1 + k.saturating_mul(copies)))
        }
        Op::Launch(Launch::Contract { k, .. }) => {
            let k = k.as_const().unwrap_or(1).max(1);
            // A step is one fragment depth of k, whatever `bk` stages.
            let depth = u64::from(fusor_ir::ir::launch::CoopGeom::COOP_DIM.max(1));
            match theta {
                // A depth step is a staged load and a barrier, whatever the
                // multiplies per subgroup.
                Some(SchedPoint::Coop { .. }) => (k.div_ceil(depth), 0),
                // A scalar-tiled lane walks every k: measured 8-10x a fragment.
                Some(SchedPoint::Sgemm(_)) => (0, k.saturating_mul(4)),
                Some(SchedPoint::Sgemv(p)) => (0, k.div_ceil(u64::from(sgemv_lanes(p, caps)))),
                _ => (0, k),
            }
        }
        Op::Launch(op @ Launch::Fold { space, axis, .. }) => {
            let theta = fold_theta(op, theta, caps);
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
        // A dense scatter walks every update per output lane.
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

/// A slab fold stage's rows per slab, reduced extent and lanes per row.
struct SlabFold {
    rows: u64,
    k: u64,
    lanes: u32,
}

/// `stage` as a fold stage of a slab of `slabs` at `block` lanes.
fn slab_fold_stage(
    graph: &EGraph,
    stage: Id,
    slabs: u32,
    block: u32,
    caps: &Caps,
) -> Option<SlabFold> {
    let Op::Launch(Launch::Fold {
        space,
        axis,
        carrier,
        ..
    }) = &graph.node(stage).op
    else {
        return None;
    };
    let k = dim_extent(*space.dims.get(*axis as usize)?).max(1);
    let rows = (iterations_of(space) / k) / u64::from(slabs.max(1));
    let lanes = slab_subgroup_width(block, rows, k, carrier, caps)
        .unwrap_or_else(|| slab_lanes_per_row(block, rows, k));
    Some(SlabFold { rows, k, lanes })
}

/// The block a slab lowers at: its widest stage's share of one slab.
fn slab_block_of(graph: &EGraph, slabs: u32, members: &[Id], caps: &Caps) -> u32 {
    let widest = members
        .iter()
        .filter_map(|s| index_space(graph, *s).iterations())
        .map(|n| n / u64::from(slabs.max(1)))
        .max()
        .unwrap_or(1);
    fusor_ir::ir::launch::slab_block(widest, caps)
}

/// Lanes per workgroup and workgroup count implied by one schedule point.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct Geometry {
    pub block: u32,
    pub workgroups: u64,
}

pub(crate) fn geometry(theta: Option<SchedPoint>, space: &IndexSpace, caps: &Caps) -> Geometry {
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
            workgroups: Tiling::new(m, n, geom.bm, geom.bn).groups(batch),
        },
        Some(SchedPoint::Sgemm(p)) => Geometry {
            block: ((p.bm / p.tm.max(1)) * (p.bn / p.tn.max(1))).max(1),
            workgroups: m.div_ceil(p.bm.max(1) as u64) * n.div_ceil(p.bn.max(1) as u64) * batch,
        },
        // One workgroup per output, or per `cols`-wide column group.
        Some(SchedPoint::Sgemv(p)) => Geometry {
            block: (p.subgroups.max(1) * width).max(1),
            workgroups: m
                .saturating_mul(batch)
                .saturating_mul(n.div_ceil(u64::from(p.cols.max(1))))
                .max(1),
        },
        // `emitted_block(lane_group)` lanes, `block / lane_group` rows.
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

/// The 3-D grid under `max_per_dim`: slab count first, then size x, so the
/// last slab is not left nearly empty.
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
type ComponentMembers = SmallVec<[Id; 4]>;

/// Why a selection cannot be ordered; `Cycle` names a member to re-select.
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

/// The node at which this selection, or its composite ownership, closes a
/// cycle: selected members may name each other's classes.
pub(crate) fn selection_cycle(graph: &EGraph, extraction: &Extraction, roots: &[Id]) -> Option<Id> {
    let resolved = roots
        .iter()
        .map(|r| select(graph, extraction, *r))
        .collect::<Result<Vec<_>>>()
        .ok()?;
    let mut slots = Vec::new();
    let attempt = walk(graph, extraction, &resolved, &mut slots)
        .and_then(|(order, operands)| cut(graph, extraction, &order, &operands, &mut slots));
    match attempt {
        Err(WalkFail::Cycle(v)) => Some(v),
        _ => None,
    }
}

fn walk(
    graph: &EGraph,
    extraction: &Extraction,
    roots: &[Id],
    slots: &mut Vec<Vec<u32>>,
) -> std::result::Result<(Vec<Id>, Operands), WalkFail> {
    const UNSEEN: u8 = 0;
    const OPEN: u8 = 1;
    const DONE: u8 = 2;

    let mut state = IdMap::with_len(graph.len(), slots);
    let mut order: Vec<Id> = Vec::new();
    let mut operands: Operands = IdMap::with_len(graph.len(), slots);
    let mut stack: Vec<Frame> = roots.iter().rev().map(|r| Frame::Enter(*r)).collect();

    let result = (|| {
        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(v) => match state.copied(v).unwrap_or(UNSEEN) {
                    DONE => {}
                    OPEN => return Err(WalkFail::Cycle(v)),
                    _ => {
                        state.insert(v, OPEN);
                        // A composite names its members by id.
                        let kids: SmallVec<[Id; 4]> = resolved_children(graph, extraction, v)
                            .map(|(c, n)| n.ok_or_else(|| unselected(graph, c)))
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
                    state.insert(v, DONE);
                    order.push(v);
                }
            }
        }
        Ok(())
    })();
    state.recycle(slots);
    if let Err(error) = result {
        operands.recycle(slots);
        return Err(error);
    }
    Ok((order, operands))
}

fn cut(
    graph: &EGraph,
    extraction: &Extraction,
    order: &[Id],
    operands: &Operands,
    slots: &mut Vec<Vec<u32>>,
) -> std::result::Result<(IdMap<Id>, Vec<ComponentMembers>), WalkFail> {
    let mut owners = IdMap::with_len(graph.len(), slots);
    let mut index_of = IdMap::with_len(graph.len(), slots);
    let result = (|| {
        // Reverse postorder: outer composites own before nested ones.
        for &id in order.iter().rev() {
            if leaf_role(graph, id) != LeafRole::NotLeaf {
                continue;
            }
            let owner = owners.copied(id).unwrap_or(id);
            owners.insert(id, owner);
            if let Some(members) = composite_members(graph, id) {
                for member in members {
                    if owners.copied(*member).is_some_and(|other| other != owner) {
                        return Err(WalkFail::Cycle(id));
                    }
                    owners.insert(*member, owner);
                }
            }
        }

        let mut groups: Vec<ComponentMembers> = Vec::new();
        for v in order {
            if leaf_role(graph, *v) != LeafRole::NotLeaf {
                continue;
            }
            let owner = owners.copied(*v).unwrap_or(*v);
            let idx = index_of.copied(owner).unwrap_or_else(|| {
                let index = groups.len() as u32;
                groups.push(SmallVec::new());
                index_of.insert(owner, index);
                index
            });
            groups[idx as usize].push(*v);
        }

        // First-appearance order is not a dependency order: sort the groups
        // as a DAG, earliest ready first.
        let n = groups.len();
        let mut deps: Vec<SmallVec<[usize; 4]>> = vec![SmallVec::new(); n];
        let mut indegree = vec![0usize; n];
        for (g, members) in groups.iter().enumerate() {
            let mut seen: SmallVec<[usize; 8]> = SmallVec::new();
            for v in members {
                for c in operands.get(*v).map(|o| o.as_slice()).unwrap_or(&[]) {
                    let Some(owner) = owners.copied(*c) else {
                        continue;
                    };
                    let d = index_of.copied(owner).unwrap() as usize;
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
                            if is_composite(graph, id)
                                && extraction.selected(graph.class_of(id)) == Some(id)
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
        let mut reordered = Vec::with_capacity(n);
        for old in &sorted {
            reordered.push(std::mem::take(&mut groups[*old]));
        }
        Ok(reordered)
    })();
    index_of.recycle(slots);
    match result {
        Ok(groups) => Ok((owners, groups)),
        Err(error) => {
            owners.recycle(slots);
            Err(error)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_component<'a>(
    graph: &EGraph,
    extraction: &Extraction,
    operands: impl Fn(Id) -> &'a [Id],
    consumer_nodes: impl Fn(Id) -> &'a [Id],
    owner: impl Fn(Id) -> Option<Id>,
    roots: &[Id],
    members: Vec<Id>,
    caps: &Caps,
    arena: &dyn ArenaPlanner,
    cache: &mut NodeCache,
) -> Result<Component> {
    let own = members.first().and_then(|m| owner(*m));
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
        quantized_work(graph, *m, operands(*m), theta_m, caps, &mut w)?;
        let padded_writes = coop_work(graph, *m, theta_m, caps, &mut w);
        let materialized = extraction.is_materialized(*m) || roots.contains(m);
        if materialized {
            writes = writes.saturating_add(padded_writes.unwrap_or_else(|| bytes_of(out)));
        }
        work = work.add(w);
    }

    // Distinct external operands and their consumers' reread factors.
    let mut ext: Vec<(Id, u64, u32)> = Vec::new();
    for m in &members {
        let iters = iterations_of(&index_space(graph, *m));
        let stream_reads = match &graph.node(*m).op {
            op @ Op::Launch(Launch::StreamFold { .. }) => {
                let Op::Launch(Launch::StreamFold {
                    producer,
                    fold,
                    operand,
                    ..
                }) = op_at(op, &dim_extent)
                else {
                    unreachable!()
                };
                let Launch::Fold {
                    space, axis, ops, ..
                } = producer.as_ref()
                else {
                    unreachable!("admitted streamed Fold")
                };
                Some((
                    ops.len(),
                    fusor_ir::semantics::work::stream_evaluations(&fold, operand)
                        .saturating_mul(dim_extent(space.dims[*axis as usize])),
                ))
            }
            _ => None,
        };
        let window = match extraction.theta.get(m) {
            Some(SchedPoint::Sgemv(p)) if p.cols > 1 => Some((p.cols, p.subgroups.max(1))),
            Some(SchedPoint::Sgemv(_)) => Some((1, 1)),
            Some(SchedPoint::Sgemm(p)) => Some((p.tn.max(1), 1)),
            _ => None,
        };
        let op = &graph.node(*m).op;
        let contract_reads = match (op, Mnkb::of(op, dim_extent), window) {
            (Op::Launch(Launch::Contract { a, .. }), Some(c), Some((columns, copies))) => {
                let rows_k = c.batch.saturating_mul(c.m).saturating_mul(c.k);
                // A is shared per subgroup (sgemv) or per register column
                // (sgemm); every output row scans B.
                let a_scans =
                    c.n.div_ceil(u64::from(columns))
                        .saturating_mul(u64::from(copies));
                Some((
                    a.len(),
                    rows_k.saturating_mul(a_scans),
                    rows_k.saturating_mul(c.n),
                ))
            }
            _ => None,
        };
        for (slot, &c) in operands(*m).iter().enumerate() {
            if owner(c) == own {
                continue;
            }
            if leaf_role(graph, c) == LeafRole::Free {
                continue;
            }
            let facts = graph.facts(c);
            let elems = elements_of(facts).max(1);
            let iters =
                contract_reads.map_or(iters, |(a_len, a, b)| if slot < a_len { a } else { b });
            let iters = stream_reads.map_or(
                iters,
                |(source_len, reads)| if slot < source_len { reads } else { iters },
            );
            let reread = iters.div_ceil(elems).max(1).min(u32::MAX as u64) as u32;
            match ext.iter_mut().find(|(id, _, _)| *id == c) {
                Some(slot) => slot.2 = slot.2.max(reread),
                None => ext.push((c, stored_bytes(graph, c), reread)),
            }
        }
    }
    ext.sort_by_key(|(id, _, _)| *id);

    let theta = extraction.theta.get(&root).copied();
    let theta = match &graph.node(root).op {
        Op::Launch(op) => fold_theta(op, theta, caps),
        _ => theta,
    };
    // What runs side by side: a group's members, else the root.
    let units: SmallVec<[(Id, Option<SchedPoint>, Geometry); 4]> = match &graph.node(root).op {
        Op::Launch(Launch::Group { members: gm, .. }) => gm
            .iter()
            .map(|m| {
                let theta = extraction.theta.get(m).copied();
                (*m, theta, member_geometry(graph, extraction, *m, caps))
            })
            .collect(),
        _ => smallvec::smallvec![(root, theta, member_geometry(graph, extraction, root, caps))],
    };
    // A group dispatches its members' workgroups at the widest block.
    let geom = if is_group(graph, root) {
        Geometry {
            block: units.iter().map(|(_, _, g)| g.block).max().unwrap_or(1),
            workgroups: units
                .iter()
                .map(|(_, _, g)| {
                    let d = distribute_workgroups(
                        g.workgroups,
                        caps.limits.max_compute_workgroups_per_dimension,
                    );
                    u64::from(d[0]) * u64::from(d[1]) * u64::from(d[2])
                })
                .sum::<u64>()
                .max(1),
        }
    } else {
        units[0].2
    };
    let mut wg_bytes = tile_bytes(graph, root, theta, caps, arena)? as u64;

    // Uncoalesced reads pay line amplification at each fold's lanes per row.
    let mut line_bytes = 0u64;
    let mut stages: SmallVec<[(Id, u32); 4]> = SmallVec::new();
    for &(m, theta, g) in &units {
        match &graph.node(m).op {
            Op::Launch(Launch::Fold { .. }) => stages.push((m, fold_lane_group(theta, caps))),
            Op::Launch(Launch::Slab { slabs, members, .. }) => {
                stages.extend(members.iter().filter_map(|s| {
                    slab_fold_stage(graph, *s, *slabs, g.block, caps).map(|f| (*s, f.lanes))
                }));
            }
            _ => {}
        }
    }
    // A coop contraction pulls each operand once per tile on the other side.
    if let Some(Mnkb { m, n, k, batch }) = Mnkb::priced(&graph.node(root).op)
        && let Some(SchedPoint::Coop { geom, .. }) = theta
    {
        let tiling = Tiling::new(m, n, geom.bm, geom.bn);
        let a_passes = u64::from(geom.n_passes.max(1));
        let elem = graph.facts(root).dtype.byte_size().max(1);
        let pulled = batch * k * (m * tiling.tiles_n * a_passes + n * tiling.tiles_m);
        let useful = batch * k * (m + n);
        line_bytes = line_bytes.saturating_add(pulled.saturating_sub(useful).saturating_mul(elem));
    }
    // An sgemv operand strided along k pulls a line per element per reread.
    if let Op::Launch(Launch::Contract { n, a, b, .. }) = &graph.node(root).op
        && let Some(SchedPoint::Sgemv(_)) = theta
    {
        let n = dim_extent(*n).max(1);
        let sides = a
            .ops
            .iter()
            .map(|o| (o, 1))
            .chain(b.ops.iter().map(|o| (o, n)));
        for (o, trailing) in sides {
            let facts = graph.facts(o.src);
            if matches!(facts.dtype, Dtype::Q(_)) {
                continue;
            }
            let line = (LINE_BYTES / facts.dtype.byte_size().max(1)).max(1);
            let amp = sgemv_k_stride(&o.layout, trailing).min(line);
            let Some(&(_, bytes, reread)) = ext
                .iter()
                .find(|(id, _, _)| graph.class_of(*id) == graph.class_of(o.src))
            else {
                continue;
            };
            line_bytes = line_bytes.saturating_add(
                bytes
                    .saturating_mul(u64::from(reread))
                    .saturating_mul(amp - 1),
            );
        }
    }

    for &m in &members {
        let ops = match &graph.node(m).op {
            Op::Launch(Launch::Fold { ops, .. } | Launch::Map { ops, .. }) => ops,
            _ => continue,
        };
        // The fold walking this member's operands: itself, else one it spans.
        let spans = |stage: Id| match &graph.node(stage).op {
            Op::Launch(Launch::Fold { space, .. }) => ops.iter().all(|op| {
                op.layout.shape().len() == space.dims.len()
                    && op
                        .layout
                        .shape()
                        .iter()
                        .zip(&space.dims)
                        .all(|(a, b)| dim_extent(*a) == dim_extent(*b))
            }),
            _ => false,
        };
        let Some(&(stage, lane_group)) = stages
            .iter()
            .find(|(s, _)| *s == m)
            .or_else(|| stages.iter().find(|(s, _)| spans(*s)))
        else {
            continue;
        };
        let Op::Launch(Launch::Fold { space, axis, .. }) = &graph.node(stage).op else {
            continue;
        };
        let dims: SmallVec<[u64; 6]> = space.dims.iter().map(|d| dim_extent(*d)).collect();
        let total: u64 = dims.iter().product();
        for op in ops {
            if owner(op.src) == own {
                continue;
            }
            let facts = graph.facts(op.src);
            if matches!(facts.dtype, Dtype::Q(_)) {
                continue;
            }
            let shape = op.layout.shape();
            let strides: Option<SmallVec<[u64; 6]>> = (matches!(op.access, AccessPlan::Alias)
                && shape.len() == dims.len()
                && shape.iter().zip(&dims).all(|(d, e)| dim_extent(*d) == *e))
            .then(|| op.layout.strides().iter().map(|s| dim_extent(*s)).collect());
            let elem = facts.dtype.byte_size().max(1);
            let amp = match strides {
                Some(strides) => operand_line_amplification(
                    &dims,
                    &strides,
                    *axis as usize,
                    lane_group,
                    caps,
                    elem,
                ),
                None if elements_of(facts) == total => {
                    fold_line_amplification(&dims, *axis as usize, lane_group, caps, elem)
                }
                None => 1,
            };
            line_bytes =
                line_bytes.saturating_add(total.saturating_mul(elem).saturating_mul(amp - 1));
        }
    }

    // Members run side by side: the chain is the longest of theirs.
    let (coop_steps, lane_steps) = units
        .iter()
        .map(|(m, theta, g)| serial_steps(graph, *m, *theta, g.block, caps))
        .fold((0, 0), |a, b| (a.0.max(b.0), a.1.max(b.1)));

    // A slab's unshared middle members live in workgroup memory as far as
    // it fits; a group's member slabs keep their own.
    let mut private: Vec<Id> = Vec::new();
    for &(slab, _, _) in &units {
        let Op::Launch(Launch::Slab { members: sm, .. }) = &graph.node(slab).op else {
            continue;
        };
        let inside: rustc_hash::FxHashSet<ClassId> =
            sm.iter().map(|m| graph.class_of(*m)).collect();
        let shared = |m: Id| {
            consumer_nodes(m)
                .iter()
                .any(|c| c != &slab && c != &root && !inside.contains(&graph.class_of(*c)))
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

/// The geometry one node launches at on its own.
fn member_geometry(graph: &EGraph, extraction: &Extraction, m: Id, caps: &Caps) -> Geometry {
    let theta = extraction.theta.get(&m).copied();
    let launch = match &graph.node(m).op {
        Op::Launch(Launch::StreamFold { fold, .. }) => Some(fold.as_ref()),
        Op::Launch(launch) => Some(launch),
        _ => None,
    };
    match launch {
        Some(Launch::Slab { slabs, members, .. }) => Geometry {
            block: slab_block_of(graph, *slabs, members, caps),
            workgroups: u64::from(*slabs).max(1),
        },
        Some(
            op @ Launch::Fold {
                space,
                axis,
                vec_axes,
                ..
            },
        ) if caps.kind == fusor_ir::device::DeviceKind::Gpu => {
            let schedule = op.fold_schedule(theta, caps).unwrap();
            let block = schedule.block;
            let rows = space
                .dims
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != *axis as usize && !vec_axes.contains(&(*i as u32)))
                .map(|(_, d)| dim_extent(*d))
                .product::<u64>();
            let rows_per_group =
                (block / schedule.strategy.lane_group(caps.subgroup_width()).max(1)).max(1);
            Geometry {
                block,
                workgroups: rows.div_ceil(u64::from(rows_per_group)).max(1),
            }
        }
        _ => geometry(theta, &index_space(graph, m), caps),
    }
}

/// True when a class has exactly one member.
pub(crate) fn is_singleton(graph: &EGraph, class: ClassId) -> bool {
    !matches!(graph.node(class.0).op, Op::Union(..))
}

/// True when a plan may select `id`: a `Leaf` or a `Launch`-level node. Cost
/// cannot decide this: a `Logical` node ties its lowered twin.
pub(crate) fn is_runnable(graph: &EGraph, id: Id) -> bool {
    if !matches!(graph.node(id).op, Op::Logical(Logical::Leaf(_)))
        && graph.level(id) != fusor_ir::ir::Level::Launch
    {
        return false;
    }
    !is_self_referential(graph, id)
}

/// True when `id` names its own e-class as an operand, so selecting it would
/// compute X by computing X.
pub(crate) fn is_self_referential(graph: &EGraph, id: Id) -> bool {
    // A composite's last member shares its class but is read by id.
    if is_composite(graph, id) {
        return false;
    }
    let class = graph.class_of(id);
    graph
        .node(id)
        .children
        .iter()
        .any(|c| graph.class_of(*c) == class)
}

/// A fold's carrier lanes and accumulator bytes; `None` when not a fold or
/// its slot extent is symbolic.
pub(crate) fn fold_footprint(graph: &EGraph, id: Id) -> Option<(u64, u64)> {
    match &graph.node(id).op {
        Op::Launch(Launch::Fold { carrier, acc, .. }) => Some((carrier.lanes()?, acc.byte_size())),
        _ => None,
    }
}

/// Whether a composite can bind its externally visible values in this graph.
pub(crate) fn composite_bindings_fit(graph: &EGraph, id: Id, caps: &Caps) -> bool {
    slab_bindings_fit(graph, id, caps) && group_bindings_fit(graph, id, caps)
}

struct BindingsCache {
    arena: u64,
    nodes: usize,
    roots: Vec<Id>,
    caps: Caps,
    fits: rustc_hash::FxHashMap<Id, bool>,
}

fn cached_bindings_fit(
    graph: &EGraph,
    id: Id,
    caps: &Caps,
    compute: impl FnOnce() -> bool,
) -> bool {
    thread_local! {
        static MEMO: std::cell::RefCell<Option<BindingsCache>> = const {
            std::cell::RefCell::new(None)
        };
    }
    let hit = MEMO.with(|memo| {
        let mut memo = memo.borrow_mut();
        if memo.as_ref().is_none_or(|cache| {
            cache.arena != graph.arena_id()
                || cache.nodes != graph.len()
                || cache.roots != graph.roots()
                || cache.caps != *caps
        }) {
            *memo = Some(BindingsCache {
                arena: graph.arena_id(),
                nodes: graph.len(),
                roots: graph.roots().to_vec(),
                caps: caps.clone(),
                fits: Default::default(),
            });
        }
        memo.as_ref().unwrap().fits.get(&id).copied()
    });
    if let Some(fit) = hit {
        return fit;
    }
    let fit = compute();
    MEMO.with(|memo| memo.borrow_mut().as_mut().unwrap().fits.insert(id, fit));
    fit
}

/// Whether group `id` can bind its members' buffers and distinct inputs.
pub(crate) fn group_bindings_fit(graph: &EGraph, id: Id, caps: &Caps) -> bool {
    let Op::Launch(Launch::Group { members, .. }) = &graph.node(id).op else {
        return true;
    };
    cached_bindings_fit(graph, id, caps, || group_members_fit(graph, members, caps))
}

fn group_members_fit(graph: &EGraph, members: &[Id], caps: &Caps) -> bool {
    let mut inputs: rustc_hash::FxHashSet<ClassId> = rustc_hash::FxHashSet::default();
    for m in members.iter() {
        let Op::Launch(Launch::Slab { members: sm, .. }) = &graph.node(*m).op else {
            inputs.extend(graph.node(*m).children.iter().map(|c| graph.class_of(*c)));
            continue;
        };
        if !slab_bindings_fit(graph, *m, caps) {
            return false;
        }
        let own: rustc_hash::FxHashSet<ClassId> = sm.iter().map(|s| graph.class_of(*s)).collect();
        for s in sm.iter() {
            inputs.extend(
                graph
                    .node(*s)
                    .children
                    .iter()
                    .map(|c| graph.class_of(*c))
                    .filter(|class| !own.contains(class)),
            );
        }
    }
    let own: rustc_hash::FxHashSet<ClassId> = members.iter().map(|m| graph.class_of(*m)).collect();
    let root_classes: rustc_hash::FxHashSet<ClassId> =
        graph.roots().iter().map(|r| graph.class_of(*r)).collect();
    let inputs = inputs
        .iter()
        .filter(|c| !own.contains(c) && own_buffer(graph, **c, &root_classes))
        .count();
    let outs = members
        .iter()
        .filter(|m| root_classes.contains(&graph.class_of(**m)))
        .count();
    2 + outs + inputs <= caps.limits.max_storage_buffers_per_shader_stage as usize
}

/// Whether slab `id` can bind under [`slab_layout`], judged on every reader
/// in the graph (a superset of the realized ones).
pub(crate) fn slab_bindings_fit(graph: &EGraph, id: Id, caps: &Caps) -> bool {
    let Op::Launch(Launch::Slab { members, .. }) = &graph.node(id).op else {
        return true;
    };
    cached_bindings_fit(graph, id, caps, || {
        let classes: rustc_hash::FxHashSet<ClassId> =
            members.iter().map(|m| graph.class_of(*m)).collect();
        let shared =
            |m: Id| graph.any_reader(graph.class_of(m), |r| !classes.contains(&graph.class_of(r)));
        slab_layout(graph, id, caps, graph.roots(), &shared).is_ok()
    })
}

/// Whether a class binds its own storage buffer (an external leaf or a
/// root) rather than the step arena.
pub(crate) fn own_buffer(
    graph: &EGraph,
    class: ClassId,
    roots: &rustc_hash::FxHashSet<ClassId>,
) -> bool {
    roots.contains(&class)
        || graph
            .members(class)
            .iter()
            .any(|m| leaf_role(graph, *m) == LeafRole::External)
}

/// A slab's private middle members and workgroup bytes: fold scratch, then
/// unshared non-root members smallest first as they fit. `Err` when the
/// storage buffers left over exceed the device's bindings.
pub(crate) fn slab_layout(
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
    let block = u64::from(slab_block_of(graph, *slabs, sm, caps));
    let slabs = u64::from((*slabs).max(1));
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
    if crate::debug::flags().no_private {
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
    // Uniforms, the arena, and every own-buffer output, input or member.
    let owns = |c: ClassId| own_buffer(graph, c, &root_classes);
    let bound = 2
        + inputs.iter().filter(|c| owns(**c)).count()
        + middle
            .iter()
            .filter(|m| !private.contains(m) && owns(graph.class_of(**m)))
            .count()
        + usize::from(owns(graph.class_of(root)));
    let limit_bufs = caps.limits.max_storage_buffers_per_shader_stage as usize;
    if bound > limit_bufs {
        return Err(Error::Plan(format!(
            "slab {root} binds {bound} storage buffers over the {limit_bufs}-buffer limit: \
             its middle members do not fit workgroup memory"
        )));
    }
    Ok((private, used))
}

/// The members of `class` a selection may take: acyclic, runnable, able to
/// bind, each filter dropped when it would leave nothing.
pub(crate) fn selectable(graph: &EGraph, class: ClassId, caps: &Caps) -> Vec<Id> {
    fn narrow(pool: Vec<Id>, keep: impl Fn(Id) -> bool) -> Vec<Id> {
        let kept: Vec<Id> = pool.iter().copied().filter(|m| keep(*m)).collect();
        if kept.is_empty() { pool } else { kept }
    }
    let pool = narrow(graph.members(class), |m| !is_self_referential(graph, m));
    let pool = narrow(pool, |m| is_runnable(graph, m));
    narrow(pool, |m| composite_bindings_fit(graph, m, caps))
}

/// The classes reachable from `roots`, ascending, and a mask of every id
/// they hold. The mask is closed under children, so a fixpoint over it
/// equals the whole-graph one restricted to it.
pub(crate) fn reachable(graph: &EGraph, roots: &[Id]) -> (Vec<ClassId>, fixedbitset::FixedBitSet) {
    let (mut classes, mask) = reachable_unsorted(graph, roots);
    classes.sort_unstable();
    (classes, mask)
}

/// [`reachable`] with the classes in discovery order.
pub(crate) fn reachable_unsorted(
    graph: &EGraph,
    roots: &[Id],
) -> (Vec<ClassId>, fixedbitset::FixedBitSet) {
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
        // Classes are disjoint, so the mask is this walk's visited set.
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
    (out, mask)
}

/// Every class in the graph, ascending.
pub(crate) fn classes(graph: &EGraph) -> Vec<ClassId> {
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use fusor_ir::cost::DeviceFacts;
    use fusor_ir::device::{DeviceKind, Limits, SubgroupWidths};
    use fusor_ir::extract::{ExtractBudget, Extractor};
    use fusor_ir::ir::Node;
    use fusor_ir::ir::launch::{
        AccessPlan, ContractSide, CoopDomain, CoopGeom, CoopSchedule, Family, Operand, SgemmDomain,
        SgemmParams, SgemvDomain, SgemvParams,
    };
    use fusor_ir::ir::logical::BufferId;
    use fusor_ir::scalar::{ScalarExpr, UnOp};
    use fusor_ir::shape::Layout;
    use rustc_hash::FxHashMap;

    struct PeakCost(crate::Roofline);

    impl CostModel for PeakCost {
        fn facts(&self) -> &DeviceFacts {
            self.0.facts()
        }

        fn launch_cost(&self, launch: &LaunchPlan<'_>) -> Picoseconds {
            self.0.launch_cost(launch)
        }

        fn node_math(
            &self,
            node: &Node,
            ins: &[ValueFacts],
            out: &ValueFacts,
            theta: Option<SchedPoint>,
        ) -> Picoseconds {
            self.0.node_math(node, ins, out, theta)
        }

        fn traffic(&self, bytes: u64, rereads: u32) -> Picoseconds {
            self.0.traffic(bytes, rereads)
        }

        fn total(&self, launches: &[LaunchPlan<'_>]) -> Picoseconds {
            self.0.total(launches)
                + launches
                    .iter()
                    .map(|l| self.launch_cost(l))
                    .max()
                    .unwrap_or_default()
        }
    }

    pub(crate) fn caps() -> Caps {
        Caps {
            kind: DeviceKind::Gpu,
            name: "replacement pricing".into(),
            limits: Limits::default(),
            subgroups: Some(SubgroupWidths { min: 32, max: 32 }),
            f16: false,
            bf16: false,
            coop: Default::default(),
            atomic_f32: false,
            workgroup_alias: false,
            mixed_precision_coop_store: false,
            pipeline_cache: false,
            timestamp_query: false,
            simd_widths: Default::default(),
            threads: 1,
        }
    }

    pub(crate) fn new_graph(arena: &Arc<fusor_tile::Planner>) -> EGraph {
        EGraph::new(fusor_ir::CoreSemantics::new(arena.clone()))
    }

    /// An f32 buffer leaf.
    pub(crate) fn buffer(graph: &mut EGraph, name: u32, shape: &[Dim]) -> Id {
        graph
            .add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
                name: BufferId(name),
                dtype: Dtype::F32,
                shape: shape.iter().copied().collect(),
            })))
            .unwrap()
    }

    pub(crate) fn alias(src: Id, layout: Layout) -> Operand {
        Operand {
            src,
            layout,
            access: AccessPlan::Alias,
        }
    }

    pub(crate) fn side(src: Id, layout: Layout) -> ContractSide {
        ContractSide::one(ScalarExpr::arg(0, Dtype::F32), alias(src, layout))
    }

    /// Each id selected for its own class.
    pub(crate) fn selecting(
        graph: &EGraph,
        ids: impl IntoIterator<Item = Id>,
    ) -> FxHashMap<ClassId, Id> {
        ids.into_iter().map(|id| (graph.class_of(id), id)).collect()
    }

    /// An f32 contraction over `[m, n, k, batch]`.
    pub(crate) fn contract(
        output: &[Dim],
        [m, n, k, batch]: [Dim; 4],
        family: Family,
        a: ContractSide,
        b: ContractSide,
        sched: ScheduleDomain,
    ) -> Op {
        Op::Launch(Launch::Contract {
            output: IndexSpace::new(output.iter().copied()),
            m,
            n,
            k,
            batch,
            family,
            post: ScalarExpr::arg(0, Dtype::F32),
            acc: Dtype::F32,
            a,
            b,
            sched,
        })
    }

    pub(crate) fn sgemv(p: SgemvParams) -> ScheduleDomain {
        ScheduleDomain::Sgemv(
            SgemvDomain {
                params: [p].into_iter().collect(),
            }
            .into(),
        )
    }

    fn sgemm(p: SgemmParams) -> ScheduleDomain {
        ScheduleDomain::Sgemm(
            SgemmDomain {
                params: [p].into_iter().collect(),
            }
            .into(),
        )
    }

    fn map(graph: &mut EGraph, src: Id, shape: &[Dim], op: UnOp) -> Id {
        graph
            .add(Op::Launch(Launch::Map {
                space: IndexSpace::new(shape.iter().copied()),
                body: ScalarExpr::un(op, ScalarExpr::arg(0, Dtype::F32)),
                ops: vec![alias(src, Layout::contiguous(shape))],
                sched: ScheduleDomain::Point,
            }))
            .unwrap()
    }

    fn composite(graph: &mut EGraph, slab: bool, members: [Id; 2]) -> Id {
        let members = members.into_iter().collect();
        let sched = ScheduleDomain::Point;
        graph
            .add(Op::Launch(if slab {
                Launch::Slab {
                    slabs: 1,
                    members,
                    sched,
                }
            } else {
                Launch::Group { members, sched }
            }))
            .unwrap()
    }

    #[test]
    fn composite_binding_cache_tracks_roots_and_device_limits() {
        let mut caps = caps();
        caps.limits.max_storage_buffers_per_shader_stage = 3;
        let mut graph = new_graph(&Arc::new(fusor_tile::Planner::new()));
        let shape = [Dim::ONE, Dim::Const(32)];
        let input = buffer(&mut graph, 0, &shape);
        let middle = map(&mut graph, input, &shape, UnOp::Neg);
        let last = map(&mut graph, middle, &shape, UnOp::Neg);
        let other = map(&mut graph, input, &shape, UnOp::Abs);
        let slab = composite(&mut graph, true, [middle, last]);
        graph.union(last, slab).unwrap();
        let group = composite(&mut graph, false, [slab, other]);
        graph.union(other, group).unwrap();
        let nodes = graph.len();
        let fits = |graph: &EGraph, caps: &Caps| {
            (
                group_bindings_fit(graph, group, caps),
                slab_bindings_fit(graph, slab, caps),
            )
        };
        assert_eq!(fits(&graph, &caps), (true, true));

        graph.add_root(slab);
        graph.add_root(other);
        assert_eq!(fits(&graph, &caps), (false, false));
        caps.limits.max_storage_buffers_per_shader_stage = 5;
        assert_eq!(fits(&graph, &caps), (true, true));
        caps.limits.max_storage_buffers_per_shader_stage = 4;
        assert_eq!(fits(&graph, &caps), (false, true));
        graph.clear_roots();
        assert_eq!(fits(&graph, &caps), (true, true));
        assert_eq!(graph.len(), nodes);
    }

    #[test]
    fn component_cache_tracks_slab_readers_roots_and_storage() {
        let caps = caps();
        let arena = Arc::new(fusor_tile::Planner::new());
        let mut graph = new_graph(&arena);
        let shape = [Dim::ONE, Dim::Const(1024)];
        let input = buffer(&mut graph, 0, &shape);
        let middle = map(&mut graph, input, &shape, UnOp::Neg);
        let last = map(&mut graph, middle, &shape, UnOp::Neg);
        let direct = map(&mut graph, input, &shape, UnOp::Abs);
        let shared = map(&mut graph, middle, &shape, UnOp::Abs);
        graph.union(direct, shared).unwrap();
        let slab = composite(&mut graph, true, [middle, last]);
        graph.union(last, slab).unwrap();
        let mut ex = Extraction {
            sigma: selecting(&graph, [input, middle, direct, slab]),
            m: Default::default(),
            theta: [middle, last, direct, shared, slab]
                .into_iter()
                .map(|id| (id, SchedPoint::Point))
                .collect(),
        };
        let mut cache = NodeCache::default();
        for (reader, fused, expose_middle, storage, private) in [
            (direct, true, false, 16384, true),
            (shared, true, false, 16384, false),
            (direct, true, false, 16384, true),
            (direct, true, true, 16384, false),
            (direct, true, false, 0, false),
            (direct, false, false, 16384, false),
            (direct, true, false, 16384, true),
        ] {
            ex.sigma.insert(graph.class_of(direct), reader);
            ex.sigma
                .insert(graph.class_of(slab), if fused { slab } else { last });
            let mut roots = vec![slab, direct];
            if expose_middle {
                roots.push(middle);
            }
            let mut caps = caps.clone();
            caps.limits.max_compute_workgroup_storage_size = storage;
            let cost = PeakCost(crate::Roofline::new(crate::facts::seed_facts(&caps)));
            let selected = Selected::new(&graph, &ex, &roots, &mut cache).unwrap();
            ex.m = selected.buffers(&graph);
            let cached = selected
                .realize(&graph, &ex, &cost, arena.as_ref(), &mut cache)
                .unwrap();
            let fresh = realize(&graph, &roots, &ex, &cost, arena.as_ref()).unwrap();
            assert_eq!(cached, fresh);
            assert_eq!(
                exact_cost(&cached, &ex, &cost),
                exact_cost(&fresh, &ex, &cost)
            );
            assert_eq!(cached.components.len(), if fused { 2 } else { 3 });
            assert_eq!(
                cached
                    .components
                    .iter()
                    .any(|c| c.private.contains(&middle)),
                private
            );
            let repeated = Selected::new(&graph, &ex, &roots, &mut cache)
                .unwrap()
                .realize(&graph, &ex, &cost, arena.as_ref(), &mut cache)
                .unwrap();
            assert!(
                cached
                    .components
                    .iter()
                    .zip(&repeated.components)
                    .all(|(a, b)| Arc::ptr_eq(a, b))
            );
        }

        let roots = [slab, direct];
        let cost = PeakCost(crate::Roofline::new(crate::facts::seed_facts(&caps)));
        let expected = realize(&graph, &roots, &ex, &cost, arena.as_ref()).unwrap();
        ex.sigma.remove(&graph.class_of(input));
        assert!(realize_with(&graph, &roots, &ex, &cost, arena.as_ref(), &mut cache).is_err());
        ex.sigma.insert(graph.class_of(input), input);
        assert_eq!(
            realize_with(&graph, &roots, &ex, &cost, arena.as_ref(), &mut cache).unwrap(),
            expected,
        );

        let search = crate::LocalSearch::new(arena.clone(), caps.clone());
        let base = search
            .replan(&graph, &roots, &mut ex, &cost, &mut NodeCache::default())
            .unwrap();
        assert!(!base.buffers.iter().any(|buffer| buffer.value == middle));
        let extended_roots = [slab, direct, middle];
        graph.clear_roots();
        for root in extended_roots {
            graph.add_root(root);
        }
        let extended = search
            .extract_seeded(
                &graph,
                &extended_roots,
                &cost,
                ExtractBudget {
                    max_move_work: 0,
                    ..ExtractBudget::default()
                },
                &base,
            )
            .unwrap();
        assert_eq!(extended.extraction.sigma[&graph.class_of(slab)], slab);
        assert!(extended.buffers.iter().any(|buffer| buffer.value == middle));
        let fresh = realize(
            &graph,
            &extended_roots,
            &extended.extraction,
            &cost,
            arena.as_ref(),
        )
        .unwrap();
        assert_eq!(
            extended.cost,
            exact_cost(&fresh, &extended.extraction, &cost)
        );
        graph.clear_roots();

        let mut alternative = graph.node(middle).op.clone();
        let Op::Launch(Launch::Map { sched, .. }) = &mut alternative else {
            unreachable!()
        };
        *sched = ScheduleDomain::Map(
            fusor_ir::ir::launch::MapDomain {
                tilings: [1, 2]
                    .map(|tm| fusor_ir::ir::launch::MapTiling {
                        dim: None,
                        tm,
                        vector: 1,
                    })
                    .into_iter()
                    .collect(),
            }
            .into(),
        );
        let alternative = graph.add(alternative).unwrap();
        graph.union(middle, alternative).unwrap();
        ex.sigma = selecting(&graph, [input, middle, shared, slab]);
        let cost = PeakCost(crate::Roofline::new(crate::facts::seed_facts(&caps)));
        let bounds = crate::lower_bound::lower_bound(&graph, &cost);
        let candidates = |ex: &Extraction, order: &[Id], mv| {
            crate::moves::candidates(
                &graph,
                ex,
                order,
                mv,
                &bounds,
                &mut crate::moves::SchedCache::new(),
                &cost,
            )
        };
        for live_slab in [false, true] {
            let roots = if live_slab {
                vec![shared, slab]
            } else {
                vec![shared]
            };
            let selected = Selected::new(&graph, &ex, &roots, &mut NodeCache::default()).unwrap();
            assert_eq!(selected.order.contains(&slab), live_slab);
            let reselect = fusor_ir::extract::Move::Reselect(graph.class_of(middle));
            assert_eq!(
                candidates(&ex, &selected.order, reselect).contains(
                    &crate::moves::Candidate::Select {
                        class: graph.class_of(middle),
                        node: alternative,
                    }
                ),
                !live_slab,
            );
            let frontier = crate::moves::frontier(&graph, &selected.order);
            assert_eq!(
                frontier.contains(&fusor_ir::extract::Move::Reselect(graph.class_of(slab))),
                live_slab,
            );
        }
        ex.sigma.insert(graph.class_of(middle), alternative);
        ex.sigma.insert(graph.class_of(direct), direct);
        let selected = Selected::new(&graph, &ex, &[direct], &mut NodeCache::default()).unwrap();
        let frontier = crate::moves::frontier(&graph, &selected.order);
        assert!(!frontier.contains(&fusor_ir::extract::Move::Reschedule(alternative)));
        for stale in [
            fusor_ir::extract::Move::Reschedule(alternative),
            fusor_ir::extract::Move::Reselect(graph.class_of(middle)),
        ] {
            assert!(candidates(&ex, &selected.order, stale).is_empty());
        }
    }

    #[test]
    fn point_folds_price_the_emitted_lanes_grid_and_scratch() {
        use fusor_ir::carrier::{Carrier, oracle};
        use fusor_ir::ir::launch::FoldDomain;
        use fusor_ir::scalar::BinOp;

        let caps = caps();
        let arena = Arc::new(fusor_tile::Planner::new());
        let cost = crate::Roofline::new(crate::facts::seed_facts(&caps));
        let sum = Carrier::binop(
            BinOp::Add,
            Carrier::binop_identity(BinOp::Add, Dtype::F32).unwrap(),
            Dtype::F32,
        );
        let mut cache = NodeCache::default();
        for (shape, axis, rows) in [([1, 1, 4096], 2, 1), ([5, 4096, 3], 1, 15)] {
            let mut graph = new_graph(&arena);
            let shape = shape.map(Dim::Const);
            let input = buffer(&mut graph, 0, &shape);
            let mut costs = Vec::new();
            let tree = SchedPoint::Fold(FoldStrat::WgTree { lane_group: 256 });
            for (carrier, theta, block, steps, scratch) in [
                (sum.clone(), SchedPoint::Point, 32, 128, 0),
                (
                    sum.clone(),
                    SchedPoint::Fold(FoldStrat::Subgroup),
                    32,
                    128,
                    0,
                ),
                (sum.clone(), tree, 256, 16, 1024),
                (
                    oracle::welford(Dtype::F32),
                    SchedPoint::Point,
                    256,
                    16,
                    3072,
                ),
            ] {
                let post = (0..carrier.width())
                    .map(|i| ScalarExpr::arg(i as u32, Dtype::F32))
                    .collect();
                let sched = match theta {
                    SchedPoint::Fold(s) => ScheduleDomain::Fold(
                        FoldDomain {
                            strategies: [s].into_iter().collect(),
                        }
                        .into(),
                    ),
                    _ => ScheduleDomain::Point,
                };
                let node = graph
                    .add(Op::Launch(Launch::Fold {
                        space: IndexSpace::new(shape),
                        axis,
                        vec_axes: Default::default(),
                        carrier,
                        acc: Dtype::F32,
                        post,
                        ops: vec![alias(input, Layout::contiguous(&shape))],
                        sched,
                    }))
                    .unwrap();
                let mut ex = Extraction {
                    sigma: selecting(&graph, [input, node]),
                    m: Default::default(),
                    theta: [(node, theta)].into_iter().collect(),
                };
                let selected = Selected::new(&graph, &ex, &[node], &mut cache).unwrap();
                ex.m = selected.buffers(&graph);
                let realized = selected
                    .realize(&graph, &ex, &cost, arena.as_ref(), &mut cache)
                    .unwrap();
                let c = &realized.components[0];
                assert_eq!(c.grid, [rows, 1, 1]);
                assert_eq!(c.block, block);
                assert_eq!(c.resident_lanes, u64::from(rows * block));
                assert_eq!(c.lane_steps, steps);
                assert_eq!(c.wg_bytes, scratch);
                costs.push(exact_cost(&realized, &ex, &cost));
            }
            assert_eq!(costs[0], costs[1]);
            assert!(costs[2] < costs[0]);
        }
    }

    #[test]
    fn matvec_traffic_counts_each_row_and_subgroup_activation_window() {
        let caps = caps();
        let arena = Arc::new(fusor_tile::Planner::new());
        let mut facts = crate::facts::seed_facts(&caps);
        facts.llc_bytes = 0;
        facts.dram_bytes_per_us = 1;
        facts.saturation_lanes = 1;
        let cost = crate::Roofline::new(facts);
        let (batch, n, k) = (2, 4097, 4096);
        let mut costs = Vec::new();
        for m in [1, 16] {
            for cols in [1, 4] {
                let mut graph = new_graph(&arena);
                let [a_shape, b_shape] = [[batch, m, k], [batch, k, n]].map(|d| d.map(Dim::Const));
                let a = buffer(&mut graph, 0, &a_shape);
                let b = buffer(&mut graph, 1, &b_shape);
                let p = SgemvParams {
                    vector: 4,
                    subgroups: 2,
                    cols,
                    parts: 1,
                    gap: 0,
                };
                let node = graph
                    .add(contract(
                        &[batch, m, n].map(Dim::Const),
                        [m, n, k, batch].map(Dim::Const),
                        Family::Sgemv,
                        side(a, Layout::contiguous(&a_shape)),
                        side(b, Layout::contiguous(&b_shape)),
                        sgemv(p),
                    ))
                    .unwrap();
                let mut ex = Extraction {
                    sigma: selecting(&graph, [a, b, node]),
                    m: Default::default(),
                    theta: [(node, SchedPoint::Sgemv(p))].into_iter().collect(),
                };
                let search = crate::LocalSearch::new(arena.clone(), caps.clone());
                let plan = search
                    .replan(&graph, &[node], &mut ex, &cost, &mut NodeCache::default())
                    .unwrap();
                let realized = realize(&graph, &[node], &ex, &cost, arena.as_ref()).unwrap();
                let component = &realized.components[0];
                let a_scans = if cols == 1 { n } else { n.div_ceil(4) * 2 };
                assert_eq!(
                    component.reads,
                    [
                        (batch * m * k * 4, a_scans as u32),
                        (batch * k * n * 4, m as u32)
                    ]
                );
                let loaded: u64 = component
                    .reads
                    .iter()
                    .map(|(bytes, scans)| bytes * u64::from(*scans))
                    .sum();
                assert_eq!(loaded, batch * m * k * (a_scans + n) * 4);
                // B is `[k, n]`: every element a lane loads along k is a line.
                assert_eq!(
                    component.line_bytes,
                    batch * k * n * 4 * m * (LINE_BYTES / 4 - 1)
                );
                costs.push(plan.cost);
            }
        }
        assert!(costs[1] < costs[0]);
        assert!(costs[3] < costs[2]);
        assert!(costs[2] > costs[0]);
    }

    #[test]
    fn symbolic_attention_cost_matches_concrete_nominal_shapes() {
        use fusor_ir::carrier::Carrier;
        use fusor_ir::ir::launch::FoldDomain;
        use fusor_ir::scalar::BinOp;
        use fusor_ir::shape::SymId;

        let caps = caps();
        let arena = Arc::new(fusor_tile::Planner::new());
        let cost = crate::Roofline::new(crate::facts::seed_facts(&caps));
        let s = Dim::Sym(SymId(0));
        let (c4, c8, c128) = (Dim::Const(4), Dim::Const(8), Dim::Const(128));
        for (symbolic, concrete) in [(s, 1024), (s * Dim::Const(2) + Dim::Sym(SymId(1)), 3072)] {
            for case in 0..4 {
                let mut results = Vec::new();
                for len in [symbolic, Dim::Const(concrete)] {
                    let mut graph = new_graph(&arena);
                    let (op, theta) = if case < 2 {
                        let (n, k) = if case == 0 { (len, c128) } else { (c128, len) };
                        let a = buffer(&mut graph, 0, &[c8, c4, k]);
                        let b = buffer(&mut graph, 1, &[c8, k, n]);
                        let p = SgemvParams {
                            vector: 4,
                            subgroups: 4,
                            cols: 4,
                            parts: 1,
                            gap: 0,
                        };
                        let op = contract(
                            &[c8, c4, n],
                            [c4, n, k, c8],
                            Family::Sgemv,
                            side(a, Layout::contiguous(&[c8, c4, k])),
                            side(b, Layout::contiguous(&[c8, k, n])),
                            sgemv(p),
                        );
                        (op, SchedPoint::Sgemv(p))
                    } else {
                        let (shape, axis) = if case == 2 {
                            ([c8, c4, len], 2)
                        } else {
                            ([c8, len, c4], 1)
                        };
                        let source = buffer(&mut graph, 0, &shape);
                        let strategy = FoldStrat::WgTree { lane_group: 256 };
                        let op = Op::Launch(Launch::Fold {
                            space: IndexSpace::new(shape),
                            axis,
                            vec_axes: Default::default(),
                            carrier: Carrier::binop(
                                BinOp::Add,
                                Carrier::binop_identity(BinOp::Add, Dtype::F32).unwrap(),
                                Dtype::F32,
                            ),
                            acc: Dtype::F32,
                            post: [ScalarExpr::arg(0, Dtype::F32)].into_iter().collect(),
                            ops: vec![alias(source, Layout::contiguous(&shape))],
                            sched: ScheduleDomain::Fold(
                                FoldDomain {
                                    strategies: [strategy].into_iter().collect(),
                                }
                                .into(),
                            ),
                        });
                        (op, SchedPoint::Fold(strategy))
                    };
                    let node = graph.add(op).unwrap();
                    let mut ex = Extraction {
                        sigma: selecting(&graph, (0..=node.0).map(Id)),
                        m: Default::default(),
                        theta: [(node, theta)].into_iter().collect(),
                    };
                    ex.m = Selected::new(&graph, &ex, &[node], &mut NodeCache::default())
                        .unwrap()
                        .buffers(&graph);
                    let realized = realize(&graph, &[node], &ex, &cost, arena.as_ref()).unwrap();
                    let c = &realized.components[0];
                    assert_eq!(
                        c.work.macs,
                        if case < 2 {
                            32 * 128 * concrete
                        } else {
                            32 * concrete
                        }
                    );
                    results.push((
                        exact_cost(&realized, &ex, &cost),
                        c.work.macs,
                        c.work.index_ops,
                        c.reads.clone(),
                        c.writes,
                        c.grid,
                        c.resident_lanes,
                        c.lane_steps,
                        c.line_bytes,
                    ));
                }
                assert_eq!(results[0], results[1], "case {case}, extent {symbolic}");
            }
        }
    }

    #[test]
    fn cooperative_traffic_counts_selected_tiles_and_padded_k() {
        let caps = caps();
        let arena = Arc::new(fusor_tile::Planner::new());
        let mut graph = new_graph(&arena);
        let [a_shape, b_shape] = [[3, 17, 17], [3, 17, 65]].map(|d| d.map(Dim::Const));
        let a_src = buffer(&mut graph, 0, &a_shape);
        let b_src = buffer(&mut graph, 1, &b_shape);
        let mut a = side(a_src, Layout::contiguous(&a_shape));
        a.pre = ScalarExpr::un(UnOp::Neg, a.pre);
        let first = CoopSchedule {
            geom: CoopGeom {
                bm: 16,
                bn: 16,
                bk: 8,
                n_passes: 1,
                subgroups: 1,
                rg: 1,
                cg: 1,
            },
            staging: 1,
        };
        let chosen = CoopSchedule {
            geom: CoopGeom {
                bn: 32,
                n_passes: 2,
                subgroups: 2,
                rg: 2,
                ..first.geom
            },
            staging: 2,
        };
        let schedules = [
            first,
            chosen,
            CoopSchedule {
                staging: 1,
                ..chosen
            },
        ];
        let node = graph
            .add(contract(
                &b_shape,
                [17, 65, 17, 3].map(Dim::Const),
                Family::Coop,
                a,
                side(b_src, Layout::contiguous(&b_shape)),
                ScheduleDomain::Coop(
                    CoopDomain {
                        schedules: schedules.into_iter().collect(),
                    }
                    .into(),
                ),
            ))
            .unwrap();
        let point = |staging| SchedPoint::Coop {
            geom: chosen.geom,
            staging,
        };
        let mut ex = Extraction {
            sigma: selecting(&graph, [a_src, b_src, node]),
            m: Default::default(),
            theta: [(node, point(chosen.staging))].into_iter().collect(),
        };
        ex.m = Selected::new(&graph, &ex, &[node], &mut NodeCache::default())
            .unwrap()
            .buffers(&graph);
        let cost = crate::Roofline::new(crate::facts::seed_facts(&caps));
        let mut cache = NodeCache::default();
        let realized = |ex: &Extraction, cache: &mut NodeCache| {
            let plan = Selected::new(&graph, ex, &[node], &mut NodeCache::default())
                .unwrap()
                .realize(&graph, ex, &cost, arena.as_ref(), cache)
                .unwrap();
            assert_eq!(
                plan,
                realize(&graph, &[node], ex, &cost, arena.as_ref()).unwrap()
            );
            plan
        };
        let plan = realized(&ex, &mut cache);
        let component = &plan.components[0];
        // 18 groups, two N passes, two 16-wide K steps. Each step writes
        // 512 operand elements and reads 256 A + 512 B elements.
        assert_eq!(component.work.wg_bytes, 18 * 2 * 2 * (512 + 256 + 512) * 4);
        assert_eq!(component.work.macs, 18 * 16 * 32 * 32 + 3 * 17 * 17 * 6);
        assert_eq!(component.writes, 18 * 16 * 32 * 4);
        assert_eq!(
            component.line_bytes,
            3 * 17 * (17 * 6 + 65 * 2 - 17 - 65) * 4
        );
        assert!(component.work.wg_bytes > component.wg_bytes * 100);
        ex.theta.insert(node, point(1));
        let single = realized(&ex, &mut cache);
        assert_eq!(
            single.components[0].work.wg_bytes,
            component.work.wg_bytes * 3 / 4
        );

        use fusor_ir::ir::kernel::{ArenaPlan, BarrierSuggestion, KernelIr};
        struct Scratch<const BYTES: u32>;
        impl<const BYTES: u32> ArenaPlanner for Scratch<BYTES> {
            fn workgroup_bytes(&self, _: &Tiles, _: &Caps) -> Result<u32> {
                Ok(BYTES)
            }
            fn arena_plan(&self, ir: &KernelIr, caps: &Caps) -> Result<ArenaPlan> {
                fusor_tile::domains::default_planner().arena_plan(ir, caps)
            }
            fn barrier_suggestions(&self, ir: &KernelIr) -> Vec<BarrierSuggestion> {
                fusor_tile::domains::default_planner().barrier_suggestions(ir)
            }
            fn verify_arena(&self, ir: &KernelIr, plan: &ArenaPlan) -> Result<()> {
                fusor_tile::domains::default_planner().verify_arena(ir, plan)
            }
            fn verify_uniformity(&self, ir: &KernelIr) -> Result<()> {
                fusor_tile::domains::default_planner().verify_uniformity(ir)
            }
        }
        let planners: [Box<dyn ArenaPlanner>; 2] =
            [Box::new(Scratch::<128>), Box::new(Scratch::<256>)];
        assert_eq!(
            planners[0].as_ref() as *const dyn ArenaPlanner as *const (),
            planners[1].as_ref() as *const dyn ArenaPlanner as *const ()
        );
        let mut cache = NodeCache::default();
        for (planner, bytes) in planners.iter().zip([128, 256]) {
            let plan =
                realize_with(&graph, &[node], &ex, &cost, planner.as_ref(), &mut cache).unwrap();
            assert_eq!(plan.components[0].wg_bytes, bytes);
        }
    }

    #[test]
    fn replacement_prices_shared_dag_and_selects_memory_savings_over_math() {
        let caps = caps();
        let mut facts = crate::facts::seed_facts(&caps);
        facts.dram_bytes_per_us = 1;
        facts.llc_bytes = 0;
        facts.saturation_lanes = 1;
        facts.lane_step_ps = 0;
        let cost = PeakCost(crate::Roofline::new(facts));
        let arena = Arc::new(fusor_tile::Planner::new());
        let mut graph = new_graph(&arena);
        let shape = [Dim::Const(16), Dim::Const(256)];
        let flipped = [shape[1], shape[0]];
        let input = buffer(&mut graph, 0, &shape);
        let weights = buffer(&mut graph, 1, &flipped);
        let shared = map(&mut graph, input, &shape, UnOp::Neg);
        let other = map(&mut graph, shared, &shape, UnOp::Neg);
        let mut contract_at = |tile, thread| {
            let params = SgemmParams {
                double_buffer: false,
                bm: tile,
                bn: tile,
                bk: 8,
                tm: thread,
                tn: thread,
            };
            let node = graph
                .add(contract(
                    &[shape[0]; 2],
                    [shape[0], shape[0], shape[1], Dim::ONE],
                    Family::Sgemm,
                    side(shared, Layout::contiguous(&shape)),
                    side(weights, Layout::contiguous(&flipped)),
                    sgemm(params),
                ))
                .unwrap();
            (node, SchedPoint::Sgemm(params))
        };
        let (small, small_theta) = contract_at(8, 1);
        let (large, large_theta) = contract_at(32, 4);
        let search = crate::LocalSearch::new(arena.clone(), caps.clone());
        let budget = ExtractBudget {
            max_move_work: 0,
            ..ExtractBudget::default()
        };
        let initial = search
            .extract(&graph, &[small, other], &cost, budget)
            .unwrap();
        graph.union(small, large).unwrap();
        let roots = [small, other];
        let mut ex = Extraction {
            sigma: selecting(&graph, [input, weights, shared, other, small]),
            m: Default::default(),
            theta: [(small, small_theta), (large, large_theta)]
                .into_iter()
                .collect(),
        };
        let buffers = |ex: &Extraction| {
            Selected::new(&graph, ex, &roots, &mut NodeCache::default())
                .unwrap()
                .buffers(&graph)
        };
        ex.m = buffers(&ex);
        let before = realize(&graph, &roots, &ex, &cost, arena.as_ref()).unwrap();
        let launch_roots = |r: &Realized| r.components.iter().map(|c| c.root).collect::<Vec<_>>();
        assert_eq!(launch_roots(&before), [shared, small, other]);
        let before_cost = exact_cost(&before, &ex, &cost);
        ex.sigma.insert(graph.class_of(small), large);
        ex.m = buffers(&ex);
        let replacement = ordinary_component(
            &graph,
            &ex,
            large,
            &cost,
            arena.as_ref(),
            &mut NodeCache::default(),
        )
        .unwrap();
        let after = realize(&graph, &roots, &ex, &cost, arena.as_ref()).unwrap();
        let after_cost = exact_cost(&after, &ex, &cost);
        assert_eq!(before.components[1].reads, [(16384, 16), (16384, 16)]);
        assert_eq!(replacement.reads, [(16384, 16), (16384, 4)]);
        assert_eq!(
            (before.components[1].line_bytes, replacement.line_bytes),
            (0, 0)
        );
        assert_eq!(launch_roots(&after), [shared, large, other]);
        assert_eq!(
            before.cost_replacing(1, &replacement, &ex, &cost),
            after_cost
        );
        assert!(after_cost < before_cost);
        assert_ne!(
            before_cost - cost.launch_cost(&before.components[1].launch(&ex))
                + cost.launch_cost(&replacement.launch(&ex)),
            after_cost
        );

        let bounds = search.lower_bound(&graph, &cost);
        assert!(bounds[small.index()] < bounds[large.index()]);
        let chosen = search.seed(&graph, &roots, &bounds, &cost).unwrap();
        assert_eq!(chosen.selected(graph.class_of(small)), Some(large));
        let extended = search
            .extract_seeded(&graph, &roots, &cost, budget, &initial)
            .unwrap();
        assert_eq!(
            extended.extraction.selected(graph.class_of(small)),
            Some(large)
        );
        assert_eq!(extended.cost, after_cost);

        let SchedPoint::Sgemm(large_params) = large_theta else {
            unreachable!()
        };
        let third_theta = SgemmParams {
            tm: 2,
            tn: 2,
            ..large_params
        };
        let mut op = graph.node(large).op.clone();
        let Op::Launch(Launch::Contract { sched, .. }) = &mut op else {
            unreachable!()
        };
        *sched = sgemm(third_theta);
        let third = graph.add(op).unwrap();
        graph.union(small, third).unwrap();
        ex.sigma.insert(graph.class_of(small), third);
        ex.theta.insert(third, SchedPoint::Sgemm(third_theta));
        let base = search
            .replan(&graph, &roots, &mut ex, &cost, &mut NodeCache::default())
            .unwrap();
        let ranked = search.launch_variant_labels(&graph, &roots, &base, 1, &cost, 0);
        let mut derived: Vec<_> = search
            .launch_variants(&graph, &roots, &base, 1, &cost, 0)
            .into_iter()
            .map(|(label, plan)| (label, plan.cost))
            .collect();
        derived.sort_by(|(a, ac), (b, bc)| ac.cmp(bc).then_with(|| a.cmp(b)));
        assert_eq!(ranked.len(), 2);
        assert_eq!(ranked, derived);
        assert_eq!(ranked[0].1, after_cost);
        assert_eq!(ranked[1].1, before_cost);
        assert_eq!(
            ranked,
            search.launch_variant_labels(&graph, &roots, &base, 1, &cost, 0)
        );
    }
}
