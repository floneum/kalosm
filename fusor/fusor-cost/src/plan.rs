//! Plan derivation: buffers, bindings, symbols and the plan hash.
//!
//! Allocation is derived from the plan: for each node in `M`,
//! `buffer_layout` gives the padded strides the selected geometry needs,
//! including split-K scratch slices. A value not in `M` gets no buffer at all.
//!
//! The plan is the cache key. `Dim::Sym` and `LeafKind::Uniform` hash as
//! the symbol's index, not its bound value, so one plan serves a whole
//! shape family.

use crate::realize::{self, Component, Realized};
use fusor_ir::Result;
use fusor_ir::cost::DeviceFacts;
use fusor_ir::dtype::Persistence;
use fusor_ir::egraph::{EGraph, Id};
use fusor_ir::error::Error;
use fusor_ir::extract::{BindKind, BindingPlan, BufferPlan, Dispatch, Extraction, Plan, PlanHash};
use fusor_ir::facts::ValueFacts;
use fusor_ir::ir::Op;
use fusor_ir::ir::launch::{Effect, Launch, SchedPoint, ScheduleDomain};
use fusor_ir::ir::logical::{LeafKind, Logical};
use fusor_ir::ir::visit::VisitMut;
use fusor_ir::scalar::{ScalarExpr, ScalarKind};
use fusor_ir::shape::{Dim, Dims, Layout, OPAQUE_SYM, SymId};
use rustc_hash::FxHasher;
use rustc_hash::{FxHashMap, FxHashSet};
use std::hash::{Hash, Hasher};

/// Everything derived from one realized extraction: buffers, launches,
/// symbols, hash and cost.
pub fn derive_plan(
    graph: &EGraph,
    extraction: &Extraction,
    realized: &Realized,
    facts: &DeviceFacts,
    cost: fusor_ir::cost::Picoseconds,
) -> Result<Plan> {
    let buffers = derive_buffers(graph, extraction, realized)?;
    let (mut dims, scalar_symbols) = classified_symbols_of(graph, realized);
    for buffer in &buffers {
        collect_layout(
            &Layout::contiguous(&graph.facts(buffer.value).shape),
            &mut dims,
        );
        collect_layout(&buffer.layout, &mut dims);
        collect_dims(&[buffer.elements], &mut dims);
    }
    dims.retain(|s| *s != OPAQUE_SYM);
    dims.sort_unstable();
    dims.dedup();
    let mut symbols = dims;
    symbols.extend(scalar_symbols.iter().copied());

    let mut launches = Vec::with_capacity(realized.components.len());
    let private: FxHashSet<Id> = realized
        .components
        .iter()
        .flat_map(|c| c.private.iter().copied())
        .collect();
    for c in &realized.components {
        // A slab member kept in workgroup memory is read by nothing outside
        // that slab; a launch binding one would read a buffer nothing wrote.
        if let Some(r) = c.external.iter().find(|r| private.contains(r)) {
            return Err(Error::Plan(format!(
                "launch {} reads {r}, a slab member kept in workgroup memory",
                c.root
            )));
        }
        launches.push(Dispatch {
            root: c.root,
            members: c.members.iter().copied().collect(),
            bindings: derive_bindings(graph, extraction, realized, c)?,
            grid: c.grid,
            block: c.block,
        });
    }

    let mut buffers = buffers;
    let arena_bytes = if facts.caps.kind == fusor_ir::device::DeviceKind::Gpu {
        pack_arena(&mut buffers, &mut launches, realized)?
    } else {
        0
    };
    let hash = plan_hash(graph, extraction, &launches, &buffers, &symbols, facts);
    Ok(Plan {
        extraction: extraction.clone(),
        launches,
        buffers,
        arena_bytes,
        symbols,
        scalar_symbols,
        hash,
        cost,
    })
}

/// Interval-color the step-local intermediates into one arena: each is
/// live from the first launch binding it to the last, and two whose ranges
/// are disjoint take the same bytes. A launch binds the arena once; typed
/// views reinterpret values through that binding. Returns the arena's size.
fn pack_arena(
    buffers: &mut [BufferPlan],
    launches: &mut [Dispatch],
    realized: &Realized,
) -> Result<u64> {
    const ALIGN: u64 = 256;
    let mut first: FxHashMap<Id, usize> = FxHashMap::default();
    let mut last: FxHashMap<Id, usize> = FxHashMap::default();
    for (ix, l) in launches.iter().enumerate() {
        for b in &l.bindings {
            first.entry(b.value).or_insert(ix);
            last.insert(b.value, ix);
        }
    }
    // Candidates: step-local, constant extent, not read back by the caller.
    let mut items: Vec<(usize, usize, u64, usize)> = Vec::new();
    for (i, b) in buffers.iter().enumerate() {
        if b.persistence != Persistence::Step || realized.is_root(b.value) {
            continue;
        }
        let Some(elements) = b.elements.as_const() else {
            continue;
        };
        let (Some(s), Some(e)) = (first.get(&b.value), last.get(&b.value)) else {
            continue;
        };
        let bytes = elements
            .saturating_mul(b.dtype.byte_size())
            .max(4)
            .div_ceil(ALIGN)
            * ALIGN;
        items.push((*s, *e, bytes, i));
    }
    items.sort_unstable_by_key(|(s, e, bytes, i)| (*s, *e, std::cmp::Reverse(*bytes), *i));
    let sizes: Vec<_> = items
        .iter()
        .map(|(_, _, bytes, _)| (*bytes, ALIGN))
        .collect();
    let (top, offsets) =
        fusor_ir::packing::pack_interference(&sizes, fusor_ir::packing::Fit::Best, |i, j| {
            items[i].0 <= items[j].1 && items[j].0 <= items[i].1
        })?;
    for ((_, _, _, i), offset) in items.iter().zip(offsets) {
        buffers[*i].arena = Some(offset);
    }
    let in_arena: FxHashSet<Id> = buffers
        .iter()
        .filter(|b| b.arena.is_some())
        .map(|b| b.value)
        .collect();
    // One binding per physical buffer. Multiple writable bindings spanning
    // the same arena are invalid in WebGPU, even for different element types.
    for l in launches.iter_mut() {
        let mut arena_binding = None;
        let mut next = 1u32;
        for b in &mut l.bindings {
            b.arena = in_arena.contains(&b.value);
            b.binding = if b.arena {
                b.kind = BindKind::ReadWrite;
                *arena_binding.get_or_insert_with(|| {
                    let n = next;
                    next += 1;
                    n
                })
            } else {
                let n = next;
                next += 1;
                n
            };
        }
    }
    Ok(top)
}

/// One [`BufferPlan`] per node in `m ∪ roots`, in realized order. Leaves are
/// excluded: an external buffer is supplied, a constant is folded, and a
/// uniform lives in binding 0.
pub fn derive_buffers(
    graph: &EGraph,
    extraction: &Extraction,
    realized: &Realized,
) -> Result<Vec<BufferPlan>> {
    let private: FxHashSet<Id> = realized
        .components
        .iter()
        .flat_map(|c| c.private.iter().copied())
        .collect();
    let mut out = Vec::new();
    for id in &realized.order {
        if realize::leaf_role(graph, *id) != realize::LeafRole::NotLeaf {
            continue;
        }
        if !extraction.is_materialized(*id) && !realized.is_root(*id) || private.contains(id) {
            continue;
        }
        let facts = graph.facts(*id);
        let theta = extraction.theta.get(id).copied();
        let (layout, elements) = buffer_layout_for(facts, &graph.node(*id).op, theta)?;
        out.push(BufferPlan {
            value: *id,
            elements,
            layout,
            dtype: facts.dtype,
            persistence: facts.persistence,
            arena: None,
        });
    }
    Ok(out)
}

/// Bindings of one launch, in binding-index order. **Binding 0 is reserved
/// for the uniform block** and is never listed here; storage bindings start
/// at 1, reads first sorted by value id, then writes.
pub fn derive_bindings(
    graph: &EGraph,
    extraction: &Extraction,
    realized: &Realized,
    component: &Component,
) -> Result<Vec<BindingPlan>> {
    let mut writes: Vec<Id> = component
        .members
        .iter()
        .copied()
        .filter(|m| {
            (extraction.is_materialized(*m) || realized.is_root(*m))
                && !component.private.contains(m)
        })
        .collect();
    writes.sort_unstable();
    writes.dedup();

    let mut reads: Vec<Id> = component.external.clone();
    reads.retain(|r| realize::leaf_role(graph, *r) != realize::LeafRole::Free);
    reads.sort_unstable();
    reads.dedup();
    // An in-place value is bound once, read-write; it must not appear twice.
    reads.retain(|r| !writes.contains(r));

    let mut out = Vec::with_capacity(reads.len() + writes.len());
    let mut binding = 1u32;
    for value in reads {
        out.push(BindingPlan {
            binding,
            value,
            kind: BindKind::Read,
            arena: false,
        });
        binding += 1;
    }
    for value in writes {
        let kind = match graph.semantics().effect(&graph.node(value).op) {
            Effect::InPlace(_) => BindKind::ReadWrite,
            Effect::Pure => BindKind::Write,
        };
        out.push(BindingPlan {
            binding,
            value,
            kind,
            arena: false,
        });
        binding += 1;
    }
    Ok(out)
}

/// Logical strides and allocation extent for a selected node. Cooperative
/// stores pad matrix groups, while retaining each group's logical axes.
pub fn buffer_layout_for(
    facts: &ValueFacts,
    op: &Op,
    theta: Option<SchedPoint>,
) -> Result<(Layout, Dim)> {
    let shape = &facts.shape;
    let (Op::Launch(Launch::Contract { m, n, batch, .. }), Some(SchedPoint::Coop { geom, .. })) =
        (op, theta)
    else {
        return Ok((Layout::contiguous(shape), layout_elements(shape)));
    };
    let constant = |dim: Dim| {
        dim.as_const()
            .ok_or_else(|| Error::Plan("cooperative matrix groups require concrete extents".into()))
    };
    let (m, n) = (constant(*m)?, constant(*n)?);
    let m_padded = m.max(1).div_ceil(u64::from(geom.bm)) * u64::from(geom.bm);
    let n_padded = n.max(1).div_ceil(u64::from(geom.bn)) * u64::from(geom.bn);
    let elements = *batch * Dim::Const(m_padded) * Dim::Const(n_padded);
    if m == 0 || n == 0 {
        return Ok((Layout::contiguous(shape), elements));
    }
    let mut strides: Dims = shape.clone();
    let mut axis = shape.len();
    for (extent, stride) in [(n, 1), (m, n_padded)] {
        let mut covered = 1;
        while covered < extent {
            axis = axis.checked_sub(1).ok_or_else(|| {
                Error::Plan("matrix group exceeds its logical output rank".into())
            })?;
            strides[axis] = Dim::Const(stride * covered);
            covered *= constant(shape[axis])?;
        }
        if covered != extent {
            return Err(Error::Plan(
                "matrix group cuts through a logical output axis".into(),
            ));
        }
    }
    let mut stride = Dim::Const(m_padded * n_padded);
    for axis in (0..axis).rev() {
        strides[axis] = stride;
        stride = stride * shape[axis];
    }
    Ok((
        Layout::from_parts(Dim::Const(0), shape, &strides)?,
        elements,
    ))
}

fn layout_elements(shape: &[Dim]) -> Dim {
    shape.iter().copied().fold(Dim::ONE, |a, b| a * b)
}

/// Every `SymId` the uniform block must carry, in binding order: dims
/// ascending, then scalars ascending, matching `Uniforms::to_bytes`.
pub fn symbols_of(graph: &EGraph, realized: &Realized) -> Vec<SymId> {
    let (mut dims, scalars) = classified_symbols_of(graph, realized);
    dims.extend(scalars);
    dims
}

/// [`symbols_of`] split into `(dims, scalars)`: the extents, offsets and
/// strides the kernels index by, and the runtime scalars they read.
pub fn classified_symbols_of(graph: &EGraph, realized: &Realized) -> (Vec<SymId>, Vec<SymId>) {
    let mut visitor = Symbols::default();
    for id in &realized.order {
        collect_dims(&graph.facts(*id).shape, &mut visitor.dims);
        graph.node(*id).op.clone().visit_mut(&mut visitor);
    }
    let Symbols {
        mut dims,
        mut scalars,
        ..
    } = visitor;

    dims.retain(|s| *s != OPAQUE_SYM);
    scalars.retain(|s| *s != OPAQUE_SYM);
    dims.sort_unstable();
    dims.dedup();
    let mut index = 0;
    while index < dims.len() {
        if let Some(fusor_ir::shape::DimExpr::Add(a, b) | fusor_ir::shape::DimExpr::Mul(a, b)) =
            dims[index].derived_expr()
        {
            for dim in [a, b] {
                if let Dim::Sym(sym) = dim
                    && !dims.contains(&sym)
                {
                    dims.push(sym);
                }
            }
        }
        index += 1;
    }
    dims.sort_unstable();
    scalars.sort_unstable();
    scalars.dedup();
    // A symbol used as an extent is bound as a dim; it must not also be
    // emitted as a scalar.
    scalars.retain(|s| !dims.contains(s));
    (dims, scalars)
}

/// `hash(realized DAG term + M + theta + DeviceFacts::fingerprint)`.
///
/// Two `FxHasher` lanes under seeds 0 and 1, folded into a `u128`. Walk
/// launches in order, then members in order; `Dim::Sym(s)` and
/// `LeafKind::Uniform { sym }` hash as the symbol's index in `symbols`,
/// never its bound value.
pub fn plan_hash(
    graph: &EGraph,
    extraction: &Extraction,
    launches: &[Dispatch],
    buffers: &[BufferPlan],
    symbols: &[SymId],
    facts: &DeviceFacts,
) -> PlanHash {
    let mut lanes = [FxHasher::default(), FxHasher::default()];
    for (seed, lane) in lanes.iter_mut().enumerate() {
        lane.write_u64(seed as u64);
    }
    let mut normalize = Normalize {
        symbols,
        scalars: FxHashMap::default(),
    };
    for b in buffers {
        let mut layout = b.layout.clone();
        layout.visit_dims_mut(&mut |d| normalize.dim(d));
        let mut elements = b.elements;
        normalize.dim(&mut elements);
        hash_both(&mut lanes, &(b.value, layout, elements, b.arena));
    }
    for launch in launches {
        hash_both(&mut lanes, &(launch.root, launch.grid, launch.block));
        for b in &launch.bindings {
            hash_both(&mut lanes, &(b.binding, b.value, b.kind as u8));
        }
        for member in &launch.members {
            let mut op = without_schedule(&graph.node(*member).op);
            op.visit_mut(&mut normalize);
            hash_both(&mut lanes, &(*member, op));
            // External buffer names do not affect a positionally rebound kernel.
            for child in graph.node(*member).children.iter() {
                if let Op::Logical(Logical::Leaf(kind)) = &graph.node(*child).op {
                    let mut leaf = Op::Logical(Logical::Leaf(kind.clone()));
                    leaf.visit_mut(&mut normalize);
                    hash_both(&mut lanes, &leaf);
                }
            }
            hash_both(
                &mut lanes,
                &(
                    extraction.is_materialized(*member),
                    extraction.theta.get(member),
                ),
            );
        }
    }
    hash_both(&mut lanes, &facts.fingerprint());
    PlanHash(((lanes[0].finish() as u128) << 64) | lanes[1].finish() as u128)
}

fn hash_both(lanes: &mut [FxHasher; 2], value: &impl Hash) {
    for lane in lanes {
        value.hash(lane);
    }
}

/// Candidate domains do not enter the key; the selected schedule does.
pub(crate) fn without_schedule(op: &Op) -> Op {
    let mut op = op.clone();
    if let Op::Launch(
        Launch::Map { sched, .. }
        | Launch::Fold { sched, .. }
        | Launch::StreamFold { sched, .. }
        | Launch::Contract { sched, .. }
        | Launch::Gather { sched, .. }
        | Launch::Scatter { sched, .. }
        | Launch::Slab { sched, .. }
        | Launch::Group { sched, .. },
    ) = &mut op
    {
        *sched = ScheduleDomain::Point;
    }
    op
}

struct Normalize<'a> {
    symbols: &'a [SymId],
    scalars: FxHashMap<u64, ScalarExpr>,
}

impl Normalize<'_> {
    fn symbol(&self, sym: SymId) -> SymId {
        SymId(
            self.symbols
                .iter()
                .position(|s| *s == sym)
                .map_or(u32::MAX, |i| i as u32),
        )
    }

    fn expression(&mut self, expr: &ScalarExpr) -> ScalarExpr {
        let key = expr.structural_hash();
        if let Some(hit) = self.scalars.get(&key) {
            return hit.clone();
        }
        let normalized = match expr.kind() {
            ScalarKind::Uniform(sym) => ScalarExpr::uniform(self.symbol(*sym), expr.dtype()),
            _ => expr.map_children(&mut |child| self.expression(child)),
        };
        self.scalars.insert(key, normalized.clone());
        normalized
    }
}

impl VisitMut for Normalize<'_> {
    fn dim(&mut self, dim: &mut Dim) {
        if let Dim::Sym(sym) = dim {
            *sym = self.symbol(*sym);
        }
    }
    fn scalar(&mut self, expr: &mut ScalarExpr) {
        *expr = self.expression(expr);
    }
    fn leaf(&mut self, leaf: &mut LeafKind) {
        match leaf {
            LeafKind::Uniform { sym, .. } => *sym = self.symbol(*sym),
            LeafKind::Buffer { name, .. }
            | LeafKind::Param { name, .. }
            | LeafKind::Quantized { name, .. } => name.0 = 0,
            _ => {}
        }
    }
}

#[derive(Default)]
struct Symbols {
    dims: Vec<SymId>,
    scalars: Vec<SymId>,
    bodies: FxHashSet<u64>,
}

impl VisitMut for Symbols {
    fn dim(&mut self, dim: &mut Dim) {
        collect_dims(&[*dim], &mut self.dims);
    }
    fn scalar(&mut self, expr: &mut ScalarExpr) {
        if self.bodies.insert(expr.structural_hash()) {
            expr.walk(&mut |e| {
                if let ScalarKind::Uniform(sym) = e.kind() {
                    self.scalars.push(*sym);
                }
            });
        }
    }
    fn leaf(&mut self, leaf: &mut LeafKind) {
        if let LeafKind::Uniform { sym, .. } = leaf {
            self.scalars.push(*sym);
        }
    }
}

fn collect_dims(dims: &[Dim], out: &mut Vec<SymId>) {
    out.extend(dims.iter().filter_map(|d| match d {
        Dim::Sym(s) => Some(*s),
        _ => None,
    }));
}

fn collect_layout(layout: &Layout, out: &mut Vec<SymId>) {
    collect_dims(&[layout.offset()], out);
    collect_dims(layout.shape(), out);
    collect_dims(layout.strides(), out);
}
