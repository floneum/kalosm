//! ABSORB — a reduction nest absorbs, into every slot's lift, the maximal
//! chain of elementwise producers its iteration space covers, minting one
//! fold. Pure substitution, so it fires under strict numerics; a fold-to-fold
//! edge is left to the extractor. [`MAP_INTO_MAP`] is the same law for a `Map`.

use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::launch::{AccessPlan, ContractSide, IndexSpace, Launch, Operand};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::rules::{MapView, access_legal_in, map_view, operand_dtypes, shift_args, splice_args};
use crate::scalar::ScalarExpr;
use crate::shape::{AxisGroup, Dim, Layout, MultiFlattenMap};
use smallvec::SmallVec;

rule!(
    ABSORB,
    level = Level::Launch,
    head = OpTag::LaunchFold,
    tag = RuleTag::Additive,
    apply = absorb,
);

rule!(
    MAP_INTO_CONTRACT,
    level = Level::Launch,
    head = OpTag::LaunchContract,
    tag = RuleTag::Additive,
    apply = map_into_contract,
);

rule!(
    MAP_INTO_MAP,
    level = Level::Launch,
    head = OpTag::LaunchMap,
    tag = RuleTag::Additive,
    apply = map_into_map,
);

rule!(
    FOLD_POST_EPILOGUE,
    level = Level::Launch,
    head = OpTag::LaunchMap,
    tag = RuleTag::Additive,
    apply = fold_post_epilogue,
);

/// A reader's operands after one splice, and the `Arg` renumbering onto them.
struct Spliced {
    ops: Vec<Operand>,
    args: Vec<ScalarExpr>,
}

/// Splice `inner` in at `slot` of `ops`. The slot must be an `Alias` reading
/// the producer densely at the iteration coordinate (`iter` = `space` minus
/// `vec_axes`); a windowed alias would silently read the whole buffer.
fn splice(
    b: &Builder<'_>,
    ops: &[Operand],
    slot: usize,
    inner: &MapView,
    space: &IndexSpace,
    iter: &IndexSpace,
    vec_axes: &[u32],
) -> Option<Spliced> {
    if !matches!(ops[slot].access, AccessPlan::Alias) {
        return None;
    }
    if !covers_for_substitution(iter, inner) {
        return None;
    }
    if !inner.ops.iter().all(|o| access_legal_in(&o.access, space)) {
        return None;
    }
    if !reads_producer_densely(&ops[slot], &inner.space.dims, space, vec_axes) {
        return None;
    }
    let (mut new_ops, args) = splice_args(b, ops, slot, inner);
    for o in &inner.ops {
        if space == &inner.space {
            new_ops.push(o.clone());
        } else {
            let (o, _) = crate::rules::rebase::effective(b, o, &inner.space);
            new_ops.push(widen_operand(&o, &inner.space, space, vec_axes)?);
        }
    }
    Some(Spliced { ops: new_ops, args })
}

/// Restate a producer read over a wider iteration space without changing its
/// address. Promoted and trailing broadcast axes contribute zero stride.
pub(crate) fn widen_operand(
    o: &Operand,
    producer: &IndexSpace,
    space: &IndexSpace,
    vec_axes: &[u32],
) -> Option<Operand> {
    if matches!(o.access, AccessPlan::Alias) && o.layout.shape() == producer.dims.as_slice() {
        let mut axis = 0;
        let strides: Vec<_> = space
            .dims
            .iter()
            .enumerate()
            .map(|(i, d)| {
                if vec_axes.contains(&(i as u32)) {
                    return Some(Dim::Const(0));
                }
                let at = axis;
                axis += 1;
                match producer.dims.get(at) {
                    None => Some(Dim::Const(0)),
                    Some(p) if p.known_eq(Dim::ONE) => Some(Dim::Const(0)),
                    Some(p) if p.known_eq(*d) => o.layout.strides().get(at).copied(),
                    _ => None,
                }
            })
            .collect::<Option<_>>()?;
        return Some(Operand {
            src: o.src,
            layout: Layout::from_parts(o.layout.offset(), &space.dims, &strides).ok()?,
            access: AccessPlan::Alias,
        });
    }
    let groups = widen_groups(&operand_groups(o)?, space, vec_axes)?;
    operand_from_groups(o, &groups, space)
}

/// Per-logical-axis groups of an operand's index map; `Gather` and `Pack`
/// cannot be restated over a wider space and decline.
pub(crate) fn operand_groups(o: &Operand) -> Option<SmallVec<[AxisGroup; 4]>> {
    match &o.access {
        AccessPlan::Unflatten(m) => Some(m.groups.clone()),
        AccessPlan::Alias => o.layout.affine_groups(),
        AccessPlan::Gather | AccessPlan::Pack { .. } => None,
    }
}

/// Restate a map over the producer's space as one over the consumer's full
/// `space`. Axes the producer does not name (promoted, or trailing past its
/// rank) get stride 0, so an absorbed producer needs no renumbering.
pub(crate) fn widen_groups(
    src: &[AxisGroup],
    space: &IndexSpace,
    vec_axes: &[u32],
) -> Option<SmallVec<[AxisGroup; 4]>> {
    let extent_of = |d: &Dim| -> Option<u32> { u32::try_from(d.as_const()?).ok() };
    let mut groups: SmallVec<[AxisGroup; 4]> = SmallVec::new();
    let mut i = 0usize;
    for (j, d) in space.dims.iter().enumerate() {
        let extent = extent_of(d)?;
        if vec_axes.contains(&(j as u32)) {
            groups.push(AxisGroup::affine(extent, 0));
            continue;
        }
        let g = src.get(i);
        i += 1;
        let Some(g) = g else {
            groups.push(AxisGroup::affine(extent, 0));
            continue;
        };
        // Width 1 against `N` is a unit-axis broadcast; any other mismatch
        // does not describe this space.
        let width: u64 = g
            .sub_axes
            .iter()
            .try_fold(1u64, |a, s| a.checked_mul(u64::from(s.extent)))?;
        if width == u64::from(extent) {
            groups.push(g.clone());
        } else if width == 1 {
            groups.push(AxisGroup::affine(extent, 0));
        } else {
            return None;
        }
    }
    Some(groups)
}

/// Spell a widened map as an operand: a plain `Alias` when every group is one
/// stride (other rules match `Alias` first), `Unflatten` only for real divmods.
pub(crate) fn operand_from_groups(
    o: &Operand,
    groups: &[AxisGroup],
    space: &IndexSpace,
) -> Option<Operand> {
    if groups.iter().all(|g| g.sub_axes.len() == 1) {
        let strides: Vec<Dim> = groups
            .iter()
            .map(|g| Dim::Const(u64::from(g.sub_axes[0].stride)))
            .collect();
        let shape: Vec<Dim> = space.dims.iter().copied().collect();
        return Some(Operand {
            src: o.src,
            layout: Layout::from_parts(o.layout.offset(), &shape, &strides).ok()?,
            access: AccessPlan::Alias,
        });
    }
    Some(Operand {
        src: o.src,
        layout: o.layout.clone(),
        access: AccessPlan::Unflatten(MultiFlattenMap {
            groups: groups.iter().cloned().collect(),
        }),
    })
}

/// Whether `o` reads a producer of `producer_shape` densely at the consumer's
/// iteration coordinate, so its body substitutes unrenumbered. Concrete maps
/// compare exactly; symbolic ones need provably dense strides.
fn reads_producer_densely(
    o: &Operand,
    producer_shape: &[Dim],
    space: &IndexSpace,
    vec_axes: &[u32],
) -> bool {
    match (
        dense_read_map(producer_shape, space, vec_axes),
        o.address_map(),
    ) {
        (Some(want), Some(got)) => want == got,
        _ => {
            if !o.layout.offset().known_eq(Dim::Const(0))
                || o.layout.shape() != space.dims.as_slice()
            {
                return false;
            }
            let dense = Layout::row_major_strides(producer_shape);
            let mut strides = dense.iter();
            space
                .dims
                .iter()
                .zip(o.layout.strides())
                .enumerate()
                .all(|(axis, (extent, stride))| {
                    let want = if vec_axes.contains(&(axis as u32)) {
                        Dim::Const(0)
                    } else {
                        strides.next().copied().unwrap_or(Dim::Const(0))
                    };
                    extent.known_eq(Dim::ONE)
                        || (want != Dim::Sym(crate::shape::OPAQUE_SYM) && stride.known_eq(want))
                })
        }
    }
}

/// The address map an operand would present if it read `producer_shape`
/// densely at the consumer's iteration coordinate.
fn dense_read_map(
    producer_shape: &[Dim],
    space: &IndexSpace,
    vec_axes: &[u32],
) -> Option<crate::ir::launch::AddressMap> {
    let src = Layout::contiguous(producer_shape).affine_groups()?;
    let groups = widen_groups(&src, space, vec_axes)?;
    Operand {
        src: Id(0),
        layout: Layout::contiguous(producer_shape),
        access: AccessPlan::Unflatten(MultiFlattenMap { groups }),
    }
    .address_map()
}

/// Absorb across an edge with a non-trivial address map (a promoted fold's):
/// the map must equal the dense read at the iteration coordinate, which is
/// sharper than `covers` when a free and the reduced axis share an extent.
fn splice_through_address_map(
    b: &Builder<'_>,
    ops: &[Operand],
    slot: usize,
    inner: &MapView,
    space: &IndexSpace,
    iter: &IndexSpace,
    vec_axes: &[u32],
) -> Option<Spliced> {
    // Unpromoted edges are the Alias path's.
    if vec_axes.is_empty() {
        return None;
    }
    // Exact equality: a prefix match would re-read the body at coordinates
    // the producer never named.
    if !iter.covers(&inner.space) || !inner.space.covers(iter) {
        return None;
    }
    let want = dense_read_map(&inner.space.dims, space, vec_axes)?;
    if ops[slot].address_map()? != want {
        return None;
    }
    let (mut new_ops, args) = splice_args(b, ops, slot, inner);
    for o in &inner.ops {
        // Collapse pure views first: a floor broadcast is a `Restride` read
        // densely, and widening that would state a stride it does not have.
        let (o, _) = crate::rules::rebase::effective(b, o, &inner.space);
        // A non-zero offset is a node `verify_launch` rejects.
        if !o.layout.offset().known_eq(Dim::Const(0)) {
            return None;
        }
        let groups = widen_groups(&operand_groups(&o)?, space, vec_axes)?;
        new_ops.push(operand_from_groups(&o, &groups, space)?);
    }
    Some(Spliced { ops: new_ops, args })
}

/// Whether `inner`'s body may be substituted into a nest iterating `iter`;
/// a body reading `IndexOf` needs the spaces to agree exactly.
fn covers_for_substitution(iter: &IndexSpace, inner: &MapView) -> bool {
    if !iter.covers(&inner.space) {
        return false;
    }
    !inner.body.reads_index_of() || inner.space.covers(iter)
}

/// The first absorbable operand slot of `ops`, spliced. [`map_view`] reads
/// only elementwise producers, so fold-to-fold edges never match.
fn absorb_step(
    b: &Builder<'_>,
    ops: &[Operand],
    space: &IndexSpace,
    iter: &IndexSpace,
    vec_axes: &[u32],
) -> Option<Spliced> {
    ops.iter().enumerate().find_map(|(i, o)| {
        let view = map_view(b, o.src)?;
        splice(b, ops, i, &view, space, iter, vec_axes)
            .or_else(|| splice_through_address_map(b, ops, i, &view, space, iter, vec_axes))
    })
}

/// Operand-list ceiling: a producer read twice by one chain widens the list.
const MAX_ABSORBED_OPERANDS: usize = 32;

/// Splice producers into `exprs` until none is left or the next list would
/// not bind; `None` when nothing absorbed. Ids strictly decrease, so it ends.
fn absorb_chain(
    b: &Builder<'_>,
    f: &Facts<'_>,
    mut cur: Vec<Operand>,
    exprs: &mut [ScalarExpr],
    step: impl Fn(&[Operand]) -> Option<Spliced>,
) -> Option<Vec<Operand>> {
    let budget = f.caps().limits.max_storage_buffers_per_shader_stage as usize;
    let mut fired = false;
    while cur.len() <= MAX_ABSORBED_OPERANDS {
        let Some(spliced) = step(&cur) else {
            break;
        };
        // One launch is one bind group; past the budget the kernel cannot
        // be created, and extraction would already have committed.
        if storage_bindings(b, &spliced.ops) > budget {
            break;
        }
        for e in exprs.iter_mut() {
            *e = e.compose(&spliced.args);
        }
        cur = spliced.ops;
        fired = true;
    }
    fired.then_some(cur)
}

/// ABSORB, greedy: absorb the maximal chain of elementwise producers into
/// every slot's lift and mint ONE fold.
pub fn absorb(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(
        k @ Launch::Fold {
            space,
            vec_axes,
            carrier,
            acc,
            ops,
            ..
        },
    ) = &node.op
    else {
        return None;
    };
    if ops.is_empty() {
        return None;
    }
    // `f.own()` is the meet over every operand, which the fused fold reads.
    if acc.accum_bits() < f.own().numeric.min_accum_bits {
        return None;
    }
    let iter = k.iter_space();
    // Every slot's lift, or a multi-slot fold computes a wrong slot.
    let mut lift: SmallVec<[ScalarExpr; 4]> = carrier.lift.clone();
    let cur = absorb_chain(b, f, ops.clone(), &mut lift, |cur| {
        absorb_step(b, cur, space, &iter, vec_axes)
    })?;
    let mut distinct = Vec::new();
    let remap: Vec<_> = cur
        .iter()
        .map(|operand| {
            if let Some(slot) = distinct.iter().position(|other| other == operand) {
                slot as u32
            } else {
                distinct.push(operand.clone());
                (distinct.len() - 1) as u32
            }
        })
        .collect();
    let mut fused = k.clone();
    if let Launch::Fold { carrier, ops, .. } = &mut fused {
        carrier.lift = lift
            .iter()
            .map(|e| crate::carrier::map_args(e, &|i| remap[i as usize]))
            .collect();
        *ops = distinct;
    }
    // A layout mismatching the nest's space is a node `verify_plan` rejects.
    crate::verify_launch::check_operand_access(&fused).ok()?;
    let fused = b.add_launch(fused).ok()?;
    b.union(id, fused).ok()
}

/// Storage bindings a launch with these operands needs: distinct non-free
/// values read, plus its output and the storage-space `Uniforms` block.
/// Two members of one class over-count, the conservative direction.
fn storage_bindings(b: &Builder<'_>, ops: &[Operand]) -> usize {
    let mut seen: SmallVec<[Id; 8]> = SmallVec::new();
    for o in ops {
        if seen.contains(&o.src) {
            continue;
        }
        if matches!(
            b.node(o.src).op,
            Op::Logical(crate::ir::logical::Logical::Leaf(
                crate::ir::logical::LeafKind::Const { .. }
                    | crate::ir::logical::LeafKind::Uniform { .. }
            ))
        ) {
            continue;
        }
        seen.push(o.src);
    }
    seen.len() + 2
}

/// MAP_INTO_MAP, greedy: absorb the maximal chain of elementwise producers
/// into this map's body and mint one map. `map_view` sees the operand's id,
/// not its class: offering every member widens the frontier for no gain.
pub fn map_into_map(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(
        op @ Launch::Map {
            space, body, ops, ..
        },
    ) = &node.op
    else {
        return None;
    };
    if ops.is_empty() {
        return None;
    }
    let mut body = [body.clone()];
    let cur = absorb_chain(b, f, ops.clone(), &mut body, |cur| {
        cur.iter().enumerate().find_map(|(i, o)| {
            let view = map_view(b, o.src)?;
            splice(b, cur, i, &view, space, space, &[])
        })
    })?;
    let [body] = body;
    let mut fused = op.clone();
    if let Launch::Map { body: b0, ops, .. } = &mut fused {
        (*b0, *ops) = (body, cur);
    }
    crate::verify_launch::check_operand_access(&fused).ok()?;
    let fused = b.add_launch(fused).ok()?;
    b.union(id, fused).ok()
}

/// Inline elementwise producers into a contraction's `a` or `b` side.
pub fn map_into_contract(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(
        op @ Launch::Contract {
            m,
            n,
            k,
            batch,
            a,
            b: rhs,
            ..
        },
    ) = &node.op
    else {
        return None;
    };
    let space = IndexSpace::new([*batch, *m, *n, *k]);
    let new_a = absorb_into_side(b, a, &space);
    let new_b = absorb_into_side(b, rhs, &space);
    if new_a.is_none() && new_b.is_none() {
        return None;
    }
    // Both sides bind in one launch.
    let budget = f.caps().limits.max_storage_buffers_per_shader_stage as usize;
    let all: Vec<Operand> = new_a
        .as_ref()
        .unwrap_or(a)
        .ops
        .iter()
        .chain(new_b.as_ref().unwrap_or(rhs).ops.iter())
        .cloned()
        .collect();
    if storage_bindings(b, &all) > budget {
        return None;
    }
    let mut fused = op.clone();
    if let Launch::Contract { a, b: rhs, .. } = &mut fused {
        *a = new_a.unwrap_or_else(|| a.clone());
        *rhs = new_b.unwrap_or_else(|| rhs.clone());
    }
    let fused = b.add_launch(fused).ok()?;
    b.union(id, fused).ok()
}

/// Absorb elementwise producers into every eligible slot of a contraction
/// side at once (one slot per fire would mint every absorption order), or
/// `None` when no slot reads one. Producer operands join the side's list.
fn absorb_into_side(
    b: &Builder<'_>,
    side: &ContractSide,
    space: &IndexSpace,
) -> Option<ContractSide> {
    let eligible = |o: &Operand| -> Option<(MapView, SmallVec<[usize; 4]>)> {
        if !matches!(o.access, AccessPlan::Alias) {
            return None;
        }
        let mut inner = map_view(b, o.src)?;
        // Fold pure-view spines into layouts now: no later rule does it on
        // this path. A spine not composing to an offset-0 layout stays.
        for p in inner.ops.iter_mut() {
            let space = IndexSpace::new(p.layout.shape().iter().copied());
            let (read, base) = crate::rules::rebase::effective(b, p, &space);
            if base != p.src && read.layout.offset().known_eq(Dim::Const(0)) {
                *p = read;
            }
        }
        if !inner
            .ops
            .iter()
            .all(|p| matches!(p.access, AccessPlan::Alias) && access_legal_in(&p.access, space))
        {
            return None;
        }
        // Never absorb a quantized read: the block decode's addressing holds
        // only for the family `lower_family` already minted it under.
        if inner
            .ops
            .iter()
            .any(|p| b.facts_of(p.src).dtype.is_quantized())
        {
            return None;
        }
        // A remaining pure-view operand's dense layout would read the base
        // buffer wrongly; nothing later re-points it.
        if inner
            .ops
            .iter()
            .any(|p| !b.trace_pure_views(p.src).views.is_empty())
        {
            return None;
        }
        // The edge must be an axis permutation of the dense value; it is
        // carried onto every absorbed operand by [`permute_layout`].
        let perm = dense_permutation(&o.layout, &inner.space.dims)?;
        // `pre` sees operand-axis coordinates, so `IndexOf` axes shift by the
        // inverse permutation (lets a causal mask ride into the contraction).
        if inner.body.reads_index_of() {
            let mut inv: SmallVec<[u32; 4]> = smallvec::smallvec![0; perm.len()];
            for (j, &i) in perm.iter().enumerate() {
                inv[i] = j as u32;
            }
            inner.body = inner
                .body
                .remap_index_axes(&|axis| inv.get(axis as usize).copied().unwrap_or(axis));
        }
        Some((inner, perm))
    };
    let plans: Vec<Option<(MapView, SmallVec<[usize; 4]>)>> =
        side.ops.iter().map(eligible).collect();
    if plans.iter().all(Option::is_none) {
        return None;
    }

    // Retained operands take the low args; producers append in slot order.
    let outer_dtypes = operand_dtypes(b, &side.ops);
    let retained = plans.iter().filter(|p| p.is_none()).count();
    let mut ops: SmallVec<[Operand; 2]> = SmallVec::new();
    for (o, plan) in side.ops.iter().zip(&plans) {
        if plan.is_none() {
            ops.push(o.clone());
        }
    }
    let mut appended = retained;
    let mut args: Vec<ScalarExpr> = Vec::with_capacity(side.ops.len());
    let mut next_retained = 0u32;
    for (j, plan) in plans.iter().enumerate() {
        match plan {
            None => {
                args.push(ScalarExpr::arg(next_retained, outer_dtypes[j]));
                next_retained += 1;
            }
            Some((inner, perm)) => {
                let inner_dtypes = operand_dtypes(b, &inner.ops);
                args.push(shift_args(&inner.body, appended as u32, &inner_dtypes));
                for p in &inner.ops {
                    ops.push(Operand {
                        src: p.src,
                        layout: permute_layout(&p.layout, perm)?,
                        access: p.access.clone(),
                    });
                }
                appended += inner.ops.len();
            }
        }
    }
    Some(ContractSide {
        pre: side.pre.compose(&args),
        ops,
    })
}

/// `perm[j] = i` when `layout`'s axis `j` walks dense `producer` axis `i`
/// (matched on `(extent, row-major stride)` pairs, offset zero), or `None`
/// when the read is not a pure permutation.
fn dense_permutation(layout: &Layout, producer: &[Dim]) -> Option<SmallVec<[usize; 4]>> {
    if !layout.offset().known_eq(Dim::Const(0)) || layout.rank() != producer.len() {
        return None;
    }
    let row_major = Layout::row_major_strides(producer);
    let mut claimed = vec![false; producer.len()];
    let mut perm: SmallVec<[usize; 4]> = SmallVec::with_capacity(producer.len());
    for (d, s) in layout.shape().iter().zip(layout.strides()) {
        let i = producer
            .iter()
            .enumerate()
            .position(|(i, pd)| !claimed[i] && d.known_eq(*pd) && s.known_eq(row_major[i]))?;
        claimed[i] = true;
        perm.push(i);
    }
    Some(perm)
}

/// `layout` with its axes reordered by `perm`; addresses are unchanged.
fn permute_layout(layout: &Layout, perm: &[usize]) -> Option<Layout> {
    if layout.rank() != perm.len() {
        return None;
    }
    let shape: SmallVec<[Dim; 6]> = perm.iter().map(|&i| layout.shape()[i]).collect();
    let strides: SmallVec<[Dim; 6]> = perm.iter().map(|&i| layout.strides()[i]).collect();
    Layout::from_parts(layout.offset(), &shape, &strides).ok()
}

/// A single-operand `Map` reading a `Fold` at the fold's *output* space is
/// that fold with a longer `post`.
pub fn fold_post_epilogue(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(Launch::Map {
        space, body, ops, ..
    }) = &node.op
    else {
        return None;
    };
    if ops.len() != 1 || !matches!(ops[0].access, AccessPlan::Alias) {
        return None;
    }
    let Op::Launch(mut extended) = b.node(ops[0].src).op.clone() else {
        return None;
    };
    let Launch::Fold { carrier, post, .. } = &mut extended else {
        return None;
    };
    // Which slot of a multi-slot fold the map meant is not recoverable.
    if carrier.width() != 1 {
        return None;
    }
    if space.dims != b.facts_of(ops[0].src).shape {
        return None;
    }
    let _ = f;
    *post = smallvec::smallvec![body.compose(&[post[0].clone()])];
    let extended = b.add_launch(extended).ok()?;
    b.union(id, extended).ok()
}
