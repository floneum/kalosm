//! ABSORB — a reduction nest absorbs a producer whose index space it covers.
//!
//! Let `F = KFold{space, axis, vec_axes, carrier C, ops}` with iteration space
//! `E(F) = space` minus `vec_axes`. Any operand `ops[i]` produced by `P` where
//! `E(F).covers(space(P))` is absorbed into every slot's lift:
//!
//! ```text
//! KFold{C, ops}  ==  KFold{C[lift[k] := lift[k]{Arg(i) := body(P)}], ops[i := ops(P)]}
//! ```
//!
//! Substitution into a lift reassociates nothing, so the law carries no
//! `reassoc` guard and fires under
//! [`NumericContract::STRICT`](crate::dtype::NumericContract::STRICT).
//!
//! The matcher is greedy: it walks the maximal chain of absorbable producers
//! and mints one fold with the fully composed lift. Intermediate partial
//! absorptions are not minted.
//!
//! A producer that is itself a reducing nest is left alone; whether to inline
//! or materialize it is the extractor's decision, not a rewrite.
//!
//! There is no reader-count check and no duplication veto: a producer read
//! twice may be absorbed by both readers. `KRegion` is the same rewrite with
//! `live_outs` non-empty.
//!
//! [`MAP_INTO_MAP`] is the same law with a `KMap` in the consumer position;
//! nothing calls `ScalarExpr::compose` at construction, so elementwise chains
//! need it to reach extraction as one node.

use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::level1::{AccessPlan, ContractSide, IndexSpace, L1, Operand};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::rules::{MapView, access_legal_in, map_view, operand_dtypes, shift_args};
use crate::scalar::ScalarExpr;
use crate::shape::{AxisGroup, Dim, Layout, MultiFlattenMap};
use smallvec::SmallVec;

rule!(
    ABSORB,
    level = Level::L1,
    head = OpTag::KFold,
    tag = RuleTag::Additive,
    apply = absorb,
);

rule!(
    MAP_INTO_CONTRACT,
    level = Level::L1,
    head = OpTag::KContract,
    tag = RuleTag::Additive,
    apply = map_into_contract,
);

rule!(
    MAP_INTO_MAP,
    level = Level::L1,
    head = OpTag::KMap,
    tag = RuleTag::Additive,
    apply = map_into_map,
);

rule!(
    FOLD_POST_EPILOGUE,
    level = Level::L1,
    head = OpTag::KMap,
    tag = RuleTag::Additive,
    apply = fold_post_epilogue,
);

rule!(
    FORM_KREGION,
    level = Level::L1,
    head = OpTag::KFold,
    tag = RuleTag::Additive,
    apply = form_kregion,
);

/// The result of splicing one elementwise producer into a reader's operand
/// list: the reader's remaining operands followed by the producer's, and the
/// substitution vector that renumbers the reader's `Arg`s onto them.
struct Spliced {
    ops: Vec<Operand>,
    args: Vec<ScalarExpr>,
}

/// Splice `inner` in at `slot` of `ops`.
///
/// The reader's iteration space must cover the producer's, every operand the
/// producer brings must satisfy the reader's access predicate, and the replaced
/// operand must be a plain alias whose `AddressMap` is the dense read of the
/// producer's shape over `space` — an alias alone can read a window, while a
/// broadcast passes. `space` is the full index space; `iter` is `space` minus
/// `vec_axes`, and every `ScalarExpr` is written against it.
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
    let base = ops.len() - 1;
    let inner_dtypes = operand_dtypes(b, &inner.ops);
    let body = shift_args(&inner.body, base as u32, &inner_dtypes);
    let outer_dtypes = operand_dtypes(b, ops);

    let mut args: Vec<ScalarExpr> = Vec::with_capacity(ops.len());
    for (j, d) in outer_dtypes.iter().enumerate() {
        args.push(match j.cmp(&slot) {
            std::cmp::Ordering::Equal => body.clone(),
            std::cmp::Ordering::Less => ScalarExpr::arg(j as u32, *d),
            std::cmp::Ordering::Greater => ScalarExpr::arg(j as u32 - 1, *d),
        });
    }
    let mut new_ops: Vec<Operand> = Vec::with_capacity(base + inner.ops.len());
    new_ops.extend(ops.iter().enumerate().filter(|(j, _)| *j != slot).map(|(_, o)| o.clone()));
    new_ops.extend(inner.ops.iter().cloned());
    Some(Spliced {
        ops: new_ops,
        args,
    })
}

/// Per-logical-axis groups of an operand's own index map, in the producer's
/// axis order. `Alias` reads them off the layout's strides; `Unflatten`
/// carries them directly. `Gather` and `Pack` derive addresses this function
/// cannot restate over a wider space, so they decline.
pub(crate) fn operand_groups(o: &Operand) -> Option<SmallVec<[AxisGroup; 4]>> {
    match &o.access {
        AccessPlan::Unflatten(m) => Some(m.groups.clone()),
        AccessPlan::Alias => o
            .layout
            .shape()
            .iter()
            .zip(o.layout.strides())
            .map(|(d, s)| {
                Some(AxisGroup::affine(
                    u32::try_from(d.as_const()?).ok()?,
                    u32::try_from(s.as_const()?).ok()?,
                ))
            })
            .collect::<Option<_>>(),
        AccessPlan::Gather | AccessPlan::Pack { .. } => None,
    }
}

/// Restate a map stated over the producer's space as one over the consumer's
/// full `space`.
///
/// The consumer's iteration axes are `space` minus `vec_axes`, in order, and
/// the producer's space is a prefix of them. Every axis the producer does not
/// name — a promoted axis, or a trailing iteration axis past the producer's
/// rank — contributes stride 0: the producer's value is re-read at every
/// position of that axis.
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
        // A group's sub-extents multiply to the axis it describes. A `1` where
        // the space has `N` is a broadcast spelled as a unit axis; anything
        // else does not describe this space.
        let width: u64 = g.sub_axes.iter().try_fold(1u64, |a, s| {
            a.checked_mul(u64::from(s.extent))
        })?;
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

/// Spell a widened map as an operand.
///
/// One `AxisGroup` per axis with one sub-axis each is a stride vector, so it is
/// minted as a plain `Alias` layout over `space`; other rules' dependence
/// queries and layout checks are written against `Alias`. `Unflatten` spells a
/// genuine divmod decomposition.
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
/// iteration coordinate — the condition under which the producer's body may be
/// substituted for `Arg(slot)` unrenumbered.
///
/// The statement is an `AddressMap` equality against [`dense_read_map`], whose
/// divisors come from const extents. A `Dim::Sym` axis has no such map, so the
/// fallback is that the offset be zero; a permuted or strided read over a
/// symbolic axis is not caught.
fn reads_producer_densely(
    o: &Operand,
    producer_shape: &[Dim],
    space: &IndexSpace,
    vec_axes: &[u32],
) -> bool {
    match (dense_read_map(producer_shape, space, vec_axes), o.address_map()) {
        (Some(want), Some(got)) => want == got,
        _ => o.layout.offset().known_eq(Dim::Const(0)),
    }
}

/// The address map an operand would present if it read `producer_shape`
/// densely at the consumer's iteration coordinate.
fn dense_read_map(
    producer_shape: &[Dim],
    space: &IndexSpace,
    vec_axes: &[u32],
) -> Option<crate::ir::level1::AddressMap> {
    let strides = Layout::row_major_strides(producer_shape);
    let src: SmallVec<[AxisGroup; 4]> = producer_shape
        .iter()
        .zip(&strides)
        .map(|(d, s)| {
            Some(AxisGroup::affine(
                u32::try_from(d.as_const()?).ok()?,
                u32::try_from(s.as_const()?).ok()?,
            ))
        })
        .collect::<Option<_>>()?;
    let groups = widen_groups(&src, space, vec_axes)?;
    Operand {
        src: Id(0),
        layout: Layout::contiguous(producer_shape),
        access: AccessPlan::Unflatten(MultiFlattenMap { groups }),
    }
    .address_map()
}

/// Absorb across an operand edge carrying a non-trivial address map.
///
/// The edge must read the producer at the consumer's iteration coordinate;
/// then the producer's body is substituted unrenumbered and its own operands
/// restated over the wider space by [`widen_groups`]. That is an equality of
/// `AddressMap`s, sharper than the prefix `covers` test beside it, and it is
/// the clause PROMOTE's output needs.
fn splice_through_address_map(
    b: &Builder<'_>,
    ops: &[Operand],
    slot: usize,
    inner: &MapView,
    space: &IndexSpace,
    iter: &IndexSpace,
    vec_axes: &[u32],
) -> Option<Spliced> {
    if vec_axes.is_empty() {
        // With no promoted axis `iter == space` and the Alias path already
        // covers every edge this one would.
        return None;
    }
    // Exact equality, both directions: `covers` is a prefix test, so it admits
    // a producer of smaller rank broadcast along the consumer's trailing
    // iteration axes, and widening across that puts the substituted body at a
    // coordinate the producer never named.
    if !iter.covers(&inner.space) || !inner.space.covers(iter) {
        return None;
    }
    let want = dense_read_map(&inner.space.dims, space, vec_axes)?;
    if ops[slot].address_map()? != want {
        return None;
    }

    let base = ops.len() - 1;
    let inner_dtypes = operand_dtypes(b, &inner.ops);
    let body = shift_args(&inner.body, base as u32, &inner_dtypes);
    let outer_dtypes = operand_dtypes(b, ops);

    let mut args: Vec<ScalarExpr> = Vec::with_capacity(ops.len());
    for (j, d) in outer_dtypes.iter().enumerate() {
        args.push(match j.cmp(&slot) {
            std::cmp::Ordering::Equal => body.clone(),
            std::cmp::Ordering::Less => ScalarExpr::arg(j as u32, *d),
            std::cmp::Ordering::Greater => ScalarExpr::arg(j as u32 - 1, *d),
        });
    }

    let mut new_ops: Vec<Operand> = Vec::with_capacity(base + inner.ops.len());
    new_ops.extend(ops.iter().enumerate().filter(|(j, _)| *j != slot).map(|(_, o)| o.clone()));
    for o in &inner.ops {
        // Collapse a pure view into the layout before widening: a broadcast
        // spelled as a `Restride` node reaches its edge with a dense layout,
        // and widening that spelling would state stride 1 on an axis the value
        // does not vary along.
        let (o, _) = crate::rules::rebase::effective(b, o, &inner.space);
        // Allocation is not described at L1, so an edge that collapsed a
        // narrowing view into a non-zero offset is a node `verify_l1` rejects.
        if !o.layout.offset().known_eq(Dim::Const(0)) {
            return None;
        }
        let groups = widen_groups(&operand_groups(&o)?, space, vec_axes)?;
        new_ops.push(operand_from_groups(&o, &groups, space)?);
    }
    Some(Spliced {
        ops: new_ops,
        args,
    })
}

/// Whether `inner`'s body may be substituted into a nest iterating `iter`.
///
/// `iter.covers(inner.space)` is a prefix test on the iteration space. When the
/// producer's body reads an `IndexOf`, the two spaces must agree exactly: only
/// then does the coordinate the body names survive substitution unrenumbered.
fn covers_for_substitution(iter: &IndexSpace, inner: &MapView) -> bool {
    if !iter.covers(&inner.space) {
        return false;
    }
    !inner.body.reads_index_of() || inner.space.covers(iter)
}

/// The first operand slot of `ops` that can be absorbed, spliced.
///
/// A producer that is itself a reducing nest is not matched: [`map_view`] reads
/// elementwise producers only. [`splice`] is tried first, then
/// [`splice_through_address_map`], the path a promoted nest needs, where
/// `space` is one rank wider than `iter`. [`splice`] does not widen onto a
/// promoted space, so there it produces a node
/// `verify_l1::check_operand_access` rejects, discarding the whole chain.
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

/// Operand-list ceiling. Absorption terminates on its own — each step replaces
/// one operand by producers with strictly smaller ids — but a producer read
/// twice by the same chain widens the list, so this bounds the width.
const MAX_ABSORBED_OPERANDS: usize = 32;

/// ABSORB, greedy: absorb the maximal chain of elementwise producers into
/// every slot's lift and mint ONE fold.
pub fn absorb(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let fused = build_absorbed_fold(b, node, f)?;
    b.union(id, fused).ok()
}

fn build_absorbed_fold(b: &mut Builder<'_>, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::L1(k @ L1::KFold {
        space,
        axis,
        vec_axes,
        carrier,
        acc,
        post,
        ops,
        sched,
    }) = &node.op
    else {
        return None;
    };
    if ops.is_empty() {
        return None;
    }
    // `f.own()` is the meet over every operand; `f.numeric(0)` reads operand
    // zero alone and is blind on the multi-operand fold this becomes.
    if acc.accum_bits() < f.own().numeric.min_accum_bits {
        return None;
    }
    let iter = k.iter_space();

    let budget = f.caps().limits.max_storage_buffers_per_shader_stage as usize;
    let mut cur: Vec<Operand> = ops.clone();
    let mut lift: SmallVec<[ScalarExpr; 4]> = carrier.lift.clone();
    let mut fired = false;
    while cur.len() <= MAX_ABSORBED_OPERANDS {
        let Some(spliced) = absorb_step(b, &cur, space, &iter, vec_axes) else {
            break;
        };
        // Stop at the last operand list the device can bind. One launch is one
        // bind group, so a fused nest reading more distinct buffers than
        // `max_storage_buffers_per_shader_stage` allows is a kernel the backend
        // cannot create.
        if storage_bindings(b, &spliced.ops) > budget {
            break;
        }
        // Substituted into every slot's lift: a carrier is one expression per
        // slot.
        lift = lift.iter().map(|l| l.compose(&spliced.args)).collect();
        cur = spliced.ops;
        fired = true;
    }
    if !fired {
        return None;
    }
    let fused = L1::KFold {
        space: space.clone(),
        axis: *axis,
        vec_axes: vec_axes.clone(),
        carrier: carrier.clone().with_lift(lift),
        acc: *acc,
        post: post.clone(),
        ops: cur,
        sched: sched.clone(),
    };
    // Invariant 5: an absorbed operand whose layout does not match this nest's
    // index space is a node `verify_plan` rejects.
    crate::verify_l1::check_operand_access(&fused).ok()?;
    b.add_l1(fused).ok()
}

/// Storage bindings one launch rooted at a nest with these operands needs: the
/// distinct non-free values it reads, its own output, and the `Uniforms` block.
///
/// `derive_bindings` reserves binding 0 for the uniform block, drops
/// `LeafRole::Free` reads and deduplicates by value. Two operands naming
/// different members of one class over-count by one, declining a legal fusion
/// rather than minting an unbindable kernel. The `Uniforms` block is a storage
/// buffer and is charged like any other — hence `+ 2`.
fn storage_bindings(b: &Builder<'_>, ops: &[Operand]) -> usize {
    let mut seen: SmallVec<[Id; 8]> = SmallVec::new();
    for o in ops {
        if seen.contains(&o.src) {
            continue;
        }
        if matches!(
            b.node(o.src).op,
            Op::L0(crate::ir::level0::L0::Leaf(
                crate::ir::level0::LeafKind::Const { .. }
                    | crate::ir::level0::LeafKind::Uniform { .. }
            ))
        ) {
            continue;
        }
        seen.push(o.src);
    }
    seen.len() + 2
}

/// MAP_INTO_MAP, greedy: absorb the maximal chain of elementwise producers
/// into this map's own body and mint ONE map.
///
/// The predicate is `ABSORB`'s and the arithmetic is `ScalarExpr::compose`.
/// There is no reader-count check. The operand is asked by id, not by class, so
/// a pure broadcast whose `KMap` spelling sits behind a `Union` can survive as
/// its own dispatch; offering every class member is a measured regression
/// against a fixed extraction move budget.
pub fn map_into_map(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::L1(L1::KMap {
        space,
        body,
        ops,
        sched,
    }) = &node.op
    else {
        return None;
    };
    if ops.is_empty() {
        return None;
    }
    let budget = f.caps().limits.max_storage_buffers_per_shader_stage as usize;
    let mut cur: Vec<Operand> = ops.clone();
    let mut expr = body.clone();
    let mut fired = false;
    // Terminates for `ABSORB`'s reason: every step replaces one operand by
    // producers with strictly smaller ids. The ceiling bounds the width a
    // producer read twice by one chain adds, not the depth.
    while cur.len() <= MAX_ABSORBED_OPERANDS {
        let Some(spliced) = cur.iter().enumerate().find_map(|(i, o)| {
            let view = map_view(b, o.src)?;
            // A `KMap` has no `vec_axes`, so its iteration space is its
            // index space and the promoted dispatch has nothing to add.
            splice(b, &cur, i, &view, space, space, &[])
        }) else {
            break;
        };
        // Stop at the last operand list the device can bind: one launch is one
        // bind group, so a fused map reading more distinct buffers than
        // `max_storage_buffers_per_shader_stage` allows cannot be created.
        if storage_bindings(b, &spliced.ops) > budget {
            break;
        }
        expr = expr.compose(&spliced.args);
        cur = spliced.ops;
        fired = true;
    }
    if !fired {
        return None;
    }
    let fused = L1::KMap {
        space: space.clone(),
        body: expr,
        ops: cur,
        sched: sched.clone(),
    };
    crate::verify_l1::check_operand_access(&fused).ok()?;
    let fused = b.add_l1(fused).ok()?;
    b.union(id, fused).ok()
}

/// Inline a single-operand elementwise producer into `pre_a` or `pre_b`.
///
/// `KContract` carries exactly two operand edges, so only a one-operand
/// producer can be absorbed without inventing a third edge.
pub fn map_into_contract(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::L1(L1::KContract {
        m,
        n,
        k,
        batch,
        family,
        post,
        acc,
        a,
        b: rhs,
        sched,
    }) = &node.op
    else {
        return None;
    };
    let space = IndexSpace::new([*batch, *m, *n, *k]);
    let new_a = absorb_into_side(b, a, &space);
    let new_b = absorb_into_side(b, rhs, &space);
    if new_a.is_none() && new_b.is_none() {
        return None;
    }
    // The same device budget `map_into_map` consults, over both sides at
    // once: a contraction binds every operand of both sides in one launch,
    // so the count that has to fit is the union, not either side's own.
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
    let fused = b
        .add_l1(L1::KContract {
            m: *m,
            n: *n,
            k: *k,
            batch: *batch,
            family: *family,
            post: post.clone(),
            acc: *acc,
            a: new_a.unwrap_or_else(|| a.clone()),
            b: new_b.unwrap_or_else(|| rhs.clone()),
            sched: sched.clone(),
        })
        .ok()?;
    b.union(id, fused).ok()
}

/// Absorb one elementwise producer into a contraction side, or `None` when
/// no slot of it reads one.
///
/// The producer's arity is not a condition: [`ContractSide`] holds a list, so a
/// multi-operand producer's operands join the side's. The slot must be a plain
/// alias and each operand the producer brings must satisfy the reader's access
/// predicate over the contraction's `(batch, m, n, k)` space. Every eligible
/// slot absorbs in one fire; one at a time would not terminate.
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
        // Fold each operand's pure-view spine into its layout before the
        // splice; the contraction path has no later rule to fold it, so a
        // carried `Restride` class materializes as its own launch. A spine that
        // does not compose to an offset-0 plain layout is left alone.
        for p in inner.ops.iter_mut() {
            if !matches!(p.access, AccessPlan::Alias) {
                continue;
            }
            let spine = b.trace_pure_views(p.src);
            if spine.views.len() != 1 {
                continue;
            }
            // Only an identity read of the view composes by substitution:
            // the operand's own strides must be the view value's dense
            // row-major set, or the composed walk is not the view's.
            let view_shape = b.facts_of(p.src).shape.clone();
            if p.layout.shape() != &view_shape[..]
                || !p.layout.offset().known_eq(Dim::Const(0))
                || p.layout
                    .strides()
                    .iter()
                    .zip(&Layout::row_major_strides(&view_shape))
                    .any(|(s, w)| !s.known_eq(*w))
            {
                continue;
            }
            let Op::L0(crate::ir::level0::L0::Restride { specs, .. }) =
                b.node(spine.views[0]).op.clone()
            else {
                continue;
            };
            let base_shape = b.facts_of(spine.base).shape.clone();
            let Some(composed) = crate::rules::composed_layout(&specs, &base_shape) else {
                continue;
            };
            // Clause 8: an L1 operand may not name a buffer offset.
            if !composed.offset().known_eq(Dim::Const(0)) {
                continue;
            }
            *p = Operand {
                src: spine.base,
                layout: composed,
                access: AccessPlan::Alias,
            };
        }
        if !inner
            .ops
            .iter()
            .all(|p| matches!(p.access, AccessPlan::Alias) && access_legal_in(&p.access, space))
        {
            return None;
        }
        // The edge may read the producer through any axis permutation of its
        // dense value, and the permutation is carried by permuting every
        // absorbed operand's own axes with it (see [`permute_layout`]). An edge
        // that is not a permutation — a window, a broadcast, an offset —
        // declines.
        let perm = dense_permutation(&o.layout, &inner.space.dims)?;
        // A body reading its own coordinates absorbs too: the contraction's
        // staging loop hands `pre` the operand-axis coordinate vector, and
        // producer axis `perm[j]` is operand axis `j`, so the axis names shift
        // by the inverse.
        if inner.body.reads_index_of() {
            let mut inv: SmallVec<[u32; 4]> = smallvec::smallvec![0; perm.len()];
            for (j, &i) in perm.iter().enumerate() {
                inv[i] = j as u32;
            }
            inner.body = inner.body.remap_index_axes(&|axis| {
                inv.get(axis as usize).copied().unwrap_or(axis)
            });
        }
        Some((inner, perm))
    };
    let plans: Vec<Option<(MapView, SmallVec<[usize; 4]>)>> =
        side.ops.iter().map(eligible).collect();
    if plans.iter().all(Option::is_none) {
        return None;
    }

    // Retained operands keep their order and take the low arg indices; each
    // absorbed producer's operands are appended in slot order, so a producer
    // body is shifted by the count of everything placed before it.
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

/// The axis order in which `layout` reads a dense value of shape `producer`,
/// or `None` when it is not a pure permutation of it.
///
/// `perm[j] = i` means the edge's axis `j` walks the producer's axis `i`: each
/// of the layout's `(extent, stride)` pairs must be exactly one producer axis's
/// `(extent, row-major stride)`, offset zero, every axis claimed once, so a
/// window, a broadcast or a gather-shaped read fails. Repeated extents stay
/// unambiguous: equal extents with equal strides address identically.
fn dense_permutation(layout: &Layout, producer: &[Dim]) -> Option<SmallVec<[usize; 4]>> {
    if !layout.offset().known_eq(Dim::Const(0)) || layout.rank() != producer.len() {
        return None;
    }
    let row_major = Layout::row_major_strides(producer);
    let mut claimed = vec![false; producer.len()];
    let mut perm: SmallVec<[usize; 4]> = SmallVec::with_capacity(producer.len());
    for (d, s) in layout.shape().iter().zip(layout.strides()) {
        let i = producer.iter().enumerate().position(|(i, pd)| {
            !claimed[i] && d.known_eq(*pd) && s.known_eq(row_major[i])
        })?;
        claimed[i] = true;
        perm.push(i);
    }
    Some(perm)
}

/// `layout` with its axes reordered by `perm` — the producer-operand layout
/// as seen through an edge that walks producer axis `perm[j]` at its own
/// axis `j`. Pure axis renaming: extents and strides travel together, so the
/// set of addresses is untouched and only the coordinate order changes.
fn permute_layout(layout: &Layout, perm: &[usize]) -> Option<Layout> {
    if layout.rank() != perm.len() {
        return None;
    }
    let shape: SmallVec<[Dim; 6]> = perm.iter().map(|&i| layout.shape()[i]).collect();
    let strides: SmallVec<[Dim; 6]> = perm.iter().map(|&i| layout.strides()[i]).collect();
    Layout::from_parts(layout.offset(), &shape, &strides).ok()
}

/// A single-operand `KMap` reading a `KFold` at the fold's *output* space is
/// that fold with a longer `post`. This plus [`map_into_fold`] is the whole
/// of row-program cluster formation.
pub fn fold_post_epilogue(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::L1(L1::KMap {
        space, body, ops, ..
    }) = &node.op
    else {
        return None;
    };
    if ops.len() != 1 || !matches!(ops[0].access, AccessPlan::Alias) {
        return None;
    }
    let Op::L1(L1::KFold {
        space: inner_space,
        axis,
        vec_axes,
        carrier,
        acc,
        post,
        ops: inner_ops,
        sched,
    }) = b.node(ops[0].src).op.clone()
    else {
        return None;
    };
    // A `KMap` body reads one value; a multi-slot fold offers several, and
    // which one the epilogue meant is not recoverable from the edge.
    if carrier.width() != 1 {
        return None;
    }
    // The epilogue must run at the fold's own output space.
    let out_shape = &b.facts_of(ops[0].src).shape;
    if space.dims.len() != out_shape.len()
        || !space
            .dims
            .iter()
            .zip(out_shape.iter())
            .all(|(a, c)| a.known_eq(*c))
    {
        return None;
    }
    let _ = f;
    let extended = b
        .add_l1(L1::KFold {
            space: inner_space,
            axis,
            vec_axes,
            carrier,
            acc,
            post: smallvec::smallvec![body.compose(&[post[0].clone()])],
            ops: inner_ops,
            sched,
        })
        .ok()?;
    b.union(id, extended).ok()
}

/// The multi-output form of [`absorb`]: the absorbed producer also escapes,
/// so the fused chain becomes a `KRegion` naming it in `live_outs`. Because
/// the region and the plain absorbed fold are both live, emitting the extra
/// buffer competes with recomputing it.
pub fn form_kregion(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::L1(k @ L1::KFold { ops, .. }) = &node.op else {
        return None;
    };
    let iter = k.iter_space();
    let (slot, _) = ops.iter().enumerate().find_map(|(i, o)| {
        if !matches!(o.access, AccessPlan::Alias) {
            return None;
        }
        let view = map_view(b, o.src)?;
        covers_for_substitution(&iter, &view).then_some((i, view))
    })?;
    let producer = ops[slot].src;
    let fused = build_absorbed_fold(b, node, f)?;
    let members: SmallVec<[Id; 8]> = smallvec::smallvec![producer, fused];
    // `live_outs: [0]` names the producer, so the schedule domain is derived
    // from the producer's index space.
    let sched = crate::rules::merge::linear_domain_of(b, producer);
    let region = b
        .add_l1(L1::KRegion {
            members,
            live_outs: smallvec::smallvec![0],
            sched,
        })
        .ok()?;
    b.union(id, region).ok()
}

#[cfg(test)]
mod tests {
    use crate::scalar::ScalarKind;
    use super::*;
    use crate::dtype::Dtype;
    use crate::scalar::BinOp;
    use crate::rules::test_support as ts;
    use crate::rules::{alias_operand_of, ident_expr};
    use crate::scalar::UnOp;
    use crate::shape::Dim;

    fn fire(g: &mut crate::egraph::EGraph, id: Id, r: &crate::egraph::Rule) -> Option<Id> {
        let caps = ts::caps();
        let node = g.node(id).clone();
        let facts = g.facts_view(id, &caps);
        let mut b = g.builder(&caps);
        (r.apply)(&mut b, id, &node, &facts)
    }

    /// Every `KFold` alternative in `id`'s class.
    fn kfolds_of(g: &crate::egraph::EGraph, id: Id) -> Vec<(Id, L1)> {
        g.chain(id)
            .into_iter()
            .filter_map(|m| match &g.node(m).op {
                Op::L1(k @ L1::KFold { .. }) => Some((m, k.clone())),
                _ => None,
            })
            .collect()
    }

    fn saturate(g: &mut crate::egraph::EGraph) -> crate::egraph::SaturationReport {
        use crate::egraph::{Saturate, SaturationBudget};
        crate::saturate::CoreSaturate
            .saturate(
                g,
                &ts::caps(),
                crate::rules::CORE_RULES,
                SaturationBudget::default(),
            )
            .unwrap()
    }

    /// ABSORB fires on a saturated graph greedily: a three-node elementwise
    /// chain under a reduction becomes one fold reading the buffer, with the
    /// whole chain in its lift, in one round.
    #[test]
    fn absorb_collapses_a_whole_chain_on_a_saturated_graph() {
        let mut g = ts::graph();
        let shape = [Dim::Const(4), Dim::Const(16)];
        let x = ts::buffer(&mut g, Dtype::F32, &shape);
        let m1 = ts::map(
            &mut g,
            ScalarExpr::un(UnOp::Exp, ScalarExpr::arg(0, Dtype::F32)),
            &[x],
        );
        let m2 = ts::map(
            &mut g,
            ScalarExpr::bin(
                BinOp::Mul,
                ScalarExpr::arg(0, Dtype::F32),
                ScalarExpr::lit(crate::dtype::Splat::F32(2.0)),
            ),
            &[m1],
        );
        let fid = ts::fold(
            &mut g,
            ts::binop_carrier(BinOp::Add, Dtype::F32),
            1,
            Dtype::F32,
            m2,
        );
        g.add_root(fid);
        let report = saturate(&mut g);
        assert!(report.saturated, "{report:?}");

        let fused = kfolds_of(&g, fid)
            .into_iter()
            .find_map(|(_, k)| match k {
                L1::KFold { carrier, ops, .. } if ops.len() == 1 && ops[0].src == x => {
                    Some(carrier)
                }
                _ => None,
            })
            .expect("a fold reading the buffer directly");
        // The merge is untouched — substitution reassociates nothing.
        assert_eq!(fused.kind(), Some(BinOp::Add));

        // The fused lift computes `2 * exp(v)` per element and the fold sums it.
        let row: Vec<f32> = vec![-1.5, 0.0, 0.25, 2.0, 3.5, -0.75];
        let want: f32 = row.iter().map(|v| 2.0 * v.exp()).sum();
        let got = row.iter().fold(fused.identity_f32(), |acc, v| {
            fused.absorb(&acc, &[*v]).unwrap()
        });
        assert!(
            (got[0] - want).abs() <= 1e-4 * want.abs(),
            "{got:?} vs {want}"
        );

        // The producers are still live members of their own classes.
        assert!(g.chain(m1).contains(&m1));
        assert!(g.chain(m2).contains(&m2));
    }

    /// ABSORB fires under `NumericContract::STRICT`, where every inexact law
    /// declines: a fake-quant chain `round(clamp(x/s, lo, hi)) * s` reduced by
    /// a plain `Fold{Add}`.
    #[test]
    fn absorb_fires_on_a_strict_quantized_chain_where_strip_declines() {
        let mut g = ts::graph();
        let shape = [Dim::Const(4), Dim::Const(512)];
        let x = ts::buffer(&mut g, Dtype::F32, &shape);
        let f32l = |v: f32| ScalarExpr::lit(crate::dtype::Splat::F32(v));
        let scaled = ScalarExpr::bin(BinOp::Div, ScalarExpr::arg(0, Dtype::F32), f32l(0.05));
        let clamped = ScalarExpr::bin(
            BinOp::Min,
            ScalarExpr::bin(BinOp::Max, scaled, f32l(-127.0)),
            f32l(127.0),
        );
        let fake_quant = ScalarExpr::bin(
            BinOp::Mul,
            ScalarExpr::round(crate::dtype::RoundMode::HalfAwayFromZero, clamped),
            f32l(0.05),
        );
        let q = ts::map(&mut g, fake_quant.clone(), &[x]);
        assert_eq!(
            g.facts(q).numeric,
            crate::dtype::NumericContract::STRICT,
            "the fixture must actually carry the strict contract"
        );
        let fid = ts::fold(
            &mut g,
            ts::binop_carrier(BinOp::Add, Dtype::F32),
            1,
            Dtype::F32,
            q,
        );
        assert_eq!(g.facts(fid).numeric, crate::dtype::NumericContract::STRICT);
        g.add_root(fid);
        let report = saturate(&mut g);

        let absorbed: u32 = report
            .fired
            .iter()
            .find(|(n, _)| *n == "ABSORB")
            .map(|(_, c)| *c)
            .unwrap_or(0);
        assert!(absorbed > 0, "ABSORB did not fire under STRICT: {report:?}");
        let fused = kfolds_of(&g, fid)
            .into_iter()
            .find_map(|(_, k)| match k {
                L1::KFold { carrier, ops, .. } if ops.len() == 1 && ops[0].src == x => {
                    Some(carrier)
                }
                _ => None,
            })
            .expect("the quantized chain fused into one fold");
        // The absorbed lift is the fake-quant body term for term, so the fused
        // fold computes the same float the unfused chain did.
        assert_eq!(fused.lift[0], fake_quant);
        assert_eq!(fused.merge, ts::binop_carrier(BinOp::Add, Dtype::F32).merge);
        assert_eq!(
            fused.identity,
            ts::binop_carrier(BinOp::Add, Dtype::F32).identity
        );

        // The inexact law on the same value declines.
        assert_eq!(
            report
                .fired
                .iter()
                .find(|(n, _)| *n == "STRIP")
                .map(|(_, c)| *c)
                .unwrap_or(0),
            0,
            "STRIP split a value whose contract forbids reassociation"
        );
    }

    /// Nearest-neighbour assignment,
    /// `Fold{Min over M}( Fold{Add over D}( (a[n,d] - b[m,d])^2 ) )`:
    /// * the squared-difference map is absorbed into the inner reduction, so
    ///   the `[N, M, D]` difference tensor is not an operand of anything;
    /// * the inner reduction is not absorbed into the outer one, so it stays a
    ///   node with its own `work()` row.
    #[test]
    fn absorb_fires_on_nearest_neighbour_assignment() {
        use crate::shape::StrideSpec;
        let (n, m, d) = (3u64, 2u64, 4u64);
        let mut g = ts::graph();
        let a = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(n), Dim::Const(d)]);
        let bb = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(m), Dim::Const(d)]);
        let a_v = ts::restride(
            &mut g,
            &[
                StrideSpec::dim(0, Dim::Const(n)),
                StrideSpec::broadcast(Dim::Const(m)),
                StrideSpec::dim(1, Dim::Const(d)),
            ],
            a,
        );
        let b_v = ts::restride(
            &mut g,
            &[
                StrideSpec::broadcast(Dim::Const(n)),
                StrideSpec::dim(0, Dim::Const(m)),
                StrideSpec::dim(1, Dim::Const(d)),
            ],
            bb,
        );
        let diff = ScalarExpr::bin(
            BinOp::Sub,
            ScalarExpr::arg(0, Dtype::F32),
            ScalarExpr::arg(1, Dtype::F32),
        );
        let sq = ts::map(
            &mut g,
            ScalarExpr::bin(BinOp::Mul, diff.clone(), diff),
            &[a_v, b_v],
        );
        let dist = ts::fold(
            &mut g,
            ts::binop_carrier(BinOp::Add, Dtype::F32),
            2,
            Dtype::F32,
            sq,
        );
        let assign = ts::fold(
            &mut g,
            ts::binop_carrier(BinOp::Min, Dtype::F32),
            1,
            Dtype::F32,
            dist,
        );
        g.add_root(assign);
        let report = saturate(&mut g);
        assert!(report.saturated, "{report:?}");

        // The [N, M, D] difference tensor is gone from the inner reduction's
        // operand list.
        let inner = kfolds_of(&g, dist)
            .into_iter()
            .find_map(|(_, k)| match k {
                L1::KFold { carrier, ops, .. }
                    if ops.len() == 2 && ops[0].src == a_v && ops[1].src == b_v =>
                {
                    Some(carrier)
                }
                _ => None,
            })
            .expect("the squared difference was not absorbed");
        assert_eq!(inner.kind(), Some(BinOp::Add));
        // The lift is (a - b)^2.
        for (p, q) in [(1.5f32, -0.5f32), (0.0, 0.0), (2.25, 2.0)] {
            let got = inner.eval_lift(&[p, q]).unwrap()[0];
            assert!((got - (p - q) * (p - q)).abs() < 1e-6);
        }

        // The outer reduction still reads the inner one as an operand edge:
        // nothing inlined a reduction into a `ScalarExpr`.
        let outers = kfolds_of(&g, assign);
        assert!(!outers.is_empty(), "the assignment did not lower");
        let dist_class = g.class_of(dist);
        for (_, k) in &outers {
            let L1::KFold { ops, carrier, .. } = k else {
                unreachable!()
            };
            assert_eq!(ops.len(), 1, "the outer fold grew an operand");
            assert_eq!(
                g.class_of(ops[0].src),
                dist_class,
                "the outer fold stopped reading the inner reduction"
            );
            // The reduction performed is still the minimum: a `Min` at slot 0.
            assert!(
                carrier.kind() == Some(BinOp::Min)
                    || matches!(
                        carrier.merge[0].kind(),
                        ScalarKind::Bin {
                            op: BinOp::Min,
                            ..
                        }
                    ),
                "the outer reduction changed operator: {:?}",
                carrier.merge[0]
            );
        }
    }

    /// A `KMap` read by two `KFold`s: both folds gain a fused alternative and
    /// the map is still a live class member. There is no duplication veto.
    #[test]
    fn map_into_fold_fires_with_two_readers() {
        let mut g = ts::graph();
        let shape = [Dim::Const(4), Dim::Const(16)];
        let x = ts::buffer(&mut g, Dtype::F32, &shape);
        let m = ts::kmap(
            &mut g,
            &shape,
            ScalarExpr::un(UnOp::Exp, ScalarExpr::arg(0, Dtype::F32)),
            vec![alias_operand_of(x, &shape)],
        );
        let mops = vec![alias_operand_of(m, &shape)];
        let f1 = ts::kfold(
            &mut g,
            &shape,
            1,
            ts::binop_carrier(BinOp::Add, Dtype::F32),
            Dtype::F32,
            ident_expr(Dtype::F32),
            mops.clone(),
        );
        let f2 = ts::kfold(
            &mut g,
            &shape,
            1,
            ts::binop_carrier(BinOp::Max, Dtype::F32),
            Dtype::F32,
            ident_expr(Dtype::F32),
            mops,
        );
        assert!(fire(&mut g, f1, &ABSORB).is_some());
        assert!(fire(&mut g, f2, &ABSORB).is_some());
        for f in [f1, f2] {
            let Op::L1(L1::KFold { carrier: before, .. }) = g.node(f).op.clone() else {
                panic!()
            };
            let members = g.chain(f);
            assert_eq!(members.len(), 2, "fold {f} did not gain an alternative");
            let alt = members.into_iter().find(|&i| i != f).unwrap();
            let Op::L1(L1::KFold { carrier, ops, .. }) = &g.node(alt).op else {
                panic!()
            };
            assert_eq!(ops.len(), 1);
            assert_eq!(ops[0].src, x);
            assert!(matches!(
                carrier.lift[0].kind(),
                crate::scalar::ScalarKind::Un { .. }
            ));
            // The merge is untouched: substitution into a lift reassociates
            // nothing.
            assert_eq!(carrier.merge, before.merge);
            assert_eq!(carrier.identity, before.identity);
        }
        // The producer is still a live member of its own class.
        assert!(g.chain(m).contains(&m));
    }

    #[test]
    fn fold_post_epilogue_extends_post() {
        let mut g = ts::graph();
        let shape = [Dim::Const(4), Dim::Const(16)];
        let x = ts::buffer(&mut g, Dtype::F32, &shape);
        let fold = ts::kfold(
            &mut g,
            &shape,
            1,
            ts::binop_carrier(BinOp::Add, Dtype::F32),
            Dtype::F32,
            ident_expr(Dtype::F32),
            vec![alias_operand_of(x, &shape)],
        );
        let out = [Dim::Const(4)];
        let epilogue = ts::kmap(
            &mut g,
            &out,
            ScalarExpr::un(UnOp::Sqrt, ScalarExpr::arg(0, Dtype::F32)),
            vec![alias_operand_of(fold, &out)],
        );
        assert!(fire(&mut g, epilogue, &FOLD_POST_EPILOGUE).is_some());
        let alt = g.chain(epilogue).into_iter().find(|&i| i != epilogue).unwrap();
        let Op::L1(L1::KFold { post, .. }) = &g.node(alt).op else {
            panic!()
        };
        assert_eq!(post.len(), 1);
        assert!(matches!(post[0].kind(), crate::scalar::ScalarKind::Un { .. }));
    }

    #[test]
    fn map_into_contract_absorbs_a_unary_pre() {
        let mut g = ts::graph();
        let ashape = [Dim::Const(4), Dim::Const(8)];
        let bshape = [Dim::Const(8), Dim::Const(2)];
        let raw = ts::buffer(&mut g, Dtype::F32, &ashape);
        let pre = ts::kmap(
            &mut g,
            &ashape,
            ScalarExpr::un(UnOp::Abs, ScalarExpr::arg(0, Dtype::F32)),
            vec![alias_operand_of(raw, &ashape)],
        );
        let rhs = ts::buffer(&mut g, Dtype::F32, &bshape);
        let c = ts::kcontract(
            &mut g,
            Dim::Const(4),
            Dim::Const(2),
            Dim::Const(8),
            ident_expr(Dtype::F32),
            alias_operand_of(pre, &ashape),
            alias_operand_of(rhs, &bshape),
        );
        assert!(fire(&mut g, c, &MAP_INTO_CONTRACT).is_some());
        let alt = g.chain(c).into_iter().find(|&i| i != c).unwrap();
        let Op::L1(L1::KContract { a, .. }) = &g.node(alt).op else {
            panic!()
        };
        assert_eq!(a.primary().src, raw);
        assert!(matches!(a.pre.kind(), crate::scalar::ScalarKind::Un { .. }));
    }

    /// A reader that names only part of its producer may not absorb it.
    ///
    /// The substituted body is written against the producer's own coordinate,
    /// so an operand whose `Alias` layout carries an offset reads a window and
    /// the fused spelling would read the whole buffer.
    #[test]
    fn map_into_map_refuses_an_operand_that_names_a_window() {
        let mut g = ts::graph();
        let whole = [Dim::Const(4)];
        let part = [Dim::Const(2)];
        let x = ts::buffer(&mut g, Dtype::F32, &whole);
        let producer = ts::kmap(
            &mut g,
            &whole,
            ScalarExpr::un(UnOp::Exp, ScalarExpr::arg(0, Dtype::F32)),
            vec![alias_operand_of(x, &whole)],
        );
        // `producer[2..4]`, spelled the way a collapsed narrowing view reaches
        // a reader: a plain alias with a non-zero offset.
        let windowed = Operand {
            src: producer,
            layout: Layout::from_parts(Dim::Const(2), &part, &[Dim::Const(1)]).unwrap(),
            access: AccessPlan::Alias,
        };
        let consumer = ts::kmap(
            &mut g,
            &part,
            ScalarExpr::un(UnOp::Tanh, ScalarExpr::arg(0, Dtype::F32)),
            vec![windowed],
        );
        assert!(
            fire(&mut g, consumer, &MAP_INTO_MAP).is_none(),
            "absorbing across an offset alias drops the window"
        );

        // The same edge at offset 0 over the producer's own extent is the
        // dense read the law states.
        let dense = ts::kmap(
            &mut g,
            &whole,
            ScalarExpr::un(UnOp::Tanh, ScalarExpr::arg(0, Dtype::F32)),
            vec![alias_operand_of(producer, &whole)],
        );
        assert!(fire(&mut g, dense, &MAP_INTO_MAP).is_some());
    }

    #[test]
    fn form_kregion_names_the_escaping_producer() {
        let mut g = ts::graph();
        let shape = [Dim::Const(4), Dim::Const(16)];
        let x = ts::buffer(&mut g, Dtype::F32, &shape);
        let m = ts::kmap(
            &mut g,
            &shape,
            ScalarExpr::un(UnOp::Exp, ScalarExpr::arg(0, Dtype::F32)),
            vec![alias_operand_of(x, &shape)],
        );
        let fold = ts::kfold(
            &mut g,
            &shape,
            1,
            ts::binop_carrier(BinOp::Add, Dtype::F32),
            Dtype::F32,
            ident_expr(Dtype::F32),
            vec![alias_operand_of(m, &shape)],
        );
        assert!(fire(&mut g, fold, &FORM_KREGION).is_some());
        let region = g
            .chain(fold)
            .into_iter()
            .find(|&i| matches!(g.node(i).op, Op::L1(L1::KRegion { .. })))
            .expect("a region alternative");
        let Op::L1(L1::KRegion {
            members, live_outs, ..
        }) = &g.node(region).op
        else {
            panic!()
        };
        assert_eq!(members[0], m);
        assert_eq!(&live_outs[..], &[0]);
    }

    /// `MAP_INTO_MAP` counts the `Uniforms` block against
    /// `max_storage_buffers_per_shader_stage`, so the widest list it will
    /// mint is `limit - 2` distinct reads: those, plus the output, plus the
    /// block. Both sides of the boundary are asserted: the widest legal list
    /// fuses, one wider declines.
    #[test]
    fn map_into_map_leaves_room_for_the_uniform_block() {
        let limit = ts::caps().limits.max_storage_buffers_per_shader_stage as usize;
        let shape = [Dim::Const(4), Dim::Const(16)];

        // `reads` distinct buffers feed a producer; the consumer reads the
        // producer and nothing else, so the fused list is exactly `reads`.
        let chain = |reads: usize| -> Option<Id> {
            let mut g = ts::graph();
            let srcs: Vec<Id> = (0..reads)
                .map(|_| ts::buffer(&mut g, Dtype::F32, &shape))
                .collect();
            // A distinct body per operand count, or hash-consing would make
            // one node of two runs.
            let body = srcs.iter().enumerate().fold(
                ScalarExpr::arg(0, Dtype::F32),
                |acc, (i, _)| {
                    if i == 0 {
                        acc
                    } else {
                        ScalarExpr::bin(BinOp::Add, acc, ScalarExpr::arg(i as u32, Dtype::F32))
                    }
                },
            );
            let producer = ts::kmap(
                &mut g,
                &shape,
                body,
                srcs.iter().map(|s| alias_operand_of(*s, &shape)).collect(),
            );
            let consumer = ts::kmap(
                &mut g,
                &shape,
                ScalarExpr::un(UnOp::Exp, ScalarExpr::arg(0, Dtype::F32)),
                vec![alias_operand_of(producer, &shape)],
            );
            fire(&mut g, consumer, &MAP_INTO_MAP)
        };

        assert!(
            chain(limit - 2).is_some(),
            "{} reads plus the output plus the Uniforms block is exactly {limit}; it fits",
            limit - 2
        );
        assert!(
            chain(limit - 1).is_none(),
            "{} reads needs {} storage buffers at a limit of {limit}",
            limit - 1,
            limit + 1
        );
    }

    /// The same boundary on the reducing consumer.
    /// `splice_through_address_map` brings the producer's *whole* operand list
    /// into a fold, so a nest can outgrow the bind group in one step.
    #[test]
    fn absorb_leaves_room_for_the_uniform_block() {
        use crate::carrier::Carrier;
        use crate::dtype::Splat;

        let limit = ts::caps().limits.max_storage_buffers_per_shader_stage as usize;
        let shape = [Dim::Const(4), Dim::Const(16)];

        // `reads` distinct buffers feed an elementwise producer; the fold
        // reads the producer and nothing else, so absorbing it leaves exactly
        // `reads` operand edges.
        let chain = |reads: usize| -> Option<Id> {
            let mut g = ts::graph();
            let srcs: Vec<Id> = (0..reads)
                .map(|_| ts::buffer(&mut g, Dtype::F32, &shape))
                .collect();
            let body = (1..reads).fold(ScalarExpr::arg(0, Dtype::F32), |acc, i| {
                ScalarExpr::bin(BinOp::Add, acc, ScalarExpr::arg(i as u32, Dtype::F32))
            });
            let producer = ts::kmap(
                &mut g,
                &shape,
                body,
                srcs.iter().map(|s| alias_operand_of(*s, &shape)).collect(),
            );
            let fold = ts::kfold(
                &mut g,
                &shape,
                1,
                Carrier::binop(BinOp::Add, Splat::F32(0.0), Dtype::F32),
                Dtype::F32,
                ScalarExpr::arg(0, Dtype::F32),
                vec![alias_operand_of(producer, &shape)],
            );
            fire(&mut g, fold, &ABSORB)
        };

        assert!(
            chain(limit - 2).is_some(),
            "{} reads plus the output plus the Uniforms block is exactly {limit}; it fits",
            limit - 2
        );
        assert!(
            chain(limit - 1).is_none(),
            "{} reads needs {} storage buffers at a limit of {limit}",
            limit - 1,
            limit + 1
        );
    }

    /// `storage_bindings` counts distinct reads plus the output plus the
    /// `Uniforms` block.
    #[test]
    fn storage_bindings_counts_the_output_and_the_uniform_block() {
        let mut g = ts::graph();
        let shape = [Dim::Const(4)];
        let a = ts::buffer(&mut g, Dtype::F32, &shape);
        let b_ = ts::buffer(&mut g, Dtype::F32, &shape);
        let caps = ts::caps();
        let builder = g.builder(&caps);

        let one = [alias_operand_of(a, &shape)];
        assert_eq!(storage_bindings(&builder, &one), 3);
        let two = [alias_operand_of(a, &shape), alias_operand_of(b_, &shape)];
        assert_eq!(storage_bindings(&builder, &two), 4);
        // The same value twice is one binding: `derive_bindings` dedups reads.
        let dup = [alias_operand_of(a, &shape), alias_operand_of(a, &shape)];
        assert_eq!(storage_bindings(&builder, &dup), 3);
    }
}
