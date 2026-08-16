//! ABSORB — a reduction nest absorbs a producer whose index space it covers.
//!
//! Let `F = Fold{space, axis, vec_axes, carrier C, ops}` with iteration space
//! `E(F) = space` minus `vec_axes`. Any operand `ops[i]` produced by `P` where
//! `E(F).covers(space(P))` is absorbed into **every slot's** lift:
//!
//! ```text
//! Fold{C, ops}  ==  Fold{C[lift[k] := lift[k]{Arg(i) := body(P)}], ops[i := ops(P)]}
//! ```
//!
//! Substitution into a lift reassociates nothing, so this law carries no
//! `reassoc` guard and fires under
//! [`NumericContract::STRICT`](crate::dtype::NumericContract::STRICT).
//!
//! Greedy: the matcher walks the maximal chain of absorbable producers and
//! mints ONE fold with the fully composed lift. Intermediate partial
//! absorptions are not minted.
//!
//! When the producer is itself a reducing nest, the edge is left alone:
//! inline-vs-materialize is the extractor's `M` bit, and the inner fold keeps
//! a real `work()` row.
//!
//! There is no reader-count check. If a producer is read twice, both readers
//! may absorb it and the pricing crate charges the recompute once per reader.
//! `Region` is the same rewrite with `live_outs` non-empty.
//!
//! [`MAP_INTO_MAP`] is the same law with a `Map` in the consumer position.

use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::launch::{
    AccessPlan, ContractSide, IndexSpace, Launch, MapDomain, Operand, ScheduleDomain,
};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::rules::{MapView, access_legal_in, map_view, operand_dtypes, shift_args};
use crate::scalar::{ScalarExpr, ScalarKind};
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

rule!(
    FORM_KREGION,
    level = Level::Launch,
    head = OpTag::LaunchFold,
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
/// Legality: the reader's *iteration* space must cover the producer's, and
/// every operand the producer brings must satisfy the reader's access
/// predicate. The operand being replaced must be a plain alias, since
/// absorbing a `Pack` or `Gather` read would silently drop the repack.
///
/// `space` is the full index space and `iter` is `space` minus `vec_axes`.
/// They differ only on a promoted fold: an operand's address map is stated
/// against the full `space`, while every `ScalarExpr` on the node — including
/// the body being substituted in — is written against `iter`.
///
/// `AccessPlan::Alias` is not on its own the condition: the substituted body
/// is written against the producer's own coordinate, so the edge must read
/// the producer densely at the consumer's iteration coordinate. An `Alias`
/// layout can carry an offset and strides — a window — and splicing across it
/// silently reads the whole buffer. The dense-read condition is checked with
/// the same helper [`splice_through_address_map`] uses. A genuine broadcast
/// still passes: [`widen_groups`] gives stride 0 on every axis the producer
/// does not name.
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
    new_ops.extend(
        ops.iter()
            .enumerate()
            .filter(|(j, _)| *j != slot)
            .map(|(_, o)| o.clone()),
    );
    new_ops.extend(inner.ops.iter().cloned());
    Some(Spliced { ops: new_ops, args })
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
/// The consumer's ITERATION axes are `space` minus `vec_axes`, in order, and
/// the producer's space is a prefix of them. Every axis the producer does not
/// name — a promoted axis, or a trailing iteration axis past the producer's
/// rank — contributes stride 0: the producer's value is re-read at every
/// position of that axis, so an absorbed producer needs no renumbering.
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
        // else is a map that does not describe this space.
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

/// Spell a widened map as an operand, preferring the **simple** spelling.
///
/// One `AxisGroup` per axis with one sub-axis each is a stride vector, so it
/// is minted as a plain `Alias` layout over `space` — every other rule's
/// dependence query, projection and layout check is written against `Alias`
/// first, and a value spelled the exotic way silently falls out of them.
/// `Unflatten` is kept for a genuine divmod decomposition.
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
/// Stated as an `AddressMap` equality against [`dense_read_map`], which
/// derives its divisors from const extents. A `Dim::Sym` axis has no such
/// map, so the fallback checks the part decidable without extents: a non-zero
/// offset is a window whatever the extents are. A permuted or strided read
/// over a symbolic axis is not caught.
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
        _ => o.layout.offset().known_eq(Dim::Const(0)),
    }
}

/// The address map an operand would present if it read `producer_shape`
/// densely at the consumer's iteration coordinate.
fn dense_read_map(
    producer_shape: &[Dim],
    space: &IndexSpace,
    vec_axes: &[u32],
) -> Option<crate::ir::launch::AddressMap> {
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

/// Absorb across an operand edge that carries a non-trivial address map.
///
/// An `Unflatten` map is pure index arithmetic. The condition is that the
/// edge reads the producer at the consumer's iteration coordinate — then the
/// producer's body may be substituted unrenumbered and each of its own
/// operands restated over the wider space by [`widen_groups`].
///
/// Checked as an equality of `AddressMap`s, which is strictly sharper than
/// the prefix `covers` test: on a shape where a free axis and the reduced
/// axis share an extent, `covers` passes spuriously and this check rejects
/// the absorption.
///
/// This is the clause PROMOTE's output needs: after promotion the iteration
/// space is the producer's space exactly, and the edge becomes absorbable
/// with no renumbering at all.
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
    // Exact equality, both directions. `covers` is a prefix test, so it also
    // admits a producer of strictly smaller rank whose value is broadcast
    // along the consumer's trailing iteration axes; the substituted body
    // would then be re-read at a coordinate the producer never named.
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
    new_ops.extend(
        ops.iter()
            .enumerate()
            .filter(|(j, _)| *j != slot)
            .map(|(_, o)| o.clone()),
    );
    for o in &inner.ops {
        // Collapse a pure view into the layout first, with the same helper the
        // dependence query uses: the floor spells a broadcast as a `Restride`
        // node with a dense reading layout, so widening the spelling would
        // state stride 1 on an axis the value does not vary along.
        let (o, _) = crate::rules::rebase::effective(b, o, &inner.space);
        // Allocation is not described at Launch, so an edge that collapsed a
        // narrowing view into a non-zero offset is a node `verify_launch` rejects.
        if !o.layout.offset().known_eq(Dim::Const(0)) {
            return None;
        }
        let groups = widen_groups(&operand_groups(&o)?, space, vec_axes)?;
        new_ops.push(operand_from_groups(&o, &groups, space)?);
    }
    Some(Spliced { ops: new_ops, args })
}

/// Whether `inner`'s body may be substituted into a nest iterating `iter`.
///
/// `iter.covers(inner.space)` is a prefix test on the iteration space. When
/// the producer's body reads an `IndexOf`, the two spaces must agree exactly:
/// only then does the coordinate the body names survive substitution
/// unrenumbered.
fn covers_for_substitution(iter: &IndexSpace, inner: &MapView) -> bool {
    if !iter.covers(&inner.space) {
        return false;
    }
    !reads_index_of(&inner.body) || inner.space.covers(iter)
}

/// Whether `e` names a loop coordinate anywhere.
fn reads_index_of(e: &ScalarExpr) -> bool {
    match e.kind() {
        ScalarKind::IndexOf(_) => true,
        ScalarKind::Un { x, .. }
        | ScalarKind::Cast { x, .. }
        | ScalarKind::Bitcast { x, .. }
        | ScalarKind::Round { x, .. }
        | ScalarKind::Splat { x, .. } => reads_index_of(x),
        ScalarKind::Bin { a, b, .. } | ScalarKind::Cmp { a, b, .. } | ScalarKind::Dot { a, b } => {
            reads_index_of(a) || reads_index_of(b)
        }
        ScalarKind::Select { c, t, f } => {
            reads_index_of(c) || reads_index_of(t) || reads_index_of(f)
        }
        ScalarKind::Arg(_) | ScalarKind::Lit(_) | ScalarKind::Uniform(_) => false,
    }
}

/// The first operand slot of `ops` that can be absorbed, spliced.
///
/// A producer that is itself a reducing nest is not matched here:
/// [`map_view`] reads elementwise producers only, so a fold-to-fold edge is
/// left alone by construction.
///
/// KNOWN GAP: the two paths restate the producer's operands differently.
/// [`splice_through_address_map`] widens each one through [`widen_groups`]
/// onto the consumer's full `space`; [`splice`] clones them at the producer's
/// own rank. On a promoted consumer the rank mismatch makes
/// [`build_absorbed_fold`]'s access check discard the whole fused chain.
/// Teaching `splice` to widen only pays once `fusor_cost::extract`'s accept
/// test is the plan's own cost — see the note there.
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

/// Operand-list ceiling. Absorption terminates on its own — each step
/// replaces one operand by producers with strictly smaller ids — but a
/// producer read twice by the same chain widens the list, so this bounds the
/// term the rule builds.
const MAX_ABSORBED_OPERANDS: usize = 32;

/// ABSORB, greedy: absorb the maximal chain of elementwise producers into
/// every slot's lift and mint ONE fold.
pub fn absorb(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let fused = build_absorbed_fold(b, node, f)?;
    b.union(id, fused).ok()
}

fn build_absorbed_fold(b: &mut Builder<'_>, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(
        k @ Launch::Fold {
            space,
            axis,
            vec_axes,
            carrier,
            acc,
            post,
            ops,
            sched,
        },
    ) = &node.op
    else {
        return None;
    };
    if ops.is_empty() {
        return None;
    }
    // `f.own()` is the meet over *every* operand. `f.numeric(0)` reads operand
    // zero alone and is blind on a multi-operand fold, which is what this
    // fold becomes the moment it absorbs anything.
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
        // Stop at the last operand list the device can bind: one launch is
        // one bind group, so a fused nest reading more distinct buffers than
        // `max_storage_buffers_per_shader_stage` allows is a kernel the
        // backend cannot create, and extraction has already committed by the
        // time `create_bind_group_layout` says so.
        if storage_bindings(b, &spliced.ops) > budget {
            break;
        }
        // Substituted into EVERY slot's lift. A carrier is one expression per
        // slot; absorbing into slot 0 alone is how a multi-slot fold silently
        // computes one right answer and one wrong one.
        lift = lift.iter().map(|l| l.compose(&spliced.args)).collect();
        cur = spliced.ops;
        fired = true;
    }
    if !fired {
        return None;
    }
    let fused = Launch::Fold {
        space: space.clone(),
        axis: *axis,
        vec_axes: vec_axes.clone(),
        carrier: carrier.clone().with_lift(lift),
        acc: *acc,
        post: post.clone(),
        ops: cur,
        sched: sched.clone(),
    };
    // Checked before minting: an absorbed operand whose layout does not match
    // this nest's index space is a node `verify_plan` would reject.
    crate::verify_launch::check_operand_access(&fused).ok()?;
    b.add_launch(fused).ok()
}

/// Storage bindings one launch rooted at a nest with these operands needs:
/// the distinct non-free values it reads, its own output, and the `Uniforms`
/// block.
///
/// `derive_bindings` reserves binding 0 for the uniform block, drops
/// `LeafRole::Free` reads and deduplicates by value. Two operands naming
/// different members of one class would over-count by one — the conservative
/// direction.
///
/// The `Uniforms` block is declared in the `storage` address space, so it is
/// charged against `max_storage_buffers_per_shader_stage` like any other
/// buffer; leaving it out of the `+ 2` is a one-buffer under-count.
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
/// into this map's own body and mint ONE map.
///
/// Nothing calls `ScalarExpr::compose` at construction, so a map chain
/// reaches saturation as separate nodes, and a launch is lowered from one
/// node — without this rule each map is its own dispatch.
///
/// No reader-count check, per this file's contract. The un-absorbed map stays
/// in the class, so materializing it once remains available.
///
/// `map_view` is asked about the operand's id, not its class: offering
/// every class member widens the extraction frontier under a fixed move
/// budget and was measured as a net regression.
pub fn map_into_map(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(Launch::Map {
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
            // A `Map` has no `vec_axes`, so its iteration space is its
            // index space and the promoted dispatch has nothing to add.
            splice(b, &cur, i, &view, space, space, &[])
        }) else {
            break;
        };
        // Stop at the last operand list the device can bind: one launch is
        // one bind group, so a fused map reading more distinct buffers than
        // `max_storage_buffers_per_shader_stage` allows is a kernel the
        // backend cannot create.
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
    let fused = Launch::Map {
        space: space.clone(),
        body: expr,
        ops: cur,
        sched: sched.clone(),
    };
    crate::verify_launch::check_operand_access(&fused).ok()?;
    let fused = b.add_launch(fused).ok()?;
    b.union(id, fused).ok()
}

/// Inline a single-operand elementwise producer into `pre_a` or `pre_b`.
///
/// `Contract` carries exactly two operand edges, so only a one-operand
/// producer can be absorbed without inventing a third edge.
pub fn map_into_contract(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(Launch::Contract {
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
    // A contraction binds every operand of both sides in one launch, so the
    // count that has to fit is the union, not either side's own.
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
        .add_launch(Launch::Contract {
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
/// The producer's own arity is not a condition — its operands join the
/// side's list. The slot must be a plain alias, and each operand the
/// producer brings must satisfy the reader's access predicate over the
/// contraction's `(batch, m, n, k)` space.
///
/// Every eligible slot absorbs in one fire. Absorbing one slot per firing
/// fills the class with every order the absorptions could happen in and the
/// graph grows without bound; one fire per side keeps the successors a chain,
/// linear in producer depth. The class still holds the un-absorbed node and
/// the fully absorbed one, which are the two the cost model chooses between.
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
        // splice: the contraction path has no later rule to fold it, so a
        // carried `Restride` class would materialize as its own launch. A
        // spine that does not compose to an offset-0 plain layout is left
        // alone: the operand stays legal, just materialized.
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
            let Op::Logical(crate::ir::logical::Logical::Restride { specs, .. }) =
                b.node(spine.views[0]).op.clone()
            else {
                continue;
            };
            let base_shape = b.facts_of(spine.base).shape.clone();
            let Some(composed) = crate::rules::composed_layout(&specs, &base_shape) else {
                continue;
            };
            // Clause 8: a Launch operand may not name a buffer offset.
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
        // A producer reading a quantized leaf never absorbs: splicing the
        // identity map `LOWER_DEQUANT` mints recreates a raw-quantized
        // contraction operand on whatever family and orientation this node
        // has, where the block decode's (row, col) addressing does not hold.
        // The raw-quantized spelling is already in the class, minted by
        // `lower_family` under the one family whose staging fill is written
        // for it.
        if inner
            .ops
            .iter()
            .any(|p| b.facts_of(p.src).dtype.is_quantized())
        {
            return None;
        }
        // Any operand still naming a pure-view class after the fold above
        // declines. Its layout was fabricated as the dense read of the view's
        // value, and inside a contraction nothing later re-points it at the
        // base: the view class materializes, or this side's matrix view reads
        // the base's buffer through the dense lie.
        if inner
            .ops
            .iter()
            .any(|p| !b.trace_pure_views(p.src).views.is_empty())
        {
            return None;
        }
        // The edge may read the producer through any axis permutation of its
        // dense value, and the permutation must survive absorption: it is
        // carried by permuting every absorbed operand's own axes with it
        // (see [`permute_layout`]). An edge that is not a permutation (a
        // window, a broadcast, an offset) declines.
        let perm = dense_permutation(&o.layout, &inner.space.dims)?;
        // A body reading its own coordinates absorbs too: the contraction's
        // staging loop hands `pre` the operand-axis coordinate vector, and
        // producer axis `perm[j]` is operand axis `j`, so the axis names
        // shift by the inverse. This lets a structural causal mask ride into
        // the contraction instead of materializing the masked scores.
        if reads_index_of(&inner.body) {
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
/// `perm[j] = i` means the edge's axis `j` walks the producer's axis `i`:
/// each of the layout's `(extent, stride)` pairs must be exactly one
/// producer axis's `(extent, row-major stride)`, offset zero, every axis
/// claimed once. The identity read — the common case — is the identity
/// permutation. A window (offset), a broadcast (stride 0 where the value has
/// none) or a gather-shaped read all fail the match and refuse absorption.
///
/// Axes may repeat an extent; matching on the *pair* keeps the bijection
/// unambiguous wherever it matters, because equal extents with equal strides
/// address identically whichever way they are paired.
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
    let Op::Launch(Launch::Fold {
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
    // A `Map` body reads one value; a multi-slot fold offers several, and
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
        .add_launch(Launch::Fold {
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

/// The linear schedule domain of a composite whose value is `landed`'s.
///
/// `verify_launch` recomputes exactly this from the composite's own inferred
/// facts, so the two cannot drift.
pub fn linear_domain_of(b: &Builder<'_>, landed: Id) -> ScheduleDomain {
    ScheduleDomain::Map(MapDomain::linear_over(b.caps(), &b.facts_of(landed).shape))
}

/// The multi-output form of [`absorb`]: the absorbed producer also escapes,
/// so the fused chain becomes a `Region` naming it in `live_outs`. Because
/// the region and the plain absorbed fold are both live, emitting the extra
/// buffer competes with recomputing it.
pub fn form_kregion(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Launch(k @ Launch::Fold { ops, .. }) = &node.op else {
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
    // `live_outs: [0]` names the producer, so the region lands the producer's
    // value and that is the index space its schedule domain is derived from —
    // the same one `verify_launch` recomputes from the region's inferred facts.
    let sched = linear_domain_of(b, producer);
    let region = b
        .add_launch(Launch::Region {
            members,
            live_outs: smallvec::smallvec![0],
            sched,
        })
        .ok()?;
    b.union(id, region).ok()
}
