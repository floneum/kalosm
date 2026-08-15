//! ABSORB — a reduction nest absorbs a producer whose index space it covers.
//!
//! Let `F = KFold{space, axis, vec_axes, carrier C, ops}` with iteration space
//! `E(F) = space` minus `vec_axes`. Any operand `ops[i]` produced by `P` where
//! `E(F).covers(space(P))` is absorbed into **every slot's** lift:
//!
//! ```text
//! KFold{C, ops}  ==  KFold{C[lift[k] := lift[k]{Arg(i) := body(P)}], ops[i := ops(P)]}
//! ```
//!
//! Substitution into a lift reassociates nothing: identity, merge,
//! associativity and every schedule survive unchanged. So this law carries
//! **no `reassoc` guard** — and requiring one would be a bug, because it would
//! kill fusion exactly on the QAT/MSQ1 path where
//! [`NumericContract::STRICT`](crate::dtype::NumericContract::STRICT) holds and
//! every inexact law declines. That exactness is the sharpest thing about it.
//!
//! **Greedy.** The matcher walks the maximal chain of absorbable producers and
//! mints ONE fold with the fully composed lift, so an `n`-map chain collapses
//! in one round instead of `n`. Intermediate partial absorptions are not
//! minted: the extractor's materialization set ranges over the *operand edges*
//! of the two extremes, which is what actually spans the fusion lattice.
//!
//! **The reduction-nesting clause is not a rewrite.** When the producer is
//! itself a reducing nest, the edge is left alone: inlining it is a nested
//! loop in one kernel body, materializing it is a buffer, and that is exactly
//! the extractor's `M` bit. Nothing enters a `ScalarExpr`, so the inner fold
//! keeps a real `work()` row and the extractor prices its recompute honestly
//! instead of a `ScalarKind::Dot` reporting one op.
//!
//! There is no reader-count check and no duplication veto. If a producer is
//! read twice, both readers may absorb it and the pricing crate charges the
//! recompute once per reader against the write and the reads it deletes.
//! `KRegion` is the same rewrite with `live_outs` non-empty.
//!
//! Elementwise-into-elementwise is `ScalarExpr::compose`, a tree
//! substitution — but **nothing calls it at construction**, so it is a rule
//! here too: [`MAP_INTO_MAP`], the same law with a `KMap` in the consumer
//! position. `Map{exp}(Map{sub}(s, m))` is lowered from one node at L1 as a
//! `KMap`.

use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::level1::{AccessPlan, ContractSide, IndexSpace, L1, MapDomain, Operand, ScheduleDomain};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::rules::{MapView, access_legal_in, map_view, operand_dtypes, shift_args};
use crate::scalar::{ScalarExpr, ScalarKind};
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
/// # `AccessPlan::Alias` is not on its own the condition
///
/// The substitution is `Arg(slot) := inner.body`, and that body is written
/// against the producer's **own** coordinate. Dropping the operand is
/// therefore sound only when the edge reads the producer at the consumer's
/// iteration coordinate — which an `Alias` does *not* imply. An `Alias` layout
/// carries an offset and a stride vector, so `x.narrow(0, 0, 3)` reaching a
/// reader as `Operand { access: Alias, layout: { offset, strides } }` passes
/// the plan check and reads a window. Splicing across it computes the
/// producer's body at the reader's coordinate and silently reads the whole
/// buffer.
///
/// That is not hypothetical. On the standard sampler's `[16] -> narrow(3)`
/// chain the fused spelling makes the one-hot pick select every row, and
/// `gather_one_hot`'s `sum(one_hot * ids)` returns `0 + 1 + ... + 15 = 120`
/// for a 16-token vocabulary — an out-of-range token id, on both
/// `sample_standard_token_respects_top_k` and every other filtered draw. The
/// fused member sat in the class unselected, so the suite was green; it is
/// selected the moment extraction searches harder.
///
/// The condition [`splice_through_address_map`] states for the promoted case
/// is the right one for this case too, and it is checked here with the same
/// helper: the edge's `AddressMap` must be the dense read of the producer's
/// shape over `space`. A genuine broadcast still passes — [`widen_groups`]
/// gives stride 0 on every axis the producer does not name — so absorbing a
/// row statistic across a broadcast edge is unaffected.
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
/// The consumer's ITERATION axes are `space` minus `vec_axes`, in order, and
/// the producer's space is a prefix of them. Every axis the producer does not
/// name — a promoted axis, or a trailing iteration axis past the producer's
/// rank — contributes stride 0, which is exactly "the producer's value is
/// re-read at every position of that axis". That is the whole content of this
/// function and the reason an absorbed producer needs no renumbering: the
/// coordinate it names is the coordinate it named before.
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
        // the space has `N` is a broadcast the layout spelled as a unit axis;
        // anything else is a map that does not describe this space, and
        // placing it here would corrupt every divisor to its left.
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

/// Spell a widened map as an operand, preferring the **simple** spelling.
///
/// One `AxisGroup` per axis with one sub-axis each is a stride vector and
/// nothing more, so it is minted as a plain `Alias` layout over `space`. That
/// is not cosmetic: every other rule's dependence query, projection and
/// layout check is written against `Alias` first, and a value spelled the
/// exotic way silently falls out of all of them. `Unflatten` is kept for a
/// genuine divmod decomposition, which is the case it exists for.
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
/// The exact statement is an `AddressMap` equality against [`dense_read_map`],
/// which derives its divisors from const extents. A `Dim::Sym` axis — a decode
/// step's runtime sequence length — has no such map, and this law reads no
/// extent, so declining there would refuse the whole symbolic chain
/// (`rebase::tests::retarget_fires_over_a_symbolic_reduction_length` pins that
/// it must not). The fallback is the part of the condition that is decidable
/// without extents: an offset is a window into a larger buffer whatever the
/// extents are, and dropping it is the failure this guard exists for. A
/// permuted or strided read over a symbolic axis is **not** caught, and this
/// is the honest statement of that limit rather than a claim the check is
/// complete.
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

/// Absorb across an operand edge that carries a non-trivial **address map**.
///
/// The shipped clause requires `AccessPlan::Alias`, whose rationale is that
/// absorbing a `Pack` or a `Gather` would silently drop real work. An
/// `Unflatten` map is neither: it is pure index arithmetic, the same kind a
/// strided `Alias` layout already carries. What actually has to be true is
/// that the edge reads the producer **at the consumer's iteration
/// coordinate** — then the producer's body may be substituted unrenumbered
/// and each of its own operands restated over the wider space by
/// [`widen_groups`].
///
/// That condition is checked as an equality of `AddressMap`s, which is
/// strictly sharper than the prefix `covers` test it accompanies: on a shape
/// where a free axis and the reduced axis happen to share an extent, `covers`
/// passes spuriously and this check is what rejects the absorption.
///
/// This is the clause PROMOTE's output needs. Before promotion the output
/// nest iterates `[.., Lq, Dh, Lk]` and the probability matrix is read with a
/// stride-0 `Dh` axis *inside* the iteration domain, so no substitution is
/// sound. After promotion `Dh` is a carrier axis, the iteration space is the
/// producer's space exactly, and the same edge becomes absorbable with no
/// renumbering at all.
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
        // With no promoted axis `iter == space` and the shipped Alias path
        // already covers every edge this one would. Declining keeps the
        // unpromoted graph byte-identical.
        return None;
    }
    // **Exact equality, both directions.** `covers` is a prefix test, so it
    // also admits a producer of strictly smaller rank whose value is broadcast
    // along the consumer's trailing iteration axes. Widening across that is
    // where a measured A/B put the attention backward's `dq` on a wrong value:
    // the substituted body then has to be re-read at a coordinate the producer
    // never named, and the operand restatement below is not enough to say so.
    // The law's own statement of this case is the equal one — the promoted
    // output nest's iteration space *is* the score space — so require it.
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
        // Collapse a pure view into the layout FIRST, with the same helper the
        // dependence query uses. The floor spells a broadcast as a `Restride`
        // node and gives the reading edge a dense layout, so widening the
        // spelling would state stride 1 on an axis the value does not vary
        // along — and every downstream invariance query would then answer
        // "varies" about a row statistic that does not. Collapsing states the
        // read.
        let (o, _) = crate::rules::rebase::effective(b, o, &inner.space);
        // Allocation is not described at L1, so an edge that collapsed a
        // narrowing view into a non-zero offset is a node `verify_l1` rejects.
        // Declining here keeps this rule from minting one.
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
/// `iter.covers(inner.space)` is the shipped prefix test, read on the
/// **iteration** space. When the producer's body reads an `IndexOf`, the two
/// spaces must agree exactly: only then does the coordinate the body names
/// survive substitution unrenumbered. Under the prefix relation alone the
/// leading axes still line up, but an exact match is what the law states and
/// it is the condition under which a frontend's `select(IndexOf(j) <= …)`
/// rides into a carrier as an ordinary predicate.
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
/// A producer that is itself a reducing nest is **not** matched here:
/// [`map_view`] reads elementwise producers only, so a fold-to-fold edge is
/// left alone by construction. That is the nesting clause — an edge property
/// the extractor decides, never a rewrite.
///
/// # On a promoted nest this order is wrong, and the measurement says so
///
/// [`splice`] is tried first and **wins on a promoted nest**, where it is not
/// sound: `space` is one rank wider than the `iter` the producer's operands
/// are written against, so `splice` appends a rank-`|iter|` alias into a
/// rank-`|space|` nest. `check_operand_access` then rejects the fold —
/// measured on the frontend's own attention chain as
/// `operand 2: Alias layout is rank 4 but the index space is rank 5` — and
/// `absorb` mints **nothing**. That is why ABSORB fires on the promoted
/// attention output nest and the derivation stalls there.
///
/// Dispatching to [`splice_through_address_map`] whenever `vec_axes` is
/// non-empty fixes it and was measured end to end. It unlocks the whole
/// chain: the promoted output nest absorbs the softmax divide (4 operands,
/// access clean), RETARGET goes 2 -> 6, HOIST starts firing, and TUPLE — with
/// its axis test corrected to compare the *iteration* axis, since a promoted
/// nest's `axis` is shifted by its carrier axes — goes 0 -> 6 and mints the
/// `[Scalar, Vector(Dh)]` flash carrier in the rank-5 space.
///
/// This dispatch **is** shipped, and it is what makes the hand-written flash
/// template deletable: `L1::KFlash`, `FlashOut` and
/// `fusor2-gpu/src/lower/flash.rs` are gone, and attention is derived from
/// TUPLE / PROMOTE / RETARGET / ABSORB / HOIST / STRIP alone — no rule names
/// attention or softmax. Measured with it live: every attention case passes on
/// **both** backends, the forward launch ceilings held at 8 / 10 / 8 the day
/// this was taken — they are 5 / 6 / 5 now, see
/// `fusor2-conformance::launch_counts` for the history — and the
/// `[Lq, Lk]` score, probability and `dp` matrices stay out of the extracted
/// plan's materialized set, which is the half of the win a launch count cannot
/// see.
///
/// Two defects that once sat behind this dispatch are fixed and are recorded
/// because both were reached through it rather than caused by it:
///
/// * The GPU promoted multi-slot path in `fusor2-gpu/src/lower/map_fold.rs`
///   computed one output row and no more, so every attention forward read `0`
///   at `[0, 0, 1, 0]` and `dq[4]` was `0`. The joint exposed that row
///   mapping; it did not introduce it.
/// * Extraction priced the joint wrongly on GPU, regressing the counts 8 -> 10
///   and 10 -> 12.
///
/// The exact-equality test in [`splice_through_address_map`] is the other half:
/// under the prefix `covers` relation alone this dispatch put the attention
/// backward's `dq` on a wrong value, and that is why the test is stated in
/// both directions rather than as the prefix the shipped `splice` uses.
/// # KNOWN GAP: [`splice`] does not widen onto a promoted space
///
/// The two paths below restate the producer's operands differently.
/// [`splice_through_address_map`] pushes each one through [`widen_groups`] onto
/// the consumer's full `space`; [`splice`] clones them at the producer's own
/// rank. On an unpromoted consumer that is the same thing. On a promoted one it
/// is not, and the mismatch is silent in this file and fatal one line later:
/// `verify_l1::check_operand_access` requires an `Alias` layout's rank to be
/// the index space's, so [`build_absorbed_fold`] — which checks before minting
/// — discards **the whole fused chain**, not just the step that widened wrong.
///
/// Measured on the attention forward chain, round 4. `ABSORB` reaches the
/// promoted output accumulator (`KFold` over the key axis with `Dh` promoted,
/// reading `v`, the shifted exponential and the row sum) through the
/// address-map path on the first step and this path on the second, and then:
///
/// ```text
/// ABSORB: check_operand_access rejected space=[2,2,3,4,4] vec=[3] ops=3:
///   operand 2: Alias layout is rank 4 but the index space is rank 5
/// ```
///
/// four times per saturation, on both backends. Teaching `splice` to widen
/// (the identity when `vec_axes` is empty, so the unpromoted graph stays
/// byte-identical) mints the node and the extractor selects it: GPU
/// `attention_forward` 5 -> 4 launches, `attention_with_lse` 6 -> 5,
/// `attention_grads_all_three` 17 -> 16.
///
/// **It is not landed, and the reason is not this rule.** The widened graph is
/// twice the size (CPU 613 -> 1348 nodes), which halves the extraction move
/// budget, and at the shipped extractor the plan gets *worse* (5 -> 8
/// launches) rather than better. It only pays once
/// `fusor2_cost::extract`'s accept test is the plan's own cost — see the note
/// there — and that change exposes six latent wrong-value/illegal-plan members
/// elsewhere. The three landing conditions are recorded in both files so the
/// next round does not have to rediscover the order.
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
/// replaces one operand by producers with strictly smaller ids, which is a
/// decreasing multiset over a well-founded order — but a producer read twice
/// by the same chain widens the list, so this bounds the term the rule builds
/// the way `MAX_NODES` bounds the graph.
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
        // Stop at the last operand list the device can bind, exactly as
        // [`map_into_map`] does. This is legality, not thrift: one launch is
        // one bind group, so a fused nest reading more distinct buffers than
        // `max_storage_buffers_per_shader_stage` allows is a kernel the
        // backend cannot create, and extraction has already committed by the
        // time `create_bind_group_layout` says so.
        //
        // The sibling rule has had this since `mirostat2` reached nine
        // bindings on a limit of eight; this one never did, and the gap is
        // reachable rather than theoretical. Measured on the attention
        // backward with a wider absorption landed:
        // `attention_backward_matches_the_analytic_adjoints [gpu]` failed
        // `verify_plan` with "launch 15 binds 10 storage buffers — 9 operands
        // plus the Uniforms block — over the 8-buffer limit". Absorbing one
        // operand fewer is a plan; a rejected bind group layout is not a
        // fallback.
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
    // Invariant 5, checked before minting rather than after selecting: an
    // absorbed operand whose layout does not match this nest's index space is
    // a node `verify_plan` would reject as a hard assert.
    crate::verify_l1::check_operand_access(&fused).ok()?;
    b.add_l1(fused).ok()
}

/// Storage bindings one launch rooted at a nest with these operands needs:
/// the distinct non-free values it reads, its own output, **and the
/// `Uniforms` block**.
///
/// `derive_bindings` reserves binding 0 for the uniform block, drops
/// `LeafRole::Free` reads (a constant is folded, a uniform lives in binding
/// 0) and deduplicates by value, so this counts what it counts. Two operands
/// naming different members of one class would over-count by one — the
/// conservative direction, which declines a legal fusion rather than minting
/// an unbindable kernel.
///
/// # The `Uniforms` block is a storage buffer
///
/// It is not listed by `derive_bindings` and it is not an operand, so it is
/// easy to leave out of the arithmetic — and leaving it out is exactly a
/// one-buffer under-count, which is the whole margin this guard has. Every
/// emitted module declares it in the `storage` address space
/// (`fusor2_gpu::bindings` derives the bind group by walking storage globals
/// and rejects a module whose binding 0 is not the read-only `Uniforms`
/// buffer), so it is charged against
/// `max_storage_buffers_per_shader_stage` like any other.
///
/// Measured, and the reason the `+ 2` is written out rather than folded into
/// the caller's comparison: with `+ 1` and a move budget large enough to
/// reach the plan, `normalization::softmax_last_dim` extracted a `KMap` with
/// seven reads and one write — eight listed bindings, at a limit of eight —
/// and died in `create_bind_group_layout` needing nine. `verify_plan`'s
/// clause 7 is the same statement as an assert on the finished plan.
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
/// **This is the rule the module header says is unnecessary, and the header
/// is wrong about the shipped frontend.** `ScalarExpr::compose` is indeed all
/// the arithmetic there is, but nothing calls it at construction: `compose`
/// has exactly two callers outside the rules — a test and `shift_args` — so
/// `Map{exp}(Map{sub}(s, m))` reaches saturation as two nodes and leaves as
/// two nodes. A launch is lowered from **one** node, so two elementwise nodes
/// are two dispatches however cheap each is. Measured on the chain
/// `attention_defn` emits: the scale, the `s - m` shift and the `exp` are
/// three separate `KMap` launches over one `[B, H, Lq, Lk]` space, and the
/// probability divide is a fourth.
///
/// It is stated as a rule rather than fixed at the frontend for the reason
/// `rules.rs` gives about the deleted flash recognizers: a law that depends on
/// which spelling the frontend happens to emit stops firing when the frontend
/// changes, and nothing notices. This one reads only "my operand is an
/// elementwise value at a space I cover", which is `ABSORB`'s own predicate
/// with a `KMap` in the consumer position.
///
/// No reader-count check, per this file's contract: if the producer is read
/// twice, both readers may absorb it and the cost model charges the recompute
/// against the write and the reads it deletes. The un-absorbed map stays in
/// the class, so materializing it once remains available at the same price it
/// had before.
///
/// # MEASURED AND REJECTED: asking the operand's class instead of its id
///
/// [`map_view`] normalizes the two spellings *one id* can carry. It does not
/// see across a class: when the frontend builds an `L0::Restride` and
/// `LOWER_RESTRIDE` mints the `KMap` spelling beside it, the consumer's
/// operand still names the restride, the `KMap` sits behind a `Union`, and a
/// `Builder` walks unions downward only — so a pure broadcast (`body =
/// Arg(0)`) survives as its own dispatch on the attention forward chain. The
/// obvious repair is to offer every member of the operand's class here.
///
/// It was built (a `Builder::class_members` accessor plus a per-member
/// `find_map`, with a step counter for termination, since a lowered spelling
/// has a *larger* id than the one the frontend built and the well-founded
/// descent above no longer holds) and measured. It is a **regression**:
/// `attention_forward` cpu 7 -> 8 and `attention_with_lse` cpu 8 -> 9, while
/// gpu improved 7 -> 6 and `attention_causal_forward` cpu 7 -> 6. More
/// alternatives in the class is not free — the extraction move budget is
/// fixed, so the local search spends the same number of moves over a wider
/// frontier and lands somewhere else. Landing it would have traded two
/// documented CPU regressions for two GPU improvements and read as green,
/// because each ceiling is the larger of the two backends.
///
/// `sink::FOLD_VIEWS_INTO_INDEX` already mints the right alternative for this
/// shape; what does not happen is the extractor reaching it. That is a search
/// problem, and `ExtractBudget::default`'s doc records what raising the
/// budget does and what blocks it.
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
    // producers with strictly smaller ids, a decreasing multiset over a
    // well-founded order. The ceiling bounds the *width* a producer read
    // twice by one chain adds, not the depth.
    //
    // `map_view` is asked about the operand's **id**, not its class, and that
    // is deliberate — see the note on `MEASURED AND REJECTED` below.
    while cur.len() <= MAX_ABSORBED_OPERANDS {
        let Some(spliced) = cur.iter().enumerate().find_map(|(i, o)| {
            let view = map_view(b, o.src)?;
            // A `KMap` has no `vec_axes`, so its iteration space is its
            // index space and the promoted dispatch has nothing to add.
            splice(b, &cur, i, &view, space, space, &[])
        }) else {
            break;
        };
        // Stop at the last operand list the device can bind. This is
        // legality, not thrift: one launch is one bind group, so a fused map
        // reading more distinct buffers than
        // `max_storage_buffers_per_shader_stage` allows is a kernel the
        // backend cannot create — `mirostat2` reached nine on a limit of
        // eight and failed in `create_bind_group_layout`, after extraction
        // had already committed. Absorbing one operand fewer is a plan; a
        // rejected bind group layout is not a fallback.
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
/// # Why the producer's own arity is not a condition
///
/// Requiring `inner.ops.len() == 1` would make the *one* producer worth
/// absorbing permanently ineligible. The
/// GGUF block decode reads its block stream through several `Restride` views
/// at once (quant plane, block scale, block minimum, group scales), so it
/// arrives here with nine operands and no rewrite collapses them to one. Such
/// a guard would not restrict absorption to safe cases; it would exclude the
/// case.
///
/// With [`ContractSide`] holding a list the producer's operands simply join
/// the side's, and the quantized staging fill stops being a backend special
/// case. Every *other* condition is unchanged: the slot must be a plain
/// alias, and each operand the producer brings must satisfy the reader's
/// access predicate over the contraction's `(batch, m, n, k)` space.
///
/// # Every eligible slot absorbs in one fire, and that is not an optimization
///
/// The obvious spelling — absorb the first eligible slot and let the additive
/// rule re-fire — is what a one-operand side could afford. There, absorbing
/// left the arity at one, so the successors of a node formed a *chain* as
/// long as the producer chain was deep.
///
/// With a list they form a lattice. A side of width `w` with `e` eligible
/// slots has `e` distinct one-slot successors, each of which has its own
/// eligible set, so the class fills with every order in which the absorptions
/// could have been performed — nodes that all denote the same value and
/// differ only in how far along each edge the rewriting got. Measured: the
/// `sampling::standard` tests stopped terminating, spinning in `saturate`
/// with the graph still growing after twenty minutes, because the sampler's
/// `lm_head` contraction reads a multi-operand `where_cond` chain.
///
/// Absorbing every eligible slot at once gives each node exactly one
/// successor per side, so the successors are a chain again and the node count
/// is linear in producer depth. What is lost is the *partial* absorptions as
/// separate alternatives — the class still holds the un-absorbed node (this
/// rule is additive) and the fully absorbed one, which are the two the cost
/// model actually chooses between.
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
        // splice. A producer's operand routinely names a `Restride` class —
        // every broadcast row statistic does — and carrying that class into
        // the contraction forces it to materialize as its own launch, since
        // the contraction path has no later rule to fold it. Composed here it
        // is a stride vector (a broadcast is stride 0) and costs nothing. A
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
        // A producer reading a *quantized* leaf never absorbs. The identity
        // map `LOWER_DEQUANT` mints is exactly such a producer, and splicing
        // it recreates a raw-quantized contraction operand — but on whatever
        // family and orientation this node happens to have, where the block
        // decode's (row, col) addressing does not hold: absorbed into an
        // Sgemm with a transposed edge, Q4K's 64x64 case read the wrong
        // blocks with [0, 0] agreeing by symmetry. The raw-quantized
        // spelling this would recreate is already in the class (the frontend
        // unions both), minted by `lower_family` under the one family whose
        // staging fill is written for it.
        if inner
            .ops
            .iter()
            .any(|p| b.facts_of(p.src).dtype.is_quantized())
        {
            return None;
        }
        // Any operand still naming a pure-view class after the fold above is
        // a refusal, not a pass-through. Its layout was fabricated as the
        // dense read of the *view's* value (`map_view`'s `L0::Map` spelling
        // has nothing else to write), and inside a contraction nothing later
        // re-points it at the base: the view class materializes — or worse,
        // this side's matrix view reads the base's buffer through the dense
        // lie. A GGUF block prefix is exactly such a spine (its offset is
        // what clause 8 stops the fold from absorbing), and carrying it here
        // is where the Q4K 64x64 conformance case got 61.8 for 108.2.
        if inner
            .ops
            .iter()
            .any(|p| !b.trace_pure_views(p.src).views.is_empty())
        {
            return None;
        }
        // The edge may read the producer through any *axis permutation* of
        // its dense value — `permuted_alias` mints exactly that for a
        // transposed or batch-reordered contraction operand — and the
        // permutation must survive absorption. It is carried by permuting
        // every absorbed operand's own axes with it (see [`permute_layout`]);
        // an edge that is not a permutation (a window, a broadcast, an
        // offset) declines, since replacing its layout with the producer's
        // would silently re-read the whole dense value.
        let perm = dense_permutation(&o.layout, &inner.space.dims)?;
        // A body reading its own coordinates absorbs too: the contraction's
        // staging loop hands `pre` the operand-axis coordinate vector (see
        // the `coords` argument of each lowering's `eval_scalar`), and
        // producer axis `perm[j]` is operand axis `j`, so the axis names
        // shift by the inverse. This is what lets a structural causal mask —
        // `select(IndexOf(k) <= IndexOf(q) + off, s, -inf)` — ride into the
        // contraction instead of forcing the masked scores to materialize.
        if reads_index_of(&inner.body) {
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
    // body is shifted by the count of everything placed before it. This is
    // `splice`'s convention, and it keeps one side's numbering independent of
    // the other's arity.
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
/// of row-program cluster formation; the reference's collect / drop-violators
/// / re-walk fixpoint over reverse-topological roots is deleted.
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

/// The linear schedule domain of a composite whose value is `landed`'s.
///
/// `verify_l1` recomputes exactly this from the composite's own inferred
/// facts, so the two cannot drift.
pub fn linear_domain_of(b: &Builder<'_>, landed: Id) -> ScheduleDomain {
    ScheduleDomain::Map(MapDomain::linear_over(
        b.caps(),
        &b.facts_of(landed).shape,
    ))
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
    // `live_outs: [0]` names the producer, so the region lands the producer's
    // value and that is the index space its schedule domain is derived from —
    // the same one `verify_l1` recomputes from the region's inferred facts.
    let sched = linear_domain_of(b, producer);
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

    /// ABSORB fires on a **saturated** graph, greedily: a three-node
    /// elementwise chain under a reduction becomes ONE fold reading the
    /// buffer, with the whole chain in its lift, in one round rather than
    /// three.
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

        // Numerically: the fused lift computes `2 * exp(v)` per element and
        // the fold sums it, against an expectation computed here.
        let row: Vec<f32> = vec![-1.5, 0.0, 0.25, 2.0, 3.5, -0.75];
        let want: f32 = row.iter().map(|v| 2.0 * v.exp()).sum();
        let got = row.iter().fold(fused.identity_f32(), |acc, v| {
            fused.absorb(&acc, &[*v]).unwrap()
        });
        assert!(
            (got[0] - want).abs() <= 1e-4 * want.abs(),
            "{got:?} vs {want}"
        );

        // The producers are still live, selectable members of their own
        // classes: this is an alternative, not a replacement.
        assert!(g.chain(m1).contains(&m1));
        assert!(g.chain(m2).contains(&m2));
    }

    /// **ABSORB fires under `NumericContract::STRICT`**, where every inexact
    /// law declines.
    ///
    /// A QAT fake-quant chain — `round(clamp(x/s, lo, hi)) * s` — reduced by a
    /// plain `Fold{Add}`. Substitution into a lift reassociates nothing, so a
    /// `reassoc` guard on this law would kill fusion exactly where it is most
    /// needed and where the byte-identical export lives. This case is the
    /// guard against someone adding one.
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
        // The numeric statement, spelled exactly: the absorbed lift is the
        // fake-quant body TERM FOR TERM. Substitution reassociates nothing, so
        // the fused fold computes the same float the unfused chain did — which
        // is what a byte-identical export needs and what a tolerance would not
        // prove. (`carrier::eval` has no `Round` arm, so a host probe could
        // not check this body at all.)
        assert_eq!(fused.lift[0], fake_quant);
        assert_eq!(fused.merge, ts::binop_carrier(BinOp::Add, Dtype::F32).merge);
        assert_eq!(
            fused.identity,
            ts::binop_carrier(BinOp::Add, Dtype::F32).identity
        );

        // The negative half: the inexact law on the same value declines.
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

    /// **The generality case: nearest-neighbour assignment.** No attention, no
    /// softmax, no fold splitting, and no rule that mentions any of them.
    ///
    /// `Fold{Min over M}( Fold{Add over D}( (a[n,d] - b[m,d])^2 ) )` — k-means
    /// assignment, k-NN, and the identical fact as never materializing an
    /// attention score matrix.
    ///
    /// Two halves, and the second is the interesting one:
    /// * the squared-difference map is absorbed into the inner reduction, so
    ///   the `[N, M, D]` difference tensor is not an operand of anything;
    /// * the inner reduction is **not** absorbed into the outer one. A
    ///   fold-to-fold edge is left alone, so `q.k` stays a real node with a
    ///   real `work()` row and the extractor prices `D` MACs of recompute
    ///   rather than the one op a `ScalarKind::Dot` would report.
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

        // Half one: the [N, M, D] difference tensor is gone from the inner
        // reduction's operand list.
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
        // Numerically the lift is (a - b)^2, evaluated here against itself.
        for (p, q) in [(1.5f32, -0.5f32), (0.0, 0.0), (2.25, 2.0)] {
            let got = inner.eval_lift(&[p, q]).unwrap()[0];
            assert!((got - (p - q) * (p - q)).abs() < 1e-6);
        }

        // Half two: the outer reduction still reads the inner one as an
        // OPERAND EDGE. Nothing inlined a reduction into a `ScalarExpr`.
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
            // Whatever else any other law did to this node, the reduction it
            // performs is still the minimum: a `Min` at every slot 0.
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

    /// Test 4. A `KMap` read by two `KFold`s: both folds gain a fused
    /// alternative and the map is still a live class member. This is the
    /// deleted duplication veto.
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
            // nothing, which is why this law carries no `reassoc` guard.
            assert_eq!(carrier.merge, before.merge);
            assert_eq!(carrier.identity, before.identity);
        }
        // The producer is still a live, selectable member of its own class.
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
    /// the fused spelling would read the whole buffer. Measured consequence
    /// before the guard: the standard sampler's `narrow` was elided and
    /// `gather_one_hot` returned the sum of every token id.
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
        // dense read the law states, and it still fires.
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
    /// block.
    ///
    /// The boundary is the test. With the block uncounted the rule accepted
    /// `limit - 1` reads, which is `limit + 1` storage buffers, and the plan
    /// reached `create_bind_group_layout` needing nine on a limit of eight —
    /// `normalization::softmax_last_dim` did exactly that once the extraction
    /// budget was large enough to select it. So both sides are asserted: the
    /// widest legal list fuses, one wider declines.
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

    /// The same boundary on the reducing consumer. `ABSORB` shipped without
    /// this check at all while `MAP_INTO_MAP` had it, and the asymmetry is not
    /// theoretical: `splice_through_address_map` brings the producer's *whole*
    /// operand list into a fold, so a nest can outgrow the bind group in one
    /// step. Measured with a wider absorption landed,
    /// `attention_backward_matches_the_analytic_adjoints` [gpu] failed
    /// `verify_plan` with "launch 15 binds 10 storage buffers — 9 operands plus
    /// the Uniforms block — over the 8-buffer limit".
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

    /// The same statement as an assert on the helper, so a future caller that
    /// reads `storage_bindings` as "operands plus output" is corrected here
    /// rather than in `create_bind_group_layout`.
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
