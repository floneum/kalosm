//! R5 — reader-rooted sinking. A pattern may match a *spine*
//! ([`crate::egraph::Builder::trace_pure_views`]), which is what makes the
//! reference's self-declared "single clearest structural gap"
//! (`sink_unary_chains_into_matmuls`, impossible there because "a generator
//! may only return a new variant for the node it was asked about") a
//! single-rooted rule here. No multi-root rule form is needed anywhere, and
//! the reference's private-view-chain sole-reader walk is deleted outright.
//!
//! Owned by W2.

use crate::dtype::Dtype;
use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::level0::L0;
use crate::ir::level1::{AccessPlan, L1, Operand};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::shape::{AxisGroup, Dim, Layout, MultiFlattenMap, SubAxis};
use smallvec::SmallVec;

rule!(
    SINK_EPILOGUE,
    level = Level::L1,
    head = OpTag::KMap,
    tag = RuleTag::Additive,
    apply = sink_epilogue,
);

rule!(
    FOLD_VIEWS_INTO_INDEX,
    level = Level::L1,
    head = OpTag::KMap,
    tag = RuleTag::Additive,
    apply = fold_views_into_index,
);

rule!(
    FOLD_VIEWS_INTO_FOLD_INDEX,
    level = Level::L1,
    head = OpTag::KFold,
    tag = RuleTag::Additive,
    apply = fold_views_into_fold_index,
);

/// `f(view(x)) == view(f(x))` when `view` is pure: a single-operand `KMap`
/// reading a contraction through a chain of restrides also equals that
/// contraction with a longer `post`, re-viewed.
///
/// The only guard is numeric: the epilogue must not round the accumulator
/// ahead of the chain, so its element type must be the accumulator's, or the
/// F16-accumulator/F32-epilogue widening pair. That is legality — whether
/// sinking pays is priced elsewhere.
pub fn sink_epilogue(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    let Op::L1(L1::KMap { body, ops, .. }) = &node.op else {
        return None;
    };
    if ops.len() != 1 || !matches!(ops[0].access, AccessPlan::Alias) {
        return None;
    }
    let spine = b.trace_pure_views(ops[0].src);
    let base = b.node(spine.base).op.clone();

    let sunk = match base {
        Op::L1(L1::KContract {
            m,
            n,
            k,
            batch,
            family,
            pre_a,
            pre_b,
            post,
            acc,
            a,
            b: rhs,
            sched,
        }) => {
            if !epilogue_preserves_accum(body.dtype(), acc) {
                return None;
            }
            b.add_l1(L1::KContract {
                m,
                n,
                k,
                batch,
                family,
                pre_a,
                pre_b,
                post: body.compose(&[post]),
                acc,
                a,
                b: rhs,
                sched,
            })
            .ok()?
        }
        Op::L1(L1::KQContract {
            fmt,
            layout,
            act,
            m,
            n,
            k,
            acc,
            post,
            a,
            b: rhs,
            sched,
        }) => {
            if !epilogue_preserves_accum(body.dtype(), acc) {
                return None;
            }
            b.add_l1(L1::KQContract {
                fmt,
                layout,
                act,
                m,
                n,
                k,
                acc,
                post: body.compose(&[post]),
                a,
                b: rhs,
                sched,
            })
            .ok()?
        }
        _ => return None,
    };

    // Re-apply the spine, innermost first. Each view keeps its own relative
    // spec vector, which is what makes a multi-node spine compose correctly.
    let mut cursor = sunk;
    for view in spine.views.iter() {
        let Op::L0(L0::Restride { specs, bounds, .. }) = b.node(*view).op.clone() else {
            return None;
        };
        cursor = b
            .add_l0(L0::Restride {
                specs,
                bounds,
                x: cursor,
            })
            .ok()?;
    }
    b.union(id, cursor).ok()
}

/// The accumulator must not be rounded ahead of the chain: an epilogue is
/// admissible when it preserves the accumulator's element type, or when it
/// is the F16-store / F32-compute widening pair.
fn epilogue_preserves_accum(epilogue: Dtype, acc: Dtype) -> bool {
    epilogue == acc
        || (acc == Dtype::F16 && epilogue == Dtype::F32)
        || (acc == Dtype::BF16 && epilogue == Dtype::F32)
}

/// Read a view through the operand's index map instead of through a
/// materialized copy.
///
/// The reference gates this on `needs_delinearize && input_reread_factor > 1`
/// inside the generator; that gate is deleted. `MultiFlattenMap::divmod_ops`
/// is the term the pricing crate charges for the divmod chain, so a reread
/// factor of two with a trivial delinearize can win on its own merits.
pub fn fold_views_into_index(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    let Op::L1(L1::KMap {
        space,
        body,
        ops,
        sched,
    }) = &node.op
    else {
        return None;
    };
    let new_ops = fold_operand_views(b, ops, space)?;
    let alt = b
        .add_l1(L1::KMap {
            space: space.clone(),
            body: body.clone(),
            ops: new_ops,
            sched: sched.clone(),
        })
        .ok()?;
    b.union(id, alt).ok()
}

/// The same law with a `KFold` in the consumer position.
///
/// [`fold_views_into_index`] states "a view is an index map, not a copy", and
/// nothing in that statement mentions the consumer's head. It shipped headed
/// at `KMap` only, and the consequence was measurable rather than theoretical:
/// on the attention forward chain the row max is broadcast back over the key
/// axis and read by **two** consumers, the `exp` map and the sum fold. The map
/// could fold the broadcast into its operand map; the fold could not, so the
/// broadcast kept a live reader, stayed in `M`, and cost a whole dispatch that
/// copied 48 floats into 192 bytes so a later kernel could read them back
/// contiguously.
///
/// A `KFold`'s operands are indexed over `space` exactly as a `KMap`'s are
/// (`verify_l1::check_operand_access` reads `index_space_of`, which returns
/// `space` for both), so the rewrite is the same rewrite. `vec_axes` needs no
/// special case: it renumbers nothing, and a promoted axis a rewritten operand
/// varies along is checked by `check_vec_axes` on the minted node, so an
/// illegal spelling is refused by `add_l1` rather than guarded here.
pub fn fold_views_into_fold_index(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    _f: &Facts<'_>,
) -> Option<Id> {
    let Op::L1(L1::KFold {
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
    let new_ops = fold_operand_views(b, ops, space)?;
    let alt = b
        .add_l1(L1::KFold {
            space: space.clone(),
            axis: *axis,
            vec_axes: vec_axes.clone(),
            carrier: carrier.clone(),
            acc: *acc,
            post: post.clone(),
            ops: new_ops,
            sched: sched.clone(),
        })
        .ok()?;
    b.union(id, alt).ok()
}

/// Every operand whose source is a single pure view, restated as an index map
/// over the view's base. `None` when no slot moved, so the caller mints
/// nothing.
fn fold_operand_views(
    b: &Builder<'_>,
    ops: &[Operand],
    space: &crate::ir::level1::IndexSpace,
) -> Option<Vec<Operand>> {
    let mut new_ops = ops.to_vec();
    let mut changed = false;
    for slot in new_ops.iter_mut() {
        if !matches!(slot.access, AccessPlan::Alias) {
            continue;
        }
        // The rewrite **replaces the operand's layout outright** with one
        // derived from the view's base, so whatever that layout said is
        // discarded. That is sound only when it said exactly "read the view
        // dense at the consuming coordinate". A permuted, strided, broadcast
        // or offset alias says something else — and since `check_operand_access`
        // only pins an `Alias`'s rank and extents, not its strides, every one
        // of those spellings reaches here.
        if !reads_its_view_densely(slot, space) {
            continue;
        }
        let spine = b.trace_pure_views(slot.src);
        if spine.views.len() != 1 {
            // A multi-node spine needs spec composition, which is only
            // decidable once every extent is known; the single-view case is
            // where the conv operand and the flat-parameter slice live.
            continue;
        }
        let Op::L0(L0::Restride { specs, .. }) = b.node(spine.views[0]).op.clone() else {
            continue;
        };
        // The map replaces the operand's layout outright, and a
        // `MultiFlattenMap`'s extents are what `Operand::address_map` derives
        // its divisors from. So the view must already span the consuming
        // index space. When it does not — an operand broadcast against the
        // space, whose `[rows, 1]` view is read over `[rows, cols]` — the
        // layout is doing work the map cannot express, and adopting the map
        // reads `flat % rows` where `flat / cols` belongs.
        if specs.len() != space.dims.len()
            || !specs
                .iter()
                .zip(&space.dims)
                .all(|(s, d)| s.size.known_eq(*d))
        {
            continue;
        }
        let base_shape = b.facts_of(spine.base).shape.clone();
        let Some((map, offset)) = unflatten_of(&specs, &base_shape) else {
            continue;
        };
        // `MultiFlattenMap` is a pure sum of stride terms and has nowhere to
        // put a base offset, so `Operand::address_map` takes it from the
        // layout. A contiguous layout says offset 0, which silently turned a
        // narrowed view (`table[2..]`) back into the whole table.
        let layout = Layout::from_parts(
            Dim::Const(offset),
            &base_shape,
            &Layout::row_major_strides(&base_shape),
        )
        .ok()?;
        *slot = Operand {
            src: spine.base,
            layout,
            access: AccessPlan::Unflatten(map),
        };
        changed = true;
    }
    changed.then_some(new_ops)
}

/// Whether `o`'s own layout is the dense row-major read of `space` at offset
/// zero — the one layout [`fold_operand_views`] may discard, because it is the
/// one the replacement map reproduces.
///
/// # Why this is a guard and not an assert
///
/// `verify_l1::check_operand_access` pins an `Alias`'s **rank and extents**
/// against the index space and says nothing about its strides or its offset.
/// So a transposed read (`strides = [1, rows]`), a broadcast one
/// (`strides = [.., 0]`) and a window (`offset != 0`) are all legal operands
/// at the extents this rule already checks, and all three arrive here.
/// Replacing them with `unflatten_of`'s map states the *view's* index
/// arithmetic in place of the operand's, which is a different address for
/// every coordinate but the first.
///
/// Measured, and this is the whole reason the guard exists: with a co-selection
/// pass in `fusor2-cost::extract` letting the search reach the fused members
/// this rule mints, the unguarded version put **29** conformance cases on wrong
/// values — every `softmax`, `layer_norm` and `rms_norm` row, `attention_qk_mask`
/// and the attention gradients, on both backends. `softmax_rows_sum_to_one`
/// reported a row summing to 1.13. The e-graph's invariant is that any member
/// of a class computes the same value, so a search that reaches further is a
/// search that reaches a wrong plan; the member has been in the graph, unequal
/// and unselected, since this rule shipped.
fn reads_its_view_densely(o: &Operand, space: &crate::ir::level1::IndexSpace) -> bool {
    if !o.layout.offset().known_eq(Dim::Const(0)) {
        return false;
    }
    if o.layout.rank() != space.dims.len() {
        return false;
    }
    if !o
        .layout
        .shape()
        .iter()
        .zip(&space.dims)
        .all(|(l, d)| l.known_eq(*d))
    {
        return false;
    }
    let want = Layout::row_major_strides(&space.dims);
    o.layout
        .strides()
        .iter()
        .zip(&want)
        .all(|(s, w)| s.known_eq(*w))
}

/// The index map a relative spec vector induces over a dense base, and the
/// base offset it starts from. Declines when an extent, stride or offset is
/// not decidable — there is no contiguous fallback here, only the alternative
/// not being minted.
///
/// The offset is returned separately because `MultiFlattenMap` is a sum of
/// stride terms with no constant slot; the caller must put it on the
/// operand's layout, which is where [`Operand::address_map`] reads it from.
fn unflatten_of(
    specs: &[crate::shape::StrideSpec],
    base_shape: &[Dim],
) -> Option<(MultiFlattenMap, u64)> {
    let base_strides = Layout::row_major_strides(base_shape);
    let mut groups: SmallVec<[AxisGroup; 4]> = SmallVec::new();
    let mut offset: u64 = 0;
    for s in specs {
        let extent = u32::try_from(s.size.as_const()?).ok()?;
        let base = base_strides.get(s.input_dim as usize)?.as_const()?;
        // A spec's offset is in units of its own input axis, so it scales by
        // that axis's stride whether or not the axis is broadcast.
        offset = offset.checked_add(s.offset.as_const()?.checked_mul(base)?)?;
        let stride = if s.multiplier == 0 {
            0
        } else {
            u32::try_from(base.checked_mul(u64::from(s.multiplier))?).ok()?
        };
        groups.push(AxisGroup {
            sub_axes: smallvec::smallvec![SubAxis { extent, stride }],
        });
    }
    Some((MultiFlattenMap { groups }, offset))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::level1::L1;
    use crate::rules::test_support as ts;
    use crate::rules::{alias_operand_of, ident_expr};
    use crate::scalar::{ScalarExpr, ScalarKind, UnOp};

    fn fire(g: &mut crate::egraph::EGraph, id: Id, r: &crate::egraph::Rule) -> Option<Id> {
        let caps = ts::caps();
        let node = g.node(id).clone();
        let facts = g.facts_view(id, &caps);
        let mut b = g.builder(&caps);
        (r.apply)(&mut b, id, &node, &facts)
    }

    /// The `KFold` arm reads a broadcast row statistic through the operand's
    /// index map instead of through the copy the floor would materialize.
    ///
    /// This is the attention forward chain's shape in miniature: a `[rows]`
    /// value broadcast back over the reduced axis and consumed by a fold. The
    /// `KMap` arm always had this; without the fold arm the broadcast keeps a
    /// reader, stays in `M` and costs a dispatch.
    #[test]
    fn fold_views_into_fold_index_reads_a_broadcast_through_the_index_map() {
        use crate::carrier::Carrier;
        use crate::dtype::Splat;
        use crate::scalar::BinOp;
        use crate::shape::StrideSpec;

        let mut g = ts::graph();
        let rows = Dim::Const(3);
        let cols = Dim::Const(4);
        let stat = ts::buffer(&mut g, Dtype::F32, &[rows]);
        // `broadcast(stat)` over `[rows, cols]`: one real axis, one stride-0.
        let bcast = ts::restride(
            &mut g,
            &[StrideSpec::dim(0, rows), StrideSpec::broadcast(cols)],
            stat,
        );
        let space = [rows, cols];
        let fold = ts::kfold(
            &mut g,
            &space,
            1,
            Carrier::binop(BinOp::Add, Splat::F32(0.0), Dtype::F32),
            Dtype::F32,
            ScalarExpr::arg(0, Dtype::F32),
            vec![alias_operand_of(bcast, &space)],
        );
        assert!(fire(&mut g, fold, &FOLD_VIEWS_INTO_FOLD_INDEX).is_some());

        let folded = g
            .chain(fold)
            .iter()
            .copied()
            .find(|&i| {
                matches!(&g.node(i).op, Op::L1(L1::KFold { ops, .. })
                    if ops.len() == 1
                        && ops[0].src == stat
                        && matches!(ops[0].access, AccessPlan::Unflatten(_)))
            })
            .expect("an alternative reading the statistic through an index map");
        let Op::L1(L1::KFold { ops, .. }) = &g.node(folded).op else {
            panic!()
        };
        let AccessPlan::Unflatten(map) = &ops[0].access else {
            panic!()
        };
        // Two axes, and the reduced one is stride 0 — the broadcast is now
        // arithmetic rather than a buffer.
        assert_eq!(map.groups.len(), 2);
        assert_eq!(map.groups[0].sub_axes[0].stride, 1);
        assert_eq!(map.groups[1].sub_axes[0].stride, 0);
        assert_eq!(map.groups[1].sub_axes[0].extent, 4);
    }

    /// The rewrite discards the operand's own layout, so it fires only where
    /// that layout was the dense read. Both halves, on one fixture: the
    /// transposed read of a view is refused, and the dense read of the same
    /// view is still rewritten.
    ///
    /// Without the guard the first assert fails — `check_operand_access` pins
    /// an `Alias`'s rank and extents and says nothing about its strides, so a
    /// transposed edge reaches the rule and comes back reading the view in
    /// row-major order.
    #[test]
    fn fold_views_into_index_refuses_an_operand_that_is_not_the_dense_read() {
        use crate::shape::StrideSpec;

        let n = Dim::Const(4);
        let square = [n, n];
        let build = |strides: &[Dim]| {
            let mut g = ts::graph();
            let base = ts::buffer(&mut g, Dtype::F32, &square);
            // A pure view of the base spanning the consuming space, which is
            // the shape this rule exists to fold into an index map.
            let view = ts::restride(
                &mut g,
                &[StrideSpec::dim(0, n), StrideSpec::dim(1, n)],
                base,
            );
            let op = Operand {
                src: view,
                layout: Layout::from_parts(Dim::Const(0), &square, strides).unwrap(),
                access: AccessPlan::Alias,
            };
            let m = ts::kmap(
                &mut g,
                &square,
                ScalarExpr::un(UnOp::Neg, ScalarExpr::arg(0, Dtype::F32)),
                vec![op],
            );
            (g, m)
        };

        // Transposed: strides [1, 4] where the dense read is [4, 1].
        let (mut g, m) = build(&[Dim::Const(1), n]);
        assert!(
            fire(&mut g, m, &FOLD_VIEWS_INTO_INDEX).is_none(),
            "a transposed alias is not the dense read, and the map that would \
             replace it reads row-major"
        );

        // Dense: the one layout the replacement map reproduces.
        let (mut g, m) = build(&[n, Dim::Const(1)]);
        assert!(fire(&mut g, m, &FOLD_VIEWS_INTO_INDEX).is_some());
    }

    /// A fold whose operand is not a view gains no alternative, so the rule
    /// is silent on the graphs it has nothing to say about.
    #[test]
    fn fold_views_into_fold_index_declines_on_a_plain_operand() {
        use crate::carrier::Carrier;
        use crate::dtype::Splat;
        use crate::scalar::BinOp;

        let mut g = ts::graph();
        let space = [Dim::Const(3), Dim::Const(4)];
        let x = ts::buffer(&mut g, Dtype::F32, &space);
        let fold = ts::kfold(
            &mut g,
            &space,
            1,
            Carrier::binop(BinOp::Add, Splat::F32(0.0), Dtype::F32),
            Dtype::F32,
            ScalarExpr::arg(0, Dtype::F32),
            vec![alias_operand_of(x, &space)],
        );
        assert!(fire(&mut g, fold, &FOLD_VIEWS_INTO_FOLD_INDEX).is_none());
    }

    /// Test 5. `KContract -> Restride -> Restride -> KMap(gelu-ish)`: the
    /// map's class gains `Restride(Restride(KContract{post}))` and the
    /// composed form survives.
    #[test]
    fn sink_epilogue_crosses_a_three_node_view_spine() {
        let mut g = ts::graph();
        let a = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(6), Dim::Const(8)]);
        let bb = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(8), Dim::Const(4)]);
        let mm = ts::kcontract(
            &mut g,
            Dim::Const(6),
            Dim::Const(4),
            Dim::Const(8),
            ident_expr(Dtype::F32),
            alias_operand_of(a, &[Dim::Const(6), Dim::Const(8)]),
            alias_operand_of(bb, &[Dim::Const(8), Dim::Const(4)]),
        );
        let v1 = ts::restride(
            &mut g,
            &[
                crate::shape::StrideSpec::dim(1, Dim::Const(4)),
                crate::shape::StrideSpec::dim(0, Dim::Const(6)),
            ],
            mm,
        );
        let v2 = ts::restride(
            &mut g,
            &[
                crate::shape::StrideSpec::dim(0, Dim::Const(4)),
                crate::shape::StrideSpec::dim(1, Dim::Const(6)),
            ],
            v1,
        );
        let out = [Dim::Const(4), Dim::Const(6)];
        let epi = ts::kmap(
            &mut g,
            &out,
            ScalarExpr::un(UnOp::Tanh, ScalarExpr::arg(0, Dtype::F32)),
            vec![alias_operand_of(v2, &out)],
        );
        assert!(fire(&mut g, epi, &SINK_EPILOGUE).is_some());

        let members = g.chain(epi);
        assert!(members.contains(&epi), "the composed form must survive");
        let alt = members
            .iter()
            .copied()
            .find(|&i| matches!(g.node(i).op, Op::L0(L0::Restride { .. })))
            .expect("a re-viewed alternative");
        // Peel the two views back off and find the contraction underneath.
        let Op::L0(L0::Restride { x: inner, .. }) = g.node(alt).op.clone() else {
            panic!()
        };
        let Op::L0(L0::Restride { x: base, .. }) = g.node(inner).op.clone() else {
            panic!("expected two stacked views")
        };
        let Op::L1(L1::KContract { post, .. }) = &g.node(base).op else {
            panic!("expected the contraction to carry the epilogue")
        };
        assert!(matches!(post.kind(), ScalarKind::Un { op: UnOp::Tanh, .. }));
    }

    #[test]
    fn sink_epilogue_refuses_to_round_the_accumulator() {
        let mut g = ts::graph();
        let a = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(2), Dim::Const(2)]);
        let bb = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(2), Dim::Const(2)]);
        let mm = ts::kcontract(
            &mut g,
            Dim::Const(2),
            Dim::Const(2),
            Dim::Const(2),
            ident_expr(Dtype::F32),
            alias_operand_of(a, &[Dim::Const(2), Dim::Const(2)]),
            alias_operand_of(bb, &[Dim::Const(2), Dim::Const(2)]),
        );
        let out = [Dim::Const(2), Dim::Const(2)];
        // An F16 epilogue over an F32 accumulator narrows it.
        let epi = ts::kmap(
            &mut g,
            &out,
            ScalarExpr::cast(Dtype::F16, ScalarExpr::arg(0, Dtype::F32)),
            vec![alias_operand_of(mm, &out)],
        );
        assert!(fire(&mut g, epi, &SINK_EPILOGUE).is_none());
        assert_eq!(g.chain(epi).len(), 1);
    }

    #[test]
    fn fold_views_into_index_replaces_a_materialized_view() {
        let mut g = ts::graph();
        let x = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(4), Dim::Const(6)]);
        let v = ts::restride(
            &mut g,
            &[
                crate::shape::StrideSpec::dim(1, Dim::Const(6)),
                crate::shape::StrideSpec::dim(0, Dim::Const(4)),
            ],
            x,
        );
        let out = [Dim::Const(6), Dim::Const(4)];
        let m = ts::kmap(
            &mut g,
            &out,
            ScalarExpr::un(UnOp::Abs, ScalarExpr::arg(0, Dtype::F32)),
            vec![alias_operand_of(v, &out)],
        );
        assert!(fire(&mut g, m, &FOLD_VIEWS_INTO_INDEX).is_some());
        let alt = g.chain(m).into_iter().find(|&i| i != m).unwrap();
        let Op::L1(L1::KMap { ops, .. }) = &g.node(alt).op else {
            panic!()
        };
        assert_eq!(ops[0].src, x);
        let AccessPlan::Unflatten(map) = &ops[0].access else {
            panic!("expected an index-mapped read")
        };
        assert_eq!(map.rank(), 2);
        assert_eq!(map.groups[0].sub_axes[0].stride, 1);
        assert_eq!(map.groups[1].sub_axes[0].stride, 6);
    }
}
