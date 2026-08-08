//! Reader-rooted sinking. Patterns match a view spine via
//! [`crate::egraph::Builder::trace_pure_views`], so sinking a unary chain into
//! a matmul across a chain of views is a single-rooted rule.

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
/// F16-accumulator/F32-epilogue widening pair.
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
                post: body.compose(&[post]),
                acc,
                a,
                b: rhs,
                sched,
            })
            .ok()?
        }
        _ => return None,
    };

    // Re-apply the spine, innermost first; each view keeps its own relative
    // spec vector.
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
/// The alternative is minted unconditionally and priced; the divmod chain is
/// charged through `MultiFlattenMap::divmod_ops`.
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
/// A `KFold`'s operands are indexed over `space` exactly as a `KMap`'s are, so
/// the rewrite is identical. `vec_axes` needs no special case: `check_vec_axes`
/// on the minted node refuses an illegal spelling through `add_l1`.
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
        // The rewrite replaces the operand's layout with one derived from the
        // view's base, which is sound only for a dense read at the consuming
        // coordinate.
        if !reads_its_view_densely(slot, space) {
            continue;
        }
        let spine = b.trace_pure_views(slot.src);
        if spine.views.len() != 1 {
            // A multi-node spine needs spec composition, decidable only once
            // every extent is known.
            continue;
        }
        let Op::L0(L0::Restride { specs, .. }) = b.node(spine.views[0]).op.clone() else {
            continue;
        };
        // `Operand::address_map` derives its divisors from the map's extents,
        // so the view must span the consuming index space.
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
        // `MultiFlattenMap` is a sum of stride terms with no slot for a base
        // offset, so the layout carries the view's offset and
        // `Operand::address_map` reads it from there.
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
/// `verify_l1::check_operand_access` pins an `Alias`'s rank and extents only,
/// so transposed, broadcast and offset aliases all reach this rule and must be
/// refused here.
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
/// not decidable.
///
/// The offset is returned separately because `MultiFlattenMap` has no constant
/// slot; the caller puts it on the operand's layout, where
/// [`Operand::address_map`] reads it from.
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
    /// index map instead of through a materialized copy: a `[rows]` value
    /// broadcast over the reduced axis and consumed by a fold.
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
    /// that layout was the dense read: the transposed read of a view is
    /// refused, the dense read of the same view is rewritten.
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

    /// `KContract -> Restride -> Restride -> KMap(gelu-ish)`: the
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
