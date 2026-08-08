//! Operand access alternatives. Access is an attribute of the *edge*, so one
//! reader may alias a strided parameter slice while another packs it — the
//! flat-parameter / gradient-concat case and the im2col operand case
//! coexisting in one graph. `LayoutPass`'s `unwrap_or_else(Layout::contiguous)`
//! has no analogue here: there is no default, only alternatives that compete.
//!
//! Each rule mints **one** alternative of the *reading* node with **one**
//! operand's access changed. `Rule::head` is a single tag, so the four rules
//! are spread across the two most common readers: three on `KMap`, where the
//! trainer's flat-parameter slices and conv window operands live, and the
//! pack rule on `KContract`, where a B-operand repack pays for itself.
//!
//! Owned by W2.

use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::level1::{AccessPlan, ContractSide, L1, Operand};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::shape::{AxisGroup, Layout, MultiFlattenMap, SubAxis};
use smallvec::SmallVec;

rule!(
    OPERAND_ALIAS,
    level = Level::L1,
    head = OpTag::KMap,
    tag = RuleTag::Additive,
    apply = operand_alias,
);

rule!(
    OPERAND_GATHER,
    level = Level::L1,
    head = OpTag::KMap,
    tag = RuleTag::Additive,
    apply = operand_gather,
);

rule!(
    OPERAND_PACK,
    level = Level::L1,
    head = OpTag::KContract,
    tag = RuleTag::Additive,
    apply = operand_pack,
);

rule!(
    OPERAND_UNFLATTEN,
    level = Level::L1,
    head = OpTag::KMap,
    tag = RuleTag::Additive,
    apply = operand_unflatten,
);

/// Rebuild a `KMap` with the first operand that `pick` rewrites replaced.
fn remap_kmap(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    pick: impl Fn(&Operand) -> Option<Operand>,
) -> Option<Id> {
    let Op::L1(L1::KMap {
        space,
        body,
        ops,
        sched,
    }) = &node.op
    else {
        return None;
    };
    let slot = ops.iter().position(|o| pick(o).is_some())?;
    let mut new_ops = ops.clone();
    new_ops[slot] = pick(&ops[slot])?;
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

/// Read this operand straight through its own strides.
///
/// # The layout is not on its own the whole access
///
/// An `Alias` addresses through `layout`'s strides and nothing else, so
/// re-spelling an edge as one is sound only when the plan it replaces
/// addresses the same way. For a [`AccessPlan::Gather`] and a
/// [`AccessPlan::Pack`] it always does — both derive their addresses from the
/// layout, which is why [`operand_gather`] declines on a dense layout as a
/// *canonicalization* — and for an [`AccessPlan::Unflatten`] minted by
/// [`operand_unflatten`] it does too, since that map is `decompose(layout)`.
///
/// It does **not** for an `Unflatten` whose map was stated independently of
/// the layout. `rules::sink::fold_operand_views` mints exactly that: the map
/// carries the *view's* index arithmetic while the layout carries only the
/// base's shape and the view's start offset, because a `MultiFlattenMap` has
/// no constant slot. Dropping that map re-reads the base densely — the
/// broadcast, transpose or window the view expressed is simply gone.
///
/// Measured, with a co-selection pass in `fusor2-cost::extract` letting the
/// search reach these members: **29** conformance cases on wrong values, every
/// `softmax`, `layer_norm` and `rms_norm` row plus `attention_qk_mask` and the
/// attention gradients, on both backends. `softmax_rows_sum_to_one` reported a
/// row summing to 1.13 because each element was divided by `l[i + j]` instead
/// of `l[i]`. Disabling either this rule or `sink::FOLD_VIEWS_INTO_INDEX`
/// alone made all 29 pass, which is the pair that has to agree.
///
/// The e-graph's invariant is that every member of a class computes the same
/// value. This member has been unequal since the two rules first coexisted;
/// only the extraction budget kept it unselected.
pub fn operand_alias(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    remap_kmap(b, id, node, |o| {
        let aliased = Operand {
            src: o.src,
            layout: o.layout.clone(),
            access: AccessPlan::Alias,
        };
        match &o.access {
            AccessPlan::Alias => None,
            // Both of these derive every address from `layout` — a `Gather` by
            // computing it per element, a `Pack` by staging the same elements
            // into a tile first — so the alias spelling addresses identically
            // whatever the extents are. Deciding this by access plan rather
            // than by `AddressMap` equality is what keeps a `Dim::Sym` edge
            // rewritable: the map is undecidable there and the equality is
            // not, but the *reason* the two agree does not read an extent.
            AccessPlan::Gather | AccessPlan::Pack { .. } => Some(aliased),
            // An `Unflatten` map is the one plan that may have been stated
            // independently of the layout, so here the equality is the
            // condition and an undecidable extent declines.
            AccessPlan::Unflatten(_) => {
                (aliased.address_map()? == o.address_map()?).then_some(aliased)
            }
        }
    })
}

/// Read this operand through a per-element address computation.
///
/// Only minted for a layout that is not already dense row-major: over a
/// contiguous layout a gather and an alias name the *same* index map, so
/// minting both would put one access in the graph twice under two spellings.
/// That is canonicalization, not a judgement about which is faster.
pub fn operand_gather(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    remap_kmap(b, id, node, |o| {
        (!matches!(o.access, AccessPlan::Gather) && !o.layout.is_contiguous()).then(|| Operand {
            src: o.src,
            layout: o.layout.clone(),
            access: AccessPlan::Gather,
        })
    })
}

/// Stage this operand into a dense tile first. Legal when the packed layout
/// is contiguous and holds exactly as many elements as the operand does.
pub fn operand_pack(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
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
    let repack = |o: &Operand| -> Option<Operand> {
        // Packing a layout that is already dense row-major stages it into a
        // byte-identical tile — the same access under two spellings.
        if matches!(o.access, AccessPlan::Pack { .. }) || o.layout.is_contiguous() {
            return None;
        }
        let into = Layout::contiguous(o.layout.shape());
        if !into.is_contiguous() || elements(&into)? != elements(&o.layout)? {
            return None;
        }
        Some(Operand {
            src: o.src,
            layout: o.layout.clone(),
            access: AccessPlan::Pack { into },
        })
    };
    // Each operand of a side is loaded through its own access plan, so packing
    // one and aliasing its neighbour is sound. Exactly one alternative is
    // minted per fire — the first packable operand in `children_of` order —
    // and the rule being additive is what lets extraction see the rest.
    let pack_first = |side: &ContractSide| -> Option<ContractSide> {
        let (i, packed) = side.ops.iter().enumerate().find_map(|(i, o)| Some((i, repack(o)?)))?;
        let mut out = side.clone();
        out.ops[i] = packed;
        Some(out)
    };
    let (new_a, new_b) = match pack_first(a) {
        Some(pa) => (pa, rhs.clone()),
        None => (a.clone(), pack_first(rhs)?),
    };
    let alt = b
        .add_l1(L1::KContract {
            m: *m,
            n: *n,
            k: *k,
            batch: *batch,
            family: *family,
            post: post.clone(),
            acc: *acc,
            a: new_a,
            b: new_b,
            sched: sched.clone(),
        })
        .ok()?;
    b.union(id, alt).ok()
}

/// Read this operand through an explicit index map. Legal only when the
/// operand's layout decomposes into decidable `AxisGroup`s; when it does not,
/// the alternative is simply not minted — there is no contiguous fallback.
///
/// A dense row-major layout decomposes into exactly the map an alias already
/// implies, so it is skipped for the same canonicalization reason as
/// [`operand_gather`].
pub fn operand_unflatten(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    remap_kmap(b, id, node, |o| {
        if matches!(o.access, AccessPlan::Unflatten(_)) || o.layout.is_contiguous() {
            return None;
        }
        let map = decompose(&o.layout)?;
        Some(Operand {
            src: o.src,
            layout: o.layout.clone(),
            access: AccessPlan::Unflatten(map),
        })
    })
}

fn elements(l: &Layout) -> Option<u64> {
    l.shape()
        .iter()
        .try_fold(1u64, |acc, d| acc.checked_mul(d.as_const()?))
}

/// One `AxisGroup` per logical axis of a decidable strided layout.
fn decompose(l: &Layout) -> Option<MultiFlattenMap> {
    let mut groups: SmallVec<[AxisGroup; 4]> = SmallVec::new();
    for (d, s) in l.shape().iter().zip(l.strides()) {
        let extent = u32::try_from(d.as_const()?).ok()?;
        let stride = u32::try_from(s.as_const()?).ok()?;
        groups.push(AxisGroup {
            sub_axes: smallvec::smallvec![SubAxis { extent, stride }],
        });
    }
    (!groups.is_empty()).then_some(MultiFlattenMap { groups })
}

/// Build a strided edge by hand, for fixtures.
#[cfg(test)]
pub(crate) fn strided_operand(
    src: Id,
    shape: &[crate::shape::Dim],
    strides: &[crate::shape::Dim],
) -> Option<Operand> {
    Some(Operand {
        src,
        layout: Layout::from_parts(crate::shape::Dim::Const(0), shape, strides).ok()?,
        access: AccessPlan::Gather,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::Dtype;
    use crate::rules::test_support as ts;
    use crate::rules::{alias_operand_of, ident_expr};
    use crate::scalar::{ScalarExpr, UnOp};
    use crate::shape::Dim;

    fn fire(g: &mut crate::egraph::EGraph, id: Id, r: &crate::egraph::Rule) -> Option<Id> {
        let caps = ts::caps();
        let node = g.node(id).clone();
        let facts = g.facts_view(id, &caps);
        let mut b = g.builder(&caps);
        (r.apply)(&mut b, id, &node, &facts)
    }

    /// Test 14. One strided producer read by two nodes: one reader's class
    /// holds an `Alias` operand, the other's a `Pack`. Access is an edge
    /// attribute, so the two never contend.
    #[test]
    fn layout_alternatives_coexist_per_edge() {
        let mut g = ts::graph();
        let shape = [Dim::Const(8), Dim::Const(4)];
        let src = ts::buffer(&mut g, Dtype::F32, &shape);
        // Reader A: an elementwise map reading it with a per-element address
        // computation, which `operand_alias` offers to straighten out.
        let gathered = strided_operand(src, &shape, &[Dim::Const(1), Dim::Const(8)]).unwrap();
        let reader_a = ts::kmap(
            &mut g,
            &shape,
            ScalarExpr::un(UnOp::Abs, ScalarExpr::arg(0, Dtype::F32)),
            vec![gathered],
        );
        assert!(fire(&mut g, reader_a, &OPERAND_ALIAS).is_some());
        let alias_alt = g
            .chain(reader_a)
            .into_iter()
            .find(|&i| i != reader_a)
            .unwrap();
        let Op::L1(L1::KMap { ops, .. }) = &g.node(alias_alt).op else {
            panic!()
        };
        assert!(matches!(ops[0].access, AccessPlan::Alias));

        // Reader B: a contraction over the same producer, which packs it.
        let rhs = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(4), Dim::Const(2)]);
        let strided_a = strided_operand(src, &shape, &[Dim::Const(1), Dim::Const(8)]).unwrap();
        let reader_b = ts::kcontract(
            &mut g,
            Dim::Const(8),
            Dim::Const(2),
            Dim::Const(4),
            ident_expr(Dtype::F32),
            strided_a,
            alias_operand_of(rhs, &[Dim::Const(4), Dim::Const(2)]),
        );
        assert!(fire(&mut g, reader_b, &OPERAND_PACK).is_some());
        let pack_alt = g
            .chain(reader_b)
            .into_iter()
            .find(|&i| i != reader_b)
            .unwrap();
        let Op::L1(L1::KContract { a, .. }) = &g.node(pack_alt).op else {
            panic!()
        };
        assert!(matches!(a.primary().access, AccessPlan::Pack { .. }));

        // And the producer is one node, read two different ways.
        assert_eq!(g.node(alias_alt).children[0], src);
        assert_eq!(g.node(pack_alt).children[0], src);
    }

    /// `OPERAND_ALIAS` refuses an `Unflatten` whose map is not its layout's
    /// own decomposition, and still rewrites one that is.
    ///
    /// The refused half is the operand `rules::sink::fold_operand_views` mints
    /// for a broadcast: the layout carries the base's `[rows]` shape and the
    /// start offset, while the map carries `[rows, cols]` with the second axis
    /// at stride 0. Aliasing that drops the broadcast and reads the base
    /// densely — 29 conformance cases' worth of wrong numbers, once extraction
    /// searches far enough to select it. See this rule's own doc.
    ///
    /// Without the guard the first assert fails: the rule fired on anything
    /// that was not already an `Alias`.
    #[test]
    fn operand_alias_refuses_a_map_its_layout_does_not_imply() {
        let rows = Dim::Const(3);
        let cols = Dim::Const(4);
        let space = [rows, cols];

        let bcast_map = MultiFlattenMap {
            groups: smallvec::smallvec![
                AxisGroup {
                    sub_axes: smallvec::smallvec![SubAxis {
                        extent: 3,
                        stride: 1
                    }]
                },
                AxisGroup {
                    sub_axes: smallvec::smallvec![SubAxis {
                        extent: 4,
                        stride: 0
                    }]
                },
            ],
        };

        let mut g = ts::graph();
        let stat = ts::buffer(&mut g, Dtype::F32, &[rows]);
        let broadcast_read = ts::kmap(
            &mut g,
            &space,
            ScalarExpr::un(UnOp::Neg, ScalarExpr::arg(0, Dtype::F32)),
            vec![Operand {
                src: stat,
                layout: Layout::contiguous(&[rows]),
                access: AccessPlan::Unflatten(bcast_map),
            }],
        );
        assert!(
            fire(&mut g, broadcast_read, &OPERAND_ALIAS).is_none(),
            "the map states a stride-0 axis the layout does not, so an alias \
             over that layout is a different value"
        );

        // The map `operand_unflatten` itself mints *is* the layout's
        // decomposition, so the alias spelling is the same access and is still
        // offered — this rule must keep doing its job.
        let transposed =
            Layout::from_parts(Dim::Const(0), &space, &[Dim::Const(1), rows]).unwrap();
        let x = ts::buffer(&mut g, Dtype::F32, &space);
        let strided_read = ts::kmap(
            &mut g,
            &space,
            ScalarExpr::un(UnOp::Neg, ScalarExpr::arg(0, Dtype::F32)),
            vec![Operand {
                src: x,
                layout: transposed.clone(),
                access: AccessPlan::Unflatten(decompose(&transposed).unwrap()),
            }],
        );
        assert!(fire(&mut g, strided_read, &OPERAND_ALIAS).is_some());

        // And a `Gather` is always re-spellable: it derives its addresses from
        // the very layout the alias would use.
        let gathered = ts::kmap(
            &mut g,
            &space,
            ScalarExpr::un(UnOp::Neg, ScalarExpr::arg(0, Dtype::F32)),
            vec![strided_operand(x, &space, &[Dim::Const(1), rows]).unwrap()],
        );
        assert!(fire(&mut g, gathered, &OPERAND_ALIAS).is_some());
    }

    /// A transposed read is a genuinely different access, so both the
    /// address-computed and the index-mapped spellings are minted. A dense
    /// row-major read is not, so neither is.
    #[test]
    fn operand_gather_and_unflatten_mint_their_own_edges() {
        let mut g = ts::graph();
        let shape = [Dim::Const(6), Dim::Const(3)];
        let src = ts::buffer(&mut g, Dtype::F32, &shape);
        let transposed =
            strided_operand(src, &shape, &[Dim::Const(1), Dim::Const(6)]).unwrap();
        let m = ts::kmap(
            &mut g,
            &shape,
            ScalarExpr::un(UnOp::Neg, ScalarExpr::arg(0, Dtype::F32)),
            vec![Operand {
                access: AccessPlan::Alias,
                ..transposed
            }],
        );
        assert!(fire(&mut g, m, &OPERAND_GATHER).is_some());
        assert!(fire(&mut g, m, &OPERAND_UNFLATTEN).is_some());
        let members = g.chain(m);
        assert_eq!(members.len(), 3);
        let mut seen_gather = false;
        let mut seen_unflatten = false;
        for id in members {
            if let Op::L1(L1::KMap { ops, .. }) = &g.node(id).op {
                match &ops[0].access {
                    AccessPlan::Gather => seen_gather = true,
                    AccessPlan::Unflatten(map) => {
                        seen_unflatten = true;
                        assert_eq!(map.groups[0].sub_axes[0].stride, 1);
                        assert_eq!(map.groups[1].sub_axes[0].stride, 6);
                    }
                    _ => {}
                }
            }
        }
        assert!(seen_gather && seen_unflatten);

        // The dense case mints nothing: those spellings are the same access.
        let dense = ts::kmap(
            &mut g,
            &shape,
            ScalarExpr::un(UnOp::Neg, ScalarExpr::arg(0, Dtype::F32)),
            vec![alias_operand_of(src, &shape)],
        );
        assert!(fire(&mut g, dense, &OPERAND_GATHER).is_none());
        assert!(fire(&mut g, dense, &OPERAND_UNFLATTEN).is_none());
        assert_eq!(g.chain(dense).len(), 1);
    }
}
