//! The [`crate::egraph::RuleTag::StrictlyLowering`] floor: one trivial,
//! always-legal Logical -> Launch lowering per Logical op. These are what the driver falls
//! back to when a saturation budget is exhausted.
//!
//! Every one emits [`ScheduleDomain::Point`]: the floor depends on no
//! schedule generator. Every one is trivially correct and trivially slow. `Leaf` needs no
//! rule.
//!
//! # This module is the only place `ScheduleDomain::Point` is minted
//!
//! `Point` is the marker the schedule rules match on. A rule that needs to mint a
//! nest carrying no schedule of its own calls [`floor_map`],
//! [`floor_alias_map`] or [`floor_fold`], and a descriptor that needs the
//! value calls [`floor_sched`].

use crate::dtype::Dtype;
use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::carrier::Carrier;
use crate::ir::logical::{Logical, Label};
use crate::ir::launch::{
    AccessPlan, GatherMode, IndexSpace, Launch, Operand, ScatterMode, ScheduleDomain,
};
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::rules::{alias_operand_of, composed_layout, ident_expr};
use crate::scalar::{BinOp, ScalarExpr};
use crate::shape::{AxisGroup, Dim, Layout, MultiFlattenMap, SubAxis};
use smallvec::SmallVec;

rule!(
    LOWER_MAP,
    level = Level::Logical,
    head = OpTag::Map,
    tag = RuleTag::StrictlyLowering,
    apply = lower_map,
);

rule!(
    LOWER_FOLD,
    level = Level::Logical,
    head = OpTag::Fold,
    tag = RuleTag::StrictlyLowering,
    apply = lower_fold,
);

rule!(
    LOWER_CONTRACT_GENERIC,
    level = Level::Logical,
    head = OpTag::Contract,
    tag = RuleTag::StrictlyLowering,
    apply = lower_contract_generic,
);

rule!(
    LOWER_RESTRIDE,
    level = Level::Logical,
    head = OpTag::Restride,
    tag = RuleTag::StrictlyLowering,
    apply = lower_restride,
);

rule!(
    LOWER_WINDOW,
    level = Level::Logical,
    head = OpTag::Window,
    tag = RuleTag::StrictlyLowering,
    apply = lower_window,
);

rule!(
    LOWER_GATHER,
    level = Level::Logical,
    head = OpTag::Gather,
    tag = RuleTag::StrictlyLowering,
    apply = lower_gather,
);

rule!(
    LOWER_SCATTER,
    level = Level::Logical,
    head = OpTag::Scatter,
    tag = RuleTag::StrictlyLowering,
    apply = lower_scatter,
);

rule!(
    LOWER_DEQUANT,
    level = Level::Logical,
    head = OpTag::Dequant,
    tag = RuleTag::StrictlyLowering,
    apply = lower_dequant,
);

rule!(
    LOWER_PROJECT,
    level = Level::Logical,
    head = OpTag::Project,
    tag = RuleTag::StrictlyLowering,
    apply = lower_project,
);

fn space_of(f: &Facts<'_>) -> IndexSpace {
    IndexSpace::new(f.own().shape.iter().copied())
}

/// The schedule a nest carries before any schedule rule has spoken.
///
/// For a *descriptor* rather than a node: [`crate::rules::tuple`] normalizes an
/// `Logical::Fold` into the `Fold` fields it would lower to, and the schedule field
/// of that normalization is this — the same value [`lower_fold`] would put
/// there. Prefer the node constructors below wherever an id is what is wanted.
pub(crate) fn floor_sched() -> ScheduleDomain {
    ScheduleDomain::Point
}

/// A `Map` minted with no schedule of its own.
///
/// The schedule rules expand it exactly as they expand a `lower_map` output:
/// a rule that restates a value does not decide how it is scheduled.
pub(crate) fn floor_map(
    b: &mut Builder<'_>,
    space: IndexSpace,
    body: ScalarExpr,
    ops: Vec<Operand>,
) -> Option<Id> {
    b.add_launch(Launch::Map {
        space,
        body,
        ops,
        sched: floor_sched(),
    })
    .ok()
}

/// The identity-body alias `Map` that re-expresses `src` at `shape` through
/// `layout`.
///
/// This is the readback spelling: a slot view of a multi-slot carrier, a
/// recovery view of a promoted fold's flattened carrier axis. It computes
/// nothing — the body is `Arg(0)` and the access is [`AccessPlan::Alias`] —
/// so it carries no schedule decision at all.
pub(crate) fn floor_alias_map(
    b: &mut Builder<'_>,
    src: Id,
    layout: Layout,
    shape: &[Dim],
    dtype: Dtype,
) -> Option<Id> {
    floor_map(
        b,
        IndexSpace::new(shape.iter().copied()),
        ident_expr(dtype),
        vec![Operand {
            src,
            layout,
            access: AccessPlan::Alias,
        }],
    )
}

/// A `Fold` minted with no schedule of its own — the nest [`lower_fold`]
/// would have minted, given these fields.
///
/// TUPLE's joint carrier is the case: the joint takes neither side's schedule,
/// because a schedule domain is not a value and a joint that inherited one
/// would be a function of which spelling the consumer's operand happened to
/// name.
#[allow(clippy::too_many_arguments)]
pub(crate) fn floor_fold(
    b: &mut Builder<'_>,
    space: IndexSpace,
    axis: u32,
    vec_axes: SmallVec<[u32; 2]>,
    carrier: Carrier,
    acc: Dtype,
    post: SmallVec<[ScalarExpr; 4]>,
    ops: Vec<Operand>,
) -> Option<Id> {
    b.add_launch(Launch::Fold {
        space,
        axis,
        vec_axes,
        carrier,
        acc,
        post,
        ops,
        sched: floor_sched(),
    })
    .ok()
}

/// `Logical::Map` -> `Launch::Map` reading every operand through its own dense
/// layout.
pub fn lower_map(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Logical(Logical::Map { expr, ins, .. }) = &node.op else {
        return None;
    };
    let ops: Vec<Operand> = ins
        .iter()
        .map(|&s| alias_operand_of(s, &b.facts_of(s).shape.clone()))
        .collect();
    let k = b
        .add_launch(Launch::Map {
            space: space_of(f),
            body: expr.clone(),
            ops,
            sched: ScheduleDomain::Point,
        })
        .ok()?;
    b.union(id, k).ok()
}

/// `Logical::Fold` -> `Launch::Fold` over the pre-reduction space with identity
/// `pre` and `post`.
pub fn lower_fold(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Logical(Logical::Fold {
        carrier,
        axis,
        acc,
        ins,
    }) = &node.op
    else {
        return None;
    };
    let in_shape = f.operand(0)?.shape.clone();
    let dtype = f.dtype(0)?;
    // The nest reads its operands at the operand dtype and accumulates at
    // `acc`, so the lift is retyped while the merge rides through untouched.
    //
    // Retyped, **not replaced**. A per-slot `Arg(0)` is right only for a
    // single-slot binop carrier, whose lift already is `Arg(0)`; Welford's
    // `(1, x, 0)` and a shift-stabilized `(x, 1)` would be silently rewritten
    // into "every slot folds the element", which reduces `n` and `m2` over the
    // data instead of over the constants they are.
    let lift: SmallVec<[ScalarExpr; 4]> = carrier
        .lift
        .iter()
        .map(|e| crate::carrier::retype_args(e, dtype))
        .collect();
    let k = b
        .add_launch(Launch::Fold {
            space: IndexSpace::new(in_shape.iter().copied()),
            axis: *axis,
            vec_axes: SmallVec::new(),
            carrier: carrier.clone().with_lift(lift),
            acc: *acc,
            post: (0..carrier.width())
                .map(|i| ScalarExpr::arg(i as u32, *acc))
                .collect(),
            ops: ins
                .iter()
                .map(|x| alias_operand_of(*x, &in_shape))
                .collect(),
            sched: ScheduleDomain::Point,
        })
        .ok()?;
    b.union(id, k).ok()
}

/// `Logical::Contract` -> `Fold { combine: Add, pre: mul(Arg0, Arg1) }`.
///
/// This is the family-free floor: no lane geometry, no tile, no split. The
/// four order-free family rules ride in a target's own rule table and are
/// appended to the core set by the driver's caller.
pub fn lower_contract_generic(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    f: &Facts<'_>,
) -> Option<Id> {
    let Op::Logical(Logical::Contract {
        spec, acc, a, b: rhs, ..
    }) = &node.op
    else {
        return None;
    };
    let a_shape = f.operand(0)?.shape.clone();
    let b_shape = f.operand(1)?.shape.clone();
    let out_shape = f.own().shape.clone();

    let contracted: SmallVec<[Label; 4]> = spec
        .a
        .iter()
        .copied()
        .filter(|l| spec.b.contains(l) && !spec.out.contains(l))
        .collect();
    let k = fold_extent(&contracted, spec, &a_shape, &b_shape)?;

    let mut space: SmallVec<[Dim; 6]> = out_shape.clone();
    space.push(k);
    let axis = out_shape.len() as u32;

    let dtype = f.dtype(0)?;
    let pre = ScalarExpr::bin(
        BinOp::Mul,
        ScalarExpr::arg(0, dtype),
        ScalarExpr::arg(1, f.dtype(1)?),
    );
    let ops = vec![
        contract_operand(*a, &spec.a, &a_shape, spec, &out_shape, &contracted, &a_shape, &b_shape)?,
        contract_operand(*rhs, &spec.b, &b_shape, spec, &out_shape, &contracted, &a_shape, &b_shape)?,
    ];
    let kf = b
        .add_launch(Launch::Fold {
            space: IndexSpace::new(space),
            axis,
            vec_axes: SmallVec::new(),
            carrier: Carrier::binop(
                BinOp::Add,
                Carrier::binop_identity(BinOp::Add, *acc)?,
                *acc,
            )
            .with_lift([pre]),
            acc: *acc,
            post: smallvec::smallvec![ident_expr(*acc)],
            ops,
            sched: ScheduleDomain::Point,
        })
        .ok()?;
    b.union(id, kf).ok()
}

/// One contraction operand read over the fold's `[out..., k]` index space.
///
/// The fold walks the output plus one merged contraction axis; the operand's
/// own axes are in `spec` order, which in general is neither. Aliasing the
/// operand's dense layout says "axis `i` of the space is axis `i` of the
/// operand", which is true only for a left operand of a canonical matmul.
/// So each space axis gets an explicit stride: the operand's stride for that label,
/// or 0 where the operand does not carry it, and the merged `k` axis
/// decomposes into one sub-axis per contracted label, most significant
/// first, which is the order `fold_extent` multiplied them in.
#[allow(clippy::too_many_arguments)]
fn contract_operand(
    src: Id,
    labels: &[Label],
    shape: &[Dim],
    spec: &crate::ir::logical::EinSpec,
    out_shape: &[Dim],
    contracted: &[Label],
    a_shape: &[Dim],
    b_shape: &[Dim],
) -> Option<Operand> {
    let strides = Layout::row_major_strides(shape);
    // A label repeated within one operand is a diagonal read: its strides add.
    let stride_of = |l: Label| -> Option<u32> {
        let mut acc: u64 = 0;
        for (i, x) in labels.iter().enumerate() {
            if *x == l {
                acc = acc.checked_add(strides.get(i)?.as_const()?)?;
            }
        }
        u32::try_from(acc).ok()
    };
    let label_extent = |l: Label| -> Option<u32> {
        let d = spec
            .a
            .iter()
            .position(|x| *x == l)
            .and_then(|i| a_shape.get(i))
            .or_else(|| {
                spec.b
                    .iter()
                    .position(|x| *x == l)
                    .and_then(|i| b_shape.get(i))
            })?;
        u32::try_from(d.as_const()?).ok()
    };

    let mut groups: SmallVec<[AxisGroup; 4]> = SmallVec::new();
    for (axis, l) in spec.out.iter().copied().enumerate() {
        let extent = u32::try_from(out_shape.get(axis)?.as_const()?).ok()?;
        groups.push(AxisGroup::affine(extent, stride_of(l)?));
    }
    let mut subs: SmallVec<[SubAxis; 2]> = SmallVec::new();
    for l in contracted {
        subs.push(SubAxis {
            extent: label_extent(*l)?,
            stride: stride_of(*l)?,
        });
    }
    if subs.is_empty() {
        subs.push(SubAxis { extent: 1, stride: 0 });
    }
    groups.push(AxisGroup { sub_axes: subs });

    Some(Operand {
        src,
        layout: Layout::contiguous(shape),
        access: AccessPlan::Unflatten(MultiFlattenMap { groups }),
    })
}

/// The single reduction extent a generic fold walks. One contracted label is
/// that label's extent; several decidable ones collapse to their product;
/// several symbolic ones have no single-loop floor and decline.
fn fold_extent(
    contracted: &[Label],
    spec: &crate::ir::logical::EinSpec,
    a: &[Dim],
    b: &[Dim],
) -> Option<Dim> {
    let extent = |l: &Label| -> Option<Dim> {
        spec.a
            .iter()
            .position(|x| x == l)
            .and_then(|i| a.get(i).copied())
            .or_else(|| {
                spec.b
                    .iter()
                    .position(|x| x == l)
                    .and_then(|i| b.get(i).copied())
            })
    };
    match contracted {
        [] => Some(Dim::ONE),
        [one] => extent(one),
        many => {
            let mut product = 1u64;
            for l in many {
                product = product.checked_mul(extent(l)?.as_const()?)?;
            }
            Some(Dim::Const(product))
        }
    }
}

/// `Logical::Restride` -> a copying `Map` whose operand carries the composed
/// view. When the composition is not decidable the operand falls to a
/// per-element address computation rather than to an invented contiguous
/// layout.
pub fn lower_restride(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Logical(Logical::Restride { specs, x, .. }) = &node.op else {
        return None;
    };
    let in_shape = f.operand(0)?.shape.clone();
    let dtype = f.dtype(0)?;
    let operand = match composed_layout(specs, &in_shape) {
        Some(layout) => Operand {
            src: *x,
            layout,
            access: AccessPlan::Alias,
        },
        None => Operand {
            src: *x,
            layout: Layout::contiguous(&in_shape),
            access: AccessPlan::Gather,
        },
    };
    let k = b
        .add_launch(Launch::Map {
            space: space_of(f),
            body: ident_expr(dtype),
            ops: vec![operand],
            sched: ScheduleDomain::Point,
        })
        .ok()?;
    b.union(id, k).ok()
}

/// `Logical::Window` -> a `Map` whose operand is read through the window's index
/// map: one `AxisGroup` per output axis, the windowed axes decomposed into
/// (position, offset) sub-axes so overlapping strides stay expressible.
pub fn lower_window(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Logical(Logical::Window { specs, x }) = &node.op else {
        return None;
    };
    let in_shape = f.operand(0)?.shape.clone();
    let out_shape = f.own().shape.clone();
    let dtype = f.dtype(0)?;
    let in_strides = Layout::row_major_strides(&in_shape);

    let access = window_map(specs, &in_shape, &out_shape, &in_strides)
        .map_or(AccessPlan::Gather, AccessPlan::Unflatten);
    let k = b
        .add_launch(Launch::Map {
            space: space_of(f),
            body: ident_expr(dtype),
            ops: vec![Operand {
                src: *x,
                layout: Layout::contiguous(&in_shape),
                access,
            }],
            sched: ScheduleDomain::Point,
        })
        .ok()?;
    b.union(id, k).ok()
}

fn window_map(
    specs: &[crate::shape::SlidingWindow],
    in_shape: &[Dim],
    out_shape: &[Dim],
    in_strides: &[Dim],
) -> Option<MultiFlattenMap> {
    let mut groups: SmallVec<[AxisGroup; 4]> = SmallVec::new();
    for (axis, extent) in out_shape.iter().enumerate() {
        let extent = u32::try_from(extent.as_const()?).ok()?;
        // Output axes past the input rank are the appended window offsets,
        // in the order `Logical::Window` pushed them.
        let (src_axis, step) = if axis < in_shape.len() {
            let step = specs
                .iter()
                .find(|w| w.axis as usize == axis)
                .map_or(1, |w| w.step);
            (axis, step)
        } else {
            let w = specs.get(axis - in_shape.len())?;
            (w.axis as usize, 1)
        };
        let base = u32::try_from(in_strides.get(src_axis)?.as_const()?).ok()?;
        groups.push(AxisGroup {
            sub_axes: smallvec::smallvec![SubAxis {
                extent,
                stride: base.checked_mul(step)?,
            }],
        });
    }
    Some(MultiFlattenMap { groups })
}

/// `Logical::Gather` -> `Gather { mode: RowPerGroup }`, the mode legal on every
/// device.
pub fn lower_gather(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Logical(Logical::Gather { axis, x, idx }) = &node.op else {
        return None;
    };
    let k = b
        .add_launch(Launch::Gather {
            space: space_of(f),
            axis: *axis,
            mode: GatherMode::RowPerGroup,
            ops: vec![
                alias_operand_of(*x, &f.operand(0)?.shape.clone()),
                alias_operand_of(*idx, &f.operand(1)?.shape.clone()),
            ],
            sched: ScheduleDomain::Point,
        })
        .ok()?;
    b.union(id, k).ok()
}

/// `Logical::Scatter` -> `Scatter { mode: SortSegment }`. Atomics and the
/// workgroup-private merge are target rules guarded on capabilities; the
/// sorted segmented reduce needs neither and is therefore the floor.
pub fn lower_scatter(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Logical(Logical::Scatter {
        axis,
        combine,
        base,
        idx,
        upd,
        ..
    }) = &node.op
    else {
        return None;
    };
    let k = b
        .add_launch(Launch::Scatter {
            space: space_of(f),
            axis: *axis,
            mode: ScatterMode::SortSegment,
            combine: *combine,
            ops: vec![
                alias_operand_of(*base, &f.operand(0)?.shape.clone()),
                alias_operand_of(*idx, &f.operand(1)?.shape.clone()),
                alias_operand_of(*upd, &f.operand(2)?.shape.clone()),
            ],
            sched: ScheduleDomain::Point,
        })
        .ok()?;
    b.union(id, k).ok()
}

/// `Logical::Dequant` -> a `Map` reading a quantized operand. The block program
/// itself lives in the format table, keyed by `(fmt, layout)`; the nest only
/// has to say that this operand decodes.
pub fn lower_dequant(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Logical(Logical::Dequant { x, .. }) = &node.op else {
        return None;
    };
    let out = f.own().dtype;
    let k = b
        .add_launch(Launch::Map {
            space: space_of(f),
            body: ident_expr(if out.is_quantized() { Dtype::F32 } else { out }),
            ops: vec![alias_operand_of(*x, &f.operand(0)?.shape.clone())],
            sched: ScheduleDomain::Point,
        })
        .ok()?;
    b.union(id, k).ok()
}

/// `Logical::Project` -> a `Map` selecting one slot of a tuple-producing
/// operand. The slot rides in the body as `Arg(slot)`.
pub fn lower_project(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    let Op::Logical(Logical::Project { slot, x }) = &node.op else {
        return None;
    };
    let k = b
        .add_launch(Launch::Map {
            space: space_of(f),
            body: ScalarExpr::arg(u32::from(*slot), f.own().dtype),
            ops: vec![alias_operand_of(*x, &f.operand(0)?.shape.clone())],
            sched: ScheduleDomain::Point,
        })
        .ok()?;
    b.union(id, k).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::logical::EinSpec;
    use crate::rules::test_support as ts;
    use crate::scalar::UnOp;
    use crate::shape::{SlidingWindow, StrideSpec};

    fn fire(g: &mut crate::egraph::EGraph, id: Id, r: &crate::egraph::Rule) -> Option<Id> {
        let caps = ts::caps();
        let node = g.node(id).clone();
        let facts = g.facts_view(id, &caps);
        let mut b = g.builder(&caps);
        (r.apply)(&mut b, id, &node, &facts)
    }

    fn l1_member(g: &crate::egraph::EGraph, id: Id) -> Id {
        g.chain(id)
            .into_iter()
            .find(|&i| g.level(i) == Level::Launch)
            .expect("an Launch member")
    }

    #[test]
    fn the_floor_covers_every_l0_op_but_leaf() {
        let mut g = ts::graph();
        let x = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(4), Dim::Const(6)]);
        let idx = ts::buffer(&mut g, Dtype::U32, &[Dim::Const(3)]);

        let m = ts::map(
            &mut g,
            ScalarExpr::un(UnOp::Exp, ScalarExpr::arg(0, Dtype::F32)),
            &[x],
        );
        assert!(fire(&mut g, m, &LOWER_MAP).is_some());
        assert!(matches!(g.node(l1_member(&g, m)).op, Op::Launch(Launch::Map { .. })));

        let fd = ts::fold(
            &mut g,
            ts::binop_carrier(BinOp::Add, Dtype::F32),
            1,
            Dtype::F32,
            x,
        );
        assert!(fire(&mut g, fd, &LOWER_FOLD).is_some());
        assert!(matches!(
            g.node(l1_member(&g, fd)).op,
            Op::Launch(Launch::Fold { .. })
        ));

        let rhs = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(6), Dim::Const(2)]);
        let ct = ts::contract(
            &mut g,
            EinSpec {
                a: smallvec::smallvec![Label(0), Label(1)],
                b: smallvec::smallvec![Label(1), Label(2)],
                out: smallvec::smallvec![Label(0), Label(2)],
            },
            Dtype::F32,
            x,
            rhs,
        );
        assert!(fire(&mut g, ct, &LOWER_CONTRACT_GENERIC).is_some());
        let kf = l1_member(&g, ct);
        let Op::Launch(Launch::Fold {
            space,
            axis,
            carrier,
            ..
        }) = &g.node(kf).op
        else {
            panic!()
        };
        assert_eq!(space.rank(), 3);
        assert_eq!(*axis, 2);
        assert_eq!(space.dims[2], Dim::Const(6));
        // The product is the carrier's lift; the merge is still a plain `Add`,
        // so the hardware collective is unchanged.
        assert!(matches!(
            carrier.lift[0].kind(),
            crate::scalar::ScalarKind::Bin {
                op: BinOp::Mul,
                ..
            }
        ));
        assert_eq!(carrier.kind(), Some(BinOp::Add));

        let rs = ts::restride(
            &mut g,
            &[
                StrideSpec::dim(1, Dim::Const(6)),
                StrideSpec::dim(0, Dim::Const(4)),
            ],
            x,
        );
        assert!(fire(&mut g, rs, &LOWER_RESTRIDE).is_some());
        let km = l1_member(&g, rs);
        let Op::Launch(Launch::Map { ops, .. }) = &g.node(km).op else {
            panic!()
        };
        assert!(matches!(ops[0].access, AccessPlan::Alias));
        assert_eq!(ops[0].layout.strides(), &[Dim::Const(1), Dim::Const(6)]);

        let win = g
            .add(Op::Logical(Logical::Window {
                specs: smallvec::smallvec![SlidingWindow::new(1, 3, 1)],
                x,
            }))
            .unwrap();
        assert!(fire(&mut g, win, &LOWER_WINDOW).is_some());
        let kw = l1_member(&g, win);
        let Op::Launch(Launch::Map { ops, .. }) = &g.node(kw).op else {
            panic!()
        };
        assert!(matches!(ops[0].access, AccessPlan::Unflatten(_)));

        let gth = g
            .add(Op::Logical(Logical::Gather { axis: 0, x, idx }))
            .unwrap();
        assert!(fire(&mut g, gth, &LOWER_GATHER).is_some());
        assert!(matches!(
            g.node(l1_member(&g, gth)).op,
            Op::Launch(Launch::Gather {
                mode: GatherMode::RowPerGroup,
                ..
            })
        ));

        let upd = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(3), Dim::Const(6)]);
        let sc = ts::scatter(&mut g, 0, x, idx, upd);
        assert!(fire(&mut g, sc, &LOWER_SCATTER).is_some());
        assert!(matches!(
            g.node(l1_member(&g, sc)).op,
            Op::Launch(Launch::Scatter {
                mode: ScatterMode::SortSegment,
                ..
            })
        ));

        let q = g
            .add(Op::Logical(Logical::Leaf(crate::ir::logical::LeafKind::Quantized {
                name: crate::ir::logical::BufferId(99),
                fmt: crate::dtype::QFmt::Q4_0,
                layout: crate::dtype::QLayout::Native,
                shape: smallvec::smallvec![Dim::Const(4), Dim::Const(32)],
            })))
            .unwrap();
        let dq = g
            .add(Op::Logical(Logical::Dequant {
                fmt: crate::dtype::QFmt::Q4_0,
                layout: crate::dtype::QLayout::Native,
                x: q,
            }))
            .unwrap();
        assert!(fire(&mut g, dq, &LOWER_DEQUANT).is_some());
        assert!(matches!(
            g.node(l1_member(&g, dq)).op,
            Op::Launch(Launch::Map { .. })
        ));

        let pj = g.add(Op::Logical(Logical::Project { slot: 1, x })).unwrap();
        assert!(fire(&mut g, pj, &LOWER_PROJECT).is_some());
        let kp = l1_member(&g, pj);
        let Op::Launch(Launch::Map { body, .. }) = &g.node(kp).op else {
            panic!()
        };
        assert!(matches!(body.kind(), crate::scalar::ScalarKind::Arg(1)));
    }

    /// Re-offering a lowering rule is one memo hit: hash-consing makes the
    /// floor idempotent, which is what lets the degraded pass ignore the
    /// fired set.
    #[test]
    fn the_floor_is_idempotent() {
        let mut g = ts::graph();
        let x = ts::buffer(&mut g, Dtype::F32, &[Dim::Const(4)]);
        let m = ts::map(
            &mut g,
            ScalarExpr::un(UnOp::Exp, ScalarExpr::arg(0, Dtype::F32)),
            &[x],
        );
        assert!(fire(&mut g, m, &LOWER_MAP).is_some());
        let after_first = g.len();
        assert!(fire(&mut g, m, &LOWER_MAP).is_some());
        assert_eq!(g.len(), after_first);
        assert_eq!(g.chain(m).len(), 2);
    }
}
