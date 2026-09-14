//! `SCATTER_ADD_AS_FOLD`: an additive scatter as a fold over the updates.
//! `out[.., bin, ..] = base[.., bin, ..] + sum_u [idx[u] == bin] upd[.., u, ..]`
//! walks every update once per bin, which is `bins` times the work of the
//! scatter — and a fold's lanes split the walk where the dense scatter runs
//! one latency-bound loop per output element.

use crate::dtype::{Dtype, Splat};
use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::launch::{AccessPlan, IndexSpace, Launch, Operand, ScheduleDomain};
use crate::ir::logical::ScatterCombine;
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::rules::ident_expr;
use crate::scalar::{BinOp, CmpOp, ScalarExpr};
use crate::shape::{Dim, Layout};
use smallvec::SmallVec;

rule!(
    SCATTER_ADD_AS_FOLD,
    level = Level::Launch,
    head = OpTag::LaunchScatter,
    tag = RuleTag::Additive,
    apply = scatter_as_fold,
);

fn contiguous_strides(shape: &[Dim]) -> Option<Vec<u64>> {
    let mut strides = vec![0u64; shape.len()];
    let mut acc = 1u64;
    for (i, d) in shape.iter().enumerate().rev() {
        strides[i] = acc;
        acc = acc.checked_mul(d.as_const()?)?;
    }
    Some(strides)
}

pub fn scatter_as_fold(b: &mut Builder<'_>, id: Id, node: &Node, _f: &Facts<'_>) -> Option<Id> {
    // The dense scatter's per-lane loop is the GPU's problem; the CPU
    // target sorts and segments.
    if std::env::var_os("FUSOR_NO_SCATTER_FOLD").is_some()
        || b.caps().kind != crate::device::DeviceKind::Gpu
    {
        return None;
    }
    let Op::Launch(Launch::Scatter { axis, combine, ops, .. }) = &node.op else {
        return None;
    };
    if *combine != ScatterCombine::Add || ops.len() != 3 {
        return None;
    }
    let (base, idx, upd) = (&ops[0], &ops[1], &ops[2]);
    if !ops.iter().all(|o| matches!(o.access, AccessPlan::Alias) && o.layout.is_contiguous()) {
        return None;
    }
    let a = *axis as usize;
    let base_shape: Vec<Dim> = base.layout.shape().to_vec();
    let idx_shape = idx.layout.shape();
    let upd_shape = upd.layout.shape();
    if idx_shape.len() != 1 || a >= base_shape.len() || upd_shape.len() != base_shape.len() {
        return None;
    }
    let updates = idx_shape[0];
    if upd_shape[a] != updates {
        return None;
    }
    let dtype = b.facts_of(base.src).dtype;
    let idx_dtype = b.facts_of(idx.src).dtype;
    if !matches!(dtype, Dtype::F32) || !matches!(idx_dtype, Dtype::U32 | Dtype::I32) {
        return None;
    }
    let rank = base_shape.len();
    let mut space: SmallVec<[Dim; 6]> = SmallVec::from_vec(base_shape.clone());
    space.push(updates);
    let base_strides = contiguous_strides(&base_shape)?;
    let upd_strides = contiguous_strides(upd_shape)?;
    let dim = |v: u64| Dim::Const(v);
    let layout = |strides: Vec<u64>| -> Option<Layout> {
        let strides: Vec<Dim> = strides.into_iter().map(dim).collect();
        Layout::from_parts(Dim::Const(0), &space, &strides).ok()
    };
    // `upd` walks the bins axis not at all and the updates axis at its own.
    let mut s_upd = upd_strides.clone();
    s_upd[a] = 0;
    s_upd.push(upd_strides[a]);
    let mut s_base = base_strides;
    s_base.push(0);
    let mut s_idx = vec![0u64; rank];
    s_idx.push(1);
    let operand = |src: Id, layout: Layout| Operand { src, layout, access: AccessPlan::Alias };
    let ops = vec![
        operand(upd.src, layout(s_upd)?),
        operand(idx.src, layout(s_idx)?),
        operand(base.src, layout(s_base)?),
    ];
    let zero = ScalarExpr::lit(Splat::F32(0.0));
    let hit = ScalarExpr::cmp(
        CmpOp::Eq,
        ScalarExpr::cast(Dtype::U32, ScalarExpr::arg(1, idx_dtype)),
        ScalarExpr::index_of(a as u32),
    );
    let first = ScalarExpr::cmp(CmpOp::Eq, ScalarExpr::index_of(rank as u32), ScalarExpr::lit(Splat::U32(0)));
    let lift = ScalarExpr::bin(
        BinOp::Add,
        ScalarExpr::select(hit, ScalarExpr::arg(0, dtype), zero.clone()),
        ScalarExpr::select(first, ScalarExpr::arg(2, dtype), zero),
    );
    let fold = b
        .add_launch(Launch::Fold {
            space: IndexSpace::new(space.iter().copied()),
            axis: rank as u32,
            vec_axes: SmallVec::new(),
            carrier: crate::carrier::Carrier::binop(
                BinOp::Add,
                crate::carrier::Carrier::binop_identity(BinOp::Add, dtype)?,
                dtype,
            )
            .with_lift([lift]),
            acc: dtype,
            post: smallvec::smallvec![ident_expr(dtype)],
            ops,
            sched: ScheduleDomain::Point,
        })
        .ok()?;
    b.union(id, fold).ok()
}
