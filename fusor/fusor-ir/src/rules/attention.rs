//! Streaming single-query attention recognized from contractions and stable
//! softmax. No model entry point or device name participates in selection.
use super::normalization::{binary, logical, unbroadcast};
use crate::carrier::Carrier;
use crate::device::DeviceKind;
use crate::dtype::{Dtype, Splat};
use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::launch::{IndexSpace, Launch, ScheduleDomain};
use crate::ir::logical::Logical;
use crate::ir::{Level, Node, OpTag};
use crate::rule;
use crate::scalar::{BinOp, CmpOp, Lit, ScalarExpr, ScalarKind, UnOp};
use crate::shape::{Dim, DimExpr, OPAQUE_SYM};

rule!(
    FUSE_STREAMING_ATTENTION,
    level = Level::Logical,
    head = OpTag::Contract,
    tag = RuleTag::Additive,
    apply = recognize,
);

fn same(b: &Builder<'_>, a: Id, c: Id) -> bool {
    b.class_of(a) == b.class_of(c)
}
fn fold(b: &Builder<'_>, id: Id, op: BinOp, axis: usize) -> Option<Id> {
    let Logical::Fold {
        carrier,
        axis: got,
        acc: Dtype::F32,
        ins,
    } = logical(b, id)?
    else {
        return None;
    };
    let expected = Carrier::binop(op, Carrier::binop_identity(op, Dtype::F32)?, Dtype::F32);
    if got as usize != axis || carrier != expected || ins.len() != 1 {
        return None;
    }
    Some(ins[0])
}
fn softmax(b: &Builder<'_>, id: Id) -> Option<Id> {
    let shape = b.facts_of(id).shape.clone();
    let axis = shape.len().checked_sub(1)?;
    let rows = &shape[..axis];
    let axes: Vec<_> = (0..axis).collect();
    let (exp, sum) = binary(b, id, BinOp::Div)?;
    let sum = unbroadcast(b, sum, rows, &axes)?;
    if !same(b, fold(b, sum, BinOp::Add, axis)?, exp) {
        return None;
    }
    let Logical::Map { expr, ins, outs: 1 } = logical(b, exp)? else {
        return None;
    };
    let ScalarKind::Un { op: UnOp::Exp, x } = expr.kind() else {
        return None;
    };
    let ScalarKind::Arg(i) = x.kind() else {
        return None;
    };
    let (scores, max) = binary(b, *ins.get(*i as usize)?, BinOp::Sub)?;
    let max = unbroadcast(b, max, rows, &axes)?;
    same(b, fold(b, max, BinOp::Max, axis)?, scores).then_some(scores)
}
/// A prefix mask is accepted only when its false branch is the exact neutral
/// value and its predicate is a scalar prefix bound on the key coordinate.
fn prefix_mask(b: &Builder<'_>, id: Id, axis: u32, neutral: f32) -> Option<(Id, ScalarExpr)> {
    let Logical::Map { expr, ins, outs: 1 } = logical(b, id)? else {
        return None;
    };
    let ScalarKind::Select { c, t, f } = expr.kind() else {
        return None;
    };
    let ScalarKind::Lit(Lit(Splat::F32(v))) = f.kind() else {
        return None;
    };
    if v.to_bits() != neutral.to_bits() {
        return None;
    }
    let ScalarKind::Arg(i) = t.kind() else {
        return None;
    };
    let ScalarKind::Cmp {
        op: CmpOp::Lt,
        a,
        b: prefix,
    } = c.kind()
    else {
        return None;
    };
    let coord = match a.kind() {
        ScalarKind::Cast { to: Dtype::F32, x } => x,
        _ => a,
    };
    if coord.kind() != &ScalarKind::IndexOf(axis) {
        return None;
    }
    if !matches!(prefix.kind(), ScalarKind::Uniform(_) | ScalarKind::Lit(_)) {
        return None;
    }
    Some((*ins.get(*i as usize)?, prefix.clone()))
}
fn dim_scalar(d: Dim) -> Option<ScalarExpr> {
    match d {
        Dim::Const(n) => Some(ScalarExpr::lit(Splat::U32(n.try_into().ok()?))),
        Dim::Sym(s) if s != OPAQUE_SYM => match s.derived_expr() {
            None => Some(ScalarExpr::uniform(s, Dtype::U32)),
            Some(DimExpr::Add(a, b)) => {
                Some(ScalarExpr::bin(BinOp::Add, dim_scalar(a)?, dim_scalar(b)?))
            }
            Some(DimExpr::Mul(a, b)) => {
                Some(ScalarExpr::bin(BinOp::Mul, dim_scalar(a)?, dim_scalar(b)?))
            }
            Some(DimExpr::DivCeil(..)) => None,
        },
        _ => None,
    }
}
fn recognize(b: &mut Builder<'_>, id: Id, _node: &Node, _f: &Facts<'_>) -> Option<Id> {
    let caps = b.caps();
    if caps.kind != DeviceKind::Gpu {
        return None;
    }
    let subgroup = caps.subgroups.filter(|s| s.is_fixed())?.assumed();
    if !subgroup.is_power_of_two() {
        return None;
    }
    let Logical::Contract {
        spec: pv,
        acc: Dtype::F32,
        a: prob,
        b: mut value,
        outs: 1,
    } = logical(b, id)?
    else {
        return None;
    };
    let shape = b.facts_of(id).shape.clone();
    let rank = shape.len();
    if !(4..=5).contains(&rank) || shape[rank - 2] != Dim::ONE {
        return None;
    }
    let width = shape[rank - 1].as_const()?;
    if width == 0 || width > 256 || !width.is_multiple_of(u64::from(subgroup)) {
        return None;
    }
    let scores = softmax(b, prob)?;
    let (scores, prefix) = if let Some((scores, prefix)) =
        prefix_mask(b, scores, (rank - 1) as u32, f32::NEG_INFINITY)
    {
        let (v, bound) = prefix_mask(b, value, 2, 0.0)?;
        if bound != prefix {
            return None;
        }
        value = v;
        (scores, Some(prefix))
    } else {
        (scores, None)
    };
    let Logical::Map { expr, ins, outs: 1 } = logical(b, scores)? else {
        return None;
    };
    let ScalarKind::Bin {
        op: BinOp::Mul,
        a,
        b: rhs,
    } = expr.kind()
    else {
        return None;
    };
    let (dot, scale) = match (a.kind(), rhs.kind()) {
        (ScalarKind::Arg(i), ScalarKind::Uniform(_) | ScalarKind::Lit(_)) => {
            (*ins.get(*i as usize)?, rhs.clone())
        }
        (ScalarKind::Uniform(_) | ScalarKind::Lit(_), ScalarKind::Arg(i)) => {
            (*ins.get(*i as usize)?, a.clone())
        }
        _ => return None,
    };
    let Logical::Contract {
        spec: qk,
        acc: Dtype::F32,
        a: query,
        b: key,
        outs: 1,
    } = logical(b, dot)?
    else {
        return None;
    };
    let qs = b.facts_of(query).shape.clone();
    let ks = b.facts_of(key).shape.clone();
    if qs != shape
        || ks.len() != 4
        || ks != b.facts_of(value).shape
        || ks[0] != shape[0]
        || ks[1] != shape[1]
        || ks[3] != shape[rank - 1]
        || ks[2] == Dim::Const(0)
    {
        return None;
    }
    // Check label relationships, not the arbitrary label numbers chosen by a frontend.
    if pv.a != qk.out
        || pv.out != qk.a
        || pv.b != qk.b
        || qk.a.len() != rank
        || qk.b.len() != 4
        || qk.out.len() != rank
        || qk.a[..rank - 1] != qk.out[..rank - 1]
        || qk.b[0] != qk.a[0]
        || qk.b[1] != qk.a[1]
        || qk.b[2] != qk.out[rank - 1]
        || qk.b[3] != qk.a[rank - 1]
        || qk.b[2] == qk.b[3]
    {
        return None;
    }
    if [id, query, key, value].iter().any(|v| {
        let f = b.facts_of(*v);
        f.dtype != Dtype::F32 || !f.numeric.reassoc || !f.numeric.contract
    }) {
        return None;
    }
    let batch = shape[0].as_const()?;
    let kv_heads = shape[1].as_const()?;
    let groups = if rank == 5 { shape[2].as_const()? } else { 1 };
    let heads = kv_heads.checked_mul(groups)?;
    if batch == 0
        || heads == 0
        || batch.checked_mul(heads)?.checked_mul(width)? > u64::from(u32::MAX)
    {
        return None;
    }
    let keys = ks[2];
    let prefix = prefix.or_else(|| dim_scalar(keys))?;
    let limits = &b.caps().limits;
    if width
        > u64::from(
            limits
                .max_compute_invocations_per_workgroup
                .min(limits.max_compute_workgroup_size[0]),
        )
    {
        return None;
    }
    let max_block = limits
        .max_compute_invocations_per_workgroup
        .min(limits.max_compute_workgroup_size[0])
        .min(256)
        .min(limits.max_compute_workgroup_storage_size / ((width as u32 + 2) * 4) * subgroup);
    let block = 1 << max_block.checked_ilog2()?;
    if block < subgroup {
        return None;
    }
    let chunk = if keys.as_const().is_some_and(|n| n <= 256) {
        0
    } else {
        256
    };
    let splits = if chunk == 0 {
        Dim::ONE
    } else {
        keys.div_ceil(u64::from(chunk))
    };
    let output = [
        Dim::Const(batch),
        Dim::Const(heads),
        Dim::ONE,
        Dim::Const(width),
    ];
    let partial = [
        output[0],
        output[1],
        splits,
        Dim::Const(width + if chunk == 0 { 0 } else { 2 }),
    ];
    let fused = b
        .add_launch(Launch::DecodeAttention {
            output: IndexSpace::new(partial),
            kv_heads: kv_heads.try_into().ok()?,
            keys,
            block,
            subgroup,
            chunk,
            scale,
            prefix,
            ops: vec![
                super::alias_operand_of(query, &output),
                super::alias_operand_of(key, &ks),
                super::alias_operand_of(value, &ks),
            ],
            sched: ScheduleDomain::Point,
        })
        .ok()?;
    let fused = if chunk == 0 {
        fused
    } else {
        b.add_launch(Launch::DecodeAttentionMerge {
            output: IndexSpace::new(output),
            splits,
            ops: vec![super::alias_operand_of(fused, &partial)],
            sched: ScheduleDomain::Point,
        })
        .ok()?
    };
    let fused = if rank == 4 {
        fused
    } else {
        b.add_launch(Launch::Map {
            space: IndexSpace::new(shape.clone()),
            body: ScalarExpr::arg(0, Dtype::F32),
            ops: vec![super::alias_operand_of(fused, &shape)],
            sched: ScheduleDomain::Point,
        })
        .ok()?
    };
    b.union(id, fused).ok()
}
