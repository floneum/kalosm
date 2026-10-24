//! Recognize normalization in ordinary Map/Fold/View algebra. The logical
//! expression remains the differentiation path; extraction prices the fused
//! implementation alongside the existing lowerings.
use crate::device::DeviceKind;
use crate::dtype::{Dtype, Splat};
use crate::egraph::{Builder, Facts, Id, RuleTag};
use crate::ir::launch::{IndexSpace, Launch, ScheduleDomain};
use crate::ir::logical::Logical;
use crate::ir::{Level, Node, Op, OpTag};
use crate::rule;
use crate::scalar::{BinOp, Lit, ScalarExpr, ScalarKind, UnOp};
use crate::shape::Dim;

rule!(
    FUSE_ROW_NORMALIZATION,
    level = Level::Logical,
    head = OpTag::Map,
    tag = RuleTag::Additive,
    apply = recognize,
);

pub(super) fn logical(b: &Builder<'_>, id: Id) -> Option<Logical> {
    b.class_members(id)
        .into_iter()
        .find_map(|m| match &b.node(m).op {
            Op::Logical(op) => Some(op.clone()),
            _ => None,
        })
}
pub(super) fn binary(b: &Builder<'_>, id: Id, wanted: BinOp) -> Option<(Id, Id)> {
    let Logical::Map { expr, ins, outs: 1 } = logical(b, id)? else {
        return None;
    };
    let ScalarKind::Bin { op, a, b } = expr.kind() else {
        return None;
    };
    let (ScalarKind::Arg(i), ScalarKind::Arg(j)) = (a.kind(), b.kind()) else {
        return None;
    };
    if *op != wanted {
        return None;
    }
    Some((*ins.get(*i as usize)?, *ins.get(*j as usize)?))
}
/// Erase only exact broadcasts. Symbolic extents may carry RuntimeMask, but
/// equal source extents and zero-offset broadcast axes prove every access here.
pub(super) fn unbroadcast(b: &Builder<'_>, id: Id, source: &[Dim], axes: &[usize]) -> Option<Id> {
    if b.facts_of(id).shape.as_slice() == source {
        return Some(id);
    }
    let Logical::Restride { specs, x, .. } = logical(b, id)? else {
        return None;
    };
    if axes.len() != source.len()
        || specs.len() != b.facts_of(id).shape.len()
        || b.facts_of(x).shape.as_slice() != source
    {
        return None;
    }
    for (i, spec) in specs.iter().enumerate() {
        if spec.offset != Dim::Const(0) {
            return None;
        }
        if let Some(axis) = axes.iter().position(|a| *a == i) {
            if spec.input_dim as usize != axis || spec.multiplier != 1 || spec.size != source[axis]
            {
                return None;
            }
        } else if spec.multiplier != 0 {
            return None;
        }
    }
    Some(x)
}
fn inverse_statistic(b: &Builder<'_>, id: Id, x: Id, width: u64) -> Option<ScalarExpr> {
    let shape = b.facts_of(x).shape.clone();
    let rows = &shape[..shape.len() - 1];
    let axes: Vec<_> = (0..rows.len()).collect();
    let statistic = unbroadcast(b, id, rows, &axes)?;
    let Logical::Map { expr, ins, outs: 1 } = logical(b, statistic)? else {
        return None;
    };
    let ScalarKind::Un {
        op: UnOp::InverseSqrt,
        x: add,
    } = expr.kind()
    else {
        return None;
    };
    let ScalarKind::Bin {
        op: BinOp::Add,
        a,
        b: rhs,
    } = add.kind()
    else {
        return None;
    };
    let (mean, eps) = match (a.kind(), rhs.kind()) {
        (ScalarKind::Arg(i), ScalarKind::Uniform(_) | ScalarKind::Lit(_)) => {
            (*ins.get(*i as usize)?, rhs.clone())
        }
        (ScalarKind::Uniform(_) | ScalarKind::Lit(_), ScalarKind::Arg(i)) => {
            (*ins.get(*i as usize)?, a.clone())
        }
        _ => return None,
    };
    let Logical::Map { expr, ins, outs: 1 } = logical(b, mean)? else {
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
    let sum = match (a.kind(), rhs.kind()) {
        (ScalarKind::Arg(i), ScalarKind::Lit(Lit(Splat::F32(n))))
        | (ScalarKind::Lit(Lit(Splat::F32(n))), ScalarKind::Arg(i))
            if n.to_bits() == (1.0 / width as f32).to_bits() =>
        {
            *ins.get(*i as usize)?
        }
        _ => return None,
    };
    let Logical::Fold {
        carrier,
        axis,
        acc: Dtype::F32,
        ins,
    } = logical(b, sum)?
    else {
        return None;
    };
    if axis as usize != rows.len()
        || carrier != crate::carrier::Carrier::binop(BinOp::Add, Splat::F32(0.0), Dtype::F32)
        || carrier.lift.as_slice() != [ScalarExpr::arg(0, Dtype::F32)]
        || ins.len() != 1
    {
        return None;
    }
    let (a, c) = binary(b, ins[0], BinOp::Mul)?;
    (b.class_of(a) == b.class_of(x) && b.class_of(c) == b.class_of(x)).then_some(eps)
}
fn recognize(b: &mut Builder<'_>, id: Id, _node: &Node, _f: &Facts<'_>) -> Option<Id> {
    if b.caps().kind != DeviceKind::Gpu {
        return None;
    }
    let (a, c) = binary(b, id, BinOp::Mul)?;
    let shape = b.facts_of(id).shape.clone();
    let width = shape.last()?.as_const()?;
    let elements = crate::shape::const_elements(&shape)?;
    if width == 0 || elements == 0 || elements > u64::from(u32::MAX) {
        return None;
    }
    for (norm, weight) in [(a, c), (c, a)] {
        let Some(weight) = unbroadcast(b, weight, &[Dim::Const(width)], &[shape.len() - 1]) else {
            continue;
        };
        let Some((a, c)) = binary(b, norm, BinOp::Mul) else {
            continue;
        };
        for (x, inv) in [(a, c), (c, a)] {
            if b.facts_of(x).shape != shape {
                continue;
            }
            let Some(epsilon) = inverse_statistic(b, inv, x, width) else {
                continue;
            };
            if [id, x, weight].iter().any(|v| {
                let f = b.facts_of(*v);
                f.dtype != Dtype::F32 || !f.numeric.reassoc || !f.numeric.contract
            }) {
                continue;
            }
            let limits = &b.caps().limits;
            let max = limits
                .max_compute_invocations_per_workgroup
                .min(limits.max_compute_workgroup_size[0])
                .min(limits.max_compute_workgroup_storage_size / 4)
                .min(512);
            let block = (1 << max.checked_ilog2()?)
                .min((width.min(512) as u32).next_power_of_two().max(32));
            if block < 32
                || width.div_ceil(u64::from(block))
                    > u64::from(crate::ir::launch::MAX_NORMALIZATION_VALUES_PER_LANE)
            {
                return None;
            }
            let mut ops = vec![
                super::alias_operand_of(x, &shape),
                super::alias_operand_of(weight, &[Dim::Const(width)]),
            ];
            if let Some((left, right)) = binary(b, x, BinOp::Add)
                && [left, right].iter().all(|v| b.facts_of(*v).shape == shape)
            {
                ops[0] = super::alias_operand_of(left, &shape);
                ops.push(super::alias_operand_of(right, &shape));
            }
            let fused = b
                .add_launch(Launch::RmsNorm {
                    output: IndexSpace::new(shape),
                    width: width as u32,
                    block,
                    epsilon,
                    ops,
                    sched: ScheduleDomain::Point,
                })
                .ok()?;
            return b.union(id, fused).ok();
        }
    }
    None
}
