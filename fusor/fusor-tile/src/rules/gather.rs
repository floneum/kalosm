//! The two gather lowerings. `index_select`, `embedding`, `gather_last`
//! and `i()` are all one `Logical::Gather`, so they share these two alternatives.

use fusor_ir::egraph::{Builder, Facts, Id, RuleTag};
use fusor_ir::ir::launch::{AccessPlan, GatherMode, IndexSpace, Launch, Operand, ScheduleDomain};
use fusor_ir::ir::logical::Logical;
use fusor_ir::ir::{Level, Node, Op, OpTag};
use fusor_ir::rule;
use fusor_ir::shape::Dim;

use crate::domains::{DomainCtx, default_planner, map_domain};
use crate::rules::adopt;
use crate::rules::contract::alias;

rule!(
    GATHER_ROW_PER_GROUP,
    level = Level::Logical,
    head = OpTag::Gather,
    tag = RuleTag::StrictlyLowering,
    apply = gather_row_per_group,
);

rule!(
    GATHER_QUANTIZED_ROWS,
    level = Level::Logical,
    head = OpTag::Gather,
    tag = RuleTag::StrictlyLowering,
    apply = gather_quantized_rows,
);

fn parts(node: &Node) -> Option<(u32, Id, Id)> {
    match &node.op {
        Op::Logical(Logical::Gather { axis, x, idx }) => Some((*axis, *x, *idx)),
        _ => None,
    }
}

fn mint(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>, mode: GatherMode) -> Option<Id> {
    let (axis, x_id, idx_id) = parts(node)?;
    let x = alias(x_id, f.operand(0)?);
    gather(b, id, f, axis, x, idx_id, mode)
}

/// A `Gather` of `x` at `idx_id`'s indices over this node's output space.
fn gather(
    b: &mut Builder<'_>,
    id: Id,
    f: &Facts<'_>,
    axis: u32,
    x: Operand,
    idx_id: Id,
    mode: GatherMode,
) -> Option<Id> {
    let idx = alias(idx_id, f.operand(1)?);
    let out: Vec<Dim> = f.own().shape.iter().copied().collect();
    let cx = DomainCtx::new(f.caps(), default_planner());
    let accesses = [x.access.clone(), idx.access.clone()];
    let op = Launch::Gather {
        space: IndexSpace::new(out.iter().copied()),
        axis,
        mode,
        ops: vec![x, idx],
        sched: ScheduleDomain::Map(map_domain(&out, &accesses, &cx).into()),
    };
    adopt(b, id, op)
}

/// One workgroup per gathered row. The universal form.
pub fn gather_row_per_group(b: &mut Builder<'_>, id: Id, node: &Node, f: &Facts<'_>) -> Option<Id> {
    parts(node)?;
    if f.operand(0)?.dtype.is_quantized() {
        return None;
    }
    mint(b, id, node, f, GatherMode::RowPerGroup)
}

/// `Gather(Dequant(q), idx)` fused: the loaders decode the quantized leaf at
/// the flat index, so only gathered rows decode. Matched on the pair, so no
/// consuming `Dequant` is left to decode twice.
pub fn gather_quantized_rows(
    b: &mut Builder<'_>,
    id: Id,
    node: &Node,
    f: &Facts<'_>,
) -> Option<Id> {
    let (axis, x_id, idx_id) = parts(node)?;
    // The source must be the dequantized leaf's own class; walk the union spine,
    // since a rule sees whichever id the frontend recorded.
    let mut leaf: Option<Id> = None;
    let mut stack: Vec<Id> = vec![x_id];
    let mut seen: Vec<Id> = Vec::new();
    while let Some(cur) = stack.pop() {
        if seen.contains(&cur) {
            continue;
        }
        seen.push(cur);
        match &b.node(cur).op {
            Op::Union(l, r) => {
                stack.push(*l);
                stack.push(*r);
            }
            Op::Logical(Logical::Dequant { x, .. }) if b.facts_of(*x).dtype.is_quantized() => {
                leaf = Some(*x);
                break;
            }
            _ => {}
        }
    }
    let leaf = leaf?;
    // Laid out over the dense space the decode-at-index loaders address.
    let x = Operand {
        src: leaf,
        layout: fusor_ir::shape::Layout::contiguous(&f.operand(0)?.shape),
        access: AccessPlan::Alias,
    };
    gather(b, id, f, axis, x, idx_id, GatherMode::QuantizedRows)
}
