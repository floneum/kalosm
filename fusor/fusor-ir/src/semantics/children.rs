//! Operand ids of every `Op`, in the order inference, verification, work
//! accounting and the cost model all expect. The one place that order is
//! written down.

use crate::ir::launch::Launch;
use crate::ir::logical::Logical;
use crate::ir::{Children, Op};

/// Operand ids of `op`. `Op::Union(a, b)` yields `[a, b]`.
pub fn children_of(op: &Op) -> Children {
    match op {
        Op::Logical(o) => children_logical(o),
        Op::Launch(o) => children_launch(o),
        Op::Union(a, b) => Children::from_slice(&[*a, *b]),
    }
}

/// Operand ids of a Logical node.
pub fn children_logical(op: &Logical) -> Children {
    match op {
        Logical::Leaf(_) => Children::new(),
        Logical::Map { ins, .. } => ins.iter().copied().collect(),
        Logical::Fold { ins, .. } => ins.iter().copied().collect(),
        Logical::Contract { a, b, .. } => Children::from_slice(&[*a, *b]),
        Logical::Restride { x, .. } => Children::from_slice(&[*x]),
        Logical::Window { x, .. } => Children::from_slice(&[*x]),
        Logical::Gather { x, idx, .. } => Children::from_slice(&[*x, *idx]),
        Logical::Scatter { base, idx, upd, .. } => Children::from_slice(&[*base, *idx, *upd]),
        Logical::Dequant { x, .. } => Children::from_slice(&[*x]),
        Logical::Project { x, .. } => Children::from_slice(&[*x]),
    }
}

/// Operand ids of a Launch node, taken from its `Operand` lists. `Contract`
/// is its A-side operands followed by its B-side ones — one each in the
/// two-buffer case that reads `[a.src, b.src]`, more once a multi-edge
/// producer has been absorbed. A region is its members and a merged wave is
/// its segments.
pub fn children_launch(op: &Launch) -> Children {
    match op {
        Launch::Map { ops, .. }
        | Launch::Fold { ops, .. }
        | Launch::Gather { ops, .. }
        | Launch::Scatter { ops, .. }
        | Launch::Ext { ops, .. } => ops.iter().map(|o| o.src).collect(),
        Launch::Contract { a, b, .. } => a.ops.iter().chain(b.ops.iter()).map(|o| o.src).collect(),
        Launch::Region { members, .. } => members.iter().copied().collect(),
    }
}
