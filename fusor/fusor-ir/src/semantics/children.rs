//! Operand ids of every `Op`, in the one order inference, verification,
//! work and cost all expect.

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

/// Operand ids of a Launch node, in [`Launch::operands`] order; a composite
/// names its members.
pub fn children_launch(op: &Launch) -> Children {
    match op {
        Launch::Slab { members, .. } | Launch::Group { members, .. } => {
            members.iter().copied().collect()
        }
        _ => op.operands().map(|o| o.src).collect(),
    }
}
