//! The adjoint table.
//!
//! `map_adjoint` differentiates a `ScalarExpr`, covering every elementwise
//! unary, comparison (zero gradient), `where_cond`, `clamp`, `gelu`,
//! `sigmoid`, `silu` and the scalar-arith family.
//!
//! Macro ops (`conv`, `rms_norm`, `attention`, `q_mat_mul`, ...) need no
//! entry: their `defn` expansion into core Logical means their adjoints
//! compose from core adjoints.

use fusor2_ir::autograd::{Adjoint, AdjointKind};
use fusor2_ir::ir::OpTag;

/// One row per differentiable core op.
pub static ADJOINTS: &[Adjoint] = &[
    Adjoint {
        op: OpTag::Contract,
        kind: AdjointKind::Analytic(crate::contract::contract_adjoint),
    },
    Adjoint {
        op: OpTag::Map,
        kind: AdjointKind::Analytic(crate::map_adjoint::map_adjoint),
    },
    Adjoint {
        op: OpTag::Restride,
        kind: AdjointKind::Structural,
    },
    Adjoint {
        op: OpTag::Window,
        kind: AdjointKind::Structural,
    },
    Adjoint {
        op: OpTag::Gather,
        kind: AdjointKind::Structural,
    },
    Adjoint {
        op: OpTag::Scatter,
        kind: AdjointKind::Structural,
    },
    Adjoint {
        op: OpTag::Fold,
        kind: AdjointKind::Structural,
    },
];

/// The row for `tag`, or `None` when the op is not differentiable (which for
/// a requires-grad parent is an error, not a silent zero).
///
/// `Leaf` terminates; `Project { slot }` routes the gradient into slot `slot`
/// of its tuple parent, which the walk does directly; `Dequant`'s input is a
/// quantized leaf that is never trainable.
pub(crate) fn adjoint_of(tag: OpTag) -> Option<&'static Adjoint> {
    ADJOINTS.iter().find(|a| a.op == tag)
}
