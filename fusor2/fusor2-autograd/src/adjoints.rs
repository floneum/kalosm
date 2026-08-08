//! The adjoint table: one row per differentiable core op.
//!
//! `map_adjoint` differentiates a `ScalarExpr`, covering the elementwise
//! unaries, the comparisons (which differentiate to zero), `where_cond`,
//! `clamp`, `gelu`, `sigmoid`, `silu` and the scalar-arith family.
//!
//! Macro ops such as `conv`, `rms_norm`, `rope`, `attention` and `q_mat_mul`
//! have no row: their `defn` expands into core L0, so their adjoints are the
//! composition of core adjoints.

use fusor2_ir::autograd::{Adjoint, AdjointKind};
use fusor2_ir::ir::{OpDefId, OpDefRegistry, OpTag};

/// The differentiable core ops and how each is differentiated.
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

/// The row for `tag`, or `None` when the op is not differentiable, which for
/// a requires-grad parent is an error rather than a silent zero.
///
/// `Leaf`, `Project` and `Dequant` have no row: `Leaf` terminates, `Project`
/// routing is done directly by the walk, and a `Dequant` input is a quantized
/// leaf that is never trainable.
pub fn adjoint_of(tag: OpTag) -> Option<&'static Adjoint> {
    ADJOINTS.iter().find(|a| a.op == tag)
}

/// [`adjoint_of`], falling through to [`fusor2_ir::ir::OpDef::adjoint`] for
/// the one open extension point.
pub fn adjoint_for(
    tag: OpTag,
    registry: &OpDefRegistry,
    def: Option<OpDefId>,
) -> Option<AdjointKind> {
    if let Some(row) = adjoint_of(tag) {
        return Some(row.kind);
    }
    if tag == OpTag::Ext {
        return registry.get(def?)?.adjoint;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_has_exactly_seven_rows() {
        assert_eq!(ADJOINTS.len(), 7);
    }

    #[test]
    fn the_seven_rows_are_the_seven_differentiable_core_ops() {
        let mut tags: Vec<OpTag> = ADJOINTS.iter().map(|a| a.op).collect();
        tags.sort();
        let mut want = vec![
            OpTag::Contract,
            OpTag::Map,
            OpTag::Restride,
            OpTag::Window,
            OpTag::Gather,
            OpTag::Scatter,
            OpTag::Fold,
        ];
        want.sort();
        assert_eq!(tags, want);
    }

    #[test]
    fn leaf_project_and_dequant_carry_no_row() {
        for tag in [OpTag::Leaf, OpTag::Project, OpTag::Dequant] {
            assert!(adjoint_of(tag).is_none(), "{tag:?} must not have a row");
        }
    }

    #[test]
    fn contract_and_map_are_analytic_and_the_rest_structural() {
        for row in ADJOINTS {
            match row.op {
                OpTag::Contract | OpTag::Map => {
                    assert!(matches!(row.kind, AdjointKind::Analytic(_)))
                }
                _ => assert!(matches!(row.kind, AdjointKind::Structural)),
            }
        }
    }

    #[test]
    fn an_unregistered_extension_has_no_adjoint() {
        let registry = OpDefRegistry::new();
        assert!(adjoint_for(OpTag::Ext, &registry, Some(OpDefId(0))).is_none());
        assert!(adjoint_for(OpTag::Ext, &registry, None).is_none());
    }
}
