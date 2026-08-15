//! `with_backwards` — a user-facing escape hatch for custom gradients.
//!
//! [`fusor2_ir::autograd::GradientSlot`] is a bare node id, never a tensor
//! handle: a closure capturing a graph handle would close an `Arc` cycle
//! pinning every cached activation for the process lifetime. Here the rule
//! is a plain `fn` pointer, so the hazard is unrepresentable.

use fusor2_ir::autograd::{AdjointFn, BackwardTarget, GradientSlot, Grads, Parent, Tape, Val};
use fusor2_ir::ir::Node;
use fusor2_ir::{Error, Result};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// A user-supplied backward attached to one value: an explicit rule over the
/// node's declared parents.
#[derive(Clone, Debug)]
pub struct CustomBackward {
    pub parents: SmallVec<[Parent; 4]>,
    pub rule: AdjointFn,
}

impl CustomBackward {
    /// Run the rule and check that it covered every requires-grad parent.
    pub fn invoke(
        &self,
        tape: &mut dyn Tape,
        node: &Node,
        grad: Val,
        ins: &[Val],
        out: Val,
    ) -> Result<Grads> {
        let grads = (self.rule)(tape, node, grad, ins, out)?;
        let targets: SmallVec<[BackwardTarget; 4]> = ins
            .iter()
            .enumerate()
            .filter_map(|(slot, v)| {
                grads.get(slot).copied().flatten().map(|g| BackwardTarget {
                    slot: GradientSlot(*v),
                    gradient: g,
                })
            })
            .collect();
        validate_parents(&self.parents, &targets)?;
        Ok(grads)
    }
}

/// Side table of user-supplied backwards, keyed by the node they belong to.
/// Consulted by the reverse walk **before** [`crate::ADJOINTS`].
#[derive(Clone, Debug, Default)]
pub struct CustomRegistry {
    rules: FxHashMap<Val, CustomBackward>,
}

impl CustomRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, value: Val) -> Option<&CustomBackward> {
        self.rules.get(&value)
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Register `rule` for `value`. Returns the previous entry, if any.
    pub fn insert(&mut self, value: Val, entry: CustomBackward) -> Option<CustomBackward> {
        self.rules.insert(value, entry)
    }
}

/// Register a user-supplied backward for `value`, declaring its parents.
///
/// The rule is a bare `fn`; the gradients it returns are slot-aligned to the
/// node's operands, exactly as [`crate::ADJOINTS`] rules are. After it runs,
/// every `Parent { requires_grad: true }` must appear among its targets.
pub fn with_backwards(
    registry: &mut CustomRegistry,
    value: Val,
    parents: &[Parent],
    rule: AdjointFn,
) -> Result<Val> {
    registry.insert(
        value,
        CustomBackward {
            parents: parents.iter().copied().collect(),
            rule,
        },
    );
    Ok(value)
}

/// Every requires-grad parent must receive a gradient. A custom rule that
/// omits one is an error, not a silent zero: the omitted parent's whole
/// subgraph would starve, and the walk's final check would report the
/// symptom rather than the cause.
pub fn validate_parents(parents: &[Parent], targets: &[BackwardTarget]) -> Result<()> {
    for parent in parents {
        if !parent.requires_grad {
            continue;
        }
        if !targets.iter().any(|t| t.slot.0 == parent.value) {
            return Err(Error::Plan(format!(
                "custom backward omitted a gradient for a parent that requires grad: {}",
                parent.value
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backward::backward_into_with;
    use crate::tape::testing::{caps, graph};
    use crate::tape::{GraphTape, TapeExt};
    use fusor2_ir::dtype::{Dtype, RoundMode, Splat};
    use fusor2_ir::egraph::{EGraph, Id};
    use fusor2_ir::ir::Op;
    use fusor2_ir::ir::level0::{BufferId, L0, LeafKind};
    use fusor2_ir::scalar::{ScalarExpr, UnOp};
    use fusor2_ir::shape::Dim;

    fn param(g: &mut EGraph, shape: &[u64]) -> Id {
        let n = g.len() as u32;
        g.add(Op::L0(L0::Leaf(LeafKind::Param {
            name: BufferId(n),
            dtype: Dtype::F32,
            shape: shape.iter().map(|d| Dim::Const(*d)).collect(),
        })))
        .unwrap()
    }

    fn ones(g: &mut EGraph, shape: &[u64]) -> Id {
        g.add(Op::L0(L0::Leaf(LeafKind::Const {
            value: Splat::F32(1.0),
            shape: shape.iter().map(|d| Dim::Const(*d)).collect(),
        })))
        .unwrap()
    }

    /// A rule that routes twice the gradient into operand 0.
    fn doubling(
        tape: &mut dyn Tape,
        _node: &Node,
        grad: Val,
        ins: &[Val],
        _out: Val,
    ) -> Result<Grads> {
        let g = tape.mul_scalar(grad, 2.0)?;
        let mut out: Grads = SmallVec::new();
        for slot in 0..ins.len() {
            out.push(if slot == 0 { Some(g) } else { None });
        }
        Ok(out)
    }

    /// A rule that returns nothing at all.
    fn starving(
        _tape: &mut dyn Tape,
        _node: &Node,
        _grad: Val,
        ins: &[Val],
        _out: Val,
    ) -> Result<Grads> {
        Ok(smallvec::smallvec![None; ins.len()])
    }

    #[test]
    fn a_custom_rule_routes_through_a_gradient_slot() {
        let mut g = graph();
        let x = param(&mut g, &[3]);
        let y = {
            let mut t = GraphTape::new(&mut g);
            t.unary(UnOp::Exp, x).unwrap()
        };
        let mut reg = CustomRegistry::new();
        with_backwards(
            &mut reg,
            y,
            &[Parent {
                value: x,
                requires_grad: true,
            }],
            doubling,
        )
        .unwrap();
        let s = ones(&mut g, &[3]);
        let got = backward_into_with(&mut g, &caps(), y, s, &[x], &reg).unwrap();
        let dx = got[0].unwrap();
        // The custom rule wins over the `Map` adjoint: `2 * seed`, not
        // `seed * exp(x)`.
        assert!(matches!(g.node(dx).op, Op::L0(L0::Map { .. })));
        match &g.node(dx).op {
            Op::L0(L0::Map { ins, .. }) => assert_eq!(ins.as_slice(), &[s]),
            _ => unreachable!(),
        }
    }

    #[test]
    fn omitting_a_requires_grad_parent_is_an_error() {
        let mut g = graph();
        let x = param(&mut g, &[3]);
        let y = {
            let mut t = GraphTape::new(&mut g);
            t.unary(UnOp::Exp, x).unwrap()
        };
        let mut reg = CustomRegistry::new();
        with_backwards(
            &mut reg,
            y,
            &[Parent {
                value: x,
                requires_grad: true,
            }],
            starving,
        )
        .unwrap();
        let s = ones(&mut g, &[3]);
        let err = backward_into_with(&mut g, &caps(), y, s, &[x], &reg).unwrap_err();
        match err {
            Error::Plan(m) => assert!(m.contains("omitted a gradient")),
            other => panic!("expected Error::Plan, got {other:?}"),
        }
    }

    #[test]
    fn a_non_grad_parent_may_be_omitted() {
        let parents = [Parent {
            value: Id(0),
            requires_grad: false,
        }];
        validate_parents(&parents, &[]).unwrap();
    }

    #[test]
    fn without_the_attribute_a_round_gives_a_zero_constant() {
        let mut g = graph();
        let x = param(&mut g, &[3]);
        let y = {
            let mut t = GraphTape::new(&mut g);
            let body = ScalarExpr::round(
                RoundMode::HalfAwayFromZero,
                ScalarExpr::arg(0, Dtype::F32),
            );
            t.map(body, &[x]).unwrap()
        };
        let s = ones(&mut g, &[3]);
        let got =
            backward_into_with(&mut g, &caps(), y, s, &[x], &CustomRegistry::new()).unwrap();
        assert!(matches!(
            g.node(got[0].unwrap()).op,
            Op::L0(L0::Leaf(LeafKind::Const { .. }))
        ));
    }
}
