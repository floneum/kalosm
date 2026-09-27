//! `with_backwards` — a user-facing escape hatch for custom gradients.
//!
//! [`fusor_ir::autograd::GradientSlot`] is a bare node id, never a tensor
//! handle: a closure capturing a graph handle would close an `Arc` cycle
//! pinning every cached activation for the process lifetime. Here the rule
//! is a plain `fn` pointer, so the hazard is unrepresentable.

use fusor_ir::autograd::{AdjointFn, Grads, Parent, Tape, Val};
use fusor_ir::ir::Node;
use fusor_ir::{Error, Result};
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
        validate_parents(&self.parents, |p| {
            ins.iter().zip(&grads).any(|(v, g)| *v == p && g.is_some())
        })?;
        Ok(grads)
    }
}

/// User-supplied adjoints, consulted before the built-in adjoint table.
pub type CustomRegistry = FxHashMap<Val, CustomBackward>;

/// Every requires-grad parent must receive a gradient; `covered` says which
/// did. A rule that omits one is an error, not a silent zero: the omitted
/// parent's whole subgraph would starve, and the walk's final check would
/// report the symptom rather than the cause.
pub fn validate_parents(parents: &[Parent], covered: impl Fn(Val) -> bool) -> Result<()> {
    match parents.iter().find(|p| p.requires_grad && !covered(p.value)) {
        Some(p) => Err(Error::Plan(format!(
            "a custom backward rule returned no gradient for parent {}, which requires one",
            p.value
        ))),
        None => Ok(()),
    }
}
