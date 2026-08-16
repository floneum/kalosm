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
