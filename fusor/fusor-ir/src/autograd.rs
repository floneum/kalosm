//! Reverse mode as a Logical -> Logical transform. Seven adjoint entries.

use crate::carrier::Carrier;
use crate::dtype::Dtype;
use crate::egraph::Id;
use crate::error::Result;
use crate::facts::ValueFacts;
use crate::ir::logical::{EinSpec, Logical};
use crate::ir::{Node, OpTag};
use crate::scalar::ScalarExpr;
use smallvec::SmallVec;

/// A value on the tape. Ids are e-graph ids: forward and backward are one
/// graph with one root set.
pub type Val = Id;

/// One gradient per parent, `None` where a parent does not require grad.
pub type Grads = SmallVec<[Option<Val>; 4]>;

/// The Logical construction surface an adjoint rule writes into. Object-safe:
/// [`AdjointFn`] takes `&mut dyn Tape`.
pub trait Tape {
    fn add(&mut self, op: Logical) -> Result<Val>;
    fn facts(&self, v: Val) -> &ValueFacts;
    fn zeros_like(&mut self, v: Val) -> Result<Val>;
    fn map(&mut self, expr: ScalarExpr, ins: &[Val]) -> Result<Val>;
    fn contract(&mut self, a: Val, b: Val, spec: EinSpec, acc: Dtype) -> Result<Val>;
    fn fold(&mut self, carrier: Carrier, axis: u32, acc: Dtype, x: Val) -> Result<Val>;

    /// A plain binary reduction — `Fold` at a single-slot [`Carrier`]. `Add`,
    /// `Mul`, `Max` and `Min` are all this.
    fn fold_binop(
        &mut self,
        op: crate::scalar::BinOp,
        axis: u32,
        acc: Dtype,
        x: Val,
    ) -> Result<Val> {
        let ident = Carrier::binop_identity(op, acc).ok_or_else(|| {
            crate::error::Error::Dtype(format!("{op:?} has no identity in {acc:?}"))
        })?;
        self.fold(Carrier::binop(op, ident, acc), axis, acc, x)
    }
    fn restride(&mut self, specs: &[crate::shape::StrideSpec], x: Val) -> Result<Val>;
    /// The declared adjoint of `Gather`. Four lowerings coexist below it;
    /// the cost model picks.
    fn scatter_add(&mut self, axis: u32, base: Val, idx: Val, upd: Val) -> Result<Val>;
    fn accumulate(&mut self, a: Val, b: Val) -> Result<Val>;
}

/// Signature of an analytic adjoint. Arguments: the tape, the primal node,
/// the incoming gradient, the primal inputs, and the primal output.
pub type AdjointFn = fn(&mut dyn Tape, &Node, Val, &[Val], Val) -> Result<Grads>;

/// How an op's reverse rule is obtained.
#[derive(Copy, Clone, Debug)]
pub enum AdjointKind {
    Analytic(AdjointFn),
    /// Derived from the op's own attributes. `Window`'s structural adjoint
    /// reads `(window, step)`: `step >= window` proves the adjoint is an
    /// elementwise mask-and-broadcast; overlapping windows give
    /// `Scatter{Add}`, itself a chain with four lowerings.
    Structural,
}

/// One row of the adjoint table.
#[derive(Copy, Clone, Debug)]
pub struct Adjoint {
    pub op: OpTag,
    pub kind: AdjointKind,
}

/// The whole reverse-mode transform. Object-safe.
pub trait Autograd: Send + Sync {
    /// The adjoint table. Seven entries.
    fn adjoints(&self) -> &'static [Adjoint];

    /// Build the backward graph for `root` with respect to `wrt`, seeded
    /// with `seed`. The result is ingested **together with** the forward as
    /// one graph with one root set.
    fn backward(
        &self,
        tape: &mut dyn Tape,
        root: Val,
        seed: Val,
        wrt: &[Val],
    ) -> Result<Vec<Option<Val>>>;
}

/// Where a user-supplied backward sends a gradient. A bare node id, never a
/// tensor handle: a closure capturing a graph handle would close an `Arc`
/// cycle pinning every cached activation for the process lifetime.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct GradientSlot(pub Val);

/// A parent declared by the user-facing escape hatch. A custom rule that
/// omits a requires-grad parent is an error.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Parent {
    pub value: Val,
    pub requires_grad: bool,
}

/// One gradient a custom backward emits.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BackwardTarget {
    pub slot: GradientSlot,
    pub gradient: Val,
}
