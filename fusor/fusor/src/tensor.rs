//! Runtime-rank tensor core: runtime dtype, one node id.
//!
//! # The surface this item consumes from `graph.rs` and `session.rs`
//!
//! Nothing else in `fusor/src/tensor*`, `fusor/src/ops/*` or
//! `fusor/src/broadcast.rs` touches the e-graph directly. Everything routes
//! through the opaque [`crate::graph::GraphRef`] handle:
//!
//! ```ignore
//! impl GraphRef {
//!     fn add_logical(&self, op: Logical) -> Result<Id>;
//!     fn facts(&self, id: Id) -> ValueFacts;
//!     fn tensor(&self, id: Id) -> Tensor;
//!     fn set_leaf_bytes(&self, id: Id, bytes: Vec<u8>);
//! }
//! ```
//!
//! Construction, graph state, and readback stay behind that handle.

pub(crate) mod construction;
pub(crate) mod readback;
pub(crate) mod typed;

use fusor_ir::dtype::{Dtype, Splat};
use fusor_ir::egraph::Id;
use fusor_ir::facts::ValueFacts;
use fusor_ir::ir::logical::{LeafKind, Logical};
use fusor_ir::scalar::ScalarExpr;
use fusor_ir::shape::{Dim, Dims, SymId};
use smallvec::SmallVec;

use crate::graph::GraphRef;
use crate::session::Backend;
use crate::{Error, Result};

pub use crate::ops::index::{IndexOp, TensorIndex, cat, stack};
pub use crate::ops::view::Extent;
pub use construction::{FromArray, arange, arange_step};
/// The rounding an explicit `round_mode` selects.
pub use fusor_ir::dtype::RoundMode;
pub use readback::{TensorSlice, ToVec};
pub use typed::{Axis, Element};

/// A value in a [`crate::Graph`] with **runtime** rank and dtype. Cloning is
/// one `Arc` bump; the node it names is immutable.
///
/// [`crate::Tensor`] — the const-rank, infallible facade — is a
/// `repr(transparent)` newtype over this. Reach for `Dyn` (via
/// [`crate::Tensor::into_dyn`] / [`crate::Tensor::as_dyn`]) when a rank or a
/// dtype is *data*: a loader that reads it from a file, a pass that walks a
/// heterogeneous list. Every op on it returns `Result`.
#[derive(Clone)]
pub struct Dyn {
    pub(crate) id: Id,
    pub(crate) graph: GraphRef,
}

/// The in-crate spelling of [`Dyn`].
pub(crate) type Tensor = Dyn;

impl Tensor {
    /// The e-graph id this tensor names. Public so conformance can assert
    /// against graph structure directly.
    pub fn id(&self) -> Id {
        self.id
    }

    /// The graph this value lives in.
    pub fn graph(&self) -> &GraphRef {
        &self.graph
    }

    /// Which backend the owning session runs on.
    pub fn backend(&self) -> Backend {
        self.graph.session().device().clone()
    }

    /// This value's inference result, cloned out of the graph. Inference ran
    /// when the node was minted; nothing here recomputes a shape.
    pub fn facts(&self) -> ValueFacts {
        self.graph.facts(self.id)
    }

    /// Runtime element type.
    pub fn dtype(&self) -> Dtype {
        self.graph.facts(self.id).dtype
    }

    /// Runtime rank. There is no rank ceiling.
    pub fn rank(&self) -> usize {
        self.graph.facts(self.id).shape.len()
    }

    /// The value's extents.
    pub fn shape(&self) -> Dims {
        self.graph.facts(self.id).shape
    }

    /// Extent of axis `i`.
    ///
    /// # Panics
    /// If `i >= self.rank()`.
    pub fn dim(&self, i: usize) -> Dim {
        let shape = self.shape();
        match shape.get(i) {
            Some(d) => *d,
            None => panic!("axis {i} out of range for rank {}", shape.len()),
        }
    }

    /// Element count, or `None` when any extent is symbolic.
    pub fn elem_count(&self) -> Option<u64> {
        self.facts().elements()
    }

    /// Materialize and re-leaf, cutting this value off from its producers.
    ///
    /// Expensive: it resolves, reads the bytes back to the host and uploads
    /// them into a fresh `Leaf::Buffer`.
    pub fn detach(&self) -> Result<Tensor> {
        let facts = self.facts();
        let bytes = self.graph.read_back(self.id)?;
        construction::upload(&self.graph, facts.dtype, &facts.shape, bytes)
    }

    /// Attach host bytes to this external leaf, invalidating any device copy.
    /// The next resolve re-uploads. This is the decode loop's per-step input
    /// path: the leaf node (and so the graph) is unchanged, only its bytes
    /// move.
    pub fn set_bytes(&self, bytes: Vec<u8>) -> Result<()> {
        if !self.is_external_leaf() {
            return Err(Error::Plan(
                "set_bytes targets an external leaf; this value is computed".into(),
            ));
        }
        let facts = self.facts();
        if let Some(elements) = facts.elements() {
            let expect = (elements * facts.dtype.byte_size()) as usize;
            if bytes.len() != expect {
                return Err(Error::Shape(format!(
                    "set_bytes got {} bytes for a {expect}-byte leaf",
                    bytes.len()
                )));
            }
        }
        self.graph.set_leaf_bytes(self.id, bytes);
        Ok(())
    }

    /// The device-side detach: rebind this external leaf to the device buffer
    /// `from` resolved into, without any host round trip.
    ///
    /// The decode loop's KV convention: the step graph reads leaf `K`,
    /// produces `K' = scatter(K, ..)`, and after the step the leaf adopts
    /// `K'`'s buffer so the *same* graph runs the next step. Requires
    /// matching dtype and shape and a resolved `from`.
    pub fn adopt_buffer(&self, from: &Tensor) -> Result<()> {
        if !self.is_external_leaf() {
            return Err(Error::Plan(
                "adopt_buffer targets an external leaf; this value is computed".into(),
            ));
        }
        let (mine, theirs) = (self.facts(), from.facts());
        if mine.dtype != theirs.dtype {
            return Err(Error::Dtype(format!(
                "adopt_buffer dtype mismatch: {:?} vs {:?}",
                mine.dtype, theirs.dtype
            )));
        }
        if mine.shape.len() != theirs.shape.len()
            || mine
                .shape
                .iter()
                .zip(theirs.shape.iter())
                .any(|(a, b)| !a.known_eq(*b))
        {
            return Err(Error::Shape(format!(
                "adopt_buffer shape mismatch: {:?} vs {:?}",
                mine.shape, theirs.shape
            )));
        }
        let buf = from.graph.device_buf(from.id).ok_or_else(|| {
            Error::Plan("adopt_buffer needs a resolved source; resolve it first".into())
        })?;
        let layout = from.graph.device_layout(from.id).map(std::sync::Arc::new);
        self.graph
            .set_device_buf_class(&[self.id], &buf, layout.as_ref());
        Ok(())
    }

    /// Drop the device buffer bound to this value's class so the next resolve
    /// re-dispatches it. See [`crate::graph::GraphRef`]'s
    /// `clear_class_device_buf`.
    pub fn clear_device_buf(&self) {
        self.graph.clear_class_device_buf(self.id);
    }

    /// Whether this value is an external leaf (`Buffer`/`Param`/`Quantized`).
    pub fn is_external_leaf(&self) -> bool {
        self.graph
            .with_egraph(|g| {
                Ok(matches!(
                    &g.node(self.id).op,
                    fusor_ir::ir::Op::Logical(Logical::Leaf(
                        LeafKind::Buffer { .. }
                            | LeafKind::Param { .. }
                            | LeafKind::Quantized { .. }
                    ))
                ))
            })
            .unwrap_or(false)
    }

    /// Mint one Logical node and wrap it. The single call site for `add_logical` in
    /// this item.
    pub(crate) fn emit(graph: &GraphRef, op: Logical) -> Result<Tensor> {
        let id = graph.add_logical(op)?;
        Ok(graph.tensor(id))
    }

    /// [`Tensor::emit`] into this value's own graph.
    pub(crate) fn emit_here(&self, op: Logical) -> Result<Tensor> {
        Self::emit(&self.graph, op)
    }

    /// One `Logical::Map` over `self`, `outs: 1`. `expr` reads the operand as
    /// `Arg(0)`.
    pub(crate) fn map1(&self, expr: ScalarExpr) -> Result<Tensor> {
        self.emit_here(Logical::Map {
            expr,
            ins: SmallVec::from_slice(&[self.id]),
            outs: 1,
        })
    }

    /// One `Logical::Map` over several operands, `outs: 1`.
    ///
    /// **Rejects** operands whose shapes are not pointwise [`Dim::known_eq`]:
    /// there is no implicit broadcasting inside the IR, callers pre-broadcast
    /// with [`crate::broadcast::broadcast_pair`].
    pub(crate) fn mapn(g: &GraphRef, expr: ScalarExpr, ins: &[&Tensor]) -> Result<Tensor> {
        let Some(first) = ins.first() else {
            return Err(Error::Shape("Map needs at least one operand".into()));
        };
        let want = first.shape();
        for other in &ins[1..] {
            let got = other.shape();
            if !dims_eq(&want, &got) {
                return Err(Error::Shape(format!(
                    "elementwise operands must have identical shape: {want:?} vs {got:?}; \
                     use the broadcasting form (add_/sub_/mul_/div_/pow_) or broadcast_as"
                )));
            }
        }
        Self::emit(
            g,
            Logical::Map {
                expr,
                ins: ins.iter().map(|t| t.id).collect(),
                outs: 1,
            },
        )
    }

    /// `Arg(i)` typed as this tensor's dtype.
    pub(crate) fn arg(&self, i: u32) -> ScalarExpr {
        ScalarExpr::arg(i, self.dtype())
    }

    /// `Arg(0)` typed as this tensor's dtype.
    pub(crate) fn arg0(&self) -> ScalarExpr {
        self.arg(0)
    }

    /// Reject a quantized operand at an op that has no quantized form.
    pub(crate) fn require_dense(&self, what: &str) -> Result<()> {
        if self.dtype().is_quantized() {
            return Err(Error::Dtype(format!(
                "{what} is not defined on the quantized dtype {:?}; dequantize first",
                self.dtype()
            )));
        }
        Ok(())
    }

    /// Range-check an axis argument.
    pub(crate) fn check_axis(&self, axis: usize, what: &str) -> Result<()> {
        if axis >= self.rank() {
            return Err(Error::Shape(format!(
                "{what}: axis {axis} out of range for rank {}",
                self.rank()
            )));
        }
        Ok(())
    }
}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let facts = self.facts();
        write!(f, "Tensor({} {:?} {:?})", self.id, facts.dtype, facts.shape)
    }
}

/// Pointwise decidable shape equality.
pub(crate) fn dims_eq(a: &[Dim], b: &[Dim]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.known_eq(*y))
}

/// A scalar operand of a scalar-arith or comparison op.
///
/// `Scalar::Uniform` reads the value out of the uniform block and never bakes
/// a literal into a kernel, so changing it recompiles nothing. A `Lit` folds
/// to a literal, which lands in the kernel key.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Scalar {
    /// A compile-time literal that participates in the kernel key.
    Lit(Splat),
    /// A runtime uniform that can change without recompilation.
    Uniform(SymId),
}

impl Scalar {
    /// The `ScalarExpr` leaf this scalar contributes, converted to `dt`.
    pub fn expr(self, dt: Dtype) -> ScalarExpr {
        match self {
            Self::Lit(v) => ScalarExpr::lit(splat_as(v, dt)),
            Self::Uniform(s) => ScalarExpr::uniform(s, dt),
        }
    }

    /// The literal value as `f64`, or `None` for a uniform.
    pub fn as_f64(self) -> Option<f64> {
        match self {
            Self::Lit(v) => Some(splat_f64(v)),
            Self::Uniform(_) => None,
        }
    }
}

impl From<f32> for Scalar {
    fn from(v: f32) -> Self {
        Self::Lit(Splat::F32(v))
    }
}
// NOTE: no `From<f64>`. With exactly one float impl, an unsuffixed literal
// unifies to `f32`, so `t.mul_scalar(2.0)` compiles; adding `From<f64>` makes
// every such call ambiguous.
impl From<half::f16> for Scalar {
    fn from(v: half::f16) -> Self {
        Self::Lit(Splat::F16(v.to_bits()))
    }
}
impl From<half::bf16> for Scalar {
    fn from(v: half::bf16) -> Self {
        Self::Lit(Splat::BF16(v.to_bits()))
    }
}
impl From<u32> for Scalar {
    fn from(v: u32) -> Self {
        Self::Lit(Splat::U32(v))
    }
}
impl From<i32> for Scalar {
    fn from(v: i32) -> Self {
        Self::Lit(Splat::I32(v))
    }
}
impl From<Splat> for Scalar {
    fn from(v: Splat) -> Self {
        Self::Lit(v)
    }
}
impl From<SymId> for Scalar {
    fn from(v: SymId) -> Self {
        Self::Uniform(v)
    }
}

/// A `Splat`'s value as `f64`. Exact for every dtype fusor has.
pub(crate) fn splat_f64(v: Splat) -> f64 {
    match v {
        Splat::F32(x) => x as f64,
        Splat::F16(bits) => half::f16::from_bits(bits).to_f64(),
        Splat::BF16(bits) => half::bf16::from_bits(bits).to_f64(),
        Splat::U32(x) => x as f64,
        Splat::I32(x) => x as f64,
    }
}

/// Retype a literal. A quantized target has no scalar literal form, so the
/// value passes through unchanged and the surrounding op rejects it.
pub(crate) fn splat_as(v: Splat, dt: Dtype) -> Splat {
    if v.dtype() == dt {
        return v;
    }
    let x = splat_f64(v);
    match dt {
        Dtype::F32 => Splat::F32(x as f32),
        Dtype::F16 => Splat::F16(half::f16::from_f64(x).to_bits()),
        Dtype::BF16 => Splat::BF16(half::bf16::from_f64(x).to_bits()),
        Dtype::U32 => Splat::U32(x as u32),
        Dtype::I32 => Splat::I32(x as i32),
        Dtype::Q(_) => v,
    }
}

/// The additive identity in `dt`.
pub(crate) fn splat_zero(dt: Dtype) -> Splat {
    splat_as(Splat::F32(0.0), dt)
}

/// The multiplicative identity in `dt`.
pub(crate) fn splat_one(dt: Dtype) -> Splat {
    splat_as(Splat::F32(1.0), dt)
}

// Graph-level acceptance tests: every assertion is against the built Logical
// term, not execution — this item mints nodes and has no backend.
