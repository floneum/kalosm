//! The one tensor type: runtime rank, runtime dtype, one node id.
//!
//! # The surface this item consumes from `graph.rs` and `session.rs`
//!
//! Nothing else in `fusor2/src/tensor*`, `fusor2/src/ops/*` or
//! `fusor2/src/broadcast.rs` touches the e-graph directly. Everything routes
//! through these four `GraphInner` methods:
//!
//! ```ignore
//! impl GraphInner {
//!     pub fn add_l0(&self, op: L0) -> Result<Id>;         // hash-cons + infer
//!     pub fn facts(&self, id: Id) -> ValueFacts;          // cloned; total
//!     pub fn tensor(self: &Arc<Self>, id: Id) -> Tensor;  // wrap an id
//!     pub fn set_leaf_bytes(&self, id: Id, bytes: Vec<u8>);
//! }
//! ```
//!
//! plus `GraphInner::{session, fresh_sym, read_back}`. [`Tensor::emit`] is the
//! **only** place a node is minted inside this item.

pub mod construction;
pub mod readback;
pub mod typed;

use fusor2_ir::dtype::{Dtype, NumericContract, Persistence, Splat};
use fusor2_ir::egraph::Id;
use fusor2_ir::facts::ValueFacts;
use fusor2_ir::ir::level0::{L0, LeafKind};
use fusor2_ir::scalar::ScalarExpr;
use fusor2_ir::shape::{Dim, Dims, SymId};
use smallvec::SmallVec;

use crate::graph::GraphRef;
use crate::session::Backend;
use crate::{Error, Result};

pub use crate::ops::index::{IndexOp, TensorIndex, cat, stack};
pub use crate::ops::view::Extent;
pub use construction::{FromArray, arange, arange_step};
pub use readback::{TensorSlice, ToVec};
pub use typed::{Axis, Element, SimdElement, Typed};
/// The rounding an explicit `round_mode` selects. Off the crate root: it is
/// the argument of exactly one op, and the root is for what a model spells.
pub use fusor2_ir::dtype::RoundMode;

/// A value in a [`crate::Graph`] with **runtime** rank and dtype. Cloning is
/// one `Arc` bump; the node it names is immutable.
///
/// This is the escape hatch, not the headline type: [`crate::Tensor`] — the
/// const-rank, infallible facade — is what a model is written in, and it is a
/// `repr(transparent)` newtype over this. Reach for `Dyn` (via
/// [`crate::Tensor::into_dyn`] / [`crate::Tensor::as_dyn`]) exactly when a
/// rank or a dtype is *data*: a loader that reads it from a file, a pass that
/// walks a heterogeneous list. Every op on it returns `Result`, because at
/// that layer a shape error genuinely is a runtime condition.
///
/// There is no `const R: usize`, no `B: Fusion`, and no dtype type parameter:
/// rank and dtype are runtime data that `verify_l0` checks.
#[derive(Clone)]
pub struct Dyn {
    pub(crate) id: Id,
    pub(crate) graph: GraphRef,
}

/// The in-crate spelling of [`Dyn`].
///
/// Every module below this one was written against the name `Tensor` when
/// there was only one tensor type. The public name is `Dyn`; this alias keeps
/// the ~40 internal `use crate::tensor::Tensor` lines meaning what they always
/// meant, and being `pub(crate)` it puts nothing back on the public surface.
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
    ///
    /// The *device* — backend plus session plus graph, the thing a constructor
    /// takes — is [`crate::Tensor::device`] on the const-rank facade. This is
    /// the selector underneath it.
    pub fn backend(&self) -> Backend {
        self.graph.session().device().clone()
    }

    /// This value's inference result, cloned out of the graph.
    /// `CoreSemantics::infer` is total and already ran, when the node was
    /// minted; nothing here recomputes a shape.
    pub fn facts(&self) -> ValueFacts {
        self.graph.facts(self.id)
    }

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
    /// If `i >= self.rank()`. Axis arguments are program structure, not data;
    /// every op in this crate range-checks before calling.
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

    /// Alternate spelling of [`Tensor::elem_count`], kept because the scaffold
    /// declared it.
    pub fn elements(&self) -> Option<u64> {
        self.elem_count()
    }

    pub fn numeric(&self) -> NumericContract {
        self.facts().numeric
    }

    pub fn persistence(&self) -> Persistence {
        self.facts().persistence
    }

    /// Rank 0 is a first-class rank: it is what a loss lives in.
    pub fn is_scalar(&self) -> bool {
        self.rank() == 0
    }

    /// Materialize and re-leaf, cutting this value off from its producers.
    ///
    /// Correct but expensive: it resolves, reads the bytes back to the host
    /// and uploads them into a fresh `Leaf::Buffer`. A device-side detach
    /// wants a session-level "adopt this buffer as a leaf" hook that does not
    /// exist yet; see the crate report.
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
                    fusor2_ir::ir::Op::L0(L0::Leaf(
                        LeafKind::Buffer { .. } | LeafKind::Param { .. } | LeafKind::Quantized { .. }
                    ))
                ))
            })
            .unwrap_or(false)
    }

    /// Wrap in the compile-time-rank facade. Purely a type-level assertion —
    /// the IR is unchanged.
    pub fn typed<const R: usize, D: Element>(self) -> Result<Typed<R, D>> {
        Typed::try_from_dyn(self)
    }

    // -- node minting -------------------------------------------------------

    /// Mint one L0 node and wrap it. The single call site for `add_l0` in
    /// this item.
    pub(crate) fn emit(graph: &GraphRef, op: L0) -> Result<Tensor> {
        let id = graph.add_l0(op)?;
        Ok(graph.tensor(id))
    }

    /// [`Tensor::emit`] into this value's own graph.
    pub(crate) fn emit_here(&self, op: L0) -> Result<Tensor> {
        Self::emit(&self.graph, op)
    }

    /// One `L0::Map` over `self`, `outs: 1`. `expr` reads the operand as
    /// `Arg(0)`.
    pub(crate) fn map1(&self, expr: ScalarExpr) -> Result<Tensor> {
        self.emit_here(L0::Map {
            expr,
            ins: SmallVec::from_slice(&[self.id]),
            outs: 1,
        })
    }

    /// One `L0::Map` over several operands, `outs: 1`.
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
            L0::Map {
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

// ---------------------------------------------------------------------------
// Scalar
// ---------------------------------------------------------------------------

/// A scalar operand of a scalar-arith or comparison op.
///
/// This single type is what deletes the trainer's `[1]`-tensor workaround:
/// `m.mul_scalar(lr)` with `lr: Scalar::Uniform(sym)` reads the learning rate
/// out of the uniform block and **never bakes a literal into a kernel**, so
/// changing it recompiles nothing. `m.mul_scalar(2.0f32)` still folds to a
/// literal, because a structural constant belongs in the kernel key.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Scalar {
    Lit(Splat),
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
// NOTE: deliberately no `From<f64>`. With exactly one float impl, trait
// selection unifies an unsuffixed literal to `f32`, so `t.mul_scalar(2.0)`
// compiles without a suffix; adding `From<f64>` makes every such call
// ambiguous.
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

/// A `Splat`'s value as `f64`. Exact for every dtype fusor2 has.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_conversions_round_trip() {
        assert_eq!(Scalar::from(2.0f32).as_f64(), Some(2.0));
        assert_eq!(Scalar::from(3u32).as_f64(), Some(3.0));
        assert_eq!(Scalar::from(-4i32).as_f64(), Some(-4.0));
        assert_eq!(Scalar::from(half::f16::from_f32(0.5)).as_f64(), Some(0.5));
        assert_eq!(Scalar::Uniform(SymId(7)).as_f64(), None);
    }

    #[test]
    fn scalar_expr_adopts_the_operand_dtype() {
        use fusor2_ir::scalar::ScalarKind;
        let e = Scalar::from(2.0f32).expr(Dtype::F16);
        assert_eq!(e.dtype(), Dtype::F16);
        assert!(matches!(e.kind(), ScalarKind::Lit(_)));

        let u = Scalar::Uniform(SymId(3)).expr(Dtype::F32);
        assert!(matches!(u.kind(), ScalarKind::Uniform(SymId(3))));
    }

    #[test]
    fn splat_helpers() {
        assert_eq!(splat_zero(Dtype::U32), Splat::U32(0));
        assert_eq!(splat_one(Dtype::I32), Splat::I32(1));
        assert_eq!(splat_one(Dtype::F16), Splat::F16(half::f16::ONE.to_bits()));
    }

    #[test]
    fn dims_eq_is_decidable_only() {
        let a = [Dim::Const(2), Dim::Sym(SymId(0))];
        let b = [Dim::Const(2), Dim::Sym(SymId(0))];
        let c = [Dim::Const(2), Dim::Sym(SymId(1))];
        assert!(dims_eq(&a, &b));
        assert!(!dims_eq(&a, &c));
    }
}

// ---------------------------------------------------------------------------
// Graph-level acceptance tests.
//
// Every assertion here is against the **built L0 term**, not against
// execution: this item mints nodes and has no backend.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod graph_tests {
    use super::*;
    use crate::graph::Graph;
    use crate::ops::index::{IndexOp, cat, stack};
    use crate::ops::view::Extent;
    use crate::session::{Backend, Session};
    use fusor2_ir::ir::level0::{L0, LeafKind, TiePolicy};
    use fusor2_ir::ir::{Op, OpTag};
    use fusor2_ir::shape::{Dim, Layout, SlidingWindow, StrideSpec};

    fn graph() -> Graph {
        let session = Session::new(Backend::cpu().expect("cpu device")).expect("session");
        Graph::new(&session)
    }

    fn dims(v: &[u64]) -> Vec<Dim> {
        v.iter().map(|&d| Dim::Const(d)).collect()
    }

    fn leaf(g: &Graph, shape: &[u64]) -> Tensor {
        g.leaf("x", &dims(shape), Dtype::F32).unwrap()
    }

    fn leaf_of(g: &Graph, shape: &[u64], dt: Dtype) -> Tensor {
        g.leaf("x", &dims(shape), dt).unwrap()
    }

    fn node_count(g: &Graph) -> usize {
        g.handle().with_egraph(|e| Ok(e.len())).unwrap()
    }

    fn tag_of(t: &Tensor) -> OpTag {
        t.graph()
            .with_egraph(|e| Ok(e.node(t.id()).op.tag()))
            .unwrap()
    }

    fn op_of(t: &Tensor) -> Op {
        t.graph()
            .with_egraph(|e| Ok(e.node(t.id()).op.clone()))
            .unwrap()
    }

    fn specs_of(t: &Tensor) -> Vec<StrideSpec> {
        match op_of(t) {
            Op::L0(L0::Restride { specs, .. }) => specs.to_vec(),
            other => panic!("expected a Restride, got {other:?}"),
        }
    }

    /// Every node in the graph, in creation order.
    fn all_ops(g: &Graph) -> Vec<Op> {
        g.handle()
            .with_egraph(|e| {
                Ok((0..e.len())
                    .map(|i| e.node(fusor2_ir::egraph::Id(i as u32)).op.clone())
                    .collect())
            })
            .unwrap()
    }

    // ---- 1 ---------------------------------------------------------------

    /// Every elementwise/scalar-arith/comparison entry point grows the graph
    /// by exactly one node, and that node is a `Map`.
    #[test]
    fn one_node_per_op() {
        let g = graph();

        macro_rules! check {
            ($build:expr) => {{
                // A fresh operand per op, so hash-consing cannot hide a node.
                let x = leaf(&g, &[2, 3]);
                let before = node_count(&g);
                let y = ($build)(&x);
                let after = node_count(&g);
                assert_eq!(after - before, 1, "{} minted {} nodes", stringify!($build), after - before);
                assert_eq!(tag_of(&y), OpTag::Map, "{}", stringify!($build));
            }};
        }

        // 21 unaries plus sqr and recip.
        check!(|x: &Tensor| x.exp().unwrap());
        check!(|x: &Tensor| x.exp2().unwrap());
        check!(|x: &Tensor| x.log().unwrap());
        check!(|x: &Tensor| x.log2().unwrap());
        check!(|x: &Tensor| x.sqrt().unwrap());
        check!(|x: &Tensor| x.inverse_sqrt().unwrap());
        check!(|x: &Tensor| x.sin().unwrap());
        check!(|x: &Tensor| x.cos().unwrap());
        check!(|x: &Tensor| x.tan().unwrap());
        check!(|x: &Tensor| x.tanh().unwrap());
        check!(|x: &Tensor| x.asin().unwrap());
        check!(|x: &Tensor| x.acos().unwrap());
        check!(|x: &Tensor| x.atan().unwrap());
        check!(|x: &Tensor| x.sinh().unwrap());
        check!(|x: &Tensor| x.cosh().unwrap());
        check!(|x: &Tensor| x.asinh().unwrap());
        check!(|x: &Tensor| x.acosh().unwrap());
        check!(|x: &Tensor| x.atanh().unwrap());
        check!(|x: &Tensor| x.abs().unwrap());
        check!(|x: &Tensor| x.neg().unwrap());
        check!(|x: &Tensor| x.sqr().unwrap());
        check!(|x: &Tensor| x.recip().unwrap());
        check!(|x: &Tensor| x.approximate_exp().unwrap());
        check!(|x: &Tensor| x.less_approximate_exp().unwrap());

        // 10 scalar-ariths.
        check!(|x: &Tensor| x.add_scalar(1.0f32).unwrap());
        check!(|x: &Tensor| x.sub_scalar(1.0f32).unwrap());
        check!(|x: &Tensor| x.rsub_scalar(1.0f32).unwrap());
        check!(|x: &Tensor| x.mul_scalar(2.0f32).unwrap());
        check!(|x: &Tensor| x.div_scalar(2.0f32).unwrap());
        check!(|x: &Tensor| x.rdiv_scalar(2.0f32).unwrap());
        check!(|x: &Tensor| x.pow_scalar(3.0f32).unwrap());
        check!(|x: &Tensor| x.max_scalar(0.0f32).unwrap());
        check!(|x: &Tensor| x.min_scalar(6.0f32).unwrap());
        check!(|x: &Tensor| x.clamp(0.0f32, 6.0f32).unwrap());

        // 12 comparisons.
        check!(|x: &Tensor| x.eq_scalar(0.0f32).unwrap());
        check!(|x: &Tensor| x.ne_scalar(0.0f32).unwrap());
        check!(|x: &Tensor| x.lt_scalar(0.0f32).unwrap());
        check!(|x: &Tensor| x.lte_scalar(0.0f32).unwrap());
        check!(|x: &Tensor| x.gt_scalar(0.0f32).unwrap());
        check!(|x: &Tensor| x.gte_scalar(0.0f32).unwrap());

        // cast and round.
        check!(|x: &Tensor| x.cast(Dtype::F16).unwrap());
        check!(|x: &Tensor| x.round().unwrap());
        check!(|x: &Tensor| x.floor().unwrap());
        check!(|x: &Tensor| x.ceil().unwrap());
        check!(|x: &Tensor| x.trunc().unwrap());

        // Same-rank binaries and where_cond need a second/third operand, so
        // count them separately.
        let a = leaf(&g, &[2, 3]);
        let b = leaf_of(&g, &[2, 3], Dtype::F16).cast(Dtype::F32).unwrap();
        for build in [
            Tensor::add as fn(&Tensor, &Tensor) -> Result<Tensor>,
            Tensor::sub,
            Tensor::mul,
            Tensor::div,
            Tensor::pow,
            Tensor::eq_tensor,
            Tensor::ne_tensor,
            Tensor::lt_tensor,
            Tensor::lte_tensor,
            Tensor::gt_tensor,
            Tensor::gte_tensor,
        ] {
            let before = node_count(&g);
            let y = build(&a, &b).unwrap();
            assert_eq!(node_count(&g) - before, 1);
            assert_eq!(tag_of(&y), OpTag::Map);
        }

        let c = leaf(&g, &[2, 3]);
        let before = node_count(&g);
        let w = a.where_cond(&b, &c).unwrap();
        assert_eq!(node_count(&g) - before, 1);
        assert_eq!(tag_of(&w), OpTag::Map);
    }

    // ---- 3 ---------------------------------------------------------------

    #[test]
    fn broadcast_right_aligned() {
        let g = graph();
        let x = leaf(&g, &[3]);
        let y = x.broadcast_as(&dims(&[2, 3])).unwrap();
        assert_eq!(tag_of(&y), OpTag::Restride);
        assert_eq!(
            specs_of(&y),
            vec![
                StrideSpec::broadcast(Dim::Const(2)),
                StrideSpec::dim(0, Dim::Const(3))
            ]
        );

        let m = leaf(&g, &[2, 1, 4]);
        let mb = m.broadcast_as(&dims(&[2, 3, 4])).unwrap();
        assert_eq!(specs_of(&mb)[1].multiplier, 0);

        let bad = leaf(&g, &[5]);
        assert!(matches!(
            bad.broadcast_as(&dims(&[2, 3])),
            Err(Error::Shape(_))
        ));

        // An unmatched target dim is insertable at position 1, not only left.
        let mid = leaf(&g, &[2, 4]);
        let midb = mid.broadcast_as(&dims(&[2, 3, 4])).unwrap();
        assert_eq!(&midb.shape()[..], &dims(&[2, 3, 4])[..]);
    }

    // ---- 4 ---------------------------------------------------------------

    #[test]
    fn no_implicit_broadcast_in_ir() {
        let g = graph();
        let a = leaf(&g, &[2, 3]);
        let b = leaf(&g, &[3]);
        assert!(a.add(&b).is_err());

        let before = node_count(&g);
        let c = a.add_(&b).unwrap();
        assert_eq!(node_count(&g) - before, 3, "Restride, Restride, Map");
        assert_eq!(tag_of(&c), OpTag::Map);
        assert_eq!(&c.shape()[..], &dims(&[2, 3])[..]);
    }

    // ---- 5 ---------------------------------------------------------------

    #[test]
    fn scalar_is_lit_or_uniform() {
        use fusor2_ir::scalar::ScalarKind;
        let g = graph();
        let x = leaf(&g, &[4]);

        let lit = x.mul_scalar(2.0f32).unwrap();
        match op_of(&lit) {
            Op::L0(L0::Map { expr, .. }) => match expr.kind() {
                ScalarKind::Bin { b, .. } => assert!(matches!(b.kind(), ScalarKind::Lit(_))),
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }

        let sym = g.handle().fresh_sym();
        let uni = x.mul_scalar(Scalar::Uniform(sym)).unwrap();
        match op_of(&uni) {
            Op::L0(L0::Map { expr, .. }) => match expr.kind() {
                ScalarKind::Bin { b, .. } => {
                    assert!(matches!(b.kind(), ScalarKind::Uniform(_)))
                }
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }

        // The node's structural identity does not depend on the bound value:
        // writing the uniform twice reuses the same node.
        g.handle().set_uniform(sym, 1e-3);
        let again = x.mul_scalar(Scalar::Uniform(sym)).unwrap();
        assert_eq!(uni.id(), again.id());
        g.handle().set_uniform(sym, 7.0);
        let third = x.mul_scalar(Scalar::Uniform(sym)).unwrap();
        assert_eq!(uni.id(), third.id());
    }

    // ---- 6 ---------------------------------------------------------------

    #[test]
    fn views_are_restride() {
        let g = graph();
        let x = leaf(&g, &[2, 3, 4]);
        for y in [
            x.reshape(&[Extent::Dim(Dim::Const(6)), Extent::Hole]).unwrap(),
            x.transpose(0, 1).unwrap(),
            x.t().unwrap(),
            x.permute(&[2, 1, 0]).unwrap(),
            x.slice(&[0..1, 1..3, 0..4]).unwrap(),
            x.narrow(1, 1, 2).unwrap(),
            x.flatten_all().unwrap(),
            x.flatten_last_n(1).unwrap(),
            x.flatten_first_n(1).unwrap(),
            x.unsqueeze(1).unwrap(),
            x.broadcast_as(&dims(&[2, 3, 4])).unwrap(),
        ] {
            assert_eq!(tag_of(&y), OpTag::Restride);
        }
        for c in x.chunk(2, 2).unwrap() {
            assert_eq!(tag_of(&c), OpTag::Restride);
        }
        let s = leaf(&g, &[1, 3, 1]);
        assert_eq!(tag_of(&s.squeeze(0).unwrap()), OpTag::Restride);
        assert_eq!(tag_of(&s.squeeze_dims(&[0, 2]).unwrap()), OpTag::Restride);
        assert_eq!(tag_of(&x.unsqueeze_dims(&[0, 4]).unwrap()), OpTag::Restride);

        // An inserted size-1 axis is an ordinary axis, not a stride-0 one.
        let u = x.unsqueeze(1).unwrap();
        assert_eq!(specs_of(&u)[1].multiplier, 1);

        // Windows are their own node kind.
        let w = x.sliding_window_view(&[SlidingWindow::new(2, 2, 2)]).unwrap();
        assert_eq!(tag_of(&w), OpTag::Window);
        assert_eq!(&w.shape()[..], &dims(&[2, 3, 2, 2])[..]);
    }

    // ---- 7 ---------------------------------------------------------------

    #[test]
    fn restride_composes_relatively() {
        use fusor2_ir::semantics::infer_l0::restride_layout;
        let g = graph();
        let x = leaf(&g, &[2, 3, 4]);
        let a = x.transpose(0, 1).unwrap();
        let b = a.slice(&[0..2, 1..2, 0..3]).unwrap();

        let want = restride_layout(
            &restride_layout(&Layout::contiguous(&dims(&[2, 3, 4])), &specs_of(&a)).unwrap(),
            &specs_of(&b),
        )
        .unwrap();

        assert_eq!(want.shape(), &dims(&[2, 1, 3])[..]);
        // transpose(0,1) over [12, 4, 1] gives [4, 12, 1]; slicing 1..2 on
        // axis 1 keeps those strides and shifts the offset by 12.
        assert_eq!(want.strides(), &dims(&[4, 12, 1])[..]);
        assert_eq!(want.offset(), Dim::Const(12));
        assert_eq!(&b.shape()[..], want.shape());
    }

    // ---- 8 ---------------------------------------------------------------

    #[test]
    fn matmul_spec() {
        use fusor2_ir::ir::level0::Label;
        let g = graph();
        let a = leaf(&g, &[2, 4, 8]);
        let b = leaf(&g, &[2, 8, 16]);
        let y = a.matmul(&b).unwrap();
        assert_eq!(tag_of(&y), OpTag::Contract);
        assert_eq!(&y.shape()[..], &dims(&[2, 4, 16])[..]);
        match op_of(&y) {
            Op::L0(L0::Contract { spec, .. }) => {
                assert_eq!(&spec.a[..], &[Label(0), Label(1), Label(2)]);
                assert_eq!(&spec.b[..], &[Label(0), Label(2), Label(3)]);
                assert_eq!(&spec.out[..], &[Label(0), Label(1), Label(3)]);
                let e = fusor2_ir::contract_spec::extents(&spec, &a.shape(), &b.shape()).unwrap();
                assert_eq!(e[&Label(2)], Dim::Const(8));
            }
            other => panic!("{other:?}"),
        }

        let p = leaf(&g, &[4, 8]);
        let q = leaf(&g, &[16, 8]);
        let t = p.matmul_t(&q).unwrap();
        assert_eq!(tag_of(&t), OpTag::Contract);
        assert_eq!(&t.shape()[..], &dims(&[4, 16])[..]);
        match op_of(&t) {
            Op::L0(L0::Contract { spec, .. }) => {
                // b is [n, k], not [k, n]; the node is otherwise identical.
                assert_eq!(&spec.b[..], &[Label(2), Label(1)]);
                assert_eq!(spec.d_lhs().out, spec.a);
                assert_eq!(spec.d_rhs().out, spec.b);
            }
            other => panic!("{other:?}"),
        }

        // No implicit batch broadcast.
        let wide = leaf(&g, &[3, 8, 16]);
        assert!(a.matmul(&wide).is_err());
    }

    // ---- 9 ---------------------------------------------------------------

    #[test]
    fn reductions() {
        let g = graph();
        let x = leaf(&g, &[2, 6, 4]);
        for y in [
            x.sum(1).unwrap(),
            x.max(1).unwrap(),
            x.min(1).unwrap(),
            x.product(1).unwrap(),
        ] {
            assert_eq!(tag_of(&y), OpTag::Fold);
            assert_eq!(&y.shape()[..], &dims(&[2, 4])[..]);
        }
        for y in [
            x.sum_keepdim(1).unwrap(),
            x.max_keepdim(1).unwrap(),
            x.min_keepdim(1).unwrap(),
            x.product_keepdim(1).unwrap(),
        ] {
            assert_eq!(&y.shape()[..], &dims(&[2, 1, 4])[..]);
        }

        // `max_with_tie` is how a parity requirement becomes a declaration.
        let split = x.max_with_tie(1, TiePolicy::SplitEvenly).unwrap();
        let first = x.max_with_tie(1, TiePolicy::FirstWins).unwrap();
        assert_ne!(split.id(), first.id());
        assert_eq!(split.id(), x.max(1).unwrap().id(), "SplitEvenly is default");

        // mean over a Const axis emits a literal; over a Sym axis, a uniform.
        use fusor2_ir::scalar::ScalarKind;
        let m = x.mean(1).unwrap();
        match op_of(&m) {
            Op::L0(L0::Map { expr, .. }) => match expr.kind() {
                ScalarKind::Bin { b, .. } => assert!(matches!(b.kind(), ScalarKind::Lit(_))),
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }
        let s = g.sym("seq");
        let dyn_x = g.leaf("d", &[Dim::Const(2), s], Dtype::F32).unwrap();
        let dm = dyn_x.mean(1).unwrap();
        match op_of(&dm) {
            Op::L0(L0::Map { expr, .. }) => match expr.kind() {
                ScalarKind::Bin { b, .. } => {
                    assert!(matches!(b.kind(), ScalarKind::Uniform(_)))
                }
                other => panic!("{other:?}"),
            },
            other => panic!("{other:?}"),
        }

        // var is exactly Map(sqr), Fold, Map(mul), Fold, Map(mul), Map(sqr),
        // Map(sub) — seven nodes, no stable-variance carrier and no extra
        // pass. Which carrier runs is a rewrite's decision, not the
        // frontend's.
        let fresh = leaf(&g, &[2, 7, 4]);
        let before = node_count(&g);
        let v = fresh.var(1).unwrap();
        assert_eq!(node_count(&g) - before, 7);
        assert_eq!(&v.shape()[..], &dims(&[2, 4])[..]);
    }

    // ---- 10 --------------------------------------------------------------

    #[test]
    fn rank_zero_first_class() {
        use fusor2_ir::dtype::Splat;
        let g = graph();
        let s = Tensor::splat(g.handle(), Splat::F32(1.0), &[]).unwrap();
        assert_eq!(s.rank(), 0);
        assert!(s.is_scalar());
        assert_eq!(s.elem_count(), Some(1));
        // Every op accepts it.
        assert_eq!(s.exp().unwrap().rank(), 0);
        assert_eq!(s.add_scalar(1.0f32).unwrap().rank(), 0);

        let v = leaf(&g, &[5]);
        let total = v.sum_all().unwrap();
        assert_eq!(total.rank(), 0);
        assert_eq!(tag_of(&total), OpTag::Fold);
    }

    // ---- 11 --------------------------------------------------------------

    #[test]
    fn scatter_substrate() {
        let g = graph();
        let parts: Vec<Tensor> = (0..3).map(|_| leaf(&g, &[2, 3])).collect();
        let before = all_ops(&g).len();
        let joined = cat(&parts, 1).unwrap();
        assert_eq!(&joined.shape()[..], &dims(&[2, 9])[..]);

        let minted = &all_ops(&g)[before..];
        let consts = minted
            .iter()
            .filter(|o| matches!(o, Op::L0(L0::Leaf(LeafKind::Const { .. }))))
            .count();
        let scatters = minted
            .iter()
            .filter(|o| {
                matches!(
                    o,
                    Op::L0(L0::Scatter {
                        combine: fusor2_ir::ir::level0::ScatterCombine::Set,
                        unique: true,
                        ..
                    })
                )
            })
            .count();
        assert_eq!(consts, 1, "one Const fill");
        assert_eq!(scatters, 3, "one Scatter{{Set, unique}} per part");

        // A zero repeat short-circuits to a single Const leaf.
        let x = leaf(&g, &[2, 3]);
        let before = all_ops(&g).len();
        let z = x.repeat(&[2, 0]).unwrap();
        assert_eq!(&z.shape()[..], &dims(&[4, 0])[..]);
        let minted = &all_ops(&g)[before..];
        assert_eq!(minted.len(), 1);
        assert!(matches!(
            minted[0],
            Op::L0(L0::Leaf(LeafKind::Const { .. }))
        ));

        // stack is unsqueeze + cat.
        let st = stack(&parts, 0).unwrap();
        assert_eq!(&st.shape()[..], &dims(&[3, 2, 3])[..]);

        // pad and resize go through the same substrate.
        let p = x.pad_with_zeros(1, 1, 2).unwrap();
        assert_eq!(&p.shape()[..], &dims(&[2, 6])[..]);
        let r = x.resize(&dims(&[3, 2])).unwrap();
        assert_eq!(&r.shape()[..], &dims(&[3, 2])[..]);
    }

    // ---- 12 --------------------------------------------------------------

    #[test]
    fn index_ops() {
        let g = graph();
        let table = leaf(&g, &[1024, 24]);
        let ids = g
            .tensor(Dtype::U32, &dims(&[3]), bytemuck::cast_slice(&[1u32, 0, 3]))
            .unwrap();

        let sel = table.index_select(0, &ids).unwrap();
        assert_eq!(tag_of(&sel), OpTag::Gather);
        assert_eq!(&sel.shape()[..], &dims(&[3, 24])[..]);

        let grid = g
            .tensor(
                Dtype::U32,
                &dims(&[2, 3]),
                bytemuck::cast_slice(&[1u32, 0, 3, 2, 2, 2]),
            )
            .unwrap();
        let emb = table.embedding(&grid).unwrap();
        assert_eq!(&emb.shape()[..], &dims(&[2, 3, 24])[..]);
        // flatten -> Gather -> reshape: the Gather is the middle node.
        assert_eq!(tag_of(&emb), OpTag::Restride);

        // gather_last on [3, 4] with [1, 0, 3] builds linear indices
        // [1, 4, 11] out of the row offsets [0, 4, 8].
        let rows = leaf(&g, &[3, 4]);
        let picked = rows.gather_last(&ids).unwrap();
        assert_eq!(tag_of(&picked), OpTag::Gather);
        assert_eq!(&picked.shape()[..], &dims(&[3])[..]);
        let offsets = crate::tensor::construction::arange_bytes(Dtype::U32, 0.0, 12.0, 4.0).unwrap();
        let offsets: Vec<u32> = offsets
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(offsets, vec![0, 4, 8]);
        let linear: Vec<u32> = offsets.iter().zip([1u32, 0, 3]).map(|(a, b)| a + b).collect();
        assert_eq!(linear, vec![1, 4, 11]);
    }

    // ---- 13 --------------------------------------------------------------

    #[test]
    fn i_indexing() {
        let g = graph();
        let x = leaf(&g, &[2, 5, 4]);
        let y = x.i((.., 2usize, ..)).unwrap();
        assert_eq!(y.rank(), 2);
        assert_eq!(&y.shape()[..], &dims(&[2, 4])[..]);

        // A zero-offset pick collapses into a single Restride.
        let before = node_count(&g);
        let z = x.i((.., 0usize, ..)).unwrap();
        assert_eq!(node_count(&g) - before, 1);
        assert_eq!(tag_of(&z), OpTag::Restride);
        assert_eq!(&z.shape()[..], &dims(&[2, 4])[..]);

        // Ranges narrow the surviving axes.
        let w = x.i((0..1, 1usize, 1..3)).unwrap();
        assert_eq!(&w.shape()[..], &dims(&[1, 2])[..]);
        assert_eq!(IndexOp::from(1..3), IndexOp::Range(1..3));
    }

    #[test]
    #[should_panic(expected = "exactly one bare usize index, got 0")]
    fn i_with_no_bare_index_panics() {
        let g = graph();
        let x = leaf(&g, &[2, 5, 4]);
        let _ = x.i((.., .., ..));
    }

    #[test]
    #[should_panic(expected = "exactly one bare usize index, got 2")]
    fn i_with_two_bare_indices_panics() {
        let g = graph();
        let x = leaf(&g, &[2, 5, 4]);
        let _ = x.i((0usize, 1usize, ..));
    }

    // ---- 16 --------------------------------------------------------------

    /// Every alias hash-conses onto its target, which is the strongest form of
    /// "structurally identical": the same node id.
    #[test]
    fn alias_surface() {
        let g = graph();
        let x = leaf(&g, &[3]);
        assert_eq!(
            x.expand(&dims(&[2, 3])).unwrap().id(),
            x.broadcast_as(&dims(&[2, 3])).unwrap().id()
        );
        assert_eq!(x.square().unwrap().id(), x.sqr().unwrap().id());
    }

    // ---- 17 --------------------------------------------------------------

    #[test]
    fn arange_step_builds_the_right_leaf() {
        let g = graph();
        let t = Tensor::arange_step(g.handle(), Dtype::F32, 5.0, 0.0, -2.0).unwrap();
        assert_eq!(&t.shape()[..], &dims(&[3])[..]);
        let bytes = g.handle().leaf_bytes(t.id()).unwrap();
        let v: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(v, vec![5.0, 3.0, 1.0]);
    }

    // ---- constructors, casts, typed --------------------------------------

    #[test]
    fn fills_are_const_leaves_with_no_upload() {
        use fusor2_ir::dtype::Splat;
        let g = graph();
        for t in [
            Tensor::zeros(g.handle(), Dtype::F32, &dims(&[2, 3])).unwrap(),
            Tensor::ones(g.handle(), Dtype::F16, &dims(&[4])).unwrap(),
            Tensor::splat(g.handle(), Splat::U32(7), &dims(&[1])).unwrap(),
            Tensor::full(g.handle(), &dims(&[1]), Splat::I32(-1)).unwrap(),
        ] {
            assert_eq!(tag_of(&t), OpTag::Leaf);
            assert!(matches!(
                op_of(&t),
                Op::L0(L0::Leaf(LeafKind::Const { .. }))
            ));
            assert!(g.handle().leaf_bytes(t.id()).is_none());
        }
    }

    #[test]
    fn every_dense_cast_pair_builds_including_float_to_u32() {
        let g = graph();
        let all = [Dtype::F32, Dtype::F16, Dtype::BF16, Dtype::U32, Dtype::I32];
        for from in all {
            let x = leaf_of(&g, &[2], from);
            for to in all {
                let y = x.cast(to).unwrap();
                assert_eq!(y.dtype(), to);
                assert_eq!(tag_of(&y), OpTag::Map);
            }
        }
    }

    #[test]
    fn typed_rejects_a_mismatch_without_panicking() {
        let g = graph();
        let x = leaf(&g, &[2, 3]);
        assert!(x.clone().typed::<2, f32>().is_ok());
        assert!(x.clone().typed::<3, f32>().is_err());
        assert!(x.typed::<2, u32>().is_err());
    }

    #[test]
    fn reshape_accepts_exactly_one_hole() {
        let g = graph();
        let x = leaf(&g, &[2, 3, 4]);
        let y = x.reshape(&[Extent::Dim(Dim::Const(6)), Extent::Hole]).unwrap();
        assert_eq!(&y.shape()[..], &dims(&[6, 4])[..]);
        assert!(x.reshape(&[Extent::Hole, Extent::Hole]).is_err());
        assert!(x.reshape(&[Extent::Dim(Dim::Const(5)), Extent::Hole]).is_err());
    }

    #[test]
    fn slice_assign_of_a_two_axis_region_uses_explicit_indices() {
        let g = graph();
        let base = leaf(&g, &[4, 4]);
        let patch = leaf(&g, &[2, 2]);
        let out = base.slice_assign(&[1..3, 1..3], &patch).unwrap();
        assert_eq!(&out.shape()[..], &dims(&[4, 4])[..]);
        // flatten, flatten, index upload, Scatter, reshape.
        assert_eq!(tag_of(&out), OpTag::Restride);
    }
}
