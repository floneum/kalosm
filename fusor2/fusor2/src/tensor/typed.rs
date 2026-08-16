//! The compile-time-rank tensor. A zero-cost newtype over [`crate::Tensor`]:
//! it has no effect on the IR, hosts no witness traits, and imposes no rank
//! ceiling.
//!
//! A rank-changing method takes its output rank as an ordinary const
//! parameter and validates it once; a bad rank is a panic that names the op.
//!
//! Every method panics instead of returning `Result`: a shape error here is a
//! bug in the model, not a runtime condition. The runtime-rank
//! [`crate::Tensor`] returns `Result` and is the layer to use when a shape is
//! data. Every panic carries the underlying [`Error`], so the two agree on
//! diagnosis and differ only on delivery.
//!
//! # Mixed precision
//!
//! Folds and contractions accumulate in [`Dtype::compute_dtype`], so an f16
//! operand accumulates in f32; [`narrow_acc`] casts the result back. Wide
//! arithmetic, narrow types.

use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Neg, Range, Rem, Sub};

use fusor2_ir::dtype::{Dtype, RoundMode, Splat};
use fusor2_ir::egraph::Id;
use fusor2_ir::shape::{Dim, SlidingWindow, StrideSpec};

use crate::device::{Device, ok};
use crate::graph::GraphRef;
use crate::ops::view::Extent;
use crate::tensor::readback::{TensorSlice, ToVec};
use crate::tensor::{Scalar, Tensor as Dyn};
use crate::{Error, Result};

mod composite;
mod construct;
mod ops;

/// A Rust scalar with a fusor2 dtype.
pub trait Element: bytemuck::Pod + Copy + Send + Sync + 'static {
    const DTYPE: Dtype;
    /// This value as the backend's splat literal.
    fn splat(self) -> Splat;
}

impl Element for f32 {
    const DTYPE: Dtype = Dtype::F32;
    fn splat(self) -> Splat {
        Splat::F32(self)
    }
}
impl Element for half::f16 {
    const DTYPE: Dtype = Dtype::F16;
    fn splat(self) -> Splat {
        Splat::F16(self.to_bits())
    }
}
impl Element for half::bf16 {
    const DTYPE: Dtype = Dtype::BF16;
    fn splat(self) -> Splat {
        Splat::BF16(self.to_bits())
    }
}
impl Element for u32 {
    const DTYPE: Dtype = Dtype::U32;
    fn splat(self) -> Splat {
        Splat::U32(self)
    }
}
impl Element for i32 {
    const DTYPE: Dtype = Dtype::I32;
    fn splat(self) -> Splat {
        Splat::I32(self)
    }
}

/// An axis argument: a literal index, or one of the from-the-end selectors.
pub trait Axis<const R: usize> {
    fn resolve(self) -> usize;
}

impl<const R: usize> Axis<R> for usize {
    fn resolve(self) -> usize {
        self
    }
}

/// `R - 1`, the last axis.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Minus1;
/// `R - 2`, the second-to-last axis.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Minus2;

impl<const R: usize> Axis<R> for Minus1 {
    fn resolve(self) -> usize {
        const { assert!(R >= 1, "Minus1 needs rank >= 1") };
        R - 1
    }
}
impl<const R: usize> Axis<R> for Minus2 {
    fn resolve(self) -> usize {
        const { assert!(R >= 2, "Minus2 needs rank >= 2") };
        R - 2
    }
}

/// A [`crate::Tensor`] whose rank and dtype are asserted at construction and
/// then tracked in the type. `repr(transparent)`: the same size and layout as
/// the runtime-rank tensor.
#[repr(transparent)]
pub struct Tensor<const R: usize, T: Element = f32> {
    raw: Dyn,
    _t: PhantomData<T>,
}

impl<const R: usize, T: Element> Clone for Tensor<R, T> {
    fn clone(&self) -> Self {
        Self {
            raw: self.raw.clone(),
            _t: PhantomData,
        }
    }
}

impl<const R: usize, T: Element> std::fmt::Debug for Tensor<R, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor<{R}, {:?}>({:?})", T::DTYPE, self.raw.id())
    }
}

/// Extents as `[usize; N]`, or an error naming the offender.
fn const_extents<const N: usize>(shape: &[Dim], what: &str) -> Result<[usize; N]> {
    if shape.len() != N {
        return Err(Error::Shape(format!(
            "{what}: value has rank {}, not {N}",
            shape.len()
        )));
    }
    let mut out = [0usize; N];
    for (slot, dim) in out.iter_mut().zip(shape) {
        *slot = dim
            .as_const()
            .ok_or_else(|| {
                Error::Shape(format!("{what}: extent {dim:?} is symbolic and has no usize"))
            })?
            .try_into()
            .map_err(|_| Error::Shape(format!("{what}: extent {dim:?} exceeds a usize")))?;
    }
    Ok(out)
}

fn dims_of<const N: usize>(shape: [usize; N]) -> Vec<Dim> {
    shape.iter().map(|&d| Dim::Const(d as u64)).collect()
}

/// Undo the accumulator promotion an accumulating op performs.
///
/// A fold and a contraction accumulate in [`Dtype::compute_dtype`], so an f16
/// operand comes back f32; the const-rank API narrows the result back to keep
/// its signatures dtype-preserving.
///
/// Narrows exactly the promotion and nothing else: any other dtype
/// disagreement falls through to [`Tensor::try_from_dyn`], which reports it. A
/// blanket `cast` here would convert a genuine mismatch into a silent
/// reinterpretation.
fn narrow_acc<T: Element>(r: Result<Dyn>) -> Result<Dyn> {
    let v = r?;
    if v.dtype() == T::DTYPE || v.dtype() != T::DTYPE.compute_dtype() {
        return Ok(v);
    }
    v.cast(T::DTYPE)
}

impl<const R: usize, T: Element> Tensor<R, T> {
    pub const RANK: usize = R;

    /// Wrap a runtime-rank value.
    ///
    /// The inverse of [`Tensor::into_dyn`].
    ///
    /// # Panics
    /// If the rank or dtype disagrees. [`Tensor::try_from_dyn`] reports
    /// instead.
    #[track_caller]
    pub fn from_dyn(raw: Dyn) -> Self {
        ok("Tensor::from_dyn", Self::try_from_dyn(raw))
    }

    /// [`Tensor::from_dyn`], reporting the mismatch instead of panicking.
    pub fn try_from_dyn(raw: Dyn) -> Result<Self> {
        if raw.rank() != R {
            return Err(Error::Shape(format!(
                "Tensor<{R}, _>: value has rank {}",
                raw.rank()
            )));
        }
        if raw.dtype() != T::DTYPE {
            return Err(Error::Dtype(format!(
                "Tensor<_, {:?}>: value has dtype {:?}",
                T::DTYPE,
                raw.dtype()
            )));
        }
        Ok(Self {
            raw,
            _t: PhantomData,
        })
    }

    #[track_caller]
    pub(super) fn wrap<const O: usize, E: Element>(what: &str, r: Result<Dyn>) -> Tensor<O, E> {
        let raw = ok(what, r);
        ok(what, Tensor::<O, E>::try_from_dyn(raw))
    }

    /// The runtime-rank value underneath. The IR never saw the wrapper.
    pub fn into_inner(self) -> Dyn {
        self.raw
    }
    /// Alias of [`Tensor::into_inner`].
    pub fn into_dyn(self) -> Dyn {
        self.raw
    }
    /// Borrow the runtime-rank value underneath.
    pub fn as_dyn(&self) -> &Dyn {
        &self.raw
    }
    pub fn id(&self) -> Id {
        self.raw.id()
    }
    pub fn graph(&self) -> &GraphRef {
        self.raw.graph()
    }
    pub fn dtype(&self) -> Dtype {
        self.raw.dtype()
    }
    pub fn rank(&self) -> usize {
        R
    }

    /// Extents, as the const-rank array a model destructures.
    ///
    /// # Panics
    /// If any extent is still symbolic. A const-rank shape is compile-time
    /// program structure; a symbolic one belongs to the runtime-rank API.
    #[track_caller]
    pub fn shape(&self) -> [usize; R] {
        ok("Tensor::shape", const_extents::<R>(&self.raw.shape(), "shape"))
    }

    /// Extent of one axis.
    #[track_caller]
    pub fn dim(&self, i: usize) -> usize {
        self.shape()[i]
    }

    /// Extents, including the symbolic ones.
    ///
    /// [`Tensor::shape`] panics on a [`Dim::Sym`]; this is the accessor for
    /// the code that has one.
    #[track_caller]
    pub fn extents(&self) -> [Dim; R] {
        let shape = self.raw.shape();
        if shape.len() != R {
            ok::<()>(
                "Tensor::extents",
                Err(Error::Shape(format!(
                    "extents: value has rank {}, not {R}",
                    shape.len()
                ))),
            );
        }
        std::array::from_fn(|i| shape[i])
    }

    /// The extent of one axis, symbolic or not.
    #[track_caller]
    pub fn extent(&self, axis: impl Axis<R>) -> Dim {
        self.extents()[axis.resolve()]
    }

    /// Element count. Panics on a symbolic extent; [`Tensor::elem_count`] is
    /// the total form.
    #[track_caller]
    pub fn elements(&self) -> usize {
        self.shape().iter().product()
    }

    /// Element count, or `None` when an extent is still symbolic.
    pub fn elem_count(&self) -> Option<u64> {
        self.raw.elem_count()
    }

    /// Replace the bytes of an external leaf in place, leaving every node
    /// built on top of it — and so the resolved plan — untouched.
    #[track_caller]
    pub fn set_bytes(&self, bytes: Vec<u8>) {
        ok("Tensor::set_bytes", self.raw.set_bytes(bytes));
    }

    /// [`Tensor::set_bytes`] from typed host elements.
    #[track_caller]
    pub fn set_elements(&self, data: &[T]) {
        self.set_bytes(bytemuck::cast_slice(data).to_vec());
    }

    /// Re-assert rank and dtype.
    #[track_caller]
    pub fn retype<const O: usize, E: Element>(self) -> Tensor<O, E> {
        ok("Tensor::retype", Tensor::<O, E>::try_from_dyn(self.raw))
    }

    /// Identity, kept for API compatibility; the e-graph owns fusion
    /// decisions. A value that is never re-leafed keeps its producers alive,
    /// so a training loop that builds a fresh tape per step accumulates nodes
    /// in the ambient graph; [`Tensor::detach`] is the correct-but-expensive
    /// way to cut that off.
    pub fn into_concrete(self) -> Self {
        self
    }

    /// Identity; see [`Tensor::into_concrete`].
    pub fn to_concrete(&self) -> Self {
        self.clone()
    }

    /// Materialize and re-leaf, cutting this value off from its producers.
    #[track_caller]
    pub fn detach(&self) -> Self {
        Self::wrap("detach", self.raw.detach())
    }
}

impl<const R: usize, T: Element> Tensor<R, T> {
    /// Upload dense host data.
    #[track_caller]
    pub fn from_slice(device: &Device, shape: [usize; R], data: &[T]) -> Self {
        let want: usize = shape.iter().product();
        if data.len() != want {
            ok::<()>(
                "Tensor::from_slice",
                Err(Error::Shape(format!(
                    "shape {shape:?} needs {want} elements, got {}",
                    data.len()
                ))),
            );
        }
        Self::wrap(
            "from_slice",
            Dyn::from_slice(
                device.handle(),
                T::DTYPE,
                &dims_of(shape),
                bytemuck::cast_slice(data),
            ),
        )
    }

    /// A zero-filled value.
    #[track_caller]
    pub fn zeros(device: &Device, shape: [usize; R]) -> Self {
        Self::wrap(
            "zeros",
            Dyn::zeros(device.handle(), T::DTYPE, &dims_of(shape)),
        )
    }

    /// A one-filled value.
    #[track_caller]
    pub fn ones(device: &Device, shape: [usize; R]) -> Self {
        Self::wrap(
            "ones",
            Dyn::ones(device.handle(), T::DTYPE, &dims_of(shape)),
        )
    }

    /// Every element `value`. Folded into the kernel; never a buffer.
    #[track_caller]
    pub fn splat(device: &Device, value: T, shape: [usize; R]) -> Self {
        Self::wrap(
            "splat",
            Dyn::splat(device.handle(), value.splat(), &dims_of(shape)),
        )
    }

    /// The device this value was built from — the same type the
    /// constructors take.
    pub fn device(&self) -> Device {
        Device::of_graph(self.raw.graph())
    }

    /// Which backend that device runs on.
    pub fn backend(&self) -> crate::session::Backend {
        self.raw.backend()
    }
}

/// Rank- and dtype-preserving unaries.
macro_rules! same {
    ($($name:ident),* $(,)?) => {
        impl<const R: usize, T: Element> Tensor<R, T> {$(
            #[doc = concat!("[`crate::Tensor::", stringify!($name), "`], rank and dtype preserved.")]
            #[track_caller]
            pub fn $name(&self) -> Self {
                // Spelled as a path, not a method call: the runtime tensor
                // also implements `std::ops::Neg`, whose by-value pick would
                // win the method probe over the inherent `Result`-returning
                // one and hand back an already-unwrapped value.
                Self::wrap(stringify!($name), Dyn::$name(&self.raw))
            }
        )*}
    };
}

same!(
    exp,
    exp2,
    log,
    log2,
    sqrt,
    sin,
    cos,
    tan,
    tanh,
    asin,
    acos,
    atan,
    sinh,
    cosh,
    asinh,
    acosh,
    atanh,
    abs,
    neg,
    sqr,
    recip,
    sign,
    tanh_exact,
    approximate_exp,
    less_approximate_exp,
    round,
    round_even,
    floor,
    ceil,
    trunc,
    zeros_like,
    ones_like,
    relu,
    sigmoid,
    silu,
    gelu,
    gelu_exact,
    softplus,
);

/// Rank- and dtype-preserving same-shape binaries.
macro_rules! same_bin {
    ($($name:ident),* $(,)?) => {
        impl<const R: usize, T: Element> Tensor<R, T> {$(
            #[doc = concat!("[`crate::Tensor::", stringify!($name), "`], shapes must match.")]
            #[track_caller]
            pub fn $name(&self, rhs: &Self) -> Self {
                // A path call, for the same reason `same!` uses one: the
                // runtime tensor implements `Add`/`Sub`/`Mul`/`Div`, and a
                // method call would resolve to the operator, not the op.
                Self::wrap(stringify!($name), Dyn::$name(&self.raw, &rhs.raw))
            }
        )*}
    };
}

same_bin!(
    add, sub, mul, div, rem, pow, maximum, minimum, eq_tensor, ne_tensor, lt_tensor, lte_tensor,
    gt_tensor, gte_tensor,
);

/// Rank-preserving scalar-arith and comparisons.
macro_rules! same_scalar {
    ($($name:ident),* $(,)?) => {
        impl<const R: usize, T: Element> Tensor<R, T> {$(
            #[doc = concat!("[`crate::Tensor::", stringify!($name), "`].")]
            #[track_caller]
            pub fn $name(&self, s: impl Into<Scalar>) -> Self {
                Self::wrap(stringify!($name), self.raw.$name(s))
            }
        )*}
    };
}

same_scalar!(
    add_scalar,
    sub_scalar,
    rsub_scalar,
    mul_scalar,
    div_scalar,
    rdiv_scalar,
    rem_scalar,
    pow_scalar,
    max_scalar,
    min_scalar,
    eq_scalar,
    ne_scalar,
    lt_scalar,
    lte_scalar,
    gt_scalar,
    gte_scalar,
);

/// The operand slot of a broadcasting binary, inferred from the argument.
pub trait Operand<const R: usize, T: Element> {
    fn operand(&self) -> &Tensor<R, T>;
}

impl<const R: usize, T: Element> Operand<R, T> for Tensor<R, T> {
    fn operand(&self) -> &Tensor<R, T> {
        self
    }
}

impl<const R: usize, T: Element> Tensor<R, T> {
    /// Broadcasting `a + b`, output rank `O = max(R, R2)`.
    #[track_caller]
    pub fn add_<const R2: usize, const O: usize, B: Operand<R2, T>>(
        &self,
        rhs: &B,
    ) -> Tensor<O, T> {
        Self::wrap("add_", self.raw.add_(&rhs.operand().raw))
    }

    /// Broadcasting `a - b`.
    #[track_caller]
    pub fn sub_<const R2: usize, const O: usize, B: Operand<R2, T>>(
        &self,
        rhs: &B,
    ) -> Tensor<O, T> {
        Self::wrap("sub_", self.raw.sub_(&rhs.operand().raw))
    }

    /// Broadcasting `a * b`.
    #[track_caller]
    pub fn mul_<const R2: usize, const O: usize, B: Operand<R2, T>>(
        &self,
        rhs: &B,
    ) -> Tensor<O, T> {
        Self::wrap("mul_", self.raw.mul_(&rhs.operand().raw))
    }

    /// Broadcasting `a / b`.
    #[track_caller]
    pub fn div_<const R2: usize, const O: usize, B: Operand<R2, T>>(
        &self,
        rhs: &B,
    ) -> Tensor<O, T> {
        Self::wrap("div_", self.raw.div_(&rhs.operand().raw))
    }

    /// Broadcasting `a ^ b`.
    #[track_caller]
    pub fn pow_<const R2: usize, const O: usize, B: Operand<R2, T>>(
        &self,
        rhs: &B,
    ) -> Tensor<O, T> {
        Self::wrap("pow_", self.raw.pow_(&rhs.operand().raw))
    }

    /// Clamp into `[lo, hi]`.
    #[track_caller]
    pub fn clamp(&self, lo: impl Into<Scalar>, hi: impl Into<Scalar>) -> Self {
        Self::wrap("clamp", self.raw.clamp(lo, hi))
    }

    /// Select elementwise: `self` is the predicate.
    #[track_caller]
    pub fn where_cond(&self, on_true: &Self, on_false: &Self) -> Self {
        Self::wrap(
            "where_cond",
            self.raw.where_cond(&on_true.raw, &on_false.raw),
        )
    }

    /// Convert the dtype; the rank is unchanged.
    #[track_caller]
    pub fn cast<E: Element>(&self) -> Tensor<R, E> {
        Self::wrap("cast", self.raw.cast(E::DTYPE))
    }

    /// Reinterpret the bytes.
    #[track_caller]
    pub fn bitcast<E: Element>(&self) -> Tensor<R, E> {
        Self::wrap("bitcast", self.raw.bitcast(E::DTYPE))
    }

    /// Set the rounding mode of a narrowing cast.
    #[track_caller]
    pub fn round_mode(&self, mode: RoundMode) -> Self {
        Self::wrap("round_mode", self.raw.round_mode(mode))
    }
}

impl<const R: usize, T: Element> Tensor<R, T> {
    /// Swap two axes.
    #[track_caller]
    pub fn transpose(&self, d0: impl Axis<R>, d1: impl Axis<R>) -> Self {
        Self::wrap("transpose", self.raw.transpose(d0.resolve(), d1.resolve()))
    }

    /// Swap the last two axes.
    #[track_caller]
    pub fn t(&self) -> Self {
        Self::wrap("t", self.raw.t())
    }

    /// Reorder every axis.
    #[track_caller]
    pub fn permute(&self, order: [usize; R]) -> Self {
        Self::wrap("permute", self.raw.permute(&order))
    }

    /// A contiguous sub-range of every axis.
    #[track_caller]
    pub fn slice(&self, ranges: [Range<usize>; R]) -> Self {
        Self::wrap("slice", self.raw.slice(&ranges))
    }

    /// `len` entries of `dim` starting at `start`.
    #[track_caller]
    pub fn narrow(&self, dim: impl Axis<R>, start: usize, len: usize) -> Self {
        Self::wrap("narrow", self.raw.narrow(dim.resolve(), start, len))
    }

    /// Split one axis into `chunks` equal pieces.
    #[track_caller]
    pub fn chunk(&self, chunks: usize, dim: impl Axis<R>) -> Vec<Self> {
        ok("chunk", self.raw.chunk(chunks, dim.resolve()))
            .into_iter()
            .map(Self::from_dyn)
            .collect()
    }

    /// Reshape into a statically known output rank.
    #[track_caller]
    pub fn reshape<const O: usize>(&self, shape: [usize; O]) -> Tensor<O, T> {
        let extents: [Extent; O] = shape.map(Extent::from);
        Self::wrap("reshape", self.raw.reshape(&extents))
    }

    /// Broadcast into a statically known output rank.
    #[track_caller]
    pub fn broadcast_as<const O: usize>(&self, target: [usize; O]) -> Tensor<O, T> {
        Self::wrap("broadcast_as", self.raw.broadcast_as(&dims_of(target)))
    }

    /// Alias of [`Tensor::broadcast_as`].
    #[track_caller]
    pub fn expand<const O: usize>(&self, target: [usize; O]) -> Tensor<O, T> {
        self.broadcast_as(target)
    }

    /// Restride into a statically known output rank.
    #[track_caller]
    pub fn restride<const O: usize>(&self, specs: [StrideSpec; O]) -> Tensor<O, T> {
        Self::wrap("restride", self.raw.restride(&specs))
    }

    /// Drop a length-1 axis; output rank `O = R - 1`.
    #[track_caller]
    pub fn squeeze<const O: usize>(&self, dim: impl Axis<R>) -> Tensor<O, T> {
        Self::wrap("squeeze", self.raw.squeeze(dim.resolve()))
    }

    /// Insert a length-1 axis; output rank `O = R + 1`.
    #[track_caller]
    pub fn unsqueeze<const O: usize>(&self, dim: usize) -> Tensor<O, T> {
        Self::wrap("unsqueeze", self.raw.unsqueeze(dim))
    }

    /// Every element in one axis.
    #[track_caller]
    pub fn flatten_all(&self) -> Tensor<1, T> {
        Self::wrap("flatten_all", self.raw.flatten_all())
    }

    /// A sliding-window view; output rank `O = R + windows`.
    #[track_caller]
    pub fn sliding_window_view<const O: usize>(&self, specs: &[SlidingWindow]) -> Tensor<O, T> {
        Self::wrap("sliding_window_view", self.raw.sliding_window_view(specs))
    }

    /// Gather rows of `dim` named by `idx`.
    #[track_caller]
    pub fn index_select(&self, dim: impl Axis<R>, idx: &Tensor<1, u32>) -> Self {
        Self::wrap(
            "index_select",
            self.raw.index_select(dim.resolve(), &idx.raw),
        )
    }

    /// Batched matrix product over the last two axes.
    ///
    /// Accumulates in [`Dtype::compute_dtype`] and narrows back, so an f16
    /// matmul has f32 accumulators.
    #[track_caller]
    pub fn matmul(&self, rhs: &Self) -> Self {
        Self::wrap("matmul", narrow_acc::<T>(self.raw.matmul(&rhs.raw)))
    }

    /// Matrix product against a transposed right-hand side.
    #[track_caller]
    pub fn matmul_t(&self, rhs: &Self) -> Self {
        Self::wrap("matmul_t", narrow_acc::<T>(self.raw.matmul_t(&rhs.raw)))
    }

}

/// Rank-reducing folds; `O` is `R - 1`.
///
/// The accumulator dtype is [`Dtype::compute_dtype`], not the operand's: an
/// f16 fold accumulates in f32 and the result is cast back, so the signature
/// stays dtype-preserving while the arithmetic stays wide.
macro_rules! reduce {
    ($($name:ident),* $(,)?) => {
        impl<const R: usize, T: Element> Tensor<R, T> {$(
            #[doc = concat!("[`crate::Tensor::", stringify!($name), "`], output rank `O = R - 1`.")]
            #[track_caller]
            pub fn $name<const O: usize>(&self, axis: impl Axis<R>) -> Tensor<O, T> {
                Self::wrap(stringify!($name), narrow_acc::<T>(self.raw.$name(axis.resolve())))
            }
        )*}
    };
}

reduce!(sum, product, max, min, mean, var, any, all, count_nonzero, norm);

/// Keepdim folds; the rank is preserved. Narrowed like `reduce!`.
macro_rules! reduce_keepdim {
    ($($name:ident),* $(,)?) => {
        impl<const R: usize, T: Element> Tensor<R, T> {$(
            #[doc = concat!("[`crate::Tensor::", stringify!($name), "`], rank preserved.")]
            #[track_caller]
            pub fn $name(&self, axis: impl Axis<R>) -> Self {
                Self::wrap(stringify!($name), narrow_acc::<T>(self.raw.$name(axis.resolve())))
            }
        )*}
    };
}

reduce_keepdim!(
    sum_keepdim,
    max_keepdim,
    mean_keepdim,
    var_keepdim,
);

impl<const R: usize, T: Element> Tensor<R, T> {
    /// `[batch, in_ch, ...spatial]` convolved with `[out_ch, in_ch, ...kernel]`.
    ///
    /// Three const parameters: `WEIGHT_RANK` is the kernel's rank, `DIFF` the
    /// number of spatial axes (and so the length of `padding` and `strides`),
    /// and `WINDOWED` the rank of the sliding-window view the lowering builds,
    /// `R + DIFF`. This is checked arithmetically.
    #[track_caller]
    pub fn conv<const WEIGHT_RANK: usize, const DIFF: usize, const WINDOWED: usize>(
        &self,
        weight: &Tensor<WEIGHT_RANK, T>,
        bias: Option<&Tensor<1, T>>,
        padding: [usize; DIFF],
        stride: [usize; DIFF],
    ) -> Self {
        if WINDOWED != R + DIFF {
            ok::<()>(
                "Tensor::conv",
                Err(Error::Shape(format!(
                    "conv::<{WEIGHT_RANK}, {DIFF}, {WINDOWED}>: the windowed view of a rank-{R} \
                     input over {DIFF} spatial axes has rank {}, not {WINDOWED}",
                    R + DIFF
                ))),
            );
        }
        let to_u32 = |v: [usize; DIFF], what: &str| -> Vec<u32> {
            v.iter()
                .map(|&x| {
                    ok(
                        what,
                        u32::try_from(x)
                            .map_err(|_| Error::Shape(format!("{what} {x} exceeds a u32"))),
                    )
                })
                .collect()
        };
        let stride = to_u32(stride, "conv stride");
        let padding = to_u32(padding, "conv padding");
        let dilation = vec![1u32; DIFF];
        Self::wrap(
            "conv",
            crate::composite::conv::conv(
                &self.raw,
                &weight.raw,
                bias.map(|b| &b.raw),
                &stride,
                &padding,
                &dilation,
            ),
        )
    }
}

macro_rules! binop {
    ($trait:ident, $method:ident, $inner:ident, $scalar:ident) => {
        impl<const R: usize, T: Element> $trait for Tensor<R, T> {
            type Output = Tensor<R, T>;
            #[track_caller]
            fn $method(self, rhs: Self) -> Self::Output {
                Tensor::<R, T>::$inner(&self, &rhs)
            }
        }
        impl<const R: usize, T: Element> $trait<&Tensor<R, T>> for Tensor<R, T> {
            type Output = Tensor<R, T>;
            #[track_caller]
            fn $method(self, rhs: &Tensor<R, T>) -> Self::Output {
                Tensor::<R, T>::$inner(&self, rhs)
            }
        }
        impl<const R: usize, T: Element> $trait<Tensor<R, T>> for &Tensor<R, T> {
            type Output = Tensor<R, T>;
            #[track_caller]
            fn $method(self, rhs: Tensor<R, T>) -> Self::Output {
                Tensor::<R, T>::$inner(self, &rhs)
            }
        }
        impl<const R: usize, T: Element> $trait<&Tensor<R, T>> for &Tensor<R, T> {
            type Output = Tensor<R, T>;
            #[track_caller]
            fn $method(self, rhs: &Tensor<R, T>) -> Self::Output {
                Tensor::<R, T>::$inner(self, rhs)
            }
        }
        impl<const R: usize, T: Element> $trait<f32> for Tensor<R, T> {
            type Output = Tensor<R, T>;
            #[track_caller]
            fn $method(self, rhs: f32) -> Self::Output {
                Tensor::<R, T>::$scalar(&self, rhs)
            }
        }
        impl<const R: usize, T: Element> $trait<f32> for &Tensor<R, T> {
            type Output = Tensor<R, T>;
            #[track_caller]
            fn $method(self, rhs: f32) -> Self::Output {
                Tensor::<R, T>::$scalar(self, rhs)
            }
        }
    };
}

binop!(Add, add, add, add_scalar);
binop!(Sub, sub, sub, sub_scalar);
binop!(Mul, mul, mul, mul_scalar);
binop!(Div, div, div, div_scalar);
binop!(Rem, rem, rem, rem_scalar);

impl<const R: usize, T: Element> Neg for Tensor<R, T> {
    type Output = Tensor<R, T>;
    #[track_caller]
    fn neg(self) -> Self::Output {
        Tensor::<R, T>::neg(&self)
    }
}

impl<const R: usize, T: Element> Neg for &Tensor<R, T> {
    type Output = Tensor<R, T>;
    #[track_caller]
    fn neg(self) -> Self::Output {
        Tensor::<R, T>::neg(self)
    }
}

/// Join values along `dim`. Every part keeps its rank.
#[track_caller]
pub fn cat<const R: usize, T: Element, I>(parts: I, dim: usize) -> Tensor<R, T>
where
    I: IntoIterator<Item = Tensor<R, T>>,
{
    let parts: Vec<Dyn> = parts.into_iter().map(Tensor::into_inner).collect();
    let raw = ok("cat", crate::ops::index::cat(&parts, dim));
    ok("cat", Tensor::try_from_dyn(raw))
}

/// Stack values into a new axis; output rank `O = R + 1`.
#[track_caller]
pub fn stack<const R: usize, const O: usize, T: Element, I>(parts: I, dim: usize) -> Tensor<O, T>
where
    I: IntoIterator<Item = Tensor<R, T>>,
{
    let parts: Vec<Dyn> = parts.into_iter().map(Tensor::into_inner).collect();
    let raw = ok("stack", crate::ops::index::stack(&parts, dim));
    ok("stack", Tensor::try_from_dyn(raw))
}

impl<const R: usize, T: Element> Tensor<R, T> {
    /// [`cat`], as an associated function.
    #[track_caller]
    pub fn cat<I>(parts: I, dim: usize) -> Self
    where
        I: IntoIterator<Item = Tensor<R, T>>,
    {
        cat(parts, dim)
    }
}

/// A host copy of a const-rank value.
///
/// [`ToVec`] on this hands back the nested `Vec` directly rather than a
/// `Result`: by the time a caller holds one, the read already succeeded.
pub struct HostSlice<const R: usize, T: Element> {
    slice: TensorSlice,
    _t: PhantomData<T>,
}

impl<const R: usize, T: Element> HostSlice<R, T> {
    pub fn slice(&self) -> &TensorSlice {
        &self.slice
    }

    /// Row-major copy of every element.
    #[track_caller]
    pub fn to_flat(&self) -> Vec<T> {
        ok("to_flat", self.slice.to_flat::<T>())
    }

    #[track_caller]
    fn extents(&self) -> [usize; R] {
        ok(
            "readback shape",
            const_extents::<R>(self.slice.shape(), "readback"),
        )
    }

    #[track_caller]
    fn at(&self, idx: &[usize]) -> T {
        match self.slice.get::<T>(idx) {
            Some(v) => v,
            None => panic!("fusor2 readback: index {idx:?} out of range"),
        }
    }
}

impl<T: Element> ToVec for HostSlice<0, T> {
    type Output = T;
    #[track_caller]
    fn to_vec(&self) -> T {
        self.at(&[])
    }
}

impl<T: Element> ToVec for HostSlice<1, T> {
    type Output = Vec<T>;
    #[track_caller]
    fn to_vec(&self) -> Vec<T> {
        let [n] = self.extents();
        (0..n).map(|i| self.at(&[i])).collect()
    }
}

impl<T: Element> ToVec for HostSlice<2, T> {
    type Output = Vec<Vec<T>>;
    #[track_caller]
    fn to_vec(&self) -> Vec<Vec<T>> {
        let [n, m] = self.extents();
        (0..n)
            .map(|i| (0..m).map(|j| self.at(&[i, j])).collect())
            .collect()
    }
}

impl<T: Element> ToVec for HostSlice<3, T> {
    type Output = Vec<Vec<Vec<T>>>;
    #[track_caller]
    fn to_vec(&self) -> Vec<Vec<Vec<T>>> {
        let [n, m, p] = self.extents();
        (0..n)
            .map(|i| {
                (0..m)
                    .map(|j| (0..p).map(|k| self.at(&[i, j, k])).collect())
                    .collect()
            })
            .collect()
    }
}

impl<const R: usize, T: Element> Tensor<R, T> {
    /// Resolve up to this value and copy it back to the host.
    ///
    /// Returns a ready future: fusor2's readback is synchronous, and the
    /// `async` shape is what a caller driving a runtime expects to await.
    pub fn as_slice(&self) -> impl Future<Output = Result<HostSlice<R, T>>> + 'static {
        std::future::ready(self.read())
    }

    /// The blocking spelling of [`Tensor::as_slice`].
    pub fn read(&self) -> Result<HostSlice<R, T>> {
        Ok(HostSlice {
            slice: self.raw.as_slice()?,
            _t: PhantomData,
        })
    }

    /// Row-major host copy.
    #[track_caller]
    pub fn to_flat(&self) -> Vec<T> {
        ok("to_flat", self.raw.to_flat::<T>())
    }

    /// The single element of a rank-0 value.
    #[track_caller]
    pub fn to_scalar(&self) -> T {
        ok("to_scalar", self.raw.to_scalar::<T>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_typed_tensor_is_one_pointer_pair() {
        assert_eq!(
            std::mem::size_of::<Tensor<3, f32>>(),
            std::mem::size_of::<Dyn>()
        );
        assert_eq!(
            std::mem::align_of::<Tensor<0, u32>>(),
            std::mem::align_of::<Dyn>()
        );
    }

    /// `device()` hands back the type the constructors take, and the round
    /// trip lands in the same graph.
    #[test]
    fn a_values_device_builds_more_values_in_the_same_graph() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();
        let x = Tensor::<2, f32>::zeros(&device, [2, 3]);
        let y = Tensor::<2, f32>::ones(&x.device(), [2, 3]);
        assert!(GraphRef::ptr_eq(x.graph(), y.graph()));
        assert_eq!(Tensor::add(&x, &y).to_flat(), vec![1.0f32; 6]);
        assert_eq!(x.device().name(), "cpu");
    }

    #[test]
    fn element_dtypes() {
        assert_eq!(<f32 as Element>::DTYPE, Dtype::F32);
        assert_eq!(<half::f16 as Element>::DTYPE, Dtype::F16);
        assert_eq!(<half::bf16 as Element>::DTYPE, Dtype::BF16);
        assert_eq!(<u32 as Element>::DTYPE, Dtype::U32);
        assert_eq!(<i32 as Element>::DTYPE, Dtype::I32);
    }

    #[test]
    fn axis_selectors_resolve() {
        assert_eq!(Axis::<4>::resolve(Minus1), 3);
        assert_eq!(Axis::<4>::resolve(Minus2), 2);
        assert_eq!(Axis::<4>::resolve(2usize), 2);
    }

    #[test]
    fn const_extents_refuses_a_symbolic_dim() {
        let sym = [Dim::Sym(fusor2_ir::shape::SymId(0))];
        assert!(const_extents::<1>(&sym, "test").is_err());
        assert_eq!(const_extents::<2>(&[Dim::Const(2), Dim::Const(3)], "t").unwrap(), [2, 3]);
        assert!(const_extents::<1>(&[Dim::Const(2), Dim::Const(3)], "t").is_err());
    }

    #[allow(dead_code)]
    fn the_dtype_parameter_defaults_to_f32(t: Tensor<2>) -> Tensor<2, f32> {
        t
    }

    /// `into_concrete` and `to_concrete` are the approved identity: the value
    /// they hand back is the *same node*, not a copy of it. If either ever
    /// materialized, the e-graph would see a different id and the fake-quant
    /// `with_backwards` boundary, which is keyed on the node the model handed
    /// it, would stop matching.
    #[test]
    fn the_fusion_erasers_hand_back_the_same_node() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();
        let a = Tensor::<1, f32>::from_slice(&device, [3], &[1.0, 2.0, 3.0]);
        let id = a.id();
        assert_eq!(a.to_concrete().id(), id);
        assert_eq!(a.clone().into_concrete().id(), id);
        // And they are not a `detach` in disguise: detach *does* re-leaf.
        let cut = a.detach();
        assert_ne!(cut.id(), id, "detach must cut the value off its producers");
        assert_eq!(cut.to_flat(), vec![1.0, 2.0, 3.0]);
    }

    /// The trainer's exact convolution turbofish, on values a hand reference
    /// covers.
    ///
    /// `model.rs` writes `x.conv::<3, 1, 4>(&weight.permute([2, 1, 0]), Some(bias),
    /// [kernel / 2], [1])`: a rank-3 input, one spatial axis, a rank-4
    /// windowed view. The kernel here is symmetric, so the value is the same
    /// whether the lowering correlates or convolves — what is under test is
    /// the plumbing and the padding, not a sign convention.
    #[test]
    fn the_trainers_conv_turbofish_computes_a_padded_1d_convolution() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();
        // [batch 1, in_ch 1, seq 4]
        let x = Tensor::<3, f32>::from_slice(&device, [1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);
        // The trainer stores [kernel, in_ch, out_ch] and permutes to
        // [out_ch, in_ch, kernel]; do the same so the permute is covered.
        let w = Tensor::<3, f32>::from_slice(&device, [3, 1, 1], &[1.0, 2.0, 1.0]);
        let bias = Tensor::<1, f32>::from_slice(&device, [1], &[0.5]);
        let kernel = w.shape()[0];

        let y = x.conv::<3, 1, 4>(&w.permute([2, 1, 0]), Some(&bias), [kernel / 2], [1]);

        assert_eq!(y.shape(), [1, 1, 4]);
        // Zero-padded [0,1,2,3,4,0] against [1,2,1], plus 0.5.
        assert_eq!(y.to_flat(), vec![4.5, 8.5, 12.5, 11.5]);
    }

    /// `WINDOWED` is checked arithmetically — it must be `R + DIFF`. A wrong one
    /// names the op and both ranks.
    #[test]
    #[should_panic(expected = "has rank 4, not 5")]
    fn conv_rejects_a_windowed_rank_that_is_not_r_plus_diff() {
        let device = Device::cpu();
        let x = Tensor::<3, f32>::from_slice(&device, [1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let w = Tensor::<3, f32>::from_slice(&device, [1, 1, 3], &[1.0, 2.0, 1.0]);
        let _ = x.conv::<3, 1, 5>(&w, None, [1], [1]);
    }

    /// Rank-changing views take the output rank as an ordinary const
    /// parameter, and each one lands on the rank its call site named.
    #[test]
    fn the_rank_changing_views_land_on_their_output_parameter() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();
        let a = Tensor::<2, f32>::from_slice(&device, [2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let flat: Tensor<1, f32> = a.reshape([6]);
        assert_eq!(flat.shape(), [6]);
        assert_eq!(flat.to_flat(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let cube: Tensor<3, f32> = a.reshape([1, 2, 3]);
        assert_eq!(cube.shape(), [1, 2, 3]);
        let dropped: Tensor<2, f32> = cube.squeeze(0usize);
        assert_eq!(dropped.shape(), [2, 3]);
        let raised: Tensor<3, f32> = a.unsqueeze(1);
        assert_eq!(raised.shape(), [2, 1, 3]);

        let wide: Tensor<3, f32> = raised.broadcast_as([2, 4, 3]);
        assert_eq!(wide.shape(), [2, 4, 3]);
        assert_eq!(a.expand([2, 3]).shape(), [2, 3]);
        assert_eq!(a.flatten_all().shape(), [6]);

        // `chunk` keeps the rank and splits one axis.
        let parts = a.chunk(3, Minus1);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].shape(), [2, 1]);
        assert_eq!(parts[2].to_flat(), vec![3.0, 6.0]);
    }

    /// `Minus1`/`Minus2` are axis *arguments*, not just a `resolve` call: the
    /// ops have to accept them where a `usize` goes.
    #[test]
    fn the_from_the_end_axis_selectors_drive_real_ops() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();
        let a = Tensor::<2, f32>::from_slice(&device, [2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Sum the last axis, then the last of what is left.
        let rows: Tensor<1, f32> = a.sum::<1>(Minus1);
        assert_eq!(rows.to_flat(), vec![6.0, 15.0]);
        let cols: Tensor<1, f32> = a.sum::<1>(Minus2);
        assert_eq!(cols.to_flat(), vec![5.0, 7.0, 9.0]);
        // `transpose(Minus2, Minus1)` is `t()`.
        assert_eq!(a.transpose(Minus2, Minus1).to_flat(), a.t().to_flat());
    }

    /// `cat` keeps the rank, `stack` raises it.
    #[test]
    fn cat_keeps_the_rank_and_stack_raises_it() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();
        let a = Tensor::<1, f32>::from_slice(&device, [2], &[1.0, 2.0]);
        let b = Tensor::<1, f32>::from_slice(&device, [2], &[3.0, 4.0]);

        let joined: Tensor<1, f32> = cat([a.clone(), b.clone()], 0);
        assert_eq!(joined.shape(), [4]);
        assert_eq!(joined.to_flat(), vec![1.0, 2.0, 3.0, 4.0]);
        // The associated-function spelling is the same value.
        assert_eq!(
            Tensor::<1, f32>::cat([a.clone(), b.clone()], 0).to_flat(),
            joined.to_flat()
        );

        let stacked: Tensor<2, f32> = stack([a, b], 0);
        assert_eq!(stacked.shape(), [2, 2]);
        assert_eq!(stacked.to_flat(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// The operand slot is inferred from the argument, so `_` is the only
    /// thing a call site ever writes there.
    #[test]
    fn a_broadcasting_binary_infers_its_operand_slot() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();
        let rows = Tensor::<2, f32>::from_slice(&device, [2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let scale = Tensor::<1, f32>::from_slice(&device, [2], &[10.0, 100.0]);
        let one = Tensor::<1, f32>::from_slice(&device, [1], &[2.0]);

        // The trainer's rank-2 spelling, and the rank-1 one from `optim.rs`.
        let scaled: Tensor<2, f32> = rows.mul_::<1, 2, _>(&scale);
        assert_eq!(scaled.to_flat(), vec![10.0, 200.0, 30.0, 400.0]);
        let halved: Tensor<1, f32> = scale.div_::<1, 1, _>(&one);
        assert_eq!(halved.to_flat(), vec![5.0, 50.0]);
        let lifted: Tensor<2, f32> = rows.add_::<1, 2, _>(&scale);
        assert_eq!(lifted.to_flat(), vec![11.0, 102.0, 13.0, 104.0]);
        assert_eq!(rows.sub_::<1, 2, _>(&scale).shape(), [2, 2]);
        // `pow` goes through exp/log, so it is compared relatively: what is
        // under test is which operand reached which slot, not the last ulp.
        let squared = scale.pow_::<1, 1, _>(&one).to_flat();
        for (got, want) in squared.iter().zip([100.0f32, 10000.0]) {
            assert!((got - want).abs() <= want * 1e-6, "pow_ gave {squared:?}");
        }
    }

    /// All six operator impls — owned/borrowed on either side, and the `f32`
    /// right-hand form — mean the same op. `optim.rs` writes four of them in
    /// one expression.
    #[test]
    fn every_operator_form_is_the_same_op() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();
        let a = Tensor::<1, f32>::from_slice(&device, [2], &[1.0, 2.0]);
        let b = Tensor::<1, f32>::from_slice(&device, [2], &[10.0, 20.0]);

        let want = vec![11.0, 22.0];
        assert_eq!((a.clone() + b.clone()).to_flat(), want);
        assert_eq!((a.clone() + &b).to_flat(), want);
        assert_eq!((&a + b.clone()).to_flat(), want);
        assert_eq!((&a + &b).to_flat(), want);
        assert_eq!((&a + 1.0f32).to_flat(), vec![2.0, 3.0]);
        assert_eq!((a.clone() + 1.0f32).to_flat(), vec![2.0, 3.0]);

        assert_eq!((&b - &a).to_flat(), vec![9.0, 18.0]);
        assert_eq!((&a * &b).to_flat(), vec![10.0, 40.0]);
        assert_eq!((&b / &a).to_flat(), vec![10.0, 10.0]);
        assert_eq!((-&a).to_flat(), vec![-1.0, -2.0]);
        assert_eq!((-a).to_flat(), vec![-1.0, -2.0]);
    }

    /// `ToVec` nests by rank and hands back the value, not a `Result`. Rank 3
    /// is the deepest impl; the trainer reads at rank 0, 1 and 2.
    #[test]
    fn a_readback_nests_as_deep_as_its_rank() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();
        let values: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let a = Tensor::<3, f32>::from_slice(&device, [2, 2, 3], &values);

        let nested: Vec<Vec<Vec<f32>>> = a.read().expect("readback").to_vec();
        assert_eq!(nested.len(), 2);
        assert_eq!(nested[1][1], vec![9.0, 10.0, 11.0]);

        let flat: Vec<f32> = a.reshape([12]).read().expect("readback").to_vec();
        assert_eq!(flat, values);
        let rows: Vec<Vec<f32>> = a.reshape([4, 3]).read().expect("readback").to_vec();
        assert_eq!(rows[3], vec![9.0, 10.0, 11.0]);
        let one: f32 = a
            .reshape([12])
            .narrow(0usize, 5, 1)
            .squeeze::<0>(0usize)
            .read()
            .expect("readback")
            .to_vec();
        assert_eq!(one, 5.0);
        // `as_slice` is the same read behind a ready future.
        assert_eq!(pollster::block_on(a.as_slice()).unwrap().to_flat(), values);
    }

    /// `narrow_acc` undoes the accumulator promotion and *only* that. A dtype
    /// disagreement that is not the promotion falls through untouched, so
    /// `try_from_dyn` reports it instead of it being silently reinterpreted.
    #[test]
    fn narrow_acc_undoes_the_promotion_and_nothing_else() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();

        // f32 result for an f16 operand: this is the promotion, so narrow it.
        let wide = Tensor::<1, f32>::from_slice(&device, [2], &[1.0, 2.0]);
        let narrowed = narrow_acc::<half::f16>(Ok(wide.clone().into_inner())).unwrap();
        assert_eq!(narrowed.dtype(), Dtype::F16);

        // f32 asked for, f32 given: untouched.
        let same = narrow_acc::<f32>(Ok(wide.clone().into_inner())).unwrap();
        assert_eq!(same.id(), wide.id());

        // u32 given where f32 was asked for is not the promotion of anything.
        // It passes through unconverted and the wrapper reports it.
        let ints = Tensor::<1, u32>::from_slice(&device, [2], &[1, 2]);
        let through = narrow_acc::<f32>(Ok(ints.into_inner())).unwrap();
        assert_eq!(through.dtype(), Dtype::U32);
        assert!(Tensor::<1, f32>::try_from_dyn(through).is_err());
    }

    /// A host-data length that disagrees with the shape is a panic naming both.
    #[test]
    #[should_panic(expected = "needs 6 elements, got 5")]
    fn from_slice_reports_a_length_that_disagrees_with_the_shape() {
        let device = Device::cpu();
        let _ = Tensor::<2, f32>::from_slice(&device, [2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    /// `try_from_dyn` reports a rank mismatch rather than panicking; `new` panics
    /// with the same diagnosis. The two differ only on delivery.
    #[test]
    fn try_from_dyn_and_from_dyn_agree_on_diagnosis() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();
        let a = Tensor::<2, f32>::zeros(&device, [2, 3]);
        let raw = a.clone().into_inner();

        let err = Tensor::<3, f32>::try_from_dyn(raw.clone()).unwrap_err();
        assert!(format!("{err}").contains("value has rank 2"), "{err}");
        let err = Tensor::<2, u32>::try_from_dyn(raw.clone()).unwrap_err();
        assert!(format!("{err}").contains("value has dtype"), "{err}");

        // `retype` is the same assertion spelled on a value that already has a
        // wrapper, and a correct one is a no-op.
        assert_eq!(a.clone().retype::<2, f32>().id(), a.id());
        assert_eq!(Tensor::<2, f32>::from_dyn(raw).id(), a.id());
    }
}
