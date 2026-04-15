//! Unified CPU/GPU tensor abstraction
//!
//! This crate provides a unified interface over `fusor-cpu` (CPU tensors with SIMD fusion)
//! and `fusor-core` (GPU tensors with compute graph batching).
//!
//! The key design is:
//! - `Tensor<CpuT, GpuT>` is a runtime dispatch enum holding either CPU or GPU version
//! - CPU kernel fusion is preserved (expression types stay lazy)
//! - GPU laziness is preserved (compute graph batching)

pub mod cache;
mod composite;
mod device;
mod error;
pub mod layers;
pub mod quantized;
mod varbuilder;

pub use varbuilder::{ShardedVarBuilder, VarBuilder};

pub use quantized::{CpuF32Tensor, QMatrix};

use std::ops::{Deref, Range};

pub use composite::{
    MaskKind, RopeCache, ToVec1, ToVec2, ToVec3, arange, arange_step, base_inverse_frequency, cat,
    stack,
};
pub use device::Device;
pub use error::Error;
pub use fusor_types::FromArray;

/// Result type for fusor operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;
use fusor_core::TensorSlice;

// Re-export from fusor-cpu
pub use fusor_cpu::{
    Abs,
    // Op types for bounds
    AbsOp,
    Acos,
    AcosOp,
    Acosh,
    AcoshOp,
    Add,
    AddOp,
    Asin,
    AsinOp,
    Asinh,
    AsinhOp,
    Atan,
    AtanOp,
    Atanh,
    AtanhOp,
    // Cast trait
    CastTo,
    ConcreteTensor,
    Cos,
    CosOp,
    Cosh,
    CoshOp,
    Div,
    DivOp,
    Exp,
    Exp2,
    Exp2Op,
    ExpOp,
    // Float operations
    FloatOps,
    // Conditional operations
    IsNonZero,
    Log,
    Log2,
    Log2Op,
    LogOp,
    MapLayout,
    // Matmul
    MatmulImpl,
    // Reduction ops
    MaxOp,
    MinOp,
    Mul,
    MulOp,
    Neg,
    NegOp,
    ResolvedTensor,
    SimdBinaryOp,
    SimdElement,
    SimdReduceOp,
    SimdUnaryOp,
    Sin,
    SinOp,
    Sinh,
    SinhOp,
    Sqrt,
    SqrtOp,
    Sub,
    SubOp,
    SumOp,
    Tan,
    TanOp,
    Tanh,
    TanhOp,
    Tensor as CpuTensor,
    // Backing trait for generic tensor type parameters
    TensorBacking,
};

pub use fusor_core::Tensor as GpuTensor;

// Re-export from fusor-core for GPU types
pub use fusor_core::{
    CastTensor, D, DataType, Dim, FloatDataType, GgufReadError, LastRank, LastRankInner, MaxRank,
    NextRank, NextRankInner, SmallerRank, WasmNotSend, WasmNotSync,
};

/// Runtime dispatch wrapper - holds either CPU or GPU version of an operation/tensor type.
///
/// This enum enables writing generic code that works with both CPU and GPU tensors
/// while preserving the benefits of each backend:
/// - CPU: Expression types stay lazy and fuse at resolve time
/// - GPU: Operations build a compute graph that batches at resolve time
#[derive(Clone)]
pub enum Tensor<const R: usize, D, B: TensorBacking<R, Elem = D> = fusor_cpu::ConcreteTensor<D, R>>
{
    Cpu(CpuTensor<R, B>),
    Gpu(GpuTensor<R, D>),
}

impl<const R: usize, D, B> Tensor<R, D, B>
where
    B: TensorBacking<R, Elem = D>,
{
    /// Returns true if this is the CPU variant.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        matches!(self, Tensor::Cpu(_))
    }

    /// Returns true if this is the GPU variant.
    #[inline]
    pub fn is_gpu(&self) -> bool {
        matches!(self, Tensor::Gpu(_))
    }

    /// Returns a reference to the CPU tensor if this is the CPU variant.
    #[inline]
    pub fn as_cpu(&self) -> Option<&CpuTensor<R, B>> {
        match self {
            Tensor::Cpu(t) => Some(t),
            _ => None,
        }
    }

    /// Returns a reference to the GPU tensor if this is the GPU variant.
    #[inline]
    pub fn as_gpu(&self) -> Option<&GpuTensor<R, D>> {
        match self {
            Tensor::Gpu(t) => Some(t),
            _ => None,
        }
    }

    /// Returns a mutable reference to the CPU tensor if this is the CPU variant.
    #[inline]
    pub fn as_cpu_mut(&mut self) -> Option<&mut CpuTensor<R, B>> {
        match self {
            Tensor::Cpu(t) => Some(t),
            _ => None,
        }
    }

    /// Returns a mutable reference to the GPU tensor if this is the GPU variant.
    #[inline]
    pub fn as_gpu_mut(&mut self) -> Option<&mut GpuTensor<R, D>> {
        match self {
            Tensor::Gpu(t) => Some(t),
            _ => None,
        }
    }

    /// Returns a mutable reference to the CPU tensor if this is the CPU variant.
    #[inline]
    pub fn to_cpu(self) -> Option<CpuTensor<R, B>> {
        match self {
            Tensor::Cpu(t) => Some(t),
            _ => None,
        }
    }

    /// Returns a mutable reference to the GPU tensor if this is the GPU variant.
    #[inline]
    pub fn to_gpu(self) -> Option<GpuTensor<R, D>> {
        match self {
            Tensor::Gpu(t) => Some(t),
            _ => None,
        }
    }

    /// Unwrap the CPU variant, panicking if this is a GPU tensor.
    #[inline]
    pub fn unwrap_cpu(self) -> CpuTensor<R, B> {
        match self {
            Tensor::Cpu(t) => t,
            Tensor::Gpu(_) => panic!("Expected CPU tensor, found GPU tensor"),
        }
    }

    /// Unwrap the GPU variant, panicking if this is a CPU tensor.
    #[inline]
    pub fn unwrap_gpu(self) -> GpuTensor<R, D> {
        match self {
            Tensor::Gpu(t) => t,
            Tensor::Cpu(_) => panic!("Expected GPU tensor, found CPU tensor"),
        }
    }

    #[inline]
    pub fn dispatch<const R2: usize, D2, B2>(
        self,
        cpu_fn: impl FnOnce(CpuTensor<R, B>) -> CpuTensor<R2, B2>,
        gpu_fn: impl FnOnce(GpuTensor<R, D>) -> GpuTensor<R2, D2>,
    ) -> Tensor<R2, D2, B2>
    where
        B2: TensorBacking<R2, Elem = D2>,
    {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(cpu_fn(t)),
            Tensor::Gpu(t) => Tensor::Gpu(gpu_fn(t)),
        }
    }

    /// Dispatch a single-tensor operation (reference variant).
    #[inline]
    pub fn dispatch_ref<const R2: usize, D2, B2>(
        &self,
        cpu_fn: impl FnOnce(&CpuTensor<R, B>) -> CpuTensor<R2, B2>,
        gpu_fn: impl FnOnce(&GpuTensor<R, D>) -> GpuTensor<R2, D2>,
    ) -> Tensor<R2, D2, B2>
    where
        B2: TensorBacking<R2, Elem = D2>,
    {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(cpu_fn(t)),
            Tensor::Gpu(t) => Tensor::Gpu(gpu_fn(t)),
        }
    }

    /// Dispatch a two-tensor operation to the appropriate backend.
    #[inline]
    pub fn dispatch_pair<const R2: usize, const R3: usize, D2, D3, B2, B3>(
        &self,
        other: &Tensor<R2, D2, B2>,
        cpu_fn: impl FnOnce(&CpuTensor<R, B>, &CpuTensor<R2, B2>) -> CpuTensor<R3, B3>,
        gpu_fn: impl FnOnce(&GpuTensor<R, D>, &GpuTensor<R2, D2>) -> GpuTensor<R3, D3>,
    ) -> Tensor<R3, D3, B3>
    where
        B2: TensorBacking<R2, Elem = D2>,
        B3: TensorBacking<R3, Elem = D3>,
    {
        match (self, other) {
            (Tensor::Cpu(a), Tensor::Cpu(b)) => Tensor::Cpu(cpu_fn(a, b)),
            (Tensor::Gpu(a), Tensor::Gpu(b)) => Tensor::Gpu(gpu_fn(a, b)),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        }
    }

    /// Dispatch a three-tensor operation to the appropriate backend.
    #[inline]
    pub fn dispatch_triple<
        const R2: usize,
        const R3: usize,
        const R4: usize,
        D2,
        D3,
        D4,
        B2,
        B3,
        B4,
    >(
        &self,
        second: &Tensor<R2, D2, B2>,
        third: &Tensor<R3, D3, B3>,
        cpu_fn: impl FnOnce(
            &CpuTensor<R, B>,
            &CpuTensor<R2, B2>,
            &CpuTensor<R3, B3>,
        ) -> CpuTensor<R4, B4>,
        gpu_fn: impl FnOnce(
            &GpuTensor<R, D>,
            &GpuTensor<R2, D2>,
            &GpuTensor<R3, D3>,
        ) -> GpuTensor<R4, D4>,
    ) -> Tensor<R4, D4, B4>
    where
        B2: TensorBacking<R2, Elem = D2>,
        B3: TensorBacking<R3, Elem = D3>,
        B4: TensorBacking<R4, Elem = D4>,
    {
        match (self, second, third) {
            (Tensor::Cpu(a), Tensor::Cpu(b), Tensor::Cpu(c)) => Tensor::Cpu(cpu_fn(a, b, c)),
            (Tensor::Gpu(a), Tensor::Gpu(b), Tensor::Gpu(c)) => Tensor::Gpu(gpu_fn(a, b, c)),
            _ => panic!("All tensors must be on the same device"),
        }
    }

    /// Dispatch a four-tensor operation to the appropriate backend.
    #[inline]
    pub fn dispatch_quad<
        const R2: usize,
        const R3: usize,
        const R4: usize,
        D2,
        D3,
        D4,
        B2,
        B3,
        B4,
    >(
        &self,
        second: &Tensor<R2, D2, B2>,
        third: &Tensor<R3, D3, B3>,
        fourth: &Tensor<R4, D4, B4>,
        cpu_fn: impl FnOnce(
            &CpuTensor<R, B>,
            &CpuTensor<R2, B2>,
            &CpuTensor<R3, B3>,
            &CpuTensor<R4, B4>,
        ) -> CpuTensor<R, ConcreteTensor<D, R>>,
        gpu_fn: impl FnOnce(
            &GpuTensor<R, D>,
            &GpuTensor<R2, D2>,
            &GpuTensor<R3, D3>,
            &GpuTensor<R4, D4>,
        ) -> GpuTensor<R, D>,
    ) -> Tensor<R, D>
    where
        D: SimdElement,
        B2: TensorBacking<R2, Elem = D2>,
        B3: TensorBacking<R3, Elem = D3>,
        B4: TensorBacking<R4, Elem = D4>,
    {
        match (self, second, third, fourth) {
            (Tensor::Cpu(a), Tensor::Cpu(b), Tensor::Cpu(c), Tensor::Cpu(d)) => {
                Tensor::Cpu(cpu_fn(a, b, c, d))
            }
            (Tensor::Gpu(a), Tensor::Gpu(b), Tensor::Gpu(c), Tensor::Gpu(d)) => {
                Tensor::Gpu(gpu_fn(a, b, c, d))
            }
            _ => panic!("All tensors must be on the same device"),
        }
    }

    /// Dispatch a two-tensor binary operation where CPU materializes the result.
    #[inline]
    pub fn dispatch_pair_concrete<const R2: usize, D2, B2>(
        &self,
        other: &Tensor<R2, D2, B2>,
        cpu_fn: impl FnOnce(&CpuTensor<R, B>, &CpuTensor<R2, B2>) -> CpuTensor<R, ConcreteTensor<D, R>>,
        gpu_fn: impl FnOnce(&GpuTensor<R, D>, &GpuTensor<R2, D2>) -> GpuTensor<R, D>,
    ) -> Tensor<R, D>
    where
        D: SimdElement,
        B2: TensorBacking<R2, Elem = D2>,
    {
        match (self, other) {
            (Tensor::Cpu(a), Tensor::Cpu(b)) => Tensor::Cpu(cpu_fn(a, b)),
            (Tensor::Gpu(a), Tensor::Gpu(b)) => Tensor::Gpu(gpu_fn(a, b)),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        }
    }

    /// Dispatch a two-tensor operation that only supports CPU (panics on GPU).
    #[inline]
    pub fn dispatch_cpu_only_pair<B2>(
        &self,
        other: &Tensor<R, D, B2>,
        cpu_fn: impl FnOnce(&CpuTensor<R, B>, &CpuTensor<R, B2>) -> CpuTensor<R, ConcreteTensor<D, R>>,
    ) -> Tensor<R, D>
    where
        D: SimdElement,
        B2: TensorBacking<R, Elem = D>,
    {
        match (self, other) {
            (Tensor::Cpu(a), Tensor::Cpu(b)) => Tensor::Cpu(cpu_fn(a, b)),
            _ => panic!("Tensor-to-tensor comparison is only supported on CPU tensors"),
        }
    }

    pub async fn as_slice(self) -> Result<TensorSlice<R, D, EitherMappedBuffer>, Error>
    where
        B: TensorBacking<R>,
        D: fusor_cpu::SimdElement + DataType,
    {
        match self {
            Tensor::Cpu(t) => Ok(t.as_slice().map_bytes(EitherMappedBuffer::Cpu)),
            Tensor::Gpu(t) => {
                let mapped = t.as_slice().await.map_err(|err| Error::Gpu(err.into()))?;
                Ok(mapped.map_bytes(EitherMappedBuffer::Gpu))
            }
        }
    }

    /// Materialize the tensor to a concrete form.
    ///
    /// For CPU tensors, this evaluates any lazy expressions.
    /// For GPU tensors, this is a no-op as GPU tensors are already concrete.
    pub fn to_concrete(&self) -> Tensor<R, D>
    where
        B: TensorBacking<R>,
        D: SimdElement,
    {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(t.to_concrete()),
            Tensor::Gpu(t) => Tensor::Gpu(t.clone()),
        }
    }

    /// Returns the shape of the tensor.
    pub fn shape(&self) -> [usize; R]
    where
        D: SimdElement + DataType,
    {
        match self {
            Tensor::Cpu(t) => t.shape(),
            Tensor::Gpu(t) => *t.shape(),
        }
    }
}

pub enum EitherMappedBuffer {
    Cpu(fusor_cpu::CpuMappedBuffer),
    Gpu(fusor_core::MappedBuffer),
}

impl Deref for EitherMappedBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            EitherMappedBuffer::Cpu(buf) => buf.deref(),
            EitherMappedBuffer::Gpu(buf) => buf.deref(),
        }
    }
}

/// Macro to implement pairwise operators for Tensor.
///
/// Generates all four combinations of owned/reference implementations:
/// - `Tensor op Tensor` (owned + owned)
/// - `&Tensor op &Tensor` (ref + ref)
/// - `Tensor op &Tensor` (owned + ref)
/// - `&Tensor op Tensor` (ref + owned)
macro_rules! impl_tensor_pairwise_op {
    ($trait:ident, $method:ident, $op:tt, $panic_msg:literal) => {
        // Owned + Owned
        impl<const R: usize, D, B, B2, O> std::ops::$trait<Tensor<R, D, O>> for Tensor<R, D, B>
        where
            CpuTensor<R, B>: std::ops::$trait<CpuTensor<R, O>, Output = CpuTensor<R, B2>>,
            GpuTensor<R, D>: std::ops::$trait<Output = GpuTensor<R, D>>,
            B: TensorBacking<R, Elem = D>,
            O: TensorBacking<R, Elem = D>,
            B2: TensorBacking<R, Elem = D>,
        {
            type Output = Tensor<R, D, B2>;

            fn $method(self, rhs: Tensor<R, D, O>) -> Self::Output {
                match (self, rhs) {
                    (Tensor::Cpu(lhs), Tensor::Cpu(rhs)) => Tensor::Cpu(lhs $op rhs),
                    (Tensor::Gpu(lhs), Tensor::Gpu(rhs)) => Tensor::Gpu(lhs $op rhs),
                    _ => panic!($panic_msg),
                }
            }
        }

        // Ref + Ref
        impl<'a, const R: usize, D, B, B2, O> std::ops::$trait<&'a Tensor<R, D, O>> for &'a Tensor<R, D, B>
        where
            &'a CpuTensor<R, B>: std::ops::$trait<&'a CpuTensor<R, O>, Output = CpuTensor<R, B2>>,
            &'a GpuTensor<R, D>: std::ops::$trait<Output = GpuTensor<R, D>>,
            B: TensorBacking<R, Elem = D>,
            O: TensorBacking<R, Elem = D>,
            B2: TensorBacking<R, Elem = D>,
        {
            type Output = Tensor<R, D, B2>;

            fn $method(self, rhs: &'a Tensor<R, D, O>) -> Self::Output {
                match (self, rhs) {
                    (Tensor::Cpu(lhs), Tensor::Cpu(rhs)) => Tensor::Cpu(lhs $op rhs),
                    (Tensor::Gpu(lhs), Tensor::Gpu(rhs)) => Tensor::Gpu(lhs $op rhs),
                    _ => panic!($panic_msg),
                }
            }
        }

        // Ref + Owned
        impl<'a, const R: usize, D, B, B2, O> std::ops::$trait<Tensor<R, D, O>> for &'a Tensor<R, D, B>
        where
            &'a CpuTensor<R, B>: std::ops::$trait<CpuTensor<R, O>, Output = CpuTensor<R, B2>>,
            &'a GpuTensor<R, D>: std::ops::$trait<GpuTensor<R, D>, Output = GpuTensor<R, D>>,
            B: TensorBacking<R, Elem = D>,
            O: TensorBacking<R, Elem = D>,
            B2: TensorBacking<R, Elem = D>,
        {
            type Output = Tensor<R, D, B2>;

            fn $method(self, rhs: Tensor<R, D, O>) -> Self::Output {
                match (self, rhs) {
                    (Tensor::Cpu(lhs), Tensor::Cpu(rhs)) => Tensor::Cpu(lhs $op rhs),
                    (Tensor::Gpu(lhs), Tensor::Gpu(rhs)) => Tensor::Gpu(lhs $op rhs),
                    _ => panic!($panic_msg),
                }
            }
        }

        // Owned + Ref
        impl<'a, const R: usize, D, B, B2, O> std::ops::$trait<&'a Tensor<R, D, O>> for Tensor<R, D, B>
        where
            CpuTensor<R, B>: std::ops::$trait<&'a CpuTensor<R, O>, Output = CpuTensor<R, B2>>,
            GpuTensor<R, D>: std::ops::$trait<&'a GpuTensor<R, D>, Output = GpuTensor<R, D>>,
            B: TensorBacking<R, Elem = D>,
            O: TensorBacking<R, Elem = D>,
            B2: TensorBacking<R, Elem = D>,
        {
            type Output = Tensor<R, D, B2>;

            fn $method(self, rhs: &'a Tensor<R, D, O>) -> Self::Output {
                match (self, rhs) {
                    (Tensor::Cpu(lhs), Tensor::Cpu(rhs)) => Tensor::Cpu(lhs $op rhs),
                    (Tensor::Gpu(lhs), Tensor::Gpu(rhs)) => Tensor::Gpu(lhs $op rhs),
                    _ => panic!($panic_msg),
                }
            }
        }
    };
}

impl_tensor_pairwise_op!(Add, add, +, "Cannot add CPU tensor to GPU tensor");
impl_tensor_pairwise_op!(Sub, sub, -, "Cannot subtract CPU tensor from GPU tensor");
impl_tensor_pairwise_op!(Mul, mul, *, "Cannot multiply CPU tensor with GPU tensor");
impl_tensor_pairwise_op!(Div, div, /, "Cannot divide CPU tensor by GPU tensor");
impl_tensor_pairwise_op!(Rem, rem, %, "Cannot perform remainder on CPU tensor with GPU tensor");

// Neg trait implementation for Tensor
impl<const R: usize, D, B, B2> std::ops::Neg for Tensor<R, D, B>
where
    CpuTensor<R, B>: std::ops::Neg<Output = CpuTensor<R, B2>>,
    GpuTensor<R, D>: std::ops::Neg<Output = GpuTensor<R, D>>,
    B: TensorBacking<R, Elem = D>,
    B2: TensorBacking<R, Elem = D>,
{
    type Output = Tensor<R, D, B2>;

    fn neg(self) -> Self::Output {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(-t),
            Tensor::Gpu(t) => Tensor::Gpu(-t),
        }
    }
}

impl<'a, const R: usize, D, B, B2> std::ops::Neg for &'a Tensor<R, D, B>
where
    &'a CpuTensor<R, B>: std::ops::Neg<Output = CpuTensor<R, B2>>,
    &'a GpuTensor<R, D>: std::ops::Neg<Output = GpuTensor<R, D>>,
    B: TensorBacking<R, Elem = D>,
    B2: TensorBacking<R, Elem = D>,
{
    type Output = Tensor<R, D, B2>;

    fn neg(self) -> Self::Output {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(-t),
            Tensor::Gpu(t) => Tensor::Gpu(-t),
        }
    }
}

// Tensor * scalar (owned)
impl<const R: usize, D, B, B2> std::ops::Mul<D> for Tensor<R, D, B>
where
    CpuTensor<R, B>: std::ops::Mul<D, Output = CpuTensor<R, B2>>,
    GpuTensor<R, D>: std::ops::Mul<D, Output = GpuTensor<R, D>>,
    B: TensorBacking<R, Elem = D>,
    B2: TensorBacking<R, Elem = D>,
    D: fusor_cpu::Scalar,
{
    type Output = Tensor<R, D, B2>;

    fn mul(self, rhs: D) -> Self::Output {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(t * rhs),
            Tensor::Gpu(t) => Tensor::Gpu(t * rhs),
        }
    }
}

// &Tensor * scalar
impl<'a, const R: usize, D, B, B2> std::ops::Mul<D> for &'a Tensor<R, D, B>
where
    &'a CpuTensor<R, B>: std::ops::Mul<D, Output = CpuTensor<R, B2>>,
    &'a GpuTensor<R, D>: std::ops::Mul<D, Output = GpuTensor<R, D>>,
    B: TensorBacking<R, Elem = D>,
    B2: TensorBacking<R, Elem = D>,
    D: fusor_cpu::Scalar,
{
    type Output = Tensor<R, D, B2>;

    fn mul(self, rhs: D) -> Self::Output {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(t * rhs),
            Tensor::Gpu(t) => Tensor::Gpu(t * rhs),
        }
    }
}

// Tensor + scalar (owned)
impl<const R: usize, D, B, B2> std::ops::Add<D> for Tensor<R, D, B>
where
    CpuTensor<R, B>: std::ops::Add<D, Output = CpuTensor<R, B2>>,
    GpuTensor<R, D>: std::ops::Add<D, Output = GpuTensor<R, D>>,
    B: TensorBacking<R, Elem = D>,
    B2: TensorBacking<R, Elem = D>,
    D: fusor_cpu::Scalar,
{
    type Output = Tensor<R, D, B2>;

    fn add(self, rhs: D) -> Self::Output {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(t + rhs),
            Tensor::Gpu(t) => Tensor::Gpu(t + rhs),
        }
    }
}

// &Tensor + scalar
impl<'a, const R: usize, D, B, B2> std::ops::Add<D> for &'a Tensor<R, D, B>
where
    &'a CpuTensor<R, B>: std::ops::Add<D, Output = CpuTensor<R, B2>>,
    &'a GpuTensor<R, D>: std::ops::Add<D, Output = GpuTensor<R, D>>,
    B: TensorBacking<R, Elem = D>,
    B2: TensorBacking<R, Elem = D>,
    D: fusor_cpu::Scalar,
{
    type Output = Tensor<R, D, B2>;

    fn add(self, rhs: D) -> Self::Output {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(t + rhs),
            Tensor::Gpu(t) => Tensor::Gpu(t + rhs),
        }
    }
}

// Broadcasting binary operations that can work with tensors of different ranks
impl<const R: usize, D, B> Tensor<R, D, B>
where
    D: SimdElement + DataType + Default,
    B: TensorBacking<R, Elem = D>,
{
    /// Broadcasting add: broadcasts both tensors to a common shape and adds them.
    pub fn add_<const R2: usize, const R3: usize, B2>(
        &self,
        second: &Tensor<R2, D, B2>,
    ) -> Tensor<R3, D, ConcreteTensor<D, R3>>
    where
        (fusor_core::Tensor<R, D>, fusor_core::Tensor<R2, D>): fusor_core::MaxRank<R3, D>,
        (ConcreteTensor<D, R>, ConcreteTensor<D, R2>): fusor_cpu::MaxRank<R3, D>,
        D: std::ops::Add<Output = D>,
        AddOp: SimdBinaryOp<D>,
        B2: TensorBacking<R2, Elem = D>,
    {
        self.dispatch_pair(second, |a, b| a.as_ref().add_(b.as_ref()), |a, b| a.add_(b))
    }

    /// Broadcasting subtract: broadcasts both tensors to a common shape and subtracts them.
    pub fn sub_<const R2: usize, const R3: usize, B2>(
        &self,
        second: &Tensor<R2, D, B2>,
    ) -> Tensor<R3, D, ConcreteTensor<D, R3>>
    where
        (fusor_core::Tensor<R, D>, fusor_core::Tensor<R2, D>): fusor_core::MaxRank<R3, D>,
        (ConcreteTensor<D, R>, ConcreteTensor<D, R2>): fusor_cpu::MaxRank<R3, D>,
        D: std::ops::Sub<Output = D>,
        SubOp: SimdBinaryOp<D>,
        B2: TensorBacking<R2, Elem = D>,
    {
        self.dispatch_pair(second, |a, b| a.as_ref().sub_(b.as_ref()), |a, b| a.sub_(b))
    }

    /// Broadcasting multiply: broadcasts both tensors to a common shape and multiplies them.
    pub fn mul_<const R2: usize, const R3: usize, B2>(
        &self,
        second: &Tensor<R2, D, B2>,
    ) -> Tensor<R3, D, ConcreteTensor<D, R3>>
    where
        (fusor_core::Tensor<R, D>, fusor_core::Tensor<R2, D>): fusor_core::MaxRank<R3, D>,
        (ConcreteTensor<D, R>, ConcreteTensor<D, R2>): fusor_cpu::MaxRank<R3, D>,
        D: std::ops::Mul<Output = D>,
        MulOp: SimdBinaryOp<D>,
        B2: TensorBacking<R2, Elem = D>,
    {
        self.dispatch_pair(second, |a, b| a.as_ref().mul_(b.as_ref()), |a, b| a.mul_(b))
    }

    /// Broadcasting divide: broadcasts both tensors to a common shape and divides them.
    pub fn div_<const R2: usize, const R3: usize, B2>(
        &self,
        second: &Tensor<R2, D, B2>,
    ) -> Tensor<R3, D, ConcreteTensor<D, R3>>
    where
        (fusor_core::Tensor<R, D>, fusor_core::Tensor<R2, D>): fusor_core::MaxRank<R3, D>,
        (ConcreteTensor<D, R>, ConcreteTensor<D, R2>): fusor_cpu::MaxRank<R3, D>,
        D: std::ops::Div<Output = D>,
        DivOp: SimdBinaryOp<D>,
        B2: TensorBacking<R2, Elem = D>,
    {
        self.dispatch_pair(second, |a, b| a.as_ref().div_(b.as_ref()), |a, b| a.div_(b))
    }

    /// Broadcasting power: broadcasts both tensors to a common shape and computes power.
    pub fn pow_<const R2: usize, const R3: usize, B2>(
        &self,
        second: &Tensor<R2, D, B2>,
    ) -> Tensor<R3, D, ConcreteTensor<D, R3>>
    where
        (fusor_core::Tensor<R, D>, fusor_core::Tensor<R2, D>): fusor_core::MaxRank<R3, D>,
        (ConcreteTensor<D, R>, ConcreteTensor<D, R2>): fusor_cpu::MaxRank<R3, D>,
        D: FloatDataType + FloatOps,
        B2: TensorBacking<R2, Elem = D>,
    {
        self.dispatch_pair(second, |a, b| a.as_ref().pow_(b.as_ref()), |a, b| a.pow_(b))
    }
}

/// Macro to implement lazy unary element-wise operations for Tensor (any backing type).
macro_rules! impl_tensor_unary_op_lazy {
    ($method:ident, $op:ident, $expr_type:ident) => {
        impl<const R: usize, D, B> Tensor<R, D, B>
        where
            D: SimdElement + DataType + FloatDataType,
            B: TensorBacking<R, Elem = D>,
            fusor_cpu::$op: fusor_cpu::SimdUnaryOp<D>,
        {
            #[doc = concat!("Element-wise ", stringify!($method), " operation (lazy for CPU).")]
            pub fn $method(&self) -> Tensor<R, D, fusor_cpu::$expr_type<D, R, &B>> {
                match self {
                    Tensor::Cpu(t) => Tensor::Cpu(t.as_ref().$method()),
                    Tensor::Gpu(t) => Tensor::Gpu(t.$method()),
                }
            }
        }
    };
}

impl_tensor_unary_op_lazy!(abs, AbsOp, Abs);
impl_tensor_unary_op_lazy!(sqrt, SqrtOp, Sqrt);
impl_tensor_unary_op_lazy!(exp, ExpOp, Exp);
impl_tensor_unary_op_lazy!(exp2, Exp2Op, Exp2);
impl_tensor_unary_op_lazy!(log, LogOp, Log);
impl_tensor_unary_op_lazy!(log2, Log2Op, Log2);
impl_tensor_unary_op_lazy!(sin, SinOp, Sin);
impl_tensor_unary_op_lazy!(cos, CosOp, Cos);
impl_tensor_unary_op_lazy!(tan, TanOp, Tan);
impl_tensor_unary_op_lazy!(tanh, TanhOp, Tanh);
impl_tensor_unary_op_lazy!(asin, AsinOp, Asin);
impl_tensor_unary_op_lazy!(acos, AcosOp, Acos);
impl_tensor_unary_op_lazy!(atan, AtanOp, Atan);
impl_tensor_unary_op_lazy!(sinh, SinhOp, Sinh);
impl_tensor_unary_op_lazy!(cosh, CoshOp, Cosh);
impl_tensor_unary_op_lazy!(asinh, AsinhOp, Asinh);
impl_tensor_unary_op_lazy!(acosh, AcoshOp, Acosh);
impl_tensor_unary_op_lazy!(atanh, AtanhOp, Atanh);

// Approximate exp operations (GPU-optimized, CPU falls back to standard exp)
impl<const R: usize, D, B> Tensor<R, D, B>
where
    D: SimdElement + DataType + FloatDataType + Default,
    B: TensorBacking<R, Elem = D>,
    fusor_cpu::ExpOp: fusor_cpu::SimdUnaryOp<D>,
{
    /// Approximate exp function (faster but less accurate on GPU, exact on CPU).
    /// Uses a polynomial approximation on GPU for better performance.
    pub fn approximate_exp(&self) -> Tensor<R, D> {
        self.dispatch_ref(|t| t.as_ref().exp().to_concrete(), |t| t.appoximate_exp())
    }

    /// Less approximate exp function (medium accuracy/speed tradeoff on GPU, exact on CPU).
    pub fn less_approximate_exp(&self) -> Tensor<R, D> {
        self.dispatch_ref(
            |t| t.as_ref().exp().to_concrete(),
            |t| t.less_appoximate_exp(),
        )
    }
}

// Exact tanh operation
impl<const R: usize, D, B> Tensor<R, D, B>
where
    D: SimdElement + DataType + FloatDataType + Default,
    B: TensorBacking<R, Elem = D>,
    fusor_cpu::TanhOp: fusor_cpu::SimdUnaryOp<D>,
{
    /// Exact tanh using (e^x - e^-x) / (e^x + e^-x).
    /// More accurate but potentially slower than built-in tanh on some platforms.
    pub fn tanh_exact(&self) -> Tensor<R, D> {
        // CPU tanh is already exact - evaluate to concrete
        self.dispatch_ref(|t| t.as_ref().tanh().to_concrete(), |t| t.tanh_exact())
    }
}

// Conditional operation (where_cond)
impl<const R: usize, D, B> Tensor<R, D, B>
where
    D: SimdElement + DataType + Default + IsNonZero,
    B: TensorBacking<R, Elem = D>,
{
    /// Conditional selection: where self != 0, select on_true, else on_false.
    pub fn where_cond<B2, B3>(
        &self,
        on_true: &Tensor<R, D, B2>,
        on_false: &Tensor<R, D, B3>,
    ) -> Tensor<R, D>
    where
        B2: TensorBacking<R, Elem = D>,
        B3: TensorBacking<R, Elem = D>,
    {
        self.dispatch_triple(
            on_true,
            on_false,
            |c, t, f| c.as_ref().where_cond(t.as_ref(), f.as_ref()),
            |c, t, f| c.clone().where_cond(t, f),
        )
    }
}

// Float operations (pow_scalar, max_scalar, min_scalar, clamp)
impl<const R: usize, D, B> Tensor<R, D, B>
where
    D: SimdElement + DataType + FloatDataType + FloatOps + Default,
    B: TensorBacking<R, Elem = D>,
{
    /// Raise each element to a power.
    pub fn pow_scalar(&self, exponent: D) -> Tensor<R, D> {
        self.dispatch_ref(
            |t| t.as_ref().pow_scalar(exponent),
            |t| t.pow_elementwise(exponent),
        )
    }

    /// Element-wise maximum with a scalar.
    pub fn max_scalar(&self, scalar: D) -> Tensor<R, D> {
        self.dispatch_ref(
            |t| t.as_ref().max_scalar(scalar),
            |t| t.max_elementwise(scalar),
        )
    }

    /// Element-wise minimum with a scalar.
    pub fn min_scalar(&self, scalar: D) -> Tensor<R, D> {
        self.dispatch_ref(
            |t| t.as_ref().min_scalar(scalar),
            |t| t.min_elementwise(scalar),
        )
    }

    /// Clamp each element to a range [min, max].
    pub fn clamp(&self, min: D, max: D) -> Tensor<R, D> {
        self.dispatch_ref(
            |t| t.as_ref().clamp(min, max),
            |t| t.max_elementwise(min).min_elementwise(max),
        )
    }

    /// Raise each element to a power (alias for pow_scalar for fusor-core API compatibility).
    pub fn pow_elementwise(&self, exponent: D) -> Tensor<R, D> {
        self.pow_scalar(exponent)
    }

    /// Element-wise maximum with a scalar (alias for max_scalar for fusor-core API compatibility).
    pub fn max_elementwise(&self, element: D) -> Tensor<R, D> {
        self.max_scalar(element)
    }

    /// Element-wise minimum with a scalar (alias for min_scalar for fusor-core API compatibility).
    pub fn min_elementwise(&self, element: D) -> Tensor<R, D> {
        self.min_scalar(element)
    }

    /// Add a scalar to each element.
    pub fn add_scalar(&self, scalar: D) -> Tensor<R, D>
    where
        D: std::ops::Add<Output = D>,
        AddOp: SimdBinaryOp<D>,
    {
        self.dispatch_ref(
            |t| t.as_ref().add_scalar(scalar).to_concrete(),
            |t| t.clone() + scalar,
        )
    }

    /// Subtract a scalar from each element.
    pub fn sub_scalar(&self, scalar: D) -> Tensor<R, D>
    where
        D: std::ops::Sub<Output = D>,
        SubOp: SimdBinaryOp<D>,
    {
        self.dispatch_ref(
            |t| t.as_ref().sub_scalar(scalar).to_concrete(),
            |t| t.clone() - scalar,
        )
    }

    /// Multiply each element by a scalar.
    pub fn mul_scalar(&self, scalar: D) -> Tensor<R, D>
    where
        D: std::ops::Mul<Output = D>,
        MulOp: SimdBinaryOp<D>,
    {
        self.dispatch_ref(
            |t| t.as_ref().mul_scalar(scalar).to_concrete(),
            |t| t.clone() * scalar,
        )
    }

    /// Divide each element by a scalar.
    pub fn div_scalar(&self, scalar: D) -> Tensor<R, D>
    where
        D: std::ops::Div<Output = D>,
        DivOp: SimdBinaryOp<D>,
    {
        self.dispatch_ref(
            |t| t.as_ref().div_scalar(scalar).to_concrete(),
            |t| t.clone() / scalar,
        )
    }
}

// Cast operation
impl<const R: usize, D, B> Tensor<R, D, B>
where
    D: SimdElement + DataType + Default,
    B: TensorBacking<R, Elem = D>,
{
    /// Cast tensor to another element type.
    pub fn cast<D2>(&self) -> Tensor<R, D2, ConcreteTensor<D2, R>>
    where
        D: CastTo<D2> + fusor_core::CastTensor<D2>,
        D2: SimdElement + DataType + Default,
    {
        self.dispatch_ref(|t| t.as_ref().cast(), |t| t.cast())
    }
}

// Index select operation
impl<const R: usize, D, B> Tensor<R, D, B>
where
    D: SimdElement + DataType + Default,
    B: TensorBacking<R, Elem = D>,
{
    /// Select elements along a dimension using indices.
    pub fn index_select<B2>(&self, dimension: usize, indices: &Tensor<1, u32, B2>) -> Tensor<R, D>
    where
        B2: TensorBacking<1, Elem = u32>,
    {
        self.dispatch_pair_concrete(
            indices,
            |t, idx| t.as_ref().index_select(dimension, idx.as_ref()),
            |t, idx| t.index_select(dimension, idx),
        )
    }
}

// Slice assign operation
impl<const R: usize, D, B> Tensor<R, D, B>
where
    D: SimdElement + DataType + Default,
    B: TensorBacking<R, Elem = D>,
{
    /// Returns a new tensor with the slice region replaced by values from the value tensor.
    pub fn slice_assign<B2>(
        &self,
        slices: [Range<usize>; R],
        value: &Tensor<R, D, B2>,
    ) -> Tensor<R, D>
    where
        B2: TensorBacking<R, Elem = D>,
    {
        let slices_clone = slices.clone();
        self.dispatch_pair(
            value,
            |t, v| t.as_ref().slice_assign(slices, v.as_ref()),
            |t, v| t.slice_assign(slices_clone, v),
        )
    }
}

// Matrix multiplication for N-dimensional tensors (N >= 2)
impl<const R: usize, D, B> Tensor<R, D, B>
where
    D: SimdElement + DataType + FloatDataType + Default + MatmulImpl,
    B: TensorBacking<R, Elem = D>,
{
    /// Matrix multiplication (batched for rank > 2)
    /// For 2D: [M, K] @ [K, N] -> [M, N]
    /// For ND: [...batch, M, K] @ [...batch, K, N] -> [...batch, M, N]
    /// Panics if R < 2
    pub fn matmul<B2>(&self, rhs: &Tensor<R, D, B2>) -> Tensor<R, D>
    where
        B2: TensorBacking<R, Elem = D>,
    {
        self.dispatch_pair(
            rhs,
            |a, b| a.as_ref().matmul(b.as_ref()),
            |a, b| a.mat_mul(b),
        )
    }

    /// Alias for matmul (for API compatibility with fusor-core)
    pub fn mat_mul<B2>(&self, rhs: &Tensor<R, D, B2>) -> Tensor<R, D>
    where
        B2: TensorBacking<R, Elem = D>,
    {
        self.matmul(rhs)
    }
}

// Quantized matrix multiplication for Tensor<R, f32>
impl<const R: usize, B> Tensor<R, f32, B>
where
    B: TensorBacking<R, Elem = f32> + TensorBacking<R>,
{
    /// Quantized matrix multiplication: self @ weights where weights is quantized.
    ///
    /// Computes `self @ weights` where `self` is an f32 tensor and `weights` is a
    /// quantized 2D tensor. This is optimized for the case where weights are stored
    /// in quantized format (e.g., from GGUF model files).
    ///
    /// # Arguments
    /// * `weights` - A quantized weight matrix (must be 2D)
    ///
    /// # Panics
    /// * If attempting to mix CPU and GPU tensors (self on CPU, weights on GPU or vice versa)
    /// * If R < 2 (matrix multiplication requires at least 2 dimensions)
    /// * If weights is not 2D
    pub fn q_mat_mul(&self, weights: &crate::QMatrix) -> Tensor<R, f32> {
        use crate::QMatrix;

        assert_eq!(
            weights.shape().len(),
            2,
            "q_mat_mul requires 2D weight tensor, got {}D",
            weights.shape().len()
        );

        match (self, weights) {
            // CPU path - dispatch based on block type
            // eval() returns Tensor<R, ConcreteTensor>, so we need .inner() to get ConcreteTensor
            (Tensor::Cpu(lhs), QMatrix::CpuQ4_0(rhs)) => Tensor::Cpu(fusor_cpu::Tensor::new(
                lhs.to_concrete().inner().q_mat_mul(rhs),
            )),
            (Tensor::Cpu(lhs), QMatrix::CpuQ5_0(rhs)) => Tensor::Cpu(fusor_cpu::Tensor::new(
                lhs.to_concrete().inner().q_mat_mul(rhs),
            )),
            (Tensor::Cpu(lhs), QMatrix::CpuQ8_0(rhs)) => Tensor::Cpu(fusor_cpu::Tensor::new(
                lhs.to_concrete().inner().q_mat_mul(rhs),
            )),
            (Tensor::Cpu(lhs), QMatrix::CpuQ4K(rhs)) => Tensor::Cpu(fusor_cpu::Tensor::new(
                lhs.to_concrete().inner().q_mat_mul(rhs),
            )),
            (Tensor::Cpu(lhs), QMatrix::CpuQ5K(rhs)) => Tensor::Cpu(fusor_cpu::Tensor::new(
                lhs.to_concrete().inner().q_mat_mul(rhs),
            )),
            (Tensor::Cpu(lhs), QMatrix::CpuQ6K(rhs)) => Tensor::Cpu(fusor_cpu::Tensor::new(
                lhs.to_concrete().inner().q_mat_mul(rhs),
            )),
            // F32 is not quantized, use regular matmul with transpose
            // Weight is [N, K] (out_features, in_features), we need input @ weight.T
            (Tensor::Cpu(lhs), QMatrix::CpuF32(t)) => {
                let shape = t.shape();
                let n = shape[0]; // out_features
                let k = shape[1]; // in_features

                // Create 2D ConcreteTensor from data and shape
                let rhs_concrete: fusor_cpu::ConcreteTensor<f32, 2> =
                    fusor_cpu::ConcreteTensor::from_parts(
                        fusor_cpu::Layout::contiguous(&[n, k]),
                        t.data().clone(),
                    );

                // Transpose weight from [N, K] to [K, N]
                let rhs_tensor = fusor_cpu::Tensor::new(rhs_concrete);
                let rhs_transposed = rhs_tensor.transpose(0, 1);

                // Reshape to R dimensions: [1, 1, ..., K, N]
                let weight_shape: [usize; R] = std::array::from_fn(|i| {
                    if i < R - 2 {
                        1 // Broadcast batch dimensions
                    } else if i == R - 2 {
                        k // K dimension
                    } else {
                        n // N dimension
                    }
                });
                let rhs_broadcast = rhs_transposed.reshape(weight_shape);

                // Do regular matmul
                let lhs_eval = lhs.to_concrete();
                let result = lhs_eval.matmul(rhs_broadcast);
                Tensor::Cpu(result)
            }

            // F16 is not quantized, convert to f32 and use regular matmul with transpose
            (Tensor::Cpu(lhs), QMatrix::CpuF16(t)) => {
                let shape = t.shape();
                let n = shape[0]; // out_features
                let k = shape[1]; // in_features

                // Convert f16 data to f32
                let f32_data: Vec<f32> = t.data().iter().map(|v| v.to_f32()).collect();
                let mut data = fusor_cpu::AVec::<f32>::with_capacity(64, f32_data.len());
                data.extend_from_slice(&f32_data);

                // Create 2D ConcreteTensor from data and shape
                let rhs_concrete: fusor_cpu::ConcreteTensor<f32, 2> =
                    fusor_cpu::ConcreteTensor::from_parts(
                        fusor_cpu::Layout::contiguous(&[n, k]),
                        data.into_boxed_slice(),
                    );

                // Transpose weight from [N, K] to [K, N]
                let rhs_tensor = fusor_cpu::Tensor::new(rhs_concrete);
                let rhs_transposed = rhs_tensor.transpose(0, 1);

                // Reshape to R dimensions: [1, 1, ..., K, N]
                let weight_shape: [usize; R] = std::array::from_fn(|i| {
                    if i < R - 2 {
                        1 // Broadcast batch dimensions
                    } else if i == R - 2 {
                        k // K dimension
                    } else {
                        n // N dimension
                    }
                });
                let rhs_broadcast = rhs_transposed.reshape(weight_shape);

                // Do regular matmul
                let lhs_eval = lhs.to_concrete();
                let result = lhs_eval.matmul(rhs_broadcast);
                Tensor::Cpu(result)
            }

            // GPU path
            (Tensor::Gpu(lhs), QMatrix::Gpu(rhs)) => Tensor::Gpu(lhs.q_mat_mul(rhs)),

            // Mixed - panic
            _ => panic!("Cannot mix CPU and GPU tensors in q_mat_mul"),
        }
    }
}

// Flatten operations
impl<const R: usize, D, B> Tensor<R, D, B>
where
    D: SimdElement + DataType + Default,
    B: TensorBacking<R, Elem = D>,
{
    /// Flatten the last FROM_END+1 dimensions into one.
    ///
    /// This follows the GPU/fusor-core semantic where FROM_END is the number of
    /// extra dimensions beyond the one being flattened into.
    /// So FROM_END=0 flattens just the last dimension (no-op),
    /// FROM_END=1 flattens the last 2 dimensions, etc.
    ///
    /// Output rank R2 = R - FROM_END.
    pub fn flatten_last_n<const FROM_END: usize, const R2: usize>(
        &self,
    ) -> Tensor<R2, D, ConcreteTensor<D, R2>>
    where
        fusor_core::Tensor<R, D>: fusor_core::SmallerRank<FROM_END, R2, D>,
    {
        match self {
            Tensor::Cpu(t) => {
                // CPU flatten_last_n takes N where output = input - N + 1
                // So we need CPU_N = FROM_END + 1 to match GPU semantics
                // Calculate new shape manually since we can't do const arithmetic
                let shape = t.shape();
                let new_shape: [usize; R2] = std::array::from_fn(|i| {
                    if i < R - 1 - FROM_END {
                        shape[i]
                    } else if i == R - 1 - FROM_END {
                        shape[R - 1 - FROM_END..].iter().product()
                    } else {
                        1
                    }
                });
                Tensor::Cpu(t.as_ref().reshape(new_shape).to_concrete())
            }
            Tensor::Gpu(t) => Tensor::Gpu(t.flatten_last_n::<FROM_END, R2>()),
        }
    }

    /// Flatten the first FROM_START+1 dimensions into one.
    ///
    /// This follows the GPU/fusor-core semantic where FROM_START is the number of
    /// extra dimensions beyond the one being flattened into.
    /// So FROM_START=0 flattens just the first dimension (no-op),
    /// FROM_START=1 flattens the first 2 dimensions, etc.
    ///
    /// Output rank R2 = R - FROM_START.
    pub fn flatten_first_n<const FROM_START: usize, const R2: usize>(
        &self,
    ) -> Tensor<R2, D, ConcreteTensor<D, R2>>
    where
        fusor_core::Tensor<R, D>: fusor_core::SmallerRank<FROM_START, R2, D>,
    {
        match self {
            Tensor::Cpu(t) => {
                // Calculate new shape: first element is product of first FROM_START+1 dims
                // remaining elements are the rest of the dimensions
                let shape = t.shape();
                let new_shape: [usize; R2] = std::array::from_fn(|i| {
                    if i == 0 {
                        shape[..=FROM_START].iter().product()
                    } else {
                        shape[i + FROM_START]
                    }
                });
                Tensor::Cpu(t.as_ref().reshape(new_shape).to_concrete())
            }
            Tensor::Gpu(t) => Tensor::Gpu(t.flatten_first_n::<FROM_START, R2>()),
        }
    }
}

// Device accessor
impl<const R: usize, D, B: TensorBacking<R, Elem = D>> Tensor<R, D, B>
where
    D: SimdElement + DataType,
{
    /// Get the device this tensor is on.
    pub fn device(&self) -> Device {
        match self {
            Tensor::Cpu(_) => Device::Cpu,
            Tensor::Gpu(t) => Device::Gpu(t.device().clone()),
        }
    }

    /// Returns the rank (number of dimensions) of the tensor.
    ///
    /// This is a const function that returns the compile-time rank R.
    #[inline]
    pub const fn rank(&self) -> usize {
        R
    }
}

// Scalar conversion
impl<const R: usize, D, B: TensorBacking<R, Elem = D>> Tensor<R, D, B>
where
    D: SimdElement + DataType + Default + Copy,
    B: TensorBacking<R>,
{
    /// Convert a scalar tensor (or get the first element) to a scalar value.
    ///
    /// This is an async operation because GPU tensors need to be mapped to CPU memory.
    pub async fn to_scalar(&self) -> Result<D, Error> {
        match self {
            Tensor::Cpu(t) => {
                let slice = t.as_ref().as_slice();
                Ok(slice.as_scalar())
            }
            Tensor::Gpu(t) => {
                let result = t.to_scalar().await.map_err(|e| Error::Gpu(e.into()))?;
                Ok(result)
            }
        }
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_gpu_or_add() {
    let a_cpu: CpuTensor<1, fusor_cpu::ConcreteTensor<f32, 1>> =
        fusor_cpu::Tensor::from_slice([3], &[1.0, 2.0, 3.0]);
    let b_cpu: CpuTensor<1, fusor_cpu::ConcreteTensor<f32, 1>> =
        fusor_cpu::Tensor::from_slice([3], &[4.0, 5.0, 6.0]);
    let device = fusor_core::Device::new().await.unwrap();
    let a_gpu: GpuTensor<1, f32> = GpuTensor::new(&device, &[1.0, 2.0, 3.0]);
    let b_gpu: GpuTensor<1, f32> = GpuTensor::new(&device, &[4.0, 5.0, 6.0]);

    let a_cpu_or: Tensor<1, f32> = Tensor::Cpu(a_cpu);
    let b_cpu_or: Tensor<1, f32> = Tensor::Cpu(b_cpu);
    let a_gpu_or: Tensor<1, f32> = Tensor::Gpu(a_gpu);
    let b_gpu_or: Tensor<1, f32> = Tensor::Gpu(b_gpu);

    let c_cpu_or = (&a_cpu_or + &b_cpu_or) * &b_cpu_or;
    println!("c_cpu_or: {:?}", c_cpu_or.as_slice().await.unwrap());
    let c_gpu_or = (&a_gpu_or + &b_gpu_or) * &b_gpu_or;
    println!("c_gpu_or: {:?}", c_gpu_or.as_slice().await.unwrap());
}

#[cfg(test)]
#[tokio::test]
#[allow(clippy::identity_op, clippy::useless_conversion)]
async fn test_matmul_cpu_vs_gpu() {
    // Create random-ish data for matmul test
    // Simulating attention: Q @ K^T with shape [batch, heads, seq_len, head_dim]
    let a_data: Vec<f32> = (0..1 * 8 * 100 * 64)
        .map(|i| (i as f32 * 0.001).sin())
        .collect();
    let b_data: Vec<f32> = (0..1 * 8 * 64 * 100)
        .map(|i| (i as f32 * 0.001).cos())
        .collect();

    // CPU version: [1, 8, 100, 64] @ [1, 8, 64, 100] -> [1, 8, 100, 100]
    let cpu_a: Tensor<4, f32> =
        Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 8, 100, 64], &a_data));
    let cpu_b: Tensor<4, f32> =
        Tensor::Cpu(fusor_cpu::Tensor::from_slice([1, 8, 64, 100], &b_data));
    let cpu_result = cpu_a.matmul(&cpu_b);
    let cpu_slice = cpu_result.as_slice().await.unwrap();

    // GPU version
    let gpu_device = Device::new().await.expect("GPU required for this test");
    let gpu_a: Tensor<4, f32> = Tensor::from_slice(&gpu_device, [1, 8, 100, 64], &a_data);
    let gpu_b: Tensor<4, f32> = Tensor::from_slice(&gpu_device, [1, 8, 64, 100], &b_data);
    let gpu_result = gpu_a.matmul(&gpu_b);
    let gpu_slice = gpu_result.as_slice().await.unwrap();

    // Compare
    assert_eq!(cpu_slice.shape(), gpu_slice.shape());
    assert_eq!(cpu_slice.shape(), &[1, 8, 100, 100]);

    let mut max_diff = 0.0f32;
    let mut sum_diff = 0.0f32;
    let mut count = 0;
    for i in 0..cpu_slice.shape()[0] {
        for j in 0..cpu_slice.shape()[1] {
            for k in 0..cpu_slice.shape()[2].min(50) {
                for l in 0..cpu_slice.shape()[3].min(50) {
                    let cpu_val: f32 = cpu_slice[[i, j, k, l]].into();
                    let gpu_val: f32 = gpu_slice[[i, j, k, l]].into();
                    let diff = (cpu_val - gpu_val).abs();
                    max_diff = max_diff.max(diff);
                    sum_diff += diff;
                    count += 1;
                }
            }
        }
    }

    eprintln!(
        "Matmul CPU vs GPU: max_diff={}, mean_diff={}",
        max_diff,
        sum_diff / count as f32
    );
    eprintln!(
        "CPU[0,0,0,0..5]: {:?}",
        (0..5)
            .map(|i| cpu_slice[[0, 0, 0, i]])
            .collect::<Vec<f32>>()
    );
    eprintln!(
        "GPU[0,0,0,0..5]: {:?}",
        (0..5)
            .map(|i| gpu_slice[[0, 0, 0, i]])
            .collect::<Vec<f32>>()
    );

    assert!(
        max_diff < 0.001,
        "Matmul CPU and GPU outputs differ too much: max_diff={}",
        max_diff
    );
}
