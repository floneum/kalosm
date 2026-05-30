//! Unified CPU/GPU tensor abstraction
//!
//! This crate provides a unified interface over `fusor-cpu` (CPU tensors with SIMD fusion)
//! and `fusor-core` (GPU tensors with compute graph batching).
//!
//! The key design is:
//! - `Tensor<R, D, B>` is a typed runtime dispatch enum holding either CPU or GPU storage
//! - CPU kernel fusion is preserved (expression types stay lazy)
//! - GPU laziness is preserved (compute graph batching)

#[cfg(not(any(feature = "cpu", feature = "gpu")))]
compile_error!("fusor requires at least one backend feature: `cpu` or `gpu`.");

pub mod cache;
mod composite;
mod cpu;
mod device;
mod error;
pub mod fusion;
mod gpu;
pub mod layers;
pub mod quantized;
mod varbuilder;

pub use varbuilder::{ShardedVarBuilder, VarBuilder};

pub use quantized::{CpuF32Tensor, QMatrix};

use std::ops::{Deref, Range};

pub use composite::{
    MaskKind, RopeCache, ToVec, ToVec1, ToVec2, ToVec3, arange, arange_step,
    base_inverse_frequency, cat, stack,
};
pub use device::Device;
pub use error::Error;
pub use fusion::{Concrete, Fusion};
pub use fusor_types::{D, Dim, FromArray, Layout, StrideSpec};

#[cfg(test)]
pub(crate) async fn gpu_device_for_test() -> Option<Device> {
    match Device::new().await {
        Ok(device) => Some(device),
        Err(err) => {
            eprintln!("skipping GPU-only test: {err}");
            None
        }
    }
}

/// Result type for fusor operations.
pub type Result<T, E = Error> = std::result::Result<T, E>;
use fusor_types::TensorSlice;

pub trait Element: crate::cpu::SimdElement + crate::gpu::DataType {}
impl<T> Element for T where T: crate::cpu::SimdElement + crate::gpu::DataType {}

pub trait FloatElement: Element + crate::cpu::FloatOps + crate::gpu::FloatDataType {}
impl<T> FloatElement for T where T: Element + crate::cpu::FloatOps + crate::gpu::FloatDataType {}

pub trait CastElement<T>: crate::cpu::CastTo<T> + crate::gpu::CastTensor<T> {}
impl<S, T> CastElement<T> for S where S: crate::cpu::CastTo<T> + crate::gpu::CastTensor<T> {}

pub trait MatmulElement: Element + crate::cpu::MatmulImpl {}
impl<T> MatmulElement for T where T: Element + crate::cpu::MatmulImpl {}

#[allow(unused_imports)]
pub use crate::cpu::{
    AbsOp, AcosOp, AcoshOp, AddOp, AsinOp, AsinhOp, AtanOp, AtanhOp, BlockQ4_0, BlockQ4K,
    BlockQ5_0, BlockQ5K, BlockQ6K, BlockQ8_0, CastTo, CosOp, CoshOp, DivOp, Exp2Op, ExpOp,
    FloatOps, GgmlType, GgufBlock, IsNonZero, Log2Op, LogOp, MatmulImpl, MaxOp, MinOp, MulOp,
    NegOp, QuantizedTensor, RemOp, SimdBinaryOp, SimdElement, SimdReduceOp, SimdUnaryOp, SinOp,
    SinhOp, SqrtOp, SubOp, SumOp, TanOp, TanhOp,
};

#[allow(unused_imports)]
pub(crate) use crate::cpu::{
    Abs, Acos, Acosh, Add, ConcreteTensor, Cos, Cosh, Div, Exp, Exp2, Log, Log2, MapLayout, Mul,
    Neg, Rem, ResolvedTensor, Sin, Sinh, Sqrt, Sub, Tan, Tanh, TypedTensor as CpuTensor,
};

pub(crate) use crate::fusion::BackendFusion as TensorBacking;
pub(crate) use crate::gpu::Tensor as GpuTensor;

#[allow(unused_imports)]
pub use crate::gpu::{
    CastTensor, DataType, FloatDataType, GgufReadError, NodeIndex, WasmNotSend, WasmNotSync,
};

pub use crate::gpu::{
    GpuMirostat2Sampler as Mirostat2Sampler, GpuMirostat2SamplerParams as Mirostat2SamplerParams,
};

/// Runtime dispatch wrapper - holds either CPU or GPU version of an operation/tensor type.
///
/// This enum enables writing generic code that works with both CPU and GPU tensors
/// while preserving the benefits of each backend:
/// - CPU: Expression types stay lazy and fuse at resolve time
/// - GPU: Operations build a compute graph that batches at resolve time
#[derive(Clone)]
pub enum Tensor<const R: usize, D, B: Fusion<R, D> = crate::fusion::Concrete<D, R>> {
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

    pub async fn as_slice(&self) -> Result<TensorSlice<R, D, EitherMappedBuffer>, Error>
    where
        B: TensorBacking<R>,
        D: crate::cpu::SimdElement + DataType,
    {
        match self {
            Tensor::Cpu(t) => Ok(t.as_slice().map_bytes(EitherMappedBuffer::Cpu)),
            Tensor::Gpu(t) => {
                let mapped = t.as_slice().await.map_err(Error::Gpu)?;
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

impl<B> Tensor<1, f32, B>
where
    B: TensorBacking<1, Elem = f32>,
{
    /// Return the top-k token ids and values sorted by descending logit.
    pub async fn top_k_pairs(&self, k: usize) -> Result<Vec<(u32, f32)>, Error> {
        match self {
            Tensor::Cpu(t) => {
                if k == 0 {
                    return Ok(Vec::new());
                }
                let values = t.as_slice();
                let mut top = Vec::<(u32, f32)>::with_capacity(k);
                for (token_id, logit) in values.as_slice().iter().copied().enumerate() {
                    if !logit.is_finite() {
                        continue;
                    }
                    if top.len() == k {
                        let Some((last_token_id, last_logit)) = top.last().copied() else {
                            continue;
                        };
                        if logit > last_logit
                            || (logit == last_logit && token_id as u32 > last_token_id)
                        {
                            top.truncate(k - 1);
                        } else {
                            continue;
                        }
                    }
                    let token_id = token_id as u32;
                    let insert = top.partition_point(|(existing_id, value)| {
                        *value > logit || (*value == logit && *existing_id > token_id)
                    });
                    top.insert(insert, (token_id, logit));
                }
                Ok(top)
            }
            Tensor::Gpu(t) => {
                let (ids, values) = t.top_k_pairs(k).await.map_err(Error::Gpu)?;
                Ok(ids.into_iter().zip(values).collect())
            }
        }
    }

    pub async fn sample_mirostat2_token(
        &self,
        sampler: &mut Mirostat2Sampler,
        previous_tokens: &[u32],
        params: Mirostat2SamplerParams,
    ) -> Result<u32, Error> {
        match self {
            Tensor::Cpu(_) => {
                let top = self.top_k_pairs(params.top_k).await?;
                Ok(top
                    .first()
                    .map(|(token_id, _)| *token_id)
                    .unwrap_or_default())
            }
            Tensor::Gpu(t) => t
                .sample_mirostat2_token(sampler, previous_tokens, params)
                .await
                .map_err(Error::Gpu),
        }
    }

    pub async fn try_sample_mirostat2_token_q_mat(
        &self,
        weights: &crate::QMatrix,
        sampler: &mut Mirostat2Sampler,
        previous_tokens: &[u32],
        params: Mirostat2SamplerParams,
    ) -> Result<Option<u32>, Error> {
        match (self, weights) {
            (Tensor::Gpu(t), crate::QMatrix::Gpu(weights)) => t
                .try_sample_mirostat2_token_q_mat(weights, sampler, previous_tokens, params)
                .await
                .map_err(Error::Gpu),
            _ => Ok(None),
        }
    }
}

pub enum EitherMappedBuffer {
    Cpu(crate::cpu::CpuMappedBuffer),
    Gpu(crate::gpu::MappedBuffer),
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

/// Macro to implement a unary operator (e.g. `Neg`) for both owned and
/// borrowed `Tensor`. Each call site previously hand-rolled a 16-line
/// `match { Cpu(t) => Cpu(op t), Gpu(t) => Gpu(op t) }` impl twice; this
/// macro emits both.
macro_rules! impl_tensor_unary_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<const R: usize, D, B, B2> std::ops::$trait for Tensor<R, D, B>
        where
            CpuTensor<R, B>: std::ops::$trait<Output = CpuTensor<R, B2>>,
            GpuTensor<R, D>: std::ops::$trait<Output = GpuTensor<R, D>>,
            B: TensorBacking<R, Elem = D>,
            B2: TensorBacking<R, Elem = D>,
        {
            type Output = Tensor<R, D, B2>;

            fn $method(self) -> Self::Output {
                match self {
                    Tensor::Cpu(t) => Tensor::Cpu($op t),
                    Tensor::Gpu(t) => Tensor::Gpu($op t),
                }
            }
        }

        impl<'a, const R: usize, D, B, B2> std::ops::$trait for &'a Tensor<R, D, B>
        where
            &'a CpuTensor<R, B>: std::ops::$trait<Output = CpuTensor<R, B2>>,
            &'a GpuTensor<R, D>: std::ops::$trait<Output = GpuTensor<R, D>>,
            B: TensorBacking<R, Elem = D>,
            B2: TensorBacking<R, Elem = D>,
        {
            type Output = Tensor<R, D, B2>;

            fn $method(self) -> Self::Output {
                match self {
                    Tensor::Cpu(t) => Tensor::Cpu($op t),
                    Tensor::Gpu(t) => Tensor::Gpu($op t),
                }
            }
        }
    };
}

impl_tensor_unary_op!(Neg, neg, -);

/// Macro to implement a `Tensor op Scalar` operator for both owned and
/// borrowed `Tensor`. Each `Add<D>`/`Mul<D>` previously needed two ~17-line
/// impls; this macro produces both with the same dispatch body.
macro_rules! impl_tensor_scalar_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<const R: usize, D, B, B2> std::ops::$trait<D> for Tensor<R, D, B>
        where
            CpuTensor<R, B>: std::ops::$trait<D, Output = CpuTensor<R, B2>>,
            GpuTensor<R, D>: std::ops::$trait<D, Output = GpuTensor<R, D>>,
            B: TensorBacking<R, Elem = D>,
            B2: TensorBacking<R, Elem = D>,
            D: crate::cpu::Scalar,
        {
            type Output = Tensor<R, D, B2>;

            fn $method(self, rhs: D) -> Self::Output {
                match self {
                    Tensor::Cpu(t) => Tensor::Cpu(t $op rhs),
                    Tensor::Gpu(t) => Tensor::Gpu(t $op rhs),
                }
            }
        }

        impl<'a, const R: usize, D, B, B2> std::ops::$trait<D> for &'a Tensor<R, D, B>
        where
            &'a CpuTensor<R, B>: std::ops::$trait<D, Output = CpuTensor<R, B2>>,
            &'a GpuTensor<R, D>: std::ops::$trait<D, Output = GpuTensor<R, D>>,
            B: TensorBacking<R, Elem = D>,
            B2: TensorBacking<R, Elem = D>,
            D: crate::cpu::Scalar,
        {
            type Output = Tensor<R, D, B2>;

            fn $method(self, rhs: D) -> Self::Output {
                match self {
                    Tensor::Cpu(t) => Tensor::Cpu(t $op rhs),
                    Tensor::Gpu(t) => Tensor::Gpu(t $op rhs),
                }
            }
        }
    };
}

impl_tensor_scalar_op!(Mul, mul, *);
impl_tensor_scalar_op!(Add, add, +);

// Broadcasting binary operations that can work with tensors of different ranks.
// Broadcasting is done at the fusor level using broadcast_as (which dispatches to
// backend restride), then same-rank operators are applied.

/// Macro to implement broadcasting binary operations for Tensor.
/// Broadcasts both tensors to a common shape using `broadcast_as` and applies the
/// same-rank operator.
macro_rules! impl_tensor_broadcast_op {
    ($trait:ident, $method:ident, $op:tt, $op_ty:ident) => {
        impl<const R: usize, D, B> Tensor<R, D, B>
        where
            D: SimdElement + DataType + Default,
            B: TensorBacking<R, Elem = D>,
        {
            #[doc = concat!(
                "Broadcasting ",
                stringify!($method),
                ": broadcasts both tensors to a common shape and applies the operation."
            )]
            pub fn $method<const R2: usize, const R3: usize, B2>(
                &self,
                second: &Tensor<R2, D, B2>,
            ) -> Tensor<R3, D>
            where
                (crate::gpu::Tensor<R, D>, crate::gpu::Tensor<R2, D>):
                    crate::gpu::MaxRank<R3, D>,
                (ConcreteTensor<D, R>, ConcreteTensor<D, R2>):
                    crate::cpu::MaxRank<R3, D>,
                D: std::ops::$trait<Output = D>,
                $op_ty: SimdBinaryOp<D>,
                B2: TensorBacking<R2, Elem = D>,
            {
                let out_shape: [usize; R3] =
                    composite::broadcast_shapes(&self.shape(), &second.shape());
                let a = self.broadcast_as(out_shape);
                let b = second.broadcast_as(out_shape);
                (&a $op &b).to_concrete()
            }
        }
    };
}

impl_tensor_broadcast_op!(Add, add_, +, AddOp);
impl_tensor_broadcast_op!(Sub, sub_, -, SubOp);
impl_tensor_broadcast_op!(Mul, mul_, *, MulOp);
impl_tensor_broadcast_op!(Div, div_, /, DivOp);

// `pow_` has a different shape from the other broadcast ops (it requires
// `FloatDataType + FloatOps`, has no SimdBinaryOp bound, and dispatches manually
// rather than going through an operator), so it stays inline.
impl<const R: usize, D, B> Tensor<R, D, B>
where
    D: SimdElement + DataType + Default,
    B: TensorBacking<R, Elem = D>,
{
    /// Broadcasting power: broadcasts both tensors to a common shape and computes power.
    pub fn pow_<const R2: usize, const R3: usize, B2>(
        &self,
        second: &Tensor<R2, D, B2>,
    ) -> Tensor<R3, D>
    where
        (crate::gpu::Tensor<R, D>, crate::gpu::Tensor<R2, D>): crate::gpu::MaxRank<R3, D>,
        (ConcreteTensor<D, R>, ConcreteTensor<D, R2>): crate::cpu::MaxRank<R3, D>,
        D: FloatDataType + FloatOps,
        B2: TensorBacking<R2, Elem = D>,
    {
        let out_shape: [usize; R3] = composite::broadcast_shapes(&self.shape(), &second.shape());
        let a = self.broadcast_as(out_shape).to_concrete();
        let b = second.broadcast_as(out_shape).to_concrete();
        match (&a, &b) {
            (Tensor::Cpu(a), Tensor::Cpu(b)) => {
                let result: Vec<D> = a
                    .inner()
                    .data()
                    .iter()
                    .zip(b.inner().data().iter())
                    .map(|(x, y)| x.powf(*y))
                    .collect();
                Tensor::Cpu(crate::cpu::TypedTensor::new(ConcreteTensor::from_slice(
                    out_shape, &result,
                )))
            }
            (Tensor::Gpu(a), Tensor::Gpu(b)) => Tensor::Gpu(a.pow_(b)),
            _ => panic!("Cannot mix CPU and GPU tensors"),
        }
    }
}

/// Macro to implement lazy unary element-wise operations for Tensor (any backing type).
macro_rules! impl_tensor_unary_op_lazy {
    ($method:ident, $op:ident, $expr_type:ident) => {
        impl<const R: usize, D, B> Tensor<R, D, B>
        where
            D: SimdElement + DataType + FloatDataType,
            B: TensorBacking<R, Elem = D>,
            crate::cpu::$op: crate::cpu::SimdUnaryOp<D>,
        {
            #[doc = concat!("Element-wise ", stringify!($method), " operation (lazy for CPU).")]
            pub fn $method(&self) -> Tensor<R, D, crate::cpu::$expr_type<D, R, &B>> {
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
    crate::cpu::ExpOp: crate::cpu::SimdUnaryOp<D>,
{
    /// Approximate exp function (faster but less accurate on GPU, exact on CPU).
    /// Uses a polynomial approximation on GPU for better performance.
    pub fn approximate_exp(&self) -> Tensor<R, D> {
        self.dispatch_ref(|t| t.as_ref().exp().to_concrete(), |t| t.approximate_exp())
    }

    /// Less approximate exp function (medium accuracy/speed tradeoff on GPU, exact on CPU).
    pub fn less_approximate_exp(&self) -> Tensor<R, D> {
        self.dispatch_ref(
            |t| t.as_ref().exp().to_concrete(),
            |t| t.less_approximate_exp(),
        )
    }
}

// Exact tanh operation
impl<const R: usize, D, B> Tensor<R, D, B>
where
    D: SimdElement + DataType + FloatDataType + Default,
    B: TensorBacking<R, Elem = D>,
    crate::cpu::TanhOp: crate::cpu::SimdUnaryOp<D>,
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
        D: CastTo<D2> + crate::gpu::CastTensor<D2>,
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
            (Tensor::Cpu(lhs), QMatrix::CpuQ4_0(rhs)) => Tensor::Cpu(crate::cpu::TypedTensor::new(
                lhs.to_concrete().inner().q_mat_mul(rhs),
            )),
            (Tensor::Cpu(lhs), QMatrix::CpuQ5_0(rhs)) => Tensor::Cpu(crate::cpu::TypedTensor::new(
                lhs.to_concrete().inner().q_mat_mul(rhs),
            )),
            (Tensor::Cpu(lhs), QMatrix::CpuQ8_0(rhs)) => Tensor::Cpu(crate::cpu::TypedTensor::new(
                lhs.to_concrete().inner().q_mat_mul(rhs),
            )),
            (Tensor::Cpu(lhs), QMatrix::CpuQ4K(rhs)) => Tensor::Cpu(crate::cpu::TypedTensor::new(
                lhs.to_concrete().inner().q_mat_mul(rhs),
            )),
            (Tensor::Cpu(lhs), QMatrix::CpuQ5K(rhs)) => Tensor::Cpu(crate::cpu::TypedTensor::new(
                lhs.to_concrete().inner().q_mat_mul(rhs),
            )),
            (Tensor::Cpu(lhs), QMatrix::CpuQ6K(rhs)) => Tensor::Cpu(crate::cpu::TypedTensor::new(
                lhs.to_concrete().inner().q_mat_mul(rhs),
            )),
            // F16/F32 are not quantized — dequantize, transpose, and use regular matmul
            (_, QMatrix::CpuF32(_))
            | (_, QMatrix::CpuF16(_))
            | (Tensor::Gpu(_), QMatrix::Gpu(_))
                if weights.ggml_type() == fusor_gguf::GgmlType::F16
                    || weights.ggml_type() == fusor_gguf::GgmlType::F32 =>
            {
                let n = weights.shape()[0]; // out_features
                let k = weights.shape()[1]; // in_features
                let dequantized: Tensor<2, f32> = weights.dequantize();
                let weight_t = dequantized.transpose(0, 1);
                let weight_shape: [usize; R] = std::array::from_fn(|i| {
                    if i < R - 2 {
                        1
                    } else if i == R - 2 {
                        k
                    } else {
                        n
                    }
                });
                let target_shape: [usize; R] = std::array::from_fn(|i| {
                    if i < R - 2 {
                        self.shape()[i]
                    } else if i == R - 2 {
                        k
                    } else {
                        n
                    }
                });
                let weight_reshaped = weight_t.reshape(weight_shape);
                let weight_broadcast = weight_reshaped.broadcast_as(target_shape);
                self.mat_mul(&weight_broadcast)
            }

            // GPU path - quantized types
            (Tensor::Gpu(lhs), QMatrix::Gpu(rhs)) => Tensor::Gpu(lhs.q_mat_mul(rhs)),

            // Mixed - panic
            _ => panic!("Cannot mix CPU and GPU tensors in q_mat_mul"),
        }
    }

    pub fn q_mat_mul_paired_silu_product(&self, weights: &crate::QMatrix) -> Tensor<R, f32> {
        use crate::QMatrix;

        assert_eq!(
            weights.shape().len(),
            2,
            "q_mat_mul_paired_silu_product requires 2D weight tensor, got {}D",
            weights.shape().len()
        );
        assert!(
            weights.shape()[0].is_multiple_of(2),
            "q_mat_mul_paired_silu_product requires an even output dimension"
        );

        match (self, weights) {
            (Tensor::Gpu(lhs), QMatrix::Gpu(rhs))
                if weights.ggml_type() == fusor_gguf::GgmlType::Q4K =>
            {
                Tensor::Gpu(lhs.q_mat_mul_paired_silu_product(rhs))
            }
            _ => {
                let pair_len = weights.shape()[0] / 2;
                let projected = self.q_mat_mul(weights);
                let gate = projected
                    .narrow(crate::D::Minus1, 0, pair_len)
                    .to_concrete();
                let up = projected
                    .narrow(crate::D::Minus1, pair_len, pair_len)
                    .to_concrete();
                (gate.silu() * up).to_concrete()
            }
        }
    }

    pub fn q_mat_mul_add2<B1, B2>(
        &self,
        weights: &crate::QMatrix,
        first: &Tensor<R, f32, B1>,
        second: &Tensor<R, f32, B2>,
    ) -> Tensor<R, f32>
    where
        B1: TensorBacking<R, Elem = f32>,
        B2: TensorBacking<R, Elem = f32>,
    {
        use crate::QMatrix;

        assert_eq!(
            weights.shape().len(),
            2,
            "q_mat_mul_add2 requires 2D weight tensor, got {}D",
            weights.shape().len()
        );

        let output_shape: [usize; R] = std::array::from_fn(|i| {
            if i + 1 == R {
                weights.shape()[0]
            } else {
                self.shape()[i]
            }
        });
        assert_eq!(
            first.shape(),
            output_shape,
            "first residual shape must match q_mat_mul output shape"
        );
        assert_eq!(
            second.shape(),
            output_shape,
            "second residual shape must match q_mat_mul output shape"
        );

        match (self, weights, first, second) {
            (Tensor::Gpu(lhs), QMatrix::Gpu(rhs), Tensor::Gpu(first), Tensor::Gpu(second))
                if weights.ggml_type() != fusor_gguf::GgmlType::F16
                    && weights.ggml_type() != fusor_gguf::GgmlType::F32 =>
            {
                Tensor::Gpu(lhs.q_mat_mul_add2(rhs, first, second))
            }
            _ => {
                let projected = self.q_mat_mul(weights);
                let with_first = (&projected + first).to_concrete();
                (&with_first + second).to_concrete()
            }
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
        crate::gpu::Tensor<R, D>: crate::gpu::SmallerRank<FROM_END, R2, D>,
    {
        let shape = self.shape();
        let new_shape: [usize; R2] = std::array::from_fn(|i| {
            if i < R - 1 - FROM_END {
                shape[i]
            } else if i == R - 1 - FROM_END {
                shape[R - 1 - FROM_END..].iter().product()
            } else {
                1
            }
        });
        self.reshape(new_shape).to_concrete()
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
        crate::gpu::Tensor<R, D>: crate::gpu::SmallerRank<FROM_START, R2, D>,
    {
        let shape = self.shape();
        let new_shape: [usize; R2] = std::array::from_fn(|i| {
            if i == 0 {
                shape[..=FROM_START].iter().product()
            } else {
                shape[i + FROM_START]
            }
        });
        self.reshape(new_shape).to_concrete()
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

    /// Return the GPU compute-graph node index, if this is a GPU tensor.
    pub fn gpu_key(&self) -> Option<NodeIndex> {
        match self {
            Tensor::Gpu(t) => Some(t.key()),
            Tensor::Cpu(_) => None,
        }
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
                let slice = t.as_slice().await.map_err(Error::Gpu)?;
                Ok(slice.as_scalar())
            }
        }
    }
}
