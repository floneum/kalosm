use std::{
    fmt::{Debug, Display},
    future::Future,
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Rem, Sub},
};

pub use fusor_core::{
    CastTensor, DataType, DataTypeEnum, Device, Dim, Error, FloatDataType, GgufReadError,
    GpuMirostat2Sampler, GpuMirostat2SamplerParams, Layout, MappedBuffer, MatMulParams, NodeIndex,
    QMatrix, Result, ShapeWithOneHole, StrideSpec, TensorSlice, WasmNotSend, WasmNotSync,
};

type CoreTensor = fusor_core::Tensor;

/// Typed facade tensor for GPU values.
///
/// The backend tensor is deliberately hidden behind this newtype. Core owns the
/// dynamic storage and graph handle; this facade carries the rank and dtype
/// proofs that `fusor` exposes to callers.
pub struct Tensor<const R: usize, D> {
    inner: CoreTensor,
    datatype: PhantomData<D>,
}

impl<const R: usize, D> Tensor<R, D> {
    #[inline]
    pub(crate) fn from_core_unchecked(inner: CoreTensor) -> Self {
        Self {
            inner,
            datatype: PhantomData,
        }
    }

    #[inline]
    pub(crate) fn as_core(&self) -> &CoreTensor {
        &self.inner
    }

    #[inline]
    pub(crate) fn into_core(self) -> CoreTensor {
        self.inner
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    #[inline]
    pub(crate) fn from_core(inner: CoreTensor) -> Self {
        inner.assert_rank::<R>();
        inner.assert_datatype::<D>();
        Self::from_core_unchecked(inner)
    }

    #[inline]
    pub fn detach(&self) -> Self {
        Self::from_core(self.inner.detach())
    }
}

impl<const R: usize, D> Clone for Tensor<R, D> {
    fn clone(&self) -> Self {
        Self::from_core_unchecked(self.inner.clone())
    }
}

impl<const R: usize, D: DataType> Display for Tensor<R, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.inner, f)
    }
}

impl<const R: usize, D: DataType> Debug for Tensor<R, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self.inner, f)
    }
}

impl<const R: usize, D, T> fusor_types::FromArray<R, D, T, Device> for Tensor<R, D>
where
    D: DataType,
    T: fusor_types::IntoFlatArray<D, R>,
{
    fn from_array(data: T, device: &Device) -> Self {
        Self::from_core(CoreTensor::new::<D, R, T>(device, data))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    #[inline]
    pub fn new<T>(device: &Device, data: T) -> Self
    where
        Self: fusor_types::FromArray<R, D, T, Device>,
    {
        fusor_types::FromArray::from_array(data, device)
    }

    #[inline]
    pub fn from_slice(device: &Device, shape: [usize; R], data: &[D]) -> Self {
        Self::from_core(CoreTensor::from_slice::<D>(device, shape, data))
    }

    #[inline]
    pub fn splat(device: &Device, value: D, shape: [usize; R]) -> Self {
        Self::from_core(CoreTensor::splat::<D>(device, value, shape))
    }

    #[inline]
    pub fn full(device: &Device, value: D, shape: [usize; R]) -> Self {
        Self::splat(device, value, shape)
    }

    #[inline]
    pub fn materialize_sync(&self) {
        self.inner.materialize_sync()
    }

    #[inline]
    pub fn materialize(&self) -> impl Future<Output = ()> + 'static {
        self.inner.materialize()
    }

    #[inline]
    pub fn count_kernels_to_resolve(&self) -> usize {
        self.inner.count_kernels_to_resolve()
    }

    #[inline]
    pub async fn as_slice(&self) -> Result<TensorSlice<R, D, MappedBuffer>> {
        self.inner.as_slice::<R, D>().await.map_err(Error::from)
    }

    #[inline]
    pub async fn to_scalar(&self) -> Result<D> {
        self.inner.to_scalar::<D>().await.map_err(Error::from)
    }

    #[inline]
    pub fn debug_assert_real(self) -> Self
    where
        D: FloatDataType,
    {
        Self::from_core(self.inner.debug_assert_real())
    }

    #[inline]
    pub fn key(&self) -> NodeIndex {
        self.inner.key()
    }

    #[inline]
    pub fn shape(&self) -> &[usize; R] {
        self.inner.shape_array::<R>()
    }

    #[inline]
    pub fn rank(&self) -> usize {
        self.inner.rank()
    }

    #[inline]
    pub fn datatype(&self) -> DataTypeEnum {
        self.inner.datatype()
    }

    #[inline]
    pub fn device(&self) -> &Device {
        self.inner.device()
    }

    #[cfg(feature = "graphvis")]
    #[inline]
    pub fn graphvis(&self) -> fusor_core::tabbycat::Graph {
        self.inner.graphvis()
    }

    #[inline]
    pub fn resize(&self, new_shape: [usize; R]) -> Self {
        Self::from_core(self.inner.resize(new_shape))
    }

    #[inline]
    pub fn reshape<const R2: usize>(&self, new_shape: impl ShapeWithOneHole<R2>) -> Tensor<R2, D> {
        let new_shape = new_shape.resolve_shape(self.shape());
        Tensor::from_core(self.inner.reshape(new_shape))
    }

    #[inline]
    pub fn restride<const R2: usize>(&self, specs: [StrideSpec; R2]) -> Tensor<R2, D> {
        Tensor::from_core(self.inner.restride(specs))
    }

    #[inline]
    pub fn restride_layout<const R2: usize>(&self, new_layout: Layout) -> Tensor<R2, D> {
        Tensor::from_core(self.inner.restride_layout(new_layout))
    }

    #[inline]
    pub fn flatten_last_n<const FROM_END: usize, const O: usize>(&self) -> Tensor<O, D> {
        Tensor::from_core(self.inner.flatten_last_n(FROM_END))
    }

    #[inline]
    pub fn flatten_first_n<const FROM_START: usize, const O: usize>(&self) -> Tensor<O, D> {
        Tensor::from_core(self.inner.flatten_first_n(FROM_START))
    }

    #[inline]
    pub fn flatten_all(&self) -> Tensor<1, D> {
        Tensor::from_core(self.inner.flatten_all())
    }

    #[inline]
    pub fn slice_assign(&self, slices: [std::ops::Range<usize>; R], value: &Self) -> Self {
        Self::from_core(self.inner.slice_assign(slices, value.as_core()))
    }

    #[inline]
    pub fn slice_assign_in_place(&self, slices: [std::ops::Range<usize>; R], value: &Self) -> Self {
        Self::from_core(self.inner.slice_assign_in_place(slices, value.as_core()))
    }

    #[inline]
    pub fn index_select(&self, dimension: usize, indexes: &Tensor<1, u32>) -> Self {
        Self::from_core(self.inner.index_select(dimension, indexes.as_core()))
    }

    #[inline]
    pub fn mat_mul(&self, other: &Self) -> Self {
        Self::from_core(self.inner.mat_mul(other.as_core()))
    }

    #[inline]
    pub fn mat_mul_with_parameters(&self, other: &Self, parameters: MatMulParams) -> Self {
        Self::from_core(
            self.inner
                .mat_mul_with_parameters(other.as_core(), parameters),
        )
    }

    #[inline]
    pub fn sum<const O: usize>(&self, dim: impl Dim<R>) -> Tensor<O, D> {
        Tensor::from_core(self.inner.sum(dim.resolve()))
    }

    #[inline]
    pub fn sum_keepdim<const O: usize>(&self, dim: impl Dim<R>) -> Self {
        Self::from_core(self.inner.sum_keepdim(dim.resolve()))
    }

    #[inline]
    pub fn max<const O: usize>(&self, dim: impl Dim<R>) -> Tensor<O, D> {
        Tensor::from_core(self.inner.max(dim.resolve()))
    }

    #[inline]
    pub fn max_keepdim<const O: usize>(&self, dim: impl Dim<R>) -> Self {
        Self::from_core(self.inner.max_keepdim(dim.resolve()))
    }

    #[inline]
    pub fn min<const O: usize>(&self, dim: impl Dim<R>) -> Tensor<O, D> {
        Tensor::from_core(self.inner.min(dim.resolve()))
    }

    #[inline]
    pub fn min_keepdim<const O: usize>(&self, dim: impl Dim<R>) -> Self {
        Self::from_core(self.inner.min_keepdim(dim.resolve()))
    }

    #[inline]
    pub fn product<const O: usize>(&self, dim: impl Dim<R>) -> Tensor<O, D> {
        Tensor::from_core(self.inner.product(dim.resolve()))
    }

    #[inline]
    pub fn product_keepdim<const O: usize>(&self, dim: impl Dim<R>) -> Self {
        Self::from_core(self.inner.product_keepdim(dim.resolve()))
    }

    #[inline]
    pub fn eq<D2: DataType>(&self, rhs: D) -> Tensor<R, D2> {
        Tensor::from_core(self.inner.eq::<D2, D>(rhs))
    }

    #[inline]
    pub fn lt<D2: DataType>(&self, rhs: D) -> Tensor<R, D2> {
        Tensor::from_core(self.inner.lt::<D2, D>(rhs))
    }

    #[inline]
    pub fn lte<D2: DataType>(&self, rhs: D) -> Tensor<R, D2> {
        Tensor::from_core(self.inner.lte::<D2, D>(rhs))
    }

    #[inline]
    pub fn mt<D2: DataType>(&self, rhs: D) -> Tensor<R, D2> {
        Tensor::from_core(self.inner.mt::<D2, D>(rhs))
    }

    #[inline]
    pub fn mte<D2: DataType>(&self, rhs: D) -> Tensor<R, D2> {
        Tensor::from_core(self.inner.mte::<D2, D>(rhs))
    }

    #[inline]
    pub fn cast<D2>(&self) -> Tensor<R, D2>
    where
        D: CastTensor<D2>,
        D2: DataType,
    {
        Tensor::from_core(self.inner.cast::<D2>())
    }
}

impl<const R: usize, D: DataType + FloatDataType> Tensor<R, D> {
    #[inline]
    pub fn less_approximate_exp(&self) -> Self {
        Self::from_core(self.inner.less_approximate_exp())
    }

    #[inline]
    pub fn approximate_exp(&self) -> Self {
        Self::from_core(self.inner.approximate_exp())
    }

    #[inline]
    pub fn exp(&self) -> Self {
        Self::from_core(self.inner.exp())
    }

    #[inline]
    pub fn exp2(&self) -> Self {
        Self::from_core(self.inner.exp2())
    }

    #[inline]
    pub fn log(&self) -> Self {
        Self::from_core(self.inner.log())
    }

    #[inline]
    pub fn log2(&self) -> Self {
        Self::from_core(self.inner.log2())
    }

    #[inline]
    pub fn pow_elementwise(&self, exponent: D) -> Self {
        Self::from_core(self.inner.pow_elementwise(exponent))
    }

    #[inline]
    pub fn sqrt(&self) -> Self {
        Self::from_core(self.inner.sqrt())
    }

    #[inline]
    pub fn sin(&self) -> Self {
        Self::from_core(self.inner.sin())
    }

    #[inline]
    pub fn cos(&self) -> Self {
        Self::from_core(self.inner.cos())
    }

    #[inline]
    pub fn tan(&self) -> Self {
        Self::from_core(self.inner.tan())
    }

    #[inline]
    pub fn asin(&self) -> Self {
        Self::from_core(self.inner.asin())
    }

    #[inline]
    pub fn acos(&self) -> Self {
        Self::from_core(self.inner.acos())
    }

    #[inline]
    pub fn atan(&self) -> Self {
        Self::from_core(self.inner.atan())
    }

    #[inline]
    pub fn sinh(&self) -> Self {
        Self::from_core(self.inner.sinh())
    }

    #[inline]
    pub fn cosh(&self) -> Self {
        Self::from_core(self.inner.cosh())
    }

    #[inline]
    pub fn tanh(&self) -> Self {
        Self::from_core(self.inner.tanh())
    }

    #[inline]
    pub fn tanh_exact(&self) -> Self {
        Self::from_core(self.inner.tanh_exact())
    }

    #[inline]
    pub fn asinh(&self) -> Self {
        Self::from_core(self.inner.asinh())
    }

    #[inline]
    pub fn acosh(&self) -> Self {
        Self::from_core(self.inner.acosh())
    }

    #[inline]
    pub fn atanh(&self) -> Self {
        Self::from_core(self.inner.atanh())
    }

    #[inline]
    pub fn abs(&self) -> Self {
        Self::from_core(self.inner.abs())
    }

    #[inline]
    pub fn max_elementwise(&self, element: D) -> Self {
        Self::from_core(self.inner.max_elementwise(element))
    }

    #[inline]
    pub fn min_elementwise(&self, element: D) -> Self {
        Self::from_core(self.inner.min_elementwise(element))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    #[inline]
    pub fn pow(&self, other: &Self) -> Self {
        Self::from_core(self.inner.pow(other.as_core()))
    }

    #[inline]
    pub fn pow_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2, D>) -> Tensor<R3, D> {
        Tensor::from_core(self.inner.pow_(second.as_core()))
    }

    #[inline]
    pub fn add_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2, D>) -> Tensor<R3, D> {
        Tensor::from_core(self.inner.add_(second.as_core()))
    }

    #[inline]
    pub fn sub_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2, D>) -> Tensor<R3, D> {
        Tensor::from_core(self.inner.sub_(second.as_core()))
    }

    #[inline]
    pub fn mul_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2, D>) -> Tensor<R3, D> {
        Tensor::from_core(self.inner.mul_(second.as_core()))
    }

    #[inline]
    pub fn div_<const R2: usize, const R3: usize>(&self, second: &Tensor<R2, D>) -> Tensor<R3, D> {
        Tensor::from_core(self.inner.div_(second.as_core()))
    }
}

impl<const R: usize> Tensor<R, f32> {
    #[inline]
    pub fn q_mat_mul(&self, other: &QMatrix) -> Self {
        Self::from_core(self.inner.q_mat_mul(other))
    }

    #[inline]
    pub fn q_mat_mul_add2(&self, other: &QMatrix, first: &Self, second: &Self) -> Self {
        Self::from_core(
            self.inner
                .q_mat_mul_add2(other, first.as_core(), second.as_core()),
        )
    }

    #[inline]
    pub fn q_mat_mul_paired_silu_product(&self, other: &QMatrix) -> Self {
        Self::from_core(self.inner.q_mat_mul_paired_silu_product(other))
    }
}

impl<const R: usize> Tensor<R, half::f16> {
    #[inline]
    pub fn q_mat_mul(&self, other: &QMatrix) -> Self {
        Self::from_core(self.inner.q_mat_mul(other))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    #[inline]
    pub fn rms_norm_fused<const W: usize, const OUT_RANK: usize>(
        &self,
        weight: &Tensor<W, D>,
        bias: Option<&Tensor<W, D>>,
        eps: f32,
    ) -> Self {
        assert_eq!(
            OUT_RANK + 1,
            R,
            "rms_norm_fused reduction rank must be input rank - 1"
        );
        Self::from_core(self.inner.rms_norm_fused(
            weight.as_core(),
            bias.map(|bias| bias.as_core()),
            eps,
        ))
    }

    #[inline]
    pub fn rms_norm_fused_no_bias<const W: usize, const OUT_RANK: usize>(
        &self,
        weight: &Tensor<W, D>,
        eps: f32,
    ) -> Self {
        self.rms_norm_fused::<W, OUT_RANK>(weight, None, eps)
    }

    #[inline]
    pub fn rms_norm_residual_fused<const W: usize, const OUT_RANK: usize>(
        &self,
        residual: &Self,
        weight: &Tensor<W, D>,
        bias: Option<&Tensor<W, D>>,
        eps: f32,
    ) -> Self {
        assert_eq!(
            OUT_RANK + 1,
            R,
            "rms_norm_residual_fused reduction rank must be input rank - 1"
        );
        Self::from_core(self.inner.rms_norm_residual_fused(
            residual.as_core(),
            weight.as_core(),
            bias.map(|bias| bias.as_core()),
            eps,
        ))
    }
}

impl Tensor<1, f32> {
    #[inline]
    pub async fn try_sample_mirostat2_token_q_mat(
        &self,
        matrix: &QMatrix,
        sampler: &mut GpuMirostat2Sampler,
        previous_tokens: &[u32],
        params: GpuMirostat2SamplerParams,
    ) -> Result<Option<u32>> {
        self.inner
            .try_sample_mirostat2_token_q_mat(matrix, sampler, previous_tokens, params)
            .await
            .map_err(Error::from)
    }

    #[inline]
    pub async fn sample_mirostat2_token(
        &self,
        sampler: &mut GpuMirostat2Sampler,
        previous_tokens: &[u32],
        params: GpuMirostat2SamplerParams,
    ) -> Result<u32> {
        self.inner
            .sample_mirostat2_token(sampler, previous_tokens, params)
            .await
            .map_err(Error::from)
    }

    #[inline]
    pub async fn top_k_pairs(&self, k: usize) -> Result<(Vec<u32>, Vec<f32>)> {
        self.inner.top_k_pairs(k).await.map_err(Error::from)
    }
}

impl<T: DataType> Tensor<4, T> {
    #[inline]
    pub fn flash_attention_causal(&self, k: &Self, v: &Self, scale: f32) -> Self {
        Self::from_core(
            self.inner
                .flash_attention_causal(k.as_core(), v.as_core(), scale),
        )
    }

    #[inline]
    pub fn flash_attention(
        &self,
        k: &Self,
        v: &Self,
        scale: f32,
        mask: Option<&Tensor<2, T>>,
    ) -> Self {
        Self::from_core(self.inner.flash_attention(
            k.as_core(),
            v.as_core(),
            scale,
            mask.map(|mask| mask.as_core()),
        ))
    }
}

impl<D: DataType> Tensor<4, D> {
    #[inline]
    pub fn rope_fused(&self, cos: &Tensor<2, D>, sin: &Tensor<2, D>) -> Self {
        Self::from_core(self.inner.rope_fused(cos.as_core(), sin.as_core()))
    }

    #[inline]
    pub fn rope_normal_fused(&self, cos: &Tensor<2, D>, sin: &Tensor<2, D>) -> Self {
        Self::from_core(self.inner.rope_normal_fused(cos.as_core(), sin.as_core()))
    }

    #[inline]
    pub fn rope_pair_fused(
        &self,
        k: &Self,
        cos: &Tensor<2, D>,
        sin: &Tensor<2, D>,
    ) -> (Self, Self) {
        let (q, k) = self
            .inner
            .rope_pair_fused(k.as_core(), cos.as_core(), sin.as_core());
        (Self::from_core(q), Self::from_core(k))
    }

    #[inline]
    pub fn rope_normal_pair_fused(
        &self,
        k: &Self,
        cos: &Tensor<2, D>,
        sin: &Tensor<2, D>,
    ) -> (Self, Self) {
        let (q, k) = self
            .inner
            .rope_normal_pair_fused(k.as_core(), cos.as_core(), sin.as_core());
        (Self::from_core(q), Self::from_core(k))
    }
}

impl<const R: usize, D: DataType> Tensor<R, D> {
    #[inline]
    pub fn softmax<const R2: usize>(&self, axis: impl Dim<R>) -> Self {
        assert_eq!(R2 + 1, R, "softmax output rank must be input rank - 1");
        Self::from_core(self.inner.softmax(axis.resolve()))
    }

    #[inline]
    pub fn softmax_last_dim<const R2: usize>(&self) -> Self {
        assert_eq!(R2 + 1, R, "softmax output rank must be input rank - 1");
        Self::from_core(self.inner.softmax_last_dim())
    }

    #[inline]
    pub fn where_cond<D2: DataType>(
        self,
        on_true: &Tensor<R, D2>,
        on_false: &Tensor<R, D2>,
    ) -> Tensor<R, D2> {
        Tensor::from_core(self.inner.where_cond(on_true.as_core(), on_false.as_core()))
    }
}

impl<const R: usize, T: DataType> Add<T> for Tensor<R, T> {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        Self::from_core(self.inner + rhs)
    }
}

impl<const R: usize, T: DataType> Add<T> for &Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn add(self, rhs: T) -> Self::Output {
        Tensor::from_core(self.as_core() + rhs)
    }
}

impl<const R: usize, T: DataType> Sub<T> for Tensor<R, T> {
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        Self::from_core(self.inner - rhs)
    }
}

impl<const R: usize, T: DataType> Mul<T> for Tensor<R, T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self::from_core(self.inner * rhs)
    }
}

impl<const R: usize, T: DataType> Mul<T> for &Tensor<R, T> {
    type Output = Tensor<R, T>;

    fn mul(self, rhs: T) -> Self::Output {
        Tensor::from_core(self.as_core() * rhs)
    }
}

impl<const R: usize, T: DataType> Div<T> for Tensor<R, T> {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        Self::from_core(self.inner / rhs)
    }
}

impl<const R: usize> Rem<u32> for Tensor<R, u32> {
    type Output = Self;

    fn rem(self, rhs: u32) -> Self::Output {
        Self::from_core(self.inner % rhs)
    }
}

macro_rules! impl_scalar_lhs {
    ($trait:ident, $method:ident, $($t:ty),* $(,)?) => {
        $(
            impl<const R: usize> $trait<Tensor<R, $t>> for $t {
                type Output = Tensor<R, $t>;

                fn $method(self, rhs: Tensor<R, $t>) -> Self::Output {
                    Tensor::from_core(self.$method(rhs.into_core()))
                }
            }
        )*
    };
}

impl_scalar_lhs!(Add, add, f32, half::f16, u32);
impl_scalar_lhs!(Sub, sub, f32, half::f16, u32);
impl_scalar_lhs!(Mul, mul, f32, half::f16, u32);
impl_scalar_lhs!(Div, div, f32, half::f16, u32);
impl_scalar_lhs!(Rem, rem, f32, half::f16, u32);

macro_rules! impl_pairwise_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<const R: usize, T: DataType> $trait<Tensor<R, T>> for Tensor<R, T> {
            type Output = Tensor<R, T>;

            fn $method(self, rhs: Tensor<R, T>) -> Self::Output {
                Tensor::from_core(self.into_core() $op rhs.into_core())
            }
        }

        impl<'a, const R: usize, T: DataType> $trait<&'a Tensor<R, T>> for &'a Tensor<R, T> {
            type Output = Tensor<R, T>;

            fn $method(self, rhs: &'a Tensor<R, T>) -> Self::Output {
                Tensor::from_core(self.as_core() $op rhs.as_core())
            }
        }

        impl<'a, const R: usize, T: DataType> $trait<Tensor<R, T>> for &'a Tensor<R, T> {
            type Output = Tensor<R, T>;

            fn $method(self, rhs: Tensor<R, T>) -> Self::Output {
                Tensor::from_core(self.as_core() $op rhs.into_core())
            }
        }

        impl<'a, const R: usize, T: DataType> $trait<&'a Tensor<R, T>> for Tensor<R, T> {
            type Output = Tensor<R, T>;

            fn $method(self, rhs: &'a Tensor<R, T>) -> Self::Output {
                Tensor::from_core(self.into_core() $op rhs.as_core())
            }
        }
    };
}

impl_pairwise_op!(Add, add, +);
impl_pairwise_op!(Sub, sub, -);
impl_pairwise_op!(Mul, mul, *);
impl_pairwise_op!(Div, div, /);

impl<const R: usize, D: DataType> Neg for Tensor<R, D> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::from_core(-self.inner)
    }
}

impl<const R: usize, D: DataType> Neg for &Tensor<R, D> {
    type Output = Tensor<R, D>;

    fn neg(self) -> Self::Output {
        Tensor::from_core(-self.as_core())
    }
}

impl<const R: usize, T: DataType> std::iter::Sum for Tensor<R, T> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        let first = iter.next().expect("Cannot sum over empty iterator");
        iter.fold(first, |acc, x| acc + x)
    }
}

impl<'a, const R: usize, T: DataType> std::iter::Sum<&'a Tensor<R, T>> for Tensor<R, T> {
    fn sum<I: Iterator<Item = &'a Tensor<R, T>>>(iter: I) -> Self {
        let mut iter = iter.cloned();
        let first = iter.next().expect("Cannot sum over empty iterator");
        iter.fold(first, |acc, x| acc + x)
    }
}

pub trait LastRankInner {
    type LastRank;
}

pub trait LastRank<const R: usize, T: DataType>: LastRankInner<LastRank = Tensor<R, T>> {}

impl<const R: usize, T: DataType, X> LastRank<R, T> for X where
    X: LastRankInner<LastRank = Tensor<R, T>>
{
}

pub trait NextRankInner {
    type NextRank;
}

pub trait NextRank<const R: usize, T: DataType>: NextRankInner<NextRank = Tensor<R, T>> {}

impl<const R: usize, T: DataType, X> NextRank<R, T> for X where
    X: NextRankInner<NextRank = Tensor<R, T>>
{
}

pub trait SmallerRankInner<const DIFF: usize> {
    type SmallerRank;
    type SmallerByArray;
}

pub trait SmallerRank<const DIFF: usize, const R: usize, T: DataType>:
    SmallerRankInner<DIFF, SmallerRank = Tensor<R, T>>
{
}

impl<const DIFF: usize, const R: usize, T: DataType, X> SmallerRank<DIFF, R, T> for X where
    X: SmallerRankInner<DIFF, SmallerRank = Tensor<R, T>>
{
}

pub trait LargerRankInner<const DIFF: usize> {
    type LargerRank;
    type LargerByArray;
}

pub trait LargerRank<const DIFF: usize, const R: usize, T: DataType>:
    LargerRankInner<DIFF, LargerRank = Tensor<R, T>>
{
}

impl<const DIFF: usize, const R: usize, T: DataType, X> LargerRank<DIFF, R, T> for X where
    X: LargerRankInner<DIFF, LargerRank = Tensor<R, T>>
{
}

pub trait MaxRankInner {
    type MaxRank;
}

pub trait MaxRank<const R: usize, T: DataType>: MaxRankInner<MaxRank = Tensor<R, T>> {}

impl<const R: usize, T: DataType, X> MaxRank<R, T> for X where
    X: MaxRankInner<MaxRank = Tensor<R, T>>
{
}

fusor_types::impl_rank_traits!(Tensor);
