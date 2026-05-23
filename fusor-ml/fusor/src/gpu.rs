#[cfg(feature = "gpu")]
pub use fusor_core::*;

#[cfg(not(feature = "gpu"))]
mod disabled {
    use std::{
        fmt::{self, Debug, Display},
        marker::PhantomData,
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign},
    };

    use bytemuck::{AnyBitPattern, NoUninit};
    pub use fusor_types::{D, Dim, Layout, StrideSpec, TensorSlice};

    pub use fusor_gguf::GgufReadError;

    pub trait DataType:
        Add<Output = Self>
        + AddAssign
        + Sub<Output = Self>
        + SubAssign
        + Mul<Output = Self>
        + MulAssign
        + Div<Output = Self>
        + DivAssign
        + PartialOrd
        + NoUninit
        + AnyBitPattern
        + Debug
        + Display
        + Send
        + Sync
        + 'static
    {
        fn zero() -> Self;
        fn one() -> Self;
    }

    pub trait FloatDataType: DataType {
        fn from_f32(value: f32) -> Self;
        fn is_finite(&self) -> bool;
    }

    impl DataType for f32 {
        fn zero() -> Self {
            0.
        }

        fn one() -> Self {
            1.
        }
    }

    impl FloatDataType for f32 {
        fn from_f32(value: f32) -> Self {
            value
        }

        fn is_finite(&self) -> bool {
            f32::is_finite(*self)
        }
    }

    impl DataType for half::f16 {
        fn zero() -> Self {
            half::f16::from_f32(0.)
        }

        fn one() -> Self {
            half::f16::from_f32(1.)
        }
    }

    impl FloatDataType for half::f16 {
        fn from_f32(value: f32) -> Self {
            half::f16::from_f32(value)
        }

        fn is_finite(&self) -> bool {
            half::f16::is_finite(*self)
        }
    }

    impl DataType for u32 {
        fn zero() -> Self {
            0
        }

        fn one() -> Self {
            1
        }
    }

    pub use fusor_types::ShapeWithOneHole;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct NodeIndex;

    #[derive(Clone, Debug)]
    pub enum Device {}

    #[derive(Clone, Debug)]
    pub struct WgpuAdapter;

    #[derive(Clone, Debug)]
    pub struct WgpuAdapterInfo;

    impl WgpuAdapter {
        pub fn get_info(&self) -> WgpuAdapterInfo {
            WgpuAdapterInfo
        }
    }

    impl Device {
        pub fn resolve_batch(&self, _keys: &[NodeIndex]) -> usize {
            match *self {}
        }

        pub fn detach_cached(&self, _keys: &[NodeIndex]) {
            match *self {}
        }

        pub fn poll_wait(&self) {
            match *self {}
        }

        pub fn wgpu_adapter(&self) -> &WgpuAdapter {
            match *self {}
        }
    }

    #[derive(Debug)]
    pub enum Error {}

    impl fmt::Display for Error {
        fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match *self {}
        }
    }

    impl std::error::Error for Error {}

    #[derive(Debug)]
    pub enum Tensor<const R: usize, T> {
        Disabled(PhantomData<T>, std::convert::Infallible),
    }

    impl<const R: usize, T> Clone for Tensor<R, T> {
        fn clone(&self) -> Self {
            match self {
                Tensor::Disabled(_, never) => match *never {},
            }
        }
    }

    #[derive(Debug)]
    pub enum MappedBuffer {}

    impl std::ops::Deref for MappedBuffer {
        type Target = [u8];

        fn deref(&self) -> &Self::Target {
            match *self {}
        }
    }

    #[derive(Clone, Debug)]
    pub enum QMatrix {}

    impl QMatrix {
        pub fn concat_rows(_matrices: &[&Self]) -> Option<Self> {
            None
        }

        pub fn datatype(&self) -> fusor_gguf::GgmlType {
            match *self {}
        }

        pub fn shape(&self) -> &[usize] {
            match *self {}
        }

        pub fn device(&self) -> &Device {
            match *self {}
        }

        pub fn from_raw_bytes(
            _device: &Device,
            _shape: impl Into<Box<[usize]>>,
            _bytes: &[u8],
            _ty: fusor_gguf::GgmlType,
        ) -> Result<Self, fusor_gguf::GgufReadError> {
            unreachable!("GPU backend is disabled")
        }

        pub fn from_parts(
            _device: &Device,
            _bytes: &[u8],
            _shape: Box<[usize]>,
            _ty: fusor_gguf::GgmlType,
        ) -> Result<Self, fusor_gguf::GgufReadError> {
            unreachable!("GPU backend is disabled")
        }

        pub fn dequantize<const R: usize, T>(&self) -> Tensor<R, T> {
            match *self {}
        }

        pub fn index_select_rows(&self, _indexes: &Tensor<1, u32>) -> Tensor<2, f32> {
            match *self {}
        }
    }

    #[derive(Debug)]
    pub enum GpuMirostat2Sampler {}

    impl GpuMirostat2Sampler {
        pub fn new(_device: &Device, _mu: f32) -> Self {
            unreachable!("GPU backend is disabled")
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub struct GpuMirostat2SamplerParams {
        pub top_k: usize,
        pub temperature: f32,
        pub repetition_penalty: f32,
        pub tau: f32,
        pub eta: f32,
        pub random: f32,
    }

    pub trait CastTensor<T>: Sized {}
    impl<S, T> CastTensor<T> for S {}

    pub trait NextRankInner {
        type NextRank: LastRankInner + NextRankInner;
    }

    pub trait NextRank<const R: usize, T>: NextRankInner<NextRank = Tensor<R, T>> {}

    impl<const R: usize, T, X> NextRank<R, T> for X where X: NextRankInner<NextRank = Tensor<R, T>> {}

    pub trait LastRankInner {
        type LastRank: NextRankInner;
    }

    pub trait LastRank<const R: usize, T>: LastRankInner<LastRank = Tensor<R, T>> {}

    impl<const R: usize, T, X> LastRank<R, T> for X where X: LastRankInner<LastRank = Tensor<R, T>> {}

    pub trait SmallerRankInner<const R: usize> {
        type SmallerRank;
        type SmallerByArray;
    }

    pub trait SmallerRank<const R: usize, const S: usize, T>:
        SmallerRankInner<R, SmallerRank = Tensor<S, T>, SmallerByArray = [usize; R]>
    {
    }

    impl<const R: usize, const S: usize, T, X> SmallerRank<R, S, T> for X where
        X: SmallerRankInner<R, SmallerRank = Tensor<S, T>, SmallerByArray = [usize; R]>
    {
    }

    pub trait LargerRankInner<const R: usize> {
        type LargerRank;
        type LargerByArray;
    }

    pub trait LargerRank<const R: usize, const S: usize, T>:
        LargerRankInner<R, LargerRank = Tensor<S, T>, LargerByArray = [usize; R]>
    {
    }

    impl<const R: usize, const S: usize, T, X> LargerRank<R, S, T> for X where
        X: LargerRankInner<R, LargerRank = Tensor<S, T>, LargerByArray = [usize; R]>
    {
    }

    pub trait MaxRankInner {
        type MaxRank;
    }

    pub trait MaxRank<const R: usize, T>: MaxRankInner<MaxRank = Tensor<R, T>> {}

    impl<const R: usize, T, X> MaxRank<R, T> for X where X: MaxRankInner<MaxRank = Tensor<R, T>> {}

    fusor_types::impl_rank_traits!(Tensor);

    pub trait IntoTensorData<const R: usize, T> {}
    impl<const R: usize, T> IntoTensorData<R, T> for &[T] {}
    impl<const R: usize, T> IntoTensorData<R, T> for &Vec<T> {}
    impl<const R: usize, T> IntoTensorData<R, T> for Vec<&[T]> {}

    macro_rules! never {
        ($value:expr) => {
            match $value {
                Tensor::Disabled(_, never) => match *never {},
            }
        };
    }

    impl<const R: usize, D, T> fusor_types::FromArray<R, D, T, Device> for Tensor<R, D>
    where
        D: DataType,
        T: fusor_types::IntoFlatArray<D, R>,
    {
        fn from_array(_data: T, _device: &Device) -> Self {
            unreachable!("GPU backend is disabled")
        }
    }

    impl<const R: usize, T> Tensor<R, T> {
        pub fn new(_device: &Device, _data: impl IntoTensorData<R, T>) -> Self {
            unreachable!("GPU backend is disabled")
        }

        pub fn from_slice(_device: &Device, _shape: [usize; R], _data: &[T]) -> Self {
            unreachable!("GPU backend is disabled")
        }

        pub fn splat(_device: &Device, _value: T, _shape: [usize; R]) -> Self {
            unreachable!("GPU backend is disabled")
        }

        pub fn shape(&self) -> &[usize; R] {
            never!(self)
        }
        pub fn device(&self) -> &Device {
            never!(self)
        }
        pub fn key(&self) -> NodeIndex {
            never!(self)
        }

        pub async fn as_slice(&self) -> Result<TensorSlice<R, T, MappedBuffer>, Error>
        where
            T: DataType,
        {
            never!(self)
        }

        pub async fn top_k_pairs(&self, _k: usize) -> Result<(Vec<u32>, Vec<f32>), Error> {
            never!(self)
        }

        pub async fn sample_mirostat2_token(
            &self,
            _sampler: &mut GpuMirostat2Sampler,
            _previous_tokens: &[u32],
            _params: GpuMirostat2SamplerParams,
        ) -> Result<u32, Error> {
            never!(self)
        }

        pub async fn try_sample_mirostat2_token_q_mat(
            &self,
            _weights: &QMatrix,
            _sampler: &mut GpuMirostat2Sampler,
            _previous_tokens: &[u32],
            _params: GpuMirostat2SamplerParams,
        ) -> Result<Option<u32>, Error> {
            never!(self)
        }

        pub fn reshape<const R2: usize>(
            &self,
            _new_shape: impl ShapeWithOneHole<R2>,
        ) -> Tensor<R2, T> {
            never!(self)
        }
        pub fn restride<const R2: usize>(&self, _specs: [StrideSpec; R2]) -> Tensor<R2, T> {
            never!(self)
        }
        pub fn restride_layout<const R2: usize>(&self, _layout: Layout) -> Tensor<R2, T> {
            never!(self)
        }
        pub fn resize(&self, _new_shape: [usize; R]) -> Tensor<R, T> {
            never!(self)
        }
        pub fn flatten_all(&self) -> Tensor<1, T> {
            never!(self)
        }
        pub fn slice_assign_in_place(
            &self,
            _slice: [std::ops::Range<usize>; R],
            _value: &Self,
        ) -> Self {
            never!(self)
        }
        pub fn slice_assign(&self, _slice: [std::ops::Range<usize>; R], _value: &Self) -> Self {
            never!(self)
        }
        pub fn softmax(&self, _axis: usize) -> Self {
            never!(self)
        }
        pub fn softmax_last_dim<const R2: usize>(&self) -> Self {
            never!(self)
        }
        pub fn sum<const R2: usize>(&self, _axis: usize) -> Tensor<R2, T> {
            never!(self)
        }
        pub fn max<const R2: usize>(&self, _axis: usize) -> Tensor<R2, T> {
            never!(self)
        }
        pub fn min<const R2: usize>(&self, _axis: usize) -> Tensor<R2, T> {
            never!(self)
        }
        pub fn product<const R2: usize>(&self, _axis: usize) -> Tensor<R2, T> {
            never!(self)
        }
        pub fn add_<const R2: usize, const R3: usize>(
            &self,
            _rhs: &Tensor<R2, T>,
        ) -> Tensor<R3, T> {
            never!(self)
        }
        pub fn sub_<const R2: usize, const R3: usize>(
            &self,
            _rhs: &Tensor<R2, T>,
        ) -> Tensor<R3, T> {
            never!(self)
        }
        pub fn mul_<const R2: usize, const R3: usize>(
            &self,
            _rhs: &Tensor<R2, T>,
        ) -> Tensor<R3, T> {
            never!(self)
        }
        pub fn div_<const R2: usize, const R3: usize>(
            &self,
            _rhs: &Tensor<R2, T>,
        ) -> Tensor<R3, T> {
            never!(self)
        }
        pub fn pow(&self, _rhs: &Self) -> Self {
            never!(self)
        }
        pub fn pow_(&self, _rhs: &Self) -> Self {
            never!(self)
        }
        pub fn cast<T2>(&self) -> Tensor<R, T2>
        where
            T: CastTensor<T2>,
        {
            never!(self)
        }
        pub fn index_select(&self, _dimension: usize, _indexes: &Tensor<1, u32>) -> Self {
            never!(self)
        }
        pub fn mat_mul(&self, _rhs: &Self) -> Self {
            never!(self)
        }
        pub fn matmul(&self, _rhs: &Self) -> Self {
            never!(self)
        }
        pub fn q_mat_mul(&self, _weights: &QMatrix) -> Self {
            never!(self)
        }
        pub fn q_mat_mul_paired_silu_product(&self, _weights: &QMatrix) -> Self {
            never!(self)
        }
        pub fn q_mat_mul_add2(&self, _weights: &QMatrix, _first: &Self, _second: &Self) -> Self {
            never!(self)
        }
        pub fn rope_fused(&self, _cos: &Tensor<2, T>, _sin: &Tensor<2, T>) -> Self {
            never!(self)
        }
        pub fn rope_normal_fused(&self, _cos: &Tensor<2, T>, _sin: &Tensor<2, T>) -> Self {
            never!(self)
        }
        pub fn rope_pair_fused(
            &self,
            _k: &Self,
            _cos: &Tensor<2, T>,
            _sin: &Tensor<2, T>,
        ) -> (Self, Self) {
            never!(self)
        }
        pub fn rope_normal_pair_fused(
            &self,
            _k: &Self,
            _cos: &Tensor<2, T>,
            _sin: &Tensor<2, T>,
        ) -> (Self, Self) {
            never!(self)
        }
        pub fn rms_norm_fused<const W: usize, const OUT_RANK: usize>(
            &self,
            _weight: &Tensor<W, T>,
            _bias: Option<&Tensor<W, T>>,
            _eps: f32,
        ) -> Self {
            never!(self)
        }
        pub fn rms_norm_fused_no_bias<const W: usize, const OUT_RANK: usize>(
            &self,
            _weight: &Tensor<W, T>,
            _eps: f32,
        ) -> Self {
            never!(self)
        }
        pub fn rms_norm_residual_fused<const W: usize, const OUT_RANK: usize>(
            &self,
            _residual: &Self,
            _weight: &Tensor<W, T>,
            _bias: Option<&Tensor<W, T>>,
            _eps: f32,
        ) -> Self {
            never!(self)
        }
        pub fn appoximate_exp(&self) -> Self {
            never!(self)
        }
        pub fn less_appoximate_exp(&self) -> Self {
            never!(self)
        }
        pub fn tanh_exact(&self) -> Self {
            never!(self)
        }
        pub fn where_cond(&self, _on_true: &Self, _on_false: &Self) -> Self {
            never!(self)
        }
        pub fn eq<D: DataType>(&self, _rhs: T) -> Tensor<R, D> {
            never!(self)
        }
        pub fn ne<D: DataType>(&self, _rhs: T) -> Tensor<R, D> {
            never!(self)
        }
        pub fn lt<D: DataType>(&self, _rhs: T) -> Tensor<R, D> {
            never!(self)
        }
        pub fn lte<D: DataType>(&self, _rhs: T) -> Tensor<R, D> {
            never!(self)
        }
        pub fn mt<D: DataType>(&self, _rhs: T) -> Tensor<R, D> {
            never!(self)
        }
        pub fn mte<D: DataType>(&self, _rhs: T) -> Tensor<R, D> {
            never!(self)
        }
        pub fn pow_elementwise(&self, _exponent: T) -> Self {
            never!(self)
        }
        pub fn max_elementwise(&self, _element: T) -> Self {
            never!(self)
        }
        pub fn min_elementwise(&self, _element: T) -> Self {
            never!(self)
        }
        pub fn to_scalar(&self) -> impl std::future::Future<Output = Result<T, Error>> + '_ {
            async move { never!(self) }
        }
        pub fn abs(&self) -> Self {
            never!(self)
        }
        pub fn sqrt(&self) -> Self {
            never!(self)
        }
        pub fn exp(&self) -> Self {
            never!(self)
        }
        pub fn exp2(&self) -> Self {
            never!(self)
        }
        pub fn log(&self) -> Self {
            never!(self)
        }
        pub fn log2(&self) -> Self {
            never!(self)
        }
        pub fn sin(&self) -> Self {
            never!(self)
        }
        pub fn cos(&self) -> Self {
            never!(self)
        }
        pub fn tan(&self) -> Self {
            never!(self)
        }
        pub fn tanh(&self) -> Self {
            never!(self)
        }
        pub fn asin(&self) -> Self {
            never!(self)
        }
        pub fn acos(&self) -> Self {
            never!(self)
        }
        pub fn atan(&self) -> Self {
            never!(self)
        }
        pub fn sinh(&self) -> Self {
            never!(self)
        }
        pub fn cosh(&self) -> Self {
            never!(self)
        }
        pub fn asinh(&self) -> Self {
            never!(self)
        }
        pub fn acosh(&self) -> Self {
            never!(self)
        }
        pub fn atanh(&self) -> Self {
            never!(self)
        }
    }

    macro_rules! binary_op {
        ($trait:ident, $method:ident) => {
            impl<const R: usize, T> $trait for Tensor<R, T> {
                type Output = Self;
                fn $method(self, _rhs: Self) -> Self::Output {
                    never!(&self)
                }
            }

            impl<'a, const R: usize, T> $trait<&'a Tensor<R, T>> for Tensor<R, T> {
                type Output = Self;
                fn $method(self, _rhs: &'a Tensor<R, T>) -> Self::Output {
                    never!(&self)
                }
            }

            impl<'a, const R: usize, T> $trait<Tensor<R, T>> for &'a Tensor<R, T> {
                type Output = Tensor<R, T>;
                fn $method(self, _rhs: Tensor<R, T>) -> Self::Output {
                    never!(self)
                }
            }

            impl<'a, const R: usize, T> $trait for &'a Tensor<R, T> {
                type Output = Tensor<R, T>;
                fn $method(self, _rhs: Self) -> Self::Output {
                    never!(self)
                }
            }

            impl<const R: usize, T> $trait<T> for Tensor<R, T> {
                type Output = Self;
                fn $method(self, _rhs: T) -> Self::Output {
                    never!(&self)
                }
            }

            impl<'a, const R: usize, T> $trait<T> for &'a Tensor<R, T> {
                type Output = Tensor<R, T>;
                fn $method(self, _rhs: T) -> Self::Output {
                    never!(self)
                }
            }
        };
    }

    binary_op!(Add, add);
    binary_op!(Sub, sub);
    binary_op!(Mul, mul);
    binary_op!(Div, div);
    binary_op!(Rem, rem);

    impl<const R: usize, T> Neg for Tensor<R, T> {
        type Output = Self;
        fn neg(self) -> Self::Output {
            never!(&self)
        }
    }

    impl<'a, const R: usize, T> Neg for &'a Tensor<R, T> {
        type Output = Tensor<R, T>;
        fn neg(self) -> Self::Output {
            never!(self)
        }
    }

    pub trait WasmNotSend: Send {}
    pub trait WasmNotSync: Sync {}
    impl<T: Send> WasmNotSend for T {}
    impl<T: Sync> WasmNotSync for T {}
}

#[cfg(not(feature = "gpu"))]
pub use disabled::*;
