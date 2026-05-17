//! Conformance tests for fusor operations across CPU and GPU backends.
//!
//! Tests are written using fusor-level ops (not raw StrideSpec) and run
//! on every available device to verify CPU/GPU parity.

use std::{
    error::Error,
    fmt::{Debug, Display},
    ops::Sub,
    ops::{Range, RangeInclusive},
    pin::Pin,
    sync::Arc,
};

use fusor::{DataType, Device, FromArray, SimdElement, Tensor};
use half::f16;
use rand::{
    RngCore, SeedableRng,
    distr::{Distribution, StandardUniform, Uniform},
    rngs::StdRng,
};
use thiserror::Error;

fn require_gpu() -> bool {
    std::env::var("FUSOR_CONFORMANCE_REQUIRE_GPU")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

/// Return all available devices: always CPU, plus GPU if available.
pub async fn available_devices() -> Vec<Device> {
    let mut devs = vec![Device::Cpu];
    match Device::gpu().await {
        Ok(gpu) => devs.push(gpu),
        Err(err) => {
            assert!(
                !require_gpu(),
                "GPU conformance is required but no GPU device was available: {err}"
            );
        }
    }
    devs
}

/// Return devices that can run f16 tensor operations.
///
/// CPU uses the scalar f16 fallback. GPUs must expose
/// `wgpu::Features::SHADER_F16`; lavapipe in Linux CI does not.
pub async fn f16_capable_devices() -> Vec<Device> {
    available_devices()
        .await
        .into_iter()
        .filter(|d| match d {
            Device::Cpu => true,
            Device::Gpu(gpu) => gpu.f16_supported(),
        })
        .collect()
}

fn index_iter<const R: usize>(shape: [usize; R]) -> impl Iterator<Item = [usize; R]> {
    let total: usize = shape.iter().product();
    (0..total).map(move |flat| {
        let mut idx = [0usize; R];
        let mut rem = flat;
        for d in (0..R).rev() {
            idx[d] = rem % shape[d];
            rem /= shape[d];
        }
        idx
    })
}

/// Assert that two f32 tensors are element-wise close within `tol`.
pub async fn eq_with<const R: usize, T: DataType + SimdElement>(
    a: &Tensor<R, T>,
    b: &Tensor<R, T>,
    eq: impl Fn(T, T) -> bool,
) -> Result<(), ItemMismatchError> {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    let shape = a.shape();
    let sa = a.as_slice().await.unwrap();
    let sb = b.as_slice().await.unwrap();

    for index in index_iter(shape) {
        let va = sa[index];
        let vb = sb[index];
        if !eq(va, vb) {
            return Err(ItemMismatchError::new(
                a.device(),
                index,
                format!("{:?}", va),
                format!("{:?}", vb),
            ));
        }
    }

    Ok(())
}

/// Assert that two f32 tensors are element-wise close within `tol`.
pub async fn approx_eq<const R: usize, T: Sub + PartialOrd + DataType + SimdElement>(
    a: &Tensor<R, T>,
    b: &Tensor<R, T>,
    tol: T,
) -> Result<(), ItemMismatchError> {
    eq_with(a, b, |va, vb| {
        let diff = if va > vb { va - vb } else { vb - va };
        diff <= tol
    })
    .await
}

/// Assert that two tensors are element-wise equal.
pub async fn exact_eq<const R: usize, T: DataType + SimdElement + PartialEq>(
    a: &Tensor<R, T>,
    b: &Tensor<R, T>,
) -> Result<(), ItemMismatchError> {
    eq_with(a, b, |va, vb| va == vb).await
}

/// Assert that two f32 tensors are element-wise close within a *relative*
/// tolerance: `|a - b| <= rel_tol * max(|a|, |b|, eps)`.
///
/// Use this when reduction outputs grow with the reduced axis size and an
/// absolute tolerance becomes meaningless (e.g. a sum of 2025 values with
/// magnitude up to 5 has expected ~5e3 but absolute roundoff scales with
/// the magnitude of the result).
pub async fn relative_eq<const R: usize>(
    a: &Tensor<R, f32>,
    b: &Tensor<R, f32>,
    rel_tol: f32,
) -> Result<(), ItemMismatchError> {
    eq_with(a, b, |va, vb| {
        let diff = (va - vb).abs();
        let scale = va.abs().max(vb.abs()).max(f32::MIN_POSITIVE);
        diff <= rel_tol * scale
    })
    .await
}

/// Assert that two f32 tensors are element-wise close within either an
/// absolute tolerance or a relative tolerance.
///
/// Use this for outputs that can be near zero for some inputs but grow large
/// enough elsewhere that absolute roundoff alone becomes brittle.
pub async fn approx_or_relative_eq<const R: usize>(
    a: &Tensor<R, f32>,
    b: &Tensor<R, f32>,
    abs_tol: f32,
    rel_tol: f32,
) -> Result<(), ItemMismatchError> {
    eq_with(a, b, |va, vb| {
        let diff = (va - vb).abs();
        let scale = va.abs().max(vb.abs()).max(f32::MIN_POSITIVE);
        diff <= abs_tol || diff <= rel_tol * scale
    })
    .await
}

/// Generate a random f32 tensor with values in [-1, 1].
pub fn random_tensor<const R: usize, T: DataType + SimdElement>(
    device: &Device,
    shape: [usize; R],
    rng: &mut StdRng,
    sample: impl Fn(&mut StdRng) -> T,
) -> Tensor<R, T> {
    let total: usize = shape.iter().product();
    let data: Vec<T> = (0..total).map(|_| sample(rng)).collect();
    Tensor::from_slice(device, shape, &data)
}

/// Generate a sequential tensor: [0, 1, 2, ...].
///
/// This uses `From<u16>` so it works for both floating-point and integer tensor types
/// used throughout fusor conformance tests.
pub fn sequential_tensor<const R: usize, T: DataType + SimdElement + From<u16>>(
    device: &Device,
    shape: [usize; R],
) -> Tensor<R, T> {
    let total: usize = shape.iter().product();
    let data: Vec<T> = (0..total)
        .map(|i| T::from(u16::try_from(i).expect("sequential tensor index fits in u16")))
        .collect();
    Tensor::from_slice(device, shape, &data)
}

#[derive(Clone, Debug)]
pub enum FuzzSizeSpec {
    Fixed(usize),
    Choices(Arc<[usize]>),
    Range { start: usize, end_exclusive: usize },
}

impl FuzzSizeSpec {
    fn sample(&self, rng: &mut StdRng) -> usize {
        match self {
            FuzzSizeSpec::Fixed(size) => *size,
            FuzzSizeSpec::Choices(choices) => {
                assert!(
                    !choices.is_empty(),
                    "fuzz size choice list must contain at least one size"
                );
                let index = (rng.next_u64() as usize) % choices.len();
                choices[index]
            }
            FuzzSizeSpec::Range {
                start,
                end_exclusive,
            } => {
                assert!(
                    start < end_exclusive,
                    "fuzz size range must not be empty: {start}..{end_exclusive}"
                );
                Uniform::new(*start, *end_exclusive)
                    .expect("validated non-empty size range")
                    .sample(rng)
            }
        }
    }
}

impl From<usize> for FuzzSizeSpec {
    fn from(value: usize) -> Self {
        Self::Fixed(value)
    }
}

impl<const N: usize> From<[usize; N]> for FuzzSizeSpec {
    fn from(value: [usize; N]) -> Self {
        Self::Choices(Arc::from(value))
    }
}

impl From<Vec<usize>> for FuzzSizeSpec {
    fn from(value: Vec<usize>) -> Self {
        Self::Choices(Arc::from(value.into_boxed_slice()))
    }
}

impl From<Box<[usize]>> for FuzzSizeSpec {
    fn from(value: Box<[usize]>) -> Self {
        Self::Choices(Arc::from(value))
    }
}

impl From<Range<usize>> for FuzzSizeSpec {
    fn from(value: Range<usize>) -> Self {
        Self::Range {
            start: value.start,
            end_exclusive: value.end,
        }
    }
}

impl From<RangeInclusive<usize>> for FuzzSizeSpec {
    fn from(value: RangeInclusive<usize>) -> Self {
        let (start, end) = value.into_inner();
        Self::Range {
            start,
            end_exclusive: end
                .checked_add(1)
                .expect("inclusive fuzz size range upper bound overflowed"),
        }
    }
}

pub trait IntoFuzzShape<const R: usize> {
    fn into_shape_specs(self) -> [FuzzSizeSpec; R];
}

impl<const R: usize> IntoFuzzShape<R> for [usize; R] {
    fn into_shape_specs(self) -> [FuzzSizeSpec; R] {
        self.map(FuzzSizeSpec::from)
    }
}

impl<const R: usize> IntoFuzzShape<R> for [FuzzSizeSpec; R] {
    fn into_shape_specs(self) -> [FuzzSizeSpec; R] {
        self
    }
}

impl<const R: usize, const N: usize> IntoFuzzShape<R> for [[usize; N]; R] {
    fn into_shape_specs(self) -> [FuzzSizeSpec; R] {
        self.map(FuzzSizeSpec::from)
    }
}

impl<const R: usize> IntoFuzzShape<R> for [Range<usize>; R] {
    fn into_shape_specs(self) -> [FuzzSizeSpec; R] {
        self.map(FuzzSizeSpec::from)
    }
}

impl<const R: usize> IntoFuzzShape<R> for [RangeInclusive<usize>; R] {
    fn into_shape_specs(self) -> [FuzzSizeSpec; R] {
        self.map(FuzzSizeSpec::from)
    }
}

#[derive(Clone)]
pub struct FuzzGenerator<const R: usize, T: SimdElement> {
    value_seed: u64,
    shape_seed: u64,
    distribution: Arc<dyn Fn(&mut rand::rngs::StdRng) -> T + Send + Sync>,
    shape_specs: [FuzzSizeSpec; R],
    phantom: std::marker::PhantomData<T>,
}

impl<const R: usize, T: SimdElement + DataType> FuzzGenerator<R, T> {
    pub fn new(shape: impl IntoFuzzShape<R>) -> Self
    where
        StandardUniform: rand::distr::Distribution<T>,
    {
        Self::with_sampler(shape, |rng| StandardUniform.sample(rng))
    }

    /// Construct a fuzz generator from an explicit sampler closure.
    ///
    /// Use this for dtypes (e.g. `f16`) where `StandardUniform` is not implemented.
    pub fn with_sampler(
        shape: impl IntoFuzzShape<R>,
        sampler: impl Fn(&mut StdRng) -> T + Send + Sync + 'static,
    ) -> Self {
        Self {
            value_seed: 0,
            shape_seed: 0,
            distribution: Arc::new(sampler),
            shape_specs: shape.into_shape_specs(),
            phantom: std::marker::PhantomData,
        }
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.value_seed = seed;
        self
    }

    pub fn with_rng(mut self, mut rng: impl RngCore) -> Self {
        self.value_seed = rng.next_u64();
        self
    }

    pub fn with_shape_seed(mut self, seed: u64) -> Self {
        self.shape_seed = seed;
        self
    }

    pub fn with_distribution(
        mut self,
        distribution: impl Distribution<T> + Send + Sync + 'static,
    ) -> Self {
        self.distribution = Arc::new(move |rng| distribution.sample(rng));
        self
    }

    fn value_seed_for_run(&self, run: usize) -> u64 {
        self.value_seed
            ^ (run as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (R as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
    }

    fn shape_seed_for_run(&self, run: usize) -> u64 {
        self.shape_seed
            ^ (run as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
            ^ (R as u64).wrapping_mul(0x94D0_49BB_1331_11EB)
    }

    fn sample_shape(&self, rng: &mut StdRng) -> [usize; R] {
        self.shape_specs
            .clone()
            .map(|shape_spec| shape_spec.sample(rng))
    }

    #[cfg(test)]
    fn shape_for_run(&self, run: usize) -> [usize; R] {
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.shape_seed_for_run(run));
        self.sample_shape(&mut rng)
    }

    fn generate_for_run(&self, device: &Device, run: usize) -> Tensor<R, T> {
        let mut shape_rng = rand::rngs::StdRng::seed_from_u64(self.shape_seed_for_run(run));
        let shape = self.sample_shape(&mut shape_rng);
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.value_seed_for_run(run));
        let base = random_tensor(device, shape, &mut rng, &*self.distribution);
        // Vary layout based on run index: even runs stay contiguous, odd runs
        // get a non-contiguous layout so operations are tested with both.
        let strategy = run % 3;
        match strategy {
            0 => base,
            1 => make_transposed(base, &mut rng, &*self.distribution),
            _ => make_sliced(base, &mut rng, &*self.distribution),
        }
    }
}

/// Generate a contiguous tensor with the last two dimensions swapped, then
/// transpose it so the result has the correct shape but non-contiguous strides.
///
/// On GPU the lazy transpose view is preserved, so the op under test sees a
/// non-contiguous stride layout. On CPU `to_concrete()` materializes the view
/// into a contiguous backing buffer, so CPU only exercises the contiguous path.
fn make_transposed<const R: usize, T: SimdElement + DataType + Default>(
    tensor: Tensor<R, T>,
    rng: &mut StdRng,
    sample: &dyn Fn(&mut StdRng) -> T,
) -> Tensor<R, T> {
    if R < 2 {
        return tensor;
    }
    let shape = tensor.shape();
    // Build a shape with the last two dims swapped.
    let transposed_shape: [usize; R] = std::array::from_fn(|i| {
        if i == R - 2 {
            shape[R - 1]
        } else if i == R - 1 {
            shape[R - 2]
        } else {
            shape[i]
        }
    });
    let device = tensor.device();
    // Generate fresh contiguous data in the transposed shape, then
    // transpose so the logical shape matches `self.shape` but strides
    // are non-contiguous (the last two dims' strides are swapped).
    let contiguous = random_tensor(&device, transposed_shape, rng, sample);
    contiguous.transpose(R - 2, R - 1).to_concrete()
}

/// Generate a larger tensor and narrow it back to the original shape,
/// producing a tensor with a non-zero offset in the underlying buffer.
///
/// Same materialization caveat as [`make_transposed`]: on GPU the narrowed
/// view reaches the op under test, but on CPU `to_concrete()` materializes it
/// into a fresh contiguous buffer.
fn make_sliced<const R: usize, T: SimdElement + DataType + Default>(
    tensor: Tensor<R, T>,
    rng: &mut StdRng,
    sample: &dyn Fn(&mut StdRng) -> T,
) -> Tensor<R, T> {
    if R == 0 {
        return tensor;
    }
    let shape = tensor.shape();
    // Pick the last dimension to pad. We prepend `pad` extra elements
    // along that dimension so the resulting narrow has a non-zero offset.
    let pad_dim = R - 1;
    let pad = 1;
    let padded_size = shape[pad_dim] + pad;
    let padded_shape: [usize; R] =
        std::array::from_fn(|i| if i == pad_dim { padded_size } else { shape[i] });
    let device = tensor.device();
    let padded = random_tensor(&device, padded_shape, rng, sample);
    // Narrow away the extra padding, creating an offset view.
    padded.narrow(pad_dim, pad, shape[pad_dim]).to_concrete()
}

impl<const R: usize> FuzzGenerator<R, f32> {
    pub fn with_positive(mut self) -> Self {
        self.distribution =
            Arc::new(move |rng| Uniform::new(0.0, 1.0).expect("0.0 < 1.0").sample(rng));
        self
    }
}

#[doc(hidden)]
pub trait GenerateFromDevice {
    type Output;
    fn generate(&mut self, device: &Device, run: usize) -> Self::Output;
}

impl<F, O> GenerateFromDevice for F
where
    F: FnMut(&Device) -> O,
{
    type Output = O;
    fn generate(&mut self, device: &Device, _run: usize) -> Self::Output {
        (self)(device)
    }
}

impl<const R: usize, T: SimdElement + DataType> GenerateFromDevice for FuzzGenerator<R, T> {
    type Output = Tensor<R, T>;
    fn generate(&mut self, device: &Device, run: usize) -> Self::Output {
        self.generate_for_run(device, run)
    }
}

#[doc(hidden)]
pub trait AsyncFnMutTuple<Args> {
    type Output;

    fn call_mut<'a>(
        &'a mut self,
        args: Args,
    ) -> Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>>;
}

macro_rules! impl_fn_mut_tuple {
    ($($type:ident),*) => {
        impl<Fn, U, Fut, $($type),*> AsyncFnMutTuple<($($type,)*)> for Fn
        where
            Fn: FnMut($($type,)*) -> Fut,
            Fut: std::future::Future<Output = U> + Send + 'static,
        {
            type Output = U;
            #[allow(non_snake_case)]
            fn call_mut<'a>(&'a mut self, ($($type,)*): ($($type,)*)) -> Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>> {
                Box::pin((self)($($type,)*))
            }
        }
    };
}

impl_fn_mut_tuple!();
impl_fn_mut_tuple!(A);
impl_fn_mut_tuple!(A, B);
impl_fn_mut_tuple!(A, B, C);
impl_fn_mut_tuple!(A, B, C, D);
impl_fn_mut_tuple!(A, B, C, D, E);
impl_fn_mut_tuple!(A, B, C, D, E, F);

#[doc(hidden)]
pub trait GenTuple {
    type Output;
    fn generate(&mut self, device: &Device, run: usize) -> Self::Output;
}

macro_rules! impl_gen_tuple {
    ($($type:ident -> $type_out:ident),*) => {
        impl<$($type, $type_out),*> GenTuple for ($($type,)*)
        where
            $(
                $type: GenerateFromDevice<Output = $type_out>,
            )*
        {
            type Output = ($($type_out,)*);
            #[allow(non_snake_case)]
            fn generate(&mut self, device: &Device, run: usize) -> Self::Output {
                let ($($type,)*) = self;
                (
                    $(
                        $type.generate(device, run),
                    )*
                )
            }
        }
    };
}

impl_gen_tuple!(A -> AOut);
impl_gen_tuple!(A -> AOut, B -> BOut);
impl_gen_tuple!(A -> AOut, B -> BOut, C -> COut);
impl_gen_tuple!(A -> AOut, B -> BOut, C -> COut, D -> DOut);
impl_gen_tuple!(A -> AOut, B -> BOut, C -> COut, D -> DOut, E -> EOut);
impl_gen_tuple!(A -> AOut, B -> BOut, C -> COut, D -> DOut, E -> EOut, F -> FOut);

#[doc(hidden)]
pub trait ResolveTensorTuple {
    type Output;

    fn resolve(self) -> impl Future<Output = Result<Self::Output, fusor::Error>> + Send + 'static;

    fn extract_device(&self) -> Device;
}

macro_rules! impl_resolve_tensor_tuple {
    ($($type:ident = $rank:ident),*) => {
        impl<$(const $rank: usize,)* $($type: DataType + SimdElement),*> ResolveTensorTuple for ($(Tensor<$rank, $type>,)*)
            where
                $(
                    fusor_types::TensorSlice<$rank, $type, fusor::EitherMappedBuffer>: fusor::ToVec<Output: Send + 'static>,
                )*
        {
            type Output = ($(<fusor_types::TensorSlice<$rank, $type, fusor::EitherMappedBuffer> as fusor::ToVec>::Output,)*);

            #[allow(non_snake_case)]
            async fn resolve(self) -> Result<Self::Output, fusor::Error> {
                let ($($type,)*) = self;
                Ok((
                    $(
                        fusor::ToVec::to_vec(&$type.as_slice().await?),
                    )*
                ))
            }

            fn extract_device(&self) -> Device {
                self.0.device()
            }
        }
    };
}

impl_resolve_tensor_tuple!(A = N1);
impl_resolve_tensor_tuple!(A = N1, B = N2);
impl_resolve_tensor_tuple!(A = N1, B = N2, C = N3);
impl_resolve_tensor_tuple!(A = N1, B = N2, C = N3, D = N4);
impl_resolve_tensor_tuple!(A = N1, B = N2, C = N3, D = N4, E = N5);
impl_resolve_tensor_tuple!(A = N1, B = N2, C = N3, D = N4, E = N5, F = N6);

#[doc(hidden)]
pub trait PushTuple<Tail> {
    type Output;
    fn push(self, new_last: Tail) -> Self::Output;
}

#[doc(hidden)]
pub trait PopTuple {
    type First;
    type Rest;
    fn pop(self) -> (Self::First, Self::Rest);
}

macro_rules! impl_push_pop_tuple {
    ($first_type:ident $(,$type:ident)*) => {
        impl<$first_type $(,$type)*> PopTuple for ($first_type, $($type,)*) {
            type First = $first_type;
            type Rest = ($($type,)*);
            #[allow(non_snake_case)]
            fn pop(self) -> (Self::First, Self::Rest) {
                let ($first_type, $($type,)*) = self;
                ($first_type, ($($type,)*))
            }
        }

        impl<$first_type, $($type,)* Tail> PushTuple<Tail> for ($first_type, $($type,)*) {
            type Output = ($first_type, $($type,)* Tail);
            #[allow(non_snake_case)]
            fn push(self, new_last: Tail) -> Self::Output {
                let (head, $($type,)*) = self;
                (head, $($type,)* new_last)
            }
        }
    };
}

impl PopTuple for () {
    type First = ();
    type Rest = ();
    fn pop(self) -> (Self::First, Self::Rest) {
        ((), ())
    }
}
impl<Tail> PushTuple<Tail> for () {
    type Output = (Tail,);
    fn push(self, new_last: Tail) -> Self::Output {
        (new_last,)
    }
}
impl_push_pop_tuple!(A);
impl_push_pop_tuple!(A, B);
impl_push_pop_tuple!(A, B, C);
impl_push_pop_tuple!(A, B, C, D);
impl_push_pop_tuple!(A, B, C, D, E);
impl_push_pop_tuple!(A, B, C, D, E, F);

#[derive(Error, Debug)]
pub struct ItemMismatchError {
    device: Device,
    position: Vec<usize>,
    expected: String,
    actual: String,
}

impl ItemMismatchError {
    pub fn new(
        device: Device,
        position: impl IntoIterator<Item = usize>,
        expected: impl ToString,
        actual: impl ToString,
    ) -> Self {
        Self {
            device,
            position: position.into_iter().collect(),
            expected: expected.to_string(),
            actual: actual.to_string(),
        }
    }
}

impl Display for ItemMismatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let position = if self.position.is_empty() {
            String::from("<scalar>")
        } else {
            format!("{:?}", self.position)
        };
        write!(
            f,
            "Item mismatch on device {:?} at {}: expected {}, got {}",
            self.device, position, self.expected, self.actual
        )
    }
}

/// ```compile_fail
/// crate::assert(|x: fusor::Tensor<2, f32>| x.sin().to_concrete())
///         .arg(FuzzGenerator::<2, f32>::new([10; 2]))
///         .build();
/// ```
pub struct AssertBuilder<T, U, Generators = (), Compare = ()> {
    baseline: Box<dyn AsyncFnMutTuple<T, Output = U>>,
    to_validate: Vec<Box<dyn AsyncFnMutTuple<T, Output = U>>>,
    generators: Generators,
    compare: Compare,
    devices: Option<Vec<Device>>,
    runs: usize,
}

impl<T, U> AssertBuilder<T, U> {
    fn new(op: impl AsyncFnMutTuple<T, Output = U> + 'static) -> Self {
        Self {
            baseline: Box::new(op),
            to_validate: Vec::new(),
            generators: (),
            compare: (),
            devices: None,
            runs: 5,
        }
    }
}

impl<T, U, Generators, Compare> AssertBuilder<T, U, Generators, Compare> {
    pub fn arg<Gen, O>(self, g: Gen) -> AssertBuilder<T, U, Generators::Output, Compare>
    where
        Generators: PushTuple<Gen>,
        Gen: GenerateFromDevice<Output = O>,
    {
        AssertBuilder {
            baseline: self.baseline,
            to_validate: self.to_validate,
            generators: self.generators.push(g),
            compare: self.compare,
            devices: self.devices,
            runs: self.runs,
        }
    }

    pub fn compare_with<Cmp>(self, cmp: Cmp) -> AssertBuilder<T, U, Generators, Cmp>
    where
        Cmp: IntoCompare<U>,
    {
        AssertBuilder {
            baseline: self.baseline,
            to_validate: self.to_validate,
            generators: self.generators,
            compare: cmp,
            devices: self.devices,
            runs: self.runs,
        }
    }

    pub fn runs(mut self, runs: usize) -> Self {
        self.runs = runs;
        self
    }

    pub fn devices(mut self, devices: impl IntoIterator<Item = Device>) -> Self {
        self.devices = Some(devices.into_iter().collect());
        self
    }

    pub fn equal_to(mut self, other: impl AsyncFnMutTuple<T, Output = U> + 'static) -> Self {
        self.to_validate.push(Box::new(other));
        self
    }

    pub fn equal_to_resolved_op(
        self,
        mut other: impl AsyncFnMutTuple<T::Output, Output = U> + Copy + Send + 'static,
    ) -> Self
    where
        T: ResolveTensorTuple,
    {
        struct UnpackedTuple<T>(T);

        impl<F, Fut, I, O> AsyncFnMutTuple<I> for UnpackedTuple<F>
        where
            F: FnMut(I) -> Fut,
            Fut: std::future::Future<Output = O> + Send + 'static,
        {
            type Output = O;
            fn call_mut<'a>(
                &'a mut self,
                input: I,
            ) -> Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>> {
                Box::pin((self.0)(input))
            }
        }

        let wrapped = move |input: T| {
            let input = input.resolve();
            async move {
                let input = input.await.unwrap();
                other.call_mut(input).await
            }
        };

        self.equal_to(UnpackedTuple(wrapped))
    }

    pub fn equal_to_resolved_with_device(
        self,
        mut other: impl AsyncFnMutTuple<<T::Output as PushTuple<Device>>::Output, Output = U>
        + Copy
        + Send
        + 'static,
    ) -> Self
    where
        T: ResolveTensorTuple,
        T::Output: PushTuple<Device>,
    {
        struct UnpackedTuple<T>(T);

        impl<F, Fut, I, O> AsyncFnMutTuple<I> for UnpackedTuple<F>
        where
            F: FnMut(I) -> Fut,
            Fut: std::future::Future<Output = O> + Send + 'static,
        {
            type Output = O;
            fn call_mut<'a>(
                &'a mut self,
                input: I,
            ) -> Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>> {
                Box::pin((self.0)(input))
            }
        }

        let wrapped = move |input: T| {
            let device = input.extract_device();
            let input = input.resolve();
            async move {
                let input = input.await.unwrap();
                other.call_mut(input.push(device)).await
            }
        };

        self.equal_to(UnpackedTuple(wrapped))
    }

    pub fn equal_to_array_op<const R: usize, D, A>(
        self,
        mut other: impl AsyncFnMutTuple<T::Output, Output = A> + Copy + Send + 'static,
    ) -> Self
    where
        T: ResolveTensorTuple,
        for<'a> U: FromArray<R, D, &'a A, Device>,
    {
        struct UnpackedTuple<T>(T);

        impl<F, Fut, I, O> AsyncFnMutTuple<I> for UnpackedTuple<F>
        where
            F: FnMut(I) -> Fut,
            Fut: std::future::Future<Output = O> + Send + 'static,
        {
            type Output = O;
            fn call_mut<'a>(
                &'a mut self,
                input: I,
            ) -> Pin<Box<dyn std::future::Future<Output = Self::Output> + Send + 'a>> {
                Box::pin((self.0)(input))
            }
        }

        let wrapped = move |input: T| {
            let device = input.extract_device();
            let input = input.resolve();
            async move {
                let input = input.await.unwrap();
                let output = other.call_mut(input).await;
                U::from_array(&output, &device)
            }
        };

        self.equal_to(UnpackedTuple(wrapped))
    }
}

impl<T, U, Generators, Compare> IntoFuture for AssertBuilder<T, U, Generators, Compare>
where
    Generators: GenTuple<Output = T> + 'static,
    Compare: IntoCompare<U>,
    T: Clone + 'static,
    U: Clone + 'static,
{
    type Output = Result<(), Compare::Error>;
    type IntoFuture = Pin<Box<dyn std::future::Future<Output = Self::Output>>>;

    fn into_future(mut self) -> Self::IntoFuture {
        let compare_fn = self.compare.into_compare();
        let future = async move {
            let devices = if let Some(devs) = self.devices {
                devs
            } else {
                available_devices().await
            };
            for run in 0..self.runs {
                for device in &devices {
                    let args = self.generators.generate(device, run);
                    let expected = self.baseline.call_mut(args.clone()).await;
                    for to_validate in &mut self.to_validate {
                        let actual = to_validate.call_mut(args.clone()).await;
                        compare_fn(&expected, &actual).await?;
                    }
                }
            }
            Ok(())
        };
        Box::pin(future)
    }
}

/// Boxed future returned by a comparator: `&'a U, &'a U -> Result<(), E>`.
/// Aliased so the comparator type signatures stay readable.
pub type CompareFut<'a, E> = Pin<Box<dyn std::future::Future<Output = Result<(), E>> + 'a>>;

#[doc(hidden)]
pub trait IntoCompare<U> {
    type Error: Error;

    fn into_compare(self)
    -> impl for<'a> Fn(&'a U, &'a U) -> CompareFut<'a, Self::Error> + 'static;
}

impl<U, Cmp, E: Error> IntoCompare<U> for Cmp
where
    Cmp: for<'a> Fn(&'a U, &'a U) -> CompareFut<'a, E> + 'static,
{
    type Error = E;

    fn into_compare(
        self,
    ) -> impl for<'a> Fn(&'a U, &'a U) -> CompareFut<'a, Self::Error> + 'static {
        self
    }
}

impl<const R: usize> IntoCompare<Tensor<R, u32>> for () {
    type Error = ItemMismatchError;

    fn into_compare(
        self,
    ) -> impl for<'a> Fn(&'a Tensor<R, u32>, &'a Tensor<R, u32>) -> CompareFut<'a, Self::Error> + 'static
    {
        |a, b| Box::pin(exact_eq(a, b))
    }
}

impl<const R: usize> IntoCompare<Tensor<R, f32>> for () {
    type Error = ItemMismatchError;

    fn into_compare(
        self,
    ) -> impl for<'a> Fn(&'a Tensor<R, f32>, &'a Tensor<R, f32>) -> CompareFut<'a, Self::Error> + 'static
    {
        |a, b| Box::pin(approx_eq(a, b, 1e-5))
    }
}

impl<const R: usize> IntoCompare<Tensor<R, f16>> for () {
    type Error = ItemMismatchError;

    fn into_compare(
        self,
    ) -> impl for<'a> Fn(&'a Tensor<R, f16>, &'a Tensor<R, f16>) -> CompareFut<'a, Self::Error> + 'static
    {
        |a, b| Box::pin(approx_eq(a, b, f16::from_f32(1e-3)))
    }
}

pub fn exact_compare<const R: usize, T>()
-> impl for<'a> Fn(&'a Tensor<R, T>, &'a Tensor<R, T>) -> CompareFut<'a, ItemMismatchError> + Clone
where
    T: DataType + SimdElement + PartialEq,
{
    |a, b| Box::pin(exact_eq(a, b))
}

pub fn approx_compare<const R: usize, T>(
    tol: T,
) -> impl for<'a> Fn(&'a Tensor<R, T>, &'a Tensor<R, T>) -> CompareFut<'a, ItemMismatchError> + Clone
where
    T: Sub<Output = T> + PartialOrd + DataType + SimdElement + Copy,
{
    move |a, b| Box::pin(approx_eq(a, b, tol))
}

/// Compare-fn factory for [`relative_eq`]: pass `rel_tol` as a fraction
/// (e.g. `1e-3` for 0.1%).
pub fn relative_compare<const R: usize>(
    rel_tol: f32,
) -> impl for<'a> Fn(&'a Tensor<R, f32>, &'a Tensor<R, f32>) -> CompareFut<'a, ItemMismatchError> + Clone
{
    move |a, b| Box::pin(relative_eq(a, b, rel_tol))
}

pub fn approx_or_relative_compare<const R: usize>(
    abs_tol: f32,
    rel_tol: f32,
) -> impl for<'a> Fn(&'a Tensor<R, f32>, &'a Tensor<R, f32>) -> CompareFut<'a, ItemMismatchError> + Clone
{
    move |a, b| Box::pin(approx_or_relative_eq(a, b, abs_tol, rel_tol))
}

pub fn assert<T, U>(op: impl AsyncFnMutTuple<T, Output = U> + 'static) -> AssertBuilder<T, U> {
    AssertBuilder::new(op)
}

#[cfg(test)]
mod api_tests {
    use fusor::{Device, Tensor};

    use crate::{FuzzGenerator, FuzzSizeSpec};

    #[tokio::test]
    async fn test_api() {
        crate::assert(async |x: fusor::Tensor<1, f32>| x.sin().to_concrete())
            .arg(FuzzGenerator::<1, f32>::new([63..=65]))
            .equal_to_resolved_with_device(async |vec: Vec<f32>, device: Device| {
                let expected = vec.iter().map(|&v| v.sin()).collect::<Vec<_>>();
                Tensor::new(&device, &expected)
            })
            .runs(10)
            .await
            .unwrap();
    }

    #[test]
    fn fuzz_generator_accepts_size_choices_and_ranges() {
        let choice_generator =
            FuzzGenerator::<2, f32>::new([[255, 256, 257], [31, 32, 33]]).with_seed(1234);
        for run in 0..24 {
            let shape = choice_generator.shape_for_run(run);
            assert!([255, 256, 257].contains(&shape[0]));
            assert!([31, 32, 33].contains(&shape[1]));
        }

        let range_generator = FuzzGenerator::<2, f32>::new([255..=257, 31..=33]).with_seed(5678);
        for run in 0..24 {
            let shape = range_generator.shape_for_run(run);
            assert!((255..=257).contains(&shape[0]));
            assert!((31..=33).contains(&shape[1]));
        }

        let mixed_generator = FuzzGenerator::<2, f32>::new([
            FuzzSizeSpec::from([255, 256, 257]),
            FuzzSizeSpec::from(63..=65),
        ])
        .with_seed(9012);
        for run in 0..24 {
            let shape = mixed_generator.shape_for_run(run);
            assert!([255, 256, 257].contains(&shape[0]));
            assert!((63..=65).contains(&shape[1]));
        }
    }

    #[test]
    fn fuzz_generator_shapes_do_not_depend_on_value_seed() {
        let first = FuzzGenerator::<2, f32>::new([255..=257, 63..=65]).with_seed(1);
        let second = FuzzGenerator::<2, f32>::new([255..=257, 63..=65]).with_seed(2);
        for run in 0..24 {
            assert_eq!(first.shape_for_run(run), second.shape_for_run(run));
        }
    }
}
