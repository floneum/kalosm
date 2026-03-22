//! Conformance tests for fusor operations across CPU and GPU backends.
//!
//! Tests are written using fusor-level ops (not raw StrideSpec) and run
//! on every available device to verify CPU/GPU parity.

use std::{
    error::Error,
    fmt::{Debug, Display},
    ops::Sub,
    pin::Pin,
};

use fusor::{DataType, Device, FromArray, SimdElement, Tensor};
use rand::{
    Rng, RngCore, SeedableRng,
    distr::{Distribution, StandardUniform, Uniform},
    rngs::StdRng,
};
use thiserror::Error;

/// Return all available devices: always CPU, plus GPU if available.
async fn devices() -> Vec<Device> {
    let mut devs = vec![Device::Cpu];
    if let Ok(gpu) = Device::gpu().await {
        devs.push(gpu);
    }
    devs
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

/// Generate a sequential f32 tensor: [0, 1, 2, ...].
pub fn sequential_tensor<const R: usize, T: DataType + SimdElement + From<usize>>(
    device: &Device,
    shape: [usize; R],
) -> Tensor<R, T> {
    let total: usize = shape.iter().product();
    let data: Vec<T> = (0..total).map(|i| T::from(i)).collect();
    Tensor::from_slice(device, shape, &data)
}

struct FuzzGenerator<const R: usize, T: SimdElement> {
    rng: rand::rngs::StdRng,
    distribution: Box<dyn Fn(&mut rand::rngs::StdRng) -> T>,
    shape: [usize; R],
    phantom: std::marker::PhantomData<T>,
}

impl<const R: usize, T: SimdElement + DataType> FuzzGenerator<R, T> {
    fn new(shape: [usize; R]) -> Self
    where
        StandardUniform: rand::distr::Distribution<T>,
    {
        Self {
            rng: rand::rngs::StdRng::from_os_rng(),
            distribution: Box::new(move |rng| StandardUniform.sample(rng)),
            shape,
            phantom: std::marker::PhantomData,
        }
    }

    fn with_rng(mut self, mut rng: impl RngCore) -> Self {
        self.rng = rand::rngs::StdRng::from_rng(&mut rng);
        self
    }

    fn with_distribution(mut self, distribution: impl Distribution<T> + 'static) -> Self {
        self.distribution = Box::new(move |rng| distribution.sample(rng));
        self
    }

    fn generate(&mut self, device: &Device) -> Tensor<R, T> {
        random_tensor(device, self.shape, &mut self.rng, &self.distribution)
    }
}

impl<const R: usize> FuzzGenerator<R, f32> {
    fn with_positive(mut self) -> Self {
        self.distribution =
            Box::new(move |rng| Uniform::new(0.0, 1.0).expect("0.0 < 1.0").sample(rng));
        self
    }
}

trait GenerateFromDevice {
    type Output;
    fn generate(&mut self, device: &Device) -> Self::Output;
}

impl<F, O> GenerateFromDevice for F
where
    F: FnMut(&Device) -> O,
{
    type Output = O;
    fn generate(&mut self, device: &Device) -> Self::Output {
        (self)(device)
    }
}

impl<const R: usize, T: SimdElement + DataType> GenerateFromDevice for FuzzGenerator<R, T> {
    type Output = Tensor<R, T>;
    fn generate(&mut self, device: &Device) -> Self::Output {
        self.generate(device)
    }
}

trait AsyncFnMutTuple<Args> {
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

trait GenTuple {
    type Output;
    fn generate(&mut self, device: &Device) -> Self::Output;
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
            fn generate(&mut self, device: &Device) -> Self::Output {
                let ($($type,)*) = self;
                (
                    $(
                        $type.generate(device),
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

trait ResolveTensorTuple {
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

            async fn resolve(self) -> Result<Self::Output, fusor::Error> {
                let ($($type,)*) = self;
                Ok((
                    $(
                        fusor::ToVec::to_vec(&$type.as_slice().await?),
                    )*
                ))
            }

            #[allow(unused)]
            fn extract_device(&self) -> Device {
                let ($($type,)*) = self;
                $(
                    let device = $type.device();
                    return device;
                )*
                unreachable!()
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

trait PushTuple<Tail> {
    type Output;
    fn push(self, new_last: Tail) -> Self::Output;
}

trait PopTuple {
    type First;
    type Rest;
    fn pop(self) -> (Self::First, Self::Rest);
}

macro_rules! impl_push_pop_tuple {
    ($first_type:ident $(,$type:ident)*) => {
        impl<$first_type $(,$type)*> PopTuple for ($first_type, $($type,)*) {
            type First = $first_type;
            type Rest = ($($type,)*);
            fn pop(self) -> (Self::First, Self::Rest) {
                let ($first_type, $($type,)*) = self;
                ($first_type, ($($type,)*))
            }
        }

        impl<$first_type, $($type,)* Tail> PushTuple<Tail> for ($first_type, $($type,)*) {
            type Output = ($first_type, $($type,)* Tail);
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
struct ItemMismatchError {
    device: Device,
    position: Vec<usize>,
    expected: String,
    actual: String,
}

impl ItemMismatchError {
    fn new(
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
        write!(
            f,
            "Item mismatch on device {:?}: expected {}, got {}",
            self.device, self.expected, self.actual
        )
    }
}

/// ```compile_fail
/// crate::assert(|x: fusor::Tensor<2, f32>| x.sin().to_concrete())
///         .arg(FuzzGenerator::new([10; 2]))
///         .build();
/// ```
struct AssertBuilder<T, U, Generators = (), Compare = ()> {
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
            runs: 1,
        }
    }
}

impl<T, U, Generators, Compare> AssertBuilder<T, U, Generators, Compare> {
    fn arg<Gen, O>(self, g: Gen) -> AssertBuilder<T, U, Generators::Output, Compare>
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

    fn compare_with<Cmp>(self, cmp: Cmp) -> AssertBuilder<T, U, Generators, Cmp>
    where
        Cmp: Fn(U, U) -> bool + 'static,
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

    fn runs(mut self, runs: usize) -> Self {
        self.runs = runs;
        self
    }

    fn devices(mut self, devices: impl IntoIterator<Item = Device>) -> Self {
        self.devices = Some(devices.into_iter().collect());
        self
    }

    fn equal_to(mut self, other: impl AsyncFnMutTuple<T, Output = U> + 'static) -> Self {
        self.to_validate.push(Box::new(other));
        self
    }

    fn equal_to_array_op<const R: usize, D, Arr>(
        self,
        mut other: impl AsyncFnMutTuple<T::Output, Output = Vec<Arr>> + Copy + Send + 'static,
    ) -> Self
    where
        T: ResolveTensorTuple,
        for<'a> U: FromArray<R, D, &'a Vec<Arr>, Device>,
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
                devices().await
            };
            for device in devices {
                for _ in 0..self.runs {
                    let args = self.generators.generate(&device);
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

trait IntoCompare<U> {
    type Error: Error;

    fn into_compare(
        self,
    ) -> impl for<'a> Fn(
        &'a U,
        &'a U,
    )
        -> Pin<Box<dyn std::future::Future<Output = Result<(), Self::Error>> + 'a>>
    + 'static;
}

impl<U, Cmp, E: Error> IntoCompare<U> for Cmp
where
    Cmp: for<'a> Fn(&'a U, &'a U) -> Pin<Box<dyn std::future::Future<Output = Result<(), E>> + 'a>>
        + 'static,
{
    type Error = E;

    fn into_compare(
        self,
    ) -> impl for<'a> Fn(
        &'a U,
        &'a U,
    )
        -> Pin<Box<dyn std::future::Future<Output = Result<(), Self::Error>> + 'a>>
    + 'static {
        self
    }
}

impl<const R: usize> IntoCompare<Tensor<R, u32>> for () {
    type Error = ItemMismatchError;

    fn into_compare(
        self,
    ) -> impl for<'a> Fn(
        &'a Tensor<R, u32>,
        &'a Tensor<R, u32>,
    )
        -> Pin<Box<dyn std::future::Future<Output = Result<(), Self::Error>> + 'a>>
    + 'static {
        |a, b| Box::pin(exact_eq(a, b))
    }
}

impl<const R: usize> IntoCompare<Tensor<R, f32>> for () {
    type Error = ItemMismatchError;

    fn into_compare(
        self,
    ) -> impl for<'a> Fn(
        &'a Tensor<R, f32>,
        &'a Tensor<R, f32>,
    )
        -> Pin<Box<dyn std::future::Future<Output = Result<(), Self::Error>> + 'a>>
    + 'static {
        |a, b| Box::pin(approx_eq(a, b, 1e-5))
    }
}

fn assert<T, U>(op: impl AsyncFnMutTuple<T, Output = U> + 'static) -> AssertBuilder<T, U> {
    AssertBuilder::new(op)
}

#[cfg(test)]
mod api_tests {
    use fusor::{FromArray, Tensor, ToVec};

    use crate::FuzzGenerator;

    #[tokio::test]
    async fn test_api() {
        crate::assert(async |x: fusor::Tensor<1, f32>| x.sin().to_concrete())
            .arg(FuzzGenerator::new([64; 1]))
            .equal_to_array_op(async |vec: Vec<f32>| {
                vec.iter().map(|&v| v.sin()).collect::<Vec<_>>()
            })
            .await
            .unwrap();
    }
}
