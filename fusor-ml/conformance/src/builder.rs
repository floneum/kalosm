use std::pin::Pin;

use fusor::{Device, FromArray};

use crate::{
    AsyncFnMutTuple, GenTuple, GenerateFromDevice, IntoCompare, PushTuple, ResolveTensorTuple,
    available_devices,
    tuple_macros::{BoxFuture, MaybeSend},
};

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
    pub(crate) fn new(op: impl AsyncFnMutTuple<T, Output = U> + 'static) -> Self {
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
        mut other: impl AsyncFnMutTuple<T::Output, Output = U> + Copy + MaybeSend + 'static,
    ) -> Self
    where
        T: ResolveTensorTuple,
        T::Output: 'static,
    {
        struct UnpackedTuple<T>(T);

        #[cfg(not(target_arch = "wasm32"))]
        impl<F, Fut, I, O> AsyncFnMutTuple<I> for UnpackedTuple<F>
        where
            F: FnMut(I) -> Fut,
            Fut: std::future::Future<Output = O> + Send + 'static,
        {
            type Output = O;
            fn call_mut<'a>(&'a mut self, input: I) -> BoxFuture<'a, Self::Output> {
                Box::pin((self.0)(input))
            }
        }

        #[cfg(target_arch = "wasm32")]
        impl<F, Fut, I, O> AsyncFnMutTuple<I> for UnpackedTuple<F>
        where
            F: FnMut(I) -> Fut,
            Fut: std::future::Future<Output = O> + 'static,
        {
            type Output = O;
            fn call_mut<'a>(&'a mut self, input: I) -> BoxFuture<'a, Self::Output> {
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
        + MaybeSend
        + 'static,
    ) -> Self
    where
        T: ResolveTensorTuple,
        T::Output: PushTuple<Device>,
        T::Output: 'static,
    {
        struct UnpackedTuple<T>(T);

        #[cfg(not(target_arch = "wasm32"))]
        impl<F, Fut, I, O> AsyncFnMutTuple<I> for UnpackedTuple<F>
        where
            F: FnMut(I) -> Fut,
            Fut: std::future::Future<Output = O> + Send + 'static,
        {
            type Output = O;
            fn call_mut<'a>(&'a mut self, input: I) -> BoxFuture<'a, Self::Output> {
                Box::pin((self.0)(input))
            }
        }

        #[cfg(target_arch = "wasm32")]
        impl<F, Fut, I, O> AsyncFnMutTuple<I> for UnpackedTuple<F>
        where
            F: FnMut(I) -> Fut,
            Fut: std::future::Future<Output = O> + 'static,
        {
            type Output = O;
            fn call_mut<'a>(&'a mut self, input: I) -> BoxFuture<'a, Self::Output> {
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
        mut other: impl AsyncFnMutTuple<T::Output, Output = A> + Copy + MaybeSend + 'static,
    ) -> Self
    where
        T: ResolveTensorTuple,
        T::Output: 'static,
        for<'a> U: FromArray<R, D, &'a A, Device>,
    {
        struct UnpackedTuple<T>(T);

        #[cfg(not(target_arch = "wasm32"))]
        impl<F, Fut, I, O> AsyncFnMutTuple<I> for UnpackedTuple<F>
        where
            F: FnMut(I) -> Fut,
            Fut: std::future::Future<Output = O> + Send + 'static,
        {
            type Output = O;
            fn call_mut<'a>(&'a mut self, input: I) -> BoxFuture<'a, Self::Output> {
                Box::pin((self.0)(input))
            }
        }

        #[cfg(target_arch = "wasm32")]
        impl<F, Fut, I, O> AsyncFnMutTuple<I> for UnpackedTuple<F>
        where
            F: FnMut(I) -> Fut,
            Fut: std::future::Future<Output = O> + 'static,
        {
            type Output = O;
            fn call_mut<'a>(&'a mut self, input: I) -> BoxFuture<'a, Self::Output> {
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
                    let baseline_device = match device {
                        Device::Cpu => device.clone(),
                        Device::Gpu(_) => Device::Cpu,
                    };
                    let expected_args = self.generators.generate(&baseline_device, run);
                    let expected = self.baseline.call_mut(expected_args).await;
                    // Each GPU device is validated as itself plus three derived
                    // handles (see `device_test_variants`): subgroups available
                    // vs disabled, crossed with a cold vs poisoned buffer pool.
                    // The no-subgroup + poisoned-pool combination is the one the
                    // web build hits; both are properties of the device handle /
                    // its allocations, not global state. Contiguous vs strided
                    // inputs are woven across `runs` by the generators.
                    for variant in device_test_variants(device) {
                        let args = self.generators.generate(&variant, run);
                        for to_validate in &mut self.to_validate {
                            let actual = to_validate.call_mut(args.clone()).await;
                            compare_fn(&expected, &actual).await?;
                        }
                    }
                }
            }
            Ok(())
        };
        Box::pin(future)
    }
}

/// The device handles every op is validated against. A GPU device is expanded
/// into the cross product of {subgroups, no subgroups} × {cold pool, poisoned
/// pool}; the cold/subgroup variants come first so they run before the poisoned
/// variants dirty the shared pool. The CPU device has no such variants.
fn device_test_variants(device: &Device) -> Vec<Device> {
    match device {
        Device::Cpu => vec![device.clone()],
        Device::Gpu(_) => vec![
            device.clone(),
            device.without_subgroups(),
            device.with_poisoned_allocations(),
            device.without_subgroups().with_poisoned_allocations(),
        ],
    }
}
