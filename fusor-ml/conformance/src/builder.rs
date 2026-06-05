use std::{future::Future, pin::Pin};

use fusor::{Device, FromArray, WasmNotSend};

use crate::{
    AsyncFnMutTuple, CaseError, CaseResult, GenTuple, GenerateFromDevice, IntoCompare, PushTuple,
    ResolveTensorTuple, available_devices, tuple_macros::BoxFuture,
};

type CaseFuture<'a> = Pin<Box<dyn Future<Output = CaseResult> + 'a>>;
type DevicesFuture = Pin<Box<dyn Future<Output = Vec<Device>>>>;

enum DeviceSelection {
    Fixed(Vec<Device>),
    Deferred(DevicesFuture),
}

pub struct AssertionCase {
    name: String,
    run: Box<dyn for<'a> FnOnce(&'a mut dyn FnMut(&str)) -> CaseFuture<'a>>,
}

#[derive(Default)]
pub struct AssertionCases {
    cases: Vec<AssertionCase>,
}

impl AssertionCases {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, case: AssertionCase) {
        self.cases.push(case);
    }

    pub fn extend(&mut self, cases: impl IntoIterator<Item = AssertionCase>) {
        self.cases.extend(cases);
    }

    pub fn into_vec(self) -> Vec<AssertionCase> {
        self.cases
    }
}

impl From<AssertionCase> for AssertionCases {
    fn from(case: AssertionCase) -> Self {
        Self { cases: vec![case] }
    }
}

impl From<Vec<AssertionCase>> for AssertionCases {
    fn from(cases: Vec<AssertionCase>) -> Self {
        Self { cases }
    }
}

impl IntoIterator for AssertionCases {
    type Item = AssertionCase;
    type IntoIter = std::vec::IntoIter<AssertionCase>;

    fn into_iter(self) -> Self::IntoIter {
        self.cases.into_iter()
    }
}

impl AssertionCase {
    pub fn new(
        name: impl Into<String>,
        run: impl for<'a> FnOnce(&'a mut dyn FnMut(&str)) -> CaseFuture<'a> + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            run: Box::new(run),
        }
    }

    pub fn from_result_future<E>(
        name: impl Into<String>,
        future: impl std::future::IntoFuture<Output = Result<(), E>> + 'static,
    ) -> Self
    where
        E: std::error::Error + 'static,
    {
        Self::new(name, |_progress| {
            Box::pin(async move {
                future.await.map_err(|err| {
                    let err: CaseError = Box::new(err);
                    err
                })
            })
        })
    }

    pub fn from_case_future(
        name: impl Into<String>,
        future: impl Future<Output = CaseResult> + 'static,
    ) -> Self {
        Self::new(name, |_progress| Box::pin(future))
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub async fn run(self) -> CaseResult {
        self.run_with_progress(&mut |_| {}).await
    }

    pub async fn run_with_progress(self, progress: &mut dyn FnMut(&str)) -> CaseResult {
        (self.run)(progress).await
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
    devices: Option<DeviceSelection>,
    runs: usize,
    baseline_on_test_device: bool,
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
            baseline_on_test_device: false,
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
            baseline_on_test_device: self.baseline_on_test_device,
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
            baseline_on_test_device: self.baseline_on_test_device,
        }
    }

    pub fn runs(mut self, runs: usize) -> Self {
        self.runs = runs;
        self
    }

    pub fn baseline_on_test_device(mut self) -> Self {
        self.baseline_on_test_device = true;
        self
    }

    pub fn devices(mut self, devices: impl IntoIterator<Item = Device>) -> Self {
        self.devices = Some(DeviceSelection::Fixed(devices.into_iter().collect()));
        self
    }

    pub fn devices_async(
        mut self,
        devices: impl std::future::IntoFuture<Output = Vec<Device>> + 'static,
    ) -> Self {
        self.devices = Some(DeviceSelection::Deferred(Box::pin(devices.into_future())));
        self
    }

    pub fn equal_to(mut self, other: impl AsyncFnMutTuple<T, Output = U> + 'static) -> Self {
        self.to_validate.push(Box::new(other));
        self
    }

    pub fn equal_to_resolved_op(
        self,
        mut other: impl AsyncFnMutTuple<T::Output, Output = U> + Copy + WasmNotSend + 'static,
    ) -> Self
    where
        T: ResolveTensorTuple,
        T::Output: 'static,
    {
        struct UnpackedTuple<T>(T);

        impl<F, Fut, I, O> AsyncFnMutTuple<I> for UnpackedTuple<F>
        where
            F: FnMut(I) -> Fut,
            Fut: std::future::Future<Output = O> + WasmNotSend + 'static,
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
        + WasmNotSend
        + 'static,
    ) -> Self
    where
        T: ResolveTensorTuple,
        T::Output: PushTuple<Device>,
        T::Output: 'static,
    {
        struct UnpackedTuple<T>(T);

        impl<F, Fut, I, O> AsyncFnMutTuple<I> for UnpackedTuple<F>
        where
            F: FnMut(I) -> Fut,
            Fut: std::future::Future<Output = O> + WasmNotSend + 'static,
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
        mut other: impl AsyncFnMutTuple<T::Output, Output = A> + Copy + WasmNotSend + 'static,
    ) -> Self
    where
        T: ResolveTensorTuple,
        T::Output: 'static,
        for<'a> U: FromArray<R, D, &'a A, Device>,
    {
        struct UnpackedTuple<T>(T);

        impl<F, Fut, I, O> AsyncFnMutTuple<I> for UnpackedTuple<F>
        where
            F: FnMut(I) -> Fut,
            Fut: std::future::Future<Output = O> + WasmNotSend + 'static,
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

impl<T, U, Generators, Compare> AssertBuilder<T, U, Generators, Compare>
where
    Generators: GenTuple<Output = T> + 'static,
    Compare: IntoCompare<U> + 'static,
    T: Clone + 'static,
    U: Clone + 'static,
{
    fn run_assertions<'a>(
        mut self,
        case_name: Option<String>,
        mut progress: Option<&'a mut dyn FnMut(&str)>,
    ) -> Pin<Box<dyn Future<Output = Result<(), Compare::Error>> + 'a>>
    where
        Self: 'a,
    {
        let compare_fn = self.compare.into_compare();
        let future = async move {
            let devices = match self.devices {
                Some(DeviceSelection::Fixed(devices)) => devices,
                Some(DeviceSelection::Deferred(devices)) => devices.await,
                None => available_devices().await,
            };
            let has_references = !self.to_validate.is_empty();
            for run in 0..self.runs {
                let run_label = self.generators.run_label(run);
                for (device_index, device) in devices.iter().enumerate() {
                    let baseline_device = match device {
                        Device::Cpu => device.clone(),
                        Device::Gpu(_) => Device::Cpu,
                    };
                    // Each GPU device is validated as itself plus three derived
                    // handles (see `device_test_variants`): subgroups available
                    // vs disabled, crossed with a cold vs poisoned buffer pool.
                    // The no-subgroup + poisoned-pool combination is the one the
                    // web build hits; both are properties of the device handle /
                    // its allocations, not global state. Contiguous vs strided
                    // inputs are woven across `runs` by the generators.
                    for (variant_index, variant) in
                        device_test_variants(device).into_iter().enumerate()
                    {
                        if let (Some(case_name), Some(progress)) =
                            (case_name.as_deref(), progress.as_mut())
                        {
                            let variant_name = assertion_variant_name(
                                case_name,
                                &run_label,
                                device_index,
                                device,
                                variant_index,
                            );
                            (**progress)(&variant_name);
                        }

                        let args = self.generators.generate(&variant, run);
                        if has_references {
                            let actual = self.baseline.call_mut(args.clone()).await;
                            for to_validate in &mut self.to_validate {
                                let expected = to_validate.call_mut(args.clone()).await;
                                compare_fn(&expected, &actual).await?;
                            }
                        } else {
                            let expected_args = if self.baseline_on_test_device {
                                self.generators.generate(&variant, run)
                            } else {
                                self.generators.generate(&baseline_device, run)
                            };
                            let expected = self.baseline.call_mut(expected_args).await;
                            let actual = self.baseline.call_mut(args).await;
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

impl<T, U, Generators, Compare> IntoFuture for AssertBuilder<T, U, Generators, Compare>
where
    Generators: GenTuple<Output = T> + 'static,
    Compare: IntoCompare<U> + 'static,
    T: Clone + 'static,
    U: Clone + 'static,
{
    type Output = Result<(), Compare::Error>;
    type IntoFuture = Pin<Box<dyn std::future::Future<Output = Self::Output>>>;

    fn into_future(self) -> Self::IntoFuture {
        self.run_assertions(None, None)
    }
}

impl<T, U, Generators, Compare> AssertBuilder<T, U, Generators, Compare>
where
    Self: IntoFuture<Output = Result<(), <Compare as IntoCompare<U>>::Error>>,
    Generators: GenTuple<Output = T> + 'static,
    Compare: IntoCompare<U> + 'static,
    T: Clone + 'static,
    U: Clone + 'static,
    <Compare as IntoCompare<U>>::Error: std::error::Error + 'static,
{
    pub fn into_case(self, name: impl Into<String>) -> AssertionCase {
        let name = name.into();
        let progress_name = name.clone();
        AssertionCase::new(name, move |progress| {
            Box::pin(async move {
                self.run_assertions(Some(progress_name), Some(progress))
                    .await
                    .map_err(|err| {
                        let err: CaseError = Box::new(err);
                        err
                    })
            })
        })
    }
}

fn assertion_variant_name(
    case_name: &str,
    run_label: &str,
    device_index: usize,
    device: &Device,
    variant_index: usize,
) -> String {
    match device {
        Device::Cpu => format!("{case_name}::{run_label}::device{device_index}_cpu"),
        Device::Gpu(_) => {
            let variant = match variant_index {
                0 => "subgroups_cold_pool",
                1 => "no_subgroups_cold_pool",
                2 => "subgroups_poisoned_pool",
                3 => "no_subgroups_poisoned_pool",
                _ => "unknown_gpu_variant",
            };
            format!("{case_name}::{run_label}::device{device_index}_gpu::{variant}")
        }
    }
}

/// The device handles every op is validated against. A GPU device is expanded
/// into the cross product of {subgroups, no subgroups} × {cold pool, poisoned
/// pool}; the cold/subgroup variants come first so they run before the poisoned
/// variants dirty the shared pool. The CPU device has no such variants.
pub(crate) fn device_test_variants(device: &Device) -> Vec<Device> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exact_value_compare;

    struct LabelledGenerator;

    impl crate::GenerateFromDevice for LabelledGenerator {
        type Output = usize;

        fn generate(&mut self, _device: &Device, run: usize) -> Self::Output {
            run
        }

        fn run_label_fragment(&self, run: usize) -> Option<&'static str> {
            Some(match run {
                0 => "alpha",
                1 => "beta",
                _ => "other",
            })
        }
    }

    #[tokio::test]
    async fn assertion_case_reports_each_builder_variant() {
        let case = crate::assert(async |_device: Device| 1usize)
            .arg(|device: &Device| device.clone())
            .equal_to(async |_device: Device| 1usize)
            .compare_with(exact_value_compare())
            .devices([Device::Cpu])
            .runs(2)
            .into_case("builder::progress");

        let mut progress = Vec::new();
        case.run_with_progress(&mut |name| progress.push(name.to_string()))
            .await
            .unwrap();

        assert_eq!(
            progress,
            [
                "builder::progress::run0::device0_cpu",
                "builder::progress::run1::device0_cpu",
            ]
        );
    }

    #[tokio::test]
    async fn assertion_case_reports_generator_run_label() {
        let case = crate::assert(async |value: usize| value)
            .arg(LabelledGenerator)
            .equal_to(async |value: usize| value)
            .compare_with(exact_value_compare())
            .devices([Device::Cpu])
            .runs(2)
            .into_case("builder::progress");

        let mut progress = Vec::new();
        case.run_with_progress(&mut |name| progress.push(name.to_string()))
            .await
            .unwrap();

        assert_eq!(
            progress,
            [
                "builder::progress::sample0_alpha::device0_cpu",
                "builder::progress::sample1_beta::device0_cpu",
            ]
        );
    }
}
