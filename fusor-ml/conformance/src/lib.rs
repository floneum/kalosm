//! Conformance tests for fusor operations across CPU and GPU backends.
//!
//! Tests are written using fusor-level ops (not raw StrideSpec) and run
//! on every available device to verify CPU/GPU parity.

mod builder;
mod comparison;
mod fuzz;
mod tuple_macros;

use fusor::{DataType, Device, SimdElement, Tensor};
use rand::rngs::StdRng;

pub use builder::AssertBuilder;
pub use comparison::{
    CompareFut, IntoCompare, ItemMismatchError, approx_compare, approx_eq, approx_or_relative_compare,
    approx_or_relative_eq, eq_with, exact_compare, exact_eq, relative_compare, relative_eq,
};
pub use fuzz::{FuzzGenerator, FuzzSizeSpec, GenerateFromDevice, IntoFuzzShape};
pub use tuple_macros::{AsyncFnMutTuple, GenTuple, PopTuple, PushTuple, ResolveTensorTuple};

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

pub fn assert<T, U>(op: impl AsyncFnMutTuple<T, Output = U> + 'static) -> AssertBuilder<T, U> {
    AssertBuilder::new(op)
}
