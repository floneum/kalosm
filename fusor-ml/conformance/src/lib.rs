//! Conformance tests for fusor operations across CPU and GPU backends.
//!
//! Tests are written using fusor-level ops (not raw StrideSpec) and run
//! on every available device to verify CPU/GPU parity.

mod builder;
mod comparison;
mod fuzz;
#[cfg(test)]
mod goldens;
mod table;
mod tuple_macros;

extern crate self as fusor_conformance;

pub mod bench;
pub mod common;
pub mod suite;

use fusor::{DataType, Device, SimdElement, Tensor};
use rand::rngs::StdRng;

use builder::AssertBuilder;
pub use builder::{AssertionCase, AssertionCases};
pub use comparison::{
    CompareFut, IntoCompare, ItemMismatchError, ValueMismatchError, approx_compare, approx_eq,
    approx_or_relative_compare, approx_or_relative_eq, eq_with, exact_compare, exact_eq,
    exact_value_compare, relative_compare, relative_eq,
};
pub use fuzz::{FuzzGenerator, FuzzSizeSpec, GenerateFromDevice, IntoFuzzShape, with_shape_specs};
pub use table::{binary_fuzz_case, unary_fuzz_case};
pub use tuple_macros::{AsyncFnMutTuple, GenTuple, PopTuple, PushTuple, ResolveTensorTuple};

/// Error returned by a conformance case. A boxed error so cases (and the shared
/// `common::assert_*` helpers) can return a comparison mismatch or a free-form
/// message uniformly, and so failures can be reported rather than panicking —
/// the browser conformance runner cannot recover from a wasm panic.
pub type CaseError = Box<dyn std::error::Error>;

/// Result of a conformance case.
pub type CaseResult = Result<(), CaseError>;

/// `assert!` that returns `Err` from a `CaseResult`-returning case instead of
/// panicking, so browser failures are reported rather than aborting the app.
#[macro_export]
macro_rules! ensure {
    ($cond:expr $(,)?) => {
        if $cond {
        } else {
            return ::core::result::Result::Err(
                format!("assertion failed: {}", ::core::stringify!($cond)).into(),
            );
        }
    };
    ($cond:expr, $($arg:tt)+) => {
        if $cond {
        } else {
            return ::core::result::Result::Err(format!($($arg)+).into());
        }
    };
}

/// `assert_eq!` counterpart of [`ensure!`].
#[macro_export]
macro_rules! ensure_eq {
    ($a:expr, $b:expr $(,)?) => {{
        let (a, b) = (&$a, &$b);
        if a != b {
            return ::core::result::Result::Err(
                format!(
                    "assertion failed: `{} == {}`\n  left: `{:?}`\n right: `{:?}`",
                    ::core::stringify!($a),
                    ::core::stringify!($b),
                    a,
                    b
                )
                .into(),
            );
        }
    }};
    ($a:expr, $b:expr, $($arg:tt)+) => {{
        let (a, b) = (&$a, &$b);
        if a != b {
            return ::core::result::Result::Err(format!($($arg)+).into());
        }
    }};
}

/// `assert_ne!` counterpart of [`ensure!`].
#[macro_export]
macro_rules! ensure_ne {
    ($a:expr, $b:expr $(,)?) => {{
        let (a, b) = (&$a, &$b);
        if a == b {
            return ::core::result::Result::Err(
                format!(
                    "assertion failed: `{} != {}`\n  both: `{:?}`",
                    ::core::stringify!($a),
                    ::core::stringify!($b),
                    a
                )
                .into(),
            );
        }
    }};
    ($a:expr, $b:expr, $($arg:tt)+) => {{
        let (a, b) = (&$a, &$b);
        if a == b {
            return ::core::result::Result::Err(format!($($arg)+).into());
        }
    }};
}

fn require_gpu() -> bool {
    std::env::var("FUSOR_CONFORMANCE_REQUIRE_GPU")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

async fn acquire_gpu() -> Option<Device> {
    match Device::gpu().await {
        Ok(gpu) => Some(gpu),
        Err(err) => {
            assert!(
                !require_gpu(),
                "GPU conformance is required but no GPU device was available: {err}"
            );
            None
        }
    }
}

/// Acquire the GPU device. `None` means no GPU is available.
///
/// In the browser the full conformance suite runs ~140 cases that each call
/// [`available_devices`]; acquiring a fresh wgpu device per case is prohibitively
/// slow, so the handle is memoized for the life of the page. Natively the cache
/// is skipped — a thread-local holding a wgpu `Device` panics when dropped during
/// thread teardown (wgpu's `Drop` touches already-destroyed thread-locals), and
/// re-acquiring per case is cheap enough off the web.
#[cfg(not(target_arch = "wasm32"))]
async fn cached_gpu() -> Option<Device> {
    acquire_gpu().await
}

#[cfg(target_arch = "wasm32")]
async fn cached_gpu() -> Option<Device> {
    thread_local! {
        static GPU: std::cell::RefCell<Option<Option<Device>>> =
            const { std::cell::RefCell::new(None) };
    }
    if let Some(cached) = GPU.with(|cell| cell.borrow().clone()) {
        return cached;
    }
    let acquired = acquire_gpu().await;
    GPU.with(|cell| *cell.borrow_mut() = Some(acquired.clone()));
    acquired
}

/// Return all available devices: always CPU, plus GPU if available.
pub async fn available_devices() -> Vec<Device> {
    let mut devs = vec![Device::Cpu];
    if let Some(gpu) = cached_gpu().await {
        devs.push(gpu);
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

/// Flatten an iterator of per-row [`AssertionCase`]/[`AssertionCases`] into one
/// [`AssertionCases`].
///
/// A consolidated producer builds a `Vec` of rows (each from
/// [`unary_fuzz_case`]/[`binary_fuzz_case`] or any builder `.into_case(..)`) and
/// folds them into a single returned `AssertionCases`, replacing repeated
/// `assertions.push(...)` boilerplate.
///
/// ```ignore
/// pub fn unary_math_ops_match_host_reference() -> AssertionCases {
///     cases_from_rows([
///         unary_fuzz_case("elementwise_ops::sin", signed(), sin_op, f32::sin, approx_compare::<2, f32>(1e-4), 3),
///         unary_fuzz_case("elementwise_ops::cos", signed(), cos_op, f32::cos, approx_compare::<2, f32>(1e-4), 3),
///     ])
/// }
/// ```
pub fn cases_from_rows<C: Into<AssertionCases>>(
    rows: impl IntoIterator<Item = C>,
) -> AssertionCases {
    let mut out = AssertionCases::new();
    for row in rows {
        out.extend(row.into());
    }
    out
}
