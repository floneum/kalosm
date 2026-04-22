//! Conformance: non-f32 dtype coverage.
//!
//! `fusor::Tensor` supports `f32`, `f16`, and `u32` at the type level, but
//! the unified element-wise/matmul/reduction APIs are only wired up for
//! `f32` (and `u32` in a few places). The pre-migration tests for
//! `f64`/`i32`/`i64`/`u8` targeted `fusor-cpu::ConcreteTensor`, a CPU-only
//! type that no longer carries tests in the unified API.
//!
//! This file restores the dtype regressions that *do* still compile through
//! the unified API:
//!
//! - **u32 element-wise add** — covered by `&Tensor<R, u32> + &Tensor<R, u32>`
//!   (was `core/src/pair_wise.rs::test_pair_wise_add_u32`).
//! - **f32 ↔ f16 cast round-trip** — `Tensor::cast::<f16>().cast::<f32>()`
//!   (was `core/src/element_wise.rs::test_f32_to_f16_cast`/`test_f16_to_f32_cast`).
//!
//! f16 element-wise / matmul tests were intentionally not restored because
//! `fusor_cpu::SimdUnaryOp<f16>` / `SimdBinaryOp<f16>` / `MatmulImpl for f16`
//! are not implemented today; bringing those tests back requires extending
//! the CPU SIMD impls first.

use fusor::Tensor;
use fusor_conformance::{approx_eq, available_devices, exact_eq};
use half::f16;

#[tokio::test]
async fn u32_pairwise_add_matches_host_reference() {
    let lhs = [1u32, 2, 3, 4, 5, 6];
    let rhs = [10u32, 20, 30, 40, 50, 60];
    let sums: Vec<u32> = lhs.iter().zip(rhs.iter()).map(|(a, b)| a + b).collect();

    for device in available_devices().await {
        let l: Tensor<2, u32> = Tensor::from_slice(&device, [3, 2], &lhs);
        let r: Tensor<2, u32> = Tensor::from_slice(&device, [3, 2], &rhs);
        let actual = (&l + &r).to_concrete();
        let expected: Tensor<2, u32> = Tensor::from_slice(&device, [3, 2], &sums);
        exact_eq(&actual, &expected).await.unwrap();
    }
}

#[tokio::test]
async fn f32_to_f16_round_trip_preserves_value() {
    let values = [0.0f32, 0.5, 1.25, -2.5, 3.75];
    for device in available_devices().await {
        let input: Tensor<1, f32> = Tensor::from_slice(&device, [5], &values);
        let round_tripped = input.cast::<f16>().cast::<f32>().to_concrete();
        let expected_values: Vec<f32> = values.iter().map(|x| f16::from_f32(*x).to_f32()).collect();
        let expected: Tensor<1, f32> = Tensor::from_slice(&device, [5], &expected_values);
        approx_eq(&round_tripped, &expected, 1e-6).await.unwrap();
    }
}
