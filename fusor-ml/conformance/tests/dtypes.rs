//! Conformance: non-f32 dtype coverage.
//!
//! `fusor::Tensor` carries `f32`, `f16`, and `u32`. f16 ops route through
//! a scalar fallback on CPU (`fusor_cpu::F16Scalar`); these tests pin that
//! the fallback agrees with the GPU path and host-side reference math.
//!
//! `f64`/`i32`/`i64`/`u8` are not part of the unified `fusor::Tensor` enum
//! and are out of scope here.

use fusor::Tensor;
use fusor_conformance::{approx_eq, available_devices, exact_eq};
use half::f16;

fn f16s(values: &[f32]) -> Vec<f16> {
    values.iter().copied().map(f16::from_f32).collect()
}

async fn assert_approx_f16<const R: usize>(
    a: &Tensor<R, f16>,
    b: &Tensor<R, f16>,
    tol: f16,
) {
    fusor_conformance::approx_eq(a, b, tol).await.unwrap();
}

// ---- u32 ----

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

// ---- f16 cast ----

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

// ---- f16 element-wise unary ----

#[tokio::test]
async fn f16_unary_ops_match_host_reference() {
    let inputs = [0.5f32, 1.0, 1.5, 2.0, -0.5, -1.0];
    let pos_inputs: Vec<f32> = inputs.iter().map(|x| x.abs() + 0.5).collect();

    for device in available_devices().await {
        let input: Tensor<1, f16> = Tensor::from_slice(&device, [6], &f16s(&inputs));

        // abs
        let actual = input.abs().to_concrete();
        let expected: Tensor<1, f16> =
            Tensor::from_slice(&device, [6], &f16s(&inputs.iter().map(|x| x.abs()).collect::<Vec<_>>()));
        assert_approx_f16(&actual, &expected, f16::from_f32(1e-3)).await;

        // sin
        let actual = input.sin().to_concrete();
        let expected: Tensor<1, f16> =
            Tensor::from_slice(&device, [6], &f16s(&inputs.iter().map(|x| x.sin()).collect::<Vec<_>>()));
        assert_approx_f16(&actual, &expected, f16::from_f32(2e-3)).await;

        // cos
        let actual = input.cos().to_concrete();
        let expected: Tensor<1, f16> =
            Tensor::from_slice(&device, [6], &f16s(&inputs.iter().map(|x| x.cos()).collect::<Vec<_>>()));
        assert_approx_f16(&actual, &expected, f16::from_f32(2e-3)).await;

        // exp
        let actual = input.exp().to_concrete();
        let expected: Tensor<1, f16> =
            Tensor::from_slice(&device, [6], &f16s(&inputs.iter().map(|x| x.exp()).collect::<Vec<_>>()));
        assert_approx_f16(&actual, &expected, f16::from_f32(1e-2)).await;

        // sqrt (positive domain)
        let pos_input: Tensor<1, f16> = Tensor::from_slice(&device, [6], &f16s(&pos_inputs));
        let actual = pos_input.sqrt().to_concrete();
        let expected: Tensor<1, f16> =
            Tensor::from_slice(&device, [6], &f16s(&pos_inputs.iter().map(|x| x.sqrt()).collect::<Vec<_>>()));
        assert_approx_f16(&actual, &expected, f16::from_f32(2e-3)).await;
    }
}

// ---- f16 element-wise binary ----

#[tokio::test]
async fn f16_pairwise_ops_match_host_reference() {
    let lhs = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5];
    let sums: Vec<f32> = lhs.iter().zip(rhs.iter()).map(|(a, b)| a + b).collect();
    let diffs: Vec<f32> = lhs.iter().zip(rhs.iter()).map(|(a, b)| a - b).collect();
    let prods: Vec<f32> = lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).collect();
    let quots: Vec<f32> = lhs.iter().zip(rhs.iter()).map(|(a, b)| a / b).collect();

    for device in available_devices().await {
        let l: Tensor<2, f16> = Tensor::from_slice(&device, [3, 2], &f16s(&lhs));
        let r: Tensor<2, f16> = Tensor::from_slice(&device, [3, 2], &f16s(&rhs));

        let actual = (&l + &r).to_concrete();
        let expected: Tensor<2, f16> = Tensor::from_slice(&device, [3, 2], &f16s(&sums));
        assert_approx_f16(&actual, &expected, f16::from_f32(1e-2)).await;

        let actual = (&l - &r).to_concrete();
        let expected: Tensor<2, f16> = Tensor::from_slice(&device, [3, 2], &f16s(&diffs));
        assert_approx_f16(&actual, &expected, f16::from_f32(1e-2)).await;

        let actual = (&l * &r).to_concrete();
        let expected: Tensor<2, f16> = Tensor::from_slice(&device, [3, 2], &f16s(&prods));
        assert_approx_f16(&actual, &expected, f16::from_f32(1e-2)).await;

        let actual = (&l / &r).to_concrete();
        let expected: Tensor<2, f16> = Tensor::from_slice(&device, [3, 2], &f16s(&quots));
        assert_approx_f16(&actual, &expected, f16::from_f32(1e-2)).await;
    }
}

// ---- f16 zeros + matmul ----

#[tokio::test]
async fn f16_zeros_matches_expected() {
    for device in available_devices().await {
        let zeros: Tensor<2, f16> = Tensor::<2, f16>::zeros(&device, [2, 3]);
        let expected: Tensor<2, f16> =
            Tensor::from_slice(&device, [2, 3], &f16s(&[0.0; 6]));
        exact_eq(&zeros, &expected).await.unwrap();
    }
}

#[tokio::test]
async fn f16_matmul_matches_host_reference() {
    // [[1],[3]] @ [[1, 2]] == [[1, 2], [3, 6]]
    let lhs = [1.0f32, 3.0];
    let rhs = [1.0f32, 2.0];
    let expected_vals = [1.0f32, 2.0, 3.0, 6.0];

    for device in available_devices().await {
        let l: Tensor<2, f16> = Tensor::from_slice(&device, [2, 1], &f16s(&lhs));
        let r: Tensor<2, f16> = Tensor::from_slice(&device, [1, 2], &f16s(&rhs));
        let actual = l.matmul(&r).to_concrete();
        let expected: Tensor<2, f16> =
            Tensor::from_slice(&device, [2, 2], &f16s(&expected_vals));
        assert_approx_f16(&actual, &expected, f16::from_f32(1e-2)).await;
    }
}

// ---- f16 reductions ----

#[tokio::test]
async fn f16_reductions_match_host_reference() {
    // 3x2 = [[1, 2], [3, 4], [5, 6]] -> sum_axis0 = [9, 12], sum_axis1 = [3, 7, 11]
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let sum_axis0 = [9.0f32, 12.0];
    let sum_axis1 = [3.0f32, 7.0, 11.0];
    let max_axis0 = [5.0f32, 6.0];
    let min_axis0 = [1.0f32, 2.0];

    for device in available_devices().await {
        let input: Tensor<2, f16> = Tensor::from_slice(&device, [3, 2], &f16s(&data));

        let actual = input.sum::<1>(0);
        let expected: Tensor<1, f16> = Tensor::from_slice(&device, [2], &f16s(&sum_axis0));
        fusor_conformance::approx_eq(&actual, &expected, f16::from_f32(1e-2))
            .await
            .unwrap();

        let actual = input.sum::<1>(1);
        let expected: Tensor<1, f16> = Tensor::from_slice(&device, [3], &f16s(&sum_axis1));
        fusor_conformance::approx_eq(&actual, &expected, f16::from_f32(1e-2))
            .await
            .unwrap();

        let actual = input.max::<1>(0);
        let expected: Tensor<1, f16> = Tensor::from_slice(&device, [2], &f16s(&max_axis0));
        fusor_conformance::approx_eq(&actual, &expected, f16::from_f32(1e-3))
            .await
            .unwrap();

        let actual = input.min::<1>(0);
        let expected: Tensor<1, f16> = Tensor::from_slice(&device, [2], &f16s(&min_axis0));
        fusor_conformance::approx_eq(&actual, &expected, f16::from_f32(1e-3))
            .await
            .unwrap();
    }
}
