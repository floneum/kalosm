//! Dtype conformance cases.
//!
//! `fusor::Tensor` carries `f32`, `f16`, and `u32`. f16 ops route through
//! a scalar fallback on CPU; these tests pin that
//! the fallback agrees with the GPU path and host-side reference math.
//!
//! `f64`/`i32`/`i64`/`u8` are not part of the unified `fusor::Tensor` enum
//! and are out of scope here.

use fusor::{Device, Tensor};
use fusor_conformance::{
    AssertionCase, AssertionCases, approx_compare, exact_compare, f16_capable_devices,
};
use half::f16;

fn f16s(values: &[f32]) -> Vec<f16> {
    values.iter().copied().map(f16::from_f32).collect()
}

// ---- u32 ----

pub fn u32_pairwise_add_matches_host_reference() -> AssertionCase {
    let lhs = [1u32, 2, 3, 4, 5, 6];
    let rhs = [10u32, 20, 30, 40, 50, 60];
    let sums: Vec<u32> = lhs.iter().zip(rhs.iter()).map(|(a, b)| a + b).collect();

    fusor_conformance::assert(move |device: Device| {
        let lhs = lhs;
        let rhs = rhs;
        async move {
            let l: Tensor<2, u32> = Tensor::from_slice(&device, [3, 2], &lhs);
            let r: Tensor<2, u32> = Tensor::from_slice(&device, [3, 2], &rhs);
            (&l + &r).to_concrete()
        }
    })
    .arg(|device: &Device| device.clone())
    .equal_to(move |device: Device| {
        let sums = sums.clone();
        async move { Tensor::from_slice(&device, [3, 2], &sums) }
    })
    .compare_with(exact_compare::<2, u32>())
    .runs(1)
    .into_case("dtypes::u32_pairwise_add_matches_host_reference")
}

// ---- f16 cast ----

pub fn f32_to_f16_round_trip_preserves_value() -> AssertionCase {
    let values = [0.0f32, 0.5, 1.25, -2.5, 3.75];
    let expected_values: Vec<f32> = values.iter().map(|x| f16::from_f32(*x).to_f32()).collect();

    fusor_conformance::assert(async |input: Tensor<1, f32>| {
        input.cast::<f16>().cast::<f32>().to_concrete()
    })
    .arg(move |device: &Device| Tensor::from_slice(device, [5], &values))
    .equal_to(move |input: Tensor<1, f32>| {
        let expected_values = expected_values.clone();
        async move { Tensor::from_slice(&input.device(), [5], &expected_values) }
    })
    .compare_with(approx_compare::<1, f32>(1e-6))
    .devices_async(f16_capable_devices())
    .runs(1)
    .into_case("dtypes::f32_to_f16_round_trip_preserves_value")
}

// ---- f16 element-wise unary ----

pub fn f16_unary_ops_match_host_reference() -> AssertionCases {
    let inputs = [0.5f32, 1.0, 1.5, 2.0, -0.5, -1.0];
    let pos_inputs: Vec<f32> = inputs.iter().map(|x| x.abs() + 0.5).collect();
    let mut assertions = AssertionCases::new();

    macro_rules! assert_unary {
        ($name:expr, $input:expr, $op:expr, $reference:expr, $tol:expr) => {{
            let input_values = $input.clone();
            let expected_values = $reference;
            assertions.push(fusor_conformance::assert($op)
                .arg(move |device: &Device| Tensor::from_slice(device, [6], &f16s(&input_values)))
                .equal_to(move |input: Tensor<1, f16>| {
                    let expected_values = expected_values.clone();
                    async move { Tensor::from_slice(&input.device(), [6], &f16s(&expected_values)) }
                })
                .compare_with(approx_compare::<1, f16>(f16::from_f32($tol)))
                .devices_async(f16_capable_devices())
                .runs(1)
                .into_case($name));
        }};
    }

    assert_unary!(
        "dtypes::f16_unary_ops_match_host_reference::abs",
        inputs.to_vec(),
        async |input: Tensor<1, f16>| input.abs().to_concrete(),
        inputs.iter().map(|x| x.abs()).collect::<Vec<_>>(),
        1e-3
    );
    assert_unary!(
        "dtypes::f16_unary_ops_match_host_reference::sin",
        inputs.to_vec(),
        async |input: Tensor<1, f16>| input.sin().to_concrete(),
        inputs.iter().map(|x| x.sin()).collect::<Vec<_>>(),
        2e-3
    );
    assert_unary!(
        "dtypes::f16_unary_ops_match_host_reference::cos",
        inputs.to_vec(),
        async |input: Tensor<1, f16>| input.cos().to_concrete(),
        inputs.iter().map(|x| x.cos()).collect::<Vec<_>>(),
        2e-3
    );
    assert_unary!(
        "dtypes::f16_unary_ops_match_host_reference::exp",
        inputs.to_vec(),
        async |input: Tensor<1, f16>| input.exp().to_concrete(),
        inputs.iter().map(|x| x.exp()).collect::<Vec<_>>(),
        1e-2
    );
    assert_unary!(
        "dtypes::f16_unary_ops_match_host_reference::sqrt",
        pos_inputs,
        async |input: Tensor<1, f16>| input.sqrt().to_concrete(),
        pos_inputs.iter().map(|x| x.sqrt()).collect::<Vec<_>>(),
        2e-3
    );
    assertions
}

// ---- f16 element-wise binary ----

pub fn f16_pairwise_ops_match_host_reference() -> AssertionCases {
    let lhs = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let rhs = [0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5];
    let sums: Vec<f32> = lhs.iter().zip(rhs.iter()).map(|(a, b)| a + b).collect();
    let diffs: Vec<f32> = lhs.iter().zip(rhs.iter()).map(|(a, b)| a - b).collect();
    let prods: Vec<f32> = lhs.iter().zip(rhs.iter()).map(|(a, b)| a * b).collect();
    let quots: Vec<f32> = lhs.iter().zip(rhs.iter()).map(|(a, b)| a / b).collect();
    let mut assertions = AssertionCases::new();

    macro_rules! assert_binary {
        ($name:expr, $op:expr, $reference:expr) => {{
            let expected_values = $reference;
            assertions.push(fusor_conformance::assert($op)
                .arg(move |device: &Device| Tensor::from_slice(device, [3, 2], &f16s(&lhs)))
                .arg(move |device: &Device| Tensor::from_slice(device, [3, 2], &f16s(&rhs)))
                .equal_to(move |l: Tensor<2, f16>, _r: Tensor<2, f16>| {
                    let expected_values = expected_values.clone();
                    async move { Tensor::from_slice(&l.device(), [3, 2], &f16s(&expected_values)) }
                })
                .compare_with(approx_compare::<2, f16>(f16::from_f32(1e-2)))
                .devices_async(f16_capable_devices())
                .runs(1)
                .into_case($name));
        }};
    }

    assert_binary!(
        "dtypes::f16_pairwise_ops_match_host_reference::add",
        async |l: Tensor<2, f16>, r: Tensor<2, f16>| (&l + &r).to_concrete(),
        sums
    );
    assert_binary!(
        "dtypes::f16_pairwise_ops_match_host_reference::sub",
        async |l: Tensor<2, f16>, r: Tensor<2, f16>| (&l - &r).to_concrete(),
        diffs
    );
    assert_binary!(
        "dtypes::f16_pairwise_ops_match_host_reference::mul",
        async |l: Tensor<2, f16>, r: Tensor<2, f16>| (&l * &r).to_concrete(),
        prods
    );
    assert_binary!(
        "dtypes::f16_pairwise_ops_match_host_reference::div",
        async |l: Tensor<2, f16>, r: Tensor<2, f16>| (&l / &r).to_concrete(),
        quots
    );
    assertions
}

// ---- f16 zeros + matmul ----

pub fn f16_zeros_matches_expected() -> AssertionCase {
    fusor_conformance::assert(async |device: Device| Tensor::<2, f16>::zeros(&device, [2, 3]))
        .arg(|device: &Device| device.clone())
        .equal_to(async |device: Device| Tensor::from_slice(&device, [2, 3], &f16s(&[0.0; 6])))
        .compare_with(exact_compare::<2, f16>())
        .devices_async(f16_capable_devices())
        .runs(1)
        .into_case("dtypes::f16_zeros_matches_expected")
}

pub fn f16_matmul_matches_host_reference() -> AssertionCase {
    // [[1],[3]] @ [[1, 2]] == [[1, 2], [3, 6]]
    let lhs = [1.0f32, 3.0];
    let rhs = [1.0f32, 2.0];
    let expected_vals = [1.0f32, 2.0, 3.0, 6.0];

    fusor_conformance::assert(async |l: Tensor<2, f16>, r: Tensor<2, f16>| {
        l.matmul(&r).to_concrete()
    })
    .arg(move |device: &Device| Tensor::from_slice(device, [2, 1], &f16s(&lhs)))
    .arg(move |device: &Device| Tensor::from_slice(device, [1, 2], &f16s(&rhs)))
    .equal_to(move |l: Tensor<2, f16>, _r: Tensor<2, f16>| async move {
        Tensor::from_slice(&l.device(), [2, 2], &f16s(&expected_vals))
    })
    .compare_with(approx_compare::<2, f16>(f16::from_f32(1e-2)))
    .devices_async(f16_capable_devices())
    .runs(1)
    .into_case("dtypes::f16_matmul_matches_host_reference")
}

// ---- f16 reductions ----

pub fn f16_reductions_match_host_reference() -> AssertionCases {
    // 3x2 = [[1, 2], [3, 4], [5, 6]] -> sum_axis0 = [9, 12], sum_axis1 = [3, 7, 11]
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let sum_axis0 = [9.0f32, 12.0];
    let sum_axis1 = [3.0f32, 7.0, 11.0];
    let max_axis0 = [5.0f32, 6.0];
    let min_axis0 = [1.0f32, 2.0];
    let mut assertions = AssertionCases::new();

    macro_rules! assert_reduction {
        ($name:expr, $op:expr, $shape:expr, $reference:expr, $tol:expr) => {{
            let expected_values = $reference;
            assertions.push(
                fusor_conformance::assert($op)
                    .arg(move |device: &Device| Tensor::from_slice(device, [3, 2], &f16s(&data)))
                    .equal_to(move |input: Tensor<2, f16>| async move {
                        Tensor::from_slice(&input.device(), $shape, &f16s(&expected_values))
                    })
                    .compare_with(approx_compare::<1, f16>(f16::from_f32($tol)))
                    .devices_async(f16_capable_devices())
                    .runs(1)
                    .into_case($name),
            );
        }};
    }

    assert_reduction!(
        "dtypes::f16_reductions_match_host_reference::sum_axis0",
        async |input: Tensor<2, f16>| input.sum::<1>(0),
        [2],
        sum_axis0,
        1e-2
    );
    assert_reduction!(
        "dtypes::f16_reductions_match_host_reference::sum_axis1",
        async |input: Tensor<2, f16>| input.sum::<1>(1),
        [3],
        sum_axis1,
        1e-2
    );
    assert_reduction!(
        "dtypes::f16_reductions_match_host_reference::max_axis0",
        async |input: Tensor<2, f16>| input.max::<1>(0),
        [2],
        max_axis0,
        1e-3
    );
    assert_reduction!(
        "dtypes::f16_reductions_match_host_reference::min_axis0",
        async |input: Tensor<2, f16>| input.min::<1>(0),
        [2],
        min_axis0,
        1e-3
    );
    assertions
}

pub fn f16_reduce_post_abs_matches_host_reference() -> AssertionCase {
    let data = [-1.0f32, -2.0, -3.0, -4.0, -5.0, -6.0];
    let expected_values = [9.0f32, 12.0];

    fusor_conformance::assert(async |input: Tensor<2, f16>| input.sum::<1>(0).abs().to_concrete())
        .arg(move |device: &Device| Tensor::from_slice(device, [3, 2], &f16s(&data)))
        .equal_to(move |input: Tensor<2, f16>| async move {
            Tensor::from_slice(&input.device(), [2], &f16s(&expected_values))
        })
        .compare_with(approx_compare::<1, f16>(f16::from_f32(1e-2)))
        .devices_async(f16_capable_devices())
        .runs(1)
        .into_case("dtypes::f16_reduce_post_abs_matches_host_reference")
}
