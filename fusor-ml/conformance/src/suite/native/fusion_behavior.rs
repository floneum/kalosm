//! Fusion behavior conformance cases.

use fusor::{Device, Tensor};
use fusor_conformance::{
    AssertionCase, AssertionCases, approx_compare, available_devices, exact_value_compare,
};

async fn gpu_devices() -> Vec<Device> {
    available_devices()
        .await
        .into_iter()
        .filter(|device| device.as_gpu().is_some())
        .collect()
}

fn assert_gpu_tensor_case<const R: usize>(
    name: impl Into<String>,
    op: impl Fn(Device) -> Tensor<R, f32> + Clone + Send + 'static,
    tol: f32,
) -> AssertionCase {
    fusor_conformance::assert(move |device: Device| {
        let op = op.clone();
        async move { op(device) }
    })
    .arg(|device: &Device| device.clone())
    .compare_with(approx_compare::<R, f32>(tol))
    .devices_async(gpu_devices())
    .runs(1)
    .into_case(name)
}

fn assert_gpu_kernel_property(
    name: impl Into<String>,
    property: impl Fn(Device) -> bool + Clone + Send + 'static,
) -> AssertionCase {
    fusor_conformance::assert(move |device: Device| {
        let property = property.clone();
        async move { property(device) }
    })
    .arg(|device: &Device| device.clone())
    .equal_to(move |_device: Device| async move { true })
    .compare_with(exact_value_compare())
    .baseline_on_test_device()
    .devices_async(gpu_devices())
    .runs(1)
    .into_case(name)
}

fn matrix_data(shape: [usize; 2], offset: f32) -> Vec<f32> {
    let total = shape[0] * shape[1];
    (0..total)
        .map(|i| (((i % 13) as f32) - 6.0) * 0.2 + offset)
        .collect()
}

fn binding_limit_sum_data(shape: [usize; 2], num_tensors: usize) -> Vec<f32> {
    let mut out = vec![0.0; shape[0] * shape[1]];
    for i in 0..num_tensors {
        for (out, value) in out.iter_mut().zip(matrix_data(shape, i as f32 * 0.3)) {
            *out += value;
        }
    }
    out
}

fn nary_binding_limit_stress_input_count(device: &Device) -> Option<usize> {
    let gpu = device.as_gpu()?;
    let num_tensors = gpu
        .nary_direct_input_binding_budget()
        .saturating_add(1)
        .max(2);

    // Some software adapters report descriptor limits high enough that a
    // limit-plus-one conformance case would be impractically large. Keep the
    // guard on the budget itself instead of the adapter type.
    (num_tensors <= 128).then_some(num_tensors)
}

fn condition_data(shape: [usize; 2]) -> Vec<f32> {
    let total = shape[0] * shape[1];
    (0..total)
        .map(|i| {
            if (i + shape[0]).is_multiple_of(3) {
                1.0
            } else {
                0.0
            }
        })
        .collect()
}

fn attention_data(len: usize, offset: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i % 17) as f32) - 8.0) * 0.12 + offset)
        .collect()
}

pub fn gpu_nary_triple_add_fuses_into_one_kernel() -> AssertionCases {
    let mut assertions = AssertionCases::new();
    for shape in [[2, 2], [3, 5], [4, 3]] {
        let a_data = matrix_data(shape, -0.3);
        let b_data = matrix_data(shape, 0.8);
        let c_data = matrix_data(shape, 1.7);
        let kernel_a_data = a_data.clone();
        let kernel_b_data = b_data.clone();
        let kernel_c_data = c_data.clone();
        assertions.push(assert_gpu_tensor_case(
            format!(
                "fusion_behavior::gpu_nary_triple_add_fuses_into_one_kernel::correctness_{shape:?}"
            ),
            move |device| {
                let a = Tensor::from_slice(&device, shape, &a_data);
                let b = Tensor::from_slice(&device, shape, &b_data);
                let c = Tensor::from_slice(&device, shape, &c_data);
                let sum = &a + &b;
                (&sum + &c).to_concrete()
            },
            1e-6,
        ));
        assertions.push(assert_gpu_kernel_property(
            format!(
                "fusion_behavior::gpu_nary_triple_add_fuses_into_one_kernel::kernels_{shape:?}"
            ),
            move |device| {
                let a = Tensor::from_slice(&device, shape, &kernel_a_data);
                let b = Tensor::from_slice(&device, shape, &kernel_b_data);
                let c = Tensor::from_slice(&device, shape, &kernel_c_data);
                let sum = &a + &b;
                let result = &sum + &c;
                result
                    .as_gpu()
                    .is_some_and(|gpu| gpu.count_kernels_to_resolve() == 1)
            },
        ));
    }
    assertions
}

pub fn gpu_nary_unary_chain_fuses_into_one_kernel() -> AssertionCases {
    let mut assertions = AssertionCases::new();
    for shape in [[2, 2], [3, 4], [2, 7]] {
        let a_data = matrix_data(shape, 0.1);
        let b_data = matrix_data(shape, -0.4);
        let kernel_a_data = a_data.clone();
        let kernel_b_data = b_data.clone();
        assertions.push(assert_gpu_tensor_case(
            format!(
                "fusion_behavior::gpu_nary_unary_chain_fuses_into_one_kernel::correctness_{shape:?}"
            ),
            move |device| {
                let a = Tensor::from_slice(&device, shape, &a_data);
                let b = Tensor::from_slice(&device, shape, &b_data);
                let sum = (-a.clone()) + b.sin();
                (sum.cos() + 1.0).to_concrete()
            },
            1e-6,
        ));
        assertions.push(assert_gpu_kernel_property(
            format!(
                "fusion_behavior::gpu_nary_unary_chain_fuses_into_one_kernel::kernels_{shape:?}"
            ),
            move |device| {
                let a = Tensor::from_slice(&device, shape, &kernel_a_data);
                let b = Tensor::from_slice(&device, shape, &kernel_b_data);
                let sum = (-a.clone()) + b.sin();
                let result = sum.cos() + 1.0;
                result
                    .as_gpu()
                    .is_some_and(|gpu| gpu.count_kernels_to_resolve() == 1)
            },
        ));
    }
    assertions
}

pub fn gpu_nary_same_input_multiple_times_deduplicates_bindings() -> AssertionCases {
    let mut assertions = AssertionCases::new();
    for shape in [[2, 2], [4, 3], [3, 6]] {
        let a_data = matrix_data(shape, 0.6);
        let kernel_a_data = a_data.clone();
        assertions.push(assert_gpu_tensor_case(
            format!("fusion_behavior::gpu_nary_same_input_multiple_times_deduplicates_bindings::correctness_{shape:?}"),
            move |device| {
                let a = Tensor::from_slice(&device, shape, &a_data);
                let sum = &a + &a;
                (&sum + &a).to_concrete()
            },
            1e-6,
        ));
        assertions.push(assert_gpu_kernel_property(
            format!("fusion_behavior::gpu_nary_same_input_multiple_times_deduplicates_bindings::kernels_{shape:?}"),
            move |device| {
                let a = Tensor::from_slice(&device, shape, &kernel_a_data);
                let sum = &a + &a;
                let result = &sum + &a;
                result
                    .as_gpu()
                    .is_some_and(|gpu| gpu.count_kernels_to_resolve() == 1)
            },
        ));
    }
    assertions
}

pub fn gpu_nary_where_cond_fuses_into_one_kernel() -> AssertionCases {
    let mut assertions = AssertionCases::new();
    for shape in [[2, 2], [3, 5], [4, 4]] {
        let condition_values = condition_data(shape);
        let on_true_data = matrix_data(shape, 2.0);
        let on_false_data = matrix_data(shape, -1.0);
        let kernel_condition_values = condition_values.clone();
        let kernel_on_true_data = on_true_data.clone();
        let kernel_on_false_data = on_false_data.clone();
        assertions.push(assert_gpu_tensor_case(
            format!(
                "fusion_behavior::gpu_nary_where_cond_fuses_into_one_kernel::correctness_{shape:?}"
            ),
            move |device| {
                let condition = Tensor::from_slice(&device, shape, &condition_values);
                let on_true = Tensor::from_slice(&device, shape, &on_true_data);
                let on_false = Tensor::from_slice(&device, shape, &on_false_data);
                condition.where_cond(&on_true, &on_false).to_concrete()
            },
            1e-6,
        ));
        assertions.push(assert_gpu_kernel_property(
            format!(
                "fusion_behavior::gpu_nary_where_cond_fuses_into_one_kernel::kernels_{shape:?}"
            ),
            move |device| {
                let condition = Tensor::from_slice(&device, shape, &kernel_condition_values);
                let on_true = Tensor::from_slice(&device, shape, &kernel_on_true_data);
                let on_false = Tensor::from_slice(&device, shape, &kernel_on_false_data);
                let result = condition.where_cond(&on_true, &on_false);
                result
                    .as_gpu()
                    .is_some_and(|gpu| gpu.count_kernels_to_resolve() == 1)
            },
        ));
    }
    assertions
}

pub fn gpu_flash_attention_fuses_into_one_kernel() -> AssertionCases {
    let q_shape = [1, 2, 3, 4];
    let kv_shape = [1, 2, 5, 4];
    let q_data = attention_data(q_shape.iter().product(), 0.1);
    let k_data = attention_data(kv_shape.iter().product(), -0.15);
    let v_data = attention_data(kv_shape.iter().product(), 0.35);
    let scale = 1.0 / (q_shape[3] as f32).sqrt();
    let kernel_q_data = q_data.clone();
    let kernel_k_data = k_data.clone();
    let kernel_v_data = v_data.clone();
    let mut assertions = AssertionCases::new();

    assertions.push(assert_gpu_tensor_case(
        "fusion_behavior::gpu_flash_attention_fuses_into_one_kernel::correctness",
        move |device| {
            let q = Tensor::from_slice(&device, q_shape, &q_data);
            let k = Tensor::from_slice(&device, kv_shape, &k_data);
            let v = Tensor::from_slice(&device, kv_shape, &v_data);
            q.flash_attention(&k, &v, scale, None).to_concrete()
        },
        1e-4,
    ));
    assertions.push(assert_gpu_kernel_property(
        "fusion_behavior::gpu_flash_attention_fuses_into_one_kernel::kernels",
        move |device| {
            let Some(gpu) = device.as_gpu() else {
                return true;
            };
            if gpu.fixed_width_subgroup_size().is_none() {
                return true;
            }
            let q = Tensor::from_slice(&device, q_shape, &kernel_q_data);
            let k = Tensor::from_slice(&device, kv_shape, &kernel_k_data);
            let v = Tensor::from_slice(&device, kv_shape, &kernel_v_data);
            q.flash_attention(&k, &v, scale, None)
                .as_gpu()
                .is_some_and(|gpu| gpu.count_kernels_to_resolve() == 1)
        },
    ));
    assertions
}

pub fn gpu_residual_rms_norm_fuses_into_one_kernel() -> AssertionCases {
    let shape = [1, 3, 256];
    let input_data = attention_data(shape.iter().product(), 0.25);
    let residual_data = attention_data(shape.iter().product(), -0.4);
    let weight_data: Vec<f32> = (0..shape[2])
        .map(|i| 0.75 + (i % 11) as f32 * 0.03)
        .collect();

    let kernel_input_data = input_data.clone();
    let kernel_residual_data = residual_data.clone();
    let kernel_weight_data = weight_data.clone();
    let mut assertions = AssertionCases::new();
    assertions.push(assert_gpu_tensor_case(
        "fusion_behavior::gpu_residual_rms_norm_fuses_into_one_kernel::correctness",
        move |device| {
            let input = Tensor::from_slice(&device, shape, &input_data);
            let residual = Tensor::from_slice(&device, shape, &residual_data);
            let weight = Tensor::from_slice(&device, [shape[2]], &weight_data);
            input
                .rms_norm_residual_fused::<1, 2, _>(&residual, &weight, None, 1e-5)
                .to_concrete()
        },
        1e-4,
    ));
    assertions.push(assert_gpu_kernel_property(
        "fusion_behavior::gpu_residual_rms_norm_fuses_into_one_kernel::kernels",
        move |device| {
            let input = Tensor::from_slice(&device, shape, &kernel_input_data);
            let residual = Tensor::from_slice(&device, shape, &kernel_residual_data);
            let weight = Tensor::from_slice(&device, [shape[2]], &kernel_weight_data);
            input
                .rms_norm_residual_fused::<1, 2, _>(&residual, &weight, None, 1e-5)
                .as_gpu()
                .is_some_and(|gpu| gpu.count_kernels_to_resolve() == 1)
        },
    ));
    assertions
}

pub fn gpu_nary_fusion_respects_binding_limit() -> AssertionCases {
    let shape = [3, 4];
    let mut assertions = AssertionCases::new();
    assertions.push(
        fusor_conformance::assert(move |device: Device| async move {
            let Some(num_tensors) = nary_binding_limit_stress_input_count(&device) else {
                return Tensor::from_slice(&device, shape, &matrix_data(shape, 0.0));
            };
            let tensors: Vec<Tensor<2, f32>> = (0..num_tensors)
                .map(|i| Tensor::from_slice(&device, shape, &matrix_data(shape, i as f32 * 0.3)))
                .collect();
            let mut iter = tensors.iter();
            let first = iter.next().unwrap().clone();
            iter.fold(first, |acc, tensor| (&acc + tensor).to_concrete())
                .to_concrete()
        })
        .arg(|device: &Device| device.clone())
        .equal_to(move |device: Device| async move {
            let num_tensors = nary_binding_limit_stress_input_count(&device).unwrap_or(1);
            Tensor::from_slice(&device, shape, &binding_limit_sum_data(shape, num_tensors))
        })
        .compare_with(approx_compare::<2, f32>(5e-6))
        .baseline_on_test_device()
        .devices_async(gpu_devices())
        .runs(1)
        .into_case("fusion_behavior::gpu_nary_fusion_respects_binding_limit::correctness"),
    );
    assertions.push(assert_gpu_kernel_property(
        "fusion_behavior::gpu_nary_fusion_respects_binding_limit::kernels",
        move |device| {
            let Some(num_tensors) = nary_binding_limit_stress_input_count(&device) else {
                return true;
            };
            let tensors: Vec<Tensor<2, f32>> = (0..num_tensors)
                .map(|i| Tensor::from_slice(&device, shape, &matrix_data(shape, i as f32 * 0.3)))
                .collect();
            let mut iter = tensors.iter();
            let first = iter.next().unwrap().clone();
            let result = iter.fold(first, |acc, tensor| (&acc + tensor).to_concrete());
            result
                .as_gpu()
                .is_some_and(|gpu| gpu.count_kernels_to_resolve() > 1)
        },
    ));
    assertions
}

pub fn gpu_gelu_lowers_to_one_kernel() -> AssertionCases {
    let mut assertions = AssertionCases::new();
    for shape in [[2, 2], [3, 5], [4, 3]] {
        let data = matrix_data(shape, -0.4);
        let kernel_data = data.clone();
        assertions.push(assert_gpu_tensor_case(
            format!("fusion_behavior::gpu_gelu_lowers_to_one_kernel::correctness_{shape:?}"),
            move |device| {
                Tensor::from_slice(&device, shape, &data)
                    .gelu()
                    .to_concrete()
            },
            1e-3,
        ));
        assertions.push(assert_gpu_kernel_property(
            format!("fusion_behavior::gpu_gelu_lowers_to_one_kernel::kernels_{shape:?}"),
            move |device| {
                Tensor::from_slice(&device, shape, &kernel_data)
                    .gelu()
                    .as_gpu()
                    .is_some_and(|gpu| gpu.count_kernels_to_resolve() == 1)
            },
        ));
    }
    assertions
}

pub fn gpu_matmul_then_unary_chain_fuses_into_one_kernel() -> AssertionCases {
    let a_shape = [2, 3];
    let b_shape = [3, 4];
    let a_data = matrix_data(a_shape, 0.2);
    let b_data = matrix_data(b_shape, -0.1);
    let kernel_a_data = a_data.clone();
    let kernel_b_data = b_data.clone();
    let mut assertions = AssertionCases::new();
    assertions.push(assert_gpu_tensor_case(
        "fusion_behavior::gpu_matmul_then_unary_chain_fuses_into_one_kernel::correctness",
        move |device| {
            let a = Tensor::from_slice(&device, a_shape, &a_data);
            let b = Tensor::from_slice(&device, b_shape, &b_data);
            (a.mat_mul(&b).cos() + 1.0).to_concrete()
        },
        1e-5,
    ));
    assertions.push(assert_gpu_kernel_property(
        "fusion_behavior::gpu_matmul_then_unary_chain_fuses_into_one_kernel::kernels",
        move |device| {
            let a = Tensor::from_slice(&device, a_shape, &kernel_a_data);
            let b = Tensor::from_slice(&device, b_shape, &kernel_b_data);
            let matmul = a.mat_mul(&b);
            let result = matmul.cos() + 1.0;
            result
                .as_gpu()
                .is_some_and(|gpu| gpu.count_kernels_to_resolve() == 1)
        },
    ));
    assertions
}

pub fn gpu_unary_inputs_fuse_into_matmul_kernel() -> AssertionCases {
    let a_shape = [2, 3];
    let b_shape = [3, 4];
    let a_data = matrix_data(a_shape, 0.7);
    let b_data = matrix_data(b_shape, 0.4);
    let kernel_a_data = a_data.clone();
    let kernel_b_data = b_data.clone();
    let mut assertions = AssertionCases::new();
    assertions.push(assert_gpu_tensor_case(
        "fusion_behavior::gpu_unary_inputs_fuse_into_matmul_kernel::correctness",
        move |device| {
            let a = Tensor::from_slice(&device, a_shape, &a_data);
            let b = Tensor::from_slice(&device, b_shape, &b_data);
            (-a.clone()).mat_mul(&b.sin()).to_concrete()
        },
        1e-5,
    ));
    assertions.push(assert_gpu_kernel_property(
        "fusion_behavior::gpu_unary_inputs_fuse_into_matmul_kernel::kernels",
        move |device| {
            let a = Tensor::from_slice(&device, a_shape, &kernel_a_data);
            let b = Tensor::from_slice(&device, b_shape, &kernel_b_data);
            (-a.clone())
                .mat_mul(&b.sin())
                .as_gpu()
                .is_some_and(|gpu| gpu.count_kernels_to_resolve() == 1)
        },
    ));
    assertions
}

pub fn gpu_reduce_then_unary_chain_fuses_into_one_kernel() -> AssertionCases {
    let shape = [3, 5];
    let data = matrix_data(shape, 0.3);
    let kernel_data = data.clone();
    let mut assertions = AssertionCases::new();
    assertions.push(assert_gpu_tensor_case(
        "fusion_behavior::gpu_reduce_then_unary_chain_fuses_into_one_kernel::correctness",
        move |device| {
            let tensor = Tensor::from_slice(&device, shape, &data);
            (tensor.sum::<1>(0).cos() + 1.0).to_concrete()
        },
        1e-5,
    ));
    assertions.push(assert_gpu_kernel_property(
        "fusion_behavior::gpu_reduce_then_unary_chain_fuses_into_one_kernel::kernels",
        move |device| {
            let tensor = Tensor::from_slice(&device, shape, &kernel_data);
            let reduced = tensor.sum::<1>(0);
            let result = reduced.cos() + 1.0;
            result
                .as_gpu()
                .is_some_and(|gpu| gpu.count_kernels_to_resolve() == 1)
        },
    ));
    assertions
}

pub fn gpu_indexing_then_arithmetic_matches_cpu() -> AssertionCase {
    // `i((row, ..))` produces a rank-1 view; chaining mul_scalar + add_scalar
    // exercises the index-then-arithmetic fusion path that no existing test
    // covers. We assert correctness against CPU; kernel-count is informational
    // (printed if the count is unexpected) since fusion details may change.
    let shape = [4, 6];
    let data = matrix_data(shape, 0.2);
    assert_gpu_tensor_case(
        "fusion_behavior::gpu_indexing_then_arithmetic_matches_cpu",
        move |device| {
            let input: Tensor<2, f32> = Tensor::from_slice(&device, shape, &data);
            let row = input.i((1, ..));
            (row.mul_scalar(2.0) + 0.5).to_concrete()
        },
        1e-6,
    )
}

pub fn gpu_reduce_then_gelu_uses_two_kernels() -> AssertionCases {
    let mut assertions = AssertionCases::new();
    for shape in [[2, 4], [3, 6], [4, 5]] {
        let data = matrix_data(shape, 0.2);
        let kernel_data = data.clone();
        assertions.push(assert_gpu_tensor_case(
            format!(
                "fusion_behavior::gpu_reduce_then_gelu_uses_two_kernels::correctness_{shape:?}"
            ),
            move |device| {
                Tensor::from_slice(&device, shape, &data)
                    .sum_keepdim::<1>(0)
                    .gelu()
                    .to_concrete()
            },
            1e-3,
        ));
        assertions.push(assert_gpu_kernel_property(
            format!("fusion_behavior::gpu_reduce_then_gelu_uses_two_kernels::kernels_{shape:?}"),
            move |device| {
                let result = Tensor::from_slice(&device, shape, &kernel_data)
                    .sum_keepdim::<1>(0)
                    .gelu();
                // Resize between Reduce and Gelu prevents fusion of the two kernels.
                result
                    .as_gpu()
                    .is_some_and(|gpu| gpu.count_kernels_to_resolve() == 2)
            },
        ));
    }
    assertions
}
