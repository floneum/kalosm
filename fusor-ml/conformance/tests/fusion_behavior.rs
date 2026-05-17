use fusor::{Device, Tensor};
use fusor_conformance::approx_eq;

async fn gpu_device() -> Option<Device> {
    Device::gpu().await.ok()
}

fn matrix_data(shape: [usize; 2], offset: f32) -> Vec<f32> {
    let total = shape[0] * shape[1];
    (0..total)
        .map(|i| (((i % 13) as f32) - 6.0) * 0.2 + offset)
        .collect()
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

#[tokio::test]
async fn gpu_nary_triple_add_fuses_into_one_kernel() {
    let Some(device) = gpu_device().await else {
        return;
    };

    for shape in [[2, 2], [3, 5], [4, 3]] {
        let a_data = matrix_data(shape, -0.3);
        let b_data = matrix_data(shape, 0.8);
        let c_data = matrix_data(shape, 1.7);
        let a = Tensor::from_slice(&device, shape, &a_data);
        let b = Tensor::from_slice(&device, shape, &b_data);
        let c = Tensor::from_slice(&device, shape, &c_data);

        let sum = &a + &b;
        let result = &sum + &c;
        assert_eq!(result.as_gpu().unwrap().count_kernels_to_resolve(), 1);
        let actual = result.to_concrete();

        let cpu_a = Tensor::from_slice(&Device::Cpu, shape, &a_data);
        let cpu_b = Tensor::from_slice(&Device::Cpu, shape, &b_data);
        let cpu_c = Tensor::from_slice(&Device::Cpu, shape, &c_data);
        let cpu_sum = &cpu_a + &cpu_b;
        let expected = (&cpu_sum + &cpu_c).to_concrete();
        approx_eq(&actual, &expected, 1e-6).await.unwrap();
    }
}

#[tokio::test]
async fn gpu_nary_unary_chain_fuses_into_one_kernel() {
    let Some(device) = gpu_device().await else {
        return;
    };

    for shape in [[2, 2], [3, 4], [2, 7]] {
        let a_data = matrix_data(shape, 0.1);
        let b_data = matrix_data(shape, -0.4);
        let a = Tensor::from_slice(&device, shape, &a_data);
        let b = Tensor::from_slice(&device, shape, &b_data);

        let sum = (-a.clone()) + b.sin();
        let result = sum.cos() + 1.0;
        assert_eq!(result.as_gpu().unwrap().count_kernels_to_resolve(), 1);
        let actual = result.to_concrete();

        let cpu_a = Tensor::from_slice(&Device::Cpu, shape, &a_data);
        let cpu_b = Tensor::from_slice(&Device::Cpu, shape, &b_data);
        let cpu_sum = (-cpu_a.clone()) + cpu_b.sin();
        let expected = (cpu_sum.cos() + 1.0).to_concrete();
        approx_eq(&actual, &expected, 1e-6).await.unwrap();
    }
}

#[tokio::test]
async fn gpu_nary_same_input_multiple_times_deduplicates_bindings() {
    let Some(device) = gpu_device().await else {
        return;
    };

    for shape in [[2, 2], [4, 3], [3, 6]] {
        let a_data = matrix_data(shape, 0.6);
        let a = Tensor::from_slice(&device, shape, &a_data);
        let sum = &a + &a;
        let result = &sum + &a;
        assert_eq!(result.as_gpu().unwrap().count_kernels_to_resolve(), 1);
        let actual = result.to_concrete();

        let cpu_a = Tensor::from_slice(&Device::Cpu, shape, &a_data);
        let cpu_sum = &cpu_a + &cpu_a;
        let expected = (&cpu_sum + &cpu_a).to_concrete();
        approx_eq(&actual, &expected, 1e-6).await.unwrap();
    }
}

#[tokio::test]
async fn gpu_nary_where_cond_fuses_into_one_kernel() {
    let Some(device) = gpu_device().await else {
        return;
    };

    for shape in [[2, 2], [3, 5], [4, 4]] {
        let condition_values = condition_data(shape);
        let on_true_data = matrix_data(shape, 2.0);
        let on_false_data = matrix_data(shape, -1.0);
        let condition = Tensor::from_slice(&device, shape, &condition_values);
        let on_true = Tensor::from_slice(&device, shape, &on_true_data);
        let on_false = Tensor::from_slice(&device, shape, &on_false_data);

        let result = condition.where_cond(&on_true, &on_false);
        assert_eq!(result.as_gpu().unwrap().count_kernels_to_resolve(), 1);
        let actual = result.to_concrete();

        let cpu_condition = Tensor::from_slice(&Device::Cpu, shape, &condition_values);
        let cpu_on_true = Tensor::from_slice(&Device::Cpu, shape, &on_true_data);
        let cpu_on_false = Tensor::from_slice(&Device::Cpu, shape, &on_false_data);
        let expected = cpu_condition
            .where_cond(&cpu_on_true, &cpu_on_false)
            .to_concrete();
        approx_eq(&actual, &expected, 1e-6).await.unwrap();
    }
}

#[tokio::test]
async fn gpu_flash_attention_fuses_into_one_kernel() {
    let Some(device) = gpu_device().await else {
        return;
    };
    let Device::Gpu(gpu) = &device else { return };
    // The streaming flash kernel is monomorphized per hardware subgroup
    // width and uses `subgroup_reduce_*`, so it can only target devices
    // that pin a single supported subgroup size. Devices that report a
    // variable range (Mesa lavapipe in Linux CI) take the composite path
    // and won't fuse — skip the fusion assertion there.
    if !gpu.subgroups_supported()
        || gpu.min_subgroup_size() != gpu.max_subgroup_size()
        || !matches!(gpu.min_subgroup_size(), 4 | 8 | 16 | 32 | 64)
    {
        return;
    }

    let q_shape = [1, 2, 3, 4];
    let kv_shape = [1, 2, 5, 4];
    let q_data = attention_data(q_shape.iter().product(), 0.1);
    let k_data = attention_data(kv_shape.iter().product(), -0.15);
    let v_data = attention_data(kv_shape.iter().product(), 0.35);
    let scale = 1.0 / (q_shape[3] as f32).sqrt();

    let q = Tensor::from_slice(&device, q_shape, &q_data);
    let k = Tensor::from_slice(&device, kv_shape, &k_data);
    let v = Tensor::from_slice(&device, kv_shape, &v_data);
    let result = q.flash_attention(&k, &v, scale, None);

    let gpu_result = result.as_gpu().unwrap();
    let kernel_count = gpu_result.count_kernels_to_resolve();
    assert_eq!(
        kernel_count,
        1,
        "flash attention graph was not fused:\n{}",
        gpu_result.graphvis()
    );
    let actual = result.to_concrete();

    let cpu_q = Tensor::from_slice(&Device::Cpu, q_shape, &q_data);
    let cpu_k = Tensor::from_slice(&Device::Cpu, kv_shape, &k_data);
    let cpu_v = Tensor::from_slice(&Device::Cpu, kv_shape, &v_data);
    let expected = cpu_q
        .flash_attention(&cpu_k, &cpu_v, scale, None)
        .to_concrete();
    approx_eq(&actual, &expected, 1e-4).await.unwrap();
}

#[tokio::test]
async fn gpu_residual_rms_norm_fuses_into_one_kernel() {
    let Some(device) = gpu_device().await else {
        return;
    };

    let shape = [1, 3, 256];
    let input_data = attention_data(shape.iter().product(), 0.25);
    let residual_data = attention_data(shape.iter().product(), -0.4);
    let weight_data: Vec<f32> = (0..shape[2])
        .map(|i| 0.75 + (i % 11) as f32 * 0.03)
        .collect();

    let input = Tensor::from_slice(&device, shape, &input_data);
    let residual = Tensor::from_slice(&device, shape, &residual_data);
    let weight = Tensor::from_slice(&device, [shape[2]], &weight_data);
    let result = input.rms_norm_residual_fused::<1, 2, _>(&residual, &weight, None, 1e-5);

    let gpu_result = result.as_gpu().unwrap();
    let kernel_count = gpu_result.count_kernels_to_resolve();
    assert_eq!(
        kernel_count,
        1,
        "residual rms norm graph was not fused:\n{}",
        gpu_result.graphvis()
    );
    let actual = result.to_concrete();

    let cpu_input = Tensor::from_slice(&Device::Cpu, shape, &input_data);
    let cpu_residual = Tensor::from_slice(&Device::Cpu, shape, &residual_data);
    let cpu_weight = Tensor::from_slice(&Device::Cpu, [shape[2]], &weight_data);
    let expected = (cpu_input + cpu_residual)
        .rms_norm_fused::<1, 2>(&cpu_weight, None, 1e-5)
        .to_concrete();
    approx_eq(&actual, &expected, 1e-4).await.unwrap();
}

#[tokio::test]
async fn gpu_nary_fusion_respects_binding_limit() {
    let Some(device) = gpu_device().await else {
        return;
    };

    let shape = [3, 4];
    let max_storage_buffers = device
        .as_gpu()
        .unwrap()
        .limits()
        .max_storage_buffers_per_shader_stage as usize;
    if max_storage_buffers > 256 {
        // DX12/WARP reports a very high storage-buffer limit. Building a
        // limit-plus-one expression tree there is not a useful conformance
        // case and can overflow the test thread stack before fusion runs.
        return;
    }
    let num_tensors = max_storage_buffers + 1;

    let tensors: Vec<Tensor<2, f32>> = (0..num_tensors)
        .map(|i| Tensor::from_slice(&device, shape, &matrix_data(shape, i as f32 * 0.3)))
        .collect();

    let mut iter = tensors.iter();
    let first = iter.next().unwrap().clone();
    let result = iter.fold(first, |acc, tensor| (&acc + tensor).to_concrete());

    let kernel_count = result.as_gpu().unwrap().count_kernels_to_resolve();
    assert!(
        kernel_count > 1,
        "expected more than one kernel when exceeding the storage binding limit, got {}",
        kernel_count
    );

    let cpu_tensors: Vec<Tensor<2, f32>> = (0..num_tensors)
        .map(|i| Tensor::from_slice(&Device::Cpu, shape, &matrix_data(shape, i as f32 * 0.3)))
        .collect();
    let mut cpu_iter = cpu_tensors.iter();
    let cpu_first = cpu_iter.next().unwrap().clone();
    let expected = cpu_iter
        .fold(cpu_first, |acc, tensor| (&acc + tensor).to_concrete())
        .to_concrete();
    approx_eq(&result.to_concrete(), &expected, 1e-6)
        .await
        .unwrap();
}

#[tokio::test]
async fn gpu_gelu_lowers_to_one_kernel() {
    let Some(device) = gpu_device().await else {
        return;
    };

    for shape in [[2, 2], [3, 5], [4, 3]] {
        let data = matrix_data(shape, -0.4);
        let tensor = Tensor::from_slice(&device, shape, &data);
        let result = tensor.gelu();
        assert_eq!(result.as_gpu().unwrap().count_kernels_to_resolve(), 1);

        let cpu = Tensor::from_slice(&Device::Cpu, shape, &data);
        let expected = cpu.gelu().to_concrete();
        approx_eq(&result.to_concrete(), &expected, 1e-3)
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn gpu_matmul_then_unary_chain_fuses_into_one_kernel() {
    let Some(device) = gpu_device().await else {
        return;
    };

    let a_shape = [2, 3];
    let b_shape = [3, 4];
    let a_data = matrix_data(a_shape, 0.2);
    let b_data = matrix_data(b_shape, -0.1);
    let a = Tensor::from_slice(&device, a_shape, &a_data);
    let b = Tensor::from_slice(&device, b_shape, &b_data);

    let matmul = a.mat_mul(&b);
    let result = matmul.cos() + 1.0;
    assert_eq!(result.as_gpu().unwrap().count_kernels_to_resolve(), 1);

    let cpu_a = Tensor::from_slice(&Device::Cpu, a_shape, &a_data);
    let cpu_b = Tensor::from_slice(&Device::Cpu, b_shape, &b_data);
    let expected = (cpu_a.mat_mul(&cpu_b).cos() + 1.0).to_concrete();
    approx_eq(&result.to_concrete(), &expected, 1e-5)
        .await
        .unwrap();
}

#[tokio::test]
async fn gpu_unary_inputs_fuse_into_matmul_kernel() {
    let Some(device) = gpu_device().await else {
        return;
    };

    let a_shape = [2, 3];
    let b_shape = [3, 4];
    let a_data = matrix_data(a_shape, 0.7);
    let b_data = matrix_data(b_shape, 0.4);
    let a = Tensor::from_slice(&device, a_shape, &a_data);
    let b = Tensor::from_slice(&device, b_shape, &b_data);

    let result = (-a.clone()).mat_mul(&b.sin());
    assert_eq!(result.as_gpu().unwrap().count_kernels_to_resolve(), 1);

    let cpu_a = Tensor::from_slice(&Device::Cpu, a_shape, &a_data);
    let cpu_b = Tensor::from_slice(&Device::Cpu, b_shape, &b_data);
    let expected = (-cpu_a.clone()).mat_mul(&cpu_b.sin()).to_concrete();
    approx_eq(&result.to_concrete(), &expected, 1e-5)
        .await
        .unwrap();
}

#[tokio::test]
async fn gpu_reduce_then_unary_chain_fuses_into_one_kernel() {
    let Some(device) = gpu_device().await else {
        return;
    };

    let shape = [3, 5];
    let data = matrix_data(shape, 0.3);
    let tensor = Tensor::from_slice(&device, shape, &data);
    let reduced = tensor.sum::<1>(0);
    let result = reduced.cos() + 1.0;
    assert_eq!(result.as_gpu().unwrap().count_kernels_to_resolve(), 1);

    let cpu = Tensor::from_slice(&Device::Cpu, shape, &data);
    let expected = (cpu.sum::<1>(0).cos() + 1.0).to_concrete();
    approx_eq(&result.to_concrete(), &expected, 1e-5)
        .await
        .unwrap();
}

#[tokio::test]
async fn gpu_indexing_then_arithmetic_matches_cpu() {
    // `i((row, ..))` produces a rank-1 view; chaining mul_scalar + add_scalar
    // exercises the index-then-arithmetic fusion path that no existing test
    // covers. We assert correctness against CPU; kernel-count is informational
    // (printed if the count is unexpected) since fusion details may change.
    let Some(device) = gpu_device().await else {
        return;
    };

    let shape = [4, 6];
    let data = matrix_data(shape, 0.2);
    let gpu_input: fusor::Tensor<2, f32> = Tensor::from_slice(&device, shape, &data);
    let row = gpu_input.i((1, ..));
    let result = row.mul_scalar(2.0) + 0.5;
    let actual = result.to_concrete();

    let cpu_input: fusor::Tensor<2, f32> = Tensor::from_slice(&Device::Cpu, shape, &data);
    let cpu_row = cpu_input.i((1, ..));
    let expected = (cpu_row.mul_scalar(2.0) + 0.5).to_concrete();
    approx_eq(&actual, &expected, 1e-6).await.unwrap();
}

#[tokio::test]
async fn gpu_reduce_then_gelu_uses_two_kernels() {
    let Some(device) = gpu_device().await else {
        return;
    };

    for shape in [[2, 4], [3, 6], [4, 5]] {
        let data = matrix_data(shape, 0.2);
        let tensor = Tensor::from_slice(&device, shape, &data);
        let reduced = tensor.sum_keepdim::<1>(0);
        let result = reduced.gelu();
        // Resize between Reduce and Gelu prevents fusion of the two kernels.
        assert_eq!(result.as_gpu().unwrap().count_kernels_to_resolve(), 2);

        let cpu = Tensor::from_slice(&Device::Cpu, shape, &data);
        let expected = cpu.sum_keepdim::<1>(0).gelu().to_concrete();
        approx_eq(&result.to_concrete(), &expected, 1e-3)
            .await
            .unwrap();
    }
}
