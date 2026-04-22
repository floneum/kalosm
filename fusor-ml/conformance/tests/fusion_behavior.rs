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
