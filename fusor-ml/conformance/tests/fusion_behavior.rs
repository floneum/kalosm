use fusor::{Device, Tensor};

async fn gpu_device() -> Option<Device> {
    Device::gpu().await.ok()
}

#[tokio::test]
async fn gpu_nary_triple_add_fuses_into_one_kernel() {
    let Some(device) = gpu_device().await else {
        return;
    };

    let a: Tensor<2, f32> = Tensor::new(&device, &[[1.0, 2.0], [3.0, 4.0]]);
    let b: Tensor<2, f32> = Tensor::new(&device, &[[5.0, 6.0], [7.0, 8.0]]);
    let c: Tensor<2, f32> = Tensor::new(&device, &[[9.0, 10.0], [11.0, 12.0]]);

    let sum = &a + &b;
    let result = &sum + &c;
    assert_eq!(result.as_gpu().unwrap().count_kernels_to_resolve(), 1);
    let output = result.as_slice().await.unwrap();
    assert_eq!(output[[0, 0]], 15.0);
}

#[tokio::test]
async fn gpu_nary_unary_chain_fuses_into_one_kernel() {
    let Some(device) = gpu_device().await else {
        return;
    };

    let a: Tensor<2, f32> = Tensor::new(&device, &[[1.0, 2.0], [3.0, 4.0]]);
    let b: Tensor<2, f32> = Tensor::new(&device, &[[0.5, 0.5], [0.5, 0.5]]);

    let sum = (-a.clone()) + b.sin();
    let result = sum.cos() + 1.0;
    assert_eq!(result.as_gpu().unwrap().count_kernels_to_resolve(), 1);
    let output = result.as_slice().await.unwrap();
    let expected_00 = ((-1.0_f32) + 0.5_f32.sin()).cos() + 1.0;
    assert!((output[[0, 0]] - expected_00).abs() < 0.001);
}

#[tokio::test]
async fn gpu_nary_same_input_multiple_times_deduplicates_bindings() {
    let Some(device) = gpu_device().await else {
        return;
    };

    let a: Tensor<2, f32> = Tensor::new(&device, &[[1.0, 2.0], [3.0, 4.0]]);
    let sum = &a + &a;
    let result = &sum + &a;
    assert_eq!(result.as_gpu().unwrap().count_kernels_to_resolve(), 1);
    let output = result.as_slice().await.unwrap();
    assert_eq!(output[[0, 0]], 3.0);
    assert_eq!(output[[1, 1]], 12.0);
}

#[tokio::test]
async fn gpu_nary_where_cond_fuses_into_one_kernel() {
    let Some(device) = gpu_device().await else {
        return;
    };

    let condition: Tensor<2, f32> = Tensor::new(&device, &[[0.0, 1.0], [1.0, 0.0]]);
    let on_true: Tensor<2, f32> = Tensor::new(&device, &[[10.0, 20.0], [30.0, 40.0]]);
    let on_false: Tensor<2, f32> = Tensor::new(&device, &[[1.0, 2.0], [3.0, 4.0]]);

    let result = condition.where_cond(&on_true, &on_false);
    assert_eq!(result.as_gpu().unwrap().count_kernels_to_resolve(), 1);
    let output = result.as_slice().await.unwrap();
    assert_eq!(output[[0, 0]], 1.0);
    assert_eq!(output[[0, 1]], 20.0);
    assert_eq!(output[[1, 0]], 30.0);
    assert_eq!(output[[1, 1]], 4.0);
}

#[tokio::test]
async fn gpu_nary_fusion_respects_binding_limit() {
    let Some(device) = gpu_device().await else {
        return;
    };

    let max_storage_buffers = device
        .as_gpu()
        .unwrap()
        .limits()
        .max_storage_buffers_per_shader_stage as usize;
    let num_tensors = max_storage_buffers + 1;

    let tensors: Vec<Tensor<2, f32>> = (0..num_tensors)
        .map(|i| {
            Tensor::new(
                &device,
                &[[i as f32, (i + 1) as f32], [(i + 2) as f32, (i + 3) as f32]],
            )
        })
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

    let output = result.as_slice().await.unwrap();
    let expected_00: f32 = (0..num_tensors).map(|i| i as f32).sum();
    let expected_11: f32 = (0..num_tensors).map(|i| (i + 3) as f32).sum();
    assert_eq!(output[[0, 0]], expected_00);
    assert_eq!(output[[1, 1]], expected_11);
}
