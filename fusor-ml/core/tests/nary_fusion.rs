use fusor_core::Device;
use fusor_core::Tensor;

#[tokio::test]
async fn test_nary_triple_add() {
    let device = Device::new().await.unwrap();
    let a = Tensor::new(&device, &[[1., 2.], [3., 4.]]);
    let b = Tensor::new(&device, &[[5., 6.], [7., 8.]]);
    let c = Tensor::new(&device, &[[9., 10.], [11., 12.]]);

    // x + y + z should be ONE kernel
    let result = &(&a + &b) + &c;
    assert_eq!(result.count_kernels_to_resolve(), 1);
    let output = result.as_slice().await.unwrap();

    assert_eq!(output[[0, 0]], 1. + 5. + 9.);
    assert_eq!(output[[1, 1]], 4. + 8. + 12.);
}

#[tokio::test]
async fn test_nary_mixed_ops() {
    let device = Device::new().await.unwrap();
    let a = Tensor::new(&device, &[[1., 2.], [3., 4.]]);
    let b = Tensor::new(&device, &[[5., 6.], [7., 8.]]);
    let c = Tensor::new(&device, &[[9., 10.], [11., 12.]]);

    // (a + b) * c should be ONE kernel
    let result = &(&a + &b) * &c;
    assert_eq!(result.count_kernels_to_resolve(), 1);
    let output = result.as_slice().await.unwrap();

    assert_eq!(output[[0, 0]], (1. + 5.) * 9.);
    assert_eq!(output[[1, 1]], (4. + 8.) * 12.);
}

#[tokio::test]
async fn test_nary_nested_pairwise() {
    let device = Device::new().await.unwrap();
    let a = Tensor::new(&device, &[[1., 2.], [3., 4.]]);
    let b = Tensor::new(&device, &[[5., 6.], [7., 8.]]);
    let c = Tensor::new(&device, &[[9., 10.], [11., 12.]]);
    let d = Tensor::new(&device, &[[1., 1.], [1., 1.]]);

    // (a + b) * (c - d) should be ONE kernel with 4 inputs
    let result = &(&a + &b) * &(&c - &d);
    assert_eq!(result.count_kernels_to_resolve(), 1);
    let output = result.as_slice().await.unwrap();

    assert_eq!(output[[0, 0]], (1. + 5.) * (9. - 1.));
    assert_eq!(output[[1, 1]], (4. + 8.) * (12. - 1.));
}

#[tokio::test]
async fn test_nary_unary_in_middle() {
    let device = Device::new().await.unwrap();
    let a = Tensor::new(&device, &[[1., 2.], [3., 4.]]);
    let b = Tensor::new(&device, &[[0.5, 0.5], [0.5, 0.5]]);

    // (-a + b.sin()).cos() + 1.0 should be ONE kernel
    let result = ((-a.clone()) + b.sin()).cos() + 1.0;
    assert_eq!(result.count_kernels_to_resolve(), 1);
    let output = result.as_slice().await.unwrap();

    let expected_00 = ((-1.0_f32) + 0.5_f32.sin()).cos() + 1.0;
    assert!((output[[0, 0]] - expected_00).abs() < 0.001);
}

#[tokio::test]
async fn test_nary_element_wise_chain_then_pairwise() {
    let device = Device::new().await.unwrap();
    let a = Tensor::new(&device, &[[1., 2.], [3., 4.]]);
    let b = Tensor::new(&device, &[[5., 6.], [7., 8.]]);

    // (a + 1.0).exp() + b.sin() should be ONE kernel
    let result = &(a + 1.0).exp() + &b.sin();
    assert_eq!(result.count_kernels_to_resolve(), 1);
    let output = result.as_slice().await.unwrap();

    let expected_00 = (1.0_f32 + 1.0).exp() + 5.0_f32.sin();
    assert!((output[[0, 0]] - expected_00).abs() < 0.001);
}

#[tokio::test]
async fn test_nary_same_input_multiple_times() {
    let device = Device::new().await.unwrap();
    let a = Tensor::new(&device, &[[1., 2.], [3., 4.]]);

    // a + a + a should deduplicate inputs
    let result = &(&a + &a) + &a;
    assert_eq!(result.count_kernels_to_resolve(), 1);
    let output = result.as_slice().await.unwrap();

    assert_eq!(output[[0, 0]], 3.0);
    assert_eq!(output[[1, 1]], 12.0);
}

#[tokio::test]
async fn test_nary_where_cond_basic() {
    let device = Device::new().await.unwrap();
    let condition = Tensor::new(&device, &[[0u32, 1], [1, 0]]);
    let on_true = Tensor::new(&device, &[[10., 20.], [30., 40.]]);
    let on_false = Tensor::new(&device, &[[1., 2.], [3., 4.]]);

    // Basic where_cond
    let result = condition.where_cond(&on_true, &on_false);
    assert_eq!(result.count_kernels_to_resolve(), 1);
    let output = result.as_slice().await.unwrap();

    assert_eq!(output[[0, 0]], 1.); // condition=0 -> on_false
    assert_eq!(output[[0, 1]], 20.); // condition=1 -> on_true
    assert_eq!(output[[1, 0]], 30.); // condition=1 -> on_true
    assert_eq!(output[[1, 1]], 4.); // condition=0 -> on_false
}

#[tokio::test]
async fn test_nary_fusion_respects_binding_limit() {
    let device = Device::new().await.unwrap();

    // Get the actual GPU storage buffer limit
    let max_storage_buffers = device.limits().max_storage_buffers_per_shader_stage as usize;

    // Create enough tensors to exceed the limit
    // We need max_storage_buffers + 1 unique inputs to exceed the limit
    // (since we also need 1 binding for output)
    let num_tensors = max_storage_buffers + 1;

    let tensors: Vec<_> = (0..num_tensors)
        .map(|i| {
            Tensor::new(
                &device,
                &[[i as f32, (i + 1) as f32], [(i + 2) as f32, (i + 3) as f32]],
            )
        })
        .collect();

    // Add all tensors together in a chain
    // This requires num_tensors input bindings + 1 output = num_tensors + 1 bindings
    // which exceeds the max_storage_buffers limit
    let result: Tensor<2, _> = tensors.iter().sum();

    // The number of kernels should be more than 1 due to the binding limit
    let kernel_count = result.count_kernels_to_resolve();
    assert!(
        kernel_count > 1,
        "Expected more than 1 kernel due to storage binding limit (max_storage_buffers={}), got {}",
        max_storage_buffers,
        kernel_count
    );

    // Verify the result is still correct
    let output = result.as_slice().await.unwrap();
    let expected_00: f32 = (0..num_tensors).map(|i| i as f32).sum();
    let expected_11: f32 = (0..num_tensors).map(|i| (i + 3) as f32).sum();
    assert_eq!(output[[0, 0]], expected_00);
    assert_eq!(output[[1, 1]], expected_11);
}
