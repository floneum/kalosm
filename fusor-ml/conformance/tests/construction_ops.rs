mod common;

use common::{assert_approx_tensors, assert_exact_tensors};
use fusor::{Device, Tensor, arange, arange_step};
use fusor_conformance::available_devices;

#[tokio::test]
async fn construction_aliases_match_on_all_devices() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

    for device in available_devices().await {
        let from_new: Tensor<2, f32> = Tensor::new(&device, &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let from_slice = Tensor::from_slice(&device, [2, 3], &data);
        assert_approx_tensors(from_new.clone(), from_slice, 0.0).await;

        let zeros = Tensor::<2, f32>::zeros(&device, [2, 3]);
        let zeros_like = from_new.zeros_like();
        assert_exact_tensors(zeros, zeros_like).await;

        let splat = Tensor::<2, f32>::splat(&device, 7.0, [2, 3]);
        let full = Tensor::<2, f32>::full(&device, [2, 3], 7.0);
        assert_exact_tensors(splat, full).await;

        let range = arange(&device, 0.0f32, 6.0).reshape([2, 3]).to_concrete();
        let range_step = arange_step(&device, 0.0f32, 12.0, 2.0)
            .reshape([2, 3])
            .to_concrete();
        assert_approx_tensors(range_step, range.mul_scalar(2.0), 1e-6).await;
    }
}

#[tokio::test]
async fn device_wrappers_and_variant_accessors_work() {
    let cpu: Tensor<1, f32> = Tensor::from_slice(&Device::Cpu, [3], &[1.0, 2.0, 3.0]);
    assert!(cpu.is_cpu());
    assert!(!cpu.is_gpu());
    assert!(cpu.as_cpu().is_some());
    assert!(cpu.as_gpu().is_none());
    assert!(cpu.clone().to_cpu().is_some());
    assert!(cpu.clone().to_gpu().is_none());
    assert_eq!(cpu.shape(), [3]);
    assert!(cpu.gpu_key().is_none());
    assert_eq!(cpu.rank(), 1);
    assert_eq!(cpu.to_scalar().await.unwrap(), 1.0);
    let cpu_concrete = cpu.to_concrete();
    assert!(cpu_concrete.is_cpu());
    let _ = cpu_concrete.clone().unwrap_cpu();

    if let Ok(gpu) = Device::gpu().await {
        let tensor: Tensor<1, f32> = Tensor::from_slice(&gpu, [3], &[1.0, 2.0, 3.0]);
        assert!(!tensor.is_cpu());
        assert!(tensor.is_gpu());
        assert!(tensor.as_cpu().is_none());
        assert!(tensor.as_gpu().is_some());
        assert!(tensor.clone().to_cpu().is_none());
        assert!(tensor.clone().to_gpu().is_some());
        assert_eq!(tensor.shape(), [3]);
        assert!(tensor.gpu_key().is_some());
        assert_eq!(tensor.rank(), 1);
        assert_eq!(tensor.to_scalar().await.unwrap(), 1.0);
        let concrete = tensor.to_concrete();
        assert!(concrete.is_gpu());
        let _ = concrete.clone().unwrap_gpu();
    }
}
