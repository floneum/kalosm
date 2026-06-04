//! Smoke tests for tensor construction aliases and device/variant accessors.

use fusor::{Device, Tensor, arange, arange_step};

use crate::{CaseResult, available_devices, ensure, ensure_eq, exact_eq};

pub async fn construction_aliases_match_on_varied_shapes() -> CaseResult {
    for device in available_devices().await {
        let vector = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let from_new_1d: Tensor<1, f32> = Tensor::new(&device, &vector);
        let from_slice_1d = Tensor::from_slice(&device, [vector.len()], &vector);
        exact_eq(&from_new_1d, &from_slice_1d).await?;

        let matrix = [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let from_new_2d: Tensor<2, f32> = Tensor::new(&device, &matrix);
        let from_slice_2d = Tensor::from_slice(&device, [3, 3], &matrix.concat());
        exact_eq(&from_new_2d, &from_slice_2d).await?;
        exact_eq(
            &Tensor::<2, f32>::zeros(&device, [3, 3]),
            &from_new_2d.zeros_like(),
        )
        .await?;
        exact_eq(
            &Tensor::<2, f32>::full(&device, [3, 3], 7.0),
            &Tensor::<2, f32>::splat(&device, 7.0, [3, 3]),
        )
        .await?;

        let cube = [
            [[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],
        ];
        let from_new_3d: Tensor<3, f32> = Tensor::new(&device, &cube);
        let from_slice_3d = Tensor::from_slice(
            &device,
            [2, 3, 2],
            &cube.concat().into_iter().flatten().collect::<Vec<_>>(),
        );
        exact_eq(&from_new_3d, &from_slice_3d).await?;
        exact_eq(
            &Tensor::<3, f32>::zeros(&device, [2, 3, 2]),
            &from_new_3d.zeros_like(),
        )
        .await?;
        exact_eq(
            &Tensor::<3, f32>::full(&device, [2, 3, 2], -2.5),
            &Tensor::<3, f32>::splat(&device, -2.5, [2, 3, 2]),
        )
        .await?;

        for &shape in &[[2, 3], [3, 4], [4, 2]] {
            let total = shape.iter().product::<usize>() as f32;
            let range = arange(&device, 0.0f32, total).reshape(shape).to_concrete();
            let range_step = arange_step(&device, 0.0f32, total * 2.0, 2.0)
                .reshape(shape)
                .to_concrete();
            let expected = range.mul_scalar(2.0).to_concrete();
            exact_eq(&range_step, &expected).await?;
        }
    }
    Ok(())
}

pub async fn device_wrappers_and_variant_accessors_work() -> CaseResult {
    let cpu: Tensor<1, f32> = Tensor::from_slice(&Device::Cpu, [5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
    ensure!(cpu.is_cpu());
    ensure!(!cpu.is_gpu());
    ensure!(cpu.as_cpu().is_some());
    ensure!(cpu.as_gpu().is_none());
    ensure!(cpu.clone().to_cpu().is_some());
    ensure!(cpu.clone().to_gpu().is_none());
    ensure_eq!(cpu.shape(), [5]);
    ensure!(cpu.gpu_key().is_none());
    ensure_eq!(cpu.rank(), 1);
    ensure_eq!(cpu.to_scalar().await.unwrap(), 1.0);
    let cpu_concrete = cpu.to_concrete();
    ensure!(cpu_concrete.is_cpu());
    let _ = cpu_concrete.clone().unwrap_cpu();

    if let Ok(gpu) = Device::gpu().await {
        let tensor: Tensor<1, f32> = Tensor::from_slice(&gpu, [5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        ensure!(!tensor.is_cpu());
        ensure!(tensor.is_gpu());
        ensure!(tensor.as_cpu().is_none());
        ensure!(tensor.as_gpu().is_some());
        ensure!(tensor.clone().to_cpu().is_none());
        ensure!(tensor.clone().to_gpu().is_some());
        ensure_eq!(tensor.shape(), [5]);
        ensure!(tensor.gpu_key().is_some());
        ensure_eq!(tensor.rank(), 1);
        ensure_eq!(tensor.to_scalar().await.unwrap(), 1.0);
        let concrete = tensor.to_concrete();
        ensure!(concrete.is_gpu());
        let _ = concrete.clone().unwrap_gpu();
    }
    Ok(())
}
