mod common;

use common::flatten2;
use fusor::{Device, Tensor, ToVec1, ToVec2, arange, arange_step};
use fusor_conformance::{FuzzGenerator, approx_compare, exact_compare};
use rand::distr::Uniform;

#[tokio::test]
async fn construction_aliases_match_on_all_devices() {
    let gen_2x3 = FuzzGenerator::<2, f32>::new([2, 3])
        .with_seed(700)
        .with_distribution(Uniform::new(-8.0, 8.0).unwrap());

    fusor_conformance::assert(async |x: Tensor<2, f32>| {
        let device = x.device();
        let values = x.as_slice().await.unwrap().to_vec2();
        Tensor::new(&device, &values)
    })
    .arg(gen_2x3.clone())
    .equal_to_resolved_with_device(async |values: Vec<Vec<f32>>, device: Device| {
        Tensor::from_slice(&device, [2, 3], &flatten2(&values))
    })
    .compare_with(approx_compare::<2, f32>(0.0))
    .runs(3)
    .await
    .unwrap();

    fusor_conformance::assert(async |x: Tensor<2, f32>| {
        Tensor::<2, f32>::zeros(&x.device(), [2, 3])
    })
    .arg(gen_2x3.clone())
    .equal_to(async |x: Tensor<2, f32>| x.zeros_like())
    .compare_with(exact_compare::<2, f32>())
    .runs(3)
    .await
    .unwrap();

    fusor_conformance::assert(async |x: Tensor<2, f32>| {
        Tensor::<2, f32>::splat(&x.device(), 7.0, [2, 3])
    })
    .arg(gen_2x3)
    .equal_to(async |x: Tensor<2, f32>| Tensor::<2, f32>::full(&x.device(), [2, 3], 7.0))
    .compare_with(exact_compare::<2, f32>())
    .runs(3)
    .await
    .unwrap();

    fusor_conformance::assert(async |device: Device| {
        arange_step(&device, 0.0f32, 12.0, 2.0)
            .reshape([2, 3])
            .to_concrete()
    })
    .arg(|device: &Device| device.clone())
    .equal_to(async |device: Device| arange(&device, 0.0f32, 6.0).reshape([2, 3]).mul_scalar(2.0))
    .compare_with(approx_compare::<2, f32>(1e-6))
    .await
    .unwrap();
}

#[tokio::test]
async fn device_wrappers_and_variant_accessors_work() {
    let fuzz = FuzzGenerator::<1, f32>::new([3])
        .with_seed(710)
        .with_distribution(Uniform::new(-4.0, 4.0).unwrap());

    fusor_conformance::assert(async |x: Tensor<1, f32>| {
        let device = x.device();
        assert_eq!(x.is_cpu(), device.is_cpu());
        assert_eq!(x.is_gpu(), device.is_gpu());
        assert_eq!(x.as_cpu().is_some(), device.is_cpu());
        assert_eq!(x.as_gpu().is_some(), device.is_gpu());
        assert_eq!(x.clone().to_cpu().is_some(), device.is_cpu());
        assert_eq!(x.clone().to_gpu().is_some(), device.is_gpu());
        assert_eq!(x.shape(), [3]);
        assert_eq!(x.rank(), 1);
        assert_eq!(x.gpu_key().is_some(), device.is_gpu());

        let values = x.as_slice().await.unwrap().to_vec1();
        assert_eq!(x.to_scalar().await.unwrap(), values[0]);

        let concrete = x.clone().to_concrete();
        assert_eq!(concrete.is_cpu(), device.is_cpu());
        assert_eq!(concrete.is_gpu(), device.is_gpu());
        if device.is_cpu() {
            let _ = concrete.clone().unwrap_cpu();
        } else {
            let _ = concrete.clone().unwrap_gpu();
        }

        x
    })
    .arg(fuzz)
    .equal_to(async |x: Tensor<1, f32>| x)
    .compare_with(approx_compare::<1, f32>(0.0))
    .runs(3)
    .await
    .unwrap();
}
