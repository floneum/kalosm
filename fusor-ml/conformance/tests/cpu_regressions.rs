use fusor::{Device, Tensor as FusorTensor};
use fusor_cpu::{Tensor as CpuTensor, TensorBacking};

#[test]
fn cpu_cast_regressions_match_expected() {
    let a = CpuTensor::from_slice([4], &[1.5f32, 2.7, -3.2, 4.9]);
    let b = a.cast::<i32>();
    assert_eq!(b.get([0]), 1);
    assert_eq!(b.get([1]), 2);
    assert_eq!(b.get([2]), -3);
    assert_eq!(b.get([3]), 4);

    let a = CpuTensor::from_slice([3], &[1i32, -2, 3]);
    let b = a.cast::<f64>();
    assert_eq!(b.get([0]), 1.0);
    assert_eq!(b.get([1]), -2.0);
    assert_eq!(b.get([2]), 3.0);

    let a = CpuTensor::from_slice([3], &[1.5f64, 2.5, 3.5]);
    let b = a.cast::<f32>();
    assert!((b.get([0]) - 1.5).abs() < 1e-6);
    assert!((b.get([1]) - 2.5).abs() < 1e-6);
    assert!((b.get([2]) - 3.5).abs() < 1e-6);

    let a = CpuTensor::from_slice([3], &[100i32, -200, 300]);
    let b = a.cast::<i64>();
    assert_eq!(b.get([0]), 100);
    assert_eq!(b.get([1]), -200);
    assert_eq!(b.get([2]), 300);

    let a = CpuTensor::from_slice([4], &[0u8, 127, 200, 255]);
    let b = a.cast::<f32>();
    assert_eq!(b.get([0]), 0.0);
    assert_eq!(b.get([1]), 127.0);
    assert_eq!(b.get([2]), 200.0);
    assert_eq!(b.get([3]), 255.0);

    let a = CpuTensor::from_slice([2, 2], &[1.1f32, 2.2, 3.3, 4.4]);
    let b = a.cast::<i32>();
    assert_eq!(b.get([0, 0]), 1);
    assert_eq!(b.get([0, 1]), 2);
    assert_eq!(b.get([1, 0]), 3);
    assert_eq!(b.get([1, 1]), 4);

    let a = CpuTensor::from_slice([3], &[1.0f32, 2.0, 3.0]);
    let b = a.cast::<f32>();
    assert_eq!(b.get([0]), 1.0);
    assert_eq!(b.get([1]), 2.0);
    assert_eq!(b.get([2]), 3.0);

    let size = 1024;
    let data: Vec<f32> = (0..size).map(|i| i as f32 + 0.5).collect();
    let a = CpuTensor::from_slice([size], &data);
    let b = a.cast::<i32>();
    for i in 0..size {
        assert_eq!(b.get([i]), i as i32);
    }
}

#[test]
fn cpu_elementwise_and_pairwise_regressions_match_expected() {
    let abs = CpuTensor::from_slice([4], &[1i32, -2, 3, -4])
        .abs()
        .to_concrete();
    assert_eq!(abs.get([0]), 1);
    assert_eq!(abs.get([1]), 2);
    assert_eq!(abs.get([2]), 3);
    assert_eq!(abs.get([3]), 4);

    let sqrt = CpuTensor::from_slice([4], &[1.0f64, 4.0, 9.0, 16.0])
        .sqrt()
        .to_concrete();
    assert_eq!(sqrt.get([0]), 1.0);
    assert_eq!(sqrt.get([1]), 2.0);
    assert_eq!(sqrt.get([2]), 3.0);
    assert_eq!(sqrt.get([3]), 4.0);

    let pow = CpuTensor::from_slice([3], &[2.0f64, 3.0, 4.0]).pow_scalar(3.0);
    assert_eq!(pow.get([0]), 8.0);
    assert_eq!(pow.get([1]), 27.0);
    assert_eq!(pow.get([2]), 64.0);

    let lhs = CpuTensor::from_slice([4], &[1i32, 2, 3, 4]);
    let rhs = CpuTensor::from_slice([4], &[10i32, 20, 30, 40]);
    let add = (&lhs + &rhs).to_concrete();
    assert_eq!(add.layout().shape(), &[4]);
    assert_eq!(add.get([0]), 11);
    assert_eq!(add.get([1]), 22);
    assert_eq!(add.get([2]), 33);
    assert_eq!(add.get([3]), 44);

    let lhs = CpuTensor::from_slice([2, 2, 2], &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let rhs = CpuTensor::from_slice([2, 2, 2], &[0.5f64, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
    let add = (&lhs + &rhs).to_concrete();
    assert_eq!(add.layout().shape(), &[2, 2, 2]);
    assert_eq!(add.get([0, 0, 0]), 1.5);
    assert_eq!(add.get([0, 0, 1]), 3.0);
    assert_eq!(add.get([1, 1, 1]), 12.0);

    let lhs = CpuTensor::from_slice([4], &[1i32, 2, 3, 4]);
    let rhs = CpuTensor::from_slice([4], &[10i32, 20, 30, 40]);
    let mul = (&lhs * &rhs).to_concrete();
    assert_eq!(mul.get([0]), 10);
    assert_eq!(mul.get([1]), 40);
    assert_eq!(mul.get([2]), 90);
    assert_eq!(mul.get([3]), 160);

    let size = 1024;
    let lhs_data: Vec<f64> = (0..size).map(|i| (i * 4) as f64).collect();
    let rhs_data: Vec<f64> = (0..size).map(|_| 2.0).collect();
    let lhs = CpuTensor::from_slice([size], &lhs_data);
    let rhs = CpuTensor::from_slice([size], &rhs_data);
    let div = (&lhs / &rhs).to_concrete();
    for i in 0..size {
        assert_eq!(div.get([i]), (i * 2) as f64);
    }
}

#[tokio::test]
async fn fusor_ne_tensor_regression_matches_expected() {
    let a: FusorTensor<1, f32> = FusorTensor::from_slice(&Device::Cpu, [4], &[1.0, 2.0, 3.0, 4.0]);
    let b: FusorTensor<1, f32> = FusorTensor::from_slice(&Device::Cpu, [4], &[1.0, 3.0, 3.0, 5.0]);

    let result = a.ne_tensor(&b);
    let slice = result.as_slice().await.unwrap();

    assert_eq!(slice[[0]], 0.0);
    assert_eq!(slice[[1]], 1.0);
    assert_eq!(slice[[2]], 0.0);
    assert_eq!(slice[[3]], 1.0);
}

#[test]
fn cpu_integer_comparison_and_conditional_regressions_match_expected() {
    let a = CpuTensor::from_slice([4], &[1i32, 2, 3, 4]);
    let b = CpuTensor::from_slice([4], &[2i32, 2, 2, 2]);

    let lt_result = a.clone().lt(b).to_concrete();
    assert_eq!(lt_result.get([0]), 1);
    assert_eq!(lt_result.get([1]), 0);
    assert_eq!(lt_result.get([2]), 0);
    assert_eq!(lt_result.get([3]), 0);

    let eq_result = a.eq_scalar(2).to_concrete();
    assert_eq!(eq_result.get([0]), 0);
    assert_eq!(eq_result.get([1]), 1);
    assert_eq!(eq_result.get([2]), 0);
    assert_eq!(eq_result.get([3]), 0);

    let cond = CpuTensor::from_slice([4], &[1i32, 0, -1, 0]);
    let on_true = CpuTensor::from_slice([4], &[10i32, 20, 30, 40]);
    let on_false = CpuTensor::from_slice([4], &[100i32, 200, 300, 400]);
    let result = cond.where_cond(on_true, on_false);

    assert_eq!(result.get([0]), 10);
    assert_eq!(result.get([1]), 200);
    assert_eq!(result.get([2]), 30);
    assert_eq!(result.get([3]), 400);
}

#[test]
fn cpu_reduction_regressions_match_expected() {
    let tensor = CpuTensor::from_slice([5], &[1.0f32, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(tensor.sum(), 15.0);

    let tensor = CpuTensor::from_slice([2, 3], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(tensor.sum(), 21.0);

    let size = 10000;
    let data: Vec<f32> = (1..=size).map(|i| i as f32).collect();
    let tensor = CpuTensor::from_slice([size], &data);
    let expected = (size * (size + 1) / 2) as f32;
    assert!((tensor.sum() - expected).abs() < 1.0);

    let tensor = CpuTensor::from_slice([8], &[3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
    assert_eq!(tensor.max(), 9.0);

    let tensor = CpuTensor::from_slice([5], &[-3.0f32, -1.0, -4.0, -1.0, -5.0]);
    assert_eq!(tensor.max(), -1.0);

    let tensor = CpuTensor::from_slice([8], &[3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
    assert_eq!(tensor.min(), 1.0);

    let tensor = CpuTensor::from_slice([5], &[-3.0f32, -1.0, -4.0, -1.0, -5.0]);
    assert_eq!(tensor.min(), -5.0);

    let tensor = CpuTensor::from_slice([5], &[1.0f32, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(tensor.prod(), 120.0);

    let tensor = CpuTensor::from_slice([2, 2], &[2.0f32, 3.0, 4.0, 5.0]);
    assert_eq!(tensor.prod(), 120.0);

    let tensor = CpuTensor::from_slice([5], &[1i32, 2, 3, 4, 5]);
    assert_eq!(tensor.clone().sum(), 15);
    assert_eq!(tensor.clone().max(), 5);
    assert_eq!(tensor.min(), 1);

    let tensor = CpuTensor::from_slice([4], &[1.0f64, 2.0, 3.0, 4.0]);
    assert_eq!(tensor.clone().sum(), 10.0);
    assert_eq!(tensor.clone().max(), 4.0);
    assert_eq!(tensor.clone().min(), 1.0);
    assert_eq!(tensor.prod(), 24.0);

    let tensor = CpuTensor::from_slice([1], &[42.0f32]);
    assert_eq!(tensor.clone().sum(), 42.0);
    assert_eq!(tensor.clone().max(), 42.0);
    assert_eq!(tensor.clone().min(), 42.0);
    assert_eq!(tensor.prod(), 42.0);

    let tensor = CpuTensor::from_slice([2, 2, 2], &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let result = tensor.sum_axis::<2>(0);
    assert_eq!(result.layout().shape(), &[2, 2]);
    assert_eq!(result.get([0, 0]), 6.0);
    assert_eq!(result.get([0, 1]), 8.0);
    assert_eq!(result.get([1, 0]), 10.0);
    assert_eq!(result.get([1, 1]), 12.0);
}

#[test]
fn cpu_index_and_matmul_regressions_match_expected() {
    let input = CpuTensor::from_slice([4], &[100i32, 200, 300, 400]);
    let indices = CpuTensor::from_slice([2], &[3u32, 1]);
    let result = input.index_select(0, indices);
    assert_eq!(result.get([0]), 400);
    assert_eq!(result.get([1]), 200);

    let lhs = CpuTensor::from_slice([2, 2], &[1.0f64, 2.0, 3.0, 4.0]);
    let rhs = CpuTensor::from_slice([2, 2], &[5.0f64, 6.0, 7.0, 8.0]);
    let result = lhs.matmul(rhs);
    assert_eq!(result.layout().shape(), &[2, 2]);
    assert_eq!(result.get([0, 0]), 19.0);
    assert_eq!(result.get([0, 1]), 22.0);
    assert_eq!(result.get([1, 0]), 43.0);
    assert_eq!(result.get([1, 1]), 50.0);
}
