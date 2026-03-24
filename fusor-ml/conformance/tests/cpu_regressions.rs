use std::{io::Error, pin::Pin, sync::Arc};

use fusor::{Device, Tensor as FusorTensor};
use fusor::{ToVec1, ToVec2, ToVec3};
use fusor_conformance::{FuzzGenerator, GenerateFromDevice, approx_compare};
use fusor_cpu::Tensor as CpuTensor;
use rand::{
    SeedableRng,
    distr::{Distribution, Uniform},
    rngs::StdRng,
};

#[derive(Clone)]
struct HostVecGenerator<T> {
    seed: u64,
    len: usize,
    distribution: Arc<dyn Fn(&mut StdRng) -> T + Send + Sync>,
}

impl<T> HostVecGenerator<T> {
    fn new(
        len: usize,
        seed: u64,
        distribution: impl Distribution<T> + Send + Sync + 'static,
    ) -> Self {
        Self {
            seed,
            len,
            distribution: Arc::new(move |rng| distribution.sample(rng)),
        }
    }

    fn seed_for_run(&self, run: usize) -> u64 {
        self.seed
            ^ (run as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (self.len as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
    }
}

impl<T: Copy + Send + 'static> GenerateFromDevice for HostVecGenerator<T> {
    type Output = Vec<T>;

    fn generate(&mut self, _device: &Device, run: usize) -> Self::Output {
        let mut rng = StdRng::seed_from_u64(self.seed_for_run(run));
        (0..self.len)
            .map(|_| (self.distribution)(&mut rng))
            .collect()
    }
}

fn flatten2<T>(values: Vec<Vec<T>>) -> Vec<T> {
    values.into_iter().flatten().collect()
}

fn flatten3<T>(values: Vec<Vec<Vec<T>>>) -> Vec<T> {
    values.into_iter().flatten().flatten().collect()
}

fn exact_vec_compare<T>() -> impl for<'a> Fn(
    &'a Vec<T>,
    &'a Vec<T>,
) -> Pin<
    Box<dyn std::future::Future<Output = Result<(), Error>> + 'a>,
> + Clone
where
    T: PartialEq + std::fmt::Debug + Send + Sync + 'static,
{
    |expected, actual| {
        Box::pin(async move {
            if expected == actual {
                Ok(())
            } else {
                Err(Error::other(format!(
                    "expected {expected:?}, got {actual:?}"
                )))
            }
        })
    }
}

fn approx_vec_compare_f32(
    tol: f32,
) -> impl for<'a> Fn(
    &'a Vec<f32>,
    &'a Vec<f32>,
) -> Pin<Box<dyn std::future::Future<Output = Result<(), Error>> + 'a>>
+ Clone {
    move |expected, actual| {
        Box::pin(async move {
            if expected.len() != actual.len() {
                return Err(Error::other(format!(
                    "length mismatch: expected {}, got {}",
                    expected.len(),
                    actual.len()
                )));
            }
            for (index, (expected, actual)) in expected.iter().zip(actual.iter()).enumerate() {
                if (expected - actual).abs() > tol {
                    return Err(Error::other(format!(
                        "mismatch at {index}: expected {expected}, got {actual}"
                    )));
                }
            }
            Ok(())
        })
    }
}

fn approx_vec_compare_f64(
    tol: f64,
) -> impl for<'a> Fn(
    &'a Vec<f64>,
    &'a Vec<f64>,
) -> Pin<Box<dyn std::future::Future<Output = Result<(), Error>> + 'a>>
+ Clone {
    move |expected, actual| {
        Box::pin(async move {
            if expected.len() != actual.len() {
                return Err(Error::other(format!(
                    "length mismatch: expected {}, got {}",
                    expected.len(),
                    actual.len()
                )));
            }
            for (index, (expected, actual)) in expected.iter().zip(actual.iter()).enumerate() {
                if (expected - actual).abs() > tol {
                    return Err(Error::other(format!(
                        "mismatch at {index}: expected {expected}, got {actual}"
                    )));
                }
            }
            Ok(())
        })
    }
}

#[tokio::test]
async fn cpu_casts_match_host_reference() {
    fusor_conformance::assert(async |values: Vec<f32>| {
        let actual = CpuTensor::from_slice([values.len()], &values).cast::<i32>();
        actual.as_slice().to_vec1()
    })
    .arg(HostVecGenerator::new(
        257,
        100,
        Uniform::new(-500.0f32, 500.0f32).unwrap(),
    ))
    .equal_to(async |values: Vec<f32>| values.into_iter().map(|value| value as i32).collect())
    .compare_with(exact_vec_compare::<i32>())
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();

    fusor_conformance::assert(async |values: Vec<i32>| {
        let actual = CpuTensor::from_slice([values.len()], &values).cast::<f64>();
        actual.as_slice().to_vec1()
    })
    .arg(HostVecGenerator::new(
        129,
        101,
        Uniform::new_inclusive(-300i32, 300i32).unwrap(),
    ))
    .equal_to(async |values: Vec<i32>| values.into_iter().map(|value| value as f64).collect())
    .compare_with(exact_vec_compare::<f64>())
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();

    fusor_conformance::assert(async |values: Vec<f64>| {
        let actual = CpuTensor::from_slice([values.len()], &values).cast::<f32>();
        actual.as_slice().to_vec1()
    })
    .arg(HostVecGenerator::new(
        129,
        102,
        Uniform::new(-100.0f64, 100.0f64).unwrap(),
    ))
    .equal_to(async |values: Vec<f64>| values.into_iter().map(|value| value as f32).collect())
    .compare_with(exact_vec_compare::<f32>())
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();

    fusor_conformance::assert(async |values: Vec<u8>| {
        let actual = CpuTensor::from_slice([values.len()], &values).cast::<f32>();
        actual.as_slice().to_vec1()
    })
    .arg(HostVecGenerator::new(
        257,
        103,
        Uniform::new_inclusive(0u8, 255u8).unwrap(),
    ))
    .equal_to(async |values: Vec<u8>| values.into_iter().map(|value| value as f32).collect())
    .compare_with(exact_vec_compare::<f32>())
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();

    const SHAPE_2D: [usize; 2] = [3, 4];
    fusor_conformance::assert(async |values: Vec<f32>| {
        let actual = CpuTensor::from_slice(SHAPE_2D, &values).cast::<i32>();
        flatten2(actual.as_slice().to_vec2())
    })
    .arg(HostVecGenerator::new(
        SHAPE_2D[0] * SHAPE_2D[1],
        104,
        Uniform::new(-20.0f32, 20.0f32).unwrap(),
    ))
    .equal_to(async |values: Vec<f32>| values.into_iter().map(|value| value as i32).collect())
    .compare_with(exact_vec_compare::<i32>())
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();
}

#[tokio::test]
async fn cpu_elementwise_and_pairwise_regressions_match_host_reference() {
    fusor_conformance::assert(async |values: Vec<i32>| {
        let actual = CpuTensor::from_slice([values.len()], &values)
            .abs()
            .to_concrete();
        actual.as_slice().to_vec1()
    })
    .arg(HostVecGenerator::new(
        129,
        110,
        Uniform::new_inclusive(-200i32, 200i32).unwrap(),
    ))
    .equal_to(async |values: Vec<i32>| values.into_iter().map(i32::abs).collect())
    .compare_with(exact_vec_compare::<i32>())
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();

    fusor_conformance::assert(async |values: Vec<f64>| {
        let actual = CpuTensor::from_slice([values.len()], &values)
            .sqrt()
            .to_concrete();
        actual.as_slice().to_vec1()
    })
    .arg(HostVecGenerator::new(
        129,
        111,
        Uniform::new(0.01f64, 100.0f64).unwrap(),
    ))
    .equal_to(async |values: Vec<f64>| values.into_iter().map(f64::sqrt).collect())
    .compare_with(approx_vec_compare_f64(1e-12))
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();

    fusor_conformance::assert(async |values: Vec<f64>| {
        let actual = CpuTensor::from_slice([values.len()], &values).pow_scalar(3.0);
        actual.as_slice().to_vec1()
    })
    .arg(HostVecGenerator::new(
        97,
        112,
        Uniform::new(-4.0f64, 4.0f64).unwrap(),
    ))
    .equal_to(async |values: Vec<f64>| values.into_iter().map(|value| value.powf(3.0)).collect())
    .compare_with(approx_vec_compare_f64(1e-10))
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();

    fusor_conformance::assert(async |lhs: Vec<i32>, rhs: Vec<i32>| {
        let lhs = CpuTensor::from_slice([lhs.len()], &lhs);
        let rhs = CpuTensor::from_slice([rhs.len()], &rhs);
        (&lhs + &rhs).to_concrete().as_slice().to_vec1()
    })
    .arg(HostVecGenerator::new(
        129,
        113,
        Uniform::new_inclusive(-100i32, 100i32).unwrap(),
    ))
    .arg(HostVecGenerator::new(
        129,
        114,
        Uniform::new_inclusive(-100i32, 100i32).unwrap(),
    ))
    .equal_to(async |lhs: Vec<i32>, rhs: Vec<i32>| {
        lhs.into_iter()
            .zip(rhs)
            .map(|(lhs, rhs)| lhs + rhs)
            .collect()
    })
    .compare_with(exact_vec_compare::<i32>())
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();

    const SHAPE_3D: [usize; 3] = [2, 2, 2];
    fusor_conformance::assert(async |lhs: Vec<f64>, rhs: Vec<f64>| {
        let lhs = CpuTensor::from_slice(SHAPE_3D, &lhs);
        let rhs = CpuTensor::from_slice(SHAPE_3D, &rhs);
        flatten3((&lhs + &rhs).to_concrete().as_slice().to_vec3())
    })
    .arg(HostVecGenerator::new(
        SHAPE_3D.iter().product(),
        115,
        Uniform::new(-10.0f64, 10.0f64).unwrap(),
    ))
    .arg(HostVecGenerator::new(
        SHAPE_3D.iter().product(),
        116,
        Uniform::new(-10.0f64, 10.0f64).unwrap(),
    ))
    .equal_to(async |lhs: Vec<f64>, rhs: Vec<f64>| {
        lhs.into_iter()
            .zip(rhs)
            .map(|(lhs, rhs)| lhs + rhs)
            .collect()
    })
    .compare_with(approx_vec_compare_f64(1e-12))
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();

    fusor_conformance::assert(async |lhs: Vec<i32>, rhs: Vec<i32>| {
        let lhs = CpuTensor::from_slice([lhs.len()], &lhs);
        let rhs = CpuTensor::from_slice([rhs.len()], &rhs);
        (&lhs * &rhs).to_concrete().as_slice().to_vec1()
    })
    .arg(HostVecGenerator::new(
        129,
        117,
        Uniform::new_inclusive(-20i32, 20i32).unwrap(),
    ))
    .arg(HostVecGenerator::new(
        129,
        118,
        Uniform::new_inclusive(-20i32, 20i32).unwrap(),
    ))
    .equal_to(async |lhs: Vec<i32>, rhs: Vec<i32>| {
        lhs.into_iter()
            .zip(rhs)
            .map(|(lhs, rhs)| lhs * rhs)
            .collect()
    })
    .compare_with(exact_vec_compare::<i32>())
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();

    fusor_conformance::assert(async |lhs: Vec<f64>, rhs: Vec<f64>| {
        let lhs = CpuTensor::from_slice([lhs.len()], &lhs);
        let rhs = CpuTensor::from_slice([rhs.len()], &rhs);
        (&lhs / &rhs).to_concrete().as_slice().to_vec1()
    })
    .arg(HostVecGenerator::new(
        129,
        119,
        Uniform::new(-50.0f64, 50.0f64).unwrap(),
    ))
    .arg(HostVecGenerator::new(
        129,
        120,
        Uniform::new(0.25f64, 8.0f64).unwrap(),
    ))
    .equal_to(async |lhs: Vec<f64>, rhs: Vec<f64>| {
        lhs.into_iter()
            .zip(rhs)
            .map(|(lhs, rhs)| lhs / rhs)
            .collect()
    })
    .compare_with(approx_vec_compare_f64(1e-12))
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();
}

#[tokio::test]
async fn fusor_ne_tensor_regression_matches_expected() {
    let fuzz = FuzzGenerator::<1, f32>::new([32])
        .with_seed(130)
        .with_distribution(Uniform::new(-3.0, 3.0).unwrap());

    fusor_conformance::assert(async |a: FusorTensor<1, f32>, b: FusorTensor<1, f32>| {
        a.ne_tensor(&b)
    })
    .arg(fuzz.clone())
    .arg(fuzz)
    .equal_to_resolved_with_device(async |a: Vec<f32>, b: Vec<f32>, device: Device| {
        let expected: Vec<f32> = a
            .into_iter()
            .zip(b)
            .map(|(lhs, rhs)| if lhs != rhs { 1.0 } else { 0.0 })
            .collect();
        FusorTensor::from_slice(&device, [expected.len()], &expected)
    })
    .compare_with(approx_compare::<1, f32>(0.0))
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();
}

#[tokio::test]
async fn cpu_integer_comparison_and_conditional_regressions_match_expected() {
    fusor_conformance::assert(async |a: Vec<i32>, b: Vec<i32>| {
        let a = CpuTensor::from_slice([a.len()], &a);
        let b = CpuTensor::from_slice([b.len()], &b);
        a.lt(b).to_concrete().as_slice().to_vec1()
    })
    .arg(HostVecGenerator::new(
        129,
        140,
        Uniform::new_inclusive(-20i32, 20i32).unwrap(),
    ))
    .arg(HostVecGenerator::new(
        129,
        141,
        Uniform::new_inclusive(-20i32, 20i32).unwrap(),
    ))
    .equal_to(async |a: Vec<i32>, b: Vec<i32>| {
        a.into_iter()
            .zip(b)
            .map(|(lhs, rhs)| if lhs < rhs { 1 } else { 0 })
            .collect()
    })
    .compare_with(exact_vec_compare::<i32>())
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();

    fusor_conformance::assert(async |values: Vec<i32>| {
        CpuTensor::from_slice([values.len()], &values)
            .eq_scalar(2)
            .to_concrete()
            .as_slice()
            .to_vec1()
    })
    .arg(HostVecGenerator::new(
        129,
        142,
        Uniform::new_inclusive(-4i32, 4i32).unwrap(),
    ))
    .equal_to(async |values: Vec<i32>| {
        values
            .into_iter()
            .map(|value| if value == 2 { 1 } else { 0 })
            .collect()
    })
    .compare_with(exact_vec_compare::<i32>())
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();

    fusor_conformance::assert(
        async |cond: Vec<i32>, on_true: Vec<i32>, on_false: Vec<i32>| {
            let cond = CpuTensor::from_slice([cond.len()], &cond);
            let on_true = CpuTensor::from_slice([on_true.len()], &on_true);
            let on_false = CpuTensor::from_slice([on_false.len()], &on_false);
            cond.where_cond(on_true, on_false)
                .to_concrete()
                .as_slice()
                .to_vec1()
        },
    )
    .arg(HostVecGenerator::new(
        129,
        143,
        Uniform::new_inclusive(-1i32, 1i32).unwrap(),
    ))
    .arg(HostVecGenerator::new(
        129,
        144,
        Uniform::new_inclusive(-100i32, 100i32).unwrap(),
    ))
    .arg(HostVecGenerator::new(
        129,
        145,
        Uniform::new_inclusive(-100i32, 100i32).unwrap(),
    ))
    .equal_to(
        async |cond: Vec<i32>, on_true: Vec<i32>, on_false: Vec<i32>| {
            cond.into_iter()
                .zip(on_true)
                .zip(on_false)
                .map(|((cond, on_true), on_false)| if cond != 0 { on_true } else { on_false })
                .collect()
        },
    )
    .compare_with(exact_vec_compare::<i32>())
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();
}

#[tokio::test]
async fn cpu_reduction_regressions_match_expected() {
    fusor_conformance::assert(async |values: Vec<f32>| {
        vec![CpuTensor::from_slice([values.len()], &values).sum()]
    })
    .arg(HostVecGenerator::new(
        1024,
        150,
        Uniform::new(-4.0f32, 4.0f32).unwrap(),
    ))
    .equal_to(async |values: Vec<f32>| vec![values.into_iter().sum::<f32>()])
    .compare_with(approx_vec_compare_f32(1e-3))
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();

    const SHAPE_2D: [usize; 2] = [8, 16];
    fusor_conformance::assert(async |values: Vec<f32>| {
        vec![CpuTensor::from_slice(SHAPE_2D, &values).sum()]
    })
    .arg(HostVecGenerator::new(
        SHAPE_2D[0] * SHAPE_2D[1],
        151,
        Uniform::new(-4.0f32, 4.0f32).unwrap(),
    ))
    .equal_to(async |values: Vec<f32>| vec![values.into_iter().sum::<f32>()])
    .compare_with(approx_vec_compare_f32(1e-4))
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();

    fusor_conformance::assert(async |values: Vec<f32>| {
        let tensor = CpuTensor::from_slice([values.len()], &values);
        vec![tensor.clone().max()]
    })
    .arg(HostVecGenerator::new(
        257,
        152,
        Uniform::new(-10.0f32, 10.0f32).unwrap(),
    ))
    .equal_to(async |values: Vec<f32>| vec![values.into_iter().fold(f32::NEG_INFINITY, f32::max)])
    .compare_with(approx_vec_compare_f32(0.0))
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();

    fusor_conformance::assert(async |values: Vec<f32>| {
        let tensor = CpuTensor::from_slice([values.len()], &values);
        vec![tensor.min()]
    })
    .arg(HostVecGenerator::new(
        257,
        153,
        Uniform::new(-10.0f32, 10.0f32).unwrap(),
    ))
    .equal_to(async |values: Vec<f32>| vec![values.into_iter().fold(f32::INFINITY, f32::min)])
    .compare_with(approx_vec_compare_f32(0.0))
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();

    fusor_conformance::assert(async |values: Vec<f32>| {
        let tensor = CpuTensor::from_slice([values.len()], &values);
        vec![tensor.prod()]
    })
    .arg(HostVecGenerator::new(
        64,
        154,
        Uniform::new(0.5f32, 1.5f32).unwrap(),
    ))
    .equal_to(async |values: Vec<f32>| vec![values.into_iter().product::<f32>()])
    .compare_with(approx_vec_compare_f32(1e-3))
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();

    fusor_conformance::assert(async |values: Vec<i32>| {
        let tensor = CpuTensor::from_slice([values.len()], &values);
        vec![tensor.clone().sum(), tensor.clone().max(), tensor.min()]
    })
    .arg(HostVecGenerator::new(
        129,
        155,
        Uniform::new_inclusive(-30i32, 30i32).unwrap(),
    ))
    .equal_to(async |values: Vec<i32>| {
        vec![
            values.iter().copied().sum::<i32>(),
            values.iter().copied().max().unwrap(),
            values.iter().copied().min().unwrap(),
        ]
    })
    .compare_with(exact_vec_compare::<i32>())
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();

    fusor_conformance::assert(async |values: Vec<f64>| {
        let tensor = CpuTensor::from_slice([values.len()], &values);
        vec![
            tensor.clone().sum(),
            tensor.clone().max(),
            tensor.clone().min(),
            tensor.prod(),
        ]
    })
    .arg(HostVecGenerator::new(
        129,
        156,
        Uniform::new(-3.0f64, 3.0f64).unwrap(),
    ))
    .equal_to(async |values: Vec<f64>| {
        vec![
            values.iter().copied().sum::<f64>(),
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            values.iter().copied().fold(f64::INFINITY, f64::min),
            values.iter().copied().product::<f64>(),
        ]
    })
    .compare_with(approx_vec_compare_f64(1e-7))
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();
}

#[tokio::test]
async fn cpu_axis_reduction_regressions_match_expected() {
    const SHAPE_3D: [usize; 3] = [2, 2, 2];
    fusor_conformance::assert(async |values: Vec<f32>| {
        let actual = CpuTensor::from_slice(SHAPE_3D, &values).sum_axis::<2>(0);
        flatten2(actual.as_slice().to_vec2())
    })
    .arg(HostVecGenerator::new(
        SHAPE_3D.iter().product(),
        160,
        Uniform::new(-6.0f32, 6.0f32).unwrap(),
    ))
    .equal_to(async |values: Vec<f32>| {
        vec![
            values[0] + values[4],
            values[1] + values[5],
            values[2] + values[6],
            values[3] + values[7],
        ]
    })
    .compare_with(approx_vec_compare_f32(1e-6))
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();
}

#[tokio::test]
async fn cpu_index_and_matmul_regressions_match_expected() {
    fusor_conformance::assert(async |input: Vec<i32>, indices: Vec<u32>| {
        let input = CpuTensor::from_slice([input.len()], &input);
        let indices_tensor = CpuTensor::from_slice([indices.len()], &indices);
        input
            .index_select(0, indices_tensor)
            .to_concrete()
            .as_slice()
            .to_vec1()
    })
    .arg(HostVecGenerator::new(
        32,
        170,
        Uniform::new_inclusive(-500i32, 500i32).unwrap(),
    ))
    .arg(HostVecGenerator::new(
        8,
        171,
        Uniform::new(0u32, 32u32).unwrap(),
    ))
    .equal_to(async |input: Vec<i32>, indices: Vec<u32>| {
        indices
            .into_iter()
            .map(|index| input[index as usize])
            .collect()
    })
    .compare_with(exact_vec_compare::<i32>())
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();

    const LHS_SHAPE: [usize; 2] = [3, 4];
    const RHS_SHAPE: [usize; 2] = [4, 2];
    fusor_conformance::assert(async |lhs: Vec<f64>, rhs: Vec<f64>| {
        let lhs = CpuTensor::from_slice(LHS_SHAPE, &lhs);
        let rhs = CpuTensor::from_slice(RHS_SHAPE, &rhs);
        flatten2(lhs.matmul(rhs).as_slice().to_vec2())
    })
    .arg(HostVecGenerator::new(
        LHS_SHAPE[0] * LHS_SHAPE[1],
        172,
        Uniform::new(-3.0f64, 3.0f64).unwrap(),
    ))
    .arg(HostVecGenerator::new(
        RHS_SHAPE[0] * RHS_SHAPE[1],
        173,
        Uniform::new(-3.0f64, 3.0f64).unwrap(),
    ))
    .equal_to(async |lhs: Vec<f64>, rhs: Vec<f64>| {
        (0..LHS_SHAPE[0])
            .flat_map(|row| {
                let lhs = lhs.clone();
                let rhs = rhs.clone();
                (0..RHS_SHAPE[1]).map(move |col| {
                    (0..LHS_SHAPE[1])
                        .map(|k| lhs[row * LHS_SHAPE[1] + k] * rhs[k * RHS_SHAPE[1] + col])
                        .sum()
                })
            })
            .collect()
    })
    .compare_with(approx_vec_compare_f64(1e-12))
    .devices([Device::Cpu])
    .runs(5)
    .await
    .unwrap();
}
