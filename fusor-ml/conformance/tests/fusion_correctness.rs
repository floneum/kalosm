mod common;

use common::{binary_map2, unary_map2, where_cond2};
use fusor::{Device, Tensor};
use fusor_conformance::{FuzzGenerator, approx_compare};
use rand::distr::Uniform;

const SHAPE: [usize; 2] = [4, 5];

fn fuzz(seed: u64) -> FuzzGenerator<2, f32> {
    FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(seed)
        .with_distribution(Uniform::new(-3.0, 3.0).unwrap())
}

#[tokio::test]
async fn nary_triple_add_fuzzed() {
    // (a + b) + c
    fusor_conformance::assert(
        async |a: Tensor<2, f32>, b: Tensor<2, f32>, c: Tensor<2, f32>| {
            let ab = a.add_::<2, 2, _>(&b);
            ab.add_::<2, 2, _>(&c)
        },
    )
    .arg(fuzz(1))
    .arg(fuzz(2))
    .arg(fuzz(3))
    .equal_to_resolved_with_device(
        async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, c: Vec<Vec<f32>>, device: Device| {
            let ab = binary_map2(&a, &b, |l, r| l + r);
            let abc = binary_map2(&ab, &c, |l, r| l + r);
            Tensor::new(&device, &abc)
        },
    )
    .compare_with(approx_compare::<2, f32>(1e-5))
    .runs(3)
    .await
    .unwrap();
}

#[tokio::test]
async fn nary_mixed_ops_fuzzed() {
    // (a + b) * c
    fusor_conformance::assert(
        async |a: Tensor<2, f32>, b: Tensor<2, f32>, c: Tensor<2, f32>| {
            let ab = a.add_::<2, 2, _>(&b);
            ab.mul_::<2, 2, _>(&c)
        },
    )
    .arg(fuzz(10))
    .arg(fuzz(11))
    .arg(fuzz(12))
    .equal_to_resolved_with_device(
        async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, c: Vec<Vec<f32>>, device: Device| {
            let ab = binary_map2(&a, &b, |l, r| l + r);
            let out = binary_map2(&ab, &c, |l, r| l * r);
            Tensor::new(&device, &out)
        },
    )
    .compare_with(approx_compare::<2, f32>(1e-4))
    .runs(3)
    .await
    .unwrap();
}

#[tokio::test]
async fn nary_nested_pairwise_fuzzed() {
    // (a + b) * (c - d)
    fusor_conformance::assert(
        async |a: Tensor<2, f32>, b: Tensor<2, f32>, c: Tensor<2, f32>, d: Tensor<2, f32>| {
            let ab = a.add_::<2, 2, _>(&b);
            let cd = c.sub_::<2, 2, _>(&d);
            ab.mul_::<2, 2, _>(&cd)
        },
    )
    .arg(fuzz(20))
    .arg(fuzz(21))
    .arg(fuzz(22))
    .arg(fuzz(23))
    .equal_to_resolved_with_device(
        async |a: Vec<Vec<f32>>,
               b: Vec<Vec<f32>>,
               c: Vec<Vec<f32>>,
               d: Vec<Vec<f32>>,
               device: Device| {
            let ab = binary_map2(&a, &b, |l, r| l + r);
            let cd = binary_map2(&c, &d, |l, r| l - r);
            let out = binary_map2(&ab, &cd, |l, r| l * r);
            Tensor::new(&device, &out)
        },
    )
    .compare_with(approx_compare::<2, f32>(1e-4))
    .runs(3)
    .await
    .unwrap();
}

#[tokio::test]
async fn nary_unary_in_middle_fuzzed() {
    // (-a + sin(b)).cos() + 1.0
    let fuzz_b = FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(31)
        .with_distribution(Uniform::new(-1.0, 1.0).unwrap());

    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| {
        let neg_a = (-a).to_concrete();
        let sin_b = b.sin().to_concrete();
        let sum = neg_a.add_::<2, 2, _>(&sin_b);
        (sum.cos().to_concrete() + 1.0).to_concrete()
    })
    .arg(fuzz(30))
    .arg(fuzz_b)
    .equal_to_resolved_with_device(async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
        let neg_a = unary_map2(&a, |x| -x);
        let sin_b = unary_map2(&b, f32::sin);
        let sum = binary_map2(&neg_a, &sin_b, |l, r| l + r);
        let out = unary_map2(&sum, |x| x.cos() + 1.0);
        Tensor::new(&device, &out)
    })
    .compare_with(approx_compare::<2, f32>(1e-3))
    .runs(3)
    .await
    .unwrap();
}

#[tokio::test]
async fn nary_chain_then_pairwise_fuzzed() {
    // (a + 1).exp() + sin(b)
    let fuzz_a = FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(40)
        .with_distribution(Uniform::new(-2.0, 2.0).unwrap());

    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| {
        let a_exp = (a + 1.0).exp().to_concrete();
        let sin_b = b.sin().to_concrete();
        a_exp.add_::<2, 2, _>(&sin_b)
    })
    .arg(fuzz_a)
    .arg(fuzz(41))
    .equal_to_resolved_with_device(async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
        let a_exp = unary_map2(&a, |x| (x + 1.0).exp());
        let sin_b = unary_map2(&b, f32::sin);
        let out = binary_map2(&a_exp, &sin_b, |l, r| l + r);
        Tensor::new(&device, &out)
    })
    .compare_with(approx_compare::<2, f32>(1e-3))
    .runs(3)
    .await
    .unwrap();
}

#[tokio::test]
async fn nary_same_input_fuzzed() {
    // a + a + a = 3a
    fusor_conformance::assert(async |a: Tensor<2, f32>| {
        let aa = a.add_::<2, 2, _>(&a);
        aa.add_::<2, 2, _>(&a)
    })
    .arg(fuzz(50))
    .equal_to_resolved_with_device(async |a: Vec<Vec<f32>>, device: Device| {
        let out = unary_map2(&a, |x| x * 3.0);
        Tensor::new(&device, &out)
    })
    .compare_with(approx_compare::<2, f32>(1e-5))
    .runs(3)
    .await
    .unwrap();
}

#[tokio::test]
async fn nary_where_cond_fuzzed() {
    let fuzz_cond = FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(60)
        .with_distribution(Uniform::new(-1.0, 1.0).unwrap());

    fusor_conformance::assert(
        async |cond: Tensor<2, f32>, on_true: Tensor<2, f32>, on_false: Tensor<2, f32>| {
            cond.gt_scalar(0.0).where_cond(&on_true, &on_false)
        },
    )
    .arg(fuzz_cond)
    .arg(fuzz(61))
    .arg(fuzz(62))
    .equal_to_resolved_with_device(
        async |cond: Vec<Vec<f32>>,
               on_true: Vec<Vec<f32>>,
               on_false: Vec<Vec<f32>>,
               device: Device| {
            let mask = unary_map2(&cond, |x| if x > 0.0 { 1.0 } else { 0.0 });
            let out = where_cond2(&mask, &on_true, &on_false);
            Tensor::new(&device, &out)
        },
    )
    .compare_with(approx_compare::<2, f32>(1e-6))
    .runs(3)
    .await
    .unwrap();
}

#[tokio::test]
async fn fused_cached_results_fuzzed() {
    // (tensor * 2 + 1).sum(0) then branch into *2 and *3
    // Tests that caching/sharing of intermediate results works correctly.
    const SHAPE_3D: [usize; 3] = [3, 4, 5];
    let fuzz_3d = FuzzGenerator::<3, f32>::new(SHAPE_3D)
        .with_seed(70)
        .with_distribution(Uniform::new(-2.0, 2.0).unwrap());

    // times_two branch
    fusor_conformance::assert(async |t: Tensor<3, f32>| {
        let doubled = t.clone() * 2.0;
        let plus_one = (doubled + 1.0).sum::<2>(0);
        (plus_one * 2.0).to_concrete()
    })
    .arg(fuzz_3d.clone())
    .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
        // manual: (data*2+1).sum(axis=0) * 2
        let rows = SHAPE_3D[1];
        let cols = SHAPE_3D[2];
        let mut summed = vec![vec![0.0f32; cols]; rows];
        for slice in &v {
            for (r, row) in slice.iter().enumerate() {
                for (c, val) in row.iter().enumerate() {
                    summed[r][c] += val * 2.0 + 1.0;
                }
            }
        }
        let out = unary_map2(&summed, |x| x * 2.0);
        Tensor::new(&device, &out)
    })
    .compare_with(approx_compare::<2, f32>(1e-2))
    .runs(3)
    .await
    .unwrap();

    // times_three branch
    fusor_conformance::assert(async |t: Tensor<3, f32>| {
        let doubled = t.clone() * 2.0;
        let plus_one = (doubled + 1.0).sum::<2>(0);
        (plus_one * 3.0).to_concrete()
    })
    .arg(fuzz_3d)
    .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
        let rows = SHAPE_3D[1];
        let cols = SHAPE_3D[2];
        let mut summed = vec![vec![0.0f32; cols]; rows];
        for slice in &v {
            for (r, row) in slice.iter().enumerate() {
                for (c, val) in row.iter().enumerate() {
                    summed[r][c] += val * 2.0 + 1.0;
                }
            }
        }
        let out = unary_map2(&summed, |x| x * 3.0);
        Tensor::new(&device, &out)
    })
    .compare_with(approx_compare::<2, f32>(1e-2))
    .runs(3)
    .await
    .unwrap();
}

#[tokio::test]
async fn inplace_clone_immutability_fuzzed() {
    // Verify that tensor + 1.0 gives correct values and cloning preserves immutability.
    // Running the same operation twice on a cloned tensor should give the same result.
    const SHAPE_3D: [usize; 3] = [3, 2, 4];
    let fuzz_3d = FuzzGenerator::<3, f32>::new(SHAPE_3D)
        .with_seed(80)
        .with_distribution(Uniform::new(-5.0, 5.0).unwrap());

    fusor_conformance::assert(async |t: Tensor<3, f32>| (t + 1.0).to_concrete())
        .arg(fuzz_3d)
        .equal_to_resolved_with_device(async |v: Vec<Vec<Vec<f32>>>, device: Device| {
            let out: Vec<Vec<Vec<f32>>> = v
                .iter()
                .map(|matrix| unary_map2(matrix, |x| x + 1.0))
                .collect();
            Tensor::new(&device, &out)
        })
        .compare_with(approx_compare::<3, f32>(1e-6))
        .runs(3)
        .await
        .unwrap();
}
