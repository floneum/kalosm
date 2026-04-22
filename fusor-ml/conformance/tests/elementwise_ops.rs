mod common;

use common::{binary_map2, unary_map2, where_cond1, where_cond2};
use fusor::{Device, Tensor};
use fusor_conformance::{FuzzGenerator, approx_compare};
use rand::distr::Uniform;

const SHAPE: [usize; 2] = [45, 45];

fn signed() -> FuzzGenerator<2, f32> {
    FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(100)
        .with_distribution(Uniform::new(-3.5, 3.5).unwrap())
}

fn positive() -> FuzzGenerator<2, f32> {
    FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(101)
        .with_distribution(Uniform::new(0.1, 3.0).unwrap())
}

fn unit() -> FuzzGenerator<2, f32> {
    FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(102)
        .with_distribution(Uniform::new(-0.95, 0.95).unwrap())
}

fn tan_domain() -> FuzzGenerator<2, f32> {
    FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(103)
        .with_distribution(Uniform::new(-0.6, 0.6).unwrap())
}

fn approx_exp_domain() -> FuzzGenerator<2, f32> {
    FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(104)
        .with_distribution(Uniform::new(-0.5, 0.5).unwrap())
}

fn acosh_domain() -> FuzzGenerator<2, f32> {
    FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(105)
        .with_distribution(Uniform::new(1.01, 3.5).unwrap())
}

macro_rules! fuzz_unary {
    ($name:ident, $gen:expr, $op:expr, $ref_fn:expr, $tol:expr) => {
        fusor_conformance::assert(async |x: Tensor<2, f32>| ($op)(x))
            .arg($gen)
            .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &unary_map2(&v, $ref_fn))
            })
            .compare_with(approx_compare::<2, f32>($tol))
            .runs(3)
            .await
            .unwrap();
    };
}

#[tokio::test]
async fn unary_math_ops_match_host_reference() {
    // abs
    fuzz_unary!(
        _abs,
        signed(),
        |x: Tensor<2, f32>| x.abs().to_concrete(),
        f32::abs,
        1e-6
    );

    // exp
    fuzz_unary!(
        _exp,
        signed(),
        |x: Tensor<2, f32>| x.exp().to_concrete(),
        f32::exp,
        1e-3
    );

    // exp2
    fuzz_unary!(
        _exp2,
        signed(),
        |x: Tensor<2, f32>| x.exp2().to_concrete(),
        f32::exp2,
        1e-3
    );

    // sin
    fuzz_unary!(
        _sin,
        signed(),
        |x: Tensor<2, f32>| x.sin().to_concrete(),
        f32::sin,
        1e-5
    );

    // cos
    fuzz_unary!(
        _cos,
        signed(),
        |x: Tensor<2, f32>| x.cos().to_concrete(),
        f32::cos,
        1e-5
    );

    // tan
    fuzz_unary!(
        _tan,
        tan_domain(),
        |x: Tensor<2, f32>| x.tan().to_concrete(),
        f32::tan,
        1e-4
    );

    // tanh
    fuzz_unary!(
        _tanh,
        signed(),
        |x: Tensor<2, f32>| x.tanh().to_concrete(),
        f32::tanh,
        1e-5
    );

    // atan
    fuzz_unary!(
        _atan,
        signed(),
        |x: Tensor<2, f32>| x.atan().to_concrete(),
        f32::atan,
        1e-5
    );

    // sinh
    fuzz_unary!(
        _sinh,
        signed(),
        |x: Tensor<2, f32>| x.sinh().to_concrete(),
        f32::sinh,
        1e-3
    );

    // cosh
    fuzz_unary!(
        _cosh,
        signed(),
        |x: Tensor<2, f32>| x.cosh().to_concrete(),
        f32::cosh,
        1e-3
    );

    // asinh
    fuzz_unary!(
        _asinh,
        signed(),
        |x: Tensor<2, f32>| x.asinh().to_concrete(),
        f32::asinh,
        1e-5
    );

    // approximate_exp
    fuzz_unary!(
        _approx_exp,
        approx_exp_domain(),
        |x: Tensor<2, f32>| x.approximate_exp(),
        f32::exp,
        6e-2
    );

    // less_approximate_exp
    fuzz_unary!(
        _less_approx_exp,
        approx_exp_domain(),
        |x: Tensor<2, f32>| x.less_approximate_exp(),
        f32::exp,
        1.5e-2
    );

    // tanh_exact
    fuzz_unary!(
        _tanh_exact,
        signed(),
        |x: Tensor<2, f32>| x.tanh_exact(),
        f32::tanh,
        1e-6
    );

    // sqr
    fuzz_unary!(
        _sqr,
        signed(),
        |x: Tensor<2, f32>| x.sqr().to_concrete(),
        |v: f32| v * v,
        1e-5
    );
}

#[tokio::test]
async fn restricted_domain_unary_ops_match_host_reference() {
    // sqrt
    fuzz_unary!(
        _sqrt,
        positive(),
        |x: Tensor<2, f32>| x.sqrt().to_concrete(),
        f32::sqrt,
        1e-6
    );

    // log
    fuzz_unary!(
        _log,
        positive(),
        |x: Tensor<2, f32>| x.log().to_concrete(),
        f32::ln,
        1e-5
    );

    // log2
    fuzz_unary!(
        _log2,
        positive(),
        |x: Tensor<2, f32>| x.log2().to_concrete(),
        f32::log2,
        1e-5
    );

    // Inverse trig / hyperbolic functions diverge from libm by up to ~6e-5
    // on the lavapipe/llvmpipe Linux CI adapter (the `unit()` distribution
    // gets close to the asymptotes where these ops are most sensitive).
    // 1e-4 covers the observed gap without papering over a real regression.

    // asin
    fuzz_unary!(
        _asin,
        unit(),
        |x: Tensor<2, f32>| x.asin().to_concrete(),
        f32::asin,
        1e-4
    );

    // acos
    fuzz_unary!(
        _acos,
        unit(),
        |x: Tensor<2, f32>| x.acos().to_concrete(),
        f32::acos,
        1e-4
    );

    // atanh
    fuzz_unary!(
        _atanh,
        unit(),
        |x: Tensor<2, f32>| x.atanh().to_concrete(),
        f32::atanh,
        1e-4
    );

    // acosh
    fuzz_unary!(
        _acosh,
        acosh_domain(),
        |x: Tensor<2, f32>| x.acosh().to_concrete(),
        f32::acosh,
        1e-4
    );
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn gelu(v: f32) -> f32 {
    0.5 * v * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (v + 0.044_715 * v.powi(3))).tanh())
}

#[tokio::test]
async fn activation_and_scalar_ops_match_host_reference() {
    // relu
    fuzz_unary!(
        _relu,
        signed(),
        |x: Tensor<2, f32>| x.relu(),
        |v: f32| v.max(0.0),
        1e-6
    );

    // silu
    fuzz_unary!(_silu, signed(), |x: Tensor<2, f32>| x.silu(), silu, 1e-5);

    // gelu
    fuzz_unary!(_gelu, signed(), |x: Tensor<2, f32>| x.gelu(), gelu, 1e-4);

    // add_scalar
    fuzz_unary!(
        _add_scalar,
        signed(),
        |x: Tensor<2, f32>| x.add_scalar(1.25),
        |v: f32| v + 1.25,
        1e-6
    );

    // sub_scalar
    fuzz_unary!(
        _sub_scalar,
        signed(),
        |x: Tensor<2, f32>| x.sub_scalar(1.25),
        |v: f32| v - 1.25,
        1e-6
    );

    // mul_scalar
    fuzz_unary!(
        _mul_scalar,
        signed(),
        |x: Tensor<2, f32>| x.mul_scalar(-1.5),
        |v: f32| v * -1.5,
        1e-5
    );

    // div_scalar
    fuzz_unary!(
        _div_scalar,
        signed(),
        |x: Tensor<2, f32>| x.div_scalar(2.0),
        |v: f32| v / 2.0,
        1e-6
    );

    // pow_scalar
    fuzz_unary!(
        _pow_scalar,
        positive(),
        |x: Tensor<2, f32>| x.pow_scalar(2.5),
        |v: f32| v.powf(2.5),
        1e-4
    );

    // max_scalar
    fuzz_unary!(
        _max_scalar,
        signed(),
        |x: Tensor<2, f32>| x.max_scalar(0.4),
        |v: f32| v.max(0.4),
        1e-6
    );

    // min_scalar
    fuzz_unary!(
        _min_scalar,
        signed(),
        |x: Tensor<2, f32>| x.min_scalar(-0.4),
        |v: f32| v.min(-0.4),
        1e-6
    );

    // clamp
    fuzz_unary!(
        _clamp,
        signed(),
        |x: Tensor<2, f32>| x.clamp(-0.75, 0.75),
        |v: f32| v.clamp(-0.75, 0.75),
        1e-6
    );
}

#[tokio::test]
async fn binary_ops_match_host_reference() {
    let gen_a = positive();
    let gen_b_1d = FuzzGenerator::<1, f32>::new([SHAPE[1]])
        .with_seed(110)
        .with_distribution(Uniform::new(0.5, 2.5).unwrap());
    let gen_b_2d = FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(111)
        .with_distribution(Uniform::new(0.5, 2.5).unwrap());

    // add broadcast 2d + 1d
    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<1, f32>| a.add_::<1, 2, _>(&b))
        .arg(gen_a.clone())
        .arg(gen_b_1d.clone())
        .equal_to_resolved_with_device(async |a: Vec<Vec<f32>>, b: Vec<f32>, device: Device| {
            let out = common::broadcast_binary_2d_1d(&a, &b, |l, r| l + r);
            Tensor::new(&device, &out)
        })
        .compare_with(approx_compare::<2, f32>(1e-6))
        .runs(3)
        .await
        .unwrap();

    // sub broadcast
    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<1, f32>| a.sub_::<1, 2, _>(&b))
        .arg(gen_a.clone())
        .arg(gen_b_1d.clone())
        .equal_to_resolved_with_device(async |a: Vec<Vec<f32>>, b: Vec<f32>, device: Device| {
            let out = common::broadcast_binary_2d_1d(&a, &b, |l, r| l - r);
            Tensor::new(&device, &out)
        })
        .compare_with(approx_compare::<2, f32>(1e-6))
        .runs(3)
        .await
        .unwrap();

    // mul broadcast
    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<1, f32>| a.mul_::<1, 2, _>(&b))
        .arg(gen_a.clone())
        .arg(gen_b_1d.clone())
        .equal_to_resolved_with_device(async |a: Vec<Vec<f32>>, b: Vec<f32>, device: Device| {
            let out = common::broadcast_binary_2d_1d(&a, &b, |l, r| l * r);
            Tensor::new(&device, &out)
        })
        .compare_with(approx_compare::<2, f32>(1e-6))
        .runs(3)
        .await
        .unwrap();

    // div broadcast
    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<1, f32>| a.div_::<1, 2, _>(&b))
        .arg(gen_a.clone())
        .arg(gen_b_1d.clone())
        .equal_to_resolved_with_device(async |a: Vec<Vec<f32>>, b: Vec<f32>, device: Device| {
            let out = common::broadcast_binary_2d_1d(&a, &b, |l, r| l / r);
            Tensor::new(&device, &out)
        })
        .compare_with(approx_compare::<2, f32>(1e-6))
        .runs(3)
        .await
        .unwrap();

    // pow elementwise 2d
    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.pow_::<2, 2, _>(&b))
        .arg(gen_a.clone())
        .arg(gen_b_2d)
        .equal_to_resolved_with_device(
            async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
                let out = binary_map2(&a, &b, |l, r| l.powf(r));
                Tensor::new(&device, &out)
            },
        )
        .compare_with(approx_compare::<2, f32>(1e-4))
        .runs(3)
        .await
        .unwrap();
}

#[tokio::test]
async fn comparison_and_conditionals_match_expected() {
    let fuzz = signed();

    // eq_scalar
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.eq_scalar(0.25))
        .arg(fuzz.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(
                &device,
                &common::compare_scalar_map2(&v, 0.25, |l, r| l == r),
            )
        })
        .compare_with(approx_compare::<2, f32>(0.0))
        .runs(3)
        .await
        .unwrap();

    // lt_scalar
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.lt_scalar(0.25))
        .arg(fuzz.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(
                &device,
                &common::compare_scalar_map2(&v, 0.25, |l, r| l < r),
            )
        })
        .compare_with(approx_compare::<2, f32>(0.0))
        .runs(3)
        .await
        .unwrap();

    // lte_scalar
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.lte_scalar(0.25))
        .arg(fuzz.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(
                &device,
                &common::compare_scalar_map2(&v, 0.25, |l, r| l <= r),
            )
        })
        .compare_with(approx_compare::<2, f32>(0.0))
        .runs(3)
        .await
        .unwrap();

    // gt_scalar
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.gt_scalar(0.25))
        .arg(fuzz.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(
                &device,
                &common::compare_scalar_map2(&v, 0.25, |l, r| l > r),
            )
        })
        .compare_with(approx_compare::<2, f32>(0.0))
        .runs(3)
        .await
        .unwrap();

    // gte_scalar
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.gte_scalar(0.25))
        .arg(fuzz.clone())
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(
                &device,
                &common::compare_scalar_map2(&v, 0.25, |l, r| l >= r),
            )
        })
        .compare_with(approx_compare::<2, f32>(0.0))
        .runs(3)
        .await
        .unwrap();

    let gen_b = FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(120)
        .with_distribution(Uniform::new(-2.0, 2.0).unwrap());

    // eq_tensor
    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.eq_tensor(&b))
        .arg(fuzz.clone())
        .arg(gen_b.clone())
        .equal_to_resolved_with_device(
            async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &common::compare_tensor_map2(&a, &b, |l, r| l == r))
            },
        )
        .compare_with(approx_compare::<2, f32>(0.0))
        .devices([Device::Cpu])
        .runs(3)
        .await
        .unwrap();

    // lt_tensor
    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.lt_tensor(&b))
        .arg(fuzz.clone())
        .arg(gen_b.clone())
        .equal_to_resolved_with_device(
            async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &common::compare_tensor_map2(&a, &b, |l, r| l < r))
            },
        )
        .compare_with(approx_compare::<2, f32>(0.0))
        .devices([Device::Cpu])
        .runs(3)
        .await
        .unwrap();

    // lte_tensor
    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.lte_tensor(&b))
        .arg(fuzz.clone())
        .arg(gen_b.clone())
        .equal_to_resolved_with_device(
            async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &common::compare_tensor_map2(&a, &b, |l, r| l <= r))
            },
        )
        .compare_with(approx_compare::<2, f32>(0.0))
        .devices([Device::Cpu])
        .runs(3)
        .await
        .unwrap();

    // gt_tensor
    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.gt_tensor(&b))
        .arg(fuzz.clone())
        .arg(gen_b.clone())
        .equal_to_resolved_with_device(
            async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &common::compare_tensor_map2(&a, &b, |l, r| l > r))
            },
        )
        .compare_with(approx_compare::<2, f32>(0.0))
        .devices([Device::Cpu])
        .runs(3)
        .await
        .unwrap();

    // gte_tensor
    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.gte_tensor(&b))
        .arg(fuzz.clone())
        .arg(gen_b.clone())
        .equal_to_resolved_with_device(
            async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &common::compare_tensor_map2(&a, &b, |l, r| l >= r))
            },
        )
        .compare_with(approx_compare::<2, f32>(0.0))
        .devices([Device::Cpu])
        .runs(3)
        .await
        .unwrap();

    // where_cond
    let gen_cond = FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(130)
        .with_distribution(Uniform::new(-1.0, 1.0).unwrap());
    let gen_on_true = FuzzGenerator::<2, f32>::new(SHAPE).with_seed(131);
    let gen_on_false = FuzzGenerator::<2, f32>::new(SHAPE).with_seed(132);

    fusor_conformance::assert(
        async |cond: Tensor<2, f32>, on_true: Tensor<2, f32>, on_false: Tensor<2, f32>| {
            cond.where_cond(&on_true, &on_false)
        },
    )
    .arg(gen_cond)
    .arg(gen_on_true)
    .arg(gen_on_false)
    .equal_to_resolved_with_device(
        async |cond: Vec<Vec<f32>>,
               on_true: Vec<Vec<f32>>,
               on_false: Vec<Vec<f32>>,
               device: Device| {
            Tensor::new(&device, &where_cond2(&cond, &on_true, &on_false))
        },
    )
    .compare_with(approx_compare::<2, f32>(0.0))
    .runs(3)
    .await
    .unwrap();
}

#[tokio::test]
async fn same_shape_binary_ops_match_host_reference() {
    let gen_a = FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(1)
        .with_distribution(Uniform::new(0.1, 3.0).unwrap());
    let gen_b = FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(2)
        .with_distribution(Uniform::new(0.1, 3.0).unwrap());

    // add
    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.add_::<2, 2, _>(&b))
        .arg(gen_a.clone())
        .arg(gen_b.clone())
        .equal_to_resolved_with_device(
            async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
                let out = binary_map2(&a, &b, |l, r| l + r);
                Tensor::new(&device, &out)
            },
        )
        .compare_with(approx_compare::<2, f32>(1e-5))
        .runs(3)
        .await
        .unwrap();

    // sub
    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.sub_::<2, 2, _>(&b))
        .arg(gen_a.clone())
        .arg(gen_b.clone())
        .equal_to_resolved_with_device(
            async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
                let out = binary_map2(&a, &b, |l, r| l - r);
                Tensor::new(&device, &out)
            },
        )
        .compare_with(approx_compare::<2, f32>(1e-5))
        .runs(3)
        .await
        .unwrap();

    // mul
    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.mul_::<2, 2, _>(&b))
        .arg(gen_a.clone())
        .arg(gen_b.clone())
        .equal_to_resolved_with_device(
            async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
                let out = binary_map2(&a, &b, |l, r| l * r);
                Tensor::new(&device, &out)
            },
        )
        .compare_with(approx_compare::<2, f32>(1e-5))
        .runs(3)
        .await
        .unwrap();

    // div
    fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.div_::<2, 2, _>(&b))
        .arg(gen_a.clone())
        .arg(gen_b.clone())
        .equal_to_resolved_with_device(
            async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
                let out = binary_map2(&a, &b, |l, r| l / r);
                Tensor::new(&device, &out)
            },
        )
        .compare_with(approx_compare::<2, f32>(1e-5))
        .runs(3)
        .await
        .unwrap();
}

#[tokio::test]
async fn large_tensor_binary_and_conditional_regressions() {
    const LARGE_SHAPE_1D: [usize; 1] = [2048];

    let gen_binary_a = FuzzGenerator::<1, f32>::new(LARGE_SHAPE_1D)
        .with_seed(140)
        .with_distribution(Uniform::new(0.5, 4.0).unwrap());
    let gen_binary_b = FuzzGenerator::<1, f32>::new(LARGE_SHAPE_1D)
        .with_seed(141)
        .with_distribution(Uniform::new(0.5, 4.0).unwrap());

    fusor_conformance::assert(async |a: Tensor<1, f32>, b: Tensor<1, f32>| a.add_::<1, 1, _>(&b))
        .arg(gen_binary_a.clone())
        .arg(gen_binary_b.clone())
        .equal_to_resolved_with_device(async |a: Vec<f32>, b: Vec<f32>, device: Device| {
            let out: Vec<f32> = a.iter().zip(b.iter()).map(|(l, r)| l + r).collect();
            Tensor::from_slice(&device, LARGE_SHAPE_1D, &out)
        })
        .compare_with(approx_compare::<1, f32>(1e-6))
        .runs(3)
        .await
        .unwrap();

    fusor_conformance::assert(async |a: Tensor<1, f32>, b: Tensor<1, f32>| a.sub_::<1, 1, _>(&b))
        .arg(gen_binary_a.clone())
        .arg(gen_binary_b.clone())
        .equal_to_resolved_with_device(async |a: Vec<f32>, b: Vec<f32>, device: Device| {
            let out: Vec<f32> = a.iter().zip(b.iter()).map(|(l, r)| l - r).collect();
            Tensor::from_slice(&device, LARGE_SHAPE_1D, &out)
        })
        .compare_with(approx_compare::<1, f32>(1e-6))
        .runs(3)
        .await
        .unwrap();

    fusor_conformance::assert(async |a: Tensor<1, f32>, b: Tensor<1, f32>| a.mul_::<1, 1, _>(&b))
        .arg(gen_binary_a.clone())
        .arg(gen_binary_b.clone())
        .equal_to_resolved_with_device(async |a: Vec<f32>, b: Vec<f32>, device: Device| {
            let out: Vec<f32> = a.iter().zip(b.iter()).map(|(l, r)| l * r).collect();
            Tensor::from_slice(&device, LARGE_SHAPE_1D, &out)
        })
        .compare_with(approx_compare::<1, f32>(1e-6))
        .runs(3)
        .await
        .unwrap();

    fusor_conformance::assert(async |a: Tensor<1, f32>, b: Tensor<1, f32>| a.div_::<1, 1, _>(&b))
        .arg(gen_binary_a.clone())
        .arg(gen_binary_b.clone())
        .equal_to_resolved_with_device(async |a: Vec<f32>, b: Vec<f32>, device: Device| {
            let out: Vec<f32> = a.iter().zip(b.iter()).map(|(l, r)| l / r).collect();
            Tensor::from_slice(&device, LARGE_SHAPE_1D, &out)
        })
        .compare_with(approx_compare::<1, f32>(1e-6))
        .runs(3)
        .await
        .unwrap();

    let gen_cmp_a = FuzzGenerator::<1, f32>::new(LARGE_SHAPE_1D)
        .with_seed(142)
        .with_distribution(Uniform::new(-10.0, 10.0).unwrap());
    let gen_cmp_b = FuzzGenerator::<1, f32>::new(LARGE_SHAPE_1D)
        .with_seed(143)
        .with_distribution(Uniform::new(-10.0, 10.0).unwrap());

    fusor_conformance::assert(async |a: Tensor<1, f32>, b: Tensor<1, f32>| a.lt_tensor(&b))
        .arg(gen_cmp_a)
        .arg(gen_cmp_b)
        .equal_to_resolved_with_device(async |a: Vec<f32>, b: Vec<f32>, device: Device| {
            let out: Vec<f32> = a
                .iter()
                .zip(b.iter())
                .map(|(l, r)| if l < r { 1.0 } else { 0.0 })
                .collect();
            Tensor::from_slice(&device, LARGE_SHAPE_1D, &out)
        })
        .compare_with(approx_compare::<1, f32>(0.0))
        .devices([Device::Cpu])
        .runs(3)
        .await
        .unwrap();

    let gen_cond = FuzzGenerator::<1, f32>::new(LARGE_SHAPE_1D)
        .with_seed(144)
        .with_distribution(Uniform::new(-1.0, 1.0).unwrap());
    let gen_true = FuzzGenerator::<1, f32>::new(LARGE_SHAPE_1D).with_seed(145);
    let gen_false = FuzzGenerator::<1, f32>::new(LARGE_SHAPE_1D).with_seed(146);

    fusor_conformance::assert(
        async |cond: Tensor<1, f32>, on_true: Tensor<1, f32>, on_false: Tensor<1, f32>| {
            cond.where_cond(&on_true, &on_false)
        },
    )
    .arg(gen_cond)
    .arg(gen_true)
    .arg(gen_false)
    .equal_to_resolved_with_device(
        async |cond: Vec<f32>, on_true: Vec<f32>, on_false: Vec<f32>, device: Device| {
            let out = where_cond1(&cond, &on_true, &on_false);
            Tensor::from_slice(&device, LARGE_SHAPE_1D, &out)
        },
    )
    .compare_with(approx_compare::<1, f32>(0.0))
    .runs(3)
    .await
    .unwrap();
}

#[tokio::test]
async fn where_cond_fuzzed() {
    const SHAPE_1D: [usize; 1] = [2048];
    // Condition: values in -1..1 so we get a mix of positive and non-positive
    let gen_cond = FuzzGenerator::<1, f32>::new(SHAPE_1D)
        .with_seed(10)
        .with_distribution(Uniform::new(-1.0, 1.0).unwrap());
    let gen_true = FuzzGenerator::<1, f32>::new(SHAPE_1D).with_seed(20);
    let gen_false = FuzzGenerator::<1, f32>::new(SHAPE_1D).with_seed(30);

    fusor_conformance::assert(
        async |cond: Tensor<1, f32>, on_true: Tensor<1, f32>, on_false: Tensor<1, f32>| {
            cond.gt_scalar(0.0).where_cond(&on_true, &on_false)
        },
    )
    .arg(gen_cond)
    .arg(gen_true)
    .arg(gen_false)
    .equal_to_resolved_with_device(
        async |cond: Vec<f32>, on_true: Vec<f32>, on_false: Vec<f32>, device: Device| {
            let out: Vec<f32> = cond
                .iter()
                .zip(on_true.iter())
                .zip(on_false.iter())
                .map(|((c, t), f)| if *c > 0.0 { *t } else { *f })
                .collect();
            Tensor::from_slice(&device, SHAPE_1D, &out)
        },
    )
    .compare_with(approx_compare::<1, f32>(1e-6))
    .runs(3)
    .await
    .unwrap();
}

#[tokio::test]
async fn large_tensor_unary_ops_fuzzed() {
    const LARGE_SHAPE: [usize; 2] = [45, 45];

    // sin
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.sin().to_concrete())
        .arg(FuzzGenerator::<2, f32>::new(LARGE_SHAPE).with_seed(1))
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(&device, &unary_map2(&v, f32::sin))
        })
        .compare_with(approx_compare::<2, f32>(1e-5))
        .runs(3)
        .await
        .unwrap();

    // cos
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.cos().to_concrete())
        .arg(FuzzGenerator::<2, f32>::new(LARGE_SHAPE).with_seed(2))
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(&device, &unary_map2(&v, f32::cos))
        })
        .compare_with(approx_compare::<2, f32>(1e-5))
        .runs(3)
        .await
        .unwrap();

    // exp (bounded range to avoid overflow)
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.exp().to_concrete())
        .arg(
            FuzzGenerator::<2, f32>::new(LARGE_SHAPE)
                .with_seed(3)
                .with_distribution(Uniform::new(-5.0, 5.0).unwrap()),
        )
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(&device, &unary_map2(&v, f32::exp))
        })
        .compare_with(approx_compare::<2, f32>(1e-3))
        .runs(3)
        .await
        .unwrap();

    // sqrt (positive only)
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.sqrt().to_concrete())
        .arg(
            FuzzGenerator::<2, f32>::new(LARGE_SHAPE)
                .with_seed(4)
                .with_positive(),
        )
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(&device, &unary_map2(&v, f32::sqrt))
        })
        .compare_with(approx_compare::<2, f32>(1e-5))
        .runs(3)
        .await
        .unwrap();

    // neg
    fusor_conformance::assert(async |x: Tensor<2, f32>| (-x).to_concrete())
        .arg(FuzzGenerator::<2, f32>::new(LARGE_SHAPE).with_seed(5))
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(&device, &unary_map2(&v, |x| -x))
        })
        .compare_with(approx_compare::<2, f32>(1e-6))
        .runs(3)
        .await
        .unwrap();

    // abs
    fusor_conformance::assert(async |x: Tensor<2, f32>| x.abs().to_concrete())
        .arg(FuzzGenerator::<2, f32>::new(LARGE_SHAPE).with_seed(6))
        .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
            Tensor::new(&device, &unary_map2(&v, f32::abs))
        })
        .compare_with(approx_compare::<2, f32>(1e-6))
        .runs(3)
        .await
        .unwrap();
}

#[tokio::test]
async fn tanh_exact_saturation_at_large_magnitudes() {
    // The default fuzz distribution rarely produces |x| > 4, but `tanh_exact`
    // must remain accurate when the input saturates the function. This pins
    // the saturation regression that the per-op test
    // `core/src/element_wise.rs::test_tanh_exact_large_values` used to cover.
    const SHAPE: [usize; 2] = [3, 2];
    let positive: Vec<Vec<f32>> = (0..SHAPE[0])
        .map(|row| {
            (0..SHAPE[1])
                .map(|col| 4.0 + (row * SHAPE[1] + col) as f32 * 1.5)
                .collect()
        })
        .collect();
    let negative: Vec<Vec<f32>> = positive
        .iter()
        .map(|row| row.iter().map(|x| -x).collect())
        .collect();

    for samples in [&positive, &negative] {
        let flat: Vec<f32> = samples.iter().flatten().copied().collect();
        let expected: Vec<f32> = flat.iter().map(|x| x.tanh()).collect();
        for device in fusor_conformance::available_devices().await {
            let input = Tensor::from_slice(&device, SHAPE, &flat);
            let actual = input.tanh_exact().to_concrete();
            let expected_tensor = Tensor::from_slice(&device, SHAPE, &expected);
            fusor_conformance::approx_eq(&actual, &expected_tensor, 1e-6)
                .await
                .unwrap();
        }
    }
}
