//! Elementwise op conformance cases.

use crate::common::{binary_map2, unary_map2, where_cond1, where_cond2};
use fusor::{Device, Tensor};
use fusor_conformance::{
    AssertionCase, AssertionCases, FuzzGenerator, approx_compare, approx_or_relative_compare,
};
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

fn eq_f32(l: f32, r: f32) -> bool {
    l == r
}

fn lt_f32(l: f32, r: f32) -> bool {
    l < r
}

fn lte_f32(l: f32, r: f32) -> bool {
    l <= r
}

fn gt_f32(l: f32, r: f32) -> bool {
    l > r
}

fn gte_f32(l: f32, r: f32) -> bool {
    l >= r
}

macro_rules! fuzz_unary {
    ($assertions:ident, $name:ident, $gen:expr, $op:expr, $ref_fn:expr, $tol:expr) => {
        fuzz_unary_compare!(
            $assertions,
            $name,
            $gen,
            $op,
            $ref_fn,
            approx_compare::<2, f32>($tol)
        );
    };
}

macro_rules! fuzz_unary_compare {
    ($assertions:ident, $name:ident, $gen:expr, $op:expr, $ref_fn:expr, $compare:expr) => {
        $assertions.push(
            fusor_conformance::assert(async |x: Tensor<2, f32>| ($op)(x))
                .arg($gen)
                .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
                    Tensor::new(&device, &unary_map2(&v, $ref_fn))
                })
                .compare_with($compare)
                .runs(3)
                .into_case(concat!("elementwise_ops::", stringify!($name))),
        );
    };
}

macro_rules! fuzz_unary_native_math {
    ($assertions:ident, $name:ident, $gen:expr, $op:expr, $ref_fn:expr, $abs_tol:expr, $rel_tol:expr) => {
        fuzz_unary_compare!(
            $assertions,
            $name,
            $gen,
            $op,
            $ref_fn,
            approx_or_relative_compare::<2>($abs_tol, $rel_tol)
        );
    };
}

macro_rules! fuzz_broadcast_binary {
    ($assertions:ident, $name:ident, $gen_a:expr, $gen_b:expr, $op:expr, $ref_fn:expr, $tol:expr) => {
        $assertions.push(
            fusor_conformance::assert($op)
                .arg($gen_a.clone())
                .arg($gen_b.clone())
                .equal_to_resolved_with_device(
                    async |a: Vec<Vec<f32>>, b: Vec<f32>, device: Device| {
                        let out = crate::common::broadcast_binary_2d_1d(&a, &b, $ref_fn);
                        Tensor::new(&device, &out)
                    },
                )
                .compare_with(approx_compare::<2, f32>($tol))
                .runs(3)
                .into_case(concat!(
                    "elementwise_ops::broadcast_binary::",
                    stringify!($name)
                )),
        );
    };
}

macro_rules! fuzz_same_shape_binary {
    ($assertions:ident, $name:ident, $gen_a:expr, $gen_b:expr, $op:expr, $ref_fn:expr, $tol:expr) => {
        $assertions.push(
            fusor_conformance::assert($op)
                .arg($gen_a.clone())
                .arg($gen_b.clone())
                .equal_to_resolved_with_device(
                    async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
                        let out = binary_map2(&a, &b, $ref_fn);
                        Tensor::new(&device, &out)
                    },
                )
                .compare_with(approx_compare::<2, f32>($tol))
                .runs(3)
                .into_case(concat!(
                    "elementwise_ops::same_shape_binary::",
                    stringify!($name)
                )),
        );
    };
}

macro_rules! fuzz_scalar_compare {
    ($assertions:ident, $name:ident, $fuzz:expr, $op:expr, $ref_fn:expr $(,)?) => {
        $assertions.push(
            fusor_conformance::assert($op)
                .arg($fuzz.clone())
                .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
                    Tensor::new(
                        &device,
                        &crate::common::compare_scalar_map2(&v, 0.25, $ref_fn),
                    )
                })
                .compare_with(approx_compare::<2, f32>(0.0))
                .runs(3)
                .into_case(concat!(
                    "elementwise_ops::scalar_compare::",
                    stringify!($name)
                )),
        );
    };
}

macro_rules! fuzz_tensor_compare {
    ($assertions:ident, $name:ident, $fuzz:expr, $gen_b:expr, $op:expr, $ref_fn:expr) => {
        $assertions.push(
            fusor_conformance::assert($op)
                .arg($fuzz.clone())
                .arg($gen_b.clone())
                .equal_to_resolved_with_device(
                    async |a: Vec<Vec<f32>>, b: Vec<Vec<f32>>, device: Device| {
                        Tensor::new(
                            &device,
                            &crate::common::compare_tensor_map2(&a, &b, $ref_fn),
                        )
                    },
                )
                .compare_with(approx_compare::<2, f32>(0.0))
                .devices([Device::Cpu])
                .runs(3)
                .into_case(concat!(
                    "elementwise_ops::tensor_compare::",
                    stringify!($name)
                )),
        );
    };
}

macro_rules! fuzz_large_binary_1d {
    ($assertions:ident, $name:ident, $shape:expr, $gen_a:expr, $gen_b:expr, $op:expr, $ref_fn:expr) => {
        $assertions.push(
            fusor_conformance::assert($op)
                .arg($gen_a.clone())
                .arg($gen_b.clone())
                .equal_to_resolved_with_device(async |a: Vec<f32>, b: Vec<f32>, device: Device| {
                    let out: Vec<f32> = a
                        .iter()
                        .copied()
                        .zip(b.iter().copied())
                        .map($ref_fn)
                        .collect();
                    Tensor::from_slice(&device, $shape, &out)
                })
                .compare_with(approx_compare::<1, f32>(1e-6))
                .runs(3)
                .into_case(concat!(
                    "elementwise_ops::large_binary_1d::",
                    stringify!($name)
                )),
        );
    };
}

pub fn unary_math_ops_match_host_reference() -> AssertionCases {
    let mut assertions = AssertionCases::new();

    // abs
    fuzz_unary!(
        assertions,
        _abs,
        signed(),
        |x: Tensor<2, f32>| x.abs().to_concrete(),
        f32::abs,
        1e-6
    );

    // Native GPU transcendental functions use backend-specific approximations.
    // Compare them with absolute-or-relative tolerances so values near zero
    // still stay tight while larger outputs are not judged by absolute error
    // alone. This avoids making Windows WARP match libm polynomial choices
    // exactly while still catching algorithmic regressions.

    // exp
    fuzz_unary_native_math!(
        assertions,
        _exp,
        signed(),
        |x: Tensor<2, f32>| x.exp().to_concrete(),
        f32::exp,
        1e-3,
        3e-4
    );

    // exp2
    fuzz_unary_native_math!(
        assertions,
        _exp2,
        signed(),
        |x: Tensor<2, f32>| x.exp2().to_concrete(),
        f32::exp2,
        1e-3,
        3e-4
    );

    // sin
    fuzz_unary_native_math!(
        assertions,
        _sin,
        signed(),
        |x: Tensor<2, f32>| x.sin().to_concrete(),
        f32::sin,
        1e-4,
        3e-4
    );

    // cos
    fuzz_unary_native_math!(
        assertions,
        _cos,
        signed(),
        |x: Tensor<2, f32>| x.cos().to_concrete(),
        f32::cos,
        1e-4,
        3e-4
    );

    // tan
    fuzz_unary_native_math!(
        assertions,
        _tan,
        tan_domain(),
        |x: Tensor<2, f32>| x.tan().to_concrete(),
        f32::tan,
        1e-4,
        3e-4
    );

    // tanh
    fuzz_unary_native_math!(
        assertions,
        _tanh,
        signed(),
        |x: Tensor<2, f32>| x.tanh().to_concrete(),
        f32::tanh,
        5e-4,
        5e-4
    );

    // atan
    fuzz_unary_native_math!(
        assertions,
        _atan,
        signed(),
        |x: Tensor<2, f32>| x.atan().to_concrete(),
        f32::atan,
        1e-4,
        3e-4
    );

    // sinh
    fuzz_unary_native_math!(
        assertions,
        _sinh,
        signed(),
        |x: Tensor<2, f32>| x.sinh().to_concrete(),
        f32::sinh,
        1e-4,
        3e-4
    );

    // cosh
    fuzz_unary_native_math!(
        assertions,
        _cosh,
        signed(),
        |x: Tensor<2, f32>| x.cosh().to_concrete(),
        f32::cosh,
        1e-4,
        5e-4
    );

    // asinh
    fuzz_unary_native_math!(
        assertions,
        _asinh,
        signed(),
        |x: Tensor<2, f32>| x.asinh().to_concrete(),
        f32::asinh,
        1e-4,
        3e-4
    );

    // approximate_exp
    fuzz_unary!(
        assertions,
        _approx_exp,
        approx_exp_domain(),
        |x: Tensor<2, f32>| x.approximate_exp(),
        f32::exp,
        6e-2
    );

    // less_approximate_exp
    fuzz_unary!(
        assertions,
        _less_approx_exp,
        approx_exp_domain(),
        |x: Tensor<2, f32>| x.less_approximate_exp(),
        f32::exp,
        1.5e-2
    );

    // tanh_exact
    fuzz_unary_native_math!(
        assertions,
        _tanh_exact,
        signed(),
        |x: Tensor<2, f32>| x.tanh_exact(),
        f32::tanh,
        5e-4,
        5e-4
    );

    // sqr
    fuzz_unary!(
        assertions,
        _sqr,
        signed(),
        |x: Tensor<2, f32>| x.sqr().to_concrete(),
        |v: f32| v * v,
        1e-5
    );
    assertions
}

pub fn restricted_domain_unary_ops_match_host_reference() -> AssertionCases {
    let mut assertions = AssertionCases::new();

    // sqrt
    fuzz_unary_native_math!(
        assertions,
        _sqrt,
        positive(),
        |x: Tensor<2, f32>| x.sqrt().to_concrete(),
        f32::sqrt,
        1e-4,
        3e-4
    );

    // log
    fuzz_unary_native_math!(
        assertions,
        _log,
        positive(),
        |x: Tensor<2, f32>| x.log().to_concrete(),
        f32::ln,
        1e-4,
        3e-4
    );

    // log2
    fuzz_unary_native_math!(
        assertions,
        _log2,
        positive(),
        |x: Tensor<2, f32>| x.log2().to_concrete(),
        f32::log2,
        1e-4,
        3e-4
    );

    // Inverse trig / hyperbolic functions diverge from libm by ~2e-4 on the
    // lavapipe/llvmpipe Linux CI adapter when the `unit()` distribution
    // samples close to the asymptotes (asin'(±0.95) ≈ 3.2, amplifying
    // input ULP error). 1e-3 covers the observed lavapipe drift while
    // still catching algorithmic regressions (which would be orders of
    // magnitude larger). macOS Metal stays well under 1e-5.

    // asin
    fuzz_unary_native_math!(
        assertions,
        _asin,
        unit(),
        |x: Tensor<2, f32>| x.asin().to_concrete(),
        f32::asin,
        1e-3,
        3e-4
    );

    // acos
    fuzz_unary_native_math!(
        assertions,
        _acos,
        unit(),
        |x: Tensor<2, f32>| x.acos().to_concrete(),
        f32::acos,
        1e-3,
        3e-4
    );

    // atanh
    fuzz_unary_native_math!(
        assertions,
        _atanh,
        unit(),
        |x: Tensor<2, f32>| x.atanh().to_concrete(),
        f32::atanh,
        1e-3,
        3e-4
    );

    // acosh
    fuzz_unary_native_math!(
        assertions,
        _acosh,
        acosh_domain(),
        |x: Tensor<2, f32>| x.acosh().to_concrete(),
        f32::acosh,
        1e-3,
        3e-4
    );
    assertions
}

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn gelu(v: f32) -> f32 {
    0.5 * v * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (v + 0.044_715 * v.powi(3))).tanh())
}

pub fn activation_and_scalar_ops_match_host_reference() -> AssertionCases {
    let mut assertions = AssertionCases::new();

    // relu
    fuzz_unary!(
        assertions,
        _relu,
        signed(),
        |x: Tensor<2, f32>| x.relu(),
        |v: f32| v.max(0.0),
        1e-6
    );

    // Windows WARP's sigmoid/exp path can leave small non-zero tails.
    fuzz_unary!(
        assertions,
        _silu,
        signed(),
        |x: Tensor<2, f32>| x.silu(),
        silu,
        2e-3
    );

    // gelu compounds the same WARP tanh drift in its polynomial approximation.
    fuzz_unary!(
        assertions,
        _gelu,
        signed(),
        |x: Tensor<2, f32>| x.gelu(),
        gelu,
        1e-3
    );

    // add_scalar
    fuzz_unary!(
        assertions,
        _add_scalar,
        signed(),
        |x: Tensor<2, f32>| x.add_scalar(1.25),
        |v: f32| v + 1.25,
        1e-6
    );

    // sub_scalar
    fuzz_unary!(
        assertions,
        _sub_scalar,
        signed(),
        |x: Tensor<2, f32>| x.sub_scalar(1.25),
        |v: f32| v - 1.25,
        1e-6
    );

    // mul_scalar
    fuzz_unary!(
        assertions,
        _mul_scalar,
        signed(),
        |x: Tensor<2, f32>| x.mul_scalar(-1.5),
        |v: f32| v * -1.5,
        1e-5
    );

    // div_scalar
    fuzz_unary!(
        assertions,
        _div_scalar,
        signed(),
        |x: Tensor<2, f32>| x.div_scalar(2.0),
        |v: f32| v / 2.0,
        1e-6
    );

    // pow_scalar
    fuzz_unary_native_math!(
        assertions,
        _pow_scalar,
        positive(),
        |x: Tensor<2, f32>| x.pow_scalar(2.5),
        |v: f32| v.powf(2.5),
        1e-4,
        3e-4
    );

    // max_scalar
    fuzz_unary!(
        assertions,
        _max_scalar,
        signed(),
        |x: Tensor<2, f32>| x.max_scalar(0.4),
        |v: f32| v.max(0.4),
        1e-6
    );

    // min_scalar
    fuzz_unary!(
        assertions,
        _min_scalar,
        signed(),
        |x: Tensor<2, f32>| x.min_scalar(-0.4),
        |v: f32| v.min(-0.4),
        1e-6
    );

    // clamp
    fuzz_unary!(
        assertions,
        _clamp,
        signed(),
        |x: Tensor<2, f32>| x.clamp(-0.75, 0.75),
        |v: f32| v.clamp(-0.75, 0.75),
        1e-6
    );
    assertions
}

pub fn binary_ops_match_host_reference() -> AssertionCases {
    let gen_a = positive();
    let gen_b_1d = FuzzGenerator::<1, f32>::new([SHAPE[1]])
        .with_seed(110)
        .with_distribution(Uniform::new(0.5, 2.5).unwrap());
    let gen_b_2d = FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(111)
        .with_distribution(Uniform::new(0.5, 2.5).unwrap());
    let mut assertions = AssertionCases::new();

    // add broadcast 2d + 1d
    fuzz_broadcast_binary!(
        assertions,
        add,
        gen_a,
        gen_b_1d,
        async |a: Tensor<2, f32>, b: Tensor<1, f32>| a.add_::<1, 2, _>(&b),
        |l, r| l + r,
        1e-6
    );

    // sub broadcast
    fuzz_broadcast_binary!(
        assertions,
        sub,
        gen_a,
        gen_b_1d,
        async |a: Tensor<2, f32>, b: Tensor<1, f32>| a.sub_::<1, 2, _>(&b),
        |l, r| l - r,
        1e-6
    );

    // mul broadcast
    fuzz_broadcast_binary!(
        assertions,
        mul,
        gen_a,
        gen_b_1d,
        async |a: Tensor<2, f32>, b: Tensor<1, f32>| a.mul_::<1, 2, _>(&b),
        |l, r| l * r,
        1e-6
    );

    // div broadcast
    fuzz_broadcast_binary!(
        assertions,
        div,
        gen_a,
        gen_b_1d,
        async |a: Tensor<2, f32>, b: Tensor<1, f32>| a.div_::<1, 2, _>(&b),
        |l, r| l / r,
        1e-6
    );

    // pow elementwise 2d
    assertions.push(
        fusor_conformance::assert(async |a: Tensor<2, f32>, b: Tensor<2, f32>| {
            a.pow_::<2, 2, _>(&b)
        })
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
        .into_case("elementwise_ops::binary_ops_match_host_reference::pow"),
    );
    assertions
}

pub fn comparison_and_conditionals_match_expected() -> AssertionCases {
    let fuzz = signed();
    let mut assertions = AssertionCases::new();

    // eq_scalar
    fuzz_scalar_compare!(
        assertions,
        eq_scalar,
        fuzz,
        async |x: Tensor<2, f32>| x.eq_scalar(0.25),
        eq_f32
    );

    // lt_scalar
    fuzz_scalar_compare!(
        assertions,
        lt_scalar,
        fuzz,
        async |x: Tensor<2, f32>| x.lt_scalar(0.25),
        lt_f32
    );

    // lte_scalar
    fuzz_scalar_compare!(
        assertions,
        lte_scalar,
        fuzz,
        async |x: Tensor<2, f32>| x.lte_scalar(0.25),
        lte_f32
    );

    // gt_scalar
    fuzz_scalar_compare!(
        assertions,
        gt_scalar,
        fuzz,
        async |x: Tensor<2, f32>| x.gt_scalar(0.25),
        gt_f32
    );

    // gte_scalar
    fuzz_scalar_compare!(
        assertions,
        gte_scalar,
        fuzz,
        async |x: Tensor<2, f32>| x.gte_scalar(0.25),
        gte_f32
    );

    let gen_b = FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(120)
        .with_distribution(Uniform::new(-2.0, 2.0).unwrap());

    // eq_tensor
    fuzz_tensor_compare!(
        assertions,
        eq_tensor,
        fuzz,
        gen_b,
        async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.eq_tensor(&b),
        eq_f32
    );

    // lt_tensor
    fuzz_tensor_compare!(
        assertions,
        lt_tensor,
        fuzz,
        gen_b,
        async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.lt_tensor(&b),
        lt_f32
    );

    // lte_tensor
    fuzz_tensor_compare!(
        assertions,
        lte_tensor,
        fuzz,
        gen_b,
        async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.lte_tensor(&b),
        lte_f32
    );

    // gt_tensor
    fuzz_tensor_compare!(
        assertions,
        gt_tensor,
        fuzz,
        gen_b,
        async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.gt_tensor(&b),
        gt_f32
    );

    // gte_tensor
    fuzz_tensor_compare!(
        assertions,
        gte_tensor,
        fuzz,
        gen_b,
        async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.gte_tensor(&b),
        gte_f32
    );

    // where_cond
    let gen_cond = FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(130)
        .with_distribution(Uniform::new(-1.0, 1.0).unwrap());
    let gen_on_true = FuzzGenerator::<2, f32>::new(SHAPE).with_seed(131);
    let gen_on_false = FuzzGenerator::<2, f32>::new(SHAPE).with_seed(132);

    assertions.push(
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
        .into_case("elementwise_ops::comparison_and_conditionals_match_expected::where_cond"),
    );
    assertions
}

pub fn same_shape_binary_ops_match_host_reference() -> AssertionCases {
    let gen_a = FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(1)
        .with_distribution(Uniform::new(0.1, 3.0).unwrap());
    let gen_b = FuzzGenerator::<2, f32>::new(SHAPE)
        .with_seed(2)
        .with_distribution(Uniform::new(0.1, 3.0).unwrap());
    let mut assertions = AssertionCases::new();

    // add
    fuzz_same_shape_binary!(
        assertions,
        add,
        gen_a,
        gen_b,
        async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.add_::<2, 2, _>(&b),
        |l, r| l + r,
        1e-5
    );

    // sub
    fuzz_same_shape_binary!(
        assertions,
        sub,
        gen_a,
        gen_b,
        async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.sub_::<2, 2, _>(&b),
        |l, r| l - r,
        1e-5
    );

    // mul
    fuzz_same_shape_binary!(
        assertions,
        mul,
        gen_a,
        gen_b,
        async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.mul_::<2, 2, _>(&b),
        |l, r| l * r,
        1e-5
    );

    // div
    fuzz_same_shape_binary!(
        assertions,
        div,
        gen_a,
        gen_b,
        async |a: Tensor<2, f32>, b: Tensor<2, f32>| a.div_::<2, 2, _>(&b),
        |l, r| l / r,
        1e-5
    );
    assertions
}

pub fn large_tensor_binary_and_conditional_regressions() -> AssertionCases {
    const LARGE_SHAPE_1D: [usize; 1] = [2048];
    let mut assertions = AssertionCases::new();

    let gen_binary_a = FuzzGenerator::<1, f32>::new(LARGE_SHAPE_1D)
        .with_seed(140)
        .with_distribution(Uniform::new(0.5, 4.0).unwrap());
    let gen_binary_b = FuzzGenerator::<1, f32>::new(LARGE_SHAPE_1D)
        .with_seed(141)
        .with_distribution(Uniform::new(0.5, 4.0).unwrap());

    fuzz_large_binary_1d!(
        assertions,
        add,
        LARGE_SHAPE_1D,
        gen_binary_a,
        gen_binary_b,
        async |a: Tensor<1, f32>, b: Tensor<1, f32>| a.add_::<1, 1, _>(&b),
        |(l, r)| l + r
    );
    fuzz_large_binary_1d!(
        assertions,
        sub,
        LARGE_SHAPE_1D,
        gen_binary_a,
        gen_binary_b,
        async |a: Tensor<1, f32>, b: Tensor<1, f32>| a.sub_::<1, 1, _>(&b),
        |(l, r)| l - r
    );
    fuzz_large_binary_1d!(
        assertions,
        mul,
        LARGE_SHAPE_1D,
        gen_binary_a,
        gen_binary_b,
        async |a: Tensor<1, f32>, b: Tensor<1, f32>| a.mul_::<1, 1, _>(&b),
        |(l, r)| l * r
    );
    fuzz_large_binary_1d!(
        assertions,
        div,
        LARGE_SHAPE_1D,
        gen_binary_a,
        gen_binary_b,
        async |a: Tensor<1, f32>, b: Tensor<1, f32>| a.div_::<1, 1, _>(&b),
        |(l, r)| l / r
    );

    let gen_cmp_a = FuzzGenerator::<1, f32>::new(LARGE_SHAPE_1D)
        .with_seed(142)
        .with_distribution(Uniform::new(-10.0, 10.0).unwrap());
    let gen_cmp_b = FuzzGenerator::<1, f32>::new(LARGE_SHAPE_1D)
        .with_seed(143)
        .with_distribution(Uniform::new(-10.0, 10.0).unwrap());

    assertions.push(
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
            .into_case(
                "elementwise_ops::large_tensor_binary_and_conditional_regressions::lt_tensor",
            ),
    );

    let gen_cond = FuzzGenerator::<1, f32>::new(LARGE_SHAPE_1D)
        .with_seed(144)
        .with_distribution(Uniform::new(-1.0, 1.0).unwrap());
    let gen_true = FuzzGenerator::<1, f32>::new(LARGE_SHAPE_1D).with_seed(145);
    let gen_false = FuzzGenerator::<1, f32>::new(LARGE_SHAPE_1D).with_seed(146);

    assertions.push(
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
        .into_case("elementwise_ops::large_tensor_binary_and_conditional_regressions::where_cond"),
    );
    assertions
}

pub fn where_cond_fuzzed() -> AssertionCase {
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
    .into_case("elementwise_ops::where_cond_fuzzed")
}

pub fn large_tensor_unary_ops_fuzzed() -> AssertionCases {
    const LARGE_SHAPE: [usize; 2] = [45, 45];
    let mut assertions = AssertionCases::new();

    // sin
    assertions.push(
        fusor_conformance::assert(async |x: Tensor<2, f32>| x.sin().to_concrete())
            .arg(FuzzGenerator::<2, f32>::new(LARGE_SHAPE).with_seed(1))
            .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &unary_map2(&v, f32::sin))
            })
            .compare_with(approx_or_relative_compare::<2>(1e-4, 3e-4))
            .runs(3)
            .into_case("elementwise_ops::large_tensor_unary_ops_fuzzed::sin"),
    );

    // cos
    assertions.push(
        fusor_conformance::assert(async |x: Tensor<2, f32>| x.cos().to_concrete())
            .arg(FuzzGenerator::<2, f32>::new(LARGE_SHAPE).with_seed(2))
            .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &unary_map2(&v, f32::cos))
            })
            .compare_with(approx_or_relative_compare::<2>(1e-4, 3e-4))
            .runs(3)
            .into_case("elementwise_ops::large_tensor_unary_ops_fuzzed::cos"),
    );

    // exp (bounded range to avoid overflow)
    assertions.push(
        fusor_conformance::assert(async |x: Tensor<2, f32>| x.exp().to_concrete())
            .arg(
                FuzzGenerator::<2, f32>::new(LARGE_SHAPE)
                    .with_seed(3)
                    .with_distribution(Uniform::new(-5.0, 5.0).unwrap()),
            )
            .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &unary_map2(&v, f32::exp))
            })
            .compare_with(approx_or_relative_compare::<2>(1e-3, 3e-4))
            .runs(3)
            .into_case("elementwise_ops::large_tensor_unary_ops_fuzzed::exp"),
    );

    // sqrt (positive only)
    assertions.push(
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
            .into_case("elementwise_ops::large_tensor_unary_ops_fuzzed::sqrt"),
    );

    // neg
    assertions.push(
        fusor_conformance::assert(async |x: Tensor<2, f32>| (-x).to_concrete())
            .arg(FuzzGenerator::<2, f32>::new(LARGE_SHAPE).with_seed(5))
            .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &unary_map2(&v, |x| -x))
            })
            .compare_with(approx_compare::<2, f32>(1e-6))
            .runs(3)
            .into_case("elementwise_ops::large_tensor_unary_ops_fuzzed::neg"),
    );

    // abs
    assertions.push(
        fusor_conformance::assert(async |x: Tensor<2, f32>| x.abs().to_concrete())
            .arg(FuzzGenerator::<2, f32>::new(LARGE_SHAPE).with_seed(6))
            .equal_to_resolved_with_device(async |v: Vec<Vec<f32>>, device: Device| {
                Tensor::new(&device, &unary_map2(&v, f32::abs))
            })
            .compare_with(approx_compare::<2, f32>(1e-6))
            .runs(3)
            .into_case("elementwise_ops::large_tensor_unary_ops_fuzzed::abs"),
    );
    assertions
}

pub fn tanh_exact_saturation_at_large_magnitudes() -> AssertionCases {
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

    let mut assertions = AssertionCases::new();
    for (kind, samples) in [("positive", &positive), ("negative", &negative)] {
        let flat: Vec<f32> = samples.iter().flatten().copied().collect();
        let expected: Vec<f32> = flat.iter().map(|x| x.tanh()).collect();
        assertions.push(
            fusor_conformance::assert(async |input: Tensor<2, f32>| {
                input.tanh_exact().to_concrete()
            })
            .arg(move |device: &Device| Tensor::from_slice(device, SHAPE, &flat))
            .equal_to(move |input: Tensor<2, f32>| {
                let expected = expected.clone();
                async move { Tensor::from_slice(&input.device(), SHAPE, &expected) }
            })
            .compare_with(approx_compare::<2, f32>(2e-4))
            .runs(1)
            .into_case(format!(
                "elementwise_ops::tanh_exact_saturation_at_large_magnitudes::{kind}"
            )),
        );
    }
    assertions
}
