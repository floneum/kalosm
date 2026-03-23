mod common;

use common::{
    assert_approx_cpu, assert_approx_devices, binary_map2, broadcast_binary_2d_1d,
    compare_scalar_map2, compare_tensor_map2, gelu, silu, unary_map2, where_cond2,
};
use fusor::Tensor;

fn signed_input() -> Vec<Vec<f32>> {
    vec![
        vec![-1.25, -0.5, 0.0, 0.75, 1.5],
        vec![2.0, -2.5, 3.0, -3.5, 0.25],
        vec![-0.1, 0.2, -0.3, 0.4, -0.5],
        vec![1.1, -1.2, 1.3, -1.4, 1.5],
    ]
}

fn tan_input() -> Vec<Vec<f32>> {
    vec![
        vec![-0.6, -0.3, 0.0, 0.2, 0.5],
        vec![0.1, -0.2, 0.3, -0.4, 0.6],
        vec![-0.55, 0.15, -0.25, 0.35, -0.45],
        vec![0.05, -0.15, 0.25, -0.35, 0.45],
    ]
}

fn approx_exp_input() -> Vec<Vec<f32>> {
    vec![
        vec![-0.4, -0.2, 0.0, 0.2, 0.4],
        vec![0.1, -0.1, 0.3, -0.3, 0.5],
        vec![-0.45, 0.15, -0.25, 0.35, -0.15],
        vec![0.05, -0.05, 0.25, -0.35, 0.45],
    ]
}

fn positive_input() -> Vec<Vec<f32>> {
    vec![
        vec![0.25, 0.5, 1.0, 1.5, 2.0],
        vec![2.5, 3.0, 0.75, 1.25, 2.25],
        vec![0.1, 0.2, 0.4, 0.8, 1.6],
        vec![1.1, 1.3, 1.7, 2.1, 2.9],
    ]
}

fn unit_input() -> Vec<Vec<f32>> {
    vec![
        vec![-0.9, -0.5, 0.0, 0.4, 0.8],
        vec![0.1, -0.2, 0.3, -0.4, 0.5],
        vec![-0.75, 0.6, -0.45, 0.2, -0.1],
        vec![0.7, -0.6, 0.55, -0.35, 0.25],
    ]
}

fn acosh_input() -> Vec<Vec<f32>> {
    vec![
        vec![1.1, 1.25, 1.5, 2.0, 2.5],
        vec![3.0, 1.75, 1.2, 2.2, 2.8],
        vec![1.05, 1.3, 1.6, 1.9, 2.4],
        vec![3.2, 2.1, 1.4, 1.8, 2.6],
    ]
}

fn rhs_1d() -> Vec<f32> {
    vec![0.5, 1.0, 1.5, 2.0, 2.5]
}

fn rhs_2d() -> Vec<Vec<f32>> {
    vec![
        vec![0.5, 1.0, 1.5, 2.0, 2.5],
        vec![1.25, 1.5, 1.75, 2.25, 2.5],
        vec![0.75, 1.25, 1.5, 1.75, 2.0],
        vec![1.1, 1.2, 1.3, 1.4, 1.5],
    ]
}

fn cond_2d() -> Vec<Vec<f32>> {
    vec![
        vec![1.0, 0.0, -2.0, 0.0, 3.0],
        vec![0.0, 4.0, 0.0, -5.0, 6.0],
        vec![7.0, 0.0, 0.0, 8.0, 0.0],
        vec![0.0, -9.0, 10.0, 0.0, 11.0],
    ]
}

fn on_true_2d() -> Vec<Vec<f32>> {
    vec![
        vec![10.0, 11.0, 12.0, 13.0, 14.0],
        vec![15.0, 16.0, 17.0, 18.0, 19.0],
        vec![20.0, 21.0, 22.0, 23.0, 24.0],
        vec![25.0, 26.0, 27.0, 28.0, 29.0],
    ]
}

fn on_false_2d() -> Vec<Vec<f32>> {
    vec![
        vec![-10.0, -11.0, -12.0, -13.0, -14.0],
        vec![-15.0, -16.0, -17.0, -18.0, -19.0],
        vec![-20.0, -21.0, -22.0, -23.0, -24.0],
        vec![-25.0, -26.0, -27.0, -28.0, -29.0],
    ]
}

#[tokio::test]
async fn unary_math_ops_match_host_reference() {
    let signed = signed_input();
    let tan_domain = tan_input();
    let approx_exp_domain = approx_exp_input();

    assert_approx_devices(
        |device| Tensor::new(device, &signed).abs().to_concrete(),
        |device| Tensor::new(device, &unary_map2(&signed, f32::abs)),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).exp().to_concrete(),
        |device| Tensor::new(device, &unary_map2(&signed, f32::exp)),
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).exp2().to_concrete(),
        |device| Tensor::new(device, &unary_map2(&signed, f32::exp2)),
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).sin().to_concrete(),
        |device| Tensor::new(device, &unary_map2(&signed, f32::sin)),
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).cos().to_concrete(),
        |device| Tensor::new(device, &unary_map2(&signed, f32::cos)),
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &tan_domain).tan().to_concrete(),
        |device| Tensor::new(device, &unary_map2(&tan_domain, f32::tan)),
        1e-4,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).tanh().to_concrete(),
        |device| Tensor::new(device, &unary_map2(&signed, f32::tanh)),
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).atan().to_concrete(),
        |device| Tensor::new(device, &unary_map2(&signed, f32::atan)),
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).sinh().to_concrete(),
        |device| Tensor::new(device, &unary_map2(&signed, f32::sinh)),
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).cosh().to_concrete(),
        |device| Tensor::new(device, &unary_map2(&signed, f32::cosh)),
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).asinh().to_concrete(),
        |device| Tensor::new(device, &unary_map2(&signed, f32::asinh)),
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &approx_exp_domain).approximate_exp(),
        |device| Tensor::new(device, &unary_map2(&approx_exp_domain, f32::exp)),
        6e-2,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &approx_exp_domain).less_approximate_exp(),
        |device| Tensor::new(device, &unary_map2(&approx_exp_domain, f32::exp)),
        1.5e-2,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).tanh_exact(),
        |device| Tensor::new(device, &unary_map2(&signed, f32::tanh)),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).sqr().to_concrete(),
        |device| Tensor::new(device, &unary_map2(&signed, |value| value * value)),
        1e-6,
    )
    .await;
}

#[tokio::test]
async fn restricted_domain_unary_ops_match_host_reference() {
    let positive = positive_input();
    let unit = unit_input();
    let acosh_domain = acosh_input();

    assert_approx_devices(
        |device| Tensor::new(device, &positive).sqrt().to_concrete(),
        |device| Tensor::new(device, &unary_map2(&positive, f32::sqrt)),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &positive).log().to_concrete(),
        |device| Tensor::new(device, &unary_map2(&positive, f32::ln)),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &positive).log2().to_concrete(),
        |device| Tensor::new(device, &unary_map2(&positive, f32::log2)),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &unit).asin().to_concrete(),
        |device| Tensor::new(device, &unary_map2(&unit, f32::asin)),
        2e-5,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &unit).acos().to_concrete(),
        |device| Tensor::new(device, &unary_map2(&unit, f32::acos)),
        2e-5,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &unit).atanh().to_concrete(),
        |device| Tensor::new(device, &unary_map2(&unit, f32::atanh)),
        2e-5,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &acosh_domain).acosh().to_concrete(),
        |device| Tensor::new(device, &unary_map2(&acosh_domain, f32::acosh)),
        2e-5,
    )
    .await;
}

#[tokio::test]
async fn activation_and_scalar_ops_match_host_reference() {
    let signed = signed_input();
    let positive = positive_input();

    assert_approx_devices(
        |device| Tensor::new(device, &signed).relu(),
        |device| Tensor::new(device, &unary_map2(&signed, |value| value.max(0.0))),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).silu(),
        |device| Tensor::new(device, &unary_map2(&signed, silu)),
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).gelu(),
        |device| Tensor::new(device, &unary_map2(&signed, gelu)),
        1e-4,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).add_scalar(1.25),
        |device| Tensor::new(device, &unary_map2(&signed, |value| value + 1.25)),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).sub_scalar(1.25),
        |device| Tensor::new(device, &unary_map2(&signed, |value| value - 1.25)),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).mul_scalar(-1.5),
        |device| Tensor::new(device, &unary_map2(&signed, |value| value * -1.5)),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).div_scalar(2.0),
        |device| Tensor::new(device, &unary_map2(&signed, |value| value / 2.0)),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &positive).pow_scalar(2.5),
        |device| Tensor::new(device, &unary_map2(&positive, |value| value.powf(2.5))),
        1e-5,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &positive).pow_elementwise(2.5),
        |device| Tensor::new(device, &positive).pow_scalar(2.5),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).max_scalar(0.4),
        |device| Tensor::new(device, &unary_map2(&signed, |value| value.max(0.4))),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).max_elementwise(0.4),
        |device| Tensor::new(device, &signed).max_scalar(0.4),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).min_scalar(-0.4),
        |device| Tensor::new(device, &unary_map2(&signed, |value| value.min(-0.4))),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).min_elementwise(-0.4),
        |device| Tensor::new(device, &signed).min_scalar(-0.4),
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).clamp(-0.75, 0.75),
        |device| {
            Tensor::new(
                device,
                &unary_map2(&signed, |value| value.clamp(-0.75, 0.75)),
            )
        },
        1e-6,
    )
    .await;
}

#[tokio::test]
async fn binary_ops_match_host_reference() {
    let lhs = positive_input();
    let rhs_broadcast = rhs_1d();
    let rhs_same_shape = rhs_2d();

    assert_approx_devices(
        |device| {
            let lhs: Tensor<2, f32> = Tensor::new(device, &lhs);
            let rhs: Tensor<1, f32> = Tensor::new(device, &rhs_broadcast);
            lhs.add_::<1, 2, _>(&rhs)
        },
        |device| {
            Tensor::new(
                device,
                &broadcast_binary_2d_1d(&lhs, &rhs_broadcast, |l, r| l + r),
            )
        },
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| {
            let lhs: Tensor<2, f32> = Tensor::new(device, &lhs);
            let rhs: Tensor<1, f32> = Tensor::new(device, &rhs_broadcast);
            lhs.sub_::<1, 2, _>(&rhs)
        },
        |device| {
            Tensor::new(
                device,
                &broadcast_binary_2d_1d(&lhs, &rhs_broadcast, |l, r| l - r),
            )
        },
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| {
            let lhs: Tensor<2, f32> = Tensor::new(device, &lhs);
            let rhs: Tensor<1, f32> = Tensor::new(device, &rhs_broadcast);
            lhs.mul_::<1, 2, _>(&rhs)
        },
        |device| {
            Tensor::new(
                device,
                &broadcast_binary_2d_1d(&lhs, &rhs_broadcast, |l, r| l * r),
            )
        },
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| {
            let lhs: Tensor<2, f32> = Tensor::new(device, &lhs);
            let rhs: Tensor<1, f32> = Tensor::new(device, &rhs_broadcast);
            lhs.div_::<1, 2, _>(&rhs)
        },
        |device| {
            Tensor::new(
                device,
                &broadcast_binary_2d_1d(&lhs, &rhs_broadcast, |l, r| l / r),
            )
        },
        1e-6,
    )
    .await;

    assert_approx_devices(
        |device| {
            let lhs: Tensor<2, f32> = Tensor::new(device, &lhs);
            let rhs: Tensor<2, f32> = Tensor::new(device, &rhs_same_shape);
            lhs.pow_::<2, 2, _>(&rhs)
        },
        |device| {
            Tensor::new(
                device,
                &binary_map2(&lhs, &rhs_same_shape, |l, r| l.powf(r)),
            )
        },
        1e-5,
    )
    .await;
}

#[tokio::test]
async fn comparison_and_conditionals_match_expected() {
    let signed = signed_input();
    let rhs_same_shape = rhs_2d();
    let cond = cond_2d();
    let on_true = on_true_2d();
    let on_false = on_false_2d();

    assert_approx_devices(
        |device| Tensor::new(device, &signed).eq_scalar(0.25),
        |device| Tensor::new(device, &compare_scalar_map2(&signed, 0.25, |l, r| l == r)),
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).lt_scalar(0.25),
        |device| Tensor::new(device, &compare_scalar_map2(&signed, 0.25, |l, r| l < r)),
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).lte_scalar(0.25),
        |device| Tensor::new(device, &compare_scalar_map2(&signed, 0.25, |l, r| l <= r)),
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).gt_scalar(0.25),
        |device| Tensor::new(device, &compare_scalar_map2(&signed, 0.25, |l, r| l > r)),
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).gte_scalar(0.25),
        |device| Tensor::new(device, &compare_scalar_map2(&signed, 0.25, |l, r| l >= r)),
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).eq(0.25),
        |device| Tensor::new(device, &signed).eq_scalar(0.25),
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).lt(0.25),
        |device| Tensor::new(device, &signed).lt_scalar(0.25),
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).lte(0.25),
        |device| Tensor::new(device, &signed).lte_scalar(0.25),
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).mt(0.25),
        |device| Tensor::new(device, &signed).gt_scalar(0.25),
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| Tensor::new(device, &signed).mte(0.25),
        |device| Tensor::new(device, &signed).gte_scalar(0.25),
        0.0,
    )
    .await;

    assert_approx_cpu(
        |device| {
            let lhs: Tensor<2, f32> = Tensor::new(device, &signed);
            let rhs: Tensor<2, f32> = Tensor::new(device, &rhs_same_shape);
            lhs.eq_tensor(&rhs)
        },
        |device| {
            Tensor::new(
                device,
                &compare_tensor_map2(&signed, &rhs_same_shape, |l, r| l == r),
            )
        },
        0.0,
    )
    .await;

    assert_approx_cpu(
        |device| {
            let lhs: Tensor<2, f32> = Tensor::new(device, &signed);
            let rhs: Tensor<2, f32> = Tensor::new(device, &rhs_same_shape);
            lhs.lt_tensor(&rhs)
        },
        |device| {
            Tensor::new(
                device,
                &compare_tensor_map2(&signed, &rhs_same_shape, |l, r| l < r),
            )
        },
        0.0,
    )
    .await;

    assert_approx_cpu(
        |device| {
            let lhs: Tensor<2, f32> = Tensor::new(device, &signed);
            let rhs: Tensor<2, f32> = Tensor::new(device, &rhs_same_shape);
            lhs.lte_tensor(&rhs)
        },
        |device| {
            Tensor::new(
                device,
                &compare_tensor_map2(&signed, &rhs_same_shape, |l, r| l <= r),
            )
        },
        0.0,
    )
    .await;

    assert_approx_cpu(
        |device| {
            let lhs: Tensor<2, f32> = Tensor::new(device, &signed);
            let rhs: Tensor<2, f32> = Tensor::new(device, &rhs_same_shape);
            lhs.gt_tensor(&rhs)
        },
        |device| {
            Tensor::new(
                device,
                &compare_tensor_map2(&signed, &rhs_same_shape, |l, r| l > r),
            )
        },
        0.0,
    )
    .await;

    assert_approx_cpu(
        |device| {
            let lhs: Tensor<2, f32> = Tensor::new(device, &signed);
            let rhs: Tensor<2, f32> = Tensor::new(device, &rhs_same_shape);
            lhs.gte_tensor(&rhs)
        },
        |device| {
            Tensor::new(
                device,
                &compare_tensor_map2(&signed, &rhs_same_shape, |l, r| l >= r),
            )
        },
        0.0,
    )
    .await;

    assert_approx_devices(
        |device| {
            let cond: Tensor<2, f32> = Tensor::new(device, &cond);
            let on_true: Tensor<2, f32> = Tensor::new(device, &on_true);
            let on_false: Tensor<2, f32> = Tensor::new(device, &on_false);
            cond.where_cond(&on_true, &on_false)
        },
        |device| Tensor::new(device, &where_cond2(&cond, &on_true, &on_false)),
        0.0,
    )
    .await;
}
