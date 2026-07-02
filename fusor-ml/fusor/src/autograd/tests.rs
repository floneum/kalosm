use super::*;
use crate::{Layout, ToVec1, ToVec2};
use fusor_types::StrideSpec;

fn assert_close(left: f32, right: f32) {
    assert!((left - right).abs() < 1e-3, "expected {right}, got {left}");
}

fn assert_slice_close(left: &[f32], right: &[f32]) {
    assert_eq!(left.len(), right.len(), "slice lengths differ");
    for (index, (left, right)) in left.iter().zip(right.iter()).enumerate() {
        assert!(
            (*left - *right).abs() < 1e-3,
            "mismatch at index {index}: expected {right}, got {left}",
        );
    }
}

async fn flatten<const R: usize>(tensor: RawTensor<R, f32>) -> Vec<f32> {
    let elements = tensor.shape().into_iter().product();
    tensor
        .reshape([elements])
        .as_slice()
        .await
        .unwrap()
        .to_vec1()
}

async fn finite_difference_gradient<const R: usize, F>(
    device: &Device,
    shape: [usize; R],
    data: &[f32],
    loss: &F,
) -> Vec<f32>
where
    F: Fn(&Graph, Tensor<R>) -> Tensor<0>,
{
    let epsilon = 1e-2f32;
    let mut numeric = Vec::with_capacity(data.len());
    for index in 0..data.len() {
        let mut perturbed = data.to_vec();
        perturbed[index] = data[index] + epsilon;
        let graph = Graph::new();
        let plus = loss(&graph, Tensor::from_slice(&graph, device, shape, &perturbed));
        let plus = plus.raw().to_scalar().await.unwrap();
        perturbed[index] = data[index] - epsilon;
        let graph = Graph::new();
        let minus = loss(&graph, Tensor::from_slice(&graph, device, shape, &perturbed));
        let minus = minus.raw().to_scalar().await.unwrap();
        numeric.push((plus - minus) / (2.0 * epsilon));
    }
    numeric
}

async fn assert_gradient_matches_finite_difference<const R: usize, F>(
    device: &Device,
    shape: [usize; R],
    data: &[f32],
    loss: F,
) where
    F: Fn(&Graph, Tensor<R>) -> Tensor<0>,
{
    let graph = Graph::new();
    let input = Tensor::from_slice(&graph, device, shape, data);
    let gradients = loss(&graph, input.clone()).backward().unwrap();
    let analytic = flatten(gradients.get(&input).unwrap()).await;
    let numeric = finite_difference_gradient(device, shape, data, &loss).await;
    assert_eq!(analytic.len(), numeric.len(), "gradient lengths differ");
    for (index, (analytic, numeric)) in analytic.iter().zip(numeric.iter()).enumerate() {
        assert!(
            (analytic - numeric).abs() < 1e-2 + 1e-2 * numeric.abs(),
            "gradient mismatch at index {index}: analytic {analytic}, finite difference {numeric}",
        );
    }
}

#[tokio::test]
async fn test_backward_squared_sum_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();

    let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
    let loss = x.sqr().sum();
    let gradients = loss.backward().unwrap();
    let dx = gradients
        .get(&x)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(dx[0], 2.0);
    assert_close(dx[1], 4.0);
    assert_close(dx[2], 6.0);
}

#[tokio::test]
async fn test_autograd_silu_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, -2.0, 0.5]);

    let output = x.silu();
    let values = output.raw().clone().as_slice().await.unwrap().to_vec1();

    let expected = [1.0f32, -2.0, 0.5].map(|v| v / (1.0 + (-v).exp()));
    for (value, expected) in values.iter().zip(expected) {
        assert_close(*value, expected);
    }

    let gradients = output.sum().backward().unwrap();
    let dx = gradients.get(&x).unwrap().as_slice().await.unwrap().to_vec1();

    let expected_grads = [1.0f32, -2.0, 0.5].map(|v| {
        let sigmoid = 1.0 / (1.0 + (-v).exp());
        sigmoid * (1.0 + v * (1.0 - sigmoid))
    });
    for (value, expected) in dx.iter().zip(expected_grads) {
        assert_close(*value, expected);
    }

    assert_gradient_matches_finite_difference(&device, [3], &[1.0, -2.0, 0.5], |_, x| {
        x.silu().sum()
    })
    .await;
}

#[tokio::test]
async fn test_autograd_gelu_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, -2.0, 0.5]);

    let output = x.gelu();
    let values = output.raw().clone().as_slice().await.unwrap().to_vec1();

    let expected = [1.0f32, -2.0, 0.5].map(|v| {
        0.5 * v
            * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (v + 0.044_715 * v.powi(3))).tanh())
    });
    for (value, expected) in values.iter().zip(expected) {
        assert_close(*value, expected);
    }

    let gradients = output.sum().backward().unwrap();
    let dx = gradients.get(&x).unwrap().as_slice().await.unwrap().to_vec1();

    let expected_grads = [1.0f32, -2.0, 0.5].map(|v| {
        let scale = (2.0 / std::f32::consts::PI).sqrt();
        let inner = scale * (v + 0.044_715 * v.powi(3));
        let tanh = inner.tanh();
        let dinner = scale * (1.0 + 3.0 * 0.044_715 * v * v);
        0.5 * (1.0 + tanh) + 0.5 * v * (1.0 - tanh * tanh) * dinner
    });
    for (value, expected) in dx.iter().zip(expected_grads) {
        assert_close(*value, expected);
    }

    assert_gradient_matches_finite_difference(&device, [3], &[1.0, -2.0, 0.5], |_, x| {
        x.gelu().sum()
    })
    .await;
}

#[tokio::test]
async fn test_backward_where_cond_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let condition: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 0.0, -2.0]);
    let on_true: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 3.0, 4.0]);
    let on_false: Tensor<1> = Tensor::new(&graph, &device, &[10.0f32, 20.0, 30.0]);

    let output = condition.where_cond(&on_true, &on_false);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.flatten_all().sum().backward().unwrap();

    let dcondition = gradients
        .get(&condition)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();
    let dtrue = gradients
        .get(&on_true)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();
    let dfalse = gradients
        .get(&on_false)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![2.0, 20.0, 4.0]);
    assert_eq!(dcondition, vec![0.0, 0.0, 0.0]);
    assert_eq!(dtrue, vec![1.0, 0.0, 1.0]);
    assert_eq!(dfalse, vec![0.0, 1.0, 0.0]);
}

#[tokio::test]
async fn test_backward_index_select_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let indices = RawTensor::from_slice(&device, [3], &[2u32, 0, 2]);

    let output = input.index_select(1, &indices);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
    let gradients = output.sum(1).sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    assert_eq!(output_values, vec![vec![3.0, 1.0, 3.0], vec![6.0, 4.0, 6.0]]);
    assert_eq!(dinput, vec![vec![1.0, 0.0, 2.0], vec![1.0, 0.0, 2.0]]);
}

#[tokio::test]
async fn test_backward_slice_assign_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> = Tensor::new(
        &graph,
        &device,
        &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    );
    let value: Tensor<2> = Tensor::new(&graph, &device, &[[10.0f32, 11.0], [12.0, 13.0]]);

    let output = input.slice_assign([0..2, 1..3], &value);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
    let gradients = output.sum(1).sum().backward().unwrap();

    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();
    let dvalue = gradients
        .get(&value)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    assert_eq!(
        output_values,
        vec![vec![1.0, 10.0, 11.0], vec![4.0, 12.0, 13.0], vec![7.0, 8.0, 9.0]]
    );
    assert_eq!(
        dinput,
        vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]]
    );
    assert_eq!(dvalue, vec![vec![1.0, 1.0], vec![1.0, 1.0]]);
}

#[tokio::test]
async fn test_backward_expand_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> = Tensor::new(&graph, &device, &[[2.0f32, 3.0, 4.0]]);

    let output = input.expand([2, 3]);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
    let gradients = output.sum(1).sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    assert_eq!(output_values, vec![vec![2.0, 3.0, 4.0], vec![2.0, 3.0, 4.0]]);
    assert_eq!(dinput, vec![vec![2.0, 2.0, 2.0]]);
}

#[tokio::test]
async fn test_backward_flatten_all_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);

    let output = input.flatten_all();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.flatten_all().sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    assert_eq!(output_values, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(dinput, vec![vec![1.0, 1.0], vec![1.0, 1.0]]);
}

#[tokio::test]
async fn test_backward_flatten_last_n_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<3> = Tensor::new(
        &graph,
        &device,
        &[
            [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
    );

    let output = input.flatten_last_n::<1, 2>();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
    let gradients = output.sum(1).sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .reshape([2, 6])
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    assert_eq!(
        output_values,
        vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]
    );
    assert_eq!(
        dinput,
        vec![vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    );
}

#[tokio::test]
async fn test_backward_flatten_first_n_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<3> = Tensor::new(
        &graph,
        &device,
        &[
            [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
        ],
    );

    let output = input.flatten_first_n::<1, 2>();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
    let gradients = output.sum(1).sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .reshape([4, 3])
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    assert_eq!(
        output_values,
        vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![10.0, 11.0, 12.0]
        ]
    );
    assert_eq!(
        dinput,
        vec![
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0]
        ]
    );
}

#[tokio::test]
async fn test_backward_narrow_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> = Tensor::new(
        &graph,
        &device,
        &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    );

    let output = input.narrow(1usize, 1, 2);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
    let gradients = output.sum(1).sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    assert_eq!(output_values, vec![vec![2.0, 3.0], vec![5.0, 6.0], vec![8.0, 9.0]]);
    assert_eq!(
        dinput,
        vec![vec![0.0, 1.0, 1.0], vec![0.0, 1.0, 1.0], vec![0.0, 1.0, 1.0]]
    );
}

#[tokio::test]
async fn test_backward_repeat_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);

    let output = input.repeat([2, 3]);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
    let gradients = output.sum(1).sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    assert_eq!(
        output_values,
        vec![
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            vec![3.0, 4.0, 3.0, 4.0, 3.0, 4.0],
            vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            vec![3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
        ]
    );
    assert_eq!(dinput, vec![vec![6.0, 6.0], vec![6.0, 6.0]]);
}

#[tokio::test]
async fn test_backward_resize_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> = Tensor::new(
        &graph,
        &device,
        &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
    );

    let output = input.resize([2, 2]);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
    let gradients = output.sum(1).sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    assert_eq!(output_values, vec![vec![1.0, 2.0], vec![4.0, 5.0]]);
    assert_eq!(
        dinput,
        vec![vec![1.0, 1.0, 0.0], vec![1.0, 1.0, 0.0], vec![0.0, 0.0, 0.0]]
    );
}

#[tokio::test]
async fn test_backward_restride_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0, 4.0]);

    let output = input.restride([StrideSpec::dim(0, 2), StrideSpec::dim(0, 3)]);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
    let gradients = output.sum(1).sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]]);
    assert_eq!(dinput, vec![1.0, 2.0, 2.0, 1.0]);
}

#[tokio::test]
async fn test_backward_restride_strided_overlap_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> =
        Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    let output = input.restride([StrideSpec::dim_with(0, 3, 2), StrideSpec::dim(0, 3)]);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
    let gradients = output.sum(1).sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(
        output_values,
        vec![vec![1.0, 2.0, 3.0], vec![3.0, 4.0, 5.0], vec![5.0, 6.0, 7.0]]
    );
    assert_eq!(dinput, vec![1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 0.0]);
}

#[tokio::test]
async fn test_backward_restride_layout_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0, 4.0, 5.0]);
    let layout = Layout::contiguous(&[5]).restride(&[
        StrideSpec::dim(0, 2).with_offset(1),
        StrideSpec::dim(0, 2),
    ]);

    let output = input.restride_layout(layout);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
    let gradients = output.sum(1).sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![vec![2.0, 3.0], vec![3.0, 4.0]]);
    assert_eq!(dinput, vec![0.0, 1.0, 2.0, 1.0, 0.0]);
}

#[tokio::test]
async fn test_backward_squeeze_dims_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<4> = Tensor::new(
        &graph,
        &device,
        &[[[[1.0f32], [2.0], [3.0]]], [[[4.0], [5.0], [6.0]]]],
    );

    let output = input.squeeze_dims::<2, 2>([1, 3]);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
    let gradients = output.sum(1).sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .reshape([2, 3])
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    assert_eq!(output_values, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    assert_eq!(dinput, vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
}

#[tokio::test]
async fn test_backward_unsqueeze_dims_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let output = input.unsqueeze_dims::<2, 4>([0, 2]);
    let output_values = output
        .raw()
        .clone()
        .reshape([2, 3])
        .as_slice()
        .await
        .unwrap()
        .to_vec2();
    let gradients = output.sum(3).sum(2).sum(1).sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    assert_eq!(output.shape(), [1, 2, 1, 3]);
    assert_eq!(output_values, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    assert_eq!(dinput, vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
}

#[tokio::test]
async fn test_backward_max_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 5.0, 5.0], [4.0, 2.0, 0.0]]);

    let output = input.max::<1>(1);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    assert_eq!(output_values, vec![5.0, 4.0]);
    assert_eq!(dinput, vec![vec![0.0, 0.5, 0.5], vec![1.0, 0.0, 0.0]]);
}

#[tokio::test]
async fn test_backward_min_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 1.0, 5.0], [4.0, 2.0, 0.0]]);

    let output = input.min::<1>(1);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    assert_eq!(output_values, vec![1.0, 0.0]);
    assert_eq!(dinput, vec![vec![0.5, 0.5, 0.0], vec![0.0, 0.0, 1.0]]);
}

#[tokio::test]
async fn test_backward_mean_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let output = input.mean::<1>(1);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    assert_eq!(output_values, vec![2.0, 5.0]);
    assert_eq!(
        dinput,
        vec![
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        ]
    );
}

#[tokio::test]
async fn test_backward_product_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> =
        Tensor::new(&graph, &device, &[[2.0f32, 3.0, 4.0], [5.0, 0.0, 7.0], [0.0, 0.0, 9.0]]);

    let output = input.product::<1>(1);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    assert_eq!(output_values, vec![24.0, 0.0, 0.0]);
    assert_eq!(
        dinput,
        vec![vec![12.0, 8.0, 6.0], vec![0.0, 35.0, 0.0], vec![0.0, 0.0, 0.0]]
    );
}

#[tokio::test]
async fn test_backward_product_keepdim_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> = Tensor::new(&graph, &device, &[[2.0f32, 3.0, 4.0], [5.0, 0.0, 7.0]]);

    let output = input.product_keepdim::<1>(1);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
    let gradients = output.sum(1).sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    assert_eq!(output_values, vec![vec![24.0], vec![0.0]]);
    assert_eq!(dinput, vec![vec![12.0, 8.0, 6.0], vec![0.0, 35.0, 0.0]]);
}

#[tokio::test]
async fn test_backward_var_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let output = input.var::<1>(1);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    assert_eq!(output_values, vec![2.0 / 3.0, 2.0 / 3.0]);
    assert_eq!(
        dinput,
        vec![
            vec![-2.0 / 3.0, 0.0, 2.0 / 3.0],
            vec![-2.0 / 3.0, 0.0, 2.0 / 3.0]
        ]
    );
}

#[tokio::test]
async fn test_backward_var_keepdim_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let output = input.var_keepdim::<1>(1);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
    let gradients = output.sum(1).sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    assert_eq!(output_values, vec![vec![2.0 / 3.0], vec![2.0 / 3.0]]);
    assert_eq!(
        dinput,
        vec![
            vec![-2.0 / 3.0, 0.0, 2.0 / 3.0],
            vec![-2.0 / 3.0, 0.0, 2.0 / 3.0]
        ]
    );
}

#[tokio::test]
async fn test_backward_clamp_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[-1.0f32, 0.0, 2.0, 5.0]);

    let output = input.clamp(0.0, 3.0);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![0.0, 0.0, 2.0, 3.0]);
    assert_eq!(dinput, vec![0.0, 0.0, 1.0, 0.0]);
}

#[tokio::test]
async fn test_backward_eq_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 1.0]);

    let output = input.eq(1.0);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![1.0, 0.0, 1.0]);
    assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_backward_eq_scalar_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[3.0f32, 2.0, 3.0]);

    let output = input.eq_scalar(3.0);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![1.0, 0.0, 1.0]);
    assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_backward_eq_tensor_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
    let rhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 0.0, 3.0]);

    let output = lhs.eq_tensor(&rhs);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dlhs = gradients
        .get(&lhs)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();
    let drhs = gradients
        .get(&rhs)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![1.0, 0.0, 1.0]);
    assert_eq!(dlhs, vec![0.0, 0.0, 0.0]);
    assert_eq!(drhs, vec![0.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_backward_gt_scalar_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

    let output = input.gt_scalar(2.0);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![0.0, 0.0, 1.0]);
    assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_backward_gt_tensor_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 3.0]);
    let rhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 1.0, 3.0]);

    let output = lhs.gt_tensor(&rhs);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dlhs = gradients
        .get(&lhs)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();
    let drhs = gradients
        .get(&rhs)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![0.0, 1.0, 0.0]);
    assert_eq!(dlhs, vec![0.0, 0.0, 0.0]);
    assert_eq!(drhs, vec![0.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_backward_gte_scalar_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

    let output = input.gte_scalar(2.0);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![0.0, 1.0, 1.0]);
    assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_backward_gte_tensor_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 3.0]);
    let rhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 4.0, 2.0]);

    let output = lhs.gte_tensor(&rhs);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dlhs = gradients
        .get(&lhs)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();
    let drhs = gradients
        .get(&rhs)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![0.0, 1.0, 1.0]);
    assert_eq!(dlhs, vec![0.0, 0.0, 0.0]);
    assert_eq!(drhs, vec![0.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_backward_lt_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

    let output = input.lt(2.0);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![1.0, 0.0, 0.0]);
    assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_backward_lt_scalar_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

    let output = input.lt_scalar(3.0);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![1.0, 1.0, 0.0]);
    assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_backward_lt_tensor_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
    let rhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 1.0, 3.0]);

    let output = lhs.lt_tensor(&rhs);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dlhs = gradients
        .get(&lhs)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();
    let drhs = gradients
        .get(&rhs)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![1.0, 0.0, 0.0]);
    assert_eq!(dlhs, vec![0.0, 0.0, 0.0]);
    assert_eq!(drhs, vec![0.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_backward_lte_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

    let output = input.lte(2.0);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![1.0, 1.0, 0.0]);
    assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_backward_lte_scalar_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

    let output = input.lte_scalar(1.0);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![1.0, 0.0, 0.0]);
    assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_backward_lte_tensor_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
    let rhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 2.0, 1.0]);

    let output = lhs.lte_tensor(&rhs);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dlhs = gradients
        .get(&lhs)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();
    let drhs = gradients
        .get(&rhs)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![1.0, 1.0, 0.0]);
    assert_eq!(dlhs, vec![0.0, 0.0, 0.0]);
    assert_eq!(drhs, vec![0.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_backward_max_elementwise_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[-1.0f32, 0.0, 2.0]);

    let output = input.max_elementwise(0.0);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![0.0, 0.0, 2.0]);
    assert_eq!(dinput, vec![0.0, 0.0, 1.0]);
}

#[tokio::test]
async fn test_backward_max_scalar_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

    let output = input.max_scalar(3.0);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![3.0, 4.0, 3.0]);
    assert_eq!(dinput, vec![0.0, 1.0, 0.0]);
}

#[tokio::test]
async fn test_backward_min_elementwise_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

    let output = input.min_elementwise(3.0);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![1.0, 3.0, 2.0]);
    assert_eq!(dinput, vec![1.0, 0.0, 1.0]);
}

#[tokio::test]
async fn test_backward_min_scalar_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

    let output = input.min_scalar(2.0);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![1.0, 2.0, 2.0]);
    assert_eq!(dinput, vec![1.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_backward_mt_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

    let output = input.mt(2.0);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![0.0, 1.0, 0.0]);
    assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_backward_mte_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

    let output = input.mte(2.0);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![0.0, 1.0, 1.0]);
    assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_backward_ne_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

    let output = input.ne(2.0);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![1.0, 1.0, 0.0]);
    assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_backward_ne_scalar_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

    let output = input.ne_scalar(4.0);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![1.0, 0.0, 1.0]);
    assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_backward_ne_tensor_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);
    let rhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 0.0, 3.0]);

    let output = lhs.ne_tensor(&rhs);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dlhs = gradients
        .get(&lhs)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();
    let drhs = gradients
        .get(&rhs)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![0.0, 1.0, 1.0]);
    assert_eq!(dlhs, vec![0.0, 0.0, 0.0]);
    assert_eq!(drhs, vec![0.0, 0.0, 0.0]);
}

#[tokio::test]
async fn test_backward_abs_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[-2.0f32, 0.0, 3.0]);

    let output = input.abs();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_eq!(output_values, vec![2.0, 0.0, 3.0]);
    assert_eq!(dinput, vec![-1.0, 0.0, 1.0]);
}

#[tokio::test]
async fn test_backward_acos_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

    let output = input.acos();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(output_values[0], 0.5f32.acos());
    assert_close(dinput[0], -1.0f32 / (1.0f32 - 0.25f32).sqrt());
}

#[tokio::test]
async fn test_backward_acosh_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32]);

    let output = input.acosh();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(output_values[0], 2.0f32.acosh());
    assert_close(dinput[0], 1.0f32 / ((2.0f32 - 1.0f32).sqrt() * (2.0f32 + 1.0f32).sqrt()));
}

#[tokio::test]
async fn test_backward_approximate_exp_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32]);

    let output = input.approximate_exp();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(output_values[0], 1.0f32.exp());
    assert_close(dinput[0], 1.0f32.exp());
}

#[tokio::test]
async fn test_backward_asin_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

    let output = input.asin();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(output_values[0], 0.5f32.asin());
    assert_close(dinput[0], 1.0f32 / (1.0f32 - 0.25f32).sqrt());
}

#[tokio::test]
async fn test_backward_asinh_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.5f32]);

    let output = input.asinh();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(output_values[0], 1.5f32.asinh());
    assert_close(dinput[0], 1.0f32 / (1.5f32 * 1.5f32 + 1.0f32).sqrt());
}

#[tokio::test]
async fn test_backward_atan_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

    let output = input.atan();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(output_values[0], 0.5f32.atan());
    assert_close(dinput[0], 1.0f32 / (1.0f32 + 0.25f32));
}

#[tokio::test]
async fn test_backward_atanh_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

    let output = input.atanh();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(output_values[0], 0.5f32.atanh());
    assert_close(dinput[0], 1.0f32 / (1.0f32 - 0.25f32));
}

#[tokio::test]
async fn test_backward_cos_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

    let output = input.cos();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(output_values[0], 0.5f32.cos());
    assert_close(dinput[0], -0.5f32.sin());
}

#[tokio::test]
async fn test_backward_cosh_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

    let output = input.cosh();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(output_values[0], 0.5f32.cosh());
    assert_close(dinput[0], 0.5f32.sinh());
}

#[tokio::test]
async fn test_backward_exp2_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32]);

    let output = input.exp2();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(output_values[0], 2.0f32.exp2());
    assert_close(dinput[0], std::f32::consts::LN_2 * 2.0f32.exp2());
}

#[tokio::test]
async fn test_backward_less_approximate_exp_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32]);

    let output = input.less_approximate_exp();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(output_values[0], 1.0f32.exp());
    assert_close(dinput[0], 1.0f32.exp());
}

#[tokio::test]
async fn test_backward_log2_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[4.0f32]);

    let output = input.log2();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(output_values[0], 4.0f32.log2());
    assert_close(dinput[0], 1.0f32 / (4.0f32 * std::f32::consts::LN_2));
}

#[tokio::test]
async fn test_backward_sin_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

    let output = input.sin();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(output_values[0], 0.5f32.sin());
    assert_close(dinput[0], 0.5f32.cos());
}

#[tokio::test]
async fn test_backward_sinh_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

    let output = input.sinh();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(output_values[0], 0.5f32.sinh());
    assert_close(dinput[0], 0.5f32.cosh());
}

#[tokio::test]
async fn test_backward_tan_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

    let output = input.tan();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(output_values[0], 0.5f32.tan());
    assert_close(dinput[0], 1.0f32 / (0.5f32.cos() * 0.5f32.cos()));
}

#[tokio::test]
async fn test_backward_tanh_exact_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

    let output = input.tanh_exact();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(output_values[0], 0.5f32.tanh());
    assert_close(dinput[0], 1.0f32 - 0.5f32.tanh().powi(2));
}

#[tokio::test]
async fn test_cast_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

    let output = input.cast::<half::f16>();
    let output_values = output.as_slice().await.unwrap().to_vec1();

    assert_close(f32::from(output_values[0]), 1.0);
    assert_close(f32::from(output_values[1]), 2.0);
    assert_close(f32::from(output_values[2]), 3.0);
}

#[tokio::test]
async fn test_arange_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();

    let output = Tensor::<1>::arange(&graph, &device, 1.0, 5.0);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();

    assert_eq!(output_values, vec![1.0, 2.0, 3.0, 4.0]);
}

#[tokio::test]
async fn test_arange_step_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();

    let output = Tensor::<1>::arange_step(&graph, &device, 1.0, 6.0, 2.0);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();

    assert_eq!(output_values, vec![1.0, 3.0, 5.0]);
}

#[tokio::test]
async fn test_full_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();

    let output: Tensor<2> = Tensor::full(&graph, &device, [2, 3], 1.5);
    let output_values = output.raw().clone().as_slice().await.unwrap();

    assert_eq!(output_values.shape(), &[2, 3]);
    for row in 0..2 {
        for col in 0..3 {
            assert_close(output_values[[row, col]], 1.5);
        }
    }
}

#[tokio::test]
async fn test_zeros_like_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);

    let output = input.zeros_like();
    let output_values = output.raw().clone().as_slice().await.unwrap();

    assert_eq!(output_values.shape(), &[2, 2]);
    assert_close(output_values[[0, 0]], 0.0);
    assert_close(output_values[[0, 1]], 0.0);
    assert_close(output_values[[1, 0]], 0.0);
    assert_close(output_values[[1, 1]], 0.0);
}

#[tokio::test]
async fn test_from_array_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();

    let output: Tensor<2> = Tensor::from_array(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);
    let output_values = output.raw().clone().as_slice().await.unwrap();

    assert_eq!(output_values.shape(), &[2, 2]);
    assert_close(output_values[[0, 0]], 1.0);
    assert_close(output_values[[0, 1]], 2.0);
    assert_close(output_values[[1, 0]], 3.0);
    assert_close(output_values[[1, 1]], 4.0);
}

#[tokio::test]
async fn test_backward_add_broadcast_api_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0]);
    let rhs: Tensor<2> = Tensor::new(&graph, &device, &[[10.0f32], [20.0]]);

    let output: Tensor<2> = lhs.add_::<2, 2>(&rhs);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
    let gradients = output.flatten_all().sum().backward().unwrap();
    let dlhs = gradients.get(&lhs).unwrap().as_slice().await.unwrap().to_vec1();
    let drhs = gradients.get(&rhs).unwrap().as_slice().await.unwrap().to_vec2();

    assert_close(output_values[0][0], 11.0);
    assert_close(output_values[0][1], 12.0);
    assert_close(output_values[1][0], 21.0);
    assert_close(output_values[1][1], 22.0);
    assert_close(dlhs[0], 2.0);
    assert_close(dlhs[1], 2.0);
    assert_close(drhs[0][0], 2.0);
    assert_close(drhs[1][0], 2.0);
}

#[tokio::test]
async fn test_backward_sub_broadcast_api_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let lhs: Tensor<2> = Tensor::new(&graph, &device, &[[3.0f32], [4.0]]);
    let rhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0]);

    let output: Tensor<2> = lhs.sub_::<1, 2>(&rhs);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
    let gradients = output.flatten_all().sum().backward().unwrap();
    let dlhs = gradients.get(&lhs).unwrap().as_slice().await.unwrap().to_vec2();
    let drhs = gradients.get(&rhs).unwrap().as_slice().await.unwrap().to_vec1();

    assert_close(output_values[0][0], 2.0);
    assert_close(output_values[0][1], 1.0);
    assert_close(output_values[1][0], 3.0);
    assert_close(output_values[1][1], 2.0);
    assert_close(dlhs[0][0], 2.0);
    assert_close(dlhs[1][0], 2.0);
    assert_close(drhs[0], -2.0);
    assert_close(drhs[1], -2.0);
}

#[tokio::test]
async fn test_backward_mul_broadcast_api_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let lhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 3.0]);
    let rhs: Tensor<2> = Tensor::new(&graph, &device, &[[10.0f32], [20.0]]);

    let output: Tensor<2> = lhs.mul_::<2, 2>(&rhs);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
    let gradients = output.flatten_all().sum().backward().unwrap();
    let dlhs = gradients.get(&lhs).unwrap().as_slice().await.unwrap().to_vec1();
    let drhs = gradients.get(&rhs).unwrap().as_slice().await.unwrap().to_vec2();

    assert_close(output_values[0][0], 20.0);
    assert_close(output_values[0][1], 30.0);
    assert_close(output_values[1][0], 40.0);
    assert_close(output_values[1][1], 60.0);
    assert_close(dlhs[0], 30.0);
    assert_close(dlhs[1], 30.0);
    assert_close(drhs[0][0], 5.0);
    assert_close(drhs[1][0], 5.0);
}

#[tokio::test]
async fn test_backward_div_broadcast_api_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let lhs: Tensor<2> = Tensor::new(&graph, &device, &[[10.0f32], [20.0]]);
    let rhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 4.0]);

    let output: Tensor<2> = lhs.div_::<1, 2>(&rhs);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
    let gradients = output.flatten_all().sum().backward().unwrap();
    let dlhs = gradients.get(&lhs).unwrap().as_slice().await.unwrap().to_vec2();
    let drhs = gradients.get(&rhs).unwrap().as_slice().await.unwrap().to_vec1();

    assert_close(output_values[0][0], 5.0);
    assert_close(output_values[0][1], 2.5);
    assert_close(output_values[1][0], 10.0);
    assert_close(output_values[1][1], 5.0);
    assert_close(dlhs[0][0], 0.75);
    assert_close(dlhs[1][0], 0.75);
    assert_close(drhs[0], -7.5);
    assert_close(drhs[1], -1.875);
}

#[tokio::test]
async fn test_backward_pow_broadcast_api_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let lhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 3.0]);
    let rhs: Tensor<2> = Tensor::new(&graph, &device, &[[2.0f32], [1.0]]);

    let output: Tensor<2> = lhs.pow_::<2, 2>(&rhs);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
    let gradients = output.flatten_all().sum().backward().unwrap();
    let dlhs = gradients.get(&lhs).unwrap().as_slice().await.unwrap().to_vec1();
    let drhs = gradients.get(&rhs).unwrap().as_slice().await.unwrap().to_vec2();

    assert_close(output_values[0][0], 4.0);
    assert_close(output_values[0][1], 9.0);
    assert_close(output_values[1][0], 2.0);
    assert_close(output_values[1][1], 3.0);
    assert_close(dlhs[0], 5.0);
    assert_close(dlhs[1], 7.0);
    assert_close(drhs[0][0], 12.660099);
    assert_close(drhs[1][0], 4.6821313);
}

#[tokio::test]
async fn test_backward_chunk_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> = Tensor::new(
        &graph,
        &device,
        &[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
    );

    let chunks = input.chunk(2, 1);
    assert_eq!(chunks.len(), 2);
    let first = chunks[0].raw().clone().as_slice().await.unwrap().to_vec2();
    let second = chunks[1].raw().clone().as_slice().await.unwrap().to_vec2();
    let loss = chunks[0]
        .flatten_all()
        .sum()
        .add(&chunks[1].flatten_all().sum().mul_scalar(2.0));
    let gradients = loss.backward().unwrap();
    let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap().to_vec2();

    assert_close(first[0][0], 1.0);
    assert_close(first[0][1], 2.0);
    assert_close(first[1][0], 5.0);
    assert_close(first[1][1], 6.0);
    assert_close(second[0][0], 3.0);
    assert_close(second[0][1], 4.0);
    assert_close(second[1][0], 7.0);
    assert_close(second[1][1], 8.0);
    assert_close(dinput[0][0], 1.0);
    assert_close(dinput[0][1], 1.0);
    assert_close(dinput[0][2], 2.0);
    assert_close(dinput[0][3], 2.0);
    assert_close(dinput[1][0], 1.0);
    assert_close(dinput[1][1], 1.0);
    assert_close(dinput[1][2], 2.0);
    assert_close(dinput[1][3], 2.0);
}

#[tokio::test]
async fn test_backward_matmul_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let lhs: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);
    let rhs: Tensor<2> = Tensor::new(&graph, &device, &[[5.0f32, 6.0], [7.0, 8.0]]);

    let output = lhs.matmul(&rhs);
    let output_values = output.raw().clone().as_slice().await.unwrap();
    let gradients = output.flatten_all().sum().backward().unwrap();
    let dlhs = gradients.get(&lhs).unwrap().as_slice().await.unwrap();
    let drhs = gradients.get(&rhs).unwrap().as_slice().await.unwrap();

    assert_eq!(output_values.shape(), &[2, 2]);
    assert_close(output_values[[0, 0]], 19.0);
    assert_close(output_values[[0, 1]], 22.0);
    assert_close(output_values[[1, 0]], 43.0);
    assert_close(output_values[[1, 1]], 50.0);

    assert_close(dlhs[[0, 0]], 11.0);
    assert_close(dlhs[[0, 1]], 15.0);
    assert_close(dlhs[[1, 0]], 11.0);
    assert_close(dlhs[[1, 1]], 15.0);

    assert_close(drhs[[0, 0]], 4.0);
    assert_close(drhs[[0, 1]], 4.0);
    assert_close(drhs[[1, 0]], 6.0);
    assert_close(drhs[[1, 1]], 6.0);
}

#[tokio::test]
async fn test_backward_t_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);

    let output = input.t();
    let output_values = output.raw().clone().as_slice().await.unwrap();
    let gradients = output.flatten_all().sum().backward().unwrap();
    let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap();

    assert_eq!(output_values.shape(), &[2, 2]);
    assert_close(output_values[[0, 0]], 1.0);
    assert_close(output_values[[0, 1]], 3.0);
    assert_close(output_values[[1, 0]], 2.0);
    assert_close(output_values[[1, 1]], 4.0);

    assert_close(dinput[[0, 0]], 1.0);
    assert_close(dinput[[0, 1]], 1.0);
    assert_close(dinput[[1, 0]], 1.0);
    assert_close(dinput[[1, 1]], 1.0);
}

#[tokio::test]
async fn test_backward_pool_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<3> = Tensor::new(&graph, &device, &[[[1.0f32, 2.0, 3.0, 4.0]]]);

    let output = input.pool::<1, 4, 5, 4>([(2, 1)], |windowed, axis| windowed.mean::<3>(axis));
    let output_values = output.raw().clone().as_slice().await.unwrap();
    let gradients = output.flatten_all().sum().backward().unwrap();
    let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap();

    assert_eq!(output_values.shape(), &[1, 1, 3]);
    assert_close(output_values[[0, 0, 0]], 1.5);
    assert_close(output_values[[0, 0, 1]], 2.5);
    assert_close(output_values[[0, 0, 2]], 3.5);

    assert_close(dinput[[0, 0, 0]], 0.5);
    assert_close(dinput[[0, 0, 1]], 1.0);
    assert_close(dinput[[0, 0, 2]], 1.0);
    assert_close(dinput[[0, 0, 3]], 0.5);
}

#[tokio::test]
async fn test_backward_pool_max_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<3> = Tensor::new(&graph, &device, &[[[1.0f32, 4.0, 2.0, 3.0]]]);

    let output = input.pool_max::<1, 4, 5, 4>([(2, 1)]);
    let output_values = output.raw().clone().as_slice().await.unwrap();
    let gradients = output.flatten_all().sum().backward().unwrap();
    let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap();

    assert_eq!(output_values.shape(), &[1, 1, 3]);
    assert_close(output_values[[0, 0, 0]], 4.0);
    assert_close(output_values[[0, 0, 1]], 4.0);
    assert_close(output_values[[0, 0, 2]], 3.0);

    assert_close(dinput[[0, 0, 0]], 0.0);
    assert_close(dinput[[0, 0, 1]], 2.0);
    assert_close(dinput[[0, 0, 2]], 0.0);
    assert_close(dinput[[0, 0, 3]], 1.0);
}

#[tokio::test]
async fn test_backward_pool_min_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<3> = Tensor::new(&graph, &device, &[[[1.0f32, 4.0, 2.0, 3.0]]]);

    let output = input.pool_min::<1, 4, 5, 4>([(2, 1)]);
    let output_values = output.raw().clone().as_slice().await.unwrap();
    let gradients = output.flatten_all().sum().backward().unwrap();
    let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap();

    assert_eq!(output_values.shape(), &[1, 1, 3]);
    assert_close(output_values[[0, 0, 0]], 1.0);
    assert_close(output_values[[0, 0, 1]], 2.0);
    assert_close(output_values[[0, 0, 2]], 2.0);

    assert_close(dinput[[0, 0, 0]], 1.0);
    assert_close(dinput[[0, 0, 1]], 0.0);
    assert_close(dinput[[0, 0, 2]], 2.0);
    assert_close(dinput[[0, 0, 3]], 0.0);
}

#[tokio::test]
async fn test_backward_q_mat_mul_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 1.0, 1.0, 1.0]]);
    let weight_bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        .into_iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let weights = crate::QMatrix::from_raw_bytes(
        &device,
        [2, 4],
        &weight_bytes,
        fusor_gguf::GgmlType::F32,
    )
    .unwrap();

    let output = input.q_mat_mul(&weights);
    let output_values = output.raw().clone().as_slice().await.unwrap();
    let gradients = output.flatten_all().sum().backward().unwrap();
    let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap();

    assert_eq!(output_values.shape(), &[1, 2]);
    assert_close(output_values[[0, 0]], 10.0);
    assert_close(output_values[[0, 1]], 26.0);

    assert_close(dinput[[0, 0]], 6.0);
    assert_close(dinput[[0, 1]], 8.0);
    assert_close(dinput[[0, 2]], 10.0);
    assert_close(dinput[[0, 3]], 12.0);
}

#[tokio::test]
async fn test_backward_stack_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let first: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0]);
    let second: Tensor<1> = Tensor::new(&graph, &device, &[3.0f32, 4.0]);

    let output = Tensor::stack::<2>(vec![first.clone(), second.clone()], 0);
    let output_values = output.raw().clone().as_slice().await.unwrap();
    let gradients = output.flatten_all().sum().backward().unwrap();
    let dfirst = gradients.get(&first).unwrap().as_slice().await.unwrap().to_vec1();
    let dsecond = gradients.get(&second).unwrap().as_slice().await.unwrap().to_vec1();

    assert_eq!(output_values.shape(), &[2, 2]);
    assert_close(output_values[[0, 0]], 1.0);
    assert_close(output_values[[0, 1]], 2.0);
    assert_close(output_values[[1, 0]], 3.0);
    assert_close(output_values[[1, 1]], 4.0);

    assert_close(dfirst[0], 1.0);
    assert_close(dfirst[1], 1.0);
    assert_close(dsecond[0], 1.0);
    assert_close(dsecond[1], 1.0);
}

#[tokio::test]
async fn test_backward_rope_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<4> =
        Tensor::new(&graph, &device, &[[[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]]);
    let cos: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 1.0], [1.0, 1.0]]);
    let sin: Tensor<2> = Tensor::new(&graph, &device, &[[0.0f32, 0.0], [0.0, 0.0]]);

    let output = input.rope(&cos, &sin);
    let output_values = output.raw().clone().as_slice().await.unwrap();
    let gradients = output.flatten_all().sum().backward().unwrap();
    let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap();
    let dcos = gradients.get(&cos).unwrap().as_slice().await.unwrap();
    let dsin = gradients.get(&sin).unwrap().as_slice().await.unwrap();

    assert_eq!(output_values.shape(), &[1, 1, 2, 4]);
    assert_close(output_values[[0, 0, 0, 0]], 1.0);
    assert_close(output_values[[0, 0, 0, 1]], 2.0);
    assert_close(output_values[[0, 0, 0, 2]], 3.0);
    assert_close(output_values[[0, 0, 0, 3]], 4.0);
    assert_close(output_values[[0, 0, 1, 0]], 5.0);
    assert_close(output_values[[0, 0, 1, 1]], 6.0);
    assert_close(output_values[[0, 0, 1, 2]], 7.0);
    assert_close(output_values[[0, 0, 1, 3]], 8.0);

    for index in [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 1, 3]] {
        assert_close(dinput[index], 1.0);
    }

    assert_close(dcos[[0, 0]], 4.0);
    assert_close(dcos[[0, 1]], 6.0);
    assert_close(dcos[[1, 0]], 12.0);
    assert_close(dcos[[1, 1]], 14.0);

    assert_close(dsin[[0, 0]], -2.0);
    assert_close(dsin[[0, 1]], -2.0);
    assert_close(dsin[[1, 0]], -2.0);
    assert_close(dsin[[1, 1]], -2.0);
}

#[tokio::test]
async fn test_backward_rope_fused_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<4> =
        Tensor::new(&graph, &device, &[[[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]]);
    let cos: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 1.0], [1.0, 1.0]]);
    let sin: Tensor<2> = Tensor::new(&graph, &device, &[[0.0f32, 0.0], [0.0, 0.0]]);

    let output = input.rope_fused(&cos, &sin);
    let output_values = output.raw().clone().as_slice().await.unwrap();
    let gradients = output.flatten_all().sum().backward().unwrap();
    let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap();
    let dcos = gradients.get(&cos).unwrap().as_slice().await.unwrap();
    let dsin = gradients.get(&sin).unwrap().as_slice().await.unwrap();

    assert_eq!(output_values.shape(), &[1, 1, 2, 4]);
    assert_close(output_values[[0, 0, 0, 0]], 1.0);
    assert_close(output_values[[0, 0, 0, 1]], 2.0);
    assert_close(output_values[[0, 0, 0, 2]], 3.0);
    assert_close(output_values[[0, 0, 0, 3]], 4.0);
    assert_close(output_values[[0, 0, 1, 0]], 5.0);
    assert_close(output_values[[0, 0, 1, 1]], 6.0);
    assert_close(output_values[[0, 0, 1, 2]], 7.0);
    assert_close(output_values[[0, 0, 1, 3]], 8.0);

    for index in [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 1, 3]] {
        assert_close(dinput[index], 1.0);
    }

    assert_close(dcos[[0, 0]], 3.0);
    assert_close(dcos[[0, 1]], 7.0);
    assert_close(dcos[[1, 0]], 11.0);
    assert_close(dcos[[1, 1]], 15.0);

    assert_close(dsin[[0, 0]], -1.0);
    assert_close(dsin[[0, 1]], -1.0);
    assert_close(dsin[[1, 0]], -1.0);
    assert_close(dsin[[1, 1]], -1.0);
}

#[tokio::test]
async fn test_backward_rope_interleaved_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<4> =
        Tensor::new(&graph, &device, &[[[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]]);
    let cos: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 1.0], [1.0, 1.0]]);
    let sin: Tensor<2> = Tensor::new(&graph, &device, &[[0.0f32, 0.0], [0.0, 0.0]]);

    let output = input.rope_interleaved(&cos, &sin);
    let output_values = output.raw().clone().as_slice().await.unwrap();
    let gradients = output.flatten_all().sum().backward().unwrap();
    let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap();
    let dcos = gradients.get(&cos).unwrap().as_slice().await.unwrap();
    let dsin = gradients.get(&sin).unwrap().as_slice().await.unwrap();

    assert_eq!(output_values.shape(), &[1, 1, 2, 4]);
    assert_close(output_values[[0, 0, 0, 0]], 1.0);
    assert_close(output_values[[0, 0, 0, 1]], 2.0);
    assert_close(output_values[[0, 0, 0, 2]], 3.0);
    assert_close(output_values[[0, 0, 0, 3]], 4.0);
    assert_close(output_values[[0, 0, 1, 0]], 5.0);
    assert_close(output_values[[0, 0, 1, 1]], 6.0);
    assert_close(output_values[[0, 0, 1, 2]], 7.0);
    assert_close(output_values[[0, 0, 1, 3]], 8.0);

    for index in [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 1, 3]] {
        assert_close(dinput[index], 1.0);
    }

    assert_close(dcos[[0, 0]], 3.0);
    assert_close(dcos[[0, 1]], 7.0);
    assert_close(dcos[[1, 0]], 11.0);
    assert_close(dcos[[1, 1]], 15.0);

    assert_close(dsin[[0, 0]], -1.0);
    assert_close(dsin[[0, 1]], -1.0);
    assert_close(dsin[[1, 0]], -1.0);
    assert_close(dsin[[1, 1]], -1.0);
}

#[tokio::test]
async fn test_backward_rope_normal_fused_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<4> =
        Tensor::new(&graph, &device, &[[[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]]);
    let cos: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 1.0], [1.0, 1.0]]);
    let sin: Tensor<2> = Tensor::new(&graph, &device, &[[0.0f32, 0.0], [0.0, 0.0]]);

    let output = input.rope_normal_fused(&cos, &sin);
    let output_values = output.raw().clone().as_slice().await.unwrap();
    let gradients = output.flatten_all().sum().backward().unwrap();
    let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap();
    let dcos = gradients.get(&cos).unwrap().as_slice().await.unwrap();
    let dsin = gradients.get(&sin).unwrap().as_slice().await.unwrap();

    assert_eq!(output_values.shape(), &[1, 1, 2, 4]);
    assert_close(output_values[[0, 0, 0, 0]], 1.0);
    assert_close(output_values[[0, 0, 0, 1]], 2.0);
    assert_close(output_values[[0, 0, 0, 2]], 3.0);
    assert_close(output_values[[0, 0, 0, 3]], 4.0);
    assert_close(output_values[[0, 0, 1, 0]], 5.0);
    assert_close(output_values[[0, 0, 1, 1]], 6.0);
    assert_close(output_values[[0, 0, 1, 2]], 7.0);
    assert_close(output_values[[0, 0, 1, 3]], 8.0);

    for index in [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 2], [0, 0, 1, 3]] {
        assert_close(dinput[index], 1.0);
    }

    assert_close(dcos[[0, 0]], 4.0);
    assert_close(dcos[[0, 1]], 6.0);
    assert_close(dcos[[1, 0]], 12.0);
    assert_close(dcos[[1, 1]], 14.0);

    assert_close(dsin[[0, 0]], -2.0);
    assert_close(dsin[[0, 1]], -2.0);
    assert_close(dsin[[1, 0]], -2.0);
    assert_close(dsin[[1, 1]], -2.0);
}

#[tokio::test]
async fn test_backward_pow_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let lhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32]);
    let rhs: Tensor<1> = Tensor::new(&graph, &device, &[3.0f32]);

    let output = lhs.pow(&rhs);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dlhs = gradients
        .get(&lhs)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();
    let drhs = gradients
        .get(&rhs)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(output_values[0], 8.0);
    assert_close(dlhs[0], 12.0);
    assert_close(drhs[0], 8.0 * 2.0f32.ln());
}

#[tokio::test]
async fn test_backward_pow_elementwise_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[3.0f32]);

    let output = input.pow_elementwise(2.0);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(output_values[0], 9.0);
    assert_close(dinput[0], 6.0);
}

#[tokio::test]
async fn test_backward_pow_scalar_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[4.0f32]);

    let output = input.pow_scalar(0.5);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(output_values[0], 2.0);
    assert_close(dinput[0], 0.25);
}

#[tokio::test]
async fn test_autograd_rms_norm_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let weight: Tensor<1> = Tensor::constant_from_raw(
        &graph,
        RawTensor::from_slice(&device, [3], &[1.0f32, 1.0, 1.0]),
    );

    let output = input.rms_norm(&weight, 1e-5);
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();

    let expected = [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].map(|row| {
        let mean_sq = row.iter().map(|value| value * value).sum::<f32>() / row.len() as f32;
        let scale = 1.0 / (mean_sq + 1e-5).sqrt();
        row.map(|value| value * scale)
    });

    for (actual_row, expected_row) in output_values.iter().zip(expected.iter()) {
        for (actual, expected) in actual_row.iter().zip(expected_row.iter()) {
            assert_close(*actual, *expected);
        }
    }

    let gradients = output.sum(1).sum().backward().unwrap();
    let dinput = gradients
        .get(&input)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    // d/dx_j sum_k x_k * (mean(x^2) + eps)^-1/2
    //     = 1/rms - x_j * sum(x) / (n * rms^3)
    let expected_grads = [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].map(|row| {
        let n = row.len() as f32;
        let mean_sq = row.iter().map(|value| value * value).sum::<f32>() / n;
        let rms = (mean_sq + 1e-5).sqrt();
        let sum = row.iter().sum::<f32>();
        row.map(|value| 1.0 / rms - value * sum / (n * rms.powi(3)))
    });
    for (actual_row, expected_row) in dinput.iter().zip(expected_grads.iter()) {
        for (actual, expected) in actual_row.iter().zip(expected_row.iter()) {
            assert_close(*actual, *expected);
        }
    }

    let fd_device = device.clone();
    assert_gradient_matches_finite_difference(
        &device,
        [2, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        move |graph, x| {
            let weight = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [3], &[1.0f32, 1.0, 1.0]),
            );
            x.rms_norm(&weight, 1e-5).sum(1).sum()
        },
    )
    .await;
}

#[tokio::test]
async fn test_backward_matmul_with_broadcast_bias_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();

    let x: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let w: Tensor<2> = Tensor::new(&graph, &device, &[[0.5f32], [1.0], [1.5]]);
    let b: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32]);

    let y = x.mat_mul(&w).add(&b.broadcast_as([2, 1]));
    let loss = y.sum(1).sum();

    let gradients = loss.backward().unwrap();
    let dw = gradients
        .get(&w)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();
    let db = gradients
        .get(&b)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec1();

    assert_close(dw[0][0], 5.0);
    assert_close(dw[1][0], 7.0);
    assert_close(dw[2][0], 9.0);
    assert_close(db[0], 2.0);
}

#[tokio::test]
async fn test_backward_embedding_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();

    let table: Tensor<2> =
        Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    let indices: RawTensor<2, u32> = RawTensor::new(&device, &[[0u32, 2u32]]);
    let embedded = table.embedding(&indices);
    let loss = embedded.sum(2).sum(1).sum();

    let gradients = loss.backward().unwrap();
    let dtable = gradients
        .get(&table)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    assert_close(dtable[0][0], 1.0);
    assert_close(dtable[0][1], 1.0);
    assert_close(dtable[1][0], 0.0);
    assert_close(dtable[1][1], 0.0);
    assert_close(dtable[2][0], 1.0);
    assert_close(dtable[2][1], 1.0);
}

#[tokio::test]
async fn test_backward_gather_last_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();

    let values: Tensor<2> =
        Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let indices: RawTensor<1, u32> = RawTensor::new(&device, &[2u32, 0u32]);
    let gathered = values.gather_last(&indices);
    let loss = gathered.sum();

    let gradients = loss.backward().unwrap();
    let dvalues = gradients
        .get(&values)
        .unwrap()
        .as_slice()
        .await
        .unwrap()
        .to_vec2();

    assert_close(dvalues[0][0], 0.0);
    assert_close(dvalues[0][1], 0.0);
    assert_close(dvalues[0][2], 1.0);
    assert_close(dvalues[1][0], 1.0);
    assert_close(dvalues[1][1], 0.0);
    assert_close(dvalues[1][2], 0.0);
}

#[tokio::test]
async fn test_backward_softmax_last_dim_fused_matches_composite_cpu() {
    let device = Device::cpu();
    let input_data = &[
        [[0.2f32, -0.4, 1.1], [0.5, 0.3, -0.7]],
        [[-1.0, 0.8, 0.6], [0.9, -0.2, 0.1]],
    ];

    let fused_graph = Graph::new();
    let fused_input: Tensor<3> = Tensor::new(&fused_graph, &device, input_data);
    let fused_output = fused_input.softmax_last_dim_fused::<2>();
    let fused_loss = fused_output.sqr().reshape([12]).sum();
    let fused_gradients = fused_loss.backward().unwrap();

    let composite_graph = Graph::new();
    let composite_input: Tensor<3> = Tensor::new(&composite_graph, &device, input_data);
    let composite_output = composite_input.softmax_last_dim::<2>();
    let composite_loss = composite_output.sqr().reshape([12]).sum();
    let composite_gradients = composite_loss.backward().unwrap();

    let fused_output = flatten(fused_output.raw().clone()).await;
    let composite_output = flatten(composite_output.raw().clone()).await;
    let fused_dx = flatten(fused_gradients.get(&fused_input).unwrap()).await;
    let composite_dx = flatten(composite_gradients.get(&composite_input).unwrap()).await;

    assert_slice_close(&fused_output, &composite_output);
    assert_slice_close(&fused_dx, &composite_dx);
}

#[tokio::test]
async fn test_backward_rms_norm_fused_matches_composite_cpu() {
    let device = Device::cpu();
    let input_data = &[
        [[0.3f32, -1.2, 0.7], [1.5, 0.1, -0.8]],
        [[-0.4, 0.9, 1.3], [0.2, -0.6, 0.5]],
    ];
    let weight_data = &[1.0f32, 0.75, 1.25];
    let eps = 1e-5;

    let fused_graph = Graph::new();
    let fused_input: Tensor<3> = Tensor::new(&fused_graph, &device, input_data);
    let fused_weight: Tensor<1> = Tensor::new(&fused_graph, &device, weight_data);
    let fused_output = fused_input.rms_norm_fused_no_bias::<2>(&fused_weight, eps);
    let fused_loss = fused_output.sqr().reshape([12]).sum();
    let fused_gradients = fused_loss.backward().unwrap();

    let composite_graph = Graph::new();
    let composite_input: Tensor<3> = Tensor::new(&composite_graph, &device, input_data);
    let composite_weight: Tensor<1> = Tensor::new(&composite_graph, &device, weight_data);
    let composite_output = composite_input.rms_norm(&composite_weight, eps);
    let composite_loss = composite_output.sqr().reshape([12]).sum();
    let composite_gradients = composite_loss.backward().unwrap();

    let fused_output = flatten(fused_output.raw().clone()).await;
    let composite_output = flatten(composite_output.raw().clone()).await;
    let fused_dx = flatten(fused_gradients.get(&fused_input).unwrap()).await;
    let composite_dx = flatten(composite_gradients.get(&composite_input).unwrap()).await;
    let fused_dw = flatten(fused_gradients.get(&fused_weight).unwrap()).await;
    let composite_dw = flatten(composite_gradients.get(&composite_weight).unwrap()).await;

    assert_slice_close(&fused_output, &composite_output);
    assert_slice_close(&fused_dx, &composite_dx);
    assert_slice_close(&fused_dw, &composite_dw);
}

#[tokio::test]
async fn test_backward_layer_norm_last_dim_fused_matches_composite_cpu() {
    let device = Device::cpu();
    let input_data = &[
        [[0.25f32, -0.5, 1.0], [1.25, -1.5, 0.75]],
        [[-0.8, 0.4, 1.2], [0.6, -0.1, -0.9]],
    ];
    let weight_data = &[1.0f32, 0.9, 1.1];
    let bias_data = &[0.1f32, -0.2, 0.05];
    let eps = 1e-5;

    let fused_graph = Graph::new();
    let fused_input: Tensor<3> = Tensor::new(&fused_graph, &device, input_data);
    let fused_weight: Tensor<1> = Tensor::new(&fused_graph, &device, weight_data);
    let fused_bias: Tensor<1> = Tensor::new(&fused_graph, &device, bias_data);
    let fused_output =
        fused_input.layer_norm_last_dim_fused::<2>(&fused_weight, Some(&fused_bias), eps);
    let fused_loss = fused_output.sqr().reshape([12]).sum();
    let fused_gradients = fused_loss.backward().unwrap();

    let composite_graph = Graph::new();
    let composite_input: Tensor<3> = Tensor::new(&composite_graph, &device, input_data);
    let composite_weight: Tensor<1> = Tensor::new(&composite_graph, &device, weight_data);
    let composite_bias: Tensor<1> = Tensor::new(&composite_graph, &device, bias_data);
    let composite_output =
        composite_input.layer_norm(&composite_weight, Some(&composite_bias), eps);
    let composite_loss = composite_output.sqr().reshape([12]).sum();
    let composite_gradients = composite_loss.backward().unwrap();

    let fused_output = flatten(fused_output.raw().clone()).await;
    let composite_output = flatten(composite_output.raw().clone()).await;
    let fused_dx = flatten(fused_gradients.get(&fused_input).unwrap()).await;
    let composite_dx = flatten(composite_gradients.get(&composite_input).unwrap()).await;
    let fused_dw = flatten(fused_gradients.get(&fused_weight).unwrap()).await;
    let composite_dw = flatten(composite_gradients.get(&composite_weight).unwrap()).await;
    let fused_db = flatten(fused_gradients.get(&fused_bias).unwrap()).await;
    let composite_db = flatten(composite_gradients.get(&composite_bias).unwrap()).await;

    assert_slice_close(&fused_output, &composite_output);
    assert_slice_close(&fused_dx, &composite_dx);
    assert_slice_close(&fused_dw, &composite_dw);
    assert_slice_close(&fused_db, &composite_db);
}

#[tokio::test]
async fn test_backward_flash_attention_matches_composite_cpu() {
    let device = Device::cpu();
    let q_data = &[[[[0.2f32, 0.6], [1.0, -0.3]]]];
    let k_data = &[[[[0.4f32, -0.7], [0.9, 0.1]]]];
    let v_data = &[[[[1.1f32, -0.5], [0.3, 0.8]]]];
    let scale = (2.0f32).sqrt();

    let fused_graph = Graph::new();
    let fused_q: Tensor<4> = Tensor::new(&fused_graph, &device, q_data);
    let fused_k: Tensor<4> = Tensor::new(&fused_graph, &device, k_data);
    let fused_v: Tensor<4> = Tensor::new(&fused_graph, &device, v_data);
    let fused_output = fused_q.flash_attention(&fused_k, &fused_v, scale, None);
    let fused_loss = fused_output.sqr().reshape([4]).sum();
    let fused_gradients = fused_loss.backward().unwrap();

    let composite_graph = Graph::new();
    let composite_q: Tensor<4> = Tensor::new(&composite_graph, &device, q_data);
    let composite_k: Tensor<4> = Tensor::new(&composite_graph, &device, k_data);
    let composite_v: Tensor<4> = Tensor::new(&composite_graph, &device, v_data);
    let composite_output =
        composite_q.flash_attention_composite(&composite_k, &composite_v, scale, None);
    let composite_loss = composite_output.sqr().reshape([4]).sum();
    let composite_gradients = composite_loss.backward().unwrap();

    let fused_output = flatten(fused_output.raw().clone()).await;
    let composite_output = flatten(composite_output.raw().clone()).await;
    let fused_dq = flatten(fused_gradients.get(&fused_q).unwrap()).await;
    let composite_dq = flatten(composite_gradients.get(&composite_q).unwrap()).await;
    let fused_dk = flatten(fused_gradients.get(&fused_k).unwrap()).await;
    let composite_dk = flatten(composite_gradients.get(&composite_k).unwrap()).await;
    let fused_dv = flatten(fused_gradients.get(&fused_v).unwrap()).await;
    let composite_dv = flatten(composite_gradients.get(&composite_v).unwrap()).await;

    assert_slice_close(&fused_output, &composite_output);
    assert_slice_close(&fused_dq, &composite_dq);
    assert_slice_close(&fused_dk, &composite_dk);
    assert_slice_close(&fused_dv, &composite_dv);
}

#[tokio::test]
async fn test_backward_mat_mul_rank3_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let lhs_data = (1..=24).map(|n| n as f32).collect::<Vec<_>>();
    let rhs_data = (1..=40).map(|n| n as f32).collect::<Vec<_>>();
    let lhs: Tensor<3> = Tensor::from_slice(&graph, &device, [2, 3, 4], &lhs_data);
    let rhs: Tensor<3> = Tensor::from_slice(&graph, &device, [2, 4, 5], &rhs_data);

    let output = lhs.mat_mul(&rhs);
    let output_values = output.raw().clone().as_slice().await.unwrap();
    let gradients = output.flatten_all().sum().backward().unwrap();
    let dlhs = flatten(gradients.get(&lhs).unwrap()).await;
    let drhs = flatten(gradients.get(&rhs).unwrap()).await;

    assert_eq!(output_values.shape(), &[2, 3, 5]);
    assert_close(output_values[[0, 0, 0]], 110.0);
    assert_close(output_values[[1, 2, 4]], 2950.0);

    // with an all-ones seed, dlhs[b, i, k] = sum_j rhs[b, k, j] and
    // drhs[b, k, j] = sum_i lhs[b, i, k]
    for b in 0..2 {
        for i in 0..3 {
            for k in 0..4 {
                let expected = (0..5).map(|j| rhs_data[b * 20 + k * 5 + j]).sum::<f32>();
                assert_close(dlhs[b * 12 + i * 4 + k], expected);
            }
        }
        for k in 0..4 {
            for j in 0..5 {
                let expected = (0..3).map(|i| lhs_data[b * 12 + i * 4 + k]).sum::<f32>();
                assert_close(drhs[b * 20 + k * 5 + j], expected);
            }
        }
    }

    let lhs_small = lhs_data.iter().map(|value| value * 0.05).collect::<Vec<_>>();
    let rhs_small = rhs_data.iter().map(|value| value * 0.03).collect::<Vec<_>>();
    let fd_device = device.clone();
    let fd_rhs = rhs_small.clone();
    assert_gradient_matches_finite_difference(
        &device,
        [2, 3, 4],
        &lhs_small,
        move |graph, lhs| {
            let rhs = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [2, 4, 5], &fd_rhs),
            );
            lhs.mat_mul(&rhs).sqr().flatten_all().sum()
        },
    )
    .await;
    let fd_device = device.clone();
    let fd_lhs = lhs_small.clone();
    assert_gradient_matches_finite_difference(
        &device,
        [2, 4, 5],
        &rhs_small,
        move |graph, rhs| {
            let lhs = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [2, 3, 4], &fd_lhs),
            );
            lhs.mat_mul(&rhs).sqr().flatten_all().sum()
        },
    )
    .await;
}

#[tokio::test]
async fn test_backward_cat_dim0_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let first_data = (1..=6).map(|n| n as f32).collect::<Vec<_>>();
    let second_data = (7..=18).map(|n| n as f32).collect::<Vec<_>>();
    let first: Tensor<3> = Tensor::from_slice(&graph, &device, [1, 2, 3], &first_data);
    let second: Tensor<3> = Tensor::from_slice(&graph, &device, [2, 2, 3], &second_data);

    let output = Tensor::cat(vec![first.clone(), second.clone()], 0);
    let output_values = flatten(output.raw().clone()).await;
    let seed_data = (0..18).map(|n| n as f32 + 10.0).collect::<Vec<_>>();
    let seed = RawTensor::from_slice(&device, [3, 2, 3], &seed_data);
    let gradients = output.backward_with(seed).unwrap();
    let dfirst = flatten(gradients.get(&first).unwrap()).await;
    let dsecond = flatten(gradients.get(&second).unwrap()).await;

    assert_eq!(output.shape(), [3, 2, 3]);
    assert_eq!(output_values, (1..=18).map(|n| n as f32).collect::<Vec<_>>());
    assert_eq!(dfirst, seed_data[..6].to_vec());
    assert_eq!(dsecond, seed_data[6..].to_vec());
}

#[tokio::test]
async fn test_backward_cat_dim1_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let first_data = (1..=6).map(|n| n as f32).collect::<Vec<_>>();
    let second_data = (10..=21).map(|n| n as f32).collect::<Vec<_>>();
    let first: Tensor<3> = Tensor::from_slice(&graph, &device, [2, 1, 3], &first_data);
    let second: Tensor<3> = Tensor::from_slice(&graph, &device, [2, 2, 3], &second_data);

    let output = Tensor::cat(vec![first.clone(), second.clone()], 1);
    let output_values = flatten(output.raw().clone()).await;
    let seed_data = (0..18).map(|n| n as f32 + 10.0).collect::<Vec<_>>();
    let seed = RawTensor::from_slice(&device, [2, 3, 3], &seed_data);
    let gradients = output.backward_with(seed).unwrap();
    let dfirst = flatten(gradients.get(&first).unwrap()).await;
    let dsecond = flatten(gradients.get(&second).unwrap()).await;

    assert_eq!(output.shape(), [2, 3, 3]);
    assert_eq!(
        output_values,
        vec![
            1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 4.0, 5.0, 6.0, 16.0, 17.0,
            18.0, 19.0, 20.0, 21.0
        ]
    );

    let mut expected_dfirst = Vec::new();
    let mut expected_dsecond = Vec::new();
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..3 {
                let value = seed_data[i * 9 + j * 3 + k];
                if j < 1 {
                    expected_dfirst.push(value);
                } else {
                    expected_dsecond.push(value);
                }
            }
        }
    }
    assert_eq!(dfirst, expected_dfirst);
    assert_eq!(dsecond, expected_dsecond);

    let first_small = first_data.iter().map(|value| value * 0.1).collect::<Vec<_>>();
    let second_small = second_data.iter().map(|value| value * 0.1).collect::<Vec<_>>();
    let fd_device = device.clone();
    let fd_second = second_small.clone();
    assert_gradient_matches_finite_difference(
        &device,
        [2, 1, 3],
        &first_small,
        move |graph, first| {
            let second = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [2, 2, 3], &fd_second),
            );
            Tensor::cat(vec![first, second], 1).sqr().flatten_all().sum()
        },
    )
    .await;
    let fd_device = device.clone();
    let fd_first = first_small.clone();
    assert_gradient_matches_finite_difference(
        &device,
        [2, 2, 3],
        &second_small,
        move |graph, second| {
            let first = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [2, 1, 3], &fd_first),
            );
            Tensor::cat(vec![first, second], 1).sqr().flatten_all().sum()
        },
    )
    .await;
}

#[tokio::test]
async fn test_backward_cat_dim2_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let first_data = (1..=4).map(|n| n as f32).collect::<Vec<_>>();
    let second_data = (5..=12).map(|n| n as f32).collect::<Vec<_>>();
    let first: Tensor<3> = Tensor::from_slice(&graph, &device, [2, 2, 1], &first_data);
    let second: Tensor<3> = Tensor::from_slice(&graph, &device, [2, 2, 2], &second_data);

    let output = Tensor::cat(vec![first.clone(), second.clone()], 2);
    let output_values = flatten(output.raw().clone()).await;
    let seed_data = (0..12).map(|n| n as f32 + 10.0).collect::<Vec<_>>();
    let seed = RawTensor::from_slice(&device, [2, 2, 3], &seed_data);
    let gradients = output.backward_with(seed).unwrap();
    let dfirst = flatten(gradients.get(&first).unwrap()).await;
    let dsecond = flatten(gradients.get(&second).unwrap()).await;

    assert_eq!(output.shape(), [2, 2, 3]);
    assert_eq!(
        output_values,
        vec![1.0, 5.0, 6.0, 2.0, 7.0, 8.0, 3.0, 9.0, 10.0, 4.0, 11.0, 12.0]
    );

    let mut expected_dfirst = Vec::new();
    let mut expected_dsecond = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..3 {
                let value = seed_data[i * 6 + j * 3 + k];
                if k < 1 {
                    expected_dfirst.push(value);
                } else {
                    expected_dsecond.push(value);
                }
            }
        }
    }
    assert_eq!(dfirst, expected_dfirst);
    assert_eq!(dsecond, expected_dsecond);

    let first_small = first_data.iter().map(|value| value * 0.1).collect::<Vec<_>>();
    let second_small = second_data.iter().map(|value| value * 0.1).collect::<Vec<_>>();
    let fd_device = device.clone();
    let fd_second = second_small.clone();
    assert_gradient_matches_finite_difference(
        &device,
        [2, 2, 1],
        &first_small,
        move |graph, first| {
            let second = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [2, 2, 2], &fd_second),
            );
            Tensor::cat(vec![first, second], 2).sqr().flatten_all().sum()
        },
    )
    .await;
    let fd_device = device.clone();
    let fd_first = first_small.clone();
    assert_gradient_matches_finite_difference(
        &device,
        [2, 2, 2],
        &second_small,
        move |graph, second| {
            let first = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [2, 2, 1], &fd_first),
            );
            Tensor::cat(vec![first, second], 2).sqr().flatten_all().sum()
        },
    )
    .await;
}

#[tokio::test]
async fn test_backward_log_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32, 1.5, 2.5]);

    let output = input.log();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap().to_vec1();

    for (value, input) in output_values.iter().zip([0.5f32, 1.5, 2.5]) {
        assert_close(*value, input.ln());
    }
    for (value, input) in dinput.iter().zip([0.5f32, 1.5, 2.5]) {
        assert_close(*value, 1.0 / input);
    }

    assert_gradient_matches_finite_difference(&device, [3], &[0.5, 1.5, 2.5], |_, x| {
        x.log().sum()
    })
    .await;
}

#[tokio::test]
async fn test_backward_neg_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[1.5f32, -2.0, 0.5]);

    let output = input.neg();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap().to_vec1();

    assert_eq!(output_values, vec![-1.5, 2.0, -0.5]);
    assert_eq!(dinput, vec![-1.0, -1.0, -1.0]);

    assert_gradient_matches_finite_difference(&device, [3], &[1.5, -2.0, 0.5], |_, x| {
        x.neg().sum()
    })
    .await;
}

#[tokio::test]
async fn test_backward_exp_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<1> = Tensor::new(&graph, &device, &[0.0f32, 0.5, -1.0]);

    let output = input.exp();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap().to_vec1();

    for (value, input) in output_values.iter().zip([0.0f32, 0.5, -1.0]) {
        assert_close(*value, input.exp());
    }
    for (value, input) in dinput.iter().zip([0.0f32, 0.5, -1.0]) {
        assert_close(*value, input.exp());
    }

    assert_gradient_matches_finite_difference(&device, [3], &[0.0, 0.5, -1.0], |_, x| {
        x.exp().sum()
    })
    .await;
}

#[tokio::test]
async fn test_backward_log_sum_exp_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let input: Tensor<2> = Tensor::new(&graph, &device, &[[0.0f32, 0.5, 1.0], [1.0, -1.0, 0.0]]);

    let output = input.exp().sum_keepdim(1).log();
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec2();
    let gradients = output.reshape([2]).sum().backward().unwrap();
    let dinput = gradients.get(&input).unwrap().as_slice().await.unwrap().to_vec2();

    // d/dx_j log(sum_k exp(x_k)) = softmax(x)_j
    let rows = [[0.0f32, 0.5, 1.0], [1.0, -1.0, 0.0]];
    for (row_index, row) in rows.iter().enumerate() {
        let sum_exp = row.iter().map(|value| value.exp()).sum::<f32>();
        assert_close(output_values[row_index][0], sum_exp.ln());
        for (column, value) in row.iter().enumerate() {
            assert_close(dinput[row_index][column], value.exp() / sum_exp);
        }
    }

    assert_gradient_matches_finite_difference(
        &device,
        [2, 3],
        &[0.0, 0.5, 1.0, 1.0, -1.0, 0.0],
        |_, x| x.exp().sum_keepdim(1).log().reshape([2]).sum(),
    )
    .await;
}

#[tokio::test]
async fn test_backward_with_backwards_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
    let y: Tensor<1> = Tensor::new(&graph, &device, &[4.0f32, 5.0, 6.0]);

    let x_target = x.clone();
    let y_target = y.clone();
    let output = x.add(&y).with_backwards([x.parent(), y.parent()], move |grad| {
        Ok(vec![
            BackwardTarget::wrt(&x_target, grad.clone().mul_scalar(2.0).to_concrete()),
            BackwardTarget::wrt(&y_target, grad.mul_scalar(-3.0).to_concrete()),
        ])
    });
    let output_values = output.raw().clone().as_slice().await.unwrap().to_vec1();
    let gradients = output.sum().backward().unwrap();
    let dx = gradients.get(&x).unwrap().as_slice().await.unwrap().to_vec1();
    let dy = gradients.get(&y).unwrap().as_slice().await.unwrap().to_vec1();

    assert_eq!(output_values, vec![5.0, 7.0, 9.0]);
    // the custom rule replaces add's backward, so the gradients are the
    // custom 2x/-3x rather than add's 1/1
    assert_eq!(dx, vec![2.0, 2.0, 2.0]);
    assert_eq!(dy, vec![-3.0, -3.0, -3.0]);
}

#[tokio::test]
async fn test_backward_with_backwards_missing_parent_errors_cpu() {
    let graph = Graph::new();
    let device = Device::cpu();
    let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
    let y: Tensor<1> = Tensor::new(&graph, &device, &[4.0f32, 5.0, 6.0]);

    let x_target = x.clone();
    let output = x.add(&y).with_backwards([x.parent(), y.parent()], move |grad| {
        Ok(vec![BackwardTarget::wrt(&x_target, grad)])
    });

    // The scheduler waits on a gradient from every child edge, so a custom
    // rule that skips a live parent must fail loudly instead of silently
    // dropping every gradient upstream of it.
    let Err(error) = output.sum().backward() else {
        panic!("backward succeeded despite an omitted parent gradient");
    };
    assert!(
        error.to_string().contains("omitted a gradient"),
        "expected missing-parent error, got: {error}",
    );
}

#[tokio::test]
async fn test_cpu_graph_drops_after_backward() {
    let graph = Graph::new();
    let weak = Arc::downgrade(&graph.inner);
    let device = Device::cpu();

    let x: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);
    let w: Tensor<2> = Tensor::new(&graph, &device, &[[0.5f32, -1.0], [1.5, 2.0]]);
    let loss = x.mat_mul(&w).sum(1).sum();
    let gradients = loss.backward().unwrap();
    assert!(gradients.get(&x).is_some());
    assert!(gradients.get(&w).is_some());

    drop(gradients);
    drop(loss);
    drop(x);
    drop(w);
    drop(graph);

    assert!(
        weak.upgrade().is_none(),
        "autograd graph stayed alive after all tensors were dropped",
    );
}

#[test]
fn test_gpu_gradients_can_detach() {
    let Ok(device) = Device::gpu_blocking() else {
        eprintln!("skipping GPU gradient detach regression test: GPU unavailable");
        return;
    };

    let graph = Graph::new();
    let x: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);
    let w: Tensor<2> = Tensor::new(&graph, &device, &[[0.5f32, -1.0], [1.5, 2.0]]);
    let gradients = x
        .mat_mul(&w)
        .sum(1)
        .sum()
        .backward()
        .unwrap()
        .into_detached();
    let dx = gradients.get(&x).expect("missing x gradient");
    let dw = gradients.get(&w).expect("missing w gradient");

    assert_eq!(
        dx.as_gpu()
            .expect("expected GPU x gradient")
            .count_kernels_to_resolve(),
        0,
        "detached x gradient should not retain backward compute graph",
    );
    assert_eq!(
        dw.as_gpu()
            .expect("expected GPU w gradient")
            .count_kernels_to_resolve(),
        0,
        "detached w gradient should not retain backward compute graph",
    );
}
