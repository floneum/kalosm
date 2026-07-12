use super::*;
use crate::{Layout, ToVec};
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

async fn test_devices() -> Vec<Device> {
    let mut devices = vec![Device::cpu()];
    match Device::gpu().await {
        Ok(gpu) => devices.push(gpu),
        Err(_) => eprintln!("skipping GPU coverage: GPU unavailable"),
    }
    devices
}

async fn flatten<const R: usize>(tensor: RawTensor<R, f32>) -> Vec<f32> {
    let elements = tensor.shape().into_iter().product();
    tensor
        .reshape([elements])
        .as_slice()
        .await
        .unwrap()
        .to_vec()
}

#[test]
fn non_grad_nodes_do_not_retain_a_backward_tape() {
    let graph = Graph::new();
    let input: Tensor<1> = graph.constant_from_data(&Device::cpu(), &[1.0f32, 2.0, 3.0]);
    let output = input.sqr().sum();
    let state = graph.inner.state.lock().unwrap();

    for node in state.nodes.values() {
        assert!(!node.requires_grad);
        assert!(
            node.parents.is_empty(),
            "non-grad nodes must not retain parent graph structure"
        );
        assert!(
            node.backward.is_none(),
            "non-grad nodes must not retain tensor-capturing backward closures"
        );
    }
    assert!(state.nodes.contains_key(&output.handle.id));
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
        let plus = loss(
            &graph,
            Tensor::from_slice(&graph, device, shape, &perturbed),
        );
        let plus = plus.raw().to_scalar().await.unwrap();
        perturbed[index] = data[index] - epsilon;
        let graph = Graph::new();
        let minus = loss(
            &graph,
            Tensor::from_slice(&graph, device, shape, &perturbed),
        );
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
async fn test_backward_squared_sum() {
    for device in test_devices().await {
        let graph = Graph::new();

        let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
        let loss = x.sqr().sum();
        let gradients = loss.backward().unwrap();
        let dx = gradients
            .get(&x)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(dx[0], 2.0);
        assert_close(dx[1], 4.0);
        assert_close(dx[2], 6.0);
    }
}

#[tokio::test]
async fn test_autograd_silu() {
    for device in test_devices().await {
        let graph = Graph::new();
        let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, -2.0, 0.5]);

        let output = x.silu();
        let values = output.raw().clone().as_slice().await.unwrap().to_vec();

        let expected = [1.0f32, -2.0, 0.5].map(|v| v / (1.0 + (-v).exp()));
        for (value, expected) in values.iter().zip(expected) {
            assert_close(*value, expected);
        }

        let gradients = output.sum().backward().unwrap();
        let dx = gradients
            .get(&x)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

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
}

#[tokio::test]
async fn test_autograd_gelu() {
    for device in test_devices().await {
        let graph = Graph::new();
        let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, -2.0, 0.5]);

        let output = x.gelu();
        let values = output.raw().clone().as_slice().await.unwrap().to_vec();

        let expected = [1.0f32, -2.0, 0.5].map(|v| {
            0.5 * v
                * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (v + 0.044_715 * v.powi(3))).tanh())
        });
        for (value, expected) in values.iter().zip(expected) {
            assert_close(*value, expected);
        }

        let gradients = output.sum().backward().unwrap();
        let dx = gradients
            .get(&x)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

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
}

#[tokio::test]
async fn test_backward_where_cond() {
    for device in test_devices().await {
        let graph = Graph::new();
        let condition: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 0.0, -2.0]);
        let on_true: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 3.0, 4.0]);
        let on_false: Tensor<1> = Tensor::new(&graph, &device, &[10.0f32, 20.0, 30.0]);

        let output = condition.where_cond(&on_true, &on_false);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.flatten_all().sum().backward().unwrap();

        let dcondition = gradients
            .get(&condition)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        let dtrue = gradients
            .get(&on_true)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        let dfalse = gradients
            .get(&on_false)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![2.0, 20.0, 4.0]);
        assert_eq!(dcondition, vec![0.0, 0.0, 0.0]);
        assert_eq!(dtrue, vec![1.0, 0.0, 1.0]);
        assert_eq!(dfalse, vec![0.0, 1.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_index_select() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let indices = RawTensor::from_slice(&device, [3], &[2u32, 0, 2]);

        let output = input.index_select(1, &indices);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(
            output_values,
            vec![vec![3.0, 1.0, 3.0], vec![6.0, 4.0, 6.0]]
        );
        assert_eq!(dinput, vec![vec![1.0, 0.0, 2.0], vec![1.0, 0.0, 2.0]]);
    }
}

#[tokio::test]
async fn test_backward_slice_assign() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<2> = Tensor::new(
            &graph,
            &device,
            &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        );
        let value: Tensor<2> = Tensor::new(&graph, &device, &[[10.0f32, 11.0], [12.0, 13.0]]);

        let output = input.slice_assign([0..2, 1..3], &value);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum(1).sum().backward().unwrap();

        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        let dvalue = gradients
            .get(&value)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(
            output_values,
            vec![
                vec![1.0, 10.0, 11.0],
                vec![4.0, 12.0, 13.0],
                vec![7.0, 8.0, 9.0]
            ]
        );
        assert_eq!(
            dinput,
            vec![
                vec![1.0, 0.0, 0.0],
                vec![1.0, 0.0, 0.0],
                vec![1.0, 1.0, 1.0]
            ]
        );
        assert_eq!(dvalue, vec![vec![1.0, 1.0], vec![1.0, 1.0]]);
    }
}

#[tokio::test]
async fn test_backward_expand() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[2.0f32, 3.0, 4.0]]);

        let output = input.expand([2, 3]);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(
            output_values,
            vec![vec![2.0, 3.0, 4.0], vec![2.0, 3.0, 4.0]]
        );
        assert_eq!(dinput, vec![vec![2.0, 2.0, 2.0]]);
    }
}

#[tokio::test]
async fn test_backward_flatten_all() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);

        let output = input.flatten_all();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(dinput, vec![vec![1.0, 1.0], vec![1.0, 1.0]]);
    }
}

#[tokio::test]
async fn test_backward_flatten_last_n() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<3> = Tensor::new(
            &graph,
            &device,
            &[
                [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
        );

        let output = input.flatten_last_n::<1, 2>();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .reshape([2, 6])
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(
            output_values,
            vec![
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
            ]
        );
        assert_eq!(
            dinput,
            vec![
                vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            ]
        );
    }
}

#[tokio::test]
async fn test_backward_flatten_first_n() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<3> = Tensor::new(
            &graph,
            &device,
            &[
                [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
        );

        let output = input.flatten_first_n::<1, 2>();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .reshape([4, 3])
            .as_slice()
            .await
            .unwrap()
            .to_vec();

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
}

#[tokio::test]
async fn test_backward_narrow() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<2> = Tensor::new(
            &graph,
            &device,
            &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        );

        let output = input.narrow(1usize, 1, 2);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(
            output_values,
            vec![vec![2.0, 3.0], vec![5.0, 6.0], vec![8.0, 9.0]]
        );
        assert_eq!(
            dinput,
            vec![
                vec![0.0, 1.0, 1.0],
                vec![0.0, 1.0, 1.0],
                vec![0.0, 1.0, 1.0]
            ]
        );
    }
}

#[tokio::test]
async fn test_backward_repeat() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);

        let output = input.repeat([2, 3]);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

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
}

#[tokio::test]
async fn test_backward_resize() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<2> = Tensor::new(
            &graph,
            &device,
            &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        );

        let output = input.resize([2, 2]);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![vec![1.0, 2.0], vec![4.0, 5.0]]);
        assert_eq!(
            dinput,
            vec![
                vec![1.0, 1.0, 0.0],
                vec![1.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0]
            ]
        );
    }
}

#[tokio::test]
async fn test_backward_restride() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0, 4.0]);

        let output = input.restride([StrideSpec::dim(0, 2), StrideSpec::dim(0, 3)]);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(
            output_values,
            vec![vec![1.0, 2.0, 3.0], vec![2.0, 3.0, 4.0]]
        );
        assert_eq!(dinput, vec![1.0, 2.0, 2.0, 1.0]);
    }
}

#[tokio::test]
async fn test_backward_restride_strided_overlap() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(
            &graph,
            &device,
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );

        let output = input.restride([StrideSpec::dim_with(0, 3, 2), StrideSpec::dim(0, 3)]);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(
            output_values,
            vec![
                vec![1.0, 2.0, 3.0],
                vec![3.0, 4.0, 5.0],
                vec![5.0, 6.0, 7.0]
            ]
        );
        assert_eq!(dinput, vec![1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_restride_layout() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let layout = Layout::contiguous(&[5])
            .restride(&[StrideSpec::dim(0, 2).with_offset(1), StrideSpec::dim(0, 2)]);

        let output: Tensor<2> = input.restride_layout(layout);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![vec![2.0, 3.0], vec![3.0, 4.0]]);
        assert_eq!(dinput, vec![0.0, 1.0, 2.0, 1.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_squeeze_dims() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<4> = Tensor::new(
            &graph,
            &device,
            &[[[[1.0f32], [2.0], [3.0]]], [[[4.0], [5.0], [6.0]]]],
        );

        let output = input.squeeze_dims::<2, 2>([1, 3]);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .reshape([2, 3])
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(
            output_values,
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]
        );
        assert_eq!(dinput, vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
    }
}

#[tokio::test]
async fn test_backward_unsqueeze_dims() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let output = input.unsqueeze_dims::<2, 4>([0, 2]);
        let output_values = output
            .raw()
            .clone()
            .reshape([2, 3])
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        let gradients = output.sum(3).sum(2).sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output.shape(), [1, 2, 1, 3]);
        assert_eq!(
            output_values,
            vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]
        );
        assert_eq!(dinput, vec![vec![1.0, 1.0, 1.0], vec![1.0, 1.0, 1.0]]);
    }
}

#[tokio::test]
async fn test_backward_max() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 5.0, 5.0], [4.0, 2.0, 0.0]]);

        let output = input.max::<1>(1);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![5.0, 4.0]);
        assert_eq!(dinput, vec![vec![0.0, 0.5, 0.5], vec![1.0, 0.0, 0.0]]);
    }
}

#[tokio::test]
async fn test_backward_min() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 1.0, 5.0], [4.0, 2.0, 0.0]]);

        let output = input.min::<1>(1);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![1.0, 0.0]);
        assert_eq!(dinput, vec![vec![0.5, 0.5, 0.0], vec![0.0, 0.0, 1.0]]);
    }
}

#[tokio::test]
async fn test_backward_mean() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let output = input.mean::<1>(1);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![2.0, 5.0]);
        assert_eq!(
            dinput,
            vec![
                vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
            ]
        );
    }
}

#[tokio::test]
async fn test_backward_product() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<2> = Tensor::new(
            &graph,
            &device,
            &[[2.0f32, 3.0, 4.0], [5.0, 0.0, 7.0], [0.0, 0.0, 9.0]],
        );

        let output = input.product::<1>(1);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![24.0, 0.0, 0.0]);
        assert_eq!(
            dinput,
            vec![
                vec![12.0, 8.0, 6.0],
                vec![0.0, 35.0, 0.0],
                vec![0.0, 0.0, 0.0]
            ]
        );
    }
}

#[tokio::test]
async fn test_backward_product_keepdim() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[2.0f32, 3.0, 4.0], [5.0, 0.0, 7.0]]);

        let output = input.product_keepdim::<1>(1);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![vec![24.0], vec![0.0]]);
        assert_eq!(dinput, vec![vec![12.0, 8.0, 6.0], vec![0.0, 35.0, 0.0]]);
    }
}

#[tokio::test]
async fn test_backward_var() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let output = input.var::<1>(1);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_slice_close(&output_values, &[2.0 / 3.0, 2.0 / 3.0]);
        assert_slice_close(
            &dinput.into_iter().flatten().collect::<Vec<_>>(),
            &[
                -2.0 / 3.0,
                0.0,
                2.0 / 3.0,
                -2.0 / 3.0,
                0.0,
                2.0 / 3.0,
            ],
        );
    }
}

#[tokio::test]
async fn test_backward_var_keepdim() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

        let output = input.var_keepdim::<1>(1);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum(1).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_slice_close(
            &output_values.into_iter().flatten().collect::<Vec<_>>(),
            &[2.0 / 3.0, 2.0 / 3.0],
        );
        assert_slice_close(
            &dinput.into_iter().flatten().collect::<Vec<_>>(),
            &[
                -2.0 / 3.0,
                0.0,
                2.0 / 3.0,
                -2.0 / 3.0,
                0.0,
                2.0 / 3.0,
            ],
        );
    }
}

#[tokio::test]
async fn test_backward_clamp() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[-1.0f32, 0.0, 2.0, 5.0]);

        let output = input.clamp(0.0, 3.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![0.0, 0.0, 2.0, 3.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 1.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_eq() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 1.0]);

        let output = input.eq(1.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![1.0, 0.0, 1.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_eq_scalar() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[3.0f32, 2.0, 3.0]);

        let output = input.eq_scalar(3.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![1.0, 0.0, 1.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_eq_tensor() {
    for device in test_devices().await {
        let graph = Graph::new();
        let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
        let rhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 0.0, 3.0]);

        let output = lhs.eq_tensor(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dlhs = gradients
            .get(&lhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        let drhs = gradients
            .get(&rhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![1.0, 0.0, 1.0]);
        assert_eq!(dlhs, vec![0.0, 0.0, 0.0]);
        assert_eq!(drhs, vec![0.0, 0.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_gt_scalar() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

        let output = input.gt_scalar(2.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![0.0, 0.0, 1.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_gt_tensor() {
    for device in test_devices().await {
        let graph = Graph::new();
        let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 3.0]);
        let rhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 1.0, 3.0]);

        let output = lhs.gt_tensor(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dlhs = gradients
            .get(&lhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        let drhs = gradients
            .get(&rhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![0.0, 1.0, 0.0]);
        assert_eq!(dlhs, vec![0.0, 0.0, 0.0]);
        assert_eq!(drhs, vec![0.0, 0.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_gte_scalar() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

        let output = input.gte_scalar(2.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![0.0, 1.0, 1.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_gte_tensor() {
    for device in test_devices().await {
        let graph = Graph::new();
        let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 3.0]);
        let rhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 4.0, 2.0]);

        let output = lhs.gte_tensor(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dlhs = gradients
            .get(&lhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        let drhs = gradients
            .get(&rhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![0.0, 1.0, 1.0]);
        assert_eq!(dlhs, vec![0.0, 0.0, 0.0]);
        assert_eq!(drhs, vec![0.0, 0.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_lt() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

        let output = input.lt(2.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![1.0, 0.0, 0.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_lt_scalar() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

        let output = input.lt_scalar(3.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![1.0, 1.0, 0.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_lt_tensor() {
    for device in test_devices().await {
        let graph = Graph::new();
        let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
        let rhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 1.0, 3.0]);

        let output = lhs.lt_tensor(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dlhs = gradients
            .get(&lhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        let drhs = gradients
            .get(&rhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![1.0, 0.0, 0.0]);
        assert_eq!(dlhs, vec![0.0, 0.0, 0.0]);
        assert_eq!(drhs, vec![0.0, 0.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_lte() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

        let output = input.lte(2.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![1.0, 1.0, 0.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_lte_scalar() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

        let output = input.lte_scalar(1.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![1.0, 0.0, 0.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_lte_tensor() {
    for device in test_devices().await {
        let graph = Graph::new();
        let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
        let rhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 2.0, 1.0]);

        let output = lhs.lte_tensor(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dlhs = gradients
            .get(&lhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        let drhs = gradients
            .get(&rhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![1.0, 1.0, 0.0]);
        assert_eq!(dlhs, vec![0.0, 0.0, 0.0]);
        assert_eq!(drhs, vec![0.0, 0.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_max_elementwise() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[-1.0f32, 0.0, 2.0]);

        let output = input.max_elementwise(0.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![0.0, 0.0, 2.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 1.0]);
    }
}

#[tokio::test]
async fn test_backward_max_scalar() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

        let output = input.max_scalar(3.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![3.0, 4.0, 3.0]);
        assert_eq!(dinput, vec![0.0, 1.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_min_elementwise() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

        let output = input.min_elementwise(3.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![1.0, 3.0, 2.0]);
        assert_eq!(dinput, vec![1.0, 0.0, 1.0]);
    }
}

#[tokio::test]
async fn test_backward_min_scalar() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

        let output = input.min_scalar(2.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![1.0, 2.0, 2.0]);
        assert_eq!(dinput, vec![1.0, 0.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_mt() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

        let output = input.mt(2.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![0.0, 1.0, 0.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_mte() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

        let output = input.mte(2.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![0.0, 1.0, 1.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_ne() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

        let output = input.ne(2.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![1.0, 1.0, 0.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_ne_scalar() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);

        let output = input.ne_scalar(4.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![1.0, 0.0, 1.0]);
        assert_eq!(dinput, vec![0.0, 0.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_ne_tensor() {
    for device in test_devices().await {
        let graph = Graph::new();
        let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 4.0, 2.0]);
        let rhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 0.0, 3.0]);

        let output = lhs.ne_tensor(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dlhs = gradients
            .get(&lhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        let drhs = gradients
            .get(&rhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![0.0, 1.0, 1.0]);
        assert_eq!(dlhs, vec![0.0, 0.0, 0.0]);
        assert_eq!(drhs, vec![0.0, 0.0, 0.0]);
    }
}

#[tokio::test]
async fn test_backward_abs() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[-2.0f32, 0.0, 3.0]);

        let output = input.abs();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![2.0, 0.0, 3.0]);
        assert_eq!(dinput, vec![-1.0, 0.0, 1.0]);
    }
}

#[tokio::test]
async fn test_backward_acos() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let output = input.acos();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0], 0.5f32.acos());
        assert_close(dinput[0], -1.0f32 / (1.0f32 - 0.25f32).sqrt());
    }
}

#[tokio::test]
async fn test_backward_acosh() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32]);

        let output = input.acosh();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0], 2.0f32.acosh());
        assert_close(
            dinput[0],
            1.0f32 / ((2.0f32 - 1.0f32).sqrt() * (2.0f32 + 1.0f32).sqrt()),
        );
    }
}

#[tokio::test]
async fn test_backward_approximate_exp() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32]);

        let output = input.approximate_exp();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0], 1.0f32.exp());
        assert_close(dinput[0], 1.0f32.exp());
    }
}

#[tokio::test]
async fn test_backward_asin() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let output = input.asin();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0], 0.5f32.asin());
        assert_close(dinput[0], 1.0f32 / (1.0f32 - 0.25f32).sqrt());
    }
}

#[tokio::test]
async fn test_backward_asinh() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.5f32]);

        let output = input.asinh();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0], 1.5f32.asinh());
        assert_close(dinput[0], 1.0f32 / (1.5f32 * 1.5f32 + 1.0f32).sqrt());
    }
}

#[tokio::test]
async fn test_backward_atan() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let output = input.atan();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0], 0.5f32.atan());
        assert_close(dinput[0], 1.0f32 / (1.0f32 + 0.25f32));
    }
}

#[tokio::test]
async fn test_backward_atanh() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let output = input.atanh();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0], 0.5f32.atanh());
        assert_close(dinput[0], 1.0f32 / (1.0f32 - 0.25f32));
    }
}

#[tokio::test]
async fn test_backward_cos() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let output = input.cos();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0], 0.5f32.cos());
        assert_close(dinput[0], -0.5f32.sin());
    }
}

#[tokio::test]
async fn test_backward_cosh() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let output = input.cosh();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0], 0.5f32.cosh());
        assert_close(dinput[0], 0.5f32.sinh());
    }
}

#[tokio::test]
async fn test_backward_exp2() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32]);

        let output = input.exp2();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0], 2.0f32.exp2());
        assert_close(dinput[0], std::f32::consts::LN_2 * 2.0f32.exp2());
    }
}

#[tokio::test]
async fn test_backward_less_approximate_exp() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32]);

        let output = input.less_approximate_exp();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0], 1.0f32.exp());
        assert_close(dinput[0], 1.0f32.exp());
    }
}

#[tokio::test]
async fn test_backward_log2() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[4.0f32]);

        let output = input.log2();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0], 4.0f32.log2());
        assert_close(dinput[0], 1.0f32 / (4.0f32 * std::f32::consts::LN_2));
    }
}

#[tokio::test]
async fn test_backward_sin() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let output = input.sin();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0], 0.5f32.sin());
        assert_close(dinput[0], 0.5f32.cos());
    }
}

#[tokio::test]
async fn test_backward_sinh() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let output = input.sinh();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0], 0.5f32.sinh());
        assert_close(dinput[0], 0.5f32.cosh());
    }
}

#[tokio::test]
async fn test_backward_tan() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let output = input.tan();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0], 0.5f32.tan());
        assert_close(dinput[0], 1.0f32 / (0.5f32.cos() * 0.5f32.cos()));
    }
}

#[tokio::test]
async fn test_backward_tanh_exact() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32]);

        let output = input.tanh_exact();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0], 0.5f32.tanh());
        assert_close(dinput[0], 1.0f32 - 0.5f32.tanh().powi(2));
    }
}

#[tokio::test]
async fn test_cast() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

        let output = input.cast::<half::f16>();
        let output_values = output.as_slice().await.unwrap().to_vec();

        assert_close(f32::from(output_values[0]), 1.0);
        assert_close(f32::from(output_values[1]), 2.0);
        assert_close(f32::from(output_values[2]), 3.0);
    }
}

#[tokio::test]
async fn test_arange() {
    for device in test_devices().await {
        let graph = Graph::new();

        let output = Tensor::<1>::arange(&graph, &device, 1.0, 5.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();

        assert_eq!(output_values, vec![1.0, 2.0, 3.0, 4.0]);
    }
}

#[tokio::test]
async fn test_arange_step() {
    for device in test_devices().await {
        let graph = Graph::new();

        let output = Tensor::<1>::arange_step(&graph, &device, 1.0, 6.0, 2.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();

        assert_eq!(output_values, vec![1.0, 3.0, 5.0]);
    }
}

#[tokio::test]
async fn test_full() {
    for device in test_devices().await {
        let graph = Graph::new();

        let output: Tensor<2> = Tensor::full(&graph, &device, [2, 3], 1.5);
        let output_values = output.raw().clone().as_slice().await.unwrap();

        assert_eq!(output_values.shape(), &[2, 3]);
        for row in 0..2 {
            for col in 0..3 {
                assert_close(output_values[[row, col]], 1.5);
            }
        }
    }
}

#[tokio::test]
async fn test_zeros_like() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);

        let output = input.zeros_like();
        let output_values = output.raw().clone().as_slice().await.unwrap();

        assert_eq!(output_values.shape(), &[2, 2]);
        assert_close(output_values[[0, 0]], 0.0);
        assert_close(output_values[[0, 1]], 0.0);
        assert_close(output_values[[1, 0]], 0.0);
        assert_close(output_values[[1, 1]], 0.0);
    }
}

#[tokio::test]
async fn test_from_array() {
    for device in test_devices().await {
        let graph = Graph::new();

        let output: Tensor<2> = Tensor::from_array(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);
        let output_values = output.raw().clone().as_slice().await.unwrap();

        assert_eq!(output_values.shape(), &[2, 2]);
        assert_close(output_values[[0, 0]], 1.0);
        assert_close(output_values[[0, 1]], 2.0);
        assert_close(output_values[[1, 0]], 3.0);
        assert_close(output_values[[1, 1]], 4.0);
    }
}

#[tokio::test]
async fn test_backward_add_broadcast_api() {
    for device in test_devices().await {
        let graph = Graph::new();
        let lhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0]);
        let rhs: Tensor<2> = Tensor::new(&graph, &device, &[[10.0f32], [20.0]]);

        let output: Tensor<2> = lhs.add_::<2, 2>(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dlhs = gradients
            .get(&lhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        let drhs = gradients
            .get(&rhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0][0], 11.0);
        assert_close(output_values[0][1], 12.0);
        assert_close(output_values[1][0], 21.0);
        assert_close(output_values[1][1], 22.0);
        assert_close(dlhs[0], 2.0);
        assert_close(dlhs[1], 2.0);
        assert_close(drhs[0][0], 2.0);
        assert_close(drhs[1][0], 2.0);
    }
}

#[tokio::test]
async fn test_backward_sub_broadcast_api() {
    for device in test_devices().await {
        let graph = Graph::new();
        let lhs: Tensor<2> = Tensor::new(&graph, &device, &[[3.0f32], [4.0]]);
        let rhs: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0]);

        let output: Tensor<2> = lhs.sub_::<1, 2>(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dlhs = gradients
            .get(&lhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        let drhs = gradients
            .get(&rhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0][0], 2.0);
        assert_close(output_values[0][1], 1.0);
        assert_close(output_values[1][0], 3.0);
        assert_close(output_values[1][1], 2.0);
        assert_close(dlhs[0][0], 2.0);
        assert_close(dlhs[1][0], 2.0);
        assert_close(drhs[0], -2.0);
        assert_close(drhs[1], -2.0);
    }
}

#[tokio::test]
async fn test_backward_mul_broadcast_api() {
    for device in test_devices().await {
        let graph = Graph::new();
        let lhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 3.0]);
        let rhs: Tensor<2> = Tensor::new(&graph, &device, &[[10.0f32], [20.0]]);

        let output: Tensor<2> = lhs.mul_::<2, 2>(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dlhs = gradients
            .get(&lhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        let drhs = gradients
            .get(&rhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0][0], 20.0);
        assert_close(output_values[0][1], 30.0);
        assert_close(output_values[1][0], 40.0);
        assert_close(output_values[1][1], 60.0);
        assert_close(dlhs[0], 30.0);
        assert_close(dlhs[1], 30.0);
        assert_close(drhs[0][0], 5.0);
        assert_close(drhs[1][0], 5.0);
    }
}

#[tokio::test]
async fn test_backward_div_broadcast_api() {
    for device in test_devices().await {
        let graph = Graph::new();
        let lhs: Tensor<2> = Tensor::new(&graph, &device, &[[10.0f32], [20.0]]);
        let rhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 4.0]);

        let output: Tensor<2> = lhs.div_::<1, 2>(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dlhs = gradients
            .get(&lhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        let drhs = gradients
            .get(&rhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0][0], 5.0);
        assert_close(output_values[0][1], 2.5);
        assert_close(output_values[1][0], 10.0);
        assert_close(output_values[1][1], 5.0);
        assert_close(dlhs[0][0], 0.75);
        assert_close(dlhs[1][0], 0.75);
        assert_close(drhs[0], -7.5);
        assert_close(drhs[1], -1.875);
    }
}

#[tokio::test]
async fn test_backward_pow_broadcast_api() {
    for device in test_devices().await {
        let graph = Graph::new();
        let lhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, 3.0]);
        let rhs: Tensor<2> = Tensor::new(&graph, &device, &[[2.0f32], [1.0]]);

        let output: Tensor<2> = lhs.pow_::<2, 2>(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dlhs = gradients
            .get(&lhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        let drhs = gradients
            .get(&rhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0][0], 4.0);
        assert_close(output_values[0][1], 9.0);
        assert_close(output_values[1][0], 2.0);
        assert_close(output_values[1][1], 3.0);
        assert_close(dlhs[0], 5.0);
        assert_close(dlhs[1], 7.0);
        assert_close(drhs[0][0], 12.660099);
        assert_close(drhs[1][0], 4.6821313);
    }
}

#[tokio::test]
async fn test_backward_chunk() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<2> = Tensor::new(
            &graph,
            &device,
            &[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        );

        let chunks = input.chunk(2, 1);
        assert_eq!(chunks.len(), 2);
        let first = chunks[0].raw().clone().as_slice().await.unwrap().to_vec();
        let second = chunks[1].raw().clone().as_slice().await.unwrap().to_vec();
        let loss = chunks[0]
            .flatten_all()
            .sum()
            .add(&chunks[1].flatten_all().sum().mul_scalar(2.0));
        let gradients = loss.backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

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
}

#[tokio::test]
async fn test_backward_matmul() {
    for device in test_devices().await {
        let graph = Graph::new();
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
}

#[tokio::test]
async fn test_backward_t() {
    for device in test_devices().await {
        let graph = Graph::new();
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
}

#[tokio::test]
async fn test_backward_pool() {
    for device in test_devices().await {
        let graph = Graph::new();
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
}

#[tokio::test]
async fn test_backward_pool_max() {
    for device in test_devices().await {
        let graph = Graph::new();
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
}

#[tokio::test]
async fn test_backward_pool_min() {
    for device in test_devices().await {
        let graph = Graph::new();
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
}

#[tokio::test]
async fn test_backward_q_mat_mul() {
    for device in test_devices().await {
        let graph = Graph::new();
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
}

#[tokio::test]
async fn test_backward_stack() {
    for device in test_devices().await {
        let graph = Graph::new();
        let first: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0]);
        let second: Tensor<1> = Tensor::new(&graph, &device, &[3.0f32, 4.0]);

        let output = Tensor::stack::<2>(vec![first.clone(), second.clone()], 0);
        let output_values = output.raw().clone().as_slice().await.unwrap();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dfirst = gradients
            .get(&first)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        let dsecond = gradients
            .get(&second)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

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
}

#[tokio::test]
async fn test_backward_rope() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<4> = Tensor::new(
            &graph,
            &device,
            &[[[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]],
        );
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

        for index in [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 2],
            [0, 0, 1, 3],
        ] {
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
}

#[tokio::test]
async fn test_backward_rope_fused() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<4> = Tensor::new(
            &graph,
            &device,
            &[[[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]],
        );
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

        for index in [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 2],
            [0, 0, 1, 3],
        ] {
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
}

#[tokio::test]
async fn test_backward_rope_interleaved() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<4> = Tensor::new(
            &graph,
            &device,
            &[[[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]],
        );
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

        for index in [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 2],
            [0, 0, 1, 3],
        ] {
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
}

#[tokio::test]
async fn test_backward_rope_normal_fused() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<4> = Tensor::new(
            &graph,
            &device,
            &[[[[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]],
        );
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

        for index in [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 2],
            [0, 0, 1, 3],
        ] {
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
}

#[tokio::test]
async fn test_backward_pow() {
    for device in test_devices().await {
        let graph = Graph::new();
        let lhs: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32]);
        let rhs: Tensor<1> = Tensor::new(&graph, &device, &[3.0f32]);

        let output = lhs.pow(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dlhs = gradients
            .get(&lhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        let drhs = gradients
            .get(&rhs)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0], 8.0);
        assert_close(dlhs[0], 12.0);
        assert_close(drhs[0], 8.0 * 2.0f32.ln());
    }
}

#[tokio::test]
async fn test_backward_pow_elementwise() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[3.0f32]);

        let output = input.pow_elementwise(2.0);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0], 9.0);
        assert_close(dinput[0], 6.0);
    }
}

#[tokio::test]
async fn test_backward_pow_scalar() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[4.0f32]);

        let output = input.pow_scalar(0.5);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(output_values[0], 2.0);
        assert_close(dinput[0], 0.25);
    }
}

#[tokio::test]
async fn test_autograd_rms_norm() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let weight: Tensor<2> = Tensor::constant_from_raw(
            &graph,
            RawTensor::from_slice(&device, [1, 3], &[1.0f32, 1.0, 1.0]),
        );

        let output = input.rms_norm(&weight, 1e-5);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();

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
            .to_vec();

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
                    RawTensor::from_slice(&fd_device, [1, 3], &[1.0f32, 1.0, 1.0]),
                );
                x.rms_norm(&weight, 1e-5).sum(1).sum()
            },
        )
        .await;
    }
}

#[tokio::test]
async fn test_backward_matmul_with_broadcast_bias() {
    for device in test_devices().await {
        let graph = Graph::new();

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
            .to_vec();
        let db = gradients
            .get(&b)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_close(dw[0][0], 5.0);
        assert_close(dw[1][0], 7.0);
        assert_close(dw[2][0], 9.0);
        assert_close(db[0], 2.0);
    }
}

#[tokio::test]
async fn test_backward_embedding() {
    for device in test_devices().await {
        let graph = Graph::new();

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
            .to_vec();

        assert_close(dtable[0][0], 1.0);
        assert_close(dtable[0][1], 1.0);
        assert_close(dtable[1][0], 0.0);
        assert_close(dtable[1][1], 0.0);
        assert_close(dtable[2][0], 1.0);
        assert_close(dtable[2][1], 1.0);
    }
}

#[tokio::test]
async fn test_backward_gather_last() {
    for device in test_devices().await {
        let graph = Graph::new();

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
            .to_vec();

        assert_close(dvalues[0][0], 0.0);
        assert_close(dvalues[0][1], 0.0);
        assert_close(dvalues[0][2], 1.0);
        assert_close(dvalues[1][0], 1.0);
        assert_close(dvalues[1][1], 0.0);
        assert_close(dvalues[1][2], 0.0);
    }
}

#[tokio::test]
async fn test_backward_softmax_last_dim_fused_matches_composite() {
    for device in test_devices().await {
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
}

#[tokio::test]
async fn test_backward_rms_norm_fused_matches_composite() {
    for device in test_devices().await {
        let input_data = &[
            [[0.3f32, -1.2, 0.7], [1.5, 0.1, -0.8]],
            [[-0.4, 0.9, 1.3], [0.2, -0.6, 0.5]],
        ];
        let weight_data = &[1.0f32, 0.75, 1.25];
        let eps = 1e-5;

        let fused_graph = Graph::new();
        let fused_input: Tensor<3> = Tensor::new(&fused_graph, &device, input_data);
        let fused_weight: Tensor<1> = Tensor::new(&fused_graph, &device, weight_data);
        let fused_output = fused_input.rms_norm_fused_no_bias::<1, 2>(&fused_weight, eps);
        let fused_loss = fused_output.sqr().reshape([12]).sum();
        let fused_gradients = fused_loss.backward().unwrap();

        let composite_graph = Graph::new();
        let composite_input: Tensor<3> = Tensor::new(&composite_graph, &device, input_data);
        let composite_weight: Tensor<3> = Tensor::from_slice(
            &composite_graph,
            &device,
            [1, 1, 3],
            weight_data,
        );
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
}

#[tokio::test]
async fn test_backward_layer_norm_last_dim_fused_matches_composite() {
    for device in test_devices().await {
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
            fused_input.layer_norm_last_dim_fused::<2, 1>(&fused_weight, Some(&fused_bias), eps);
        let fused_loss = fused_output.sqr().reshape([12]).sum();
        let fused_gradients = fused_loss.backward().unwrap();

        let composite_graph = Graph::new();
        let composite_input: Tensor<3> = Tensor::new(&composite_graph, &device, input_data);
        let composite_weight: Tensor<3> = Tensor::from_slice(
            &composite_graph,
            &device,
            [1, 1, 3],
            weight_data,
        );
        let composite_bias: Tensor<3> = Tensor::from_slice(
            &composite_graph,
            &device,
            [1, 1, 3],
            bias_data,
        );
        let composite_output =
            composite_input.layer_norm(&composite_weight, Some(&composite_bias), eps, true);
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
}

#[tokio::test]
async fn test_backward_flash_attention_matches_composite() {
    for device in test_devices().await {
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
}

#[tokio::test]
async fn test_backward_mat_mul_rank3() {
    for device in test_devices().await {
        let graph = Graph::new();
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

        let lhs_small = lhs_data
            .iter()
            .map(|value| value * 0.05)
            .collect::<Vec<_>>();
        let rhs_small = rhs_data
            .iter()
            .map(|value| value * 0.03)
            .collect::<Vec<_>>();
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
}

#[tokio::test]
async fn test_backward_cat_dim0() {
    for device in test_devices().await {
        let graph = Graph::new();
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
        assert_eq!(
            output_values,
            (1..=18).map(|n| n as f32).collect::<Vec<_>>()
        );
        assert_eq!(dfirst, seed_data[..6].to_vec());
        assert_eq!(dsecond, seed_data[6..].to_vec());
    }
}

#[tokio::test]
async fn test_backward_cat_dim1() {
    for device in test_devices().await {
        let graph = Graph::new();
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
                1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 4.0, 5.0, 6.0, 16.0, 17.0, 18.0,
                19.0, 20.0, 21.0
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

        let first_small = first_data
            .iter()
            .map(|value| value * 0.1)
            .collect::<Vec<_>>();
        let second_small = second_data
            .iter()
            .map(|value| value * 0.1)
            .collect::<Vec<_>>();
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
                Tensor::cat(vec![first, second], 1)
                    .sqr()
                    .flatten_all()
                    .sum()
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
                Tensor::cat(vec![first, second], 1)
                    .sqr()
                    .flatten_all()
                    .sum()
            },
        )
        .await;
    }
}

#[tokio::test]
async fn test_backward_cat_dim2() {
    for device in test_devices().await {
        let graph = Graph::new();
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
            vec![
                1.0, 5.0, 6.0, 2.0, 7.0, 8.0, 3.0, 9.0, 10.0, 4.0, 11.0, 12.0
            ]
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

        let first_small = first_data
            .iter()
            .map(|value| value * 0.1)
            .collect::<Vec<_>>();
        let second_small = second_data
            .iter()
            .map(|value| value * 0.1)
            .collect::<Vec<_>>();
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
                Tensor::cat(vec![first, second], 2)
                    .sqr()
                    .flatten_all()
                    .sum()
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
                Tensor::cat(vec![first, second], 2)
                    .sqr()
                    .flatten_all()
                    .sum()
            },
        )
        .await;
    }
}

#[tokio::test]
async fn test_backward_log() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.5f32, 1.5, 2.5]);

        let output = input.log();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

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
}

#[tokio::test]
async fn test_backward_neg() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.5f32, -2.0, 0.5]);

        let output = input.neg();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![-1.5, 2.0, -0.5]);
        assert_eq!(dinput, vec![-1.0, -1.0, -1.0]);

        assert_gradient_matches_finite_difference(&device, [3], &[1.5, -2.0, 0.5], |_, x| {
            x.neg().sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_backward_exp() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[0.0f32, 0.5, -1.0]);

        let output = input.exp();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

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
}

#[tokio::test]
async fn test_backward_log_sum_exp() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<2> =
            Tensor::new(&graph, &device, &[[0.0f32, 0.5, 1.0], [1.0, -1.0, 0.0]]);

        let output = input.exp().sum_keepdim(1).log();
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.reshape([2]).sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

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
}

#[tokio::test]
async fn test_backward_with_backwards() {
    for device in test_devices().await {
        let graph = Graph::new();
        let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
        let y: Tensor<1> = Tensor::new(&graph, &device, &[4.0f32, 5.0, 6.0]);

        let x_target = x.clone();
        let y_target = y.clone();
        let output = x
            .add(&y)
            .with_backwards([x.parent(), y.parent()], move |grad| {
                Ok(vec![
                    BackwardTarget::wrt(&x_target, grad.clone().mul_scalar(2.0).to_concrete()),
                    BackwardTarget::wrt(&y_target, grad.mul_scalar(-3.0).to_concrete()),
                ])
            });
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dx = gradients
            .get(&x)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        let dy = gradients
            .get(&y)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_eq!(output_values, vec![5.0, 7.0, 9.0]);
        // the custom rule replaces add's backward, so the gradients are the
        // custom 2x/-3x rather than add's 1/1
        assert_eq!(dx, vec![2.0, 2.0, 2.0]);
        assert_eq!(dy, vec![-3.0, -3.0, -3.0]);
    }
}

#[tokio::test]
async fn test_backward_with_backwards_missing_parent_errors() {
    for device in test_devices().await {
        let graph = Graph::new();
        let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
        let y: Tensor<1> = Tensor::new(&graph, &device, &[4.0f32, 5.0, 6.0]);

        let x_target = x.clone();
        let output = x
            .add(&y)
            .with_backwards([x.parent(), y.parent()], move |grad| {
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
}

#[tokio::test]
async fn test_graph_drops_after_backward() {
    for device in test_devices().await {
        let graph = Graph::new();
        let weak = Arc::downgrade(&graph.inner);

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

/// 8x8 XOR grid plus deterministic LCG weight init for a 2-`hidden`-2 MLP,
/// shared by the XOR training tests.
fn xor_training_data(hidden: usize) -> (Vec<f32>, Vec<u32>, Vec<f32>, Vec<f32>) {
    let mut features = Vec::with_capacity(128);
    let mut labels = Vec::with_capacity(64);
    for row in 0..8 {
        for column in 0..8 {
            let x = -0.875 + 0.25 * row as f32;
            let y = -0.875 + 0.25 * column as f32;
            features.extend([x, y]);
            labels.push(u32::from((x > 0.0) != (y > 0.0)));
        }
    }

    let mut state = 42u64;
    let mut next_uniform = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as f32 / (1u64 << 32) as f32 - 0.5
    };
    let w1_init: Vec<f32> = (0..2 * hidden).map(|_| next_uniform()).collect();
    let w2_init: Vec<f32> = (0..hidden * 2).map(|_| next_uniform() * 0.5).collect();
    (features, labels, w1_init, w2_init)
}

/// End-to-end training: a 2-16-2 MLP learns XOR over an 8x8 grid of 2D
/// points with softmax cross-entropy and full-batch SGD. Exercises the
/// whole tape (matmul, broadcast bias, relu, softmax, gather, log,
/// reduce) plus the detach/update loop across many resolves per device.
#[tokio::test]
async fn test_train_xor_classifier() {
    const SAMPLES: usize = 64;
    const HIDDEN: usize = 16;
    const STEPS: usize = 500;
    const LEARNING_RATE: f32 = 1.0;

    let (features, labels, w1_init, w2_init) = xor_training_data(HIDDEN);

    for (device, name) in test_devices().await.into_iter().zip(["cpu", "gpu"]) {
        let inputs = RawTensor::from_slice(&device, [SAMPLES, 2], &features);
        let targets = RawTensor::from_slice(&device, [SAMPLES], &labels);

        let mut w1 = RawTensor::from_slice(&device, [2, HIDDEN], &w1_init);
        let mut b1 = RawTensor::zeros(&device, [HIDDEN]);
        let mut w2 = RawTensor::from_slice(&device, [HIDDEN, 2], &w2_init);
        let mut b2 = RawTensor::zeros(&device, [2]);

        let mut final_loss = f32::INFINITY;
        for step in 0..STEPS {
            let graph = Graph::new();
            let x = Tensor::constant_from_raw(&graph, inputs.clone());
            let w1_t = Tensor::from_raw(&graph, w1.clone());
            let b1_t = Tensor::from_raw(&graph, b1.clone());
            let w2_t = Tensor::from_raw(&graph, w2.clone());
            let b2_t = Tensor::from_raw(&graph, b2.clone());

            let hidden = b1_t.add_::<2, 2>(&x.mat_mul(&w1_t)).relu();
            let logits = b2_t.add_::<2, 2>(&hidden.mat_mul(&w2_t));
            // Numerically stable cross-entropy: log softmax via log-sum-exp
            // so a saturated class cannot underflow to log(0).
            let shifted = logits.sub_::<2, 2>(&logits.max_keepdim::<1>(1));
            let log_sum_exp = shifted.exp().sum_keepdim(1).log();
            let label_log_probs = shifted.sub_::<2, 2>(&log_sum_exp).gather_last(&targets);
            let loss: Tensor<0> = label_log_probs.sum().mul_scalar(-1.0 / SAMPLES as f32);

            let loss_value = flatten(loss.raw().clone()).await[0];
            let gradients = loss.backward().unwrap().into_detached();
            let dw1 = gradients.get(&w1_t).unwrap();
            let db1 = gradients.get(&b1_t).unwrap();
            let dw2 = gradients.get(&w2_t).unwrap();
            let db2 = gradients.get(&b2_t).unwrap();

            w1 = (w1 - dw1 * LEARNING_RATE).to_concrete();
            b1 = (b1 - db1 * LEARNING_RATE).to_concrete();
            w2 = (w2 - dw2 * LEARNING_RATE).to_concrete();
            b2 = (b2 - db2 * LEARNING_RATE).to_concrete();

            final_loss = loss_value;
            if step % 100 == 0 {
                eprintln!("[{name}] step {step}: loss {loss_value:.4}");
            }
        }
        eprintln!("[{name}] final loss {final_loss:.4}");

        let graph = Graph::new();
        let x = Tensor::constant_from_raw(&graph, inputs.clone());
        let w1_t = Tensor::constant_from_raw(&graph, w1.clone());
        let b1_t = Tensor::constant_from_raw(&graph, b1.clone());
        let w2_t = Tensor::constant_from_raw(&graph, w2.clone());
        let b2_t = Tensor::constant_from_raw(&graph, b2.clone());
        let hidden = b1_t.add_::<2, 2>(&x.mat_mul(&w1_t)).relu();
        let logits = b2_t.add_::<2, 2>(&hidden.mat_mul(&w2_t));
        let logits = logits.raw().clone().as_slice().await.unwrap().to_vec();
        let correct = logits
            .iter()
            .zip(&labels)
            .filter(|(row, label)| u32::from(row[1] > row[0]) == **label)
            .count();
        eprintln!("[{name}] accuracy {correct}/{SAMPLES}");

        assert!(
            final_loss < 0.1,
            "training did not converge: final loss {final_loss}",
        );
        assert_eq!(correct, SAMPLES, "classifier misclassified training points");
    }
}

#[tokio::test]
async fn test_autograd_sigmoid() {
    for device in test_devices().await {
        let graph = Graph::new();
        let inputs = [-2.0f32, -0.5, 0.0, 1.0, 3.0];
        let x: Tensor<1> = Tensor::new(&graph, &device, &inputs);

        let output = x.sigmoid();
        let values = output.raw().clone().as_slice().await.unwrap().to_vec();

        let expected = inputs.map(|v| 1.0 / (1.0 + (-v).exp()));
        for (value, expected) in values.iter().zip(expected) {
            assert_close(*value, expected);
        }

        let gradients = output.sum().backward().unwrap();
        let dx = gradients
            .get(&x)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        let expected_grads = inputs.map(|v| {
            let sigmoid = 1.0 / (1.0 + (-v).exp());
            sigmoid * (1.0 - sigmoid)
        });
        for (value, expected) in dx.iter().zip(expected_grads) {
            assert_close(*value, expected);
        }

        assert_gradient_matches_finite_difference(&device, [5], &inputs, |_, x| x.sigmoid().sum())
            .await;
    }
}

#[tokio::test]
async fn test_autograd_to_concrete() {
    for device in test_devices().await {
        let graph = Graph::new();
        let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, -2.0, 3.0]);

        let output = x.mul_scalar(2.0).to_concrete();
        let values = output.raw().clone().as_slice().await.unwrap().to_vec();
        assert_eq!(values, vec![2.0, -4.0, 6.0]);

        let gradients = output.sqr().sum().backward().unwrap();
        let dx = gradients
            .get(&x)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        assert_slice_close(&dx, &[8.0, -16.0, 24.0]);

        let graph = Graph::new();
        let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, -2.0, 3.0]);
        let gradients = x.to_concrete().sqr().sum().backward().unwrap();
        let dx = gradients
            .get(&x)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        assert_slice_close(&dx, &[2.0, -4.0, 6.0]);

        assert_gradient_matches_finite_difference(&device, [3], &[1.0f32, -2.0, 3.0], |_, x| {
            x.to_concrete().sqr().sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_autograd_index_select_rank_generic() {
    for device in test_devices().await {
        let graph = Graph::new();
        let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0, 4.0]);
        let indices = RawTensor::from_slice(&device, [3], &[2u32, 0, 2]);
        let selected = x.index_select(0, &indices);
        assert_eq!(
            selected.raw().clone().as_slice().await.unwrap().to_vec(),
            vec![3.0, 1.0, 3.0]
        );
        let gradients = selected.sum().backward().unwrap();
        let dx = flatten(gradients.get(&x).unwrap()).await;
        assert_eq!(dx, vec![1.0, 0.0, 2.0, 0.0]);

        let graph = Graph::new();
        let data: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let x: Tensor<3> = Tensor::from_slice(&graph, &device, [2, 3, 2], &data);
        let indices = RawTensor::from_slice(&device, [4], &[2u32, 0, 1, 0]);
        let selected = x.index_select(1, &indices);
        assert_eq!(selected.shape(), [2, 4, 2]);
        assert_eq!(
            flatten(selected.raw().clone()).await,
            vec![4.0, 5.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 10.0, 11.0, 6.0, 7.0, 8.0, 9.0, 6.0, 7.0]
        );
        let gradients = selected.flatten_all().sum().backward().unwrap();
        let dx = flatten(gradients.get(&x).unwrap()).await;
        assert_eq!(
            dx,
            vec![2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]
        );

        assert_gradient_matches_finite_difference(
            &device,
            [2, 3],
            &[0.5f32, -1.0, 2.0, 3.0, -0.5, 1.5],
            |graph, x| {
                let indices = RawTensor::from_slice(&x.device(), [2], &[2u32, 0]);
                let weights = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&x.device(), [2, 2], &[1.0f32, 2.0, 3.0, 4.0]),
                );
                x.index_select(1, &indices)
                    .mul(&weights)
                    .flatten_all()
                    .sum()
            },
        )
        .await;

        assert_gradient_matches_finite_difference(
            &device,
            [2, 3],
            &[0.5f32, -1.0, 2.0, 3.0, -0.5, 1.5],
            |graph, x| {
                let indices = RawTensor::from_slice(&x.device(), [3], &[1u32, 1, 0]);
                let weights = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(
                        &x.device(),
                        [3, 3],
                        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                    ),
                );
                x.index_select(0, &indices)
                    .mul(&weights)
                    .flatten_all()
                    .sum()
            },
        )
        .await;
    }
}

#[tokio::test]
async fn test_autograd_i_indexing() {
    for device in test_devices().await {
        let graph = Graph::new();
        let x: Tensor<2> =
            Tensor::from_slice(&graph, &device, [2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let row = x.i((1, ..));
        assert_eq!(
            row.raw().clone().as_slice().await.unwrap().to_vec(),
            vec![4.0, 5.0, 6.0]
        );
        let gradients = row.sum().backward().unwrap();
        let dx = flatten(gradients.get(&x).unwrap()).await;
        assert_eq!(dx, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);

        let graph = Graph::new();
        let x: Tensor<2> =
            Tensor::from_slice(&graph, &device, [2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let column = x.i((0..2, 1));
        assert_eq!(
            column.raw().clone().as_slice().await.unwrap().to_vec(),
            vec![2.0, 5.0]
        );
        let gradients = column.sum().backward().unwrap();
        let dx = flatten(gradients.get(&x).unwrap()).await;
        assert_eq!(dx, vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);

        let graph = Graph::new();
        let data: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let x: Tensor<3> = Tensor::from_slice(&graph, &device, [2, 2, 3], &data);
        let plane = x.i((.., 1, ..));
        assert_eq!(plane.shape(), [2, 3]);
        assert_eq!(
            flatten(plane.raw().clone()).await,
            vec![3.0, 4.0, 5.0, 9.0, 10.0, 11.0]
        );
        let gradients = plane.flatten_all().sum().backward().unwrap();
        let dx = flatten(gradients.get(&x).unwrap()).await;
        assert_eq!(
            dx,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        );

        let graph = Graph::new();
        let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
        let x: Tensor<4> = Tensor::from_slice(&graph, &device, [2, 2, 2, 2], &data);
        let cube = x.i((1, .., .., ..));
        assert_eq!(cube.shape(), [2, 2, 2]);
        assert_eq!(
            flatten(cube.raw().clone()).await,
            (8..16).map(|v| v as f32).collect::<Vec<_>>()
        );
        let gradients = cube.flatten_all().sum().backward().unwrap();
        let dx = flatten(gradients.get(&x).unwrap()).await;
        let expected: Vec<f32> = (0..16).map(|v| if v < 8 { 0.0 } else { 1.0 }).collect();
        assert_eq!(dx, expected);

        assert_gradient_matches_finite_difference(
            &device,
            [2, 3],
            &[0.5f32, -1.0, 2.0, 3.0, -0.5, 1.5],
            |graph, x| {
                let weights = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&x.device(), [3], &[1.0f32, 2.0, 3.0]),
                );
                x.i((1, ..)).mul(&weights).sum()
            },
        )
        .await;
    }
}

#[tokio::test]
async fn test_backward_squeeze() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<3> =
            Tensor::from_slice(&graph, &device, [2, 1, 2], &[1.0, 2.0, 3.0, 4.0]);
        let output: Tensor<2> = input.squeeze::<2>(1);
        assert_eq!(output.shape(), [2, 2]);
        let values = output.raw().clone().as_slice().await.unwrap().to_vec();
        assert_eq!(values, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

        let gradients = output.sqr().flatten_all().sum().backward().unwrap();
        let dx = flatten(gradients.get(&input).unwrap()).await;
        assert_slice_close(&dx, &[2.0, 4.0, 6.0, 8.0]);

        assert_gradient_matches_finite_difference(
            &device,
            [2, 1, 2],
            &[1.0, 2.0, 3.0, 4.0],
            |_, x| x.squeeze::<2>(1).sqr().flatten_all().sum(),
        )
        .await;
    }
}

#[tokio::test]
async fn test_backward_unsqueeze() {
    for device in test_devices().await {
        let graph = Graph::new();
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input: Tensor<3> = Tensor::from_slice(&graph, &device, [2, 2, 2], &data);
        let output: Tensor<4> = input.unsqueeze::<4>(1);
        assert_eq!(output.shape(), [2, 1, 2, 2]);
        assert_eq!(flatten(output.raw().clone()).await, data.to_vec());

        let gradients = output.sqr().flatten_all().sum().backward().unwrap();
        let dx = flatten(gradients.get(&input).unwrap()).await;
        assert_slice_close(&dx, &data.map(|value| 2.0 * value));

        assert_gradient_matches_finite_difference(&device, [2, 2, 2], &data, |_, x| {
            x.unsqueeze::<4>(3).sqr().flatten_all().sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_backward_cat_rank1() {
    for device in test_devices().await {
        let graph = Graph::new();
        let first: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0]);
        let second: Tensor<1> = Tensor::new(&graph, &device, &[3.0f32, 4.0, 5.0]);
        let output = Tensor::cat(vec![first.clone(), second.clone()], 0);
        assert_eq!(output.shape(), [5]);
        let values = output.raw().clone().as_slice().await.unwrap().to_vec();
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let gradients = output.sqr().sum().backward().unwrap();
        assert_slice_close(&flatten(gradients.get(&first).unwrap()).await, &[2.0, 4.0]);
        assert_slice_close(
            &flatten(gradients.get(&second).unwrap()).await,
            &[6.0, 8.0, 10.0],
        );

        assert_gradient_matches_finite_difference(&device, [2], &[1.0, 2.0], |graph, x| {
            let other = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&x.device(), [3], &[3.0, 4.0, 5.0]),
            );
            Tensor::cat(vec![x, other], 0).sqr().sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_backward_cat_rank2() {
    for device in test_devices().await {
        let graph = Graph::new();
        let first: Tensor<2> =
            Tensor::from_slice(&graph, &device, [2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let second: Tensor<2> = Tensor::from_slice(&graph, &device, [2, 1], &[5.0, 6.0]);
        let output = Tensor::cat(vec![first.clone(), second.clone()], 1);
        assert_eq!(output.shape(), [2, 3]);
        let values = output.raw().clone().as_slice().await.unwrap().to_vec();
        assert_eq!(values, vec![vec![1.0, 2.0, 5.0], vec![3.0, 4.0, 6.0]]);

        let gradients = output.sqr().flatten_all().sum().backward().unwrap();
        assert_slice_close(
            &flatten(gradients.get(&first).unwrap()).await,
            &[2.0, 4.0, 6.0, 8.0],
        );
        assert_slice_close(&flatten(gradients.get(&second).unwrap()).await, &[10.0, 12.0]);

        assert_gradient_matches_finite_difference(&device, [2, 1], &[5.0, 6.0], |graph, x| {
            let other = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&x.device(), [2, 2], &[1.0, 2.0, 3.0, 4.0]),
            );
            Tensor::cat(vec![other, x], 1).sqr().flatten_all().sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_backward_pad_with_zeros() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
        let output = input.pad_with_zeros(0, 1, 2);
        assert_eq!(output.shape(), [6]);
        let values = output.raw().clone().as_slice().await.unwrap().to_vec();
        assert_eq!(values, vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0]);

        let gradients = output.sqr().sum().backward().unwrap();
        assert_slice_close(
            &flatten(gradients.get(&input).unwrap()).await,
            &[2.0, 4.0, 6.0],
        );

        let passthrough = input.pad_with_zeros(0, 0, 0);
        let gradients = passthrough.sqr().sum().backward().unwrap();
        assert_slice_close(
            &flatten(gradients.get(&input).unwrap()).await,
            &[2.0, 4.0, 6.0],
        );

        assert_gradient_matches_finite_difference(&device, [3], &[1.0, 2.0, 3.0], |_, x| {
            x.pad_with_zeros(0, 1, 2).sqr().sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_backward_pad_axis() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<2> =
            Tensor::from_slice(&graph, &device, [2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let output = input.pad_axis(1, 1);
        assert_eq!(output.shape(), [2, 4]);
        let values = output.raw().clone().as_slice().await.unwrap().to_vec();
        assert_eq!(
            values,
            vec![vec![0.0, 1.0, 2.0, 0.0], vec![0.0, 3.0, 4.0, 0.0]]
        );

        let gradients = output.sqr().flatten_all().sum().backward().unwrap();
        assert_slice_close(
            &flatten(gradients.get(&input).unwrap()).await,
            &[2.0, 4.0, 6.0, 8.0],
        );

        assert_gradient_matches_finite_difference(
            &device,
            [2, 2],
            &[1.0, 2.0, 3.0, 4.0],
            |_, x| x.pad_axis(0, 2).sqr().flatten_all().sum(),
        )
        .await;
    }
}

#[tokio::test]
async fn test_backward_sliding_window_view() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let output: Tensor<2> =
            input.sliding_window_view::<1, 2>([fusor_types::SlidingWindow::new(0, 3, 1)]);
        assert_eq!(output.shape(), [3, 3]);
        let values = output.raw().clone().as_slice().await.unwrap().to_vec();
        assert_eq!(
            values,
            vec![
                vec![1.0, 2.0, 3.0],
                vec![2.0, 3.0, 4.0],
                vec![3.0, 4.0, 5.0]
            ]
        );

        let gradients = output.flatten_all().sum().backward().unwrap();
        assert_slice_close(
            &flatten(gradients.get(&input).unwrap()).await,
            &[1.0, 2.0, 3.0, 2.0, 1.0],
        );

        assert_gradient_matches_finite_difference(
            &device,
            [5],
            &[1.0, 2.0, 3.0, 4.0, 5.0],
            |_, x| {
                x.sliding_window_view::<1, 2>([fusor_types::SlidingWindow::new(0, 3, 1)])
                    .sqr()
                    .flatten_all()
                    .sum()
            },
        )
        .await;
    }
}

#[tokio::test]
async fn test_backward_sliding_window_view_strided() {
    for device in test_devices().await {
        let graph = Graph::new();
        let data: Vec<f32> = (1..=10).map(|value| value as f32).collect();
        let input: Tensor<2> = Tensor::from_slice(&graph, &device, [2, 5], &data);
        let output: Tensor<3> =
            input.sliding_window_view::<1, 3>([fusor_types::SlidingWindow::new(1, 3, 2)]);
        assert_eq!(output.shape(), [2, 2, 3]);
        assert_eq!(
            flatten(output.raw().clone()).await,
            vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 9.0, 10.0]
        );

        let gradients = output.flatten_all().sum().backward().unwrap();
        assert_slice_close(
            &flatten(gradients.get(&input).unwrap()).await,
            &[1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
        );

        assert_gradient_matches_finite_difference(&device, [2, 5], &data, |_, x| {
            x.sliding_window_view::<1, 3>([fusor_types::SlidingWindow::new(1, 3, 2)])
                .sqr()
                .flatten_all()
                .sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_backward_sum_axis() {
    for device in test_devices().await {
        let graph = Graph::new();

        let x: Tensor<2> =
            Tensor::from_slice(&graph, &device, [2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let summed = x.sum(1);
        let forward = summed.raw().clone().as_slice().await.unwrap().to_vec();
        assert_slice_close(&forward, &[6.0, 15.0]);

        let weight =
            Tensor::constant_from_raw(&graph, RawTensor::from_slice(&device, [2], &[1.0, 2.0]));
        let loss = summed.mul(&weight).sum();
        let gradients = loss.backward().unwrap();
        let dx = flatten(gradients.get(&x).unwrap()).await;
        assert_slice_close(&dx, &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        let graph = Graph::new();
        let x: Tensor<2> =
            Tensor::from_slice(&graph, &device, [2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let summed = x.sum(0);
        let forward = summed.raw().clone().as_slice().await.unwrap().to_vec();
        assert_slice_close(&forward, &[5.0, 7.0, 9.0]);

        let weight = Tensor::constant_from_raw(
            &graph,
            RawTensor::from_slice(&device, [3], &[1.0, 2.0, 3.0]),
        );
        let loss = summed.mul(&weight).sum();
        let gradients = loss.backward().unwrap();
        let dx = flatten(gradients.get(&x).unwrap()).await;
        assert_slice_close(&dx, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_gradient_matches_finite_difference(&device, [2, 3], &data, |_, x| {
            x.sum(1).sqr().sum()
        })
        .await;
        assert_gradient_matches_finite_difference(&device, [2, 3], &data, |_, x| {
            x.sum(0).sqr().sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_backward_sum_high_rank() {
    for device in test_devices().await {
        let graph = Graph::new();

        let data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let x: Tensor<5> = Tensor::from_slice(&graph, &device, [2, 1, 2, 1, 2], &data);
        let summed = x.sum(2);
        assert_eq!(summed.shape(), [2, 1, 1, 2]);
        let forward = flatten(summed.raw().clone()).await;
        assert_slice_close(&forward, &[4.0, 6.0, 12.0, 14.0]);

        let loss = summed.sqr().sum(3).sum(2).sum(1).sum();
        let gradients = loss.backward().unwrap();
        let dx = flatten(gradients.get(&x).unwrap()).await;
        assert_slice_close(&dx, &[8.0, 12.0, 8.0, 12.0, 24.0, 28.0, 24.0, 28.0]);

        assert_gradient_matches_finite_difference(&device, [2, 1, 2, 1, 2], &data, |_, x| {
            x.sum(2).sqr().sum(3).sum(2).sum(1).sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_backward_sum_keepdim() {
    for device in test_devices().await {
        let graph = Graph::new();

        let x: Tensor<2> =
            Tensor::from_slice(&graph, &device, [2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let summed = x.sum_keepdim(1);
        assert_eq!(summed.shape(), [2, 1]);
        let forward = flatten(summed.raw().clone()).await;
        assert_slice_close(&forward, &[6.0, 15.0]);

        let weight = Tensor::constant_from_raw(
            &graph,
            RawTensor::from_slice(&device, [2, 1], &[1.0, 2.0]),
        );
        let loss = summed.mul(&weight).flatten_all().sum();
        let gradients = loss.backward().unwrap();
        let dx = flatten(gradients.get(&x).unwrap()).await;
        assert_slice_close(&dx, &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        let graph = Graph::new();
        let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
        let summed = x.sum_keepdim(0);
        assert_eq!(summed.shape(), [1]);
        let forward = summed.raw().clone().as_slice().await.unwrap().to_vec();
        assert_slice_close(&forward, &[6.0]);
        let gradients = summed.sqr().sum().backward().unwrap();
        let dx = gradients
            .get(&x)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        assert_slice_close(&dx, &[12.0, 12.0, 12.0]);

        let graph = Graph::new();
        let data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let x: Tensor<5> = Tensor::from_slice(&graph, &device, [2, 1, 2, 1, 2], &data);
        let summed = x.sum_keepdim(2);
        assert_eq!(summed.shape(), [2, 1, 1, 1, 2]);
        let forward = flatten(summed.raw().clone()).await;
        assert_slice_close(&forward, &[4.0, 6.0, 12.0, 14.0]);
        let gradients = summed.flatten_all().sum().backward().unwrap();
        let dx = flatten(gradients.get(&x).unwrap()).await;
        assert_slice_close(&dx, &[1.0; 8]);

        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_gradient_matches_finite_difference(&device, [2, 3], &data, |_, x| {
            x.sum_keepdim(0).sqr().flatten_all().sum()
        })
        .await;
        assert_gradient_matches_finite_difference(&device, [3], &data[..3], |_, x| {
            x.sum_keepdim(0).sqr().sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_backward_q_mat_mul_rank1() {
    for device in test_devices().await {
        let graph = Graph::new();
        let input: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0, 4.0]);
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
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        let gradients = output.sum().backward().unwrap();
        let dinput = gradients
            .get(&input)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();

        assert_slice_close(&output_values, &[30.0, 70.0]);
        assert_slice_close(&dinput, &[6.0, 8.0, 10.0, 12.0]);

        assert_gradient_matches_finite_difference(&device, [4], &[0.5, -1.0, 2.0, 0.25], |_, x| {
            x.q_mat_mul(&weights).sum()
        })
        .await;
    }
}

fn composite_ramp<const R: usize>(graph: &Graph, device: &Device, shape: [usize; R]) -> Tensor<R> {
    let elements: usize = shape.iter().product();
    Tensor::constant_from_raw(
        graph,
        crate::arange(device, 1.0, elements as f32 + 1.0)
            .reshape(shape)
            .to_concrete(),
    )
}

#[tokio::test]
async fn test_autograd_conv1d() {
    for device in test_devices().await {
        let x_data: Vec<f32> = (0..8).map(|i| (i as f32 * 0.7).sin()).collect();
        let w_data: Vec<f32> = (0..8).map(|i| (i as f32 * 0.3).cos()).collect();
        let b_data = [0.5f32, -1.0];

        let graph = Graph::new();
        let x: Tensor<3> = Tensor::from_slice(&graph, &device, [1, 2, 4], &x_data);
        let w: Tensor<3> = Tensor::from_slice(&graph, &device, [2, 2, 2], &w_data);
        let b: Tensor<1> = Tensor::from_slice(&graph, &device, [2], &b_data);
        let output = x.conv(&w, Some(&b), [1], [1]);

        let raw_x = RawTensor::from_slice(&device, [1, 2, 4], &x_data);
        let raw_w = RawTensor::from_slice(&device, [2, 2, 2], &w_data);
        let raw_b = RawTensor::from_slice(&device, [2], &b_data);
        let expected = raw_x.conv(&raw_w, Some(&raw_b), [1], [1]);
        assert_slice_close(
            &flatten(output.raw().clone()).await,
            &flatten(expected).await,
        );

        let fd_device = device.clone();
        let w_fd = w_data.clone();
        assert_gradient_matches_finite_difference(
            &device,
            [1, 2, 4],
            &x_data,
            move |graph, x| {
                let w = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [2, 2, 2], &w_fd),
                );
                let b = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [2], &[0.5f32, -1.0]),
                );
                let out = x.conv(&w, Some(&b), [1], [1]);
                out.mul(&composite_ramp(graph, &fd_device, out.shape()))
                    .flatten_all()
                    .sum()
            },
        )
        .await;

        let fd_device = device.clone();
        let x_fd = x_data.clone();
        assert_gradient_matches_finite_difference(
            &device,
            [2, 2, 2],
            &w_data,
            move |graph, w| {
                let x = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [1, 2, 4], &x_fd),
                );
                let out = x.conv(&w, None, [1], [1]);
                out.mul(&composite_ramp(graph, &fd_device, out.shape()))
                    .flatten_all()
                    .sum()
            },
        )
        .await;

        let fd_device = device.clone();
        let x_fd = x_data.clone();
        let w_fd = w_data.clone();
        assert_gradient_matches_finite_difference(&device, [2], &b_data, move |graph, b| {
            let x = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [1, 2, 4], &x_fd),
            );
            let w = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [2, 2, 2], &w_fd),
            );
            let out = x.conv(&w, Some(&b), [1], [1]);
            out.mul(&composite_ramp(graph, &fd_device, out.shape()))
                .flatten_all()
                .sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_autograd_conv2d_strided() {
    for device in test_devices().await {
        let x_data: Vec<f32> = (0..18).map(|i| (i as f32 * 0.41).sin()).collect();
        let w_data: Vec<f32> = (0..16).map(|i| (i as f32 * 0.23).cos()).collect();

        let graph = Graph::new();
        let x: Tensor<4> = Tensor::from_slice(&graph, &device, [1, 2, 3, 3], &x_data);
        let w: Tensor<4> = Tensor::from_slice(&graph, &device, [2, 2, 2, 2], &w_data);
        let output = x.conv(&w, None, [1, 1], [2, 2]);

        let raw_x = RawTensor::from_slice(&device, [1, 2, 3, 3], &x_data);
        let raw_w = RawTensor::from_slice(&device, [2, 2, 2, 2], &w_data);
        let expected = raw_x.conv(&raw_w, None, [1, 1], [2, 2]);
        assert_slice_close(
            &flatten(output.raw().clone()).await,
            &flatten(expected).await,
        );

        let fd_device = device.clone();
        let w_fd = w_data.clone();
        assert_gradient_matches_finite_difference(
            &device,
            [1, 2, 3, 3],
            &x_data,
            move |graph, x| {
                let w = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [2, 2, 2, 2], &w_fd),
                );
                let out = x.conv(&w, None, [1, 1], [2, 2]);
                out.mul(&composite_ramp(graph, &fd_device, out.shape()))
                    .flatten_all()
                    .sum()
            },
        )
        .await;

        let fd_device = device.clone();
        let x_fd = x_data.clone();
        assert_gradient_matches_finite_difference(
            &device,
            [2, 2, 2, 2],
            &w_data,
            move |graph, w| {
                let x = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [1, 2, 3, 3], &x_fd),
                );
                let out = x.conv(&w, None, [1, 1], [2, 2]);
                out.mul(&composite_ramp(graph, &fd_device, out.shape()))
                    .flatten_all()
                    .sum()
            },
        )
        .await;
    }
}

#[tokio::test]
async fn test_autograd_grouped_conv() {
    for device in test_devices().await {
        let x_data: Vec<f32> = (0..16).map(|i| (i as f32 * 0.57).sin()).collect();
        let w_data: Vec<f32> = (0..16).map(|i| (i as f32 * 0.31).cos()).collect();
        let b_data = [0.2f32, -0.4, 0.6, -0.8];

        let graph = Graph::new();
        let x: Tensor<3> = Tensor::from_slice(&graph, &device, [1, 4, 4], &x_data);
        let w: Tensor<3> = Tensor::from_slice(&graph, &device, [4, 2, 2], &w_data);
        let b: Tensor<1> = Tensor::from_slice(&graph, &device, [4], &b_data);
        let output = x.grouped_conv(&w, Some(&b), [1], [2], 2);

        let raw_x = RawTensor::from_slice(&device, [1, 4, 4], &x_data);
        let raw_w = RawTensor::from_slice(&device, [4, 2, 2], &w_data);
        let raw_b = RawTensor::from_slice(&device, [4], &b_data);
        let expected = raw_x.grouped_conv(&raw_w, Some(&raw_b), [1], [2], 2);
        assert_slice_close(
            &flatten(output.raw().clone()).await,
            &flatten(expected).await,
        );

        let fd_device = device.clone();
        let w_fd = w_data.clone();
        assert_gradient_matches_finite_difference(
            &device,
            [1, 4, 4],
            &x_data,
            move |graph, x| {
                let w = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [4, 2, 2], &w_fd),
                );
                let b = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [4], &[0.2f32, -0.4, 0.6, -0.8]),
                );
                let out = x.grouped_conv(&w, Some(&b), [1], [2], 2);
                out.mul(&composite_ramp(graph, &fd_device, out.shape()))
                    .flatten_all()
                    .sum()
            },
        )
        .await;

        let fd_device = device.clone();
        let x_fd = x_data.clone();
        assert_gradient_matches_finite_difference(
            &device,
            [4, 2, 2],
            &w_data,
            move |graph, w| {
                let x = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [1, 4, 4], &x_fd),
                );
                let out = x.grouped_conv(&w, None, [1], [2], 2);
                out.mul(&composite_ramp(graph, &fd_device, out.shape()))
                    .flatten_all()
                    .sum()
            },
        )
        .await;
    }
}

#[tokio::test]
async fn test_autograd_upsample_nearest2d() {
    for device in test_devices().await {
        let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
        let graph = Graph::new();
        let x: Tensor<4> = Tensor::from_slice(&graph, &device, [1, 2, 2, 3], &data);
        let output = x.upsample_nearest2d(2, 3);
        assert_eq!(output.shape(), [1, 2, 4, 9]);

        let raw_x = RawTensor::from_slice(&device, [1, 2, 2, 3], &data);
        let expected = raw_x.upsample_nearest2d(2, 3);
        assert_slice_close(
            &flatten(output.raw().clone()).await,
            &flatten(expected).await,
        );

        let gradients = output.flatten_all().sum().backward().unwrap();
        let dx = flatten(gradients.get(&x).unwrap()).await;
        for value in dx {
            assert_close(value, 6.0);
        }

        let fd_device = device.clone();
        assert_gradient_matches_finite_difference(
            &device,
            [1, 2, 2, 3],
            &data,
            move |graph, x| {
                let out = x.upsample_nearest2d(2, 3);
                out.mul(&composite_ramp(graph, &fd_device, out.shape()))
                    .flatten_all()
                    .sum()
            },
        )
        .await;
    }
}

#[tokio::test]
async fn test_autograd_softmax_slow() {
    for device in test_devices().await {
        let data = [1.0f32, 2.0, 3.0, -1.0, 0.5, 0.0];
        let graph = Graph::new();
        let x: Tensor<2> = Tensor::from_slice(&graph, &device, [2, 3], &data);
        let output = x.softmax_slow(1);

        let raw_x = RawTensor::from_slice(&device, [2, 3], &data);
        let expected = raw_x.softmax_slow::<1>(1);
        assert_slice_close(
            &flatten(output.raw().clone()).await,
            &flatten(expected).await,
        );

        let last = x.softmax_slow_last_dim();
        assert_slice_close(
            &flatten(last.raw().clone()).await,
            &flatten(output.raw().clone()).await,
        );

        let fd_device = device.clone();
        assert_gradient_matches_finite_difference(&device, [2, 3], &data, move |graph, x| {
            x.softmax_slow(1)
                .mul(&composite_ramp(graph, &fd_device, [2, 3]))
                .flatten_all()
                .sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_autograd_layer_norm_rms_norm_rank4() {
    for device in test_devices().await {
        let data: Vec<f32> = (0..12).map(|i| (i as f32 * 0.83).sin() * 2.0).collect();
        let w_data = [0.5f32, 1.5, 2.0];
        let b_data = [0.1f32, -0.2, 0.3];
        let eps = 1e-5;

        let graph = Graph::new();
        let x: Tensor<4> = Tensor::from_slice(&graph, &device, [1, 2, 2, 3], &data);
        let w: Tensor<4> = Tensor::from_slice(&graph, &device, [1, 1, 1, 3], &w_data);
        let b: Tensor<4> = Tensor::from_slice(&graph, &device, [1, 1, 1, 3], &b_data);
        let output = x.layer_norm(&w, Some(&b), eps, true);

        let raw_x = RawTensor::from_slice(&device, [1, 2, 2, 3], &data);
        let raw_w = RawTensor::from_slice(&device, [3], &w_data);
        let raw_b = RawTensor::from_slice(&device, [3], &b_data);
        let expected = raw_x.layer_norm_last_dim_fused::<3, 1, _, _>(&raw_w, Some(&raw_b), eps);
        assert_slice_close(
            &flatten(output.raw().clone()).await,
            &flatten(expected).await,
        );

        let fd_device = device.clone();
        assert_gradient_matches_finite_difference(
            &device,
            [1, 2, 2, 3],
            &data,
            move |graph, x| {
                let w = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [1, 1, 1, 3], &[0.5f32, 1.5, 2.0]),
                );
                let b = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [1, 1, 1, 3], &[0.1f32, -0.2, 0.3]),
                );
                let out = x.layer_norm(&w, Some(&b), eps, true);
                out.mul(&composite_ramp(graph, &fd_device, out.shape()))
                    .flatten_all()
                    .sum()
            },
        )
        .await;

        let rms = x.rms_norm(&w, eps);
        let expected_rms = RawTensor::from_slice(&device, [1, 2, 2, 3], &data)
            .rms_norm_fused::<1, 3>(&raw_w, None, eps);
        assert_slice_close(
            &flatten(rms.raw().clone()).await,
            &flatten(expected_rms).await,
        );

        let fd_device = device.clone();
        assert_gradient_matches_finite_difference(
            &device,
            [1, 2, 2, 3],
            &data,
            move |graph, x| {
                let w = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [1, 1, 1, 3], &[0.5f32, 1.5, 2.0]),
                );
                let out = x.rms_norm(&w, eps);
                out.mul(&composite_ramp(graph, &fd_device, out.shape()))
                    .flatten_all()
                    .sum()
            },
        )
        .await;
    }
}

#[tokio::test]
async fn test_autograd_rms_norm_residual_fused() {
    for device in test_devices().await {
        let x_data: Vec<f32> = (0..12).map(|i| (i as f32 * 0.29).sin()).collect();
        let r_data: Vec<f32> = (0..12).map(|i| (i as f32 * 0.61).cos()).collect();
        let w_data = [0.5f32, 1.5, 2.0];
        let b_data = [0.1f32, -0.2, 0.3];
        let eps = 1e-5;

        let graph = Graph::new();
        let x: Tensor<3> = Tensor::from_slice(&graph, &device, [2, 2, 3], &x_data);
        let r: Tensor<3> = Tensor::from_slice(&graph, &device, [2, 2, 3], &r_data);
        let w: Tensor<1> = Tensor::from_slice(&graph, &device, [3], &w_data);
        let b: Tensor<1> = Tensor::from_slice(&graph, &device, [3], &b_data);

        let raw_x = RawTensor::from_slice(&device, [2, 2, 3], &x_data);
        let raw_r = RawTensor::from_slice(&device, [2, 2, 3], &r_data);
        let raw_w = RawTensor::from_slice(&device, [3], &w_data);
        let raw_b = RawTensor::from_slice(&device, [3], &b_data);

        let output = x.rms_norm_residual_fused(&r, &w, Some(&b), eps);
        let expected = raw_x.rms_norm_residual_fused::<1, 2, _>(&raw_r, &raw_w, Some(&raw_b), eps);
        assert_slice_close(
            &flatten(output.raw().clone()).await,
            &flatten(expected).await,
        );

        let no_bias = x.rms_norm_residual_fused(&r, &w, None, eps);
        let expected_no_bias = RawTensor::from_slice(&device, [2, 2, 3], &x_data)
            .rms_norm_residual_fused::<1, 2, _>(&raw_r, &raw_w, None, eps);
        assert_slice_close(
            &flatten(no_bias.raw().clone()).await,
            &flatten(expected_no_bias).await,
        );

        let gradients = output.flatten_all().sum().backward().unwrap();
        assert!(gradients.get(&x).is_some());
        assert!(gradients.get(&r).is_some());
        assert!(gradients.get(&w).is_some());
        assert!(gradients.get(&b).is_some());

        let fd_device = device.clone();
        let r_fd = r_data.clone();
        assert_gradient_matches_finite_difference(
            &device,
            [2, 2, 3],
            &x_data,
            move |graph, x| {
                let r = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [2, 2, 3], &r_fd),
                );
                let w = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [3], &[0.5f32, 1.5, 2.0]),
                );
                let b = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [3], &[0.1f32, -0.2, 0.3]),
                );
                let out = x.rms_norm_residual_fused(&r, &w, Some(&b), eps);
                out.mul(&composite_ramp(graph, &fd_device, out.shape()))
                    .flatten_all()
                    .sum()
            },
        )
        .await;

        let fd_device = device.clone();
        let x_fd = x_data.clone();
        assert_gradient_matches_finite_difference(
            &device,
            [2, 2, 3],
            &r_data,
            move |graph, r| {
                let x = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [2, 2, 3], &x_fd),
                );
                let w = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [3], &[0.5f32, 1.5, 2.0]),
                );
                let out = x.rms_norm_residual_fused(&r, &w, None, eps);
                out.mul(&composite_ramp(graph, &fd_device, out.shape()))
                    .flatten_all()
                    .sum()
            },
        )
        .await;

        let fd_device = device.clone();
        let x_fd = x_data.clone();
        let r_fd = r_data.clone();
        assert_gradient_matches_finite_difference(&device, [3], &w_data, move |graph, w| {
            let x = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [2, 2, 3], &x_fd),
            );
            let r = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [2, 2, 3], &r_fd),
            );
            let b = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [3], &[0.1f32, -0.2, 0.3]),
            );
            let out = x.rms_norm_residual_fused(&r, &w, Some(&b), eps);
            out.mul(&composite_ramp(graph, &fd_device, out.shape()))
                .flatten_all()
                .sum()
        })
        .await;

        let fd_device = device.clone();
        let x_fd = x_data.clone();
        let r_fd = r_data.clone();
        assert_gradient_matches_finite_difference(&device, [3], &b_data, move |graph, b| {
            let x = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [2, 2, 3], &x_fd),
            );
            let r = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [2, 2, 3], &r_fd),
            );
            let w = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [3], &[0.5f32, 1.5, 2.0]),
            );
            let out = x.rms_norm_residual_fused(&r, &w, Some(&b), eps);
            out.mul(&composite_ramp(graph, &fd_device, out.shape()))
                .flatten_all()
                .sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_autograd_rope_pair_fused() {
    for device in test_devices().await {
        let q_data: Vec<f32> = (0..24).map(|i| (i as f32 * 0.37).sin()).collect();
        let k_data: Vec<f32> = (0..12).map(|i| (i as f32 * 0.53).cos()).collect();
        let cos_data: Vec<f32> = (0..6).map(|i| (i as f32 * 0.7).cos()).collect();
        let sin_data: Vec<f32> = (0..6).map(|i| (i as f32 * 0.7).sin()).collect();

        let graph = Graph::new();
        let q: Tensor<4> = Tensor::from_slice(&graph, &device, [1, 2, 3, 4], &q_data);
        let k: Tensor<4> = Tensor::from_slice(&graph, &device, [1, 1, 3, 4], &k_data);
        let cos: Tensor<2> =
            Tensor::constant_from_raw(&graph, RawTensor::from_slice(&device, [3, 2], &cos_data));
        let sin: Tensor<2> =
            Tensor::constant_from_raw(&graph, RawTensor::from_slice(&device, [3, 2], &sin_data));
        let (q_out, k_out) = q.rope_pair_fused(&k, &cos, &sin);

        let raw_q = RawTensor::from_slice(&device, [1, 2, 3, 4], &q_data);
        let raw_k = RawTensor::from_slice(&device, [1, 1, 3, 4], &k_data);
        let raw_cos = RawTensor::from_slice(&device, [3, 2], &cos_data);
        let raw_sin = RawTensor::from_slice(&device, [3, 2], &sin_data);
        let (expected_q, expected_k) = raw_q.rope_pair_fused(&raw_k, &raw_cos, &raw_sin);
        assert_slice_close(
            &flatten(q_out.raw().clone()).await,
            &flatten(expected_q).await,
        );
        assert_slice_close(
            &flatten(k_out.raw().clone()).await,
            &flatten(expected_k).await,
        );

        let (normal_q, normal_k) = q.rope_normal_pair_fused(&k, &cos, &sin);
        let (expected_nq, expected_nk) = RawTensor::from_slice(&device, [1, 2, 3, 4], &q_data)
            .rope_normal_pair_fused(&raw_k, &raw_cos, &raw_sin);
        assert_slice_close(
            &flatten(normal_q.raw().clone()).await,
            &flatten(expected_nq).await,
        );
        assert_slice_close(
            &flatten(normal_k.raw().clone()).await,
            &flatten(expected_nk).await,
        );

        let fd_device = device.clone();
        let k_fd = k_data.clone();
        let cos_fd = cos_data.clone();
        let sin_fd = sin_data.clone();
        assert_gradient_matches_finite_difference(
            &device,
            [1, 2, 3, 4],
            &q_data,
            move |graph, q| {
                let k = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [1, 1, 3, 4], &k_fd),
                );
                let cos = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [3, 2], &cos_fd),
                );
                let sin = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [3, 2], &sin_fd),
                );
                let (q_out, k_out) = q.rope_pair_fused(&k, &cos, &sin);
                let q_loss = q_out
                    .mul(&composite_ramp(graph, &fd_device, q_out.shape()))
                    .flatten_all()
                    .sum();
                q_loss.add(&k_out.flatten_all().sum())
            },
        )
        .await;

        let fd_device = device.clone();
        let q_fd = q_data.clone();
        let cos_fd = cos_data.clone();
        let sin_fd = sin_data.clone();
        assert_gradient_matches_finite_difference(
            &device,
            [1, 1, 3, 4],
            &k_data,
            move |graph, k| {
                let q = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [1, 2, 3, 4], &q_fd),
                );
                let cos = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [3, 2], &cos_fd),
                );
                let sin = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [3, 2], &sin_fd),
                );
                let (_, k_out) = q.rope_normal_pair_fused(&k, &cos, &sin);
                k_out
                    .mul(&composite_ramp(graph, &fd_device, k_out.shape()))
                    .flatten_all()
                    .sum()
            },
        )
        .await;
    }
}

#[tokio::test]
async fn test_autograd_rope_cache_forward() {
    for device in test_devices().await {
        let q_data: Vec<f32> = (0..24).map(|i| (i as f32 * 0.37).sin()).collect();
        let k_data: Vec<f32> = (0..24).map(|i| (i as f32 * 0.53).cos()).collect();
        let cache = crate::RopeCache::new(4, 8, 10000.0, &device).unwrap();

        let graph = Graph::new();
        let q: Tensor<4> = Tensor::from_slice(&graph, &device, [1, 2, 3, 4], &q_data);
        let k: Tensor<4> = Tensor::from_slice(&graph, &device, [1, 2, 3, 4], &k_data);

        let raw_q = RawTensor::from_slice(&device, [1, 2, 3, 4], &q_data);
        let raw_k = RawTensor::from_slice(&device, [1, 2, 3, 4], &k_data);

        let (q_out, k_out) = q.rope_cache_forward(&k, &cache, 2);
        let (expected_q, expected_k) = cache.forward(&raw_q, &raw_k, 2);
        assert_slice_close(
            &flatten(q_out.raw().clone()).await,
            &flatten(expected_q).await,
        );
        assert_slice_close(
            &flatten(k_out.raw().clone()).await,
            &flatten(expected_k).await,
        );

        let (qi_out, ki_out) = q.rope_cache_forward_interleaved(&k, &cache, 2);
        let (expected_qi, expected_ki) = cache.forward_interleaved(
            &RawTensor::from_slice(&device, [1, 2, 3, 4], &q_data),
            &RawTensor::from_slice(&device, [1, 2, 3, 4], &k_data),
            2,
        );
        assert_slice_close(
            &flatten(qi_out.raw().clone()).await,
            &flatten(expected_qi).await,
        );
        assert_slice_close(
            &flatten(ki_out.raw().clone()).await,
            &flatten(expected_ki).await,
        );

        let fd_device = device.clone();
        let k_fd = k_data.clone();
        let fd_cache = cache.clone();
        assert_gradient_matches_finite_difference(
            &device,
            [1, 2, 3, 4],
            &q_data,
            move |graph, q| {
                let k = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [1, 2, 3, 4], &k_fd),
                );
                let (q_out, _) = q.rope_cache_forward(&k, &fd_cache, 2);
                q_out
                    .mul(&composite_ramp(graph, &fd_device, q_out.shape()))
                    .flatten_all()
                    .sum()
            },
        )
        .await;
    }
}

#[tokio::test]
async fn test_autograd_ones() {
    for device in test_devices().await {
        let graph = Graph::new();

        let ones: Tensor<1> = Tensor::ones(&graph, &device, [3]);
        let forward = ones.raw().clone().as_slice().await.unwrap().to_vec();
        assert_eq!(forward, vec![1.0, 1.0, 1.0]);

        let x: Tensor<1> = Tensor::new(&graph, &device, &[2.0f32, -3.0, 4.0]);
        let loss = x.mul(&ones).sum();
        assert_close(loss.raw().to_scalar().await.unwrap(), 3.0);
        let gradients = loss.backward().unwrap();
        let dx = gradients
            .get(&x)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        assert_slice_close(&dx, &[1.0, 1.0, 1.0]);
        let dones = gradients
            .get(&ones)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        assert_slice_close(&dones, &[2.0, -3.0, 4.0]);
    }
}

#[tokio::test]
async fn test_autograd_ones_like() {
    for device in test_devices().await {
        let graph = Graph::new();

        let x: Tensor<2> = Tensor::new(&graph, &device, &[[1.0f32, 2.0], [3.0, 4.0]]);
        let ones = x.ones_like();
        assert_eq!(ones.shape(), [2, 2]);
        assert_eq!(flatten(ones.raw().clone()).await, vec![1.0, 1.0, 1.0, 1.0]);

        let loss = x.mul(&ones).flatten_all().sum();
        assert_close(loss.raw().to_scalar().await.unwrap(), 10.0);
        let gradients = loss.backward().unwrap();
        let dx = flatten(gradients.get(&x).unwrap()).await;
        assert_slice_close(&dx, &[1.0, 1.0, 1.0, 1.0]);
        let dones = flatten(gradients.get(&ones).unwrap()).await;
        assert_slice_close(&dones, &[1.0, 2.0, 3.0, 4.0]);
    }
}

#[tokio::test]
async fn test_backward_mat_mul_rank4() {
    for device in test_devices().await {
        let graph = Graph::new();
        let lhs_data = (1..=24).map(|n| n as f32).collect::<Vec<_>>();
        let rhs_data = (1..=24).map(|n| n as f32).collect::<Vec<_>>();
        let lhs: Tensor<4> = Tensor::from_slice(&graph, &device, [2, 2, 2, 3], &lhs_data);
        let rhs: Tensor<4> = Tensor::from_slice(&graph, &device, [2, 2, 3, 2], &rhs_data);

        let output = lhs.mat_mul(&rhs);
        let output_values = output.raw().clone().as_slice().await.unwrap();
        let gradients = output.flatten_all().sum().backward().unwrap();
        let dlhs = flatten(gradients.get(&lhs).unwrap()).await;
        let drhs = flatten(gradients.get(&rhs).unwrap()).await;

        assert_eq!(output_values.shape(), &[2, 2, 2, 2]);
        assert_close(output_values[[0, 0, 0, 0]], 22.0);
        assert_close(output_values[[1, 1, 1, 1]], 1522.0);

        // with an all-ones seed, dlhs[b, i, k] = sum_j rhs[b, k, j] and
        // drhs[b, k, j] = sum_i lhs[b, i, k]
        for batch in 0..4 {
            for i in 0..2 {
                for k in 0..3 {
                    let expected = (0..2)
                        .map(|j| rhs_data[batch * 6 + k * 2 + j])
                        .sum::<f32>();
                    assert_close(dlhs[batch * 6 + i * 3 + k], expected);
                }
            }
            for k in 0..3 {
                for j in 0..2 {
                    let expected = (0..2)
                        .map(|i| lhs_data[batch * 6 + i * 3 + k])
                        .sum::<f32>();
                    assert_close(drhs[batch * 6 + k * 2 + j], expected);
                }
            }
        }

        let lhs_small = lhs_data
            .iter()
            .map(|value| value * 0.05)
            .collect::<Vec<_>>();
        let rhs_small = rhs_data
            .iter()
            .map(|value| value * 0.03)
            .collect::<Vec<_>>();
        let fd_device = device.clone();
        let fd_rhs = rhs_small.clone();
        assert_gradient_matches_finite_difference(
            &device,
            [2, 2, 2, 3],
            &lhs_small,
            move |graph, lhs| {
                let rhs = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [2, 2, 3, 2], &fd_rhs),
                );
                lhs.mat_mul(&rhs).sqr().flatten_all().sum()
            },
        )
        .await;
        let fd_device = device.clone();
        let fd_lhs = lhs_small.clone();
        assert_gradient_matches_finite_difference(
            &device,
            [2, 2, 3, 2],
            &rhs_small,
            move |graph, rhs| {
                let lhs = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [2, 2, 2, 3], &fd_lhs),
                );
                lhs.mat_mul(&rhs).sqr().flatten_all().sum()
            },
        )
        .await;
    }
}

#[tokio::test]
async fn test_autograd_std_ops_add_sub() {
    for device in test_devices().await {
        let graph = Graph::new();
        let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
        let y: Tensor<1> = Tensor::new(&graph, &device, &[4.0f32, -5.0, 8.0]);

        for add in [
            &x + &y,
            x.clone() + y.clone(),
            &x + y.clone(),
            x.clone() + &y,
        ] {
            let values = add.raw().clone().as_slice().await.unwrap().to_vec();
            assert_slice_close(&values, &[5.0, -3.0, 11.0]);
        }
        for sub in [
            &x - &y,
            x.clone() - y.clone(),
            &x - y.clone(),
            x.clone() - &y,
        ] {
            let values = sub.raw().clone().as_slice().await.unwrap().to_vec();
            assert_slice_close(&values, &[-3.0, 7.0, -5.0]);
        }

        let loss = ((&x + &y) * (&x - &y)).sum();
        let gradients = loss.backward().unwrap();
        let dx = gradients
            .get(&x)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        let dy = gradients
            .get(&y)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        assert_slice_close(&dx, &[2.0, 4.0, 6.0]);
        assert_slice_close(&dy, &[-8.0, 10.0, -16.0]);

        assert_gradient_matches_finite_difference(&device, [3], &[1.0, 2.0, 3.0], |graph, x| {
            let y = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&x.device(), [3], &[4.0, -5.0, 8.0]),
            );
            ((&x + &y) * (&x - &y)).sum()
        })
        .await;
        assert_gradient_matches_finite_difference(&device, [3], &[4.0, -5.0, 8.0], |graph, y| {
            let x = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&y.device(), [3], &[1.0, 2.0, 3.0]),
            );
            ((&x + &y) * (&x - &y)).sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_autograd_std_ops_mul_div() {
    for device in test_devices().await {
        let graph = Graph::new();
        let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);
        let y: Tensor<1> = Tensor::new(&graph, &device, &[4.0f32, -5.0, 8.0]);

        for mul in [
            &x * &y,
            x.clone() * y.clone(),
            &x * y.clone(),
            x.clone() * &y,
        ] {
            let values = mul.raw().clone().as_slice().await.unwrap().to_vec();
            assert_slice_close(&values, &[4.0, -10.0, 24.0]);
        }
        for div in [
            &x / &y,
            x.clone() / y.clone(),
            &x / y.clone(),
            x.clone() / &y,
        ] {
            let values = div.raw().clone().as_slice().await.unwrap().to_vec();
            assert_slice_close(&values, &[0.25, -0.4, 0.375]);
        }

        let loss = ((&x * &y) + (&x / &y)).sum();
        let gradients = loss.backward().unwrap();
        let dx = gradients
            .get(&x)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        let dy = gradients
            .get(&y)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        assert_slice_close(&dx, &[4.25, -5.2, 8.125]);
        assert_slice_close(&dy, &[0.9375, 1.92, 2.953125]);

        assert_gradient_matches_finite_difference(&device, [3], &[1.0, 2.0, 3.0], |graph, x| {
            let y = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&x.device(), [3], &[4.0, -5.0, 8.0]),
            );
            ((&x * &y) + (&x / &y)).sum()
        })
        .await;
        assert_gradient_matches_finite_difference(&device, [3], &[4.0, -5.0, 8.0], |graph, y| {
            let x = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&y.device(), [3], &[1.0, 2.0, 3.0]),
            );
            ((&x * &y) + (&x / &y)).sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_autograd_std_ops_neg() {
    for device in test_devices().await {
        let graph = Graph::new();
        let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, -2.0, 3.0]);

        for neg in [-&x, -x.clone()] {
            let values = neg.raw().clone().as_slice().await.unwrap().to_vec();
            assert_slice_close(&values, &[-1.0, 2.0, -3.0]);
        }

        let loss = ((-&x) * &x).sum();
        let gradients = loss.backward().unwrap();
        let dx = gradients
            .get(&x)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        assert_slice_close(&dx, &[-2.0, 4.0, -6.0]);

        assert_gradient_matches_finite_difference(&device, [3], &[1.0, -2.0, 3.0], |_, x| {
            ((-&x) * &x).sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_autograd_std_ops_scalar() {
    for device in test_devices().await {
        let graph = Graph::new();
        let x: Tensor<1> = Tensor::new(&graph, &device, &[1.0f32, 2.0, 3.0]);

        for mul in [&x * 2.5, x.clone() * 2.5] {
            let values = mul.raw().clone().as_slice().await.unwrap().to_vec();
            assert_slice_close(&values, &[2.5, 5.0, 7.5]);
        }
        for add in [&x + 1.5, x.clone() + 1.5] {
            let values = add.raw().clone().as_slice().await.unwrap().to_vec();
            assert_slice_close(&values, &[2.5, 3.5, 4.5]);
        }
        for sub in [&x - 0.5, x.clone() - 0.5] {
            let values = sub.raw().clone().as_slice().await.unwrap().to_vec();
            assert_slice_close(&values, &[0.5, 1.5, 2.5]);
        }
        for div in [&x / 2.0, x.clone() / 2.0] {
            let values = div.raw().clone().as_slice().await.unwrap().to_vec();
            assert_slice_close(&values, &[0.5, 1.0, 1.5]);
        }

        let loss = ((&x * 3.0) + 2.0).sum();
        let gradients = loss.backward().unwrap();
        let dx = gradients
            .get(&x)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        assert_slice_close(&dx, &[3.0, 3.0, 3.0]);

        assert_gradient_matches_finite_difference(&device, [3], &[1.0, 2.0, 3.0], |_, x| {
            ((&x * 3.0) + 2.0).sum()
        })
        .await;

        assert_gradient_matches_finite_difference(&device, [3], &[1.0, 2.0, 3.0], |_, x| {
            ((&x - 1.5) / 4.0).sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_autograd_layer_norm_rank_generic() {
    for device in test_devices().await {
        let eps = 1e-5f32;
        let x_rows = [[1.0f32, 2.0, 4.0], [-1.0, 0.5, 3.0]];
        let x_data = [1.0f32, 2.0, 4.0, -1.0, 0.5, 3.0];
        let w_data = [0.5f32, 1.0, 1.5];
        let b_data = [0.1f32, -0.2, 0.3];
        for remove_mean in [true, false] {
            let graph = Graph::new();
            let x: Tensor<2> = Tensor::from_slice(&graph, &device, [2, 3], &x_data);
            let w: Tensor<2> = Tensor::from_slice(&graph, &device, [1, 3], &w_data);
            let b: Tensor<2> = Tensor::from_slice(&graph, &device, [1, 3], &b_data);

            let output = x.layer_norm(&w, Some(&b), eps, remove_mean);
            let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
            for (row_index, row) in x_rows.iter().enumerate() {
                let mean = if remove_mean {
                    row.iter().sum::<f32>() / 3.0
                } else {
                    0.0
                };
                let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 3.0;
                let std = (var + eps).sqrt();
                for column in 0..3 {
                    let expected = (row[column] - mean) / std * w_data[column] + b_data[column];
                    assert_close(output_values[row_index][column], expected);
                }
            }

            let fd_device = device.clone();
            assert_gradient_matches_finite_difference(
                &device,
                [2, 3],
                &x_data,
                move |graph, x| {
                    let w = Tensor::constant_from_raw(
                        graph,
                        RawTensor::from_slice(&fd_device, [1, 3], &w_data),
                    );
                    let b = Tensor::constant_from_raw(
                        graph,
                        RawTensor::from_slice(&fd_device, [1, 3], &b_data),
                    );
                    x.layer_norm(&w, Some(&b), eps, remove_mean)
                        .flatten_all()
                        .sum()
                },
            )
            .await;

            let fd_device = device.clone();
            assert_gradient_matches_finite_difference(
                &device,
                [1, 3],
                &w_data,
                move |graph, w| {
                    let x = Tensor::constant_from_raw(
                        graph,
                        RawTensor::from_slice(&fd_device, [2, 3], &x_data),
                    );
                    let b = Tensor::constant_from_raw(
                        graph,
                        RawTensor::from_slice(&fd_device, [1, 3], &b_data),
                    );
                    x.layer_norm(&w, Some(&b), eps, remove_mean)
                        .flatten_all()
                        .sum()
                },
            )
            .await;

            let fd_device = device.clone();
            assert_gradient_matches_finite_difference(
                &device,
                [1, 3],
                &b_data,
                move |graph, b| {
                    let x = Tensor::constant_from_raw(
                        graph,
                        RawTensor::from_slice(&fd_device, [2, 3], &x_data),
                    );
                    let w = Tensor::constant_from_raw(
                        graph,
                        RawTensor::from_slice(&fd_device, [1, 3], &w_data),
                    );
                    x.layer_norm(&w, Some(&b), eps, remove_mean)
                        .flatten_all()
                        .sum()
                },
            )
            .await;
        }
    }
}

#[tokio::test]
async fn test_autograd_rms_norm_rank_generic_weight() {
    for device in test_devices().await {
        let eps = 1e-5f32;
        let x_rows = [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let x_data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w_data = [0.5f32, 1.0, 1.5];
        let graph = Graph::new();
        let x: Tensor<2> = Tensor::from_slice(&graph, &device, [2, 3], &x_data);
        let w: Tensor<2> = Tensor::from_slice(&graph, &device, [1, 3], &w_data);

        let output = x.rms_norm(&w, eps);
        let output_values = output.raw().clone().as_slice().await.unwrap().to_vec();
        for (row_index, row) in x_rows.iter().enumerate() {
            let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / 3.0;
            let rms = (mean_sq + eps).sqrt();
            for column in 0..3 {
                assert_close(
                    output_values[row_index][column],
                    row[column] / rms * w_data[column],
                );
            }
        }

        let gradients = output.flatten_all().sum().backward().unwrap();
        let dw = gradients
            .get(&w)
            .unwrap()
            .as_slice()
            .await
            .unwrap()
            .to_vec();
        for column in 0..3 {
            let mut expected = 0.0f32;
            for row in x_rows.iter() {
                let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / 3.0;
                expected += row[column] / (mean_sq + eps).sqrt();
            }
            assert_close(dw[0][column], expected);
        }

        let fd_device = device.clone();
        assert_gradient_matches_finite_difference(&device, [2, 3], &x_data, move |graph, x| {
            let w = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [1, 3], &w_data),
            );
            x.rms_norm(&w, eps).flatten_all().sum()
        })
        .await;

        let fd_device = device.clone();
        assert_gradient_matches_finite_difference(&device, [1, 3], &w_data, move |graph, w| {
            let x = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [2, 3], &x_data),
            );
            x.rms_norm(&w, eps).flatten_all().sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_autograd_rms_norm_fused_weight_rank() {
    for device in test_devices().await {
        let eps = 1e-5f32;
        let x_rows = [[1.0f32, -2.0, 3.0], [0.5, 4.0, -1.5]];
        let x_data = [1.0f32, -2.0, 3.0, 0.5, 4.0, -1.5];
        let w_data = [0.5f32, 1.0, 1.5];
        let b_data = [0.1f32, -0.2, 0.3];
        let graph = Graph::new();
        let x: Tensor<2> = Tensor::from_slice(&graph, &device, [2, 3], &x_data);
        let w1: Tensor<1> = Tensor::from_slice(&graph, &device, [3], &w_data);
        let b1: Tensor<1> = Tensor::from_slice(&graph, &device, [3], &b_data);
        let w2: Tensor<2> = Tensor::from_slice(&graph, &device, [1, 3], &w_data);

        let biased = x.rms_norm_fused::<1, 1>(&w1, Some(&b1), eps);
        let biased_values = biased.raw().clone().as_slice().await.unwrap().to_vec();
        let no_bias = x.rms_norm_fused_no_bias::<2, 1>(&w2, eps);
        let no_bias_values = no_bias.raw().clone().as_slice().await.unwrap().to_vec();
        for (row_index, row) in x_rows.iter().enumerate() {
            let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / 3.0;
            let rms = (mean_sq + eps).sqrt();
            for column in 0..3 {
                let scaled = row[column] / rms * w_data[column];
                assert_close(biased_values[row_index][column], scaled + b_data[column]);
                assert_close(no_bias_values[row_index][column], scaled);
            }
        }

        let fd_device = device.clone();
        assert_gradient_matches_finite_difference(&device, [2, 3], &x_data, move |graph, x| {
            let w = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [3], &w_data),
            );
            let b = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [3], &b_data),
            );
            x.rms_norm_fused::<1, 1>(&w, Some(&b), eps)
                .flatten_all()
                .sum()
        })
        .await;

        let fd_device = device.clone();
        assert_gradient_matches_finite_difference(&device, [1, 3], &w_data, move |graph, w| {
            let x = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [2, 3], &x_data),
            );
            x.rms_norm_fused_no_bias::<2, 1>(&w, eps).flatten_all().sum()
        })
        .await;

        let fd_device = device.clone();
        assert_gradient_matches_finite_difference(&device, [3], &b_data, move |graph, b| {
            let x = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [2, 3], &x_data),
            );
            let w = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [3], &w_data),
            );
            x.rms_norm_fused::<1, 1>(&w, Some(&b), eps)
                .flatten_all()
                .sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_autograd_rms_norm_residual_fused_weight_rank() {
    for device in test_devices().await {
        let eps = 1e-5f32;
        let x_data = [1.0f32, -2.0, 3.0, 0.5, 4.0, -1.5];
        let r_data = [0.5f32, 1.5, -1.0, 2.0, -0.5, 1.0];
        let w_data = [0.5f32, 1.0, 1.5];
        let b_data = [0.1f32, -0.2, 0.3];
        let graph = Graph::new();
        let x: Tensor<2> = Tensor::from_slice(&graph, &device, [2, 3], &x_data);
        let r: Tensor<2> = Tensor::from_slice(&graph, &device, [2, 3], &r_data);
        let w: Tensor<2> = Tensor::from_slice(&graph, &device, [1, 3], &w_data);
        let b: Tensor<2> = Tensor::from_slice(&graph, &device, [1, 3], &b_data);

        let biased = x.rms_norm_residual_fused::<2, 1>(&r, &w, Some(&b), eps);
        let biased_values = biased.raw().clone().as_slice().await.unwrap().to_vec();
        let no_bias = x.rms_norm_residual_fused::<2, 1>(&r, &w, None, eps);
        let no_bias_values = no_bias.raw().clone().as_slice().await.unwrap().to_vec();
        for row_index in 0..2 {
            let combined: Vec<f32> = (0..3)
                .map(|column| x_data[row_index * 3 + column] + r_data[row_index * 3 + column])
                .collect();
            let mean_sq = combined.iter().map(|v| v * v).sum::<f32>() / 3.0;
            let rms = (mean_sq + eps).sqrt();
            for column in 0..3 {
                let scaled = combined[column] / rms * w_data[column];
                assert_close(biased_values[row_index][column], scaled + b_data[column]);
                assert_close(no_bias_values[row_index][column], scaled);
            }
        }

        // Input/residual gradients are covered by
        // test_autograd_rms_norm_residual_fused; only the rank-2 weight/bias
        // path is new here.
        let fd_device = device.clone();
        assert_gradient_matches_finite_difference(&device, [1, 3], &w_data, move |graph, w| {
            let x = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [2, 3], &x_data),
            );
            let r = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [2, 3], &r_data),
            );
            let b = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [1, 3], &b_data),
            );
            x.rms_norm_residual_fused::<2, 1>(&r, &w, Some(&b), eps)
                .flatten_all()
                .sum()
        })
        .await;

        let fd_device = device.clone();
        assert_gradient_matches_finite_difference(&device, [1, 3], &b_data, move |graph, b| {
            let x = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [2, 3], &x_data),
            );
            let r = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [2, 3], &r_data),
            );
            let w = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [1, 3], &w_data),
            );
            x.rms_norm_residual_fused::<2, 1>(&r, &w, Some(&b), eps)
                .flatten_all()
                .sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_autograd_layer_norm_last_dim_fused_weight_rank() {
    for device in test_devices().await {
        let eps = 1e-5f32;
        let x_rows = [[1.0f32, 2.0, 4.0], [-1.0, 0.5, 3.0]];
        let x_data = [1.0f32, 2.0, 4.0, -1.0, 0.5, 3.0];
        let w_data = [0.5f32, 1.0, 1.5];
        let b_data = [0.1f32, -0.2, 0.3];
        let graph = Graph::new();
        let x: Tensor<2> = Tensor::from_slice(&graph, &device, [2, 3], &x_data);
        let w1: Tensor<1> = Tensor::from_slice(&graph, &device, [3], &w_data);
        let b1: Tensor<1> = Tensor::from_slice(&graph, &device, [3], &b_data);
        let w2: Tensor<2> = Tensor::from_slice(&graph, &device, [1, 3], &w_data);
        let b2: Tensor<2> = Tensor::from_slice(&graph, &device, [1, 3], &b_data);

        let rank1 = x.layer_norm_last_dim_fused::<1, 1>(&w1, Some(&b1), eps);
        let rank1_values = rank1.raw().clone().as_slice().await.unwrap().to_vec();
        let rank2 = x.layer_norm_last_dim_fused::<1, 2>(&w2, Some(&b2), eps);
        let rank2_values = rank2.raw().clone().as_slice().await.unwrap().to_vec();
        for (row_index, row) in x_rows.iter().enumerate() {
            let mean = row.iter().sum::<f32>() / 3.0;
            let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / 3.0;
            let std = (var + eps).sqrt();
            for column in 0..3 {
                let expected = (row[column] - mean) / std * w_data[column] + b_data[column];
                assert_close(rank1_values[row_index][column], expected);
                assert_close(rank2_values[row_index][column], expected);
            }
        }

        let fd_device = device.clone();
        assert_gradient_matches_finite_difference(&device, [2, 3], &x_data, move |graph, x| {
            let w = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [1, 3], &w_data),
            );
            let b = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [1, 3], &b_data),
            );
            x.layer_norm_last_dim_fused::<1, 2>(&w, Some(&b), eps)
                .flatten_all()
                .sum()
        })
        .await;

        let fd_device = device.clone();
        assert_gradient_matches_finite_difference(&device, [1, 3], &w_data, move |graph, w| {
            let x = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [2, 3], &x_data),
            );
            x.layer_norm_last_dim_fused::<1, 2>(&w, None, eps)
                .flatten_all()
                .sum()
        })
        .await;

        let fd_device = device.clone();
        assert_gradient_matches_finite_difference(&device, [1, 3], &b_data, move |graph, b| {
            let x = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [2, 3], &x_data),
            );
            let w = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [1, 3], &w_data),
            );
            x.layer_norm_last_dim_fused::<1, 2>(&w, Some(&b), eps)
                .flatten_all()
                .sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_autograd_layer_linear_forward_matches_inference() {
    for device in test_devices().await {
        let weight_data = [
            0.5f32, -1.0, 0.25, 2.0, 1.5, -0.75, 0.1, 0.4, -0.2, 0.9, -1.3, 0.6,
        ];
        let bias_data = [0.3f32, -0.6, 1.1];
        let input_data = [
            1.0f32, -2.0, 0.5, 0.25, 0.75, 1.5, -0.5, 2.0, -1.25, 0.4, 0.8, -0.3, 0.15, -0.9, 1.2,
            0.7,
        ];
        let weight_bytes: Vec<u8> = weight_data
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let qweight = |device: &Device| {
            crate::QMatrix::from_raw_bytes(device, [3, 4], &weight_bytes, fusor_gguf::GgmlType::F32)
                .unwrap()
        };
        let raw_bias = RawTensor::from_slice(&device, [3], &bias_data);
        let inference = crate::layers::Linear::new(qweight(&device), Some(raw_bias));
        let inference_no_bias = crate::layers::Linear::<f32>::new(qweight(&device), None);

        let graph = Graph::new();
        let layer = layers::Linear::new(
            Tensor::from_slice(&graph, &device, [3, 4], &weight_data),
            Some(Tensor::from_slice(&graph, &device, [3], &bias_data)),
        );
        assert_eq!(layer.in_features(), 4);
        assert_eq!(layer.out_features(), 3);

        let raw_input = RawTensor::from_slice(&device, [2, 2, 4], &input_data);
        let input = Tensor::constant_from_raw(&graph, raw_input.clone());
        let output = layer.forward(&input);
        assert_eq!(output.shape(), [2, 2, 3]);
        let expected = flatten(inference.forward(&raw_input)).await;
        assert_slice_close(&flatten(output.raw().clone()).await, &expected);

        let raw_input_2d = RawTensor::from_slice(&device, [4, 4], &input_data);
        let input_2d = Tensor::constant_from_raw(&graph, raw_input_2d.clone());
        let output_2d = layer.forward(&input_2d);
        assert_eq!(output_2d.shape(), [4, 3]);
        let expected_2d = flatten(inference.forward(&raw_input_2d)).await;
        assert_slice_close(&flatten(output_2d.raw().clone()).await, &expected_2d);

        let layer_no_bias = layers::Linear::new(
            Tensor::from_slice(&graph, &device, [3, 4], &weight_data),
            None,
        );
        let output_no_bias = layer_no_bias.forward(&input);
        let expected_no_bias = flatten(inference_no_bias.forward(&raw_input)).await;
        assert_slice_close(
            &flatten(output_no_bias.raw().clone()).await,
            &expected_no_bias,
        );
    }
}

#[tokio::test]
async fn test_autograd_layer_linear_weight_gradient() {
    for device in test_devices().await {
        let weight_data = [
            0.5f32, -1.0, 0.25, 2.0, 1.5, -0.75, 0.1, 0.4, -0.2, 0.9, -1.3, 0.6,
        ];
        let bias_data = [0.3f32, -0.6, 1.1];
        let input_data = [
            1.0f32, -2.0, 0.5, 0.25, 0.75, 1.5, -0.5, 2.0, -1.25, 0.4, 0.8, -0.3, 0.15, -0.9, 1.2,
            0.7,
        ];
        assert_gradient_matches_finite_difference(
            &device,
            [3, 4],
            &weight_data,
            |graph, weight| {
                let bias = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&device, [3], &bias_data),
                );
                let layer = layers::Linear::new(weight, Some(bias));
                let input = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&device, [2, 2, 4], &input_data),
                );
                layer.forward(&input).sqr().flatten_all().sum()
            },
        )
        .await;
    }
}

#[tokio::test]
async fn test_autograd_layer_linear_bias_gradient() {
    for device in test_devices().await {
        let weight_data = [
            0.5f32, -1.0, 0.25, 2.0, 1.5, -0.75, 0.1, 0.4, -0.2, 0.9, -1.3, 0.6,
        ];
        let bias_data = [0.3f32, -0.6, 1.1];
        let input_data = [
            1.0f32, -2.0, 0.5, 0.25, 0.75, 1.5, -0.5, 2.0, -1.25, 0.4, 0.8, -0.3, 0.15, -0.9, 1.2,
            0.7,
        ];
        assert_gradient_matches_finite_difference(&device, [3], &bias_data, |graph, bias| {
            let weight = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&device, [3, 4], &weight_data),
            );
            let layer = layers::Linear::new(weight, Some(bias));
            let input = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&device, [2, 2, 4], &input_data),
            );
            layer.forward(&input).sqr().flatten_all().sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_autograd_layer_linear_from_inference() {
    for device in test_devices().await {
        let weight_data = [
            0.5f32, -1.0, 0.25, 2.0, 1.5, -0.75, 0.1, 0.4, -0.2, 0.9, -1.3, 0.6,
        ];
        let bias_data = [0.3f32, -0.6, 1.1];
        let input_data = [
            1.0f32, -2.0, 0.5, 0.25, 0.75, 1.5, -0.5, 2.0, -1.25, 0.4, 0.8, -0.3, 0.15, -0.9, 1.2,
            0.7,
        ];
        let weight_bytes: Vec<u8> = weight_data
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let qweight = crate::QMatrix::from_raw_bytes(
            &device,
            [3, 4],
            &weight_bytes,
            fusor_gguf::GgmlType::F32,
        )
        .unwrap();
        let raw_bias = RawTensor::from_slice(&device, [3], &bias_data);
        let inference = crate::layers::Linear::new(qweight, Some(raw_bias));

        let graph = Graph::new();
        let layer = layers::Linear::from_inference(&graph, &inference);
        assert_eq!(layer.in_features(), 4);
        assert_eq!(layer.out_features(), 3);

        let raw_input = RawTensor::from_slice(&device, [2, 2, 4], &input_data);
        let input = Tensor::constant_from_raw(&graph, raw_input.clone());
        let output = layer.forward(&input);
        let expected = flatten(inference.forward(&raw_input)).await;
        assert_slice_close(&flatten(output.raw().clone()).await, &expected);

        let gradients = output.sqr().flatten_all().sum().backward().unwrap();
        let dweight = gradients.get(layer.weight()).unwrap();
        assert_eq!(dweight.shape(), [3, 4]);
        let dbias = gradients.get(layer.bias().unwrap()).unwrap();
        let mut expected_dbias = [0.0f32; 3];
        for row in expected.chunks(3) {
            for (acc, value) in expected_dbias.iter_mut().zip(row) {
                *acc += 2.0 * value;
            }
        }
        assert_slice_close(&flatten(dbias).await, &expected_dbias);
    }
}

#[tokio::test]
async fn test_autograd_layer_embedding_forward_parity() {
    let weights: Vec<f32> = (0..12).map(|index| (index as f32 * 0.7).sin()).collect();
    for device in test_devices().await {
        let table = RawTensor::from_slice(&device, [4, 3], &weights);
        let raw_layer = crate::layers::Embedding::new_from_tensor(table.clone());
        let graph = Graph::new();
        let layer =
            layers::Embedding::new_from_tensor(graph.leaf(table));
        assert_eq!(layer.num_embeddings(), 4);
        assert_eq!(layer.embedding_dim(), 3);

        let indices: RawTensor<2, u32> = RawTensor::from_slice(&device, [2, 2], &[0, 2, 1, 3]);
        let expected: RawTensor<3, f32> = raw_layer.forward(&indices);
        let output = layer.forward(&indices);
        assert_eq!(output.shape(), [2, 2, 3]);
        assert_slice_close(&flatten(output.raw().clone()).await, &flatten(expected).await);

        let flat_indices: RawTensor<1, u32> = RawTensor::from_slice(&device, [3], &[2, 0, 2]);
        let expected_flat: RawTensor<2, f32> = raw_layer.forward(&flat_indices);
        let output_flat = layer.forward(&flat_indices);
        assert_eq!(output_flat.shape(), [3, 3]);
        assert_slice_close(
            &flatten(output_flat.raw().clone()).await,
            &flatten(expected_flat).await,
        );
    }
}

#[tokio::test]
async fn test_autograd_layer_embedding_weight_gradient() {
    let weights: Vec<f32> = (0..12).map(|index| (index as f32 * 0.7).sin()).collect();
    for device in test_devices().await {
        let indices: RawTensor<2, u32> = RawTensor::from_slice(&device, [2, 2], &[0, 2, 1, 2]);
        assert_gradient_matches_finite_difference(&device, [4, 3], &weights, |_graph, table| {
            layers::Embedding::new_from_tensor(table)
                .forward(&indices)
                .sqr()
                .flatten_all()
                .sum()
        })
        .await;

        let flat_indices: RawTensor<1, u32> = RawTensor::from_slice(&device, [3], &[3, 1, 3]);
        assert_gradient_matches_finite_difference(&device, [4, 3], &weights, |_graph, table| {
            layers::Embedding::new_from_tensor(table)
                .forward(&flat_indices)
                .sqr()
                .flatten_all()
                .sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_autograd_layer_embedding_from_inference() {
    let weights: Vec<f32> = (0..12).map(|index| index as f32 * 0.25 - 1.0).collect();
    let bytes: Vec<u8> = weights
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    for device in test_devices().await {
        let quantized = crate::QMatrix::from_raw_bytes(
            &device,
            [4usize, 3],
            &bytes,
            fusor_gguf::GgmlType::F32,
        )
        .unwrap();
        let quantized_layer = crate::layers::Embedding::<f32>::new(quantized);
        let dense_layer = crate::layers::Embedding::<f32>::new_from_tensor(RawTensor::from_slice(
            &device,
            [4, 3],
            &weights,
        ));
        for raw_layer in [&quantized_layer, &dense_layer] {
            let indices: RawTensor<2, u32> = RawTensor::from_slice(&device, [2, 2], &[3, 0, 2, 2]);
            let expected: RawTensor<3, f32> = raw_layer.forward(&indices);

            let graph = Graph::new();
            let layer =
                layers::Embedding::from_inference(&graph, raw_layer);
            assert_eq!(layer.num_embeddings(), 4);
            assert_eq!(layer.embedding_dim(), 3);
            assert_slice_close(&flatten(layer.embeddings().raw().clone()).await, &weights);
            let output = layer.forward(&indices);
            assert_slice_close(&flatten(output.raw().clone()).await, &flatten(expected).await);

            let loss = layer.forward(&indices).sqr().flatten_all().sum();
            let gradients = loss.backward().unwrap();
            assert!(gradients.get(layer.embeddings()).is_some());
        }
    }
}

#[tokio::test]
async fn test_autograd_layer_layer_norm_forward_matches_inference() {
    for device in test_devices().await {
        let weight_data = [0.5f32, 1.5, -1.0, 2.0];
        let bias_data = [0.1f32, -0.2, 0.3, 0.4];
        let input_data: Vec<f32> = (0..24).map(|index| (index as f32 * 0.7).sin()).collect();

        let inference = crate::layers::LayerNorm::new(
            RawTensor::from_slice(&device, [4], &weight_data),
            Some(RawTensor::from_slice(&device, [4], &bias_data)),
            1e-5,
        );
        let raw_input = RawTensor::from_slice(&device, [2, 3, 4], &input_data);
        let raw_input_2d = RawTensor::from_slice(&device, [6, 4], &input_data);
        let expected = flatten(inference.forward(&raw_input)).await;
        let expected_2d = flatten(inference.forward(&raw_input_2d)).await;

        let graph = Graph::new();
        let layer = layers::LayerNorm::new(
            Tensor::from_slice(&graph, &device, [4], &weight_data),
            Some(Tensor::from_slice(&graph, &device, [4], &bias_data)),
            1e-5,
        );
        let input = Tensor::from_slice(&graph, &device, [2, 3, 4], &input_data);
        let input_2d = Tensor::from_slice(&graph, &device, [6, 4], &input_data);
        assert_slice_close(&flatten(layer.forward(&input).into_raw()).await, &expected);
        assert_slice_close(
            &flatten(layer.forward(&input).into_raw()).await,
            &expected,
        );
        assert_slice_close(
            &flatten(layer.forward(&input_2d).into_raw()).await,
            &expected_2d,
        );
    }
}

#[tokio::test]
async fn test_autograd_layer_layer_norm_nd_forward_matches_inference() {
    for device in test_devices().await {
        let weight_data = [0.5f32, 1.5, -1.0, 2.0];
        let bias_data = [0.1f32, -0.2, 0.3, 0.4];
        let axis_weight_data = [0.5f32, 1.5, -1.0];
        let axis_bias_data = [0.1f32, -0.2, 0.3];
        let input_data: Vec<f32> = (0..24).map(|index| (index as f32 * 0.7).sin()).collect();
        let raw_input = RawTensor::from_slice(&device, [2, 3, 4], &input_data);

        let inference = crate::layers::LayerNormNd::new(
            RawTensor::from_slice(&device, [4], &weight_data),
            Some(RawTensor::from_slice(&device, [4], &bias_data)),
            1e-5,
        );
        let expected = flatten(inference.forward::<3, 2, _>(&raw_input)).await;
        let raw_input_2d = RawTensor::from_slice(&device, [6, 4], &input_data);
        let expected_2d = flatten(inference.forward(&raw_input_2d)).await;

        let inference_axis = crate::layers::LayerNormNd::new_over_axis(
            RawTensor::from_slice(&device, [3], &axis_weight_data),
            Some(RawTensor::from_slice(&device, [3], &axis_bias_data)),
            1,
            1e-5,
        );
        let expected_axis = flatten(inference_axis.forward::<3, 2, _>(&raw_input)).await;

        let graph = Graph::new();
        let input = Tensor::from_slice(&graph, &device, [2, 3, 4], &input_data);
        let input_2d = Tensor::from_slice(&graph, &device, [6, 4], &input_data);
        let layer = layers::LayerNormNd::new(
            Tensor::from_slice(&graph, &device, [4], &weight_data),
            Some(Tensor::from_slice(&graph, &device, [4], &bias_data)),
            1e-5,
        );
        assert_slice_close(
            &flatten(layer.forward::<3, 2>(&input).into_raw()).await,
            &expected,
        );
        assert_slice_close(
            &flatten(layer.forward_fused(&input).into_raw()).await,
            &expected,
        );
        assert_slice_close(
            &flatten(layer.forward(&input_2d).into_raw()).await,
            &expected_2d,
        );

        let layer_axis = layers::LayerNormNd::new_over_axis(
            Tensor::from_slice(&graph, &device, [3], &axis_weight_data),
            Some(Tensor::from_slice(&graph, &device, [3], &axis_bias_data)),
            1,
            1e-5,
        );
        assert_slice_close(
            &flatten(layer_axis.forward::<3, 2>(&input).into_raw()).await,
            &expected_axis,
        );
        assert_slice_close(
            &flatten(layer_axis.forward_fused(&input).into_raw()).await,
            &expected_axis,
        );
    }
}

#[tokio::test]
async fn test_autograd_layer_layer_norm_parameter_gradients() {
    for device in test_devices().await {
        let weight_data = [0.5f32, 1.5, -1.0, 2.0];
        let bias_data = [0.1f32, -0.2, 0.3, 0.4];
        let input_data: Vec<f32> = (0..24).map(|index| (index as f32 * 0.7).sin()).collect();

        assert_gradient_matches_finite_difference(&device, [4], &weight_data, |graph, weight| {
            let bias = Tensor::from_slice(graph, &device, [4], &bias_data);
            let input = Tensor::from_slice(graph, &device, [2, 3, 4], &input_data);
            layers::LayerNorm::new(weight, Some(bias), 1e-5)
                .forward(&input)
                .flatten_all()
                .sum()
        })
        .await;

        assert_gradient_matches_finite_difference(&device, [4], &bias_data, |graph, bias| {
            let weight = Tensor::from_slice(graph, &device, [4], &weight_data);
            let input = Tensor::from_slice(graph, &device, [2, 3, 4], &input_data);
            layers::LayerNorm::new(weight, Some(bias), 1e-5)
                .forward(&input)
                .flatten_all()
                .sum()
        })
        .await;

        assert_gradient_matches_finite_difference(&device, [4], &weight_data, |graph, weight| {
            let bias = Tensor::from_slice(graph, &device, [4], &bias_data);
            let input = Tensor::from_slice(graph, &device, [2, 3, 4], &input_data);
            layers::LayerNorm::new(weight, Some(bias), 1e-5)
                .forward(&input)
                .flatten_all()
                .sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_autograd_layer_layer_norm_nd_parameter_gradients() {
    for device in test_devices().await {
        let weight_data = [0.5f32, 1.5, -1.0];
        let bias_data = [0.1f32, -0.2, 0.3];
        let input_data: Vec<f32> = (0..24).map(|index| (index as f32 * 0.7).sin()).collect();

        assert_gradient_matches_finite_difference(&device, [3], &weight_data, |graph, weight| {
            let bias = Tensor::from_slice(graph, &device, [3], &bias_data);
            let input = Tensor::from_slice(graph, &device, [2, 3, 4], &input_data);
            layers::LayerNormNd::new_over_axis(weight, Some(bias), 1, 1e-5)
                .forward::<3, 2>(&input)
                .flatten_all()
                .sum()
        })
        .await;

        assert_gradient_matches_finite_difference(&device, [3], &bias_data, |graph, bias| {
            let weight = Tensor::from_slice(graph, &device, [3], &weight_data);
            let input = Tensor::from_slice(graph, &device, [2, 3, 4], &input_data);
            layers::LayerNormNd::new_over_axis(weight, Some(bias), 1, 1e-5)
                .forward::<3, 2>(&input)
                .flatten_all()
                .sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_autograd_layer_layer_norm_from_inference_roundtrip() {
    for device in test_devices().await {
        let weight_data = [0.5f32, 1.5, -1.0, 2.0];
        let bias_data = [0.1f32, -0.2, 0.3, 0.4];
        let axis_weight_data = [0.5f32, 1.5, -1.0];
        let input_data: Vec<f32> = (0..24).map(|index| (index as f32 * 0.7).sin()).collect();
        let raw_input = RawTensor::from_slice(&device, [2, 3, 4], &input_data);

        let inference = crate::layers::LayerNorm::new(
            RawTensor::from_slice(&device, [4], &weight_data),
            Some(RawTensor::from_slice(&device, [4], &bias_data)),
            1e-5,
        );
        let expected = flatten(inference.forward(&raw_input)).await;

        let graph = Graph::new();
        let input = Tensor::constant_from_raw(&graph, raw_input.clone());
        let layer = layers::LayerNorm::from_inference(&graph, &inference);
        let output = layer.forward(&input);
        assert_slice_close(&flatten(output.raw().clone()).await, &expected);

        let gradients = output.flatten_all().sum().backward().unwrap();
        assert_eq!(gradients.get(layer.weight()).unwrap().shape(), [4]);
        assert_eq!(gradients.get(layer.bias().unwrap()).unwrap().shape(), [4]);
        assert!(gradients.get(&input).is_none());

        let inference_nd = crate::layers::LayerNormNd::new(
            RawTensor::from_slice(&device, [4], &weight_data),
            Some(RawTensor::from_slice(&device, [4], &bias_data)),
            1e-5,
        );
        let expected_nd = flatten(inference_nd.forward::<3, 2, _>(&raw_input)).await;
        let layer_nd = layers::LayerNormNd::from_inference(&graph, &inference_nd);
        assert_slice_close(
            &flatten(layer_nd.forward::<3, 2>(&input).into_raw()).await,
            &expected_nd,
        );

        let inference_axis = crate::layers::LayerNormNd::new_over_axis(
            RawTensor::from_slice(&device, [3], &axis_weight_data),
            None,
            1,
            1e-5,
        );
        let expected_axis = flatten(inference_axis.forward::<3, 2, _>(&raw_input)).await;
        let layer_axis =
            layers::LayerNormNd::from_inference_over_axis(&graph, &inference_axis, 1);
        assert_slice_close(
            &flatten(layer_axis.forward::<3, 2>(&input).into_raw()).await,
            &expected_axis,
        );
    }
}

#[tokio::test]
async fn test_autograd_layer_rms_norm_forward_parity() {
    let weight_data = [0.5f32, 1.5, -0.75, 2.0];
    let bias_data = [0.1f32, -0.2, 0.3, -0.4];
    let input_data: Vec<f32> = (0..24).map(|i| (i as f32 * 0.37).sin()).collect();
    let residual_data: Vec<f32> = (0..24).map(|i| (i as f32 * 0.61).cos()).collect();
    for device in test_devices().await {
        let raw_weight = RawTensor::from_slice(&device, [4], &weight_data);
        let raw_bias = RawTensor::from_slice(&device, [4], &bias_data);
        let raw_layer =
            crate::layers::RmsNorm::new(raw_weight.clone(), Some(raw_bias.clone()), 1e-5);

        let graph = Graph::new();
        let layer = layers::RmsNorm::new(
            graph.leaf(raw_weight),
            Some(graph.leaf(raw_bias)),
            1e-5,
        );

        let input_2d = RawTensor::from_slice(&device, [6, 4], &input_data);
        let expected = flatten(raw_layer.forward(&input_2d)).await;
        let output = layer.forward(&Tensor::constant_from_raw(&graph, input_2d));
        assert_slice_close(&flatten(output.raw().clone()).await, &expected);

        let input_3d = RawTensor::from_slice(&device, [2, 3, 4], &input_data);
        let expected = flatten(raw_layer.forward(&input_3d)).await;
        let output = layer.forward(&Tensor::constant_from_raw(&graph, input_3d.clone()));
        assert_slice_close(&flatten(output.raw().clone()).await, &expected);

        let input_4d = RawTensor::from_slice(&device, [2, 1, 3, 4], &input_data);
        let expected = flatten(raw_layer.forward(&input_4d)).await;
        let output = layer.forward(&Tensor::constant_from_raw(&graph, input_4d));
        assert_slice_close(&flatten(output.raw().clone()).await, &expected);

        let residual_3d = RawTensor::from_slice(&device, [2, 3, 4], &residual_data);
        let expected = flatten(raw_layer.forward_residual_f32(&input_3d, &residual_3d)).await;
        let output = layer.forward_residual(
            &Tensor::constant_from_raw(&graph, input_3d),
            &Tensor::constant_from_raw(&graph, residual_3d),
        );
        assert_slice_close(&flatten(output.raw().clone()).await, &expected);
    }
}

#[tokio::test]
async fn test_autograd_layer_rms_norm_weight_gradient() {
    let weight_data = [0.5f32, 1.5, -0.75, 2.0];
    let bias_data = [0.1f32, -0.2, 0.3, -0.4];
    let input_data: Vec<f32> = (0..24).map(|i| (i as f32 * 0.37).sin() + 0.25).collect();
    let residual_data: Vec<f32> = (0..24).map(|i| (i as f32 * 0.61).cos()).collect();
    for device in test_devices().await {
        let input_2d = RawTensor::from_slice(&device, [6, 4], &input_data);
        assert_gradient_matches_finite_difference(&device, [4], &weight_data, |graph, weight| {
            let layer = layers::RmsNorm::new(weight, None, 1e-5);
            layer
                .forward(&Tensor::constant_from_raw(graph, input_2d.clone()))
                .sqr()
                .flatten_all()
                .sum()
        })
        .await;

        let input_3d = RawTensor::from_slice(&device, [2, 3, 4], &input_data);
        let bias = RawTensor::from_slice(&device, [4], &bias_data);
        assert_gradient_matches_finite_difference(&device, [4], &weight_data, |graph, weight| {
            let layer = layers::RmsNorm::new(
                weight,
                Some(Tensor::constant_from_raw(graph, bias.clone())),
                1e-5,
            );
            layer
                .forward(&Tensor::constant_from_raw(graph, input_3d.clone()))
                .sqr()
                .flatten_all()
                .sum()
        })
        .await;

        let residual_3d = RawTensor::from_slice(&device, [2, 3, 4], &residual_data);
        assert_gradient_matches_finite_difference(&device, [4], &weight_data, |graph, weight| {
            let layer = layers::RmsNorm::new(
                weight,
                Some(Tensor::constant_from_raw(graph, bias.clone())),
                1e-5,
            );
            layer
                .forward_residual(
                    &Tensor::constant_from_raw(graph, input_3d.clone()),
                    &Tensor::constant_from_raw(graph, residual_3d.clone()),
                )
                .sqr()
                .flatten_all()
                .sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_autograd_layer_rms_norm_bias_gradient() {
    let weight_data = [0.5f32, 1.5, -0.75, 2.0];
    let bias_data = [0.1f32, -0.2, 0.3, -0.4];
    let input_data: Vec<f32> = (0..24).map(|i| (i as f32 * 0.37).sin() + 0.25).collect();
    let residual_data: Vec<f32> = (0..24).map(|i| (i as f32 * 0.61).cos()).collect();
    for device in test_devices().await {
        let input_3d = RawTensor::from_slice(&device, [2, 3, 4], &input_data);
        let weight = RawTensor::from_slice(&device, [4], &weight_data);
        assert_gradient_matches_finite_difference(&device, [4], &bias_data, |graph, bias| {
            let layer = layers::RmsNorm::new(
                Tensor::constant_from_raw(graph, weight.clone()),
                Some(bias),
                1e-5,
            );
            layer
                .forward(&Tensor::constant_from_raw(graph, input_3d.clone()))
                .sqr()
                .flatten_all()
                .sum()
        })
        .await;

        let residual_3d = RawTensor::from_slice(&device, [2, 3, 4], &residual_data);
        assert_gradient_matches_finite_difference(&device, [4], &bias_data, |graph, bias| {
            let layer = layers::RmsNorm::new(
                Tensor::constant_from_raw(graph, weight.clone()),
                Some(bias),
                1e-5,
            );
            layer
                .forward_residual(
                    &Tensor::constant_from_raw(graph, input_3d.clone()),
                    &Tensor::constant_from_raw(graph, residual_3d.clone()),
                )
                .sqr()
                .flatten_all()
                .sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_autograd_layer_rms_norm_from_inference() {
    let weight_data = [0.5f32, 1.5, -0.75, 2.0];
    let bias_data = [0.1f32, -0.2, 0.3, -0.4];
    let input_data: Vec<f32> = (0..24).map(|i| (i as f32 * 0.37).sin()).collect();
    for device in test_devices().await {
        let raw_layer = crate::layers::RmsNorm::new(
            RawTensor::from_slice(&device, [4], &weight_data),
            Some(RawTensor::from_slice(&device, [4], &bias_data)),
            1e-5,
        );
        let graph = Graph::new();
        let layer = layers::RmsNorm::from_inference(&graph, &raw_layer);
        assert_eq!(layer.eps(), raw_layer.eps());

        let input = RawTensor::from_slice(&device, [2, 3, 4], &input_data);
        let expected = flatten(raw_layer.forward(&input)).await;
        let output = layer.forward(&Tensor::constant_from_raw(&graph, input));
        assert_slice_close(&flatten(output.raw().clone()).await, &expected);

        let gradients = output.flatten_all().sum().backward().unwrap();
        let weight_gradient = gradients.get(layer.weight()).unwrap();
        let bias_gradient = gradients.get(layer.bias().unwrap()).unwrap();
        assert_eq!(weight_gradient.shape(), [4]);
        assert_eq!(bias_gradient.shape(), [4]);
    }
}

#[tokio::test]
async fn test_autograd_layer_conv_nd_1d_forward_parity() {
    for device in test_devices().await {
        let w_data: Vec<f32> = (0..12).map(|i| (i as f32 * 0.37).cos()).collect();
        let b_data = [0.3f32, -0.7];
        let x_data: Vec<f32> = (0..15).map(|i| (i as f32 * 0.61).sin()).collect();
        let config = crate::layers::ConvNdConfig {
            padding: [1],
            stride: [2],
            groups: 1,
        };

        let graph = Graph::new();
        let layer = layers::ConvNd::<1, 3>::new(
            Tensor::from_slice(&graph, &device, [2, 3, 2], &w_data),
            Some(Tensor::from_slice(&graph, &device, [2], &b_data)),
            config,
        );
        let x: Tensor<3> = Tensor::from_slice(&graph, &device, [1, 3, 5], &x_data);
        let output = layer.forward(&x);

        let raw_layer = crate::layers::ConvNd::<1, 3, f32>::new(
            RawTensor::from_slice(&device, [2, 3, 2], &w_data),
            Some(RawTensor::from_slice(&device, [2], &b_data)),
            config,
        );
        let expected = raw_layer.forward(&RawTensor::from_slice(&device, [1, 3, 5], &x_data));
        assert_slice_close(
            &flatten(output.raw().clone()).await,
            &flatten(expected).await,
        );
        assert_eq!(layer.in_channels(), 3);
        assert_eq!(layer.out_channels(), 2);
        assert_eq!(layer.config().stride, [2]);
    }
}

#[tokio::test]
async fn test_autograd_layer_conv_nd_1d_parameter_gradients() {
    for device in test_devices().await {
        let w_data: Vec<f32> = (0..12).map(|i| (i as f32 * 0.37).cos()).collect();
        let b_data = [0.3f32, -0.7];
        let x_data: Vec<f32> = (0..15).map(|i| (i as f32 * 0.61).sin()).collect();
        let config = crate::layers::ConvNdConfig {
            padding: [1],
            stride: [2],
            groups: 1,
        };

        let graph = Graph::new();
        let layer = layers::ConvNd::<1, 3>::new(
            Tensor::from_slice(&graph, &device, [2, 3, 2], &w_data),
            Some(Tensor::from_slice(&graph, &device, [2], &b_data)),
            config,
        );
        let x = Tensor::constant_from_raw(
            &graph,
            RawTensor::from_slice(&device, [1, 3, 5], &x_data),
        );
        let gradients = layer.forward(&x).flatten_all().sum().backward().unwrap();
        assert!(gradients.get(layer.weight()).is_some());
        assert!(gradients.get(layer.bias().unwrap()).is_some());

        let fd_device = device.clone();
        let x_fd = x_data.clone();
        let b_fd = b_data;
        assert_gradient_matches_finite_difference(
            &device,
            [2, 3, 2],
            &w_data,
            move |graph, w| {
                let layer = layers::ConvNd::<1, 3>::new(
                    w,
                    Some(Tensor::constant_from_raw(
                        graph,
                        RawTensor::from_slice(&fd_device, [2], &b_fd),
                    )),
                    config,
                );
                let x = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [1, 3, 5], &x_fd),
                );
                let out = layer.forward(&x);
                out.mul(&composite_ramp(graph, &fd_device, out.shape()))
                    .flatten_all()
                    .sum()
            },
        )
        .await;

        let fd_device = device.clone();
        let x_fd = x_data.clone();
        let w_fd = w_data.clone();
        assert_gradient_matches_finite_difference(&device, [2], &b_data, move |graph, b| {
            let layer = layers::ConvNd::<1, 3>::new(
                Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [2, 3, 2], &w_fd),
                ),
                Some(b),
                config,
            );
            let x = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [1, 3, 5], &x_fd),
            );
            let out = layer.forward(&x);
            out.mul(&composite_ramp(graph, &fd_device, out.shape()))
                .flatten_all()
                .sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_autograd_layer_conv_nd_forward_parity() {
    for device in test_devices().await {
        {
            let x_data: Vec<f32> = (0..18).map(|i| (i as f32 * 0.41).sin()).collect();
            let w_data: Vec<f32> = (0..16).map(|i| (i as f32 * 0.23).cos()).collect();
            let b_data = [0.5f32, -1.0];
            let config = crate::layers::ConvNdConfig {
                padding: [1, 1],
                stride: [2, 2],
                groups: 1,
            };

            let raw_layer = crate::layers::ConvNd::<2, 4, f32>::new(
                RawTensor::from_slice(&device, [2, 2, 2, 2], &w_data),
                Some(RawTensor::from_slice(&device, [2], &b_data)),
                config,
            );
            let expected =
                raw_layer.forward(&RawTensor::from_slice(&device, [1, 2, 3, 3], &x_data));

            let graph = Graph::new();
            let layer = layers::ConvNd::<2, 4>::new(
                Tensor::from_slice(&graph, &device, [2, 2, 2, 2], &w_data),
                Some(Tensor::from_slice(&graph, &device, [2], &b_data)),
                config,
            );
            let x: Tensor<4> = Tensor::from_slice(&graph, &device, [1, 2, 3, 3], &x_data);
            let output = layer.forward(&x);

            assert_slice_close(
                &flatten(output.raw().clone()).await,
                &flatten(expected).await,
            );
        }

        {
            let x_data: Vec<f32> = (0..16).map(|i| (i as f32 * 0.57).sin()).collect();
            let w_data: Vec<f32> = (0..16).map(|i| (i as f32 * 0.31).cos()).collect();
            let b_data = [0.2f32, -0.4, 0.6, -0.8];
            let config = crate::layers::ConvNdConfig {
                padding: [1],
                stride: [2],
                groups: 2,
            };

            let raw_layer = crate::layers::ConvNd::<1, 3, f32>::new(
                RawTensor::from_slice(&device, [4, 2, 2], &w_data),
                Some(RawTensor::from_slice(&device, [4], &b_data)),
                config,
            );
            let expected = raw_layer.forward(&RawTensor::from_slice(&device, [1, 4, 4], &x_data));

            let graph = Graph::new();
            let layer = layers::ConvNd::<1, 3>::new(
                Tensor::from_slice(&graph, &device, [4, 2, 2], &w_data),
                Some(Tensor::from_slice(&graph, &device, [4], &b_data)),
                config,
            );
            let x: Tensor<3> = Tensor::from_slice(&graph, &device, [1, 4, 4], &x_data);
            let output = layer.forward(&x);

            assert_slice_close(
                &flatten(output.raw().clone()).await,
                &flatten(expected).await,
            );
        }
    }
}

#[tokio::test]
async fn test_autograd_layer_conv_nd_parameter_gradients() {
    for device in test_devices().await {
        let x_data: Vec<f32> = (0..16).map(|i| (i as f32 * 0.57).sin()).collect();
        let w_data: Vec<f32> = (0..16).map(|i| (i as f32 * 0.31).cos()).collect();
        let b_data = [0.2f32, -0.4, 0.6, -0.8];
        let config = crate::layers::ConvNdConfig {
            padding: [1],
            stride: [2],
            groups: 2,
        };

        {
            let graph = Graph::new();
            let layer = layers::ConvNd::<1, 3>::new(
                Tensor::from_slice(&graph, &device, [4, 2, 2], &w_data),
                Some(Tensor::from_slice(&graph, &device, [4], &b_data)),
                config,
            );
            let x = Tensor::constant_from_raw(
                &graph,
                RawTensor::from_slice(&device, [1, 4, 4], &x_data),
            );
            let gradients = layer.forward(&x).flatten_all().sum().backward().unwrap();
            assert_eq!(gradients.get(layer.weight()).unwrap().shape(), [4, 2, 2]);
            assert_eq!(gradients.get(layer.bias().unwrap()).unwrap().shape(), [4]);
        }

        let fd_device = device.clone();
        let x_fd = x_data.clone();
        assert_gradient_matches_finite_difference(
            &device,
            [4, 2, 2],
            &w_data,
            move |graph, w| {
                let layer = layers::ConvNd::<1, 3>::new(
                    w,
                    Some(Tensor::from_slice(graph, &fd_device, [4], &b_data)),
                    config,
                );
                let x = Tensor::constant_from_raw(
                    graph,
                    RawTensor::from_slice(&fd_device, [1, 4, 4], &x_fd),
                );
                let out = layer.forward(&x);
                out.mul(&composite_ramp(graph, &fd_device, out.shape()))
                    .flatten_all()
                    .sum()
            },
        )
        .await;

        let fd_device = device.clone();
        let x_fd = x_data.clone();
        let w_fd = w_data.clone();
        assert_gradient_matches_finite_difference(&device, [4], &b_data, move |graph, b| {
            let layer = layers::ConvNd::<1, 3>::new(
                Tensor::from_slice(graph, &fd_device, [4, 2, 2], &w_fd),
                Some(b),
                config,
            );
            let x = Tensor::constant_from_raw(
                graph,
                RawTensor::from_slice(&fd_device, [1, 4, 4], &x_fd),
            );
            let out = layer.forward(&x);
            out.mul(&composite_ramp(graph, &fd_device, out.shape()))
                .flatten_all()
                .sum()
        })
        .await;
    }
}

#[tokio::test]
async fn test_autograd_layer_conv_nd_from_inference() {
    for device in test_devices().await {
        let x_data: Vec<f32> = (0..16).map(|i| (i as f32 * 0.57).sin()).collect();
        let w_data: Vec<f32> = (0..16).map(|i| (i as f32 * 0.31).cos()).collect();
        let b_data = [0.2f32, -0.4, 0.6, -0.8];
        let config = crate::layers::ConvNdConfig {
            padding: [1],
            stride: [2],
            groups: 2,
        };
        let raw_layer = crate::layers::ConvNd::<1, 3, f32>::new(
            RawTensor::from_slice(&device, [4, 2, 2], &w_data),
            Some(RawTensor::from_slice(&device, [4], &b_data)),
            config,
        );

        let graph = Graph::new();
        let layer = layers::ConvNd::<1, 3>::from_inference(&graph, &raw_layer);
        assert_slice_close(&flatten(layer.weight().raw().clone()).await, &w_data);
        assert_slice_close(&flatten(layer.bias().unwrap().raw().clone()).await, &b_data);

        let x_raw = RawTensor::from_slice(&device, [1, 4, 4], &x_data);
        let expected = raw_layer.forward(&x_raw);
        let output = layer.forward(&Tensor::constant_from_raw(&graph, x_raw));
        assert_slice_close(
            &flatten(output.raw().clone()).await,
            &flatten(expected).await,
        );

        let gradients = output.flatten_all().sum().backward().unwrap();
        assert!(gradients.get(layer.weight()).is_some());
        assert!(gradients.get(layer.bias().unwrap()).is_some());

        let raw_no_bias = crate::layers::ConvNd::<1, 3, f32>::new(
            RawTensor::from_slice(&device, [4, 2, 2], &w_data),
            None,
            config,
        );
        let imported = layers::ConvNd::<1, 3>::from_inference(&graph, &raw_no_bias);
        assert!(imported.bias().is_none());
        let x_raw = RawTensor::from_slice(&device, [1, 4, 4], &x_data);
        let expected = raw_no_bias.forward(&x_raw);
        let output = imported.forward(&Tensor::constant_from_raw(&graph, x_raw));
        assert_slice_close(
            &flatten(output.raw().clone()).await,
            &flatten(expected).await,
        );
    }
}

/// End-to-end training with the trainable autograd layers: the same 2-16-2
/// XOR MLP as [`test_train_xor_classifier`], but built from two
/// `layers::Linear` layers with gradients fetched through the
/// layer's `weight()`/`bias()` parameter handles.
#[tokio::test]
async fn test_train_xor_with_layers() {
    const SAMPLES: usize = 64;
    const HIDDEN: usize = 16;
    const STEPS: usize = 500;
    const LEARNING_RATE: f32 = 1.0;

    let (features, labels, w1_init, w2_init) = xor_training_data(HIDDEN);

    for (device, name) in test_devices().await.into_iter().zip(["cpu", "gpu"]) {
        let inputs = RawTensor::from_slice(&device, [SAMPLES, 2], &features);
        let targets = RawTensor::from_slice(&device, [SAMPLES], &labels);

        // Linear stores weights as (out_features, in_features).
        let mut w1 = RawTensor::from_slice(&device, [HIDDEN, 2], &w1_init);
        let mut b1 = RawTensor::zeros(&device, [HIDDEN]);
        let mut w2 = RawTensor::from_slice(&device, [2, HIDDEN], &w2_init);
        let mut b2 = RawTensor::zeros(&device, [2]);

        let mut final_loss = f32::INFINITY;
        for step in 0..STEPS {
            let graph = Graph::new();
            let x = Tensor::constant_from_raw(&graph, inputs.clone());
            let layer1 = layers::Linear::new(
                graph.leaf(w1.clone()),
                Some(graph.leaf(b1.clone())),
            );
            let layer2 = layers::Linear::new(
                graph.leaf(w2.clone()),
                Some(graph.leaf(b2.clone())),
            );

            let hidden = layer1.forward(&x).relu();
            let logits = layer2.forward(&hidden);
            // Numerically stable cross-entropy: log softmax via log-sum-exp
            // so a saturated class cannot underflow to log(0).
            let shifted = logits.sub_::<2, 2>(&logits.max_keepdim::<1>(1));
            let log_sum_exp = shifted.exp().sum_keepdim(1).log();
            let label_log_probs = shifted.sub_::<2, 2>(&log_sum_exp).gather_last(&targets);
            let loss: Tensor<0> = label_log_probs.sum().mul_scalar(-1.0 / SAMPLES as f32);

            let loss_value = flatten(loss.raw().clone()).await[0];
            let gradients = loss.backward().unwrap().into_detached();
            let dw1 = gradients.get(layer1.weight()).unwrap();
            let db1 = gradients.get(layer1.bias().unwrap()).unwrap();
            let dw2 = gradients.get(layer2.weight()).unwrap();
            let db2 = gradients.get(layer2.bias().unwrap()).unwrap();

            w1 = (w1 - dw1 * LEARNING_RATE).to_concrete();
            b1 = (b1 - db1 * LEARNING_RATE).to_concrete();
            w2 = (w2 - dw2 * LEARNING_RATE).to_concrete();
            b2 = (b2 - db2 * LEARNING_RATE).to_concrete();

            final_loss = loss_value;
            if step % 100 == 0 {
                eprintln!("[{name}] step {step}: loss {loss_value:.4}");
            }
        }
        eprintln!("[{name}] final loss {final_loss:.4}");

        let graph = Graph::new();
        let x = Tensor::constant_from_raw(&graph, inputs.clone());
        let layer1 = layers::Linear::new(
            Tensor::constant_from_raw(&graph, w1.clone()),
            Some(Tensor::constant_from_raw(&graph, b1.clone())),
        );
        let layer2 = layers::Linear::new(
            Tensor::constant_from_raw(&graph, w2.clone()),
            Some(Tensor::constant_from_raw(&graph, b2.clone())),
        );
        let hidden = layer1.forward(&x).relu();
        let logits = layer2.forward(&hidden);
        let logits = logits.raw().clone().as_slice().await.unwrap().to_vec();
        let correct = logits
            .iter()
            .zip(&labels)
            .filter(|(row, label)| u32::from(row[1] > row[0]) == **label)
            .count();
        eprintln!("[{name}] accuracy {correct}/{SAMPLES}");

        assert!(
            final_loss < 0.1,
            "training did not converge: final loss {final_loss}",
        );
        assert_eq!(correct, SAMPLES, "classifier misclassified training points");
    }
}
