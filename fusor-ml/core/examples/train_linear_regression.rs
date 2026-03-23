use fusor_core::{Device, StrideSpec, Tensor};

const LEARNING_RATE: f32 = 0.05;
const EPOCHS: usize = 80;

fn main() {
    pollster::block_on(async {
        let device = Device::new().await.unwrap();

        // Learn y = 2x + 1 from a tiny synthetic dataset.
        let inputs: Tensor<2, f32> = Tensor::new(&device, &[[0.0], [1.0], [2.0], [3.0], [4.0]]);
        let targets: Tensor<2, f32> = Tensor::new(&device, &[[1.0], [3.0], [5.0], [7.0], [9.0]]);

        let mut weight: Tensor<2, f32> = Tensor::new(&device, &[[0.0]]);
        let mut bias: Tensor<2, f32> = Tensor::new(&device, &[[0.0]]);

        for epoch in 0..EPOCHS {
            let batch_size = inputs.shape()[0] as f32;
            let bias_broadcast: Tensor<2, f32> = bias.restride([
                StrideSpec::dim_with(0, inputs.shape()[0], 0),
                StrideSpec::dim(1, 1),
            ]);
            let prediction = inputs.mat_mul(&weight) + &bias_broadcast;
            let error = &prediction - &targets;
            let squared_error = &error * &error;
            let loss: Tensor<0, f32> = squared_error.sum::<1>(0).sum::<0>(0) / batch_size;

            let loss_value = loss.to_scalar().await.unwrap();
            let error_host = error.as_slice().await.unwrap();
            let inputs_host = inputs.as_slice().await.unwrap();

            let mut weight_grad_value = 0.0f32;
            let mut bias_grad_value = 0.0f32;
            for row in 0..inputs.shape()[0] {
                let err = error_host[[row, 0]];
                weight_grad_value += inputs_host[[row, 0]] * err;
                bias_grad_value += err;
            }
            let grad_scale = 2.0 / batch_size;
            weight_grad_value *= grad_scale;
            bias_grad_value *= grad_scale;

            let weight_grad: Tensor<2, f32> = Tensor::new(&device, &[[weight_grad_value]]);
            let bias_grad: Tensor<2, f32> = Tensor::new(&device, &[[bias_grad_value]]);

            // Apply a simple SGD update.
            let next_weight = &weight - &(weight_grad * LEARNING_RATE);
            let next_bias = &bias - &(bias_grad * LEARNING_RATE);

            // Recreate the parameter tensors from host values so each SGD step starts a fresh graph.
            let next_weight_host = next_weight.as_slice().await.unwrap();
            let next_bias_host = next_bias.as_slice().await.unwrap();
            let weight_value = next_weight_host[[0, 0]];
            let bias_value = next_bias_host[[0, 0]];
            weight = Tensor::new(&device, &[[weight_value]]);
            bias = Tensor::new(&device, &[[bias_value]]);

            if epoch % 10 == 0 || epoch + 1 == EPOCHS {
                println!(
                    "epoch {:>2}: loss={:.6} weight={:.4} bias={:.4}",
                    epoch + 1,
                    loss_value,
                    weight_value,
                    bias_value,
                );
            }
        }

        let final_prediction = inputs.mat_mul(&weight)
            + &bias.restride([
                StrideSpec::dim_with(0, inputs.shape()[0], 0),
                StrideSpec::dim(1, 1),
            ]);
        println!(
            "final predictions: {:?}",
            final_prediction.as_slice().await.unwrap()
        );
    });
}
