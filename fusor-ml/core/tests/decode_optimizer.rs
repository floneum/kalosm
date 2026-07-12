//! Decode regression for the default aggressive optimizer.

use fusor_core::{Device, QMatrix, Tensor};
use fusor_gguf::GgmlType;

const N: usize = 4;
const K: usize = 8;
const QMATMULS: usize = 16;

fn weight(device: &Device) -> (QMatrix, Vec<f32>) {
    let values = (0..N * K)
        .map(|i| 0.1 + i as f32 * 0.05)
        .collect::<Vec<_>>();
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let matrix =
        QMatrix::from_parts(device, &bytes, vec![N, K].into_boxed_slice(), GgmlType::F32).unwrap();
    (matrix, values)
}

#[test]
fn decode_runs_the_full_optimizer_by_default() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let (weight, weights) = weight(&device);
        let input = [1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let activation = Tensor::from_slice(&device, [1, K], &input);
        let outputs = (0..QMATMULS)
            .map(|_| activation.q_mat_mul(&weight))
            .collect::<Vec<_>>();
        let mut total = outputs[0].clone();
        for output in &outputs[1..] {
            total = &total + output;
        }

        assert!(
            total.count_kernels_to_resolve() < QMATMULS * 2 - 1,
            "decode should take the automatic planning and fusion path",
        );

        let actual = total.as_slice::<2, f32>().await.unwrap();
        for column in 0..N {
            let one_matmul = (0..K)
                .map(|k| input[k] * weights[column * K + k])
                .sum::<f32>();
            let expected = one_matmul * QMATMULS as f32;
            let tolerance = 1e-4 * expected.abs().max(1.0);
            assert!(
                (actual[[0, column]] - expected).abs() <= tolerance,
                "column {column}: got {}, expected {expected}",
                actual[[0, column]],
            );
        }
    });
}
