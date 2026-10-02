#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
use fusor::{Device, Tensor};

#[test]
fn row_collectives_cover_tails_and_private_intermediates() {
    // Fresh devices keep these fixed shapes from promoting into one symbolic
    // family. Softmax needs both max and sum intermediates inside each slab.
    for (rows, width) in [(2, 29), (5, 65), (17, 96), (4, 512)] {
        let device = Device::gpu_blocking().unwrap();
        let input: Vec<f32> = (0..rows * width)
            .map(|i| ((i * 17 + i / width * 11) % 37) as f32 * 0.13 - 2.4)
            .collect();
        let x = Tensor::<2, f32>::from_slice(&device, [rows, width], &input);
        let y = x.softmax(1);
        let got = y.to_vec_f32();
        for (row, (source, result)) in input.chunks(width).zip(got.chunks(width)).enumerate() {
            let max = source.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = source.iter().map(|x| (x - max).exp()).collect();
            let sum: f32 = exp.iter().sum();
            for (column, (a, b)) in result.iter().zip(exp).enumerate() {
                assert!(
                    (a - b / sum).abs() < 2e-6,
                    "rows={rows} width={width} row={row} column={column}: {a} vs {}",
                    b / sum
                );
            }
        }
    }
}
