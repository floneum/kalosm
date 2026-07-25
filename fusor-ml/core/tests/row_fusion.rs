//! Aggressive row-fusion regression. Interleaved-view scalar absorption is
//! part of the one default optimizer path for every graph size.

use fusor_core::{Device, Tensor};

fn pattern(len: usize, scale: f32) -> Vec<f32> {
    (0..len).map(|i| ((i as f32) * scale).sin()).collect()
}

#[test]
fn keepdim_scalar_chain_sandwich_fuses_to_single_kernel() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        // Layer-norm-forward shape: `x - broadcast(sum_keepdim(x) / h)`.
        // The `sum_keepdim` unsqueeze sits *between* the scalar division and
        // the reduce, so absorbing the mean as a row phase requires the
        // interleaved-view scalar walk.
        let (b, s, h) = (4usize, 8usize, 64usize);
        let x_values = pattern(b * s * h, 0.17);
        let x = Tensor::from_slice(&device, [b, s, h], &x_values);

        let mean = &x.sum_keepdim(2) / (h as f32);
        let centered = &x - &mean.broadcast_as([b, s, h]);
        assert!(
            centered.resolves_in::<1>(),
            "keepdim mean chain must fold into one row program"
        );

        let slice = centered.as_slice::<3, f32>().await.unwrap();
        for (batch, row) in [(0usize, 0usize), (1, 5), (b - 1, s - 1)] {
            let base = batch * s * h + row * h;
            let mean: f32 = (0..h).map(|col| x_values[base + col]).sum::<f32>() / h as f32;
            for col in [0usize, 31, h - 1] {
                let expected = x_values[base + col] - mean;
                let actual = slice[[batch, row, col]];
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "[{batch}, {row}, {col}]: got {actual}, expected {expected}"
                );
            }
        }
    });
}
