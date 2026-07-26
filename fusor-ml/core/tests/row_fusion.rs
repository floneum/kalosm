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

/// Lane groups narrower than the workgroup make the axis stride in chunks and
/// pin one row per group of lanes, so every axis length exercises a different
/// (group width, chunk count, masked tail) triple. Sweep the widths around the
/// subgroup width and the chunking thresholds against a host reference.
#[test]
fn row_reductions_match_host_across_axis_widths() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        // The no-subgroup sibling takes the shared-memory reduction tree, the
        // only path the web build has.
        sweep_axis_widths(&device).await;
        sweep_axis_widths(&device.without_subgroups()).await;
    });
}

async fn sweep_axis_widths(device: &Device) {
    const ROWS: usize = 37;
    for k in [
        1usize, 2, 3, 8, 16, 31, 32, 33, 63, 64, 65, 100, 128, 129, 200, 256, 257, 384, 512, 1000,
    ] {
        let values = pattern(ROWS * k, 0.31);
        let x = Tensor::from_slice(device, [ROWS, k], &values);

        let sums = x.sum(1).as_slice::<1, f32>().await.unwrap();
        let softmax = x.softmax_last_dim().as_slice::<2, f32>().await.unwrap();
        for row in [0usize, 1, ROWS / 2, ROWS - 1] {
            let base = row * k;
            let span = &values[base..base + k];

            let expected: f32 = span.iter().sum();
            let actual = sums[[row]];
            let tolerance = 1e-4 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() < tolerance,
                "k={k} row={row}: sum {actual}, expected {expected}"
            );

            let max = span.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator: f32 = span.iter().map(|v| (v - max).exp()).sum();
            for col in [0usize, k / 2, k - 1] {
                let expected = (span[col] - max).exp() / denominator;
                let actual = softmax[[row, col]];
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "k={k} row={row} col={col}: softmax {actual}, expected {expected}"
                );
            }
        }
    }
}
