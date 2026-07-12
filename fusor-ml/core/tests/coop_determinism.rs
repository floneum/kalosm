//! The cooperative pair-loop matmul must be bit-deterministic run to run.
//!
//! Guards the expression-sinking class of miscompile: naga backends must
//! force-bake `CooperativeLoad` at its `Emit` point (like `Load`), or
//! single-use simdgroup loads inline into the deferred accumulator chain and
//! sink past the workgroup barrier, racing the next tile refill. That bug
//! produced nondeterministic 8x8-fragment corruption at a rate proportional
//! to workgroups x k-iterations — invisible to small conformance shapes, so
//! this test uses a grid large enough to expose it (it failed 10/10 rounds
//! against the broken backend).

use fusor_core::{Device, StrideSpec, Tensor};

#[test]
fn coop_matmul_is_run_to_run_deterministic() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };
        let (m, k, n) = (2048usize, 576, 128);
        let a_host: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.137).sin() * 0.5).collect();
        let weight_host: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.071).sin() * 0.1).collect();
        let a = Tensor::from_slice(&device, [m, k], &a_host);
        let weight = Tensor::from_slice(&device, [n, k], &weight_host);
        // The transposed-B restride keeps B staging on the strided path,
        // matching the configuration that exposed the original race.
        let b_mat = weight.restride([StrideSpec::dim(1, k), StrideSpec::dim(0, n)]);

        let reference = a.mat_mul(&b_mat).as_slice::<2, f32>().await.unwrap();
        for round in 0..3 {
            let repeat = a.mat_mul(&b_mat).as_slice::<2, f32>().await.unwrap();
            let mut diffs = 0usize;
            for mi in 0..m {
                for ni in 0..n {
                    if (repeat[[mi, ni]] - reference[[mi, ni]]).abs() > 1e-6 {
                        diffs += 1;
                    }
                }
            }
            assert_eq!(
                diffs, 0,
                "round {round}: {diffs} elements differ between identical dispatches"
            );
        }
    });
}
