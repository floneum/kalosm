//! Ordinary tensor updates must not compile shape-specific copy shaders.
#![cfg(not(target_arch = "wasm32"))]

#[test]
#[ignore = "requires a live GPU"]
fn general_contiguous_updates_use_copy_commands() {
    use fusor::{Device, Tensor};
    let _guard = fusor_conformance::harness::gpu_test_guard();
    let device = pollster::block_on(Device::gpu()).unwrap();
    let target = device.backend().gpu_target().unwrap();
    let initial_compiles = target.launcher().pipeline_compiles();
    for width in [3, 7, 16, 33, 65] {
        let values: Vec<f32> = (0..2 * width * 4).map(|i| i as f32 + 1.).collect();
        let input = Tensor::from_slice(&device, [2, width, 4], &values);
        let padded = input.as_dyn().pad_with_zeros(1, 2, 3).unwrap();
        let actual = padded.to_vec_f32().unwrap();
        let mut expected = vec![0.; 2 * (width + 5) * 4];
        for row in 0..2 {
            let start = (row * (width + 5) + 2) * 4;
            expected[start..start + width * 4]
                .copy_from_slice(&values[row * width * 4..(row + 1) * width * 4]);
        }
        assert_eq!(actual, expected, "padding width {width}");
        let base_values: Vec<f32> = (0..expected.len()).map(|i| -(i as f32) - 1.).collect();
        let base = Tensor::from_slice(&device, [2, width + 5, 4], &base_values);
        let assigned = base
            .as_dyn()
            .slice_assign(&[0..2, 2..width + 2, 0..4], input.as_dyn())
            .unwrap();
        let mut assigned_expected = base_values.clone();
        for row in 0..2 {
            let start = (row * (width + 5) + 2) * 4;
            assigned_expected[start..start + width * 4]
                .copy_from_slice(&values[row * width * 4..(row + 1) * width * 4]);
        }
        assert_eq!(assigned.to_vec_f32().unwrap(), assigned_expected);
        assert_eq!(
            base.to_vec_f32(),
            base_values,
            "range assignment preserves its source"
        );
    }
    assert_eq!(
        target.launcher().pipeline_compiles() - initial_compiles,
        0,
        "contiguous updates should use WebGPU copy commands with the ordinary device"
    );
}

#[test]
#[ignore = "requires a live GPU"]
fn copy_lowering_preserves_compute_roots_and_strided_updates() {
    use fusor::{Device, Tensor};
    let _guard = fusor_conformance::harness::gpu_test_guard();
    let device = pollster::block_on(Device::gpu()).unwrap();
    let input = Tensor::from_slice(&device, [2, 3], &[1., 2., 3., 4., 5., 6.]);
    let computed = input.mul_scalar(3.);
    let transposed = computed.transpose(0, 1);
    let padded = transposed.as_dyn().pad_with_zeros(0, 1, 1).unwrap();
    device
        .session()
        .resolve(&[computed.as_dyn().clone(), padded.clone()])
        .unwrap();
    assert_eq!(computed.to_vec_f32(), vec![3., 6., 9., 12., 15., 18.]);
    assert_eq!(
        padded.to_vec_f32().unwrap(),
        vec![0., 0., 3., 12., 6., 15., 9., 18., 0., 0.]
    );

    // Unaligned F16 rows must retain the ordinary compute fallback.
    let half = Tensor::from_slice(&device, [2, 3], &[half::f16::ONE; 6]);
    let padded = half.as_dyn().pad_with_zeros(1, 1, 1).unwrap();
    assert_eq!(
        padded.to_vec_f32().unwrap(),
        vec![0., 1., 1., 1., 0., 0., 1., 1., 1., 0.]
    );
}

#[test]
#[ignore = "requires a live GPU"]
fn general_range_updates_preserve_gradients() {
    use fusor::{Session, session::Backend};
    use fusor_conformance::{harness, suite};
    let _guard = harness::gpu_test_guard();
    let session = Session::new(pollster::block_on(Backend::gpu()).unwrap()).unwrap();
    let cases = suite::registry();
    for name in [
        "pad_axis",
        "pad_with_zeros",
        "slice_assign",
        "cat_dim1",
        "repeat",
        "resize",
    ] {
        let case = cases.iter().find(|c| c.short() == name).unwrap();
        let outcome = harness::run_one(case, &session);
        eprintln!("{name}: {outcome:?}");
        assert!(outcome.is_pass(), "{name}: {outcome:?}");
    }
}
