#![cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
use fusor::{Device, Tensor, program::TrainingProgram};

fn floats(bytes: Vec<u8>) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}

#[test]
fn a_parallel_global_reduction_requires_a_dispatch_boundary() {
    use fusor::program::ProgramOptions;
    let device = Device::gpu_blocking().unwrap();
    let values: Vec<f32> = (0..97).map(|i| i as f32 * 0.01).collect();
    let x = Tensor::<1, f32>::from_slice(&device, [97], &values);
    let squares = x.sqr();
    let sum = squares.sum::<0>(0);
    let mut program = pollster::block_on(TrainingProgram::compile_with_options(
        &[squares.as_dyn().clone(), sum.as_dyn().clone()],
        &[],
        ProgramOptions {
            workgroups: Some(3),
            ..Default::default()
        },
    ))
    .unwrap();
    assert_eq!(program.stats().kernels, 2);
    program.run().unwrap();
    let got = floats(pollster::block_on(program.read(sum.as_dyn())).unwrap())[0];
    assert!((got - values.iter().map(|x| x * x).sum::<f32>()).abs() < 1e-4);
}

#[test]
fn ordinary_grouped_training_widens_subgroup_reductions() {
    let device = Device::gpu_blocking().unwrap();
    let x = Tensor::<2, f32>::from_slice(&device, [4, 32], &[0.1; 128]);
    let a = Tensor::<2, f32>::from_slice(&device, [32, 32], &[0.05; 1024]);
    let b = Tensor::<2, f32>::from_slice(&device, [32, 8], &[0.01; 256]);
    let y = x.matmul(&a).tanh().matmul(&b);
    let loss = y.sqr().sum::<1>(1).sum::<0>(0);
    let params = vec![a.as_dyn().clone(), b.as_dyn().clone()];
    let grads = device
        .graph()
        .backward_with(loss.as_dyn(), &params)
        .unwrap();
    let mut roots = vec![loss.as_dyn().clone()];
    for p in &params {
        roots.push(
            p.sub(&grads.get(p).unwrap().mul_scalar(0.01f32).unwrap())
                .unwrap(),
        );
    }
    device.session().resolve(&roots).unwrap();
    let expected = (0.16f32.tanh() * 0.32).powi(2) * 32.;
    assert!((loss.to_vec_f32()[0] - expected).abs() < 1e-5);
}

#[test]
fn long_queue_and_observation_preserve_state() {
    let device = Device::gpu_blocking().unwrap();
    let state = Tensor::<1, f32>::from_slice(&device, [1], &[0.]);
    let next = state.as_dyn().add_scalar(1f32).unwrap();
    let mut program = pollster::block_on(TrainingProgram::compile(
        std::slice::from_ref(&next),
        &[(state.as_dyn().clone(), next.clone())],
    ))
    .unwrap();
    pollster::block_on(async {
        for _ in 0..97 {
            program.run_async().await.unwrap();
        }
        assert_eq!(
            floats(program.read(state.as_dyn()).await.unwrap()),
            vec![97.]
        );
    });
    assert_eq!(program.dispatch_count(), 97);
    program.export(&[state.as_dyn().clone()]).unwrap();
    assert_eq!(state.to_vec_f32(), vec![97.]);
    program.run().unwrap();
    assert_eq!(
        floats(pollster::block_on(program.read(state.as_dyn())).unwrap()),
        vec![98.]
    );
    assert_eq!(
        state.to_vec_f32(),
        vec![97.],
        "exports are independent snapshots"
    );
    let foreign = Tensor::<1, f32>::from_slice(&Device::gpu_blocking().unwrap(), [1], &[0.]);
    assert!(
        program
            .write(foreign.as_dyn(), &0f32.to_le_bytes())
            .is_err()
    );
    assert!(
        pollster::block_on(TrainingProgram::compile(
            &[foreign.as_dyn().clone()],
            &[(state.as_dyn().clone(), state.as_dyn().clone())]
        ))
        .is_err()
    );
}

#[test]
fn long_parameter_gradient_uses_each_matrix_tile_once() {
    use fusor::program::ProgramOptions;
    let device = Device::gpu_blocking().unwrap();
    for k_len in [257, 512] {
        let a: Vec<f32> = (0..k_len * 17)
            .map(|i| (i % 13) as f32 * 0.01 - 0.06)
            .collect();
        let b: Vec<f32> = (0..k_len * 19)
            .map(|i| (i % 17) as f32 * 0.01 - 0.08)
            .collect();
        let x = Tensor::<2, f32>::from_slice(&device, [k_len, 17], &a);
        let y = Tensor::<2, f32>::from_slice(&device, [k_len, 19], &b);
        let gradient = x.transpose(0, 1).matmul(&y);
        let update = gradient.as_dyn().mul_scalar(-0.01f32).unwrap();
        for (matrix_acceleration, subgroup_acceleration) in
            [(false, false), (false, true), (true, true)]
        {
            let mut program = pollster::block_on(TrainingProgram::compile_with_options(
                std::slice::from_ref(&update),
                &[],
                ProgramOptions {
                    workgroups: Some(64),
                    matrix_acceleration,
                    subgroup_acceleration,
                    ..Default::default()
                },
            ))
            .unwrap();
            program.run().unwrap();
            let got = floats(pollster::block_on(program.read(&update)).unwrap());
            for r in 0..17 {
                for c in 0..19 {
                    let want: f32 = (0..k_len)
                        .map(|k| a[k * 17 + r] * b[k * 19 + c] * -0.01)
                        .sum();
                    assert!((got[r * 19 + c] - want).abs() < 1e-5);
                }
            }
            if k_len == 512 {
                assert_eq!(
                    program.stats().stages,
                    3,
                    "split contraction, partial sum, and update"
                );
            }
            assert!(
                program.stats().kernels >= 2,
                "a tiled gradient and linear consumer require a dispatch boundary"
            );
        }
    }
}

#[test]
fn simultaneous_feedback_snapshots_swaps_and_retained_views() {
    let device = Device::gpu_blocking().unwrap();
    let a = Tensor::<2, f32>::from_slice(&device, [2, 3], &[1., 2., 3., 4., 5., 6.]);
    let b = Tensor::<2, f32>::from_slice(&device, [2, 3], &[7., 8., 9., 10., 11., 12.]);
    let view = a.transpose(0, 1);
    let mut program = pollster::block_on(TrainingProgram::compile(
        &[view.as_dyn().clone()],
        &[
            (a.as_dyn().clone(), b.as_dyn().clone()),
            (b.as_dyn().clone(), a.as_dyn().clone()),
        ],
    ))
    .unwrap();
    for step in 0..4 {
        program.run().unwrap();
        let (old_a, old_b) = if step % 2 == 0 {
            (
                vec![1., 2., 3., 4., 5., 6.],
                vec![7., 8., 9., 10., 11., 12.],
            )
        } else {
            (
                vec![7., 8., 9., 10., 11., 12.],
                vec![1., 2., 3., 4., 5., 6.],
            )
        };
        assert_eq!(
            floats(pollster::block_on(program.read(a.as_dyn())).unwrap()),
            old_b
        );
        assert_eq!(
            floats(pollster::block_on(program.read(b.as_dyn())).unwrap()),
            old_a
        );
        assert_eq!(
            floats(pollster::block_on(program.read(view.as_dyn())).unwrap()),
            [old_a[0], old_a[3], old_a[1], old_a[4], old_a[2], old_a[5]]
        );
    }
}

#[test]
fn parallel_tail_matmul_reduction_and_feedback_match_host() {
    use fusor::program::ProgramOptions;
    let device = Device::gpu_blocking().unwrap();
    let data: Vec<f32> = (0..65 * 17)
        .map(|i| ((i % 31) as f32 - 15.) * 0.02)
        .collect();
    let weights: Vec<f32> = (0..17 * 19)
        .map(|i| ((i % 23) as f32 - 11.) * 0.03)
        .collect();
    let x = Tensor::<2, f32>::from_slice(&device, [65, 17], &data);
    let w = Tensor::<2, f32>::from_slice(&device, [17, 19], &weights);
    let product = x.matmul(&w);
    let row = product.sqr().sum::<1>(1);
    let loss = row.sum::<0>(0);
    for (groups, matrix_acceleration) in [
        (1, false),
        (3, false),
        (16, false),
        (1, true),
        (3, true),
        (16, true),
    ] {
        let options = ProgramOptions {
            workgroups: Some(groups),
            matrix_acceleration,
            max_region_stages: 2,
            ..Default::default()
        };
        let mut program = pollster::block_on(TrainingProgram::compile_with_options(
            &[
                product.as_dyn().clone(),
                row.as_dyn().clone(),
                loss.as_dyn().clone(),
            ],
            &[],
            options,
        ))
        .unwrap();
        // Each variant uses different input data: stale arena bytes from a
        // prior successful variant must not mask a missed output tile.
        let data: Vec<f32> = data
            .iter()
            .map(|v| v + groups as f32 * 0.01 + f32::from(matrix_acceleration) * 0.005)
            .collect();
        program
            .write(x.as_dyn(), bytemuck::cast_slice(&data))
            .unwrap();
        assert!(pollster::block_on(program.read(product.as_dyn())).is_err());
        program.run().unwrap();
        let got = floats(pollster::block_on(program.read(product.as_dyn())).unwrap());
        let mut expected_rows = vec![0.; 65];
        for r in 0..65 {
            for c in 0..19 {
                let expected: f32 = (0..17)
                    .map(|k| data[r * 17 + k] * weights[k * 19 + c])
                    .sum();
                assert!(
                    (got[r * 19 + c] - expected).abs() < 1e-5,
                    "groups={groups} r={r} c={c}: {} vs {expected}",
                    got[r * 19 + c]
                );
                expected_rows[r] += expected * expected;
            }
        }
        let got = floats(pollster::block_on(program.read(row.as_dyn())).unwrap());
        for (a, b) in got.iter().zip(&expected_rows) {
            assert!((a - b).abs() < 1e-5);
        }
        let got = floats(pollster::block_on(program.read(loss.as_dyn())).unwrap());
        assert!((got[0] - expected_rows.iter().sum::<f32>()).abs() < 1e-4);
    }
}

#[test]
fn fused_training_updates_state_and_uses_changing_inputs() {
    let device = Device::gpu_blocking().expect("GPU is required for the program integration suite");
    let weight = Tensor::<2, f32>::from_slice(&device, [2, 2], &[0.2, -0.3, 0.4, 0.1]);
    let input = Tensor::<2, f32>::from_slice(&device, [2, 2], &[1., 2., -1., 3.]);
    let predicted = input.matmul(&weight);
    let loss = predicted.sqr().sum::<1>(1).sum::<0>(0);
    let gradients = device
        .graph()
        .backward_with(loss.as_dyn(), &[weight.as_dyn().clone()])
        .unwrap();
    let gradient = gradients.get(weight.as_dyn()).unwrap();
    let update = weight
        .as_dyn()
        .sub(&gradient.mul_scalar(0.01f32).unwrap())
        .unwrap();
    let mut program = pollster::block_on(TrainingProgram::compile(
        &[loss.as_dyn().clone(), update.clone()],
        &[(weight.as_dyn().clone(), update)],
    ))
    .unwrap();
    let mut w = [0.2f32, -0.3, 0.4, 0.1];
    for step in 0..6 {
        let x = [1. + step as f32 * 0.1, 2., -1., 3.];
        program
            .write(input.as_dyn(), bytemuck::cast_slice(&x))
            .unwrap();
        program.run().unwrap();
        let mut expected_loss = 0.;
        let mut g = [0.; 4];
        for r in 0..2 {
            for c in 0..2 {
                let y = x[r * 2] * w[c] + x[r * 2 + 1] * w[2 + c];
                expected_loss += y * y;
                for k in 0..2 {
                    g[k * 2 + c] += 2. * y * x[r * 2 + k];
                }
            }
        }
        for i in 0..4 {
            w[i] -= 0.01 * g[i];
        }
        let bytes = pollster::block_on(program.read(loss.as_dyn())).unwrap();
        let got = f32::from_le_bytes(bytes.try_into().unwrap());
        assert!(
            (got - expected_loss).abs() < 1e-5,
            "{got} vs {expected_loss}"
        );
        let bytes = pollster::block_on(program.read(weight.as_dyn())).unwrap();
        let got: Vec<_> = bytes
            .chunks_exact(4)
            .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
            .collect();
        for (a, b) in got.iter().zip(w) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
    assert_eq!(program.dispatch_count(), 6);
    assert!(program.write(input.as_dyn(), &[0; 4]).is_err());
    program.export(&[weight.as_dyn().clone()]).unwrap();
    let got = weight.to_vec_f32();
    for (a, b) in got.iter().zip(w) {
        assert!((a - b).abs() < 1e-5);
    }
}

#[test]
fn independent_workloads_pack_without_cross_job_reads_or_reused_live_storage() {
    use fusor::program::ProgramOptions;
    let device = Device::gpu_blocking().unwrap();
    let x = Tensor::<2, f32>::from_slice(&device, [65, 17], &[0.; 65 * 17]);
    let w = Tensor::<2, f32>::from_slice(&device, [17, 19], &[0.25; 17 * 19]);
    let y = Tensor::<2, f32>::from_slice(&device, [33, 9], &[0.5; 33 * 9]);
    let z = Tensor::<2, f32>::from_slice(&device, [9, 7], &[-0.125; 9 * 7]);
    let a = x.matmul(&w);
    let b = y.matmul(&z);
    let squares = x.sqr();
    let rows = a.sum::<1>(1);
    for (matrix_acceleration, subgroup_acceleration) in
        [(false, false), (false, true), (true, true)]
    {
        let mut p = pollster::block_on(TrainingProgram::compile_with_options(
            &[
                a.as_dyn().clone(),
                b.as_dyn().clone(),
                squares.as_dyn().clone(),
                rows.as_dyn().clone(),
            ],
            &[],
            ProgramOptions {
                workgroups: Some(3),
                matrix_acceleration,
                subgroup_acceleration,
                ..Default::default()
            },
        ))
        .unwrap();
        assert_eq!(
            p.stats().kernels,
            2,
            "independent tiles share a dispatch; their consumer waits"
        );
        for step in 1..=3 {
            let data: Vec<f32> = (0..65 * 17)
                .map(|i| (i % 23) as f32 * 0.01 * step as f32)
                .collect();
            p.write(x.as_dyn(), bytemuck::cast_slice(&data)).unwrap();
            p.run().unwrap();
            for (got, want) in floats(pollster::block_on(p.read(squares.as_dyn())).unwrap())
                .iter()
                .zip(&data)
            {
                assert!((got - want * want).abs() < 1e-6);
            }
            for got in floats(pollster::block_on(p.read(b.as_dyn())).unwrap()) {
                assert!((got + 0.5625).abs() < 1e-6);
            }
            let got = floats(pollster::block_on(p.read(a.as_dyn())).unwrap());
            let sums = floats(pollster::block_on(p.read(rows.as_dyn())).unwrap());
            for row in 0..65 {
                let want = data[row * 17..(row + 1) * 17].iter().sum::<f32>() * 0.25;
                for col in 0..19 {
                    assert!((got[row * 19 + col] - want).abs() < 1e-5);
                }
                assert!((sums[row] - want * 19.).abs() < 1e-4);
            }
        }
    }
}

#[test]
fn collective_reductions_cover_strides_tails_and_integer_bits() {
    use fusor::program::ProgramOptions;
    let device = Device::gpu_blocking().unwrap();
    let data: Vec<f32> = (0..257 * 5)
        .map(|i| (i % 29) as f32 * 0.01 - 0.13)
        .collect();
    let x = Tensor::<2, f32>::from_slice(&device, [257, 5], &data);
    let strided = x.sum::<1>(0);
    let contiguous = x.transpose(0, 1).sum::<1>(1);
    let ints: Vec<i32> = (0..3 * 257).map(|i| i % 17 - 8).collect();
    let y = Tensor::<2, i32>::from_slice(&device, [3, 257], &ints);
    let integer_sum = y.sum::<1>(1);
    for workgroups in [1, 3, 64] {
        for (matrix_acceleration, subgroup_acceleration) in
            [(false, false), (false, true), (true, true)]
        {
            let mut p = pollster::block_on(TrainingProgram::compile_with_options(
                &[
                    strided.as_dyn().clone(),
                    contiguous.as_dyn().clone(),
                    integer_sum.as_dyn().clone(),
                ],
                &[],
                ProgramOptions {
                    workgroups: Some(workgroups),
                    matrix_acceleration,
                    subgroup_acceleration,
                    ..Default::default()
                },
            ))
            .unwrap();
            p.run().unwrap();
            for root in [strided.as_dyn(), contiguous.as_dyn()] {
                let got = floats(pollster::block_on(p.read(root)).unwrap());
                for col in 0..5 {
                    let want = (0..257).map(|row| data[row * 5 + col]).sum::<f32>();
                    assert!((got[col] - want).abs() < 1e-4);
                }
            }
            let got = pollster::block_on(p.read(integer_sum.as_dyn())).unwrap();
            for (row, word) in got.chunks_exact(4).enumerate() {
                assert_eq!(
                    i32::from_le_bytes(word.try_into().unwrap()),
                    ints[row * 257..(row + 1) * 257].iter().sum::<i32>()
                );
            }
        }
    }
}

#[test]
fn indexed_reductions_preserve_order_outer_axes_and_changing_indices() {
    use fusor::program::ProgramOptions;
    let device = Device::gpu_blocking().unwrap();
    for count in [257, 1001] {
        let base_data: Vec<f32> = (0..3 * 97 * 33).map(|i| (i % 7) as f32 * 0.25).collect();
        let base = Tensor::<3, f32>::from_slice(&device, [3, 97, 33], &base_data);
        let indices = Tensor::<1, u32>::from_slice(&device, [count], &vec![0; count]);
        let updates =
            Tensor::<3, f32>::from_slice(&device, [3, count, 33], &vec![0.; 3 * count * 33]);
        let result = base
            .as_dyn()
            .scatter_add(1, indices.as_dyn(), updates.as_dyn())
            .unwrap();
        for (matrix_acceleration, subgroup_acceleration) in
            [(false, false), (false, true), (true, true)]
        {
            let mut p = pollster::block_on(TrainingProgram::compile_with_options(
                std::slice::from_ref(&result),
                &[],
                ProgramOptions {
                    workgroups: Some(3),
                    matrix_acceleration,
                    subgroup_acceleration,
                    ..Default::default()
                },
            ))
            .unwrap();
            for step in 0..3 {
                let idx: Vec<u32> = (0..count)
                    .map(|k| match if step == 2 { 17 } else { (k + step) % 17 } {
                        0 => u32::MAX,
                        1 => 97,
                        17 => 0,
                        _ => ((k + step) % 13) as u32,
                    })
                    .collect();
                // Reordering these f32 additions changes the answer. Missing
                // rows retain their nonzero base; row 256 reuses WG scratch.
                let upd: Vec<f32> = (0..3 * count * 33)
                    .map(|i| match ((i / 33) % count) / 13 % 3 {
                        0 => 1e8,
                        1 => (i % 33 + 1) as f32,
                        _ => -1e8,
                    })
                    .collect();
                let mut want = base_data.clone();
                for outer in 0..3 {
                    for k in 0..count {
                        if idx[k] < 97 {
                            for col in 0..33 {
                                want[(outer * 97 + idx[k] as usize) * 33 + col] +=
                                    upd[(outer * count + k) * 33 + col];
                            }
                        }
                    }
                }
                p.write(indices.as_dyn(), bytemuck::cast_slice(&idx))
                    .unwrap();
                p.write(updates.as_dyn(), bytemuck::cast_slice(&upd))
                    .unwrap();
                p.run().unwrap();
                assert_eq!(floats(pollster::block_on(p.read(&result)).unwrap()), want);
            }
        }
    }
}

#[test]
fn indexed_reductions_preserve_signed_integer_bits() {
    use fusor::program::ProgramOptions;
    let device = Device::gpu_blocking().unwrap();
    let base = Tensor::<2, i32>::from_slice(&device, [65, 33], &vec![-7; 65 * 33]);
    let indices = Tensor::<1, i32>::from_slice(
        &device,
        [257],
        &(0..257)
            .map(|i| if i % 3 == 0 { -1 } else { 64 })
            .collect::<Vec<_>>(),
    );
    let updates = Tensor::<2, i32>::from_slice(&device, [257, 33], &vec![-3; 257 * 33]);
    let result = base
        .as_dyn()
        .scatter_add(0, indices.as_dyn(), updates.as_dyn())
        .unwrap();
    for (matrix_acceleration, subgroup_acceleration) in
        [(false, false), (false, true), (true, true)]
    {
        let mut p = pollster::block_on(TrainingProgram::compile_with_options(
            std::slice::from_ref(&result),
            &[],
            ProgramOptions {
                workgroups: Some(3),
                matrix_acceleration,
                subgroup_acceleration,
                ..Default::default()
            },
        ))
        .unwrap();
        p.run().unwrap();
        let bytes = pollster::block_on(p.read(&result)).unwrap();
        for (i, word) in bytes.chunks_exact(4).enumerate() {
            assert_eq!(
                i32::from_le_bytes(word.try_into().unwrap()),
                if i / 33 == 64 { -7 - 3 * 171 } else { -7 }
            );
        }
    }
}

#[test]
fn matrix_tiles_reuse_initialized_scratch_with_retained_branches() {
    use fusor::program::ProgramOptions;
    let device = Device::gpu_blocking().unwrap();
    // More tiles than workgroups and an odd number of K tiles exercise scratch
    // reuse; retained branches check that later consumers observe every tile.
    let a = Tensor::<2, f32>::from_slice(&device, [544, 33], &vec![0.; 544 * 33]);
    let b = Tensor::<2, f32>::from_slice(&device, [33, 256], &vec![0.125; 33 * 256]);
    let y = a.matmul(&b);
    let squared = y.sqr();
    let shifted = y.as_dyn().add_scalar(0.25f32).unwrap();
    let mixed = squared.as_dyn().add(&shifted).unwrap();
    let transposed = y.transpose(0, 1).as_dyn().add_scalar(0.5f32).unwrap();
    for (matrix_acceleration, subgroup_acceleration) in
        [(false, false), (false, true), (true, true)]
    {
        let mut p = pollster::block_on(TrainingProgram::compile_with_options(
            &[
                y.as_dyn().clone(),
                squared.as_dyn().clone(),
                mixed.clone(),
                transposed.clone(),
            ],
            &[],
            ProgramOptions {
                workgroups: Some(3),
                matrix_acceleration,
                subgroup_acceleration,
                ..Default::default()
            },
        ))
        .unwrap();
        for step in 1..=3 {
            let data: Vec<f32> = (0..544 * 33)
                .map(|i| ((i % 17) as f32 - 8.) * step as f32 * 0.03125)
                .collect();
            p.write(a.as_dyn(), bytemuck::cast_slice(&data)).unwrap();
            p.run().unwrap();
            let ys = floats(pollster::block_on(p.read(y.as_dyn())).unwrap());
            let sq = floats(pollster::block_on(p.read(squared.as_dyn())).unwrap());
            let mix = floats(pollster::block_on(p.read(&mixed)).unwrap());
            let tr = floats(pollster::block_on(p.read(&transposed)).unwrap());
            for row in 0..544 {
                let want = data[row * 33..(row + 1) * 33].iter().sum::<f32>() * 0.125;
                for col in 0..256 {
                    let i = row * 256 + col;
                    assert_eq!(ys[i], want);
                    assert_eq!(sq[i], want * want);
                    assert_eq!(mix[i], want * want + (want + 0.25));
                    assert_eq!(tr[col * 544 + row], want + 0.5);
                }
            }
        }
    }
}

#[test]
fn composed_view_addresses_keep_gather_bounds_before_simplification() {
    use fusor::program::ProgramOptions;
    let device = Device::gpu_blocking().unwrap();
    let data: Vec<f32> = (0..7 * 5 * 9).map(|i| i as f32 * 0.5).collect();
    let x = Tensor::<3, f32>::from_slice(&device, [7, 5, 9], &data);
    let view = x
        .as_dyn()
        .transpose(0, 2)
        .unwrap()
        .narrow(0, 1, 7)
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let idx = Tensor::<1, u32>::from_slice(&device, [5], &[0, 6, 7, u32::MAX, 2]);
    let selected = view.index_select(1, idx.as_dyn()).unwrap();
    let output = selected.add_scalar(0.25f32).unwrap();
    for workgroups in [1, 3] {
        let mut p = pollster::block_on(TrainingProgram::compile_with_options(
            std::slice::from_ref(&output),
            &[],
            ProgramOptions {
                workgroups: Some(workgroups),
                ..Default::default()
            },
        ))
        .unwrap();
        p.run().unwrap();
        let got = floats(pollster::block_on(p.read(&output)).unwrap());
        for mid in 0..5 {
            for (k, picked) in [0, 6, 7, u32::MAX, 2].into_iter().enumerate() {
                for first in 0..7 {
                    let want = if picked < 7 {
                        data[(first * 5 + mid) * 9 + picked as usize + 1]
                    } else {
                        0.
                    };
                    assert_eq!(got[(mid * 5 + k) * 7 + first], want + 0.25);
                }
            }
        }
    }
}

#[test]
fn matrix_coordinates_cover_multiple_axes_and_permuted_outputs() {
    use fusor::program::ProgramOptions;
    use fusor_ir::ir::logical::{EinSpec, Label};
    let device = Device::gpu_blocking().unwrap();
    let a: Vec<f32> = (0..6 * 10 * 21).map(|i| (i % 17) as f32 / 16.).collect();
    let b: Vec<f32> = (0..6 * 21 * 22).map(|i| (i % 13) as f32 / 16.).collect();
    let x = Tensor::<6, f32>::from_slice(&device, [2, 3, 2, 5, 3, 7], &a);
    let y = Tensor::<6, f32>::from_slice(&device, [2, 3, 3, 7, 2, 11], &b);
    let product = x
        .as_dyn()
        .contract(
            y.as_dyn(),
            EinSpec {
                a: [0, 1, 2, 3, 4, 5].map(Label).into_iter().collect(),
                b: [0, 1, 4, 5, 6, 7].map(Label).into_iter().collect(),
                out: [1, 0, 3, 7, 2, 6].map(Label).into_iter().collect(),
            },
            fusor::Dtype::F32,
        )
        .unwrap();
    for matrix_acceleration in [false, true] {
        let mut program = pollster::block_on(TrainingProgram::compile_with_options(
            std::slice::from_ref(&product),
            &[],
            ProgramOptions {
                matrix_acceleration,
                ..Default::default()
            },
        ))
        .unwrap();
        program.run().unwrap();
        let actual = floats(pollster::block_on(program.read(&product)).unwrap());
        for batch in 0..6 {
            for row in 0..10 {
                for col in 0..22 {
                    let expected: f32 = (0..21)
                        .map(|k| a[(batch * 10 + row) * 21 + k] * b[(batch * 21 + k) * 22 + col])
                        .sum();
                    let index = (((((batch % 3 * 2 + batch / 3) * 5 + row % 5) * 11 + col % 11)
                        * 2
                        + row / 5)
                        * 2)
                        + col / 11;
                    assert!(
                        (actual[index] - expected).abs() < 1e-5,
                        "batch {batch} row {row} col {col}: {} vs {expected}",
                        actual[index]
                    );
                }
            }
        }
    }
}
