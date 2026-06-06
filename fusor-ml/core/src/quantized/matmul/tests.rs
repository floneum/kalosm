use super::*;

#[cfg(test)]
mod selection_tests {
    use super::*;
    use crate::kernel_selection::{CooperativeMatrixCaps, assert_selector_generates};

    fn caps(high_tile_limits: bool) -> KernelDeviceCaps {
        KernelDeviceCaps {
            max_compute_invocations_per_workgroup: if high_tile_limits { 1024 } else { 256 },
            max_compute_workgroup_storage_size: if high_tile_limits {
                64 * 1024
            } else {
                16 * 1024
            },
            ..KernelDeviceCaps::test_caps()
        }
    }

    fn no_coop_caps(high_tile_limits: bool) -> KernelDeviceCaps {
        KernelDeviceCaps {
            cooperative_matrix: CooperativeMatrixCaps::default(),
            ..caps(high_tile_limits)
        }
    }

    fn no_subgroup_caps(high_tile_limits: bool) -> KernelDeviceCaps {
        KernelDeviceCaps {
            subgroups_supported: false,
            cooperative_matrix: CooperativeMatrixCaps::default(),
            ..caps(high_tile_limits)
        }
    }

    fn variable_subgroup_caps(high_tile_limits: bool) -> KernelDeviceCaps {
        KernelDeviceCaps {
            cooperative_matrix: CooperativeMatrixCaps::default(),
            min_subgroup_size: 8,
            max_subgroup_size: 32,
            ..caps(high_tile_limits)
        }
    }

    fn ctx(format: tile_ir::GgmlQuantFormat) -> QMatmulDirectCtx {
        QMatmulDirectCtx { format }
    }

    const fn qtile(bm: u32, bn: u32) -> CoopTile {
        CoopTile::new(bm, bn, QMATMUL_COOP_BK)
    }

    #[test]
    fn qmatmul_direct_selector_generates_each_variant() {
        let selector = qmatmul_direct_selector();
        let q4 = tile_ir::GgmlQuantFormat::Q4_0;
        let cases = [
            (
                QMatmulPath::Q5SmallSingleRow,
                ctx(tile_ir::GgmlQuantFormat::Q5_0),
                caps(false),
            ),
            (QMatmulPath::SingleRow, ctx(q4), caps(false)),
            (
                QMatmulPath::Q8Wide(qtile(64, 128)),
                ctx(tile_ir::GgmlQuantFormat::Q8_0),
                caps(true),
            ),
            (
                QMatmulPath::Tile {
                    tile: qtile(128, 128),
                    cached: false,
                },
                ctx(q4),
                caps(true),
            ),
            (
                QMatmulPath::Tile {
                    tile: qtile(128, 64),
                    cached: false,
                },
                ctx(q4),
                caps(false),
            ),
            (
                QMatmulPath::Tile {
                    tile: qtile(64, 128),
                    cached: false,
                },
                ctx(q4),
                caps(false),
            ),
            (
                QMatmulPath::Tile {
                    tile: qtile(64, 64),
                    cached: true,
                },
                ctx(q4),
                caps(false),
            ),
            (
                QMatmulPath::Tile {
                    tile: qtile(64, 64),
                    cached: false,
                },
                ctx(q4),
                caps(false),
            ),
            (QMatmulPath::Workgroup, ctx(q4), no_coop_caps(false)),
        ];
        assert_selector_generates(&selector, cases);
    }

    #[test]
    fn qmatmul_direct_selector_requires_coop_capability_for_coop_tiles() {
        let selector = qmatmul_direct_selector();
        let shape = KernelShape::new([128, 4096, 5120]);
        let q4k = tile_ir::GgmlQuantFormat::Q4K;
        assert_eq!(
            selector.select(shape, &ctx(q4k), no_coop_caps(true)),
            Some(QMatmulPath::Workgroup)
        );
        assert_eq!(
            selector.select(shape, &ctx(q4k), caps(true)),
            Some(QMatmulPath::Tile {
                tile: qtile(128, 128),
                cached: false
            })
        );
        assert!(!qmatmul_coop_supported(no_coop_caps(true)));
        assert_eq!(
            qmatmul_m_pad_target_for_caps(48, 5120, no_coop_caps(true)),
            None
        );
        assert_eq!(
            qmatmul_m_pad_target_for_caps(48, 5120, caps(true)),
            Some(128)
        );
    }

    #[test]
    fn single_row_direct_path_requires_trusted_runtime_subgroups_not_coop_matrix() {
        let format = tile_ir::GgmlQuantFormat::Q4K;
        let k = 4096;
        let n = 8192;
        let qgemv_supported = |caps| qgemv_subgroup_supported(format, k, n, caps);

        assert!(!qmatmul_path_requires_coop(QMatmulPath::SingleRow));
        assert!(!qmatmul_path_requires_coop(QMatmulPath::Q5SmallSingleRow));
        assert!(!qmatmul_path_requires_coop(QMatmulPath::Workgroup));
        assert!(qmatmul_path_requires_coop(QMatmulPath::Tile {
            tile: qtile(64, 64),
            cached: false,
        }));
        assert!(qgemv_supported(no_coop_caps(false)));
        assert!(qgemv_supported(variable_subgroup_caps(false)));

        let selector = qmatmul_direct_selector();
        let caps = no_coop_caps(false);
        assert!(caps.subgroups_supported);
        assert!(!qmatmul_coop_supported(caps));
        assert_eq!(
            selector.select(
                KernelShape::new([1, 4096, 8192]),
                &ctx(tile_ir::GgmlQuantFormat::Q4K),
                caps,
            ),
            Some(QMatmulPath::SingleRow)
        );
        assert_eq!(
            selector.select(
                KernelShape::new([1, 4096, 8192]),
                &ctx(tile_ir::GgmlQuantFormat::Q4K),
                no_subgroup_caps(false),
            ),
            Some(QMatmulPath::Workgroup)
        );
    }

    #[test]
    fn coop_acc_init_only_claims_shapes_the_coop_path_will_take() {
        assert!(qmatmul_variant_supports_coop_acc_init(
            QMatmulPath::Tile {
                tile: qtile(64, 128),
                cached: false
            },
            64,
            512,
            128,
            true,
        ));
        assert!(!qmatmul_variant_supports_coop_acc_init(
            QMatmulPath::Tile {
                tile: qtile(64, 128),
                cached: false
            },
            63,
            512,
            128,
            true,
        ));
        assert!(!qmatmul_variant_supports_coop_acc_init(
            QMatmulPath::Tile {
                tile: qtile(64, 64),
                cached: false
            },
            2,
            512,
            4,
            true,
        ));
        assert!(!qmatmul_variant_supports_coop_acc_init(
            QMatmulPath::Tile {
                tile: qtile(64, 128),
                cached: false
            },
            64,
            510,
            128,
            true,
        ));
        assert!(!qmatmul_variant_supports_coop_acc_init(
            QMatmulPath::Tile {
                tile: qtile(64, 128),
                cached: false
            },
            64,
            512,
            128,
            false,
        ));
    }

    #[test]
    fn custom_accumulator_offsets_must_cover_output_width() {
        assert!(qmatmul_custom_accumulator_offsets_cover_output(1, 9, 10, 1));
        assert!(qmatmul_custom_accumulator_offsets_cover_output(
            1, 10, 10, 0
        ));
        assert!(!qmatmul_custom_accumulator_offsets_cover_output(
            1, 5, 10, 1
        ));
        assert!(!qmatmul_custom_accumulator_offsets_cover_output(
            1, 10, 10, 1
        ));
        assert!(!qmatmul_custom_accumulator_offsets_cover_output(
            1,
            u32::MAX,
            10,
            1
        ));
        assert!(!qmatmul_custom_accumulator_offsets_cover_output(
            2, 9, 10, 1
        ));
    }

    #[test]
    fn indexed_post_accumulator_offsets_require_subgroup_direct_qgemv_support() {
        let format = tile_ir::GgmlQuantFormat::Q4KNative;
        let m = 1;
        let k = 4096;
        let n = 4096;
        let supported = |caps, max_workgroups| {
            let variant = select_qmatmul_direct_variant(format, m, k, n, caps);
            qmatmul_custom_accumulator_offsets_supported(
                format,
                variant,
                m,
                k,
                n,
                n * 2,
                n,
                max_workgroups,
            )
        };

        assert!(supported(caps(false), 65_535));
        assert!(!supported(no_subgroup_caps(false), 65_535));
        assert!(!supported(caps(false), 1));
    }
}

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use std::{mem::size_of, sync::Arc};

    use fusor_gguf::{BlockQ4_0, BlockQ4K, BlockQ5_0, BlockQ5K, BlockQ6K, BlockQ8_0, GgufBlock};
    use fusor_tile_ir_runtime::DirectKernelBinding;

    use super::*;
    use crate::{compute_graph::ComputeGraphInner, mir::workgroup_shape::WorkgroupShape};

    fn push_f16(bytes: &mut Vec<u8>, value: f32) {
        bytes.extend_from_slice(&half::f16::from_f32(value).to_le_bytes());
    }

    fn packed_nibble_byte(low: usize, high: usize) -> u8 {
        ((low & 0x0F) as u8) | (((high & 0x0F) as u8) << 4)
    }

    fn padded_copy_size(size: u64) -> u64 {
        let align_mask = wgpu::COPY_BUFFER_ALIGNMENT - 1;
        ((size + align_mask) & !align_mask).max(wgpu::COPY_BUFFER_ALIGNMENT)
    }

    fn patterned_q4k_bytes(shape: [usize; 2]) -> Vec<u8> {
        let block_count = shape.iter().product::<usize>() / BlockQ4K::BLOCK_SIZE;
        let mut bytes = Vec::with_capacity(block_count * size_of::<BlockQ4K>());
        for block in 0..block_count {
            push_f16(&mut bytes, 0.004);
            push_f16(&mut bytes, 0.0005);
            for i in 0..BlockQ4K::SCALES_SIZE {
                bytes.push((((block * 5 + i * 3) % 24) + 1) as u8);
            }
            for i in 0..BlockQ4K::WEIGHTS_SIZE {
                bytes.push(packed_nibble_byte(
                    10 + ((block + i * 2) % 6),
                    11 + ((block * 3 + i) % 5),
                ));
            }
        }
        bytes
    }

    fn patterned_q6k_bytes(shape: [usize; 2]) -> Vec<u8> {
        let block_count = shape.iter().product::<usize>() / BlockQ6K::BLOCK_SIZE;
        let mut bytes = Vec::with_capacity(block_count * size_of::<BlockQ6K>());
        for block in 0..block_count {
            for i in 0..BlockQ6K::WEIGHTS_LOW_BITS_SIZE {
                bytes.push(packed_nibble_byte(
                    block * 5 + i * 3 + 1,
                    block * 7 + i * 11 + 2,
                ));
            }
            for i in 0..BlockQ6K::WEIGHTS_HIGH_BITS_SIZE {
                bytes.push(((block * 17 + i * 9 + 0x12) & 0xFF) as u8);
            }
            for i in 0..BlockQ6K::SCALES_SIZE {
                let scale = ((block * 5 + i * 2) % 7 + 1) as i8;
                bytes.push(scale as u8);
            }
            push_f16(&mut bytes, 0.0035);
        }
        bytes
    }

    #[test]
    fn qmatmul_direct_kernel_binds_compact_quantized_weight_buffer() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            let weight_shape = [128usize, 256usize];
            let element_count = weight_shape.iter().product::<usize>();
            let block_count = element_count / BlockQ4_0::BLOCK_SIZE;
            let raw_bytes = vec![0; block_count * size_of::<BlockQ4_0>()];
            let matrix =
                QMatrix::from_parts(&device, &raw_bytes, weight_shape.into(), GgmlType::Q4_0)
                    .unwrap();

            let compact_len = block_count
                * match matrix.storage_layout() {
                    QMatrixStorageLayout::Native => size_of::<<BlockQ4_0 as GgufBlock>::AsBytes>(),
                    QMatrixStorageLayout::GpuF32Scales => {
                        size_of::<<BlockQ4_0 as GgufBlock>::BytesF32>()
                    }
                };
            let dense_len = element_count * size_of::<f32>();
            assert_eq!(matrix.buffer().size(), padded_copy_size(compact_len as u64));
            assert!(matrix.buffer().size() < padded_copy_size(dense_len as u64));

            let input =
                TensorData::new_for_shape(&device, &[1, weight_shape[1]], DataTypeEnum::F32);
            let output =
                TensorData::new_for_shape(&device, &[1, weight_shape[0]], DataTypeEnum::F32);
            let graph = ComputeGraphInner::new_for_test(device.downgrade());
            let operation = QMatMulOperation {
                input_datatype: DataTypeEnum::F32,
                input: NodeIndex::new(0),
                matrix: matrix.clone(),
                in_shape: Box::new([1, weight_shape[1]]),
                out_shape: Box::new([1, weight_shape[0]]),
                pre_element_wise_expr: None,
                post_element_wise_expr: None,
                post_accumulator_offsets: Box::new([]),
            };
            let kernel = operation
                .build_direct_kernel(
                    &graph,
                    &WorkgroupShape::new(256, 1, 1),
                    &[input.into(), matrix.clone().into(), output.into()],
                )
                .expect("qmatmul should build a direct quantized kernel");

            let bindings = kernel.bindings_for_test();
            assert_eq!(bindings.len(), 3);
            let DirectKernelBinding {
                binding,
                buffer,
                read_only,
            } = &bindings[1];
            assert_eq!(*binding, 1);
            assert!(*read_only);
            assert!(Arc::ptr_eq(buffer, matrix.buffer()));
        });
    }

    #[test]
    fn q4k_multirow_qmatmul_builds_direct_kernel_when_scalar_grid_would_exceed_cap() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            let weight_shape = [14336usize, 4096usize];
            let input_shape = [1usize, 48usize, weight_shape[1]];
            let output_shape = [1usize, 48usize, weight_shape[0]];
            let element_count = weight_shape.iter().product::<usize>();
            let block_count = element_count / BlockQ4K::BLOCK_SIZE;
            let raw_bytes = vec![0; block_count * size_of::<BlockQ4K>()];
            let matrix =
                QMatrix::from_parts(&device, &raw_bytes, weight_shape.into(), GgmlType::Q4K)
                    .unwrap();

            let input = TensorData::new_for_shape(&device, &input_shape, DataTypeEnum::F32);
            let output = TensorData::new_for_shape(&device, &output_shape, DataTypeEnum::F32);
            let graph = ComputeGraphInner::new_for_test(device.downgrade());
            let operation = QMatMulOperation {
                input_datatype: DataTypeEnum::F32,
                input: NodeIndex::new(0),
                matrix: matrix.clone(),
                in_shape: input_shape.into(),
                out_shape: output_shape.into(),
                pre_element_wise_expr: None,
                post_element_wise_expr: None,
                post_accumulator_offsets: Box::new([]),
            };

            operation
                .build_direct_kernel(
                    &graph,
                    &WorkgroupShape::new(32, 1, 1),
                    &[input.into(), matrix.into(), output.into()],
                )
                .expect("Q4K multi-row qmatmul should build a direct kernel");
        });
    }

    #[test]
    fn q4k_multirow_qmatmul_zero_weights_produce_zero_output_when_grid_exceeds_old_cap() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            let weight_shape = [14336usize, 4096usize];
            let input_shape = [1usize, 48usize, weight_shape[1]];
            let element_count = weight_shape.iter().product::<usize>();
            let block_count = element_count / BlockQ4K::BLOCK_SIZE;
            let raw_bytes = vec![0; block_count * size_of::<BlockQ4K>()];
            let matrix =
                QMatrix::from_parts(&device, &raw_bytes, weight_shape.into(), GgmlType::Q4K)
                    .unwrap();
            let input_values = vec![0.25f32; input_shape.iter().product()];
            let input = Tensor::from_slice::<f32>(&device, input_shape, &input_values);

            let result = input.q_mat_mul(&matrix).as_slice::<3, f32>().await.unwrap();

            assert_eq!(result.shape(), &[1, 48, weight_shape[0]]);
            assert!(
                result.as_slice().iter().all(|value| *value == 0.0),
                "zero Q4K weights should produce zero output for the multi-row Llama shape"
            );
        });
    }

    fn patterned_q8_0_bytes(shape: [usize; 2]) -> Vec<u8> {
        let block_count = shape.iter().product::<usize>() / BlockQ8_0::BLOCK_SIZE;
        let mut bytes = Vec::with_capacity(block_count * size_of::<BlockQ8_0>());
        for block in 0..block_count {
            push_f16(&mut bytes, 0.01);
            for i in 0..BlockQ8_0::WEIGHTS_SIZE {
                let value = (((block * 5 + i * 3) % 64) as i32 - 32) as i8;
                bytes.push(value as u8);
            }
        }
        bytes
    }

    fn patterned_q5_0_bytes(shape: [usize; 2]) -> Vec<u8> {
        let block_count = shape.iter().product::<usize>() / BlockQ5_0::BLOCK_SIZE;
        let mut bytes = Vec::with_capacity(block_count * size_of::<BlockQ5_0>());
        for block in 0..block_count {
            push_f16(&mut bytes, 0.01);
            for i in 0..BlockQ5_0::WEIGHTS_HIGH_BITS_SIZE {
                bytes.push(((block * 7 + i * 13) & 0xFF) as u8);
            }
            for i in 0..BlockQ5_0::WEIGHTS_LOW_BITS_SIZE {
                bytes.push(packed_nibble_byte(block + i, block * 2 + i + 1));
            }
        }
        bytes
    }

    #[test]
    fn q4k_native_dequantize_matches_patterned_blocks() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            let shape = [4usize, 4096usize];
            let raw_bytes = patterned_q4k_bytes(shape);
            let matrix =
                QMatrix::from_parts(&device, &raw_bytes, shape.into(), GgmlType::Q4K).unwrap();
            let blocks: &[BlockQ4K] = bytemuck::cast_slice(&raw_bytes);

            let result = matrix
                .dequantize::<f32>()
                .as_slice::<2, f32>()
                .await
                .unwrap();

            let blocks_per_row = shape[1] / BlockQ4K::BLOCK_SIZE;
            for row in 0..shape[0] {
                for offset in 0..shape[1] {
                    let block = &blocks[row * blocks_per_row + offset / BlockQ4K::BLOCK_SIZE];
                    let expected = block.dequantize();
                    let actual = result[[row, offset]];
                    let expected = expected.as_ref()[offset % BlockQ4K::BLOCK_SIZE];
                    assert!(
                        (actual - expected).abs() <= 1e-6,
                        "row={row} offset={offset} actual={actual} expected={expected}"
                    );
                }
            }
        });
    }

    #[test]
    fn q4k_paired_silu_single_row_no_subgroup_resolves() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            let device = device.without_subgroups();

            let weight_shape = [64usize, 512usize];
            let raw_bytes = patterned_q4k_bytes(weight_shape);
            let matrix =
                QMatrix::from_parts(&device, &raw_bytes, weight_shape.into(), GgmlType::Q4K)
                    .unwrap();
            let input_values = (0..weight_shape[1])
                .map(|index| {
                    let bucket = (index.wrapping_mul(29).wrapping_add(5)) % 61;
                    (bucket as f32 - 30.0) * 0.001
                })
                .collect::<Vec<_>>();
            let input = Tensor::from_slice::<f32>(&device, [1, weight_shape[1]], &input_values);

            let result = input
                .q_mat_mul_paired_silu_product(&matrix)
                .as_slice::<2, f32>()
                .await
                .unwrap();

            assert_eq!(result.shape(), &[1, weight_shape[0] / 2]);
            assert!(result.as_slice().iter().all(|value| value.is_finite()));
        });
    }

    #[test]
    fn q4k_large_single_row_qgemv_handles_tail_columns_with_subgroups() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            if !device.subgroups_supported() {
                return;
            }

            let weight_shape = [8193usize, 4096usize];
            let blocks_per_row = weight_shape[1] / BlockQ4K::BLOCK_SIZE;
            let raw_bytes = patterned_q4k_bytes(weight_shape);
            let matrix =
                QMatrix::from_parts(&device, &raw_bytes, weight_shape.into(), GgmlType::Q4K)
                    .unwrap();
            let blocks: &[BlockQ4K] = bytemuck::cast_slice(&raw_bytes);
            let input_values = (0..weight_shape[1])
                .map(|index| {
                    let bucket = (index.wrapping_mul(37).wrapping_add(11)) % 101;
                    (bucket as f32 - 50.0) * 0.0025
                })
                .collect::<Vec<_>>();
            let input = Tensor::from_slice::<f32>(&device, [1, weight_shape[1]], &input_values);

            let result = input.q_mat_mul(&matrix).as_slice::<2, f32>().await.unwrap();

            assert_eq!(result.shape(), &[1, weight_shape[0]]);
            for col in [0usize, 1, 63, 64, 511, 1024, 8191, 8192] {
                let expected = (0..blocks_per_row)
                    .map(|block_col| {
                        let block = &blocks[col * blocks_per_row + block_col];
                        let weights = block.dequantize();
                        weights
                            .as_ref()
                            .iter()
                            .enumerate()
                            .map(|(offset, weight)| {
                                input_values[block_col * BlockQ4K::BLOCK_SIZE + offset] * *weight
                            })
                            .sum::<f32>()
                    })
                    .sum::<f32>();
                let actual = result[[0, col]];
                assert!(
                    (actual - expected).abs() <= 1e-2_f32.max(expected.abs() * 1.0e-4),
                    "col={col} actual={actual} expected={expected}"
                );
            }
        });
    }

    // Regression test for native decode gibberish: the single-row (m=1, decode)
    // subgroup qgemv must agree with the no-subgroup path, which is the
    // reference web/native-without-subgroups run and is known-correct. A
    // divergence here is exactly the per-token logit corruption that turns
    // generation into token-salad. Covers each quantized format the model uses,
    // at a Llama-ish projection shape.
    // Isolates whether the cooperative-matrix bug is quantized-specific (the
    // dequantizing tile fill) or general to the coop matmul codegen: a plain
    // dense f32 matmul at coop-eligible dims must also agree between the
    // subgroup (coop) and no-subgroup paths. If THIS diverges, the bug is the
    // cooperative-matrix path itself (independent of quantization).
    #[test]
    fn dense_coop_matmul_subgroup_matches_no_subgroup() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            if !device.subgroups_supported() {
                return;
            }
            let no_sg = device.without_subgroups();

            let (m, k, n) = (128usize, 256usize, 128usize);
            let a = (0..m * k)
                .map(|i| ((i.wrapping_mul(31).wrapping_add(7)) % 97) as f32 * 0.003 - 0.14)
                .collect::<Vec<_>>();
            let b = (0..k * n)
                .map(|i| ((i.wrapping_mul(53).wrapping_add(3)) % 89) as f32 * 0.004 - 0.17)
                .collect::<Vec<_>>();

            let out_sg = {
                let a_t = Tensor::from_slice::<f32>(&device, [m, k], &a);
                let b_t = Tensor::from_slice::<f32>(&device, [k, n], &b);
                let s = a_t.mat_mul(&b_t).as_slice::<2, f32>().await.unwrap();
                (0..m)
                    .flat_map(|r| (0..n).map(move |c| (r, c)))
                    .map(|(r, c)| s[[r, c]])
                    .collect::<Vec<_>>()
            };
            let out_no = {
                let a_t = Tensor::from_slice::<f32>(&no_sg, [m, k], &a);
                let b_t = Tensor::from_slice::<f32>(&no_sg, [k, n], &b);
                let s = a_t.mat_mul(&b_t).as_slice::<2, f32>().await.unwrap();
                (0..m)
                    .flat_map(|r| (0..n).map(move |c| (r, c)))
                    .map(|(r, c)| s[[r, c]])
                    .collect::<Vec<_>>()
            };

            let mut worst = 0.0f32;
            let mut wi = 0usize;
            let (mut wa, mut wb) = (0.0f32, 0.0f32);
            for (i, (x, y)) in out_sg.iter().zip(&out_no).enumerate() {
                let err = (x - y).abs();
                if err > worst {
                    worst = err;
                    wi = i;
                    wa = *x;
                    wb = *y;
                }
            }
            assert!(
                worst <= 1.0e-3 + wb.abs() * 1.0e-3,
                "dense coop matmul diverges from no-subgroup at {wi}: subgroup={wa} no_subgroup={wb} abs_err={worst}"
            );
        });
    }

    #[test]
    fn single_row_qgemv_subgroup_matches_no_subgroup() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            if !device.subgroups_supported() {
                return;
            }
            let no_sg = device.without_subgroups();

            // (name, [n, k], bytes, format). The large-block K-quants (q4k/q6k)
            // are exercised at the ffn k=4096 regime; the small-block formats
            // (q8_0/q5_0) at the k=896 projection regime the model actually uses
            // for its attention/mlp projections (896 is not a multiple of 256, so
            // those weights can only be block-32 formats).
            let cases: [(&str, [usize; 2], Vec<u8>, GgmlType); 5] = [
                (
                    "q4k",
                    [896, 4096],
                    patterned_q4k_bytes([896, 4096]),
                    GgmlType::Q4K,
                ),
                (
                    "q6k",
                    [896, 4096],
                    patterned_q6k_bytes([896, 4096]),
                    GgmlType::Q6K,
                ),
                (
                    "q8_0",
                    [896, 896],
                    patterned_q8_0_bytes([896, 896]),
                    GgmlType::Q8_0,
                ),
                (
                    "q5_0",
                    [896, 896],
                    patterned_q5_0_bytes([896, 896]),
                    GgmlType::Q5_0,
                ),
                // lm-head / tied-embedding shape: huge n, q5_0, k=hidden. Produces
                // the logits every token, so a wrong subgroup qgemv here is exactly
                // "wrong token sampled => gibberish".
                (
                    "q5_0_lmhead",
                    [151936, 896],
                    patterned_q5_0_bytes([151936, 896]),
                    GgmlType::Q5_0,
                ),
            ];

            async fn row(t: Tensor) -> Vec<f32> {
                let slice = t.as_slice::<2, f32>().await.unwrap();
                let cols = slice.shape()[1];
                (0..cols).map(|col| slice[[0, col]]).collect()
            }
            async fn rows(t: Tensor) -> Vec<f32> {
                let slice = t.as_slice::<2, f32>().await.unwrap();
                let shape = slice.shape();
                let (m, cols) = (shape[0], shape[1]);
                let mut out = Vec::with_capacity(m * cols);
                for r in 0..m {
                    for c in 0..cols {
                        out.push(slice[[r, c]]);
                    }
                }
                out
            }
            // The two paths use different reduction kernels, so allow FP rounding
            // slack; gibberish (a stale/miscompiled subgroup kernel) is orders of
            // magnitude larger.
            fn assert_match(name: &str, op: &str, sg: &[f32], no: &[f32]) {
                assert_eq!(sg.len(), no.len(), "{name}/{op}: length mismatch");
                let mut worst = 0.0f32;
                let (mut wc, mut wa, mut wb) = (0usize, 0.0f32, 0.0f32);
                for (col, (a, b)) in sg.iter().zip(no).enumerate() {
                    let err = (a - b).abs();
                    if err > worst {
                        worst = err;
                        wc = col;
                        wa = *a;
                        wb = *b;
                    }
                }
                assert!(
                    worst <= 1.0e-2 + wb.abs() * 1.0e-2,
                    "{name}/{op}: subgroup qgemv diverges from no-subgroup at col {wc}: \
                     subgroup={wa} no_subgroup={wb} abs_err={worst}"
                );
            }

            for (name, weight_shape, raw_bytes, ty) in cases {
                let [n, k] = weight_shape;
                let input_values = (0..k)
                    .map(|index| {
                        let bucket = (index.wrapping_mul(37).wrapping_add(11)) % 101;
                        (bucket as f32 - 50.0) * 0.0025
                    })
                    .collect::<Vec<_>>();
                let extra = (0..n)
                    .map(|i| (i % 13) as f32 * 0.01 - 0.06)
                    .collect::<Vec<_>>();
                let mat_sg =
                    QMatrix::from_parts(&device, &raw_bytes, weight_shape.into(), ty).unwrap();
                let mat_no =
                    QMatrix::from_parts(&no_sg, &raw_bytes, weight_shape.into(), ty).unwrap();
                let in_sg = Tensor::from_slice::<f32>(&device, [1, k], &input_values);
                let in_no = Tensor::from_slice::<f32>(&no_sg, [1, k], &input_values);

                // Plain qgemv.
                assert_match(
                    name,
                    "plain",
                    &row(in_sg.q_mat_mul(&mat_sg)).await,
                    &row(in_no.q_mat_mul(&mat_no)).await,
                );

                // Fused paired-silu-product epilogue (the MLP gate/up matmul) —
                // the dominant matmul in decode and the one the previous fast
                // cache could not key.
                assert_match(
                    name,
                    "silu_product",
                    &row(in_sg.q_mat_mul_paired_silu_product(&mat_sg)).await,
                    &row(in_no.q_mat_mul_paired_silu_product(&mat_no)).await,
                );

                // Fused add2 epilogue (residual add after a projection).
                let first_sg = Tensor::from_slice::<f32>(&device, [1, n], &extra);
                let second_sg = Tensor::from_slice::<f32>(&device, [1, n], &extra);
                let first_no = Tensor::from_slice::<f32>(&no_sg, [1, n], &extra);
                let second_no = Tensor::from_slice::<f32>(&no_sg, [1, n], &extra);
                assert_match(
                    name,
                    "add2",
                    &row(in_sg.q_mat_mul_add2(&mat_sg, &first_sg, &second_sg)).await,
                    &row(in_no.q_mat_mul_add2(&mat_no, &first_no, &second_no)).await,
                );

                // Multi-row (prefill) regime: the prompt is processed at m>1 with
                // a different (tile/coop) subgroup kernel than decode's qgemv. A
                // wrong prefill matmul poisons the KV cache, so every later decode
                // token reads bad context and produces gibberish.
                let m = 16usize;
                let multi = (0..m * k)
                    .map(|i| {
                        let bucket = (i.wrapping_mul(53).wrapping_add(7)) % 97;
                        (bucket as f32 - 48.0) * 0.0021
                    })
                    .collect::<Vec<_>>();
                let multi_sg = Tensor::from_slice::<f32>(&device, [m, k], &multi);
                let multi_no = Tensor::from_slice::<f32>(&no_sg, [m, k], &multi);
                assert_match(
                    name,
                    "plain_multirow",
                    &rows(multi_sg.q_mat_mul(&mat_sg)).await,
                    &rows(multi_no.q_mat_mul(&mat_no)).await,
                );
            }
        });
    }

    #[test]
    fn q4k_native_small_multirow_qmatmul_matches_one_hot_reference() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            let weight_shape = [4usize, 4096usize];
            let input_shape = [8usize, weight_shape[1]];
            let selected_k = 777usize;
            let selected_block_in_row = selected_k / BlockQ4K::BLOCK_SIZE;
            let selected_offset = selected_k % BlockQ4K::BLOCK_SIZE;
            let blocks_per_row = weight_shape[1] / BlockQ4K::BLOCK_SIZE;
            let raw_bytes = patterned_q4k_bytes(weight_shape);
            let matrix =
                QMatrix::from_parts(&device, &raw_bytes, weight_shape.into(), GgmlType::Q4K)
                    .unwrap();
            let blocks: &[BlockQ4K] = bytemuck::cast_slice(&raw_bytes);
            let selected_weights = (0..weight_shape[0])
                .map(|row| {
                    let block = &blocks[row * blocks_per_row + selected_block_in_row];
                    block.dequantize().as_ref()[selected_offset]
                })
                .collect::<Vec<_>>();
            let mut input_values = vec![0.0f32; input_shape.iter().product()];
            for row in 0..input_shape[0] {
                input_values[row * weight_shape[1] + selected_k] = 0.125 + row as f32 * 0.01;
            }
            let input = Tensor::from_slice::<f32>(&device, input_shape, &input_values);
            let result = input.q_mat_mul(&matrix).as_slice::<2, f32>().await.unwrap();

            assert_eq!(result.shape(), &[input_shape[0], weight_shape[0]]);
            for row in 0..input_shape[0] {
                let scale = input_values[row * weight_shape[1] + selected_k];
                for col in 0..weight_shape[0] {
                    let actual = result[[row, col]];
                    let expected = scale * selected_weights[col];
                    assert!(
                        (actual - expected).abs() <= 1e-3,
                        "row={row} col={col} actual={actual} expected={expected}"
                    );
                }
            }
        });
    }

    #[test]
    fn q4k_native_small_multirow_qmatmul_zero_input_produces_zero() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            let weight_shape = [4usize, 4096usize];
            let input_shape = [8usize, weight_shape[1]];
            let raw_bytes = patterned_q4k_bytes(weight_shape);
            let matrix =
                QMatrix::from_parts(&device, &raw_bytes, weight_shape.into(), GgmlType::Q4K)
                    .unwrap();
            let input_values = vec![0.0f32; input_shape.iter().product()];
            let input = Tensor::from_slice::<f32>(&device, input_shape, &input_values);

            let result = input.q_mat_mul(&matrix).as_slice::<2, f32>().await.unwrap();

            assert_eq!(result.shape(), &[input_shape[0], weight_shape[0]]);
            for value in result.as_slice() {
                assert_eq!(*value, 0.0);
            }
        });
    }

    #[test]
    fn q4k_multirow_qmatmul_large_grid_matches_one_hot_reference() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            let weight_shape = [14336usize, 4096usize];
            let input_shape = [1usize, 48usize, weight_shape[1]];
            let selected_k = 777usize;
            let selected_block_in_row = selected_k / BlockQ4K::BLOCK_SIZE;
            let selected_offset = selected_k % BlockQ4K::BLOCK_SIZE;
            let blocks_per_row = weight_shape[1] / BlockQ4K::BLOCK_SIZE;
            let raw_bytes = patterned_q4k_bytes(weight_shape);
            let matrix =
                QMatrix::from_parts(&device, &raw_bytes, weight_shape.into(), GgmlType::Q4K)
                    .unwrap();
            let blocks: &[BlockQ4K] = bytemuck::cast_slice(&raw_bytes);
            let selected_weights = (0..weight_shape[0])
                .map(|row| {
                    let block = &blocks[row * blocks_per_row + selected_block_in_row];
                    block.dequantize().as_ref()[selected_offset]
                })
                .collect::<Vec<_>>();
            let mut input_values = vec![0.0f32; input_shape.iter().product()];
            for row in 0..input_shape[1] {
                input_values[row * weight_shape[1] + selected_k] = 0.125 + row as f32 * 0.01;
            }
            let input = Tensor::from_slice::<f32>(&device, input_shape, &input_values);

            let result = input.q_mat_mul(&matrix).as_slice::<3, f32>().await.unwrap();

            assert_eq!(result.shape(), &[1, 48, weight_shape[0]]);
            for row in 0..input_shape[1] {
                let scale = input_values[row * weight_shape[1] + selected_k];
                for col in [0usize, 1, 63, 64, 511, 1024, 8191, 14335] {
                    let actual = result[[0, row, col]];
                    let expected = scale * selected_weights[col];
                    assert!(
                        (actual - expected).abs() <= 1e-3,
                        "row={row} col={col} actual={actual} expected={expected}"
                    );
                }
            }
        });
    }

    #[test]
    fn q4k_multirow_qmatmul_large_grid_matches_dense_sampled_columns() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            let weight_shape = [14336usize, 4096usize];
            let input_shape = [1usize, 48usize, weight_shape[1]];
            let blocks_per_row = weight_shape[1] / BlockQ4K::BLOCK_SIZE;
            let raw_bytes = patterned_q4k_bytes(weight_shape);
            let matrix =
                QMatrix::from_parts(&device, &raw_bytes, weight_shape.into(), GgmlType::Q4K)
                    .unwrap();
            let blocks: &[BlockQ4K] = bytemuck::cast_slice(&raw_bytes);
            let input_values = (0..input_shape.iter().product::<usize>())
                .map(|index| {
                    let bucket = (index.wrapping_mul(37).wrapping_add(11)) % 101;
                    (bucket as f32 - 50.0) * 0.0025
                })
                .collect::<Vec<_>>();
            let input = Tensor::from_slice::<f32>(&device, input_shape, &input_values);

            let result = input.q_mat_mul(&matrix).as_slice::<3, f32>().await.unwrap();

            assert_eq!(result.shape(), &[1, 48, weight_shape[0]]);
            for row in [0usize, 1, 7, 17, 31, 47] {
                let input_row = &input_values[row * weight_shape[1]..(row + 1) * weight_shape[1]];
                for col in [0usize, 1, 63, 64, 511, 1024, 8191, 14335] {
                    let expected = (0..blocks_per_row)
                        .map(|block_col| {
                            let block = &blocks[col * blocks_per_row + block_col];
                            let weights = block.dequantize();
                            weights
                                .as_ref()
                                .iter()
                                .enumerate()
                                .map(|(offset, weight)| {
                                    input_row[block_col * BlockQ4K::BLOCK_SIZE + offset] * *weight
                                })
                                .sum::<f32>()
                        })
                        .sum::<f32>();
                    let actual = result[[0, row, col]];
                    assert!(
                        (actual - expected).abs() <= 1e-2_f32.max(expected.abs() * 1.0e-4),
                        "row={row} col={col} actual={actual} expected={expected}"
                    );
                }
            }
        });
    }

    #[test]
    fn q6k_large_qgemv_matches_one_hot_reference_when_grid_exceeds_old_cap() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            let weight_shape = [32768usize, 4096usize];
            let selected_k = 777usize;
            let selected_block_in_row = selected_k / BlockQ6K::BLOCK_SIZE;
            let selected_offset = selected_k % BlockQ6K::BLOCK_SIZE;
            let blocks_per_row = weight_shape[1] / BlockQ6K::BLOCK_SIZE;
            let raw_bytes = patterned_q6k_bytes(weight_shape);
            let matrix =
                QMatrix::from_parts(&device, &raw_bytes, weight_shape.into(), GgmlType::Q6K)
                    .unwrap();
            let selected_weights = (0..weight_shape[0])
                .map(|row| {
                    let block_index = row * blocks_per_row + selected_block_in_row;
                    let offset = block_index * size_of::<BlockQ6K>();
                    let block = unsafe {
                        std::ptr::read_unaligned(raw_bytes.as_ptr().add(offset).cast::<BlockQ6K>())
                    };
                    block.dequantize().as_ref()[selected_offset]
                })
                .collect::<Vec<_>>();
            let mut input_values = vec![0.0f32; weight_shape[1]];
            input_values[selected_k] = 0.25;
            let input = Tensor::from_slice::<f32>(&device, [1, weight_shape[1]], &input_values);

            let result = input.q_mat_mul(&matrix).as_slice::<2, f32>().await.unwrap();

            assert_eq!(result.shape(), &[1, weight_shape[0]]);
            for col in [0usize, 1, 63, 64, 511, 1024, 8191, 16384, 32767] {
                let actual = result[[0, col]];
                let expected = input_values[selected_k] * selected_weights[col];
                assert!(
                    (actual - expected).abs() <= 1e-3,
                    "col={col} actual={actual} expected={expected}"
                );
            }
        });
    }

    #[test]
    fn q6k_large_qgemv_matches_dense_sampled_columns() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            let weight_shape = [32768usize, 4096usize];
            let blocks_per_row = weight_shape[1] / BlockQ6K::BLOCK_SIZE;
            let raw_bytes = patterned_q6k_bytes(weight_shape);
            let matrix =
                QMatrix::from_parts(&device, &raw_bytes, weight_shape.into(), GgmlType::Q6K)
                    .unwrap();
            let input_values = (0..weight_shape[1])
                .map(|index| {
                    let bucket = (index.wrapping_mul(31).wrapping_add(7)) % 103;
                    (bucket as f32 - 51.0) * 0.002
                })
                .collect::<Vec<_>>();
            let input = Tensor::from_slice::<f32>(&device, [1, weight_shape[1]], &input_values);

            let result = input.q_mat_mul(&matrix).as_slice::<2, f32>().await.unwrap();

            assert_eq!(result.shape(), &[1, weight_shape[0]]);
            for col in [0usize, 1, 63, 64, 511, 1024, 8191, 16384, 32767] {
                let expected = (0..blocks_per_row)
                    .map(|block_col| {
                        let block_index = col * blocks_per_row + block_col;
                        let offset = block_index * size_of::<BlockQ6K>();
                        let block = unsafe {
                            std::ptr::read_unaligned(
                                raw_bytes.as_ptr().add(offset).cast::<BlockQ6K>(),
                            )
                        };
                        block
                            .dequantize()
                            .as_ref()
                            .iter()
                            .enumerate()
                            .map(|(block_offset, weight)| {
                                input_values[block_col * BlockQ6K::BLOCK_SIZE + block_offset]
                                    * *weight
                            })
                            .sum::<f32>()
                    })
                    .sum::<f32>();
                let actual = result[[0, col]];
                assert!(
                    (actual - expected).abs() <= 1e-2_f32.max(expected.abs() * 1.0e-4),
                    "col={col} actual={actual} expected={expected}"
                );
            }
        });
    }

    #[test]
    fn q6k_multirow_ffn_down_shape_matches_dense_sampled_columns() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            let weight_shape = [4096usize, 14336usize];
            let input_shape = [1usize, 48usize, weight_shape[1]];
            let blocks_per_row = weight_shape[1] / BlockQ6K::BLOCK_SIZE;
            let raw_bytes = patterned_q6k_bytes(weight_shape);
            let matrix =
                QMatrix::from_parts(&device, &raw_bytes, weight_shape.into(), GgmlType::Q6K)
                    .unwrap();
            let input_values = (0..input_shape.iter().product::<usize>())
                .map(|index| {
                    let bucket = (index.wrapping_mul(29).wrapping_add(5)) % 97;
                    (bucket as f32 - 48.0) * 0.0015
                })
                .collect::<Vec<_>>();
            let input = Tensor::from_slice::<f32>(&device, input_shape, &input_values);

            let result = input.q_mat_mul(&matrix).as_slice::<3, f32>().await.unwrap();

            assert_eq!(result.shape(), &[1, 48, weight_shape[0]]);
            for row in [0usize, 1, 7, 17, 31, 47] {
                let input_row = &input_values[row * weight_shape[1]..(row + 1) * weight_shape[1]];
                for col in [0usize, 1, 63, 64, 511, 1024, 2047, 4095] {
                    let expected = (0..blocks_per_row)
                        .map(|block_col| {
                            let block_index = col * blocks_per_row + block_col;
                            let offset = block_index * size_of::<BlockQ6K>();
                            let block = unsafe {
                                std::ptr::read_unaligned(
                                    raw_bytes.as_ptr().add(offset).cast::<BlockQ6K>(),
                                )
                            };
                            block
                                .dequantize()
                                .as_ref()
                                .iter()
                                .enumerate()
                                .map(|(block_offset, weight)| {
                                    input_row[block_col * BlockQ6K::BLOCK_SIZE + block_offset]
                                        * *weight
                                })
                                .sum::<f32>()
                        })
                        .sum::<f32>();
                    let actual = result[[0, row, col]];
                    assert!(
                        (actual - expected).abs() <= 1e-2_f32.max(expected.abs() * 1.0e-4),
                        "row={row} col={col} actual={actual} expected={expected}"
                    );
                }
            }
        });
    }

    #[test]
    fn qmatmul_accepts_dense_f32_qmatrix_without_generic_fallback() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            let weights = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let matrix = QMatrix::from_parts(
                &device,
                bytemuck::cast_slice(&weights),
                Box::new([2usize, 4usize]),
                GgmlType::F32,
            )
            .unwrap();
            let input_rows = vec![vec![1.0f32, 2.0, 3.0, 4.0]];
            let input = Tensor::new::<f32, 2, _>(&device, &input_rows);

            let result = input.q_mat_mul(&matrix).as_slice::<2, f32>().await.unwrap();
            assert_eq!(result.shape(), &[1, 2]);
            assert!((result[[0, 0]] - 30.0).abs() < 1e-4);
            assert!((result[[0, 1]] - 70.0).abs() < 1e-4);
        });
    }

    #[test]
    fn q5_0_qgemv_matches_expected_values() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            fn q5_0_block(scale: f32, high_bits: [u8; 4], low_bits: u8) -> Vec<u8> {
                let mut bytes = Vec::with_capacity(22);
                bytes.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
                bytes.extend_from_slice(&high_bits);
                bytes.extend(std::iter::repeat_n(low_bits, 16));
                bytes
            }

            let mut raw_bytes = Vec::new();
            raw_bytes.extend(q5_0_block(1.0, [0xff; 4], 0x11));
            raw_bytes.extend(q5_0_block(1.0, [0x00; 4], 0xff));
            let matrix =
                QMatrix::from_parts(&device, &raw_bytes, Box::new([2, 32]), GgmlType::Q5_0)
                    .unwrap();
            let input_rows = vec![(1..=32).map(|value| value as f32).collect::<Vec<_>>()];
            let input = Tensor::new::<f32, 2, _>(&device, &input_rows);

            let result = input.q_mat_mul(&matrix).as_slice::<2, f32>().await.unwrap();

            assert_eq!(result.shape(), &[1, 2]);
            assert!((result[[0, 0]] - 528.0).abs() < 1e-3);
            assert!((result[[0, 1]] + 528.0).abs() < 1e-3);
        });
    }

    #[test]
    fn q4_0_qgemv_matches_expected_values() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            fn q4_0_block(scale: f32, packed: u8) -> Vec<u8> {
                let mut bytes = Vec::with_capacity(18);
                bytes.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
                bytes.extend(std::iter::repeat_n(packed, 16));
                bytes
            }

            let mut raw_bytes = Vec::new();
            raw_bytes.extend(q4_0_block(1.0, 0x99));
            raw_bytes.extend(q4_0_block(1.0, 0x77));
            let matrix =
                QMatrix::from_parts(&device, &raw_bytes, Box::new([2, 32]), GgmlType::Q4_0)
                    .unwrap();
            let input_rows = vec![(1..=32).map(|value| value as f32).collect::<Vec<_>>()];
            let input = Tensor::new::<f32, 2, _>(&device, &input_rows);

            let result = input.q_mat_mul(&matrix).as_slice::<2, f32>().await.unwrap();

            assert_eq!(result.shape(), &[1, 2]);
            assert!((result[[0, 0]] - 528.0).abs() < 1e-3);
            assert!((result[[0, 1]] + 528.0).abs() < 1e-3);
        });
    }

    #[test]
    fn q8_0_qgemv_matches_expected_values() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            fn q8_0_block(scale: f32, value: i8) -> Vec<u8> {
                let mut bytes = Vec::with_capacity(34);
                bytes.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
                bytes.extend(std::iter::repeat_n(value as u8, 32));
                bytes
            }

            let mut raw_bytes = Vec::new();
            raw_bytes.extend(q8_0_block(1.0, 1));
            raw_bytes.extend(q8_0_block(1.0, -1));
            let matrix =
                QMatrix::from_parts(&device, &raw_bytes, Box::new([2, 32]), GgmlType::Q8_0)
                    .unwrap();
            let input_rows = vec![(1..=32).map(|value| value as f32).collect::<Vec<_>>()];
            let input = Tensor::new::<f32, 2, _>(&device, &input_rows);

            let result = input.q_mat_mul(&matrix).as_slice::<2, f32>().await.unwrap();

            assert_eq!(result.shape(), &[1, 2]);
            assert!((result[[0, 0]] - 528.0).abs() < 1e-3);
            assert!((result[[0, 1]] + 528.0).abs() < 1e-3);
        });
    }

    #[test]
    fn q5k_qgemv_matches_expected_values() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };

            let mut raw_bytes = Vec::with_capacity(size_of::<BlockQ5K>());
            push_f16(&mut raw_bytes, 1.0);
            push_f16(&mut raw_bytes, 0.0);
            raw_bytes.extend([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
            raw_bytes.extend(std::iter::repeat_n(0, BlockQ5K::QH_SIZE));
            raw_bytes.extend(std::iter::repeat_n(0x11, BlockQ5K::QS_SIZE));
            let matrix =
                QMatrix::from_parts(&device, &raw_bytes, Box::new([1, 256]), GgmlType::Q5K)
                    .unwrap();
            let input_rows = vec![(1..=256).map(|value| value as f32).collect::<Vec<_>>()];
            let input = Tensor::new::<f32, 2, _>(&device, &input_rows);

            let result = input.q_mat_mul(&matrix).as_slice::<2, f32>().await.unwrap();

            assert_eq!(result.shape(), &[1, 1]);
            assert!((result[[0, 0]] - 32896.0).abs() < 1e-2);
        });
    }

    #[test]
    fn f16_qmatmul_casts_through_f32_direct_path() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            if !device.f16_supported() {
                return;
            }

            let weight_shape = [4usize, BlockQ8_0::BLOCK_SIZE];
            let block_count = weight_shape.iter().product::<usize>() / BlockQ8_0::BLOCK_SIZE;
            let raw_bytes = vec![0; block_count * size_of::<BlockQ8_0>()];
            let matrix =
                QMatrix::from_parts(&device, &raw_bytes, weight_shape.into(), GgmlType::Q8_0)
                    .unwrap();
            let input_rows = vec![vec![half::f16::from_f32(0.25); weight_shape[1]]];
            let input = Tensor::new::<half::f16, 2, _>(&device, &input_rows);

            let result = input
                .q_mat_mul(&matrix)
                .as_slice::<2, half::f16>()
                .await
                .unwrap();

            assert_eq!(result.shape(), &[1, weight_shape[0]]);
            assert!(
                result
                    .as_slice()
                    .iter()
                    .take(weight_shape[0])
                    .all(|value| *value == half::f16::from_f32(0.0))
            );
        });
    }
}
