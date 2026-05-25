use super::*;

#[cfg(test)]
mod selection_tests {
    use super::*;
    use crate::kernel_selection::assert_selector_generates;

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

    fn ctx(format: tile_ir::GgmlQuantFormat, y_supports_coop: bool) -> QMatmulDirectCtx {
        QMatmulDirectCtx {
            format,
            y_supports_coop,
        }
    }

    #[test]
    fn qmatmul_direct_selector_generates_each_variant() {
        let selector = qmatmul_direct_selector();
        let q4 = tile_ir::GgmlQuantFormat::Q4_0;
        let cases = [
            (
                QMatmulPath::Q5SmallSingleRow,
                ctx(tile_ir::GgmlQuantFormat::Q5_0, false),
                caps(false),
            ),
            (QMatmulPath::SingleRow, ctx(q4, false), caps(false)),
            (
                QMatmulPath::Q8Wide(QCoopTile::new(64, 128)),
                ctx(tile_ir::GgmlQuantFormat::Q8_0, false),
                caps(true),
            ),
            (QMatmulPath::Tile { tile: QCoopTile::new(128, 128), cached: false }, ctx(q4, true), caps(true)),
            (QMatmulPath::Tile { tile: QCoopTile::new(128, 64), cached: false }, ctx(q4, true), caps(false)),
            (QMatmulPath::Tile { tile: QCoopTile::new(64, 128), cached: false }, ctx(q4, true), caps(false)),
            (
                QMatmulPath::Tile { tile: QCoopTile::new(64, 64), cached: true },
                ctx(q4, true),
                caps(false),
            ),
            (QMatmulPath::Tile { tile: QCoopTile::new(64, 64), cached: false }, ctx(q4, false), caps(false)),
        ];
        assert_selector_generates(&selector, cases);
    }

    #[test]
    fn coop_acc_init_only_claims_shapes_the_coop_path_will_take() {
        assert!(qmatmul_variant_supports_coop_acc_init(
            QMatmulPath::Tile { tile: QCoopTile::new(64, 128), cached: false },
            64,
            512,
            128,
            true,
        ));
        assert!(!qmatmul_variant_supports_coop_acc_init(
            QMatmulPath::Tile { tile: QCoopTile::new(64, 128), cached: false },
            63,
            512,
            128,
            true,
        ));
        assert!(!qmatmul_variant_supports_coop_acc_init(
            QMatmulPath::Tile { tile: QCoopTile::new(64, 64), cached: false },
            2,
            512,
            4,
            true,
        ));
        assert!(!qmatmul_variant_supports_coop_acc_init(
            QMatmulPath::Tile { tile: QCoopTile::new(64, 128), cached: false },
            64,
            510,
            128,
            true,
        ));
        assert!(!qmatmul_variant_supports_coop_acc_init(
            QMatmulPath::Tile { tile: QCoopTile::new(64, 128), cached: false },
            64,
            512,
            128,
            false,
        ));
    }

}

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use std::{mem::size_of, sync::Arc};

    use fusor_gguf::{BlockQ4_0, BlockQ4K, BlockQ6K, BlockQ8_0, GgufBlock};

    use super::*;
    use crate::{
        compute_graph::ComputeGraphInner, mir::kernel_backend::DirectKernelBinding,
        mir::workgroup_shape::WorkgroupShape,
    };

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

            let compact_len = block_count * size_of::<<BlockQ4_0 as GgufBlock>::BytesF32>();
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
                paired: None,
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
                paired: None,
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
