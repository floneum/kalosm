//! Conformance: panic-on-misuse regressions.
//!
//! These tests pin the panic behavior of operations that fail loudly when
//! given malformed inputs. They were originally `cpu/tests/matmul.rs::test_matmul_shape_mismatch`,
//! `cpu/src/slice_assign.rs::{test_slice_assign_out_of_bounds, test_slice_assign_shape_mismatch}`,
//! and `cpu/src/quantized.rs::{test_invalid_shape_not_multiple_of_block_size, test_invalid_block_count}`.

use fusor::{Device, GgmlType, QMatrix, Tensor};

#[test]
#[should_panic]
fn matmul_shape_mismatch_panics() {
    let lhs: Tensor<2, f32> = Tensor::from_slice(&Device::Cpu, [2, 3], &[1.0; 6]);
    let rhs: Tensor<2, f32> = Tensor::from_slice(&Device::Cpu, [2, 2], &[1.0; 4]);
    let _ = lhs.matmul(&rhs);
}

#[test]
#[should_panic]
fn slice_assign_out_of_bounds_panics() {
    let target: Tensor<2, f32> = Tensor::from_slice(&Device::Cpu, [3, 3], &[0.0; 9]);
    let value: Tensor<2, f32> = Tensor::from_slice(&Device::Cpu, [2, 2], &[1.0; 4]);
    // The slice `[2..4, 0..2]` exceeds the row count (3) of the target.
    let _ = target.slice_assign([2..4, 0..2], &value).to_concrete();
}

#[test]
#[should_panic]
fn slice_assign_shape_mismatch_panics() {
    let target: Tensor<2, f32> = Tensor::from_slice(&Device::Cpu, [3, 3], &[0.0; 9]);
    // Value is 3x2 but the slice region is 2x2 — shape mismatch.
    let value: Tensor<2, f32> = Tensor::from_slice(&Device::Cpu, [3, 2], &[1.0; 6]);
    let _ = target.slice_assign([0..2, 0..2], &value).to_concrete();
}

#[test]
#[should_panic(expected = "Innermost dimension")]
fn quantized_invalid_inner_dim_panics() {
    // Q4_0 block size is 32 elements / 18 bytes; an inner dim of 33 is not a
    // multiple. We supply 2 valid 18-byte blocks (matches the
    // total_elements / block_size count for [2, 33]) so the bytemuck cast
    // succeeds and the inner-dim assertion is the first to fire.
    let bytes = vec![0u8; 36];
    let _ = QMatrix::from_raw_bytes(&Device::Cpu, [2usize, 33], &bytes, GgmlType::Q4_0);
}

#[test]
#[should_panic(expected = "assertion `left == right` failed")]
fn quantized_invalid_block_count_panics() {
    // Q4_0 block size is 32 elements -> shape [2, 64] needs 4 blocks.
    // Each Q4_0 block is 18 bytes, so 36 bytes only supplies 2 blocks.
    // The block-count assertion in `from_blocks` should fire.
    let bytes = vec![0u8; 36];
    let _ = QMatrix::from_raw_bytes(&Device::Cpu, [2usize, 64], &bytes, GgmlType::Q4_0);
}
