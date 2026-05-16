//! Quantized dequantization program kernels.

use fusor_tile_ir::tile::{Program, Storage};
use fusor_tile_ir::{F32, Layout, MemoryLevel, QuantizedMatrix, Shape, StorageView, WorkgroupAxis};

/// Lane-per-element dequantization.
///
/// Emits one f32 per quantized element of `b` and writes them to a row-major
/// `y` of `b.rows * b.cols` floats.
pub fn qdequantize(
    program: &mut Program,
    b: &QuantizedMatrix,
    y: &Storage<F32, 1>,
    workgroups_x: u32,
) {
    const BLOCK: usize = 256;
    assert!(
        workgroups_x > 0,
        "qdequantize workgroups_x must be non-zero"
    );
    assert_eq!(
        y.view().layout.element_count().get(),
        b.rows
            .checked_mul(b.cols)
            .expect("qdequantize output element count overflow"),
        "qdequantize output must contain one dense f32 per quantized element"
    );
    assert!(
        y.view().layout.is_row_major(),
        "qdequantize output must be row-major"
    );

    let total = b
        .rows
        .checked_mul(b.cols)
        .expect("qdequantize output element count overflow");
    let workgroups = total.div_ceil(BLOCK as u32);
    let dispatch_y = workgroups.div_ceil(workgroups_x);
    let y = Storage::<F32, 2>::from_view(StorageView {
        buffer: y.view().buffer,
        offset: y.view().offset,
        layout: Layout::contiguous(MemoryLevel::Storage, Shape::new([1, total])),
    });
    program.program_grid::<BLOCK>([workgroups_x, dispatch_y, 1], |program| {
        let lane = program.lane();
        let linear_group = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * workgroups_x;
        let flat = linear_group * BLOCK as u32 + lane;
        let mask = flat.lt(total);
        let value = program.load_quantized(
            b,
            flat.clone() % b.rows,
            flat.clone() / b.rows,
            mask.clone(),
            0.0,
        );
        program.store(y.at((0, flat)), value, mask);
    });
}
