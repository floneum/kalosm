//! Quantized dequantization program kernels.

use fusor_tile_ir::tile::{Program, Storage};
use fusor_tile_ir::{
    ElementType, Layout, MemoryLevel, QuantizedMatrix, Shape, StorageView, WorkgroupAxis,
};

/// Lane-per-element dequantization.
///
/// Emits one dense f32/f16 element per quantized element of `b` and writes it
/// to a row-major `y` of `b.rows * b.cols` elements.
pub fn qdequantize(program: &mut Program, b: &QuantizedMatrix, y: &Storage, workgroups_x: u32) {
    const BLOCK: u32 = 256;
    assert!(
        workgroups_x > 0,
        "qdequantize workgroups_x must be non-zero"
    );
    assert_eq!(
        y.view().layout.element_count().get(),
        b.rows
            .checked_mul(b.cols)
            .expect("qdequantize output element count overflow"),
        "qdequantize output must contain one dense element per quantized element"
    );
    assert!(
        y.view().layout.is_row_major(),
        "qdequantize output must be row-major"
    );
    assert!(
        matches!(y.view().buffer.element, ElementType::F32 | ElementType::F16),
        "qdequantize output must be f32 or f16"
    );

    let total = b
        .rows
        .checked_mul(b.cols)
        .expect("qdequantize output element count overflow");
    let workgroups = total.div_ceil(BLOCK);
    let dispatch_y = workgroups.div_ceil(workgroups_x);
    let y = Storage::from_view(StorageView {
        buffer: y.view().buffer.clone(),
        offset: y.view().offset,
        layout: Layout::contiguous(MemoryLevel::Storage, Shape::new([1, total])),
    });
    program.program_grid(BLOCK, [workgroups_x, dispatch_y, 1], |program| {
        let lane = program.lane();
        let linear_group = program.program_id(WorkgroupAxis::X)
            + program.program_id(WorkgroupAxis::Y) * workgroups_x;
        let flat = linear_group * BLOCK + lane;
        let mask = flat.lt(total);
        let value = program.load_quantized(
            b,
            flat.clone() % b.rows,
            flat.clone() / b.rows,
            mask.clone(),
            0.0,
        );
        program.store_cast(y.at((0, flat)), value, mask);
    });
}
