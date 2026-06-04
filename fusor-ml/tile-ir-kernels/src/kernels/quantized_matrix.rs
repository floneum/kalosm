//! Quantized matrix program inputs.

use fusor_tile_ir::tile::{Program, Storage};
use fusor_tile_ir::{ElementType, GgmlQuantFormat, KernelBuilder, QuantizedMatrix, Shape};

/// Declare a quantized matrix on a [`KernelBuilder`] and remember its runtime
/// binding.
///
/// Equivalent to pushing `binding` and then calling [`quantized_matrix`] on
/// the underlying [`Program`].
pub fn quantized_matrix_for<B>(
    kb: &mut KernelBuilder<B>,
    binding: B,
    format: GgmlQuantFormat,
    rows: u32,
    cols: u32,
) -> QuantizedMatrix {
    kb.push_binding(binding);
    quantized_matrix(kb.program(), format, rows, cols)
}

/// Allocate a quantized matrix backing buffer and return its kernel handle.
///
/// ```
/// use fusor_tile_ir::{tile, GgmlQuantFormat};
/// use fusor_tile_ir_kernels::quantized_matrix;
///
/// let ir = tile::build(|program| {
///     let q = quantized_matrix(program, GgmlQuantFormat::Q4K, 256, 16);
///     assert_eq!(q.rows, 256);
///     assert_eq!(q.cols, 16);
/// });
/// # let _ = ir;
/// ```
pub fn quantized_matrix(
    program: &mut Program,
    format: GgmlQuantFormat,
    rows: u32,
    cols: u32,
) -> QuantizedMatrix {
    assert!(
        rows > 0 && cols > 0,
        "quantized matrix shape must be non-zero"
    );
    let total_elements = rows
        .checked_mul(cols)
        .expect("quantized matrix element count overflow");
    let blocks = total_elements.div_ceil(format.block_elements());
    let words = blocks
        .checked_mul(format.block_bytes())
        .map(|bytes| bytes.div_ceil(4))
        .expect("quantized matrix word count overflow");
    let data: Storage = program.storage_read(ElementType::U32, Shape::new([words]));
    QuantizedMatrix {
        data: data.view().clone(),
        format,
        rows,
        cols,
    }
}
