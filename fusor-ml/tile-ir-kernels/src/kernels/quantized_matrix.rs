//! Quantized matrix program inputs.

use fusor_tile_ir::tile::{Program, Storage};
use fusor_tile_ir::{GgmlQuantFormat, KernelBuilder, QuantizedMatrix, Shape, U32};

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
    assert_eq!(
        rows % format.block_elements(),
        0,
        "quantized rows/K dimension must be a multiple of the format block size"
    );
    let blocks_per_col = rows / format.block_elements();
    let words = blocks_per_col
        .checked_mul(cols)
        .and_then(|blocks| blocks.checked_mul(format.block_words()))
        .expect("quantized matrix word count overflow");
    let data: Storage<U32, 1> = program.storage_read(Shape::new([words]));
    QuantizedMatrix {
        data: data.view().clone(),
        format,
        rows,
        cols,
    }
}
