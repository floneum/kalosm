//! Support for GGUF quantized tensors
//!
//! This module provides `QuantizedTensor` for storing and operating on
//! quantized data from GGUF files. It supports:
//! - Multiple quantization types (Q4_0, Q5_0, Q8_0, Q4K, Q6K)
//! - Eager full dequantization to f32
//! - Lazy dequantization via the `Dequantize` expression type
//! - Efficient block-by-block matrix multiplication

use aligned_vec::{ABox, AVec};
use bytemuck::Pod;
use fusor_gguf::GgufBlock;
use pulp::Simd;

use fusor_types::Layout;

use crate::expr::materialize_expr;
use crate::reduce::{SimdReduceOp, SumOp};
use crate::{ConcreteTensor, MAX_SIMD_LANES, ResolvedTensor, SimdElement, TensorBacking};

/// A tensor storing quantized blocks.
///
/// `QuantizedTensor<B>` stores data in quantized block format where `B` is
/// the block type (e.g., `BlockQ4_0`). The rank is dynamic at runtime.
///
/// The innermost dimension must be a multiple of the block size. For example,
/// a [3, 256] tensor with Q4_0 quantization (block size 32) stores 3 rows of
/// 8 blocks each.
#[derive(Clone)]
pub struct QuantizedTensor<B: GgufBlock> {
    /// The logical shape in elements (not blocks)
    element_shape: Box<[usize]>,
    /// The quantized blocks stored in row-major order
    blocks: ABox<[B]>,
}

impl<B: GgufBlock> QuantizedTensor<B> {
    /// Create a quantized tensor from pre-existing blocks.
    ///
    /// # Arguments
    /// * `element_shape` - The logical shape in elements (not blocks).
    ///   The innermost dimension must be a multiple of `B::BLOCK_SIZE`.
    /// * `blocks` - The quantized blocks in row-major order.
    ///
    /// # Panics
    /// Panics if:
    /// - The innermost dimension is not a multiple of the block size
    /// - The number of blocks doesn't match the shape
    pub fn from_blocks(element_shape: impl Into<Box<[usize]>>, blocks: ABox<[B]>) -> Self {
        let element_shape = element_shape.into();
        let rank = element_shape.len();
        assert!(rank > 0, "Tensor must have at least rank 1");
        let inner_dim = element_shape[rank - 1];
        assert!(
            inner_dim % B::BLOCK_SIZE == 0,
            "Innermost dimension ({}) must be a multiple of block size ({})",
            inner_dim,
            B::BLOCK_SIZE
        );

        let expected_blocks = Self::compute_block_count(&element_shape);
        assert_eq!(
            blocks.len(),
            expected_blocks,
            "Expected {} blocks for shape {:?}, got {}",
            expected_blocks,
            element_shape,
            blocks.len()
        );

        Self {
            element_shape,
            blocks,
        }
    }

    /// Create a quantized tensor from raw bytes.
    ///
    /// This interprets the bytes as a slice of blocks using bytemuck.
    ///
    /// # Arguments
    /// * `element_shape` - The logical shape in elements (not blocks).
    /// * `bytes` - Raw bytes that will be cast to blocks.
    ///
    /// # Panics
    /// Panics if:
    /// - The bytes length is not a multiple of the block size
    /// - The innermost dimension is not a multiple of the block size
    /// - The number of blocks doesn't match the shape
    pub fn from_raw_bytes(element_shape: impl Into<Box<[usize]>>, bytes: &[u8]) -> Self {
        let blocks_slice: &[B] = pulp::bytemuck::cast_slice(bytes);
        let mut vec: AVec<B> = AVec::with_capacity(64, blocks_slice.len());
        vec.extend_from_slice(blocks_slice);
        Self::from_blocks(element_shape, vec.into_boxed_slice())
    }

    /// Compute the number of blocks needed for a given element shape.
    fn compute_block_count(element_shape: &[usize]) -> usize {
        let total_elements: usize = element_shape.iter().product();
        total_elements / B::BLOCK_SIZE
    }

    /// Returns the logical element shape (not block shape).
    pub fn element_shape(&self) -> &[usize] {
        &self.element_shape
    }

    /// Returns the total number of logical elements.
    pub fn element_count(&self) -> usize {
        self.element_shape.iter().product()
    }

    /// Returns the number of blocks.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Returns a reference to the underlying blocks.
    pub fn blocks(&self) -> &[B] {
        &self.blocks
    }

    /// Eagerly dequantize the entire tensor to f32.
    ///
    /// This allocates a new `ConcreteTensor<f32, R>` and dequantizes all blocks.
    /// For large tensors, consider using `dequantize_lazy()` instead.
    ///
    /// # Panics
    /// Panics if the tensor's rank doesn't match R.
    pub fn dequantize<const R: usize>(&self) -> ConcreteTensor<f32, R> {
        let shape: [usize; R] = self
            .element_shape
            .as_ref()
            .try_into()
            .expect("Shape length mismatch in dequantize");
        let layout = fusor_types::Layout::contiguous(&shape);
        let n = layout.num_elements();
        let mut vec: AVec<f32> = AVec::with_capacity(64, n);

        for block in self.blocks.iter() {
            let dequantized = block.dequantize();
            vec.extend_from_slice(dequantized.as_ref());
        }

        ConcreteTensor::from_parts(layout, vec.into_boxed_slice())
    }

    /// Create a lazy dequantization expression.
    ///
    /// This returns a `Dequantize` expression that implements `Expr`,
    /// allowing it to be composed with other operations before materialization.
    ///
    /// # Panics
    /// Panics if the tensor's rank doesn't match R.
    pub fn dequantize_lazy<const R: usize>(&self) -> Dequantize<'_, B, R> {
        assert_eq!(
            self.element_shape.len(),
            R,
            "Tensor rank {} doesn't match expected rank {}",
            self.element_shape.len(),
            R
        );
        Dequantize { source: self }
    }
}

/// Lazy dequantization expression.
///
/// This implements `Expr` for lazy evaluation of dequantized values.
/// Instead of dequantizing the entire tensor upfront, values are
/// dequantized on-demand during expression evaluation.
pub struct Dequantize<'a, B: GgufBlock, const R: usize> {
    source: &'a QuantizedTensor<B>,
}

impl<B: GgufBlock, const R: usize> crate::LazyBacking for Dequantize<'_, B, R>
where
    B::Dequantized: AsRef<[f32]>,
{
    type Elem = f32;

    #[inline(always)]
    fn eval_scalar(&self, idx: usize) -> f32 {
        let block_idx = idx / B::BLOCK_SIZE;
        let elem_idx = idx % B::BLOCK_SIZE;
        self.source.blocks[block_idx].dequantize().as_ref()[elem_idx]
    }

    #[inline(always)]
    fn eval_simd<S: Simd>(&self, _simd: S, base_idx: usize) -> <f32 as SimdElement>::Simd<S> {
        // Block boundaries don't typically align with SIMD lanes,
        // so we fall back to scalar gathering
        let lane_count =
            std::mem::size_of::<<f32 as SimdElement>::Simd<S>>() / std::mem::size_of::<f32>();
        let mut temp = [0.0f32; MAX_SIMD_LANES];
        for (i, temp_elem) in temp.iter_mut().enumerate().take(lane_count) {
            *temp_elem = self.eval_scalar(base_idx + i);
        }
        let (simd_vec, _) = f32::as_simd::<S>(&temp[..lane_count]);
        simd_vec[0]
    }
}

impl<B: GgufBlock, const R: usize> TensorBacking<R> for Dequantize<'_, B, R>
where
    B::Dequantized: AsRef<[f32]>,
{
    fn layout(&self) -> Layout {
        // The layout of the dequantized tensor matches the source tensor's element shape
        let shape: [usize; R] = self
            .source
            .element_shape
            .as_ref()
            .try_into()
            .expect("Shape length mismatch in Dequantize::layout");
        Layout::contiguous(&shape)
    }

    fn to_concrete(&self) -> ConcreteTensor<f32, R> {
        let shape: [usize; R] = self
            .source
            .element_shape
            .as_ref()
            .try_into()
            .expect("Shape length mismatch in Dequantize::to_concrete");
        materialize_expr(self, shape)
    }
}

/// Matrix multiplication with a quantized RHS.
///
/// Computes `self @ rhs` where `self` is an f32 tensor and `rhs` is quantized.
/// This processes blocks one at a time to avoid the memory cost of full dequantization.
/// Supports batched inputs: `[batch_dims..., M, K] @ [K, N] -> [batch_dims..., M, N]`
impl<const R: usize> ConcreteTensor<f32, R> {
    /// Matrix multiplication: self ([batch_dims..., M, K]) @ rhs (K x N) -> ([batch_dims..., M, N])
    ///
    /// This is optimized for the case where the RHS (weights) are quantized.
    /// Instead of dequantizing the entire RHS matrix, it processes block-by-block
    /// with SIMD acceleration.
    ///
    /// The RHS must be 2D (K x N), while the LHS can have arbitrary batch dimensions.
    ///
    /// # Panics
    /// Panics if rhs is not 2D.
    pub fn q_mat_mul<B: GgufBlock + Sync>(&self, rhs: &QuantizedTensor<B>) -> ConcreteTensor<f32, R>
    where
        B::Dequantized: AsRef<[f32]>,
        B::ActivationBlock: Pod + Send + Sync,
    {
        const { assert!(R >= 2, "q_mat_mul requires at least 2 dimensions") };

        let rhs_shape = rhs.element_shape();
        assert_eq!(
            rhs_shape.len(),
            2,
            "q_mat_mul requires 2D weight tensor, got {}D",
            rhs_shape.len()
        );

        let lhs_shape = self.layout().shape();
        let m = lhs_shape[R - 2];
        let k = lhs_shape[R - 1];
        // Weight is stored as [out_features, in_features] to match GPU convention
        let n = rhs_shape[0]; // out_features
        let k2 = rhs_shape[1]; // in_features

        assert_eq!(
            k, k2,
            "Matrix dimension mismatch: lhs columns ({}) != weight in_features ({})",
            k, k2
        );

        // Output shape: preserve batch dims, replace last two with [M, N]
        let mut out_shape: [usize; R] = [0; R];
        out_shape.copy_from_slice(lhs_shape);
        out_shape[R - 1] = n;

        let mut output = ConcreteTensor::<f32, R>::zeros(out_shape);

        // Compute batch size (product of all dims except last 2)
        let batch_size: usize = if R > 2 {
            lhs_shape[..R - 2].iter().product()
        } else {
            1
        };

        let lhs_matrix_size = m * k;
        let out_matrix_size = m * n;

        // Weight is [N, K], so blocks per row of weight = K / BLOCK_SIZE
        let blocks_per_weight_row = k / B::BLOCK_SIZE;

        let lhs_contiguous = self.layout().is_contiguous();

        if lhs_contiguous {
            // Fast path: LHS is contiguous
            let lhs_data = self.data();
            let out_data = output.data_mut();

            for b in 0..batch_size {
                let lhs_slice = &lhs_data[b * lhs_matrix_size..(b + 1) * lhs_matrix_size];
                let out_slice = &mut out_data[b * out_matrix_size..(b + 1) * out_matrix_size];

                pulp::Arch::new().dispatch(QMatmulSimd {
                    lhs_data: lhs_slice,
                    rhs_blocks: rhs.blocks(),
                    out_data: out_slice,
                    m,
                    k,
                    n,
                    blocks_per_weight_row,
                    _phantom: std::marker::PhantomData::<B>,
                });
            }
        } else {
            // Slow path: LHS is not contiguous, need to extract each batch to contiguous memory
            let batch_dims = &lhs_shape[..R - 2];
            let mut batch_indices = vec![0usize; R - 2];

            for b in 0..batch_size {
                // Extract this batch's matrix to contiguous memory
                let mut lhs_batch = vec![0.0f32; lhs_matrix_size];
                for i in 0..m {
                    for l in 0..k {
                        let mut lhs_idx_arr = [0usize; R];
                        for (idx, &bi) in batch_indices.iter().enumerate() {
                            lhs_idx_arr[idx] = bi;
                        }
                        lhs_idx_arr[R - 2] = i;
                        lhs_idx_arr[R - 1] = l;
                        let lhs_idx = self.layout().linear_index(&lhs_idx_arr);
                        lhs_batch[i * k + l] = self.data()[lhs_idx];
                    }
                }

                let out_slice =
                    &mut output.data_mut()[b * out_matrix_size..(b + 1) * out_matrix_size];

                pulp::Arch::new().dispatch(QMatmulSimd {
                    lhs_data: &lhs_batch,
                    rhs_blocks: rhs.blocks(),
                    out_data: out_slice,
                    m,
                    k,
                    n,
                    blocks_per_weight_row,
                    _phantom: std::marker::PhantomData::<B>,
                });

                // Increment batch indices (like a multi-digit counter)
                for d in (0..batch_indices.len()).rev() {
                    batch_indices[d] += 1;
                    if batch_indices[d] < batch_dims[d] {
                        break;
                    }
                    batch_indices[d] = 0;
                }
            }
        }

        output
    }
}

/// SIMD-accelerated quantized matmul kernel
struct QMatmulSimd<'a, B: GgufBlock> {
    lhs_data: &'a [f32],
    rhs_blocks: &'a [B],
    out_data: &'a mut [f32],
    m: usize,
    k: usize,
    n: usize,
    /// Number of blocks per row of the weight matrix [N, K]
    blocks_per_weight_row: usize,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: GgufBlock + Sync> pulp::WithSimd for QMatmulSimd<'_, B>
where
    B::Dequantized: AsRef<[f32]>,
    B::ActivationBlock: Pod + Send + Sync,
{
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self {
            lhs_data,
            rhs_blocks,
            out_data,
            m,
            k,
            n,
            blocks_per_weight_row,
            ..
        } = self;

        // Use f32 dequantize path: dequantize weight blocks to f32 and use SIMD mul_add.
        // This avoids quantizing activations (which introduces compounding error across layers).
        // Special fast path for m=1 (common inference case): parallelize over output columns
        if m == 1 {
            let n_threads = crate::parallel::num_threads();

            // For small n or single-threaded, don't parallelize
            if n < 64 || n_threads == 1 {
                process_row_simd_tiled::<B, S>(
                    simd,
                    lhs_data,
                    rhs_blocks,
                    out_data,
                    n,
                    blocks_per_weight_row,
                );
            } else {
                // Parallelize over output column chunks using scoped threads
                const CHUNK_SIZE: usize = 32;
                let total_chunks = n.div_ceil(CHUNK_SIZE);
                let chunks_per_thread = total_chunks.div_ceil(n_threads);
                let elements_per_thread = chunks_per_thread * CHUNK_SIZE;

                std::thread::scope(|scope| {
                    let mut remaining = out_data;
                    let mut start_n = 0;

                    for thread_id in 0..n_threads {
                        if remaining.is_empty() {
                            break;
                        }

                        let this_size = if thread_id == n_threads - 1 {
                            remaining.len()
                        } else {
                            elements_per_thread.min(remaining.len())
                        };

                        let (thread_chunk, rest) = remaining.split_at_mut(this_size);
                        remaining = rest;
                        let thread_start_n = start_n;
                        start_n += this_size;

                        scope.spawn(move || {
                            // Process each CHUNK_SIZE piece within this thread
                            for (i, out_chunk) in thread_chunk.chunks_mut(CHUNK_SIZE).enumerate() {
                                let chunk_start = thread_start_n + i * CHUNK_SIZE;
                                process_row_simd_range::<B, S>(
                                    simd,
                                    lhs_data,
                                    rhs_blocks,
                                    out_chunk,
                                    chunk_start,
                                    out_chunk.len(),
                                    blocks_per_weight_row,
                                );
                            }
                        });
                    }
                });
            }
        } else if m >= 4 {
            let n_threads = crate::parallel::num_threads();

            if n_threads == 1 {
                // Sequential processing
                for i in 0..m {
                    let lhs_row = &lhs_data[i * k..(i + 1) * k];
                    let out_row = &mut out_data[i * n..(i + 1) * n];
                    process_row_simd_tiled::<B, S>(
                        simd,
                        lhs_row,
                        rhs_blocks,
                        out_row,
                        n,
                        blocks_per_weight_row,
                    );
                }
            } else {
                // Process rows in parallel using scoped threads
                let rows_per_thread = m.div_ceil(n_threads);

                std::thread::scope(|scope| {
                    let mut remaining_out = out_data;
                    let mut row_offset = 0;

                    for thread_id in 0..n_threads {
                        if remaining_out.is_empty() {
                            break;
                        }

                        let this_rows = if thread_id == n_threads - 1 {
                            m - row_offset
                        } else {
                            rows_per_thread.min(m - row_offset)
                        };

                        let this_size = this_rows * n;
                        let (thread_out, rest) = remaining_out.split_at_mut(this_size);
                        remaining_out = rest;
                        let thread_row_offset = row_offset;
                        row_offset += this_rows;

                        scope.spawn(move || {
                            for i in 0..this_rows {
                                let global_row = thread_row_offset + i;
                                let lhs_row = &lhs_data[global_row * k..(global_row + 1) * k];
                                let out_row = &mut thread_out[i * n..(i + 1) * n];
                                process_row_simd_tiled::<B, S>(
                                    simd,
                                    lhs_row,
                                    rhs_blocks,
                                    out_row,
                                    n,
                                    blocks_per_weight_row,
                                );
                            }
                        });
                    }
                });
            }
        } else {
            // Sequential processing for small matrices (m=2,3)
            for i in 0..m {
                let lhs_row = &lhs_data[i * k..(i + 1) * k];
                let out_row = &mut out_data[i * n..(i + 1) * n];
                process_row_simd_tiled::<B, S>(
                    simd,
                    lhs_row,
                    rhs_blocks,
                    out_row,
                    n,
                    blocks_per_weight_row,
                );
            }
        }
    }
}

/// Process a range of output columns for m=1 parallelization using integer dot products.
/// Uses NEON intrinsics on aarch64 for efficient i8 x i8 -> i32 computation.
#[allow(dead_code)]
#[inline(always)]
fn process_row_integer_range<B: GgufBlock>(
    lhs_row: &[f32],
    rhs_blocks: &[B],
    out_chunk: &mut [f32],
    start_n: usize,
    chunk_n: usize,
    blocks_per_weight_row: usize,
) where
    B::ActivationBlock: Pod,
{
    // Quantize activations once for all output columns
    let act_blocks: Vec<B::ActivationBlock> = (0..blocks_per_weight_row)
        .map(|block_idx| {
            let start = block_idx * B::BLOCK_SIZE;
            let chunk = &lhs_row[start..start + B::BLOCK_SIZE];
            B::quantize_activation(chunk)
        })
        .collect();

    for (i, out_elem) in out_chunk.iter_mut().enumerate().take(chunk_n) {
        let n_out = start_n + i;
        let mut sum = 0.0f32;
        for (block_idx, act_block) in act_blocks.iter().enumerate() {
            let weight_block_idx = n_out * blocks_per_weight_row + block_idx;
            sum += rhs_blocks[weight_block_idx].vec_dot(act_block);
        }
        *out_elem = sum;
    }
}

/// Process a range of output columns for m=1 parallelization
#[inline(always)]
fn process_row_simd_range<B: GgufBlock, S: Simd>(
    simd: S,
    lhs_row: &[f32],
    rhs_blocks: &[B],
    out_chunk: &mut [f32],
    start_n: usize,
    chunk_n: usize,
    blocks_per_weight_row: usize,
) where
    B::Dequantized: AsRef<[f32]>,
{
    for (i, out_elem) in out_chunk.iter_mut().enumerate().take(chunk_n) {
        let n_out = start_n + i;
        *out_elem =
            compute_dot_product::<B, S>(simd, lhs_row, rhs_blocks, n_out, blocks_per_weight_row);
    }
}

/// Process a single output row using integer dot products with 4-way tiling.
/// Uses NEON intrinsics on aarch64 for efficient i8 x i8 -> i32 computation.
#[allow(dead_code)]
#[inline(always)]
fn process_row_integer_tiled<B: GgufBlock>(
    lhs_row: &[f32],
    rhs_blocks: &[B],
    out_row: &mut [f32],
    n: usize,
    blocks_per_weight_row: usize,
) where
    B::ActivationBlock: Pod,
{
    // Step 1: Quantize activations to Q8 blocks (once per row)
    let mut act_blocks: Vec<B::ActivationBlock> = Vec::with_capacity(blocks_per_weight_row);
    for block_idx in 0..blocks_per_weight_row {
        let start = block_idx * B::BLOCK_SIZE;
        let chunk = &lhs_row[start..start + B::BLOCK_SIZE];
        act_blocks.push(B::quantize_activation(chunk));
    }

    // Step 2: 4-way tiled output loop using integer dot products
    const TILE: usize = 4;
    let n_tiles = n / TILE;

    for tile in 0..n_tiles {
        let base = tile * TILE;
        let mut acc = [0.0f32; TILE];

        for block_idx in 0..blocks_per_weight_row {
            let act = &act_blocks[block_idx];

            // Compute 4 dot products
            acc[0] += rhs_blocks[base * blocks_per_weight_row + block_idx].vec_dot(act);
            acc[1] += rhs_blocks[(base + 1) * blocks_per_weight_row + block_idx].vec_dot(act);
            acc[2] += rhs_blocks[(base + 2) * blocks_per_weight_row + block_idx].vec_dot(act);
            acc[3] += rhs_blocks[(base + 3) * blocks_per_weight_row + block_idx].vec_dot(act);
        }

        out_row[base..base + TILE].copy_from_slice(&acc);
    }

    // Handle remainder
    for j in (n_tiles * TILE)..n {
        let mut sum = 0.0f32;
        for block_idx in 0..blocks_per_weight_row {
            sum +=
                rhs_blocks[j * blocks_per_weight_row + block_idx].vec_dot(&act_blocks[block_idx]);
        }
        out_row[j] = sum;
    }
}

/// Process a single output row with SIMD using 4-way tiling for better ILP
#[inline(always)]
fn process_row_simd_tiled<B: GgufBlock, S: Simd>(
    simd: S,
    lhs_row: &[f32],
    rhs_blocks: &[B],
    out_row: &mut [f32],
    n: usize,
    blocks_per_weight_row: usize,
) where
    B::Dequantized: AsRef<[f32]>,
{
    // Process 4 output columns at a time for better instruction-level parallelism
    const TILE: usize = 4;
    let n_tiles = n / TILE;
    let n_remainder = n % TILE;

    for tile in 0..n_tiles {
        let base = tile * TILE;

        // Initialize 4 accumulators
        let mut acc0 = simd.splat_f32s(0.0);
        let mut acc1 = simd.splat_f32s(0.0);
        let mut acc2 = simd.splat_f32s(0.0);
        let mut acc3 = simd.splat_f32s(0.0);
        let mut scalar_acc = [0.0f32; TILE];

        // Process all blocks, accumulating into all 4 outputs
        for block_idx in 0..blocks_per_weight_row {
            let input_block_start = block_idx * B::BLOCK_SIZE;
            let input_block = &lhs_row[input_block_start..input_block_start + B::BLOCK_SIZE];
            let (inp_simd, inp_tail) = S::as_simd_f32s(input_block);

            // Dequantize and accumulate for each of the 4 output columns
            let deq0 = rhs_blocks[base * blocks_per_weight_row + block_idx].dequantize();
            let deq1 = rhs_blocks[(base + 1) * blocks_per_weight_row + block_idx].dequantize();
            let deq2 = rhs_blocks[(base + 2) * blocks_per_weight_row + block_idx].dequantize();
            let deq3 = rhs_blocks[(base + 3) * blocks_per_weight_row + block_idx].dequantize();

            let (deq0_simd, deq0_tail) = S::as_simd_f32s(deq0.as_ref());
            let (deq1_simd, deq1_tail) = S::as_simd_f32s(deq1.as_ref());
            let (deq2_simd, deq2_tail) = S::as_simd_f32s(deq2.as_ref());
            let (deq3_simd, deq3_tail) = S::as_simd_f32s(deq3.as_ref());

            // SIMD accumulation for all 4 outputs
            for (i, &inp_vec) in inp_simd.iter().enumerate() {
                acc0 = simd.mul_add_f32s(inp_vec, deq0_simd[i], acc0);
                acc1 = simd.mul_add_f32s(inp_vec, deq1_simd[i], acc1);
                acc2 = simd.mul_add_f32s(inp_vec, deq2_simd[i], acc2);
                acc3 = simd.mul_add_f32s(inp_vec, deq3_simd[i], acc3);
            }

            // Scalar tail
            for (i, &inp_val) in inp_tail.iter().enumerate() {
                scalar_acc[0] += inp_val * deq0_tail[i];
                scalar_acc[1] += inp_val * deq1_tail[i];
                scalar_acc[2] += inp_val * deq2_tail[i];
                scalar_acc[3] += inp_val * deq3_tail[i];
            }
        }

        // Reduce and store results
        out_row[base] = <SumOp as SimdReduceOp<f32>>::reduce_simd_vec(simd, acc0) + scalar_acc[0];
        out_row[base + 1] =
            <SumOp as SimdReduceOp<f32>>::reduce_simd_vec(simd, acc1) + scalar_acc[1];
        out_row[base + 2] =
            <SumOp as SimdReduceOp<f32>>::reduce_simd_vec(simd, acc2) + scalar_acc[2];
        out_row[base + 3] =
            <SumOp as SimdReduceOp<f32>>::reduce_simd_vec(simd, acc3) + scalar_acc[3];
    }

    // Handle remainder
    for i in 0..n_remainder {
        let n_out = n_tiles * TILE + i;
        out_row[n_out] =
            compute_dot_product::<B, S>(simd, lhs_row, rhs_blocks, n_out, blocks_per_weight_row);
    }
}

/// Compute a single dot product for one output column
#[inline(always)]
fn compute_dot_product<B: GgufBlock, S: Simd>(
    simd: S,
    lhs_row: &[f32],
    rhs_blocks: &[B],
    n_out: usize,
    blocks_per_weight_row: usize,
) -> f32
where
    B::Dequantized: AsRef<[f32]>,
{
    let mut acc = simd.splat_f32s(0.0);
    let mut scalar_acc = 0.0f32;

    for block_idx in 0..blocks_per_weight_row {
        let weight_block_idx = n_out * blocks_per_weight_row + block_idx;
        let input_block_start = block_idx * B::BLOCK_SIZE;

        let dequantized = rhs_blocks[weight_block_idx].dequantize();
        let dequantized_slice = dequantized.as_ref();
        let input_block = &lhs_row[input_block_start..input_block_start + B::BLOCK_SIZE];

        let (inp_simd, inp_tail) = S::as_simd_f32s(input_block);
        let (deq_simd, deq_tail) = S::as_simd_f32s(dequantized_slice);

        for (&inp_vec, &deq_vec) in inp_simd.iter().zip(deq_simd.iter()) {
            acc = simd.mul_add_f32s(inp_vec, deq_vec, acc);
        }

        for (&inp_val, &deq_val) in inp_tail.iter().zip(deq_tail.iter()) {
            scalar_acc += inp_val * deq_val;
        }
    }

    <SumOp as SimdReduceOp<f32>>::reduce_simd_vec(simd, acc) + scalar_acc
}
