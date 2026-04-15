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
    fn with_simd<S: Simd>(self, _simd: S) -> Self::Output {
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

        if m == 1 {
            // m=1 (token generation): memory-bandwidth bound, parallelize over output columns.
            // Scale thread count based on work size: each thread needs enough vec_dot calls
            // to amortize std::thread::scope's thread creation overhead (~10µs per thread).
            let max_threads = crate::parallel::num_threads();
            let total_work = n * blocks_per_weight_row;
            let n_threads = (total_work / 16384).min(max_threads).max(1);

            if n_threads == 1 {
                process_row_integer_tiled::<B>(
                    lhs_data,
                    rhs_blocks,
                    out_data,
                    n,
                    blocks_per_weight_row,
                );
            } else {
                // Same CHUNK_SIZE=32 aligned thread distribution as before,
                // but each thread quantizes activations once instead of per chunk.
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
                            // Quantize activations ONCE per thread (not per chunk).
                            let act_blocks: Vec<B::ActivationBlock> = (0..blocks_per_weight_row)
                                .map(|block_idx| {
                                    let start = block_idx * B::BLOCK_SIZE;
                                    B::quantize_activation(&lhs_data[start..start + B::BLOCK_SIZE])
                                })
                                .collect();

                            for (i, out_chunk) in thread_chunk.chunks_mut(CHUNK_SIZE).enumerate() {
                                let chunk_start = thread_start_n + i * CHUNK_SIZE;
                                let chunk_n = out_chunk.len();
                                for (idx, out_elem) in
                                    out_chunk.iter_mut().enumerate().take(chunk_n)
                                {
                                    let n_out = chunk_start + idx;
                                    let mut sum = 0.0f32;
                                    for (block_idx, act_block) in act_blocks.iter().enumerate() {
                                        sum += rhs_blocks
                                            [n_out * blocks_per_weight_row + block_idx]
                                            .vec_dot(act_block);
                                    }
                                    *out_elem = sum;
                                }
                            }
                        });
                    }
                });
            }
        } else {
            // Multi-row path (m≥2): use outer-loop unrolled 3×4 tiling.
            // Weight blocks are loaded once and reused across 3 LHS rows,
            // reducing memory traffic by ~3× compared to row-at-a-time processing.
            let n_threads = crate::parallel::num_threads();

            // Use at most m/3 threads so each thread gets ≥3 rows,
            // maximizing benefit of the 3-row kernel.
            let effective_threads = (m / 3).min(n_threads).max(1);

            if effective_threads <= 1 {
                process_multi_row_tiled::<B>(
                    lhs_data,
                    rhs_blocks,
                    out_data,
                    m,
                    k,
                    n,
                    blocks_per_weight_row,
                );
            } else {
                let rows_per_thread = m.div_ceil(effective_threads);

                std::thread::scope(|scope| {
                    let mut remaining_out = out_data;
                    let mut row_offset = 0;

                    for thread_id in 0..effective_threads {
                        if remaining_out.is_empty() {
                            break;
                        }

                        let this_rows = if thread_id == effective_threads - 1 {
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
                            process_multi_row_tiled::<B>(
                                &lhs_data
                                    [thread_row_offset * k..(thread_row_offset + this_rows) * k],
                                rhs_blocks,
                                thread_out,
                                this_rows,
                                k,
                                n,
                                blocks_per_weight_row,
                            );
                        });
                    }
                });
            }
        }
    }
}

/// Process a range of output columns for m=1 parallelization
#[allow(dead_code)]
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

/// Process a range of output columns using pre-quantized activation blocks.
/// Uses 4-way column tiling for instruction-level parallelism.
#[allow(dead_code)]
#[inline(always)]
fn process_range_with_acts<B: GgufBlock>(
    act_blocks: &[B::ActivationBlock],
    rhs_blocks: &[B],
    out_chunk: &mut [f32],
    start_n: usize,
    blocks_per_weight_row: usize,
) where
    B::ActivationBlock: Pod,
{
    let chunk_n = out_chunk.len();
    const NR: usize = 4;
    let n_tiles = chunk_n / NR;

    for tile in 0..n_tiles {
        let local_off = tile * NR;
        let col = start_n + local_off;
        let mut acc = [0.0f32; NR];

        for (block_idx, act) in act_blocks.iter().enumerate() {
            acc[0] += rhs_blocks[col * blocks_per_weight_row + block_idx].vec_dot(act);
            acc[1] += rhs_blocks[(col + 1) * blocks_per_weight_row + block_idx].vec_dot(act);
            acc[2] += rhs_blocks[(col + 2) * blocks_per_weight_row + block_idx].vec_dot(act);
            acc[3] += rhs_blocks[(col + 3) * blocks_per_weight_row + block_idx].vec_dot(act);
        }

        out_chunk[local_off..local_off + NR].copy_from_slice(&acc);
    }

    // Handle remainder
    for (i, block) in out_chunk
        .iter_mut()
        .enumerate()
        .take(chunk_n)
        .skip(n_tiles * NR)
    {
        let n_out = start_n + i;
        let mut sum = 0.0f32;
        for (block_idx, act) in act_blocks.iter().enumerate() {
            sum += rhs_blocks[n_out * blocks_per_weight_row + block_idx].vec_dot(act);
        }
        *block = sum;
    }
}

/// Process 3 LHS rows × all N output columns using 3×4 outer-loop unrolled tiling.
///
/// This is the key optimization from llamafile's matmul approach: by processing
/// 3 rows simultaneously, each weight block is loaded once from memory and reused
/// across all 3 rows. This reduces memory traffic for weights by ~3× compared to
/// processing one row at a time.
///
/// Layout: lhs_data contains 3 contiguous rows of k elements each.
///         out_data contains 3 contiguous rows of n elements each.
#[inline(always)]
fn process_3rows_integer_tiled<B: GgufBlock>(
    lhs_data: &[f32],
    rhs_blocks: &[B],
    out_data: &mut [f32],
    k: usize,
    n: usize,
    blocks_per_weight_row: usize,
) where
    B::ActivationBlock: Pod,
{
    // Pre-quantize all 3 rows
    let mut act0: Vec<B::ActivationBlock> = Vec::with_capacity(blocks_per_weight_row);
    let mut act1: Vec<B::ActivationBlock> = Vec::with_capacity(blocks_per_weight_row);
    let mut act2: Vec<B::ActivationBlock> = Vec::with_capacity(blocks_per_weight_row);

    for block_idx in 0..blocks_per_weight_row {
        let s = block_idx * B::BLOCK_SIZE;
        act0.push(B::quantize_activation(&lhs_data[s..s + B::BLOCK_SIZE]));
        act1.push(B::quantize_activation(
            &lhs_data[k + s..k + s + B::BLOCK_SIZE],
        ));
        act2.push(B::quantize_activation(
            &lhs_data[2 * k + s..2 * k + s + B::BLOCK_SIZE],
        ));
    }

    // 3×4 tiled inner loop
    const NR: usize = 4;
    let n_tiles = n / NR;

    for tile in 0..n_tiles {
        let col = tile * NR;

        // 12 explicit accumulators (3 rows × 4 cols) to help register allocation
        let (mut a00, mut a01, mut a02, mut a03) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut a10, mut a11, mut a12, mut a13) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut a20, mut a21, mut a22, mut a23) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

        for block_idx in 0..blocks_per_weight_row {
            // Load 4 weight blocks ONCE (shared across all 3 rows)
            let w0 = &rhs_blocks[col * blocks_per_weight_row + block_idx];
            let w1 = &rhs_blocks[(col + 1) * blocks_per_weight_row + block_idx];
            let w2 = &rhs_blocks[(col + 2) * blocks_per_weight_row + block_idx];
            let w3 = &rhs_blocks[(col + 3) * blocks_per_weight_row + block_idx];

            // Row 0: 4 dot products
            let a = &act0[block_idx];
            a00 += w0.vec_dot(a);
            a01 += w1.vec_dot(a);
            a02 += w2.vec_dot(a);
            a03 += w3.vec_dot(a);

            // Row 1: reusing same weight blocks from cache
            let a = &act1[block_idx];
            a10 += w0.vec_dot(a);
            a11 += w1.vec_dot(a);
            a12 += w2.vec_dot(a);
            a13 += w3.vec_dot(a);

            // Row 2: reusing same weight blocks from cache
            let a = &act2[block_idx];
            a20 += w0.vec_dot(a);
            a21 += w1.vec_dot(a);
            a22 += w2.vec_dot(a);
            a23 += w3.vec_dot(a);
        }

        // Store results for all 3 rows
        out_data[col] = a00;
        out_data[col + 1] = a01;
        out_data[col + 2] = a02;
        out_data[col + 3] = a03;
        out_data[n + col] = a10;
        out_data[n + col + 1] = a11;
        out_data[n + col + 2] = a12;
        out_data[n + col + 3] = a13;
        out_data[2 * n + col] = a20;
        out_data[2 * n + col + 1] = a21;
        out_data[2 * n + col + 2] = a22;
        out_data[2 * n + col + 3] = a23;
    }

    // Handle remainder columns
    for j in (n_tiles * NR)..n {
        let (mut s0, mut s1, mut s2) = (0.0f32, 0.0f32, 0.0f32);
        for block_idx in 0..blocks_per_weight_row {
            let w = &rhs_blocks[j * blocks_per_weight_row + block_idx];
            s0 += w.vec_dot(&act0[block_idx]);
            s1 += w.vec_dot(&act1[block_idx]);
            s2 += w.vec_dot(&act2[block_idx]);
        }
        out_data[j] = s0;
        out_data[n + j] = s1;
        out_data[2 * n + j] = s2;
    }
}

/// Process 2 LHS rows × all N output columns using 2×4 outer-loop unrolled tiling.
/// Same approach as the 3-row version but for the remainder when m % 3 == 2.
#[inline(always)]
fn process_2rows_integer_tiled<B: GgufBlock>(
    lhs_data: &[f32],
    rhs_blocks: &[B],
    out_data: &mut [f32],
    k: usize,
    n: usize,
    blocks_per_weight_row: usize,
) where
    B::ActivationBlock: Pod,
{
    let mut act0: Vec<B::ActivationBlock> = Vec::with_capacity(blocks_per_weight_row);
    let mut act1: Vec<B::ActivationBlock> = Vec::with_capacity(blocks_per_weight_row);

    for block_idx in 0..blocks_per_weight_row {
        let s = block_idx * B::BLOCK_SIZE;
        act0.push(B::quantize_activation(&lhs_data[s..s + B::BLOCK_SIZE]));
        act1.push(B::quantize_activation(
            &lhs_data[k + s..k + s + B::BLOCK_SIZE],
        ));
    }

    const NR: usize = 4;
    let n_tiles = n / NR;

    for tile in 0..n_tiles {
        let col = tile * NR;

        let (mut a00, mut a01, mut a02, mut a03) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let (mut a10, mut a11, mut a12, mut a13) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

        for block_idx in 0..blocks_per_weight_row {
            let w0 = &rhs_blocks[col * blocks_per_weight_row + block_idx];
            let w1 = &rhs_blocks[(col + 1) * blocks_per_weight_row + block_idx];
            let w2 = &rhs_blocks[(col + 2) * blocks_per_weight_row + block_idx];
            let w3 = &rhs_blocks[(col + 3) * blocks_per_weight_row + block_idx];

            let a = &act0[block_idx];
            a00 += w0.vec_dot(a);
            a01 += w1.vec_dot(a);
            a02 += w2.vec_dot(a);
            a03 += w3.vec_dot(a);

            let a = &act1[block_idx];
            a10 += w0.vec_dot(a);
            a11 += w1.vec_dot(a);
            a12 += w2.vec_dot(a);
            a13 += w3.vec_dot(a);
        }

        out_data[col] = a00;
        out_data[col + 1] = a01;
        out_data[col + 2] = a02;
        out_data[col + 3] = a03;
        out_data[n + col] = a10;
        out_data[n + col + 1] = a11;
        out_data[n + col + 2] = a12;
        out_data[n + col + 3] = a13;
    }

    for j in (n_tiles * NR)..n {
        let (mut s0, mut s1) = (0.0f32, 0.0f32);
        for block_idx in 0..blocks_per_weight_row {
            let w = &rhs_blocks[j * blocks_per_weight_row + block_idx];
            s0 += w.vec_dot(&act0[block_idx]);
            s1 += w.vec_dot(&act1[block_idx]);
        }
        out_data[j] = s0;
        out_data[n + j] = s1;
    }
}

/// Process m rows using outer-loop unrolled tiling.
/// Groups rows into sets of 3 for maximum weight reuse, with
/// 2-row and 1-row fallbacks for the remainder.
#[inline(always)]
fn process_multi_row_tiled<B: GgufBlock>(
    lhs_data: &[f32],
    rhs_blocks: &[B],
    out_data: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    blocks_per_weight_row: usize,
) where
    B::ActivationBlock: Pod,
{
    const MR: usize = 3;
    let full_groups = m / MR;
    let remainder = m % MR;

    for g in 0..full_groups {
        let row = g * MR;
        process_3rows_integer_tiled::<B>(
            &lhs_data[row * k..(row + MR) * k],
            rhs_blocks,
            &mut out_data[row * n..(row + MR) * n],
            k,
            n,
            blocks_per_weight_row,
        );
    }

    let rem_start = full_groups * MR;
    match remainder {
        2 => {
            process_2rows_integer_tiled::<B>(
                &lhs_data[rem_start * k..(rem_start + 2) * k],
                rhs_blocks,
                &mut out_data[rem_start * n..(rem_start + 2) * n],
                k,
                n,
                blocks_per_weight_row,
            );
        }
        1 => {
            let lhs_row = &lhs_data[rem_start * k..(rem_start + 1) * k];
            let out_row = &mut out_data[rem_start * n..(rem_start + 1) * n];
            process_row_integer_tiled::<B>(lhs_row, rhs_blocks, out_row, n, blocks_per_weight_row);
        }
        _ => {}
    }
}

/// Process a single output row with SIMD using 4-way tiling for better ILP
#[allow(dead_code)]
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
#[allow(dead_code)]
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

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::{LazyBacking, TensorBacking};
    use fusor_gguf::BlockQ8_0;
    use pulp::bytemuck;

    /// Helper to create a Q8_0 block from scale and data using bytemuck
    fn make_q8_0_block(scale: f32, data: [i8; 32]) -> BlockQ8_0 {
        // Q8_0 layout: scale (f16, 2 bytes) + data (32 i8, 32 bytes) = 34 bytes
        let mut bytes = [0u8; 34];
        let scale_f16 = half::f16::from_f32(scale);
        bytes[0..2].copy_from_slice(&scale_f16.to_le_bytes());
        bytes[2..34].copy_from_slice(bytemuck::cast_slice(&data));
        *bytemuck::from_bytes(&bytes)
    }

    #[test]
    fn test_quantized_tensor_creation() {
        // Create a simple Q8_0 tensor (block size 32)
        // Shape [2, 64] = 128 elements = 4 blocks
        let shape = [2, 64];
        let num_blocks = 4;

        // Create some test blocks
        let mut blocks_vec: AVec<BlockQ8_0> = AVec::with_capacity(64, num_blocks);
        for i in 0..num_blocks {
            let mut data = [0i8; 32];
            for j in 0..32 {
                data[j] = ((i * 32 + j) % 128) as i8;
            }
            blocks_vec.push(make_q8_0_block(1.0, data));
        }
        let blocks = blocks_vec.into_boxed_slice();

        let tensor = QuantizedTensor::from_blocks(shape, blocks);

        assert_eq!(tensor.element_shape(), &[2, 64]);
        assert_eq!(tensor.element_count(), 128);
        assert_eq!(tensor.block_count(), 4);
    }

    #[test]
    fn test_dequantize_q8_0() {
        // Q8_0: scale * i8 values
        let shape = [1, 32];
        let mut data = [0i8; 32];
        for i in 0..32 {
            data[i] = i as i8;
        }

        let mut blocks_vec: AVec<BlockQ8_0> = AVec::with_capacity(64, 1);
        blocks_vec.push(make_q8_0_block(0.5, data));
        let blocks = blocks_vec.into_boxed_slice();

        let quantized = QuantizedTensor::from_blocks(shape, blocks);
        let dequantized = quantized.dequantize::<2>();

        // Verify dequantization: each value should be scale * data[i] = 0.5 * i
        for i in 0..32 {
            let expected = 0.5 * (i as f32);
            let actual = dequantized.get([0, i]);
            assert!(
                (actual - expected).abs() < 1e-6,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_lazy_dequantize() {
        let shape = [1, 32];
        let mut data = [0i8; 32];
        for i in 0..32 {
            data[i] = (i as i8) - 16;
        }

        let mut blocks_vec: AVec<BlockQ8_0> = AVec::with_capacity(64, 1);
        blocks_vec.push(make_q8_0_block(2.0, data));
        let blocks = blocks_vec.into_boxed_slice();

        let quantized = QuantizedTensor::from_blocks(shape, blocks);
        let lazy = quantized.dequantize_lazy::<2>();

        // Test layout methods
        assert_eq!(lazy.layout().num_elements(), 32);
        assert_eq!(lazy.layout().shape(), &[1, 32]);
        assert!(lazy.layout().is_contiguous());

        // Test scalar evaluation
        for i in 0..32 {
            let expected = 2.0 * ((i as f32) - 16.0);
            let actual = lazy.eval_scalar(i);
            assert!(
                (actual - expected).abs() < 1e-6,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_matmul_quantized_simple() {
        // LHS: [2, 32] f32 tensor (all ones)
        // RHS: [32, 32] quantized (identity-like pattern)
        let lhs = ConcreteTensor::<f32, 2>::from_slice([2, 32], &vec![1.0f32; 64]);

        // Create a Q8_0 tensor where each block dequantizes to known values
        let shape = [32, 32];
        let num_blocks = 32; // 32 rows * 32 cols / 32 block_size = 32 blocks

        let mut blocks_vec: AVec<BlockQ8_0> = AVec::with_capacity(64, num_blocks);
        for _ in 0..num_blocks {
            let data = [1i8; 32]; // Each element is 1.0 after dequant
            blocks_vec.push(make_q8_0_block(1.0, data));
        }
        let blocks = blocks_vec.into_boxed_slice();
        let rhs = QuantizedTensor::from_blocks(shape, blocks);

        let result = lhs.q_mat_mul(&rhs);

        // Each output element should be sum of 32 ones = 32.0
        assert_eq!(result.layout().shape(), &[2, 32]);
        for i in 0..2 {
            for j in 0..32 {
                let val = result.get([i, j]);
                // Allow for quantization error from Q8_0 activation quantization
                assert!(
                    (val - 32.0).abs() < 0.1,
                    "Mismatch at [{}, {}]: expected 32.0, got {}",
                    i,
                    j,
                    val
                );
            }
        }
    }

    #[test]
    fn test_matmul_quantized_vs_dequantize() {
        // Compare matmul_quantized with dequantize + regular matmul
        // qmatmul computes: C = A @ W.T where W is [N, K] (out_features, in_features)
        // So to compare with regular matmul, we need: C = A @ W_dequant.T
        let lhs = ConcreteTensor::<f32, 2>::from_slice(
            [3, 64],
            &(0..192).map(|x| x as f32 * 0.1).collect::<Vec<_>>(),
        );

        // Create a quantized tensor with shape [N, K] = [out_features, in_features]
        // N = 64 (out_features), K = 64 (in_features)
        let shape = [64, 64];
        let num_blocks = 64 * 64 / 32; // 128 blocks

        let mut blocks_vec: AVec<BlockQ8_0> = AVec::with_capacity(64, num_blocks);
        for i in 0..num_blocks {
            let mut data = [0i8; 32];
            for j in 0..32 {
                data[j] = ((i + j) % 100) as i8 - 50;
            }
            blocks_vec.push(make_q8_0_block(0.1, data));
        }
        let blocks = blocks_vec.into_boxed_slice();
        let rhs = QuantizedTensor::from_blocks(shape, blocks);

        // Compute using quantized matmul
        let result_quantized = lhs.q_mat_mul(&rhs);

        // Compute using dequantize + regular matmul
        // Since qmatmul does A @ W.T and W is [N, K], we need A @ W.T
        // Regular matmul expects [K, N], so we transpose dequantized weight
        let rhs_dequantized = rhs.dequantize::<2>();
        let rhs_tensor = crate::Tensor::new(rhs_dequantized);
        let rhs_transposed = rhs_tensor.transpose(0, 1).to_concrete();
        let result_dequantized = lhs.matmul_ref(rhs_transposed.inner());

        // Results should match
        assert_eq!(
            result_quantized.layout().shape(),
            result_dequantized.layout().shape()
        );
        for i in 0..3 {
            for j in 0..64 {
                let q_val = result_quantized.get([i, j]);
                let d_val = result_dequantized.get([i, j]);
                // Allow up to 3% relative error due to activation quantization
                let tolerance = d_val.abs().max(1.0) * 0.03;
                assert!(
                    (q_val - d_val).abs() < tolerance,
                    "Mismatch at [{}, {}]: quantized={}, dequantized={}, diff={}",
                    i,
                    j,
                    q_val,
                    d_val,
                    (q_val - d_val).abs()
                );
            }
        }
    }

    #[test]
    fn test_from_raw_bytes() {
        // Test creating a quantized tensor from raw bytes
        let shape = [1, 32];

        // Create raw bytes for one Q8_0 block: scale (f16) + data (32 i8)
        let mut bytes = vec![0u8; 34];
        let scale_f16 = half::f16::from_f32(1.0);
        bytes[0..2].copy_from_slice(&scale_f16.to_le_bytes());
        for i in 0..32 {
            bytes[2 + i] = i as u8;
        }

        let tensor = QuantizedTensor::<BlockQ8_0>::from_raw_bytes(shape, &bytes);

        assert_eq!(tensor.element_shape(), &[1, 32]);
        assert_eq!(tensor.block_count(), 1);

        // Verify dequantization
        let dequantized = tensor.dequantize::<2>();
        for i in 0..32 {
            let expected = i as f32;
            let actual = dequantized.get([0, i]);
            assert!(
                (actual - expected).abs() < 1e-6,
                "Mismatch at index {}: expected {}, got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    #[should_panic(expected = "Innermost dimension")]
    fn test_invalid_shape_not_multiple_of_block_size() {
        let shape = [2, 33]; // 33 is not a multiple of 32
        let blocks_vec: AVec<BlockQ8_0> = AVec::new(64);
        let blocks = blocks_vec.into_boxed_slice();
        let _ = QuantizedTensor::from_blocks(shape, blocks);
    }

    #[test]
    #[should_panic(expected = "Expected")]
    fn test_invalid_block_count() {
        let shape = [2, 64]; // Expects 4 blocks
        let mut blocks_vec: AVec<BlockQ8_0> = AVec::with_capacity(64, 2);
        for _ in 0..2 {
            blocks_vec.push(make_q8_0_block(1.0, [0i8; 32]));
        }
        let blocks = blocks_vec.into_boxed_slice();
        let _ = QuantizedTensor::from_blocks(shape, blocks);
    }

    #[test]
    fn test_batched_q_mat_mul_3d() {
        // Test batched q_mat_mul with 3D input: [batch, M, K] @ [N, K].T -> [batch, M, N]
        // Weight W has shape [N, K] (out_features, in_features), matching GPU convention
        let batch = 2;
        let m = 3;
        let k = 64;
        let n = 32;

        // Create batched LHS: [2, 3, 64]
        let lhs_data: Vec<f32> = (0..(batch * m * k)).map(|x| (x as f32) * 0.01).collect();
        let lhs = ConcreteTensor::<f32, 3>::from_slice([batch, m, k], &lhs_data);

        // Create quantized RHS: [N, K] = [32, 64] (out_features, in_features)
        let rhs_shape = [n, k];
        let num_blocks = n * k / 32;
        let mut blocks_vec: AVec<BlockQ8_0> = AVec::with_capacity(64, num_blocks);
        for i in 0..num_blocks {
            let mut data = [0i8; 32];
            for j in 0..32 {
                data[j] = ((i + j) % 100) as i8 - 50;
            }
            blocks_vec.push(make_q8_0_block(0.1, data));
        }
        let blocks = blocks_vec.into_boxed_slice();
        let rhs = QuantizedTensor::from_blocks(rhs_shape, blocks);

        // Compute using batched q_mat_mul
        let result = lhs.q_mat_mul(&rhs);

        // Verify output shape
        assert_eq!(result.layout().shape(), &[batch, m, n]);

        // Compare with dequantize + regular batched matmul for each batch
        // qmatmul computes: C = A @ W.T where W is [N, K]
        let rhs_dequantized = rhs.dequantize::<2>();
        for b in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    // Compute expected value: C[i, j] = sum_l A[i, l] * W[j, l]
                    let mut expected = 0.0f32;
                    for l in 0..k {
                        expected += lhs.get([b, i, l]) * rhs_dequantized.get([j, l]);
                    }
                    let actual = result.get([b, i, j]);
                    // Combined error from Q8_0 weights + Q8_0 activation quantization
                    let tolerance = expected.abs().max(1.0) * 0.03; // 3% relative error
                    assert!(
                        (actual - expected).abs() < tolerance,
                        "Mismatch at [{}, {}, {}]: expected={}, actual={}, diff={}",
                        b,
                        i,
                        j,
                        expected,
                        actual,
                        (actual - expected).abs()
                    );
                }
            }
        }
    }

    #[test]
    fn test_batched_q_mat_mul_4d() {
        // Test batched q_mat_mul with 4D input: [b1, b2, M, K] @ [N, K].T -> [b1, b2, M, N]
        // Weight W has shape [N, K] (out_features, in_features), matching GPU convention
        let b1 = 2;
        let b2 = 3;
        let m = 2;
        let k = 32;
        let n = 32;

        // Create batched LHS: [2, 3, 2, 32]
        let lhs_data: Vec<f32> = (0..(b1 * b2 * m * k)).map(|x| (x as f32) * 0.02).collect();
        let lhs = ConcreteTensor::<f32, 4>::from_slice([b1, b2, m, k], &lhs_data);

        // Create quantized RHS: [N, K] = [32, 32] (out_features, in_features)
        let rhs_shape = [n, k];
        let num_blocks = n * k / 32;
        let mut blocks_vec: AVec<BlockQ8_0> = AVec::with_capacity(64, num_blocks);
        for i in 0..num_blocks {
            let mut data = [0i8; 32];
            for j in 0..32 {
                data[j] = ((i * 2 + j) % 80) as i8 - 40;
            }
            blocks_vec.push(make_q8_0_block(0.05, data));
        }
        let blocks = blocks_vec.into_boxed_slice();
        let rhs = QuantizedTensor::from_blocks(rhs_shape, blocks);

        // Compute using batched q_mat_mul
        let result = lhs.q_mat_mul(&rhs);

        // Verify output shape
        assert_eq!(result.layout().shape(), &[b1, b2, m, n]);

        // Compare with dequantize for spot checks
        // qmatmul computes: C = A @ W.T where W is [N, K]
        let rhs_dequantized = rhs.dequantize::<2>();
        for bi in 0..b1 {
            for bj in 0..b2 {
                for i in 0..m {
                    for j in 0..n {
                        // Compute expected value: C[i, j] = sum_l A[i, l] * W[j, l]
                        let mut expected = 0.0f32;
                        for l in 0..k {
                            expected += lhs.get([bi, bj, i, l]) * rhs_dequantized.get([j, l]);
                        }
                        let actual = result.get([bi, bj, i, j]);
                        // Combined error from Q8_0 weights + Q8_0 activation quantization
                        let tolerance = expected.abs().max(1.0) * 0.06; // 6% relative error
                        assert!(
                            (actual - expected).abs() < tolerance,
                            "Mismatch at [{}, {}, {}, {}]: expected={}, actual={}, diff={}",
                            bi,
                            bj,
                            i,
                            j,
                            expected,
                            actual,
                            (actual - expected).abs()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_q4_0_matmul() {
        // Test Q4_0 quantization specifically, since that's what the whisper model uses
        // Weight W has shape [N, K] (out_features, in_features), matching GPU convention
        use fusor_gguf::BlockQ4_0;

        let m = 2;
        let k = 64;
        let n = 32;

        // Create LHS: [2, 64]
        let lhs_data: Vec<f32> = (0..(m * k)).map(|x| (x as f32) * 0.01 - 0.5).collect();
        let lhs = ConcreteTensor::<f32, 2>::from_slice([m, k], &lhs_data);

        // Create quantized RHS using Q4_0: [N, K] = [32, 64]
        // Q4_0 block: 2 bytes (f16 scale) + 16 bytes (32 4-bit values packed) = 18 bytes
        let rhs_shape = [n, k];
        let num_blocks = n * k / 32;

        // Manually create Q4_0 blocks with known patterns
        let mut raw_bytes = Vec::with_capacity(num_blocks * std::mem::size_of::<BlockQ4_0>());
        for block_idx in 0..num_blocks {
            // Set scale to 0.1
            let scale_f16 = half::f16::from_f32(0.1);
            raw_bytes.extend_from_slice(&scale_f16.to_le_bytes());

            // Pack 32 4-bit values: low nibble (indices 0-15), high nibble (indices 16-31)
            // Create a simple pattern where values are (block_idx + i) % 16 centered at 8
            for i in 0..16 {
                let low_val = ((block_idx + i) % 16) as u8; // indices 0-15
                let high_val = ((block_idx + i + 8) % 16) as u8; // indices 16-31
                let packed = low_val | (high_val << 4);
                raw_bytes.push(packed);
            }
        }

        let rhs = QuantizedTensor::<BlockQ4_0>::from_raw_bytes(rhs_shape, &raw_bytes);

        // Compute using q_mat_mul
        let result = lhs.q_mat_mul(&rhs);

        // Verify output shape
        assert_eq!(result.layout().shape(), &[m, n]);

        // Compare with dequantize + manual matmul
        let rhs_dequantized = rhs.dequantize::<2>();
        for i in 0..m {
            for j in 0..n {
                // qmatmul computes: C[i, j] = sum_l A[i, l] * W[j, l]
                let mut expected = 0.0f32;
                for l in 0..k {
                    expected += lhs.get([i, l]) * rhs_dequantized.get([j, l]);
                }
                let actual = result.get([i, j]);
                // Combined error from Q4_0 weights + Q8_0 activation quantization
                // is higher than just dequantizing weights (~1-2% error is expected)
                let tolerance = expected.abs().max(1.0) * 0.02; // 2% relative error
                assert!(
                    (actual - expected).abs() < tolerance,
                    "Q4_0 mismatch at [{}, {}]: expected={}, actual={}, diff={}",
                    i,
                    j,
                    expected,
                    actual,
                    (actual - expected).abs()
                );
            }
        }
    }

    #[test]
    fn test_q4_0_matmul_realistic_size() {
        // Test Q4_0 with whisper model-like dimensions
        // Linear layer: input [1, seq, 384] @ weight [384, 384].T + bias
        use fusor_gguf::BlockQ4_0;

        let batch = 1;
        let seq_len = 4;
        let in_features = 384;
        let out_features = 384;

        // Create LHS: [1, 4, 384] with random-ish values
        let lhs_data: Vec<f32> = (0..(batch * seq_len * in_features))
            .map(|x| (x as f32 * 0.1).sin() * 2.0)
            .collect();
        let lhs = ConcreteTensor::<f32, 3>::from_slice([batch, seq_len, in_features], &lhs_data);

        // Create quantized weights: [out_features, in_features] = [384, 384]
        let weight_shape = [out_features, in_features];
        let num_blocks = out_features * in_features / 32;

        // Create realistic Q4_0 weights with varied scales and data
        let mut raw_bytes = Vec::with_capacity(num_blocks * std::mem::size_of::<BlockQ4_0>());
        for block_idx in 0..num_blocks {
            // Vary scale based on block position
            let scale = 0.05 + (block_idx as f32 * 0.0001);
            let scale_f16 = half::f16::from_f32(scale);
            raw_bytes.extend_from_slice(&scale_f16.to_le_bytes());

            // Create varied data pattern
            for i in 0..16 {
                let low_val = ((block_idx * 3 + i * 7) % 16) as u8;
                let high_val = ((block_idx * 5 + i * 11 + 4) % 16) as u8;
                let packed = low_val | (high_val << 4);
                raw_bytes.push(packed);
            }
        }

        let weights = QuantizedTensor::<BlockQ4_0>::from_raw_bytes(weight_shape, &raw_bytes);

        // Compute using q_mat_mul
        let result = lhs.q_mat_mul(&weights);

        // Verify output shape
        assert_eq!(result.layout().shape(), &[batch, seq_len, out_features]);

        // Compare with dequantize + manual matmul for the first few positions
        let weights_dequantized = weights.dequantize::<2>();
        for b in 0..batch {
            for s in 0..seq_len.min(2) {
                for o in 0..out_features.min(10) {
                    let mut expected = 0.0f32;
                    for i in 0..in_features {
                        expected += lhs.get([b, s, i]) * weights_dequantized.get([o, i]);
                    }
                    let actual = result.get([b, s, o]);
                    assert!(
                        (actual - expected).abs() < 0.1,
                        "Realistic Q4_0 mismatch at [{}, {}, {}]: expected={}, actual={}, diff={}",
                        b,
                        s,
                        o,
                        expected,
                        actual,
                        (actual - expected).abs()
                    );
                }
            }
        }
    }

    #[test]
    fn test_batched_q_mat_mul_matches_unbatched() {
        // Verify that batched results match unbatched when run individually
        // Weight W has shape [N, K] (out_features, in_features), matching GPU convention
        let m = 2;
        let k = 64;
        let n = 32;

        // Create two separate 2D matrices
        let lhs1_data: Vec<f32> = (0..(m * k)).map(|x| (x as f32) * 0.01).collect();
        let lhs2_data: Vec<f32> = (0..(m * k)).map(|x| (x as f32) * 0.02 + 0.5).collect();

        let lhs1 = ConcreteTensor::<f32, 2>::from_slice([m, k], &lhs1_data);
        let lhs2 = ConcreteTensor::<f32, 2>::from_slice([m, k], &lhs2_data);

        // Create batched version: [2, m, k]
        let mut batched_data = lhs1_data.clone();
        batched_data.extend(&lhs2_data);
        let lhs_batched = ConcreteTensor::<f32, 3>::from_slice([2, m, k], &batched_data);

        // Create quantized RHS: [N, K] = [32, 64] (out_features, in_features)
        let rhs_shape = [n, k];
        let num_blocks = n * k / 32;
        let mut blocks_vec: AVec<BlockQ8_0> = AVec::with_capacity(64, num_blocks);
        for i in 0..num_blocks {
            let mut data = [0i8; 32];
            for j in 0..32 {
                data[j] = ((i + j * 3) % 100) as i8 - 50;
            }
            blocks_vec.push(make_q8_0_block(0.1, data));
        }
        let blocks = blocks_vec.into_boxed_slice();
        let rhs = QuantizedTensor::from_blocks(rhs_shape, blocks);

        // Compute separately
        let result1 = lhs1.q_mat_mul(&rhs);
        let result2 = lhs2.q_mat_mul(&rhs);

        // Compute batched
        let result_batched = lhs_batched.q_mat_mul(&rhs);

        // Verify shapes
        assert_eq!(result1.layout().shape(), &[m, n]);
        assert_eq!(result_batched.layout().shape(), &[2, m, n]);

        // Verify values match
        for i in 0..m {
            for j in 0..n {
                let v1 = result1.get([i, j]);
                let v1_batched = result_batched.get([0, i, j]);
                assert!(
                    (v1 - v1_batched).abs() < 1e-6,
                    "Batch 0 mismatch at [{}, {}]: unbatched={}, batched={}",
                    i,
                    j,
                    v1,
                    v1_batched
                );

                let v2 = result2.get([i, j]);
                let v2_batched = result_batched.get([1, i, j]);
                assert!(
                    (v2 - v2_batched).abs() < 1e-6,
                    "Batch 1 mismatch at [{}, {}]: unbatched={}, batched={}",
                    i,
                    j,
                    v2,
                    v2_batched
                );
            }
        }
    }
}
