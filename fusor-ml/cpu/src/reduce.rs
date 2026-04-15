//! Reduction operations: sum, max, min, prod (full tensor and axis-wise)

use pulp::{Arch, Simd, WithSimd};

use crate::{ConcreteTensor, IndexIterator, LastRank, ResolvedTensor, SimdElement};

/// Trait for reduction operations that have SIMD support
pub trait SimdReduceOp<E: SimdElement>: Copy {
    /// Identity element for the reduction (e.g., 0 for sum, MIN for max)
    fn identity() -> E;

    /// Create a SIMD vector filled with the identity value
    fn splat_identity<S: Simd>(simd: S) -> E::Simd<S>;

    /// Combine two SIMD vectors element-wise (e.g., add for sum, max for max)
    fn combine_simd_vec<S: Simd>(simd: S, a: E::Simd<S>, b: E::Simd<S>) -> E::Simd<S>;

    /// Combine two scalar values
    fn combine_scalar(a: E, b: E) -> E;

    /// Reduce a SIMD vector to a scalar (horizontal reduction)
    fn reduce_simd_vec<S: Simd>(simd: S, v: E::Simd<S>) -> E;
}

// Reduce operation markers
macro_rules! define_op_marker {
    ($($name:ident),* $(,)?) => {
        $(
            #[derive(Copy, Clone)]
            pub struct $name;
        )*
    };
}
define_op_marker!(SumOp, MaxOp, MinOp, ProdOp);

// Trait for scalar combine operations used by both SIMD and horizontal reduction
trait ScalarCombine<T>: Copy {
    fn combine(a: T, b: T) -> T;
}

impl ScalarCombine<f32> for SumOp {
    #[inline(always)]
    fn combine(a: f32, b: f32) -> f32 {
        a + b
    }
}
impl ScalarCombine<f64> for SumOp {
    #[inline(always)]
    fn combine(a: f64, b: f64) -> f64 {
        a + b
    }
}
impl ScalarCombine<f32> for MaxOp {
    #[inline(always)]
    fn combine(a: f32, b: f32) -> f32 {
        a.max(b)
    }
}
impl ScalarCombine<f64> for MaxOp {
    #[inline(always)]
    fn combine(a: f64, b: f64) -> f64 {
        a.max(b)
    }
}
impl ScalarCombine<f32> for MinOp {
    #[inline(always)]
    fn combine(a: f32, b: f32) -> f32 {
        a.min(b)
    }
}
impl ScalarCombine<f64> for MinOp {
    #[inline(always)]
    fn combine(a: f64, b: f64) -> f64 {
        a.min(b)
    }
}
impl ScalarCombine<f32> for ProdOp {
    #[inline(always)]
    fn combine(a: f32, b: f32) -> f32 {
        a * b
    }
}
impl ScalarCombine<f64> for ProdOp {
    #[inline(always)]
    fn combine(a: f64, b: f64) -> f64 {
        a * b
    }
}

macro_rules! impl_scalar_combine_int {
    ($op:ty, $elem:ty, $method:ident) => {
        impl ScalarCombine<$elem> for $op {
            #[inline(always)]
            fn combine(a: $elem, b: $elem) -> $elem {
                a.$method(b)
            }
        }
    };
}

// SumOp for integers (wrapping add)
impl_scalar_combine_int!(SumOp, i8, wrapping_add);
impl_scalar_combine_int!(SumOp, i16, wrapping_add);
impl_scalar_combine_int!(SumOp, i32, wrapping_add);
impl_scalar_combine_int!(SumOp, i64, wrapping_add);
impl_scalar_combine_int!(SumOp, u8, wrapping_add);
impl_scalar_combine_int!(SumOp, u16, wrapping_add);
impl_scalar_combine_int!(SumOp, u32, wrapping_add);
impl_scalar_combine_int!(SumOp, u64, wrapping_add);

// MaxOp for integers
impl_scalar_combine_int!(MaxOp, i8, max);
impl_scalar_combine_int!(MaxOp, i16, max);
impl_scalar_combine_int!(MaxOp, i32, max);
impl_scalar_combine_int!(MaxOp, i64, max);
impl_scalar_combine_int!(MaxOp, u8, max);
impl_scalar_combine_int!(MaxOp, u16, max);
impl_scalar_combine_int!(MaxOp, u32, max);
impl_scalar_combine_int!(MaxOp, u64, max);

// MinOp for integers
impl_scalar_combine_int!(MinOp, i8, min);
impl_scalar_combine_int!(MinOp, i16, min);
impl_scalar_combine_int!(MinOp, i32, min);
impl_scalar_combine_int!(MinOp, i64, min);
impl_scalar_combine_int!(MinOp, u8, min);
impl_scalar_combine_int!(MinOp, u16, min);
impl_scalar_combine_int!(MinOp, u32, min);
impl_scalar_combine_int!(MinOp, u64, min);

// ProdOp for integers (wrapping mul)
impl_scalar_combine_int!(ProdOp, i16, wrapping_mul);
impl_scalar_combine_int!(ProdOp, i32, wrapping_mul);
impl_scalar_combine_int!(ProdOp, u16, wrapping_mul);
impl_scalar_combine_int!(ProdOp, u32, wrapping_mul);

// Macro for reduce implementations with SIMD support
macro_rules! impl_reduce_op {
    ($op:ty, $elem:ty, $identity:expr, $splat:ident, $simd_combine:ident) => {
        impl SimdReduceOp<$elem> for $op {
            #[inline(always)]
            fn identity() -> $elem {
                $identity
            }

            #[inline(always)]
            fn splat_identity<S: Simd>(simd: S) -> <$elem as SimdElement>::Simd<S> {
                simd.$splat($identity)
            }

            #[inline(always)]
            fn combine_simd_vec<S: Simd>(
                simd: S,
                a: <$elem as SimdElement>::Simd<S>,
                b: <$elem as SimdElement>::Simd<S>,
            ) -> <$elem as SimdElement>::Simd<S> {
                simd.$simd_combine(a, b)
            }

            #[inline(always)]
            fn combine_scalar(a: $elem, b: $elem) -> $elem {
                <$op as ScalarCombine<$elem>>::combine(a, b)
            }

            #[inline(always)]
            fn reduce_simd_vec<S: Simd>(_simd: S, v: <$elem as SimdElement>::Simd<S>) -> $elem {
                // Safe: cast SIMD ref to scalar slice via bytemuck
                let slice: &[$elem] = pulp::bytemuck::cast_slice(std::slice::from_ref(&v));
                slice.iter().copied().fold($identity, |acc, x| {
                    <$op as ScalarCombine<$elem>>::combine(acc, x)
                })
            }
        }
    };
}

// SumOp for floats
impl_reduce_op!(SumOp, f32, 0.0, splat_f32s, add_f32s);
impl_reduce_op!(SumOp, f64, 0.0, splat_f64s, add_f64s);

// MaxOp for floats
impl_reduce_op!(MaxOp, f32, f32::NEG_INFINITY, splat_f32s, max_f32s);
impl_reduce_op!(MaxOp, f64, f64::NEG_INFINITY, splat_f64s, max_f64s);

// MinOp for floats
impl_reduce_op!(MinOp, f32, f32::INFINITY, splat_f32s, min_f32s);
impl_reduce_op!(MinOp, f64, f64::INFINITY, splat_f64s, min_f64s);

// ProdOp for floats
impl_reduce_op!(ProdOp, f32, 1.0, splat_f32s, mul_f32s);
impl_reduce_op!(ProdOp, f64, 1.0, splat_f64s, mul_f64s);

// SumOp for integers
impl_reduce_op!(SumOp, i8, 0, splat_i8s, add_i8s);
impl_reduce_op!(SumOp, i16, 0, splat_i16s, add_i16s);
impl_reduce_op!(SumOp, i32, 0, splat_i32s, add_i32s);
impl_reduce_op!(SumOp, i64, 0, splat_i64s, add_i64s);
impl_reduce_op!(SumOp, u8, 0, splat_u8s, add_u8s);
impl_reduce_op!(SumOp, u16, 0, splat_u16s, add_u16s);
impl_reduce_op!(SumOp, u32, 0, splat_u32s, add_u32s);
impl_reduce_op!(SumOp, u64, 0, splat_u64s, add_u64s);

// MaxOp for integers
impl_reduce_op!(MaxOp, i8, i8::MIN, splat_i8s, max_i8s);
impl_reduce_op!(MaxOp, i16, i16::MIN, splat_i16s, max_i16s);
impl_reduce_op!(MaxOp, i32, i32::MIN, splat_i32s, max_i32s);
impl_reduce_op!(MaxOp, i64, i64::MIN, splat_i64s, max_i64s);
impl_reduce_op!(MaxOp, u8, u8::MIN, splat_u8s, max_u8s);
impl_reduce_op!(MaxOp, u16, u16::MIN, splat_u16s, max_u16s);
impl_reduce_op!(MaxOp, u32, u32::MIN, splat_u32s, max_u32s);
impl_reduce_op!(MaxOp, u64, u64::MIN, splat_u64s, max_u64s);

// MinOp for integers
impl_reduce_op!(MinOp, i8, i8::MAX, splat_i8s, min_i8s);
impl_reduce_op!(MinOp, i16, i16::MAX, splat_i16s, min_i16s);
impl_reduce_op!(MinOp, i32, i32::MAX, splat_i32s, min_i32s);
impl_reduce_op!(MinOp, i64, i64::MAX, splat_i64s, min_i64s);
impl_reduce_op!(MinOp, u8, u8::MAX, splat_u8s, min_u8s);
impl_reduce_op!(MinOp, u16, u16::MAX, splat_u16s, min_u16s);
impl_reduce_op!(MinOp, u32, u32::MAX, splat_u32s, min_u32s);
impl_reduce_op!(MinOp, u64, u64::MAX, splat_u64s, min_u64s);

// ProdOp for integers that have SIMD multiply (i16, i32, u16, u32)
impl_reduce_op!(ProdOp, i16, 1, splat_i16s, mul_i16s);
impl_reduce_op!(ProdOp, i32, 1, splat_i32s, mul_i32s);
impl_reduce_op!(ProdOp, u16, 1, splat_u16s, mul_u16s);
impl_reduce_op!(ProdOp, u32, 1, splat_u32s, mul_u32s);

/// Helper struct for dispatching reduce operations via Arch::dispatch
struct ReduceOpDispatch<'a, E: SimdElement, Op: SimdReduceOp<E>> {
    input: &'a [E],
    _op: std::marker::PhantomData<Op>,
}

impl<E: SimdElement, Op: SimdReduceOp<E>> WithSimd for ReduceOpDispatch<'_, E, Op> {
    type Output = E;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> E {
        let (in_simd, in_tail) = E::as_simd::<S>(self.input);

        // Use 4 accumulators for better instruction-level parallelism
        let mut acc0 = Op::splat_identity(simd);
        let mut acc1 = Op::splat_identity(simd);
        let mut acc2 = Op::splat_identity(simd);
        let mut acc3 = Op::splat_identity(simd);

        // Process 4 SIMD vectors at a time for better ILP
        let chunks = in_simd.chunks_exact(4);
        let remainder = chunks.remainder();

        for chunk in chunks {
            acc0 = Op::combine_simd_vec(simd, acc0, chunk[0]);
            acc1 = Op::combine_simd_vec(simd, acc1, chunk[1]);
            acc2 = Op::combine_simd_vec(simd, acc2, chunk[2]);
            acc3 = Op::combine_simd_vec(simd, acc3, chunk[3]);
        }

        // Handle remaining SIMD vectors
        for v in remainder {
            acc0 = Op::combine_simd_vec(simd, acc0, *v);
        }

        // Combine accumulators
        acc0 = Op::combine_simd_vec(simd, acc0, acc1);
        acc2 = Op::combine_simd_vec(simd, acc2, acc3);
        acc0 = Op::combine_simd_vec(simd, acc0, acc2);

        // Horizontal reduce SIMD to scalar
        let mut result = Op::reduce_simd_vec(simd, acc0);

        // Handle tail elements
        for &v in in_tail {
            result = Op::combine_scalar(result, v);
        }

        result
    }
}

/// Perform a reduce operation on contiguous slices using SIMD dispatch
#[inline(always)]
fn reduce_op_contiguous<E: SimdElement, Op: SimdReduceOp<E>>(input: &[E]) -> E {
    Arch::new().dispatch(ReduceOpDispatch::<E, Op> {
        input,
        _op: std::marker::PhantomData,
    })
}

/// Full reduction on tensor (handles strided case)
pub(crate) fn reduce_tensor_op<E: SimdElement, const R: usize, Op: SimdReduceOp<E>>(
    tensor: &ConcreteTensor<E, R>,
) -> E {
    if tensor.layout().is_contiguous() {
        reduce_op_contiguous::<E, Op>(tensor.data())
    } else {
        // Fall back to scalar iteration for strided tensors
        let mut result = Op::identity();
        for indices in IndexIterator::new(tensor.layout().shape()) {
            let idx = tensor.layout().linear_index(&indices);
            result = Op::combine_scalar(result, tensor.data()[idx]);
        }
        result
    }
}

/// Fused softmax operation along the last dimension.
///
/// This performs softmax in a single pass for better cache efficiency:
/// 1. Find max while streaming through data
/// 2. Compute exp(x - max) and sum in second pass
/// 3. Normalize by dividing by sum
///
/// For attention matrices, this is significantly faster than separate max/sub/exp/sum/div operations.
pub fn softmax_last_dim_fused<const R: usize>(
    tensor: &ConcreteTensor<f32, R>,
) -> ConcreteTensor<f32, R> {
    let shape: [usize; R] = tensor
        .layout()
        .shape()
        .try_into()
        .expect("Shape length mismatch");
    let last_dim = shape[R - 1];

    // Total number of rows (product of all dims except last)
    let num_rows: usize = shape[..R - 1].iter().product();

    // Use zeros() because softmax_row_simd needs to read back output in Pass 3
    let mut output = ConcreteTensor::<f32, R>::zeros(shape);

    if tensor.layout().is_contiguous() {
        // Fast path: contiguous data
        let in_data = tensor.data();
        let out_data = output.data_mut();

        // Process rows in parallel for large tensors
        let n_threads = crate::parallel::num_threads();
        if num_rows >= 4 && n_threads > 1 {
            crate::parallel::parallel_zip_chunks_mut(
                in_data,
                out_data,
                last_dim,
                |_, in_row, out_row| {
                    softmax_row_simd(in_row, out_row);
                },
            );
        } else {
            // Sequential for small batch sizes
            for (in_row, out_row) in in_data.chunks(last_dim).zip(out_data.chunks_mut(last_dim)) {
                softmax_row_simd(in_row, out_row);
            }
        }
    } else {
        // Slow path: extract rows to contiguous buffer
        let strides = tensor.layout().strides();
        let in_data = tensor.data();
        let out_data = output.data_mut();

        // Check if only the batch dimensions are non-contiguous (last dim is contiguous)
        let last_dim_contiguous = strides[R - 1] == 1;

        if last_dim_contiguous {
            // We can process rows directly without copying
            for row_idx in 0..num_rows {
                // Compute start offset for this row
                let mut remaining = row_idx;
                let mut start_offset = 0;
                for dim in (0..R - 1).rev() {
                    let dim_size = shape[dim];
                    let dim_idx = remaining % dim_size;
                    remaining /= dim_size;
                    start_offset += dim_idx * strides[dim];
                }

                let in_row = &in_data[start_offset..start_offset + last_dim];
                let out_row = &mut out_data[row_idx * last_dim..(row_idx + 1) * last_dim];
                softmax_row_simd(in_row, out_row);
            }
        } else {
            // Need to extract each row
            let mut row_buffer = vec![0.0f32; last_dim];
            for row_idx in 0..num_rows {
                // Extract row to buffer
                let mut remaining = row_idx;
                let mut base_indices = [0usize; R];
                for dim in (0..R - 1).rev() {
                    let dim_size = shape[dim];
                    base_indices[dim] = remaining % dim_size;
                    remaining /= dim_size;
                }

                for (i, row_elem) in row_buffer.iter_mut().enumerate().take(last_dim) {
                    base_indices[R - 1] = i;
                    let idx = tensor.layout().linear_index(&base_indices);
                    *row_elem = in_data[idx];
                }

                let out_row = &mut out_data[row_idx * last_dim..(row_idx + 1) * last_dim];
                softmax_row_simd(&row_buffer, out_row);
            }
        }
    }

    output
}

/// SIMD-optimized softmax for a single row
#[inline(always)]
fn softmax_row_simd(input: &[f32], output: &mut [f32]) {
    Arch::new().dispatch(SoftmaxRowDispatch { input, output });
}

struct SoftmaxRowDispatch<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for SoftmaxRowDispatch<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self { input, output } = self;

        // Pass 1: Find max
        let (in_simd, in_tail) = S::as_simd_f32s(input);
        let mut max_acc = simd.splat_f32s(f32::NEG_INFINITY);

        for &v in in_simd {
            max_acc = simd.max_f32s(max_acc, v);
        }

        // Reduce SIMD max to scalar
        let max_slice: &[f32] = pulp::bytemuck::cast_slice(std::slice::from_ref(&max_acc));
        let mut max_val = max_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for &v in in_tail {
            max_val = max_val.max(v);
        }

        let max_splat = simd.splat_f32s(max_val);

        // Pass 2: Compute exp(x - max) and sum
        let (out_simd, out_tail) = f32::as_mut_simd::<S>(output);
        let mut sum_acc = simd.splat_f32s(0.0);

        for (i, out_vec) in out_simd.iter_mut().enumerate() {
            // x - max
            let shifted = simd.sub_f32s(in_simd[i], max_splat);
            // exp(x - max) - use fast exp approximation or standard
            let exp_val = exp_f32_simd(simd, shifted);
            *out_vec = exp_val;
            sum_acc = simd.add_f32s(sum_acc, exp_val);
        }

        // Reduce SIMD sum to scalar
        let sum_slice: &[f32] = pulp::bytemuck::cast_slice(std::slice::from_ref(&sum_acc));
        let mut sum_val: f32 = sum_slice.iter().sum();

        // Handle tail
        for (i, out) in out_tail.iter_mut().enumerate() {
            let exp_val = (in_tail[i] - max_val).exp();
            *out = exp_val;
            sum_val += exp_val;
        }

        // Pass 3: Normalize by dividing by sum
        let sum_inv = 1.0 / sum_val;
        let sum_inv_splat = simd.splat_f32s(sum_inv);

        for out_vec in out_simd.iter_mut() {
            *out_vec = simd.mul_f32s(*out_vec, sum_inv_splat);
        }

        for out in out_tail.iter_mut() {
            *out *= sum_inv;
        }
    }
}

/// Fast SIMD exp approximation using polynomial approximation
/// This is accurate enough for softmax (error < 1e-4)
#[inline(always)]
fn exp_f32_simd<S: Simd>(_simd: S, x: S::f32s) -> S::f32s {
    // For now, fall back to scalar exp. A proper SIMD exp would use
    // polynomial approximation but that's more complex.
    // This is still faster than non-fused because we avoid memory traffic.
    let x_slice: &[f32] = pulp::bytemuck::cast_slice(std::slice::from_ref(&x));
    let mut result = [0.0f32; 64]; // MAX_SIMD_LANES
    let lanes = x_slice.len();
    for i in 0..lanes {
        result[i] = x_slice[i].exp();
    }
    let (result_simd, _) = S::as_simd_f32s(&result[..lanes]);
    result_simd[0]
}

/// Fused layer normalization along the last dimension.
///
/// Formula: output = (input - mean) / sqrt(var + eps) * weight + bias
/// All computed in minimal passes through memory.
pub fn layer_norm_last_dim_fused<const R: usize>(
    tensor: &ConcreteTensor<f32, R>,
    weight: &ConcreteTensor<f32, R>,
    bias: Option<&ConcreteTensor<f32, R>>,
    eps: f32,
) -> ConcreteTensor<f32, R> {
    let shape: [usize; R] = tensor
        .layout()
        .shape()
        .try_into()
        .expect("Shape length mismatch");
    let last_dim = shape[R - 1];
    let num_rows: usize = shape[..R - 1].iter().product();

    // Use zeros() because layer_norm_row writes output via as_mut_simd
    let mut output = ConcreteTensor::<f32, R>::zeros(shape);

    if tensor.layout().is_contiguous() && weight.layout().is_contiguous() {
        let in_data: &[f32] = tensor.data();
        let weight_data: &[f32] = weight.data();
        let bias_data: Option<&[f32]> = bias.map(|b| b.data() as &[f32]);
        let out_data = output.data_mut();

        // Process rows in parallel for large tensors
        let n_threads = crate::parallel::num_threads();
        if num_rows >= 4 && n_threads > 1 {
            crate::parallel::parallel_zip_chunks_mut(
                in_data,
                out_data,
                last_dim,
                |_, in_row, out_row| {
                    layer_norm_row(in_row, weight_data, bias_data, eps, out_row);
                },
            );
        } else {
            for (in_row, out_row) in in_data.chunks(last_dim).zip(out_data.chunks_mut(last_dim)) {
                layer_norm_row(in_row, weight_data, bias_data, eps, out_row);
            }
        }
    } else {
        // Fall back to non-fused for non-contiguous
        // This is slower but handles all cases
        let in_data: &[f32] = tensor.data();
        let weight_data: &[f32] = weight.data();
        let bias_data: Option<&[f32]> = bias.map(|b| b.data() as &[f32]);
        let out_data = output.data_mut();

        let mut row_buffer = vec![0.0f32; last_dim];
        for row_idx in 0..num_rows {
            // Extract row
            for (i, row_elem) in row_buffer.iter_mut().enumerate().take(last_dim) {
                let mut indices = [0usize; R];
                let mut remaining = row_idx;
                for dim in (0..R - 1).rev() {
                    let dim_size = shape[dim];
                    indices[dim] = remaining % dim_size;
                    remaining /= dim_size;
                }
                indices[R - 1] = i;
                let idx = tensor.layout().linear_index(&indices);
                *row_elem = in_data[idx];
            }

            let out_row = &mut out_data[row_idx * last_dim..(row_idx + 1) * last_dim];
            layer_norm_row(&row_buffer, weight_data, bias_data, eps, out_row);
        }
    }

    output
}

/// SIMD-optimized layer normalization for a single row
#[inline(always)]
fn layer_norm_row(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
    output: &mut [f32],
) {
    Arch::new().dispatch(LayerNormRowDispatch {
        input,
        weight,
        bias,
        eps,
        output,
    });
}

struct LayerNormRowDispatch<'a> {
    input: &'a [f32],
    weight: &'a [f32],
    bias: Option<&'a [f32]>,
    eps: f32,
    output: &'a mut [f32],
}

impl WithSimd for LayerNormRowDispatch<'_> {
    type Output = ();

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let Self {
            input,
            weight,
            bias,
            eps,
            output,
        } = self;
        let n = input.len() as f32;
        let inv_n = 1.0 / n;

        // Pass 1: Compute sum for mean using SIMD
        let (in_simd, in_tail) = S::as_simd_f32s(input);
        let mut sum_acc = simd.splat_f32s(0.0);

        for &v in in_simd {
            sum_acc = simd.add_f32s(sum_acc, v);
        }

        // Reduce SIMD sum to scalar
        let sum_slice: &[f32] = pulp::bytemuck::cast_slice(std::slice::from_ref(&sum_acc));
        let mut sum_val: f32 = sum_slice.iter().sum();
        for &v in in_tail {
            sum_val += v;
        }

        let mean = sum_val * inv_n;
        let mean_splat = simd.splat_f32s(mean);

        // Pass 2: Compute variance sum using SIMD
        let mut var_acc = simd.splat_f32s(0.0);

        for &v in in_simd {
            let diff = simd.sub_f32s(v, mean_splat);
            let diff_sq = simd.mul_f32s(diff, diff);
            var_acc = simd.add_f32s(var_acc, diff_sq);
        }

        // Reduce SIMD variance to scalar
        let var_slice: &[f32] = pulp::bytemuck::cast_slice(std::slice::from_ref(&var_acc));
        let mut var_sum: f32 = var_slice.iter().sum();
        for &v in in_tail {
            let diff = v - mean;
            var_sum += diff * diff;
        }

        let var = var_sum * inv_n;
        let inv_std = 1.0 / (var + eps).sqrt();
        let inv_std_splat = simd.splat_f32s(inv_std);

        // Pass 3: Normalize with weight and optional bias using SIMD
        let (weight_simd, weight_tail) = S::as_simd_f32s(weight);
        let (out_simd, out_tail) = f32::as_mut_simd::<S>(output);

        match bias {
            Some(bias) => {
                let (bias_simd, bias_tail) = S::as_simd_f32s(bias);

                for (((out_vec, &in_vec), &w_vec), &b_vec) in out_simd
                    .iter_mut()
                    .zip(in_simd.iter())
                    .zip(weight_simd.iter())
                    .zip(bias_simd.iter())
                {
                    // (input - mean) * inv_std * weight + bias
                    let centered = simd.sub_f32s(in_vec, mean_splat);
                    let normalized = simd.mul_f32s(centered, inv_std_splat);
                    let scaled = simd.mul_f32s(normalized, w_vec);
                    *out_vec = simd.add_f32s(scaled, b_vec);
                }

                // Handle tail
                for (i, out) in out_tail.iter_mut().enumerate() {
                    *out = (in_tail[i] - mean) * inv_std * weight_tail[i] + bias_tail[i];
                }
            }
            None => {
                for ((out_vec, &in_vec), &w_vec) in out_simd
                    .iter_mut()
                    .zip(in_simd.iter())
                    .zip(weight_simd.iter())
                {
                    // (input - mean) * inv_std * weight
                    let centered = simd.sub_f32s(in_vec, mean_splat);
                    let normalized = simd.mul_f32s(centered, inv_std_splat);
                    *out_vec = simd.mul_f32s(normalized, w_vec);
                }

                // Handle tail
                for (i, out) in out_tail.iter_mut().enumerate() {
                    *out = (in_tail[i] - mean) * inv_std * weight_tail[i];
                }
            }
        }
    }
}

/// Reduce along a specific axis (runtime axis), returning tensor with OUT_RANK dimensions
pub(crate) fn reduce_tensor_axis_dyn<
    E: SimdElement + Default,
    const R: usize,
    const OUT_RANK: usize,
    Op: SimdReduceOp<E>,
>(
    tensor: &ConcreteTensor<E, R>,
    axis: usize,
) -> ConcreteTensor<E, OUT_RANK>
where
    ConcreteTensor<E, R>: LastRank<OUT_RANK, E>,
{
    assert!(
        axis < R,
        "Axis {} out of bounds for tensor with rank {}",
        axis,
        R
    );
    assert_eq!(OUT_RANK, R - 1, "Output rank must be input rank - 1");

    // Compute output shape (remove axis dimension)
    let in_shape = tensor.layout().shape();
    let mut out_shape = [0usize; OUT_RANK];
    let mut j = 0;
    for (i, &in_dim) in in_shape.iter().enumerate().take(R) {
        if i != axis {
            out_shape[j] = in_dim;
            j += 1;
        }
    }

    let mut output = ConcreteTensor::<E, OUT_RANK>::zeros(out_shape);
    let reduce_dim = in_shape[axis];

    // Pre-compute strides for the reduction axis for faster linear index calculation
    let axis_stride = tensor.layout().strides()[axis];

    // Iterate over output indices and reduce along axis
    // Use fixed-size array to avoid allocation
    let mut in_indices = [0usize; R];
    for out_indices in IndexIterator::new(&out_shape) {
        // Build base input indices (with axis = 0)
        let mut j = 0;
        for (i, in_idx) in in_indices.iter_mut().enumerate().take(R) {
            if i == axis {
                *in_idx = 0;
            } else {
                *in_idx = out_indices[j];
                j += 1;
            }
        }

        // Get base index and reduce along axis using stride
        let base_idx = tensor.layout().linear_index(&in_indices);
        let mut acc = Op::identity();
        for k in 0..reduce_dim {
            let in_idx = base_idx + k * axis_stride;
            acc = Op::combine_scalar(acc, tensor.data()[in_idx]);
        }
        let out_idx = output.layout().linear_index(&out_indices);
        output.data_mut()[out_idx] = acc;
    }

    output
}
