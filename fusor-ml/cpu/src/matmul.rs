//! Matrix multiplication operations

use std::ops::{Add as StdAdd, Mul as StdMul};

use crate::{ConcreteTensor, ResolvedTensor, SimdElement};

/// Trait for dispatching matmul to the appropriate implementation
pub trait MatmulImpl:
    SimdElement + Default + StdAdd<Output = Self> + StdMul<Output = Self>
{
    fn matmul_contiguous(
        lhs: &[Self],
        rhs: &[Self],
        out: &mut [Self],
        m: usize,
        k: usize,
        n: usize,
    );
}

/// Parameters for matrix multiplication
struct MatmulParams<'a, T> {
    lhs: &'a [T],
    rhs: &'a [T],
    out: &'a mut [T],
    m: usize,
    k: usize,
    n: usize,
    zero: T,
    one: T,
}

/// Optimized matmul for contiguous tensors using gemm crate
#[inline(always)]
fn matmul_contiguous_gemm<T: 'static>(params: MatmulParams<'_, T>) {
    let MatmulParams {
        lhs,
        rhs,
        out,
        m,
        k,
        n,
        zero,
        one,
    } = params;
    // gemm computes: dst := alpha×dst + beta×lhs×rhs
    // We want: out = lhs × rhs
    // So: read_dst = false (ignore existing dst), beta = 1.0
    // Note: gemm expects (column_stride, row_stride) order
    // For row-major: col_stride = 1, row_stride = num_cols
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            out.as_mut_ptr(),
            1,          // dst_cs: col stride (row-major = 1)
            n as isize, // dst_rs: row stride (row-major = num_cols)
            false,      // read_dst: false = overwrite, don't accumulate
            lhs.as_ptr(),
            1,          // lhs_cs: col stride
            k as isize, // lhs_rs: row stride (num cols of lhs = k)
            rhs.as_ptr(),
            1,                           // rhs_cs: col stride
            n as isize,                  // rhs_rs: row stride (num cols of rhs = n)
            zero,                        // alpha (ignored when read_dst = false)
            one,                         // beta
            false,                       // conj_dst
            false,                       // conj_lhs
            false,                       // conj_rhs
            gemm::Parallelism::Rayon(0), // Use all available threads
        );
    }
}

/// Naive matmul implementation for types without gemm support
#[inline]
fn matmul_naive<T>(lhs: &[T], rhs: &[T], out: &mut [T], m: usize, k: usize, n: usize)
where
    T: Copy + Default + StdAdd<Output = T> + StdMul<Output = T>,
{
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::default();
            for l in 0..k {
                sum = sum + lhs[i * k + l] * rhs[l * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}

impl MatmulImpl for f32 {
    #[inline(always)]
    fn matmul_contiguous(
        lhs: &[Self],
        rhs: &[Self],
        out: &mut [Self],
        m: usize,
        k: usize,
        n: usize,
    ) {
        matmul_contiguous_gemm(MatmulParams {
            lhs,
            rhs,
            out,
            m,
            k,
            n,
            zero: 0.0,
            one: 1.0,
        });
    }
}

impl MatmulImpl for f64 {
    #[inline(always)]
    fn matmul_contiguous(
        lhs: &[Self],
        rhs: &[Self],
        out: &mut [Self],
        m: usize,
        k: usize,
        n: usize,
    ) {
        matmul_contiguous_gemm(MatmulParams {
            lhs,
            rhs,
            out,
            m,
            k,
            n,
            zero: 0.0,
            one: 1.0,
        });
    }
}

// Fallback implementation for other types
macro_rules! impl_matmul_naive {
    ($($t:ty),*) => {
        $(
            impl MatmulImpl for $t {
                #[inline]
                fn matmul_contiguous(lhs: &[Self], rhs: &[Self], out: &mut [Self], m: usize, k: usize, n: usize) {
                    matmul_naive(lhs, rhs, out, m, k, n);
                }
            }
        )*
    };
}

impl_matmul_naive!(i8, i16, i32, i64, u8, u16, u32, u64);

/// Extract a batch matrix from a non-contiguous tensor into a contiguous buffer
#[inline]
fn extract_batch_matrix<T: SimdElement + Copy, const R: usize>(
    tensor: &ConcreteTensor<T, R>,
    batch_indices: &[usize],
    rows: usize,
    cols: usize,
    buffer: &mut [T],
) {
    let layout = tensor.layout();
    let data = tensor.data();

    // Pre-compute the base offset from batch indices
    let mut base_idx = [0usize; R];
    for (idx, &b) in batch_indices.iter().enumerate() {
        base_idx[idx] = b;
    }

    // Check if the last two dimensions are contiguous (common case for transposed matrices)
    let strides = layout.strides();
    let last_two_contiguous = R >= 2 && strides[R - 1] == 1 && strides[R - 2] == cols;

    if last_two_contiguous {
        // Fast path: copy entire rows at once
        base_idx[R - 2] = 0;
        base_idx[R - 1] = 0;
        let start_offset = layout.linear_index(&base_idx);
        let src = &data[start_offset..start_offset + rows * cols];
        buffer[..rows * cols].copy_from_slice(src);
    } else {
        // General path: extract element by element, but optimize inner loop
        for i in 0..rows {
            base_idx[R - 2] = i;
            for j in 0..cols {
                base_idx[R - 1] = j;
                let idx = layout.linear_index(&base_idx);
                buffer[i * cols + j] = data[idx];
            }
        }
    }
}

/// Generic batched matrix multiplication for N-dimensional tensors (N >= 2)
/// Shape: [...batch_dims, M, K] @ [...batch_dims, K, N] -> [...batch_dims, M, N]
/// All batch dimensions must match between lhs and rhs.
pub fn batched_matmul<T: SimdElement + MatmulImpl, const R: usize>(
    lhs: &ConcreteTensor<T, R>,
    rhs: &ConcreteTensor<T, R>,
) -> ConcreteTensor<T, R> {
    const {
        assert!(
            R >= 2,
            "Matrix multiplication requires at least 2 dimensions"
        )
    };

    let lhs_shape = lhs.layout().shape();
    let rhs_shape = rhs.layout().shape();

    // Last two dimensions are matrix dimensions
    let m = lhs_shape[R - 2];
    let k = lhs_shape[R - 1];
    let k2 = rhs_shape[R - 2];
    let n = rhs_shape[R - 1];

    assert_eq!(
        k, k2,
        "Matrix dimension mismatch: lhs columns ({}) != rhs rows ({})",
        k, k2
    );

    // Check batch dimensions match
    for i in 0..(R - 2) {
        assert_eq!(
            lhs_shape[i], rhs_shape[i],
            "Batch dimension {} mismatch: {} != {}",
            i, lhs_shape[i], rhs_shape[i]
        );
    }

    // Compute output shape
    let mut out_shape: [usize; R] = lhs_shape.try_into().expect("Shape length mismatch");
    out_shape[R - 1] = n;

    let mut output = ConcreteTensor::<T, R>::zeros(out_shape);

    // Compute total batch size (product of all batch dimensions)
    let batch_size: usize = lhs_shape[..R - 2].iter().product();

    let lhs_contiguous = lhs.layout().is_contiguous();
    let rhs_contiguous = rhs.layout().is_contiguous();
    let lhs_matrix_size = m * k;
    let rhs_matrix_size = k * n;
    let out_matrix_size = m * n;

    if lhs_contiguous && rhs_contiguous {
        // Fast path: iterate over flattened batch dimension
        for b in 0..batch_size {
            let lhs_offset = b * lhs_matrix_size;
            let rhs_offset = b * rhs_matrix_size;
            let out_offset = b * out_matrix_size;

            T::matmul_contiguous(
                &lhs.data()[lhs_offset..lhs_offset + lhs_matrix_size],
                &rhs.data()[rhs_offset..rhs_offset + rhs_matrix_size],
                &mut output.data_mut()[out_offset..out_offset + out_matrix_size],
                m,
                k,
                n,
            );
        }
    } else {
        // Optimized path for non-contiguous tensors:
        // Extract each batch matrix to contiguous memory, then use optimized matmul
        let batch_dims = &lhs_shape[..R - 2];
        let mut batch_indices = vec![0usize; R - 2];

        // Reusable buffers for non-contiguous extraction
        let mut lhs_buffer = if !lhs_contiguous {
            vec![T::default(); lhs_matrix_size]
        } else {
            Vec::new()
        };
        let mut rhs_buffer = if !rhs_contiguous {
            vec![T::default(); rhs_matrix_size]
        } else {
            Vec::new()
        };

        for b in 0..batch_size {
            // Get contiguous data for this batch
            let lhs_slice: &[T] = if lhs_contiguous {
                let offset = b * lhs_matrix_size;
                &lhs.data()[offset..offset + lhs_matrix_size]
            } else {
                // Extract non-contiguous lhs to buffer
                extract_batch_matrix(lhs, &batch_indices, m, k, &mut lhs_buffer);
                &lhs_buffer
            };

            let rhs_slice: &[T] = if rhs_contiguous {
                let offset = b * rhs_matrix_size;
                &rhs.data()[offset..offset + rhs_matrix_size]
            } else {
                // Extract non-contiguous rhs to buffer
                extract_batch_matrix(rhs, &batch_indices, k, n, &mut rhs_buffer);
                &rhs_buffer
            };

            let out_offset = b * out_matrix_size;
            T::matmul_contiguous(
                lhs_slice,
                rhs_slice,
                &mut output.data_mut()[out_offset..out_offset + out_matrix_size],
                m,
                k,
                n,
            );

            // Increment batch indices
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

/// Matrix multiplication for N-dimensional tensors (N >= 2)
/// For 2D: [M, K] @ [K, N] -> [M, N]
/// For ND: [...batch, M, K] @ [...batch, K, N] -> [...batch, M, N]
impl<T: SimdElement + MatmulImpl, const R: usize> ConcreteTensor<T, R> {
    /// Matrix multiplication (batched for rank > 2)
    /// Panics if R < 2
    #[inline]
    pub fn matmul_ref(&self, rhs: &Self) -> Self {
        batched_matmul(self, rhs)
    }
}
