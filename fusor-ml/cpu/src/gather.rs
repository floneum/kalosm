//! SIMD gather operations for loading non-contiguous tensor elements
//!
//! This module provides efficient gather operations that use hardware SIMD
//! gather instructions (AVX2, AVX-512) when available, with fallback to
//! scalar loading on other architectures.

use pulp::Simd;

use crate::MAX_SIMD_LANES;
#[allow(unused_imports)]
use crate::SimdElement;

// Architecture-specific gather implementations
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86_gather {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// Check if AVX2 is available at runtime
    #[inline]
    pub fn has_avx2() -> bool {
        #[cfg(target_feature = "avx2")]
        {
            true
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            is_x86_feature_detected!("avx2")
        }
    }

    /// AVX2 gather for f32 (8 lanes)
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn gather_f32_avx2(slice: &[f32], indices: &[usize]) -> [f32; 8] {
        unsafe {
            let base_ptr = slice.as_ptr();
            let idx_i32: [i32; 8] = [
                indices[0] as i32,
                indices[1] as i32,
                indices[2] as i32,
                indices[3] as i32,
                indices[4] as i32,
                indices[5] as i32,
                indices[6] as i32,
                indices[7] as i32,
            ];
            let idx_vec = _mm256_loadu_si256(idx_i32.as_ptr() as *const __m256i);
            let gathered = _mm256_i32gather_ps::<4>(base_ptr, idx_vec);

            let mut result = [0.0f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), gathered);
            result
        }
    }

    /// AVX2 gather for f64 (4 lanes)
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn gather_f64_avx2(slice: &[f64], indices: &[usize]) -> [f64; 4] {
        unsafe {
            let base_ptr = slice.as_ptr();
            let idx_i32: [i32; 4] = [
                indices[0] as i32,
                indices[1] as i32,
                indices[2] as i32,
                indices[3] as i32,
            ];
            let idx_vec = _mm_loadu_si128(idx_i32.as_ptr() as *const __m128i);
            let gathered = _mm256_i32gather_pd::<8>(base_ptr, idx_vec);

            let mut result = [0.0f64; 4];
            _mm256_storeu_pd(result.as_mut_ptr(), gathered);
            result
        }
    }

    /// AVX2 gather for i32 (8 lanes)
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn gather_i32_avx2(slice: &[i32], indices: &[usize]) -> [i32; 8] {
        unsafe {
            let base_ptr = slice.as_ptr();
            let idx_i32: [i32; 8] = [
                indices[0] as i32,
                indices[1] as i32,
                indices[2] as i32,
                indices[3] as i32,
                indices[4] as i32,
                indices[5] as i32,
                indices[6] as i32,
                indices[7] as i32,
            ];
            let idx_vec = _mm256_loadu_si256(idx_i32.as_ptr() as *const __m256i);
            let gathered = _mm256_i32gather_epi32::<4>(base_ptr, idx_vec);

            let mut result = [0i32; 8];
            _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, gathered);
            result
        }
    }

    /// AVX2 gather for i64 (4 lanes)
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn gather_i64_avx2(slice: &[i64], indices: &[usize]) -> [i64; 4] {
        unsafe {
            let base_ptr = slice.as_ptr();
            let idx_i32: [i32; 4] = [
                indices[0] as i32,
                indices[1] as i32,
                indices[2] as i32,
                indices[3] as i32,
            ];
            let idx_vec = _mm_loadu_si128(idx_i32.as_ptr() as *const __m128i);
            let gathered = _mm256_i32gather_epi64::<8>(base_ptr, idx_vec);

            let mut result = [0i64; 4];
            _mm256_storeu_si256(result.as_mut_ptr() as *mut __m256i, gathered);
            result
        }
    }
}

/// Generic gather implementation that dispatches to optimized versions when available.
///
/// # Safety
/// All indices must be valid indices into the slice.
#[inline(always)]
pub unsafe fn gather_impl<T, S: Simd>(
    _simd: S,
    slice: &[T],
    indices: &[usize],
    lane_count: usize,
) -> T::Simd<S>
where
    T: crate::SimdElement,
{
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        // Use type_id to dispatch to specialized implementations
        use std::any::TypeId;
        let type_id = TypeId::of::<T>();

        if x86_gather::has_avx2() {
            // SAFETY: Caller guarantees type T matches and indices are valid.
            // The transmutes are safe because we checked the TypeId matches.
            unsafe {
                if type_id == TypeId::of::<f32>() && lane_count == 8 {
                    let slice_f32: &[f32] = std::mem::transmute(slice);
                    let result = x86_gather::gather_f32_avx2(slice_f32, indices);
                    let (simd_vec, _) = f32::as_simd::<S>(&result);
                    return std::mem::transmute_copy(&simd_vec[0]);
                }
                if type_id == TypeId::of::<f64>() && lane_count == 4 {
                    let slice_f64: &[f64] = std::mem::transmute(slice);
                    let result = x86_gather::gather_f64_avx2(slice_f64, indices);
                    let (simd_vec, _) = f64::as_simd::<S>(&result);
                    return std::mem::transmute_copy(&simd_vec[0]);
                }
                if type_id == TypeId::of::<i32>() && lane_count == 8 {
                    let slice_i32: &[i32] = std::mem::transmute(slice);
                    let result = x86_gather::gather_i32_avx2(slice_i32, indices);
                    let (simd_vec, _) = i32::as_simd::<S>(&result);
                    return std::mem::transmute_copy(&simd_vec[0]);
                }
                if type_id == TypeId::of::<i64>() && lane_count == 4 {
                    let slice_i64: &[i64] = std::mem::transmute(slice);
                    let result = x86_gather::gather_i64_avx2(slice_i64, indices);
                    let (simd_vec, _) = i64::as_simd::<S>(&result);
                    return std::mem::transmute_copy(&simd_vec[0]);
                }
            }
        }
    }

    // Fallback to scalar gather
    let mut temp = [T::default(); MAX_SIMD_LANES];
    for i in 0..lane_count {
        temp[i] = slice[indices[i]];
    }
    let (simd_vec, _) = T::as_simd::<S>(&temp[..lane_count]);
    simd_vec[0]
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::SimdElement;
    use pulp::Arch;

    #[test]
    fn test_gather_f32() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();

        // Gather every other element
        let mut indices = [0usize; MAX_SIMD_LANES];
        for i in 0..MAX_SIMD_LANES {
            indices[i] = i * 2;
        }

        struct GatherTest<'a> {
            data: &'a [f32],
            indices: &'a [usize],
            result: &'a mut [f32],
            lane_count: &'a mut usize,
        }

        impl pulp::WithSimd for GatherTest<'_> {
            type Output = ();

            fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                let lane_count = std::mem::size_of::<<f32 as SimdElement>::Simd<S>>()
                    / std::mem::size_of::<f32>();
                *self.lane_count = lane_count;

                unsafe {
                    let gathered = f32::gather_unchecked(simd, self.data, self.indices, lane_count);

                    // Store result
                    let (out_simd, _) = f32::as_mut_simd::<S>(self.result);
                    out_simd[0] = gathered;
                }
            }
        }

        let mut result = vec![0.0f32; MAX_SIMD_LANES];
        let mut lane_count = 0;
        Arch::new().dispatch(GatherTest {
            data: &data,
            indices: &indices,
            result: &mut result,
            lane_count: &mut lane_count,
        });

        // Verify results for the actual number of SIMD lanes
        for i in 0..lane_count {
            assert_eq!(result[i], (i * 2) as f32, "Mismatch at index {}", i);
        }
    }

    #[test]
    fn test_gather_strided() {
        let data: Vec<f32> = (0..100).map(|i| i as f32).collect();

        // Gather with stride 3, starting at offset 5
        let mut indices = [0usize; MAX_SIMD_LANES];
        for i in 0..MAX_SIMD_LANES {
            indices[i] = 5 + i * 3;
        }

        struct GatherTest<'a> {
            data: &'a [f32],
            indices: &'a [usize],
            result: &'a mut [f32],
            lane_count: &'a mut usize,
        }

        impl pulp::WithSimd for GatherTest<'_> {
            type Output = ();

            fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
                let lane_count = std::mem::size_of::<<f32 as SimdElement>::Simd<S>>()
                    / std::mem::size_of::<f32>();
                *self.lane_count = lane_count;

                unsafe {
                    let gathered = f32::gather_unchecked(simd, self.data, self.indices, lane_count);

                    let (out_simd, _) = f32::as_mut_simd::<S>(self.result);
                    out_simd[0] = gathered;
                }
            }
        }

        let mut result = vec![0.0f32; MAX_SIMD_LANES];
        let mut lane_count = 0;
        Arch::new().dispatch(GatherTest {
            data: &data,
            indices: &indices,
            result: &mut result,
            lane_count: &mut lane_count,
        });

        // Verify results for the actual number of SIMD lanes
        for i in 0..lane_count {
            let expected = (5 + i * 3) as f32;
            assert_eq!(result[i], expected, "Mismatch at index {}", i);
        }
    }
}
