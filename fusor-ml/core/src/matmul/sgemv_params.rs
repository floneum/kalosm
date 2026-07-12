//! Measured (m, k) bucket table for the vector-family fallback. This family
//! only runs where cooperative matrices are unavailable (no-subgroup devices
//! and the WebGPU baseline), so the cells are measurement, not derivation —
//! rederiving them requires benching those targets. Structural legality of
//! every cell is pinned by `fallback_family_params_are_legal_everywhere`.

use crate::sgemv::SgemvParams;

#[inline]
pub fn gemv_parameters(m: usize, _: usize, k: usize) -> SgemvParams {
    let m = m as f32;
    let k = k as f32;
    match (m as u32, k as u32) {
        (..=512, ..=512) => SgemvParams::new(16, 4, 16),
        (..=512, ..=1024) => SgemvParams::new(2, 4, 1),
        (..=512, ..=2048) => SgemvParams::new(8, 4, 2),
        (..=512, ..=4096) => SgemvParams::new(8, 4, 2),
        (..=512, _) => SgemvParams::new(8, 4, 2),

        (..=1024, ..=512) => SgemvParams::new(8, 4, 8),
        (..=1024, ..=1024) => SgemvParams::new(16, 4, 1),
        (..=1024, ..=2048) => SgemvParams::new(8, 4, 16),
        (..=1024, ..=4096) => SgemvParams::new(8, 4, 16),
        (..=1024, _) => SgemvParams::new(8, 4, 16),

        (..=2048, ..=512) => SgemvParams::new(16, 4, 1),
        (..=2048, ..=1024) => SgemvParams::new(8, 4, 32),
        (..=2048, ..=2048) => SgemvParams::new(8, 4, 2),
        (..=2048, ..=4096) => SgemvParams::new(8, 4, 8),
        (..=2048, _) => SgemvParams::new(32, 2, 8),

        (..=4096, ..=512) => SgemvParams::new(8, 2, 1),
        (..=4096, ..=1024) => SgemvParams::new(16, 4, 1),
        (..=4096, ..=2048) => SgemvParams::new(32, 2, 16),
        (..=4096, ..=4096) => SgemvParams::new(8, 4, 16),
        (..=4096, _) => SgemvParams::new(32, 2, 32),

        (_, _) => SgemvParams::new(8, 4, 16),
    }
}
