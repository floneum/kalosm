//! Shape-selection policy for kernel dispatch. Pure functions and shape keys.
//!
//! Const-generic monomorphization stays in the dispatch macros that consume
//! these shape keys (see `kernels/qgemv.rs`). The
//! compiler must see the const literals at the dispatch site to monomorphize,
//! so this module returns a small shape key, and a `match` in the builder picks
//! the literal generic
//! arguments.
//!
//! Shape selection is automatic and deterministic. Tests exercise the pure
//! policy functions directly; production has no environment-forced geometry.

use fusor_tile_ir::{GgmlQuantFormat, SubgroupToken};

// ===== qgemv shapes (Q4K and Q6K ggml paths) =====

pub(crate) const fn qgemv_subgroups_per_workgroup(format: GgmlQuantFormat) -> u32 {
    match format {
        GgmlQuantFormat::Q4K
        | GgmlQuantFormat::Q4KNative
        | GgmlQuantFormat::Q6K
        | GgmlQuantFormat::Q6KNative
        | GgmlQuantFormat::Q8_0
        | GgmlQuantFormat::Q8_0Native
        | GgmlQuantFormat::Q8_1 => 4,
        _ => 2,
    }
}

/// Shape-aware subgroup count used by the qgemv dispatch policy.
const fn qgemv_subgroups_per_workgroup_for_shape(
    format: GgmlQuantFormat,
    rows: u32,
    _cols: u32,
) -> u32 {
    match format {
        format if format.is_q6k_family() && rows > 4096 => 8,
        _ => qgemv_subgroups_per_workgroup(format),
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct QgemvShape {
    pub subgroups: u32,
    pub cols_per_subgroup: u32,
}

impl QgemvShape {
    const fn new(subgroups: u32, cols_per_subgroup: u32) -> Self {
        Self {
            subgroups,
            cols_per_subgroup,
        }
    }

    pub const fn cols_per_workgroup(self) -> u32 {
        self.subgroups * self.cols_per_subgroup
    }
}

/// The workgroup geometry the qgemv builder will emit for `(format, rows,
/// output_cols)` — the single source for both the kernel body and the
/// dispatch grid. `rows` is the contraction depth (K) and `output_cols` the
/// epilogue-adjusted output width (N). Callers computing a dispatch must use
/// this (never a re-derived approximation): the launched grid and the
/// kernel's internal `qgemv_grid` agree by construction only when both come
/// from here.
pub fn qgemv_selected_shape(format: GgmlQuantFormat, rows: u32, output_cols: u32) -> QgemvShape {
    match format {
        GgmlQuantFormat::Q8_0 | GgmlQuantFormat::Q8_0Native => {
            if output_cols >= 8192 {
                QgemvShape::new(4, 8)
            } else {
                QgemvShape::new(4, 4)
            }
        }
        GgmlQuantFormat::Q8_1 => QgemvShape::new(4, 4),
        GgmlQuantFormat::Q4K | GgmlQuantFormat::Q4KNative => {
            if rows <= 4096 && (4096..8192).contains(&output_cols) {
                q4k_default_mid(rows, output_cols)
            } else if rows <= 4096 && output_cols <= 4096 {
                QgemvShape::new(8, 4)
            } else if rows <= 4096 && output_cols >= 8192 {
                q4k_default_large(rows, output_cols)
            } else if rows > 4096 && output_cols <= 4096 {
                q4k_default_tall(rows, output_cols)
            } else if qgemv_subgroups_per_workgroup_for_shape(format, rows, output_cols) == 8 {
                QgemvShape::new(8, 8)
            } else {
                QgemvShape::new(4, 8)
            }
        }
        GgmlQuantFormat::Q5_0 | GgmlQuantFormat::Q5_0Native => QgemvShape::new(2, 4),
        GgmlQuantFormat::Q4_0
        | GgmlQuantFormat::Q4_0Native
        | GgmlQuantFormat::Q4_1
        | GgmlQuantFormat::Q5_1
        | GgmlQuantFormat::Q2K => QgemvShape::new(2, 4),
        GgmlQuantFormat::Q3K | GgmlQuantFormat::Q8K => QgemvShape::new(2, 2),
        GgmlQuantFormat::Q5K | GgmlQuantFormat::Q5KNative => QgemvShape::new(2, 1),
        GgmlQuantFormat::Q6K | GgmlQuantFormat::Q6KNative => {
            if rows <= 4096 && output_cols >= 8192 {
                q6k_default_large(rows, output_cols)
            } else if rows > 4096 && output_cols <= 4096 {
                q6k_default_tall(rows, output_cols)
            } else if qgemv_subgroups_per_workgroup_for_shape(format, rows, output_cols) == 4 {
                QgemvShape::new(4, 4)
            } else {
                QgemvShape::new(8, 4)
            }
        }
    }
}

/// Subgroup-width range advertised by the target adapter for one generated
/// kernel. The generated shader reads the actual runtime subgroup size; the
/// max is only used for the fixed workgroup allocation passed to WGSL.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SubgroupConfig {
    token: SubgroupToken,
    min_size: u32,
    max_size: u32,
}

impl SubgroupConfig {
    pub const fn new(token: SubgroupToken, min_size: u32, max_size: u32) -> Self {
        assert!(min_size > 0, "subgroup min size must be non-zero");
        assert!(max_size >= min_size, "subgroup size range must be ordered");
        Self {
            token,
            min_size,
            max_size,
        }
    }

    pub const fn fixed(token: SubgroupToken, size: u32) -> Self {
        Self::new(token, size, size)
    }

    pub const fn token(self) -> SubgroupToken {
        self.token
    }

    pub const fn min_size(self) -> u32 {
        self.min_size
    }

    pub const fn max_size(self) -> u32 {
        self.max_size
    }

    pub const fn is_fixed(self) -> bool {
        self.min_size == self.max_size
    }

    pub const fn block_for_subgroups(self, subgroups: u32) -> u32 {
        subgroups * self.max_size
    }

    pub const fn supports_lanes_per_item(self, lanes: u32) -> bool {
        self.min_size >= lanes
            && self.min_size.is_multiple_of(lanes)
            && self.max_size.is_multiple_of(lanes)
    }
}

// ----- Q4K mid (rows<=4096, 4096<=cols<8192) -----

/// Default Q4K mid-shape: cols==5120 → 4x3, cols==6144 → 8x2, else 2x2.
pub(crate) const fn q4k_default_mid(_rows: u32, cols: u32) -> QgemvShape {
    if cols == 5120 {
        return QgemvShape::new(4, 3);
    }
    if cols == 6144 {
        return QgemvShape::new(8, 2);
    }
    QgemvShape::new(2, 2)
}

// ----- Q4K large (rows<=4096, cols>=8192) -----

/// Default Q4K large-shape: cols<=16_384 → 8x4, else 2x4.
pub(crate) const fn q4k_default_large(_rows: u32, cols: u32) -> QgemvShape {
    if cols <= 16_384 {
        QgemvShape::new(8, 4)
    } else {
        QgemvShape::new(2, 4)
    }
}

// ----- Q4K tall (rows>4096, cols<=4096) -----

/// Default Q4K tall-shape: 4x2.
pub(crate) const fn q4k_default_tall(_rows: u32, _cols: u32) -> QgemvShape {
    QgemvShape::new(4, 2)
}

// ----- Q6K large (rows<=4096, cols>=8192) -----

/// Default Q6K large-shape: cols<=16_384 → 2x2, else 2x4.
pub(crate) const fn q6k_default_large(_rows: u32, cols: u32) -> QgemvShape {
    if cols <= 16_384 {
        QgemvShape::new(2, 2)
    } else {
        QgemvShape::new(2, 4)
    }
}

// ----- Q6K tall (rows>4096, cols<=4096) -----

/// Default Q6K tall-shape: 2x2.
pub(crate) const fn q6k_default_tall(_rows: u32, _cols: u32) -> QgemvShape {
    QgemvShape::new(2, 2)
}

#[cfg(test)]
mod tests {
    //! Snapshot tests pinning the automatic `(format, rows, cols) → ShapeKey`
    //! mapping.
    use super::*;

    /// The selected shape is the single source for dispatch and kernel
    /// geometry; these cells pin it where the deleted core-side re-derivation
    /// used to disagree with the builder (over-dispatching masked workgroups
    /// and permanently missing the prebuilt-pipeline fast path).
    #[test]
    fn selected_shape_pins_previously_desynced_cells() {
        use GgmlQuantFormat as F;
        let cols = |f, k, n| qgemv_selected_shape(f, k, n).cols_per_workgroup();
        // Q4K large: the old dispatch ladder said 8; the builder emits 32.
        assert_eq!(cols(F::Q4K, 2048, 8192), 32);
        assert_eq!(cols(F::Q4K, 4096, 11008), 32);
        // Q4K mid: old ladder said 4 for every mid shape; the builder varies.
        assert_eq!(cols(F::Q4K, 4096, 4097), 4);
        assert_eq!(cols(F::Q4K, 4096, 5120), 12);
        assert_eq!(cols(F::Q4K, 4096, 6144), 16);
        // Q6K large <=16384: old ladder said 8; the builder emits 4.
        assert_eq!(cols(F::Q6K, 2048, 8192), 4);
        assert_eq!(cols(F::Q6K, 2048, 16385), 8);
        // Q8_0 wide: old ladder said 32; the builder emits 32 (4x8).
        assert_eq!(cols(F::Q8_0, 1024, 8192), 32);
        // SmolLM2 decode cells.
        assert_eq!(cols(F::Q4K, 576, 1536), 32);
        assert_eq!(cols(F::Q4K, 1536, 576), 32);
        assert_eq!(cols(F::Q6K, 576, 49152), 8);
    }

    #[test]
    fn q4k_mid_default_unchanged() {
        // Uses the inline `if b.cols == 5120 / 6144` branches from
        // qgemv_tile (kernels/qgemv.rs).
        assert_eq!(q4k_default_mid(4096, 4096), QgemvShape::new(2, 2));
        assert_eq!(q4k_default_mid(4096, 5120), QgemvShape::new(4, 3));
        assert_eq!(q4k_default_mid(4096, 6144), QgemvShape::new(8, 2));
        assert_eq!(q4k_default_mid(2048, 7000), QgemvShape::new(2, 2));
    }

    #[test]
    fn q4k_large_default_selected() {
        // Uses the mid-size Q4K branch from kernels/qgemv.rs.
        assert_eq!(q4k_default_large(4096, 8192), QgemvShape::new(8, 4));
        assert_eq!(q4k_default_large(4096, 16_384), QgemvShape::new(8, 4));
        assert_eq!(q4k_default_large(4096, 16_385), QgemvShape::new(2, 4));
        assert_eq!(q4k_default_large(4096, 32_768), QgemvShape::new(2, 4));
    }

    #[test]
    fn q4k_tall_default_unchanged() {
        // Constant 4x2 from kernels/qgemv.rs.
        assert_eq!(q4k_default_tall(8192, 4096), QgemvShape::new(4, 2));
        assert_eq!(q4k_default_tall(16_384, 2048), QgemvShape::new(4, 2));
    }

    #[test]
    fn q6k_large_default_unchanged() {
        // Uses the large/tall Q6K branches from kernels/qgemv.rs.
        assert_eq!(q6k_default_large(4096, 8192), QgemvShape::new(2, 2));
        assert_eq!(q6k_default_large(4096, 16_384), QgemvShape::new(2, 2));
        assert_eq!(q6k_default_large(4096, 16_385), QgemvShape::new(2, 4));
    }

    #[test]
    fn q6k_tall_default_unchanged() {
        // Constant 2x2 from kernels/qgemv.rs.
        assert_eq!(q6k_default_tall(8192, 4096), QgemvShape::new(2, 2));
    }
}
