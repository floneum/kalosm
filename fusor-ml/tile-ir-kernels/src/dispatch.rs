//! Shape-selection policy for kernel dispatch. Pure functions and enums.
//!
//! Const-generic monomorphization stays in the dispatch macros that consume
//! these enums (see `kernels/qgemv.rs`). The
//! compiler must see the const literals at the dispatch site to monomorphize,
//! so this module never returns a runtime tile triple — it returns a
//! ShapeKey enum, and a `match` in the builder picks the literal generic
//! arguments.
//!
//! The mapping from environment variables and `(rows, cols)` heuristics to
//! ShapeKeys mirrors the table that previously lived inline in the builder
//! methods. Snapshot tests in this module pin the current behavior so that
//! the move from inline shape-tables to policy functions is observably a
//! no-op.
//!
//! Env vars (all preserved):
//!   - `FUSOR_Q4K_MID_TILE`   (rows<=4096, 4096<=cols<8192)
//!   - `FUSOR_Q4K_LARGE_TILE` (rows<=4096, cols>=8192)
//!   - `FUSOR_Q4K_TALL_TILE`  (rows>4096,  cols<=4096)
//!   - `FUSOR_Q6K_LARGE_TILE` (rows<=4096, cols>=8192)
//!   - `FUSOR_Q6K_TALL_TILE`  (rows>4096,  cols<=4096)

use fusor_tile_ir::GgmlQuantFormat;

// ===== qgemv shapes (Q4K and Q6K ggml paths) =====

/// Default qgemv output columns handled by one workgroup for `format`.
pub const fn qgemv_cols_per_workgroup(format: GgmlQuantFormat) -> u32 {
    qgemv_subgroups_per_workgroup(format) * qgemv_cols_per_subgroup(format)
}

/// Shape-aware qgemv output columns handled by one workgroup.
///
/// This includes the Q4K/Q6K GGML specializations whose column grouping
/// depends on both K (`rows`) and N (`cols`).
pub fn qgemv_cols_per_workgroup_for_shape(
    format: GgmlQuantFormat,
    rows: u32,
    cols: u32,
) -> u32 {
    if matches!(format, GgmlQuantFormat::Q4K) && rows <= 4096 && (4096..8192).contains(&cols) {
        return q4k_mid_override(q4k_default_mid(rows, cols)).cols_per_workgroup();
    }

    if matches!(format, GgmlQuantFormat::Q4K) && rows <= 4096 && cols >= 8192 {
        return q4k_large_override(q4k_default_large(rows, cols)).cols_per_workgroup();
    }

    if matches!(format, GgmlQuantFormat::Q4K) && rows > 4096 && cols <= 4096 {
        return q4k_tall_override(q4k_default_tall(rows, cols)).cols_per_workgroup();
    }

    if matches!(format, GgmlQuantFormat::Q6K) && rows <= 4096 && cols >= 8192 {
        return q6k_large_override(q6k_default_large(rows, cols)).cols_per_workgroup();
    }

    if matches!(format, GgmlQuantFormat::Q6K) && rows > 4096 && cols <= 4096 {
        return q6k_tall_override(q6k_default_tall(rows, cols)).cols_per_workgroup();
    }

    qgemv_subgroups_per_workgroup_for_shape(format, rows, cols) * qgemv_cols_per_subgroup(format)
}

pub(crate) const fn qgemv_cols_per_subgroup(format: GgmlQuantFormat) -> u32 {
    match format {
        GgmlQuantFormat::Q2K => 4,
        GgmlQuantFormat::Q4_0 | GgmlQuantFormat::Q4_1 | GgmlQuantFormat::Q5_1 => 4,
        GgmlQuantFormat::Q5_0 => 4,
        GgmlQuantFormat::Q3K | GgmlQuantFormat::Q8K => 2,
        GgmlQuantFormat::Q4K => 8,
        GgmlQuantFormat::Q6K => 4,
        GgmlQuantFormat::Q8_0 | GgmlQuantFormat::Q8_1 => 4,
        GgmlQuantFormat::Q5K => 1,
    }
}

pub(crate) const fn qgemv_subgroups_per_workgroup(format: GgmlQuantFormat) -> u32 {
    match format {
        GgmlQuantFormat::Q4K
        | GgmlQuantFormat::Q6K
        | GgmlQuantFormat::Q8_0
        | GgmlQuantFormat::Q8_1 => 4,
        _ => 2,
    }
}

/// Shape-aware subgroup count used by the qgemv dispatch policy.
pub const fn qgemv_subgroups_per_workgroup_for_shape(
    format: GgmlQuantFormat,
    rows: u32,
    _cols: u32,
) -> u32 {
    match format {
        GgmlQuantFormat::Q6K if rows > 4096 => 8,
        _ => qgemv_subgroups_per_workgroup(format),
    }
}

/// Tile shape for `qgemv_q4k_ggml::<SUBGROUPS, COLS_PER_SUBGROUP, BLOCK>`.
/// The `1x4_32` etc. tiles are only reachable through the Q4K large-tile
/// override list; the default selection never emits them.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum QgemvShapeQ4K {
    Ggml1x4_32,
    Ggml1x8_32,
    Ggml2x2_64,
    Ggml2x3_64,
    Ggml2x4_64,
    Ggml2x8_64,
    Ggml4x1_128,
    Ggml4x2_128,
    Ggml4x3_128,
    Ggml4x4_128,
    Ggml4x8_128,
    Ggml8x1_256,
    Ggml8x2_256,
    Ggml8x4_256,
}

impl QgemvShapeQ4K {
    const fn cols_per_workgroup(self) -> u32 {
        match self {
            Self::Ggml1x4_32 => 4,
            Self::Ggml1x8_32 => 8,
            Self::Ggml2x2_64 => 4,
            Self::Ggml2x3_64 => 6,
            Self::Ggml2x4_64 => 8,
            Self::Ggml2x8_64 => 16,
            Self::Ggml4x1_128 => 4,
            Self::Ggml4x2_128 => 8,
            Self::Ggml4x3_128 => 12,
            Self::Ggml4x4_128 => 16,
            Self::Ggml4x8_128 => 32,
            Self::Ggml8x1_256 => 8,
            Self::Ggml8x2_256 => 16,
            Self::Ggml8x4_256 => 32,
        }
    }
}

/// Tile shape for `qgemv_q6k_ggml::<SUBGROUPS, COLS_PER_SUBGROUP, BLOCK>`.
/// The Q6K override lists use the standard tile set (no 2x3/4x3, no
/// 1x_/4x1/8x1 entries).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum QgemvShapeQ6K {
    Ggml2x2_64,
    Ggml2x4_64,
    Ggml2x8_64,
    Ggml4x2_128,
    Ggml4x4_128,
    Ggml4x8_128,
    Ggml8x2_256,
    Ggml8x4_256,
}

impl QgemvShapeQ6K {
    const fn cols_per_workgroup(self) -> u32 {
        match self {
            Self::Ggml2x2_64 => 4,
            Self::Ggml2x4_64 => 8,
            Self::Ggml2x8_64 => 16,
            Self::Ggml4x2_128 => 8,
            Self::Ggml4x4_128 => 16,
            Self::Ggml4x8_128 => 32,
            Self::Ggml8x2_256 => 16,
            Self::Ggml8x4_256 => 32,
        }
    }
}

// ----- Q4K mid (rows<=4096, 4096<=cols<8192) -----

/// Default Q4K mid-shape: cols==5120 → 4x3, cols==6144 → 8x2, else 2x2.
pub(crate) const fn q4k_default_mid(_rows: u32, cols: u32) -> QgemvShapeQ4K {
    if cols == 5120 {
        return QgemvShapeQ4K::Ggml4x3_128;
    }
    if cols == 6144 {
        return QgemvShapeQ4K::Ggml8x2_256;
    }
    QgemvShapeQ4K::Ggml2x2_64
}

/// Apply `FUSOR_Q4K_MID_TILE` if set; otherwise return the default. The set
/// of accepted env values is exactly the inline `qgemv_ggml_env!` table that
/// used to live in `qgemv_tile`.
pub(crate) fn q4k_mid_override(default: QgemvShapeQ4K) -> QgemvShapeQ4K {
    match std::env::var("FUSOR_Q4K_MID_TILE").as_deref() {
        Ok("ggml_2x2") => QgemvShapeQ4K::Ggml2x2_64,
        Ok("ggml_2x3") => QgemvShapeQ4K::Ggml2x3_64,
        Ok("ggml_2x4") => QgemvShapeQ4K::Ggml2x4_64,
        Ok("ggml_2x8") => QgemvShapeQ4K::Ggml2x8_64,
        Ok("ggml_4x2") => QgemvShapeQ4K::Ggml4x2_128,
        Ok("ggml_4x3") => QgemvShapeQ4K::Ggml4x3_128,
        Ok("ggml_4x4") => QgemvShapeQ4K::Ggml4x4_128,
        Ok("ggml_4x8") => QgemvShapeQ4K::Ggml4x8_128,
        Ok("ggml_8x2") => QgemvShapeQ4K::Ggml8x2_256,
        Ok("ggml_8x4") => QgemvShapeQ4K::Ggml8x4_256,
        _ => default,
    }
}

// ----- Q4K large (rows<=4096, cols>=8192) -----

/// Default Q4K large-shape: cols<=16_384 → 8x4, else 2x4.
pub(crate) const fn q4k_default_large(_rows: u32, cols: u32) -> QgemvShapeQ4K {
    if cols <= 16_384 {
        QgemvShapeQ4K::Ggml8x4_256
    } else {
        QgemvShapeQ4K::Ggml2x4_64
    }
}

/// Apply `FUSOR_Q4K_LARGE_TILE` if set. Carries the same tile list as the
/// inline macro: adds 1x4/1x8/4x1/8x1 (no 2x3/4x3 entries).
pub(crate) fn q4k_large_override(default: QgemvShapeQ4K) -> QgemvShapeQ4K {
    match std::env::var("FUSOR_Q4K_LARGE_TILE").as_deref() {
        Ok("ggml_1x4") => QgemvShapeQ4K::Ggml1x4_32,
        Ok("ggml_1x8") => QgemvShapeQ4K::Ggml1x8_32,
        Ok("ggml_2x2") => QgemvShapeQ4K::Ggml2x2_64,
        Ok("ggml_2x4") => QgemvShapeQ4K::Ggml2x4_64,
        Ok("ggml_2x8") => QgemvShapeQ4K::Ggml2x8_64,
        Ok("ggml_4x1") => QgemvShapeQ4K::Ggml4x1_128,
        Ok("ggml_4x2") => QgemvShapeQ4K::Ggml4x2_128,
        Ok("ggml_4x4") => QgemvShapeQ4K::Ggml4x4_128,
        Ok("ggml_4x8") => QgemvShapeQ4K::Ggml4x8_128,
        Ok("ggml_8x1") => QgemvShapeQ4K::Ggml8x1_256,
        Ok("ggml_8x2") => QgemvShapeQ4K::Ggml8x2_256,
        Ok("ggml_8x4") => QgemvShapeQ4K::Ggml8x4_256,
        _ => default,
    }
}

// ----- Q4K tall (rows>4096, cols<=4096) -----

/// Default Q4K tall-shape: 4x2.
pub(crate) const fn q4k_default_tall(_rows: u32, _cols: u32) -> QgemvShapeQ4K {
    QgemvShapeQ4K::Ggml4x2_128
}

/// Apply `FUSOR_Q4K_TALL_TILE` if set. Standard 8-tile set.
pub(crate) fn q4k_tall_override(default: QgemvShapeQ4K) -> QgemvShapeQ4K {
    match std::env::var("FUSOR_Q4K_TALL_TILE").as_deref() {
        Ok("ggml_2x2") => QgemvShapeQ4K::Ggml2x2_64,
        Ok("ggml_2x4") => QgemvShapeQ4K::Ggml2x4_64,
        Ok("ggml_2x8") => QgemvShapeQ4K::Ggml2x8_64,
        Ok("ggml_4x2") => QgemvShapeQ4K::Ggml4x2_128,
        Ok("ggml_4x4") => QgemvShapeQ4K::Ggml4x4_128,
        Ok("ggml_4x8") => QgemvShapeQ4K::Ggml4x8_128,
        Ok("ggml_8x2") => QgemvShapeQ4K::Ggml8x2_256,
        Ok("ggml_8x4") => QgemvShapeQ4K::Ggml8x4_256,
        _ => default,
    }
}

// ----- Q6K large (rows<=4096, cols>=8192) -----

/// Default Q6K large-shape: cols<=16_384 → 2x2, else 2x4.
pub(crate) const fn q6k_default_large(_rows: u32, cols: u32) -> QgemvShapeQ6K {
    if cols <= 16_384 {
        QgemvShapeQ6K::Ggml2x2_64
    } else {
        QgemvShapeQ6K::Ggml2x4_64
    }
}

/// Apply `FUSOR_Q6K_LARGE_TILE` if set. Standard 8-tile set.
pub(crate) fn q6k_large_override(default: QgemvShapeQ6K) -> QgemvShapeQ6K {
    q6k_standard_override("FUSOR_Q6K_LARGE_TILE", default)
}

// ----- Q6K tall (rows>4096, cols<=4096) -----

/// Default Q6K tall-shape: 2x2.
pub(crate) const fn q6k_default_tall(_rows: u32, _cols: u32) -> QgemvShapeQ6K {
    QgemvShapeQ6K::Ggml2x2_64
}

/// Apply `FUSOR_Q6K_TALL_TILE` if set. Standard 8-tile set.
pub(crate) fn q6k_tall_override(default: QgemvShapeQ6K) -> QgemvShapeQ6K {
    q6k_standard_override("FUSOR_Q6K_TALL_TILE", default)
}

/// Q6K's "standard 8-tile set" override table. Both `q6k_large_override` and
/// `q6k_tall_override` accept the same set of env values; the only difference
/// is which env var name they read.
fn q6k_standard_override(var: &str, default: QgemvShapeQ6K) -> QgemvShapeQ6K {
    match std::env::var(var).as_deref() {
        Ok("ggml_2x2") => QgemvShapeQ6K::Ggml2x2_64,
        Ok("ggml_2x4") => QgemvShapeQ6K::Ggml2x4_64,
        Ok("ggml_2x8") => QgemvShapeQ6K::Ggml2x8_64,
        Ok("ggml_4x2") => QgemvShapeQ6K::Ggml4x2_128,
        Ok("ggml_4x4") => QgemvShapeQ6K::Ggml4x4_128,
        Ok("ggml_4x8") => QgemvShapeQ6K::Ggml4x8_128,
        Ok("ggml_8x2") => QgemvShapeQ6K::Ggml8x2_256,
        Ok("ggml_8x4") => QgemvShapeQ6K::Ggml8x4_256,
        _ => default,
    }
}

#[cfg(test)]
mod tests {
    //! Snapshot tests pinning the current `(format, rows, cols, env) →
    //! ShapeKey` mapping. These must continue to pass after the inline
    //! `qgemv_ggml_env!` invocations and `if b.cols == ...` heuristics in
    //! `kernels/qgemv.rs` are replaced with calls into this module.
    //!
    //! Env-var tests use a serial mutex because `std::env::set_var` is
    //! process-global. They also unset the variable on entry to avoid
    //! cross-test contamination from a developer's shell.
    use super::*;
    use std::sync::Mutex;
    use std::sync::OnceLock;
    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn with_env<R>(var: &str, value: Option<&str>, f: impl FnOnce() -> R) -> R {
        let _guard = env_lock().lock().unwrap_or_else(|e| e.into_inner());
        let prior = std::env::var(var).ok();
        // SAFETY: tests serialize on env_lock above; no other thread reads or
        // mutates this var while the closure runs.
        unsafe {
            match value {
                Some(v) => std::env::set_var(var, v),
                None => std::env::remove_var(var),
            }
        }
        let out = f();
        unsafe {
            match prior {
                Some(p) => std::env::set_var(var, p),
                None => std::env::remove_var(var),
            }
        }
        out
    }

    #[test]
    fn q4k_mid_default_unchanged() {
        // Uses the inline `if b.cols == 5120 / 6144` branches from
        // qgemv_tile (kernels/qgemv.rs).
        assert_eq!(q4k_default_mid(4096, 4096), QgemvShapeQ4K::Ggml2x2_64);
        assert_eq!(q4k_default_mid(4096, 5120), QgemvShapeQ4K::Ggml4x3_128);
        assert_eq!(q4k_default_mid(4096, 6144), QgemvShapeQ4K::Ggml8x2_256);
        assert_eq!(q4k_default_mid(2048, 7000), QgemvShapeQ4K::Ggml2x2_64);
    }

    #[test]
    fn q4k_large_default_selected() {
        // Uses the mid-size Q4K branch from kernels/qgemv.rs.
        assert_eq!(q4k_default_large(4096, 8192), QgemvShapeQ4K::Ggml8x4_256);
        assert_eq!(q4k_default_large(4096, 16_384), QgemvShapeQ4K::Ggml8x4_256);
        assert_eq!(q4k_default_large(4096, 16_385), QgemvShapeQ4K::Ggml2x4_64);
        assert_eq!(q4k_default_large(4096, 32_768), QgemvShapeQ4K::Ggml2x4_64);
    }

    #[test]
    fn q4k_tall_default_unchanged() {
        // Constant 4x2 from kernels/qgemv.rs.
        assert_eq!(q4k_default_tall(8192, 4096), QgemvShapeQ4K::Ggml4x2_128);
        assert_eq!(q4k_default_tall(16_384, 2048), QgemvShapeQ4K::Ggml4x2_128);
    }

    #[test]
    fn q6k_large_default_unchanged() {
        // Uses the large/tall Q6K branches from kernels/qgemv.rs.
        assert_eq!(q6k_default_large(4096, 8192), QgemvShapeQ6K::Ggml2x2_64);
        assert_eq!(q6k_default_large(4096, 16_384), QgemvShapeQ6K::Ggml2x2_64);
        assert_eq!(q6k_default_large(4096, 16_385), QgemvShapeQ6K::Ggml2x4_64);
    }

    #[test]
    fn q6k_tall_default_unchanged() {
        // Constant 2x2 from kernels/qgemv.rs.
        assert_eq!(q6k_default_tall(8192, 4096), QgemvShapeQ6K::Ggml2x2_64);
    }

    #[test]
    fn q4k_mid_override_table_unchanged() {
        with_env("FUSOR_Q4K_MID_TILE", None, || {
            assert_eq!(
                q4k_mid_override(QgemvShapeQ4K::Ggml2x2_64),
                QgemvShapeQ4K::Ggml2x2_64
            );
        });
        let cases = [
            ("ggml_2x2", QgemvShapeQ4K::Ggml2x2_64),
            ("ggml_2x3", QgemvShapeQ4K::Ggml2x3_64),
            ("ggml_2x4", QgemvShapeQ4K::Ggml2x4_64),
            ("ggml_2x8", QgemvShapeQ4K::Ggml2x8_64),
            ("ggml_4x2", QgemvShapeQ4K::Ggml4x2_128),
            ("ggml_4x3", QgemvShapeQ4K::Ggml4x3_128),
            ("ggml_4x4", QgemvShapeQ4K::Ggml4x4_128),
            ("ggml_4x8", QgemvShapeQ4K::Ggml4x8_128),
            ("ggml_8x2", QgemvShapeQ4K::Ggml8x2_256),
            ("ggml_8x4", QgemvShapeQ4K::Ggml8x4_256),
        ];
        for (val, expect) in cases {
            with_env("FUSOR_Q4K_MID_TILE", Some(val), || {
                assert_eq!(
                    q4k_mid_override(QgemvShapeQ4K::Ggml4x4_128),
                    expect,
                    "FUSOR_Q4K_MID_TILE={val}"
                );
            });
        }
        // Unrecognized value falls through to default.
        with_env("FUSOR_Q4K_MID_TILE", Some("nonsense"), || {
            assert_eq!(
                q4k_mid_override(QgemvShapeQ4K::Ggml4x4_128),
                QgemvShapeQ4K::Ggml4x4_128
            );
        });
    }

    #[test]
    fn q4k_large_override_table_unchanged() {
        let cases = [
            ("ggml_1x4", QgemvShapeQ4K::Ggml1x4_32),
            ("ggml_1x8", QgemvShapeQ4K::Ggml1x8_32),
            ("ggml_2x2", QgemvShapeQ4K::Ggml2x2_64),
            ("ggml_2x4", QgemvShapeQ4K::Ggml2x4_64),
            ("ggml_2x8", QgemvShapeQ4K::Ggml2x8_64),
            ("ggml_4x1", QgemvShapeQ4K::Ggml4x1_128),
            ("ggml_4x2", QgemvShapeQ4K::Ggml4x2_128),
            ("ggml_4x4", QgemvShapeQ4K::Ggml4x4_128),
            ("ggml_4x8", QgemvShapeQ4K::Ggml4x8_128),
            ("ggml_8x1", QgemvShapeQ4K::Ggml8x1_256),
            ("ggml_8x2", QgemvShapeQ4K::Ggml8x2_256),
            ("ggml_8x4", QgemvShapeQ4K::Ggml8x4_256),
        ];
        for (val, expect) in cases {
            with_env("FUSOR_Q4K_LARGE_TILE", Some(val), || {
                assert_eq!(
                    q4k_large_override(QgemvShapeQ4K::Ggml4x4_128),
                    expect,
                    "FUSOR_Q4K_LARGE_TILE={val}"
                );
            });
        }
    }

    #[test]
    fn q4k_tall_override_table_unchanged() {
        let cases = [
            ("ggml_2x2", QgemvShapeQ4K::Ggml2x2_64),
            ("ggml_2x4", QgemvShapeQ4K::Ggml2x4_64),
            ("ggml_2x8", QgemvShapeQ4K::Ggml2x8_64),
            ("ggml_4x2", QgemvShapeQ4K::Ggml4x2_128),
            ("ggml_4x4", QgemvShapeQ4K::Ggml4x4_128),
            ("ggml_4x8", QgemvShapeQ4K::Ggml4x8_128),
            ("ggml_8x2", QgemvShapeQ4K::Ggml8x2_256),
            ("ggml_8x4", QgemvShapeQ4K::Ggml8x4_256),
        ];
        for (val, expect) in cases {
            with_env("FUSOR_Q4K_TALL_TILE", Some(val), || {
                assert_eq!(
                    q4k_tall_override(QgemvShapeQ4K::Ggml4x2_128),
                    expect,
                    "FUSOR_Q4K_TALL_TILE={val}"
                );
            });
        }
    }

    #[test]
    fn q6k_large_override_table_unchanged() {
        let cases = [
            ("ggml_2x2", QgemvShapeQ6K::Ggml2x2_64),
            ("ggml_2x4", QgemvShapeQ6K::Ggml2x4_64),
            ("ggml_2x8", QgemvShapeQ6K::Ggml2x8_64),
            ("ggml_4x2", QgemvShapeQ6K::Ggml4x2_128),
            ("ggml_4x4", QgemvShapeQ6K::Ggml4x4_128),
            ("ggml_4x8", QgemvShapeQ6K::Ggml4x8_128),
            ("ggml_8x2", QgemvShapeQ6K::Ggml8x2_256),
            ("ggml_8x4", QgemvShapeQ6K::Ggml8x4_256),
        ];
        for (val, expect) in cases {
            with_env("FUSOR_Q6K_LARGE_TILE", Some(val), || {
                assert_eq!(
                    q6k_large_override(QgemvShapeQ6K::Ggml2x2_64),
                    expect,
                    "FUSOR_Q6K_LARGE_TILE={val}"
                );
            });
        }
    }

    #[test]
    fn q6k_tall_override_table_unchanged() {
        let cases = [
            ("ggml_2x2", QgemvShapeQ6K::Ggml2x2_64),
            ("ggml_2x4", QgemvShapeQ6K::Ggml2x4_64),
            ("ggml_2x8", QgemvShapeQ6K::Ggml2x8_64),
            ("ggml_4x2", QgemvShapeQ6K::Ggml4x2_128),
            ("ggml_4x4", QgemvShapeQ6K::Ggml4x4_128),
            ("ggml_4x8", QgemvShapeQ6K::Ggml4x8_128),
            ("ggml_8x2", QgemvShapeQ6K::Ggml8x2_256),
            ("ggml_8x4", QgemvShapeQ6K::Ggml8x4_256),
        ];
        for (val, expect) in cases {
            with_env("FUSOR_Q6K_TALL_TILE", Some(val), || {
                assert_eq!(
                    q6k_tall_override(QgemvShapeQ6K::Ggml2x2_64),
                    expect,
                    "FUSOR_Q6K_TALL_TILE={val}"
                );
            });
        }
    }
}
