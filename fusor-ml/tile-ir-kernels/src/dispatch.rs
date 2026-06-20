//! Shape-selection policy for kernel dispatch. Pure functions and shape keys.
//!
//! Const-generic monomorphization stays in the dispatch macros that consume
//! these shape keys (see `kernels/qgemv.rs`). The
//! compiler must see the const literals at the dispatch site to monomorphize,
//! so this module returns a small shape key, and a `match` in the builder picks
//! the literal generic
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

use fusor_tile_ir::{GgmlQuantFormat, SubgroupToken};

// ===== qgemv shapes (Q4K and Q6K ggml paths) =====

const fn is_q4k_family(format: GgmlQuantFormat) -> bool {
    format.is_q4k_family()
}

const fn is_q6k_family(format: GgmlQuantFormat) -> bool {
    format.is_q6k_family()
}

/// Default qgemv output columns handled by one workgroup for `format`.
pub const fn qgemv_cols_per_workgroup(format: GgmlQuantFormat) -> u32 {
    qgemv_subgroups_per_workgroup(format) * qgemv_cols_per_subgroup(format)
}

/// Shape-aware qgemv output columns handled by one workgroup.
///
/// This includes the Q4K/Q6K GGML specializations whose column grouping
/// depends on both K (`rows`) and N (`cols`).
pub fn qgemv_cols_per_workgroup_for_shape(format: GgmlQuantFormat, rows: u32, cols: u32) -> u32 {
    if matches!(format, GgmlQuantFormat::Q4_0 | GgmlQuantFormat::Q4_0Native) {
        return q4_0_override(q4_0_default(rows, cols)).cols_per_workgroup();
    }

    if is_q4k_family(format) && rows <= 4096 && (4096..8192).contains(&cols) {
        return q4k_mid_override(q4k_default_mid(rows, cols)).cols_per_workgroup();
    }

    if is_q4k_family(format) && rows <= 4096 && cols >= 8192 {
        return q4k_large_override(q4k_default_large(rows, cols)).cols_per_workgroup();
    }

    if is_q4k_family(format) && rows > 4096 && cols <= 4096 {
        return q4k_tall_override(q4k_default_tall(rows, cols)).cols_per_workgroup();
    }

    if is_q6k_family(format) && rows <= 4096 && cols >= 8192 {
        return q6k_large_override(q6k_default_large(rows, cols)).cols_per_workgroup();
    }

    if is_q6k_family(format) && rows > 4096 && cols <= 4096 {
        return q6k_tall_override(q6k_default_tall(rows, cols)).cols_per_workgroup();
    }

    qgemv_subgroups_per_workgroup_for_shape(format, rows, cols) * qgemv_cols_per_subgroup(format)
}

pub(crate) const fn qgemv_cols_per_subgroup(format: GgmlQuantFormat) -> u32 {
    match format {
        GgmlQuantFormat::Q2K => 4,
        GgmlQuantFormat::Q4_0
        | GgmlQuantFormat::Q4_0Native
        | GgmlQuantFormat::Q4_1
        | GgmlQuantFormat::Q5_1 => 4,
        GgmlQuantFormat::Q5_0 | GgmlQuantFormat::Q5_0Native => 4,
        GgmlQuantFormat::Q3K | GgmlQuantFormat::Q8K => 2,
        GgmlQuantFormat::Q4K | GgmlQuantFormat::Q4KNative => 8,
        GgmlQuantFormat::Q6K | GgmlQuantFormat::Q6KNative => 4,
        GgmlQuantFormat::Q8_0 | GgmlQuantFormat::Q8_0Native | GgmlQuantFormat::Q8_1 => 4,
        GgmlQuantFormat::Q5K | GgmlQuantFormat::Q5KNative => 1,
    }
}

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
pub fn qgemv_subgroups_per_workgroup_for_shape(
    format: GgmlQuantFormat,
    rows: u32,
    cols: u32,
) -> u32 {
    match format {
        GgmlQuantFormat::Q4_0 | GgmlQuantFormat::Q4_0Native => {
            q4_0_override(q4_0_default(rows, cols)).subgroups
        }
        format if format.is_q6k_family() && rows > 4096 => 8,
        _ => qgemv_subgroups_per_workgroup(format),
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct QgemvShape {
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

    pub(crate) const fn cols_per_workgroup(self) -> u32 {
        self.subgroups * self.cols_per_subgroup
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

/// Apply `FUSOR_Q4K_MID_TILE` if set; otherwise return the default. The set
/// of accepted env values is exactly the inline `qgemv_ggml_env!` table that
/// used to live in `qgemv_tile`.
/// Per-context env override tables. Each entry maps an env-var token to the
/// `QgemvShape` it selects. Different contexts (mid / large / tall, Q4K vs
/// Q6K) accept different subsets of the 14 total shapes.
const Q4K_MID_TILES: &[(&str, QgemvShape)] = &[
    ("ggml_2x2", QgemvShape::new(2, 2)),
    ("ggml_2x3", QgemvShape::new(2, 3)),
    ("ggml_2x4", QgemvShape::new(2, 4)),
    ("ggml_2x8", QgemvShape::new(2, 8)),
    ("ggml_4x2", QgemvShape::new(4, 2)),
    ("ggml_4x3", QgemvShape::new(4, 3)),
    ("ggml_4x4", QgemvShape::new(4, 4)),
    ("ggml_4x8", QgemvShape::new(4, 8)),
    ("ggml_8x2", QgemvShape::new(8, 2)),
    ("ggml_8x4", QgemvShape::new(8, 4)),
];

const Q4K_LARGE_TILES: &[(&str, QgemvShape)] = &[
    ("ggml_1x4", QgemvShape::new(1, 4)),
    ("ggml_1x8", QgemvShape::new(1, 8)),
    ("ggml_2x2", QgemvShape::new(2, 2)),
    ("ggml_2x4", QgemvShape::new(2, 4)),
    ("ggml_2x8", QgemvShape::new(2, 8)),
    ("ggml_4x1", QgemvShape::new(4, 1)),
    ("ggml_4x2", QgemvShape::new(4, 2)),
    ("ggml_4x4", QgemvShape::new(4, 4)),
    ("ggml_4x8", QgemvShape::new(4, 8)),
    ("ggml_8x1", QgemvShape::new(8, 1)),
    ("ggml_8x2", QgemvShape::new(8, 2)),
    ("ggml_8x4", QgemvShape::new(8, 4)),
];

const STANDARD_8_TILES: &[(&str, QgemvShape)] = &[
    ("ggml_2x2", QgemvShape::new(2, 2)),
    ("ggml_2x4", QgemvShape::new(2, 4)),
    ("ggml_2x8", QgemvShape::new(2, 8)),
    ("ggml_4x2", QgemvShape::new(4, 2)),
    ("ggml_4x4", QgemvShape::new(4, 4)),
    ("ggml_4x8", QgemvShape::new(4, 8)),
    ("ggml_8x2", QgemvShape::new(8, 2)),
    ("ggml_8x4", QgemvShape::new(8, 4)),
];

const Q4_0_TILES: &[(&str, QgemvShape)] = &[
    ("ggml_1x4", QgemvShape::new(1, 4)),
    ("ggml_1x8", QgemvShape::new(1, 8)),
    ("ggml_2x2", QgemvShape::new(2, 2)),
    ("ggml_2x4", QgemvShape::new(2, 4)),
    ("ggml_2x8", QgemvShape::new(2, 8)),
    ("ggml_4x2", QgemvShape::new(4, 2)),
    ("ggml_4x4", QgemvShape::new(4, 4)),
    ("ggml_4x8", QgemvShape::new(4, 8)),
    ("ggml_8x1", QgemvShape::new(8, 1)),
    ("ggml_8x2", QgemvShape::new(8, 2)),
    ("ggml_8x4", QgemvShape::new(8, 4)),
];

fn env_tile_override(var: &str, table: &[(&str, QgemvShape)], default: QgemvShape) -> QgemvShape {
    let Ok(value) = std::env::var(var) else {
        return default;
    };
    table
        .iter()
        .find(|(name, _)| *name == value)
        .map(|(_, shape)| *shape)
        .unwrap_or(default)
}

pub(crate) fn q4k_mid_override(default: QgemvShape) -> QgemvShape {
    env_tile_override("FUSOR_Q4K_MID_TILE", Q4K_MID_TILES, default)
}

pub(crate) const fn q4_0_default(_rows: u32, _cols: u32) -> QgemvShape {
    QgemvShape::new(4, 4)
}

pub(crate) fn q4_0_override(default: QgemvShape) -> QgemvShape {
    env_tile_override("FUSOR_Q4_0_TILE", Q4_0_TILES, default)
}

pub(crate) fn q4_0_values_per_lane(default: u32) -> u32 {
    match std::env::var("FUSOR_Q4_0_VALUES_PER_LANE").ok().as_deref() {
        Some("8") => 8,
        Some("16") => 16,
        Some("32") => 32,
        _ => default,
    }
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

/// Apply `FUSOR_Q4K_LARGE_TILE` if set. Carries the same tile list as the
/// inline macro: adds 1x4/1x8/4x1/8x1 (no 2x3/4x3 entries).
pub(crate) fn q4k_large_override(default: QgemvShape) -> QgemvShape {
    env_tile_override("FUSOR_Q4K_LARGE_TILE", Q4K_LARGE_TILES, default)
}

// ----- Q4K tall (rows>4096, cols<=4096) -----

/// Default Q4K tall-shape: 4x2.
pub(crate) const fn q4k_default_tall(_rows: u32, _cols: u32) -> QgemvShape {
    QgemvShape::new(4, 2)
}

/// Apply `FUSOR_Q4K_TALL_TILE` if set. Standard 8-tile set.
pub(crate) fn q4k_tall_override(default: QgemvShape) -> QgemvShape {
    env_tile_override("FUSOR_Q4K_TALL_TILE", STANDARD_8_TILES, default)
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

/// Apply `FUSOR_Q6K_LARGE_TILE` if set. Standard 8-tile set.
pub(crate) fn q6k_large_override(default: QgemvShape) -> QgemvShape {
    env_tile_override("FUSOR_Q6K_LARGE_TILE", STANDARD_8_TILES, default)
}

// ----- Q6K tall (rows>4096, cols<=4096) -----

/// Default Q6K tall-shape: 2x2.
pub(crate) const fn q6k_default_tall(_rows: u32, _cols: u32) -> QgemvShape {
    QgemvShape::new(2, 2)
}

/// Apply `FUSOR_Q6K_TALL_TILE` if set. Standard 8-tile set.
pub(crate) fn q6k_tall_override(default: QgemvShape) -> QgemvShape {
    env_tile_override("FUSOR_Q6K_TALL_TILE", STANDARD_8_TILES, default)
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

    #[test]
    fn q4k_mid_override_table_unchanged() {
        with_env("FUSOR_Q4K_MID_TILE", None, || {
            assert_eq!(
                q4k_mid_override(QgemvShape::new(2, 2)),
                QgemvShape::new(2, 2)
            );
        });
        let cases = [
            ("ggml_2x2", QgemvShape::new(2, 2)),
            ("ggml_2x3", QgemvShape::new(2, 3)),
            ("ggml_2x4", QgemvShape::new(2, 4)),
            ("ggml_2x8", QgemvShape::new(2, 8)),
            ("ggml_4x2", QgemvShape::new(4, 2)),
            ("ggml_4x3", QgemvShape::new(4, 3)),
            ("ggml_4x4", QgemvShape::new(4, 4)),
            ("ggml_4x8", QgemvShape::new(4, 8)),
            ("ggml_8x2", QgemvShape::new(8, 2)),
            ("ggml_8x4", QgemvShape::new(8, 4)),
        ];
        for (val, expect) in cases {
            with_env("FUSOR_Q4K_MID_TILE", Some(val), || {
                assert_eq!(
                    q4k_mid_override(QgemvShape::new(4, 4)),
                    expect,
                    "FUSOR_Q4K_MID_TILE={val}"
                );
            });
        }
        // Unrecognized value falls through to default.
        with_env("FUSOR_Q4K_MID_TILE", Some("nonsense"), || {
            assert_eq!(
                q4k_mid_override(QgemvShape::new(4, 4)),
                QgemvShape::new(4, 4)
            );
        });
    }

    #[test]
    fn q4k_large_override_table_unchanged() {
        let cases = [
            ("ggml_1x4", QgemvShape::new(1, 4)),
            ("ggml_1x8", QgemvShape::new(1, 8)),
            ("ggml_2x2", QgemvShape::new(2, 2)),
            ("ggml_2x4", QgemvShape::new(2, 4)),
            ("ggml_2x8", QgemvShape::new(2, 8)),
            ("ggml_4x1", QgemvShape::new(4, 1)),
            ("ggml_4x2", QgemvShape::new(4, 2)),
            ("ggml_4x4", QgemvShape::new(4, 4)),
            ("ggml_4x8", QgemvShape::new(4, 8)),
            ("ggml_8x1", QgemvShape::new(8, 1)),
            ("ggml_8x2", QgemvShape::new(8, 2)),
            ("ggml_8x4", QgemvShape::new(8, 4)),
        ];
        for (val, expect) in cases {
            with_env("FUSOR_Q4K_LARGE_TILE", Some(val), || {
                assert_eq!(
                    q4k_large_override(QgemvShape::new(4, 4)),
                    expect,
                    "FUSOR_Q4K_LARGE_TILE={val}"
                );
            });
        }
    }

    #[test]
    fn q4k_tall_override_table_unchanged() {
        let cases = [
            ("ggml_2x2", QgemvShape::new(2, 2)),
            ("ggml_2x4", QgemvShape::new(2, 4)),
            ("ggml_2x8", QgemvShape::new(2, 8)),
            ("ggml_4x2", QgemvShape::new(4, 2)),
            ("ggml_4x4", QgemvShape::new(4, 4)),
            ("ggml_4x8", QgemvShape::new(4, 8)),
            ("ggml_8x2", QgemvShape::new(8, 2)),
            ("ggml_8x4", QgemvShape::new(8, 4)),
        ];
        for (val, expect) in cases {
            with_env("FUSOR_Q4K_TALL_TILE", Some(val), || {
                assert_eq!(
                    q4k_tall_override(QgemvShape::new(4, 2)),
                    expect,
                    "FUSOR_Q4K_TALL_TILE={val}"
                );
            });
        }
    }

    #[test]
    fn q6k_large_override_table_unchanged() {
        let cases = [
            ("ggml_2x2", QgemvShape::new(2, 2)),
            ("ggml_2x4", QgemvShape::new(2, 4)),
            ("ggml_2x8", QgemvShape::new(2, 8)),
            ("ggml_4x2", QgemvShape::new(4, 2)),
            ("ggml_4x4", QgemvShape::new(4, 4)),
            ("ggml_4x8", QgemvShape::new(4, 8)),
            ("ggml_8x2", QgemvShape::new(8, 2)),
            ("ggml_8x4", QgemvShape::new(8, 4)),
        ];
        for (val, expect) in cases {
            with_env("FUSOR_Q6K_LARGE_TILE", Some(val), || {
                assert_eq!(
                    q6k_large_override(QgemvShape::new(2, 2)),
                    expect,
                    "FUSOR_Q6K_LARGE_TILE={val}"
                );
            });
        }
    }

    #[test]
    fn q6k_tall_override_table_unchanged() {
        let cases = [
            ("ggml_2x2", QgemvShape::new(2, 2)),
            ("ggml_2x4", QgemvShape::new(2, 4)),
            ("ggml_2x8", QgemvShape::new(2, 8)),
            ("ggml_4x2", QgemvShape::new(4, 2)),
            ("ggml_4x4", QgemvShape::new(4, 4)),
            ("ggml_4x8", QgemvShape::new(4, 8)),
            ("ggml_8x2", QgemvShape::new(8, 2)),
            ("ggml_8x4", QgemvShape::new(8, 4)),
        ];
        for (val, expect) in cases {
            with_env("FUSOR_Q6K_TALL_TILE", Some(val), || {
                assert_eq!(
                    q6k_tall_override(QgemvShape::new(2, 2)),
                    expect,
                    "FUSOR_Q6K_TALL_TILE={val}"
                );
            });
        }
    }
}
