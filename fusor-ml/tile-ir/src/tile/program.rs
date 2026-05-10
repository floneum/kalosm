#![allow(unused_imports)]
use std::marker::PhantomData;
use std::ops::{Add, BitAnd, BitXor, Div, Mul, Rem, Sub};

use crate::ir::{
    BlockDequantId, BufferAccess, BufferDecl, BufferRef, CoopFragmentId,
    CoopOperandRole, DynamicOffset, F32Bits, F32Vec4, Im2ColNhwcMap, KernelIr, Layout, LocalDecl,
    LocalRef, MemoryLevel, Numeric, Op,
    QuantizedVecDotKind, Shape, StorageIndexMap, StorageView, TileBinaryOp, TileCompareOp,
    TileDecl, TileExpr, TileIndexExpr, TileIndexedStoreStmt, TileLevel, TileLinearLoadExpr,
    TileLiteral, TileLoadExpr, TileMaskExpr, TileOrigin, TileProgramOp, TileQuantizedLoadExpr,
    TileReduceOp, TileRef, TileScalarExpr, TileStmt, TileStoreStmt, TileUnaryOp, TileVec4LoadExpr,
    WorkgroupAxis, WorkgroupOffset, F32, U32,
};
use crate::dispatch::{qmatmul_path, QmatmulPath};
use crate::quantized::{GgmlQuantFormat, QuantizedMatrix};
use super::*;
use super::types::{matrix_shape, cooperative_store_layout_supported};
use super::grid::{qgemv_grid, store_qgemv_sums, q4k_ggml_activations, dot4_sum};

/// Monomorphization dispatcher for `qmatmul_perf`. Stays as a macro so the
/// const literals are visible at the call site for the compiler to
/// instantiate. Policy lives in `crate::dispatch::qmatmul_path`.
macro_rules! dispatch_qmatmul {
    ($program:expr, $path:expr, $a:expr, $b:expr, $y:expr, $fallthrough:block) => {
        match $path {
            QmatmulPath::Coop64x64 => {
                return $program.qmatmul_perf::<64, 64, 32, 2, 2, 128>($a, $b, $y);
            }
            QmatmulPath::Coop64x128 => {
                return $program.qmatmul_perf::<64, 128, 32, 2, 4, 256>($a, $b, $y);
            }
            QmatmulPath::Coop128x64 => {
                return $program.qmatmul_perf::<128, 64, 32, 4, 2, 256>($a, $b, $y);
            }
            QmatmulPath::Coop128x128 => {
                return $program.qmatmul_perf::<128, 128, 32, 4, 4, 512>($a, $b, $y);
            }
            QmatmulPath::Scalar => $fallthrough,
        }
    };
}

macro_rules! storage_accessors {
    ($read:ident, $write:ident($($arg:ident: $ty:ty),*) => ($layout:expr, $offset:expr, $index_map:expr)) => {
        pub fn $read<T: Numeric, const R: usize>(&mut self, $($arg: $ty),*) -> Storage<T, R> {
            self.storage_with_layout_and_access($layout, $offset, $index_map, BufferAccess::Read)
        }

        pub fn $write<T: Numeric, const R: usize>(&mut self, $($arg: $ty),*) -> Storage<T, R> {
            self.storage_with_layout_and_access(
                $layout,
                $offset,
                $index_map,
                BufferAccess::ReadWrite,
            )
        }
    };
}

macro_rules! erased_storage_accessors {
    ($read:ident, $write:ident($($arg:ident: $ty:ty),*) => ($layout:expr, $offset:expr)) => {
        pub fn $read<const R: usize>(
            &mut self,
            element: crate::ElementType,
            $($arg: $ty),*
        ) -> ErasedStorage<R> {
            ErasedStorage {
                view: self.storage_view_with_layout_and_access::<R>(
                    element,
                    $layout,
                    $offset,
                    None,
                    BufferAccess::Read,
                ),
            }
        }

        pub fn $write<const R: usize>(
            &mut self,
            element: crate::ElementType,
            $($arg: $ty),*
        ) -> ErasedStorage<R> {
            ErasedStorage {
                view: self.storage_view_with_layout_and_access::<R>(
                    element,
                    $layout,
                    $offset,
                    None,
                    BufferAccess::ReadWrite,
                ),
            }
        }
    };
}

pub struct Program {
    pub(crate) ir: KernelIr,
    /// Builder-only counter for fresh `BufferId`s. Lives here (not on
    /// `KernelIr`) because the finished IR is immutable data — the counter
    /// is only needed during construction.
    pub(crate) next_buffer: u32,
    /// Builder-only counter for fresh `TileId`s. Same reasoning as
    /// `next_buffer`.
    pub(crate) next_tile: u32,
    /// Builder-only counter for fresh `LocalId`s. Same reasoning as
    /// `next_buffer`.
    pub(crate) next_local: u32,
    /// Builder-only counter for fresh `BlockDequantId`s. Lives here (not on
    /// `KernelIr`) because these ids are SSA-scoped names allocated by the
    /// builder and never observed off the finished IR.
    pub(crate) next_block_dequant: u32,
    /// Builder-only counter for fresh `CoopFragmentId`s. Same reasoning as
    /// `next_block_dequant`.
    pub(crate) next_coop_fragment: u32,
}

impl Program {
    /// Create an empty builder. Most callers should use [`build`] instead;
    /// this is for [`crate::kernel_builder::KernelBuilder`] which owns the
    /// program plus a parallel binding list.
    pub fn new() -> Self {
        Self {
            ir: KernelIr::default(),
            next_buffer: 0,
            next_tile: 0,
            next_local: 0,
            next_block_dequant: 0,
            next_coop_fragment: 0,
        }
    }

    /// Consume the builder and return the constructed [`KernelIr`].
    pub fn into_ir(self) -> KernelIr {
        self.ir
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

impl Program {
    storage_accessors!(
        storage_read,
        storage_write(shape: Shape) => (
            Layout::contiguous(MemoryLevel::Storage, shape),
            0,
            None
        )
    );
    storage_accessors!(
        storage_read_with_layout,
        storage_write_with_layout(layout: Layout) => (layout, 0, None)
    );
    storage_accessors!(
        storage_read_with_layout_offset,
        storage_write_with_layout_offset(layout: Layout, offset: u32) => (layout, offset, None)
    );
    storage_accessors!(
        storage_read_with_layout_offset_and_index_map,
        storage_write_with_layout_offset_and_index_map(
            layout: Layout,
            offset: u32,
            index_map: StorageIndexMap
        ) => (layout, offset, Some(index_map))
    );

    fn storage_with_layout_and_access<T: Numeric, const R: usize>(
        &mut self,
        layout: Layout,
        offset: u32,
        index_map: Option<StorageIndexMap>,
        access: BufferAccess,
    ) -> Storage<T, R> {
        let view = self.storage_view_with_layout_and_access::<R>(
            T::ELEMENT,
            layout,
            offset,
            index_map,
            access,
        );
        Storage {
            view,
            _ty: PhantomData,
        }
    }

    erased_storage_accessors!(
        storage_read_element_with_layout_offset,
        storage_write_element_with_layout_offset(layout: Layout, offset: u32) => (layout, offset)
    );

    fn storage_view_with_layout_and_access<const R: usize>(
        &mut self,
        element: crate::ElementType,
        layout: Layout,
        offset: u32,
        index_map: Option<StorageIndexMap>,
        access: BufferAccess,
    ) -> StorageView {
        assert_eq!(
            layout.memory_level(),
            MemoryLevel::Storage,
            "storage tensors must use MemoryLevel::Storage"
        );
        assert_eq!(layout.shape().rank(), R, "storage rank mismatch");
        let buffer = self.alloc_buffer_element(element, layout.clone(), access);
        StorageView {
            buffer,
            offset,
            dynamic_offsets: vec![None; layout.shape().rank()],
            layout,
            index_map,
        }
    }

    pub fn quantized_matrix(
        &mut self,
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
        let data = self.storage_read::<U32, 1>(Shape::new([words]));
        QuantizedMatrix {
            data: data.view,
            format,
            rows,
            cols,
        }
    }

    pub fn qmatmul<const BM: usize, const BN: usize, const BK: usize>(
        &mut self,
        a: &Storage<F32, 2>,
        b: &QuantizedMatrix,
        y: &Storage<F32, 2>,
        vector_width: u32,
    ) {
        self.qmatmul_options::<BM, BN, BK>(a, b, y, vector_width, true, 1);
    }

    pub fn qgemv<const BN: usize, const BK: usize>(
        &mut self,
        a: &Storage<F32, 2>,
        b: &QuantizedMatrix,
        y: &Storage<F32, 2>,
        vector_width: u32,
        workgroups_x: u32,
    ) {
        self.qmatmul_options::<1, BN, BK>(a, b, y, vector_width, true, workgroups_x);
    }

    pub fn qmatmul_options<const BM: usize, const BN: usize, const BK: usize>(
        &mut self,
        a: &Storage<F32, 2>,
        b: &QuantizedMatrix,
        y: &Storage<F32, 2>,
        vector_width: u32,
        use_qgemv: bool,
        workgroups_x: u32,
    ) {
        assert!(
            BM > 0 && BN > 0 && BK > 0,
            "qmatmul tile shape must be non-zero"
        );
        assert!(vector_width > 0, "qmatmul vector width must be non-zero");
        assert!(workgroups_x > 0, "qmatmul workgroups_x must be non-zero");
        let [m, k] = matrix_shape(&a.view.layout);
        let [y_m, y_n] = matrix_shape(&y.view.layout);
        assert_eq!(k, b.rows, "qmatmul K dimensions must match");
        assert_eq!(m, y_m, "qmatmul output row count must match A");
        assert_eq!(b.cols, y_n, "qmatmul output column count must match B");

        if m == 1 && use_qgemv {
            self.qgemv_tile::<BN, BK>(a, b, y, workgroups_x);
        } else {
            self.qmatmul_tile::<BM, BN, BK>(a, b, y);
        }
    }

    fn qmatmul_tile<const BM: usize, const BN: usize, const BK: usize>(
        &mut self,
        a: &Storage<F32, 2>,
        b: &QuantizedMatrix,
        y: &Storage<F32, 2>,
    ) {
        const LANES: usize = 256;
        assert!(
            BM > 0 && BN > 0 && BK > 0,
            "qmatmul tile shape must be non-zero"
        );
        let [m, k] = matrix_shape(&a.view.layout);

        // Cooperative-matrix perf-parity path. Policy in `crate::dispatch`
        // selects which `qmatmul_perf::<...>` monomorphization to call (or
        // `Scalar` to fall through to the lane-mapped tile body below).
        let path = qmatmul_path(
            BM,
            BN,
            BK,
            m as usize % BM == 0,
            b.cols as usize % BN == 0,
            k as usize % BK == 0,
            cooperative_store_layout_supported(&y.view.layout),
        );
        dispatch_qmatmul!(self, path, a, b, y, {});

        if BM * BN * BK != LANES || !BK.is_power_of_two() {
            self.qmatmul_tile::<8, 4, 8>(a, b, y);
            return;
        }
        let k_iterations = k.div_ceil(BK as u32);
        self.program_grid::<LANES>(
            [b.cols.div_ceil(BN as u32), m.div_ceil(BM as u32), 1],
            |program| {
                let lane = program.arange();
                let k_lane = lane.clone() % BK as u32;
                let output_lane = lane / BK as u32;
                let row_lane = output_lane.clone() / BN as u32;
                let col_lane = output_lane % BN as u32;
                let row = program.program_id(WorkgroupAxis::Y) * BM as u32 + row_lane;
                let col = program.program_id(WorkgroupAxis::X) * BN as u32 + col_lane;
                let loop_index = program.loop_index();
                let k_index = loop_index * BK as u32 + k_lane.clone();
                let mask = row.lt(m).and(col.lt(b.cols)).and(k_index.lt(k));
                let a_value = program.load(a.at(&row, &k_index), mask.clone(), 0.0);
                let b_value = program.load_quantized(b, &k_index, &col, mask.clone(), 0.0);
                let partial = program.loop_fold(
                    TileReduceOp::Sum,
                    k_iterations,
                    a_value * b_value,
                    TileLiteral::F32(F32Bits::new(0.0)),
                );
                let sum = program.group_reduce_sum::<BK>(partial);
                let store_mask = k_lane.eq(0).and(row.lt(m)).and(col.lt(b.cols));
                program.store(y.at(row, col), sum, store_mask);
            },
        );
    }

    /// Cooperative-matrix qmatmul body. Each workgroup produces one BMxBN
    /// output tile via an interleaved `ROW_GROUPS x COL_GROUPS` grid of
    /// subgroups, each holding `(32*32)/(8*8)` = 16 cooperative-matrix
    /// accumulators. `BLOCK == ROW_GROUPS * COL_GROUPS * 32`.
    fn qmatmul_perf<
        const BM: usize,
        const BN: usize,
        const BK: usize,
        const ROW_GROUPS: u32,
        const COL_GROUPS: u32,
        const BLOCK: usize,
    >(
        &mut self,
        a: &Storage<F32, 2>,
        b: &QuantizedMatrix,
        y: &Storage<F32, 2>,
    ) {
        const COOP_DIM: u32 = 8;
        const SUBGROUP_SIZE: u32 = 32;
        const SUBGROUP_ROWS: u32 = 32;
        const SUBGROUP_COLS: u32 = 32;
        debug_assert_eq!(ROW_GROUPS * SUBGROUP_ROWS, BM as u32);
        debug_assert_eq!(COL_GROUPS * SUBGROUP_COLS, BN as u32);
        debug_assert_eq!(ROW_GROUPS * COL_GROUPS * SUBGROUP_SIZE, BLOCK as u32);

        let [m, k] = matrix_shape(&a.view.layout);
        let n = b.cols;
        let n_grid_x = n / BN as u32;
        let n_grid_y = m / BM as u32;
        let k_iterations = k / BK as u32;

        let a_tile = self.alloc_workgroup_tile_f32(BM as u32, BK as u32);
        let b_tile = self.alloc_workgroup_tile_f32(BK as u32, BN as u32);
        let b_clone = b.clone();
        let a_clone = a.clone();
        let y_clone = y.clone();

        const TILE_ROWS_PER_SG: u32 = SUBGROUP_ROWS / 8;
        const TILE_COLS_PER_SG: u32 = SUBGROUP_COLS / 8;

        self.program_grid::<BLOCK>([n_grid_x, n_grid_y, 1], |program| {
            let row_base = program.program_id(WorkgroupAxis::Y) * BM as u32;
            let col_base = program.program_id(WorkgroupAxis::X) * BN as u32;
            let subgroup_id = program.subgroup_id();
            let sg_row = subgroup_id.clone() / COL_GROUPS;
            let sg_col = subgroup_id % COL_GROUPS;
            let sg_row_base = sg_row * SUBGROUP_ROWS;
            let sg_col_base = sg_col * SUBGROUP_COLS;

            // Allocate and zero per-fragment accumulators. Each subgroup uses
            // the same accumulator schema; runtime indexing via subgroup_id
            // selects the data each subgroup operates on.
            let accs: Vec<Vec<CoopAcc>> = (0..TILE_ROWS_PER_SG)
                .map(|_| {
                    (0..TILE_COLS_PER_SG)
                        .map(|_| {
                            let acc = program.alloc_coop_acc();
                            program.zero_coop_acc(&acc);
                            acc
                        })
                        .collect()
                })
                .collect();

            program.while_true(k_iterations, |program| {
                let k_base = program.loop_index() * BK as u32;
                program.copy_storage_to_tile(a_tile, &a_clone, &row_base, &k_base);
                program.copy_quant_to_tile(b_tile, &b_clone, &k_base, &col_base);
                program.workgroup_barrier();

                // Per kk-step: load each row's A fragment once and each col's
                // B fragment once, then MMA the full row × col grid against
                // the cached SSA handles. This keeps the kk-step at
                // ROW × A-loads + COL × B-loads + ROW*COL × MMAs (matching
                // the accelerator path), instead of (ROW*COL) of each.
                let kk_steps = (BK as u32) / COOP_DIM;
                for kk in 0..kk_steps {
                    let a_frags: Vec<_> = (0..TILE_ROWS_PER_SG)
                        .map(|r| {
                            program.coop_load_a(
                                a_tile,
                                sg_row_base.clone() + r * COOP_DIM,
                                kk * COOP_DIM,
                            )
                        })
                        .collect();
                    let b_frags: Vec<_> = (0..TILE_COLS_PER_SG)
                        .map(|c| {
                            program.coop_load_b(
                                b_tile,
                                kk * COOP_DIM,
                                sg_col_base.clone() + c * COOP_DIM,
                            )
                        })
                        .collect();
                    for r in 0..TILE_ROWS_PER_SG {
                        for c in 0..TILE_COLS_PER_SG {
                            program.coop_mma(
                                &accs[r as usize][c as usize],
                                &a_frags[r as usize],
                                &b_frags[c as usize],
                            );
                        }
                    }
                }
                program.workgroup_barrier();
            });

            // Cooperative-store each fragment to its global tile location.
            for r in 0..TILE_ROWS_PER_SG {
                for c in 0..TILE_COLS_PER_SG {
                    let row = row_base.clone() + sg_row_base.clone() + r * COOP_DIM;
                    let col = col_base.clone() + sg_col_base.clone() + c * COOP_DIM;
                    program.coop_store(&accs[r as usize][c as usize], &y_clone, row, col);
                }
            }
        });
    }

    pub fn matmul<const BK: usize>(
        &mut self,
        a: &Storage<F32, 2>,
        b: &Storage<F32, 2>,
        y: &Storage<F32, 2>,
    ) {
        const TILE_ROWS: usize = 16;
        const TILE_COLS: usize = 16;
        const LANES: usize = TILE_ROWS * TILE_COLS;
        assert!(BK > 0, "matmul K tile shape must be non-zero");
        let [m, k] = matrix_shape(&a.view.layout);
        let [b_k, n] = matrix_shape(&b.view.layout);
        let [y_m, y_n] = matrix_shape(&y.view.layout);
        assert_eq!(k, b_k, "matmul K dimensions must match");
        assert_eq!(m, y_m, "matmul output row count must match A");
        assert_eq!(n, y_n, "matmul output column count must match B");

        self.program_grid::<LANES>(
            [
                n.div_ceil(TILE_COLS as u32),
                m.div_ceil(TILE_ROWS as u32),
                1,
            ],
            |program| {
                let lane_tile = program.lane_tile_2d::<TILE_ROWS, TILE_COLS>();
                let row = program.program_id(WorkgroupAxis::Y) * TILE_ROWS as u32 + lane_tile.row();
                let col = program.program_id(WorkgroupAxis::X) * TILE_COLS as u32 + lane_tile.col();
                let mask = row.lt(m).and(col.lt(n));
                let k_index = program.loop_index();
                let a_value = program.load(a.at(&row, &k_index), mask.clone(), 0.0);
                let b_value = program.load(b.at(&k_index, &col), mask.clone(), 0.0);
                let sum = program.loop_fold(
                    TileReduceOp::Sum,
                    k,
                    a_value * b_value,
                    TileLiteral::F32(F32Bits::new(0.0)),
                );
                program.store(y.at(row, col), sum, mask);
            },
        );
    }

    pub fn gemv<
        const ROWS_PER_WORKGROUP: usize,
        const VALUES_PER_LANE: usize,
        const BLOCK: usize,
    >(
        &mut self,
        a: &Storage<F32, 2>,
        x: &Storage<F32, 2>,
        y: &Storage<F32, 2>,
    ) {
        const SUBGROUP_SIZE: u32 = 32;
        assert!(
            ROWS_PER_WORKGROUP > 0 && VALUES_PER_LANE > 0,
            "gemv tile shape must be non-zero"
        );
        assert_eq!(
            ROWS_PER_WORKGROUP as u32 * SUBGROUP_SIZE,
            BLOCK as u32,
            "gemv maps one output row to each subgroup"
        );
        assert!(
            VALUES_PER_LANE % 4 == 0,
            "gemv values per lane must be divisible by dot4 width"
        );
        let [m, k] = matrix_shape(&a.view.layout);
        let [x_k, n] = matrix_shape(&x.view.layout);
        let [y_m, y_n] = matrix_shape(&y.view.layout);
        assert_eq!(k, x_k, "gemv K dimensions must match");
        assert_eq!(n, 1, "gemv expects a single RHS column");
        assert_eq!(m, y_m, "gemv output row count must match A");
        assert_eq!(y_n, 1, "gemv output must have a single column");

        let k_per_iter = SUBGROUP_SIZE * VALUES_PER_LANE as u32;
        let k_iterations = k.div_ceil(k_per_iter);
        self.program_grid::<BLOCK>([m.div_ceil(ROWS_PER_WORKGROUP as u32), 1, 1], |program| {
            let row = program.program_id(WorkgroupAxis::X) * ROWS_PER_WORKGROUP as u32
                + program.subgroup_id();
            let lane = program.subgroup_lane();
            let row_in_bounds = row.lt(m);
            let zero = TileLiteral::F32(F32Bits::new(0.0));
            let [sum] =
                program.loop_fold_n::<1, _>(TileReduceOp::Sum, k_iterations, [zero], |program| {
                    let k_base =
                        program.loop_index() * k_per_iter + lane.clone() * VALUES_PER_LANE as u32;
                    let a_values: [Tile<BLOCK>; VALUES_PER_LANE] = std::array::from_fn(|i| {
                        let k_index = k_base.clone() + i as u32;
                        let mask = row_in_bounds.clone().and(k_index.lt(k));
                        program.load(a.at(&row, k_index), mask, 0.0)
                    });
                    let x_values: [Tile<BLOCK>; VALUES_PER_LANE] = std::array::from_fn(|i| {
                        let k_index = k_base.clone() + i as u32;
                        program.load(x.at(k_index.clone(), 0), k_index.lt(k), 0.0)
                    });
                    [dot4_sum(program, &a_values, &x_values)]
                });
            let reduced = program.subgroup_reduce_sum(sum);
            let mask = lane.eq(0).and(row_in_bounds);
            program.store(y.at(row, 0), reduced, mask);
        });
    }

    pub fn qdequantize(&mut self, b: &QuantizedMatrix, y: &Storage<F32, 1>, workgroups_x: u32) {
        const BLOCK: usize = 256;
        assert!(
            workgroups_x > 0,
            "qdequantize workgroups_x must be non-zero"
        );
        assert_eq!(
            y.view.layout.element_count().get(),
            b.rows
                .checked_mul(b.cols)
                .expect("qdequantize output element count overflow"),
            "qdequantize output must contain one dense f32 per quantized element"
        );
        assert!(
            y.view.layout.is_row_major(),
            "qdequantize output must be row-major"
        );

        let total = b
            .rows
            .checked_mul(b.cols)
            .expect("qdequantize output element count overflow");
        let workgroups = total.div_ceil(BLOCK as u32);
        let dispatch_y = workgroups.div_ceil(workgroups_x);
        let y = Storage::<F32, 2> {
            view: StorageView {
                buffer: y.view.buffer,
                offset: y.view.offset,
                dynamic_offsets: vec![None, None],
                layout: Layout::contiguous(MemoryLevel::Storage, Shape::new([1, total])),
                index_map: None,
            },
            _ty: PhantomData,
        };
        self.program_grid::<BLOCK>([workgroups_x, dispatch_y, 1], |program| {
            let lane = program.arange();
            let linear_group = program.program_id(WorkgroupAxis::X)
                + program.program_id(WorkgroupAxis::Y) * workgroups_x;
            let flat = linear_group * BLOCK as u32 + lane;
            let mask = flat.lt(total);
            let value = program.load_quantized(
                b,
                flat.clone() % b.rows,
                flat.clone() / b.rows,
                mask.clone(),
                0.0,
            );
            program.store(y.at(0, flat), value, mask);
        });
    }

    pub fn program_grid<const BLOCK: usize>(
        &mut self,
        grid: [u32; 3],
        body: impl FnOnce(&mut TileBlock<'_, BLOCK>),
    ) {
        assert!(BLOCK > 0, "tile block size must be non-zero");
        assert!(
            BLOCK <= 1024 && BLOCK.is_power_of_two(),
            "tile block size must be a power of two at most 1024"
        );
        let mut block = TileBlock {
            program: self,
            grid,
            body: Vec::new(),
            stmt_stack: Vec::new(),
        };
        body(&mut block);
        block.program.ir.body.push(Op::TileProgram(TileProgramOp {
            grid,
            block: BLOCK as u32,
            body: block.body,
        }));
    }

    fn alloc_buffer_element(
        &mut self,
        element: crate::ElementType,
        layout: Layout,
        access: BufferAccess,
    ) -> BufferRef {
        let id = crate::BufferId(self.next_buffer);
        self.next_buffer += 1;
        let buffer = BufferRef::new(id, element);
        self.ir.buffers.push(BufferDecl {
            id,
            element,
            layout,
            access,
        });
        buffer
    }

    pub(super) fn next_block_dequant_id(&mut self) -> BlockDequantId {
        let id = BlockDequantId(self.next_block_dequant);
        self.next_block_dequant += 1;
        id
    }

    pub(super) fn next_coop_fragment_id(&mut self) -> CoopFragmentId {
        let id = CoopFragmentId(self.next_coop_fragment);
        self.next_coop_fragment += 1;
        id
    }

    pub(super) fn alloc_local<T: Numeric>(&mut self) -> LocalRef {
        self.alloc_local_element(T::ELEMENT)
    }

    pub(super) fn alloc_local_element(&mut self, element: crate::ElementType) -> LocalRef {
        let id = crate::LocalId(self.next_local);
        self.next_local += 1;
        let local = LocalRef::new(id, element);
        self.ir.locals.push(LocalDecl { id, element });
        local
    }

    /// Allocate a workgroup-scope f32 tile of shape `[rows, cols]`.
    pub fn alloc_workgroup_tile_f32(&mut self, rows: u32, cols: u32) -> TileRef {
        self.alloc_tile::<F32>(
            Layout::contiguous(MemoryLevel::Workgroup, Shape::new([rows, cols])),
            TileLevel::Workgroup,
        )
    }

    /// Allocate a rank-1 workgroup-scope scratch array.
    pub fn alloc_workgroup_array<T: Numeric>(&mut self, len: u32) -> TileRef {
        self.alloc_tile::<T>(
            Layout::contiguous(MemoryLevel::Workgroup, Shape::new([len])),
            TileLevel::Workgroup,
        )
    }

    pub(super) fn alloc_tile<T: Numeric>(&mut self, layout: Layout, level: TileLevel) -> TileRef {
        let id = crate::TileId(self.next_tile);
        self.next_tile += 1;
        let tile = TileRef::new(id, T::ELEMENT);
        self.ir.tiles.push(TileDecl {
            id,
            element: T::ELEMENT,
            layout,
            level,
            origin: TileOrigin::Allocation,
        });
        tile
    }
}
