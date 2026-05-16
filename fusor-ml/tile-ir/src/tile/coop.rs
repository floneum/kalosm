use std::marker::PhantomData;

use super::*;
use crate::ir::{
    CoopElement, CoopMatrixRole, CoopOperandRole, ElementType, Expr, Numeric, TileRef, TileStmt,
};

/// Workgroup tile coordinates for `TileBlock::mma_from_tiles`.
pub struct CoopTileLoad {
    tile: TileRef,
    row: Box<Expr>,
    col: Box<Expr>,
}

/// Cooperative-matrix operand role for generic fragment loads.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CoopRole {
    /// Load an A operand fragment.
    A,
    /// Load a B operand fragment.
    B,
}

impl From<CoopRole> for CoopOperandRole {
    fn from(value: CoopRole) -> Self {
        match value {
            CoopRole::A => Self::A,
            CoopRole::B => Self::B,
        }
    }
}

impl CoopTileLoad {
    /// Create a cooperative tile-load descriptor.
    pub fn new(tile: TileRef, row: impl IntoIndex, col: impl IntoIndex) -> Self {
        Self {
            tile,
            row: row.into_index(),
            col: col.into_index(),
        }
    }
}

impl TileBlock<'_> {
    /// Allocate a cooperative-matrix accumulator.
    ///
    /// ```
    /// use fusor_tile_ir::{tile, F32};
    ///
    /// let ir = tile::build(|program| {
    ///     program.program_grid::<32>([1, 1, 1], |block| {
    ///         let acc = block.alloc_coop_acc::<F32, 8, 8>();
    ///         block.zero_coop_acc(&acc);
    ///     });
    /// });
    /// # let _ = ir;
    /// ```
    pub fn alloc_coop_acc<T: CoopElement, const ROWS: usize, const COLS: usize>(
        &mut self,
    ) -> CoopAcc<T, ROWS, COLS> {
        assert!(
            ROWS == 8 || ROWS == 16,
            "cooperative-matrix rows must be 8 or 16"
        );
        assert!(
            COLS == 8 || COLS == 16,
            "cooperative-matrix columns must be 8 or 16"
        );
        let local = self.program.alloc_local_element(ElementType::coop_matrix(
            T::SCALAR,
            CoopMatrixRole::C,
            ROWS as u32,
            COLS as u32,
        ));
        CoopAcc {
            local,
            _ty: PhantomData,
        }
    }

    /// Zero an accumulator before MMA use.
    pub fn zero_coop_acc<T, const ROWS: usize, const COLS: usize>(
        &mut self,
        acc: &CoopAcc<T, ROWS, COLS>,
    ) {
        self.push_stmt(TileStmt::ZeroCoopAcc { acc: acc.local });
    }

    /// Copy a dense storage tile into workgroup memory.
    pub fn copy_storage_to_tile<T: Numeric>(
        &mut self,
        dst: TileRef,
        src: &Storage<T, 2>,
        row_offset: impl IntoIndex,
        col_offset: impl IntoIndex,
    ) {
        self.push_stmt(TileStmt::CopyToWorkgroupTile {
            dst,
            src: crate::ir::CopySource::Storage(src.view.clone()),
            row_offset: row_offset.into_index(),
            col_offset: col_offset.into_index(),
        });
    }

    /// Copy and dequantize a quantized matrix tile into workgroup memory.
    pub fn copy_quant_to_tile(
        &mut self,
        dst: TileRef,
        src: &crate::quantized::QuantizedMatrix,
        row_offset: impl IntoIndex,
        col_offset: impl IntoIndex,
    ) {
        self.push_stmt(TileStmt::CopyToWorkgroupTile {
            dst,
            src: crate::ir::CopySource::Quantized(src.clone()),
            row_offset: row_offset.into_index(),
            col_offset: col_offset.into_index(),
        });
    }

    /// `acc += A * B` using two cooperative tile-load descriptors.
    ///
    /// Convenience wrapper that emits typed A/B fragment loads, then
    /// [`coop_mma`](Self::coop_mma). For MMAs that share an A or B operand
    /// across the inner row x col grid, prefer the explicit load calls so
    /// fragment handles can be reused.
    pub fn mma_from_tiles<T: CoopElement, const ROWS: usize, const COLS: usize>(
        &mut self,
        acc: &CoopAcc<T, ROWS, COLS>,
        a: CoopTileLoad,
        b: CoopTileLoad,
    ) {
        let a = self.coop_load::<T, ROWS, COLS>(CoopRole::A, a);
        let b = self.coop_load::<T, ROWS, COLS>(CoopRole::B, b);
        self.coop_mma(acc, &a, &b);
    }

    /// Build a cooperative tile-load descriptor for later use.
    pub fn coop_tile_load(
        &self,
        tile: TileRef,
        row: impl IntoIndex,
        col: impl IntoIndex,
    ) -> CoopTileLoad {
        CoopTileLoad::new(tile, row, col)
    }

    /// Cooperatively load a fragment from a workgroup tile.
    pub fn coop_load<T: CoopElement, const ROWS: usize, const COLS: usize>(
        &mut self,
        role: CoopRole,
        load: CoopTileLoad,
    ) -> CoopFragment<T, ROWS, COLS> {
        assert!(
            ROWS == 8 || ROWS == 16,
            "cooperative-matrix rows must be 8 or 16"
        );
        assert!(
            COLS == 8 || COLS == 16,
            "cooperative-matrix columns must be 8 or 16"
        );
        let id = self.program.next_coop_fragment_id();
        let role = CoopOperandRole::from(role);
        self.push_stmt(TileStmt::LoadCoop {
            id,
            role,
            scalar: T::SCALAR,
            rows: ROWS as u32,
            cols: COLS as u32,
            tile: load.tile,
            row: load.row,
            col: load.col,
        });
        CoopFragment {
            id,
            role,
            _ty: PhantomData,
        }
    }

    /// `acc += a * b` where `a`/`b` are fragments previously loaded with the
    /// same scalar and cooperative shape.
    pub fn coop_mma<T, const ROWS: usize, const COLS: usize>(
        &mut self,
        acc: &CoopAcc<T, ROWS, COLS>,
        a: &CoopFragment<T, ROWS, COLS>,
        b: &CoopFragment<T, ROWS, COLS>,
    ) {
        assert_eq!(
            a.role,
            CoopOperandRole::A,
            "coop_mma A operand must be an A-role fragment"
        );
        assert_eq!(
            b.role,
            CoopOperandRole::B,
            "coop_mma B operand must be a B-role fragment"
        );
        self.push_stmt(TileStmt::Mma {
            acc: acc.local,
            a: a.id,
            b: b.id,
        });
    }

    /// Cooperatively store `acc` to `dst` at (row, col).
    pub fn coop_store<T: CoopElement, const ROWS: usize, const COLS: usize>(
        &mut self,
        acc: &CoopAcc<T, ROWS, COLS>,
        dst: &Storage<T, 2>,
        row: impl IntoIndex,
        col: impl IntoIndex,
    ) {
        self.push_stmt(TileStmt::StoreCoopAcc {
            acc: acc.local,
            dst: dst.view.clone(),
            row: row.into_index(),
            col: col.into_index(),
        });
    }
}
