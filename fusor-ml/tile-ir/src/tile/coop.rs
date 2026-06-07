use super::block::load_coop_acc;
use super::value::WorkgroupTile;
use super::value::{boxed_index, CoopAcc};
use super::{Storage, Tile, TileBlock};
use crate::ir::{Addr, CoopMatrixRole, CoopSrc, ElementType, Expr, ExprKind, ScalarElement, Stmt};

impl TileBlock<'_> {
    /// Allocate a cooperative-matrix accumulator local (role `C`). Coop
    /// accumulators are mutable private locals; zero/set/mma are composed
    /// through [`coop_store_local`](Self::coop_store_local):
    /// - zero: `coop_store_local(acc, coop_zero(..))`
    /// - set:  `coop_store_local(acc, coop_load_c_broadcast(..))`
    /// - mma:  `coop_store_local(acc, coop_mma(a, b, coop_load_local(acc)))`
    pub(crate) fn alloc_coop_acc(
        &mut self,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
    ) -> CoopAcc {
        assert!(
            rows == 8 || rows == 16,
            "cooperative-matrix rows must be 8 or 16"
        );
        assert!(
            cols == 8 || cols == 16,
            "cooperative-matrix columns must be 8 or 16"
        );
        self.program.alloc_coop_acc(scalar, rows, cols)
    }

    /// Load the current SSA value of a coop accumulator.
    pub(crate) fn coop_load_local(&self, acc: &CoopAcc) -> Tile {
        load_coop_acc(acc)
    }

    /// Store a coop value into an accumulator. The lowerer chains MMA stores
    /// through the acc-value SSA memo (1 Load + N MMA + 1 Store per iteration).
    pub(crate) fn coop_store_local(&mut self, acc: &CoopAcc, value: Tile) {
        self.push_stmt(Stmt::StoreLocal {
            dst: acc.decl().clone(),
            value: value.into_expr(),
        });
    }

    /// A zeroed coop-`C` accumulator value. Composed via
    /// `coop_store_local(acc, coop_zero(..))` for the zero-init case.
    ///
    /// Shaped as a `Cast` of an f32 zero literal to the coop-`C` element type:
    /// it introduces no spurious decls, and the lowerer lowers a cast to a
    /// cooperative-matrix type as `Expression::ZeroValue` (there is no real
    /// scalar→fragment cast).
    pub(crate) fn coop_zero(&self, scalar: ScalarElement, rows: u32, cols: u32) -> Tile {
        let coop = ElementType::coop_matrix(scalar, CoopMatrixRole::C, rows, cols);
        Tile::new(
            ExprKind::Cast {
                value: Box::new(Expr::new(
                    ExprKind::Literal(crate::ir::TileLiteral::f32(0.0)),
                    ElementType::F32,
                )),
                to: coop,
            },
            coop,
        )
    }

    /// Cooperatively load an A-role fragment from a region of a workgroup tile.
    pub(crate) fn coop_load_a(
        &self,
        tile: &WorkgroupTile,
        row: impl Into<Tile>,
        col: impl Into<Tile>,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
    ) -> Tile {
        self.coop_load_tile(CoopMatrixRole::A, tile, row, col, scalar, rows, cols)
    }

    /// Cooperatively load a B-role fragment from a region of a workgroup tile.
    pub(crate) fn coop_load_b(
        &self,
        tile: &WorkgroupTile,
        row: impl Into<Tile>,
        col: impl Into<Tile>,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
    ) -> Tile {
        self.coop_load_tile(CoopMatrixRole::B, tile, row, col, scalar, rows, cols)
    }

    #[allow(clippy::too_many_arguments)]
    fn coop_load_tile(
        &self,
        role: CoopMatrixRole,
        tile: &WorkgroupTile,
        row: impl Into<Tile>,
        col: impl Into<Tile>,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
    ) -> Tile {
        assert!(rows == 8 || rows == 16, "coop rows must be 8 or 16");
        assert!(cols == 8 || cols == 16, "coop cols must be 8 or 16");
        Tile::new(
            ExprKind::CoopLoad {
                role,
                scalar,
                rows,
                cols,
                src: CoopSrc::TileRegion {
                    tile: tile.decl().clone(),
                    row: boxed_index(row),
                    col: boxed_index(col),
                },
            },
            ElementType::coop_matrix(scalar, role, rows, cols),
        )
    }

    /// Cooperatively load a C-role fragment from a rank-1 storage vector,
    /// broadcasting the selected columns across all fragment rows.
    pub(crate) fn coop_load_c_broadcast(
        &self,
        src: &Storage,
        col: impl Into<Tile>,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
    ) -> Tile {
        assert!(rows == 8 || rows == 16, "coop rows must be 8 or 16");
        assert!(cols == 8 || cols == 16, "coop cols must be 8 or 16");
        Tile::new(
            ExprKind::CoopLoad {
                role: CoopMatrixRole::C,
                scalar,
                rows,
                cols,
                src: CoopSrc::BroadcastCol {
                    src: src.view().clone(),
                    col: boxed_index(col),
                },
            },
            ElementType::coop_matrix(scalar, CoopMatrixRole::C, rows, cols),
        )
    }

    /// `a * b + c` over cooperative fragments — value-producing. Compose with
    /// `coop_store_local(acc, coop_mma(a, b, coop_load_local(acc)))`.
    pub(crate) fn coop_mma(&self, a: Tile, b: Tile, c: Tile) -> Tile {
        let ty = c.element();
        Tile::new(
            ExprKind::CoopMma {
                a: Box::new(a.into_expr()),
                b: Box::new(b.into_expr()),
                c: Box::new(c.into_expr()),
            },
            ty,
        )
    }

    /// Cooperatively store an accumulator to `dst` at `(row, col)`. A distinct
    /// collective primitive — never a per-lane store.
    pub(crate) fn coop_store(
        &mut self,
        acc: &CoopAcc,
        dst: &Storage,
        row: impl Into<Tile>,
        col: impl Into<Tile>,
    ) {
        self.push_stmt(Stmt::CoopStore {
            acc: acc.decl().clone(),
            dst: dst.view().clone(),
            addr: Addr::Rc2 {
                row: boxed_index(row),
                col: boxed_index(col),
            },
        });
    }
}
