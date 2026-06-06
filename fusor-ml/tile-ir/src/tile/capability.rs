use super::value::{CoopAcc, WorkgroupTile};
use super::{Storage, Tile, TileBlock};
use crate::ir::{ScalarElement, TileReduceOp};

/// Capability token for tile-IR operations that require WebGPU subgroup
/// support.
///
/// Safe constructors live in higher-level device code. The unchecked
/// constructor exists so crates that own device capability validation can pass
/// that proof into tile-IR without adding a dependency from tile-IR back to the
/// runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SubgroupToken {
    _private: (),
}

impl SubgroupToken {
    /// Construct a subgroup token without checking device capabilities.
    pub fn new_unchecked() -> Self {
        Self { _private: () }
    }

    /// `@builtin(subgroup_id)`.
    pub fn subgroup_id(self, program: &TileBlock<'_>) -> Tile {
        program.subgroup_id()
    }

    /// `@builtin(subgroup_invocation_id)`.
    pub fn subgroup_lane(self, program: &TileBlock<'_>) -> Tile {
        program.subgroup_lane()
    }

    /// `@builtin(subgroup_size)`.
    pub fn subgroup_size(self, program: &TileBlock<'_>) -> Tile {
        program.subgroup_size()
    }

    /// `@builtin(num_subgroups)`.
    pub fn num_subgroups(self, program: &TileBlock<'_>) -> Tile {
        program.num_subgroups()
    }

    /// Reduction across one subgroup.
    pub fn subgroup_reduce(self, program: &TileBlock<'_>, op: TileReduceOp, value: Tile) -> Tile {
        program.subgroup_reduce(op, value)
    }

    /// Sum reduction across one subgroup.
    pub fn subgroup_reduce_sum(self, program: &TileBlock<'_>, value: Tile) -> Tile {
        self.subgroup_reduce(program, TileReduceOp::Sum, value)
    }

    /// Max reduction across one subgroup.
    pub fn subgroup_reduce_max(self, program: &TileBlock<'_>, value: Tile) -> Tile {
        self.subgroup_reduce(program, TileReduceOp::Max, value)
    }

    /// Min reduction across one subgroup.
    pub fn subgroup_reduce_min(self, program: &TileBlock<'_>, value: Tile) -> Tile {
        self.subgroup_reduce(program, TileReduceOp::Min, value)
    }
}

/// Capability token for tile-IR cooperative-matrix operations.
///
/// Safe constructors live in higher-level device code. The unchecked
/// constructor exists so crates that own device capability validation can pass
/// that proof into tile-IR without adding a dependency from tile-IR back to the
/// runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CoopMatrixToken {
    _private: (),
}

impl CoopMatrixToken {
    /// Construct a cooperative-matrix token without checking device
    /// capabilities.
    pub fn new_unchecked() -> Self {
        Self { _private: () }
    }

    /// Allocate a cooperative-matrix accumulator local.
    pub fn alloc_coop_acc(
        self,
        program: &mut TileBlock<'_>,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
    ) -> CoopAcc {
        program.alloc_coop_acc(scalar, rows, cols)
    }

    /// Load the current SSA value of a coop accumulator.
    pub fn load_local_coop(self, program: &TileBlock<'_>, acc: &CoopAcc) -> Tile {
        program.load_local_coop(acc)
    }

    /// Store a coop value into an accumulator.
    pub fn store_local_coop(self, program: &mut TileBlock<'_>, acc: &CoopAcc, value: Tile) {
        program.store_local_coop(acc, value);
    }

    /// A zeroed coop-`C` accumulator value.
    pub fn coop_zero(
        self,
        program: &TileBlock<'_>,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
    ) -> Tile {
        program.coop_zero(scalar, rows, cols)
    }

    /// Cooperatively load an A-role fragment from a workgroup tile.
    #[allow(clippy::too_many_arguments)]
    pub fn coop_load_a(
        self,
        program: &TileBlock<'_>,
        tile: &WorkgroupTile,
        row: impl Into<Tile>,
        col: impl Into<Tile>,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
    ) -> Tile {
        program.coop_load_a(tile, row, col, scalar, rows, cols)
    }

    /// Cooperatively load a B-role fragment from a workgroup tile.
    #[allow(clippy::too_many_arguments)]
    pub fn coop_load_b(
        self,
        program: &TileBlock<'_>,
        tile: &WorkgroupTile,
        row: impl Into<Tile>,
        col: impl Into<Tile>,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
    ) -> Tile {
        program.coop_load_b(tile, row, col, scalar, rows, cols)
    }

    /// Cooperatively load a C-role fragment from a rank-1 storage vector.
    pub fn coop_load_c_broadcast(
        self,
        program: &TileBlock<'_>,
        src: &Storage,
        col: impl Into<Tile>,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
    ) -> Tile {
        program.coop_load_c_broadcast(src, col, scalar, rows, cols)
    }

    /// `a * b + c` over cooperative fragments.
    pub fn coop_mma(self, program: &TileBlock<'_>, a: Tile, b: Tile, c: Tile) -> Tile {
        program.coop_mma(a, b, c)
    }

    /// Cooperatively store an accumulator to `dst` at `(row, col)`.
    pub fn coop_store(
        self,
        program: &mut TileBlock<'_>,
        acc: &CoopAcc,
        dst: &Storage,
        row: impl Into<Tile>,
        col: impl Into<Tile>,
    ) {
        program.coop_store(acc, dst, row, col);
    }
}
