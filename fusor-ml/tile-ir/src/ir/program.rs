use super::{Addr, Buffer, Expr, Local, StorageView, Tile};
use crate::{LowerError, NagaKernel};

/// A typed kernel IR emitted by the tile builder.
///
/// The IR is a self-contained tree: declarations are `Rc`-owned at their use
/// sites. Buffers are also retained in declaration order so runtime binding
/// lists produced by [`crate::KernelBuilder`] stay aligned even when a declared
/// storage is optimized out of the body.
#[derive(Clone, Debug)]
pub struct KernelIr {
    /// Storage declarations in builder declaration order.
    pub(crate) buffers: Vec<Buffer>,
    /// Dispatch grid.
    pub grid: [u32; 3],
    /// Workgroup invocation count.
    pub block: u32,
    /// Program statements.
    pub body: Vec<Stmt>,
}

impl Default for KernelIr {
    fn default() -> Self {
        Self {
            buffers: Vec::new(),
            grid: [1, 1, 1],
            block: 0,
            body: Vec::new(),
        }
    }
}

impl KernelIr {
    /// The statements that form the kernel body.
    pub fn body(&self) -> &[Stmt] {
        &self.body
    }

    /// Lower this IR into a validated Naga module.
    pub fn lower_to_naga(&self) -> Result<NagaKernel, LowerError> {
        crate::lower::lower_to_naga(self)
    }
}

/// One accumulator carried by a counted `Stmt::Loop`.
#[derive(Clone, Debug)]
pub struct Accumulator {
    /// Local carrying the accumulator (owned by the loop).
    pub local: Local,
    /// Initial value.
    pub init: Expr,
    /// Update expression evaluated each iteration.
    pub update: Expr,
}

/// One ordered statement in a tile program.
#[derive(Clone, Debug)]
pub enum Stmt {
    /// Per-lane masked storage write — dense or indexed (the `addr` variant
    /// selects), never a collective store.
    Store {
        /// Destination view.
        dst: StorageView,
        /// Linear or rank-2 address.
        addr: Addr,
        /// Stored value.
        value: Expr,
        /// Store mask.
        mask: Box<Expr>,
    },
    /// Store to a private per-invocation local. Used for first-write SSA
    /// bindings and rebinds, and as the coop-accumulator verb (zero = store a
    /// coop-zero, set = store a `CoopLoad{role:C}`, mma = store a `CoopMma`).
    StoreLocal {
        /// Destination local.
        dst: Local,
        /// Stored value.
        value: Expr,
    },
    /// Store to a workgroup scratch tile at a dynamic flat index.
    StoreTile {
        /// Destination tile.
        dst: Tile,
        /// Flat index into the tile.
        index: Box<Expr>,
        /// Stored value.
        value: Expr,
    },
    /// Fill a workgroup tile from `value` (a masked `Load`, dense or quant).
    /// Coords are derived by the lowerer; the vec4 fast path is internal.
    FillTile {
        /// Destination tile.
        dst: Tile,
        /// Per-element source value (typically a masked `Load`).
        value: Expr,
    },
    /// Cooperatively store an accumulator to a global storage view. A distinct
    /// subgroup-collective primitive — never lowered as a per-lane `Store`.
    CoopStore {
        /// Accumulator local.
        acc: Local,
        /// Destination view.
        dst: StorageView,
        /// Destination address.
        addr: Addr,
    },
    /// Per-invocation control flow.
    If {
        /// Bool condition.
        condition: Expr,
        /// Statements run when true.
        accept: Vec<Stmt>,
        /// Statements run when false.
        reject: Vec<Stmt>,
    },
    /// One loop node, two forms:
    /// - `count: Some(..)` => counted, iterating `0..count` into `index`.
    /// - `count: None` => unstructured; body may contain `Break`/`Return`.
    Loop {
        /// Iteration count, or `None` for an unstructured loop.
        count: Option<Expr>,
        /// Loop index local (owned by the loop), when counted.
        index: Option<Local>,
        /// Carried accumulators.
        accumulators: Vec<Accumulator>,
        /// Loop body.
        body: Vec<Stmt>,
    },
    /// Break out of the innermost `Loop` (data-dependent early exit).
    Break,
    /// Return from the kernel entry point.
    Return,
    /// Workgroup-scope memory barrier.
    Barrier,
}
