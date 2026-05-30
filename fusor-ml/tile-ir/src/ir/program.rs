use super::*;
use crate::{LowerError, NagaKernel, QuantizedMatrix};

/// A typed kernel IR emitted by the tile builder.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct KernelIr {
    pub(crate) buffers: Vec<BufferDecl>,
    pub(crate) tiles: Vec<TileDecl>,
    pub(crate) locals: Vec<LocalRef>,
    pub(crate) body: TileProgramOp,
}

impl KernelIr {
    /// Storage buffer declarations bound by the kernel.
    pub fn buffers(&self) -> &[BufferDecl] {
        &self.buffers
    }

    /// Workgroup tile declarations allocated by the kernel.
    pub fn tiles(&self) -> &[TileDecl] {
        &self.tiles
    }

    /// Private scalar/vector locals allocated by tiled programs.
    pub fn locals(&self) -> &[LocalRef] {
        &self.locals
    }

    /// The single tile program that forms the kernel body.
    pub fn body(&self) -> &TileProgramOp {
        &self.body
    }

    /// Lower this IR into a validated Naga module.
    pub fn lower_to_naga(&self) -> Result<NagaKernel, LowerError> {
        crate::lower::lower_to_naga(self)
    }
}

/// Multiplicand role for a cooperatively-loaded matrix fragment.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CoopOperandRole {
    /// Left-hand MMA operand.
    A,
    /// Right-hand MMA operand.
    B,
    /// Accumulator/post-MMA operand.
    C,
}

/// A Triton-like source tile program over one workgroup tile.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TileProgramOp {
    /// Dispatch grid.
    pub grid: [u32; 3],
    /// Workgroup invocation count.
    pub block: u32,
    /// Program statements.
    pub body: Vec<TileStmt>,
}

impl Default for TileProgramOp {
    fn default() -> Self {
        Self {
            grid: [1, 1, 1],
            block: 0,
            body: Vec::new(),
        }
    }
}

/// Source of a `TileStmt::CopyToWorkgroupTile`. The lowerer dispatches the
/// per-element copy on this variant — `Quantized` dequantizes on the fly.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CopySource {
    /// Dense storage source.
    Storage(StorageView),
    /// Quantized matrix source.
    Quantized(QuantizedMatrix),
}

/// One ordered statement in a tile program.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TileStmt {
    /// Per-lane masked storage write.
    Store(TileStoreStmt),
    /// Per-lane masked rank-1 storage write.
    StoreIndexed(TileIndexedStoreStmt),
    /// Store to a private per-invocation local. The kernel-builder uses this
    /// for both first-write SSA bindings (via `program.bind`) and rebinds.
    StoreLocal { dst: LocalRef, value: Expr },
    /// Store to a workgroup scratch tile at a dynamic flat index.
    StoreWorkgroup {
        dst: TileRef,
        index: Box<Expr>,
        value: Expr,
    },
    /// Per-invocation control flow.
    If {
        condition: Expr,
        accept: Vec<TileStmt>,
        reject: Vec<TileStmt>,
    },
    /// Unstructured loop. Body may contain `Break` and `Return`.
    Loop { body: Vec<TileStmt> },
    /// Break out of the innermost `Loop`.
    Break,
    /// Return from the kernel entry point.
    Return,
    /// Zero-initialize a coop accumulator. `acc` is a local declared with
    /// element `ElementType::CoopMatrix { role: C, .. }`.
    ZeroCoopAcc { acc: LocalRef },
    /// Cooperatively copy a workgroup-tile-sized region of a storage view into
    /// a workgroup tile (one element per invocation per pass). `row_offset`
    /// and `col_offset` are evaluated in the surrounding scope.
    CopyToWorkgroupTile {
        dst: TileRef,
        src: CopySource,
        row_offset: Box<Expr>,
        col_offset: Box<Expr>,
    },
    /// Workgroup-scope memory barrier.
    Barrier,
    /// Cooperatively load an 8x8 fragment from a workgroup tile and bind it to
    /// `id` for subsequent `Mma` references in the same scope.
    LoadCoop {
        id: CoopFragmentId,
        role: CoopOperandRole,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
        tile: TileRef,
        row: Box<Expr>,
        col: Box<Expr>,
    },
    /// Cooperatively load a C-role fragment from a rank-1 storage vector,
    /// broadcasting the selected columns across all rows.
    LoadCoopBroadcast {
        id: CoopFragmentId,
        role: CoopOperandRole,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
        src: StorageView,
        col: Box<Expr>,
    },
    /// `acc += a * b` where `a`/`b` are previously loaded fragments.
    Mma {
        acc: LocalRef,
        a: CoopFragmentId,
        b: CoopFragmentId,
    },
    /// Initialize an accumulator from a C-role cooperative fragment.
    SetCoopAcc { acc: LocalRef, c: CoopFragmentId },
    /// Cooperatively store an accumulator to a global storage view.
    StoreCoopAcc {
        acc: LocalRef,
        dst: StorageView,
        row: Box<Expr>,
        col: Box<Expr>,
    },
    /// Iterator-driven loop over `0..count` that carries named, mutable
    /// accumulators.
    Fold {
        count: Box<Expr>,
        iter_var: LocalId,
        body: Vec<TileStmt>,
        accumulators: Vec<FoldAccumulator>,
    },
}

/// One accumulator carried by a `TileStmt::Fold`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FoldAccumulator {
    /// Local id carrying the accumulator.
    pub name: LocalId,
    /// Accumulator element type.
    pub element: ElementType,
    /// Initial value.
    pub init: Expr,
    /// Update expression evaluated each iteration.
    pub update: Expr,
}

/// A masked tile store emitted by a source tile program.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TileStoreStmt {
    /// Destination view.
    pub dst: StorageView,
    /// Row index.
    pub row: Box<Expr>,
    /// Column index.
    pub col: Box<Expr>,
    /// Stored value.
    pub value: Expr,
    /// Store mask.
    pub mask: Box<Expr>,
}

/// A masked rank-1 store emitted by a source tile program.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TileIndexedStoreStmt {
    /// Destination view.
    pub dst: StorageView,
    /// Linear index.
    pub index: Box<Expr>,
    /// Stored value.
    pub value: Expr,
    /// Store mask.
    pub mask: Box<Expr>,
}
