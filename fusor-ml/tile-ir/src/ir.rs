use crate::{LowerError, NagaKernel, QuantizedMatrix};

macro_rules! id_newtype {
    ($(#[$meta:meta])* $vis:vis $name:ident $(, $derive:ident)*) => {
        $(#[$meta])*
        #[derive(Copy, Clone, Debug, PartialEq, Eq $(, $derive)*)]
        $vis struct $name(pub(crate) u32);

        impl $name {
            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

macro_rules! numeric_markers {
    ($(($(#[$meta:meta])* $name:ident, $element:expr)),+ $(,)?) => {
        $(
            $(#[$meta])*
            #[derive(Copy, Clone, Debug)]
            pub struct $name;

            impl Numeric for $name {
                const ELEMENT: ElementType = $element;
            }
        )+
    };
}

/// A typed kernel IR emitted by the tile builder.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelIr {
    pub(crate) buffers: Vec<BufferDecl>,
    pub(crate) tiles: Vec<TileDecl>,
    pub(crate) locals: Vec<LocalDecl>,
    pub(crate) body: TileProgramOp,
}

id_newtype!(
    /// Identifier of a cooperatively-loaded fragment (SSA-cached).
    pub CoopFragmentId, Hash
);

/// Multiplicand role for a cooperatively-loaded matrix fragment.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CoopOperandRole {
    A,
    B,
}

id_newtype!(
    /// Identifier shared by lanes of one fused quantized-block dequant.
    pub BlockDequantId, Hash
);

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
    pub fn locals(&self) -> &[LocalDecl] {
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

    /// Return the dispatch grid for kernels that lower to a single tile program.
    pub fn single_tile_program_grid(&self) -> Option<[u32; 3]> {
        Some(self.body.grid)
    }

    /// Best-effort element-type inference for a `Expr`, used by builder
    /// helpers like `pin` that need to allocate a typed local before the
    /// lowerer runs. Falls back to `F32` for variants that need additional
    /// context.
    pub(crate) fn tile_expr_element(&self, expr: &Expr) -> ElementType {
        match expr {
            Expr::Load(load) => load.src.buffer.element,
            Expr::LoadLinear(load) => load.src.buffer.element,
            Expr::LoadWorkgroup { src, .. } => src.element,
            Expr::LoadLocal(local) => local.element,
            Expr::QuantizedLoad(_) => ElementType::F32,
            Expr::Literal(value) => value.element(),
            Expr::Builtin(_) => ElementType::U32,
            Expr::Reduce { scratch, .. } => scratch.element,
            Expr::Unary { value, .. } | Expr::Binary { left: value, .. } => {
                self.tile_expr_element(value)
            }
            Expr::Cast { to, .. } => *to,
            Expr::Bitcast { to, .. } => *to,
            Expr::Select { accept, .. } => self.tile_expr_element(accept),
            Expr::Compare { output, .. } => *output,
            Expr::GroupReduce { scratch, .. } => scratch.element,
            Expr::SubgroupReduce { value, .. } => self.tile_expr_element(value),
            Expr::QuantizedBlockLane { .. } => ElementType::F32,
            Expr::Vec4Dot { .. } | Expr::QuantizedDot { .. } => ElementType::F32,
            Expr::Compose4 { .. } => ElementType::F32Vec4,
        }
    }
}

/// A storage buffer declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BufferDecl {
    pub id: BufferId,
    pub element: ElementType,
    pub layout: Layout,
    pub access: BufferAccess,
}

/// A storage buffer reference.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BufferRef {
    pub id: BufferId,
    pub element: ElementType,
}

impl BufferRef {
    /// Create a typed reference to an existing buffer declaration.
    pub const fn new(id: BufferId, element: ElementType) -> Self {
        Self { id, element }
    }
}

id_newtype!(
    /// A storage buffer identifier.
    pub BufferId
);

/// Access required for a storage buffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BufferAccess {
    Read,
    ReadWrite,
}

/// A typed workgroup tile declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileDecl {
    pub id: TileId,
    pub element: ElementType,
    pub layout: Layout,
    pub level: TileLevel,
    pub origin: TileOrigin,
}

/// A typed reference to a tile declaration.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TileRef {
    pub id: TileId,
    pub element: ElementType,
}

/// A private per-invocation local declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LocalDecl {
    pub id: LocalId,
    pub element: ElementType,
}

/// A typed private local reference.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LocalRef {
    pub id: LocalId,
    pub element: ElementType,
}

impl LocalRef {
    pub const fn new(id: LocalId, element: ElementType) -> Self {
        Self { id, element }
    }
}

id_newtype!(
    /// A private local identifier.
    pub LocalId, Hash
);

impl TileRef {
    /// Create a typed reference to an existing tile declaration.
    pub const fn new(id: TileId, element: ElementType) -> Self {
        Self { id, element }
    }
}

/// A shaped view into a storage buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageView {
    pub buffer: BufferRef,
    pub offset: u32,
    pub layout: Layout,
    pub dynamic_offsets: Vec<Option<DynamicOffset>>,
    pub index_map: Option<StorageIndexMap>,
}

impl StorageView {
    /// Construct a storage view with no dynamic workgroup offset.
    pub fn root(buffer: BufferRef, layout: Layout) -> Self {
        let dynamic_offsets = vec![None; layout.shape().rank()];
        Self {
            buffer,
            offset: 0,
            layout,
            dynamic_offsets,
            index_map: None,
        }
    }
}

/// Non-affine logical-to-storage mappings for matrix views.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StorageIndexMap {
    Im2ColNhwc(Im2ColNhwcMap),
    FlattenedMatrix(FlattenedMatrixMap),
}

/// Rank-N tensor viewed as a rank-2 matrix by flattening every axis except the
/// final column axis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlattenedMatrixMap {
    pub prefix_shape: Vec<u32>,
    pub prefix_strides: Vec<u32>,
    pub column_stride: u32,
}

/// NHWC convolution activation view lowered as an im2col matrix.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Im2ColNhwcMap {
    pub out_h: u32,
    pub out_w: u32,
    pub kernel_h: u32,
    pub kernel_w: u32,
    pub channels: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub dilation_h: u32,
    pub dilation_w: u32,
    pub batch_stride: u32,
    pub row_stride: u32,
    pub col_stride: u32,
    pub channel_stride: u32,
}

/// Dynamic coordinate offset used by storage views.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DynamicOffset {
    /// Offset derived from `@builtin(workgroup_id)`.
    Workgroup(WorkgroupOffset),
    /// Offset derived from the innermost IR loop induction variable.
    Loop(LoopOffset),
}

impl From<WorkgroupOffset> for DynamicOffset {
    fn from(offset: WorkgroupOffset) -> Self {
        Self::Workgroup(offset)
    }
}

impl From<LoopOffset> for DynamicOffset {
    fn from(offset: LoopOffset) -> Self {
        Self::Loop(offset)
    }
}

/// Dynamic coordinate offset derived from `@builtin(workgroup_id)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct WorkgroupOffset {
    pub axis: WorkgroupAxis,
    pub scale: u32,
}

impl WorkgroupOffset {
    /// Offset an axis by `workgroup_id.axis * scale`.
    pub const fn new(axis: WorkgroupAxis, scale: u32) -> Self {
        Self { axis, scale }
    }
}

/// Dynamic coordinate offset derived from the current loop induction.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct LoopOffset {
    pub scale: u32,
}

impl LoopOffset {
    /// Offset an axis by `loop_index * scale`.
    pub const fn new(scale: u32) -> Self {
        Self { scale }
    }
}

/// Axis of `@builtin(workgroup_id)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum WorkgroupAxis {
    X,
    Y,
    Z,
}

impl WorkgroupAxis {
    pub(crate) const fn index(self) -> u32 {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}

/// A Triton-like source tile program over one workgroup tile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileProgramOp {
    pub grid: [u32; 3],
    pub block: u32,
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CopySource {
    Storage(StorageView),
    Quantized(QuantizedMatrix),
}

/// One ordered statement in a tile program.
#[derive(Clone, Debug, PartialEq, Eq)]
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
    /// element `ElementType::CoopMatrixF32 { .. }`.
    ZeroCoopAcc { acc: LocalRef },
    /// Cooperatively copy a workgroup-tile-sized region of a storage view into
    /// a workgroup tile (one element per invocation per pass). `row_offset`
    /// and `col_offset` are evaluated in the surrounding scope (e.g. inside a
    /// K loop body they may reference `loop_index`). The lowerer dispatches on
    /// `src` — `CopySource::Quantized` dequantizes on the fly and requires an
    /// `f32` `dst` tile.
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
        tile: TileRef,
        row: Box<Expr>,
        col: Box<Expr>,
    },
    /// `acc += a * b` where `a`/`b` are previously loaded fragments. Letting
    /// the user load fragments separately lets one A/B load be reused across
    /// many MMAs (e.g. across the inner row × col grid in qmatmul). `acc` is
    /// a local declared with element `ElementType::CoopMatrixF32 { .. }`.
    Mma {
        acc: LocalRef,
        a: CoopFragmentId,
        b: CoopFragmentId,
    },
    /// Cooperatively store an accumulator to a global storage view. `acc` is
    /// a local declared with element `ElementType::CoopMatrixF32 { .. }`.
    StoreCoopAcc {
        acc: LocalRef,
        dst: StorageView,
        row: Box<Expr>,
        col: Box<Expr>,
    },
    /// Iterator-driven loop over `0..count` that carries named, mutable
    /// accumulators. Each `accumulator.init` is evaluated in the surrounding
    /// scope and stored into the accumulator local. Inside `body` and inside
    /// each `accumulator.update`, references to the iterator value are
    /// `LoadLocal(LocalRef { id: iter_var, element: U32 })`, and references
    /// to the in-flight accumulator value are `LoadLocal` of the
    /// accumulator's name. After the statement finishes, the accumulator
    /// locals hold the final values and may be read as ordinary `LoadLocal`.
    Fold {
        count: Box<Expr>,
        iter_var: LocalId,
        body: Vec<TileStmt>,
        accumulators: Vec<FoldAccumulator>,
    },
}

/// One accumulator carried by a `TileStmt::Fold`. `init` is evaluated once
/// in the surrounding scope. `update` is evaluated each iteration with
/// `iter_var` and the current accumulator values in scope (via
/// `LoadLocal`). The result of `update` becomes the accumulator's value for
/// the next iteration; after the loop, it is the final value visible
/// outside the fold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldAccumulator {
    pub name: LocalId,
    pub element: ElementType,
    pub init: Expr,
    pub update: Expr,
}

/// A masked tile store emitted by a source tile program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileStoreStmt {
    pub dst: StorageView,
    pub row: Box<Expr>,
    pub col: Box<Expr>,
    pub value: Expr,
    pub mask: Box<Expr>,
}

/// A masked rank-1 store emitted by a source tile program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileIndexedStoreStmt {
    pub dst: StorageView,
    pub index: Box<Expr>,
    pub value: Expr,
    pub mask: Box<Expr>,
}

/// Floating point literal stored by bits so IR equality remains exact.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct F32Bits(pub u32);

impl F32Bits {
    pub fn new(value: f32) -> Self {
        Self(value.to_bits())
    }

    pub fn get(self) -> f32 {
        f32::from_bits(self.0)
    }
}

/// A typed scalar literal stored by bits so IR equality remains exact. Vector
/// constants are not literals — they are built by composing scalar literals
/// (e.g. `Expr::Compose4`).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TileLiteral {
    F32(F32Bits),
    F16(u16),
    U32(u32),
    Bool(bool),
}

impl TileLiteral {
    pub const fn element(self) -> ElementType {
        match self {
            Self::F32(_) => ElementType::F32,
            Self::F16(_) => ElementType::F16,
            Self::U32(_) => ElementType::U32,
            Self::Bool(_) => ElementType::Bool,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TileUnaryOp {
    Exp,
    Exp2,
    Log,
    Log2,
    Sqrt,
    InverseSqrt,
    Sin,
    Cos,
    Tan,
    Tanh,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Asinh,
    Acosh,
    Atanh,
    Abs,
    Neg,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TileBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    Min,
    Max,
    BitAnd,
    BitOr,
    BitXor,
    LogicalAnd,
    LogicalOr,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TileReduceOp {
    Sum,
    Product,
    Max,
    Min,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TileCompareOp {
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

id_newtype!(
    /// A tiny tile identifier for the typed IR.
    pub TileId
);

/// Element types represented by the typed IR.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ElementType {
    F32,
    F16,
    U32,
    F32Vec4,
    Bool,
    /// Cooperative-matrix accumulator type (the `C`-role fragment) of the
    /// given shape. Only `8x8` f32 is currently supported by the lowerer; the
    /// shape is carried explicitly so a single `LocalDecl` describes the full
    /// fragment type.
    CoopMatrixF32 {
        rows: u32,
        cols: u32,
    },
}

/// Numeric element markers that can appear in the typed IR.
pub trait Numeric {
    const ELEMENT: ElementType;
}

numeric_markers!(
    (
        /// A sample numeric marker.
        F32,
        ElementType::F32
    ),
    (
        /// Half-precision floating point storage marker.
        F16,
        ElementType::F16
    ),
    (
        /// Packed u32 storage marker.
        U32,
        ElementType::U32
    ),
    (
        /// Four packed f32 values stored as one storage element.
        F32Vec4,
        ElementType::F32Vec4
    ),
    (
        /// Boolean private/control value marker.
        Bool,
        ElementType::Bool
    ),
);

mod expr;
pub use expr::{
    Builtin, DotK, Expr, PackedActivations, TileLinearLoadExpr, TileLoadExpr,
    TileQuantizedLoadExpr,
};

mod layout;
pub use layout::{Layout, MemoryLevel, Shape, Strides, TileLevel, TileOrigin};
