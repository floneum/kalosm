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
    pub(crate) body: Block,
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

/// What to iterate over inside a `Fold`. Initially just `Range`; future
/// variants (Chunks, Strided, Zip) compose without changing existing loop
/// shapes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileIter {
    /// Counted range `0..count` where `count` is a dynamic expression.
    Range { count: Box<TileExpr> },
}

/// Built-in u32 quantities that show up as leaves in index/address arithmetic.
/// Promoted to `TileExpr::Builtin` so a single expression type can host both
/// per-lane data and indexing math.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Builtin {
    /// `@builtin(local_invocation_index)` — flat lane within the workgroup.
    Lane,
    /// Current iteration counter of the innermost structured `Fold` / `Loop`.
    LoopIndex,
    /// `@builtin(workgroup_id).{x|y|z}`.
    ProgramId(WorkgroupAxis),
    /// `@builtin(subgroup_id)`.
    SubgroupId,
    /// `@builtin(subgroup_invocation_id)` — lane within the subgroup.
    SubgroupLane,
    /// `@builtin(subgroup_size)` — runtime subgroup size.
    SubgroupSize,
    /// `@builtin(num_subgroups)` — number of subgroups per workgroup.
    NumSubgroups,
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
    pub fn locals(&self) -> &[LocalDecl] {
        &self.locals
    }

    /// The structured root body of the kernel.
    pub fn body(&self) -> &Block {
        &self.body
    }

    /// Lower this IR into a validated Naga module.
    pub fn lower_to_naga(&self) -> Result<NagaKernel, LowerError> {
        crate::lower::lower_to_naga(self)
    }

    /// Return the dispatch grid for kernels that lower to a single tile program.
    pub fn single_tile_program_grid(&self) -> Option<[u32; 3]> {
        let [Op::TileProgram(program)] = self.body.ops.as_slice() else {
            return None;
        };
        Some(program.grid)
    }

    /// Best-effort element-type inference for a `TileExpr`, used by builder
    /// helpers like `pin` that need to allocate a typed local before the
    /// lowerer runs. Falls back to `F32` for variants that need additional
    /// context.
    pub(crate) fn tile_expr_element(&self, expr: &TileExpr) -> ElementType {
        match expr {
            TileExpr::Load(load) => load.src.buffer.element,
            TileExpr::LoadLinear(load) => load.src.buffer.element,
            TileExpr::LoadVec4(_) => ElementType::F32Vec4,
            TileExpr::LoadWorkgroup { src, .. } => src.element,
            TileExpr::LoadLocal(local) => local.element,
            TileExpr::QuantizedLoad(_) | TileExpr::Full(_) => ElementType::F32,
            TileExpr::Literal(value) => value.element(),
            TileExpr::Index(_) => ElementType::U32,
            TileExpr::Builtin(_) => ElementType::U32,
            TileExpr::Scalar(scalar) => match scalar {
                TileScalarExpr::Reduce { scratch, .. }
                | TileScalarExpr::LoopReduce { scratch, .. } => scratch.element,
                TileScalarExpr::Literal(value) => value.element(),
            },
            TileExpr::Unary { value, .. } | TileExpr::Binary { left: value, .. } => {
                self.tile_expr_element(value)
            }
            TileExpr::Sum { values } => values
                .first()
                .map(|value| self.tile_expr_element(value))
                .unwrap_or(ElementType::F32),
            TileExpr::Cast { to, .. } => *to,
            TileExpr::Bitcast { to, .. } => *to,
            TileExpr::Select { accept, .. } => self.tile_expr_element(accept),
            TileExpr::Compare { output, .. } => *output,
            TileExpr::LoopFold { initial, .. } => initial.element(),
            TileExpr::GroupReduce { scratch, .. } => scratch.element,
            TileExpr::SubgroupReduce { value, .. } => self.tile_expr_element(value),
            TileExpr::QuantizedBlockLane { .. } => ElementType::F32,
            TileExpr::Vec4Dot { .. }
            | TileExpr::QuantizedQ8_0Dot8 { .. }
            | TileExpr::QuantizedVecDot { .. }
            | TileExpr::QuantizedQ4KGgmlDot { .. }
            | TileExpr::QuantizedQ6KGgmlDot { .. } => ElementType::F32,
            TileExpr::Vec4Splat { .. } | TileExpr::Compose4 { .. } => ElementType::F32Vec4,
        }
    }
}

/// A structured sequence of typed IR operations.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Block {
    ops: Vec<Op>,
}

impl Block {
    /// Construct an empty block.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a block from an operation list.
    pub fn from_ops(ops: Vec<Op>) -> Self {
        Self { ops }
    }

    /// Operations in this block.
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    pub(crate) fn push(&mut self, op: Op) {
        self.ops.push(op);
    }
}

/// A typed IR operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Op {
    /// Triton-like source tile program over one workgroup tile.
    TileProgram(TileProgramOp),
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

/// One ordered statement in a tile program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileStmt {
    /// Per-lane masked storage write.
    Store(TileStoreStmt),
    /// Per-lane masked rank-1 storage write.
    StoreIndexed(TileIndexedStoreStmt),
    /// Store to a private per-invocation local.
    StoreLocal { dst: LocalRef, value: TileExpr },
    /// Bind `value` to a fresh local. Subsequent reads in the rest of this
    /// statement vec use `TileExpr::LoadLocal(LocalRef { id: name.id, .. })`.
    /// Lowers to a Store of `value` into the local; the local must be
    /// declared in `KernelIr.locals`.
    Let { name: LocalRef, value: TileExpr },
    /// Store to a workgroup scratch tile at a dynamic flat index.
    StoreWorkgroup {
        dst: TileRef,
        index: TileIndexExpr,
        value: TileExpr,
    },
    /// Per-invocation control flow.
    If {
        condition: TileExpr,
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
    /// K loop body they may reference `loop_index`).
    CopyToWorkgroupTile {
        dst: TileRef,
        src: StorageView,
        row_offset: TileIndexExpr,
        col_offset: TileIndexExpr,
    },
    /// Same as `CopyToWorkgroupTile` but dequantizing on the fly from a packed
    /// quantized matrix. `dst` must be an f32 workgroup tile.
    CopyQuantToWorkgroupTile {
        dst: TileRef,
        src: QuantizedMatrix,
        row_offset: TileIndexExpr,
        col_offset: TileIndexExpr,
    },
    /// Workgroup-scope memory barrier.
    Barrier,
    /// Cooperatively load an 8x8 fragment from a workgroup tile and bind it to
    /// `id` for subsequent `Mma` references in the same scope.
    LoadCoop {
        id: CoopFragmentId,
        role: CoopOperandRole,
        tile: TileRef,
        row: TileIndexExpr,
        col: TileIndexExpr,
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
        row: TileIndexExpr,
        col: TileIndexExpr,
    },
    /// Temporary generic loop form. Lowering emits `loop { ... }` with an
    /// explicit top-of-loop break when `TileIndexExpr::LoopIndex` reaches
    /// `max_iterations`.
    WhileTrue {
        max_iterations: u32,
        body: Vec<TileStmt>,
    },
    /// Iterator-driven loop that carries named, mutable accumulators. Each
    /// `accumulator.init` is evaluated in the surrounding scope and stored
    /// into the accumulator local. Inside `body` and inside each
    /// `accumulator.update`, references to the iterator value are
    /// `LoadLocal(LocalRef { id: iter_var, element: U32 })`, and references
    /// to the in-flight accumulator value are `LoadLocal` of the
    /// accumulator's name. After the statement finishes, the accumulator
    /// locals hold the final values and may be read as ordinary `LoadLocal`.
    Fold {
        iter: TileIter,
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
    pub init: TileExpr,
    pub update: TileExpr,
}

/// A masked tile store emitted by a source tile program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileStoreStmt {
    pub dst: StorageView,
    pub row: TileIndexExpr,
    pub col: TileIndexExpr,
    pub value: TileExpr,
    pub mask: TileMaskExpr,
}

/// A masked rank-1 store emitted by a source tile program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileIndexedStoreStmt {
    pub dst: StorageView,
    pub index: TileIndexExpr,
    pub value: TileExpr,
    pub mask: TileMaskExpr,
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

/// A typed scalar literal stored by bits so IR equality remains exact.
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

/// A rank-1 tile expression evaluated lane-wise.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileExpr {
    Load(TileLoadExpr),
    LoadLinear(TileLinearLoadExpr),
    LoadVec4(TileVec4LoadExpr),
    LoadWorkgroup {
        src: TileRef,
        index: TileIndexExpr,
    },
    LoadLocal(LocalRef),
    QuantizedLoad(TileQuantizedLoadExpr),
    Full(F32Bits),
    Literal(TileLiteral),
    Index(TileIndexExpr),
    Scalar(TileScalarExpr),
    /// A built-in u32 quantity (lane id, loop index, program id, subgroup
    /// builtins). Promoted from `TileIndexExpr` leaves so the same expression
    /// type spans index arithmetic and per-lane data.
    Builtin(Builtin),
    Unary {
        op: TileUnaryOp,
        value: Box<TileExpr>,
    },
    Binary {
        op: TileBinaryOp,
        left: Box<TileExpr>,
        right: Box<TileExpr>,
    },
    /// Left-associated sum of a flat value list. This represents long
    /// unrolled accumulations without forcing the lowerer to recurse through
    /// a deep binary tree.
    Sum {
        values: Vec<Box<TileExpr>>,
    },
    Cast {
        value: Box<TileExpr>,
        to: ElementType,
    },
    Bitcast {
        value: Box<TileExpr>,
        to: ElementType,
    },
    Select {
        condition: Box<TileExpr>,
        accept: Box<TileExpr>,
        reject: Box<TileExpr>,
    },
    Compare {
        op: TileCompareOp,
        left: Box<TileExpr>,
        right: Box<TileExpr>,
        output: ElementType,
    },
    LoopFold {
        op: TileReduceOp,
        iterations: u32,
        value: Box<TileExpr>,
        initial: TileLiteral,
    },
    GroupReduce {
        op: TileReduceOp,
        value: Box<TileExpr>,
        scratch: TileRef,
        group_size: u32,
    },
    /// Reduction across the lanes of one subgroup. Lowers to
    /// `subgroupAdd`/`subgroupMax`/`subgroupMin` — no shared-memory tree, no
    /// workgroup-shape divisibility constraint.
    SubgroupReduce {
        op: TileReduceOp,
        value: Box<TileExpr>,
    },
    /// One lane of a fused N-wide quantized dequant. All lanes of the same
    /// `id` share the block scale lookup; the lowerer emits the helper once
    /// and reuses the result across lanes.
    QuantizedBlockLane {
        id: BlockDequantId,
        src: QuantizedMatrix,
        k_base: TileIndexExpr,
        col: TileIndexExpr,
        mask: TileMaskExpr,
        fill: F32Bits,
        block_n: u32,
        lane: u32,
    },
    /// Dot product between two `vec4<f32>` expressions.
    Vec4Dot {
        left: Box<TileExpr>,
        right: Box<TileExpr>,
    },
    /// `vec4<f32>(value, value, value, value)`.
    Vec4Splat {
        value: Box<TileExpr>,
    },
    /// `vec4<f32>(values[0], values[1], values[2], values[3])`. Combined with
    /// `Vec4Dot` this expresses the fused 4-way dot product the qgemv
    /// accelerator emits.
    Compose4 {
        values: [Box<TileExpr>; 4],
    },
    QuantizedQ8_0Dot8 {
        a: [Box<TileExpr>; 8],
        src: QuantizedMatrix,
        k_base: TileIndexExpr,
        col: TileIndexExpr,
        mask: TileMaskExpr,
        fill: F32Bits,
    },
    QuantizedVecDot {
        kind: QuantizedVecDotKind,
        a: Vec<Box<TileExpr>>,
        src: QuantizedMatrix,
        k_base: TileIndexExpr,
        col: TileIndexExpr,
        mask: TileMaskExpr,
        fill: F32Bits,
        block_n: u32,
    },
    QuantizedQ4KGgmlDot {
        a_low: Vec<Box<TileExpr>>,
        a_high: Vec<Box<TileExpr>>,
        sums: Vec<Box<TileExpr>>,
        src: QuantizedMatrix,
        block: TileIndexExpr,
        iq: TileIndexExpr,
        ir: TileIndexExpr,
        col: TileIndexExpr,
        mask: TileMaskExpr,
        fill: F32Bits,
    },
    QuantizedQ6KGgmlDot {
        a: Vec<Box<TileExpr>>,
        src: QuantizedMatrix,
        block: TileIndexExpr,
        ip: TileIndexExpr,
        il: TileIndexExpr,
        col: TileIndexExpr,
        mask: TileMaskExpr,
        fill: F32Bits,
    },
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum QuantizedVecDotKind {
    Q8Activation,
    Q4KF32,
}

/// A masked rank-1 tile load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileLoadExpr {
    pub src: StorageView,
    pub row: TileIndexExpr,
    pub col: TileIndexExpr,
    pub mask: TileMaskExpr,
    pub fill: TileLiteral,
}

/// A masked rank-1 vec4 load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileVec4LoadExpr {
    pub src: StorageView,
    pub index: TileIndexExpr,
    pub mask: TileMaskExpr,
    pub fill: F32Bits,
}

/// A masked rank-1 storage load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileLinearLoadExpr {
    pub src: StorageView,
    pub index: TileIndexExpr,
    pub mask: TileMaskExpr,
    pub fill: TileLiteral,
}

/// A masked dequantizing rank-1 tile load from a packed quantized matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileQuantizedLoadExpr {
    pub src: QuantizedMatrix,
    pub row: TileIndexExpr,
    pub col: TileIndexExpr,
    pub mask: TileMaskExpr,
    pub fill: F32Bits,
}

/// A scalar value derived from a tile expression.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileScalarExpr {
    Reduce {
        op: TileReduceOp,
        value: Box<TileExpr>,
        scratch: TileRef,
    },
    LoopReduce {
        op: TileReduceOp,
        iterations: u32,
        value: Box<TileExpr>,
        scratch: TileRef,
    },
    Literal(TileLiteral),
}

/// Integer index expression over program ids and the current lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileIndexExpr {
    Lane,
    LoopIndex,
    ProgramId(WorkgroupAxis),
    /// `@builtin(subgroup_id)` — index of the subgroup within the workgroup.
    SubgroupId,
    /// `@builtin(subgroup_invocation_id)` — lane within the current subgroup.
    SubgroupLane,
    /// `@builtin(subgroup_size)` — runtime subgroup size.
    SubgroupSize,
    /// `@builtin(num_subgroups)` — number of subgroups per workgroup.
    NumSubgroups,
    Literal(u32),
    Add(Box<TileIndexExpr>, Box<TileIndexExpr>),
    Mul(Box<TileIndexExpr>, u32),
    Div(Box<TileIndexExpr>, u32),
    Mod(Box<TileIndexExpr>, u32),
    Value(Box<TileExpr>),
}

/// Boolean mask expression.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileMaskExpr {
    True,
    Compare {
        op: TileCompareOp,
        left: TileIndexExpr,
        right: TileIndexExpr,
    },
    And(Box<TileMaskExpr>, Box<TileMaskExpr>),
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

mod layout;
pub use layout::{Layout, MemoryLevel, Shape, Strides, TileLevel, TileOrigin};
