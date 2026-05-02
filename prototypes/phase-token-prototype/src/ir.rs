use std::num::NonZeroU32;

use crate::{LowerError, NagaKernel, QuantizedMatrix};

/// A typed kernel IR emitted by the prototype builder.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelIr {
    pub(crate) buffers: Vec<BufferDecl>,
    pub(crate) tiles: Vec<TileDecl>,
    pub(crate) body: Block,
    pub(crate) next_buffer: u32,
    pub(crate) next_tile: u32,
    pub(crate) next_block_dequant: u32,
    /// Side table of pinned subexpressions, indexed by `PinId`. The lowerer
    /// emits each entry into a private local on first reference and reuses the
    /// load on subsequent references in the same scope.
    pub(crate) pinned_values: Vec<TileExpr>,
    /// Side table of multi-output loop-fold groups, indexed by
    /// `LoopFoldGroupId`. Lowering materializes one Naga loop per group with N
    /// parallel accumulators.
    pub(crate) loop_fold_groups: Vec<LoopFoldGroup>,
    /// Declared cooperative-matrix accumulators. Each entry maps to an 8x8 f32
    /// CooperativeMatrix-typed function local.
    pub(crate) coop_accs: Vec<CoopAccDecl>,
    /// Counter for cooperative-matrix fragment SSA names.
    pub(crate) next_coop_fragment: u32,
}

/// Identifier of a cooperative-matrix accumulator local.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CoopAccId(pub(crate) u32);

impl CoopAccId {
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Declaration of a cooperative-matrix accumulator. Currently only 8x8 f32
/// `C`-role fragments are supported.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CoopAccDecl {
    pub id: CoopAccId,
    pub rows: u32,
    pub cols: u32,
}

/// Identifier of a cooperatively-loaded fragment (SSA-cached).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct CoopFragmentId(pub(crate) u32);

impl CoopFragmentId {
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Multiplicand role for a cooperatively-loaded matrix fragment.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CoopOperandRole {
    A,
    B,
}

/// Identifier of a pinned subexpression.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PinId(pub(crate) u32);

impl PinId {
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Identifier of a multi-output loop-fold group.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LoopFoldGroupId(pub(crate) u32);

impl LoopFoldGroupId {
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// A K-loop that accumulates N parallel reductions sharing one body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopFoldGroup {
    pub iterations: u32,
    pub op: TileReduceOp,
    pub initials: Vec<TileLiteral>,
    pub bodies: Vec<TileExpr>,
}

/// Identifier shared by lanes of one fused quantized-block dequant.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BlockDequantId(pub(crate) u32);

impl BlockDequantId {
    pub const fn index(self) -> usize {
        self.0 as usize
    }
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

    /// The structured root body of the kernel.
    pub fn body(&self) -> &Block {
        &self.body
    }

    /// Cooperative-matrix accumulators declared by the kernel body.
    pub fn coop_accs(&self) -> &[CoopAccDecl] {
        &self.coop_accs
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

/// A storage buffer identifier.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BufferId(pub(crate) u32);

impl BufferId {
    /// The dense index for this buffer declaration.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

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
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
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
    /// Zero-initialize a coop accumulator.
    ZeroCoopAcc { id: CoopAccId },
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
    /// many MMAs (e.g. across the inner row × col grid in qmatmul).
    Mma {
        acc: CoopAccId,
        a: CoopFragmentId,
        b: CoopFragmentId,
    },
    /// Cooperatively store an accumulator to a global storage view.
    StoreCoopAcc {
        acc: CoopAccId,
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
}

impl TileLiteral {
    pub const fn element(self) -> ElementType {
        match self {
            Self::F32(_) => ElementType::F32,
            Self::F16(_) => ElementType::F16,
            Self::U32(_) => ElementType::U32,
        }
    }
}

/// A rank-1 tile expression evaluated lane-wise.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileExpr {
    Load(TileLoadExpr),
    QuantizedLoad(TileQuantizedLoadExpr),
    Full(F32Bits),
    Literal(TileLiteral),
    Index(TileIndexExpr),
    Scalar(TileScalarExpr),
    Unary {
        op: TileUnaryOp,
        value: Box<TileExpr>,
    },
    Binary {
        op: TileBinaryOp,
        left: Box<TileExpr>,
        right: Box<TileExpr>,
    },
    Cast {
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
    /// Fused 4-way dot product. Lowers to a single `Math::Dot` over composed
    /// `vec4<f32>` operands — the same pattern the qgemv accelerator emits.
    Dot4 {
        a: [Box<TileExpr>; 4],
        b: [Box<TileExpr>; 4],
    },
    /// Reference to a pinned subexpression. The first reference in a scope
    /// lowers the bound value into a private local; subsequent references in
    /// the same scope reuse it.
    PinnedRef {
        id: PinId,
    },
    /// Reference to one accumulator output of a multi-output loop-fold group.
    /// The first reference materializes the shared K-loop; subsequent
    /// references reuse the per-accumulator local.
    LoopFoldGroupOutput {
        group: LoopFoldGroupId,
        lane: u32,
    },
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
}

/// A tiny tile identifier for the typed IR.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct TileId(pub(crate) u32);

impl TileId {
    /// The dense index for this tile declaration.
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Element types represented by the typed IR.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ElementType {
    F32,
    F16,
    U32,
}

/// A sample numeric marker.
#[derive(Copy, Clone, Debug)]
pub struct F32;

/// Half-precision floating point storage marker.
#[derive(Copy, Clone, Debug)]
pub struct F16;

/// Packed u32 storage marker.
#[derive(Copy, Clone, Debug)]
pub struct U32;

/// Numeric element markers that can appear in the typed IR.
pub trait Numeric {
    const ELEMENT: ElementType;
}

impl Numeric for F32 {
    const ELEMENT: ElementType = ElementType::F32;
}

impl Numeric for F16 {
    const ELEMENT: ElementType = ElementType::F16;
}

impl Numeric for U32 {
    const ELEMENT: ElementType = ElementType::U32;
}

/// A concrete layout for a tile-like value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    strides: Strides,
    memory_level: MemoryLevel,
}

impl Layout {
    /// Construct a row-major contiguous layout in a memory level.
    pub fn contiguous(memory_level: MemoryLevel, shape: Shape) -> Self {
        let strides = Strides::row_major_for(&shape);
        Self {
            shape,
            strides,
            memory_level,
        }
    }

    /// Construct an explicit strided layout in a memory level.
    pub fn strided(memory_level: MemoryLevel, shape: Shape, strides: Strides) -> Self {
        assert_eq!(
            shape.rank(),
            strides.rank(),
            "layout shape and strides must have the same rank"
        );
        Self {
            shape,
            strides,
            memory_level,
        }
    }

    /// Logical shape of the tile.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Logical strides of the tile.
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// The memory level where this tile is represented.
    pub const fn memory_level(&self) -> MemoryLevel {
        self.memory_level
    }

    /// Total number of logical elements addressed by this layout.
    pub fn element_count(&self) -> NonZeroU32 {
        self.shape.element_count()
    }

    /// Number of elements required to back this layout, including padding
    /// implied by non-contiguous strides.
    pub fn allocation_element_count(&self) -> NonZeroU32 {
        let last_index = self
            .shape
            .dims()
            .iter()
            .zip(self.strides.values())
            .try_fold(0u32, |acc, (dim, stride)| {
                let extent = dim.get().checked_sub(1)?;
                acc.checked_add(extent.checked_mul(*stride)?)
            })
            .and_then(|index| index.checked_add(1))
            .expect("layout allocation span overflow");
        NonZeroU32::new(last_index).expect("layout rank is non-zero")
    }

    /// True when the strides match row-major contiguous order.
    pub fn is_row_major(&self) -> bool {
        self.strides == Strides::row_major_for(&self.shape)
    }

    /// True when the strides match column-major contiguous order.
    pub fn is_col_major(&self) -> bool {
        self.strides == Strides::col_major_for(&self.shape)
    }

    /// True when the strides are a standard contiguous row- or column-major layout.
    pub fn is_contiguous(&self) -> bool {
        self.is_row_major() || self.is_col_major()
    }
}

/// The logical shape of a tile-level operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<NonZeroU32>,
}

impl Shape {
    /// Construct a tile shape from positive dimension sizes.
    pub fn new<const R: usize>(dims: [u32; R]) -> Self {
        assert!(R > 0, "tile shape must have at least one dimension");
        Self {
            dims: dims
                .into_iter()
                .map(|dim| NonZeroU32::new(dim).expect("tile dimensions must be non-zero"))
                .collect(),
        }
    }

    /// Construct the default one-dimensional tile shape.
    pub fn tile() -> Self {
        Self::new([32])
    }

    /// Rank of the logical shape.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Dimension sizes.
    pub fn dims(&self) -> &[NonZeroU32] {
        &self.dims
    }

    /// Number of logical elements in the tile.
    pub fn element_count(&self) -> NonZeroU32 {
        let elements = self
            .dims
            .iter()
            .fold(1u32, |acc, dim| acc.checked_mul(dim.get()).unwrap());
        NonZeroU32::new(elements).expect("shape rank is non-zero")
    }
}

/// Logical strides for a tile layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Strides {
    values: Vec<u32>,
}

impl Strides {
    /// Construct explicit strides.
    pub fn new<const R: usize>(values: [u32; R]) -> Self {
        assert!(R > 0, "strides must have at least one dimension");
        Self {
            values: values.into_iter().collect(),
        }
    }

    /// Construct row-major contiguous strides for a shape.
    pub fn row_major_for(shape: &Shape) -> Self {
        let mut values = vec![1; shape.rank()];
        let dims = shape.dims();
        for axis in (0..shape.rank() - 1).rev() {
            values[axis] = values[axis + 1] * dims[axis + 1].get();
        }
        Self { values }
    }

    /// Construct column-major contiguous strides for a shape.
    pub fn col_major_for(shape: &Shape) -> Self {
        let mut values = vec![1; shape.rank()];
        let dims = shape.dims();
        for axis in 1..shape.rank() {
            values[axis] = values[axis - 1] * dims[axis - 1].get();
        }
        Self { values }
    }

    /// Rank of the stride vector.
    pub fn rank(&self) -> usize {
        self.values.len()
    }

    /// Stride values.
    pub fn values(&self) -> &[u32] {
        &self.values
    }
}

/// Where a layout lives in the GPU memory hierarchy.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MemoryLevel {
    Storage,
    Uniform,
    Workgroup,
    Private,
}

/// The execution hierarchy level that owns a tile.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TileLevel {
    Workgroup,
}

/// Whether a tile declaration owns storage.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TileOrigin {
    Allocation,
}

impl Default for Layout {
    fn default() -> Self {
        Self::contiguous(MemoryLevel::Workgroup, Shape::tile())
    }
}
