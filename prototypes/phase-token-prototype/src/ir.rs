use std::num::NonZeroU32;

use crate::{LowerError, NagaKernel};

/// A typed kernel IR emitted by the prototype builder.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KernelIr {
    pub(crate) buffers: Vec<BufferDecl>,
    pub(crate) tiles: Vec<TileDecl>,
    pub(crate) body: Block,
    pub(crate) next_buffer: u32,
    pub(crate) next_tile: u32,
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

/// GGML quantization formats represented by the prototype qmatmul path.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum GgmlQuantFormat {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
}

impl GgmlQuantFormat {
    pub const fn block_elements(self) -> u32 {
        match self {
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,
        }
    }

    /// Number of u32 words in the f32-scale shader layout for one block.
    pub const fn block_words(self) -> u32 {
        match self {
            Self::Q4_0 => 5,
            Self::Q4_1 => 6,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 9,
            Self::Q8_1 => 10,
            Self::Q2K => 22,
            Self::Q3K => 28,
            Self::Q4K => 37,
            Self::Q5K => 45,
            Self::Q6K => 53,
            Self::Q8K => 73,
        }
    }

    pub const fn qgemv_cols_per_workgroup(self) -> u32 {
        self.qgemv_subgroups_per_workgroup() * self.qgemv_cols_per_subgroup()
    }

    pub const fn qgemv_cols_per_subgroup(self) -> u32 {
        match self {
            Self::Q2K => 4,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_1 => 4,
            Self::Q5_0 => 4,
            Self::Q3K | Self::Q4K | Self::Q8K => 2,
            Self::Q6K => 1,
            Self::Q8_0 | Self::Q8_1 => 4,
            Self::Q5K => 1,
        }
    }

    pub const fn qgemv_subgroups_per_workgroup(self) -> u32 {
        match self {
            Self::Q4K | Self::Q8_0 | Self::Q8_1 => 4,
            _ => 2,
        }
    }
}

/// A packed quantized storage matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantizedMatrix {
    pub data: StorageView,
    pub format: GgmlQuantFormat,
    pub rows: u32,
    pub cols: u32,
}

/// A small Triton-like source tile program.
///
/// The first lowering target supports one-dimensional lane tiles: each
/// invocation owns one lane, reductions use scratch tiles, and storage accesses
/// are expressed through typed index expressions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileProgramOp {
    pub grid: [u32; 3],
    pub block: u32,
    pub stores: Vec<TileStoreProgramOp>,
    pub accelerator: Option<TileProgramAccelerator>,
}

/// A structured tile-program body lowered with backend tile acceleration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TileProgramAccelerator {
    QMatmul(TileQMatmulProgramOp),
    QGemv(TileQGemvProgramOp),
}

/// Cooperative tile qmatmul body. The operation remains inside
/// [`Op::TileProgram`], but is kept structured so the lowerer can emit native
/// 8x8 cooperative matrix fragments instead of scalarizing the tile program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileQMatmulProgramOp {
    pub a: StorageView,
    pub b: QuantizedMatrix,
    pub y: StorageView,
    pub a_tile: TileRef,
    pub b_tile: TileRef,
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
}

/// Subgroup tile qgemv body for the single-row qmatmul case. It is represented
/// as a [`TileProgram`] accelerator so the public op surface stays uniform.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileQGemvProgramOp {
    pub a: StorageView,
    pub b: QuantizedMatrix,
    pub y: StorageView,
    pub workgroups_x: u32,
}

/// A masked tile store emitted by a source tile program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TileStoreProgramOp {
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
