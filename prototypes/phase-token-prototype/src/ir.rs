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

    /// Expand high-level GEMM operations into nested partition/MMA operations.
    ///
    /// This is the prototype's split between a Triton/TileLang-like source IR
    /// and a lower tile IR where subgroup/thread partitioning is explicit.
    pub fn expand_gemm_to_mma(&self) -> Self {
        let mut expanded = Self {
            buffers: self.buffers.clone(),
            tiles: self.tiles.clone(),
            body: Block::new(),
            next_buffer: self.next_buffer,
            next_tile: self.next_tile,
        };
        expanded.body = expanded.expand_block_gemm_to_mma(&self.body);
        expanded
    }

    /// Lower this IR into a validated Naga module.
    pub fn lower_to_naga(&self) -> Result<NagaKernel, LowerError> {
        crate::lower::lower_to_naga(self)
    }

    fn expand_block_gemm_to_mma(&mut self, block: &Block) -> Block {
        let mut expanded = Block::new();
        for op in block.ops() {
            expanded.push(match op {
                Op::Gemm(op) => self.expand_gemm_op(*op),
                Op::Loop(op) => Op::Loop(LoopOp {
                    kind: op.kind,
                    body: self.expand_block_gemm_to_mma(&op.body),
                }),
                Op::Partition(op) => Op::Partition(PartitionOp {
                    bindings: op.bindings.clone(),
                    level: op.level,
                    body: self.expand_block_gemm_to_mma(&op.body),
                }),
                op => op.clone(),
            });
        }
        expanded
    }

    fn expand_gemm_op(&mut self, op: GemmOp) -> Op {
        let [m, k] = self.tile_shape_2d(op.a);
        let [k_b, n] = self.tile_shape_2d(op.b);
        let [m_acc, n_acc] = self.tile_shape_2d(op.acc);
        assert_eq!(k, k_b, "gemm K dimensions must match");
        assert_eq!(m, m_acc, "gemm M dimension must match accumulator");
        assert_eq!(n, n_acc, "gemm N dimension must match accumulator");
        assert_eq!(m % op.tiling.subgroup_m, 0, "M must divide subgroup_m");
        assert_eq!(n % op.tiling.subgroup_n, 0, "N must divide subgroup_n");
        assert_eq!(
            op.tiling.subgroup_m % op.tiling.thread_m,
            0,
            "subgroup_m must divide thread_m"
        );
        assert_eq!(
            op.tiling.subgroup_n % op.tiling.thread_n,
            0,
            "subgroup_n must divide thread_n"
        );

        let mut ops = Vec::new();
        for subgroup_m in (0..m).step_by(op.tiling.subgroup_m as usize) {
            for subgroup_n in (0..n).step_by(op.tiling.subgroup_n as usize) {
                let a_subgroup = self.alloc_partition_view(
                    op.a,
                    TileLevel::Subgroup,
                    Shape::new([op.tiling.subgroup_m, op.tiling.subgroup_k]),
                    [subgroup_m, 0],
                );
                let b_subgroup = self.alloc_partition_view(
                    op.b,
                    TileLevel::Subgroup,
                    Shape::new([op.tiling.subgroup_k, op.tiling.subgroup_n]),
                    [0, subgroup_n],
                );
                let acc_subgroup = self.alloc_partition_view(
                    op.acc,
                    TileLevel::Subgroup,
                    Shape::new([op.tiling.subgroup_m, op.tiling.subgroup_n]),
                    [subgroup_m, subgroup_n],
                );

                for thread_m in (0..op.tiling.subgroup_m).step_by(op.tiling.thread_m as usize) {
                    for thread_n in (0..op.tiling.subgroup_n).step_by(op.tiling.thread_n as usize) {
                        let a_thread = self.alloc_partition_view(
                            a_subgroup,
                            TileLevel::Thread,
                            Shape::new([op.tiling.thread_m, op.tiling.thread_k]),
                            [thread_m, 0],
                        );
                        let b_thread = self.alloc_partition_view(
                            b_subgroup,
                            TileLevel::Thread,
                            Shape::new([op.tiling.thread_k, op.tiling.thread_n]),
                            [0, thread_n],
                        );
                        let acc_thread = self.alloc_partition_view(
                            acc_subgroup,
                            TileLevel::Thread,
                            Shape::new([op.tiling.thread_m, op.tiling.thread_n]),
                            [thread_m, thread_n],
                        );
                        ops.push(Op::Mma(MmaOp {
                            a: a_thread,
                            b: b_thread,
                            acc: acc_thread,
                            level: TileLevel::Thread,
                            backend: op.backend,
                        }));
                    }
                }
            }
        }

        Op::Block(BlockOp {
            body: Block::from_ops(ops),
        })
    }

    fn alloc_partition_view(
        &mut self,
        source: TileRef,
        level: TileLevel,
        shape: Shape,
        origin: [u32; 2],
    ) -> TileRef {
        let source_decl = self.tile_decl(source);
        let layout = Layout::strided(
            source_decl.layout.memory_level(),
            shape,
            source_decl.layout.strides().clone(),
        );
        let element = source_decl.element;
        let id = TileId(self.next_tile);
        self.next_tile += 1;
        let view = TileRef::new(id, element);
        self.tiles.push(TileDecl {
            id,
            element,
            layout,
            level,
            origin: TileOrigin::View {
                source,
                mapping: ViewMapping::Partition { level, origin },
            },
        });
        view
    }

    fn tile_shape_2d(&self, tile: TileRef) -> [u32; 2] {
        let shape = self.tile_decl(tile).layout.shape();
        assert_eq!(shape.rank(), 2, "gemm tiles must be rank-2");
        [shape.dims()[0].get(), shape.dims()[1].get()]
    }

    fn tile_decl(&self, tile: TileRef) -> &TileDecl {
        let decl = self
            .tiles
            .get(tile.id.index())
            .expect("tile reference must point at a declared tile");
        assert_eq!(
            decl.element, tile.element,
            "tile reference element must match its declaration"
        );
        decl
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
    /// A structural block introduced by an expansion pass.
    Block(BlockOp),
    /// Fill a tile with a scalar value.
    FillTile(FillTileOp),
    /// A subgroup-cooperative load into a workgroup tile.
    CooperativeLoad(CooperativeLoadOp),
    /// A semantic partition of one or more tiles to a lower execution level.
    Partition(PartitionOp),
    /// A control barrier.
    Barrier(BarrierOp),
    /// High-level tiled GEMM over parent tiles/fragments.
    Gemm(GemmOp),
    /// Row-parallel matrix-vector multiply over storage tensors.
    Gemv(GemvOp),
    /// Matrix multiply-accumulate over tile operands.
    Mma(MmaOp),
    /// Store a tile to a storage buffer view.
    StoreTile(StoreTileOp),
    /// A structured loop.
    Loop(LoopOp),
}

/// A structured block operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockOp {
    pub body: Block,
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

/// A subgroup-cooperative load into a workgroup tile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CooperativeLoadOp {
    pub dst: TileRef,
    pub src: StorageView,
    pub level: TileLevel,
}

/// A store from a tile to storage.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoreTileOp {
    pub src: TileRef,
    pub dst: StorageView,
}

/// A shaped view into a storage buffer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StorageView {
    pub buffer: BufferRef,
    pub offset: u32,
    pub layout: Layout,
    pub dynamic_offsets: Vec<Option<DynamicOffset>>,
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
        }
    }
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

/// Fill a tile with a scalar value.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FillTileOp {
    pub dst: TileRef,
    pub value: FillValue,
}

/// Literal fill values represented by the prototype IR.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FillValue {
    Zero,
}

/// A barrier operation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BarrierOp {
    pub scope: BarrierScope,
}

/// Matrix multiply-accumulate over shaped tiles.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MmaOp {
    pub a: TileRef,
    pub b: TileRef,
    pub acc: TileRef,
    pub level: TileLevel,
    pub backend: MmaBackend,
}

/// High-level GEMM over shared/block operands and an accumulator fragment.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GemmOp {
    pub a: TileRef,
    pub b: TileRef,
    pub acc: TileRef,
    pub tiling: GemmTiling,
    pub backend: MmaBackend,
}

/// A row-parallel GEMV operation.
///
/// The lowering assigns one workgroup to one output row. Invocations within the
/// workgroup cooperatively reduce the K dimension into `partials`, then lane 0
/// writes the row result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GemvOp {
    pub a: StorageView,
    pub x: StorageView,
    pub y: StorageView,
    pub partials: TileRef,
    pub rows_per_workgroup: u32,
    pub vector_width: u32,
}

/// Concrete tiling plan used when lowering a high-level GEMM into tile MMA.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct GemmTiling {
    pub subgroup_m: u32,
    pub subgroup_n: u32,
    pub subgroup_k: u32,
    pub thread_m: u32,
    pub thread_n: u32,
    pub thread_k: u32,
}

impl GemmTiling {
    /// A conservative portable tiling plan for the prototype FMA lowering.
    pub fn portable(m: u32, n: u32, k: u32) -> Self {
        let subgroup_m = m.min(16);
        let subgroup_n = n.min(16);
        let subgroup_k = k;
        Self {
            subgroup_m,
            subgroup_n,
            subgroup_k,
            thread_m: subgroup_m.min(4),
            thread_n: subgroup_n.min(4),
            thread_k: subgroup_k,
        }
    }
}

/// MMA lowering backend requested by the IR.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MmaBackend {
    FmaPortable,
    SubgroupMatrix,
}

/// A structured tile partition.
///
/// The body is emitted once by the Rust builder, but semantically describes
/// code that runs over child tile views at `level`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartitionOp {
    pub bindings: Vec<PartitionBinding>,
    pub level: TileLevel,
    pub body: Block,
}

/// One source tile and the child tile view produced by a partition.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PartitionBinding {
    pub source: TileRef,
    pub view: TileRef,
}

/// A structured loop operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopOp {
    pub kind: LoopKind,
    pub body: Block,
}

/// The loop form represented by this prototype.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LoopKind {
    RangeStep { induction: Dim, iterations: u32 },
}

/// A symbolic dimension used only to make the loop API look like an IR builder.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Dim(pub u32);

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

    /// Construct the default one-dimensional subgroup tile shape.
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
    Subgroup,
    Thread,
}

/// Whether a tile declaration owns storage or is a view of another tile.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TileOrigin {
    Allocation,
    View {
        source: TileRef,
        mapping: ViewMapping,
    },
}

/// How a view relates to its source tile.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ViewMapping {
    Partition { level: TileLevel, origin: [u32; 2] },
}

impl Default for Layout {
    fn default() -> Self {
        Self::contiguous(MemoryLevel::Workgroup, Shape::tile())
    }
}

/// The synchronization scope for a barrier.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BarrierScope {
    Workgroup,
}
