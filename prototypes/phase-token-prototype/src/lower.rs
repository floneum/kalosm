use std::fmt;

use naga::{
    AddressSpace, Arena, ArraySize, Barrier, BinaryOperator, Binding, Block, BuiltIn,
    CooperativeData, CooperativeRole, CooperativeSize, EntryPoint, Expression, Function,
    FunctionArgument, GlobalVariable, Handle, Literal, LocalVariable, MathFunction,
    MemoryDecorations, Module, Range, ResourceBinding, Scalar, ShaderStage, Span, Statement,
    StorageAccess, Type, TypeInner, VectorSize,
};

use crate::{
    BarrierScope, BufferAccess, BufferId, DynamicOffset, ElementType, GemmOp, GemvOp, KernelIr,
    Layout, MemoryLevel, MmaOp, Op, StorageView, TileId, TileOrigin, TileRef, ViewMapping,
};

const LOCAL_INVOCATION_INDEX_ARG: u32 = 0;
const WORKGROUP_ID_ARG: u32 = 1;
const SUBGROUP_ID_ARG: u32 = 2;
const DEFAULT_WORKGROUP_INVOCATIONS: u32 = 256;
const DEFAULT_WORKGROUP_SIZE: [u32; 3] = [16, 16, 1];
const GEMV_WORKGROUP_INVOCATIONS: u32 = 128;
const GEMV_WORKGROUP_SIZE: [u32; 3] = [128, 1, 1];
const COOP_MATRIX_WORKGROUP_INVOCATIONS: u32 = 32;
const COOP_MATRIX_WORKGROUP_SIZE: [u32; 3] = [32, 1, 1];
const COOP_MATRIX_DIM: u32 = 8;
const COOP_MATRIX_SIZE: CooperativeSize = CooperativeSize::Eight;
const PREFER_COOP_MATRIX_GEMM: bool = true;
// The staged cooperative path now matches the MLX-style 64x64/4-subgroup shape
// closely enough to beat direct cooperative loads on the current F32 Metal path.
const PREFER_SHARED_COOP_GEMM: bool = true;
// Correct but slower for scalar F32 on current wgpu/Metal; keep it opt-in until
// the cost model can pick it only for hardware/backends where it wins.
const PREFER_SHARED_GEMM: bool = false;
const COOP_MATRIX_OUTER_UNROLL: u32 = 1;
const PREFER_LINEAR_BASE_HOIST: bool = false;
const COOPERATIVE_LOAD_WIDTH: u32 = 8;

pub(crate) fn lower_to_naga(ir: &KernelIr) -> Result<NagaKernel, LowerError> {
    Lowerer::new(ir).lower()
}

/// A validated Naga lowering result.
pub struct NagaKernel {
    module: Module,
    info: naga::valid::ModuleInfo,
}

impl NagaKernel {
    /// The generated Naga module.
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// Naga validation metadata for the generated module.
    pub fn info(&self) -> &naga::valid::ModuleInfo {
        &self.info
    }
}

/// Errors produced by the Naga lowering pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerError {
    /// An event referenced a tile that was never allocated.
    UnknownTile(TileId),
    /// An operation referenced a storage buffer that was never declared.
    UnknownBuffer(BufferId),
    /// The Naga lowerer cannot emit this memory level.
    UnsupportedMemoryLevel(MemoryLevel),
    /// The typed IR operation is outside the supported lowering subset.
    UnsupportedOperation(&'static str),
    /// An operation used a tile as a different element type than its declaration.
    TileElementMismatch {
        tile: TileId,
        declared: ElementType,
        used: ElementType,
    },
    /// Naga rejected the generated module.
    Validation(String),
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTile(tile) => write!(f, "unknown tile {:?}", tile),
            Self::UnknownBuffer(buffer) => write!(f, "unknown buffer {:?}", buffer),
            Self::UnsupportedMemoryLevel(memory) => {
                write!(f, "unsupported memory level {:?}", memory)
            }
            Self::UnsupportedOperation(op) => write!(f, "unsupported operation {op}"),
            Self::TileElementMismatch {
                tile,
                declared,
                used,
            } => write!(
                f,
                "tile {:?} declared as {:?} but used as {:?}",
                tile, declared, used
            ),
            Self::Validation(error) => write!(f, "naga validation failed: {error}"),
        }
    }
}

impl std::error::Error for LowerError {}

struct Lowerer<'a> {
    ir: &'a KernelIr,
    module: Module,
    f32_ty: Handle<Type>,
    f32_vec4_ty: Handle<Type>,
    u32_ty: Handle<Type>,
    u32_vec3_ty: Handle<Type>,
    coop_f32_a_ty: Option<Handle<Type>>,
    coop_f32_b_ty: Option<Handle<Type>>,
    coop_f32_c_ty: Option<Handle<Type>>,
    buffer_globals: Vec<Option<Handle<GlobalVariable>>>,
    tile_globals: Vec<Option<Handle<GlobalVariable>>>,
    tile_locals: Vec<Option<Handle<LocalVariable>>>,
    live_tiles: Vec<bool>,
    loop_index_local: Option<Handle<LocalVariable>>,
    workgroup_invocations: u32,
    workgroup_size: [u32; 3],
    max_gemv_rows: u32,
    uses_coop_gemm: bool,
    coop_subgroups: u32,
    uses_subgroup_id: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct GemmDescriptor {
    a: TileRef,
    b: TileRef,
    acc: TileRef,
}

impl From<&GemmOp> for GemmDescriptor {
    fn from(op: &GemmOp) -> Self {
        Self {
            a: op.a,
            b: op.b,
            acc: op.acc,
        }
    }
}

struct FusedGemmParts<'ops> {
    fill: &'ops crate::FillTileOp,
    loop_op: &'ops crate::LoopOp,
    store: &'ops crate::StoreTileOp,
    gemm: GemmDescriptor,
    a_load: &'ops crate::CooperativeLoadOp,
    b_load: &'ops crate::CooperativeLoadOp,
}

#[derive(Copy, Clone)]
struct ScratchLocals {
    tile_index: Handle<LocalVariable>,
    linear_index: Handle<LocalVariable>,
    store_index: Handle<LocalVariable>,
    loop_index: Handle<LocalVariable>,
    mma_i: Handle<LocalVariable>,
    mma_j: Handle<LocalVariable>,
    mma_k: Handle<LocalVariable>,
    mma_sum: Handle<LocalVariable>,
    mma_sum_1: Handle<LocalVariable>,
    mma_sum_2: Option<Handle<LocalVariable>>,
    mma_sum_3: Option<Handle<LocalVariable>>,
    mma_sum_4: Option<Handle<LocalVariable>>,
    mma_sum_5: Option<Handle<LocalVariable>>,
    mma_sum_6: Option<Handle<LocalVariable>>,
    mma_sum_7: Option<Handle<LocalVariable>>,
    coop_accs: [Option<Handle<LocalVariable>>; 16],
}

#[derive(Copy, Clone)]
enum CoopPartition {
    Single,
    Columns,
    Rows,
    InterleavedGrid { row_groups: u32, col_groups: u32 },
}

mod analysis;
mod block;
mod control;
mod gemm_coop;
mod gemm_scalar;
mod gemm_shared_scalar;
mod gemm_storage_scalar;
mod gemm_wide;
mod gemv;
mod indexing;
mod math;
mod ops;
mod setup;
