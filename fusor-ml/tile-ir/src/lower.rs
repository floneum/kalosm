use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;

use naga::{
    AddressSpace, Arena, ArraySize, Barrier, BinaryOperator, Binding, Block, BuiltIn,
    CollectiveOperation, EntryPoint, Expression, Function, FunctionArgument, GlobalVariable,
    Handle, Literal, LocalVariable, MathFunction, Module, Range, ResourceBinding, Scalar,
    ScalarKind, ShaderStage, Span, Statement, StorageAccess, SubgroupOperation, Type, TypeInner,
    VectorSize,
};

use crate::ir::{
    BlockDequantId, BufferAccess, BufferId, CoopFragmentId, CoopOperandRole, CopySource,
    DotK, ElementType, Expr, FlattenedMatrixMap, Im2ColNhwcMap, KernelIr,
    Layout, LocalId, LocalRef, MemoryLevel, PackedActivations, StorageIndexMap, StorageView,
    TileBinaryOp, TileCompareOp, TileId, TileLinearLoadExpr, TileLiteral, TileLoadExpr,
    LoadSource, TileProgramOp, TileReduceOp, TileRef, TileStmt,
    TileStoreStmt, TileUnaryOp,
};
use crate::quantized::{GgmlQuantFormat, QuantizedMatrix};

const LOCAL_INVOCATION_INDEX_ARG: u32 = 0;
const WORKGROUP_ID_ARG: u32 = 1;
const SUBGROUP_ID_ARG: u32 = 2;
const SUBGROUP_INVOCATION_ID_ARG: u32 = 3;
const SUBGROUP_SIZE_ARG: u32 = 4;
const NUM_SUBGROUPS_ARG: u32 = 5;
const DEFAULT_WORKGROUP_INVOCATIONS: u32 = 256;
const DEFAULT_WORKGROUP_SIZE: [u32; 3] = [16, 16, 1];

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
    /// An operation referenced a private local that was never allocated.
    UnknownLocal(LocalId),
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
    /// An operation used a local as a different element type than its declaration.
    LocalElementMismatch {
        local: LocalId,
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
            Self::UnknownLocal(local) => write!(f, "unknown local {:?}", local),
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
            Self::LocalElementMismatch {
                local,
                declared,
                used,
            } => write!(
                f,
                "local {:?} declared as {:?} but used as {:?}",
                local, declared, used
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
    i32_ty: Handle<Type>,
    i32_vec4_ty: Handle<Type>,
    f16_ty: Option<Handle<Type>>,
    u32_ty: Handle<Type>,
    bool_ty: Handle<Type>,
    u32_vec3_ty: Handle<Type>,
    buffer_globals: Vec<Option<Handle<GlobalVariable>>>,
    tile_globals: Vec<Option<Handle<GlobalVariable>>>,
    tile_locals: Vec<Option<Handle<LocalVariable>>>,
    private_locals: Vec<Option<Handle<LocalVariable>>>,
    live_tiles: Vec<bool>,
    loop_index_local: Option<Handle<LocalVariable>>,
    workgroup_invocations: u32,
    workgroup_size: [u32; 3],
    subgroup_usage: analysis::SubgroupIndexUsage,
    block_dequant_cache: RefCell<HashMap<BlockDequantId, Vec<Handle<Expression>>>>,
    q8_activation_pack_cache: RefCell<HashMap<Vec<Handle<Expression>>, Q8ActivationPacks>>,
    coop_c_ty: Option<Handle<Type>>,
    /// SSA-cached cooperatively-loaded fragments. The producer
    /// (`TileStmt::LoadCoop`) inserts its fresh `CoopFragmentId`; consumers
    /// (`TileStmt::Mma`) read by id within the same scope. Cleared at
    /// boundaries via `snapshot_coop_loop_caches`.
    coop_fragment_cache: RefCell<HashMap<CoopFragmentId, Handle<Expression>>>,
    /// Latest SSA value of each cooperative accumulator, keyed by the
    /// accumulator's `LocalId`. Lets MMAs chain through SSA — `mma(c=load)`
    /// once, then `mma(c=last_ssa)` for the rest of the scope, with a single
    /// Store at scope end.
    coop_acc_value_cache: RefCell<HashMap<LocalId, Handle<Expression>>>,
    uses_cooperative_matrix: bool,
}

#[derive(Clone)]
struct Q8ActivationPacks {
    len: usize,
    scales: [Handle<LocalVariable>; 4],
    packs: [Handle<LocalVariable>; 4],
    sums_i32: [Handle<LocalVariable>; 4],
}

struct Q8ActivationPackValues {
    scales: Vec<Handle<Expression>>,
    packs: Vec<Handle<Expression>>,
    sums_i32: Vec<Handle<Expression>>,
}

struct TileLoopCacheSnapshot {
    block_dequant: Vec<(BlockDequantId, Vec<Handle<Expression>>)>,
}

struct CoopLoopCacheSnapshot {
    fragments: Vec<(CoopFragmentId, Handle<Expression>)>,
    acc_values: Vec<(LocalId, Handle<Expression>)>,
}

#[derive(Copy, Clone)]
struct ScratchLocals {
    loop_index: Handle<LocalVariable>,
    values: [Handle<LocalVariable>; 5],
    spills: [[Handle<LocalVariable>; 32]; 5],
    block_dequant: [Handle<LocalVariable>; 16],
    q8_activation_scales: [Handle<LocalVariable>; 4],
    q8_activation_packs: [Handle<LocalVariable>; 4],
    q8_activation_sums_i32: [Handle<LocalVariable>; 4],
}

mod analysis;
mod block;
mod control;
mod coop;
mod indexing;
mod math;
mod quantized;
mod setup;
mod tile_program;
