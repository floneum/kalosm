use std::fmt;

use naga::{
    AddressSpace, Arena, ArraySize, Barrier, BinaryOperator, Binding, Block, BuiltIn, EntryPoint,
    Expression, Function, FunctionArgument, GlobalVariable, Handle, Literal, LocalVariable,
    MathFunction, Module, Range, ResourceBinding, Scalar, ScalarKind, ShaderStage, Span, Statement,
    StorageAccess, Type, TypeInner, VectorSize,
};

use crate::ir::{
    BufferAccess, BufferId, DynamicOffset, ElementType, FlattenedMatrixMap, GgmlQuantFormat,
    Im2ColNhwcMap, KernelIr, Layout, MemoryLevel, Op, QuantizedMatrix, StorageIndexMap,
    StorageView, TileBinaryOp, TileCompareOp, TileExpr, TileId, TileIndexExpr, TileLiteral,
    TileLoadExpr, TileMaskExpr, TileOrigin, TileProgramOp, TileQuantizedLoadExpr, TileReduceOp,
    TileRef, TileScalarExpr, TileUnaryOp,
};

const LOCAL_INVOCATION_INDEX_ARG: u32 = 0;
const WORKGROUP_ID_ARG: u32 = 1;
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
    f16_ty: Option<Handle<Type>>,
    u32_ty: Handle<Type>,
    u32_vec3_ty: Handle<Type>,
    buffer_globals: Vec<Option<Handle<GlobalVariable>>>,
    tile_globals: Vec<Option<Handle<GlobalVariable>>>,
    tile_locals: Vec<Option<Handle<LocalVariable>>>,
    live_tiles: Vec<bool>,
    loop_index_local: Option<Handle<LocalVariable>>,
    workgroup_invocations: u32,
    workgroup_size: [u32; 3],
}

#[derive(Copy, Clone)]
struct ScratchLocals {
    loop_index: Handle<LocalVariable>,
    values: [Handle<LocalVariable>; 3],
    spills: [[Handle<LocalVariable>; 32]; 3],
}

mod analysis;
mod block;
mod control;
mod indexing;
mod math;
mod quantized;
mod setup;
mod tile_program;
