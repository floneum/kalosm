use std::cell::RefCell;
use std::fmt;

use naga::{
    AddressSpace, Arena, ArraySize, Barrier, BinaryOperator, Binding, Block, BuiltIn,
    CollectiveOperation, EntryPoint, Expression, Function, FunctionArgument, GlobalVariable,
    Handle, Literal, LocalVariable, MathFunction, Module, Range, ResourceBinding, Scalar,
    ScalarKind, ShaderStage, Span, Statement, StorageAccess, SubgroupOperation, Type, TypeInner,
    VectorSize,
};
use rustc_hash::FxHashMap;

use crate::ir::{
    Accumulator, Addr, AxisGroup, Buffer, BufferAccess, Builtin, CoopMatrixRole, CoopSrc,
    ElementType, Expr, ExprKind, KernelIr, Layout, Local, LocalDecl, MemoryLevel, MultiFlattenMap,
    Node, QuantActivation, ReduceKind, ScalarElement, Source, Stmt, StorageView, Tile,
    TileBinaryOp, TileCompareOp, TileLiteral, TileReduceOp, TileUnaryOp,
};
use crate::quantized::{GgmlQuantFormat, QuantizedMatrix};

const LOCAL_INVOCATION_INDEX_ARG: u32 = 0;
const WORKGROUP_ID_ARG: u32 = 1;
const DEFAULT_WORKGROUP_INVOCATIONS: u32 = 256;
const DEFAULT_WORKGROUP_SIZE: [u32; 3] = [16, 16, 1];

pub(crate) fn lower_to_naga(ir: &KernelIr) -> Result<NagaKernel, LowerError> {
    Lowerer::new(ir)?.lower()
}

/// A validated Naga lowering result.
#[derive(Debug)]
pub struct NagaKernel {
    module: Module,
    info: naga::valid::ModuleInfo,
    wgsl_extensions: WgslExtensions,
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

    /// WGSL extensions required when this kernel is serialized to WGSL.
    pub fn wgsl_extensions(&self) -> WgslExtensions {
        self.wgsl_extensions
    }

    /// Extension directives to prepend before Naga's WGSL output.
    pub fn wgsl_extension_prelude(&self) -> &'static str {
        self.wgsl_extensions.prelude()
    }
}

/// WGSL extension directives required by a lowered kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct WgslExtensions {
    subgroups: bool,
}

impl WgslExtensions {
    pub(crate) fn new(subgroups: bool) -> Self {
        Self { subgroups }
    }

    /// Whether this kernel needs `enable subgroups;` in WGSL.
    pub fn subgroups(self) -> bool {
        self.subgroups
    }

    /// Text that must appear before Naga's serialized WGSL declarations.
    pub fn prelude(self) -> &'static str {
        if self.subgroups {
            "enable subgroups;\n\n"
        } else {
            ""
        }
    }
}

/// Errors produced by the Naga lowering pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerError {
    /// The Naga lowerer cannot emit this memory level.
    UnsupportedMemoryLevel(MemoryLevel),
    /// The typed IR operation is outside the supported lowering subset.
    UnsupportedOperation(&'static str),
    /// Naga rejected the generated module.
    Validation(String),
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedMemoryLevel(memory) => {
                write!(f, "unsupported memory level {:?}", memory)
            }
            Self::UnsupportedOperation(op) => write!(f, "unsupported operation {op}"),
            Self::Validation(error) => write!(f, "naga validation failed: {error}"),
        }
    }
}

impl std::error::Error for LowerError {}

/// Demand-allocated private scratch local kinds. Locals are interned lazily by
/// `(ScratchKind, ElementType, depth)`; each distinct key allocates exactly one
/// `LocalVariable`, preserving the local count (which only drops, never grows)
/// from the old eagerly-allocated `ScratchLocals`.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub(super) enum ScratchKind {
    /// The counted/unstructured loop index (`u32`).
    LoopIndex,
    /// A masked-value spill local (one per masked load/reduce).
    Value,
    /// A reduce-accumulator spill local (deepens with nesting).
    Spill,
    /// A dequantized block-lane value local (`f32`).
    BlockDequant,
    /// A Q8 activation scale local (`f32`).
    Q8Scale,
    /// A Q8 activation pack local (`u32`).
    Q8Pack,
    /// A Q8 activation sum local (`i32`).
    Q8Sum,
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

struct Lowerer<'a> {
    ir: &'a KernelIr,
    module: Module,
    /// One `ElementType -> Handle<Type>` interner, replacing the ~18 cached
    /// type-handle fields. `f16` types are gated up front so an f16 handle on a
    /// non-f16 adapter yields `UnsupportedOperation` rather than mis-lowering.
    types: RefCell<FxHashMap<ElementType, Handle<Type>>>,
    f32_ty: Handle<Type>,
    f32_vec4_ty: Handle<Type>,
    i32_ty: Handle<Type>,
    i32_vec4_ty: Handle<Type>,
    u32_ty: Handle<Type>,
    u32_vec3_ty: Handle<Type>,
    /// `true` once `uses_f16` is decided up front (not lazy first-use).
    uses_f16: bool,
    /// Buffer / tile global variables, keyed by `Rc::as_ptr` of the decl.
    globals: RefCell<FxHashMap<*const (), Handle<GlobalVariable>>>,
    /// Private locals (program locals + private-level tiles), keyed by
    /// `Rc::as_ptr` of the decl.
    locals: RefCell<FxHashMap<*const (), Handle<LocalVariable>>>,
    /// Demand-allocated scratch locals interned by `(kind, element, depth)`.
    scratch: RefCell<FxHashMap<(ScratchKind, ElementType, u32), Handle<LocalVariable>>>,
    /// The function-local arena, owned during lowering so scratch can be
    /// demand-allocated through shared `&self` and moved into the function at
    /// the end. Kept disjoint from `function.expressions`.
    func_locals: RefCell<Arena<LocalVariable>>,
    /// `q8_activation_pack_cache`: dedups the activation pack of one set of
    /// activation handles within a store/quant scope.
    q8_activation_pack_cache: RefCell<FxHashMap<Vec<Handle<Expression>>, Q8ActivationPacks>>,
    /// Latest SSA value of each cooperative accumulator, keyed by the
    /// accumulator local's `Rc::as_ptr`. Lets MMAs chain through SSA — one
    /// `Load` then N `CooperativeMultiplyAdd`, one `Store` at scope end.
    coop_acc_value_cache: RefCell<FxHashMap<*const LocalDecl, Handle<Expression>>>,
    /// `Shared(Dequantize)` -> the N projected lane handles, keyed on
    /// `Rc::as_ptr` of the shared `Node`. The dequant helper runs once.
    dequant_memo: RefCell<FxHashMap<*const Node, Vec<Handle<Expression>>>>,
    /// `Shared(_)` emit-once, keyed on `Rc::as_ptr` of the shared `Node`.
    expr_memo: RefCell<FxHashMap<*const Node, Handle<Expression>>>,
    workgroup_invocations: u32,
    workgroup_size: [u32; 3],
    caps: analysis::Capabilities,
    subgroup_id_arg: Option<u32>,
    subgroup_invocation_id_arg: Option<u32>,
    subgroup_size_arg: Option<u32>,
    num_subgroups_arg: Option<u32>,
    /// First-use-ordered declarations from the one analysis walk, emitted as the
    /// global/local arenas (buffers sorted by binding at global-creation time).
    buffer_decls: Vec<Buffer>,
    tile_decls: Vec<Tile>,
    local_decls: Vec<Local>,
}

/// Snapshot of the per-iteration caches that are scoped to one loop iteration:
/// `dequant_memo` + `expr_memo` (block-lane / shared SSA) and the coop
/// acc-value + q8 activation caches. Drained at the iteration boundary and
/// restored on exit.
struct LoopCacheSnapshot {
    dequant_memo: Vec<(*const Node, Vec<Handle<Expression>>)>,
    expr_memo: Vec<(*const Node, Handle<Expression>)>,
}

struct CoopLoopCacheSnapshot {
    acc_values: Vec<(*const LocalDecl, Handle<Expression>)>,
}

/// Pointer-key helpers for the Rc-keyed decl maps.
fn buffer_key(buffer: &Buffer) -> *const () {
    std::rc::Rc::as_ptr(buffer) as *const ()
}

fn tile_key(tile: &Tile) -> *const () {
    std::rc::Rc::as_ptr(tile) as *const ()
}

fn local_key(local: &Local) -> *const () {
    std::rc::Rc::as_ptr(local) as *const ()
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
