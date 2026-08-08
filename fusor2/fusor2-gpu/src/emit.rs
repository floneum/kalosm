//! `KernelIr` -> naga `Module`.
//!
//! naga IR is already an IR, so there is no target dialect between L2 and it:
//! a dialect there would host zero rewrite rules.
//!
//! One up-front [`Analysis`] walk decides every capability the module will
//! declare *before* a single expression is lowered. Gating f16 here rather
//! than lazily is what makes an f16 handle on a non-f16 adapter fail with
//! [`EmitError::MissingCapability`] instead of mis-lowering.

pub mod coop;
pub mod expr;
pub mod quantized;
pub mod reduce;
pub mod stmt;
pub mod types;

use fusor2_ir::device::Caps;
use fusor2_ir::ir::level2::{
    Accumulator, Addr, ArenaPlan, Buffer, Builtin, CoopSrc, ElementType, KernelIr, Local, MemReads,
    ReduceKind, ScalarElement, Source, Stmt, Tile, TileExpr, TileExprKind,
};
use fusor2_ir::target::EmitError;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::bindings::{BindingDesc, bindings_from_module};

/// `@builtin(local_invocation_index)`, always argument 0.
pub const LOCAL_INVOCATION_INDEX_ARG: u32 = 0;
/// `@builtin(workgroup_id)`, always argument 1.
pub const WORKGROUP_ID_ARG: u32 = 1;
/// Lanes assumed when `KernelIr::block` is zero.
pub const DEFAULT_WORKGROUP_INVOCATIONS: u32 = 256;

/// The memory spaces a memo entry is stamped against, in stamp order.
pub(crate) const MEM_SPACES: [MemReads; 3] =
    [MemReads::STORAGE, MemReads::TILE, MemReads::LOCAL];

/// One write counter per entry of [`MEM_SPACES`].
pub(crate) type MemStamp = [u32; MEM_SPACES.len()];

/// A validated module plus everything the launcher needs and cannot re-derive.
#[derive(Debug)]
pub struct EmittedModule {
    pub module: naga::Module,
    pub info: naga::valid::ModuleInfo,
    /// Derived from the module's storage globals — the *only* source of
    /// binding order in this crate.
    pub bindings: Vec<BindingDesc>,
    pub workgroup_size: [u32; 3],
    /// Whether the WGSL serialization needs `enable subgroups;`.
    pub subgroups: bool,
}

/// Emit one kernel. The result is validated before
/// `create_shader_module_trusted` is allowed anywhere near it.
///
/// The workgroup-arena plan comes from the *same* memoized `arena_plan` the L1
/// footprint check and the occupancy term read, never a local estimator — that
/// identity is what makes "extraction commits a plan that fails L2
/// verification" unstateable.
pub fn emit(ir: &KernelIr, caps: &Caps) -> Result<naga::Module, EmitError> {
    let planner = fusor2_tile::Planner::shared();
    let plan = <dyn fusor2_ir::ir::level2::ArenaPlanner>::arena_plan(&*planner, ir, caps)
        .map_err(|e| EmitError::Unsupported(format!("arena_plan: {e}")))?;
    Ok(emit_module(ir, caps, &plan)?.module)
}

/// Emit with a plan the caller already has, for callers holding the plan from
/// admission that must not recompute it.
pub fn emit_module(
    ir: &KernelIr,
    caps: &Caps,
    plan: &ArenaPlan,
) -> Result<EmittedModule, EmitError> {
    Emitter::new(ir, caps, plan)?.finish()
}

// Analysis

/// One walk of the whole body, run before any expression is lowered. Everything
/// the module declares — types, entry-point arguments, atomic buffer types and
/// the validation capability set — is decided here.
#[derive(Debug, Default)]
pub struct Analysis {
    pub uses_f16: bool,
    pub uses_bf16: bool,
    /// A `u32` holding two packed f16s is unpacked with `Unpack2x16Float`,
    /// which needs `SHADER_FLOAT16_IN_FLOAT32` but **not** `SHADER_F16`.
    pub unpacks_f16: bool,
    pub uses_coop: bool,
    pub uses_subgroup_collective: bool,
    pub subgroup_id: bool,
    pub subgroup_lane: bool,
    pub subgroup_size: bool,
    pub num_subgroups: bool,
    /// Bindings a `Stmt::AtomicAdd` targets. Typed `array<atomic<T>>` up front.
    pub atomic_buffers: FxHashSet<u32>,
    /// First-use-ordered, deduplicated declaration lists.
    pub buffers: Vec<Buffer>,
    pub tiles: Vec<Tile>,
    pub locals: Vec<Local>,
}

impl Analysis {
    /// Whether the module needs the `SUBGROUP` validation capability.
    pub fn uses_subgroups(&self) -> bool {
        self.uses_subgroup_collective
            || self.subgroup_id
            || self.subgroup_lane
            || self.subgroup_size
            || self.num_subgroups
    }

    pub fn run(ir: &KernelIr) -> Self {
        let mut a = Self::default();
        let mut seen = Seen::default();
        for buffer in &ir.buffers {
            a.note_buffer(buffer, &mut seen);
        }
        for stmt in &ir.body {
            a.stmt(stmt, &mut seen);
        }
        // Cooperative lowering addresses fragments per subgroup, so it needs a
        // subgroup id whether or not the body asked for one.
        a.subgroup_id |= a.uses_coop;
        a
    }

    fn note_buffer(&mut self, b: &Buffer, seen: &mut Seen) {
        if seen.buffers.insert(key(b)) {
            self.buffers.push(b.clone());
        }
        self.note_element(b.element);
    }

    fn note_tile(&mut self, t: &Tile, seen: &mut Seen) {
        if seen.tiles.insert(key(t)) {
            self.tiles.push(t.clone());
        }
        self.note_element(t.element);
    }

    fn note_local(&mut self, l: &Local, seen: &mut Seen) {
        if seen.locals.insert(key(l)) {
            self.locals.push(l.clone());
        }
        self.note_element(l.element);
    }

    fn note_element(&mut self, e: ElementType) {
        let scalar = match e {
            ElementType::Scalar(s) => s,
            ElementType::Vector { scalar, .. } => scalar,
            ElementType::CoopMatrix { scalar, .. } => {
                self.uses_coop = true;
                scalar
            }
        };
        match scalar {
            ScalarElement::F16 => self.uses_f16 = true,
            ScalarElement::BF16 => self.uses_bf16 = true,
            _ => {}
        }
    }

    fn stmt(&mut self, stmt: &Stmt, seen: &mut Seen) {
        match stmt {
            Stmt::Store {
                dst,
                addr,
                value,
                mask,
            } => {
                self.note_buffer(&dst.buffer, seen);
                self.addr(addr, seen);
                self.expr(value, seen);
                self.expr(mask, seen);
            }
            Stmt::AtomicAdd {
                dst,
                addr,
                value,
                mask,
            } => {
                self.atomic_buffers.insert(dst.buffer.binding);
                self.note_buffer(&dst.buffer, seen);
                self.addr(addr, seen);
                self.expr(value, seen);
                self.expr(mask, seen);
            }
            Stmt::StoreLocal { dst, value } => {
                self.note_local(dst, seen);
                self.expr(value, seen);
            }
            Stmt::StoreTile { dst, index, value } => {
                self.note_tile(dst, seen);
                self.expr(index, seen);
                self.expr(value, seen);
            }
            Stmt::FillTile { dst, value, bounds } => {
                self.note_tile(dst, seen);
                self.expr(value, seen);
                for b in bounds.iter().flatten() {
                    self.expr(b, seen);
                }
            }
            Stmt::CoopStore { acc, dst, addr } => {
                self.uses_coop = true;
                self.expr(acc, seen);
                self.note_buffer(&dst.buffer, seen);
                self.addr(addr, seen);
            }
            Stmt::CoopStoreTile {
                acc,
                tile,
                row,
                col,
            } => {
                self.uses_coop = true;
                self.expr(acc, seen);
                self.note_tile(tile, seen);
                self.expr(row, seen);
                self.expr(col, seen);
            }
            Stmt::If {
                condition,
                accept,
                reject,
            } => {
                self.expr(condition, seen);
                for s in accept.iter().chain(reject) {
                    self.stmt(s, seen);
                }
            }
            Stmt::Loop {
                count,
                index,
                accumulators,
                body,
            } => {
                if let Some(c) = count {
                    self.expr(c, seen);
                }
                if let Some(i) = index {
                    self.note_local(i, seen);
                }
                for Accumulator {
                    local,
                    init,
                    update,
                } in accumulators
                {
                    self.note_local(local, seen);
                    self.expr(init, seen);
                    self.expr(update, seen);
                }
                for s in body {
                    self.stmt(s, seen);
                }
            }
            Stmt::Reduce {
                kind,
                values,
                merge,
                outs,
                scratch,
                ..
            } => {
                if matches!(&**kind, ReduceKind::Subgroup) {
                    self.uses_subgroup_collective = true;
                }
                if let ReduceKind::Loop { index, .. } = &**kind {
                    self.note_local(index, seen);
                }
                for tile in scratch {
                    self.note_tile(tile, seen);
                }
                for local in merge.lhs.iter().chain(&merge.rhs).chain(outs) {
                    self.note_local(local, seen);
                }
                for e in values.iter().chain(&merge.body) {
                    self.expr(e, seen);
                }
            }
            Stmt::Break | Stmt::Return | Stmt::Barrier | Stmt::StorageBarrier => {}
        }
    }

    fn addr(&mut self, addr: &Addr, seen: &mut Seen) {
        match addr {
            Addr::Linear(e) => self.expr(e, seen),
            Addr::Rc2 { row, col } => {
                self.expr(row, seen);
                self.expr(col, seen);
            }
        }
    }

    fn expr(&mut self, e: &TileExpr, seen: &mut Seen) {
        // The hash-cons makes repeated subtrees cheap to skip, and the walk is
        // therefore linear in *distinct* nodes rather than in tree size.
        if !seen.exprs.insert(e.clone()) {
            return;
        }
        self.note_element(e.element());
        match e.kind() {
            TileExprKind::Literal(_) | TileExprKind::CoopZero { .. } => {}
            TileExprKind::Builtin(b) => match b {
                Builtin::SubgroupId => self.subgroup_id = true,
                Builtin::SubgroupLane => self.subgroup_lane = true,
                Builtin::SubgroupSize => self.subgroup_size = true,
                Builtin::NumSubgroups => self.num_subgroups = true,
                Builtin::Lane | Builtin::ProgramId(_) => {}
            },
            TileExprKind::LoadLocal(l) => self.note_local(l, seen),
            TileExprKind::Load {
                src,
                addr,
                mask,
                fill,
            } => {
                self.source(src, seen);
                self.addr(addr, seen);
                self.expr(mask, seen);
                self.expr(fill, seen);
            }
            TileExprKind::LoadTile { tile, index } => {
                self.note_tile(tile, seen);
                self.expr(index, seen);
            }
            TileExprKind::Unary { op, value, .. } => {
                if matches!(op, fusor2_ir::scalar::UnOp::Unpack2x16Float) {
                    self.unpacks_f16 = true;
                }
                self.expr(value, seen);
            }
            TileExprKind::Binary { left, right, .. }
            | TileExprKind::Compare { left, right, .. }
            | TileExprKind::Dot { left, right } => {
                self.expr(left, seen);
                self.expr(right, seen);
            }
            TileExprKind::Round { value, .. } => self.expr(value, seen),
            TileExprKind::Cast { value, to } | TileExprKind::Bitcast { value, to } => {
                self.note_element(*to);
                self.expr(value, seen);
            }
            TileExprKind::Select {
                condition,
                accept,
                reject,
            } => {
                self.expr(condition, seen);
                self.expr(accept, seen);
                self.expr(reject, seen);
            }
            TileExprKind::Vec { parts, .. } => {
                for p in parts {
                    self.expr(p, seen);
                }
            }
            TileExprKind::VecComponent { vector, .. } => self.expr(vector, seen),
            TileExprKind::Reduce { kind, value, .. } => {
                match &**kind {
                    ReduceKind::Subgroup => self.uses_subgroup_collective = true,
                    ReduceKind::Workgroup { scratch, .. } => self.note_tile(scratch, seen),
                    ReduceKind::Loop { index, scratch, .. } => {
                        self.note_local(index, seen);
                        self.note_tile(scratch, seen);
                    }
                }
                self.expr(value, seen);
            }
            TileExprKind::CoopLoad { src, .. } => {
                self.uses_coop = true;
                match &**src {
                    CoopSrc::TileRegion { tile, row, col, .. } => {
                        self.note_tile(tile, seen);
                        self.expr(row, seen);
                        self.expr(col, seen);
                    }
                    CoopSrc::BroadcastCol { src, col } => {
                        self.note_buffer(&src.buffer, seen);
                        self.expr(col, seen);
                    }
                }
            }
            TileExprKind::CoopMma { a, b, c } => {
                self.uses_coop = true;
                self.expr(a, seen);
                self.expr(b, seen);
                self.expr(c, seen);
            }
        }
    }

    fn source(&mut self, src: &Source, seen: &mut Seen) {
        match src {
            Source::Storage(v) => self.note_buffer(&v.buffer, seen),
            Source::Quantized(q) => self.quantized(q, seen),
        }
    }

    fn quantized(&mut self, q: &fusor2_ir::ir::level2::QuantizedView, seen: &mut Seen) {
        self.note_buffer(&q.data.buffer, seen);
        // Native-layout scales are half floats read out of a u32 word with
        // `Unpack2x16Float`; that needs SHADER_FLOAT16_IN_FLOAT32, not
        // SHADER_F16.
        if q.layout == fusor2_ir::dtype::QLayout::Native {
            self.unpacks_f16 = true;
        }
    }
}

#[derive(Default)]
struct Seen {
    buffers: FxHashSet<usize>,
    tiles: FxHashSet<usize>,
    locals: FxHashSet<usize>,
    exprs: FxHashSet<TileExpr>,
}

/// `Arc`-identity key for a declaration.
pub(crate) fn key<T>(v: &std::sync::Arc<T>) -> usize {
    std::sync::Arc::as_ptr(v) as *const () as usize
}

// Scratch

/// Demand-allocated private scratch kinds, interned by
/// `(kind, element, depth)`. Only keys that are actually used allocate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScratchKind {
    LoopIndex,
    /// A masked-value spill local (one per masked load/reduce).
    Value,
    /// A reduce-accumulator spill local; deepens with nesting.
    Spill,
    /// An atomic compare-exchange loop's `old`/`ok` slots.
    Atomic,
}

// Emitter

/// Per-kernel emission state: the naga arenas plus the L2 -> naga handle maps.
pub struct Emitter<'a> {
    pub caps: &'a Caps,
    pub module: naga::Module,
    pub(crate) ir: &'a KernelIr,
    pub(crate) plan: &'a ArenaPlan,
    pub(crate) analysis: Analysis,
    pub(crate) exprs: naga::Arena<naga::Expression>,
    pub(crate) fn_locals: naga::Arena<naga::LocalVariable>,
    /// Buffer decl key -> storage global.
    pub(crate) buffer_globals: FxHashMap<usize, naga::Handle<naga::GlobalVariable>>,
    /// Tile decl key -> how that tile is backed in workgroup memory.
    pub(crate) tile_backing: FxHashMap<usize, types::TileBacking>,
    /// Private tiles and program locals.
    pub(crate) local_handles: FxHashMap<usize, naga::Handle<naga::LocalVariable>>,
    pub(crate) scratch:
        FxHashMap<(ScratchKind, ElementType, u32), naga::Handle<naga::LocalVariable>>,
    /// Expressions to force into a named temporary, drained into
    /// [`naga::Function::named_expressions`] by [`Emitter::finish`].
    ///
    /// A backend inlines a single-use expression into its consumer, so a run
    /// of them nests one inside the next and Metal's front end refuses past
    /// 256 brackets. Naming one emits `const auto _n = <expr>;` — an SSA
    /// binding, not a memory round trip, which is what makes it cheaper than
    /// spilling through a [`Self::scratch`] local.
    pub(crate) forced_names: Vec<(naga::Handle<naga::Expression>, String)>,
    /// Hash-consed expression memo. `TileExpr: Hash + Eq` is O(1) through its
    /// cached structural hash, so two identical subtrees built separately
    /// merge here — which pointer-identity memoization cannot do.
    ///
    /// Each entry carries the [`Self::mem_epoch`] it was created at. A key
    /// that reads memory is a hit only while every space it reads is still at
    /// that epoch — see [`Emitter::expr`].
    pub(crate) memo: FxHashMap<TileExpr, (naga::Handle<naga::Expression>, MemStamp)>,
    /// Write counter per memory space, indexed by [`MEM_SPACES`] order:
    /// storage, workgroup tile, private local.
    ///
    /// **Monotonic, and deliberately not part of [`expr::Scope`].** A store
    /// inside an `If` or a loop body must invalidate the *enclosing* block's
    /// memoized reads as well, and restoring a saved counter on scope exit
    /// would resurrect exactly the stale entries the store invalidated.
    pub(crate) mem_epoch: MemStamp,
    /// Latest SSA value of each cooperative accumulator local: one `Load`,
    /// N MMAs, one `Store` per scope.
    pub(crate) coop_acc: FxHashMap<usize, naga::Handle<naga::Expression>>,
    pub(crate) workgroup_invocations: u32,
    pub(crate) workgroup_size: [u32; 3],
    /// Argument indices of the four optional subgroup builtins, in the fixed
    /// order they are appended.
    pub(crate) subgroup_args: [Option<u32>; 4],
    pub(crate) depth: u32,
    pub(crate) u32_ty: naga::Handle<naga::Type>,
    pub(crate) u32_vec3_ty: naga::Handle<naga::Type>,
}

impl<'a> Emitter<'a> {
    pub fn new(ir: &'a KernelIr, caps: &'a Caps, plan: &'a ArenaPlan) -> Result<Self, EmitError> {
        let analysis = Analysis::run(ir);

        // Capability gates, all decided before a single type is interned.
        if analysis.uses_f16 && !caps.f16 {
            return Err(EmitError::MissingCapability("shader-f16"));
        }
        if analysis.uses_bf16 && !caps.bf16 {
            return Err(EmitError::MissingCapability("shader-bf16"));
        }
        if analysis.uses_coop && caps.coop.is_empty() {
            return Err(EmitError::MissingCapability("cooperative-matrix"));
        }
        if analysis.uses_subgroups() && caps.subgroups.is_none() {
            return Err(EmitError::MissingCapability("subgroups"));
        }

        let workgroup_invocations = if ir.block > 0 {
            ir.block
        } else {
            DEFAULT_WORKGROUP_INVOCATIONS
        };
        if ir.block != 0 && ir.block != workgroup_invocations {
            return Err(EmitError::Unsupported(
                "kernel block must match the workgroup size".into(),
            ));
        }
        if workgroup_invocations > caps.limits.max_compute_invocations_per_workgroup {
            return Err(EmitError::LimitExceeded(format!(
                "block {workgroup_invocations} exceeds max_compute_invocations_per_workgroup {}",
                caps.limits.max_compute_invocations_per_workgroup
            )));
        }

        let mut module = naga::Module::default();
        let prelude = types::intern_prelude(&mut module, &analysis)?;

        Ok(Self {
            caps,
            module,
            ir,
            plan,
            analysis,
            exprs: naga::Arena::new(),
            fn_locals: naga::Arena::new(),
            buffer_globals: FxHashMap::default(),
            tile_backing: FxHashMap::default(),
            local_handles: FxHashMap::default(),
            scratch: FxHashMap::default(),
            forced_names: Vec::new(),
            memo: FxHashMap::default(),
            mem_epoch: MemStamp::default(),
            coop_acc: FxHashMap::default(),
            workgroup_invocations,
            workgroup_size: [workgroup_invocations, 1, 1],
            subgroup_args: [None; 4],
            depth: 0,
            u32_ty: prelude.u32_ty,
            u32_vec3_ty: prelude.u32_vec3_ty,
        })
    }

    /// Build the globals, the one `main` entry point, and validate.
    pub fn finish(mut self) -> Result<EmittedModule, EmitError> {
        types::create_storage_globals(&mut self)?;
        types::create_workgroup_globals(&mut self)?;
        types::create_private_locals(&mut self)?;

        let mut arguments = vec![
            builtin_arg(self.u32_ty, naga::BuiltIn::LocalInvocationIndex),
            builtin_arg(self.u32_vec3_ty, naga::BuiltIn::WorkGroupId),
        ];
        // Fixed order: subgroup_id, subgroup_invocation_id, subgroup_size,
        // num_subgroups; each appended only when used.
        let optional = [
            (self.analysis.subgroup_id, naga::BuiltIn::SubgroupId),
            (
                self.analysis.subgroup_lane,
                naga::BuiltIn::SubgroupInvocationId,
            ),
            (self.analysis.subgroup_size, naga::BuiltIn::SubgroupSize),
            (self.analysis.num_subgroups, naga::BuiltIn::NumSubgroups),
        ];
        for (slot, (used, builtin)) in optional.into_iter().enumerate() {
            if used {
                self.subgroup_args[slot] = Some(arguments.len() as u32);
                arguments.push(builtin_arg(self.u32_ty, builtin));
            }
        }

        let mut body = naga::Block::new();
        let mut inner = naga::Block::new();
        let stmts = self.ir.body.clone();
        for stmt in &stmts {
            self.stmt(stmt, &mut inner)?;
        }
        self.flush_coop_acc(&mut inner);
        body.push(naga::Statement::Block(inner), naga::Span::default());
        body.push(
            naga::Statement::Return { value: None },
            naga::Span::default(),
        );

        let mut function = naga::Function {
            name: None,
            arguments,
            result: None,
            local_variables: std::mem::take(&mut self.fn_locals),
            expressions: std::mem::take(&mut self.exprs),
            named_expressions: Default::default(),
            body,
            diagnostic_filter_leaf: None,
        };
        for (handle, name) in std::mem::take(&mut self.forced_names) {
            function.named_expressions.insert(handle, name);
        }

        let workgroup_size = self.workgroup_size;
        let subgroups = self.analysis.uses_subgroups();
        self.module.entry_points.push(naga::EntryPoint {
            name: "main".into(),
            stage: naga::ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size,
            workgroup_size_overrides: None,
            function,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });

        let info = self.validate()?;
        let bindings = bindings_from_module(&self.module);
        Ok(EmittedModule {
            module: self.module,
            info,
            bindings,
            workgroup_size,
            subgroups,
        })
    }

    /// Run naga's validator with exactly the capabilities the analysis raised.
    /// A failure here is a compiler bug, not a user error — it is what licenses
    /// the trusted shader-module path, and it is never a silent fallback.
    pub fn validate(&self) -> Result<naga::valid::ModuleInfo, EmitError> {
        use naga::valid::Capabilities as C;
        let mut caps = C::empty();
        if self.analysis.uses_f16 {
            caps |= C::SHADER_FLOAT16;
        }
        if self.analysis.unpacks_f16 || self.analysis.uses_f16 {
            caps |= C::SHADER_FLOAT16_IN_FLOAT32;
        }
        if self.analysis.uses_subgroups() {
            caps |= C::SUBGROUP;
        }
        if self.analysis.uses_coop {
            caps |= C::COOPERATIVE_MATRIX;
        }
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps)
            .validate(&self.module)
            .map_err(|e| EmitError::Validation(format!("{e:#?}")))
    }
}

fn builtin_arg(ty: naga::Handle<naga::Type>, builtin: naga::BuiltIn) -> naga::FunctionArgument {
    naga::FunctionArgument {
        name: None,
        ty,
        binding: Some(naga::Binding::BuiltIn(builtin)),
    }
}

// Test kit

/// Fixture builders and a minimal dispatcher, shared by every emit test in
/// this crate. Test-only: nothing here is compiled into a release build, and
/// the launcher proper lives in `crate::launch`.
#[cfg(test)]
pub(crate) mod testkit {
    use super::*;
    use fusor2_ir::device::{Caps, DeviceKind, Limits, SubgroupWidths};
    use fusor2_ir::ir::level2::{
        ArenaMode, BufferAccess, BufferDecl, LocalDecl, MemoryLevel, Placement, StorageView,
        TileDecl, TileLayout, TileLiteral,
    };
    use std::sync::Arc;

    pub fn f32e() -> ElementType {
        ElementType::Scalar(ScalarElement::F32)
    }
    pub fn u32e() -> ElementType {
        ElementType::Scalar(ScalarElement::U32)
    }
    pub fn boole() -> ElementType {
        ElementType::Scalar(ScalarElement::Bool)
    }
    pub fn f16e() -> ElementType {
        ElementType::Scalar(ScalarElement::F16)
    }

    pub fn caps(f16: bool, subgroups: bool) -> Caps {
        Caps {
            kind: DeviceKind::Gpu,
            name: "test".into(),
            limits: Limits::default(),
            subgroups: subgroups.then_some(SubgroupWidths { min: 32, max: 32 }),
            f16,
            bf16: false,
            coop: Default::default(),
            atomic_f32: true,
            workgroup_alias: false,
            mixed_precision_coop_store: false,
            pipeline_cache: false,
            timestamp_query: false,
            simd_widths: Default::default(),
            threads: 1,
        }
    }

    pub fn buffer(binding: u32, element: ElementType, len: u32, write: bool) -> Buffer {
        Arc::new(BufferDecl {
            binding,
            element,
            layout: TileLayout::contiguous(MemoryLevel::Storage, &[len]),
            access: if write {
                BufferAccess::ReadWrite
            } else {
                BufferAccess::Read
            },
        })
    }

    pub fn view(buffer: &Buffer, extents: &[u32]) -> StorageView {
        StorageView {
            buffer: buffer.clone(),
            offset: 0,
            layout: TileLayout::contiguous(MemoryLevel::Storage, extents),
        }
    }

    pub fn tile(element: ElementType, extents: &[u32]) -> Tile {
        Arc::new(TileDecl::new(
            element,
            TileLayout::contiguous(MemoryLevel::Workgroup, extents),
            "t",
        ))
    }

    pub fn local(element: ElementType) -> Local {
        Arc::new(LocalDecl::new(element))
    }

    pub fn lit_u32(v: u32) -> TileExpr {
        TileExpr::new(TileExprKind::Literal(TileLiteral::U32(v)), u32e())
    }
    pub fn lit_f32(v: f32) -> TileExpr {
        TileExpr::new(TileExprKind::Literal(TileLiteral::F32(v.to_bits())), f32e())
    }
    pub fn tru() -> TileExpr {
        TileExpr::new(TileExprKind::Literal(TileLiteral::Bool(true)), boole())
    }
    pub fn lane() -> TileExpr {
        TileExpr::new(TileExprKind::Builtin(Builtin::Lane), u32e())
    }
    pub fn load(v: &StorageView, index: TileExpr) -> TileExpr {
        let element = v.buffer.element;
        TileExpr::new(
            TileExprKind::Load {
                src: Source::Storage(v.clone()),
                addr: Box::new(Addr::Linear(index)),
                mask: tru(),
                fill: lit_f32(0.0),
            },
            element,
        )
    }
    pub fn store(v: &StorageView, index: TileExpr, value: TileExpr) -> Stmt {
        Stmt::Store {
            dst: v.clone(),
            addr: Addr::Linear(index),
            value,
            mask: tru(),
        }
    }
    pub fn un(op: fusor2_ir::scalar::UnOp, x: TileExpr) -> TileExpr {
        let ty = x.element();
        TileExpr::new(
            TileExprKind::Unary {
                op,
                value: x,
                numeric: fusor2_ir::dtype::NumericContract::RELAXED,
            },
            ty,
        )
    }
    pub fn bin(
        op: fusor2_ir::scalar::BinOp,
        a: TileExpr,
        b: TileExpr,
        numeric: fusor2_ir::dtype::NumericContract,
    ) -> TileExpr {
        let ty = a.element();
        TileExpr::new(
            TileExprKind::Binary {
                op,
                left: a,
                right: b,
                numeric,
            },
            ty,
        )
    }

    /// The plan a kernel with no shared tiles gets: every tile keeps its own
    /// allocation.
    pub fn no_plan() -> ArenaPlan {
        ArenaPlan {
            mode: ArenaMode::Regions,
            total_bytes: 0,
            placements: Default::default(),
            barriers_inserted: Default::default(),
        }
    }

    /// A `Regions` plan that puts every listed tile in one shared allocation.
    pub fn shared_region(tiles: &[Tile]) -> ArenaPlan {
        let mut placements = smallvec::SmallVec::new();
        let mut total = 0u32;
        for t in tiles {
            let bytes =
                t.layout.element_count() as u32 * t.element.workgroup_array_stride().unwrap_or(4);
            total = total.max(bytes);
            placements.push(Placement {
                tile: t.clone(),
                byte_offset: 0,
                byte_len: bytes,
            });
        }
        ArenaPlan {
            mode: ArenaMode::Regions,
            total_bytes: total,
            placements,
            barriers_inserted: Default::default(),
        }
    }

    /// A `ByteArena` plan that packs the listed tiles end to end.
    pub fn byte_arena(tiles: &[Tile]) -> ArenaPlan {
        let mut placements = smallvec::SmallVec::new();
        let mut offset = 0u32;
        for t in tiles {
            let bytes =
                t.layout.element_count() as u32 * t.element.workgroup_array_stride().unwrap_or(4);
            placements.push(Placement {
                tile: t.clone(),
                byte_offset: offset,
                byte_len: bytes,
            });
            offset += bytes;
        }
        ArenaPlan {
            mode: ArenaMode::ByteArena,
            total_bytes: offset,
            placements,
            barriers_inserted: Default::default(),
        }
    }

    /// The one `main` function of an emitted module.
    pub fn main_fn(module: &naga::Module) -> &naga::Function {
        &module.entry_points[0].function
    }

    pub fn count_exprs(module: &naga::Module, f: impl Fn(&naga::Expression) -> bool) -> usize {
        main_fn(module)
            .expressions
            .iter()
            .filter(|(_, e)| f(e))
            .count()
    }

    /// A device probed exactly the way the shipped path probes it.
    pub fn gpu() -> Option<crate::device::GpuDevice> {
        crate::device::gpu_blocking(&crate::device::DeviceOptions::default()).ok()
    }

    /// Emit, compile and dispatch `ir`, returning binding `out`'s contents.
    ///
    /// `inputs` is positional against the derived binding list, so the test
    /// exercises the same derivation the launcher does.
    pub fn run(
        gpu: &crate::device::GpuDevice,
        ir: &KernelIr,
        plan: &ArenaPlan,
        inputs: &[Vec<u8>],
        out: usize,
    ) -> Vec<u8> {
        run_emitted(
            gpu,
            ir,
            emit_module(ir, gpu.caps(), plan).expect("emit"),
            inputs,
            out,
        )
    }

    /// As [`run`], for a module the caller already emitted.
    pub fn run_emitted(
        gpu: &crate::device::GpuDevice,
        ir: &KernelIr,
        emitted: EmittedModule,
        inputs: &[Vec<u8>],
        out: usize,
    ) -> Vec<u8> {
        let device = gpu.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(ir.name),
            source: wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(emitted.module)),
        });
        let entries = crate::bindings::layout_entries(&emitted.bindings);
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let buffers: Vec<wgpu::Buffer> = inputs
            .iter()
            .map(|bytes| {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: bytes.len().max(4) as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                gpu.queue().write_buffer(&buffer, 0, bytes);
                buffer
            })
            .collect();
        let bind_entries = zip_buffers(&emitted.bindings, &buffers).expect("zip buffers");
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &bind_entries,
        });

        let size = inputs[out].len() as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(ir.grid[0], ir.grid[1], ir.grid[2]);
        }
        encoder.copy_buffer_to_buffer(&buffers[out], 0, &staging, 0, size);
        gpu.queue().submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map").expect("map ok");
        let data = slice.get_mapped_range().to_vec();
        staging.unmap();
        data
    }

    pub fn f32s(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    pub fn bytes_of(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }
    /// Binding 0 is always the read-only `Uniforms` storage buffer.
    pub fn uniforms() -> Vec<u8> {
        vec![0u8; 16]
    }

    /// Zip a derived binding list positionally with the caller's buffers.
    ///
    /// **Binding 0 is always the read-only `Uniforms` storage buffer.** A
    /// module whose binding 0 is writable is rejected here rather than at
    /// dispatch, and a length mismatch is a validation failure, never a
    /// silent truncation.
    pub fn zip_buffers<'a>(
        slots: &[crate::bindings::BindingDesc],
        buffers: &'a [wgpu::Buffer],
    ) -> Result<Vec<wgpu::BindGroupEntry<'a>>, EmitError> {
        check_uniform_binding(slots)?;
        if slots.len() != buffers.len() {
            return Err(EmitError::Validation(format!(
                "{} derived bindings against {} buffers",
                slots.len(),
                buffers.len()
            )));
        }
        Ok(slots
            .iter()
            .zip(buffers)
            .map(|(slot, buffer)| wgpu::BindGroupEntry {
                binding: slot.binding,
                resource: buffer.as_entire_binding(),
            })
            .collect())
    }

    /// Assert the `Uniforms` invariant: binding 0 exists and is read-only.
    pub fn check_uniform_binding(
        slots: &[crate::bindings::BindingDesc],
    ) -> Result<(), EmitError> {
        match slots.first() {
            Some(first) if first.binding == 0 && first.read_only => Ok(()),
            Some(first) if first.binding == 0 => Err(EmitError::Validation(
                "binding 0 is the Uniforms block and must be read-only".into(),
            )),
            Some(_) | None => Err(EmitError::Validation(
                "binding 0 must be the Uniforms storage buffer".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::testkit::*;
    use super::*;
    use fusor2_ir::dtype::NumericContract;
    use fusor2_ir::ir::level2::{ArenaMode, ReduceKind, TileReduceOp};
    use fusor2_ir::scalar::{BinOp, UnOp};

    /// out[lane] = exp(in[lane]) + 1
    fn elementwise() -> KernelIr {
        let uni = buffer(0, u32e(), 4, false);
        let src = buffer(1, f32e(), 256, false);
        let dst = buffer(2, f32e(), 256, true);
        let sv = view(&src, &[256]);
        let dv = view(&dst, &[256]);
        let value = bin(
            BinOp::Add,
            un(UnOp::Exp, load(&sv, lane())),
            lit_f32(1.0),
            NumericContract::RELAXED,
        );
        KernelIr {
            buffers: vec![uni, src, dst],
            grid: [1, 1, 1],
            block: 256,
            body: vec![store(&dv, lane(), value)],
            byte_arena: None,
            name: "elementwise",
        }
    }

    /// A workgroup-tree sum written by lane 0.
    fn tree_sum() -> KernelIr {
        let uni = buffer(0, u32e(), 4, false);
        let src = buffer(1, f32e(), 256, false);
        let dst = buffer(2, f32e(), 1, true);
        let sv = view(&src, &[256]);
        let dv = view(&dst, &[1]);
        let scratch = tile(f32e(), &[256]);
        let total = TileExpr::new(
            TileExprKind::Reduce {
                op: TileReduceOp::Sum,
                kind: Box::new(ReduceKind::Workgroup {
                    scratch,
                    group_size: 256,
                }),
                value: load(&sv, lane()),
            },
            f32e(),
        );
        let is_zero = TileExpr::new(
            TileExprKind::Compare {
                op: fusor2_ir::scalar::CmpOp::Eq,
                left: lane(),
                right: lit_u32(0),
            },
            boole(),
        );
        KernelIr {
            buffers: vec![uni, src, dst],
            grid: [1, 1, 1],
            block: 256,
            body: vec![Stmt::Store {
                dst: dv,
                addr: Addr::Linear(lit_u32(0)),
                value: total,
                mask: is_zero,
            }],
            byte_arena: None,
            name: "tree_sum",
        }
    }

    /// Two workgroup tiles that a plan may or may not share.
    fn two_tiles() -> (KernelIr, Tile, Tile) {
        let uni = buffer(0, u32e(), 4, false);
        let dst = buffer(1, f32e(), 256, true);
        let dv = view(&dst, &[256]);
        let a = tile(f32e(), &[256]);
        let b = tile(f32e(), &[256]);
        let body = vec![
            Stmt::StoreTile {
                dst: a.clone(),
                index: lane(),
                value: lit_f32(2.0),
            },
            Stmt::Barrier,
            Stmt::StoreTile {
                dst: b.clone(),
                index: lane(),
                value: TileExpr::new(
                    TileExprKind::LoadTile {
                        tile: a.clone(),
                        index: lane(),
                    },
                    f32e(),
                ),
            },
            Stmt::Barrier,
            store(
                &dv,
                lane(),
                TileExpr::new(
                    TileExprKind::LoadTile {
                        tile: b.clone(),
                        index: lane(),
                    },
                    f32e(),
                ),
            ),
        ];
        (
            KernelIr {
                buffers: vec![uni, dst],
                grid: [1, 1, 1],
                block: 256,
                body,
                byte_arena: None,
                name: "two_tiles",
            },
            a,
            b,
        )
    }

    /// One `main`, the right workgroup size, the right argument list.
    #[test]
    fn every_module_has_one_main() {
        let caps = caps(false, true);
        let (tiles_ir, _, _) = two_tiles();
        for ir in [elementwise(), tree_sum(), tiles_ir] {
            let emitted = emit_module(&ir, &caps, &no_plan()).expect(ir.name);
            let m = &emitted.module;
            assert_eq!(m.entry_points.len(), 1, "{}", ir.name);
            assert_eq!(m.entry_points[0].name, "main");
            assert_eq!(m.entry_points[0].stage, naga::ShaderStage::Compute);
            assert_eq!(m.entry_points[0].workgroup_size, [ir.block, 1, 1]);
            let args = &main_fn(m).arguments;
            assert!(args.len() >= 2);
            assert_eq!(
                args[0].binding,
                Some(naga::Binding::BuiltIn(naga::BuiltIn::LocalInvocationIndex))
            );
            assert_eq!(
                args[1].binding,
                Some(naga::Binding::BuiltIn(naga::BuiltIn::WorkGroupId))
            );
            // Subgroup arguments appear only when the body uses them; none of
            // these fixtures does.
            assert_eq!(args.len(), 2, "{} grew a subgroup argument", ir.name);
            assert!(!emitted.subgroups);
        }
    }

    /// A subgroup collective is the one fixture that grows argument 2.
    #[test]
    fn subgroup_args_appear_only_when_used() {
        let uni = buffer(0, u32e(), 4, false);
        let src = buffer(1, f32e(), 256, false);
        let dst = buffer(2, f32e(), 1, true);
        let sv = view(&src, &[256]);
        let dv = view(&dst, &[1]);
        let total = TileExpr::new(
            TileExprKind::Reduce {
                op: TileReduceOp::Sum,
                kind: Box::new(ReduceKind::Subgroup),
                value: load(&sv, lane()),
            },
            f32e(),
        );
        let ir = KernelIr {
            buffers: vec![uni, src, dst],
            grid: [1, 1, 1],
            block: 32,
            body: vec![store(&dv, lit_u32(0), total)],
            byte_arena: None,
            name: "subgroup_sum",
        };
        let emitted = emit_module(&ir, &caps(false, true), &no_plan()).expect("emit");
        assert!(emitted.subgroups);
        // A collective needs no builtin argument of its own.
        assert_eq!(main_fn(&emitted.module).arguments.len(), 2);
        // ... and the module must fail on a device without subgroups.
        assert_eq!(
            emit_module(&ir, &caps(false, false), &no_plan()).unwrap_err(),
            EmitError::MissingCapability("subgroups")
        );
    }

    /// f16 is gated up front, not lazily.
    #[test]
    fn f16_without_capability_is_unsupported() {
        let uni = buffer(0, u32e(), 4, false);
        let src = buffer(1, f16e(), 256, false);
        let dst = buffer(2, f16e(), 256, true);
        let sv = view(&src, &[256]);
        let dv = view(&dst, &[256]);
        let ir = KernelIr {
            buffers: vec![uni, src, dst],
            grid: [1, 1, 1],
            block: 256,
            body: vec![store(&dv, lane(), load(&sv, lane()))],
            byte_arena: None,
            name: "f16_copy",
        };
        assert_eq!(
            emit_module(&ir, &caps(false, true), &no_plan()).unwrap_err(),
            EmitError::MissingCapability("shader-f16")
        );

        // With the capability it emits, and the 2-byte float appears.
        let ok = emit_module(&ir, &caps(true, true), &no_plan()).expect("f16 emit");
        assert!(has_two_byte_float(&ok.module));
        // A kernel that never mentions f16 never interns one.
        let plain = emit_module(&elementwise(), &caps(true, true), &no_plan()).expect("emit");
        assert!(!has_two_byte_float(&plain.module));
    }

    fn has_two_byte_float(module: &naga::Module) -> bool {
        module.types.iter().any(|(_, t)| {
            matches!(
                t.inner,
                naga::TypeInner::Scalar(naga::Scalar {
                    kind: naga::ScalarKind::Float,
                    width: 2
                })
            )
        })
    }

    /// The type arena is deterministic.
    #[test]
    fn type_arena_is_deterministic() {
        let caps = caps(false, true);
        for ir in [elementwise(), tree_sum()] {
            let a = emit_module(&ir, &caps, &no_plan()).expect("a").module;
            let b = emit_module(&ir, &caps, &no_plan()).expect("b").module;
            assert_eq!(format!("{a:#?}"), format!("{b:#?}"), "{}", ir.name);
        }
        // Two structurally identical IRs built independently agree as well,
        // which is what the hash-consed memo buys over pointer identity.
        let x = emit_module(&elementwise(), &caps, &no_plan())
            .expect("x")
            .module;
        let y = emit_module(&elementwise(), &caps, &no_plan())
            .expect("y")
            .module;
        assert_eq!(format!("{x:#?}"), format!("{y:#?}"));
    }

    /// The byte arena is a footprint choice, never a numeric one.
    #[test]
    fn byte_arena_absence_is_only_footprint() {
        let (ir, a, b) = two_tiles();
        let regions = shared_region(&[a.clone(), b.clone()]);
        let arena = byte_arena(&[a, b]);
        assert_eq!(regions.mode, ArenaMode::Regions);
        assert_eq!(arena.mode, ArenaMode::ByteArena);
        // Only the footprint differs: the shared region overlaps the two
        // tiles, the byte arena packs them end to end.
        assert!(arena.total_bytes > regions.total_bytes);

        let caps = caps(false, true);
        let m_regions = emit_module(&ir, &caps, &regions).expect("regions");
        let m_arena = emit_module(&ir, &caps, &arena).expect("arena");
        // No `WorkgroupAlias` is emitted in either module: released naga has
        // no such decoration, so the arena is index arithmetic instead.
        assert_eq!(workgroup_globals(&m_regions.module), 1);
        assert_eq!(workgroup_globals(&m_arena.module), 1);

        let Some(gpu) = gpu() else {
            eprintln!("no wgpu adapter; skipping the numeric half");
            return;
        };
        let inputs = vec![uniforms(), bytes_of(&[0.0; 256])];
        let out_regions = f32s(&run(&gpu, &ir, &regions, &inputs, 1));
        let out_arena = f32s(&run(&gpu, &ir, &arena, &inputs, 1));
        assert_eq!(out_regions, vec![2.0f32; 256]);
        assert_eq!(out_regions, out_arena);
    }

    fn workgroup_globals(module: &naga::Module) -> usize {
        module
            .global_variables
            .iter()
            .filter(|(_, g)| g.space == naga::AddressSpace::WorkGroup)
            .count()
    }

    /// A kernel wider than the baseline's invocation limit is refused rather
    /// than silently clamped.
    #[test]
    fn oversize_block_is_a_limit_error() {
        let mut ir = elementwise();
        ir.block = 1024;
        assert!(matches!(
            emit_module(&ir, &caps(false, true), &no_plan()),
            Err(EmitError::LimitExceeded(_))
        ));
    }
}
