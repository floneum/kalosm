//! `KernelIr` -> naga `Module`.
//!
//! One up-front [`Analysis`] walk decides every capability the module will
//! declare *before* a single expression is lowered. Gating f16 up front is
//! what makes an f16 handle on a non-f16 adapter fail with
//! [`EmitError::MissingCapability`] instead of mis-lowering.

pub(crate) mod coop;
pub(crate) mod expr;
pub(crate) mod quantized;
pub(crate) mod reduce;
pub(crate) mod stmt;
pub(crate) mod types;

use fusor_ir::device::Caps;
use fusor_ir::ir::kernel::{
    Accumulator, Addr, ArenaPlan, Buffer, Builtin, CoopSrc, ElementType, KernelIr, Local, MemReads,
    ReduceKind, ScalarElement, Source, Stmt, Tile, TileExpr, TileExprKind,
};
use fusor_ir::target::EmitError;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::bindings::{BindingDesc, bindings_from_module};

/// `@builtin(local_invocation_index)`, always argument 0.
pub(crate) const LOCAL_INVOCATION_INDEX_ARG: u32 = 0;
/// `@builtin(workgroup_id)`, always argument 1.
pub(crate) const WORKGROUP_ID_ARG: u32 = 1;
/// Lanes assumed when `KernelIr::block` is zero.
pub(crate) const DEFAULT_WORKGROUP_INVOCATIONS: u32 = 256;

/// The memory spaces a memo entry is stamped against, in stamp order.
pub(crate) const MEM_SPACES: [MemReads; 3] = [MemReads::STORAGE, MemReads::TILE, MemReads::LOCAL];

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
/// The workgroup-arena plan comes from the *same* memoized `arena_plan` the Launch
/// footprint check and the occupancy term read, never a local estimator — that
/// identity is what makes "extraction commits a plan that fails Kernel
/// verification" unstateable.
pub fn emit(ir: &KernelIr, caps: &Caps) -> Result<EmittedModule, EmitError> {
    let planner = fusor_tile::Planner::shared();
    let plan = <dyn fusor_ir::ir::kernel::ArenaPlanner>::arena_plan(&*planner, ir, caps)
        .map_err(|e| EmitError::Unsupported(format!("arena_plan: {e}")))?;
    emit_module(ir, caps, &plan)
}

/// Emit with a plan the caller already has, which must not be recomputed.
pub(crate) fn emit_module(
    ir: &KernelIr,
    caps: &Caps,
    plan: &ArenaPlan,
) -> Result<EmittedModule, EmitError> {
    Emitter::new(ir, caps, plan)?.finish()
}

/// Whether a `ReduceKind::Workgroup` tree at `group_size` on a `block`-lane
/// kernel is emitted as the subgroup two-stage (one collective per subgroup,
/// partials staged through scratch, two barriers) instead of the barrier
/// tree (`2 + log2(block)` barriers). `width` is the *fixed* subgroup width,
/// `None` when the device has none: a varying width would make `block/width`
/// a guess, and a guessed slot count is a race, not a reduction.
///
/// Grouped trees (`group_size < block`) and non-scalar/bool elements keep
/// the tree — the collective reduces the whole subgroup, which crosses group
/// boundaries, and `subgroupAdd` on bool is undefined.
pub(crate) fn collective_tree(
    width: Option<u32>,
    block: u32,
    group_size: u32,
    element: ElementType,
) -> bool {
    let Some(w) = width else { return false };
    matches!(element, ElementType::Scalar(s) if s != ScalarElement::Bool)
        && group_size == block
        && block >= w
        && block.is_multiple_of(w)
}

/// One walk of the whole body, run before any expression is lowered. Everything
/// the module declares — types, entry-point arguments, atomic buffer types and
/// the validation capability set — is decided here.
#[derive(Debug, Default)]
pub(crate) struct Analysis {
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
    pub num_workgroups: bool,
    /// Bindings a `Stmt::AtomicAdd` targets. Typed `array<atomic<T>>` up front.
    pub atomic_buffers: FxHashSet<u32>,
    /// First-use-ordered, deduplicated declaration lists.
    pub buffers: Vec<Buffer>,
    pub tiles: Vec<Tile>,
    pub locals: Vec<Local>,
    /// The device's fixed subgroup width, when it has one — what decides
    /// whether a workgroup tree upgrades to the subgroup two-stage
    /// ([`collective_tree`]), so the walk flags the builtins that path reads.
    fixed_width: Option<u32>,
    /// The kernel's workgroup size, for the same decision.
    block: u32,
}

impl Analysis {
    /// Whether the module needs the `SUBGROUP` validation capability.
    pub(crate) fn uses_subgroups(&self) -> bool {
        self.uses_subgroup_collective
            || self.subgroup_id
            || self.subgroup_lane
            || self.subgroup_size
            || self.num_subgroups
    }

    pub(crate) fn run(ir: &KernelIr, caps: &Caps) -> Self {
        let mut a = Self::default();
        a.fixed_width = caps.subgroups.filter(|s| s.is_fixed()).map(|s| s.assumed());
        a.block = if ir.block > 0 {
            ir.block
        } else {
            DEFAULT_WORKGROUP_INVOCATIONS
        };
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

    /// A workgroup tree the emitter will upgrade to the subgroup two-stage
    /// reads the collective and — past one subgroup — the subgroup id and
    /// lane, none of which appear in the IR itself.
    fn note_tree(&mut self, group_size: u32, element: ElementType) {
        if collective_tree(self.fixed_width, self.block, group_size, element) {
            self.uses_subgroup_collective = true;
            if Some(self.block) != self.fixed_width {
                self.subgroup_id = true;
                self.subgroup_lane = true;
            }
        }
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
                fast,
                outs,
                scratch,
            } => {
                if matches!(&**kind, ReduceKind::Subgroup) {
                    self.uses_subgroup_collective = true;
                }
                // A single-slot fold with a hardware operator takes the
                // expression path (see `Stmt::Reduce` emission), so the same
                // two-stage upgrade applies to it.
                if values.len() == 1 && fast.is_some() {
                    match &**kind {
                        ReduceKind::Workgroup { group_size, .. }
                        | ReduceKind::Loop { group_size, .. } => {
                            self.note_tree(*group_size, values[0].element());
                        }
                        ReduceKind::Subgroup => {}
                    }
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
                Builtin::NumWorkgroups(_) => self.num_workgroups = true,
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
                if matches!(op, fusor_ir::scalar::UnOp::Unpack2x16Float) {
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
                    ReduceKind::Workgroup {
                        scratch,
                        group_size,
                    } => {
                        self.note_tile(scratch, seen);
                        self.note_tree(*group_size, value.element());
                    }
                    ReduceKind::Loop {
                        index,
                        scratch,
                        group_size,
                        ..
                    } => {
                        self.note_local(index, seen);
                        self.note_tile(scratch, seen);
                        self.note_tree(*group_size, value.element());
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

    fn quantized(&mut self, q: &fusor_ir::ir::kernel::QuantizedView, seen: &mut Seen) {
        self.note_buffer(&q.data.buffer, seen);
        // Native-layout scales are half floats read out of a u32 word with
        // `Unpack2x16Float`; that needs SHADER_FLOAT16_IN_FLOAT32, not
        // SHADER_F16.
        if q.layout == fusor_ir::dtype::QLayout::Native {
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

/// Demand-allocated private scratch kinds, interned by
/// `(kind, element, depth)`. Only keys that are actually used allocate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum ScratchKind {
    LoopIndex,
    /// A masked-value spill local (one per masked load/reduce).
    Value,
    /// A reduce-accumulator spill local; deepens with nesting.
    Spill,
}

/// Per-kernel emission state: the naga arenas plus the Kernel -> naga handle maps.
pub(crate) struct Emitter<'a> {
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
    /// binding, not a memory round trip.
    pub(crate) forced_names: Vec<(naga::Handle<naga::Expression>, String)>,
    /// Hash-consed expression memo. `TileExpr: Hash + Eq` is O(1) through its
    /// cached structural hash, so two identical subtrees built separately
    /// merge here.
    ///
    /// Each entry carries the [`Self::mem_epoch`] it was created at. A key
    /// that reads memory is a hit only while every space it reads is still at
    /// that epoch — see [`Emitter::expr`].
    pub(crate) memo: FxHashMap<TileExpr, (naga::Handle<naga::Expression>, MemStamp)>,
    /// Write counter per memory space, indexed by [`MEM_SPACES`] order:
    /// storage, workgroup tile, private local.
    ///
    /// **Monotonic, and not part of [`expr::Scope`].** A store
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
    /// Argument index of `@builtin(num_workgroups)`, when the body reads it.
    pub(crate) num_workgroups_arg: Option<u32>,
    pub(crate) depth: u32,
    pub(crate) u32_ty: naga::Handle<naga::Type>,
    pub(crate) u32_vec3_ty: naga::Handle<naga::Type>,
}

impl<'a> Emitter<'a> {
    pub(crate) fn new(
        ir: &'a KernelIr,
        caps: &'a Caps,
        plan: &'a ArenaPlan,
    ) -> Result<Self, EmitError> {
        let analysis = Analysis::run(ir, caps);

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
            num_workgroups_arg: None,
            depth: 0,
            u32_ty: prelude.u32_ty,
            u32_vec3_ty: prelude.u32_vec3_ty,
        })
    }

    /// Build the globals, the one `main` entry point, and validate.
    pub(crate) fn finish(mut self) -> Result<EmittedModule, EmitError> {
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
        if self.analysis.num_workgroups {
            self.num_workgroups_arg = Some(arguments.len() as u32);
            arguments.push(builtin_arg(self.u32_vec3_ty, naga::BuiltIn::NumWorkGroups));
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
    pub(crate) fn validate(&self) -> Result<naga::valid::ModuleInfo, EmitError> {
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
