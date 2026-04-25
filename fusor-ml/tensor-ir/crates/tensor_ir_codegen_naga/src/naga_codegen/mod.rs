//! Lower a `DispatchProgram` into a `naga::Module`.
//!
//! Each `DispatchInfo` becomes a `@compute` entry point. Pure expressions
//! in the e-graph are lowered recursively (and memoized) into Naga
//! expressions; effectful skeleton statements become Naga statements.

use std::collections::{HashMap, HashSet};

use egg::{EGraph, Id};
use naga::{
    AddressSpace, Arena, ArraySize, BinaryOperator, Binding, Block, BuiltIn, EntryPoint,
    Expression, Function, FunctionArgument, GlobalVariable, Handle, Literal, LocalVariable, Module,
    ResourceBinding, Scalar, ScalarKind, ShaderStage, Span, Statement, StorageAccess, Type,
    TypeInner, VectorSize,
};

use crate::analysis::TensorAnalysis;
use crate::language::TensorIr;
use crate::skeleton::{DispatchInfo, DispatchProgram, OutputElement};
use crate::types::{BinderKind, BufferRef, IndexLevel, MemTier};
use crate::{Verified, verify};

/// Cached type handles used throughout codegen.
struct TypeCache {
    f32: Handle<Type>,
    u32: Handle<Type>,
    i32: Handle<Type>,
    bool: Handle<Type>,
    vec3_u32: Handle<Type>,
    /// `array<f32>` — runtime-sized storage buffer element type.
    rt_array_f32: Handle<Type>,
}

/// Per-dispatch codegen context.
///
/// Holds the Naga function being built (expressions, locals, body) plus
/// mappings from e-graph Ids and Var names to Naga expression handles.
struct CodegenCtx<'a> {
    egraph: &'a EGraph<TensorIr, TensorAnalysis>,
    types: &'a TypeCache,

    /// Number of simdgroups per physical workgroup for this dispatch.
    simdgroups: u32,
    /// Lanes per simdgroup, sourced from `DeviceProfile::simd_width`.
    simd_width: u32,

    /// Extracted node choices: canonical Id → the specific `TensorIr` node
    /// the extractor selected. Used instead of picking an arbitrary e-node.
    chosen_nodes: &'a HashMap<Id, TensorIr>,

    // Naga function parts (built incrementally, moved into Function at the end).
    expressions: Arena<Expression>,
    local_variables: Arena<LocalVariable>,
    body: Block,

    /// Memoized: canonical e-graph Id → naga expression handle.
    id_cache: HashMap<Id, Handle<Expression>>,

    /// Canonical ids currently being lowered. Used to detect accidental
    /// recursive chosen-node cycles before they overflow the stack.
    lowering_set: HashSet<Id>,
    lowering_stack: Vec<Id>,

    /// Theta results: canonical e-graph Id → accumulator local-variable
    /// pointers, one per slot (scalar accumulators are one-slot tuples).
    /// Unlike `id_cache` (which gets cleared when entering/leaving loops),
    /// this persists across scope boundaries. When a Theta's output is
    /// referenced again at an outer scope, we emit a fresh load from its
    /// accumulator instead of re-lowering the entire loop.
    theta_acc_ptrs: HashMap<Id, Vec<Handle<Expression>>>,

    /// Stack of in-scope binder frames. `binder_stack.last()` is the innermost
    /// binder currently being lowered. `VarRef::Bound { kind, slot, depth }`
    /// walks the stack from the top, keeping only frames whose kind matches,
    /// and takes the `depth`-th of those. Per-kind depth numbering means a
    /// Theta ref never resolves through a Dispatch frame and vice versa.
    binder_stack: Vec<BinderFrame>,

    /// Buffer handles: typed `MemTier` → (`global_var` handle, `is_workgroup`).
    buffer_map: HashMap<MemTier, (Handle<GlobalVariable>, bool)>,

    /// Per-simdgroup read stride for scaled tg buffers, keyed on the
    /// threadgroup-tier `BufferRef`.
    tg_sg_read_strides: HashMap<BufferRef, u32>,

    /// Built-in expressions (created once at entry-point setup).
    local_invocation_id: Handle<Expression>,
    workgroup_id: Handle<Expression>,

    /// Track which expression handles have been emitted to avoid double-emit.
    emitted: HashSet<Handle<Expression>>,

    /// Named expressions for cleaner WGSL output.
    named_expressions: naga::FastIndexMap<Handle<Expression>, String>,
}

/// In-scope bindings for one `Theta` being lowered. The accumulator is
/// uniformly represented as a tuple — a scalar accumulator is just a
/// one-slot tuple, so `handles.len() == types.len()` always, and a bare
/// `Var(Acc)` resolves to slot 0 when arity is 1.
///
/// Pushed on entry to the Theta's update body and popped on exit.
#[derive(Debug, Clone)]
pub(super) struct ThetaFrame {
    pub(super) iter: Handle<Expression>,
    pub(super) acc_handles: Vec<Handle<Expression>>,
    pub(super) acc_types: Vec<Handle<Type>>,
}

/// In-scope bindings for the enclosing `Dispatch`.
///
/// Pushed once at dispatch setup and popped at the end. Dispatch cannot
/// nest, so depth is always 0 for Dispatch-kind refs.
#[derive(Debug, Clone, Copy)]
pub(super) struct DispatchFrame {
    pub(super) lane: Handle<Expression>,
    pub(super) simdgroup: Handle<Expression>,
    pub(super) workgroup: Handle<Expression>,
}

/// Tagged binder frame — one variant per [`BinderKind`].
///
/// Per-kind depth resolution: `lookup_var` walks `binder_stack` top-down,
/// keeps only frames whose `kind()` matches the ref's kind, and takes the
/// `depth`-th of those.
#[derive(Debug, Clone)]
pub(super) enum BinderFrame {
    Theta(ThetaFrame),
    Dispatch(DispatchFrame),
}

impl BinderFrame {
    pub(super) fn kind(&self) -> BinderKind {
        match self {
            Self::Theta(_) => BinderKind::Theta,
            Self::Dispatch(_) => BinderKind::Dispatch,
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════

/// Lower a verified `DispatchProgram` into a `naga::Module`.
///
/// Takes a [`Verified`] handle so codegen can rely on every `Var` reference
/// being bound (no runtime "unbound variable" panic) and the chosen
/// extraction being acyclic. Use [`verify`] to obtain a `Verified` from a
/// `&DispatchProgram`.
///
/// Each dispatch becomes a compute entry point named `dispatch_0`, `dispatch_1`, etc.
/// Storage buffers are bound at `group=0, binding=0..n` (inputs then output).
#[must_use]
pub fn lower_dispatch_program(verified: Verified<'_>) -> Module {
    let program = verified.program();
    let mut module = Module::default();

    let types = register_types(&mut module);

    for (i, dispatch) in program.dispatches.iter().enumerate() {
        lower_single_dispatch(
            &mut module,
            &program.egraph,
            dispatch,
            i,
            &types,
            &program.chosen_nodes,
            program.device.simd_width,
        );
    }

    module
}

/// Convenience: verify, lower, and immediately emit WGSL text.
///
/// Runs [`verify`], then the Naga validator, then the WGSL backend writer.
/// Returns `Err` if verification, validation, or writing fails.
///
/// # Errors
///
/// Returns an error if any stage fails.
pub fn lower_to_wgsl(program: &DispatchProgram) -> Result<String, String> {
    let verified = verify(program).map_err(|e| format!("verification error: {e}"))?;
    let module = lower_dispatch_program(verified);
    module_to_wgsl(&module)
}

/// Convert a `naga::Module` to WGSL text.
///
/// # Errors
///
/// Returns an error if validation or WGSL emission fails.
pub fn module_to_wgsl(module: &Module) -> Result<String, String> {
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    let mut validator = Validator::new(ValidationFlags::empty(), Capabilities::all());
    let info = validator
        .validate(module)
        .map_err(|e| format!("Naga validation error: {e:?}"))?;

    let mut writer =
        naga::back::wgsl::Writer::new(String::new(), naga::back::wgsl::WriterFlags::empty());
    writer
        .write(module, &info)
        .map_err(|e| format!("Naga WGSL write error: {e}"))?;
    Ok(writer.finish())
}

/// Convenience: verify, lower, and immediately emit MSL text.
///
/// # Errors
///
/// Returns an error if verification, validation, or MSL emission fails.
pub fn lower_to_msl(program: &DispatchProgram) -> Result<String, String> {
    let verified = verify(program).map_err(|e| format!("verification error: {e}"))?;
    let module = lower_dispatch_program(verified);
    module_to_msl(&module)
}

/// Convert a `naga::Module` to Metal Shading Language text.
///
/// # Errors
///
/// Returns an error if validation or MSL emission fails.
pub fn module_to_msl(module: &Module) -> Result<String, String> {
    use naga::back::msl;
    use naga::valid::{Capabilities, ValidationFlags, Validator};

    let mut validator = Validator::new(ValidationFlags::empty(), Capabilities::all());
    let info = validator
        .validate(module)
        .map_err(|e| format!("Naga validation error: {e:?}"))?;

    let options = msl::Options {
        zero_initialize_workgroup_memory: false,
        force_loop_bounding: false,
        ..msl::Options::default()
    };
    let pipeline_options = msl::PipelineOptions {
        allow_and_force_point_size: false,
        entry_point: None,
        vertex_pulling_transform: false,
        vertex_buffer_mappings: Vec::new(),
    };
    let (source, _) = msl::write_string(module, &info, &options, &pipeline_options)
        .map_err(|e| format!("Naga MSL write error: {e}"))?;
    Ok(source)
}

// ═══════════════════════════════════════════════════════════════
// Type registration
// ═══════════════════════════════════════════════════════════════

fn register_types(module: &mut Module) -> TypeCache {
    let f32 = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar {
                kind: ScalarKind::Float,
                width: 4,
            }),
        },
        Span::UNDEFINED,
    );
    let u32_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar {
                kind: ScalarKind::Uint,
                width: 4,
            }),
        },
        Span::UNDEFINED,
    );
    let i32_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar {
                kind: ScalarKind::Sint,
                width: 4,
            }),
        },
        Span::UNDEFINED,
    );
    let bool_ty = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Scalar(Scalar {
                kind: ScalarKind::Bool,
                width: 1,
            }),
        },
        Span::UNDEFINED,
    );
    let vec3_u32 = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Vector {
                size: VectorSize::Tri,
                scalar: Scalar {
                    kind: ScalarKind::Uint,
                    width: 4,
                },
            },
        },
        Span::UNDEFINED,
    );
    let rt_array_f32 = module.types.insert(
        Type {
            name: None,
            inner: TypeInner::Array {
                base: f32,
                size: ArraySize::Dynamic,
                stride: 4,
            },
        },
        Span::UNDEFINED,
    );

    TypeCache {
        f32,
        u32: u32_ty,
        i32: i32_ty,
        bool: bool_ty,
        vec3_u32,
        rt_array_f32,
    }
}

// ═══════════════════════════════════════════════════════════════
// Per-dispatch lowering
// ═══════════════════════════════════════════════════════════════

fn lower_single_dispatch(
    module: &mut Module,
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    dispatch: &DispatchInfo,
    dispatch_idx: usize,
    types: &TypeCache,
    chosen_nodes: &HashMap<Id, TensorIr>,
    simd_width: u32,
) {
    let mut buffer_map = HashMap::new();
    let output_binding = register_input_buffers(module, dispatch, types, &mut buffer_map);
    let output_gv = register_output_buffer(module, types, &mut buffer_map, output_binding);
    declare_workgroup_buffers(module, dispatch, types, &mut buffer_map);

    let mut ctx = build_codegen_ctx(
        egraph,
        dispatch,
        types,
        chosen_nodes,
        buffer_map,
        simd_width,
    );
    emit_dispatch_outputs(&mut ctx, &dispatch.outputs, output_gv);
    module.entry_points.push(build_dispatch_entry_point(
        ctx,
        dispatch_idx,
        dispatch,
        types,
        simd_width,
    ));
}

fn register_input_buffers(
    module: &mut Module,
    dispatch: &DispatchInfo,
    types: &TypeCache,
    buffer_map: &mut HashMap<MemTier, (Handle<GlobalVariable>, bool)>,
) -> u32 {
    let mut binding_idx = 0;
    for (index, _) in dispatch.inputs.iter().enumerate() {
        let slot = u32::try_from(index).expect("input slot fits in u32");
        let buf_ref = BufferRef::Input(slot);
        let gv = module.global_variables.append(
            GlobalVariable {
                name: Some(format!("input_{index}")),
                space: AddressSpace::Storage {
                    access: StorageAccess::LOAD,
                },
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: binding_idx,
                }),
                ty: types.rt_array_f32,
                init: None,
            },
            Span::UNDEFINED,
        );
        buffer_map.insert(MemTier::Device(buf_ref), (gv, false));
        binding_idx += 1;
    }
    binding_idx
}

fn register_output_buffer(
    module: &mut Module,
    types: &TypeCache,
    buffer_map: &mut HashMap<MemTier, (Handle<GlobalVariable>, bool)>,
    binding_idx: u32,
) -> Handle<GlobalVariable> {
    let output_gv = module.global_variables.append(
        GlobalVariable {
            name: Some("output".into()),
            space: AddressSpace::Storage {
                access: StorageAccess::LOAD | StorageAccess::STORE,
            },
            binding: Some(ResourceBinding {
                group: 0,
                binding: binding_idx,
            }),
            ty: types.rt_array_f32,
            init: None,
        },
        Span::UNDEFINED,
    );
    buffer_map.insert(MemTier::Device(BufferRef::Output(0)), (output_gv, false));
    output_gv
}

fn build_codegen_ctx<'a>(
    egraph: &'a EGraph<TensorIr, TensorAnalysis>,
    dispatch: &DispatchInfo,
    types: &'a TypeCache,
    chosen_nodes: &'a HashMap<Id, TensorIr>,
    buffer_map: HashMap<MemTier, (Handle<GlobalVariable>, bool)>,
    simd_width: u32,
) -> CodegenCtx<'a> {
    let mut expressions = Arena::new();
    let local_invocation_id = expressions.append(Expression::FunctionArgument(0), Span::UNDEFINED);
    let workgroup_id = expressions.append(Expression::FunctionArgument(1), Span::UNDEFINED);
    let tg_sg_read_strides = dispatch
        .tg_buffers
        .iter()
        .filter(|buffer| buffer.sg_read_stride > 0)
        .map(|buffer| (buffer.tg_name, buffer.sg_read_stride))
        .collect();

    CodegenCtx {
        egraph,
        types,
        simdgroups: dispatch.simdgroups,
        simd_width,
        chosen_nodes,
        expressions,
        local_variables: Arena::new(),
        body: Block::new(),
        id_cache: HashMap::new(),
        lowering_set: HashSet::new(),
        lowering_stack: Vec::new(),
        theta_acc_ptrs: HashMap::new(),
        binder_stack: Vec::new(),
        buffer_map,
        tg_sg_read_strides,
        local_invocation_id,
        workgroup_id,
        emitted: HashSet::new(),
        named_expressions: naga::FastIndexMap::default(),
    }
}

fn emit_dispatch_outputs(
    ctx: &mut CodegenCtx<'_>,
    outputs: &[OutputElement],
    output_gv: Handle<GlobalVariable>,
) {
    // Push the Dispatch binder frame once at the start of codegen. Holds the
    // three thread-index expressions so `lookup_var` can resolve
    // `VarRef::Bound { kind: Dispatch, .. }` without special-casing in the
    // caller.
    let lane = ctx.lower_index(IndexLevel::Lane);
    let simdgroup = ctx.lower_index(IndexLevel::Simdgroup);
    let workgroup = ctx.lower_index(IndexLevel::Workgroup);
    ctx.binder_stack.push(BinderFrame::Dispatch(DispatchFrame {
        lane,
        simdgroup,
        workgroup,
    }));

    let out_gv_expr = ctx
        .expressions
        .append(Expression::GlobalVariable(output_gv), Span::UNDEFINED);
    for output in outputs {
        emit_output_store(ctx, out_gv_expr, output);
    }
}

fn emit_output_store(
    ctx: &mut CodegenCtx<'_>,
    out_gv_expr: Handle<Expression>,
    output: &OutputElement,
) {
    let output_idx = ctx.lower_expr(output.addr_id);
    let output_val = ctx.lower_expr(output.value_id);
    let out_ptr = ctx.emit_access(out_gv_expr, output_idx);

    if ctx.egraph[output.addr_id].data.dep.contains_lane() {
        ctx.body.push(
            Statement::Store {
                pointer: out_ptr,
                value: output_val,
            },
            Span::UNDEFINED,
        );
        return;
    }

    emit_lane_zero_output_store(ctx, out_ptr, output_val);
}

fn emit_lane_zero_output_store(
    ctx: &mut CodegenCtx<'_>,
    out_ptr: Handle<Expression>,
    output_val: Handle<Expression>,
) {
    let lane_id = ctx.lower_index(IndexLevel::Lane);
    let zero = ctx.emit_literal(Literal::U32(0));
    let is_lane_0 = ctx.emit_binary(BinaryOperator::Equal, lane_id, zero);

    let outer_body = std::mem::replace(&mut ctx.body, Block::new());
    ctx.body.push(
        Statement::Store {
            pointer: out_ptr,
            value: output_val,
        },
        Span::UNDEFINED,
    );
    let if_body = std::mem::replace(&mut ctx.body, outer_body);

    ctx.body.push(
        Statement::If {
            condition: is_lane_0,
            accept: if_body,
            reject: Block::new(),
        },
        Span::UNDEFINED,
    );
}

fn build_dispatch_entry_point(
    ctx: CodegenCtx<'_>,
    dispatch_idx: usize,
    dispatch: &DispatchInfo,
    types: &TypeCache,
    simd_width: u32,
) -> EntryPoint {
    let function = Function {
        name: Some(format!("dispatch_{dispatch_idx}")),
        arguments: vec![
            FunctionArgument {
                name: Some("local_invocation_id".into()),
                ty: types.vec3_u32,
                binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationId)),
            },
            FunctionArgument {
                name: Some("workgroup_id".into()),
                ty: types.vec3_u32,
                binding: Some(Binding::BuiltIn(BuiltIn::WorkGroupId)),
            },
        ],
        result: None,
        local_variables: ctx.local_variables,
        expressions: ctx.expressions,
        named_expressions: ctx.named_expressions,
        body: ctx.body,
        diagnostic_filter_leaf: None,
    };

    EntryPoint {
        name: format!("dispatch_{dispatch_idx}"),
        stage: ShaderStage::Compute,
        early_depth_test: None,
        workgroup_size: [dispatch.simdgroups * simd_width, 1, 1],
        workgroup_size_overrides: None,
        function,
        mesh_info: None,
        task_payload: None,
        incoming_ray_payload: None,
    }
}

fn binary_args(args: &[Id]) -> Option<[Id; 2]> {
    match args {
        [lhs, rhs] => Some([*lhs, *rhs]),
        _ => None,
    }
}

/// Declare workgroup-scoped global variables for threadgroup buffers.
///
/// Uses the pre-computed `tg_buffers` from the skeleton builder, which
/// already has correct sizes from tile dimension analysis.
fn declare_workgroup_buffers(
    module: &mut Module,
    dispatch: &DispatchInfo,
    types: &TypeCache,
    buf_map: &mut HashMap<MemTier, (Handle<GlobalVariable>, bool)>,
) {
    for buf_info in &dispatch.tg_buffers {
        let key = MemTier::Threadgroup(buf_info.tg_name);
        if buf_map.contains_key(&key) {
            continue;
        }
        let fixed_arr = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Array {
                    base: types.f32,
                    size: ArraySize::Constant(
                        std::num::NonZeroU32::new(buf_info.size.max(1)).unwrap(),
                    ),
                    stride: 4,
                },
            },
            Span::UNDEFINED,
        );
        let gv = module.global_variables.append(
            GlobalVariable {
                name: Some(format!("{}_tg", buf_info.tg_name)),
                space: AddressSpace::WorkGroup,
                binding: None,
                ty: fixed_arr,
                init: None,
            },
            Span::UNDEFINED,
        );
        buf_map.insert(key, (gv, true));
    }
}

// ═══════════════════════════════════════════════════════════════
// Expression lowering (e-graph Id → Handle<Expression>)
// ═══════════════════════════════════════════════════════════════

mod emit;
mod lower;
