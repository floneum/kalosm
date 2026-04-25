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
    ResourceBinding, Scalar, ShaderStage, Span, Statement, StorageAccess, Type, TypeInner,
    VectorSize,
};

use crate::analysis::TensorAnalysis;
use crate::language::TensorIr;
use crate::skeleton::{DispatchInfo, DispatchProgram, OutputElement};
use crate::types::{BinderKind, BufferRef, DType, IndexLevel, MemTier};
use crate::{Verified, verify};

pub(super) const MAX_DISPATCH_WORKGROUPS_PER_DIMENSION: u32 = 65_535;

/// Lazy interner for Naga module type declarations.
///
/// The cache key is the full [`Type`] value, so named structs and every
/// [`TypeInner`] variant can share the same path instead of needing one field
/// per type shape that codegen happens to use today.
#[derive(Default)]
struct TypeCache {
    handles: HashMap<Type, Handle<Type>>,
}

impl TypeCache {
    fn get(&mut self, module: &mut Module, ty: Type) -> Handle<Type> {
        if let Some(&handle) = self.handles.get(&ty) {
            return handle;
        }

        let handle = module.types.insert(ty.clone(), Span::UNDEFINED);
        self.handles.insert(ty, handle);
        handle
    }

    fn unnamed(&mut self, module: &mut Module, inner: TypeInner) -> Handle<Type> {
        self.get(module, Type { name: None, inner })
    }

    fn scalar(&mut self, module: &mut Module, scalar: Scalar) -> Handle<Type> {
        self.unnamed(module, TypeInner::Scalar(scalar))
    }

    fn scalar_for_dtype(&mut self, module: &mut Module, dtype: DType) -> Handle<Type> {
        self.scalar(module, Self::dtype_scalar(dtype))
    }

    fn vector(&mut self, module: &mut Module, size: VectorSize, scalar: Scalar) -> Handle<Type> {
        self.unnamed(module, TypeInner::Vector { size, scalar })
    }

    fn array(
        &mut self,
        module: &mut Module,
        base: Handle<Type>,
        size: ArraySize,
        stride: u32,
    ) -> Handle<Type> {
        self.unnamed(module, TypeInner::Array { base, size, stride })
    }

    fn runtime_array(&mut self, module: &mut Module, dtype: DType) -> Handle<Type> {
        let scalar = Self::storage_scalar(dtype);
        let base = self.scalar(module, scalar);
        self.array(module, base, ArraySize::Dynamic, dtype.byte_size())
    }

    const fn dtype_scalar(dtype: DType) -> Scalar {
        match dtype {
            DType::F16 => Scalar::F16,
            DType::F32 => Scalar::F32,
            DType::U32 => Scalar::U32,
            DType::I32 => Scalar::I32,
            DType::Bool => Scalar::BOOL,
        }
    }

    const fn storage_scalar(dtype: DType) -> Scalar {
        match dtype {
            // Naga/WGSL bools are not host-shareable, so bool tensor storage
            // uses the same 32-bit word representation as `DType::byte_size`.
            DType::Bool => Scalar::U32,
            _ => Self::dtype_scalar(dtype),
        }
    }
}

/// Per-dispatch codegen context.
///
/// Holds the Naga function being built (expressions, locals, body) plus
/// mappings from e-graph Ids and Var names to Naga expression handles.
struct CodegenCtx<'a> {
    module: &'a mut Module,
    egraph: &'a EGraph<TensorIr, TensorAnalysis>,
    types: &'a mut TypeCache,

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

    /// Scalar element dtype for each buffer visible to this dispatch.
    buffer_dtypes: HashMap<MemTier, DType>,

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
    let mut types = TypeCache::default();

    for (i, dispatch) in program.dispatches.iter().enumerate() {
        lower_single_dispatch(
            &mut module,
            &program.egraph,
            dispatch,
            i,
            &mut types,
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
// Per-dispatch lowering
// ═══════════════════════════════════════════════════════════════

fn lower_single_dispatch(
    module: &mut Module,
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    dispatch: &DispatchInfo,
    dispatch_idx: usize,
    types: &mut TypeCache,
    chosen_nodes: &HashMap<Id, TensorIr>,
    simd_width: u32,
) {
    let mut buffer_map = HashMap::new();
    let input_dtypes = dispatch
        .inputs
        .iter()
        .map(|input| dtype_for_id(egraph, *input))
        .collect::<Vec<_>>();
    let output_dtype = dispatch_output_dtype(egraph, dispatch);
    let mut buffer_dtypes = build_buffer_dtypes(dispatch, &input_dtypes, output_dtype);
    let output_binding =
        register_input_buffers(module, dispatch, types, &input_dtypes, &mut buffer_map);
    let output_gv =
        register_output_buffer(module, types, output_dtype, &mut buffer_map, output_binding);
    declare_workgroup_buffers(
        module,
        dispatch,
        types,
        &input_dtypes,
        output_dtype,
        &mut buffer_map,
        &mut buffer_dtypes,
    );

    let mut ctx = build_codegen_ctx(
        module,
        egraph,
        dispatch,
        types,
        chosen_nodes,
        buffer_map,
        buffer_dtypes,
        simd_width,
    );
    emit_dispatch_outputs(&mut ctx, &dispatch.outputs, output_gv);
    emit_dispatch_bounds_guard(&mut ctx, dispatch);
    let entry_point = build_dispatch_entry_point(ctx, dispatch_idx, dispatch, simd_width);
    module.entry_points.push(entry_point);
}

fn dtype_for_id(egraph: &EGraph<TensorIr, TensorAnalysis>, id: Id) -> DType {
    egraph[egraph.find(id)].data.dtype.unwrap_or(DType::F32)
}

fn dispatch_output_dtype(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    dispatch: &DispatchInfo,
) -> DType {
    egraph[egraph.find(dispatch.semantic_output_id)]
        .data
        .dtype
        .or_else(|| {
            dispatch
                .outputs
                .first()
                .and_then(|output| egraph[egraph.find(output.value_id)].data.dtype)
        })
        .unwrap_or(DType::F32)
}

fn dtype_for_buffer_ref(input_dtypes: &[DType], output_dtype: DType, buffer: BufferRef) -> DType {
    match buffer {
        BufferRef::Input(index) => input_dtypes
            .get(index as usize)
            .copied()
            .unwrap_or(DType::F32),
        BufferRef::Output(_) => output_dtype,
    }
}

fn build_buffer_dtypes(
    dispatch: &DispatchInfo,
    input_dtypes: &[DType],
    output_dtype: DType,
) -> HashMap<MemTier, DType> {
    let mut dtypes = HashMap::new();
    for (index, dtype) in input_dtypes.iter().enumerate() {
        dtypes.insert(MemTier::Device(BufferRef::Input(index as u32)), *dtype);
    }
    dtypes.insert(MemTier::Device(BufferRef::Output(0)), output_dtype);
    for buffer in &dispatch.tg_buffers {
        let dtype = dtype_for_buffer_ref(input_dtypes, output_dtype, buffer.device_name);
        dtypes.insert(MemTier::Threadgroup(buffer.tg_name), dtype);
    }
    dtypes
}

fn register_input_buffers(
    module: &mut Module,
    dispatch: &DispatchInfo,
    types: &mut TypeCache,
    input_dtypes: &[DType],
    buffer_map: &mut HashMap<MemTier, (Handle<GlobalVariable>, bool)>,
) -> u32 {
    let mut binding_idx = 0;
    for (index, _) in dispatch.inputs.iter().enumerate() {
        let slot = u32::try_from(index).expect("input slot fits in u32");
        let buf_ref = BufferRef::Input(slot);
        let dtype = input_dtypes.get(index).copied().unwrap_or(DType::F32);
        let ty = types.runtime_array(module, dtype);
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
                ty,
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
    types: &mut TypeCache,
    output_dtype: DType,
    buffer_map: &mut HashMap<MemTier, (Handle<GlobalVariable>, bool)>,
    binding_idx: u32,
) -> Handle<GlobalVariable> {
    let ty = types.runtime_array(module, output_dtype);
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
            ty,
            init: None,
        },
        Span::UNDEFINED,
    );
    buffer_map.insert(MemTier::Device(BufferRef::Output(0)), (output_gv, false));
    output_gv
}

fn build_codegen_ctx<'a>(
    module: &'a mut Module,
    egraph: &'a EGraph<TensorIr, TensorAnalysis>,
    dispatch: &DispatchInfo,
    types: &'a mut TypeCache,
    chosen_nodes: &'a HashMap<Id, TensorIr>,
    buffer_map: HashMap<MemTier, (Handle<GlobalVariable>, bool)>,
    buffer_dtypes: HashMap<MemTier, DType>,
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
        module,
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
        buffer_dtypes,
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

fn emit_dispatch_bounds_guard(ctx: &mut CodegenCtx<'_>, dispatch: &DispatchInfo) {
    let physical_workgroups = dispatch.workgroups / dispatch.simdgroups.max(1);
    if physical_workgroups <= MAX_DISPATCH_WORKGROUPS_PER_DIMENSION {
        return;
    }

    let workgroup = ctx.lower_index(IndexLevel::Workgroup);
    let bound = ctx.emit_literal(Literal::U32(dispatch.workgroups));
    let in_bounds = ctx.emit_binary(BinaryOperator::Less, workgroup, bound);
    let accept = std::mem::replace(&mut ctx.body, Block::new());
    ctx.body.push(
        Statement::If {
            condition: in_bounds,
            accept,
            reject: Block::new(),
        },
        Span::UNDEFINED,
    );
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
    simd_width: u32,
) -> EntryPoint {
    let vec3_u32 = ctx.types.vector(ctx.module, VectorSize::Tri, Scalar::U32);
    let function = Function {
        name: Some(format!("dispatch_{dispatch_idx}")),
        arguments: vec![
            FunctionArgument {
                name: Some("local_invocation_id".into()),
                ty: vec3_u32,
                binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationId)),
            },
            FunctionArgument {
                name: Some("workgroup_id".into()),
                ty: vec3_u32,
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
    types: &mut TypeCache,
    input_dtypes: &[DType],
    output_dtype: DType,
    buf_map: &mut HashMap<MemTier, (Handle<GlobalVariable>, bool)>,
    dtype_map: &mut HashMap<MemTier, DType>,
) {
    for buf_info in &dispatch.tg_buffers {
        let key = MemTier::Threadgroup(buf_info.tg_name);
        if buf_map.contains_key(&key) {
            continue;
        }
        let dtype = dtype_for_buffer_ref(input_dtypes, output_dtype, buf_info.device_name);
        let base = types.scalar_for_dtype(module, dtype);
        let fixed_arr = types.array(
            module,
            base,
            ArraySize::Constant(std::num::NonZeroU32::new(buf_info.size.max(1)).unwrap()),
            dtype.byte_size(),
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
        dtype_map.insert(key, dtype);
    }
}

// ═══════════════════════════════════════════════════════════════
// Expression lowering (e-graph Id → Handle<Expression>)
// ═══════════════════════════════════════════════════════════════

mod emit;
mod lower;

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use super::*;

    #[test]
    fn type_cache_is_lazy_and_dedupes_types() {
        let mut module = Module::default();
        let mut cache = TypeCache::default();

        assert_eq!(module.types.len(), 0);

        let f32_a = cache.scalar(&mut module, Scalar::F32);
        let f32_b = cache.scalar(&mut module, Scalar::F32);
        assert_eq!(f32_a, f32_b);
        assert_eq!(module.types.len(), 1);

        let array_a = cache.array(
            &mut module,
            f32_a,
            ArraySize::Constant(NonZeroU32::new(4).unwrap()),
            4,
        );
        let array_b = cache.array(
            &mut module,
            f32_a,
            ArraySize::Constant(NonZeroU32::new(4).unwrap()),
            4,
        );
        assert_eq!(array_a, array_b);
        assert_eq!(module.types.len(), 2);
    }

    #[test]
    fn type_cache_keys_include_the_full_type() {
        let mut module = Module::default();
        let mut cache = TypeCache::default();

        let u32_ty = cache.scalar(&mut module, Scalar::U32);
        let inner = TypeInner::Struct {
            members: vec![naga::StructMember {
                name: Some("value".into()),
                ty: u32_ty,
                binding: None,
                offset: 0,
            }],
            span: 4,
        };

        let named_a = cache.get(
            &mut module,
            Type {
                name: Some("A".into()),
                inner: inner.clone(),
            },
        );
        let named_a_again = cache.get(
            &mut module,
            Type {
                name: Some("A".into()),
                inner: inner.clone(),
            },
        );
        let named_b = cache.get(
            &mut module,
            Type {
                name: Some("B".into()),
                inner,
            },
        );

        assert_eq!(named_a, named_a_again);
        assert_ne!(named_a, named_b);
        assert_eq!(module.types.len(), 3);
    }
}
