//! Lower tensor IR programs into a `naga::Module`.
//!
//! Effectful `Program` terms become one or more `@compute` entry points. Pure
//! value expressions are lowered recursively into Naga expressions; effect
//! nodes provide dispatch order, stores, barriers, and buffer identity.

use std::collections::{HashMap, HashSet};

use egg::{EGraph, Id, Language, RecExpr};
use naga::{
    AddressSpace, Arena, ArraySize, BinaryOperator, Binding, Block, BuiltIn, EntryPoint,
    Expression, Function, FunctionArgument, GlobalVariable, Handle, Literal, LocalVariable, Module,
    ResourceBinding, Scalar, ShaderStage, Span, Statement, StorageAccess, Type, TypeInner,
    VectorSize,
};

use crate::analysis::TensorAnalysis;
use crate::language::{EffectNode, SimdNode, TensorIr, extract_list};
use crate::types::{
    BinaryOp, BinderKind, BufferRef, DType, DeviceProfile, Dim, IndexLevel, MemTier, ScalarValue,
    VarRef, slots,
};

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
    shape_params_gv: Handle<GlobalVariable>,

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

/// Lower an extracted effectful program rooted at `EffectNode::Program`.
///
/// This is the e-graph-native backend path: dispatch order, stores, barriers,
/// and buffer identities are read from the extracted effect term directly.
///
/// # Errors
///
/// Returns an error when the root is not an effect program or the effect tree
/// contains malformed nodes.
pub fn lower_effect_program(
    expr: &RecExpr<TensorIr>,
    device: &DeviceProfile,
) -> Result<Module, String> {
    let (egraph, chosen_nodes, root) = materialize_extracted_program(expr);
    let TensorIr::Effect(EffectNode::Program { children }) =
        chosen_node(&egraph, &chosen_nodes, root)
    else {
        return Err("extracted root must be EffectNode::Program".into());
    };

    let mut dispatches = Vec::new();
    collect_effect_dispatches(&egraph, &chosen_nodes, children[1], &mut dispatches)?;
    if dispatches.is_empty() {
        return Err("effect program contains no dispatches".into());
    }

    let mut module = Module::default();
    let mut types = TypeCache::default();
    let device_buffer_dtypes = collect_program_device_buffer_dtypes(&egraph, &chosen_nodes, root);
    let device_buffers = collect_program_device_buffers(&egraph, &chosen_nodes, root);
    let tg_buffers = collect_program_threadgroup_buffer_layouts(
        &egraph,
        &chosen_nodes,
        root,
        &dispatches,
        &device_buffer_dtypes,
        device.simd_width,
    )?;

    for (i, dispatch) in dispatches.iter().enumerate() {
        lower_effect_dispatch(
            &mut module,
            &egraph,
            &chosen_nodes,
            dispatch,
            i,
            &mut types,
            &device_buffers,
            &device_buffer_dtypes,
            &tg_buffers,
            device.simd_width,
        )?;
    }

    Ok(module)
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

#[derive(Debug, Clone)]
struct EffectDispatchPlan {
    workgroups: Dim,
    simdgroups: u32,
    state: Id,
    body: Id,
}

fn materialize_extracted_program(
    expr: &RecExpr<TensorIr>,
) -> (EGraph<TensorIr, TensorAnalysis>, HashMap<Id, TensorIr>, Id) {
    let mut egraph = EGraph::<TensorIr, TensorAnalysis>::default();
    let mut id_map = Vec::with_capacity(expr.as_ref().len());
    let mut chosen_nodes = HashMap::with_capacity(expr.as_ref().len());

    for node in expr.as_ref() {
        let remapped = node.clone().map_children(|id| id_map[usize::from(id)]);
        let id = egraph.add(remapped.clone());
        id_map.push(id);
        chosen_nodes.insert(egraph.find(id), remapped);
    }
    egraph.rebuild();

    let root = egraph.find(
        *id_map
            .last()
            .expect("extracted program should contain at least one node"),
    );
    (egraph, chosen_nodes, root)
}

fn chosen_node(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen_nodes: &HashMap<Id, TensorIr>,
    id: Id,
) -> TensorIr {
    let canonical = egraph.find(id);
    chosen_nodes
        .get(&canonical)
        .cloned()
        .unwrap_or_else(|| egraph[canonical].nodes[0].clone())
}

fn collect_effect_dispatches(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen_nodes: &HashMap<Id, TensorIr>,
    id: Id,
    out: &mut Vec<EffectDispatchPlan>,
) -> Result<(), String> {
    match chosen_node(egraph, chosen_nodes, id) {
        TensorIr::Effect(EffectNode::Seq(list_id)) => {
            for step in extract_list(egraph, list_id) {
                collect_effect_dispatches(egraph, chosen_nodes, step, out)?;
            }
        }
        TensorIr::Effect(EffectNode::Dispatch {
            workgroups,
            simdgroups,
            children: [state, body],
        }) => out.push(EffectDispatchPlan {
            workgroups,
            simdgroups,
            state,
            body,
        }),
        TensorIr::Effect(EffectNode::Token) => {}
        TensorIr::Effect(EffectNode::Program { children }) => {
            collect_effect_dispatches(egraph, chosen_nodes, children[1], out)?;
        }
        other => {
            return Err(format!(
                "expected effect Seq/Dispatch while collecting program order, found {other:?}"
            ));
        }
    }
    Ok(())
}

fn collect_program_device_buffers(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen_nodes: &HashMap<Id, TensorIr>,
    root: Id,
) -> Vec<BufferRef> {
    let mut buffers = HashSet::new();
    collect_buffers(egraph, chosen_nodes, root, &mut buffers, false);
    let mut buffers = buffers.into_iter().collect::<Vec<_>>();
    buffers.sort();
    buffers
}

#[derive(Debug, Clone, Copy)]
struct WorkgroupBufferLayout {
    elements: u32,
    dtype: DType,
}

fn collect_program_threadgroup_buffer_layouts(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen_nodes: &HashMap<Id, TensorIr>,
    root: Id,
    dispatches: &[EffectDispatchPlan],
    device_buffer_dtypes: &HashMap<BufferRef, DType>,
    simd_width: u32,
) -> Result<HashMap<BufferRef, WorkgroupBufferLayout>, String> {
    let mut layouts = HashMap::new();
    for dispatch in dispatches {
        refine_threadgroup_layouts_from_effect(
            egraph,
            chosen_nodes,
            dispatch.body,
            dispatch.simdgroups,
            simd_width,
            device_buffer_dtypes,
            &mut layouts,
        )?;
    }
    let mut buffers = HashSet::new();
    collect_buffers(egraph, chosen_nodes, root, &mut buffers, true);
    for buffer in buffers {
        layouts.entry(buffer).or_insert(WorkgroupBufferLayout {
            elements: 4096,
            dtype: DType::F32,
        });
    }
    Ok(layouts)
}

fn refine_threadgroup_layouts_from_effect(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen_nodes: &HashMap<Id, TensorIr>,
    id: Id,
    simdgroups: u32,
    simd_width: u32,
    device_buffer_dtypes: &HashMap<BufferRef, DType>,
    layouts: &mut HashMap<BufferRef, WorkgroupBufferLayout>,
) -> Result<(), String> {
    match chosen_node(egraph, chosen_nodes, id) {
        TensorIr::Effect(EffectNode::Token) => {}
        TensorIr::Effect(EffectNode::Seq(list_id)) => {
            for step in extract_list(egraph, list_id) {
                refine_threadgroup_layouts_from_effect(
                    egraph,
                    chosen_nodes,
                    step,
                    simdgroups,
                    simd_width,
                    device_buffer_dtypes,
                    layouts,
                )?;
            }
        }
        TensorIr::Effect(EffectNode::Store { tier, children }) => {
            refine_threadgroup_layouts_from_effect(
                egraph,
                chosen_nodes,
                children[2],
                simdgroups,
                simd_width,
                device_buffer_dtypes,
                layouts,
            )?;
            if let MemTier::Threadgroup(buffer) = tier {
                record_threadgroup_store_layout(
                    egraph,
                    chosen_nodes,
                    buffer,
                    children[0],
                    children[1],
                    None,
                    simdgroups,
                    simd_width,
                    device_buffer_dtypes,
                    layouts,
                )?;
            } else {
                let mut seen = HashSet::new();
                refine_threadgroup_layouts_from_value(
                    egraph,
                    chosen_nodes,
                    children[1],
                    simdgroups,
                    simd_width,
                    device_buffer_dtypes,
                    layouts,
                    &mut seen,
                )?;
            }
        }
        TensorIr::Effect(EffectNode::StoreIf { tier, children }) => {
            refine_threadgroup_layouts_from_effect(
                egraph,
                chosen_nodes,
                children[3],
                simdgroups,
                simd_width,
                device_buffer_dtypes,
                layouts,
            )?;
            if let MemTier::Threadgroup(buffer) = tier {
                record_threadgroup_store_layout(
                    egraph,
                    chosen_nodes,
                    buffer,
                    children[1],
                    children[2],
                    Some(children[0]),
                    simdgroups,
                    simd_width,
                    device_buffer_dtypes,
                    layouts,
                )?;
            } else {
                let mut seen = HashSet::new();
                refine_threadgroup_layouts_from_value(
                    egraph,
                    chosen_nodes,
                    children[2],
                    simdgroups,
                    simd_width,
                    device_buffer_dtypes,
                    layouts,
                    &mut seen,
                )?;
            }
        }
        TensorIr::Effect(EffectNode::Barrier { state, .. }) => {
            refine_threadgroup_layouts_from_effect(
                egraph,
                chosen_nodes,
                state,
                simdgroups,
                simd_width,
                device_buffer_dtypes,
                layouts,
            )?;
        }
        other => {
            return Err(format!(
                "expected effect state while sizing threadgroup buffers, found {other:?}"
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn refine_threadgroup_layouts_from_value(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen_nodes: &HashMap<Id, TensorIr>,
    id: Id,
    simdgroups: u32,
    simd_width: u32,
    device_buffer_dtypes: &HashMap<BufferRef, DType>,
    layouts: &mut HashMap<BufferRef, WorkgroupBufferLayout>,
    seen: &mut HashSet<Id>,
) -> Result<(), String> {
    let canonical = egraph.find(id);
    if !seen.insert(canonical) {
        return Ok(());
    }
    let node = chosen_node(egraph, chosen_nodes, canonical);
    match &node {
        TensorIr::Simd(SimdNode::Load {
            tier: MemTier::Threadgroup(buffer),
            children,
        }) => {
            record_threadgroup_load_layout(
                egraph,
                chosen_nodes,
                *buffer,
                canonical,
                children[0],
                simdgroups,
                simd_width,
                layouts,
            );
            refine_threadgroup_layouts_from_value(
                egraph,
                chosen_nodes,
                children[1],
                simdgroups,
                simd_width,
                device_buffer_dtypes,
                layouts,
                seen,
            )?;
        }
        TensorIr::Simd(SimdNode::Store { tier, children }) => {
            refine_threadgroup_layouts_from_value(
                egraph,
                chosen_nodes,
                children[2],
                simdgroups,
                simd_width,
                device_buffer_dtypes,
                layouts,
                seen,
            )?;
            if let MemTier::Threadgroup(buffer) = tier {
                record_threadgroup_store_layout(
                    egraph,
                    chosen_nodes,
                    *buffer,
                    children[0],
                    children[1],
                    None,
                    simdgroups,
                    simd_width,
                    device_buffer_dtypes,
                    layouts,
                )?;
            }
        }
        TensorIr::Simd(SimdNode::StoreIf { tier, children }) => {
            refine_threadgroup_layouts_from_value(
                egraph,
                chosen_nodes,
                children[3],
                simdgroups,
                simd_width,
                device_buffer_dtypes,
                layouts,
                seen,
            )?;
            if let MemTier::Threadgroup(buffer) = tier {
                record_threadgroup_store_layout(
                    egraph,
                    chosen_nodes,
                    *buffer,
                    children[1],
                    children[2],
                    Some(children[0]),
                    simdgroups,
                    simd_width,
                    device_buffer_dtypes,
                    layouts,
                )?;
            }
        }
        TensorIr::Simd(SimdNode::Barrier { state, .. }) => {
            refine_threadgroup_layouts_from_value(
                egraph,
                chosen_nodes,
                *state,
                simdgroups,
                simd_width,
                device_buffer_dtypes,
                layouts,
                seen,
            )?;
        }
        _ => {
            for child in node.children() {
                refine_threadgroup_layouts_from_value(
                    egraph,
                    chosen_nodes,
                    *child,
                    simdgroups,
                    simd_width,
                    device_buffer_dtypes,
                    layouts,
                    seen,
                )?;
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn record_threadgroup_load_layout(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen_nodes: &HashMap<Id, TensorIr>,
    buffer: BufferRef,
    load: Id,
    addr: Id,
    simdgroups: u32,
    simd_width: u32,
    layouts: &mut HashMap<BufferRef, WorkgroupBufferLayout>,
) {
    let elements = {
        let mut memo = HashMap::new();
        expr_upper_bound(
            egraph,
            chosen_nodes,
            addr,
            simdgroups,
            simd_width,
            &mut memo,
        )
        .map(|max_addr| max_addr.saturating_add(1))
        .unwrap_or(4096)
        .max(1)
    };
    let dtype = egraph[egraph.find(load)].data.dtype.unwrap_or(DType::F32);
    layouts
        .entry(buffer)
        .and_modify(|layout| {
            layout.elements = layout.elements.max(elements);
            if layout.dtype == DType::F32 {
                layout.dtype = dtype;
            }
        })
        .or_insert(WorkgroupBufferLayout { elements, dtype });
}

#[allow(clippy::too_many_arguments)]
fn record_threadgroup_store_layout(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen_nodes: &HashMap<Id, TensorIr>,
    buffer: BufferRef,
    addr: Id,
    value: Id,
    cond: Option<Id>,
    simdgroups: u32,
    simd_width: u32,
    device_buffer_dtypes: &HashMap<BufferRef, DType>,
    layouts: &mut HashMap<BufferRef, WorkgroupBufferLayout>,
) -> Result<(), String> {
    let elements = cond
        .and_then(|cond| guarded_store_extent(egraph, chosen_nodes, addr, cond))
        .or_else(|| {
            let mut memo = HashMap::new();
            expr_upper_bound(
                egraph,
                chosen_nodes,
                addr,
                simdgroups,
                simd_width,
                &mut memo,
            )
            .map(|max_addr| max_addr.saturating_add(1))
        })
        .unwrap_or(4096)
        .max(1);
    let mut memo = HashMap::new();
    let dtype =
        infer_effect_value_dtype(egraph, chosen_nodes, value, device_buffer_dtypes, &mut memo)
            .or(egraph[egraph.find(value)].data.dtype)
            .unwrap_or(DType::F32);

    match layouts.get_mut(&buffer) {
        Some(existing) => {
            if existing.dtype != dtype {
                if existing.dtype == DType::F32 {
                    existing.dtype = dtype;
                } else {
                    return Err(format!(
                        "threadgroup buffer {buffer} is stored with inconsistent dtypes {} and {dtype}",
                        existing.dtype
                    ));
                }
            }
            existing.elements = existing.elements.max(elements);
        }
        None => {
            layouts.insert(buffer, WorkgroupBufferLayout { elements, dtype });
        }
    }
    Ok(())
}

fn guarded_store_extent(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen_nodes: &HashMap<Id, TensorIr>,
    addr: Id,
    cond: Id,
) -> Option<u32> {
    match chosen_node(egraph, chosen_nodes, cond) {
        TensorIr::BinOp(BinaryOp::Lt, [lhs, rhs]) if egraph.find(lhs) == egraph.find(addr) => {
            const_u32(egraph, chosen_nodes, rhs)
        }
        _ => None,
    }
}

fn const_u32(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen_nodes: &HashMap<Id, TensorIr>,
    id: Id,
) -> Option<u32> {
    match chosen_node(egraph, chosen_nodes, id) {
        TensorIr::Const(ScalarValue::U32(value)) => Some(value),
        _ => match &egraph[egraph.find(id)].data.constant {
            Some(ScalarValue::U32(value)) => Some(*value),
            _ => None,
        },
    }
}

fn expr_upper_bound(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen_nodes: &HashMap<Id, TensorIr>,
    id: Id,
    simdgroups: u32,
    simd_width: u32,
    memo: &mut HashMap<Id, Option<u32>>,
) -> Option<u32> {
    let canonical = egraph.find(id);
    if let Some(bound) = memo.get(&canonical) {
        return *bound;
    }
    let bound = match chosen_node(egraph, chosen_nodes, canonical) {
        TensorIr::Const(ScalarValue::U32(value)) => Some(value),
        TensorIr::Simd(SimdNode::Var(VarRef::Bound {
            kind: BinderKind::Dispatch,
            slot: slots::DISPATCH_LANE,
            depth: 0,
        })) => Some(simd_width.saturating_sub(1)),
        TensorIr::Simd(SimdNode::Var(VarRef::Bound {
            kind: BinderKind::Dispatch,
            slot: slots::DISPATCH_SIMDGROUP,
            depth: 0,
        })) => Some(simdgroups.saturating_sub(1)),
        TensorIr::Simd(SimdNode::Var(VarRef::Bound {
            kind: BinderKind::Dispatch,
            slot: slots::DISPATCH_WORKGROUP,
            depth: 0,
        })) => None,
        TensorIr::BinOp(op, [lhs, rhs]) => {
            let lhs_bound =
                expr_upper_bound(egraph, chosen_nodes, lhs, simdgroups, simd_width, memo);
            let rhs_bound =
                expr_upper_bound(egraph, chosen_nodes, rhs, simdgroups, simd_width, memo);
            match op {
                BinaryOp::Add => lhs_bound.zip(rhs_bound).map(|(a, b)| a.saturating_add(b)),
                BinaryOp::Sub => lhs_bound,
                BinaryOp::Mul => lhs_bound.zip(rhs_bound).map(|(a, b)| a.saturating_mul(b)),
                BinaryOp::Div => lhs_bound
                    .zip(const_u32(egraph, chosen_nodes, rhs))
                    .and_then(|(lhs, rhs)| (rhs != 0).then_some(lhs / rhs)),
                BinaryOp::Mod => const_u32(egraph, chosen_nodes, rhs)
                    .map(|rhs| rhs.saturating_sub(1))
                    .zip(lhs_bound)
                    .map(|(rhs, lhs)| rhs.min(lhs)),
                BinaryOp::Min => lhs_bound.zip(rhs_bound).map(|(a, b)| a.min(b)),
                BinaryOp::Max => lhs_bound.zip(rhs_bound).map(|(a, b)| a.max(b)),
                _ => None,
            }
        }
        _ => egraph[canonical]
            .data
            .address_profile
            .and_then(|profile| profile.max_value),
    };
    memo.insert(canonical, bound);
    bound
}

fn collect_program_device_buffer_dtypes(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen_nodes: &HashMap<Id, TensorIr>,
    root: Id,
) -> HashMap<BufferRef, DType> {
    let mut dtypes = HashMap::new();
    for node in chosen_nodes.values() {
        if let TensorIr::HighLevel(crate::language::HighLevelNode::Input { id, dtype, .. }) = node {
            dtypes.entry(BufferRef::Input(*id)).or_insert(*dtype);
            dtypes.entry(BufferRef::External(*id)).or_insert(*dtype);
        }
    }
    let mut memo = HashMap::new();
    collect_buffer_dtypes(egraph, chosen_nodes, root, &mut dtypes, &mut memo);
    dtypes
}

fn collect_buffer_dtypes(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen_nodes: &HashMap<Id, TensorIr>,
    id: Id,
    dtypes: &mut HashMap<BufferRef, DType>,
    memo: &mut HashMap<Id, Option<DType>>,
) {
    let node = chosen_node(egraph, chosen_nodes, id);
    match &node {
        TensorIr::HighLevel(crate::language::HighLevelNode::Input { id, dtype, .. }) => {
            let external = BufferRef::External(*id);
            dtypes.entry(external).or_insert(*dtype);
            dtypes.entry(BufferRef::Input(*id)).or_insert(*dtype);
        }
        TensorIr::Simd(SimdNode::Load {
            tier: MemTier::Device(buffer),
            ..
        }) => {
            if let Some(dtype) = egraph[egraph.find(id)].data.dtype {
                dtypes.entry(*buffer).or_insert(dtype);
                if let BufferRef::External(index) = buffer {
                    dtypes.entry(BufferRef::Input(*index)).or_insert(dtype);
                }
            }
        }
        TensorIr::Effect(EffectNode::Store { tier, children }) => {
            if let MemTier::Device(buffer) = tier {
                let value = children[1];
                if let Some(dtype) =
                    infer_effect_value_dtype(egraph, chosen_nodes, value, dtypes, memo)
                {
                    dtypes.entry(*buffer).or_insert(dtype);
                }
            }
        }
        TensorIr::Effect(EffectNode::StoreIf { tier, children }) => {
            if let MemTier::Device(buffer) = tier {
                let value = children[2];
                if let Some(dtype) =
                    infer_effect_value_dtype(egraph, chosen_nodes, value, dtypes, memo)
                {
                    dtypes.entry(*buffer).or_insert(dtype);
                }
            }
        }
        _ => {}
    }
    for child in node.children() {
        collect_buffer_dtypes(egraph, chosen_nodes, *child, dtypes, memo);
    }
}

fn infer_effect_value_dtype(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen_nodes: &HashMap<Id, TensorIr>,
    id: Id,
    buffer_dtypes: &HashMap<BufferRef, DType>,
    memo: &mut HashMap<Id, Option<DType>>,
) -> Option<DType> {
    let canonical = egraph.find(id);
    if let Some(dtype) = memo.get(&canonical) {
        return *dtype;
    }
    let node = chosen_node(egraph, chosen_nodes, canonical);
    let dtype = match node {
        TensorIr::Const(crate::types::ScalarValue::F16(_)) => Some(DType::F16),
        TensorIr::Const(crate::types::ScalarValue::F32(_)) => Some(DType::F32),
        TensorIr::Const(crate::types::ScalarValue::U32(_)) => Some(DType::U32),
        TensorIr::Const(crate::types::ScalarValue::I32(_)) => Some(DType::I32),
        TensorIr::Const(crate::types::ScalarValue::Bool(_)) => Some(DType::Bool),
        TensorIr::Simd(SimdNode::Load {
            tier: MemTier::Device(buffer),
            ..
        }) => buffer_dtypes
            .get(&buffer)
            .or_else(|| match buffer {
                BufferRef::Input(index) => buffer_dtypes.get(&BufferRef::External(index)),
                BufferRef::External(index) => buffer_dtypes.get(&BufferRef::Input(index)),
                _ => None,
            })
            .copied(),
        TensorIr::Simd(SimdNode::Load { tier, .. }) => {
            egraph[egraph.find(id)].data.dtype.or_else(|| match tier {
                MemTier::Threadgroup(buffer) => buffer_dtypes.get(&buffer).copied(),
                MemTier::Device(_) => None,
            })
        }
        TensorIr::UnOp(op, arg) => match op {
            crate::types::UnaryOp::CastF16 => Some(DType::F16),
            crate::types::UnaryOp::CastF32 => Some(DType::F32),
            crate::types::UnaryOp::CastI32 => Some(DType::I32),
            crate::types::UnaryOp::CastU32 => Some(DType::U32),
            crate::types::UnaryOp::CastBool | crate::types::UnaryOp::Not => Some(DType::Bool),
            _ => infer_effect_value_dtype(egraph, chosen_nodes, arg, buffer_dtypes, memo),
        },
        TensorIr::BinOp(_, [lhs, _]) => {
            infer_effect_value_dtype(egraph, chosen_nodes, lhs, buffer_dtypes, memo)
        }
        TensorIr::TernOp(crate::types::TernaryOp::Fma, [arg, _, _]) => {
            infer_effect_value_dtype(egraph, chosen_nodes, arg, buffer_dtypes, memo)
        }
        TensorIr::TernOp(crate::types::TernaryOp::Select, [_, accept, _]) => {
            infer_effect_value_dtype(egraph, chosen_nodes, accept, buffer_dtypes, memo)
        }
        TensorIr::Simd(SimdNode::Theta {
            children: [init, ..],
        }) => infer_effect_value_dtype(egraph, chosen_nodes, init, buffer_dtypes, memo),
        TensorIr::Dispatch(crate::language::DispatchNode::Extract { tuple, .. }) => {
            infer_effect_value_dtype(egraph, chosen_nodes, tuple, buffer_dtypes, memo)
        }
        _ => egraph[canonical].data.dtype,
    };
    memo.insert(canonical, dtype);
    dtype
}

fn collect_buffers(
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen_nodes: &HashMap<Id, TensorIr>,
    id: Id,
    out: &mut HashSet<BufferRef>,
    threadgroup: bool,
) {
    let node = chosen_node(egraph, chosen_nodes, id);
    match &node {
        TensorIr::Simd(SimdNode::Load { tier, .. })
        | TensorIr::Effect(EffectNode::Store { tier, .. })
        | TensorIr::Effect(EffectNode::StoreIf { tier, .. }) => match (threadgroup, tier) {
            (false, MemTier::Device(buffer)) | (true, MemTier::Threadgroup(buffer)) => {
                out.insert(*buffer);
            }
            _ => {}
        },
        _ => {}
    }
    for child in node.children() {
        collect_buffers(egraph, chosen_nodes, *child, out, threadgroup);
    }
}

fn lower_effect_dispatch(
    module: &mut Module,
    egraph: &EGraph<TensorIr, TensorAnalysis>,
    chosen_nodes: &HashMap<Id, TensorIr>,
    dispatch: &EffectDispatchPlan,
    dispatch_idx: usize,
    types: &mut TypeCache,
    device_buffers: &[BufferRef],
    device_buffer_dtypes: &HashMap<BufferRef, DType>,
    tg_buffers: &HashMap<BufferRef, WorkgroupBufferLayout>,
    simd_width: u32,
) -> Result<(), String> {
    let mut buffer_map = HashMap::new();
    let mut buffer_dtypes = HashMap::new();
    register_effect_device_buffers(
        module,
        types,
        device_buffers,
        device_buffer_dtypes,
        &mut buffer_map,
        &mut buffer_dtypes,
    );
    let shape_params_gv = register_shape_params_buffer(
        module,
        types,
        u32::try_from(device_buffers.len()).unwrap_or(0),
    );
    declare_effect_workgroup_buffers(
        module,
        types,
        tg_buffers,
        &mut buffer_map,
        &mut buffer_dtypes,
    );

    let mut ctx = build_effect_codegen_ctx(
        module,
        egraph,
        dispatch,
        types,
        chosen_nodes,
        buffer_map,
        shape_params_gv,
        buffer_dtypes,
        simd_width,
    );
    emit_effect_dispatch_body(&mut ctx, dispatch)?;
    emit_effect_bounds_guard(&mut ctx, dispatch);
    let entry_point = build_effect_entry_point(ctx, dispatch_idx, dispatch, simd_width);
    module.entry_points.push(entry_point);
    Ok(())
}

fn register_effect_device_buffers(
    module: &mut Module,
    types: &mut TypeCache,
    buffers: &[BufferRef],
    buffer_dtype_lookup: &HashMap<BufferRef, DType>,
    buffer_map: &mut HashMap<MemTier, (Handle<GlobalVariable>, bool)>,
    dtype_map: &mut HashMap<MemTier, DType>,
) {
    for (binding_idx, buffer) in buffers.iter().copied().enumerate() {
        let dtype = buffer_dtype_lookup
            .get(&buffer)
            .or_else(|| match buffer {
                BufferRef::Input(index) => buffer_dtype_lookup.get(&BufferRef::External(index)),
                BufferRef::External(index) => buffer_dtype_lookup.get(&BufferRef::Input(index)),
                _ => None,
            })
            .copied()
            .unwrap_or(DType::F32);
        let ty = types.runtime_array(module, dtype);
        let gv = module.global_variables.append(
            GlobalVariable {
                name: Some(format!("buffer_{buffer}")),
                space: AddressSpace::Storage {
                    access: StorageAccess::LOAD | StorageAccess::STORE,
                },
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: binding_idx as u32,
                }),
                ty,
                init: None,
            },
            Span::UNDEFINED,
        );
        let tier = MemTier::Device(buffer);
        buffer_map.insert(tier, (gv, false));
        dtype_map.insert(tier, dtype);
        if let BufferRef::External(index) = buffer {
            let alias = MemTier::Device(BufferRef::Input(index));
            buffer_map.insert(alias, (gv, false));
            dtype_map.insert(alias, dtype);
        }
    }
}

fn register_shape_params_buffer(
    module: &mut Module,
    types: &mut TypeCache,
    binding_idx: u32,
) -> Handle<GlobalVariable> {
    let ty = types.runtime_array(module, DType::U32);
    module.global_variables.append(
        GlobalVariable {
            name: Some("shape_params".into()),
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
    )
}

fn declare_effect_workgroup_buffers(
    module: &mut Module,
    types: &mut TypeCache,
    buffers: &HashMap<BufferRef, WorkgroupBufferLayout>,
    buffer_map: &mut HashMap<MemTier, (Handle<GlobalVariable>, bool)>,
    dtype_map: &mut HashMap<MemTier, DType>,
) {
    let mut buffers = buffers.iter().collect::<Vec<_>>();
    buffers.sort_by_key(|(buffer, _)| **buffer);
    for (buffer, layout) in buffers {
        let tier = MemTier::Threadgroup(*buffer);
        let dtype = layout.dtype;
        let base = types.scalar_for_dtype(module, dtype);
        let Some(elements) = std::num::NonZeroU32::new(layout.elements.max(1)) else {
            continue;
        };
        let fixed_arr = types.array(
            module,
            base,
            ArraySize::Constant(elements),
            dtype.byte_size(),
        );
        let gv = module.global_variables.append(
            GlobalVariable {
                name: Some(format!("{buffer}_tg")),
                space: AddressSpace::WorkGroup,
                binding: None,
                ty: fixed_arr,
                init: None,
            },
            Span::UNDEFINED,
        );
        buffer_map.insert(tier, (gv, true));
        dtype_map.insert(tier, dtype);
    }
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

fn build_effect_codegen_ctx<'a>(
    module: &'a mut Module,
    egraph: &'a EGraph<TensorIr, TensorAnalysis>,
    dispatch: &EffectDispatchPlan,
    types: &'a mut TypeCache,
    chosen_nodes: &'a HashMap<Id, TensorIr>,
    buffer_map: HashMap<MemTier, (Handle<GlobalVariable>, bool)>,
    shape_params_gv: Handle<GlobalVariable>,
    buffer_dtypes: HashMap<MemTier, DType>,
    simd_width: u32,
) -> CodegenCtx<'a> {
    let mut expressions = Arena::new();
    let local_invocation_id = expressions.append(Expression::FunctionArgument(0), Span::UNDEFINED);
    let workgroup_id = expressions.append(Expression::FunctionArgument(1), Span::UNDEFINED);

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
        shape_params_gv,
        buffer_dtypes,
        tg_sg_read_strides: HashMap::new(),
        local_invocation_id,
        workgroup_id,
        emitted: HashSet::new(),
        named_expressions: naga::FastIndexMap::default(),
    }
}

fn emit_effect_dispatch_body(
    ctx: &mut CodegenCtx<'_>,
    dispatch: &EffectDispatchPlan,
) -> Result<(), String> {
    let lane = ctx.lower_index(IndexLevel::Lane);
    let simdgroup = ctx.lower_index(IndexLevel::Simdgroup);
    let workgroup = ctx.lower_index(IndexLevel::Workgroup);
    ctx.binder_stack.push(BinderFrame::Dispatch(DispatchFrame {
        lane,
        simdgroup,
        workgroup,
    }));
    emit_effect_node(ctx, dispatch.state)?;
    emit_effect_node(ctx, dispatch.body)?;
    ctx.binder_stack.pop();
    Ok(())
}

fn emit_effect_node(ctx: &mut CodegenCtx<'_>, id: Id) -> Result<(), String> {
    match ctx.select_lowering_node(ctx.egraph.find(id)) {
        TensorIr::Effect(EffectNode::Token) => {}
        TensorIr::Effect(EffectNode::Seq(list_id)) => {
            for step in extract_list(ctx.egraph, list_id) {
                emit_effect_node(ctx, step)?;
            }
        }
        TensorIr::Effect(EffectNode::Store { tier, children }) => {
            emit_effect_node(ctx, children[2])?;
            emit_effect_store(ctx, &tier, children[0], children[1]);
        }
        TensorIr::Effect(EffectNode::StoreIf { tier, children }) => {
            emit_effect_node(ctx, children[3])?;
            ctx.emit_store_if(&tier, children[0], children[1], children[2]);
        }
        TensorIr::Effect(EffectNode::Barrier { state, .. }) => {
            emit_effect_node(ctx, state)?;
            ctx.body.push(
                Statement::ControlBarrier(naga::Barrier::WORK_GROUP),
                Span::UNDEFINED,
            );
        }
        other => {
            return Err(format!(
                "expected effect node in dispatch body, found {other:?}"
            ));
        }
    }
    Ok(())
}

fn emit_effect_store(ctx: &mut CodegenCtx<'_>, tier: &MemTier, addr_id: Id, value_id: Id) {
    if matches!(tier, MemTier::Device(_))
        && !ctx.egraph[ctx.egraph.find(addr_id)]
            .data
            .dep
            .contains_lane()
    {
        let addr_h = ctx.lower_expr(addr_id);
        let val_h = ctx.lower_expr(value_id);
        let (gv, _) = *ctx
            .buffer_map
            .get(tier)
            .unwrap_or_else(|| panic!("unknown buffer for store: {tier}"));
        let gv_expr = ctx
            .expressions
            .append(Expression::GlobalVariable(gv), Span::UNDEFINED);
        let ptr = ctx.emit_access(gv_expr, addr_h);
        emit_lane_zero_output_store(ctx, ptr, val_h);
    } else {
        ctx.emit_store(tier, addr_id, value_id);
    }
}

fn emit_effect_bounds_guard(ctx: &mut CodegenCtx<'_>, dispatch: &EffectDispatchPlan) {
    let Some(workgroups) = dispatch.workgroups.as_const() else {
        return;
    };
    let physical_workgroups = workgroups.div_ceil(dispatch.simdgroups.max(1));
    if physical_workgroups <= MAX_DISPATCH_WORKGROUPS_PER_DIMENSION {
        return;
    }

    let workgroup = ctx.lower_index(IndexLevel::Workgroup);
    let bound = ctx.lower_dim(&dispatch.workgroups);
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

fn build_effect_entry_point(
    ctx: CodegenCtx<'_>,
    dispatch_idx: usize,
    dispatch: &EffectDispatchPlan,
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
