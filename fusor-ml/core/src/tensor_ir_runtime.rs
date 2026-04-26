use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use tensor_ir::language::EffectNode;
use tensor_ir::{
    BufferRef, DType, Dim, MemTier, ShapeParams, StageConfig, StagedPipeline, TensorExprProgram,
    TensorIr, lower_to_wgsl,
};
use wgpu::{Buffer, CommandEncoder, ComputePipeline};

use crate::{DataTypeEnum, Device, TensorData};

const MAX_DISPATCH_WORKGROUPS_PER_DIMENSION: u32 = 65_535;

pub(crate) fn device_profile(device: &Device) -> tensor_ir::DeviceProfile {
    let limits = device.limits();
    // tensor_ir's current elementwise lowering requires the logical element
    // count to be divisible by the configured SIMD width. Fusor tensors are
    // arbitrary-shaped, so use scalar lanes until tensor_ir grows tail masks.
    let simd_width = 1;
    let max_simdgroups = 1;
    tensor_ir::DeviceProfile {
        simd_width,
        max_threadgroup_bytes: limits.max_compute_workgroup_storage_size,
        // Keep Fusor on single-output dispatches until tensor_ir's generic
        // register blocking handles batched contractions with broadcasted
        // operands soundly.
        max_registers_per_lane: 1,
        max_simdgroups,
        max_workgroup_size: limits.max_compute_invocations_per_workgroup,
    }
}

pub(crate) fn execute(
    device: &Device,
    expr: &TensorExprProgram,
    inputs: &[TensorData],
    output_shape: &[usize],
    output_datatype: DataTypeEnum,
    command_encoder: &mut CommandEncoder,
) -> Result<TensorData, String> {
    ensure_device_supports_dtype(device, output_datatype)?;
    for input in inputs {
        ensure_device_supports_dtype(device, input.datatype())?;
    }

    let mut config = StageConfig::default();
    config.runner.device = device_profile(device);
    config.runner.iter_limit = 10;
    config.runner.node_limit = 50_000;
    config.runner.time_limit_secs = 30;
    config.candidate_limit = Some(1);
    let pipeline = StagedPipeline::new(config);
    let (kernel, plan, mut wgsl) =
        lower_valid_kernel(&pipeline, expr, output_datatype).map_err(|error| {
            let extraction = error
                .report
                .extraction
                .as_ref()
                .map(|extraction| format!("{:?}", extraction.candidate_validation))
                .unwrap_or_else(|| "no extraction report".to_string());
            format!("{} ({extraction})", error.message)
        })?;
    if wgsl.contains("f16") && !wgsl.contains("enable f16;") {
        if !device.f16_supported() {
            return Err("tensor_ir runtime requires SHADER_F16 for f16 shader operations".into());
        }
        wgsl = format!("enable f16;\n{wgsl}");
    }
    let shader = device.create_shader_module(wgsl.as_str());

    let mut device_buffers: HashMap<BufferRef, Arc<Buffer>> = HashMap::default();
    let shape_params = ShapeParams::new(
        output_shape
            .iter()
            .map(|dim| {
                u32::try_from(*dim)
                    .map_err(|_| format!("tensor_ir output dimension {dim} exceeds u32::MAX"))
            })
            .collect::<Result<Vec<_>, _>>()?,
    );
    let shape_param_words = shape_params.storage_words();
    let shape_params_buffer = device.create_buffer_init(
        bytemuck::cast_slice(&shape_param_words),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    for buffer in &plan.device_buffers {
        match *buffer {
            BufferRef::External(index) | BufferRef::Input(index) => {
                let input = inputs.get(index as usize).ok_or_else(|| {
                    format!(
                        "tensor_ir dispatch expected input_{index}, but only {} inputs were provided",
                        inputs.len()
                    )
                })?;
                device_buffers.insert(*buffer, input.buffer().clone());
            }
            BufferRef::Tensor(_) | BufferRef::Output(_) => {
                let mut elems = plan
                    .buffer_elements(buffer, kernel.device(), &shape_params)
                    .unwrap_or_else(|| output_shape.iter().product::<usize>().max(1));
                if plan.outputs.contains(buffer) {
                    elems = elems.max(output_shape.iter().product::<usize>());
                }
                let datatype = plan
                    .buffer_dtypes
                    .get(buffer)
                    .copied()
                    .unwrap_or(output_datatype);
                ensure_device_supports_dtype(device, datatype)?;
                let bytes = (elems * datatype.element_size()).max(1) as u64;
                let bytes = bytes.next_multiple_of(4);
                let mut usage = wgpu::BufferUsages::STORAGE;
                if plan.outputs.contains(buffer) {
                    usage |= wgpu::BufferUsages::COPY_SRC;
                }
                let storage = device.create_buffer(bytes, usage);
                device_buffers.insert(*buffer, storage);
            }
        }
    }

    for (dispatch_index, dispatch) in plan.dispatches.iter().enumerate() {
        let entry_point = format!("dispatch_{dispatch_index}_");
        let active_bindings = active_wgsl_bindings_for_entrypoint(&wgsl, &entry_point);
        let pipeline =
            device
                .wgpu_device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Fusor Tensor IR Pipeline"),
                    layout: None,
                    module: &shader,
                    entry_point: Some(&entry_point),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let dispatch_workgroups = dispatch.workgroups.eval_u32(&shape_params).ok_or_else(|| {
            "fusor tensor_ir runtime is missing shape parameters for dynamic workgroups".to_string()
        })?;

        let bind_group = create_bind_group(
            device,
            &plan,
            &device_buffers,
            &active_bindings,
            &pipeline,
            shape_params_buffer.as_ref(),
        )?;

        {
            let mut pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fusor Tensor IR Dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let physical_workgroups = dispatch_workgroups.div_ceil(dispatch.simdgroups.max(1));
            let (x, y, z) = dispatch_grid(physical_workgroups);
            pass.dispatch_workgroups(x, y, z);
        }
    }

    let final_ref = plan
        .outputs
        .first()
        .ok_or_else(|| "tensor_ir program produced no output".to_string())?;
    let final_output = device_buffers
        .get(final_ref)
        .cloned()
        .ok_or_else(|| format!("tensor_ir final output buffer {final_ref} was not allocated"))?;
    Ok(TensorData::new_from_buffer(
        device,
        final_output,
        output_shape,
        output_datatype,
    ))
}

fn lower_valid_kernel(
    pipeline: &StagedPipeline,
    expr: &TensorExprProgram,
    output_datatype: DataTypeEnum,
) -> Result<(tensor_ir::KernelProgram, EffectRuntimePlan, String), tensor_ir::LoweringError> {
    let mut errors = Vec::new();

    match pipeline.lower_candidates(expr, 16) {
        Ok(candidates) => {
            for candidate in candidates {
                match compile_valid_candidate(pipeline, candidate, output_datatype) {
                    Ok(compiled) => return Ok(compiled),
                    Err(error) => errors.push(error),
                }
            }
        }
        Err(error) => errors.push(error),
    }

    let report = lower_with_report(pipeline, expr)
        .map(|(_, report)| report)
        .unwrap_or_else(|error| error.report);
    let message = format!(
        "no tensor_ir candidate produced a valid WGSL shader: {}",
        errors
            .last()
            .cloned()
            .unwrap_or_else(|| "no candidates returned".to_string())
    );
    Err(tensor_ir::LoweringError::new(message, report))
}

fn compile_valid_candidate(
    _pipeline: &StagedPipeline,
    kernel: tensor_ir::KernelProgram,
    output_datatype: DataTypeEnum,
) -> Result<(tensor_ir::KernelProgram, EffectRuntimePlan, String), String> {
    let plan = EffectRuntimePlan::from_kernel(&kernel, output_datatype)?;
    let wgsl = lower_to_wgsl(&kernel)?;
    Ok((kernel, plan, wgsl))
}

fn dispatch_grid(physical_workgroups: u32) -> (u32, u32, u32) {
    if physical_workgroups <= MAX_DISPATCH_WORKGROUPS_PER_DIMENSION {
        return (physical_workgroups, 1, 1);
    }

    let y = physical_workgroups.div_ceil(MAX_DISPATCH_WORKGROUPS_PER_DIMENSION);
    (MAX_DISPATCH_WORKGROUPS_PER_DIMENSION, y, 1)
}

fn ensure_device_supports_dtype(device: &Device, datatype: DataTypeEnum) -> Result<(), String> {
    if datatype == DataTypeEnum::F16 && !device.f16_supported() {
        return Err("tensor_ir runtime requires SHADER_F16 for f16 tensors".to_string());
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct EffectRuntimeDispatch {
    workgroups: Dim,
    simdgroups: u32,
    stores: Vec<BufferRef>,
}

#[derive(Debug, Clone)]
struct EffectRuntimePlan {
    dispatches: Vec<EffectRuntimeDispatch>,
    device_buffers: Vec<BufferRef>,
    buffer_dtypes: HashMap<BufferRef, DataTypeEnum>,
    outputs: Vec<BufferRef>,
}

impl EffectRuntimePlan {
    fn from_kernel(
        kernel: &tensor_ir::KernelProgram,
        output_datatype: DataTypeEnum,
    ) -> Result<Self, String> {
        let nodes = kernel.extracted().as_ref();
        let root_idx = nodes
            .len()
            .checked_sub(1)
            .ok_or_else(|| "tensor_ir program is empty".to_string())?;
        let TensorIr::Effect(EffectNode::Program {
            children: [_, body, outputs],
        }) = &nodes[root_idx]
        else {
            return Err("tensor_ir runtime expected an EffectNode::Program root".to_string());
        };

        let outputs = tensor_ir::language::extract_recexpr_list(nodes, *outputs)
            .into_iter()
            .map(|id| tensor_marker_to_buffer(nodes, id))
            .collect::<Result<Vec<_>, _>>()?;
        let mut dispatches = Vec::new();
        collect_effect_dispatches(nodes, *body, &mut dispatches)?;
        if dispatches.is_empty() {
            return Err("tensor_ir program contains no dispatches".to_string());
        }

        let mut device_buffers = HashSet::new();
        collect_device_buffers(nodes, egg::Id::from(root_idx), &mut device_buffers);
        for output in &outputs {
            device_buffers.insert(*output);
        }
        let mut device_buffers = device_buffers.into_iter().collect::<Vec<_>>();
        device_buffers.sort();

        let mut buffer_dtypes = HashMap::new();
        collect_device_buffer_dtypes(kernel, nodes, egg::Id::from(root_idx), &mut buffer_dtypes)?;
        for output in &outputs {
            let datatype = buffer_dtypes
                .get(output)
                .copied()
                .unwrap_or(output_datatype);
            if datatype != output_datatype {
                return Err(format!(
                    "tensor_ir output dtype mismatch: program produced {datatype}, caller expected {output_datatype}"
                ));
            }
            buffer_dtypes.insert(*output, output_datatype);
        }

        Ok(Self {
            dispatches,
            device_buffers,
            buffer_dtypes,
            outputs,
        })
    }

    fn buffer_elements(
        &self,
        buffer: &BufferRef,
        device: &tensor_ir::DeviceProfile,
        shape_params: &ShapeParams,
    ) -> Option<usize> {
        let mut elems = None;
        for dispatch in &self.dispatches {
            if !dispatch.stores.contains(buffer) {
                continue;
            }
            let workgroups = dispatch.workgroups.eval_u32(shape_params)?.max(1);
            let stores = dispatch.stores.len().max(1);
            let dispatch_elems = workgroups as usize * device.simd_width as usize * stores;
            elems = Some(elems.map_or(dispatch_elems, |old: usize| old.max(dispatch_elems)));
        }
        elems
    }
}

fn tensor_marker_to_buffer(nodes: &[TensorIr], id: egg::Id) -> Result<BufferRef, String> {
    match nodes.get(usize::from(id)) {
        Some(TensorIr::Const(tensor_ir::ScalarValue::U32(raw))) => {
            Ok(BufferRef::Tensor(tensor_ir::TensorId(*raw)))
        }
        other => Err(format!(
            "program output marker must be a tensor id, got {other:?}"
        )),
    }
}

fn collect_effect_dispatches(
    nodes: &[TensorIr],
    id: egg::Id,
    out: &mut Vec<EffectRuntimeDispatch>,
) -> Result<(), String> {
    match &nodes[usize::from(id)] {
        TensorIr::Effect(EffectNode::Seq(list)) => {
            for child in tensor_ir::language::extract_recexpr_list(nodes, *list) {
                collect_effect_dispatches(nodes, child, out)?;
            }
        }
        TensorIr::Effect(EffectNode::Dispatch {
            workgroups,
            simdgroups,
            children: [_, body],
        }) => {
            let mut stores = Vec::new();
            collect_effect_stores(nodes, *body, &mut stores)?;
            out.push(EffectRuntimeDispatch {
                workgroups: workgroups.clone(),
                simdgroups: *simdgroups,
                stores,
            });
        }
        TensorIr::Effect(EffectNode::Token) => {}
        other => {
            return Err(format!(
                "program body contains non-effect dispatch node {other:?}"
            ));
        }
    }
    Ok(())
}

fn collect_effect_stores(
    nodes: &[TensorIr],
    id: egg::Id,
    stores: &mut Vec<BufferRef>,
) -> Result<(), String> {
    match &nodes[usize::from(id)] {
        TensorIr::Effect(EffectNode::Token) => {}
        TensorIr::Effect(EffectNode::Store { tier, children }) => {
            collect_effect_stores(nodes, children[2], stores)?;
            if let MemTier::Device(buffer) = tier {
                stores.push(*buffer);
            }
        }
        TensorIr::Effect(EffectNode::StoreIf { tier, children }) => {
            collect_effect_stores(nodes, children[3], stores)?;
            if let MemTier::Device(buffer) = tier {
                stores.push(*buffer);
            }
        }
        TensorIr::Effect(EffectNode::Barrier { state, .. }) => {
            collect_effect_stores(nodes, *state, stores)?;
        }
        other => {
            return Err(format!(
                "dispatch body contains non-effect state node {other:?}"
            ));
        }
    }
    Ok(())
}

fn collect_device_buffers(nodes: &[TensorIr], id: egg::Id, buffers: &mut HashSet<BufferRef>) {
    let node = &nodes[usize::from(id)];
    match node {
        TensorIr::Simd(tensor_ir::SimdNode::Load {
            tier: MemTier::Device(buffer),
            ..
        })
        | TensorIr::Effect(EffectNode::Store {
            tier: MemTier::Device(buffer),
            ..
        })
        | TensorIr::Effect(EffectNode::StoreIf {
            tier: MemTier::Device(buffer),
            ..
        }) => {
            buffers.insert(*buffer);
        }
        _ => {}
    }
    for child in egg::Language::children(node) {
        collect_device_buffers(nodes, *child, buffers);
    }
}

fn collect_device_buffer_dtypes(
    kernel: &tensor_ir::KernelProgram,
    nodes: &[TensorIr],
    id: egg::Id,
    dtypes: &mut HashMap<BufferRef, DataTypeEnum>,
) -> Result<(), String> {
    let node = &nodes[usize::from(id)];
    match node {
        TensorIr::HighLevel(tensor_ir::HighLevelNode::Input { id, dtype, .. }) => {
            let datatype = dtype_to_datatype(*dtype)?;
            insert_buffer_dtype(dtypes, BufferRef::External(*id), datatype)?;
            insert_buffer_dtype(dtypes, BufferRef::Input(*id), datatype)?;
        }
        TensorIr::Simd(tensor_ir::SimdNode::Load {
            tier: MemTier::Device(buffer),
            ..
        }) => {
            if let Some(dtype) = extracted_node_dtype(kernel, id) {
                insert_buffer_dtype(dtypes, *buffer, dtype_to_datatype(dtype)?)?;
                if let BufferRef::External(index) = buffer {
                    insert_buffer_dtype(
                        dtypes,
                        BufferRef::Input(*index),
                        dtype_to_datatype(dtype)?,
                    )?;
                }
            }
        }
        TensorIr::Effect(EffectNode::Store { tier, children }) => {
            if let MemTier::Device(buffer) = tier
                && let Some(dtype) = infer_value_dtype(kernel, nodes, children[1], dtypes)?
            {
                insert_buffer_dtype(dtypes, *buffer, dtype)?;
            }
        }
        TensorIr::Effect(EffectNode::StoreIf { tier, children }) => {
            if let MemTier::Device(buffer) = tier
                && let Some(dtype) = infer_value_dtype(kernel, nodes, children[2], dtypes)?
            {
                insert_buffer_dtype(dtypes, *buffer, dtype)?;
            }
        }
        _ => {}
    }
    for child in egg::Language::children(node) {
        collect_device_buffer_dtypes(kernel, nodes, *child, dtypes)?;
    }
    Ok(())
}

fn insert_buffer_dtype(
    dtypes: &mut HashMap<BufferRef, DataTypeEnum>,
    buffer: BufferRef,
    datatype: DataTypeEnum,
) -> Result<(), String> {
    if let Some(existing) = dtypes.insert(buffer, datatype)
        && existing != datatype
    {
        return Err(format!(
            "tensor_ir buffer {buffer} has inconsistent dtypes {existing} and {datatype}"
        ));
    }
    Ok(())
}

fn extracted_node_dtype(kernel: &tensor_ir::KernelProgram, id: egg::Id) -> Option<DType> {
    kernel
        .extracted_program()
        .eclass_for_node()
        .get(usize::from(id))
        .copied()
        .flatten()
        .and_then(|eclass| kernel.egraph()[kernel.egraph().find(eclass)].data.dtype)
}

fn infer_value_dtype(
    kernel: &tensor_ir::KernelProgram,
    nodes: &[TensorIr],
    id: egg::Id,
    buffer_dtypes: &HashMap<BufferRef, DataTypeEnum>,
) -> Result<Option<DataTypeEnum>, String> {
    if let Some(dtype) = extracted_node_dtype(kernel, id) {
        return dtype_to_datatype(dtype).map(Some);
    }
    let dtype = match &nodes[usize::from(id)] {
        TensorIr::Const(tensor_ir::ScalarValue::F16(_)) => Some(DataTypeEnum::F16),
        TensorIr::Const(tensor_ir::ScalarValue::F32(_)) => Some(DataTypeEnum::F32),
        TensorIr::Const(tensor_ir::ScalarValue::U32(_)) => Some(DataTypeEnum::U32),
        TensorIr::Const(tensor_ir::ScalarValue::I32(_) | tensor_ir::ScalarValue::Bool(_)) => {
            return Err("fusor tensor_ir runtime cannot materialize i32/bool buffers".to_string());
        }
        TensorIr::Simd(tensor_ir::SimdNode::Load {
            tier: MemTier::Device(buffer),
            ..
        }) => buffer_dtypes
            .get(buffer)
            .or_else(|| match buffer {
                BufferRef::Input(index) => buffer_dtypes.get(&BufferRef::External(*index)),
                BufferRef::External(index) => buffer_dtypes.get(&BufferRef::Input(*index)),
                _ => None,
            })
            .copied(),
        TensorIr::UnOp(op, arg) => match op {
            tensor_ir::UnaryOp::CastF16 => Some(DataTypeEnum::F16),
            tensor_ir::UnaryOp::CastF32 => Some(DataTypeEnum::F32),
            tensor_ir::UnaryOp::CastU32 => Some(DataTypeEnum::U32),
            tensor_ir::UnaryOp::CastI32
            | tensor_ir::UnaryOp::CastBool
            | tensor_ir::UnaryOp::Not => {
                return Err(
                    "fusor tensor_ir runtime cannot materialize i32/bool buffers".to_string(),
                );
            }
            _ => infer_value_dtype(kernel, nodes, *arg, buffer_dtypes)?,
        },
        TensorIr::BinOp(op, [lhs, _])
            if !matches!(
                op,
                tensor_ir::BinaryOp::Lt
                    | tensor_ir::BinaryOp::Le
                    | tensor_ir::BinaryOp::Gt
                    | tensor_ir::BinaryOp::Ge
                    | tensor_ir::BinaryOp::Eq
                    | tensor_ir::BinaryOp::Neq
            ) =>
        {
            infer_value_dtype(kernel, nodes, *lhs, buffer_dtypes)?
        }
        TensorIr::TernOp(tensor_ir::TernaryOp::Fma, [arg, _, _]) => {
            infer_value_dtype(kernel, nodes, *arg, buffer_dtypes)?
        }
        TensorIr::TernOp(tensor_ir::TernaryOp::Select, [_, accept, _]) => {
            infer_value_dtype(kernel, nodes, *accept, buffer_dtypes)?
        }
        TensorIr::Simd(tensor_ir::SimdNode::Theta {
            children: [init, ..],
        }) => infer_value_dtype(kernel, nodes, *init, buffer_dtypes)?,
        TensorIr::Dispatch(tensor_ir::DispatchNode::Extract { tuple, .. }) => {
            infer_value_dtype(kernel, nodes, *tuple, buffer_dtypes)?
        }
        _ => None,
    };
    Ok(dtype)
}

fn dtype_to_datatype(dtype: DType) -> Result<DataTypeEnum, String> {
    match dtype {
        DType::F16 => Ok(DataTypeEnum::F16),
        DType::F32 => Ok(DataTypeEnum::F32),
        DType::U32 => Ok(DataTypeEnum::U32),
        DType::I32 | DType::Bool => Err(format!(
            "fusor tensor_ir runtime cannot materialize {dtype} tensors"
        )),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn lower_with_report(
    pipeline: &StagedPipeline,
    expr: &TensorExprProgram,
) -> Result<(tensor_ir::KernelProgram, tensor_ir::LoweringReport), tensor_ir::LoweringError> {
    let pipeline = pipeline.clone();
    let expr = expr.clone();
    let expr_nodes = expr.nodes().len();
    std::thread::Builder::new()
        .name("fusor-tensor-ir-lowering".to_string())
        .stack_size(128 * 1024 * 1024)
        .spawn(move || pipeline.lower_with_report(&expr))
        .map_err(|error| {
            tensor_ir::LoweringError::new(
                format!("failed to spawn tensor_ir lowering thread: {error}"),
                tensor_ir::LoweringReport::new(expr_nodes),
            )
        })?
        .join()
        .map_err(|_| {
            tensor_ir::LoweringError::new(
                "tensor_ir lowering thread panicked".to_string(),
                tensor_ir::LoweringReport::new(expr_nodes),
            )
        })?
}

#[cfg(target_arch = "wasm32")]
fn lower_with_report(
    pipeline: &StagedPipeline,
    expr: &TensorExprProgram,
) -> Result<(tensor_ir::KernelProgram, tensor_ir::LoweringReport), tensor_ir::LoweringError> {
    pipeline.lower_with_report(expr)
}

fn create_bind_group(
    device: &Device,
    plan: &EffectRuntimePlan,
    device_buffers: &HashMap<BufferRef, Arc<Buffer>>,
    active_bindings: &HashSet<u32>,
    pipeline: &ComputePipeline,
    shape_params_buffer: &Buffer,
) -> Result<wgpu::BindGroup, String> {
    let mut entries = Vec::new();

    for (binding, buffer_ref) in plan.device_buffers.iter().enumerate() {
        if !active_bindings.contains(&(binding as u32)) {
            continue;
        }
        let buffer = device_buffers
            .get(buffer_ref)
            .ok_or_else(|| format!("missing tensor_ir buffer binding for {buffer_ref}"))?;
        entries.push(wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        });
    }
    let shape_binding = plan.device_buffers.len() as u32;
    if active_bindings.contains(&shape_binding) {
        entries.push(wgpu::BindGroupEntry {
            binding: shape_binding,
            resource: shape_params_buffer.as_entire_binding(),
        });
    }

    Ok(device
        .wgpu_device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fusor Tensor IR Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        }))
}

fn active_wgsl_bindings_for_entrypoint(wgsl: &str, entry_point: &str) -> HashSet<u32> {
    let mut binding_names = Vec::new();
    let mut pending_binding = None;
    for line in wgsl.lines() {
        if let Some(binding_attr) = line.split("@binding(").nth(1)
            && let Some(raw) = binding_attr.split(')').next()
            && let Ok(binding) = raw.trim().parse()
        {
            pending_binding = Some(binding);
        }
        if let Some(binding) = pending_binding
            && let Some(name_part) = line.split("var<").nth(1)
            && let Some(name) = name_part
                .split('>')
                .nth(1)
                .and_then(|rest| rest.split(':').next())
                .map(str::trim)
        {
            binding_names.push((binding, name.to_string()));
            pending_binding = None;
        }
    }

    let body = wgsl
        .split(&format!("fn {entry_point}("))
        .nth(1)
        .unwrap_or(wgsl);
    let mut bindings = HashSet::new();
    for (binding, name) in binding_names {
        if body.contains(&name) {
            bindings.insert(binding);
        }
    }
    bindings
}
