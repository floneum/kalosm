use std::{collections::HashMap, sync::Arc};

use tensor_ir::{
    DType, DispatchProgram, LoweringOptions, SimdProgram, StageConfig, StagedPipeline,
    TensorExprProgram, TensorIr, lower_dispatch_program, module_to_wgsl, verify,
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
    let (simd, mut wgsl) =
        lower_valid_simd_program(&pipeline, expr, output_datatype).map_err(|error| {
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
    let program = simd.dispatch_program();
    let final_dispatch_index = program
        .dispatches
        .len()
        .checked_sub(1)
        .ok_or_else(|| "tensor_ir program produced no dispatches".to_string())?;
    let final_datatype = dispatch_output_datatype(program, final_dispatch_index)?;
    if final_datatype != output_datatype {
        return Err(format!(
            "tensor_ir output dtype mismatch: program produced {final_datatype}, caller expected {output_datatype}"
        ));
    }
    let shader = device.create_shader_module(wgsl);

    let mut produced_buffers: HashMap<egg::Id, Arc<Buffer>> = HashMap::default();
    let mut final_output = None;

    for (dispatch_index, dispatch) in program.dispatches.iter().enumerate() {
        let pipeline =
            device
                .wgpu_device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Fusor Tensor IR Pipeline"),
                    layout: None,
                    module: &shader,
                    entry_point: Some(&format!("dispatch_{dispatch_index}_")),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let dispatch_datatype = dispatch_output_datatype(program, dispatch_index)?;
        let output_buffer = if dispatch_index + 1 == program.dispatches.len() {
            let required_elems =
                (dispatch.workgroups * program.device.simd_width) as usize * dispatch.outputs.len();
            let shape_elems = output_shape.iter().product::<usize>();
            TensorData::new_for_shape(device, &[required_elems.max(shape_elems)], output_datatype)
                .buffer()
                .clone()
        } else {
            let output_elems =
                (dispatch.workgroups * program.device.simd_width) as usize * dispatch.outputs.len();
            TensorData::new_for_shape(device, &[output_elems], dispatch_datatype)
                .buffer()
                .clone()
        };

        let bind_group = create_bind_group(
            device,
            program,
            inputs,
            &produced_buffers,
            &pipeline,
            dispatch_index,
            output_buffer.clone(),
        )?;

        {
            let mut pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Fusor Tensor IR Dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let physical_workgroups = dispatch.workgroups / dispatch.simdgroups.max(1);
            let (x, y, z) = dispatch_grid(physical_workgroups);
            pass.dispatch_workgroups(x, y, z);
        }

        produced_buffers.insert(
            program.egraph.find(dispatch.semantic_output_id),
            output_buffer.clone(),
        );
        if dispatch_index + 1 == program.dispatches.len() {
            final_output = Some(output_buffer);
        }
    }

    let final_output =
        final_output.ok_or_else(|| "tensor_ir program produced no output".to_string())?;
    Ok(TensorData::new_from_buffer(
        device,
        final_output,
        output_shape,
        output_datatype,
    ))
}

fn lower_valid_simd_program(
    pipeline: &StagedPipeline,
    expr: &TensorExprProgram,
    output_datatype: DataTypeEnum,
) -> Result<(SimdProgram, String), tensor_ir::LoweringError> {
    let (kernel, report) = lower_with_report(pipeline, expr)?;
    let mut errors = Vec::new();
    match compile_valid_candidate(pipeline, kernel, output_datatype) {
        Ok(compiled) => return Ok(compiled),
        Err(error) => errors.push(error),
    }

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

    let mut fallback_config = pipeline.config().clone();
    fallback_config.runner.lowering = LoweringOptions::readable();
    fallback_config.runner.iter_limit = 1;
    let fallback = StagedPipeline::new(fallback_config);
    if fallback.config().runner.lowering != pipeline.config().runner.lowering {
        match lower_with_report(&fallback, expr) {
            Ok((candidate, _)) => {
                match compile_valid_candidate(&fallback, candidate, output_datatype) {
                    Ok(compiled) => return Ok(compiled),
                    Err(error) => errors.push(error),
                }
            }
            Err(error) => errors.push(error.message),
        }
        match fallback.lower_candidates(expr, 16) {
            Ok(candidates) => {
                for candidate in candidates {
                    match compile_valid_candidate(&fallback, candidate, output_datatype) {
                        Ok(compiled) => return Ok(compiled),
                        Err(error) => errors.push(error),
                    }
                }
            }
            Err(error) => errors.push(error),
        }
    }

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
    pipeline: &StagedPipeline,
    kernel: tensor_ir::KernelProgram,
    output_datatype: DataTypeEnum,
) -> Result<(SimdProgram, String), String> {
    let simd = pipeline.compile(kernel)?;
    let program = simd.dispatch_program();
    let final_dispatch_index = program
        .dispatches
        .len()
        .checked_sub(1)
        .ok_or_else(|| "tensor_ir program produced no dispatches".to_string())?;
    let final_datatype = dispatch_output_datatype(program, final_dispatch_index)?;
    if final_datatype != output_datatype {
        return Err(format!(
            "tensor_ir output dtype mismatch: program produced {final_datatype}, caller expected {output_datatype}"
        ));
    }
    let verified = verify(program).map_err(|error| format!("verification error: {error}"))?;
    let module = lower_dispatch_program(verified);
    let wgsl = module_to_wgsl(&module)?;
    Ok((simd, wgsl))
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

fn dispatch_output_datatype(
    program: &DispatchProgram,
    dispatch_index: usize,
) -> Result<DataTypeEnum, String> {
    let dispatch = &program.dispatches[dispatch_index];
    let dtype = program.egraph[program.egraph.find(dispatch.semantic_output_id)]
        .data
        .dtype
        .or_else(|| {
            dispatch.outputs.first().and_then(|output| {
                program.egraph[program.egraph.find(output.value_id)]
                    .data
                    .dtype
            })
        })
        .unwrap_or(DType::F32);
    dtype_to_datatype(dtype)
}

fn create_bind_group(
    device: &Device,
    program: &DispatchProgram,
    inputs: &[TensorData],
    produced_buffers: &HashMap<egg::Id, Arc<Buffer>>,
    pipeline: &ComputePipeline,
    dispatch_index: usize,
    output_buffer: Arc<Buffer>,
) -> Result<wgpu::BindGroup, String> {
    let dispatch = &program.dispatches[dispatch_index];
    let input_buffers = dispatch
        .inputs
        .iter()
        .map(|input_id| resolve_dispatch_input_buffer(program, *input_id, inputs, produced_buffers))
        .collect::<Result<Vec<_>, _>>()?;
    let mut entries = Vec::new();

    for (binding, buffer) in input_buffers.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        });
    }
    entries.push(wgpu::BindGroupEntry {
        binding: dispatch.inputs.len() as u32,
        resource: output_buffer.as_entire_binding(),
    });

    Ok(device
        .wgpu_device()
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Fusor Tensor IR Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        }))
}

fn resolve_dispatch_input_buffer(
    program: &DispatchProgram,
    input_id: egg::Id,
    external_inputs: &[TensorData],
    produced_buffers: &HashMap<egg::Id, Arc<Buffer>>,
) -> Result<Arc<Buffer>, String> {
    let canonical = program.egraph.find(input_id);
    if let Some(buffer) = produced_buffers.get(&canonical) {
        return Ok(buffer.clone());
    }

    for node in program.egraph[canonical].iter() {
        match node {
            TensorIr::HighLevel(tensor_ir::HighLevelNode::Input { id, .. })
            | TensorIr::Simd(tensor_ir::SimdNode::Load {
                tier: tensor_ir::MemTier::Device(tensor_ir::BufferRef::Input(id)),
                ..
            }) => {
                let index = *id as usize;
                let input = external_inputs.get(index).ok_or_else(|| {
                    format!(
                        "tensor_ir dispatch expected input_{index}, but only {} inputs were provided",
                        external_inputs.len()
                    )
                })?;
                return Ok(input.buffer().clone());
            }
            _ => {}
        }
    }

    Err(format!(
        "unresolved tensor_ir dispatch input: {canonical:?}"
    ))
}
