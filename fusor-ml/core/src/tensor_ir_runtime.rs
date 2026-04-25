use std::{collections::HashMap, sync::Arc};

use tensor_ir::{
    DispatchProgram, StageConfig, StagedPipeline, TensorExprProgram, TensorIr, lower_to_wgsl,
};
use wgpu::{Buffer, CommandEncoder, ComputePipeline};

use crate::{DataTypeEnum, Device, TensorData};

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
        max_registers_per_lane: 128,
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
    if output_datatype != DataTypeEnum::F32 {
        return Err(format!(
            "tensor_ir runtime currently requires f32 outputs, got {output_datatype}"
        ));
    }
    if inputs
        .iter()
        .any(|input| input.datatype() != DataTypeEnum::F32)
    {
        return Err("tensor_ir runtime currently requires f32 tensor inputs".to_string());
    }

    let mut config = StageConfig::default();
    config.runner.device = device_profile(device);
    let pipeline = StagedPipeline::new(config);
    let kernel = pipeline.lower_with_report(expr).map_err(|error| {
        let extraction = error
            .report
            .extraction
            .as_ref()
            .map(|extraction| format!("{:?}", extraction.candidate_validation))
            .unwrap_or_else(|| "no extraction report".to_string());
        format!("{} ({extraction})", error.message)
    })?;
    let simd = pipeline.compile(kernel.0)?;
    let program = simd.dispatch_program();

    let wgsl = lower_to_wgsl(program)?;
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
            TensorData::new_for_shape(device, &[output_elems], output_datatype)
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
            pass.dispatch_workgroups(physical_workgroups, 1, 1);
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
