use crate::{
    Device,
    tensor::{DataTypeEnum, TensorData},
};

use super::GPU_SAMPLER_PREVIOUS_TOKENS;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ProcessorParams {
    temperature: f32,
    repetition_penalty: f32,
    previous_len: u32,
    _padding: u32,
}

pub(crate) fn fixed_previous_tokens_data(
    device: &Device,
    previous_tokens: &[u32],
) -> (TensorData, u32) {
    let len = previous_tokens.len().min(GPU_SAMPLER_PREVIOUS_TOKENS);
    let previous_tokens = &previous_tokens[previous_tokens.len().saturating_sub(len)..];
    let mut fixed = [0u32; GPU_SAMPLER_PREVIOUS_TOKENS];
    fixed[..len].copy_from_slice(previous_tokens);
    let buffer = device.create_buffer_init(
        bytemuck::cast_slice(&fixed),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    (
        TensorData::new_from_buffer(
            device,
            buffer,
            &[GPU_SAMPLER_PREVIOUS_TOKENS],
            DataTypeEnum::U32,
        ),
        len as u32,
    )
}

pub(crate) fn processor_params_data(
    device: &Device,
    temperature: f32,
    repetition_penalty: f32,
    previous_len: u32,
) -> TensorData {
    let params = ProcessorParams {
        temperature,
        repetition_penalty,
        previous_len,
        _padding: 0,
    };
    let buffer = device.create_buffer_init(
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    TensorData::new_from_buffer(device, buffer, &[1], DataTypeEnum::U32)
}
