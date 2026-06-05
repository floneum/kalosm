use std::hash::Hash;

use fusor_tile_ir_kernels as tile_ir_kernels;

use crate::{
    Device,
    mir::kernel_backend,
    sampling::{GPU_SAMPLE_RESULT_WORDS, GpuStandardSamplerParams, TOP_K_BLOCK},
    tensor::{DataTypeEnum, TensorData},
};
use wgpu::CommandEncoder;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct StandardSamplerParams {
    random: f32,
    top_p: f32,
    min_p: f32,
    _padding: f32,
}

struct StandardSamplerSortedTopKKernelVariant;

fn normalized_probability(value: f32, default: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        default
    }
}

fn standard_sampler_params_data(device: &Device, params: GpuStandardSamplerParams) -> TensorData {
    let params = StandardSamplerParams {
        random: params.random.clamp(0.0, 0.999_999_94),
        top_p: normalized_probability(params.top_p, 1.0),
        min_p: normalized_probability(params.min_p, 0.0),
        _padding: 0.0,
    };
    let buffer = device.create_buffer_init(
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    TensorData::new_from_buffer(device, buffer, &[4], DataTypeEnum::F32)
}

pub(crate) fn sample_from_sorted_top_k_data_with_encoder(
    ids: &TensorData,
    values: &TensorData,
    params: GpuStandardSamplerParams,
    exactness_flag: Option<&TensorData>,
    encoder: Option<&mut CommandEncoder>,
) -> Option<TensorData> {
    if ids.datatype() != DataTypeEnum::U32 || values.datatype() != DataTypeEnum::F32 {
        return None;
    }
    if ids.layout().rank() != 1 || values.layout().rank() != 1 {
        return None;
    }
    if let Some(flag) = exactness_flag
        && (flag.datatype() != DataTypeEnum::U32
            || flag.layout().rank() != 1
            || flag.layout().shape()[0] == 0
            || !values.device().is_same_device(flag.device()))
    {
        return None;
    }

    let top_k = params
        .top_k
        .min(ids.layout().shape()[0])
        .min(values.layout().shape()[0]);
    if top_k == 0 {
        return None;
    }
    let device = values.device();
    let params = standard_sampler_params_data(device, params);
    let has_exactness_flag = exactness_flag.is_some();
    let output = TensorData::new_for_shape(device, &[GPU_SAMPLE_RESULT_WORDS], DataTypeEnum::U32);
    let meta = tile_ir_kernels::Mirostat2Meta {
        top_k: top_k.try_into().ok()?,
        ids_offset: ids.layout().offset().try_into().ok()?,
        ids_stride: ids.layout().strides()[0].try_into().ok()?,
        values_offset: values.layout().offset().try_into().ok()?,
        values_stride: values.layout().strides()[0].try_into().ok()?,
        has_exactness_flag,
    };
    let cache_key = kernel_backend::KernelCacheKey::from_hash_inputs(|state| {
        kernel_backend::KernelVariantKey::of::<StandardSamplerSortedTopKKernelVariant>()
            .hash(state);
        TOP_K_BLOCK.hash(state);
        top_k.hash(state);
        ids.layout().offset().hash(state);
        ids.layout().shape().hash(state);
        ids.layout().strides().hash(state);
        values.layout().offset().hash(state);
        values.layout().shape().hash(state);
        values.layout().strides().hash(state);
        has_exactness_flag.hash(state);
    });
    let kernel = kernel_backend::run_kernel(
        device.kernel_cache(),
        "sample_standard_sorted_top_k_f32",
        cache_key,
        [1, 1, 1],
        |kb| {
            tile_ir_kernels::standard_sampler(
                kb,
                tile_ir_kernels::StandardSampler {
                    ids: ids.as_kernel_tensor_ref(),
                    values: values.as_kernel_tensor_ref(),
                    params: params.as_kernel_tensor_ref(),
                    output: output.as_kernel_tensor_ref(),
                    exactness_flag: exactness_flag.map(|t| t.as_kernel_tensor_ref()),
                    meta,
                },
            )
        },
    )?;

    if let Some(encoder) = encoder {
        kernel.run(device.kernel_cache(), encoder);
    } else {
        let mut encoder =
            device
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sample_standard_sorted_top_k_f32 encoder"),
                });
        kernel.run(device.kernel_cache(), &mut encoder);
        device.wgpu_queue().submit(Some(encoder.finish()));
    }

    Some(output)
}
