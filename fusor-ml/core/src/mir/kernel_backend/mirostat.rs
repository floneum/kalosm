use fusor_tile_ir as tile_ir;

use crate::{
    Device,
    mir::{direct_kernel::DirectKernelBinding, kernel_backend},
    sampling::{
        GPU_SAMPLE_RESULT_WORDS, GpuMirostat2Sampler, GpuMirostat2SamplerParams, TOP_K_BLOCK,
    },
    tensor::{DataTypeEnum, TensorData},
};
use wgpu::CommandEncoder;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Mirostat2Params {
    tau: f32,
    eta: f32,
    random: f32,
    _padding: f32,
}

fn mirostat2_params_data(device: &Device, params: GpuMirostat2SamplerParams) -> TensorData {
    let params = Mirostat2Params {
        tau: params.tau,
        eta: params.eta,
        random: params.random.clamp(0.0, 0.999_999_94),
        _padding: 0.0,
    };
    let buffer = device.create_buffer_init(
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    TensorData::new_from_buffer(device, buffer, &[1], DataTypeEnum::U32)
}

pub(crate) fn sample_from_sorted_top_k_data_with_encoder(
    ids: &TensorData,
    values: &TensorData,
    sampler: &mut GpuMirostat2Sampler,
    params: GpuMirostat2SamplerParams,
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
    let params = mirostat2_params_data(device, params);
    let has_exactness_flag = exactness_flag.is_some();
    let output = TensorData::new_for_shape(device, &[GPU_SAMPLE_RESULT_WORDS], DataTypeEnum::U32);
    let meta = tile_ir::Mirostat2Meta {
        top_k: top_k.try_into().ok()?,
        ids_offset: ids.layout().offset().try_into().ok()?,
        ids_stride: ids.layout().strides()[0].try_into().ok()?,
        values_offset: values.layout().offset().try_into().ok()?,
        values_stride: values.layout().strides()[0].try_into().ok()?,
        has_exactness_flag,
    };
    let cache_key = format!(
        "sample_mirostat2_sorted_top_k_f32:backend-lowered:block={TOP_K_BLOCK}:top_k={top_k}:ids={:?}:values={:?}:exact={has_exactness_flag}",
        ids.layout(),
        values.layout()
    );
    let mut bindings = vec![
        DirectKernelBinding::Storage {
            binding: 0,
            buffer: ids.buffer().clone(),
            read_only: true,
        },
        DirectKernelBinding::Storage {
            binding: 1,
            buffer: values.buffer().clone(),
            read_only: true,
        },
        DirectKernelBinding::Storage {
            binding: 2,
            buffer: sampler.state.buffer().clone(),
            read_only: false,
        },
        DirectKernelBinding::Storage {
            binding: 3,
            buffer: params.buffer().clone(),
            read_only: true,
        },
        DirectKernelBinding::Storage {
            binding: 4,
            buffer: output.buffer().clone(),
            read_only: false,
        },
    ];
    if let Some(flag) = exactness_flag {
        bindings.push(DirectKernelBinding::Storage {
            binding: 5,
            buffer: flag.buffer().clone(),
            read_only: true,
        });
    }

    let kernel = kernel_backend::dynamic_kernel_from_ir(
        device,
        "sample_mirostat2_sorted_top_k_f32",
        cache_key,
        || tile_ir::kernels::mirostat2(meta),
        bindings,
        [1, 1, 1],
    )?;

    if let Some(encoder) = encoder {
        kernel.run(device, encoder);
    } else {
        let mut encoder =
            device
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sample_mirostat2_sorted_top_k_f32 encoder"),
                });
        kernel.run(device, &mut encoder);
        device.wgpu_queue().submit(Some(encoder.finish()));
    }

    Some(output)
}
