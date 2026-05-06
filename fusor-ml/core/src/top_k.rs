use std::num::NonZeroU32;

use crate::{
    Device, Layout,
    mir::direct_kernel::{DirectKernel, DirectKernelBinding},
    mir::tile_direct::{
        flatten_matrix_layout, tile_storage_read_with_direct_layout,
        tile_storage_write_with_direct_layout,
    },
    quantized::QMatrix,
    tensor::{DataTypeEnum, TensorData},
};
use fusor_gguf::GgmlType;
use phase_token_prototype as tile_ir;
use wgpu::{
    CommandEncoder,
    naga::{
        AddressSpace, Arena, ArraySize, Barrier, BinaryOperator, Binding, Block, BuiltIn,
        EntryPoint, Expression, Function, FunctionArgument, GlobalVariable, Handle, Literal,
        LocalVariable, MathFunction, Module, Range, ResourceBinding, Scalar, ScalarKind,
        ShaderStage, Span, Statement, StorageAccess, Type, TypeInner, VectorSize,
    },
};

const TOP_K_BLOCK: u32 = 256;
pub(crate) const TOP_K_CHUNK: usize = TOP_K_BLOCK as usize;
pub(crate) const MIN_TOP_K_CANDIDATES_PER_CHUNK: usize = 64;
const MAX_F32: f32 = 3.4028234663852886e38;
const NEG_MAX_F32: f32 = -3.4028234663852886e38;
const GPU_SAMPLER_PREVIOUS_TOKENS: usize = 64;

#[derive(Clone, Copy, Debug)]
pub struct GpuMirostat2SamplerParams {
    pub top_k: usize,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub tau: f32,
    pub eta: f32,
    pub random: f32,
}

#[derive(Clone, Debug)]
pub struct GpuMirostat2Sampler {
    state: TensorData,
}

impl GpuMirostat2Sampler {
    pub fn new(device: &Device, mu: f32) -> Self {
        let state = TensorData::new_splat(device, &[1], mu);
        Self { state }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ProcessorParams {
    temperature: f32,
    repetition_penalty: f32,
    previous_len: u32,
    _padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Mirostat2Params {
    tau: f32,
    eta: f32,
    random: f32,
    _padding: f32,
}

pub(crate) async fn mirostat2_sample_token_to_host(
    input: &TensorData,
    sampler: &mut GpuMirostat2Sampler,
    previous_tokens: &[u32],
    params: GpuMirostat2SamplerParams,
) -> Result<Option<u32>, wgpu::BufferAsyncError> {
    sample_processed_logits_to_host(
        input,
        sampler,
        previous_tokens,
        params,
        None,
        "mirostat2 sampled token download",
    )
    .await
}

pub(crate) async fn qmat_mirostat2_sample_token_to_host(
    hidden: &TensorData,
    matrix: &QMatrix,
    sampler: &mut GpuMirostat2Sampler,
    previous_tokens: &[u32],
    params: GpuMirostat2SamplerParams,
) -> Result<Option<u32>, wgpu::BufferAsyncError> {
    if hidden.datatype() != DataTypeEnum::F32 || hidden.layout().rank() != 1 {
        return Ok(None);
    }
    let hidden_len = hidden.layout().shape()[0];
    let [vocab_len, hidden_matrix_len] = matrix.shape() else {
        return Ok(None);
    };
    if hidden_len != *hidden_matrix_len || *vocab_len == 0 {
        return Ok(None);
    }
    if !hidden.device().is_same_device(matrix.device()) {
        return Ok(None);
    }

    let device = hidden.device();
    let mut encoder =
        device
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("qmat_mirostat2_sample_token_to_host encoder"),
            });

    let Some(logits) = qmat_logits_data_with_encoder(hidden, matrix, &mut encoder) else {
        return Ok(None);
    };
    sample_processed_logits_to_host(
        &logits,
        sampler,
        previous_tokens,
        params,
        Some(encoder),
        "qmat mirostat2 sampled token download",
    )
    .await
}

async fn sample_processed_logits_to_host(
    input: &TensorData,
    sampler: &mut GpuMirostat2Sampler,
    previous_tokens: &[u32],
    params: GpuMirostat2SamplerParams,
    mut initial_encoder: Option<CommandEncoder>,
    download_label: &'static str,
) -> Result<Option<u32>, wgpu::BufferAsyncError> {
    if input.datatype() != DataTypeEnum::F32 || input.layout().rank() != 1 {
        return Ok(None);
    }

    let input_len = input.layout().shape()[0];
    let top_k = params.top_k.min(input_len);
    if top_k == 0 {
        return Ok(None);
    }

    let chunks = input_len.div_ceil(TOP_K_CHUNK);
    let mut candidate_count = initial_sampler_candidate_count(top_k, chunks);
    loop {
        let device = input.device();
        let mut encoder = initial_encoder.take().unwrap_or_else(|| {
            device
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("mirostat2_sample_token_to_host encoder"),
                })
        });

        let output_per_chunk = sampler_output_per_chunk(candidate_count);
        let Some((chunk_ids, chunk_values)) = chunk_top_k_pair_data_with_processors_with_encoder(
            input,
            previous_tokens,
            params.temperature,
            params.repetition_penalty,
            candidate_count,
            output_per_chunk,
            Some(&mut encoder),
        ) else {
            return Ok(None);
        };
        let Some((ids, values)) = merge_sorted_chunk_top_k_pair_data_with_encoder(
            &chunk_ids,
            &chunk_values,
            chunks,
            candidate_count,
            output_per_chunk,
            input_len,
            top_k,
            Some(&mut encoder),
        ) else {
            return Ok(None);
        };
        let exactness_flag = if candidate_count < top_k && candidate_count < TOP_K_CHUNK {
            let Some(flag) = top_k_exactness_flag_data_with_encoder(
                &values,
                &chunk_values,
                chunks,
                candidate_count,
                output_per_chunk,
                top_k,
                Some(&mut encoder),
            ) else {
                return Ok(None);
            };
            Some(flag)
        } else {
            None
        };
        let Some(output) = sample_from_sorted_top_k_data_with_encoder(
            &ids,
            &values,
            sampler,
            params,
            exactness_flag.as_ref(),
            Some(&mut encoder),
        ) else {
            return Ok(None);
        };

        let download = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
            size: std::mem::size_of::<u32>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
            label: Some(download_label),
        });
        encoder.copy_buffer_to_buffer(
            output.buffer(),
            0,
            &download,
            0,
            std::mem::size_of::<u32>() as u64,
        );
        device.wgpu_queue().submit(Some(encoder.finish()));

        let (sender, receiver) = futures_channel::oneshot::channel();
        download
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                _ = sender.send(result);
            });
        #[cfg(not(target_arch = "wasm32"))]
        device.poll_wait();
        receiver.await.map_err(|_| wgpu::BufferAsyncError)??;

        let view = download.slice(..).get_mapped_range();
        let token = view
            .get(..std::mem::size_of::<u32>())
            .map(bytemuck::from_bytes::<u32>)
            .copied()
            .unwrap_or_default();
        drop(view);
        download.unmap();

        if token != u32::MAX {
            return Ok(Some(token));
        }

        let next = next_sampler_candidate_count(candidate_count, top_k);
        if next == candidate_count {
            return Ok(None);
        }
        candidate_count = next;
    }
}

fn qmat_logits_data_with_encoder(
    hidden: &TensorData,
    matrix: &QMatrix,
    encoder: &mut CommandEncoder,
) -> Option<TensorData> {
    if hidden.datatype() != DataTypeEnum::F32 || hidden.layout().rank() != 1 {
        return None;
    }
    let hidden_len = hidden.layout().shape()[0];
    let hidden_stride = hidden.layout().strides()[0];
    let [vocab_len, matrix_hidden_len] = matrix.shape() else {
        return None;
    };
    if hidden_len != *matrix_hidden_len || *vocab_len == 0 {
        return None;
    }

    let device = hidden.device();
    let logits = TensorData::new_for_shape(device, &[*vocab_len], DataTypeEnum::F32);
    let hidden_2d = TensorData::new_from_parts(
        device,
        hidden.buffer().clone(),
        Layout::from_parts(
            hidden.layout().offset(),
            Box::new([1, hidden_len]),
            Box::new([0, hidden_stride]),
        ),
        DataTypeEnum::F32,
    );
    let logits_2d = TensorData::new_from_parts(
        device,
        logits.buffer().clone(),
        Layout::from_parts(0, Box::new([1, *vocab_len]), Box::new([*vocab_len, 1])),
        DataTypeEnum::F32,
    );
    let kernel = qmat_logits_direct_kernel(&hidden_2d, matrix, &logits_2d)?;
    kernel.run(device, encoder);

    Some(logits)
}

fn qmat_logits_direct_kernel(
    input: &TensorData,
    matrix: &QMatrix,
    output: &TensorData,
) -> Option<DirectKernel> {
    if input.datatype() != DataTypeEnum::F32 || output.datatype() != DataTypeEnum::F32 {
        return None;
    }
    let format = qmat_direct_quant_format(matrix)?;
    let a_view = flatten_matrix_layout(input.layout())?;
    let y_view = flatten_matrix_layout(output.layout())?;
    let m = a_view.rows;
    let k = a_view.cols;
    let y_m = y_view.rows;
    let n = y_view.cols;
    if m != 1 || y_m != 1 || k != matrix.shape()[1] as u32 || n != matrix.shape()[0] as u32 {
        return None;
    }

    let device = input.device();
    let limits = device.limits();
    let max_workgroups = limits.max_compute_workgroups_per_dimension;
    let qgemv_cols_per_workgroup = qgemv_cols_per_workgroup_for_direct(format, k, n);
    let qgemv_workgroups = n.div_ceil(qgemv_cols_per_workgroup);
    let [workgroups_x, _] = split_workgroups_2d(qgemv_workgroups, max_workgroups)?;
    let dispatch_size = [workgroups_x, qgemv_workgroups.div_ceil(workgroups_x), 1];
    if dispatch_size.iter().any(|dim| *dim > max_workgroups) {
        return None;
    }

    let cache_key = format!(
        "q_mat_logits_for_sampler:direct:{format:?}:m={m}:k={k}:n={n}:dispatch={dispatch_size:?}:{:?}:{:?}",
        input.layout(),
        output.layout()
    );
    let module = if let Some(module) = device.naga_module_cache().write().get(&cache_key) {
        module.clone()
    } else {
        let ir = tile_ir::tile::build(move |phase| {
            let a = tile_storage_read_with_direct_layout(phase, a_view);
            let b = phase.quantized_matrix(format, k, n);
            let y = tile_storage_write_with_direct_layout(phase, y_view);
            if format == tile_ir::GgmlQuantFormat::Q5_0 && k <= 1024 && n <= 4096 {
                phase.qgemv::<8, 32>(&a, &b, &y, 4, workgroups_x);
            } else {
                phase.qgemv::<4, 64>(&a, &b, &y, 4, workgroups_x);
            }
        });
        let module = ir.lower_to_naga().ok()?.module().clone();
        device
            .naga_module_cache()
            .write()
            .get_or_insert(cache_key.clone(), || module.clone())
            .clone()
    };

    Some(DirectKernel::new_with_cache_key(
        "q_mat_logits_for_sampler",
        cache_key,
        module,
        vec![
            DirectKernelBinding::Storage {
                binding: 0,
                buffer: input.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 1,
                buffer: matrix.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 2,
                buffer: output.buffer().clone(),
                read_only: false,
            },
        ],
        dispatch_size,
    ))
}

fn qmat_direct_quant_format(matrix: &QMatrix) -> Option<tile_ir::GgmlQuantFormat> {
    Some(match matrix.datatype() {
        GgmlType::Q4_0 => tile_ir::GgmlQuantFormat::Q4_0,
        GgmlType::Q4_1 => tile_ir::GgmlQuantFormat::Q4_1,
        GgmlType::Q5_0 => tile_ir::GgmlQuantFormat::Q5_0,
        GgmlType::Q5_1 => tile_ir::GgmlQuantFormat::Q5_1,
        GgmlType::Q8_0 => tile_ir::GgmlQuantFormat::Q8_0,
        GgmlType::Q8_1 => tile_ir::GgmlQuantFormat::Q8_1,
        GgmlType::Q2K => tile_ir::GgmlQuantFormat::Q2K,
        GgmlType::Q3K => tile_ir::GgmlQuantFormat::Q3K,
        GgmlType::Q4K => tile_ir::GgmlQuantFormat::Q4K,
        GgmlType::Q5K => tile_ir::GgmlQuantFormat::Q5K,
        GgmlType::Q6K => tile_ir::GgmlQuantFormat::Q6K,
        GgmlType::Q8K => tile_ir::GgmlQuantFormat::Q8K,
        GgmlType::F16 | GgmlType::F32 => return None,
    })
}

fn ceil_div_u32(x: u32, divisor: u32) -> u32 {
    x.div_ceil(divisor)
}

fn split_workgroups_2d(
    total_workgroups: u32,
    max_workgroups_per_dimension: u32,
) -> Option<[u32; 2]> {
    if total_workgroups == 0 {
        return Some([1, 1]);
    }

    let max_workgroups_per_dimension = max_workgroups_per_dimension.max(1);
    let x = total_workgroups.min(max_workgroups_per_dimension);
    let y = ceil_div_u32(total_workgroups, x);
    (y <= max_workgroups_per_dimension).then_some([x, y])
}

fn initial_sampler_candidate_count(top_k: usize, chunks: usize) -> usize {
    top_k
        .div_ceil(chunks)
        .max(MIN_TOP_K_CANDIDATES_PER_CHUNK)
        .min(top_k)
        .min(TOP_K_CHUNK)
}

fn sampler_output_per_chunk(candidate_count: usize) -> usize {
    if candidate_count >= TOP_K_CHUNK {
        TOP_K_CHUNK
    } else {
        candidate_count + 1
    }
}

fn next_sampler_candidate_count(candidate_count: usize, top_k: usize) -> usize {
    candidate_count
        .saturating_mul(2)
        .min(top_k)
        .min(TOP_K_CHUNK)
}

fn qgemv_cols_per_workgroup_for_direct(format: tile_ir::GgmlQuantFormat, k: u32, n: u32) -> u32 {
    if format == tile_ir::GgmlQuantFormat::Q4K && k <= 4096 && n >= 4096 && n < 8192 {
        return 4;
    }

    if format == tile_ir::GgmlQuantFormat::Q4K && k <= 4096 && n >= 8192 {
        return 8;
    }

    if format == tile_ir::GgmlQuantFormat::Q4K && k > 4096 && n <= 4096 {
        return 8;
    }

    if format == tile_ir::GgmlQuantFormat::Q6K && k <= 4096 && n >= 8192 {
        return 8;
    }

    if format == tile_ir::GgmlQuantFormat::Q6K && k > 4096 && n <= 4096 {
        return 4;
    }

    let qgemv_uses_accelerator = format == tile_ir::GgmlQuantFormat::Q4K
        || format == tile_ir::GgmlQuantFormat::Q6K
        || (format == tile_ir::GgmlQuantFormat::Q5_0
            && k.checked_mul(n)
                .is_some_and(|elements| elements >= 4 * 1024 * 1024))
        || (format == tile_ir::GgmlQuantFormat::Q8_0 && k <= 1024 && n >= 8192);

    if qgemv_uses_accelerator {
        if format == tile_ir::GgmlQuantFormat::Q8_0 && k <= 1024 && n >= 8192 {
            4 * 8
        } else {
            format.qgemv_cols_per_workgroup_for_shape(k, n)
        }
    } else if format == tile_ir::GgmlQuantFormat::Q5_0 && k <= 1024 && n <= 4096 {
        8
    } else {
        4
    }
}

fn fixed_previous_tokens_data(device: &Device, previous_tokens: &[u32]) -> (TensorData, u32) {
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

fn processor_params_data(
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

fn sample_from_sorted_top_k_data_with_encoder(
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
    let ids_offset = ids.layout().offset();
    let ids_stride = ids.layout().strides()[0];
    let values_offset = values.layout().offset();
    let values_stride = values.layout().strides()[0];
    let device = values.device();
    let params = mirostat2_params_data(device, params);
    let output = TensorData::new_for_shape(device, &[1], DataTypeEnum::U32);
    let has_exactness_flag = exactness_flag.is_some();
    let exactness_binding = if has_exactness_flag {
        "@group(0) @binding(5) var<storage, read> exactness_flag: array<u32>;"
    } else {
        ""
    };
    let exactness_check = if has_exactness_flag {
        r#"
    if (exactness_flag[0] == 0u) {
        output[0] = 4294967295u;
        return;
    }
"#
    } else {
        ""
    };
    let cache_key = format!(
        "sample_mirostat2_sorted_top_k_f32:block={TOP_K_BLOCK}:top_k={top_k}:ids={:?}:values={:?}:exact={has_exactness_flag}",
        ids.layout(),
        values.layout()
    );
    let reduce_start = TOP_K_BLOCK / 2;
    let source = format!(
        r#"
struct Mirostat2Params {{
    tau: f32,
    eta: f32,
    random: f32,
    _padding: f32,
}};

@group(0) @binding(0) var<storage, read> ids: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<f32>;
@group(0) @binding(2) var<storage, read_write> state: array<f32>;
@group(0) @binding(3) var<storage, read> params: Mirostat2Params;
@group(0) @binding(4) var<storage, read_write> output: array<u32>;
{exactness_binding}

var<workgroup> scratch: array<f32, {TOP_K_BLOCK}>;

fn top_value(index: u32) -> f32 {{
    return values[{values_offset}u + index * {values_stride}u];
}}

fn top_id(index: u32) -> u32 {{
    return ids[{ids_offset}u + index * {ids_stride}u];
}}

@compute @workgroup_size({TOP_K_BLOCK})
fn main(@builtin(local_invocation_index) lane: u32) {{
{exactness_check}
    let max_value = top_value(0u);
    var local_sum = 0.0;
    var index = lane;
    loop {{
        if (index >= {top_k}u) {{
            break;
        }}
        local_sum = local_sum + exp(top_value(index) - max_value);
        index = index + {TOP_K_BLOCK}u;
    }}
    scratch[lane] = local_sum;
    workgroupBarrier();

    var reduce_step = {reduce_start}u;
    loop {{
        if (reduce_step == 0u) {{
            break;
        }}
        if (lane < reduce_step) {{
            scratch[lane] = scratch[lane] + scratch[lane + reduce_step];
        }}
        workgroupBarrier();
        reduce_step = reduce_step / 2u;
    }}

    if (lane != 0u) {{
        return;
    }}

    let total = max(scratch[0], 1.0e-20);
    let mu = state[0];
    var cutoff = 0u;
    var scan = 0u;
    loop {{
        if (scan >= {top_k}u) {{
            cutoff = 1u;
            break;
        }}
        let probability = exp(top_value(scan) - max_value) / total;
        if (-log2(max(probability, 1.0e-20)) > mu) {{
            cutoff = max(scan, 1u);
            break;
        }}
        scan = scan + 1u;
    }}

    var cutoff_sum = 0.0;
    scan = 0u;
    loop {{
        if (scan >= cutoff) {{
            break;
        }}
        cutoff_sum = cutoff_sum + exp(top_value(scan) - max_value);
        scan = scan + 1u;
    }}
    cutoff_sum = max(cutoff_sum, 1.0e-20);

    let threshold = params.random * cutoff_sum;
    var cumulative = 0.0;
    var selected = top_id(0u);
    var selected_probability = exp(top_value(0u) - max_value) / cutoff_sum;
    scan = 0u;
    loop {{
        if (scan >= cutoff) {{
            break;
        }}
        let weight = exp(top_value(scan) - max_value);
        cumulative = cumulative + weight;
        if (cumulative >= threshold) {{
            selected = top_id(scan);
            selected_probability = weight / cutoff_sum;
            break;
        }}
        scan = scan + 1u;
    }}

    state[0] = mu - params.eta * (-log2(max(selected_probability, 1.0e-20)) - params.tau);
    output[0] = selected;
}}
"#
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

    let kernel = DirectKernel::new_wgsl_with_cache_key(
        "sample_mirostat2_sorted_top_k_f32",
        cache_key,
        source,
        bindings,
        [1, 1, 1],
    );

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

#[derive(Clone, Copy)]
struct TopKExactnessMeta {
    chunks: u32,
    candidate_count: u32,
    output_per_chunk: u32,
    top_k: u32,
    top_values_offset: u32,
    top_values_stride: u32,
    chunk_values_offset: u32,
    chunk_values_stride: u32,
}

fn top_k_exactness_flag_data_with_encoder(
    top_values: &TensorData,
    chunk_values: &TensorData,
    chunks: usize,
    candidate_count: usize,
    output_per_chunk: usize,
    top_k: usize,
    encoder: Option<&mut CommandEncoder>,
) -> Option<TensorData> {
    if top_values.datatype() != DataTypeEnum::F32
        || chunk_values.datatype() != DataTypeEnum::F32
        || top_values.layout().rank() != 1
        || chunk_values.layout().rank() != 1
        || !top_values.device().is_same_device(chunk_values.device())
        || top_k == 0
        || top_values.layout().shape()[0] < top_k
        || candidate_count >= output_per_chunk
    {
        return None;
    }

    let device = top_values.device();
    let flag = TensorData::new_for_shape(device, &[1], DataTypeEnum::U32);
    let meta = TopKExactnessMeta {
        chunks: chunks.try_into().ok()?,
        candidate_count: candidate_count.try_into().ok()?,
        output_per_chunk: output_per_chunk.try_into().ok()?,
        top_k: top_k.try_into().ok()?,
        top_values_offset: top_values.layout().offset().try_into().ok()?,
        top_values_stride: top_values.layout().strides()[0].try_into().ok()?,
        chunk_values_offset: chunk_values.layout().offset().try_into().ok()?,
        chunk_values_stride: chunk_values.layout().strides()[0].try_into().ok()?,
    };
    let cache_key = format!(
        "prove_top_k_exact_f32:block={TOP_K_BLOCK}:chunks={chunks}:candidate_count={candidate_count}:output_per_chunk={output_per_chunk}:top_k={top_k}:top={:?}:chunk={:?}",
        top_values.layout(),
        chunk_values.layout()
    );
    let module = if let Some(module) = device.naga_module_cache().write().get(&cache_key) {
        module.clone()
    } else {
        let module = TopKExactnessModuleBuilder::new(meta).build()?;
        device
            .naga_module_cache()
            .write()
            .get_or_insert(cache_key.clone(), || module.clone())
            .clone()
    };
    let kernel = DirectKernel::new_with_cache_key(
        "prove_top_k_exact_f32",
        cache_key,
        module,
        vec![
            DirectKernelBinding::Storage {
                binding: 0,
                buffer: top_values.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 1,
                buffer: chunk_values.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 2,
                buffer: flag.buffer().clone(),
                read_only: false,
            },
        ],
        [1, 1, 1],
    );

    if let Some(encoder) = encoder {
        kernel.run(device, encoder);
    } else {
        let mut encoder =
            device
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("prove_top_k_exact_f32 encoder"),
                });
        kernel.run(device, &mut encoder);
        device.wgpu_queue().submit(Some(encoder.finish()));
    }

    Some(flag)
}

pub(crate) fn chunk_top_k_pair_data(
    input: &TensorData,
    candidate_count: usize,
    output_per_chunk: usize,
) -> Option<(TensorData, TensorData)> {
    chunk_top_k_pair_data_with_encoder(input, candidate_count, output_per_chunk, None)
}

fn chunk_top_k_pair_data_with_encoder(
    input: &TensorData,
    candidate_count: usize,
    output_per_chunk: usize,
    encoder: Option<&mut CommandEncoder>,
) -> Option<(TensorData, TensorData)> {
    chunk_top_k_pair_data_inner_with_encoder(
        input,
        candidate_count,
        output_per_chunk,
        None,
        encoder,
    )
}

fn chunk_top_k_pair_data_with_processors_with_encoder(
    input: &TensorData,
    previous_tokens: &[u32],
    temperature: f32,
    repetition_penalty: f32,
    candidate_count: usize,
    output_per_chunk: usize,
    encoder: Option<&mut CommandEncoder>,
) -> Option<(TensorData, TensorData)> {
    let device = input.device();
    let (previous_tokens, previous_len) = fixed_previous_tokens_data(device, previous_tokens);
    let params = processor_params_data(device, temperature, repetition_penalty, previous_len);
    chunk_top_k_pair_data_inner_with_encoder(
        input,
        candidate_count,
        output_per_chunk,
        Some((&previous_tokens, &params)),
        encoder,
    )
}

fn chunk_top_k_pair_data_inner_with_encoder(
    input: &TensorData,
    candidate_count: usize,
    output_per_chunk: usize,
    processors: Option<(&TensorData, &TensorData)>,
    encoder: Option<&mut CommandEncoder>,
) -> Option<(TensorData, TensorData)> {
    if input.datatype() != DataTypeEnum::F32 || input.layout().rank() != 1 {
        return None;
    }

    let input_len = input.layout().shape()[0];
    let chunks = input_len.div_ceil(TOP_K_CHUNK);
    let output_len = chunks.checked_mul(output_per_chunk)?;
    let device = input.device();
    let ids = TensorData::new_for_shape(device, &[output_len], DataTypeEnum::U32);
    let values = TensorData::new_for_shape(device, &[output_len], DataTypeEnum::F32);
    if candidate_count == 0 || output_per_chunk == 0 || input_len == 0 {
        return Some((ids, values));
    }

    let input_offset = input.layout().offset();
    let input_stride = input.layout().strides()[0];
    let has_processors = processors.is_some();
    let cache_key = format!(
        "chunk_top_k_pairs_f32:block={TOP_K_BLOCK}:chunk={TOP_K_CHUNK}:len={input_len}:candidate_count={candidate_count}:output_per_chunk={output_per_chunk}:offset={input_offset}:stride={input_stride}:processors={has_processors}"
    );
    let module = if let Some(module) = device.naga_module_cache().write().get(&cache_key) {
        module.clone()
    } else {
        let module = TopKModuleBuilder::new(
            input_len.try_into().ok()?,
            output_per_chunk.try_into().ok()?,
            input_offset.try_into().ok()?,
            input_stride.try_into().ok()?,
            has_processors,
        )
        .build()?;
        device
            .naga_module_cache()
            .write()
            .get_or_insert(cache_key.clone(), || module.clone())
            .clone()
    };

    let kernel = if let Some((previous_tokens, params)) = processors {
        let bindings = vec![
            DirectKernelBinding::Storage {
                binding: 0,
                buffer: input.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 1,
                buffer: ids.buffer().clone(),
                read_only: false,
            },
            DirectKernelBinding::Storage {
                binding: 2,
                buffer: values.buffer().clone(),
                read_only: false,
            },
            DirectKernelBinding::Storage {
                binding: 3,
                buffer: previous_tokens.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 4,
                buffer: params.buffer().clone(),
                read_only: true,
            },
        ];
        DirectKernel::new_with_cache_key(
            "chunk_top_k_pairs_f32",
            cache_key,
            module,
            bindings,
            [chunks.try_into().ok()?, 1, 1],
        )
    } else {
        DirectKernel::new_with_cache_key(
            "chunk_top_k_pairs_f32",
            cache_key,
            module,
            vec![
                DirectKernelBinding::Storage {
                    binding: 0,
                    buffer: input.buffer().clone(),
                    read_only: true,
                },
                DirectKernelBinding::Storage {
                    binding: 1,
                    buffer: ids.buffer().clone(),
                    read_only: false,
                },
                DirectKernelBinding::Storage {
                    binding: 2,
                    buffer: values.buffer().clone(),
                    read_only: false,
                },
            ],
            [chunks.try_into().ok()?, 1, 1],
        )
    };

    if let Some(encoder) = encoder {
        kernel.run(device, encoder);
    } else {
        let mut encoder =
            device
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("chunk_top_k_pairs_f32 encoder"),
                });
        kernel.run(device, &mut encoder);
        device.wgpu_queue().submit(Some(encoder.finish()));
    }

    Some((ids, values))
}

pub(crate) fn merge_sorted_chunk_top_k_pair_data(
    input_ids: &TensorData,
    input_values: &TensorData,
    chunks: usize,
    chunk_len: usize,
    chunk_stride: usize,
    input_len: usize,
    k: usize,
) -> Option<(TensorData, TensorData)> {
    merge_sorted_chunk_top_k_pair_data_with_encoder(
        input_ids,
        input_values,
        chunks,
        chunk_len,
        chunk_stride,
        input_len,
        k,
        None,
    )
}

fn merge_sorted_chunk_top_k_pair_data_with_encoder(
    input_ids: &TensorData,
    input_values: &TensorData,
    chunks: usize,
    chunk_len: usize,
    chunk_stride: usize,
    input_len: usize,
    k: usize,
    encoder: Option<&mut CommandEncoder>,
) -> Option<(TensorData, TensorData)> {
    if input_ids.datatype() != DataTypeEnum::U32 || input_values.datatype() != DataTypeEnum::F32 {
        return None;
    }
    if input_ids.layout().rank() != 1 || input_values.layout().rank() != 1 {
        return None;
    }
    let input_ids_len = input_ids.layout().shape()[0];
    let input_values_len = input_values.layout().shape()[0];
    let expected_len = if chunks == 0 {
        0
    } else {
        (chunks - 1)
            .checked_mul(chunk_stride)?
            .checked_add(chunk_len)?
    };
    if input_ids_len < expected_len || input_values_len < expected_len {
        return None;
    }

    let device = input_values.device();
    let output_len = k.min(input_len);
    let ids = TensorData::new_for_shape(device, &[output_len], DataTypeEnum::U32);
    let values = TensorData::new_for_shape(device, &[output_len], DataTypeEnum::F32);
    if chunks == 0 || chunk_len == 0 || output_len == 0 {
        return Some((ids, values));
    }

    let cache_key = format!(
        "merge_sorted_chunk_top_k_pairs_f32:block={TOP_K_BLOCK}:chunks={chunks}:chunk_len={chunk_len}:chunk_stride={chunk_stride}:input_len={input_len}:k={output_len}:ids={:?}:values={:?}",
        input_ids.layout(),
        input_values.layout()
    );
    let module = if let Some(module) = device.naga_module_cache().write().get(&cache_key) {
        module.clone()
    } else {
        let module = MergeTopKModuleBuilder::new(
            chunks.try_into().ok()?,
            chunk_len.try_into().ok()?,
            chunk_stride.try_into().ok()?,
            input_len.try_into().ok()?,
            output_len.try_into().ok()?,
        )
        .build()?;
        device
            .naga_module_cache()
            .write()
            .get_or_insert(cache_key.clone(), || module.clone())
            .clone()
    };

    let kernel = DirectKernel::new_with_cache_key(
        "merge_sorted_chunk_top_k_pairs_f32",
        cache_key,
        module,
        vec![
            DirectKernelBinding::Storage {
                binding: 0,
                buffer: input_ids.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 1,
                buffer: input_values.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 2,
                buffer: ids.buffer().clone(),
                read_only: false,
            },
            DirectKernelBinding::Storage {
                binding: 3,
                buffer: values.buffer().clone(),
                read_only: false,
            },
        ],
        [1, 1, 1],
    );

    if let Some(encoder) = encoder {
        kernel.run(device, encoder);
    } else {
        let mut encoder =
            device
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("merge_sorted_chunk_top_k_pairs_f32 encoder"),
                });
        kernel.run(device, &mut encoder);
        device.wgpu_queue().submit(Some(encoder.finish()));
    }

    Some((ids, values))
}

struct TopKModuleBuilder {
    input_len: u32,
    output_per_chunk: u32,
    input_offset: u32,
    input_stride: u32,
    processors: bool,
}

struct TopKGlobals {
    input: Handle<GlobalVariable>,
    output_ids: Handle<GlobalVariable>,
    output_values: Handle<GlobalVariable>,
    previous_tokens: Option<Handle<GlobalVariable>>,
    processor_params: Option<Handle<GlobalVariable>>,
    scratch_values: Handle<GlobalVariable>,
    scratch_ids: Handle<GlobalVariable>,
}

struct TopKLocals {
    current_value: Handle<LocalVariable>,
    current_id: Handle<LocalVariable>,
    previous_index: Handle<LocalVariable>,
    repeated: Handle<LocalVariable>,
}

struct TopKExactnessModuleBuilder {
    meta: TopKExactnessMeta,
}

#[derive(Clone, Copy)]
struct TopKExactnessGlobals {
    top_values: Handle<GlobalVariable>,
    chunk_values: Handle<GlobalVariable>,
    flag: Handle<GlobalVariable>,
    scratch: Handle<GlobalVariable>,
}

#[derive(Clone, Copy)]
struct TopKExactnessLocals {
    chunk: Handle<LocalVariable>,
    inexact: Handle<LocalVariable>,
}

struct MergeTopKModuleBuilder {
    chunks: u32,
    chunk_len: u32,
    chunk_stride: u32,
    input_len: u32,
    k: u32,
}

struct MergeTopKGlobals {
    input_ids: Handle<GlobalVariable>,
    input_values: Handle<GlobalVariable>,
    output_ids: Handle<GlobalVariable>,
    output_values: Handle<GlobalVariable>,
    chunk_positions: Handle<GlobalVariable>,
    scratch_values: Handle<GlobalVariable>,
    scratch_ids: Handle<GlobalVariable>,
    scratch_chunks: Handle<GlobalVariable>,
}

struct MergeTopKLocals {
    rank: Handle<LocalVariable>,
    scan_chunk: Handle<LocalVariable>,
    local_best_value: Handle<LocalVariable>,
    local_best_id: Handle<LocalVariable>,
    local_best_chunk: Handle<LocalVariable>,
    reduce_step: Handle<LocalVariable>,
}

impl TopKExactnessModuleBuilder {
    fn new(meta: TopKExactnessMeta) -> Self {
        Self { meta }
    }

    fn build(self) -> Option<Module> {
        let mut module = Module::default();
        let f32_ty = module.types.insert(
            Type {
                name: Some("TopKExactF32".into()),
                inner: TypeInner::Scalar(Scalar::F32),
            },
            Span::default(),
        );
        let u32_ty = module.types.insert(
            Type {
                name: Some("TopKExactU32".into()),
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::default(),
        );
        let f32_storage_ty = module.types.insert(
            Type {
                name: Some("TopKExactF32Buffer".into()),
                inner: TypeInner::Array {
                    base: f32_ty,
                    size: ArraySize::Dynamic,
                    stride: 4,
                },
            },
            Span::default(),
        );
        let u32_storage_ty = module.types.insert(
            Type {
                name: Some("TopKExactU32Buffer".into()),
                inner: TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Dynamic,
                    stride: 4,
                },
            },
            Span::default(),
        );
        let scratch_ty = module.types.insert(
            Type {
                name: Some("TopKExactScratch".into()),
                inner: TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Constant(NonZeroU32::new(TOP_K_BLOCK)?),
                    stride: 4,
                },
            },
            Span::default(),
        );

        let globals = TopKExactnessGlobals {
            top_values: Self::storage_global(&mut module, "top_values", 0, f32_storage_ty, true),
            chunk_values: Self::storage_global(
                &mut module,
                "chunk_values",
                1,
                f32_storage_ty,
                true,
            ),
            flag: Self::storage_global(&mut module, "flag", 2, u32_storage_ty, false),
            scratch: Self::workgroup_global(&mut module, "scratch", scratch_ty),
        };

        let mut function = Function {
            name: Some("main".into()),
            arguments: vec![FunctionArgument {
                name: Some("local_invocation_index".into()),
                ty: u32_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)),
            }],
            ..Function::default()
        };
        let locals = TopKExactnessLocals {
            chunk: Self::local(&mut function, "chunk", u32_ty),
            inexact: Self::local(&mut function, "inexact", u32_ty),
        };
        function.body = self.entry_body(&mut function.expressions, globals, locals);
        function
            .body
            .push(Statement::Return { value: None }, Span::default());

        module.entry_points.push(EntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [TOP_K_BLOCK, 1, 1],
            workgroup_size_overrides: None,
            function,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });

        Some(module)
    }

    fn entry_body(
        &self,
        expressions: &mut Arena<Expression>,
        globals: TopKExactnessGlobals,
        locals: TopKExactnessLocals,
    ) -> Block {
        let mut body = Block::new();
        let lane = expressions.append(Expression::FunctionArgument(0), Span::default());
        let threshold_rank = self.u32_lit(expressions, self.meta.top_k - 1);
        let threshold_index = self.index1(
            expressions,
            &mut body,
            self.meta.top_values_offset,
            self.meta.top_values_stride,
            threshold_rank,
        );
        let threshold =
            self.load_storage(expressions, &mut body, globals.top_values, threshold_index);
        let threshold_finite = self.is_finite(expressions, &mut body, threshold);

        let zero = self.u32_lit(expressions, 0);
        self.store_local(expressions, &mut body, locals.inexact, zero);
        self.store_local(expressions, &mut body, locals.chunk, lane);
        self.append_scan_loop(
            expressions,
            &mut body,
            globals,
            locals,
            threshold,
            threshold_finite,
        );

        let inexact = self.load_local(expressions, &mut body, locals.inexact);
        self.store_workgroup(expressions, &mut body, globals.scratch, lane, inexact);
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let mut stride = TOP_K_BLOCK / 2;
        while stride > 0 {
            let participates = self.lt_lit(expressions, &mut body, lane, stride);
            let mut accept = Block::new();
            let rhs_index = self.add_lit(expressions, &mut accept, lane, stride);
            let lhs = self.load_workgroup(expressions, &mut accept, globals.scratch, lane);
            let rhs = self.load_workgroup(expressions, &mut accept, globals.scratch, rhs_index);
            let merged = self.bin(
                expressions,
                &mut accept,
                BinaryOperator::InclusiveOr,
                lhs,
                rhs,
            );
            self.store_workgroup(expressions, &mut accept, globals.scratch, lane, merged);
            body.push(
                Statement::If {
                    condition: participates,
                    accept,
                    reject: Block::new(),
                },
                Span::default(),
            );
            body.push(
                Statement::ControlBarrier(Barrier::WORK_GROUP),
                Span::default(),
            );
            stride /= 2;
        }

        let lane_zero = self.bin(expressions, &mut body, BinaryOperator::Equal, lane, zero);
        let mut store_accept = Block::new();
        let root = self.load_workgroup(expressions, &mut store_accept, globals.scratch, zero);
        let exact = self.bin(
            expressions,
            &mut store_accept,
            BinaryOperator::Equal,
            root,
            zero,
        );
        let mut exact_accept = Block::new();
        let one = self.u32_lit(expressions, 1);
        self.store_storage(expressions, &mut exact_accept, globals.flag, zero, one);
        let mut exact_reject = Block::new();
        self.store_storage(expressions, &mut exact_reject, globals.flag, zero, zero);
        store_accept.push(
            Statement::If {
                condition: exact,
                accept: exact_accept,
                reject: exact_reject,
            },
            Span::default(),
        );
        body.push(
            Statement::If {
                condition: lane_zero,
                accept: store_accept,
                reject: Block::new(),
            },
            Span::default(),
        );

        body
    }

    fn append_scan_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: TopKExactnessGlobals,
        locals: TopKExactnessLocals,
        threshold: Handle<Expression>,
        threshold_finite: Handle<Expression>,
    ) {
        let mut loop_body = Block::new();
        let chunk = self.load_local(expressions, &mut loop_body, locals.chunk);
        let chunks = self.u32_lit(expressions, self.meta.chunks);
        let done = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::GreaterEqual,
            chunk,
            chunks,
        );
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let bound_rank = self.mul_lit(
            expressions,
            &mut loop_body,
            chunk,
            self.meta.output_per_chunk,
        );
        let bound_rank = self.add_lit(
            expressions,
            &mut loop_body,
            bound_rank,
            self.meta.candidate_count,
        );
        let bound_index = self.index1(
            expressions,
            &mut loop_body,
            self.meta.chunk_values_offset,
            self.meta.chunk_values_stride,
            bound_rank,
        );
        let bound = self.load_storage(
            expressions,
            &mut loop_body,
            globals.chunk_values,
            bound_index,
        );
        let bound_finite = self.is_finite(expressions, &mut loop_body, bound);
        let bound_ge_threshold = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::GreaterEqual,
            bound,
            threshold,
        );
        let finite_bound_inexact = self.and(
            expressions,
            &mut loop_body,
            bound_finite,
            bound_ge_threshold,
        );
        let finite_inexact = self.and(
            expressions,
            &mut loop_body,
            threshold_finite,
            finite_bound_inexact,
        );
        let false_lit = self.bool_lit(expressions, false);
        let threshold_not_finite = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Equal,
            threshold_finite,
            false_lit,
        );
        let nonfinite_inexact = self.and(
            expressions,
            &mut loop_body,
            threshold_not_finite,
            bound_finite,
        );
        let inexact = self.or(
            expressions,
            &mut loop_body,
            finite_inexact,
            nonfinite_inexact,
        );
        let mut inexact_accept = Block::new();
        let one = self.u32_lit(expressions, 1);
        self.store_local(expressions, &mut inexact_accept, locals.inexact, one);
        loop_body.push(
            Statement::If {
                condition: inexact,
                accept: inexact_accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        let next_chunk = self.add_lit(expressions, &mut loop_body, chunk, TOP_K_BLOCK);
        self.store_local(expressions, &mut loop_body, locals.chunk, next_chunk);
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }

    fn storage_global(
        module: &mut Module,
        name: &str,
        binding: u32,
        ty: Handle<Type>,
        read_only: bool,
    ) -> Handle<GlobalVariable> {
        module.global_variables.append(
            GlobalVariable {
                name: Some(name.into()),
                space: AddressSpace::Storage {
                    access: if read_only {
                        StorageAccess::LOAD
                    } else {
                        StorageAccess::LOAD | StorageAccess::STORE
                    },
                },
                binding: Some(ResourceBinding { group: 0, binding }),
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    fn workgroup_global(
        module: &mut Module,
        name: &str,
        ty: Handle<Type>,
    ) -> Handle<GlobalVariable> {
        module.global_variables.append(
            GlobalVariable {
                name: Some(name.into()),
                space: AddressSpace::WorkGroup,
                binding: None,
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    fn local(function: &mut Function, name: &str, ty: Handle<Type>) -> Handle<LocalVariable> {
        function.local_variables.append(
            LocalVariable {
                name: Some(name.into()),
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    fn index1(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        offset: u32,
        stride: u32,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let scaled = if stride == 1 {
            index
        } else {
            self.mul_lit(expressions, body, index, stride)
        };
        self.add_lit(expressions, body, scaled, offset)
    }

    fn is_finite(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
    ) -> Handle<Expression> {
        let self_equal = self.bin(expressions, body, BinaryOperator::Equal, value, value);
        let abs = self.emit(
            expressions,
            body,
            Expression::Math {
                fun: MathFunction::Abs,
                arg: value,
                arg1: None,
                arg2: None,
                arg3: None,
            },
        );
        let max = self.f32_lit(expressions, MAX_F32);
        let finite_magnitude = self.bin(expressions, body, BinaryOperator::LessEqual, abs, max);
        self.and(expressions, body, self_equal, finite_magnitude)
    }

    fn load_storage(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let ptr = self.ptr(expressions, body, Expression::GlobalVariable(global), index);
        self.emit(expressions, body, Expression::Load { pointer: ptr })
    }

    fn store_storage(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
        value: Handle<Expression>,
    ) {
        let pointer = self.ptr(expressions, body, Expression::GlobalVariable(global), index);
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    fn load_workgroup(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let ptr = self.ptr(expressions, body, Expression::GlobalVariable(global), index);
        self.emit(expressions, body, Expression::Load { pointer: ptr })
    }

    fn store_workgroup(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
        value: Handle<Expression>,
    ) {
        let pointer = self.ptr(expressions, body, Expression::GlobalVariable(global), index);
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    fn ptr(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        base: Expression,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let base = expressions.append(base, Span::default());
        self.emit(expressions, body, Expression::Access { base, index })
    }

    fn load_local(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        local: Handle<LocalVariable>,
    ) -> Handle<Expression> {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        self.emit(expressions, body, Expression::Load { pointer })
    }

    fn store_local(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        local: Handle<LocalVariable>,
        value: Handle<Expression>,
    ) {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    fn lt_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Less, value, rhs)
    }

    fn add_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        if literal == 0 {
            value
        } else {
            let rhs = self.u32_lit(expressions, literal);
            self.bin(expressions, body, BinaryOperator::Add, value, rhs)
        }
    }

    fn mul_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Multiply, value, rhs)
    }

    fn and(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(expressions, body, BinaryOperator::LogicalAnd, left, right)
    }

    fn or(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(expressions, body, BinaryOperator::LogicalOr, left, right)
    }

    fn bin(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(expressions, body, Expression::Binary { op, left, right })
    }

    fn emit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        expression: Expression,
    ) -> Handle<Expression> {
        let handle = expressions.append(expression, Span::default());
        body.push(
            Statement::Emit(Range::new_from_bounds(handle, handle)),
            Span::default(),
        );
        handle
    }

    fn f32_lit(&self, expressions: &mut Arena<Expression>, value: f32) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::F32(value)), Span::default())
    }

    fn u32_lit(&self, expressions: &mut Arena<Expression>, value: u32) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::U32(value)), Span::default())
    }

    fn bool_lit(&self, expressions: &mut Arena<Expression>, value: bool) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::Bool(value)), Span::default())
    }
}

impl MergeTopKModuleBuilder {
    fn new(chunks: u32, chunk_len: u32, chunk_stride: u32, input_len: u32, k: u32) -> Self {
        Self {
            chunks,
            chunk_len,
            chunk_stride,
            input_len,
            k,
        }
    }

    fn build(self) -> Option<Module> {
        let mut module = Module::default();
        let f32_ty = module.types.insert(
            Type {
                name: Some("MergeTopKF32".into()),
                inner: TypeInner::Scalar(Scalar::F32),
            },
            Span::default(),
        );
        let u32_ty = module.types.insert(
            Type {
                name: Some("MergeTopKU32".into()),
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::default(),
        );
        let f32_storage_ty = module.types.insert(
            Type {
                name: Some("MergeTopKF32Buffer".into()),
                inner: TypeInner::Array {
                    base: f32_ty,
                    size: ArraySize::Dynamic,
                    stride: 4,
                },
            },
            Span::default(),
        );
        let u32_storage_ty = module.types.insert(
            Type {
                name: Some("MergeTopKU32Buffer".into()),
                inner: TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Dynamic,
                    stride: 4,
                },
            },
            Span::default(),
        );
        let chunk_positions_ty = module.types.insert(
            Type {
                name: Some("MergeTopKChunkPositions".into()),
                inner: TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Constant(NonZeroU32::new(self.chunks)?),
                    stride: 4,
                },
            },
            Span::default(),
        );
        let scratch_f32_ty = module.types.insert(
            Type {
                name: Some("MergeTopKScratchF32".into()),
                inner: TypeInner::Array {
                    base: f32_ty,
                    size: ArraySize::Constant(NonZeroU32::new(TOP_K_BLOCK)?),
                    stride: 4,
                },
            },
            Span::default(),
        );
        let scratch_u32_ty = module.types.insert(
            Type {
                name: Some("MergeTopKScratchU32".into()),
                inner: TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Constant(NonZeroU32::new(TOP_K_BLOCK)?),
                    stride: 4,
                },
            },
            Span::default(),
        );

        let globals = MergeTopKGlobals {
            input_ids: Self::storage_global(&mut module, "input_ids", 0, u32_storage_ty, true),
            input_values: Self::storage_global(
                &mut module,
                "input_values",
                1,
                f32_storage_ty,
                true,
            ),
            output_ids: Self::storage_global(&mut module, "output_ids", 2, u32_storage_ty, false),
            output_values: Self::storage_global(
                &mut module,
                "output_values",
                3,
                f32_storage_ty,
                false,
            ),
            chunk_positions: Self::workgroup_global(
                &mut module,
                "chunk_positions",
                chunk_positions_ty,
            ),
            scratch_values: Self::workgroup_global(&mut module, "scratch_values", scratch_f32_ty),
            scratch_ids: Self::workgroup_global(&mut module, "scratch_ids", scratch_u32_ty),
            scratch_chunks: Self::workgroup_global(&mut module, "scratch_chunks", scratch_u32_ty),
        };

        let mut function = Function {
            name: Some("main".into()),
            arguments: vec![FunctionArgument {
                name: Some("local_invocation_index".into()),
                ty: u32_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)),
            }],
            ..Function::default()
        };
        let locals = MergeTopKLocals {
            rank: Self::local(&mut function, "rank", u32_ty),
            scan_chunk: Self::local(&mut function, "scan_chunk", u32_ty),
            local_best_value: Self::local(&mut function, "local_best_value", f32_ty),
            local_best_id: Self::local(&mut function, "local_best_id", u32_ty),
            local_best_chunk: Self::local(&mut function, "local_best_chunk", u32_ty),
            reduce_step: Self::local(&mut function, "reduce_step", u32_ty),
        };

        function.body = self.entry_body(&mut function.expressions, globals, locals);
        function
            .body
            .push(Statement::Return { value: None }, Span::default());
        module.entry_points.push(EntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [TOP_K_BLOCK, 1, 1],
            workgroup_size_overrides: None,
            function,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });

        Some(module)
    }

    fn entry_body(
        &self,
        expressions: &mut Arena<Expression>,
        globals: MergeTopKGlobals,
        locals: MergeTopKLocals,
    ) -> Block {
        let mut body = Block::new();
        let lane = expressions.append(Expression::FunctionArgument(0), Span::default());

        self.store_local(expressions, &mut body, locals.scan_chunk, lane);
        let mut init_body = Block::new();
        let chunk = self.load_local(expressions, &mut init_body, locals.scan_chunk);
        let done = self.ge_lit(expressions, &mut init_body, chunk, self.chunks);
        init_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );
        let zero = self.u32_lit(expressions, 0);
        self.store_storage(
            expressions,
            &mut init_body,
            globals.chunk_positions,
            chunk,
            zero,
        );
        let chunk = self.load_local(expressions, &mut init_body, locals.scan_chunk);
        let next = self.add_lit(expressions, &mut init_body, chunk, TOP_K_BLOCK);
        self.store_local(expressions, &mut init_body, locals.scan_chunk, next);
        body.push(
            Statement::Loop {
                body: init_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let zero = self.u32_lit(expressions, 0);
        self.store_local(expressions, &mut body, locals.rank, zero);
        let mut rank_body = Block::new();
        let rank = self.load_local(expressions, &mut rank_body, locals.rank);
        let done = self.ge_lit(expressions, &mut rank_body, rank, self.k);
        rank_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let neg_max = self.f32_lit(expressions, NEG_MAX_F32);
        let invalid = self.u32_lit(expressions, u32::MAX);
        self.store_local(
            expressions,
            &mut rank_body,
            locals.local_best_value,
            neg_max,
        );
        self.store_local(expressions, &mut rank_body, locals.local_best_id, invalid);
        self.store_local(
            expressions,
            &mut rank_body,
            locals.local_best_chunk,
            invalid,
        );
        self.store_local(expressions, &mut rank_body, locals.scan_chunk, lane);

        self.append_scan_chunks_loop(expressions, &mut rank_body, &globals, &locals);
        self.store_local_best_to_scratch(expressions, &mut rank_body, &globals, &locals, lane);
        self.append_reduce_loop(expressions, &mut rank_body, &globals, &locals, lane);
        self.store_rank_output(expressions, &mut rank_body, &globals, &locals, lane);

        let rank = self.load_local(expressions, &mut rank_body, locals.rank);
        let next_rank = self.add_lit(expressions, &mut rank_body, rank, 1);
        self.store_local(expressions, &mut rank_body, locals.rank, next_rank);
        body.push(
            Statement::Loop {
                body: rank_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );

        body
    }

    fn append_scan_chunks_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: &MergeTopKGlobals,
        locals: &MergeTopKLocals,
    ) {
        let mut scan_body = Block::new();
        let chunk = self.load_local(expressions, &mut scan_body, locals.scan_chunk);
        let done = self.ge_lit(expressions, &mut scan_body, chunk, self.chunks);
        scan_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let position =
            self.load_storage(expressions, &mut scan_body, globals.chunk_positions, chunk);
        let chunk_len = self.u32_lit(expressions, self.chunk_len);
        let in_chunk = self.bin(
            expressions,
            &mut scan_body,
            BinaryOperator::Less,
            position,
            chunk_len,
        );
        let mut candidate_accept = Block::new();
        let chunk_offset =
            self.mul_lit(expressions, &mut candidate_accept, chunk, self.chunk_stride);
        let index = self.bin(
            expressions,
            &mut candidate_accept,
            BinaryOperator::Add,
            chunk_offset,
            position,
        );
        let id = self.load_storage(expressions, &mut candidate_accept, globals.input_ids, index);
        let input_len = self.u32_lit(expressions, self.input_len);
        let valid_id = self.bin(
            expressions,
            &mut candidate_accept,
            BinaryOperator::Less,
            id,
            input_len,
        );
        let value = self.load_storage(
            expressions,
            &mut candidate_accept,
            globals.input_values,
            index,
        );
        let finite = self.is_finite(expressions, &mut candidate_accept, value);
        let valid = self.and(expressions, &mut candidate_accept, valid_id, finite);
        let best_value =
            self.load_local(expressions, &mut candidate_accept, locals.local_best_value);
        let best_id = self.load_local(expressions, &mut candidate_accept, locals.local_best_id);
        let better = self.better_candidate(
            expressions,
            &mut candidate_accept,
            value,
            id,
            best_value,
            best_id,
        );
        let should_update = self.and(expressions, &mut candidate_accept, valid, better);
        let mut update = Block::new();
        self.store_local(expressions, &mut update, locals.local_best_value, value);
        self.store_local(expressions, &mut update, locals.local_best_id, id);
        self.store_local(expressions, &mut update, locals.local_best_chunk, chunk);
        candidate_accept.push(
            Statement::If {
                condition: should_update,
                accept: update,
                reject: Block::new(),
            },
            Span::default(),
        );
        scan_body.push(
            Statement::If {
                condition: in_chunk,
                accept: candidate_accept,
                reject: Block::new(),
            },
            Span::default(),
        );

        let chunk = self.load_local(expressions, &mut scan_body, locals.scan_chunk);
        let next = self.add_lit(expressions, &mut scan_body, chunk, TOP_K_BLOCK);
        self.store_local(expressions, &mut scan_body, locals.scan_chunk, next);
        body.push(
            Statement::Loop {
                body: scan_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }

    fn store_local_best_to_scratch(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: &MergeTopKGlobals,
        locals: &MergeTopKLocals,
        lane: Handle<Expression>,
    ) {
        let value = self.load_local(expressions, body, locals.local_best_value);
        let id = self.load_local(expressions, body, locals.local_best_id);
        let chunk = self.load_local(expressions, body, locals.local_best_chunk);
        self.store_storage(expressions, body, globals.scratch_values, lane, value);
        self.store_storage(expressions, body, globals.scratch_ids, lane, id);
        self.store_storage(expressions, body, globals.scratch_chunks, lane, chunk);
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
    }

    fn append_reduce_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: &MergeTopKGlobals,
        locals: &MergeTopKLocals,
        lane: Handle<Expression>,
    ) {
        let half_block = self.u32_lit(expressions, TOP_K_BLOCK / 2);
        self.store_local(expressions, body, locals.reduce_step, half_block);

        let mut reduce_body = Block::new();
        let step = self.load_local(expressions, &mut reduce_body, locals.reduce_step);
        let done = self.eq_lit(expressions, &mut reduce_body, step, 0);
        reduce_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let participates = self.bin(
            expressions,
            &mut reduce_body,
            BinaryOperator::Less,
            lane,
            step,
        );
        let mut accept = Block::new();
        let other_index = self.bin(expressions, &mut accept, BinaryOperator::Add, lane, step);
        let other_value = self.load_storage(
            expressions,
            &mut accept,
            globals.scratch_values,
            other_index,
        );
        let other_id =
            self.load_storage(expressions, &mut accept, globals.scratch_ids, other_index);
        let other_chunk = self.load_storage(
            expressions,
            &mut accept,
            globals.scratch_chunks,
            other_index,
        );
        let current_value =
            self.load_storage(expressions, &mut accept, globals.scratch_values, lane);
        let current_id = self.load_storage(expressions, &mut accept, globals.scratch_ids, lane);
        let better = self.better_candidate(
            expressions,
            &mut accept,
            other_value,
            other_id,
            current_value,
            current_id,
        );
        let mut better_accept = Block::new();
        self.store_storage(
            expressions,
            &mut better_accept,
            globals.scratch_values,
            lane,
            other_value,
        );
        self.store_storage(
            expressions,
            &mut better_accept,
            globals.scratch_ids,
            lane,
            other_id,
        );
        self.store_storage(
            expressions,
            &mut better_accept,
            globals.scratch_chunks,
            lane,
            other_chunk,
        );
        accept.push(
            Statement::If {
                condition: better,
                accept: better_accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        reduce_body.push(
            Statement::If {
                condition: participates,
                accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        reduce_body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        let step = self.load_local(expressions, &mut reduce_body, locals.reduce_step);
        let two = self.u32_lit(expressions, 2);
        let next_step = self.bin(
            expressions,
            &mut reduce_body,
            BinaryOperator::Divide,
            step,
            two,
        );
        self.store_local(expressions, &mut reduce_body, locals.reduce_step, next_step);
        body.push(
            Statement::Loop {
                body: reduce_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }

    fn store_rank_output(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: &MergeTopKGlobals,
        locals: &MergeTopKLocals,
        lane: Handle<Expression>,
    ) {
        let lane_zero = self.eq_lit(expressions, body, lane, 0);
        let mut accept = Block::new();
        let zero = self.u32_lit(expressions, 0);
        let selected_value =
            self.load_storage(expressions, &mut accept, globals.scratch_values, zero);
        let zero = self.u32_lit(expressions, 0);
        let selected_id = self.load_storage(expressions, &mut accept, globals.scratch_ids, zero);
        let zero = self.u32_lit(expressions, 0);
        let selected_chunk =
            self.load_storage(expressions, &mut accept, globals.scratch_chunks, zero);
        let rank = self.load_local(expressions, &mut accept, locals.rank);
        self.store_storage(
            expressions,
            &mut accept,
            globals.output_values,
            rank,
            selected_value,
        );
        self.store_storage(
            expressions,
            &mut accept,
            globals.output_ids,
            rank,
            selected_id,
        );

        let chunks = self.u32_lit(expressions, self.chunks);
        let valid_chunk = self.bin(
            expressions,
            &mut accept,
            BinaryOperator::Less,
            selected_chunk,
            chunks,
        );
        let mut advance = Block::new();
        let position = self.load_storage(
            expressions,
            &mut advance,
            globals.chunk_positions,
            selected_chunk,
        );
        let next_position = self.add_lit(expressions, &mut advance, position, 1);
        self.store_storage(
            expressions,
            &mut advance,
            globals.chunk_positions,
            selected_chunk,
            next_position,
        );
        accept.push(
            Statement::If {
                condition: valid_chunk,
                accept: advance,
                reject: Block::new(),
            },
            Span::default(),
        );
        body.push(
            Statement::If {
                condition: lane_zero,
                accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
    }

    fn storage_global(
        module: &mut Module,
        name: &str,
        binding: u32,
        ty: Handle<Type>,
        read_only: bool,
    ) -> Handle<GlobalVariable> {
        module.global_variables.append(
            GlobalVariable {
                name: Some(name.into()),
                space: AddressSpace::Storage {
                    access: if read_only {
                        StorageAccess::LOAD
                    } else {
                        StorageAccess::LOAD | StorageAccess::STORE
                    },
                },
                binding: Some(ResourceBinding { group: 0, binding }),
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    fn workgroup_global(
        module: &mut Module,
        name: &str,
        ty: Handle<Type>,
    ) -> Handle<GlobalVariable> {
        module.global_variables.append(
            GlobalVariable {
                name: Some(name.into()),
                space: AddressSpace::WorkGroup,
                binding: None,
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    fn local(function: &mut Function, name: &str, ty: Handle<Type>) -> Handle<LocalVariable> {
        function.local_variables.append(
            LocalVariable {
                name: Some(name.into()),
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    fn is_finite(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
    ) -> Handle<Expression> {
        let self_equal = self.bin(expressions, body, BinaryOperator::Equal, value, value);
        let abs = self.emit(
            expressions,
            body,
            Expression::Math {
                fun: MathFunction::Abs,
                arg: value,
                arg1: None,
                arg2: None,
                arg3: None,
            },
        );
        let max = self.f32_lit(expressions, MAX_F32);
        let finite_magnitude = self.bin(expressions, body, BinaryOperator::LessEqual, abs, max);
        self.and(expressions, body, self_equal, finite_magnitude)
    }

    fn better_candidate(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        id: Handle<Expression>,
        best_value: Handle<Expression>,
        best_id: Handle<Expression>,
    ) -> Handle<Expression> {
        let value_greater = self.bin(
            expressions,
            body,
            BinaryOperator::Greater,
            value,
            best_value,
        );
        let value_equal = self.bin(expressions, body, BinaryOperator::Equal, value, best_value);
        let id_greater = self.bin(expressions, body, BinaryOperator::Greater, id, best_id);
        let equal_and_id = self.and(expressions, body, value_equal, id_greater);
        self.or(expressions, body, value_greater, equal_and_id)
    }

    fn load_storage(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let ptr = self.storage_ptr(expressions, body, global, index);
        self.emit(expressions, body, Expression::Load { pointer: ptr })
    }

    fn store_storage(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
        value: Handle<Expression>,
    ) {
        let pointer = self.storage_ptr(expressions, body, global, index);
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    fn storage_ptr(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let base = expressions.append(Expression::GlobalVariable(global), Span::default());
        self.emit(expressions, body, Expression::Access { base, index })
    }

    fn load_local(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        local: Handle<LocalVariable>,
    ) -> Handle<Expression> {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        self.emit(expressions, body, Expression::Load { pointer })
    }

    fn store_local(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        local: Handle<LocalVariable>,
        value: Handle<Expression>,
    ) {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    fn add_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        if literal == 0 {
            value
        } else {
            let rhs = self.u32_lit(expressions, literal);
            self.bin(expressions, body, BinaryOperator::Add, value, rhs)
        }
    }

    fn mul_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Multiply, value, rhs)
    }

    fn ge_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::GreaterEqual, value, rhs)
    }

    fn eq_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Equal, value, rhs)
    }

    fn and(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(expressions, body, BinaryOperator::LogicalAnd, left, right)
    }

    fn or(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(expressions, body, BinaryOperator::LogicalOr, left, right)
    }

    fn bin(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(expressions, body, Expression::Binary { op, left, right })
    }

    fn emit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        expression: Expression,
    ) -> Handle<Expression> {
        let handle = expressions.append(expression, Span::default());
        body.push(
            Statement::Emit(Range::new_from_bounds(handle, handle)),
            Span::default(),
        );
        handle
    }

    fn f32_lit(&self, expressions: &mut Arena<Expression>, value: f32) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::F32(value)), Span::default())
    }

    fn u32_lit(&self, expressions: &mut Arena<Expression>, value: u32) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::U32(value)), Span::default())
    }
}

impl TopKModuleBuilder {
    fn new(
        input_len: u32,
        output_per_chunk: u32,
        input_offset: u32,
        input_stride: u32,
        processors: bool,
    ) -> Self {
        Self {
            input_len,
            output_per_chunk,
            input_offset,
            input_stride,
            processors,
        }
    }

    fn build(self) -> Option<Module> {
        let mut module = Module::default();
        let bool_ty = module.types.insert(
            Type {
                name: Some("TopKBool".into()),
                inner: TypeInner::Scalar(Scalar::BOOL),
            },
            Span::default(),
        );
        let f32_ty = module.types.insert(
            Type {
                name: Some("TopKF32".into()),
                inner: TypeInner::Scalar(Scalar::F32),
            },
            Span::default(),
        );
        let u32_ty = module.types.insert(
            Type {
                name: Some("TopKU32".into()),
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::default(),
        );
        let u32_vec3_ty = module.types.insert(
            Type {
                name: Some("TopKWorkgroupId".into()),
                inner: TypeInner::Vector {
                    size: VectorSize::Tri,
                    scalar: Scalar::U32,
                },
            },
            Span::default(),
        );
        let f32_storage_ty = module.types.insert(
            Type {
                name: Some("TopKF32Buffer".into()),
                inner: TypeInner::Array {
                    base: f32_ty,
                    size: ArraySize::Dynamic,
                    stride: 4,
                },
            },
            Span::default(),
        );
        let u32_storage_ty = module.types.insert(
            Type {
                name: Some("TopKU32Buffer".into()),
                inner: TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Dynamic,
                    stride: 4,
                },
            },
            Span::default(),
        );
        let scratch_f32_ty = module.types.insert(
            Type {
                name: Some("TopKScratchF32".into()),
                inner: TypeInner::Array {
                    base: f32_ty,
                    size: ArraySize::Constant(NonZeroU32::new(TOP_K_BLOCK)?),
                    stride: 4,
                },
            },
            Span::default(),
        );
        let scratch_u32_ty = module.types.insert(
            Type {
                name: Some("TopKScratchU32".into()),
                inner: TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Constant(NonZeroU32::new(TOP_K_BLOCK)?),
                    stride: 4,
                },
            },
            Span::default(),
        );

        let globals = TopKGlobals {
            input: Self::storage_global(&mut module, "input", 0, f32_storage_ty, true),
            output_ids: Self::storage_global(&mut module, "output_ids", 1, u32_storage_ty, false),
            output_values: Self::storage_global(
                &mut module,
                "output_values",
                2,
                f32_storage_ty,
                false,
            ),
            previous_tokens: self.processors.then(|| {
                Self::storage_global(&mut module, "previous_tokens", 3, u32_storage_ty, true)
            }),
            processor_params: self.processors.then(|| {
                Self::storage_global(&mut module, "processor_params", 4, u32_storage_ty, true)
            }),
            scratch_values: Self::workgroup_global(&mut module, "scratch_values", scratch_f32_ty),
            scratch_ids: Self::workgroup_global(&mut module, "scratch_ids", scratch_u32_ty),
        };

        let mut function = Function {
            name: Some("main".into()),
            arguments: vec![
                FunctionArgument {
                    name: Some("local_invocation_index".into()),
                    ty: u32_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)),
                },
                FunctionArgument {
                    name: Some("workgroup_id".into()),
                    ty: u32_vec3_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::WorkGroupId)),
                },
            ],
            ..Function::default()
        };
        let locals = TopKLocals {
            current_value: Self::local(&mut function, "current_value", f32_ty),
            current_id: Self::local(&mut function, "current_id", u32_ty),
            previous_index: Self::local(&mut function, "previous_index", u32_ty),
            repeated: Self::local(&mut function, "repeated", bool_ty),
        };

        function.body = self.entry_body(&mut function.expressions, globals, locals);
        function
            .body
            .push(Statement::Return { value: None }, Span::default());
        module.entry_points.push(EntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [TOP_K_BLOCK, 1, 1],
            workgroup_size_overrides: None,
            function,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });

        Some(module)
    }

    fn entry_body(
        &self,
        expressions: &mut Arena<Expression>,
        globals: TopKGlobals,
        locals: TopKLocals,
    ) -> Block {
        let mut body = Block::new();
        let lane = expressions.append(Expression::FunctionArgument(0), Span::default());
        let workgroup_id = expressions.append(Expression::FunctionArgument(1), Span::default());
        let chunk = self.emit(
            expressions,
            &mut body,
            Expression::AccessIndex {
                base: workgroup_id,
                index: 0,
            },
        );
        let neg_max = self.f32_lit(expressions, NEG_MAX_F32);
        let invalid_id = self.u32_lit(expressions, u32::MAX);
        self.store_local(expressions, &mut body, locals.current_value, neg_max);
        self.store_local(expressions, &mut body, locals.current_id, invalid_id);

        let chunk_base = self.mul_lit(expressions, &mut body, chunk, TOP_K_CHUNK as u32);
        let token_id = self.bin(
            expressions,
            &mut body,
            BinaryOperator::Add,
            chunk_base,
            lane,
        );
        let input_len = self.u32_lit(expressions, self.input_len);
        let token_valid = self.bin(
            expressions,
            &mut body,
            BinaryOperator::Less,
            token_id,
            input_len,
        );
        let mut load_accept = Block::new();
        let input_index = if self.input_stride == 1 {
            self.add_lit(expressions, &mut load_accept, token_id, self.input_offset)
        } else {
            let scaled = self.mul_lit(expressions, &mut load_accept, token_id, self.input_stride);
            self.add_lit(expressions, &mut load_accept, scaled, self.input_offset)
        };
        let value = self.load_storage(expressions, &mut load_accept, globals.input, input_index);
        let raw_finite = self.is_finite(expressions, &mut load_accept, value);
        let mut finite_accept = Block::new();
        let value = self.apply_processors(
            expressions,
            &mut finite_accept,
            &globals,
            &locals,
            value,
            token_id,
        );
        let finite = self.is_finite(expressions, &mut finite_accept, value);
        let mut processed_finite_accept = Block::new();
        self.store_local(
            expressions,
            &mut processed_finite_accept,
            locals.current_value,
            value,
        );
        self.store_local(
            expressions,
            &mut processed_finite_accept,
            locals.current_id,
            token_id,
        );
        finite_accept.push(
            Statement::If {
                condition: finite,
                accept: processed_finite_accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        load_accept.push(
            Statement::If {
                condition: raw_finite,
                accept: finite_accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        body.push(
            Statement::If {
                condition: token_valid,
                accept: load_accept,
                reject: Block::new(),
            },
            Span::default(),
        );

        let current_value = self.load_local(expressions, &mut body, locals.current_value);
        let current_id = self.load_local(expressions, &mut body, locals.current_id);
        self.store_storage(
            expressions,
            &mut body,
            globals.scratch_values,
            lane,
            current_value,
        );
        self.store_storage(
            expressions,
            &mut body,
            globals.scratch_ids,
            lane,
            current_id,
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let mut size = 2;
        while size <= TOP_K_BLOCK {
            let mut stride = size / 2;
            while stride > 0 {
                self.append_bitonic_stage(expressions, &mut body, &globals, lane, size, stride);
                stride /= 2;
            }
            size *= 2;
        }

        let output_per_chunk = self.u32_lit(expressions, self.output_per_chunk);
        let writes_output = self.bin(
            expressions,
            &mut body,
            BinaryOperator::Less,
            lane,
            output_per_chunk,
        );
        let mut write_accept = Block::new();
        let chunk_base = self.mul_lit(expressions, &mut write_accept, chunk, self.output_per_chunk);
        let output_index = self.bin(
            expressions,
            &mut write_accept,
            BinaryOperator::Add,
            chunk_base,
            lane,
        );
        let selected_value =
            self.load_storage(expressions, &mut write_accept, globals.scratch_values, lane);
        let selected_id =
            self.load_storage(expressions, &mut write_accept, globals.scratch_ids, lane);
        self.store_storage(
            expressions,
            &mut write_accept,
            globals.output_values,
            output_index,
            selected_value,
        );
        self.store_storage(
            expressions,
            &mut write_accept,
            globals.output_ids,
            output_index,
            selected_id,
        );
        body.push(
            Statement::If {
                condition: writes_output,
                accept: write_accept,
                reject: Block::new(),
            },
            Span::default(),
        );

        body
    }

    fn append_bitonic_stage(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: &TopKGlobals,
        lane: Handle<Expression>,
        size: u32,
        stride: u32,
    ) {
        let stride_lit = self.u32_lit(expressions, stride);
        let partner = self.bin(
            expressions,
            body,
            BinaryOperator::ExclusiveOr,
            lane,
            stride_lit,
        );
        let current_value = self.load_storage(expressions, body, globals.scratch_values, lane);
        let current_id = self.load_storage(expressions, body, globals.scratch_ids, lane);
        let partner_value = self.load_storage(expressions, body, globals.scratch_values, partner);
        let partner_id = self.load_storage(expressions, body, globals.scratch_ids, partner);

        let stride_lit = self.u32_lit(expressions, stride);
        let lane_stride_bits = self.bin(expressions, body, BinaryOperator::And, lane, stride_lit);
        let size_lit = self.u32_lit(expressions, size);
        let lane_size_bits = self.bin(expressions, body, BinaryOperator::And, lane, size_lit);
        let zero = self.u32_lit(expressions, 0);
        let lower_lane = self.bin(
            expressions,
            body,
            BinaryOperator::Equal,
            lane_stride_bits,
            zero,
        );
        let descending = self.bin(
            expressions,
            body,
            BinaryOperator::Equal,
            lane_size_bits,
            zero,
        );
        let want_better = self.bin(
            expressions,
            body,
            BinaryOperator::Equal,
            lower_lane,
            descending,
        );

        let partner_better = self.better_candidate(
            expressions,
            body,
            partner_value,
            partner_id,
            current_value,
            current_id,
        );
        let current_better = self.better_candidate(
            expressions,
            body,
            current_value,
            current_id,
            partner_value,
            partner_id,
        );
        let false_lit = self.bool_lit(expressions, false);
        let want_worse = self.bin(
            expressions,
            body,
            BinaryOperator::Equal,
            want_better,
            false_lit,
        );
        let choose_better_partner = self.and(expressions, body, want_better, partner_better);
        let choose_worse_partner = self.and(expressions, body, want_worse, current_better);
        let choose_partner = self.or(
            expressions,
            body,
            choose_better_partner,
            choose_worse_partner,
        );

        let mut accept = Block::new();
        self.store_storage(
            expressions,
            &mut accept,
            globals.scratch_values,
            lane,
            partner_value,
        );
        self.store_storage(
            expressions,
            &mut accept,
            globals.scratch_ids,
            lane,
            partner_id,
        );
        body.push(
            Statement::If {
                condition: choose_partner,
                accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
    }

    fn apply_processors(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: &TopKGlobals,
        locals: &TopKLocals,
        value: Handle<Expression>,
        token_id: Handle<Expression>,
    ) -> Handle<Expression> {
        let (Some(previous_tokens), Some(processor_params)) =
            (globals.previous_tokens, globals.processor_params)
        else {
            return value;
        };

        self.store_local(expressions, body, locals.current_value, value);
        let zero_u32 = self.u32_lit(expressions, 0);
        self.store_local(expressions, body, locals.previous_index, zero_u32);
        let false_lit = self.bool_lit(expressions, false);
        self.store_local(expressions, body, locals.repeated, false_lit);

        let previous_len_index = self.u32_lit(expressions, 2);
        let previous_len =
            self.load_storage(expressions, body, processor_params, previous_len_index);
        let mut scan_body = Block::new();
        let previous_index = self.load_local(expressions, &mut scan_body, locals.previous_index);
        let done = self.bin(
            expressions,
            &mut scan_body,
            BinaryOperator::GreaterEqual,
            previous_index,
            previous_len,
        );
        scan_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );
        let previous_index = self.load_local(expressions, &mut scan_body, locals.previous_index);
        let previous_token =
            self.load_storage(expressions, &mut scan_body, previous_tokens, previous_index);
        let repeated = self.bin(
            expressions,
            &mut scan_body,
            BinaryOperator::Equal,
            previous_token,
            token_id,
        );
        let mut repeated_accept = Block::new();
        let true_lit = self.bool_lit(expressions, true);
        self.store_local(expressions, &mut repeated_accept, locals.repeated, true_lit);
        repeated_accept.push(Statement::Break, Span::default());
        scan_body.push(
            Statement::If {
                condition: repeated,
                accept: repeated_accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        let previous_index = self.load_local(expressions, &mut scan_body, locals.previous_index);
        let next_previous_index = self.add_lit(expressions, &mut scan_body, previous_index, 1);
        self.store_local(
            expressions,
            &mut scan_body,
            locals.previous_index,
            next_previous_index,
        );
        body.push(
            Statement::Loop {
                body: scan_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );

        let repetition_penalty =
            self.load_processor_param_f32(expressions, body, processor_params, 1);
        let repeated = self.load_local(expressions, body, locals.repeated);
        let one = self.f32_lit(expressions, 1.0);
        let penalty_gt_one = self.bin(
            expressions,
            body,
            BinaryOperator::Greater,
            repetition_penalty,
            one,
        );
        let should_apply_penalty = self.and(expressions, body, repeated, penalty_gt_one);
        let mut penalty_accept = Block::new();
        let current = self.load_local(expressions, &mut penalty_accept, locals.current_value);
        let zero = self.f32_lit(expressions, 0.0);
        let non_positive = self.bin(
            expressions,
            &mut penalty_accept,
            BinaryOperator::LessEqual,
            current,
            zero,
        );
        let mut non_positive_accept = Block::new();
        let current = self.load_local(expressions, &mut non_positive_accept, locals.current_value);
        let penalized = self.bin(
            expressions,
            &mut non_positive_accept,
            BinaryOperator::Multiply,
            current,
            repetition_penalty,
        );
        self.store_local(
            expressions,
            &mut non_positive_accept,
            locals.current_value,
            penalized,
        );
        let mut positive_accept = Block::new();
        let current = self.load_local(expressions, &mut positive_accept, locals.current_value);
        let penalized = self.bin(
            expressions,
            &mut positive_accept,
            BinaryOperator::Divide,
            current,
            repetition_penalty,
        );
        self.store_local(
            expressions,
            &mut positive_accept,
            locals.current_value,
            penalized,
        );
        penalty_accept.push(
            Statement::If {
                condition: non_positive,
                accept: non_positive_accept,
                reject: positive_accept,
            },
            Span::default(),
        );
        body.push(
            Statement::If {
                condition: should_apply_penalty,
                accept: penalty_accept,
                reject: Block::new(),
            },
            Span::default(),
        );

        let temperature = self.load_processor_param_f32(expressions, body, processor_params, 0);
        let zero = self.f32_lit(expressions, 0.0);
        let temp_nonzero = self.bin(
            expressions,
            body,
            BinaryOperator::NotEqual,
            temperature,
            zero,
        );
        let mut temperature_accept = Block::new();
        let current = self.load_local(expressions, &mut temperature_accept, locals.current_value);
        let adjusted = self.bin(
            expressions,
            &mut temperature_accept,
            BinaryOperator::Divide,
            current,
            temperature,
        );
        self.store_local(
            expressions,
            &mut temperature_accept,
            locals.current_value,
            adjusted,
        );
        body.push(
            Statement::If {
                condition: temp_nonzero,
                accept: temperature_accept,
                reject: Block::new(),
            },
            Span::default(),
        );

        self.load_local(expressions, body, locals.current_value)
    }

    fn storage_global(
        module: &mut Module,
        name: &str,
        binding: u32,
        ty: Handle<Type>,
        read_only: bool,
    ) -> Handle<GlobalVariable> {
        module.global_variables.append(
            GlobalVariable {
                name: Some(name.into()),
                space: AddressSpace::Storage {
                    access: if read_only {
                        StorageAccess::LOAD
                    } else {
                        StorageAccess::LOAD | StorageAccess::STORE
                    },
                },
                binding: Some(ResourceBinding { group: 0, binding }),
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    fn workgroup_global(
        module: &mut Module,
        name: &str,
        ty: Handle<Type>,
    ) -> Handle<GlobalVariable> {
        module.global_variables.append(
            GlobalVariable {
                name: Some(name.into()),
                space: AddressSpace::WorkGroup,
                binding: None,
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    fn local(function: &mut Function, name: &str, ty: Handle<Type>) -> Handle<LocalVariable> {
        function.local_variables.append(
            LocalVariable {
                name: Some(name.into()),
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    fn is_finite(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
    ) -> Handle<Expression> {
        let self_equal = self.bin(expressions, body, BinaryOperator::Equal, value, value);
        let abs = self.emit(
            expressions,
            body,
            Expression::Math {
                fun: MathFunction::Abs,
                arg: value,
                arg1: None,
                arg2: None,
                arg3: None,
            },
        );
        let max = self.f32_lit(expressions, MAX_F32);
        let finite_magnitude = self.bin(expressions, body, BinaryOperator::LessEqual, abs, max);
        self.and(expressions, body, self_equal, finite_magnitude)
    }

    fn better_candidate(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        id: Handle<Expression>,
        best_value: Handle<Expression>,
        best_id: Handle<Expression>,
    ) -> Handle<Expression> {
        let value_greater = self.bin(
            expressions,
            body,
            BinaryOperator::Greater,
            value,
            best_value,
        );
        let value_equal = self.bin(expressions, body, BinaryOperator::Equal, value, best_value);
        let id_greater = self.bin(expressions, body, BinaryOperator::Greater, id, best_id);
        let equal_and_id = self.and(expressions, body, value_equal, id_greater);
        self.or(expressions, body, value_greater, equal_and_id)
    }

    fn load_storage(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let ptr = self.storage_ptr(expressions, body, global, index);
        self.emit(expressions, body, Expression::Load { pointer: ptr })
    }

    fn load_processor_param_f32(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: u32,
    ) -> Handle<Expression> {
        let index = self.u32_lit(expressions, index);
        let bits = self.load_storage(expressions, body, global, index);
        self.emit(
            expressions,
            body,
            Expression::As {
                expr: bits,
                kind: ScalarKind::Float,
                convert: None,
            },
        )
    }

    fn store_storage(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
        value: Handle<Expression>,
    ) {
        let pointer = self.storage_ptr(expressions, body, global, index);
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    fn storage_ptr(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let base = expressions.append(Expression::GlobalVariable(global), Span::default());
        self.emit(expressions, body, Expression::Access { base, index })
    }

    fn load_local(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        local: Handle<LocalVariable>,
    ) -> Handle<Expression> {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        self.emit(expressions, body, Expression::Load { pointer })
    }

    fn store_local(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        local: Handle<LocalVariable>,
        value: Handle<Expression>,
    ) {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    fn add_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        if literal == 0 {
            value
        } else {
            let rhs = self.u32_lit(expressions, literal);
            self.bin(expressions, body, BinaryOperator::Add, value, rhs)
        }
    }

    fn mul_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Multiply, value, rhs)
    }

    fn and(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(expressions, body, BinaryOperator::LogicalAnd, left, right)
    }

    fn or(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(expressions, body, BinaryOperator::LogicalOr, left, right)
    }

    fn bin(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(expressions, body, Expression::Binary { op, left, right })
    }

    fn emit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        expression: Expression,
    ) -> Handle<Expression> {
        let handle = expressions.append(expression, Span::default());
        body.push(
            Statement::Emit(Range::new_from_bounds(handle, handle)),
            Span::default(),
        );
        handle
    }

    fn f32_lit(&self, expressions: &mut Arena<Expression>, value: f32) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::F32(value)), Span::default())
    }

    fn u32_lit(&self, expressions: &mut Arena<Expression>, value: u32) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::U32(value)), Span::default())
    }

    fn bool_lit(&self, expressions: &mut Arena<Expression>, value: bool) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::Bool(value)), Span::default())
    }
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use crate::{DataTypeEnum, Device, Tensor, TensorData, quantized::QMatrix};
    use fusor_gguf::{BlockQ4_0, GgmlType};

    use super::{
        GpuMirostat2Sampler, GpuMirostat2SamplerParams,
        chunk_top_k_pair_data_with_processors_with_encoder, mirostat2_sample_token_to_host,
    };

    #[tokio::test]
    async fn top_k_pairs_match_cpu_sorted_order() {
        let device = Device::new().await.unwrap();
        let values = [
            0.25,
            f32::NAN,
            7.0,
            -3.0,
            f32::INFINITY,
            2.5,
            9.0,
            f32::NEG_INFINITY,
            8.5,
            9.0,
            6.0,
            -1.0,
        ];
        let tensor = Tensor::new(&device, values.as_slice());
        let (ids, logits) = tensor.top_k_pairs(5).await.unwrap();

        let mut expected = values
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| right.0.cmp(&left.0))
        });
        expected.truncate(5);

        let actual = ids
            .into_iter()
            .zip(logits)
            .map(|(id, value)| (id as usize, value))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn processed_chunk_top_k_applies_temperature_and_repetition_penalty() {
        let device = Device::new().await.unwrap();
        let values = [
            4.0,
            -2.0,
            3.5,
            8.0,
            f32::NAN,
            1.0,
            5.0,
            -1.5,
            6.5,
            7.0,
            f32::NEG_INFINITY,
            0.5,
        ];
        let buffer = device.create_buffer_init(
            bytemuck::cast_slice(&values),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let data = TensorData::new_from_buffer(&device, buffer, &[values.len()], DataTypeEnum::F32);
        let previous_tokens = [0, 3, 9];
        let (ids, logits) = chunk_top_k_pair_data_with_processors_with_encoder(
            &data,
            &previous_tokens,
            0.5,
            2.0,
            5,
            5,
            None,
        )
        .unwrap();
        let ids = Tensor::<1, u32>::from(ids).as_slice().await.unwrap();
        let logits = Tensor::<1, f32>::from(logits).as_slice().await.unwrap();

        let mut expected = values
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(token_id, mut value)| {
                if !value.is_finite() {
                    return None;
                }
                if previous_tokens.contains(&(token_id as u32)) {
                    if value <= 0.0 {
                        value *= 2.0;
                    } else {
                        value /= 2.0;
                    }
                }
                value /= 0.5;
                Some((token_id as u32, value))
            })
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| right.0.cmp(&left.0))
        });
        expected.truncate(5);

        let actual = ids
            .as_slice()
            .iter()
            .copied()
            .zip(logits.as_slice().iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    fn cpu_mirostat2_selected_token(
        values: &[f32],
        mu: f32,
        params: GpuMirostat2SamplerParams,
    ) -> u32 {
        let mut top = values
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(token_id, value)| {
                value
                    .is_finite()
                    .then_some((token_id as u32, value / params.temperature))
            })
            .collect::<Vec<_>>();
        top.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| right.0.cmp(&left.0))
        });
        top.truncate(params.top_k.min(top.len()));

        let max_value = top[0].1;
        let total = top
            .iter()
            .map(|(_, value)| (*value - max_value).exp())
            .sum::<f32>()
            .max(1.0e-20);
        let mut cutoff = 0usize;
        for (scan, (_, value)) in top.iter().enumerate() {
            let probability = (*value - max_value).exp() / total;
            if -probability.max(1.0e-20).log2() > mu {
                cutoff = scan.max(1);
                break;
            }
        }
        if cutoff == 0 {
            cutoff = 1;
        }

        let cutoff_sum = top
            .iter()
            .take(cutoff)
            .map(|(_, value)| (*value - max_value).exp())
            .sum::<f32>()
            .max(1.0e-20);
        let threshold = params.random.clamp(0.0, 0.999_999_94) * cutoff_sum;
        let mut cumulative = 0.0;
        let mut selected = top[0].0;
        for (token_id, value) in top.iter().take(cutoff) {
            cumulative += (*value - max_value).exp();
            if cumulative >= threshold {
                selected = *token_id;
                break;
            }
        }
        selected
    }

    #[tokio::test]
    async fn mirostat2_sampler_uses_exact_top_k_when_candidates_cluster() {
        let device = Device::new().await.unwrap();
        let mut values = vec![0.0f32; 512];
        for (token_id, value) in values.iter_mut().take(128).enumerate() {
            *value = 10.0 - token_id as f32 * 0.005;
        }
        let buffer = device.create_buffer_init(
            bytemuck::cast_slice(&values),
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let data = TensorData::new_from_buffer(&device, buffer, &[values.len()], DataTypeEnum::F32);
        let mu = 7.297_829;
        let params = GpuMirostat2SamplerParams {
            top_k: 128,
            temperature: 1.0,
            repetition_penalty: 1.0,
            tau: 5.0,
            eta: 0.1,
            random: 0.99,
        };
        let expected = cpu_mirostat2_selected_token(&values, mu, params);
        assert!(
            expected >= 64,
            "test setup should select a token missing from a 64-candidate chunk"
        );

        let mut sampler = GpuMirostat2Sampler::new(&device, mu);
        let token = mirostat2_sample_token_to_host(&data, &mut sampler, &[], params)
            .await
            .unwrap();

        assert_eq!(token, Some(expected));
    }

    #[tokio::test]
    async fn top_k_pairs_merge_path_match_cpu_sorted_order() {
        let device = Device::new().await.unwrap();
        let values = (0..4096)
            .map(|index| {
                if index % 997 == 0 {
                    f32::NAN
                } else if index % 991 == 0 {
                    f32::INFINITY
                } else if index % 983 == 0 {
                    f32::NEG_INFINITY
                } else {
                    let coarse = ((index * 37) % 251) as f32;
                    let tied = (index % 17) as f32 * 0.001;
                    coarse - tied
                }
            })
            .collect::<Vec<_>>();
        let tensor = Tensor::new(&device, values.as_slice());
        let (ids, logits) = tensor.top_k_pairs(16).await.unwrap();

        let mut expected = values
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| right.0.cmp(&left.0))
        });
        expected.truncate(16);

        let actual = ids
            .into_iter()
            .zip(logits)
            .map(|(id, value)| (id as usize, value))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn top_k_pairs_large_vocab_merge_path_matches_cpu_sorted_order() {
        let device = Device::new().await.unwrap();
        let values = (0..128_256)
            .map(|index| {
                if index % 65_521 == 0 {
                    f32::NAN
                } else if index % 32_749 == 0 {
                    f32::INFINITY
                } else if index % 32_719 == 0 {
                    f32::NEG_INFINITY
                } else {
                    let coarse = ((index * 97) % 4093) as f32;
                    let tied = (index % 31) as f32 * 0.0001;
                    coarse - tied
                }
            })
            .collect::<Vec<_>>();
        let tensor = Tensor::new(&device, values.as_slice());
        let (ids, logits) = tensor.top_k_pairs(512).await.unwrap();

        let mut expected = values
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| right.0.cmp(&left.0))
        });
        expected.truncate(512);

        let actual = ids
            .into_iter()
            .zip(logits)
            .map(|(id, value)| (id as usize, value))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn qmat_mirostat2_sample_token_uses_direct_sampler_path() {
        let device = Device::new().await.unwrap();
        let hidden = Tensor::new(&device, vec![1.0f32; 32].as_slice());
        let element_count = 8 * 32;
        let block_count = element_count / BlockQ4_0::BLOCK_SIZE;
        let raw_bytes = vec![0u8; block_count * size_of::<BlockQ4_0>()];
        let matrix =
            QMatrix::from_parts(&device, &raw_bytes, Box::new([8, 32]), GgmlType::Q4_0).unwrap();
        let mut sampler = GpuMirostat2Sampler::new(&device, 10.0);
        let params = GpuMirostat2SamplerParams {
            top_k: 4,
            temperature: 1.0,
            repetition_penalty: 1.0,
            tau: 5.0,
            eta: 0.1,
            random: 0.0,
        };

        let token = hidden
            .try_sample_mirostat2_token_q_mat(&matrix, &mut sampler, &[], params)
            .await
            .unwrap();

        assert_eq!(token, Some(7));
    }
}
