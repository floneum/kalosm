use std::{borrow::Cow, env, sync::mpsc, time::Instant};

use phase_token_prototype::{
    tile, GgmlQuantFormat, KernelIr, Layout, MemoryLevel, Shape, Strides, F32,
};
use wgpu::util::DeviceExt;

const GEMM_M: usize = 1024;
const GEMM_N: usize = 1024;
const GEMM_K: usize = 1024;
const GEMV_N: usize = 4096;
const GEMV_K: usize = 4096;
const WARMUP_BATCHES: usize = 2;
const MEASURED_BATCHES: usize = 5;
const GEMM_DISPATCHES_PER_BATCH: usize = 50;
const GEMV_DISPATCHES_PER_BATCH: usize = 100;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    pollster::block_on(run())
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mode = parse_mode()?;
    let formats = parse_formats()?;
    let (layout, layout_arg_present) = parse_layout()?;
    let shape = parse_shape_override(mode, layout_arg_present)?;
    let tile_override = parse_tile_override(mode, layout_arg_present, shape)?;

    for format in formats {
        let harness = Harness::new(format, mode, layout, shape, tile_override).await?;

        for _ in 0..WARMUP_BATCHES {
            harness.run_batch(mode.dispatches_per_batch())?;
        }

        let mut samples = Vec::with_capacity(MEASURED_BATCHES);
        for _ in 0..MEASURED_BATCHES {
            let command_buffer = harness.encode_batch(mode.dispatches_per_batch());
            let started = Instant::now();
            harness.submit_and_wait(command_buffer)?;
            samples.push(started.elapsed().as_secs_f64() / mode.dispatches_per_batch() as f64);
        }

        let output = harness.read_output()?;
        let max_abs_error = sampled_max_abs_error(
            &output,
            &harness.y_layout,
            &harness.a,
            harness.m,
            harness.n,
            harness.k,
        );
        if max_abs_error > 1.0e-3 {
            let (row, col, actual, expected) = first_sample_mismatch(
                &output,
                &harness.y_layout,
                &harness.a,
                harness.m,
                harness.n,
                harness.k,
                1.0e-3,
            )
            .unwrap_or((
                0,
                0,
                matrix_value(&output, &harness.y_layout, 0, 0),
                cpu_dot_ones(&harness.a, harness.k, 0),
            ));
            return Err(format!(
                "qmatmul {format:?} {mode:?} mismatch at ({row}, {col}): gpu={actual} cpu={expected} max_abs_error={max_abs_error}",
            )
            .into());
        }

        samples.sort_by(f64::total_cmp);
        let total_dispatches = MEASURED_BATCHES * mode.dispatches_per_batch();
        let mean_s = samples.iter().sum::<f64>() / samples.len() as f64;
        let p50_s = percentile(&samples, 0.50);
        let p90_s = percentile(&samples, 0.90);
        let min_s = samples[0];
        let max_s = samples[samples.len() - 1];
        let flops_per_dispatch = 2.0 * harness.m as f64 * harness.n as f64 * harness.k as f64;
        let weight_bytes = harness.b_words.len() as f64 * std::mem::size_of::<u32>() as f64;
        let io_bytes = weight_bytes
            + (harness.a_physical_len + harness.y_physical_len) as f64
                * std::mem::size_of::<f32>() as f64;

        println!(
            "adapter: {} ({:?})",
            harness.adapter_info.name, harness.adapter_info.backend
        );
        println!(
            "features: subgroup={} cooperative_matrix={} subgroup_min={} subgroup_max={}",
            harness.device_features.contains(wgpu::Features::SUBGROUP),
            harness
                .device_features
                .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX),
            harness.adapter_info.subgroup_min_size,
            harness.adapter_info.subgroup_max_size,
        );
        println!(
            "limits: max_invocations={} max_workgroup_storage={}",
            harness
                .device
                .limits()
                .max_compute_invocations_per_workgroup,
            harness.device.limits().max_compute_workgroup_storage_size,
        );
        println!(
            "bench_qmatmul_{mode:?}: {format:?} A={}x{} B={}x{} -> Y={}x{}",
            harness.m, harness.k, harness.k, harness.n, harness.m, harness.n
        );
        if matches!(mode, BenchMode::Gemm) {
            println!(
                "tile: BM={} BN={} BK={}",
                harness.tile_m, harness.tile_n, harness.tile_k
            );
        }
        println!("layout: {:?}", harness.layout);
        println!("dispatches: {total_dispatches} measured, {WARMUP_BATCHES} warmup batches");
        println!("max_abs_error: {max_abs_error:.6}");
        println!("mean_dispatch_time_us: {:.3}", mean_s * 1.0e6);
        println!("p50_dispatch_time_us: {:.3}", p50_s * 1.0e6);
        println!("p90_dispatch_time_us: {:.3}", p90_s * 1.0e6);
        println!("min_dispatch_time_us: {:.3}", min_s * 1.0e6);
        println!("max_dispatch_time_us: {:.3}", max_s * 1.0e6);
        println!(
            "effective_gflops: {:.6}",
            flops_per_dispatch / mean_s / 1.0e9
        );
        println!(
            "effective_tflops: {:.6}",
            flops_per_dispatch / mean_s / 1.0e12
        );
        println!(
            "packed_weight_bandwidth_gb_s: {:.6}",
            weight_bytes / mean_s / 1.0e9
        );
        println!(
            "effective_io_bandwidth_gb_s: {:.6}",
            io_bytes / mean_s / 1.0e9
        );
        println!();
    }

    Ok(())
}

#[derive(Copy, Clone, Debug)]
enum BenchMode {
    Gemm,
    Gemv,
}

#[derive(Copy, Clone, Debug)]
enum LayoutVariant {
    Contiguous,
    Padded,
    Transposed,
    Skewed,
    Im2Col,
}

#[derive(Copy, Clone, Debug)]
struct TilePlan {
    bm: usize,
    bn: usize,
    bk: usize,
}

impl BenchMode {
    fn dispatches_per_batch(self) -> usize {
        match self {
            Self::Gemm => GEMM_DISPATCHES_PER_BATCH,
            Self::Gemv => GEMV_DISPATCHES_PER_BATCH,
        }
    }
}

struct Harness {
    m: usize,
    n: usize,
    k: usize,
    tile_m: usize,
    tile_n: usize,
    tile_k: usize,
    layout: LayoutVariant,
    adapter_info: wgpu::AdapterInfo,
    device_features: wgpu::Features,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    dispatch_grid: [u32; 3],
    y_buffer: wgpu::Buffer,
    readback: wgpu::Buffer,
    y_layout: Layout,
    a: Vec<f32>,
    a_physical_len: usize,
    y_physical_len: usize,
    b_words: Vec<u32>,
}

impl Harness {
    async fn new(
        format: GgmlQuantFormat,
        mode: BenchMode,
        layout: LayoutVariant,
        shape: (usize, usize, usize),
        tile_override: Option<TilePlan>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (m, n, k) = shape;
        let tile_plan = match mode {
            BenchMode::Gemm => tile_override.unwrap_or_else(|| qgemm_tile_shape(format)),
            BenchMode::Gemv => TilePlan {
                bm: 1,
                bn: qgemv_cols_per_workgroup(format),
                bk: k,
            },
        };
        let ir = qmatmul_ir(
            format,
            m,
            n,
            k,
            layout,
            tile_override.filter(|_| matches!(mode, BenchMode::Gemm)),
        );
        let dispatch_grid = ir
            .single_tile_program_grid()
            .ok_or("qmatmul bench expects one tile program")?;
        let lowered = ir.lower_to_naga()?;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::default());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await?;
        let adapter_info = adapter.get_info();
        let adapter_features = adapter.features();
        let mut required_features = wgpu::Features::empty();
        if adapter_features.contains(wgpu::Features::SUBGROUP) {
            required_features |= wgpu::Features::SUBGROUP;
        }
        if adapter_features.contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX) {
            required_features |= wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("phase-token-prototype qmatmul bench device"),
                required_features,
                required_limits: adapter.limits(),
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                ..Default::default()
            })
            .await?;
        let device_features = device.features();

        let y_layout = output_layout(m, n, layout);
        let (a, a_physical) = activation_data(m, k, layout);
        let y_physical_len = allocation_len(&y_layout);
        let b_words = pack_ones(format, k, n);
        let a_buffer = storage_buffer_f32(&device, "A", &a_physical, wgpu::BufferUsages::empty());
        let b_buffer = storage_buffer_u32(&device, "Bq", &b_words, wgpu::BufferUsages::empty());
        let y_buffer = storage_buffer_f32(
            &device,
            "Y",
            &vec![0.0_f32; y_physical_len],
            wgpu::BufferUsages::COPY_SRC,
        );
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Y readback"),
            size: byte_len::<f32>(y_physical_len),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = unsafe {
            device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some("lowered qmatmul"),
                    source: wgpu::ShaderSource::Naga(Cow::Owned(lowered.module().clone())),
                },
                wgpu::ShaderRuntimeChecks::unchecked(),
            )
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("qmatmul buffers"),
            entries: &storage_bindings(&[true, true, false]),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("qmatmul pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("qmatmul pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("qmatmul bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            m,
            n,
            k,
            tile_m: tile_plan.bm,
            tile_n: tile_plan.bn,
            tile_k: tile_plan.bk,
            layout,
            adapter_info,
            device_features,
            device,
            queue,
            pipeline,
            bind_group,
            dispatch_grid,
            y_buffer,
            readback,
            y_layout,
            a,
            a_physical_len: a_physical.len(),
            y_physical_len,
            b_words,
        })
    }

    fn run_batch(&self, dispatches: usize) -> Result<(), Box<dyn std::error::Error>> {
        let command_buffer = self.encode_batch(dispatches);
        self.submit_and_wait(command_buffer)
    }

    fn encode_batch(&self, dispatches: usize) -> wgpu::CommandBuffer {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("qmatmul bench encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("qmatmul bench pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            for _ in 0..dispatches {
                let [x, y, z] = self.dispatch_grid;
                pass.dispatch_workgroups(x, y, z);
            }
        }
        encoder.finish()
    }

    fn submit_and_wait(
        &self,
        command_buffer: wgpu::CommandBuffer,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.queue.submit(Some(command_buffer));
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        Ok(())
    }

    fn read_output(&self) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("qmatmul readback encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &self.y_buffer,
            0,
            &self.readback,
            0,
            byte_len::<f32>(self.y_physical_len),
        );
        self.queue.submit(Some(encoder.finish()));

        let slice = self.readback.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;

        let mapped = slice.get_mapped_range();
        let output = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        self.readback.unmap();
        Ok(output)
    }
}

fn qmatmul_ir(
    format: GgmlQuantFormat,
    m: usize,
    n: usize,
    k: usize,
    layout: LayoutVariant,
    tile_override: Option<TilePlan>,
) -> KernelIr {
    tile::build(move |phase| {
        let a = if matches!(layout, LayoutVariant::Im2Col) {
            let shape = im2col_shape(m, k);
            let input = phase.storage_read_with_layout::<F32, 4>(im2col_input_layout(&shape));
            input.im2col_nhwc(
                [shape.out_h as u32, shape.out_w as u32],
                [shape.kernel_h as u32, shape.kernel_w as u32],
                [1, 1],
                [1, 1],
            )
        } else {
            phase.storage_read_with_layout::<F32, 2>(activation_layout(m, k, layout))
        };
        let b = phase.quantized_matrix(format, k as u32, n as u32);
        let y = phase.storage_write_with_layout::<F32, 2>(output_layout(m, n, layout));
        if m == 1 {
            phase.qgemv::<4, 64>(&a, &b, &y, 4, 1);
        } else if let Some(tile) = tile_override {
            match (tile.bm, tile.bn, tile.bk) {
                (32, 32, 32) => phase.qmatmul::<32, 32, 32>(&a, &b, &y, 4),
                (32, 64, 32) => phase.qmatmul::<32, 64, 32>(&a, &b, &y, 4),
                (64, 64, 32) => phase.qmatmul::<64, 64, 32>(&a, &b, &y, 4),
                (32, 128, 32) => phase.qmatmul::<32, 128, 32>(&a, &b, &y, 4),
                (64, 128, 32) => phase.qmatmul::<64, 128, 32>(&a, &b, &y, 4),
                (128, 64, 32) => phase.qmatmul::<128, 64, 32>(&a, &b, &y, 4),
                (128, 96, 32) => phase.qmatmul::<128, 96, 32>(&a, &b, &y, 4),
                (64, 192, 32) => phase.qmatmul::<64, 192, 32>(&a, &b, &y, 4),
                (128, 128, 16) => phase.qmatmul::<128, 128, 16>(&a, &b, &y, 4),
                (128, 128, 32) => phase.qmatmul::<128, 128, 32>(&a, &b, &y, 4),
                (8, 8, 4) => phase.qmatmul::<8, 8, 4>(&a, &b, &y, 4),
                (8, 4, 8) => phase.qmatmul::<8, 4, 8>(&a, &b, &y, 4),
                (4, 8, 8) => phase.qmatmul::<4, 8, 8>(&a, &b, &y, 4),
                (16, 4, 4) => phase.qmatmul::<16, 4, 4>(&a, &b, &y, 4),
                (4, 16, 4) => phase.qmatmul::<4, 16, 4>(&a, &b, &y, 4),
                _ => {
                    panic!(
                        "typed tile bench supports cooperative BM/BN/BK of 32/32/32, 32/64/32, 64/64/32, 32/128/32, 64/128/32, 128/64/32, 128/96/32, 64/192/32, 128/128/16, 128/128/32 or scalar 8/8/4, 8/4/8, 4/8/8, 16/4/4, 4/16/4"
                    )
                }
            }
        } else {
            phase.qmatmul::<64, 64, 32>(&a, &b, &y, 4);
        }
    })
}

fn qgemm_tile_shape(format: GgmlQuantFormat) -> TilePlan {
    match format {
        GgmlQuantFormat::Q4_0
        | GgmlQuantFormat::Q4_1
        | GgmlQuantFormat::Q5_0
        | GgmlQuantFormat::Q5_1
        | GgmlQuantFormat::Q8_0
        | GgmlQuantFormat::Q8_1
        | GgmlQuantFormat::Q2K
        | GgmlQuantFormat::Q3K
        | GgmlQuantFormat::Q4K
        | GgmlQuantFormat::Q5K
        | GgmlQuantFormat::Q6K
        | GgmlQuantFormat::Q8K => TilePlan {
            bm: 64,
            bn: 64,
            bk: 32,
        },
    }
}

fn qgemv_cols_per_workgroup(format: GgmlQuantFormat) -> usize {
    format.qgemv_cols_per_workgroup() as usize
}

fn pack_ones(format: GgmlQuantFormat, rows: usize, cols: usize) -> Vec<u32> {
    assert_eq!(rows % format.block_elements() as usize, 0);
    let block_words = format.block_words() as usize;
    let blocks_per_col = rows / format.block_elements() as usize;
    let mut words = Vec::with_capacity(cols * blocks_per_col * block_words);
    for _ in 0..cols * blocks_per_col {
        match format {
            GgmlQuantFormat::Q4_0 => pack_q4_0_ones(&mut words),
            GgmlQuantFormat::Q4_1 => pack_q4_1_ones(&mut words),
            GgmlQuantFormat::Q5_0 => pack_q5_0_ones(&mut words),
            GgmlQuantFormat::Q5_1 => pack_q5_1_ones(&mut words),
            GgmlQuantFormat::Q8_0 => pack_q8_0_ones(&mut words),
            GgmlQuantFormat::Q8_1 => pack_q8_1_ones(&mut words),
            GgmlQuantFormat::Q2K => pack_q2k_ones(&mut words),
            GgmlQuantFormat::Q3K => pack_q3k_ones(&mut words),
            GgmlQuantFormat::Q4K => pack_q4k_ones(&mut words),
            GgmlQuantFormat::Q5K => pack_q5k_ones(&mut words),
            GgmlQuantFormat::Q6K => pack_q6k_ones(&mut words),
            GgmlQuantFormat::Q8K => pack_q8k_ones(&mut words),
        }
    }
    words
}

fn pack_q4_0_ones(words: &mut Vec<u32>) {
    words.push(0.25_f32.to_bits());
    words.extend([0xcccc_cccc; 4]);
}

fn pack_q4_1_ones(words: &mut Vec<u32>) {
    words.push(0.25_f32.to_bits());
    words.push(0.0_f32.to_bits());
    words.extend([0x4444_4444; 4]);
}

fn pack_q5_0_ones(words: &mut Vec<u32>) {
    words.push(0.25_f32.to_bits());
    words.push(0xffff_ffff);
    words.extend([0x4444_4444; 4]);
}

fn pack_q5_1_ones(words: &mut Vec<u32>) {
    words.push(0.25_f32.to_bits());
    words.push(0.0_f32.to_bits());
    words.push(0);
    words.extend([0x4444_4444; 4]);
}

fn pack_q8_0_ones(words: &mut Vec<u32>) {
    words.push(0.25_f32.to_bits());
    words.extend([0x0404_0404; 8]);
}

fn pack_q8_1_ones(words: &mut Vec<u32>) {
    words.push(0.25_f32.to_bits());
    words.push(0.0_f32.to_bits());
    words.extend([0x0404_0404; 8]);
}

fn pack_q2k_ones(words: &mut Vec<u32>) {
    words.extend([0x0404_0404; 4]);
    words.extend([0x5555_5555; 16]);
    words.push(0.25_f32.to_bits());
    words.push(0.0_f32.to_bits());
}

fn pack_q3k_ones(words: &mut Vec<u32>) {
    words.extend([0xffff_ffff; 8]);
    words.extend([0xffff_ffff; 16]);
    words.push(0x1111_1111);
    words.push(0x1111_1111);
    words.push(0xaaaa_aaaa);
    words.push((1.0_f32 / 3.0).to_bits());
}

fn pack_q4k_ones(words: &mut Vec<u32>) {
    words.push(0.25_f32.to_bits());
    words.push(0.0_f32.to_bits());
    words.push(0x0404_0404);
    words.push(0);
    words.push(0x0404_0404);
    words.extend([0x1111_1111; 32]);
}

fn pack_q5k_ones(words: &mut Vec<u32>) {
    words.push(0.25_f32.to_bits());
    words.push(0.0_f32.to_bits());
    words.push(0x0404_0404);
    words.push(0);
    words.push(0x0404_0404);
    words.extend([0; 8]);
    words.extend([0x1111_1111; 32]);
}

fn pack_q6k_ones(words: &mut Vec<u32>) {
    words.extend([0x4444_4444; 32]);
    words.extend([0xaaaa_aaaa; 16]);
    words.extend([0x0101_0101; 4]);
    words.push(0.25_f32.to_bits());
}

fn pack_q8k_ones(words: &mut Vec<u32>) {
    words.push(0.25_f32.to_bits());
    words.extend([0x0404_0404; 64]);
    words.extend([0; 8]);
}

fn storage_buffer_f32(
    device: &wgpu::Device,
    label: &'static str,
    data: &[f32],
    extra_usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra_usage,
    })
}

fn storage_buffer_u32(
    device: &wgpu::Device,
    label: &'static str,
    data: &[u32],
    extra_usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra_usage,
    })
}

fn storage_bindings(read_only: &[bool]) -> Vec<wgpu::BindGroupLayoutEntry> {
    read_only
        .iter()
        .enumerate()
        .map(|(binding, read_only)| wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: *read_only,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}

fn make_a(m: usize, k_size: usize) -> Vec<f32> {
    (0..m)
        .flat_map(|row| {
            (0..k_size).map(move |k| {
                let value = ((row * 13 + k * 7) % 31) as i32 - 15;
                value as f32 * 0.03125
            })
        })
        .collect()
}

fn activation_data(m: usize, k: usize, layout: LayoutVariant) -> (Vec<f32>, Vec<f32>) {
    if matches!(layout, LayoutVariant::Im2Col) {
        let shape = im2col_shape(m, k);
        let input = make_a(shape.input_h * shape.input_w, shape.channels);
        let a = im2col_nhwc_matrix(&input, &shape);
        return (a, input);
    }

    let a = make_a(m, k);
    let a_physical = pack_matrix(&a, m, k, &activation_layout(m, k, layout));
    (a, a_physical)
}

fn matrix_layout(rows: usize, cols: usize, layout: LayoutVariant) -> Layout {
    let shape = shape([rows, cols]);
    match layout {
        LayoutVariant::Contiguous | LayoutVariant::Skewed | LayoutVariant::Im2Col => {
            Layout::contiguous(MemoryLevel::Storage, shape)
        }
        LayoutVariant::Padded if rows == 1 && cols > 1 => Layout::strided(
            MemoryLevel::Storage,
            shape,
            Strides::new([(cols * 2 + 3) as u32, 2]),
        ),
        LayoutVariant::Padded => Layout::strided(
            MemoryLevel::Storage,
            shape,
            Strides::new([(cols + 5) as u32, 1]),
        ),
        LayoutVariant::Transposed => {
            let strides = Strides::col_major_for(&shape);
            Layout::strided(MemoryLevel::Storage, shape, strides)
        }
    }
}

fn activation_layout(rows: usize, cols: usize, layout: LayoutVariant) -> Layout {
    if matches!(layout, LayoutVariant::Im2Col) {
        return im2col_input_layout(&im2col_shape(rows, cols));
    }
    if matches!(layout, LayoutVariant::Skewed) {
        let shape = shape([rows, cols]);
        return Layout::strided(
            MemoryLevel::Storage,
            shape,
            Strides::new([2, (rows * 2 + 7) as u32]),
        );
    }
    matrix_layout(rows, cols, layout)
}

fn output_layout(rows: usize, cols: usize, layout: LayoutVariant) -> Layout {
    if matches!(layout, LayoutVariant::Skewed | LayoutVariant::Im2Col) {
        return Layout::contiguous(MemoryLevel::Storage, shape([rows, cols]));
    }
    matrix_layout(rows, cols, layout)
}

#[derive(Copy, Clone)]
struct Im2ColShape {
    input_h: usize,
    input_w: usize,
    channels: usize,
    out_h: usize,
    out_w: usize,
    kernel_h: usize,
    kernel_w: usize,
}

fn im2col_shape(m: usize, k: usize) -> Im2ColShape {
    if m == GEMM_M && k == GEMM_K {
        return Im2ColShape {
            input_h: 35,
            input_w: 35,
            channels: 64,
            out_h: 32,
            out_w: 32,
            kernel_h: 4,
            kernel_w: 4,
        };
    }
    assert_eq!(m, 1, "im2col benchmark shape only supports GEMM or GEMV");
    Im2ColShape {
        input_h: 1,
        input_w: 1,
        channels: k,
        out_h: 1,
        out_w: 1,
        kernel_h: 1,
        kernel_w: 1,
    }
}

fn im2col_input_layout(spec: &Im2ColShape) -> Layout {
    Layout::contiguous(
        MemoryLevel::Storage,
        shape([1, spec.input_h, spec.input_w, spec.channels]),
    )
}

fn im2col_nhwc_matrix(input: &[f32], shape: &Im2ColShape) -> Vec<f32> {
    let m = shape.out_h * shape.out_w;
    let k = shape.kernel_h * shape.kernel_w * shape.channels;
    let mut matrix = vec![0.0; m * k];
    for oh in 0..shape.out_h {
        for ow in 0..shape.out_w {
            let row = oh * shape.out_w + ow;
            for kh in 0..shape.kernel_h {
                for kw in 0..shape.kernel_w {
                    for c in 0..shape.channels {
                        let col = (kh * shape.kernel_w + kw) * shape.channels + c;
                        let input_index =
                            ((oh + kh) * shape.input_w + ow + kw) * shape.channels + c;
                        matrix[row * k + col] = input[input_index];
                    }
                }
            }
        }
    }
    matrix
}

fn allocation_len(layout: &Layout) -> usize {
    layout.allocation_element_count().get() as usize
}

fn matrix_index(layout: &Layout, row: usize, col: usize) -> usize {
    let strides = layout.strides().values();
    row * strides[0] as usize + col * strides[1] as usize
}

fn matrix_value(values: &[f32], layout: &Layout, row: usize, col: usize) -> f32 {
    values[matrix_index(layout, row, col)]
}

fn pack_matrix(logical: &[f32], rows: usize, cols: usize, layout: &Layout) -> Vec<f32> {
    assert_eq!(logical.len(), rows * cols);
    let mut physical = vec![17.0; allocation_len(layout)];
    for row in 0..rows {
        for col in 0..cols {
            physical[matrix_index(layout, row, col)] = logical[row * cols + col];
        }
    }
    physical
}

fn sampled_max_abs_error(
    actual: &[f32],
    y_layout: &Layout,
    a: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> f32 {
    sample_rows(m)
        .into_iter()
        .map(|row| {
            let expected = cpu_dot_ones(a, k, row);
            sample_cols(n)
                .into_iter()
                .map(|col| (matrix_value(actual, y_layout, row, col) - expected).abs())
                .fold(0.0, f32::max)
        })
        .fold(0.0, f32::max)
}

fn first_sample_mismatch(
    actual: &[f32],
    y_layout: &Layout,
    a: &[f32],
    m: usize,
    n: usize,
    k: usize,
    tolerance: f32,
) -> Option<(usize, usize, f32, f32)> {
    for row in sample_rows(m) {
        let expected = cpu_dot_ones(a, k, row);
        for col in sample_cols(n) {
            let actual = matrix_value(actual, y_layout, row, col);
            if (actual - expected).abs() > tolerance {
                return Some((row, col, actual, expected));
            }
        }
    }
    None
}

fn sample_rows(m: usize) -> Vec<usize> {
    let mut rows = vec![0, m / 2, m.saturating_sub(1)];
    for i in 0..32 {
        rows.push((i * 97) % m);
    }
    rows
}

fn sample_cols(n: usize) -> Vec<usize> {
    let mut cols = vec![0, n / 2, n.saturating_sub(1)];
    for i in 0..32 {
        cols.push((i * 193) % n);
    }
    cols
}

fn cpu_dot_ones(a: &[f32], k_size: usize, row: usize) -> f32 {
    let mut sum = 0.0;
    for k in 0..k_size {
        sum += a[row * k_size + k];
    }
    sum
}

fn parse_mode() -> Result<BenchMode, Box<dyn std::error::Error>> {
    match env::args().nth(1).as_deref() {
        None | Some("gemm") => Ok(BenchMode::Gemm),
        Some("gemv") => Ok(BenchMode::Gemv),
        Some(other) => Err(format!("unknown mode {other:?}; expected gemm or gemv").into()),
    }
}

fn parse_formats() -> Result<Vec<GgmlQuantFormat>, Box<dyn std::error::Error>> {
    let Some(raw) = env::args().nth(2) else {
        return Ok(all_formats().to_vec());
    };
    raw.split(',').map(parse_format).collect()
}

fn parse_layout() -> Result<(LayoutVariant, bool), Box<dyn std::error::Error>> {
    match env::args().nth(3).as_deref() {
        Some("contiguous") => Ok((LayoutVariant::Contiguous, true)),
        Some("padded") => Ok((LayoutVariant::Padded, true)),
        Some("transposed") => Ok((LayoutVariant::Transposed, true)),
        Some("skewed") => Ok((LayoutVariant::Skewed, true)),
        Some("im2col") => Ok((LayoutVariant::Im2Col, true)),
        _ => Ok((LayoutVariant::Contiguous, false)),
    }
}

fn parse_shape_override(
    mode: BenchMode,
    layout_arg_present: bool,
) -> Result<(usize, usize, usize), Box<dyn std::error::Error>> {
    let args = env::args().collect::<Vec<_>>();
    let start = if layout_arg_present { 4 } else { 3 };
    let defaults = match mode {
        BenchMode::Gemm => (GEMM_M, GEMM_N, GEMM_K),
        BenchMode::Gemv => (1, GEMV_N, GEMV_K),
    };
    if matches!(mode, BenchMode::Gemv) {
        if args.len() <= start {
            return Ok(defaults);
        }
        if args.len() != start + 3 {
            return Err(
                "usage: bench_qmatmul gemv [formats] [contiguous|padded|transposed|skewed|im2col] [M N K]"
                    .into(),
            );
        }
        let m = args[start].parse::<usize>()?;
        let n = args[start + 1].parse::<usize>()?;
        let k = args[start + 2].parse::<usize>()?;
        if m == 0 || n == 0 || k == 0 {
            return Err("benchmark dimensions must be non-zero".into());
        }
        if m != 1 {
            return Err("gemv benchmark requires M=1".into());
        }
        return Ok((m, n, k));
    }
    if args.len() <= start + 3 {
        return Ok(defaults);
    }
    if args.len() != start + 6 {
        return Err(
            "usage: bench_qmatmul [gemm|gemv] [formats] [contiguous|padded|transposed|skewed|im2col] [BM BN BK [M N K]]"
                .into(),
        );
    }
    let m = args[start + 3].parse::<usize>()?;
    let n = args[start + 4].parse::<usize>()?;
    let k = args[start + 5].parse::<usize>()?;
    if m == 0 || n == 0 || k == 0 {
        return Err("benchmark dimensions must be non-zero".into());
    }
    if matches!(mode, BenchMode::Gemv) && m != 1 {
        return Err("gemv benchmark requires M=1".into());
    }
    Ok((m, n, k))
}

fn parse_tile_override(
    mode: BenchMode,
    layout_arg_present: bool,
    shape: (usize, usize, usize),
) -> Result<Option<TilePlan>, Box<dyn std::error::Error>> {
    let args = env::args().collect::<Vec<_>>();
    let start = if layout_arg_present { 4 } else { 3 };
    if args.len() <= start {
        return Ok(None);
    }
    if !matches!(mode, BenchMode::Gemm) {
        return Ok(None);
    }
    if args.len() != start + 3 && args.len() != start + 6 {
        return Err(
            "usage: bench_qmatmul [gemm|gemv] [formats] [contiguous|padded|transposed|skewed|im2col] [BM BN BK [M N K]]"
                .into(),
        );
    }
    let bm = args[start].parse::<usize>()?;
    let bn = args[start + 1].parse::<usize>()?;
    let bk = args[start + 2].parse::<usize>()?;
    let (m, n, k) = shape;
    if bm == 0 || bn == 0 || bk == 0 || m % bm != 0 || n % bn != 0 || k % bk != 0 {
        return Err(format!("tile must divide {m}x{n}x{k}; got BM={bm} BN={bn} BK={bk}",).into());
    }
    if !matches!(
        (bm, bn, bk),
        (32, 32, 32)
            | (32, 64, 32)
            | (64, 64, 32)
            | (32, 128, 32)
            | (64, 128, 32)
            | (128, 64, 32)
            | (128, 96, 32)
            | (64, 192, 32)
            | (128, 128, 16)
            | (128, 128, 32)
            | (8, 8, 4)
            | (8, 4, 8)
            | (4, 8, 8)
            | (16, 4, 4)
            | (4, 16, 4)
    ) {
        return Err(format!(
            "typed tile bench supports cooperative BM/BN/BK of 32/32/32, 32/64/32, 64/64/32, 32/128/32, 64/128/32, 128/64/32, 128/96/32, 64/192/32, 128/128/16, 128/128/32 or scalar 8/8/4, 8/4/8, 4/8/8, 16/4/4, 4/16/4; got {bm}/{bn}/{bk}",
        )
        .into());
    }
    Ok(Some(TilePlan { bm, bn, bk }))
}

fn parse_format(raw: &str) -> Result<GgmlQuantFormat, Box<dyn std::error::Error>> {
    match raw {
        "q4_0" | "Q4_0" => Ok(GgmlQuantFormat::Q4_0),
        "q4_1" | "Q4_1" => Ok(GgmlQuantFormat::Q4_1),
        "q5_0" | "Q5_0" => Ok(GgmlQuantFormat::Q5_0),
        "q5_1" | "Q5_1" => Ok(GgmlQuantFormat::Q5_1),
        "q8_0" | "Q8_0" => Ok(GgmlQuantFormat::Q8_0),
        "q8_1" | "Q8_1" => Ok(GgmlQuantFormat::Q8_1),
        "q2k" | "Q2K" => Ok(GgmlQuantFormat::Q2K),
        "q3k" | "Q3K" => Ok(GgmlQuantFormat::Q3K),
        "q4k" | "Q4K" => Ok(GgmlQuantFormat::Q4K),
        "q5k" | "Q5K" => Ok(GgmlQuantFormat::Q5K),
        "q6k" | "Q6K" => Ok(GgmlQuantFormat::Q6K),
        "q8k" | "Q8K" => Ok(GgmlQuantFormat::Q8K),
        other => Err(format!("unknown quant format {other:?}").into()),
    }
}

fn all_formats() -> &'static [GgmlQuantFormat] {
    &[
        GgmlQuantFormat::Q4_0,
        GgmlQuantFormat::Q4_1,
        GgmlQuantFormat::Q5_0,
        GgmlQuantFormat::Q5_1,
        GgmlQuantFormat::Q8_0,
        GgmlQuantFormat::Q8_1,
        GgmlQuantFormat::Q2K,
        GgmlQuantFormat::Q3K,
        GgmlQuantFormat::Q4K,
        GgmlQuantFormat::Q5K,
        GgmlQuantFormat::Q6K,
        GgmlQuantFormat::Q8K,
    ]
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[index]
}

fn byte_len<T>(len: usize) -> u64 {
    (len * std::mem::size_of::<T>()) as u64
}

fn shape<const R: usize>(dims: [usize; R]) -> Shape {
    Shape::new(dims.map(|dim| dim as u32))
}
