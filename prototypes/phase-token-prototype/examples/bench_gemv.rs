use std::{borrow::Cow, sync::mpsc, time::Instant};

use phase_token_prototype::{build, KernelIr, Shape, F32};
use wgpu::util::DeviceExt;

const M: usize = 4096;
const K: usize = 4096;
const ROWS_PER_WORKGROUP: usize = 4;
const WORKGROUP_PARTIALS: usize = 128 * ROWS_PER_WORKGROUP;
const VECTOR_WIDTH: u32 = 1;
const WARMUP_BATCHES: usize = 4;
const MEASURED_BATCHES: usize = 10;
const DISPATCHES_PER_BATCH: usize = 100;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    pollster::block_on(run())
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let harness = Harness::new().await?;

    for _ in 0..WARMUP_BATCHES {
        harness.run_batch(DISPATCHES_PER_BATCH)?;
    }

    let mut samples = Vec::with_capacity(MEASURED_BATCHES);
    for _ in 0..MEASURED_BATCHES {
        let command_buffer = harness.encode_batch(DISPATCHES_PER_BATCH);
        let started = Instant::now();
        harness.submit_and_wait(command_buffer)?;
        samples.push(started.elapsed().as_secs_f64() / DISPATCHES_PER_BATCH as f64);
    }

    let output = harness.read_output()?;
    let max_abs_error = sampled_max_abs_error(&output, &harness.a, &harness.x);
    if max_abs_error > 1.0e-3 {
        let (row, actual, expected) = first_sample_mismatch(
            &output, &harness.a, &harness.x, 1.0e-3,
        )
        .unwrap_or((0, output[0], cpu_dot(&harness.a, &harness.x, 0)));
        return Err(format!(
            "gemv mismatch at row {row}: gpu={actual} cpu={expected} max_abs_error={max_abs_error}",
        )
        .into());
    }

    samples.sort_by(f64::total_cmp);
    let total_dispatches = MEASURED_BATCHES * DISPATCHES_PER_BATCH;
    let mean_s = samples.iter().sum::<f64>() / samples.len() as f64;
    let p50_s = percentile(&samples, 0.50);
    let p90_s = percentile(&samples, 0.90);
    let min_s = samples[0];
    let max_s = samples[samples.len() - 1];
    let flops_per_dispatch = 2.0 * M as f64 * K as f64;
    let bytes_per_dispatch = ((M * K + K + M) * std::mem::size_of::<f32>()) as f64;

    println!(
        "adapter: {} ({:?})",
        harness.adapter_info.name, harness.adapter_info.backend
    );
    println!(
        "bench_gemv: {M}x{K} f32 matrix times {K} vector, {} workgroups per dispatch",
        M / ROWS_PER_WORKGROUP
    );
    println!("dispatches: {total_dispatches} measured, {WARMUP_BATCHES} warmup batches");
    println!("max_abs_error: {max_abs_error:.6}");
    println!("rows_per_workgroup: {ROWS_PER_WORKGROUP}");
    println!("vector_width: {VECTOR_WIDTH}");
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
        "effective_bandwidth_gb_s: {:.6}",
        bytes_per_dispatch / mean_s / 1.0e9
    );
    println!("note: GEMV uses the typed row-parallel Gemv IR op.");
    println!("note: this times pre-encoded batch submit-to-completion on the host.");

    Ok(())
}

struct Harness {
    adapter_info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    y_buffer: wgpu::Buffer,
    readback: wgpu::Buffer,
    a: Vec<f32>,
    x: Vec<f32>,
}

impl Harness {
    async fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let ir = gemv_ir();
        let lowered = ir.lower_to_naga()?;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await?;
        let adapter_info = adapter.get_info();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("phase-token-prototype gemv bench device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })
            .await?;

        let a = make_a();
        let x = make_x();
        let a_buffer = storage_buffer(&device, "A", &a, wgpu::BufferUsages::empty());
        let x_buffer = storage_buffer(&device, "x", &x, wgpu::BufferUsages::empty());
        let y_buffer = storage_buffer(
            &device,
            "y",
            &vec![0.0_f32; M],
            wgpu::BufferUsages::COPY_SRC,
        );
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("y readback"),
            size: byte_len::<f32>(M),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = unsafe {
            device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some("lowered gemv"),
                    source: wgpu::ShaderSource::Naga(Cow::Owned(lowered.module().clone())),
                },
                wgpu::ShaderRuntimeChecks::unchecked(),
            )
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gemv buffers"),
            entries: &storage_bindings(&[true, true, false]),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gemv pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gemv pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gemv bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: y_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            adapter_info,
            device,
            queue,
            pipeline,
            bind_group,
            y_buffer,
            readback,
            a,
            x,
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
                label: Some("gemv bench encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gemv bench pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            for _ in 0..dispatches {
                pass.dispatch_workgroups((M / ROWS_PER_WORKGROUP) as u32, 1, 1);
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
                label: Some("gemv readback encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.y_buffer, 0, &self.readback, 0, byte_len::<f32>(M));
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

fn gemv_ir() -> KernelIr {
    build(|mut phase| {
        let a_full = phase.storage_tensor_read::<F32>(shape([M, K]));
        let x_full = phase.storage_tensor_read::<F32>(shape([K, 1]));
        let y_full = phase.storage_tensor::<F32>(shape([M, 1]));
        let partials = phase.alloc_workgroup_tile::<F32>(shape([WORKGROUP_PARTIALS]));
        phase.gemv_tiled(
            &a_full,
            &x_full,
            &y_full,
            partials,
            ROWS_PER_WORKGROUP as u32,
            VECTOR_WIDTH,
        );
        phase.finish()
    })
}

fn storage_buffer(
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

fn make_a() -> Vec<f32> {
    (0..M)
        .flat_map(|row| {
            (0..K).map(move |k| {
                let value = ((row * 13 + k * 7) % 31) as i32 - 15;
                value as f32 * 0.03125
            })
        })
        .collect()
}

fn make_x() -> Vec<f32> {
    (0..K)
        .map(|k| {
            let value = ((k * 11) % 29) as i32 - 14;
            value as f32 * 0.03125
        })
        .collect()
}

fn sampled_max_abs_error(actual: &[f32], a: &[f32], x: &[f32]) -> f32 {
    sample_rows()
        .into_iter()
        .map(|row| (actual[row] - cpu_dot(a, x, row)).abs())
        .fold(0.0, f32::max)
}

fn first_sample_mismatch(
    actual: &[f32],
    a: &[f32],
    x: &[f32],
    tolerance: f32,
) -> Option<(usize, f32, f32)> {
    for row in sample_rows() {
        let expected = cpu_dot(a, x, row);
        if (actual[row] - expected).abs() > tolerance {
            return Some((row, actual[row], expected));
        }
    }
    None
}

fn sample_rows() -> Vec<usize> {
    let mut rows = vec![0, 1, M / 2, M - 2, M - 1];
    for i in 0..64 {
        rows.push((i * 97) % M);
    }
    rows
}

fn cpu_dot(a: &[f32], x: &[f32], row: usize) -> f32 {
    let mut sum = 0.0;
    for k in 0..K {
        sum += a[row * K + k] * x[k];
    }
    sum
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
