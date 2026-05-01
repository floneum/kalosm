use std::{borrow::Cow, sync::mpsc, time::Instant};

use phase_token_prototype::{tile, KernelIr, Shape, WorkgroupAxis, F32};
use wgpu::util::DeviceExt;

const WARMUP_BATCHES: usize = 2;
const MEASURED_BATCHES: usize = 5;
const DISPATCHES_PER_BATCH: usize = 50;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    pollster::block_on(run())
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    bench::<128>(100).await?;
    Ok(())
}

async fn bench<const BLOCK: usize>(size: usize) -> Result<(), Box<dyn std::error::Error>> {
    let harness = Harness::new::<BLOCK>(size).await?;

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
    let max_abs_error = max_abs_error(&harness.input, &output, size);
    if max_abs_error > 1.0e-5 {
        return Err(format!("tile softmax mismatch: max_abs_error={max_abs_error}").into());
    }

    samples.sort_by(f64::total_cmp);
    let mean_s = samples.iter().sum::<f64>() / samples.len() as f64;
    let p50_s = percentile(&samples, 0.50);
    let p90_s = percentile(&samples, 0.90);
    println!("adapter: {}", harness.adapter_info.name);
    println!("bench_tile_softmax: {size}x{size}, BLOCK={BLOCK}");
    println!(
        "dispatches: {} measured, {WARMUP_BATCHES} warmup batches",
        MEASURED_BATCHES * DISPATCHES_PER_BATCH
    );
    println!("max_abs_error: {max_abs_error:.6}");
    println!("mean_dispatch_time_us: {:.3}", mean_s * 1.0e6);
    println!("p50_dispatch_time_us: {:.3}", p50_s * 1.0e6);
    println!("p90_dispatch_time_us: {:.3}", p90_s * 1.0e6);
    println!();
    Ok(())
}

struct Harness {
    size: usize,
    adapter_info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    output_buffer: wgpu::Buffer,
    readback: wgpu::Buffer,
    input: Vec<f32>,
}

impl Harness {
    async fn new<const BLOCK: usize>(size: usize) -> Result<Self, Box<dyn std::error::Error>> {
        let ir = softmax_ir::<BLOCK>(size as u32, size as u32);
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
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("phase-token-prototype tile softmax bench device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                ..Default::default()
            })
            .await?;

        let input = softmax_input(size);
        let input_buffer = storage_buffer(&device, "X", &input, wgpu::BufferUsages::empty());
        let output_buffer = storage_buffer(
            &device,
            "Y",
            &vec![0.0_f32; size * size],
            wgpu::BufferUsages::COPY_SRC,
        );
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Y readback"),
            size: byte_len::<f32>(size * size),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = unsafe {
            device.create_shader_module_trusted(
                wgpu::ShaderModuleDescriptor {
                    label: Some("lowered tile softmax"),
                    source: wgpu::ShaderSource::Naga(Cow::Owned(lowered.module().clone())),
                },
                wgpu::ShaderRuntimeChecks::unchecked(),
            )
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile softmax buffers"),
            entries: &storage_bindings(&[true, false]),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tile softmax pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("tile softmax pipeline"),
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
            label: Some("tile softmax bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            size,
            adapter_info,
            device,
            queue,
            pipeline,
            bind_group,
            output_buffer,
            readback,
            input,
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
                label: Some("tile softmax bench encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tile softmax bench pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            for _ in 0..dispatches {
                pass.dispatch_workgroups(1, self.size as u32, 1);
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
                label: Some("tile softmax readback encoder"),
            });
        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &self.readback,
            0,
            byte_len::<f32>(self.size * self.size),
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

fn softmax_ir<const BLOCK: usize>(rows: u32, cols: u32) -> KernelIr {
    assert!(cols <= BLOCK as u32, "one tile covers one row");
    tile::build(|phase| {
        let x = phase.storage_read::<F32, 2>(Shape::new([rows, cols]));
        let y = phase.storage_write::<F32, 2>(Shape::new([rows, cols]));

        phase.program_grid::<BLOCK>([1, rows, 1], |program| {
            let row = program.program_id(WorkgroupAxis::Y);
            let col = program.arange();
            let mask = col.lt(cols);
            let values = program.load(x.at(&row, &col), mask.clone(), -3.4028235e38);
            let max = program.reduce_max(values.clone());
            let exp = (values - max).exp();
            let sum = program.reduce_sum(exp.clone());

            program.store(y.at(row, col), exp / sum, mask);
        });
    })
}

fn softmax_input(size: usize) -> Vec<f32> {
    (0..size * size)
        .map(|index| ((index * 13 + index / 7) % 31) as f32 / 8.0)
        .collect()
}

fn max_abs_error(input: &[f32], output: &[f32], size: usize) -> f32 {
    let mut max_abs = 0.0_f32;
    for row in 0..size {
        let row_input = &input[row * size..(row + 1) * size];
        let row_output = &output[row * size..(row + 1) * size];
        let max = row_input.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum = row_input
            .iter()
            .map(|value| (*value - max).exp())
            .sum::<f32>();
        for (value, actual) in row_input.iter().zip(row_output) {
            let expected = (*value - max).exp() / sum;
            max_abs = max_abs.max((actual - expected).abs());
        }
    }
    max_abs
}

fn storage_buffer(
    device: &wgpu::Device,
    label: &str,
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

fn byte_len<T>(elements: usize) -> u64 {
    (elements * std::mem::size_of::<T>()) as u64
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[index]
}
