//! GPU runtime for executing and benchmarking `DispatchProgram`s via wgpu.

use crate::{DispatchProgram, TensorIr, naga_codegen};
use std::borrow::Cow;
use std::collections::HashMap;
use std::time::Instant;
use wgpu::util::DeviceExt;

const MAX_DISPATCH_WORKGROUPS_PER_DIMENSION: u32 = 65_535;

/// A GPU context that can execute `DispatchProgram`s.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
}

/// Benchmark configuration for a GPU program.
#[derive(Debug, Clone, Copy)]
pub struct ProgramBenchmarkConfig {
    /// Warmup dispatch submissions that are not included in timing results.
    pub warmup_runs: u32,
    /// Number of timestamped timing submissions to record.
    pub timing_runs: u32,
}

impl Default for ProgramBenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_runs: 3,
            timing_runs: 10,
        }
    }
}

/// GPU-only timing results in microseconds.
#[derive(Debug, Clone)]
pub struct GpuBenchmarkResult {
    pub samples_gpu_us: Vec<f64>,
    pub min_gpu_us: f64,
    pub median_gpu_us: f64,
    pub max_gpu_us: f64,
}

/// Host wall-clock timing results in microseconds.
#[derive(Debug, Clone)]
pub struct HostBenchmarkResult {
    pub samples_host_us: Vec<f64>,
    pub min_host_us: f64,
    pub median_host_us: f64,
    pub max_host_us: f64,
}

struct TimestampQueryResources {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    destination_buffer: wgpu::Buffer,
}

impl TimestampQueryResources {
    fn new(device: &wgpu::Device, count: u32) -> Self {
        let bytes = (count as u64) * (std::mem::size_of::<u64>() as u64);
        Self {
            query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("tensor_ir_benchmark_query_set"),
                count,
                ty: wgpu::QueryType::Timestamp,
            }),
            resolve_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tensor_ir_benchmark_query_resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
                mapped_at_creation: false,
            }),
            destination_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tensor_ir_benchmark_query_readback"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }),
        }
    }

    fn encode_resolve(&self, encoder: &mut wgpu::CommandEncoder, count: u32) {
        encoder.resolve_query_set(&self.query_set, 0..count, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.destination_buffer,
            0,
            (count as u64) * (std::mem::size_of::<u64>() as u64),
        );
    }

    fn read_results(&self, device: &wgpu::Device) -> Result<Vec<u64>, String> {
        let slice = self.destination_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .map_err(|_| "timestamp map channel closed".to_string())?
            .map_err(|err| format!("timestamp map failed: {err:?}"))?;

        let data = slice.get_mapped_range();
        let timestamps = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.destination_buffer.unmap();
        Ok(timestamps)
    }
}

fn runtime_required_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    let supported = adapter.features();
    let mut required = wgpu::Features::empty();
    if supported.contains(wgpu::Features::SUBGROUP) {
        required |= wgpu::Features::SUBGROUP;
    }
    if supported.contains(wgpu::Features::SUBGROUP_BARRIER) {
        required |= wgpu::Features::SUBGROUP_BARRIER;
    }
    if supported.contains(wgpu::Features::TIMESTAMP_QUERY) {
        required |= wgpu::Features::TIMESTAMP_QUERY;
    }
    required
}

impl GpuContext {
    /// Create a new GPU context, requesting a default adapter and device.
    pub fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("no GPU adapter found");
        let required_features = runtime_required_features(&adapter);
        let adapter_info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("tensor_ir runtime"),
            required_features,
            ..Default::default()
        }))
        .expect("failed to create device");
        Self {
            device,
            queue,
            adapter_info,
        }
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Measure GPU execution time using timestamp queries.
    ///
    /// This requires `TIMESTAMP_QUERY` support on the selected adapter.
    /// Returns microsecond timings summed across all dispatches in the program.
    pub fn benchmark(
        &self,
        program: &DispatchProgram,
        inputs: &[&[f32]],
        config: ProgramBenchmarkConfig,
    ) -> Result<GpuBenchmarkResult, String> {
        if program.dispatches.is_empty() {
            return Err("dispatch program has no runnable dispatches".into());
        }
        if config.timing_runs == 0 {
            return Err("timing_runs must be greater than zero".into());
        }
        if !self
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return Err("TIMESTAMP_QUERY is not supported by this GPU/device".into());
        }

        let prepared = self.prepare_program(program, inputs, false)?;
        for _ in 0..config.warmup_runs {
            self.submit_program(program, &prepared, None)?;
        }

        let timestamps_per_run = (program.dispatches.len() as u32) * 2;
        let timestamp_queries =
            TimestampQueryResources::new(&self.device, config.timing_runs * timestamps_per_run);

        for run_index in 0..config.timing_runs {
            self.submit_program(program, &prepared, Some((&timestamp_queries, run_index)))?;
        }

        let mut resolve_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("tensor_ir_benchmark_query_resolve_encoder"),
                });
        timestamp_queries.encode_resolve(
            &mut resolve_encoder,
            config.timing_runs * timestamps_per_run,
        );
        self.queue.submit(std::iter::once(resolve_encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

        let timestamp_period_ns = self.queue.get_timestamp_period() as f64;
        let timestamps = timestamp_queries.read_results(&self.device)?;
        let mut samples_gpu_us = Vec::with_capacity(config.timing_runs as usize);
        for run_index in 0..config.timing_runs as usize {
            let run_start = run_index * timestamps_per_run as usize;
            let mut elapsed_us = 0.0;
            for dispatch_index in 0..program.dispatches.len() {
                let start_tick = timestamps[run_start + dispatch_index * 2];
                let end_tick = timestamps[run_start + dispatch_index * 2 + 1];
                elapsed_us +=
                    end_tick.wrapping_sub(start_tick) as f64 * timestamp_period_ns / 1_000.0;
            }
            samples_gpu_us.push(elapsed_us);
        }

        samples_gpu_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(GpuBenchmarkResult {
            min_gpu_us: samples_gpu_us[0],
            median_gpu_us: samples_gpu_us[samples_gpu_us.len() / 2],
            max_gpu_us: *samples_gpu_us.last().unwrap(),
            samples_gpu_us,
        })
    }

    /// Measure host wall-clock time for prepared dispatch submissions.
    ///
    /// This prepares shaders, pipelines, bind groups, and buffers once, then
    /// measures command encoding, submission, and synchronization for each run.
    /// It does not read the output buffer back to the host.
    pub fn benchmark_host(
        &self,
        program: &DispatchProgram,
        inputs: &[&[f32]],
        config: ProgramBenchmarkConfig,
    ) -> Result<HostBenchmarkResult, String> {
        if program.dispatches.is_empty() {
            return Err("dispatch program has no runnable dispatches".into());
        }
        if config.timing_runs == 0 {
            return Err("timing_runs must be greater than zero".into());
        }

        let prepared = self.prepare_program(program, inputs, false)?;
        for _ in 0..config.warmup_runs {
            self.submit_program(program, &prepared, None)?;
        }

        let mut samples_host_us = Vec::with_capacity(config.timing_runs as usize);
        for _ in 0..config.timing_runs {
            let start = Instant::now();
            self.submit_program(program, &prepared, None)?;
            samples_host_us.push(start.elapsed().as_secs_f64() * 1e6);
        }

        samples_host_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(HostBenchmarkResult {
            min_host_us: samples_host_us[0],
            median_host_us: samples_host_us[samples_host_us.len() / 2],
            max_host_us: *samples_host_us.last().unwrap(),
            samples_host_us,
        })
    }

    /// Measure amortized host wall-clock time for repeated prepared dispatch
    /// submissions batched into one command buffer per sample.
    ///
    /// Each sample encodes `batch_size` sequential executions, submits once,
    /// waits once, then reports elapsed time divided by `batch_size`. This
    /// measures the runtime shape used when graph execution avoids a
    /// synchronize-after-every-op boundary.
    pub fn benchmark_host_batched(
        &self,
        program: &DispatchProgram,
        inputs: &[&[f32]],
        config: ProgramBenchmarkConfig,
        batch_size: u32,
    ) -> Result<HostBenchmarkResult, String> {
        if program.dispatches.is_empty() {
            return Err("dispatch program has no runnable dispatches".into());
        }
        if config.timing_runs == 0 {
            return Err("timing_runs must be greater than zero".into());
        }
        if batch_size == 0 {
            return Err("batch_size must be greater than zero".into());
        }

        let prepared = self.prepare_program(program, inputs, false)?;
        for _ in 0..config.warmup_runs {
            self.submit_program_repeated(program, &prepared, batch_size)?;
        }

        let mut samples_host_us = Vec::with_capacity(config.timing_runs as usize);
        for _ in 0..config.timing_runs {
            let start = Instant::now();
            self.submit_program_repeated(program, &prepared, batch_size)?;
            samples_host_us.push(start.elapsed().as_secs_f64() * 1e6 / f64::from(batch_size));
        }

        samples_host_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        Ok(HostBenchmarkResult {
            min_host_us: samples_host_us[0],
            median_host_us: samples_host_us[samples_host_us.len() / 2],
            max_host_us: *samples_host_us.last().unwrap(),
            samples_host_us,
        })
    }

    /// Execute a `DispatchProgram` on the GPU.
    ///
    /// `inputs` must match the number and order of device-buffer inputs
    /// declared by the program's first dispatch (e.g., `input_0`, `input_1`,
    /// ...). Multi-dispatch programs run their dispatches sequentially,
    /// wiring each dispatch's output buffer into downstream consumers that
    /// reference the same semantic e-class (see `resolve_dispatch_input_buffer`).
    ///
    /// Returns the final dispatch's output buffer contents as `Vec<f32>`.
    pub fn execute(&self, program: &DispatchProgram, inputs: &[&[f32]]) -> Vec<f32> {
        assert!(
            !program.dispatches.is_empty(),
            "dispatch program must contain at least one dispatch"
        );

        let prepared = self
            .prepare_program(program, inputs, true)
            .expect("failed to prepare dispatch program");
        let final_output = prepared
            .final_output_buffer
            .as_ref()
            .expect("program should produce a final output buffer");
        let staging_buffer = prepared
            .staging_buffer
            .as_ref()
            .expect("program should create a staging buffer");
        let output_bytes = final_output.size();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tensor_ir encoder"),
            });
        self.encode_program(&mut encoder, program, &prepared, None);
        encoder.copy_buffer_to_buffer(&final_output, 0, &staging_buffer, 0, output_bytes);
        self.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv()
            .expect("channel closed")
            .expect("buffer mapping failed");

        let data = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        result
    }

    fn prepare_program(
        &self,
        program: &DispatchProgram,
        inputs: &[&[f32]],
        readback_final_output: bool,
    ) -> Result<PreparedDispatchProgram, String> {
        let module =
            naga_codegen::lower_dispatch_program(naga_codegen::verify(program).expect("verify"));
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("tensor_ir dispatch"),
                source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
            });

        let external_input_buffers: Vec<_> = inputs
            .iter()
            .enumerate()
            .map(|(index, data)| {
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("input_{index}")),
                        contents: bytemuck::cast_slice(data),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            })
            .collect();

        let mut produced_buffers: HashMap<egg::Id, wgpu::Buffer> = HashMap::new();
        let mut pipelines = Vec::with_capacity(program.dispatches.len());
        let mut bind_groups = Vec::with_capacity(program.dispatches.len());
        let mut final_output_buffer = None;

        for (dispatch_index, dispatch) in program.dispatches.iter().enumerate() {
            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("tensor_ir pipeline"),
                    layout: None,
                    module: &shader,
                    entry_point: Some(&format!("dispatch_{dispatch_index}")),
                    compilation_options: Default::default(),
                    cache: None,
                });

            let output_elems =
                (dispatch.workgroups * program.device.simd_width) as usize * dispatch.outputs.len();
            let output_bytes = (output_elems * std::mem::size_of::<f32>()) as u64;
            let mut output_usage = wgpu::BufferUsages::STORAGE;
            if readback_final_output && dispatch_index + 1 == program.dispatches.len() {
                output_usage |= wgpu::BufferUsages::COPY_SRC;
            }
            let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("output_{dispatch_index}")),
                size: output_bytes,
                usage: output_usage,
                mapped_at_creation: false,
            });

            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let mut input_bind_buffers = Vec::with_capacity(dispatch.inputs.len());
            for input_id in &dispatch.inputs {
                input_bind_buffers.push(resolve_dispatch_input_buffer(
                    program,
                    *input_id,
                    inputs.len(),
                    &external_input_buffers,
                    &produced_buffers,
                )?);
            }
            let mut entries = Vec::new();
            for (binding, buffer) in input_bind_buffers.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                });
            }
            entries.push(wgpu::BindGroupEntry {
                binding: dispatch.inputs.len() as u32,
                resource: output_buffer.as_entire_binding(),
            });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tensor_ir bind group"),
                layout: &bind_group_layout,
                entries: &entries,
            });

            produced_buffers.insert(
                program.egraph.find(dispatch.semantic_output_id),
                output_buffer.clone(),
            );
            if dispatch_index + 1 == program.dispatches.len() {
                final_output_buffer = Some(output_buffer.clone());
            }
            pipelines.push(pipeline);
            bind_groups.push(bind_group);
        }

        let staging_buffer = if readback_final_output {
            let final_output_buffer = final_output_buffer
                .as_ref()
                .ok_or_else(|| "dispatch program did not produce an output buffer".to_string())?;
            Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tensor_ir staging"),
                size: final_output_buffer.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        Ok(PreparedDispatchProgram {
            pipelines,
            bind_groups,
            final_output_buffer,
            staging_buffer,
        })
    }

    fn submit_program(
        &self,
        program: &DispatchProgram,
        prepared: &PreparedDispatchProgram,
        timestamp_run: Option<(&TimestampQueryResources, u32)>,
    ) -> Result<(), String> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tensor_ir benchmark encoder"),
            });
        self.encode_program(&mut encoder, program, prepared, timestamp_run);
        self.queue.submit(std::iter::once(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        Ok(())
    }

    fn submit_program_repeated(
        &self,
        program: &DispatchProgram,
        prepared: &PreparedDispatchProgram,
        repetitions: u32,
    ) -> Result<(), String> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tensor_ir batched benchmark encoder"),
            });
        for _ in 0..repetitions {
            self.encode_program(&mut encoder, program, prepared, None);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        Ok(())
    }

    fn encode_program(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        program: &DispatchProgram,
        prepared: &PreparedDispatchProgram,
        timestamp_run: Option<(&TimestampQueryResources, u32)>,
    ) {
        let timestamps_per_run = (program.dispatches.len() as u32) * 2;

        for (dispatch_index, ((dispatch, pipeline), bind_group)) in program
            .dispatches
            .iter()
            .zip(prepared.pipelines.iter())
            .zip(prepared.bind_groups.iter())
            .enumerate()
        {
            let timestamp_writes = timestamp_run.map(|(queries, run_index)| {
                let base = run_index * timestamps_per_run + (dispatch_index as u32) * 2;
                wgpu::ComputePassTimestampWrites {
                    query_set: &queries.query_set,
                    beginning_of_pass_write_index: Some(base),
                    end_of_pass_write_index: Some(base + 1),
                }
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tensor_ir benchmark pass"),
                timestamp_writes,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            let physical_workgroups = dispatch.workgroups / dispatch.simdgroups.max(1);
            let (x, y, z) = dispatch_grid(physical_workgroups);
            pass.dispatch_workgroups(x, y, z);
        }
    }
}

fn dispatch_grid(physical_workgroups: u32) -> (u32, u32, u32) {
    if physical_workgroups <= MAX_DISPATCH_WORKGROUPS_PER_DIMENSION {
        return (physical_workgroups, 1, 1);
    }

    let y = physical_workgroups.div_ceil(MAX_DISPATCH_WORKGROUPS_PER_DIMENSION);
    (MAX_DISPATCH_WORKGROUPS_PER_DIMENSION, y, 1)
}

impl Default for GpuContext {
    fn default() -> Self {
        Self::new()
    }
}

struct PreparedDispatchProgram {
    pipelines: Vec<wgpu::ComputePipeline>,
    bind_groups: Vec<wgpu::BindGroup>,
    final_output_buffer: Option<wgpu::Buffer>,
    staging_buffer: Option<wgpu::Buffer>,
}

fn resolve_dispatch_input_buffer(
    program: &DispatchProgram,
    input_id: egg::Id,
    provided_inputs: usize,
    external_input_buffers: &[wgpu::Buffer],
    produced_buffers: &HashMap<egg::Id, wgpu::Buffer>,
) -> Result<wgpu::Buffer, String> {
    let canonical = program.egraph.find(input_id);
    if let Some(buffer) = produced_buffers.get(&canonical) {
        return Ok(buffer.clone());
    }

    for node in program.egraph[canonical].iter() {
        match node {
            TensorIr::HighLevel(crate::HighLevelNode::Input { id, .. }) => {
                let index = *id as usize;
                if index >= provided_inputs {
                    return Err(format!(
                        "expected at least {} external inputs, got {}",
                        index + 1,
                        provided_inputs
                    ));
                }
                return Ok(external_input_buffers[index].clone());
            }
            // A `Load { tier: Device(Input(n)), .. }` in the input's eclass is
            // equivalent to referencing external input `n`. Phase-1 lowering
            // can fold `Restride(Input(n))` directly into a per-lane Load in
            // the consumer's input slot, leaving the eclass without a raw
            // `HighLevel::Input` node to match above — but the device-buffer
            // identity is still carried by the Load's `BufferRef::Input`.
            TensorIr::Simd(crate::SimdNode::Load {
                tier: crate::MemTier::Device(crate::BufferRef::Input(id)),
                ..
            }) => {
                let index = *id as usize;
                if index >= provided_inputs {
                    return Err(format!(
                        "expected at least {} external inputs, got {}",
                        index + 1,
                        provided_inputs
                    ));
                }
                return Ok(external_input_buffers[index].clone());
            }
            _ => {}
        }
    }

    if std::env::var("TENSOR_IR_DEBUG_DISPATCH").is_ok() {
        let nodes: Vec<_> = program.egraph[canonical].iter().cloned().collect();
        eprintln!(
            "unresolved input canonical={canonical:?} nodes={:#?}",
            nodes
        );
    }
    Err(format!("unresolved dispatch input: {canonical:?}"))
}
