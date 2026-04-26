//! GPU runtime for executing e-graph-native effect programs via wgpu.

use crate::{EffectNode, TensorIr, naga_codegen};
use egg::{Id, Language, RecExpr};
use std::borrow::Cow;
use std::collections::{BTreeSet, HashMap};
use std::time::Instant;
use tensor_ir_frontend::types::{BufferRef, DeviceProfile, ScalarValue, ShapeParams, TensorId};
use wgpu::util::DeviceExt;

const MAX_DISPATCH_WORKGROUPS_PER_DIMENSION: u32 = 65_535;

/// Borrowed effect program ready for direct runtime execution.
pub struct EffectProgram<'a> {
    pub expr: &'a RecExpr<TensorIr>,
    pub device: &'a DeviceProfile,
    pub output_elements: usize,
}

#[derive(Debug, Clone)]
struct DispatchPlan {
    workgroups: tensor_ir_frontend::types::Dim,
    simdgroups: u32,
}

#[derive(Debug, Clone)]
struct RuntimePlan {
    dispatches: Vec<DispatchPlan>,
    device_buffers: Vec<BufferRef>,
    output_buffers: Vec<BufferRef>,
}

/// A GPU context that can execute e-graph-native effect programs.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
}

/// Benchmark configuration for a GPU program.
#[derive(Debug, Clone, Copy)]
pub struct ProgramBenchmarkConfig {
    pub warmup_runs: u32,
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

    pub fn benchmark(
        &self,
        program: &EffectProgram<'_>,
        inputs: &[&[f32]],
        shape_params: &ShapeParams,
        config: ProgramBenchmarkConfig,
    ) -> Result<GpuBenchmarkResult, String> {
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

        let prepared = self.prepare_program(program, inputs, shape_params, false)?;
        for _ in 0..config.warmup_runs {
            self.submit_program(&prepared, None)?;
        }

        let timestamps_per_run = (prepared.dispatches.len() as u32) * 2;
        let timestamp_queries =
            TimestampQueryResources::new(&self.device, config.timing_runs * timestamps_per_run);

        for run_index in 0..config.timing_runs {
            self.submit_program(&prepared, Some((&timestamp_queries, run_index)))?;
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
            for dispatch_index in 0..prepared.dispatches.len() {
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

    pub fn benchmark_host(
        &self,
        program: &EffectProgram<'_>,
        inputs: &[&[f32]],
        shape_params: &ShapeParams,
        config: ProgramBenchmarkConfig,
    ) -> Result<HostBenchmarkResult, String> {
        if config.timing_runs == 0 {
            return Err("timing_runs must be greater than zero".into());
        }

        let prepared = self.prepare_program(program, inputs, shape_params, false)?;
        for _ in 0..config.warmup_runs {
            self.submit_program(&prepared, None)?;
        }

        let mut samples_host_us = Vec::with_capacity(config.timing_runs as usize);
        for _ in 0..config.timing_runs {
            let start = Instant::now();
            self.submit_program(&prepared, None)?;
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

    pub fn execute(
        &self,
        program: &EffectProgram<'_>,
        inputs: &[&[f32]],
        shape_params: &ShapeParams,
    ) -> Vec<f32> {
        let prepared = self
            .prepare_program(program, inputs, shape_params, true)
            .expect("failed to prepare effect program");
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
        self.encode_program(&mut encoder, &prepared, None);
        encoder.copy_buffer_to_buffer(&final_output, 0, staging_buffer, 0, output_bytes);
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
        program: &EffectProgram<'_>,
        inputs: &[&[f32]],
        shape_params: &ShapeParams,
        readback_final_output: bool,
    ) -> Result<PreparedEffectProgram, String> {
        let runtime_plan = RuntimePlan::from_expr(program.expr)?;
        if runtime_plan.dispatches.is_empty() {
            return Err("effect program has no runnable dispatches".into());
        }

        let module = naga_codegen::lower_effect_program(program.expr, program.device)?;
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("tensor_ir effect program"),
                source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
            });

        let output_bytes = ((program.output_elements * std::mem::size_of::<f32>()) as u64).max(4);
        let mut buffers = HashMap::new();
        for buffer in &runtime_plan.device_buffers {
            let wgpu_buffer = if let Some(input_index) = buffer.input_index() {
                let data = inputs.get(input_index as usize).ok_or_else(|| {
                    format!(
                        "expected external input {input_index}, got {}",
                        inputs.len()
                    )
                })?;
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("input_{input_index}")),
                        contents: bytemuck::cast_slice(data),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            } else {
                let mut usage = wgpu::BufferUsages::STORAGE;
                if readback_final_output && runtime_plan.output_buffers.contains(buffer) {
                    usage |= wgpu::BufferUsages::COPY_SRC;
                }
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("buffer_{buffer}")),
                    size: output_bytes,
                    usage,
                    mapped_at_creation: false,
                })
            };
            buffers.insert(*buffer, wgpu_buffer);
        }

        let shape_param_words = shape_params.storage_words();
        let shape_params_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("shape_params"),
                    contents: bytemuck::cast_slice(&shape_param_words),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let mut pipelines = Vec::with_capacity(runtime_plan.dispatches.len());
        let mut bind_groups = Vec::with_capacity(runtime_plan.dispatches.len());
        let mut physical_workgroups = Vec::with_capacity(runtime_plan.dispatches.len());

        for (dispatch_index, dispatch) in runtime_plan.dispatches.iter().enumerate() {
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
            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let mut entries = Vec::new();
            for (binding, buffer_ref) in runtime_plan.device_buffers.iter().enumerate() {
                let buffer = buffers
                    .get(buffer_ref)
                    .ok_or_else(|| format!("missing buffer {buffer_ref}"))?;
                entries.push(wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                });
            }
            entries.push(wgpu::BindGroupEntry {
                binding: runtime_plan.device_buffers.len() as u32,
                resource: shape_params_buffer.as_entire_binding(),
            });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tensor_ir bind group"),
                layout: &bind_group_layout,
                entries: &entries,
            });
            let workgroups = dispatch.workgroups.eval_u32(shape_params).ok_or_else(|| {
                format!(
                    "missing shape parameter while evaluating dispatch {dispatch_index} workgroups"
                )
            })?;
            pipelines.push(pipeline);
            bind_groups.push(bind_group);
            physical_workgroups.push(workgroups.div_ceil(dispatch.simdgroups.max(1)));
        }

        let final_output_buffer = runtime_plan
            .output_buffers
            .last()
            .and_then(|buffer| buffers.get(buffer))
            .cloned();
        let staging_buffer = if readback_final_output {
            let final_output_buffer = final_output_buffer
                .as_ref()
                .ok_or_else(|| "effect program did not produce an output buffer".to_string())?;
            Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("tensor_ir staging"),
                size: final_output_buffer.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        Ok(PreparedEffectProgram {
            dispatches: runtime_plan.dispatches,
            pipelines,
            bind_groups,
            physical_workgroups,
            final_output_buffer,
            staging_buffer,
        })
    }

    fn submit_program(
        &self,
        prepared: &PreparedEffectProgram,
        timestamp_run: Option<(&TimestampQueryResources, u32)>,
    ) -> Result<(), String> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tensor_ir benchmark encoder"),
            });
        self.encode_program(&mut encoder, prepared, timestamp_run);
        self.queue.submit(std::iter::once(encoder.finish()));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        Ok(())
    }

    fn encode_program(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        prepared: &PreparedEffectProgram,
        timestamp_run: Option<(&TimestampQueryResources, u32)>,
    ) {
        let timestamps_per_run = (prepared.dispatches.len() as u32) * 2;
        for (dispatch_index, (pipeline, bind_group)) in prepared
            .pipelines
            .iter()
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
            let (x, y, z) = dispatch_grid(prepared.physical_workgroups[dispatch_index]);
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

struct PreparedEffectProgram {
    dispatches: Vec<DispatchPlan>,
    pipelines: Vec<wgpu::ComputePipeline>,
    bind_groups: Vec<wgpu::BindGroup>,
    physical_workgroups: Vec<u32>,
    final_output_buffer: Option<wgpu::Buffer>,
    staging_buffer: Option<wgpu::Buffer>,
}

impl RuntimePlan {
    fn from_expr(expr: &RecExpr<TensorIr>) -> Result<Self, String> {
        let root = Id::from(expr.as_ref().len() - 1);
        let TensorIr::Effect(EffectNode::Program { children }) = &expr[root] else {
            return Err("runtime program root must be EffectNode::Program".into());
        };
        let mut dispatches = Vec::new();
        collect_dispatches(expr, children[1], &mut dispatches)?;

        let mut device_buffers = BTreeSet::new();
        collect_device_buffers(expr, root, &mut device_buffers);
        let output_buffers = collect_program_outputs(expr, children[2]);

        Ok(Self {
            dispatches,
            device_buffers: device_buffers.into_iter().collect(),
            output_buffers,
        })
    }
}

fn collect_dispatches(
    expr: &RecExpr<TensorIr>,
    id: Id,
    out: &mut Vec<DispatchPlan>,
) -> Result<(), String> {
    match &expr[id] {
        TensorIr::Effect(EffectNode::Token) => {}
        TensorIr::Effect(EffectNode::Seq(list)) => {
            for step in extract_list_recexpr(expr, *list) {
                collect_dispatches(expr, step, out)?;
            }
        }
        TensorIr::Effect(EffectNode::Dispatch {
            workgroups,
            simdgroups,
            ..
        }) => out.push(DispatchPlan {
            workgroups: workgroups.clone(),
            simdgroups: *simdgroups,
        }),
        other => return Err(format!("expected effect dispatch/seq, found {other:?}")),
    }
    Ok(())
}

fn collect_device_buffers(expr: &RecExpr<TensorIr>, id: Id, out: &mut BTreeSet<BufferRef>) {
    match &expr[id] {
        TensorIr::Simd(crate::SimdNode::Load {
            tier: tensor_ir_frontend::types::MemTier::Device(buffer),
            ..
        })
        | TensorIr::Effect(EffectNode::Store {
            tier: tensor_ir_frontend::types::MemTier::Device(buffer),
            ..
        })
        | TensorIr::Effect(EffectNode::StoreIf {
            tier: tensor_ir_frontend::types::MemTier::Device(buffer),
            ..
        }) => {
            out.insert(*buffer);
        }
        _ => {}
    }
    for child in expr[id].children() {
        collect_device_buffers(expr, *child, out);
    }
}

fn collect_program_outputs(expr: &RecExpr<TensorIr>, outputs: Id) -> Vec<BufferRef> {
    extract_list_recexpr(expr, outputs)
        .into_iter()
        .filter_map(|id| match &expr[id] {
            TensorIr::Const(ScalarValue::U32(tensor_id)) => {
                Some(BufferRef::Tensor(TensorId(*tensor_id)))
            }
            _ => None,
        })
        .collect()
}

fn extract_list_recexpr(expr: &RecExpr<TensorIr>, mut id: Id) -> Vec<Id> {
    let mut out = Vec::new();
    loop {
        match &expr[id] {
            TensorIr::Cons([head, tail]) => {
                out.push(*head);
                id = *tail;
            }
            TensorIr::Nil => break,
            _ => break,
        }
    }
    out
}
