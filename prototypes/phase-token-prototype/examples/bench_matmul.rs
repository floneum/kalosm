use std::{borrow::Cow, sync::mpsc, time::Instant};

use phase_token_prototype::{
    build,
    kernels::gemm::{self, GemmTilePlan},
    DynamicOffset, KernelIr, Layout, LoopOffset, MemoryLevel, Shape, Strides, WorkgroupAxis,
    WorkgroupOffset, F32,
};
use wgpu::util::DeviceExt;

const M: usize = 1024;
const N: usize = 1024;
const K: usize = 1024;
const DEFAULT_BM: usize = 64;
const DEFAULT_BN: usize = 64;
const DEFAULT_BK: usize = 16;
const SHARED_PAD: usize = 4;
const WARMUP_BATCHES: usize = 2;
const MEASURED_BATCHES: usize = 10;
const DISPATCHES_PER_BATCH: usize = 200;
const TIMESTAMP_QUERIES_PER_BATCH: usize = DISPATCHES_PER_BATCH * 2;
const USE_PER_DISPATCH_TIMESTAMPS: bool = false;
const USE_MSL_PASSTHROUGH: bool = false;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    pollster::block_on(run())
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let tile = parse_tile_shape()?;
    let harness = Harness::new(tile).await?;

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

    let mut gpu_samples = Vec::with_capacity(MEASURED_BATCHES);
    for _ in 0..MEASURED_BATCHES {
        let command_buffer = harness.encode_timed_batch(DISPATCHES_PER_BATCH);
        harness.submit_and_wait(command_buffer)?;
        gpu_samples.extend(harness.read_gpu_dispatch_seconds(DISPATCHES_PER_BATCH)?);
    }

    let output = harness.read_output()?;
    let max_abs_error = sampled_max_abs_error(&output, &harness.a, &harness.b);
    if max_abs_error > 1.0e-3 {
        let (index, actual, expected) = first_sample_mismatch(
            &output, &harness.a, &harness.b, 1.0e-3,
        )
        .unwrap_or((0, output[0], cpu_dot(&harness.a, &harness.b, 0, 0)));
        return Err(format!(
            "matmul mismatch at {index}: gpu={actual} cpu={expected} max_abs_error={max_abs_error}",
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
    let flops_per_dispatch = 2.0 * M as f64 * N as f64 * K as f64;

    println!(
        "adapter: {} ({:?})",
        harness.adapter_info.name, harness.adapter_info.backend
    );
    println!(
        "bench_matmul: {M}x{N}x{K} f32, {}x{} workgroup grid per dispatch",
        N / tile.bn,
        M / tile.bm
    );
    println!("tile: BM={} BN={} BK={}", tile.bm, tile.bn, tile.bk);
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
    if !gpu_samples.is_empty() {
        gpu_samples.sort_by(f64::total_cmp);
        let gpu_mean_s = gpu_samples.iter().sum::<f64>() / gpu_samples.len() as f64;
        let gpu_p50_s = percentile(&gpu_samples, 0.50);
        let gpu_p90_s = percentile(&gpu_samples, 0.90);
        let gpu_min_s = gpu_samples[0];
        let gpu_max_s = gpu_samples[gpu_samples.len() - 1];
        println!(
            "gpu_timestamp_samples: {} valid / {} measured",
            gpu_samples.len(),
            MEASURED_BATCHES
        );
        println!("gpu_mean_dispatch_time_us: {:.3}", gpu_mean_s * 1.0e6);
        println!("gpu_p50_dispatch_time_us: {:.3}", gpu_p50_s * 1.0e6);
        println!("gpu_p90_dispatch_time_us: {:.3}", gpu_p90_s * 1.0e6);
        println!("gpu_min_dispatch_time_us: {:.3}", gpu_min_s * 1.0e6);
        println!("gpu_max_dispatch_time_us: {:.3}", gpu_max_s * 1.0e6);
        println!(
            "gpu_effective_tflops: {:.6}",
            flops_per_dispatch / gpu_mean_s / 1.0e12
        );
        println!(
            "gpu_p50_effective_tflops: {:.6}",
            flops_per_dispatch / gpu_p50_s / 1.0e12
        );
        println!(
            "gpu_p90_effective_tflops: {:.6}",
            flops_per_dispatch / gpu_p90_s / 1.0e12
        );
    }
    println!("note: this times pre-encoded batch submit-to-completion on the host.");
    println!("note: this requests wgpu's experimental cooperative matrix feature.");

    Ok(())
}

#[derive(Copy, Clone)]
struct TileShape {
    bm: usize,
    bn: usize,
    bk: usize,
}

struct Harness {
    tile: TileShape,
    adapter_info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    timestamp_inside_passes: bool,
    timestamp_inside_encoders: bool,
    query_set: wgpu::QuerySet,
    timestamp_buffer: wgpu::Buffer,
    timestamp_readback: wgpu::Buffer,
    timestamp_period_ns: f64,
    c_buffer: wgpu::Buffer,
    readback: wgpu::Buffer,
    a: Vec<f32>,
    b: Vec<f32>,
}

impl Harness {
    async fn new(tile: TileShape) -> Result<Self, Box<dyn std::error::Error>> {
        let ir = matmul_ir(tile);
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
        let cooperative_matrix_properties = adapter.cooperative_matrix_properties();
        let has_cooperative_matrix = adapter
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX);
        let needs_subgroups = tile.bn > 32 || tile.bm > 32;
        let has_subgroups = adapter.features().contains(wgpu::Features::SUBGROUP);
        let has_timestamp_query = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let has_timestamp_inside_passes = adapter
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
        let has_timestamp_inside_encoders = adapter
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS);
        let has_passthrough = adapter
            .features()
            .contains(wgpu::Features::PASSTHROUGH_SHADERS);
        if !has_cooperative_matrix {
            return Err(format!(
                "adapter {} does not expose EXPERIMENTAL_COOPERATIVE_MATRIX; properties: {:?}",
                adapter_info.name, cooperative_matrix_properties
            )
            .into());
        }
        if needs_subgroups && !has_subgroups {
            return Err(format!(
                "adapter {} does not expose SUBGROUP, required for multi-subgroup matmul tiles",
                adapter_info.name
            )
            .into());
        }
        if cooperative_matrix_properties.is_empty() {
            return Err(format!(
                "adapter {} exposes EXPERIMENTAL_COOPERATIVE_MATRIX but reports no cooperative matrix properties",
                adapter_info.name
            )
            .into());
        }
        if !has_timestamp_query {
            return Err(format!(
                "adapter {} does not expose TIMESTAMP_QUERY, required by this benchmark",
                adapter_info.name
            )
            .into());
        }
        if USE_MSL_PASSTHROUGH && !has_passthrough {
            return Err(format!(
                "adapter {} does not expose PASSTHROUGH_SHADERS, required by this benchmark's raw MSL compiler path",
                adapter_info.name
            )
            .into());
        }
        let mut required_features = wgpu::Features::EXPERIMENTAL_COOPERATIVE_MATRIX;
        required_features |= wgpu::Features::TIMESTAMP_QUERY;
        if USE_PER_DISPATCH_TIMESTAMPS && has_timestamp_inside_passes {
            required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES;
        }
        if USE_PER_DISPATCH_TIMESTAMPS && has_timestamp_inside_encoders {
            required_features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        }
        if USE_MSL_PASSTHROUGH {
            required_features |= wgpu::Features::PASSTHROUGH_SHADERS;
        }
        if needs_subgroups {
            required_features |= wgpu::Features::SUBGROUP;
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("phase-token-prototype bench device"),
                required_features,
                required_limits: wgpu::Limits::default(),
                experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
                ..Default::default()
            })
            .await?;

        let a = make_a();
        let b = make_b();
        let a_buffer = storage_buffer(&device, "A", &a, wgpu::BufferUsages::empty());
        let b_buffer = storage_buffer(&device, "B", &b, wgpu::BufferUsages::empty());
        let c_buffer = storage_buffer(
            &device,
            "C",
            &vec![0.0_f32; M * N],
            wgpu::BufferUsages::COPY_SRC,
        );
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("C readback"),
            size: byte_len::<f32>(M * N),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("matmul timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: TIMESTAMP_QUERIES_PER_BATCH as u32,
        });
        let timestamp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul timestamp resolve"),
            size: byte_len::<u64>(TIMESTAMP_QUERIES_PER_BATCH),
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let timestamp_readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul timestamp readback"),
            size: byte_len::<u64>(TIMESTAMP_QUERIES_PER_BATCH),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let timestamp_period_ns = queue.get_timestamp_period() as f64;

        let (shader, entry_point) = if USE_MSL_PASSTHROUGH {
            let workgroup_size = lowered.module().entry_points[0].workgroup_size;
            let (msl, entry_point) = matmul_msl(&lowered)?;
            let shader = unsafe {
                device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                    label: Some("lowered matmul msl"),
                    num_workgroups: (workgroup_size[0], workgroup_size[1], workgroup_size[2]),
                    msl: Some(Cow::Owned(msl)),
                    ..Default::default()
                })
            };
            (shader, entry_point)
        } else {
            let shader = unsafe {
                device.create_shader_module_trusted(
                    wgpu::ShaderModuleDescriptor {
                        label: Some("lowered matmul"),
                        source: wgpu::ShaderSource::Naga(Cow::Owned(lowered.module().clone())),
                    },
                    wgpu::ShaderRuntimeChecks::unchecked(),
                )
            };
            (shader, "main".to_string())
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matmul buffers"),
            entries: &storage_bindings(&[true, true, false]),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("matmul pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matmul pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(&entry_point),
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul bind group"),
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
                    resource: c_buffer.as_entire_binding(),
                },
            ],
        });

        Ok(Self {
            tile,
            adapter_info,
            device,
            queue,
            pipeline,
            bind_group,
            timestamp_inside_passes: has_timestamp_inside_passes,
            timestamp_inside_encoders: has_timestamp_inside_encoders,
            query_set,
            timestamp_buffer,
            timestamp_readback,
            timestamp_period_ns,
            c_buffer,
            readback,
            a,
            b,
        })
    }

    fn run_batch(&self, dispatches: usize) -> Result<(), Box<dyn std::error::Error>> {
        let command_buffer = self.encode_batch(dispatches);
        self.submit_and_wait(command_buffer)
    }

    fn encode_batch(&self, dispatches: usize) -> wgpu::CommandBuffer {
        self.encode_batch_inner(dispatches, false)
    }

    fn encode_timed_batch(&self, dispatches: usize) -> wgpu::CommandBuffer {
        self.encode_batch_inner(dispatches, true)
    }

    fn encode_batch_inner(&self, dispatches: usize, timestamp: bool) -> wgpu::CommandBuffer {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul bench encoder"),
            });
        if timestamp
            && USE_PER_DISPATCH_TIMESTAMPS
            && !self.timestamp_inside_passes
            && self.timestamp_inside_encoders
        {
            for dispatch in 0..dispatches {
                encoder.write_timestamp(&self.query_set, (dispatch * 2) as u32);
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("matmul timed dispatch pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &self.bind_group, &[]);
                    pass.dispatch_workgroups(
                        (N / self.tile.bn) as u32,
                        (M / self.tile.bm) as u32,
                        1,
                    );
                }
                encoder.write_timestamp(&self.query_set, (dispatch * 2 + 1) as u32);
            }
        } else {
            let timestamp_writes = (timestamp
                && (!USE_PER_DISPATCH_TIMESTAMPS
                    || (!self.timestamp_inside_passes && !self.timestamp_inside_encoders)))
                .then_some(wgpu::ComputePassTimestampWrites {
                    query_set: &self.query_set,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul bench pass"),
                timestamp_writes,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            for dispatch in 0..dispatches {
                if timestamp && USE_PER_DISPATCH_TIMESTAMPS {
                    pass.write_timestamp(&self.query_set, (dispatch * 2) as u32);
                }
                pass.dispatch_workgroups((N / self.tile.bn) as u32, (M / self.tile.bm) as u32, 1);
                if timestamp && USE_PER_DISPATCH_TIMESTAMPS {
                    pass.write_timestamp(&self.query_set, (dispatch * 2 + 1) as u32);
                }
            }
        }
        if timestamp {
            let query_count = if !USE_PER_DISPATCH_TIMESTAMPS
                || (!self.timestamp_inside_passes && !self.timestamp_inside_encoders)
            {
                2
            } else {
                dispatches * 2
            };
            encoder.resolve_query_set(
                &self.query_set,
                0..query_count as u32,
                &self.timestamp_buffer,
                0,
            );
            encoder.copy_buffer_to_buffer(
                &self.timestamp_buffer,
                0,
                &self.timestamp_readback,
                0,
                byte_len::<u64>(query_count),
            );
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
                label: Some("matmul readback encoder"),
            });
        encoder.copy_buffer_to_buffer(&self.c_buffer, 0, &self.readback, 0, byte_len::<f32>(M * N));
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

    fn read_gpu_dispatch_seconds(
        &self,
        dispatches: usize,
    ) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
        let slice = self.timestamp_readback.slice(..);
        let (tx, rx) = mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        self.device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;

        let mapped = slice.get_mapped_range();
        let timestamps = bytemuck::cast_slice::<u8, u64>(&mapped);
        let sample_count = if USE_PER_DISPATCH_TIMESTAMPS
            && (self.timestamp_inside_passes || self.timestamp_inside_encoders)
        {
            dispatches
        } else {
            1
        };
        let per_dispatch = USE_PER_DISPATCH_TIMESTAMPS
            && (self.timestamp_inside_passes || self.timestamp_inside_encoders);
        let samples = timestamps
            .chunks_exact(2)
            .take(sample_count)
            .filter_map(|pair| {
                let elapsed_ticks = pair[1].checked_sub(pair[0])?;
                (elapsed_ticks > 0).then(|| {
                    let seconds = elapsed_ticks as f64 * self.timestamp_period_ns * 1.0e-9;
                    if per_dispatch {
                        seconds
                    } else {
                        seconds / dispatches as f64
                    }
                })
            })
            .collect();
        drop(mapped);
        self.timestamp_readback.unmap();
        Ok(samples)
    }
}

fn matmul_ir(tile: TileShape) -> KernelIr {
    build(move |mut phase| {
        let a_full = phase.storage_tensor_read::<F32>(shape([M, K]));
        let b_full = phase.storage_tensor_read::<F32>(shape([K, N]));
        let c_full = phase.storage_tensor::<F32>(shape([M, N]));
        let a_in = a_full.dynamic_tile_2d(
            shape([tile.bm, tile.bk]),
            Some(DynamicOffset::Workgroup(WorkgroupOffset::new(
                WorkgroupAxis::Y,
                tile.bm as u32,
            ))),
            Some(DynamicOffset::Loop(LoopOffset::new(tile.bk as u32))),
        );
        let b_in = b_full.dynamic_tile_2d(
            shape([tile.bk, tile.bn]),
            Some(DynamicOffset::Loop(LoopOffset::new(tile.bk as u32))),
            Some(DynamicOffset::Workgroup(WorkgroupOffset::new(
                WorkgroupAxis::X,
                tile.bn as u32,
            ))),
        );
        let c_out = c_full.workgroup_tile_2d(
            shape([tile.bm, tile.bn]),
            Some(WorkgroupOffset::new(WorkgroupAxis::Y, tile.bm as u32)),
            Some(WorkgroupOffset::new(WorkgroupAxis::X, tile.bn as u32)),
        );
        let mut acc = phase.alloc_fragment::<F32>(shape([tile.bm, tile.bn]));
        phase.fill_zero(&mut acc);
        let acc_out = acc;

        phase.range_step_count(
            (K / tile.bk) as u32,
            |mut phase, _| {
                let a = phase
                    .alloc_tile_with_layout::<F32>(workgroup_layout([tile.bm, tile.bk], tile.bk));
                let b = phase
                    .alloc_tile_with_layout::<F32>(workgroup_layout([tile.bk, tile.bn], tile.bn));
                let pending = phase.cooperative_load_pair(a, &a_in, b, &b_in);
                let (a, b, mut phase) = pending.sync_tiles();

                gemm::tiled(
                    &mut phase,
                    &a,
                    &b,
                    &mut acc,
                    GemmTilePlan::portable(tile.bm as u32, tile.bn as u32, tile.bk as u32),
                );
                phase.sync_end()
            },
            |mut phase| {
                phase.store_fragment_to_storage(&acc_out, &c_out);
                phase.finish()
            },
        )
    })
}

fn matmul_msl(
    lowered: &phase_token_prototype::NagaKernel,
) -> Result<(String, String), Box<dyn std::error::Error>> {
    let mut resources = naga::back::msl::BindingMap::default();
    for binding in 0..3 {
        resources.insert(
            naga::ResourceBinding { group: 0, binding },
            naga::back::msl::BindTarget {
                buffer: Some(binding as u8),
                ..Default::default()
            },
        );
    }

    let mut options = naga::back::msl::Options::default();
    options.lang_version = (2, 3);
    options.zero_initialize_workgroup_memory = false;
    options.force_loop_bounding = false;
    options.bounds_check_policies = naga::proc::BoundsCheckPolicies {
        index: naga::proc::BoundsCheckPolicy::Unchecked,
        buffer: naga::proc::BoundsCheckPolicy::Unchecked,
        image_load: naga::proc::BoundsCheckPolicy::Unchecked,
        binding_array: naga::proc::BoundsCheckPolicy::Unchecked,
    };
    options.per_entry_point_map = naga::back::msl::EntryPointResourceMap::from([(
        "main".to_string(),
        naga::back::msl::EntryPointResources {
            resources,
            ..Default::default()
        },
    )]);

    let pipeline_options = naga::back::msl::PipelineOptions {
        entry_point: Some((naga::ShaderStage::Compute, "main".into())),
        allow_and_force_point_size: false,
        vertex_pulling_transform: false,
        vertex_buffer_mappings: Vec::new(),
    };
    let (mut msl, info) = naga::back::msl::write_string(
        lowered.module(),
        lowered.info(),
        &options,
        &pipeline_options,
    )?;
    msl = msl.replace(
        "metal::simdgroup_float8x8 NagaCooperativeLoad",
        "static inline metal::simdgroup_float8x8 NagaCooperativeLoad",
    );
    msl = msl.replace(
        "metal::simdgroup_float8x8 NagaCooperativeMultiplyAdd",
        "static inline metal::simdgroup_float8x8 NagaCooperativeMultiplyAdd",
    );
    let entry_point = info.entry_point_names[0]
        .as_ref()
        .map_err(|error| format!("MSL entry point translation failed: {error}"))?
        .clone();
    Ok((msl, entry_point))
}

fn parse_tile_shape() -> Result<TileShape, Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let bm = parse_arg(args.next(), DEFAULT_BM, "BM")?;
    let bn = parse_arg(args.next(), DEFAULT_BN, "BN")?;
    let bk = parse_arg(args.next(), DEFAULT_BK, "BK")?;
    if args.next().is_some() {
        return Err("usage: cargo run --example bench_matmul -- [BM BN BK]".into());
    }
    let tile = TileShape { bm, bn, bk };
    if tile.bm == 0
        || tile.bn == 0
        || tile.bk == 0
        || M % tile.bm != 0
        || N % tile.bn != 0
        || K % tile.bk != 0
    {
        return Err(format!(
            "tile must divide {M}x{N}x{K}; got BM={} BN={} BK={}",
            tile.bm, tile.bn, tile.bk
        )
        .into());
    }
    Ok(tile)
}

fn parse_arg(
    value: Option<String>,
    default: usize,
    name: &'static str,
) -> Result<usize, Box<dyn std::error::Error>> {
    match value {
        Some(value) => value.parse::<usize>().map_err(|error| {
            format!("{name} must be a positive integer, got {value:?}: {error}").into()
        }),
        None => Ok(default),
    }
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

fn make_b() -> Vec<f32> {
    (0..K)
        .flat_map(|k| {
            (0..N).map(move |col| {
                let value = ((k * 11 + col * 5) % 29) as i32 - 14;
                value as f32 * 0.03125
            })
        })
        .collect()
}

fn sampled_max_abs_error(actual: &[f32], a: &[f32], b: &[f32]) -> f32 {
    sample_points()
        .into_iter()
        .map(|(row, col)| (actual[row * N + col] - cpu_dot(a, b, row, col)).abs())
        .fold(0.0, f32::max)
}

fn first_sample_mismatch(
    actual: &[f32],
    a: &[f32],
    b: &[f32],
    tolerance: f32,
) -> Option<(usize, f32, f32)> {
    for (row, col) in sample_points() {
        let index = row * N + col;
        let expected = cpu_dot(a, b, row, col);
        if (actual[index] - expected).abs() > tolerance {
            return Some((index, actual[index], expected));
        }
    }
    None
}

fn sample_points() -> Vec<(usize, usize)> {
    let mut points = vec![
        (0, 0),
        (0, N - 1),
        (M - 1, 0),
        (M - 1, N - 1),
        (M / 2, N / 2),
    ];
    for i in 0..64 {
        points.push(((i * 37) % M, (i * 101) % N));
    }
    points
}

fn cpu_dot(a: &[f32], b: &[f32], row: usize, col: usize) -> f32 {
    let mut sum = 0.0;
    for k in 0..K {
        sum += a[row * K + k] * b[k * N + col];
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

fn workgroup_layout(dims: [usize; 2], cols: usize) -> Layout {
    Layout::strided(
        MemoryLevel::Workgroup,
        shape(dims),
        Strides::new([(cols + SHARED_PAD) as u32, 1]),
    )
}
