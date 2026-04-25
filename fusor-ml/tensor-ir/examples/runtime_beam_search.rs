#[cfg(not(feature = "runtime"))]
fn main() {
    eprintln!(
        "Enable the runtime feature: cargo run -r -p tensor_ir --features runtime --example runtime_beam_search"
    );
    std::process::exit(1);
}

#[cfg(feature = "runtime")]
use std::borrow::Cow;
#[cfg(feature = "runtime")]
use std::collections::HashSet;
#[cfg(feature = "runtime")]
use std::panic::{self, AssertUnwindSafe};
#[cfg(feature = "runtime")]
use std::time::Instant;

#[cfg(feature = "runtime")]
use tensor_ir::language::extract_recexpr_list;
#[cfg(feature = "runtime")]
use tensor_ir::*;
#[cfg(feature = "runtime")]
use wgpu::util::DeviceExt;

#[cfg(feature = "runtime")]
const SIMD_WIDTH: u32 = 32;

#[cfg(feature = "runtime")]
struct BenchmarkHarness {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[cfg(feature = "runtime")]
fn benchmark_required_features(adapter: &wgpu::Adapter) -> wgpu::Features {
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

#[cfg(feature = "runtime")]
struct CandidateMeasurement {
    median_us: f64,
    max_err: f32,
    timing_kind: &'static str,
}

#[cfg(feature = "runtime")]
struct RunnableCandidate {
    cost: f64,
    expr: egg::RecExpr<TensorIr>,
    program: DispatchProgram,
}

#[cfg(feature = "runtime")]
struct TimestampQueryResources {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    destination_buffer: wgpu::Buffer,
}

#[cfg(feature = "runtime")]
impl TimestampQueryResources {
    fn new(device: &wgpu::Device, count: u32) -> Self {
        let bytes = (count as u64) * (std::mem::size_of::<u64>() as u64);
        Self {
            query_set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("runtime_beam_query_set"),
                count,
                ty: wgpu::QueryType::Timestamp,
            }),
            resolve_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runtime_beam_query_resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
                mapped_at_creation: false,
            }),
            destination_buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("runtime_beam_query_readback"),
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

#[cfg(feature = "runtime")]
impl BenchmarkHarness {
    fn new() -> Self {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("no GPU adapter found");
        let required_features = benchmark_required_features(&adapter);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("runtime_beam_search"),
            required_features,
            ..Default::default()
        }))
        .expect("failed to create device");
        Self { device, queue }
    }

    fn benchmark(
        &self,
        program: &DispatchProgram,
        inputs: &[&[f32]],
        shape_params: &ShapeParams,
        expected: &[f32],
        warmup_runs: u32,
        timing_runs: u32,
    ) -> Result<CandidateMeasurement, String> {
        if program.dispatches.is_empty() {
            return Err("candidate rejected during lowering".into());
        }

        let verified =
            tensor_ir::naga_codegen::verify(program).map_err(|e| format!("verify: {e}"))?;
        let module = panic::catch_unwind(AssertUnwindSafe(|| lower_dispatch_program(verified)))
            .map_err(format_panic_payload)?;
        let shader = panic::catch_unwind(AssertUnwindSafe(|| {
            self.device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("runtime_beam_candidate"),
                    source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
                })
        }))
        .map_err(format_panic_payload)?;

        let external_input_buffers: Vec<_> = inputs
            .iter()
            .enumerate()
            .map(|(index, data)| {
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("candidate_input_{index}")),
                        contents: bytemuck::cast_slice(data),
                        usage: wgpu::BufferUsages::STORAGE,
                    })
            })
            .collect();

        let mut produced_buffers: std::collections::HashMap<egg::Id, wgpu::Buffer> =
            std::collections::HashMap::new();
        let mut pipelines = Vec::new();
        let mut bind_groups = Vec::new();
        let mut output_buffers = Vec::new();
        let shape_param_words = shape_params.storage_words();
        let shape_params_buffer =
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("shape_params"),
                    contents: bytemuck::cast_slice(&shape_param_words),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        for (dispatch_index, dispatch) in program.dispatches.iter().enumerate() {
            let pipeline = panic::catch_unwind(AssertUnwindSafe(|| {
                self.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("runtime_beam_pipeline"),
                        layout: None,
                        module: &shader,
                        entry_point: Some(&format!("dispatch_{dispatch_index}")),
                        compilation_options: Default::default(),
                        cache: None,
                    })
            }))
            .map_err(format_panic_payload)?;

            let dispatch_workgroups =
                dispatch.workgroups.eval_u32(shape_params).ok_or_else(|| {
                    format!("missing shape parameter for dispatch {dispatch_index} workgroups")
                })?;
            let dispatch_elems =
                (dispatch_workgroups * SIMD_WIDTH) as usize * dispatch.outputs.len();
            let output_elems = dispatch_elems.max(expected.len());
            let output_bytes = (output_elems * std::mem::size_of::<f32>()) as u64;
            let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("candidate_output_{dispatch_index}")),
                size: output_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let bind_group_layout = pipeline.get_bind_group_layout(0);
            let mut input_bind_buffers = Vec::new();
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
            entries.push(wgpu::BindGroupEntry {
                binding: dispatch.inputs.len() as u32 + 1,
                resource: shape_params_buffer.as_entire_binding(),
            });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("runtime_beam_bind_group"),
                layout: &bind_group_layout,
                entries: &entries,
            });

            produced_buffers.insert(
                program.egraph.find(dispatch.outputs[0].value_id),
                output_buffer.clone(),
            );
            pipelines.push(pipeline);
            bind_groups.push(bind_group);
            output_buffers.push(output_buffer);
        }

        let final_output_buffer = output_buffers
            .last()
            .cloned()
            .ok_or_else(|| "candidate rejected during lowering".to_string())?;
        let final_output_bytes = final_output_buffer.size();
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("candidate_staging"),
            size: final_output_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let measured_runs = timing_runs.max(1);
        let timestamps_per_run = (program.dispatches.len() as u32) * 2;
        let timestamp_queries = if self
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            && timestamps_per_run > 0
        {
            Some(TimestampQueryResources::new(
                &self.device,
                measured_runs * timestamps_per_run,
            ))
        } else {
            None
        };

        let run_once = |read_back: bool,
                        query_run_index: Option<u32>|
         -> Result<(f64, Option<Vec<f32>>), String> {
            let start = Instant::now();
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("runtime_beam_encoder"),
                });
            for (dispatch_index, ((dispatch, pipeline), bind_group)) in program
                .dispatches
                .iter()
                .zip(&pipelines)
                .zip(&bind_groups)
                .enumerate()
            {
                let timestamp_writes = query_run_index.and_then(|run_index| {
                    timestamp_queries.as_ref().map(|queries| {
                        let base = run_index * timestamps_per_run + (dispatch_index as u32) * 2;
                        wgpu::ComputePassTimestampWrites {
                            query_set: &queries.query_set,
                            beginning_of_pass_write_index: Some(base),
                            end_of_pass_write_index: Some(base + 1),
                        }
                    })
                });
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("runtime_beam_pass"),
                    timestamp_writes,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                // Physical workgroups = virtual workgroups / simdgroups.
                // Each physical workgroup contains `simdgroups` simdgroups.
                let dispatch_workgroups =
                    dispatch.workgroups.eval_u32(shape_params).ok_or_else(|| {
                        format!("missing shape parameter for dispatch {dispatch_index} workgroups")
                    })?;
                let physical_workgroups = dispatch_workgroups.div_ceil(dispatch.simdgroups.max(1));
                pass.dispatch_workgroups(physical_workgroups, 1, 1);
            }
            if read_back {
                encoder.copy_buffer_to_buffer(
                    &final_output_buffer,
                    0,
                    &staging_buffer,
                    0,
                    final_output_bytes,
                );
            }
            self.queue.submit(std::iter::once(encoder.finish()));
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

            let result = if read_back {
                let slice = staging_buffer.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |result| {
                    let _ = tx.send(result);
                });
                let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
                rx.recv()
                    .map_err(|_| "staging map channel closed".to_string())?
                    .map_err(|err| format!("staging map failed: {err:?}"))?;

                let data = slice.get_mapped_range();
                let output: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                drop(data);
                staging_buffer.unmap();
                Some(output)
            } else {
                None
            };

            Ok((start.elapsed().as_secs_f64() * 1_000_000.0, result))
        };

        for _ in 0..warmup_runs {
            let _ = run_once(false, None)?;
        }

        let mut host_timings = Vec::with_capacity(measured_runs as usize);
        for run_index in 0..measured_runs {
            let (elapsed_us, _output) = run_once(false, Some(run_index))?;
            host_timings.push(elapsed_us);
        }

        let mut timings = if let Some(queries) = &timestamp_queries {
            let mut resolve_encoder =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("runtime_beam_query_resolve_encoder"),
                    });
            queries.encode_resolve(&mut resolve_encoder, measured_runs * timestamps_per_run);
            self.queue.submit(std::iter::once(resolve_encoder.finish()));
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

            let timestamp_period_ns = self.queue.get_timestamp_period() as f64;
            let timestamps = queries.read_results(&self.device)?;
            let mut gpu_timings = Vec::with_capacity(measured_runs as usize);
            for run_index in 0..measured_runs as usize {
                let run_start = run_index * timestamps_per_run as usize;
                let mut elapsed_us = 0.0;
                for dispatch_index in 0..program.dispatches.len() {
                    let start_tick = timestamps[run_start + dispatch_index * 2];
                    let end_tick = timestamps[run_start + dispatch_index * 2 + 1];
                    elapsed_us +=
                        end_tick.wrapping_sub(start_tick) as f64 * timestamp_period_ns / 1_000.0;
                }
                gpu_timings.push(elapsed_us);
            }
            gpu_timings
        } else {
            host_timings
        };
        timings.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_us = timings[timings.len() / 2];
        let timing_kind = if timestamp_queries.is_some() {
            "gpu"
        } else {
            "submit+wait"
        };

        let (_validation_us, final_output) = run_once(true, None)?;

        let output = final_output.ok_or_else(|| "missing readback output".to_string())?;
        if output.len() < expected.len() {
            return Err(format!(
                "output buffer too small: expected at least {}, got {}",
                expected.len(),
                output.len()
            ));
        }
        let max_err = output[..expected.len()]
            .iter()
            .zip(expected)
            .map(|(gpu, cpu)| (gpu - cpu).abs())
            .fold(0.0f32, f32::max);

        // Debug: print first few error positions when there are errors
        if max_err >= 1e-3 {
            let n = expected.len();
            let dim = (n as f64).sqrt() as usize; // assume square matrix
            let mut err_count = 0;
            for (i, (gpu, cpu)) in output[..n].iter().zip(expected).enumerate() {
                let err = (gpu - cpu).abs();
                if err >= 1e-3 && err_count < 30 {
                    let row = i / dim;
                    let col = i % dim;
                    eprintln!(
                        "  ERR at [{row},{col}] (flat={i}): gpu={gpu:.6} cpu={cpu:.6} diff={err:.6}"
                    );
                    err_count += 1;
                }
            }
            if err_count > 0 {
                let total_errs = output[..n]
                    .iter()
                    .zip(expected)
                    .filter(|(g, c)| (*g - *c).abs() >= 1e-3)
                    .count();
                eprintln!("  Total error positions: {total_errs} / {n}");
            }
        }

        Ok(CandidateMeasurement {
            median_us,
            max_err,
            timing_kind,
        })
    }
}

#[cfg(feature = "runtime")]
fn format_panic_payload(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(msg) = payload.downcast_ref::<&'static str>() {
        format!("codegen panic: {msg}")
    } else if let Some(msg) = payload.downcast_ref::<String>() {
        format!("codegen panic: {msg}")
    } else {
        "codegen panic".to_string()
    }
}

#[cfg(feature = "runtime")]
fn resolve_dispatch_input_buffer(
    program: &DispatchProgram,
    input_id: egg::Id,
    provided_inputs: usize,
    external_input_buffers: &[wgpu::Buffer],
    produced_buffers: &std::collections::HashMap<egg::Id, wgpu::Buffer>,
) -> Result<wgpu::Buffer, String> {
    let canonical = program.egraph.find(input_id);
    if let Some(buffer) = produced_buffers.get(&canonical) {
        return Ok(buffer.clone());
    }

    for node in program.egraph[canonical].iter() {
        if let TensorIr::HighLevel(HighLevelNode::Input { id, .. }) = node {
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
    }

    Err(format!("unresolved dispatch input: {canonical:?}"))
}

#[cfg(feature = "runtime")]
fn extract_workload_arg(args: &[String]) -> (String, Vec<String>) {
    let mut kind = String::from("matmul");
    let mut positional = Vec::with_capacity(args.len());
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--workload" => {
                if let Some(value) = iter.next() {
                    kind = value.clone();
                }
            }
            other if other.starts_with("--workload=") => {
                kind = other["--workload=".len()..].to_string();
            }
            _ => positional.push(arg.clone()),
        }
    }
    (kind, positional)
}

#[cfg(feature = "runtime")]
fn parse_arg<T: std::str::FromStr>(args: &[String], index: usize, default: T) -> T {
    args.get(index)
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
}

#[cfg(feature = "runtime")]
#[path = "common/refs.rs"]
mod common_refs;
#[cfg(feature = "runtime")]
use common_refs::build_workload;

#[cfg(feature = "runtime")]
fn is_runnable_dispatch_candidate(
    egraph: &TensorEGraph,
    expr: &egg::RecExpr<TensorIr>,
    provided_inputs: usize,
) -> bool {
    let Some(root) = expr.as_ref().last() else {
        return false;
    };

    let expected_dispatches = match root {
        TensorIr::Dispatch(DispatchNode::Dispatch { .. }) => 1,
        TensorIr::Dispatch(DispatchNode::Seq(list_id)) => {
            extract_recexpr_list(expr.as_ref(), *list_id).len()
        }
        _ => return false,
    };

    let program = build_dispatch_program_from_extracted(
        expr,
        egraph.clone(),
        &DeviceProfile::default(),
        &tensor_ir::LoweringOptions::default(),
    );
    if program.dispatches.len() != expected_dispatches {
        return false;
    }

    let mut produced = HashSet::new();
    for dispatch in &program.dispatches {
        for input in &dispatch.inputs {
            let canonical = program.egraph.find(*input);
            if produced.contains(&canonical) {
                continue;
            }

            let mut resolved = false;
            for node in program.egraph[canonical].iter() {
                if let TensorIr::HighLevel(HighLevelNode::Input { id, .. }) = node {
                    resolved = (*id as usize) < provided_inputs;
                    if resolved {
                        break;
                    }
                }
            }

            if !resolved {
                return false;
            }
        }
        produced.insert(program.egraph.find(dispatch.semantic_output_id));
    }

    true
}

#[cfg(feature = "runtime")]
fn parse_dump_candidate_filter() -> Option<HashSet<usize>> {
    let raw = std::env::var("TENSOR_IR_DUMP_CANDIDATES").ok()?;
    let indices: HashSet<_> = raw
        .split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .collect();
    if indices.is_empty() {
        None
    } else {
        Some(indices)
    }
}

#[cfg(feature = "runtime")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    // Accept `--workload <name>` anywhere in argv; strip it before positional parsing
    // so existing numeric-argument callers keep working.
    let (workload_kind, positional) = extract_workload_arg(&args);
    let m: u32 = parse_arg(&positional, 1, 64);
    let n: u32 = parse_arg(&positional, 2, 64);
    let k: u32 = parse_arg(&positional, 3, 64);
    let beam_width: usize = parse_arg(&positional, 4, 8);
    let candidate_limit: usize = parse_arg(&positional, 5, 32);
    let warmup_runs: u32 = parse_arg(&positional, 6, 2);
    let timing_runs: u32 = parse_arg(&positional, 7, 5);
    let iter_limit: usize = parse_arg(&positional, 8, 10);
    let node_limit: usize = parse_arg(&positional, 9, 50_000);
    let time_limit_secs: u64 = parse_arg(&positional, 10, 30);

    let workload = match build_workload(&workload_kind, m, n, k) {
        Ok(w) => w,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    };

    println!("=== Runtime Beam Search ({}) ===", workload.name);
    println!("  Dims: m={m} n={n} k={k}");
    println!("  Beam width: {beam_width}");
    println!("  Candidate limit: {candidate_limit}");
    println!("  Warmup runs: {warmup_runs}");
    println!("  Timing runs: {timing_runs}");
    println!("  Saturation iter limit: {iter_limit}");
    println!("  Saturation node limit: {node_limit}");
    println!("  Saturation time limit: {time_limit_secs}s");
    println!();

    let provided_inputs = workload.inputs.len();
    let mut egraph = TensorEGraph::default();
    let root = egraph.add_expr(&workload.expr);
    egraph.rebuild();

    let runner = RunnerConfig {
        iter_limit,
        node_limit,
        time_limit_secs,
        device: DeviceProfile::default(),
        lowering: tensor_ir::LoweringOptions::default(),
    };
    let egraph = saturate(egraph, &runner);

    let beam_cfg = BeamConfig {
        beam_width,
        ..Default::default()
    };
    let raw_candidates = beam_extract_candidates(
        &egraph,
        root,
        &beam_cfg,
        candidate_limit.saturating_mul(8).max(candidate_limit),
    );
    println!("=== Raw Candidates (before filtering) ===");
    for (index, (cost, expr)) in raw_candidates.iter().enumerate() {
        let root_node = expr
            .as_ref()
            .last()
            .map(|node| format!("{node}"))
            .unwrap_or_else(|| "<empty>".into());
        let runnable = is_runnable_dispatch_candidate(&egraph, expr, provided_inputs);
        let dev_loads = expr
            .as_ref()
            .iter()
            .filter(|n| {
                matches!(
                    n,
                    TensorIr::Simd(SimdNode::Load {
                        tier: MemTier::Device(_),
                        ..
                    })
                )
            })
            .count();
        let tg_loads = expr
            .as_ref()
            .iter()
            .filter(|n| {
                matches!(
                    n,
                    TensorIr::Simd(SimdNode::Load {
                        tier: MemTier::Threadgroup(_),
                        ..
                    })
                )
            })
            .count();
        let thetas = expr
            .as_ref()
            .iter()
            .filter(|n| matches!(n, TensorIr::Simd(SimdNode::Theta { .. })))
            .count();
        // Show simdgroups for runnable tiled candidates
        let sg_info = if runnable {
            let prog = build_dispatch_program_from_extracted(
                expr,
                egraph.clone(),
                &DeviceProfile::default(),
                &tensor_ir::LoweringOptions::default(),
            );
            if let Some(d) = prog.dispatches.first() {
                format!(" sg={}", d.simdgroups)
            } else {
                String::new()
            }
        } else {
            String::new()
        };
        println!(
            "  [{index}] cost={cost:.1} root={root_node} dev={dev_loads} tg={tg_loads} thetas={thetas}{sg_info} runnable={runnable}"
        );
    }
    println!();
    let mut candidates = Vec::new();
    for (cost, expr) in raw_candidates.into_iter() {
        if !is_runnable_dispatch_candidate(&egraph, &expr, 2) {
            continue;
        }

        let program = build_dispatch_program_from_extracted(
            &expr,
            egraph.clone(),
            &DeviceProfile::default(),
            &tensor_ir::LoweringOptions::default(),
        );
        candidates.push(RunnableCandidate {
            cost,
            expr,
            program,
        });
        if candidates.len() >= candidate_limit {
            break;
        }
    }
    println!("=== Candidates ===");
    for (index, candidate) in candidates.iter().enumerate() {
        println!(
            "  [{index}] cost={:.1} root={}",
            candidate.cost,
            candidate
                .expr
                .as_ref()
                .last()
                .map(|node| format!("{node}"))
                .unwrap_or_else(|| "<empty>".into())
        );
    }
    println!();

    let input_refs: Vec<&[f32]> = workload.inputs.iter().map(Vec::as_slice).collect();
    let expected = workload.expected.as_slice();
    let harness = BenchmarkHarness::new();
    let dump_candidates = parse_dump_candidate_filter();

    let mut winner = None;
    println!("  Runnable dispatch candidates: {}", candidates.len());
    println!("=== Runtime Measurements ===");
    for (index, candidate) in candidates.iter().enumerate() {
        if dump_candidates
            .as_ref()
            .map(|set| set.contains(&index))
            .unwrap_or(false)
        {
            eprintln!("=== Candidate {index} Expr ===");
            eprintln!("{}", candidate.expr);
            eprintln!("=== Candidate {index} Program ===");
            eprintln!("{}", candidate.program);
            match lower_to_wgsl(&candidate.program) {
                Ok(wgsl) => {
                    eprintln!("=== Candidate {index} WGSL ===");
                    eprintln!("{wgsl}");
                }
                Err(err) => {
                    eprintln!("=== Candidate {index} WGSL Error ===");
                    eprintln!("{err}");
                }
            }
        }
        match harness.benchmark(
            &candidate.program,
            &input_refs,
            &workload.shape_params,
            expected,
            warmup_runs,
            timing_runs,
        ) {
            Ok(measurement) => {
                let status = if measurement.max_err < 1e-3 {
                    "ok"
                } else {
                    "bad-output"
                };
                println!(
                    "  [{index}] cost={:.1} time={:.1} us ({}) max_err={:.6} {status}",
                    candidate.cost,
                    measurement.median_us,
                    measurement.timing_kind,
                    measurement.max_err
                );
                if status == "ok"
                    && winner
                        .as_ref()
                        .map(|(_, best_time, _, _)| measurement.median_us < *best_time)
                        .unwrap_or(true)
                {
                    winner = Some((
                        index,
                        measurement.median_us,
                        candidate.cost,
                        candidate.expr.clone(),
                    ));
                }
            }
            Err(err) => {
                println!("  [{index}] cost={:.1} failed: {err}", candidate.cost);
            }
        }
    }
    println!();

    let Some((winner_index, winner_time, winner_cost, winner_expr)) = winner else {
        eprintln!("No valid runtime candidate succeeded");
        std::process::exit(1);
    };

    let winner_program = build_dispatch_program_from_extracted(
        &winner_expr,
        egraph,
        &DeviceProfile::default(),
        &tensor_ir::LoweringOptions::default(),
    );
    let wgsl = lower_to_wgsl(&winner_program).expect("winner should lower to WGSL");

    println!("=== Winner ===");
    println!("  Candidate: {winner_index}");
    println!("  Synthetic cost: {winner_cost:.1}");
    println!("  Median runtime: {winner_time:.1} us");
    println!();
    println!("=== Winner WGSL ===");
    println!("{wgsl}");
}
