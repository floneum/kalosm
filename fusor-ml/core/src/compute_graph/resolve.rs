use std::{
    collections::VecDeque,
    sync::Arc,
    time::{Duration, Instant},
};

use crate::{
    mir::{direct_kernel::PreparedDirectDispatch, inputs::MirValue, operation::Operation},
    nary_wise::{ExtractedUnaryChain, NaryExpr, NaryOperation, UnaryFunctionChain},
    quantized::matmul::QMatMulOperation,
    tensor::TensorData,
};
use petgraph::algo::toposort;
use petgraph::stable_graph::StableGraph;
use rustc_hash::{FxHashMap, FxHashSet};

use super::{ComputeGraphInner, ComputeGraphNode, ComputeGraphNodeVariant, NodeIndex};

pub(crate) struct ResolverResult {
    pub(crate) data: TensorData,
    pub(crate) total_kernels: usize,
}

struct DispatchRecord {
    dispatch: PreparedDirectDispatch,
    name: Option<String>,
    category: Option<String>,
}

struct DispatchMetadata {
    name: Option<String>,
    category: Option<String>,
}

struct CopyBufferRecord {
    source: Arc<wgpu::Buffer>,
    destination: Arc<wgpu::Buffer>,
    source_offset: u64,
    destination_offset: u64,
    size: u64,
}

enum CommandRecord {
    Dispatch(DispatchRecord),
    CopyBuffer(CopyBufferRecord),
}

#[derive(Default)]
struct KernelProfileAggregate {
    count: usize,
    total_ns: f64,
    max_ns: f64,
}

impl KernelProfileAggregate {
    fn record(&mut self, ns: f64) {
        self.count += 1;
        self.total_ns += ns;
        self.max_ns = self.max_ns.max(ns);
    }
}

#[derive(Default)]
struct ResolveHostProfile {
    build_execution_graph: Duration,
    optimize: Duration,
    toposort: Duration,
    queue_lowering: Duration,
    consumer_count: Duration,
    encoder_create: Duration,
    map_layout: Duration,
    inputs: Duration,
    output: Duration,
    workgroup: Duration,
    build_kernel: Duration,
    prepare_dispatch: Duration,
    release: Duration,
    timestamp_setup: Duration,
    encode: Duration,
    submit: Duration,
    profile_readback: Duration,
}

#[derive(Default)]
struct ResolveHostCategoryProfile {
    count: usize,
    inputs: Duration,
    output: Duration,
    workgroup: Duration,
    build_kernel: Duration,
    prepare_dispatch: Duration,
}

impl ResolveHostProfile {
    fn print(&self, total: Duration, queued_ops: usize, kernels: usize) {
        eprintln!(
            "resolve_host_profile queued_ops={queued_ops} kernels={kernels} total={total:?} \
build_execution_graph={:?} optimize={:?} toposort={:?} queue_lowering={:?} \
consumer_count={:?} encoder_create={:?} map_layout={:?} inputs={:?} output={:?} \
workgroup={:?} build_kernel={:?} prepare_dispatch={:?} release={:?} \
timestamp_setup={:?} encode={:?} submit={:?} profile_readback={:?}",
            self.build_execution_graph,
            self.optimize,
            self.toposort,
            self.queue_lowering,
            self.consumer_count,
            self.encoder_create,
            self.map_layout,
            self.inputs,
            self.output,
            self.workgroup,
            self.build_kernel,
            self.prepare_dispatch,
            self.release,
            self.timestamp_setup,
            self.encode,
            self.submit,
            self.profile_readback,
        );
    }
}

fn print_host_category_profile(profile: FxHashMap<&'static str, ResolveHostCategoryProfile>) {
    let mut profile = profile
        .into_iter()
        .map(|(category, profile)| {
            (
                category,
                profile.count,
                profile.inputs,
                profile.output,
                profile.workgroup,
                profile.build_kernel,
                profile.prepare_dispatch,
            )
        })
        .collect::<Vec<_>>();
    profile.sort_by(|a, b| b.5.cmp(&a.5));
    eprintln!("resolve_host_category_profile {profile:?}");
}

fn node_category(variant: &ComputeGraphNodeVariant) -> &'static str {
    match variant {
        ComputeGraphNodeVariant::Nary(_) => "nary",
        ComputeGraphNodeVariant::SliceAssign(_) => "slice_assign",
        ComputeGraphNodeVariant::Resize(_) => "resize",
        ComputeGraphNodeVariant::MapLayout(_) => "map_layout",
        ComputeGraphNodeVariant::Dequantize(_) => "dequantize",
        ComputeGraphNodeVariant::QEmbedding(_) => "q_embedding",
        ComputeGraphNodeVariant::MatMul(_) => "matmul",
        ComputeGraphNodeVariant::QMatMul(_) => "q_matmul",
        ComputeGraphNodeVariant::Tensor(_) => "tensor",
        ComputeGraphNodeVariant::Reduce(_) => "reduce",
        ComputeGraphNodeVariant::RmsNorm(_) => "rms_norm",
        ComputeGraphNodeVariant::FlashAttention(_) => "flash_attention",
    }
}

#[derive(Debug, Clone)]
struct ExecutionNode {
    inner_idx: NodeIndex,
    variant: ComputeGraphNodeVariant,
}

type ExecutionGraph = StableGraph<ExecutionNode, ()>;
type ExecutionNodeIndex = petgraph::graph::NodeIndex;

fn dispatch_category(name: &str) -> String {
    name.split('_').take(2).collect::<Vec<_>>().join("_")
}

fn padded_query_buffer_size(size: u64) -> u64 {
    let align_mask = wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT - 1;
    ((size + align_mask) & !align_mask).max(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT)
}

fn print_gpu_kernel_profile(
    records: &[DispatchMetadata],
    timestamps: &[u64],
    timestamp_period_ns: f64,
    timestamp_mode: &str,
) {
    let mut category_profile = FxHashMap::<String, KernelProfileAggregate>::default();
    let mut name_profile = FxHashMap::<String, KernelProfileAggregate>::default();
    let mut accounted_ns = 0.0;

    for (index, record) in records.iter().enumerate() {
        let begin = timestamps.get(index * 2).copied().unwrap_or_default();
        let end = timestamps.get(index * 2 + 1).copied().unwrap_or(begin);
        let ns = end.saturating_sub(begin) as f64 * timestamp_period_ns;
        accounted_ns += ns;
        if let Some(category) = &record.category {
            category_profile
                .entry(category.clone())
                .or_default()
                .record(ns);
        }
        if let Some(name) = &record.name {
            name_profile.entry(name.clone()).or_default().record(ns);
        }
    }

    let span_ns = match (timestamps.first(), timestamps.last()) {
        (Some(first), Some(last)) => last.saturating_sub(*first) as f64 * timestamp_period_ns,
        _ => 0.0,
    };

    let mut categories = category_profile
        .into_iter()
        .map(|(name, aggregate)| {
            (
                name,
                aggregate.count,
                aggregate.total_ns / 1_000_000.0,
                aggregate.total_ns / aggregate.count as f64 / 1_000.0,
                aggregate.max_ns / 1_000.0,
            )
        })
        .collect::<Vec<_>>();
    categories.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

    let mut names = name_profile
        .into_iter()
        .map(|(name, aggregate)| {
            (
                name,
                aggregate.count,
                aggregate.total_ns / 1_000_000.0,
                aggregate.total_ns / aggregate.count as f64 / 1_000.0,
                aggregate.max_ns / 1_000.0,
            )
        })
        .collect::<Vec<_>>();
    names.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    names.truncate(32);

    eprintln!(
        "resolve_gpu_kernel_profile mode={} kernels={} accounted_ms={:.3} span_ms={:.3} timestamp_period_ns={:.3}",
        timestamp_mode,
        records.len(),
        accounted_ns / 1_000_000.0,
        span_ns / 1_000_000.0,
        timestamp_period_ns
    );
    eprintln!("resolve_gpu_kernel_categories {categories:?}");
    eprintln!("resolve_gpu_kernel_top_names {names:?}");
}

pub(crate) struct Resolver {
    execution_graph: ExecutionGraph,
    node_mapping: FxHashMap<NodeIndex, ExecutionNodeIndex>,
    targets: Vec<NodeIndex>,
    resolved_set: FxHashSet<NodeIndex>,
}

impl Resolver {
    pub(crate) fn new(graph: &mut ComputeGraphInner, target: NodeIndex) -> Self {
        Self::new_batch(graph, vec![target])
    }

    pub(crate) fn new_batch(graph: &mut ComputeGraphInner, targets: Vec<NodeIndex>) -> Self {
        let resolved_set = graph
            .nodes
            .nodes
            .node_indices()
            .filter(|&idx| {
                graph
                    .nodes
                    .nodes
                    .node_weight(idx)
                    .map(|n| n.cached.is_some())
                    .unwrap_or(false)
            })
            .collect();
        Self {
            targets,
            execution_graph: Default::default(),
            node_mapping: Default::default(),
            resolved_set,
        }
    }

    pub(crate) fn run(
        &mut self,
        graph: &mut ComputeGraphInner,
        _removed: &mut Vec<ComputeGraphNode>,
    ) -> ResolverResult {
        let host_trace = std::env::var_os("FUSOR_TRACE_RESOLVE_HOST").is_some();
        let host_category_trace = std::env::var_os("FUSOR_TRACE_RESOLVE_HOST_CATEGORIES").is_some();
        let host_total_start = host_trace.then(Instant::now);
        let mut host_profile = ResolveHostProfile::default();
        let mut host_category_profile =
            FxHashMap::<&'static str, ResolveHostCategoryProfile>::default();
        let device = graph.device();
        let max_subgroup_size = device.max_subgroup_size();

        // Pass 1: Build execution graph for all targets
        {
            let start = host_trace.then(Instant::now);
            let targets = self.targets.clone();
            for &target in &targets {
                self.build_execution_graph(graph, target);
            }
            if let Some(start) = start {
                host_profile.build_execution_graph += start.elapsed();
            }
        }

        // Pass 2: Apply Rewrite Rules
        {
            let start = host_trace.then(Instant::now);
            self.optimize(graph);
            if let Some(start) = start {
                host_profile.optimize += start.elapsed();
            }
        }

        // Pass 3: Topological Sort
        let sorted_nodes = {
            let start = host_trace.then(Instant::now);
            let sorted_nodes = toposort(&self.execution_graph, None)
                .unwrap_or_else(|_| panic!("Cycle detected in execution graph"));
            if let Some(start) = start {
                host_profile.toposort += start.elapsed();
            }
            sorted_nodes
        };

        // Pass 4: Execution
        // Extract operations in order.
        let target_set: FxHashSet<NodeIndex> = self.targets.iter().copied().collect();
        let mut queued_operations = Vec::with_capacity(sorted_nodes.len());

        {
            let start = host_trace.then(Instant::now);
            for idx in sorted_nodes {
                let node = &self.execution_graph[idx];
                // Handle Tensor caching explicitly here
                if let ComputeGraphNodeVariant::Tensor(data) = &node.variant {
                    graph.set_cached_result(node.inner_idx, data.clone());
                    continue;
                }

                if let Some(op) = self.lower_node(node) {
                    queued_operations.push((node.inner_idx, op));
                }
            }
            if let Some(start) = start {
                host_profile.queue_lowering += start.elapsed();
            }
        }
        let queued_operation_count = queued_operations.len();

        // Build a remaining-consumer count. For each queued operation, we use
        // the Operation's visit_dependencies (which reflects post-optimization
        // fused dependencies) to count how many future operations read each
        // inner NodeIndex.
        let mut remaining_consumers: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        {
            let start = host_trace.then(Instant::now);
            for (_, op) in &queued_operations {
                op.visit_dependencies(&mut |dep| {
                    *remaining_consumers.entry(dep).or_insert(0) += 1;
                });
            }
            if let Some(start) = start {
                host_profile.consumer_count += start.elapsed();
            }
        }

        // Record all kernels for this resolve into one command encoder. The
        // encoder is submitted once at the end so host-side materialization is
        // the synchronization boundary.
        let mut command_encoder = {
            let start = host_trace.then(Instant::now);
            let command_encoder =
                device
                    .wgpu_device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Resolver Encoder"),
                    });
            if let Some(start) = start {
                host_profile.encoder_create += start.elapsed();
            }
            command_encoder
        };

        let trace = std::env::var_os("FUSOR_TRACE_DECODE").is_some()
            || std::env::var_os("FUSOR_TRACE_RESOLVE").is_some();
        let trace_names = std::env::var_os("FUSOR_TRACE_DECODE_NAMES").is_some();
        let profile_gpu_kernels = std::env::var_os("FUSOR_TRACE_GPU_KERNELS").is_some();
        let collect_dispatch_metadata = trace || profile_gpu_kernels;
        let mut commands = Vec::<CommandRecord>::with_capacity(queued_operations.len());
        let mut dispatch_categories = FxHashMap::<String, usize>::default();
        let mut dispatch_names = FxHashMap::<String, usize>::default();
        for (node, operation) in queued_operations {
            let operation_category = host_category_trace
                .then(|| {
                    graph
                        .nodes
                        .nodes
                        .node_weight(node)
                        .map(|node| node_category(&node.variant))
                })
                .flatten();
            // Map layout isn't really a kernel. Resolve it immediately
            let map_layout = if let Some(node_data) = graph.nodes.nodes.node_weight(node) {
                match &node_data.variant {
                    ComputeGraphNodeVariant::MapLayout(map_layout) => Some(map_layout.clone()),
                    ComputeGraphNodeVariant::Resize(resize) => resize.lower(graph),
                    _ => None,
                }
            } else {
                None
            };
            if let Some(map_layout) = map_layout {
                let start = host_trace.then(Instant::now);
                let result = map_layout.run(graph);
                // Cache the result
                graph.set_cached_result(node, result);
                // Map-layout nodes are resolved immediately — release any
                // input buffers that are no longer needed.
                // Use graph.visit_dependencies for map_layout since they
                // are not lowered to Operations.
                Self::release_dead_intermediates_from_graph(
                    graph,
                    &[node],
                    &mut remaining_consumers,
                    &target_set,
                );
                if let Some(start) = start {
                    host_profile.map_layout += start.elapsed();
                }
            } else {
                let slice_copy = graph.nodes.nodes.node_weight(node).and_then(|node_data| {
                    let ComputeGraphNodeVariant::SliceAssign(slice_assign) = &node_data.variant
                    else {
                        return None;
                    };
                    Self::try_prepare_in_place_slice_assign_copy(graph, slice_assign)
                });
                if let Some((output, copies)) = slice_copy {
                    graph.set_cached_result(node, output);
                    commands.extend(
                        copies
                            .into_iter()
                            .map(|copy| CommandRecord::CopyBuffer(copy)),
                    );
                    let start = host_trace.then(Instant::now);
                    Self::release_dead_intermediates(
                        graph,
                        &[(node, operation)],
                        &mut remaining_consumers,
                        &target_set,
                    );
                    if let Some(start) = start {
                        host_profile.release += start.elapsed();
                    }
                    continue;
                }

                let start = host_trace.then(Instant::now);
                let new_inputs = operation.inputs(graph);
                if let Some(start) = start {
                    let elapsed = start.elapsed();
                    host_profile.inputs += elapsed;
                    if let Some(category) = operation_category {
                        host_category_profile.entry(category).or_default().inputs += elapsed;
                    }
                }
                let start = host_trace.then(Instant::now);
                let result = operation.output(graph, &new_inputs);
                let MirValue::Tensor(resolved) = result else {
                    panic!("Kernel input value is not a tensor");
                };
                graph.set_cached_result(node, resolved);
                if let Some(start) = start {
                    let elapsed = start.elapsed();
                    host_profile.output += elapsed;
                    if let Some(category) = operation_category {
                        host_category_profile.entry(category).or_default().output += elapsed;
                    }
                }

                let start = host_trace.then(Instant::now);
                let constraints = operation.workgroup_shape_constraints(&device);
                let workgroup_shape = constraints.solve(max_subgroup_size).unwrap_or_else(|| {
                    panic!("Failed to find a valid workgroup shape for constraints {constraints:?}")
                });
                if let Some(start) = start {
                    let elapsed = start.elapsed();
                    host_profile.workgroup += elapsed;
                    if let Some(category) = operation_category {
                        host_category_profile.entry(category).or_default().workgroup += elapsed;
                    }
                }
                let start = host_trace.then(Instant::now);
                let Some(direct_kernel) =
                    operation.build_direct_kernel(graph, &workgroup_shape, &new_inputs)
                else {
                    panic!(
                        "operation did not provide a direct kernel: {}",
                        operation.name()
                    );
                };
                if let Some(start) = start {
                    let elapsed = start.elapsed();
                    host_profile.build_kernel += elapsed;
                    if let Some(category) = operation_category {
                        let profile = host_category_profile.entry(category).or_default();
                        profile.count += 1;
                        profile.build_kernel += elapsed;
                    }
                }
                let start = host_trace.then(Instant::now);
                if let Some(dispatch) = direct_kernel.prepare_dispatch(&device) {
                    if let Some(start) = start {
                        let elapsed = start.elapsed();
                        host_profile.prepare_dispatch += elapsed;
                        if let Some(category) = operation_category {
                            host_category_profile
                                .entry(category)
                                .or_default()
                                .prepare_dispatch += elapsed;
                        }
                    }
                    let (name, category) = if collect_dispatch_metadata {
                        let name = operation.name();
                        let category = dispatch_category(&name);
                        if trace {
                            *dispatch_categories.entry(category.clone()).or_default() += 1;
                            if trace_names {
                                *dispatch_names.entry(name.clone()).or_default() += 1;
                            }
                        }
                        (Some(name), Some(category))
                    } else {
                        (None, None)
                    };
                    commands.push(CommandRecord::Dispatch(DispatchRecord {
                        dispatch,
                        name,
                        category,
                    }));
                } else if let Some(start) = start {
                    let elapsed = start.elapsed();
                    host_profile.prepare_dispatch += elapsed;
                    if let Some(category) = operation_category {
                        host_category_profile
                            .entry(category)
                            .or_default()
                            .prepare_dispatch += elapsed;
                    }
                }
                let start = host_trace.then(Instant::now);
                Self::release_dead_intermediates(
                    graph,
                    &[(node, operation)],
                    &mut remaining_consumers,
                    &target_set,
                );
                if let Some(start) = start {
                    host_profile.release += start.elapsed();
                }
            };
        }

        let total_kernels = commands
            .iter()
            .filter(|command| matches!(command, CommandRecord::Dispatch(_)))
            .count();
        if trace {
            let mut categories = dispatch_categories.into_iter().collect::<Vec<_>>();
            categories.sort_by(|a, b| a.0.cmp(&b.0));
            eprintln!("resolve_dispatch_categories {categories:?}");
            if trace_names {
                let mut names = dispatch_names.into_iter().collect::<Vec<_>>();
                names.sort_by(|a, b| a.0.cmp(&b.0));
                eprintln!("resolve_dispatch_names {names:?}");
            }
        }
        let dispatch_metadata = commands
            .iter()
            .filter_map(|command| match command {
                CommandRecord::Dispatch(record) => Some(DispatchMetadata {
                    name: record.name.clone(),
                    category: record.category.clone(),
                }),
                CommandRecord::CopyBuffer(_) => None,
            })
            .collect::<Vec<_>>();
        let query_count = (total_kernels * 2) as u32;
        let profile_inside_pass_timestamps = profile_gpu_kernels
            && device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
        let query_resources = if total_kernels > 0 {
            let profiling_supported = profile_gpu_kernels
                && device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
                && total_kernels * 2 <= wgpu::QUERY_SET_MAX_QUERIES as usize;
            if profiling_supported {
                let start = host_trace.then(Instant::now);
                let query_set = device
                    .wgpu_device()
                    .create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("Resolver Kernel Timestamp Queries"),
                        ty: wgpu::QueryType::Timestamp,
                        count: query_count,
                    });
                let raw_query_size = query_count as u64 * wgpu::QUERY_SIZE as u64;
                let query_buffer_size = padded_query_buffer_size(raw_query_size);
                let query_buffer = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Resolver Kernel Timestamp Resolve Buffer"),
                    size: query_buffer_size,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let readback_buffer = device.wgpu_device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Resolver Kernel Timestamp Readback Buffer"),
                    size: query_buffer_size,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                if let Some(start) = start {
                    host_profile.timestamp_setup += start.elapsed();
                }
                Some((query_set, query_buffer, readback_buffer, raw_query_size))
            } else {
                if profile_gpu_kernels {
                    eprintln!(
                        "resolve_gpu_kernel_profile unavailable timestamp_features={:?} kernels={}",
                        device.features(),
                        total_kernels
                    );
                }
                None
            }
        } else {
            None
        };

        if !commands.is_empty() {
            let encode_start = host_trace.then(Instant::now);
            let mut dispatch_index = 0usize;
            let mut command_index = 0usize;
            while command_index < commands.len() {
                match &commands[command_index] {
                    CommandRecord::CopyBuffer(copy) => {
                        command_encoder.copy_buffer_to_buffer(
                            &copy.source,
                            copy.source_offset,
                            &copy.destination,
                            copy.destination_offset,
                            copy.size,
                        );
                        command_index += 1;
                    }
                    CommandRecord::Dispatch(_) => {
                        if let Some((query_set, _, _, _)) = &query_resources {
                            if profile_inside_pass_timestamps {
                                let mut pass = command_encoder.begin_compute_pass(
                                    &wgpu::ComputePassDescriptor {
                                        label: Some("Resolver Direct Kernels"),
                                        timestamp_writes: None,
                                    },
                                );
                                while command_index < commands.len() {
                                    let CommandRecord::Dispatch(record) = &commands[command_index]
                                    else {
                                        break;
                                    };
                                    pass.write_timestamp(query_set, (dispatch_index * 2) as u32);
                                    record.dispatch.run(&mut pass);
                                    pass.write_timestamp(
                                        query_set,
                                        (dispatch_index * 2 + 1) as u32,
                                    );
                                    dispatch_index += 1;
                                    command_index += 1;
                                }
                            } else {
                                let CommandRecord::Dispatch(record) = &commands[command_index]
                                else {
                                    unreachable!();
                                };
                                let mut pass = command_encoder.begin_compute_pass(
                                    &wgpu::ComputePassDescriptor {
                                        label: Some("Resolver Direct Kernel"),
                                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                                            query_set,
                                            beginning_of_pass_write_index: Some(
                                                (dispatch_index * 2) as u32,
                                            ),
                                            end_of_pass_write_index: Some(
                                                (dispatch_index * 2 + 1) as u32,
                                            ),
                                        }),
                                    },
                                );
                                record.dispatch.run(&mut pass);
                                dispatch_index += 1;
                                command_index += 1;
                            }
                        } else {
                            let mut pass =
                                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                    label: Some("Resolver Direct Kernels"),
                                    timestamp_writes: None,
                                });
                            while command_index < commands.len() {
                                let CommandRecord::Dispatch(record) = &commands[command_index]
                                else {
                                    break;
                                };
                                record.dispatch.run(&mut pass);
                                command_index += 1;
                            }
                        }
                    }
                }
            }

            if let Some((query_set, query_buffer, readback_buffer, raw_query_size)) =
                &query_resources
            {
                command_encoder.resolve_query_set(query_set, 0..query_count, query_buffer, 0);
                command_encoder.copy_buffer_to_buffer(
                    query_buffer,
                    0,
                    readback_buffer,
                    0,
                    *raw_query_size,
                );
            }
            if let Some(start) = encode_start {
                host_profile.encode += start.elapsed();
            }
        }

        // Submit any remaining commands.
        let submit_start = host_trace.then(Instant::now);
        device.wgpu_queue().submit(Some(command_encoder.finish()));
        if let Some(start) = submit_start {
            host_profile.submit += start.elapsed();
        }
        if let Some((_, _, readback_buffer, raw_query_size)) = &query_resources {
            let profile_readback_start = host_trace.then(Instant::now);
            let slice = readback_buffer.slice(..*raw_query_size);
            let (sender, receiver) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
            device.poll_wait();
            match receiver.recv() {
                Ok(Ok(())) => {
                    let view = slice.get_mapped_range();
                    let timestamps = bytemuck::cast_slice::<u8, u64>(&view);
                    print_gpu_kernel_profile(
                        &dispatch_metadata,
                        timestamps,
                        device.wgpu_queue().get_timestamp_period() as f64,
                        if profile_inside_pass_timestamps {
                            "inside_pass"
                        } else {
                            "pass_boundary"
                        },
                    );
                    drop(view);
                    readback_buffer.unmap();
                }
                Ok(Err(error)) => {
                    eprintln!("resolve_gpu_kernel_profile map_failed {error:?}");
                }
                Err(error) => {
                    eprintln!("resolve_gpu_kernel_profile map_channel_failed {error:?}");
                }
            }
            if let Some(start) = profile_readback_start {
                host_profile.profile_readback += start.elapsed();
            }
        }
        device.reset_initialized_buffers();

        let data = graph
            .get_result(self.targets[0])
            .expect("Target result not cached");
        if let Some(start) = host_total_start {
            host_profile.print(start.elapsed(), queued_operation_count, total_kernels);
            if host_category_trace {
                print_host_category_profile(host_category_profile);
            }
        }
        ResolverResult {
            data,
            total_kernels,
        }
    }

    /// After a kernel flush produces cached results for a set of nodes,
    /// decrement the remaining-consumer count for each of their inputs. When
    /// an input's count reaches zero and it is neither a target node nor held
    /// by a live tensor handle, drop its cached result to free the GPU buffer.
    ///
    /// Uses `op.visit_dependencies()` to match the post-optimization
    /// dependencies that were used to build the consumer counts.
    fn release_dead_intermediates(
        graph: &mut ComputeGraphInner,
        produced_ops: &[(NodeIndex, Arc<dyn Operation>)],
        remaining_consumers: &mut FxHashMap<NodeIndex, usize>,
        targets: &FxHashSet<NodeIndex>,
    ) {
        for (_, op) in produced_ops {
            op.visit_dependencies(&mut |dep| {
                if let Some(count) = remaining_consumers.get_mut(&dep) {
                    *count = count.saturating_sub(1);
                    if *count == 0 && !targets.contains(&dep) && !graph.has_live_reference(dep) {
                        // All consumers within this execution have been
                        // processed — free the cached buffer.
                        if let Some(node) = graph.nodes.nodes.node_weight_mut(dep) {
                            node.cached = None;
                        }
                    }
                }
            });
        }
    }

    /// Like `release_dead_intermediates` but uses the compute graph's
    /// `visit_dependencies` instead of an Operation's. Used for map-layout
    /// and resize nodes that are resolved immediately without being lowered
    /// to an Operation.
    fn release_dead_intermediates_from_graph(
        graph: &mut ComputeGraphInner,
        produced_nodes: &[NodeIndex],
        remaining_consumers: &mut FxHashMap<NodeIndex, usize>,
        targets: &FxHashSet<NodeIndex>,
    ) {
        for &produced in produced_nodes {
            let mut deps = Vec::new();
            graph.visit_dependencies(produced, &mut |dep| {
                deps.push(dep);
            });
            for dep in deps {
                if let Some(count) = remaining_consumers.get_mut(&dep) {
                    *count = count.saturating_sub(1);
                    if *count == 0 && !targets.contains(&dep) && !graph.has_live_reference(dep) {
                        if let Some(node) = graph.nodes.nodes.node_weight_mut(dep) {
                            node.cached = None;
                        }
                    }
                }
            }
        }
    }

    fn try_prepare_in_place_slice_assign_copy(
        graph: &ComputeGraphInner,
        operation: &crate::slice_assign::SliceAssignOperation,
    ) -> Option<(TensorData, Vec<CopyBufferRecord>)> {
        if !operation.in_place {
            return None;
        }
        let input = graph.get_cached_result(operation.input)?;
        let value = graph.get_cached_result(operation.value)?;
        if input.datatype() != value.datatype() || operation.slices.len() != input.layout().rank() {
            return None;
        }

        let output = input.slice(&operation.slices);
        if output.layout().shape() != value.layout().shape()
            || !output.layout().inner_dim_contiguous()
            || !value.layout().inner_dim_contiguous()
        {
            return None;
        }

        let element_size = input.datatype().element_size();
        let shape = value.layout().shape();
        let row_elems = *shape.last()?;
        let copy_size = row_elems.checked_mul(element_size)? as u64;
        if copy_size == 0 || copy_size % wgpu::COPY_BUFFER_ALIGNMENT != 0 {
            return None;
        }

        let outer_rank = shape.len().saturating_sub(1);
        let outer_count = shape[..outer_rank]
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))?;
        let source_strides = value.layout().strides();
        let destination_strides = output.layout().strides();
        let source_base = value.layout().offset();
        let destination_base = output.layout().offset();
        let mut copies = Vec::with_capacity(outer_count);

        for linear in 0..outer_count {
            let mut remaining = linear;
            let mut source_element = source_base;
            let mut destination_element = destination_base;
            for dim in (0..outer_rank).rev() {
                let dim_len = shape[dim];
                let index = if dim_len == 0 { 0 } else { remaining % dim_len };
                remaining = if dim_len == 0 { 0 } else { remaining / dim_len };
                source_element = source_element.checked_add(index * source_strides[dim])?;
                destination_element =
                    destination_element.checked_add(index * destination_strides[dim])?;
            }

            let source_offset = source_element.checked_mul(element_size)? as u64;
            let destination_offset = destination_element.checked_mul(element_size)? as u64;
            if source_offset % wgpu::COPY_BUFFER_ALIGNMENT != 0
                || destination_offset % wgpu::COPY_BUFFER_ALIGNMENT != 0
            {
                return None;
            }
            copies.push(CopyBufferRecord {
                source: value.buffer().clone(),
                destination: input.buffer().clone(),
                source_offset,
                destination_offset,
                size: copy_size,
            });
        }

        Some((input.clone(), copies))
    }

    fn build_execution_graph(
        &mut self,
        graph: &ComputeGraphInner,
        node: NodeIndex,
    ) -> Option<ExecutionNodeIndex> {
        if self.resolved_set.contains(&node) {
            return None;
        }
        if let Some(&idx) = self.node_mapping.get(&node) {
            return Some(idx);
        }

        let node_data = graph
            .nodes
            .nodes
            .node_weight(node)
            .expect("Node not found in graph");
        let variant = node_data.variant.clone();

        // Add to execution graph
        let exec_idx = self.execution_graph.add_node(ExecutionNode {
            inner_idx: node,
            variant: variant.clone(),
        });
        self.node_mapping.insert(node, exec_idx);

        // Find dependencies
        let mut dependencies = Vec::new();
        variant.visit_dependencies(&mut |dependency| {
            dependencies.push(dependency);
        });

        for dependency in dependencies {
            if let Some(dep_exec_idx) = self.build_execution_graph(graph, dependency) {
                self.execution_graph.add_edge(dep_exec_idx, exec_idx, ());
            }
        }

        Some(exec_idx)
    }

    fn lower_node(&self, node: &ExecutionNode) -> Option<Arc<dyn Operation>> {
        match &node.variant {
            ComputeGraphNodeVariant::Nary(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::MatMul(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::Reduce(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::RmsNorm(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::FlashAttention(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::MapLayout(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::Resize(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::SliceAssign(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::QEmbedding(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::QMatMul(op) => Some(Arc::new(QMatMulOperation::new(
                op.input_datatype,
                &op.in_shape,
                op.input,
                op.matrix.clone(),
            ))),
            ComputeGraphNodeVariant::Dequantize(op) => Some(Arc::new(op.clone())),
            ComputeGraphNodeVariant::Tensor(_) => None, // Handled in execution loop
        }
    }

    // --- Rewrite Engine ---

    fn optimize(&mut self, graph: &mut ComputeGraphInner) {
        // Initialize worklist with all nodes
        let mut worklist: VecDeque<ExecutionNodeIndex> =
            self.execution_graph.node_indices().collect();
        let mut in_worklist: FxHashSet<ExecutionNodeIndex> = worklist.iter().copied().collect();

        while let Some(node_idx) = worklist.pop_front() {
            in_worklist.remove(&node_idx);

            if !self.execution_graph.contains_node(node_idx) {
                continue;
            }

            // Collect neighbors before optimization (they may need re-processing)
            let neighbors: Vec<_> = self
                .execution_graph
                .neighbors_undirected(node_idx)
                .collect();

            // 1. Fuse naries together (combine expression trees)
            // 2. Try to fuse resulting nary into specialized ops (reduce, matmul, etc.)
            let changed = self.try_fuse_naries(graph, node_idx)
                || self.try_fuse_into_reduce(graph, node_idx)
                || self.try_fuse_into_matmul(graph, node_idx);

            if changed {
                // Re-add the current node to worklist if it still exists
                if self.execution_graph.contains_node(node_idx) && !in_worklist.contains(&node_idx)
                {
                    worklist.push_back(node_idx);
                    in_worklist.insert(node_idx);
                }

                // Re-add neighbors that might be affected by this change
                for neighbor in neighbors {
                    if self.execution_graph.contains_node(neighbor)
                        && !in_worklist.contains(&neighbor)
                    {
                        worklist.push_back(neighbor);
                        in_worklist.insert(neighbor);
                    }
                }

                // Also add new neighbors that may have been created
                if self.execution_graph.contains_node(node_idx) {
                    for neighbor in self.execution_graph.neighbors_undirected(node_idx) {
                        if !in_worklist.contains(&neighbor) {
                            worklist.push_back(neighbor);
                            in_worklist.insert(neighbor);
                        }
                    }
                }
            }
        }
    }

    // Helpers
    fn add_physical_dependencies(
        &self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
        inputs: &[NodeIndex],
    ) {
        let inner_idx = self.execution_graph[node_idx].inner_idx;
        for &input in inputs {
            graph.nodes.nodes.add_edge(input, inner_idx, ());
        }
    }

    fn get_input_node_in_exec_graph(&self, inner_input: NodeIndex) -> Option<ExecutionNodeIndex> {
        self.node_mapping.get(&inner_input).copied()
    }

    fn check_cached(&self, graph: &ComputeGraphInner, inner_idx: NodeIndex) -> bool {
        graph.get_cached_result(inner_idx).is_some()
    }

    fn remove_node_if_dead(&mut self, node_idx: ExecutionNodeIndex) {
        if !self.execution_graph.contains_node(node_idx) {
            return;
        }
        if self
            .execution_graph
            .neighbors_directed(node_idx, petgraph::Direction::Outgoing)
            .count()
            == 0
        {
            // Collect incoming neighbors before removing
            let incoming: Vec<_> = self
                .execution_graph
                .neighbors_directed(node_idx, petgraph::Direction::Incoming)
                .collect();
            self.execution_graph.remove_node(node_idx);
            // Recursively check if dependencies are now dead
            for dep in incoming {
                self.remove_node_if_dead(dep);
            }
        }
    }

    // Rules

    /// Fuse a Nary operation with all of its Nary inputs.
    fn try_fuse_naries(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let node_variant = self.execution_graph[node_idx].variant.clone();

        let ComputeGraphNodeVariant::Nary(nary) = node_variant else {
            return false;
        };

        // Collect all fusible nary inputs
        let mut expression = nary.expression.clone();
        let mut all_inputs = nary.inputs.clone();
        let mut fused_execs = Vec::new();

        // Get the max storage buffers limit from GPU
        let max_storage_bindings =
            graph.device().limits().max_storage_buffers_per_shader_stage as usize;

        for (input_idx, &input_inner) in nary.inputs.iter().enumerate() {
            if self.check_cached(graph, input_inner) {
                continue;
            }
            let Some(input_exec) = self.get_input_node_in_exec_graph(input_inner) else {
                continue;
            };
            // Check if the node still exists (it may have been removed during optimization)
            if !self.execution_graph.contains_node(input_exec) {
                continue;
            }
            let ComputeGraphNodeVariant::Nary(input_nary) =
                &self.execution_graph[input_exec].variant
            else {
                continue;
            };

            // Inline: offset input nary's indices to append after current inputs
            let offset = all_inputs.len();
            let inlined = Self::offset_input_indices(&input_nary.expression, offset);
            let (new_expression, success) =
                Self::substitute_input_in_expr(&expression, input_idx, &inlined);

            // Only fuse if substitution was successful
            // If not, the expression still references the original input which must remain
            if success {
                // Check if fusing would exceed the GPU's per-stage buffer limit.
                // On Metal, `max_buffers_per_stage` is 31 and covers ALL buffer
                // types (storage + uniform) plus an implicit sizes buffer added
                // by wgpu. Each unique tensor input needs at least 1 storage
                // binding, and there will also be a small number of uniform
                // "info" bindings (for tensor shape/stride metadata), plus the
                // output storage binding, plus the wgpu sizes buffer.
                //
                // We use `max_storage_bindings` (which equals `max_buffers_per_stage`
                // on Metal = 31) as the hard ceiling and reserve slots for:
                //   - 1 output storage binding
                //   - 1 wgpu sizes buffer
                //   - up to `info_headroom` uniform info bindings
                let info_headroom = 4usize;
                let max_fused_inputs = max_storage_bindings.saturating_sub(2 + info_headroom);

                // Count unique inputs after potential merge (duplicates share a binding)
                let unique_inputs: FxHashSet<_> = all_inputs
                    .iter()
                    .chain(input_nary.inputs.iter())
                    .copied()
                    .collect();

                if unique_inputs.len() >= max_fused_inputs {
                    // Skip fusion - would exceed GPU binding limit
                    continue;
                }

                expression = new_expression;
                all_inputs.extend(input_nary.inputs.iter().copied());
                fused_execs.push((input_exec, input_nary.inputs.clone()));
            }
        }

        if fused_execs.is_empty() {
            return false;
        }

        // Deduplicate and remove unused inputs
        let (final_inputs, final_expression) = Self::deduplicate_inputs(all_inputs, expression);

        let new_nary = NaryOperation {
            inputs: final_inputs.clone(),
            expression: final_expression,
            shape: nary.shape.clone(),
            output_datatype: nary.output_datatype,
        };

        self.execution_graph[node_idx].variant = ComputeGraphNodeVariant::Nary(new_nary.clone());

        // Update graph edges
        for (input_exec, new_inputs) in fused_execs {
            if let Some(edge) = self.execution_graph.find_edge(input_exec, node_idx) {
                self.execution_graph.remove_edge(edge);
            }
            for &new_input in &new_inputs {
                if let Some(exec) = self.get_input_node_in_exec_graph(new_input)
                    && self.execution_graph.find_edge(exec, node_idx).is_none()
                {
                    self.execution_graph.add_edge(exec, node_idx, ());
                }
            }
            self.remove_node_if_dead(input_exec);
        }

        self.add_physical_dependencies(graph, node_idx, &new_nary.inputs);
        true
    }

    /// Add offset to all input indices in an expression.
    fn offset_input_indices(expr: &NaryExpr, offset: usize) -> NaryExpr {
        match expr {
            NaryExpr::Op { children, function } => NaryExpr::Op {
                children: children
                    .iter()
                    .map(|c| Self::offset_input_indices(c, offset))
                    .collect(),
                function: function.clone(),
            },
            NaryExpr::IndexedInput { input_idx, indices } => NaryExpr::IndexedInput {
                input_idx: input_idx + offset,
                indices: indices
                    .iter()
                    .map(|c| Self::offset_input_indices(c, offset))
                    .collect(),
            },
            NaryExpr::DimIndex(dim) => NaryExpr::DimIndex(*dim),
            NaryExpr::Scalar(value) => NaryExpr::Scalar(*value),
        }
    }

    /// Substitute IndexedInput(target_idx) with element-wise access with the replacement expression.
    /// Returns (new_expression, success) where success is true if all references to target_idx
    /// were successfully substituted. If false, the input should NOT be removed from the graph.
    fn substitute_input_in_expr(
        expr: &NaryExpr,
        target_idx: usize,
        replacement: &NaryExpr,
    ) -> (NaryExpr, bool) {
        /// Helper to extract input_idx from an IndexedInput with element-wise access
        fn get_elementwise_input_idx(expr: &NaryExpr) -> Option<usize> {
            match expr {
                NaryExpr::IndexedInput { input_idx, indices }
                    if NaryExpr::is_elementwise_indices(indices) =>
                {
                    Some(*input_idx)
                }
                _ => None,
            }
        }

        match expr {
            NaryExpr::Op { children, function } => {
                let mut all_success = true;
                let new_children: Vec<_> = children
                    .iter()
                    .map(|c| {
                        let (new_c, success) =
                            Self::substitute_input_in_expr(c, target_idx, replacement);
                        all_success &= success;
                        new_c
                    })
                    .collect();
                (
                    NaryExpr::Op {
                        children: new_children,
                        function: function.clone(),
                    },
                    all_success,
                )
            }
            NaryExpr::IndexedInput { input_idx, indices } => {
                if *input_idx == target_idx {
                    // Check if this is element-wise access
                    if NaryExpr::is_elementwise_indices(indices) {
                        // Element-wise can be fully replaced with any expression
                        (replacement.clone(), true)
                    } else {
                        // Custom indexing can only substitute if replacement is also element-wise
                        if let Some(new_idx) = get_elementwise_input_idx(replacement) {
                            let mut all_success = true;
                            let new_indices: Vec<_> = indices
                                .iter()
                                .map(|c| {
                                    let (new_c, success) =
                                        Self::substitute_input_in_expr(c, target_idx, replacement);
                                    all_success &= success;
                                    new_c
                                })
                                .collect();
                            (
                                NaryExpr::IndexedInput {
                                    input_idx: new_idx,
                                    indices: new_indices,
                                },
                                all_success,
                            )
                        } else {
                            // Cannot fuse complex expression into custom indexed input
                            let all_success = false;
                            let new_indices: Vec<_> = indices
                                .iter()
                                .map(|c| {
                                    let (new_c, _) =
                                        Self::substitute_input_in_expr(c, target_idx, replacement);
                                    new_c
                                })
                                .collect();
                            (
                                NaryExpr::IndexedInput {
                                    input_idx: *input_idx,
                                    indices: new_indices,
                                },
                                all_success,
                            )
                        }
                    }
                } else {
                    // Recurse into the index expressions
                    let mut all_success = true;
                    let new_indices: Vec<_> = indices
                        .iter()
                        .map(|c| {
                            let (new_c, s) =
                                Self::substitute_input_in_expr(c, target_idx, replacement);
                            all_success &= s;
                            new_c
                        })
                        .collect();
                    (
                        NaryExpr::IndexedInput {
                            input_idx: *input_idx,
                            indices: new_indices,
                        },
                        all_success,
                    )
                }
            }
            NaryExpr::DimIndex(dim) => (NaryExpr::DimIndex(*dim), true),
            NaryExpr::Scalar(value) => (NaryExpr::Scalar(*value), true),
        }
    }

    /// Remove unused inputs and deduplicate, returning new inputs and remapped expression.
    fn deduplicate_inputs(inputs: Vec<NodeIndex>, expr: NaryExpr) -> (Vec<NodeIndex>, NaryExpr) {
        // Collect which input indices are actually used
        let mut used_indices = FxHashSet::default();
        Self::collect_used_inputs(&expr, &mut used_indices);

        // Build mapping: old index -> new index, and collect only used inputs
        let mut new_inputs = Vec::new();
        let mut old_to_new = FxHashMap::default();

        for old_idx in used_indices.iter().copied().collect::<Vec<_>>() {
            let node = inputs[old_idx];
            // Check if this node already exists in new_inputs (deduplication)
            let new_idx = if let Some(existing) = new_inputs.iter().position(|&n| n == node) {
                existing
            } else {
                let idx = new_inputs.len();
                new_inputs.push(node);
                idx
            };
            old_to_new.insert(old_idx, new_idx);
        }

        let new_expr = Self::remap_input_indices(&expr, &old_to_new);
        (new_inputs, new_expr)
    }

    fn collect_used_inputs(expr: &NaryExpr, used: &mut FxHashSet<usize>) {
        match expr {
            NaryExpr::Op { children, .. } => {
                for child in children {
                    Self::collect_used_inputs(child, used);
                }
            }
            NaryExpr::IndexedInput { input_idx, indices } => {
                used.insert(*input_idx);
                for c in indices {
                    Self::collect_used_inputs(c, used);
                }
            }
            NaryExpr::DimIndex(_) => {}
            NaryExpr::Scalar(_) => {}
        }
    }

    fn remap_input_indices(expr: &NaryExpr, mapping: &FxHashMap<usize, usize>) -> NaryExpr {
        match expr {
            NaryExpr::Op { children, function } => NaryExpr::Op {
                children: children
                    .iter()
                    .map(|c| Self::remap_input_indices(c, mapping))
                    .collect(),
                function: function.clone(),
            },
            NaryExpr::IndexedInput { input_idx, indices } => NaryExpr::IndexedInput {
                input_idx: mapping[input_idx],
                indices: indices
                    .iter()
                    .map(|c| Self::remap_input_indices(c, mapping))
                    .collect(),
            },
            NaryExpr::DimIndex(dim) => NaryExpr::DimIndex(*dim),
            NaryExpr::Scalar(value) => NaryExpr::Scalar(*value),
        }
    }

    /// Try to extract a unary function chain from a node variant.
    /// Only Nary ops with a single input and element-wise access can be converted.
    fn try_get_unary_chain(variant: &ComputeGraphNodeVariant) -> Option<ExtractedUnaryChain> {
        match variant {
            ComputeGraphNodeVariant::Nary(nary) => nary.try_extract_unary_chain(),
            _ => None,
        }
    }

    fn try_fuse_into_reduce(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let node_variant = self.execution_graph[node_idx].variant.clone();

        let Some(el_op) = Self::try_get_unary_chain(&node_variant) else {
            return false;
        };

        let input_inner = el_op.value;
        if self.check_cached(graph, input_inner) {
            return false;
        }

        let Some(input_exec_idx) = self.get_input_node_in_exec_graph(input_inner) else {
            return false;
        };

        let input_variant = self.execution_graph[input_exec_idx].variant.clone();
        let ComputeGraphNodeVariant::Reduce(reduce_op) = input_variant else {
            return false;
        };

        let mut new_reduce = reduce_op.clone();
        let mut existing_post = new_reduce.post_element_wise.functions.clone();
        existing_post.extend(el_op.functions.functions.iter().cloned());
        new_reduce.post_element_wise =
            UnaryFunctionChain::new(existing_post, reduce_op.post_element_wise.input_datatype());

        self.execution_graph[node_idx].variant =
            ComputeGraphNodeVariant::Reduce(new_reduce.clone());

        let reduce_input_inner = reduce_op.value;
        if let Some(reduce_input_exec) = self.get_input_node_in_exec_graph(reduce_input_inner) {
            self.execution_graph
                .add_edge(reduce_input_exec, node_idx, ());
        }

        if let Some(edge) = self.execution_graph.find_edge(input_exec_idx, node_idx) {
            self.execution_graph.remove_edge(edge);
        }
        self.add_physical_dependencies(graph, node_idx, &[reduce_input_inner]);
        self.remove_node_if_dead(input_exec_idx);
        true
    }

    fn try_fuse_into_matmul(
        &mut self,
        graph: &mut ComputeGraphInner,
        node_idx: ExecutionNodeIndex,
    ) -> bool {
        let node_variant = self.execution_graph[node_idx].variant.clone();

        // Post-op: fuse elementwise after matmul
        if let Some(el_op) = Self::try_get_unary_chain(&node_variant) {
            let input_inner = el_op.value;
            if !self.check_cached(graph, input_inner)
                && let Some(input_exec_idx) = self.get_input_node_in_exec_graph(input_inner)
            {
                let input_variant = self.execution_graph[input_exec_idx].variant.clone();
                if let ComputeGraphNodeVariant::MatMul(matmul_op) = input_variant {
                    let mut new_matmul = matmul_op.clone();
                    let mut existing_post = new_matmul.post_element_wise.functions.clone();
                    existing_post.extend(el_op.functions.functions.iter().cloned());
                    new_matmul.post_element_wise = UnaryFunctionChain::new(
                        existing_post,
                        matmul_op.post_element_wise.input_datatype(),
                    );

                    self.execution_graph[node_idx].variant =
                        ComputeGraphNodeVariant::MatMul(new_matmul.clone());

                    let (first_inner, second_inner) = (matmul_op.first, matmul_op.second);
                    if let Some(idx) = self.get_input_node_in_exec_graph(first_inner) {
                        self.execution_graph.add_edge(idx, node_idx, ());
                    }
                    if let Some(idx) = self.get_input_node_in_exec_graph(second_inner) {
                        self.execution_graph.add_edge(idx, node_idx, ());
                    }
                    if let Some(edge) = self.execution_graph.find_edge(input_exec_idx, node_idx) {
                        self.execution_graph.remove_edge(edge);
                    }
                    self.add_physical_dependencies(graph, node_idx, &[first_inner, second_inner]);
                    self.remove_node_if_dead(input_exec_idx);
                    return true;
                }
            }
        }

        // Pre-op: fuse elementwise before matmul inputs
        if let ComputeGraphNodeVariant::MatMul(matmul_op) = &node_variant {
            let mut new_matmul = matmul_op.clone();
            let mut changed = false;

            // Check first input
            if !self.check_cached(graph, matmul_op.first)
                && let Some(first_exec) = self.get_input_node_in_exec_graph(matmul_op.first)
                && let Some(el_op) =
                    Self::try_get_unary_chain(&self.execution_graph[first_exec].variant)
            {
                new_matmul.first = el_op.value;
                let mut functions = el_op.functions.functions.clone();
                functions.extend(new_matmul.pre_element_wise[0].functions.iter().cloned());
                new_matmul.pre_element_wise[0] =
                    UnaryFunctionChain::new(functions, el_op.functions.input_datatype());
                changed = true;
            }

            // Check second input
            if !self.check_cached(graph, matmul_op.second)
                && let Some(second_exec) = self.get_input_node_in_exec_graph(matmul_op.second)
                && let Some(el_op) =
                    Self::try_get_unary_chain(&self.execution_graph[second_exec].variant)
            {
                new_matmul.second = el_op.value;
                let mut functions = el_op.functions.functions.clone();
                functions.extend(new_matmul.pre_element_wise[1].functions.iter().cloned());
                new_matmul.pre_element_wise[1] =
                    UnaryFunctionChain::new(functions, el_op.functions.input_datatype());
                changed = true;
            }

            if changed {
                self.execution_graph[node_idx].variant =
                    ComputeGraphNodeVariant::MatMul(new_matmul.clone());

                if new_matmul.first != matmul_op.first {
                    let old = self.get_input_node_in_exec_graph(matmul_op.first).unwrap();
                    if let Some(edge) = self.execution_graph.find_edge(old, node_idx) {
                        self.execution_graph.remove_edge(edge);
                    }
                    if let Some(new) = self.get_input_node_in_exec_graph(new_matmul.first) {
                        self.execution_graph.add_edge(new, node_idx, ());
                    }
                    self.remove_node_if_dead(old);
                }
                if new_matmul.second != matmul_op.second {
                    let old = self.get_input_node_in_exec_graph(matmul_op.second).unwrap();
                    if let Some(edge) = self.execution_graph.find_edge(old, node_idx) {
                        self.execution_graph.remove_edge(edge);
                    }
                    if let Some(new) = self.get_input_node_in_exec_graph(new_matmul.second) {
                        self.execution_graph.add_edge(new, node_idx, ());
                    }
                    self.remove_node_if_dead(old);
                }
                self.add_physical_dependencies(
                    graph,
                    node_idx,
                    &[new_matmul.first, new_matmul.second],
                );
                return true;
            }
        }

        false
    }
}
