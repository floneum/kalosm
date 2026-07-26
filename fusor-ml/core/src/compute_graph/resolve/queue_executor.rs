//! Execution of the resolver's lowered operation queue.
//!
//! The executor owns serial input gathering and output allocation, parallel
//! kernel-plan building, then ordered recording and command preparation. It
//! is used by every resolved graph; horizontal merging only changes individual
//! queue entries.

use super::merge_horizontal::MergedSegments;
use super::*;
use crate::mir::kernel_backend::DirectKernel;

/// Every node that observes `node`'s value, transitively.
///
/// Observations chain: one pass can record `a` as an observation of `b` and a
/// later one record `b` as an observation of `c`, and then caching `c` has to
/// reach `a` too. Following only the direct entries leaves the far end of such
/// a chain uncached, and its readers resolve to nothing.
fn observations_of(
    node: NodeIndex,
    shared_outputs: &FxHashMap<NodeIndex, Vec<NodeIndex>>,
) -> Vec<NodeIndex> {
    let mut observations = Vec::new();
    let mut pending = vec![node];
    let mut seen = FxHashSet::default();
    while let Some(current) = pending.pop() {
        let Some(direct) = shared_outputs.get(&current) else {
            continue;
        };
        for &observation in direct {
            if seen.insert(observation) {
                observations.push(observation);
                pending.push(observation);
            }
        }
    }
    observations
}

fn cache_output(
    graph: &mut ComputeGraphInner,
    node: NodeIndex,
    result: &TensorData,
    shared_outputs: &FxHashMap<NodeIndex, Vec<NodeIndex>>,
) {
    graph.set_cached_result(node, result.clone());
    for observation in observations_of(node, shared_outputs) {
        graph.set_cached_result(observation, result.clone());
    }
}

fn record_shared_outputs(
    recorder: &std::cell::RefCell<flush_replay::PlanRecorder>,
    node: NodeIndex,
    result: &TensorData,
    shared_outputs: &FxHashMap<NodeIndex, Vec<NodeIndex>>,
) {
    for observation in observations_of(node, shared_outputs) {
        recorder
            .borrow_mut()
            .record_shared_alias(observation, result, node);
    }
}

/// One entry of the three-phase queue, preserving queue order.
enum QueueStep {
    View {
        node: NodeIndex,
        result: TensorData,
        deps: Vec<NodeIndex>,
    },
    CopyAssign {
        node: NodeIndex,
        copies: Vec<CopyBufferRecord>,
        op: QueuedOperation,
    },
    Work(usize),
}

enum QueueWorkKind {
    Operation {
        inputs: Vec<MirValue>,
        workgroup_shape: crate::mir::workgroup_shape::WorkgroupShape,
        resolved: TensorData,
        /// Node whose dead buffer this output claimed, if any.
        claimed_from: Option<NodeIndex>,
    },
    Merged {
        segment_inputs: Vec<Vec<MirValue>>,
        /// One entry per segment output, in segment/statement order (regions
        /// contribute several outputs per segment), with the node whose dead
        /// buffer the output claimed, if any.
        outputs: Vec<(NodeIndex, TensorData, Option<NodeIndex>)>,
    },
}

struct QueueWork {
    node: NodeIndex,
    op: QueuedOperation,
    kind: QueueWorkKind,
    built: std::sync::Mutex<Option<BuiltWork>>,
}

struct BuiltWork {
    kernels: Vec<DirectKernel>,
    prepared: Vec<Option<(PreparedDirectDispatch, String)>>,
}

#[cfg(not(target_arch = "wasm32"))]
const MIN_PARALLEL_BUILD_QUEUE: usize = 16;
#[cfg(not(target_arch = "wasm32"))]
const MIN_PARALLEL_BUILD_REMAINDER: usize = 4;
#[cfg(not(target_arch = "wasm32"))]
const COLD_BUILD_THRESHOLD: std::time::Duration = std::time::Duration::from_millis(1);

#[cfg(not(target_arch = "wasm32"))]
fn should_parallelize_build_remainder(
    worker_count: usize,
    remaining: usize,
    probe_elapsed: std::time::Duration,
) -> bool {
    worker_count > 1
        && remaining >= MIN_PARALLEL_BUILD_REMAINDER
        && probe_elapsed >= COLD_BUILD_THRESHOLD
}

pub(super) fn merged_plan_cache_key(
    merged: &MergedSegments,
    segment_inputs: &[Vec<MirValue>],
) -> crate::mir::kernel_backend::KernelCacheKey {
    // Declared here because its `TypeId` — declaration site and all — stamps
    // every merged plan key already in the persistent store.
    struct MergedPlanKernelVariant;
    super::plan_cache::merged_segments_key(
        crate::mir::kernel_backend::KernelVariantKey::of::<MergedPlanKernelVariant>(),
        merged,
        segment_inputs,
    )
}

fn build_queue_work(
    work: &QueueWork,
    graph: &ComputeGraphInner,
    device: &crate::Device,
) -> BuiltWork {
    let build_timer = std::time::Instant::now();
    let kernels = match (&work.op, &work.kind) {
        (
            QueuedOperation::Operation(operation),
            QueueWorkKind::Operation {
                inputs,
                workgroup_shape,
                ..
            },
        ) => {
            let build_kernels = || {
                operation
                    .build_direct_kernel_plan(graph, workgroup_shape, inputs)
                    .unwrap_or_else(|error| panic!("{error}"))
                    .into_kernels()
            };
            let kernel_key = structural_kernel_key(operation.as_ref(), inputs, workgroup_shape);
            let kernels = super::run::resolve_cached_kernel_plan(
                device.kernel_cache(),
                kernel_key,
                super::run::kernel_plan_binding_buffers(inputs),
                build_kernels,
            );
            kernels
        }
        (QueuedOperation::Merged(merged), QueueWorkKind::Merged { segment_inputs, .. }) => {
            // Merged kernels go through the same plan cache as single ops:
            // buffers are presented flattened in segment order, and the
            // insert path verifies that order matches the kernel's true
            // binding order (folded or deduplicated plans silently skip).
            let expected: Vec<std::sync::Arc<wgpu::Buffer>> = segment_inputs
                .iter()
                .flatten()
                .filter_map(|value| match value {
                    MirValue::Tensor(tensor) => Some(tensor.buffer().clone()),
                    MirValue::QMatrix(matrix) => Some(matrix.buffer().clone()),
                    MirValue::Integer(_) | MirValue::Float(_) => None,
                })
                .collect();
            let plan_key = merged_plan_cache_key(merged, segment_inputs);
            if let Some(kernels) = device.kernel_cache().kernel_plan_cache().get_many(
                device.kernel_cache(),
                plan_key,
                &[&expected],
            ) {
                return finish_queue_build(build_timer, kernels, device);
            }
            let built = match merged {
                MergedSegments::Row(segments) => {
                    crate::row_program::build_merged_row_program_kernel(
                        graph,
                        &segments
                            .iter()
                            .map(|(_, op)| op.clone())
                            .collect::<Vec<_>>(),
                        segment_inputs,
                    )
                }
                MergedSegments::MatMul(segments) => crate::matmul::build_merged_matmul_kernel(
                    graph,
                    &segments
                        .iter()
                        .map(|(_, op)| op.clone())
                        .collect::<Vec<_>>(),
                    segment_inputs,
                ),
                MergedSegments::Region(segments) => crate::nary_direct::build_merged_region_kernel(
                    graph,
                    &segments
                        .iter()
                        .map(|(_, op)| op.clone())
                        .collect::<Vec<_>>(),
                    segment_inputs,
                ),
            };
            match built {
                Some(kernel) => {
                    device.kernel_cache().kernel_plan_cache().insert_many(
                        plan_key,
                        std::slice::from_ref(&kernel),
                        &[&expected],
                    );
                    vec![kernel]
                }
                None if matches!(merged, MergedSegments::Region(_)) => {
                    // Region fallback: one standalone region kernel per
                    // segment. Replay records the resulting kernel batch.
                    let MergedSegments::Region(segments) = merged else {
                        unreachable!("matched above");
                    };
                    let kernels = segments
                        .iter()
                        .zip(segment_inputs)
                        .map(|((_, op), inputs)| {
                            crate::nary_direct::build_merged_region_kernel(
                                graph,
                                std::slice::from_ref(op),
                                std::slice::from_ref(inputs),
                            )
                            .unwrap_or_else(|| {
                                panic!("region fallback did not provide a kernel: {}", op.name())
                            })
                        })
                        .collect();
                    kernels
                }
                None => {
                    // Fallback: per-segment kernels. Replay records the
                    // resulting kernel batch with the same output slots.
                    let max_subgroup_size = device.max_subgroup_size();
                    let kernels = merged
                        .segment_ops()
                        .into_iter()
                        .zip(segment_inputs)
                        .flat_map(|((_, op), inputs)| {
                            let constraints = op.workgroup_shape_constraints(device);
                            let workgroup_shape = constraints
                                .solve(max_subgroup_size, &device.limits())
                                .unwrap_or_else(|| {
                                    panic!("failed to solve workgroup shape for merged fallback")
                                });
                            op.build_direct_kernel_plan(graph, &workgroup_shape, inputs)
                                .unwrap_or_else(|error| panic!("{error}"))
                                .into_kernels()
                        })
                        .collect();
                    kernels
                }
            }
        }
        _ => unreachable!("queue work kind matches its queued operation"),
    };
    finish_queue_build(build_timer, kernels, device)
}

/// Prepare dispatches (which also compiles shaders and pipelines, here on
/// the parallel build workers) and assemble the phase-2 result.
fn finish_queue_build(
    build_timer: std::time::Instant,
    kernels: Vec<crate::mir::kernel_backend::DirectKernel>,
    device: &crate::Device,
) -> BuiltWork {
    let prepared = kernels
        .iter()
        .map(|kernel| {
            kernel
                .prepare_dispatch(device.kernel_cache())
                .map(|dispatch| (dispatch, kernel.name().to_string()))
        })
        .collect();
    if device.config().trace_build_times {
        let total = build_timer.elapsed();
        if total.as_millis() >= 2 {
            eprintln!(
                "build_time total={total:?} first={}",
                kernels.first().map(|k| k.name()).unwrap_or("")
            );
        }
    }
    BuiltWork { kernels, prepared }
}

impl Resolver {
    /// Three-phase queue execution for all resolved graphs: serial input
    /// gathering and output caching (queue order), parallel kernel building
    /// and dispatch preparation, then serial recording, encoding, and
    /// release accounting in exactly the original queue order.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn execute_queue(
        recorder: Option<&std::cell::RefCell<flush_replay::PlanRecorder>>,
        graph: &mut ComputeGraphInner,
        device: &crate::Device,
        max_subgroup_size: u32,
        queued_operations: Vec<(NodeIndex, QueuedOperation)>,
        remaining_consumers: &mut FxHashMap<NodeIndex, usize>,
        target_set: &FxHashSet<NodeIndex>,
        shared_outputs: &FxHashMap<NodeIndex, Vec<NodeIndex>>,
        ledger: &mut super::alloc_reuse::BufferLedger,
        commands: &mut Vec<CommandRecord>,
        host_profile: &mut ResolveHostProfile,
        host_trace: bool,
        on_dispatch_name: &mut dyn FnMut(&str) -> Option<String>,
    ) {
        // Phase 1: gather inputs, allocate outputs, cache results.
        let gather_start = host_trace.then(Instant::now);
        let mut steps = Vec::with_capacity(queued_operations.len());
        let mut work: Vec<QueueWork> = Vec::new();
        for (node, queued_operation) in queued_operations {
            let view_result = if let Some(node_data) = graph.nodes.nodes.node_weight(node) {
                match &node_data.variant {
                    ComputeGraphNodeVariant::View(view) => graph
                        .get_cached_result(view.input)
                        .and_then(|input| view.try_map_tensor(input)),
                    _ => None,
                }
            } else {
                None
            };
            if let Some(result) = view_result {
                let mut deps = Vec::new();
                graph.visit_dependencies(node, &mut |dep| deps.push(dep));
                cache_output(graph, node, &result, shared_outputs);
                ledger.note_transient(result.buffer());
                ledger.consume(graph, &deps, target_set);
                steps.push(QueueStep::View { node, result, deps });
                continue;
            }
            let slice_copy = graph.nodes.nodes.node_weight(node).and_then(|node_data| {
                let ComputeGraphNodeVariant::Assign(slice_assign) = &node_data.variant else {
                    return None;
                };
                Self::try_prepare_in_place_slice_assign_copy(graph, slice_assign)
            });
            if let Some((output, copies)) = slice_copy {
                cache_output(graph, node, &output, shared_outputs);
                ledger.note_transient(output.buffer());
                for copy in &copies {
                    ledger.note_transient(&copy.source);
                    ledger.note_transient(&copy.destination);
                }
                let mut deps = Vec::new();
                queued_operation.visit_dependencies(&mut |dep| deps.push(dep));
                ledger.consume(graph, &deps, target_set);
                steps.push(QueueStep::CopyAssign {
                    node,
                    copies,
                    op: queued_operation,
                });
                continue;
            }
            match &queued_operation {
                QueuedOperation::Operation(operation) => {
                    let mut inputs = operation.inputs(graph);
                    let output_value = operation.output(graph, &inputs);
                    let MirValue::Tensor(mut resolved) = output_value else {
                        panic!("Kernel input value is not a tensor");
                    };
                    // Cache the output before the death accounting: a
                    // source is only releasable once every alive-uncached
                    // descendant (this very operation) is cached.
                    cache_output(graph, node, &resolved, shared_outputs);
                    let mut deps = Vec::new();
                    queued_operation.visit_dependencies(&mut |dep| deps.push(dep));
                    ledger.consume(graph, &deps, target_set);
                    let mut claimed_from = None;
                    if ledger.enabled() {
                        let out_ptr = Arc::as_ptr(resolved.buffer()) as usize;
                        let forbidden: FxHashSet<usize> = inputs
                            .iter()
                            .filter_map(|value| match value {
                                MirValue::Tensor(tensor) => {
                                    let ptr = Arc::as_ptr(tensor.buffer()) as usize;
                                    (ptr != out_ptr).then_some(ptr)
                                }
                                _ => None,
                            })
                            .collect();
                        if let Some(swapped) = ledger.try_claim(node, &resolved, &forbidden) {
                            for value in inputs.iter_mut() {
                                if let MirValue::Tensor(tensor) = value
                                    && Arc::as_ptr(tensor.buffer()) as usize == out_ptr
                                {
                                    *value = swapped.clone().into();
                                }
                            }
                            resolved = swapped;
                            claimed_from = ledger.chosen_source(node);
                            cache_output(graph, node, &resolved, shared_outputs);
                        }
                    }
                    ledger.note_alloc(&resolved);
                    for value in &inputs {
                        if let MirValue::Tensor(tensor) = value {
                            ledger.note_transient(tensor.buffer());
                        }
                    }
                    ledger.note_transient(resolved.buffer());
                    let constraints = operation.workgroup_shape_constraints(device);
                    let workgroup_shape = constraints
                        .solve(max_subgroup_size, &device.limits())
                        .unwrap_or_else(|| {
                            panic!(
                                "Failed to find a valid workgroup shape for constraints {constraints:?}"
                            )
                        });
                    steps.push(QueueStep::Work(work.len()));
                    work.push(QueueWork {
                        node,
                        op: queued_operation,
                        kind: QueueWorkKind::Operation {
                            inputs,
                            workgroup_shape,
                            resolved,
                            claimed_from,
                        },
                        built: std::sync::Mutex::new(None),
                    });
                }
                QueuedOperation::Merged(merged) => {
                    let mut segment_inputs: Vec<Vec<MirValue>> = Vec::new();
                    let mut outputs: Vec<(NodeIndex, TensorData, Option<NodeIndex>)> = Vec::new();
                    if let MergedSegments::Region(segments) = merged {
                        let device = graph.device();
                        // A segment may write an output over one of its own
                        // input buffers only when no other segment of this
                        // dispatch binds that buffer (concurrent workgroups)
                        // — count cached-buffer pointers across the whole
                        // dispatch and require the source to be unique.
                        let mut dispatch_ptr_uses: FxHashMap<usize, u32> = FxHashMap::default();
                        for (_, op) in segments {
                            for idx in &op.inputs {
                                if let Some(cached) = graph.get_cached_result(*idx) {
                                    *dispatch_ptr_uses
                                        .entry(Arc::as_ptr(cached.buffer()) as usize)
                                        .or_insert(0) += 1;
                                }
                            }
                        }
                        // Segments share one unsynchronized dispatch, so a
                        // scratch claim must avoid every segment's reads, not
                        // just the claiming segment's own.
                        let dispatch_reads: FxHashSet<usize> =
                            dispatch_ptr_uses.keys().copied().collect();
                        for (_, op) in segments {
                            let values: Vec<MirValue> = op
                                .inputs
                                .iter()
                                .map(|idx| {
                                    graph
                                        .get_result(*idx)
                                        .expect("region inputs resolve before the region")
                                        .into()
                                })
                                .collect();
                            // Register the gathered input clones before any
                            // claim so the reference accounting that guards
                            // in-place claims sees them.
                            for value in &values {
                                if let MirValue::Tensor(tensor) = value {
                                    ledger.note_transient(tensor.buffer());
                                }
                            }
                            let mut values = values;
                            let reads = op.input_read_summary();
                            let mut slot_claimed = vec![false; op.inputs.len()];
                            // Cache every output before the death accounting:
                            // sources are only releasable once this region
                            // (their last alive-uncached descendant) counts
                            // as cached.
                            let mut fresh_outputs = Vec::new();
                            for statement in &op.statements {
                                let Some(out_node) = statement.output else {
                                    continue;
                                };
                                let output = TensorData::new_for_shape(
                                    &device,
                                    &op.shape,
                                    statement.datatype,
                                );
                                cache_output(graph, out_node, &output, shared_outputs);
                                fresh_outputs.push(output);
                            }
                            {
                                let mut deps = Vec::new();
                                op.visit_dependencies(&mut |dep| deps.push(dep));
                                ledger.consume(graph, &deps, target_set);
                            }
                            let mut fresh_outputs = fresh_outputs.into_iter();
                            for (position, statement) in op.statements.iter().enumerate() {
                                let Some(out_node) = statement.output else {
                                    continue;
                                };
                                let mut output = fresh_outputs
                                    .next()
                                    .expect("one fresh output per statement");
                                let mut claimed_from = None;
                                // Write in place over an input this statement
                                // is the last reader of: per-thread the load
                                // precedes the store and threads own disjoint
                                // elements, so identity reads stay exact.
                                for (slot, source) in op.inputs.iter().enumerate() {
                                    if slot_claimed[slot]
                                        || !reads[slot].identity_only
                                        || reads[slot].last_reader != Some(position)
                                    {
                                        continue;
                                    }
                                    let unique = graph
                                        .get_cached_result(*source)
                                        .map(|cached| Arc::as_ptr(cached.buffer()) as usize)
                                        .and_then(|ptr| dispatch_ptr_uses.get(&ptr))
                                        == Some(&1);
                                    if !unique {
                                        continue;
                                    }
                                    if let Some(swapped) = ledger.try_claim_in_place(
                                        out_node, &output, *source, graph, target_set,
                                    ) {
                                        output = swapped;
                                        claimed_from = Some(*source);
                                        slot_claimed[slot] = true;
                                        break;
                                    }
                                }
                                if claimed_from.is_none()
                                    && let Some(swapped) =
                                        ledger.try_claim(out_node, &output, &dispatch_reads)
                                {
                                    output = swapped;
                                    claimed_from = ledger.chosen_source(out_node);
                                }
                                if claimed_from.is_some() {
                                    cache_output(graph, out_node, &output, shared_outputs);
                                }
                                ledger.note_alloc(&output);
                                ledger.note_transient(output.buffer());
                                values.push(output.clone().into());
                                outputs.push((out_node, output, claimed_from));
                            }
                            segment_inputs.push(values);
                        }
                    } else {
                        for (seg_node, op) in merged.segment_ops() {
                            let inputs = op.inputs(graph);
                            let MirValue::Tensor(output) = op.output(graph, &inputs) else {
                                panic!("merged segment output is not a tensor");
                            };
                            cache_output(graph, seg_node, &output, shared_outputs);
                            ledger.note_alloc(&output);
                            for value in &inputs {
                                if let MirValue::Tensor(tensor) = value {
                                    ledger.note_transient(tensor.buffer());
                                }
                            }
                            ledger.note_transient(output.buffer());
                            outputs.push((seg_node, output, None));
                            segment_inputs.push(inputs);
                        }
                        let mut deps = Vec::new();
                        queued_operation.visit_dependencies(&mut |dep| deps.push(dep));
                        ledger.consume(graph, &deps, target_set);
                    }
                    steps.push(QueueStep::Work(work.len()));
                    work.push(QueueWork {
                        node,
                        op: queued_operation,
                        kind: QueueWorkKind::Merged {
                            segment_inputs,
                            outputs,
                        },
                        built: std::sync::Mutex::new(None),
                    });
                }
            }
        }
        // Allocation is complete: releases past this point free buffers no
        // claim can use anymore.
        ledger.freeze();
        if let Some(start) = gather_start {
            host_profile.inputs += start.elapsed();
        }

        // Phase 2: build kernels and prepare dispatches. Builds are pure
        // functions of (operation, layouts, buffers); the shared kernel
        // caches are internally synchronized.
        let build_start = host_trace.then(Instant::now);
        #[cfg(target_arch = "wasm32")]
        for item in &work {
            *item.built.lock().unwrap() = Some(build_queue_work(item, graph, device));
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let workers = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                .min(work.len())
                .max(1);
            // Tiny queues build serially. Larger queues probe on the caller
            // thread until a build is measurably cold. A warm plan-cache hit
            // costs much less than creating a fresh worker cohort, while the
            // first shader/pipeline miss still moves the remaining cold work
            // onto parallel workers.
            if workers <= 1 || work.len() < MIN_PARALLEL_BUILD_QUEUE {
                for item in &work {
                    *item.built.lock().unwrap() = Some(build_queue_work(item, graph, device));
                }
            } else {
                let mut next_index = 0;
                while let Some(item) = work.get(next_index) {
                    let probe_start = std::time::Instant::now();
                    *item.built.lock().unwrap() = Some(build_queue_work(item, graph, device));
                    next_index += 1;
                    if should_parallelize_build_remainder(
                        workers,
                        work.len() - next_index,
                        probe_start.elapsed(),
                    ) {
                        break;
                    }
                }

                if next_index < work.len() {
                    let remaining_workers = workers.min(work.len() - next_index);
                    let next = std::sync::atomic::AtomicUsize::new(next_index);
                    let graph_ref: &ComputeGraphInner = graph;
                    std::thread::scope(|scope| {
                        for _ in 0..remaining_workers {
                            scope.spawn(|| {
                                loop {
                                    let index =
                                        next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                    let Some(item) = work.get(index) else { break };
                                    let built = build_queue_work(item, graph_ref, device);
                                    *item.built.lock().unwrap() = Some(built);
                                }
                            });
                        }
                    });
                }
            }
        }
        if let Some(start) = build_start {
            host_profile.build_kernel += start.elapsed();
        }

        // Phase 3: record, encode, and release in queue order.
        let encode_start = host_trace.then(Instant::now);
        let mut consumers = super::execution::NodeConsumers {
            counts: remaining_consumers,
            targets: target_set,
        };
        for step in steps {
            match step {
                QueueStep::View { node, result, deps } => {
                    if let Some(recorder) = recorder {
                        recorder
                            .borrow_mut()
                            .record_view_alias(node, &result, &deps);
                        record_shared_outputs(recorder, node, &result, shared_outputs);
                    }
                    super::execution::release_consumed(
                        graph,
                        &mut consumers,
                        Some(ledger),
                        |release| deps.into_iter().for_each(release),
                    );
                }
                QueueStep::CopyAssign { node, copies, op } => {
                    if let Some(recorder) = recorder {
                        let output = graph
                            .get_cached_result(node)
                            .expect("copy-assign output cached in phase 1")
                            .clone();
                        recorder.borrow_mut().record_copy_assign(node, &output, &op);
                    }
                    commands.extend(copies.into_iter().map(CommandRecord::CopyBuffer));
                    super::execution::release_consumed(
                        graph,
                        &mut consumers,
                        Some(ledger),
                        |release| op.visit_dependencies(release),
                    );
                }
                QueueStep::Work(index) => {
                    let item = &work[index];
                    let built = item
                        .built
                        .lock()
                        .unwrap()
                        .take()
                        .expect("queue work built in phase 2");
                    if let Some(recorder) = recorder {
                        match (&item.op, &item.kind) {
                            (
                                QueuedOperation::Operation(_),
                                QueueWorkKind::Operation {
                                    resolved,
                                    claimed_from,
                                    ..
                                },
                            ) => {
                                recorder.borrow_mut().record_dispatch(
                                    item.node,
                                    &built.kernels,
                                    resolved,
                                    &item.op,
                                    *claimed_from,
                                );
                                record_shared_outputs(
                                    recorder,
                                    item.node,
                                    resolved,
                                    shared_outputs,
                                );
                            }
                            (
                                QueuedOperation::Merged(merged),
                                QueueWorkKind::Merged { outputs, .. },
                            ) => {
                                let node_outputs: Vec<(NodeIndex, &TensorData, Option<NodeIndex>)> =
                                    outputs
                                        .iter()
                                        .map(|(node, output, claimed)| (*node, output, *claimed))
                                        .collect();
                                recorder.borrow_mut().record_merged_dispatch(
                                    &node_outputs,
                                    &built.kernels,
                                    merged,
                                );
                                for (node, output, _) in outputs {
                                    record_shared_outputs(recorder, *node, output, shared_outputs);
                                }
                            }
                            _ => unreachable!("queue work kind matches its queued operation"),
                        }
                    }
                    for (dispatch, name) in built.prepared.into_iter().flatten() {
                        let category = on_dispatch_name(&name);
                        commands.push(CommandRecord::Dispatch(DispatchRecord {
                            dispatch,
                            name,
                            category,
                        }));
                    }
                    super::execution::release_consumed(
                        graph,
                        &mut consumers,
                        Some(ledger),
                        |release| item.op.visit_dependencies(release),
                    );
                }
            }
        }
        if let Some(start) = encode_start {
            host_profile.prepare_dispatch += start.elapsed();
        }
    }
}

/// Where a profiled resolve writes its per-dispatch timestamps.
pub(super) struct TimestampPlan<'a> {
    pub(super) query_set: &'a wgpu::QuerySet,
    /// Timestamps ride inside the shared passes; without the feature every
    /// dispatch takes its own pass and the writes land on its boundaries.
    pub(super) inside_pass: bool,
}

/// Encode a command stream with the resolver's pass and submit chunking.
///
/// Intermediate chunks are handed to `submit_chunk`; the final encoder is
/// returned so the caller can append tail work before its synchronization
/// boundary and final submit.
pub(super) fn encode_command_records(
    device: &crate::Device,
    commands: &[CommandRecord],
    total_kernels: usize,
    timestamps: Option<TimestampPlan<'_>>,
    mut command_encoder: wgpu::CommandEncoder,
    mut submit_chunk: impl FnMut(wgpu::CommandEncoder, bool),
) -> wgpu::CommandEncoder {
    let dispatches_per_pass = super::run::dispatches_per_pass(device, total_kernels);
    let dispatches_per_submit = super::run::dispatches_per_submit(device, total_kernels);
    let wait_after_chunk_submit = device.backend() == wgpu::Backend::Metal;
    let mut command_index = 0usize;
    let mut dispatch_index = 0usize;
    let mut dispatches_in_submit = 0usize;
    let mut encoder_has_commands = false;
    let mut pass_segments = 0usize;
    let mut copy_records = 0usize;

    while command_index < commands.len() {
        if encoder_has_commands && dispatches_in_submit >= dispatches_per_submit {
            let next_encoder =
                device
                    .wgpu_device()
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Resolver Encoder"),
                    });
            let ready_encoder = std::mem::replace(&mut command_encoder, next_encoder);
            submit_chunk(ready_encoder, wait_after_chunk_submit);
            encoder_has_commands = false;
            dispatches_in_submit = 0;
        }

        match &commands[command_index] {
            CommandRecord::CopyBuffer(copy) => {
                command_encoder.copy_buffer_to_buffer(
                    &copy.source,
                    copy.source_offset,
                    &copy.destination,
                    copy.destination_offset,
                    copy.size,
                );
                copy_records += 1;
                encoder_has_commands = true;
                command_index += 1;
            }
            CommandRecord::Dispatch(record) => {
                if let Some(plan) = &timestamps
                    && !plan.inside_pass
                {
                    let mut pass =
                        command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some(record.name.as_str()),
                            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                                query_set: plan.query_set,
                                beginning_of_pass_write_index: Some((dispatch_index * 2) as u32),
                                end_of_pass_write_index: Some((dispatch_index * 2 + 1) as u32),
                            }),
                        });
                    record.dispatch.run(&mut pass);
                    drop(pass);
                    pass_segments += 1;
                    dispatch_index += 1;
                    dispatches_in_submit += 1;
                    encoder_has_commands = true;
                    command_index += 1;
                    continue;
                }

                let mut pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Resolver Direct Kernels"),
                    timestamp_writes: None,
                });
                pass_segments += 1;
                let mut pass_dispatches = 0usize;
                while command_index < commands.len()
                    && pass_dispatches < dispatches_per_pass
                    && dispatches_in_submit < dispatches_per_submit
                {
                    let CommandRecord::Dispatch(record) = &commands[command_index] else {
                        break;
                    };
                    if let Some(plan) = &timestamps {
                        pass.write_timestamp(plan.query_set, (dispatch_index * 2) as u32);
                    }
                    pass.push_debug_group(&record.name);
                    record.dispatch.run(&mut pass);
                    pass.pop_debug_group();
                    if let Some(plan) = &timestamps {
                        pass.write_timestamp(plan.query_set, (dispatch_index * 2 + 1) as u32);
                    }
                    dispatch_index += 1;
                    dispatches_in_submit += 1;
                    command_index += 1;
                    pass_dispatches += 1;
                    encoder_has_commands = true;
                }
            }
        }
    }
    if cfg!(target_arch = "wasm32") || device.config().trace_resolve_host {
        tracing::info!(
            "resolve_pass_layout kernels={total_kernels} passes={pass_segments} copies={copy_records}"
        );
    }

    command_encoder
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::{Device, Tensor};

    #[test]
    fn warm_build_probe_keeps_queue_serial() {
        assert!(!should_parallelize_build_remainder(
            8,
            32,
            COLD_BUILD_THRESHOLD - std::time::Duration::from_nanos(1),
        ));
    }

    #[test]
    fn cold_build_probe_parallelizes_useful_remainder() {
        assert!(should_parallelize_build_remainder(
            8,
            MIN_PARALLEL_BUILD_REMAINDER,
            COLD_BUILD_THRESHOLD,
        ));
    }

    #[test]
    fn cold_build_probe_does_not_spawn_for_tiny_remainder() {
        assert!(!should_parallelize_build_remainder(
            8,
            MIN_PARALLEL_BUILD_REMAINDER - 1,
            COLD_BUILD_THRESHOLD,
        ));
        assert!(!should_parallelize_build_remainder(
            1,
            32,
            COLD_BUILD_THRESHOLD,
        ));
    }

    #[test]
    fn shared_eclass_dispatches_once_and_caches_every_observation() {
        pollster::block_on(async {
            let Ok(device) = Device::new().await else {
                return;
            };
            let input = Tensor::new(&device, &[1.0f32, 2.0, 3.0, 4.0]);
            let left = &input * 2.0;
            let right = &input * 2.0;
            let targets = vec![left.data().key, right.data().key];
            let (kernels, same_buffer) = device.compute_graph().with_mut(|graph| {
                let mut resolver = Resolver::new_batch(graph, targets.clone());
                let mut removed = Vec::new();
                let result = resolver.run(graph, &mut removed);
                let left = graph.get_cached_result(targets[0]).unwrap();
                let right = graph.get_cached_result(targets[1]).unwrap();
                (
                    result.total_kernels,
                    std::sync::Arc::ptr_eq(left.buffer(), right.buffer()),
                )
            });
            device.poll_wait();
            assert_eq!(kernels, 1);
            assert!(same_buffer);
        });
    }
}
