use super::*;

impl Resolver {
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
            let optimize_limit = optimize_node_limit();
            let skip_large_graph_optimize =
                optimize_limit != 0 && self.execution_graph.node_count() > optimize_limit;
            let skip_decode_optimize = skip_large_graph_optimize
                && self.is_single_token_qmatmul_graph()
                && std::env::var_os("FUSOR_RESOLVE_OPTIMIZE_DECODE_GRAPHS").is_none();
            if std::env::var_os("FUSOR_RESOLVE_SKIP_OPTIMIZE").is_none() {
                if skip_large_graph_optimize {
                    self.optimize_large_graph(graph);
                } else {
                    self.optimize(graph);
                }
            }
            if let Some(start) = start {
                host_profile.optimize += start.elapsed();
            }
            if host_trace && skip_large_graph_optimize {
                eprintln!(
                    "resolve_host_profile optimize_large_graph node_count={} limit={optimize_limit} skipped_decode={skip_decode_optimize}",
                    self.execution_graph.node_count(),
                );
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
        for (node, queued_operation) in queued_operations {
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
                    commands.extend(copies.into_iter().map(CommandRecord::CopyBuffer));
                    let start = host_trace.then(Instant::now);
                    Self::release_dead_intermediates(
                        graph,
                        &[&queued_operation],
                        &mut remaining_consumers,
                        &target_set,
                    );
                    if let Some(start) = start {
                        host_profile.release += start.elapsed();
                    }
                    continue;
                }

                let start = host_trace.then(Instant::now);
                let new_inputs = queued_operation.inputs(graph);
                if let Some(start) = start {
                    let elapsed = start.elapsed();
                    host_profile.inputs += elapsed;
                    if let Some(category) = operation_category {
                        host_category_profile.entry(category).or_default().inputs += elapsed;
                    }
                }
                if let QueuedOperation::QMatMul(qmatmul) = &queued_operation {
                    let start = host_trace.then(Instant::now);
                    let result = qmatmul.output(graph, &new_inputs);
                    let MirValue::Tensor(resolved) = result else {
                        panic!("QMatMul output value is not a tensor");
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
                    let constraints = qmatmul.workgroup_shape_constraints(&device);
                    let workgroup_shape = constraints.solve(max_subgroup_size).unwrap_or_else(|| {
                        panic!(
                            "Failed to find a valid qmatmul workgroup shape for constraints {constraints:?}"
                        )
                    });
                    if let Some(start) = start {
                        let elapsed = start.elapsed();
                        host_profile.workgroup += elapsed;
                        if let Some(category) = operation_category {
                            host_category_profile.entry(category).or_default().workgroup += elapsed;
                        }
                    }

                    let start = host_trace.then(Instant::now);
                    let direct_kernel_plan = qmatmul
                        .build_direct_kernels(graph, &workgroup_shape, &new_inputs)
                        .unwrap_or_else(|error| panic!("{error}"));
                    if let Some(start) = start {
                        let elapsed = start.elapsed();
                        host_profile.build_kernel += elapsed;
                        if let Some(category) = operation_category {
                            let profile = host_category_profile.entry(category).or_default();
                            profile.count += direct_kernel_plan.dispatch_count();
                            profile.build_kernel += elapsed;
                        }
                    }

                    for direct_kernel in direct_kernel_plan.into_kernels() {
                        let start = host_trace.then(Instant::now);
                        if let Some(dispatch) =
                            direct_kernel.prepare_dispatch(device.kernel_cache())
                        {
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
                                let name = qmatmul.name();
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
                    }

                    let start = host_trace.then(Instant::now);
                    Self::release_dead_intermediates(
                        graph,
                        &[&queued_operation],
                        &mut remaining_consumers,
                        &target_set,
                    );
                    if let Some(start) = start {
                        host_profile.release += start.elapsed();
                    }
                    continue;
                }

                let QueuedOperation::Generic(operation) = &queued_operation else {
                    unreachable!("qmatmul resolver arm returned above");
                };
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
                if let Some(dispatch) = direct_kernel.prepare_dispatch(device.kernel_cache()) {
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
                    &[&queued_operation],
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
            let dispatches_per_pass = dispatches_per_pass(total_kernels);
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
                        if let Some((query_set, _, _, _)) = &query_resources
                            && !profile_inside_pass_timestamps
                        {
                            if let CommandRecord::Dispatch(record) = &commands[command_index] {
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
                            }
                            dispatch_index += 1;
                            command_index += 1;
                            continue;
                        }

                        let mut pass =
                            command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("Resolver Direct Kernels"),
                                timestamp_writes: None,
                            });
                        let mut pass_dispatches = 0usize;
                        while command_index < commands.len() {
                            if pass_dispatches >= dispatches_per_pass {
                                break;
                            }
                            let CommandRecord::Dispatch(record) = &commands[command_index] else {
                                break;
                            };
                            if let Some((query_set, _, _, _)) = &query_resources {
                                pass.write_timestamp(query_set, (dispatch_index * 2) as u32);
                            }
                            record.dispatch.run(&mut pass);
                            if let Some((query_set, _, _, _)) = &query_resources {
                                pass.write_timestamp(query_set, (dispatch_index * 2 + 1) as u32);
                            }
                            dispatch_index += 1;
                            command_index += 1;
                            pass_dispatches += 1;
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
}

fn dispatches_per_pass(total_kernels: usize) -> usize {
    if let Ok(value) = std::env::var("FUSOR_RESOLVE_DISPATCHES_PER_PASS")
        && let Ok(parsed) = value.parse::<usize>()
        && parsed > 0
    {
        return parsed;
    }

    if total_kernels >= 1024 { 1 } else { usize::MAX }
}
