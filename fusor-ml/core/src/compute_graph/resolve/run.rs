use super::*;

impl Resolver {
    pub(crate) fn run(
        &mut self,
        graph: &mut ComputeGraphInner,
        _removed: &mut Vec<ComputeGraphNode>,
    ) -> ResolverResult {
        let (result, ()) = self.run_with_tail(graph, _removed, |_, _| ());
        result
    }

    pub(crate) fn run_with_tail<T>(
        &mut self,
        graph: &mut ComputeGraphInner,
        _removed: &mut Vec<ComputeGraphNode>,
        tail: impl FnOnce(&TensorData, &mut wgpu::CommandEncoder) -> T,
    ) -> (ResolverResult, T) {
        let host_trace =
            cfg!(target_arch = "wasm32") || std::env::var_os("FUSOR_TRACE_RESOLVE_HOST").is_some();
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
                tracing::info!(
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
        let first_target = self.targets[0];
        let target_set: FxHashSet<NodeIndex> = self.targets.iter().copied().collect();
        let mut first_target_data = None;
        let mut queued_operations = Vec::with_capacity(sorted_nodes.len());

        {
            let start = host_trace.then(Instant::now);
            for idx in sorted_nodes {
                let node = &self.execution_graph[idx];
                // Handle Tensor caching explicitly here
                if let ExecutionVariant::Tensor(data) = &node.variant {
                    let data = data.clone();
                    if node.inner_idx == first_target {
                        first_target_data = Some(data.clone());
                    }
                    graph.set_cached_result(node.inner_idx, data);
                    continue;
                }

                if let Some(op) = self.lower_node(idx, node) {
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
        let plan_cache_enabled = device.kernel_cache().direct_plan_cache().enabled();
        for (node, queued_operation) in queued_operations {
            let operation_category = host_category_trace
                .then(|| {
                    graph
                        .nodes
                        .nodes
                        .node_weight(node)
                        .map(|node| node_category_inner(&node.variant))
                })
                .flatten();
            // A view that composes with its input's buffer layout isn't a
            // kernel. Resolve it immediately as a zero-cost buffer view;
            // anything else (fill regions, non-composable reshapes) falls
            // through to the gather kernel below.
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
                let start = host_trace.then(Instant::now);
                if node == first_target {
                    first_target_data = Some(result.clone());
                }
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
                    let ComputeGraphNodeVariant::Assign(slice_assign) = &node_data.variant else {
                        return None;
                    };
                    Self::try_prepare_in_place_slice_assign_copy(graph, slice_assign)
                });
                if let Some((output, copies)) = slice_copy {
                    if node == first_target {
                        first_target_data = Some(output.clone());
                    }
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
                    if node == first_target {
                        first_target_data = Some(resolved.clone());
                    }
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
                    let workgroup_shape = constraints
                        .solve(max_subgroup_size, &device.limits())
                        .unwrap_or_else(|| {
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
                    let build_kernels = || {
                        qmatmul
                            .build_direct_kernels(graph, &workgroup_shape, &new_inputs)
                            .unwrap_or_else(|error| panic!("{error}"))
                            .into_kernels()
                    };
                    let kernels = if plan_cache_enabled {
                        let kernel_key =
                            structural_kernel_key(qmatmul.as_ref(), &new_inputs, &workgroup_shape);
                        resolve_cached_direct_plan(
                            device.kernel_cache().direct_plan_cache(),
                            kernel_key,
                            direct_plan_binding_buffers(&new_inputs),
                            build_kernels,
                        )
                    } else {
                        build_kernels()
                    };
                    if let Some(start) = start {
                        let elapsed = start.elapsed();
                        host_profile.build_kernel += elapsed;
                        if let Some(category) = operation_category {
                            let profile = host_category_profile.entry(category).or_default();
                            profile.count += kernels.len();
                            profile.build_kernel += elapsed;
                        }
                    }

                    for direct_kernel in kernels {
                        let start = host_trace.then(Instant::now);
                        if let Some(dispatch) =
                            direct_kernel.prepare_dispatch(device.kernel_cache())
                        {
                            let name = direct_kernel.name().to_string();
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
                            let category = collect_dispatch_metadata.then(|| {
                                let category = dispatch_category(&name);
                                if trace {
                                    *dispatch_categories.entry(category.clone()).or_default() += 1;
                                    if trace_names {
                                        *dispatch_names.entry(name.clone()).or_default() += 1;
                                    }
                                }
                                category
                            });
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
                if node == first_target {
                    first_target_data = Some(resolved.clone());
                }
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
                let workgroup_shape = constraints
                    .solve(max_subgroup_size, &device.limits())
                    .unwrap_or_else(|| {
                        panic!(
                            "Failed to find a valid workgroup shape for constraints {constraints:?}"
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
                let build_kernels = || {
                    vec![
                        operation
                            .build_direct_kernel(graph, &workgroup_shape, &new_inputs)
                            .unwrap_or_else(|| {
                                panic!(
                                    "operation did not provide a direct kernel: {}",
                                    operation.name()
                                )
                            }),
                    ]
                };
                let kernels = if plan_cache_enabled {
                    let kernel_key =
                        structural_kernel_key(operation.as_ref(), &new_inputs, &workgroup_shape);
                    resolve_cached_direct_plan(
                        device.kernel_cache().direct_plan_cache(),
                        kernel_key,
                        direct_plan_binding_buffers(&new_inputs),
                        build_kernels,
                    )
                } else {
                    build_kernels()
                };
                if let Some(start) = start {
                    let elapsed = start.elapsed();
                    host_profile.build_kernel += elapsed;
                    if let Some(category) = operation_category {
                        let profile = host_category_profile.entry(category).or_default();
                        profile.count += kernels.len();
                        profile.build_kernel += elapsed;
                    }
                }
                for direct_kernel in kernels {
                    let start = host_trace.then(Instant::now);
                    if let Some(dispatch) = direct_kernel.prepare_dispatch(device.kernel_cache()) {
                        let name = direct_kernel.name().to_string();
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
                        let category = collect_dispatch_metadata.then(|| {
                            let category = dispatch_category(&name);
                            if trace {
                                *dispatch_categories.entry(category.clone()).or_default() += 1;
                                if trace_names {
                                    *dispatch_names.entry(name.clone()).or_default() += 1;
                                }
                            }
                            category
                        });
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
            };
        }

        let total_kernels = commands
            .iter()
            .filter(|command| matches!(command, CommandRecord::Dispatch(_)))
            .count();
        if trace {
            let mut categories = dispatch_categories.into_iter().collect::<Vec<_>>();
            categories.sort_by(|a, b| a.0.cmp(&b.0));
            tracing::info!("resolve_dispatch_categories {categories:?}");
            if trace_names {
                let mut names = dispatch_names.into_iter().collect::<Vec<_>>();
                names.sort_by(|a, b| a.0.cmp(&b.0));
                tracing::info!("resolve_dispatch_names {names:?}");
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        let dispatch_metadata = commands
            .iter()
            .filter_map(|command| match command {
                CommandRecord::Dispatch(record) => Some(DispatchMetadata {
                    name: profile_gpu_kernels.then(|| record.name.clone()),
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
                    tracing::warn!(
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
            let dispatches_per_submit = dispatches_per_submit(total_kernels, device.backend());
            let wait_after_chunk_submit = device.backend() == wgpu::Backend::Metal;
            let mut dispatches_in_submit = 0usize;
            let mut encoder_has_commands = false;
            while command_index < commands.len() {
                if encoder_has_commands && dispatches_in_submit >= dispatches_per_submit {
                    let next_encoder = resolver_command_encoder(&device);
                    let ready_encoder = std::mem::replace(&mut command_encoder, next_encoder);
                    submit_resolver_encoder(
                        &device,
                        ready_encoder,
                        wait_after_chunk_submit,
                        host_trace,
                        &mut host_profile,
                    );
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
                        encoder_has_commands = true;
                        command_index += 1;
                    }
                    CommandRecord::Dispatch(_) => {
                        if let Some((query_set, _, _, _)) = &query_resources
                            && !profile_inside_pass_timestamps
                        {
                            if let CommandRecord::Dispatch(record) = &commands[command_index] {
                                let mut pass = command_encoder.begin_compute_pass(
                                    &wgpu::ComputePassDescriptor {
                                        label: Some(record.name.as_str()),
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
                            dispatches_in_submit += 1;
                            encoder_has_commands = true;
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
                            if dispatches_in_submit >= dispatches_per_submit {
                                break;
                            }
                            let CommandRecord::Dispatch(record) = &commands[command_index] else {
                                break;
                            };
                            if let Some((query_set, _, _, _)) = &query_resources {
                                pass.write_timestamp(query_set, (dispatch_index * 2) as u32);
                            }
                            pass.push_debug_group(&record.name);
                            record.dispatch.run(&mut pass);
                            pass.pop_debug_group();
                            if let Some((query_set, _, _, _)) = &query_resources {
                                pass.write_timestamp(query_set, (dispatch_index * 2 + 1) as u32);
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

        let produced_first_target = first_target_data.is_some();
        let data = match first_target_data.or_else(|| graph.get_result(first_target)) {
            Some(data) => data,
            None => {
                let target_state = graph.nodes.nodes.node_weight(first_target).map(|node| {
                    (
                        node_category_inner(&node.variant),
                        node.reference_count,
                        node.live_descendant_count,
                        node.cached.is_some(),
                    )
                });
                panic!(
                    "Target result not cached target={first_target:?} produced_first_target={produced_first_target} target_state={target_state:?} execution_nodes={} queued_operations={queued_operation_count} kernels={total_kernels}",
                    self.execution_graph.node_count(),
                );
            }
        };
        let tail_result = tail(&data, &mut command_encoder);

        // Submit any remaining commands.
        submit_resolver_encoder(
            &device,
            command_encoder,
            false,
            host_trace,
            &mut host_profile,
        );
        #[cfg(not(target_arch = "wasm32"))]
        {
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
                        tracing::warn!("resolve_gpu_kernel_profile map_failed {error:?}");
                    }
                    Err(error) => {
                        tracing::warn!("resolve_gpu_kernel_profile map_channel_failed {error:?}");
                    }
                }
                if let Some(start) = profile_readback_start {
                    host_profile.profile_readback += start.elapsed();
                }
            }
        }
        device.reset_initialized_buffers();

        if let Some(start) = host_total_start {
            host_profile.print(start.elapsed(), queued_operation_count, total_kernels);
            if host_category_trace {
                print_host_category_profile(host_category_profile);
            }
        }
        (
            ResolverResult {
                data,
                total_kernels,
            },
            tail_result,
        )
    }
}

fn resolve_cached_direct_plan(
    plan_cache: &fusor_tile_ir_runtime::DirectPlanCache,
    cache_key: crate::mir::kernel_backend::KernelCacheKey,
    binding_buffers: Vec<Vec<std::sync::Arc<wgpu::Buffer>>>,
    build: impl FnOnce() -> Vec<crate::mir::kernel_backend::DirectKernel>,
) -> Vec<crate::mir::kernel_backend::DirectKernel> {
    let binding_slices = binding_buffers
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    plan_cache
        .try_get_or_insert_many(cache_key, &binding_slices, || {
            Ok::<_, std::convert::Infallible>(build())
        })
        .expect("infallible direct plan cache build failed")
}

fn direct_plan_binding_buffers(inputs: &[MirValue]) -> Vec<Vec<std::sync::Arc<wgpu::Buffer>>> {
    let buffers = inputs
        .iter()
        .filter_map(|input| match input {
            MirValue::Tensor(tensor) => Some(tensor.buffer().clone()),
            MirValue::QMatrix(matrix) => Some(matrix.buffer().clone()),
            MirValue::Integer(_) | MirValue::Float(_) => None,
        })
        .collect();
    vec![buffers]
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

fn dispatches_per_submit(total_kernels: usize, backend: wgpu::Backend) -> usize {
    if let Ok(value) = std::env::var("FUSOR_RESOLVE_DISPATCHES_PER_SUBMIT")
        && let Ok(parsed) = value.parse::<usize>()
        && parsed > 0
    {
        return parsed;
    }

    if backend == wgpu::Backend::Metal && total_kernels > 8 {
        8
    } else {
        usize::MAX
    }
}

fn resolver_command_encoder(device: &crate::Device) -> wgpu::CommandEncoder {
    device
        .wgpu_device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Resolver Encoder"),
        })
}

fn submit_resolver_encoder(
    device: &crate::Device,
    command_encoder: wgpu::CommandEncoder,
    wait: bool,
    host_trace: bool,
    host_profile: &mut ResolveHostProfile,
) {
    let submit_start = host_trace.then(Instant::now);
    device.wgpu_queue().submit(Some(command_encoder.finish()));
    if wait {
        device.poll_wait();
    }
    if let Some(start) = submit_start {
        host_profile.submit += start.elapsed();
    }
}
