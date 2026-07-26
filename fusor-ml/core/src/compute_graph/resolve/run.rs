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
        let device = graph.device();
        let host_trace = cfg!(target_arch = "wasm32") || device.config().trace_resolve_host;
        let host_total_start = host_trace.then(Instant::now);
        let mut host_profile = ResolveHostProfile::default();
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
            #[cfg(feature = "graphvis")]
            if let Some(dir) = &device.config().dump_stages {
                super::visualize::dump_stage(
                    dir,
                    &self.execution_graph,
                    super::visualize::Stage::Built,
                );
            }
        }

        // Pass 2: Apply Rewrite Rules
        {
            let start = host_trace.then(Instant::now);
            let node_count = self.execution_graph.node_count();
            self.optimize(graph);
            if let Some(start) = start {
                host_profile.optimize += start.elapsed();
            }
            if host_trace {
                let phases = self.optimize_phases;
                tracing::info!(
                    "resolve_optimize_phases node_count={node_count} recognition={:?} extraction={:?} physical={:?}",
                    phases.recognition,
                    phases.extraction,
                    phases.physical,
                );
            }
        }

        // Pass 3: Fusion-friendly topological sort.
        let sorted_nodes = {
            let start = host_trace.then(Instant::now);
            let sorted_nodes = merge_horizontal::fusion_toposort(&self.execution_graph);
            if let Some(start) = start {
                host_profile.toposort += start.elapsed();
            }
            sorted_nodes
        };

        // Pass 4: Execution
        // Extract operations in order.
        let mut target_set: FxHashSet<NodeIndex> = self.targets.iter().copied().collect();
        target_set.extend(self.shared_outputs.keys().copied());
        let mut queued_operations = Vec::with_capacity(sorted_nodes.len());

        {
            let start = host_trace.then(Instant::now);
            let mut merger = merge_horizontal::HorizontalMerger::new(&device);
            for idx in sorted_nodes {
                let node = &self.execution_graph[idx];
                // Handle Tensor caching explicitly here
                if let ExecutionVariant::Tensor(data) = &node.variant {
                    if let Some(recorder) = &self.recorder {
                        recorder
                            .borrow_mut()
                            .record_tensor_leaf(node.inner_idx, data);
                    }
                    graph.set_cached_result(node.inner_idx, data.clone());
                    continue;
                }

                let lowered = self.lower_node(idx, node);
                merger.push(node, lowered, &self.shared_outputs, &mut queued_operations);
            }
            merger.finish(&mut queued_operations);
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

        let trace = device.config().trace_decode || device.config().trace_resolve;
        let trace_names = device.config().trace_decode_names;
        let profile_gpu_kernels = device.config().trace_gpu_kernels;
        let collect_dispatch_metadata = trace || profile_gpu_kernels;
        let mut commands = Vec::<CommandRecord>::with_capacity(queued_operations.len());
        let mut dispatch_categories = FxHashMap::<String, usize>::default();
        let mut dispatch_names = FxHashMap::<String, usize>::default();
        // Every graph takes the three-phase queue runner: serial input
        // gathering, parallel kernel building, serial recording/encoding.
        // Compatible independent operations may merge across the queue; each
        // operation kind retains its own shape, dependency and binding gates.
        let mut ledger = super::alloc_reuse::BufferLedger::new(&device, Some(&remaining_consumers));
        if let Some(recorder) = &self.recorder {
            ledger.register_recorder_pins(recorder.borrow().pinned_ptrs());
        }
        Self::execute_queue(
            self.recorder.as_ref(),
            graph,
            &device,
            max_subgroup_size,
            queued_operations,
            &mut remaining_consumers,
            &target_set,
            &self.shared_outputs,
            &mut ledger,
            &mut commands,
            &mut host_profile,
            host_trace,
            &mut |name: &str| {
                collect_dispatch_metadata.then(|| {
                    let category = dispatch_category(name);
                    if trace {
                        *dispatch_categories.entry(category.clone()).or_default() += 1;
                        if trace_names {
                            *dispatch_names.entry(name.to_string()).or_default() += 1;
                        }
                    }
                    category
                })
            },
        );
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
            command_encoder = super::queue_executor::encode_command_records(
                &device,
                &commands,
                total_kernels,
                query_resources.as_ref().map(|(query_set, _, _, _)| {
                    super::queue_executor::TimestampPlan {
                        query_set,
                        inside_pass: profile_inside_pass_timestamps,
                    }
                }),
                command_encoder,
                |encoder, wait| {
                    submit_resolver_encoder(&device, encoder, wait, host_trace, &mut host_profile);
                },
            );

            if let Some(start) = encode_start {
                host_profile.encode += start.elapsed();
            }
        }

        let data = graph
            .get_result(self.targets[0])
            .expect("Target result not cached");
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
            if let Some((query_set, query_buffer, readback_buffer, raw_query_size)) =
                &query_resources
            {
                let profile_readback_start = host_trace.then(Instant::now);
                // A command buffer's counter samples are not guaranteed visible to
                // `resolve_query_set` until it has completed: Metal's writeback of the
                // final compute encoder's stage-boundary samples races a resolve encoded
                // behind it, leaving those slots zero. Resolving from a command buffer
                // submitted after the sampling work retired removes the race.
                device.poll_wait();
                let mut resolve_encoder =
                    device
                        .wgpu_device()
                        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("Resolver Timestamp Resolve"),
                        });
                resolve_encoder.resolve_query_set(query_set, 0..query_count, query_buffer, 0);
                resolve_encoder.copy_buffer_to_buffer(
                    query_buffer,
                    0,
                    readback_buffer,
                    0,
                    *raw_query_size,
                );
                device.wgpu_queue().submit(Some(resolve_encoder.finish()));
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
                        let profile = collect_gpu_kernel_profile(
                            &dispatch_metadata,
                            timestamps,
                            device.wgpu_queue().get_timestamp_period() as f64,
                            if profile_inside_pass_timestamps {
                                "inside_pass"
                            } else {
                                "pass_boundary"
                            },
                        );
                        device.record_kernel_profile(profile);
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

pub(super) fn resolve_cached_kernel_plan(
    kernel_cache: &fusor_tile_ir_runtime::KernelCache,
    cache_key: crate::mir::kernel_backend::KernelCacheKey,
    binding_buffers: Vec<Vec<std::sync::Arc<wgpu::Buffer>>>,
    build: impl FnOnce() -> Vec<crate::mir::kernel_backend::DirectKernel>,
) -> Vec<crate::mir::kernel_backend::DirectKernel> {
    let binding_slices = binding_buffers
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    kernel_cache
        .kernel_plan_cache()
        .try_get_or_insert_many(kernel_cache, cache_key, &binding_slices, || {
            Ok::<_, std::convert::Infallible>(build())
        })
        .expect("infallible kernel plan cache build failed")
}

pub(super) fn kernel_plan_binding_buffers(
    inputs: &[MirValue],
) -> Vec<Vec<std::sync::Arc<wgpu::Buffer>>> {
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

pub(super) fn dispatches_per_pass(device: &crate::Device, total_kernels: usize) -> usize {
    if let Some(value) = device.config().resolve_dispatches_per_pass {
        return value;
    }

    if total_kernels >= 1024 { 1 } else { usize::MAX }
}

pub(super) fn dispatches_per_submit(device: &crate::Device, total_kernels: usize) -> usize {
    if let Some(value) = device.config().resolve_dispatches_per_submit {
        return value;
    }

    // Chunked submits exist to bound in-flight memory on giant training
    // graphs; small inference graphs must stay a single submit.
    if device.backend() == wgpu::Backend::Metal && total_kernels >= 1024 {
        256
    } else {
        usize::MAX
    }
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
