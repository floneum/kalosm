use std::sync::Arc;

use wgpu::{CommandEncoder, ComputePass, PipelineCompilationOptions};

use crate::cache::{
    CachedKernel, DirectDynamicBindGroupKey, DirectStorage3BindGroupKey, KernelCache,
};

#[derive(Clone, Debug)]
pub struct DirectKernelBinding {
    pub binding: u32,
    pub buffer: Arc<wgpu::Buffer>,
    pub read_only: bool,
}

#[derive(Debug)]
enum DirectKernelKind {
    /// Generic path: bindings are derived from the kernel's lowered storage
    /// declarations; the pipeline is built lazily from the cached shader on
    /// first dispatch.
    Dynamic {
        cached: Arc<CachedKernel>,
        bindings: Vec<DirectKernelBinding>,
    },
    /// Hot-path specialization for the singleton 3-buffer (input, weight,
    /// output) layout. The pipeline is pre-built (typically once per
    /// quantized matrix) so dispatch skips the kernel-cache LRU entirely.
    Storage3 {
        pipeline: wgpu::ComputePipeline,
        input: Arc<wgpu::Buffer>,
        weight: Arc<wgpu::Buffer>,
        output: Arc<wgpu::Buffer>,
    },
}

#[derive(Debug)]
pub struct DirectKernel {
    name: String,
    dispatch_size: [u32; 3],
    kind: DirectKernelKind,
}

pub struct PreparedDirectDispatch {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    _buffers: Vec<Arc<wgpu::Buffer>>,
    dispatch_size: [u32; 3],
}

impl DirectKernel {
    pub fn from_cached(
        name: impl Into<String>,
        cached: Arc<CachedKernel>,
        bindings: Vec<DirectKernelBinding>,
        dispatch_size: [u32; 3],
    ) -> Self {
        Self {
            name: name.into(),
            dispatch_size,
            kind: DirectKernelKind::Dynamic { cached, bindings },
        }
    }

    pub fn from_prepared_three_buffer_pipeline(
        name: impl Into<String>,
        pipeline: wgpu::ComputePipeline,
        input: Arc<wgpu::Buffer>,
        weight: Arc<wgpu::Buffer>,
        output: Arc<wgpu::Buffer>,
        dispatch_size: [u32; 3],
    ) -> Self {
        Self {
            name: name.into(),
            dispatch_size,
            kind: DirectKernelKind::Storage3 {
                pipeline,
                input,
                weight,
                output,
            },
        }
    }

    pub fn run(&self, cache: &KernelCache, command_encoder: &mut CommandEncoder) {
        let Some(dispatch) = self.prepare_dispatch(cache) else {
            return;
        };
        let mut pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&self.name),
            timestamp_writes: None,
        });
        dispatch.run(&mut pass);
    }

    pub fn prepare_dispatch(&self, cache: &KernelCache) -> Option<PreparedDirectDispatch> {
        let [x, y, z] = self.dispatch_size;
        if x * y * z == 0 {
            return None;
        }

        match &self.kind {
            DirectKernelKind::Storage3 {
                pipeline,
                input,
                weight,
                output,
            } => {
                let bind_group_layout = cache.direct_three_buffer_bind_group_layout();
                let bind_group_key = DirectStorage3BindGroupKey::new(input, weight, output);
                let bind_group = cache
                    .direct_three_buffer_bind_group_cache
                    .write()
                    .get_or_insert(bind_group_key, || {
                        let bind_entries = [
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: input.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: weight.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: output.as_entire_binding(),
                            },
                        ];
                        cache.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some(&self.name),
                            layout: &bind_group_layout,
                            entries: &bind_entries,
                        })
                    })
                    .clone();
                Some(PreparedDirectDispatch {
                    pipeline: pipeline.clone(),
                    bind_group,
                    _buffers: vec![input.clone(), weight.clone(), output.clone()],
                    dispatch_size: self.dispatch_size,
                })
            }
            DirectKernelKind::Dynamic { cached, bindings } => {
                let layout_entries = bindings
                    .iter()
                    .map(|binding| wgpu::BindGroupLayoutEntry {
                        binding: binding.binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage {
                                read_only: binding.read_only,
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect::<Vec<_>>();

                let bind_group_layout = cache
                    .bind_group_layout_cache
                    .write()
                    .get_or_insert_ref(&layout_entries, || {
                        cache
                            .device
                            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                                label: Some(&self.name),
                                entries: &layout_entries,
                            })
                    })
                    .clone();

                let bind_group_key = DirectDynamicBindGroupKey::new(
                    bindings
                        .iter()
                        .map(|b| (b.binding, b.read_only, b.buffer.clone())),
                );
                let bind_group = cache
                    .direct_dynamic_bind_group_cache
                    .write()
                    .get_or_insert(bind_group_key, || {
                        let bind_entries = bindings
                            .iter()
                            .map(|b| wgpu::BindGroupEntry {
                                binding: b.binding,
                                resource: b.buffer.as_entire_binding(),
                            })
                            .collect::<Vec<_>>();
                        cache.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some(&self.name),
                            layout: &bind_group_layout,
                            entries: &bind_entries,
                        })
                    })
                    .clone();

                let pipeline_layout = cache
                    .pipeline_layout_cache
                    .write()
                    .get_or_insert_ref(&bind_group_layout, || {
                        cache
                            .device
                            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                                label: Some(&self.name),
                                bind_group_layouts: &[Some(&bind_group_layout)],
                                immediate_size: 0,
                            })
                    })
                    .clone();

                let shader = cache.shader_for(cached);
                let pipeline = cached
                    .pipeline
                    .get_or_init(|| {
                        cache
                            .device
                            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                                label: Some(&self.name),
                                layout: Some(&pipeline_layout),
                                module: shader,
                                entry_point: Some("main"),
                                cache: cache.wgpu_cache.as_ref(),
                                compilation_options: PipelineCompilationOptions {
                                    zero_initialize_workgroup_memory: false,
                                    ..Default::default()
                                },
                            })
                    })
                    .clone();

                Some(PreparedDirectDispatch {
                    pipeline,
                    bind_group,
                    _buffers: bindings
                        .iter()
                        .map(|binding| binding.buffer.clone())
                        .collect(),
                    dispatch_size: self.dispatch_size,
                })
            }
        }
    }

    pub fn bindings_for_test(&self) -> Vec<DirectKernelBinding> {
        match &self.kind {
            DirectKernelKind::Dynamic { bindings, .. } => bindings.clone(),
            DirectKernelKind::Storage3 {
                input,
                weight,
                output,
                ..
            } => vec![
                DirectKernelBinding {
                    binding: 0,
                    buffer: input.clone(),
                    read_only: true,
                },
                DirectKernelBinding {
                    binding: 1,
                    buffer: weight.clone(),
                    read_only: true,
                },
                DirectKernelBinding {
                    binding: 2,
                    buffer: output.clone(),
                    read_only: false,
                },
            ],
        }
    }
}

impl PreparedDirectDispatch {
    pub fn run<'a>(&'a self, pass: &mut ComputePass<'a>) {
        let [x, y, z] = self.dispatch_size;
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(x, y, z);
    }
}
