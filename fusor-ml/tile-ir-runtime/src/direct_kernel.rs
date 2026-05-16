use std::sync::Arc;

use wgpu::{CommandEncoder, ComputePass, PipelineCompilationOptions};

use crate::cache::{
    CachedKernel, DirectDynamicBindGroupKey, DirectStorage3BindGroupKey, KernelCache,
    KernelCacheKey,
};

#[derive(Clone, Debug)]
pub enum DirectKernelBinding {
    Storage {
        binding: u32,
        buffer: Arc<wgpu::Buffer>,
        read_only: bool,
    },
}

#[derive(Debug)]
enum DirectKernelBindings {
    Dynamic(Vec<DirectKernelBinding>),
    Storage3 {
        input: Arc<wgpu::Buffer>,
        weight: Arc<wgpu::Buffer>,
        output: Arc<wgpu::Buffer>,
    },
}

#[derive(Debug)]
pub struct DirectKernel {
    name: String,
    cache_key: KernelCacheKey,
    source: Option<Arc<wgpu::naga::Module>>,
    prepared_pipeline: Option<wgpu::ComputePipeline>,
    bindings: DirectKernelBindings,
    dispatch_size: [u32; 3],
}

pub struct PreparedDirectDispatch {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    dispatch_size: [u32; 3],
}

impl DirectKernel {
    pub fn from_naga(
        name: impl Into<String>,
        cache_key: KernelCacheKey,
        naga: Arc<wgpu::naga::Module>,
        bindings: Vec<DirectKernelBinding>,
        dispatch_size: [u32; 3],
    ) -> Self {
        Self {
            name: name.into(),
            cache_key,
            source: Some(naga),
            prepared_pipeline: None,
            bindings: DirectKernelBindings::Dynamic(bindings),
            dispatch_size,
        }
    }

    pub fn from_prepared_three_buffer_pipeline(
        name: impl Into<String>,
        cache_key: KernelCacheKey,
        pipeline: wgpu::ComputePipeline,
        input: Arc<wgpu::Buffer>,
        weight: Arc<wgpu::Buffer>,
        output: Arc<wgpu::Buffer>,
        dispatch_size: [u32; 3],
    ) -> Self {
        Self {
            name: name.into(),
            cache_key,
            source: None,
            prepared_pipeline: Some(pipeline),
            bindings: DirectKernelBindings::Storage3 {
                input,
                weight,
                output,
            },
            dispatch_size,
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

        if let DirectKernelBindings::Storage3 {
            input,
            weight,
            output,
        } = &self.bindings
        {
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
            let pipeline_layout = cache.direct_three_buffer_pipeline_layout();
            let pipeline = self.pipeline_with_layout(cache, &pipeline_layout);
            return Some(PreparedDirectDispatch {
                pipeline,
                bind_group,
                dispatch_size: self.dispatch_size,
            });
        }

        let DirectKernelBindings::Dynamic(bindings) = &self.bindings else {
            unreachable!("storage3 direct kernels are handled above");
        };

        let layout_entries = bindings
            .iter()
            .map(|binding| match binding {
                DirectKernelBinding::Storage {
                    binding, read_only, ..
                } => wgpu::BindGroupLayoutEntry {
                    binding: *binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: *read_only,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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

        let bind_group_key = DirectDynamicBindGroupKey::new(bindings.iter().map(|binding| {
            let DirectKernelBinding::Storage {
                binding,
                buffer,
                read_only,
            } = binding;
            (*binding, *read_only, buffer.clone())
        }));
        let bind_group = cache
            .direct_dynamic_bind_group_cache
            .write()
            .get_or_insert(bind_group_key, || {
                let bind_entries = bindings
                    .iter()
                    .map(|binding| match binding {
                        DirectKernelBinding::Storage {
                            binding, buffer, ..
                        } => wgpu::BindGroupEntry {
                            binding: *binding,
                            resource: buffer.as_entire_binding(),
                        },
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

        let pipeline = self.pipeline_with_layout(cache, &pipeline_layout);
        Some(PreparedDirectDispatch {
            pipeline,
            bind_group,
            dispatch_size: self.dispatch_size,
        })
    }

    fn pipeline_with_layout(
        &self,
        cache: &KernelCache,
        pipeline_layout: &wgpu::PipelineLayout,
    ) -> wgpu::ComputePipeline {
        if let Some(pipeline) = &self.prepared_pipeline {
            return pipeline.clone();
        }
        let naga = self
            .source
            .as_ref()
            .expect("direct kernel without a prepared pipeline needs a naga source");
        let cached = cache.get_or_insert_kernel(self.cache_key, || naga.clone());
        let shader = cache.shader_for(&cached);
        cached
            .pipeline
            .get_or_init(|| {
                cache
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(&self.name),
                        layout: Some(pipeline_layout),
                        module: shader,
                        entry_point: Some("main"),
                        cache: cache.wgpu_cache.as_ref(),
                        compilation_options: PipelineCompilationOptions {
                            zero_initialize_workgroup_memory: false,
                            ..Default::default()
                        },
                    })
            })
            .clone()
    }

    pub fn bindings_for_test(&self) -> Vec<DirectKernelBinding> {
        match &self.bindings {
            DirectKernelBindings::Dynamic(bindings) => bindings.clone(),
            DirectKernelBindings::Storage3 {
                input,
                weight,
                output,
            } => vec![
                DirectKernelBinding::Storage {
                    binding: 0,
                    buffer: input.clone(),
                    read_only: true,
                },
                DirectKernelBinding::Storage {
                    binding: 1,
                    buffer: weight.clone(),
                    read_only: true,
                },
                DirectKernelBinding::Storage {
                    binding: 2,
                    buffer: output.clone(),
                    read_only: false,
                },
            ],
        }
    }
}

impl KernelCache {
    pub(crate) fn shader_for<'a>(&self, cached: &'a Arc<CachedKernel>) -> &'a wgpu::ShaderModule {
        cached
            .shader
            .get_or_init(|| self.create_naga_shader_module(cached.naga.as_ref().clone()))
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
