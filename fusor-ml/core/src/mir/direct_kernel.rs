use std::sync::Arc;

use wgpu::{CommandEncoder, ComputePass, PipelineCompilationOptions};

use crate::{
    Device, DirectDynamicBindGroupKey, DirectStorage3BindGroupKey,
    mir::kernel_backend::KernelCacheKey,
};

#[derive(Clone, Debug)]
pub(crate) enum DirectKernelBinding {
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
enum DirectKernelSource {
    Naga(Arc<wgpu::naga::Module>),
}

#[derive(Debug)]
pub(crate) struct DirectKernel {
    name: String,
    cache_key: KernelCacheKey,
    source: Option<DirectKernelSource>,
    prepared_pipeline: Option<wgpu::ComputePipeline>,
    bindings: DirectKernelBindings,
    dispatch_size: [u32; 3],
}

pub(crate) struct PreparedDirectDispatch {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    dispatch_size: [u32; 3],
}

impl DirectKernel {
    pub(super) fn new_with_arc_module(
        name: impl Into<String>,
        cache_key: KernelCacheKey,
        module: Arc<wgpu::naga::Module>,
        bindings: Vec<DirectKernelBinding>,
        dispatch_size: [u32; 3],
    ) -> Self {
        Self {
            name: name.into(),
            cache_key,
            source: Some(DirectKernelSource::Naga(module)),
            prepared_pipeline: None,
            bindings: DirectKernelBindings::Dynamic(bindings),
            dispatch_size,
        }
    }

    pub(super) fn new_storage3_with_prepared_pipeline(
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

    pub(crate) fn run(&self, device: &Device, command_encoder: &mut CommandEncoder) {
        let Some(dispatch) = self.prepare_dispatch(device) else {
            return;
        };
        let mut pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&self.name),
            timestamp_writes: None,
        });
        dispatch.run(&mut pass);
    }

    pub(crate) fn prepare_dispatch(&self, device: &Device) -> Option<PreparedDirectDispatch> {
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
            let bind_group_layout = device.direct_storage3_bind_group_layout();
            let bind_group_key = DirectStorage3BindGroupKey::new(input, weight, output);
            let bind_group = device
                .direct_storage3_bind_group_cache()
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
                    device
                        .wgpu_device()
                        .create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some(&self.name),
                            layout: &bind_group_layout,
                            entries: &bind_entries,
                        })
                })
                .clone();
            let pipeline_layout = device.direct_storage3_pipeline_layout();
            let pipeline = self.pipeline_with_layout(device, pipeline_layout);
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

        let bind_group_layout = device
            .bind_group_layout_cache()
            .write()
            .get_or_insert_ref(&layout_entries, || {
                device
                    .wgpu_device()
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
        let bind_group = device
            .direct_dynamic_bind_group_cache()
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
                device
                    .wgpu_device()
                    .create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&self.name),
                        layout: &bind_group_layout,
                        entries: &bind_entries,
                    })
            })
            .clone();

        let pipeline_layout = device
            .pipeline_layout_cache()
            .write()
            .get_or_insert_ref(&bind_group_layout, || {
                device
                    .wgpu_device()
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(&self.name),
                        bind_group_layouts: &[Some(&bind_group_layout)],
                        immediate_size: 0,
                    })
            })
            .clone();

        let pipeline = self.pipeline_with_layout(device, pipeline_layout);
        Some(PreparedDirectDispatch {
            pipeline,
            bind_group,
            dispatch_size: self.dispatch_size,
        })
    }

    fn pipeline_with_layout(
        &self,
        device: &Device,
        pipeline_layout: wgpu::PipelineLayout,
    ) -> wgpu::ComputePipeline {
        if let Some(pipeline) = &self.prepared_pipeline {
            pipeline.clone()
        } else {
            let source = self
                .source
                .as_ref()
                .expect("direct kernel without a prepared pipeline needs a shader source");
            let module = device
                .shader_module_cache()
                .write()
                .get_or_insert_ref(&self.cache_key, || match source {
                    DirectKernelSource::Naga(module_ir) => {
                        device.create_naga_shader_module(module_ir.as_ref().clone())
                    }
                })
                .clone();
            device
                .compute_pipeline_cache()
                .write()
                .get_or_insert((pipeline_layout.clone(), module.clone()), || {
                    device
                        .wgpu_device()
                        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                            label: Some(&self.name),
                            layout: Some(&pipeline_layout),
                            module: &module,
                            entry_point: Some("main"),
                            cache: device.wgpu_cache(),
                            compilation_options: PipelineCompilationOptions {
                                zero_initialize_workgroup_memory: false,
                                ..Default::default()
                            },
                        })
                })
                .clone()
        }
    }

    #[cfg(test)]
    pub(crate) fn bindings_for_test(&self) -> Vec<DirectKernelBinding> {
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

impl PreparedDirectDispatch {
    pub(crate) fn run<'a>(&'a self, pass: &mut ComputePass<'a>) {
        let [x, y, z] = self.dispatch_size;
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(x, y, z);
    }
}
