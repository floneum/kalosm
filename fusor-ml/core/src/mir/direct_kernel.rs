use std::sync::Arc;

use wgpu::{CommandEncoder, PipelineCompilationOptions};

use crate::Device;

#[derive(Clone, Debug)]
pub(crate) enum DirectKernelBinding {
    Storage {
        binding: u32,
        buffer: Arc<wgpu::Buffer>,
        read_only: bool,
    },
}

#[derive(Debug)]
pub(crate) struct DirectKernel {
    name: String,
    cache_key: String,
    module: wgpu::naga::Module,
    bindings: Vec<DirectKernelBinding>,
    dispatch_size: [u32; 3],
}

impl DirectKernel {
    pub(crate) fn new_with_cache_key(
        name: impl Into<String>,
        cache_key: impl Into<String>,
        module: wgpu::naga::Module,
        bindings: Vec<DirectKernelBinding>,
        dispatch_size: [u32; 3],
    ) -> Self {
        Self {
            name: name.into(),
            cache_key: cache_key.into(),
            module,
            bindings,
            dispatch_size,
        }
    }

    pub(crate) fn run(&self, device: &Device, command_encoder: &mut CommandEncoder) {
        let [x, y, z] = self.dispatch_size;
        if x * y * z == 0 {
            return;
        }

        let layout_entries = self
            .bindings
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

        let bind_entries = self
            .bindings
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
        let bind_group = device
            .wgpu_device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&self.name),
                layout: &bind_group_layout,
                entries: &bind_entries,
            });

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

        let module = device
            .shader_module_cache()
            .write()
            .get_or_insert_ref(&self.cache_key, || {
                device.create_naga_shader_module(self.module.clone())
            })
            .clone();
        let pipeline = device
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
            .clone();

        let mut pass = command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&self.name),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(x, y, z);
    }

    #[cfg(test)]
    pub(crate) fn bindings_for_test(&self) -> &[DirectKernelBinding] {
        &self.bindings
    }
}
