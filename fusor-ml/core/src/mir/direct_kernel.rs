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

pub(crate) fn direct_storage_array_size(_allocation_len: u32) -> wgpu::naga::ArraySize {
    // Direct kernels bind whole storage buffers. Keep the shader type runtime-sized so Naga
    // does not try to lay out very large model buffers as fixed-size shader types.
    wgpu::naga::ArraySize::Dynamic
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

#[cfg(test)]
mod tests {
    use wgpu::naga::{
        AddressSpace, GlobalVariable, ResourceBinding, Scalar, ScalarKind, Span, StorageAccess,
        Type, TypeInner, valid,
    };

    use super::*;

    #[test]
    fn direct_storage_arrays_validate_above_naga_fixed_type_limit() {
        assert!(matches!(
            direct_storage_array_size(valid::MAX_TYPE_SIZE / 4 + 1),
            wgpu::naga::ArraySize::Dynamic
        ));

        let mut module = wgpu::naga::Module::default();
        let f32_ty = module.types.insert(
            Type {
                name: Some("f32".into()),
                inner: TypeInner::Scalar(Scalar {
                    kind: ScalarKind::Float,
                    width: 4,
                }),
            },
            Span::default(),
        );
        let buffer_ty = module.types.insert(
            Type {
                name: Some("HugeDirectBuffer".into()),
                inner: TypeInner::Array {
                    base: f32_ty,
                    size: direct_storage_array_size(valid::MAX_TYPE_SIZE / 4 + 1),
                    stride: 4,
                },
            },
            Span::default(),
        );
        module.global_variables.append(
            GlobalVariable {
                name: Some("huge_direct_buffer".into()),
                space: AddressSpace::Storage {
                    access: StorageAccess::LOAD,
                },
                binding: Some(ResourceBinding {
                    group: 0,
                    binding: 0,
                }),
                ty: buffer_ty,
                init: None,
            },
            Span::default(),
        );

        valid::Validator::new(valid::ValidationFlags::all(), valid::Capabilities::empty())
            .validate(&module)
            .unwrap();
    }
}
