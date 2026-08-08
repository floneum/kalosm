//! Bind groups are derived from the emitted module's storage globals, sorted by
//! binding, read-only from the absence of `StorageAccess::STORE`, zipped
//! positionally with the builder's buffer list.
//!
//! One `main`, one bind group, whole-buffer bindings. Binding 0 is always the
//! `Uniforms` storage buffer; this walks storage globals, so a
//! uniform-address-space block would not be found.

/// One derived binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindingDesc {
    pub binding: u32,
    pub read_only: bool,
    /// The global's name in the emitted module, for diagnostics only.
    pub name: Option<String>,
}

/// Walk `module`'s storage globals in binding order. The only source of
/// binding order in the crate.
pub fn bindings_from_module(module: &naga::Module) -> Vec<BindingDesc> {
    let mut out: Vec<BindingDesc> = module
        .global_variables
        .iter()
        .filter_map(|(_, global)| {
            let naga::AddressSpace::Storage { access } = global.space else {
                return None;
            };
            let binding = global.binding.as_ref()?;
            Some(BindingDesc {
                binding: binding.binding,
                read_only: !access.contains(naga::StorageAccess::STORE),
                name: global.name.clone(),
            })
        })
        .collect();
    out.sort_by_key(|b| b.binding);
    out
}

/// The wgpu layout entries for a derived binding list.
pub fn layout_entries(bindings: &[BindingDesc]) -> Vec<wgpu::BindGroupLayoutEntry> {
    bindings
        .iter()
        .map(|slot| wgpu::BindGroupLayoutEntry {
            binding: slot.binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage {
                    read_only: slot.read_only,
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor2_ir::target::EmitError;
    use naga::{
        AddressSpace, ArraySize, GlobalVariable, Module, ResourceBinding, Scalar, Span,
        StorageAccess, Type, TypeInner,
    };

    fn module_with(bindings: &[(u32, bool)]) -> Module {
        let mut module = Module::default();
        let base = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::F32),
            },
            Span::default(),
        );
        let array = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Array {
                    base,
                    size: ArraySize::Dynamic,
                    stride: 4,
                },
            },
            Span::default(),
        );
        for &(binding, writable) in bindings {
            let access = if writable {
                StorageAccess::LOAD | StorageAccess::STORE
            } else {
                StorageAccess::LOAD
            };
            module.global_variables.append(
                GlobalVariable {
                    name: Some(format!("b{binding}")),
                    space: AddressSpace::Storage { access },
                    binding: Some(ResourceBinding { group: 0, binding }),
                    ty: array,
                    init: None,
                    memory_decorations: naga::MemoryDecorations::empty(),
                },
                Span::default(),
            );
        }
        module
    }

    /// Bindings are derived and sorted, read-only is inferred, and a bad
    /// uniform slot is rejected.
    #[test]
    fn bindings_sorted_and_read_only_derived() {
        use crate::emit::testkit::check_uniform_binding;
        // Declared out of order: 2 (writable), 0, 1.
        let module = module_with(&[(2, true), (0, false), (1, false)]);
        let slots = bindings_from_module(&module);
        assert_eq!(
            slots
                .iter()
                .map(|s| (s.binding, s.read_only))
                .collect::<Vec<_>>(),
            vec![(0, true), (1, true), (2, false)]
        );

        let entries = layout_entries(&slots);
        assert_eq!(entries.len(), 3);
        assert!(matches!(
            entries[0].ty,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            }
        ));
        assert!(entries.iter().all(|e| e.count.is_none()));
        assert!(
            entries
                .iter()
                .all(|e| e.visibility == wgpu::ShaderStages::COMPUTE)
        );

        // A writable binding 0 is refused: binding 0 is the Uniforms block.
        let bad = bindings_from_module(&module_with(&[(0, true), (1, false)]));
        assert!(matches!(
            check_uniform_binding(&bad),
            Err(EmitError::Validation(_))
        ));

        // A module with no binding 0 at all is refused too.
        let missing = bindings_from_module(&module_with(&[(1, false)]));
        assert!(matches!(
            check_uniform_binding(&missing),
            Err(EmitError::Validation(_))
        ));
    }

    /// Needs live `wgpu::Buffer`s, so it runs only when an adapter exists.
    #[test]
    fn zip_length_mismatch_is_a_validation_error() {
        use crate::emit::testkit::zip_buffers;
        let Ok(gpu) = crate::device::gpu_blocking(&crate::device::DeviceOptions::default()) else {
            eprintln!("no wgpu adapter; skipping");
            return;
        };
        let slots = bindings_from_module(&module_with(&[(0, false), (1, false), (2, true)]));
        let make = |i: u32| {
            gpu.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("t"),
                size: 16 + i as u64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let two = [make(0), make(1)];
        assert!(matches!(
            zip_buffers(&slots, &two),
            Err(EmitError::Validation(_))
        ));
        let three = [make(0), make(1), make(2)];
        let entries = zip_buffers(&slots, &three).expect("three buffers for three slots");
        assert_eq!(
            entries.iter().map(|e| e.binding).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }
}
