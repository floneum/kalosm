//! Bind groups derived from the emitted module's storage globals in binding
//! order, read-only when not `STORE`. Binding 0 is the `Uniforms` storage
//! buffer: a uniform-space block would escape this walk.

/// One derived binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindingDesc {
    pub binding: u32,
    pub read_only: bool,
}
/// Walk `module`'s storage globals in binding order: the crate's only source
/// of binding order.
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
            })
        })
        .collect();
    out.sort_by_key(|b| b.binding);
    out
}

/// The wgpu layout entries for a derived binding list.
pub(crate) fn layout_entries(bindings: &[BindingDesc]) -> Vec<wgpu::BindGroupLayoutEntry> {
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
