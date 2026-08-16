//! Bind groups are derived from the emitted module's storage globals,
//! sorted by binding, read-only from the absence of `StorageAccess::STORE`,
//! zipped positionally with the builder's buffer list.
//!
//! One `main`, one bind group, whole-buffer bindings. Binding 0 is always the
//! `Uniforms` **storage** buffer — a uniform-address-space block would break
//! this mechanism, which walks storage globals.


/// One derived binding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindingDesc {
    pub binding: u32,
    pub read_only: bool,
    /// The global's name in the emitted module, for diagnostics only.
    pub name: Option<String>,
}
/// Walk `module`'s storage globals in binding order.
///
/// This is the only source of binding order in the crate: nothing else may
/// enumerate buffers for a dispatch.
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
