use std::sync::Arc;

use phase_token_prototype as tile_ir;

use crate::{
    Device,
    mir::direct_kernel::{DirectKernel, DirectKernelBinding},
};

pub(crate) mod flash_attention;
pub(crate) mod mirostat;
pub(crate) mod rms_norm;
pub(crate) mod sampling_topk;

#[derive(Clone, Debug)]
pub(crate) struct CompiledKernelModule {
    module: Arc<wgpu::naga::Module>,
}

fn compiled_module(module: wgpu::naga::Module) -> CompiledKernelModule {
    CompiledKernelModule {
        module: Arc::new(module),
    }
}

fn compile_ir(ir: tile_ir::KernelIr) -> Option<CompiledKernelModule> {
    Some(compiled_module(ir.lower_to_naga().ok()?.module().clone()))
}

fn cached_kernel_module(
    device: &Device,
    cache_key: impl Into<String>,
    build_module: impl FnOnce() -> Option<CompiledKernelModule>,
) -> Option<CompiledKernelModule> {
    let cache_key = cache_key.into();
    if let Some(module) = device.naga_module_cache().write().get(&cache_key) {
        return Some(CompiledKernelModule {
            module: Arc::new(module.clone()),
        });
    }

    let compiled = build_module()?;
    device
        .naga_module_cache()
        .write()
        .get_or_insert(cache_key, || compiled.module.as_ref().clone());
    Some(compiled)
}

pub(crate) fn cached_kernel_ir(
    device: &Device,
    cache_key: impl Into<String>,
    build_ir: impl FnOnce() -> Option<tile_ir::KernelIr>,
) -> Option<CompiledKernelModule> {
    cached_kernel_module(device, cache_key, || compile_ir(build_ir()?))
}

pub(super) fn cached_backend_naga_module(
    device: &Device,
    cache_key: impl Into<String>,
    build_module: impl FnOnce() -> Option<wgpu::naga::Module>,
) -> Option<CompiledKernelModule> {
    cached_kernel_module(device, cache_key, || Some(compiled_module(build_module()?)))
}

pub(super) fn dynamic_kernel_from_backend_naga_module(
    device: &Device,
    name: impl Into<String>,
    cache_key: impl Into<String>,
    build_module: impl FnOnce() -> Option<wgpu::naga::Module>,
    bindings: Vec<DirectKernelBinding>,
    dispatch_size: [u32; 3],
) -> Option<DirectKernel> {
    let cache_key = cache_key.into();
    let module = cached_backend_naga_module(device, cache_key.clone(), build_module)?;
    Some(dynamic_kernel_from_module(
        name,
        cache_key,
        module,
        bindings,
        dispatch_size,
    ))
}

pub(crate) fn dynamic_kernel_from_ir(
    device: &Device,
    name: impl Into<String>,
    cache_key: impl Into<String>,
    build_ir: impl FnOnce() -> Option<tile_ir::KernelIr>,
    bindings: Vec<DirectKernelBinding>,
    dispatch_size: [u32; 3],
) -> Option<DirectKernel> {
    let cache_key = cache_key.into();
    let module = cached_kernel_ir(device, cache_key.clone(), build_ir)?;
    Some(dynamic_kernel_from_module(
        name,
        cache_key,
        module,
        bindings,
        dispatch_size,
    ))
}

pub(crate) fn dynamic_kernel_from_module(
    name: impl Into<String>,
    cache_key: impl Into<String>,
    module: CompiledKernelModule,
    bindings: Vec<DirectKernelBinding>,
    dispatch_size: [u32; 3],
) -> DirectKernel {
    DirectKernel::new_with_arc_module(name, cache_key, module.module, bindings, dispatch_size)
}

pub(crate) fn storage3_kernel_with_prepared_pipeline(
    name: impl Into<String>,
    cache_key: impl Into<String>,
    pipeline: wgpu::ComputePipeline,
    input: Arc<wgpu::Buffer>,
    weight: Arc<wgpu::Buffer>,
    output: Arc<wgpu::Buffer>,
    dispatch_size: [u32; 3],
) -> DirectKernel {
    DirectKernel::new_storage3_with_prepared_pipeline(
        name,
        cache_key,
        pipeline,
        input,
        weight,
        output,
        dispatch_size,
    )
}

pub(crate) fn prepare_storage3_pipeline(
    device: &Device,
    name: &str,
    module: &CompiledKernelModule,
) -> wgpu::ComputePipeline {
    let shader = device.create_naga_shader_module(module.module.as_ref().clone());
    let pipeline_layout = device.direct_storage3_pipeline_layout();
    device
        .wgpu_device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            cache: device.wgpu_cache(),
            compilation_options: wgpu::PipelineCompilationOptions {
                zero_initialize_workgroup_memory: false,
                ..Default::default()
            },
        })
}

pub(crate) fn storage3_pipeline_from_ir(
    device: &Device,
    name: &str,
    cache_key: impl Into<String>,
    build_ir: impl FnOnce() -> Option<tile_ir::KernelIr>,
) -> Option<wgpu::ComputePipeline> {
    let module = cached_kernel_ir(device, cache_key, build_ir)?;
    Some(prepare_storage3_pipeline(device, name, &module))
}

pub(crate) fn storage3_pipeline_from_cached_module(
    device: &Device,
    name: &str,
    cache_key: &str,
) -> Option<wgpu::ComputePipeline> {
    let module = device
        .naga_module_cache()
        .write()
        .get(cache_key)
        .map(|module| CompiledKernelModule {
            module: Arc::new(module.clone()),
        })?;
    Some(prepare_storage3_pipeline(device, name, &module))
}
