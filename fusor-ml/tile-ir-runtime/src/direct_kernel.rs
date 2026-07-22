use std::sync::Arc;

use wgpu::{CommandEncoder, ComputePass, PipelineCompilationOptions};

use crate::cache::{CachedDirectBindGroup, CachedKernel, DirectDynamicBindGroupKey, KernelCache};

#[derive(Clone, Debug)]
pub struct DirectKernelBinding {
    pub binding: u32,
    pub buffer: Arc<wgpu::Buffer>,
    pub read_only: bool,
}

#[derive(Debug, Clone)]
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
        /// The lowered kernel behind the pipeline, when the construction
        /// site still had it (the per-matrix decode pipeline cache keeps
        /// only the pipeline). Plans need it to persist across processes.
        cached: Option<Arc<CachedKernel>>,
        input: Arc<wgpu::Buffer>,
        weight: Arc<wgpu::Buffer>,
        output: Arc<wgpu::Buffer>,
    },
    /// Ordered multi-dispatch kernel. This is used by operations that need a
    /// scratch-buffer pass followed by a reduction pass.
    Sequence(Vec<DirectKernel>),
}

#[derive(Debug, Clone)]
pub struct DirectKernel {
    name: String,
    dispatch_size: [u32; 3],
    kind: DirectKernelKind,
}

#[derive(Debug, Clone)]
struct DirectKernelTemplateBinding {
    binding: u32,
    read_only: bool,
}

#[derive(Debug, Clone)]
enum DirectKernelTemplateKind {
    Dynamic {
        cached: Arc<CachedKernel>,
        bindings: Vec<DirectKernelTemplateBinding>,
    },
    Storage3 {
        pipeline: wgpu::ComputePipeline,
        cached: Option<Arc<CachedKernel>>,
    },
    Sequence(Vec<DirectKernelTemplate>),
}

/// A direct-kernel template with all bound buffers stripped out.
///
/// This preserves the lowered shader/pipeline metadata needed to rebuild a
/// dispatchable [`DirectKernel`] while avoiding retention of per-run input and
/// output buffers.
#[derive(Debug, Clone)]
pub struct DirectKernelTemplate {
    name: String,
    dispatch_size: [u32; 3],
    kind: DirectKernelTemplateKind,
}

impl DirectKernelTemplate {
    /// The serializable form of this template, or `None` for kernels whose
    /// pipeline cannot be rebuilt from a module alone.
    pub(crate) fn to_disk(&self) -> Option<crate::disk_cache::DiskTemplate> {
        let kind = match &self.kind {
            DirectKernelTemplateKind::Dynamic { cached, bindings } => {
                crate::disk_cache::DiskTemplateKind::Dynamic {
                    module: cached.kernel.module().clone(),
                    subgroups: cached.kernel.subgroups(),
                    bindings: bindings
                        .iter()
                        .map(|binding| (binding.binding, binding.read_only))
                        .collect(),
                }
            }
            DirectKernelTemplateKind::Storage3 { cached, .. } => {
                let cached = cached.as_ref()?;
                crate::disk_cache::DiskTemplateKind::Storage3 {
                    module: cached.kernel.module().clone(),
                    subgroups: cached.kernel.subgroups(),
                }
            }
            DirectKernelTemplateKind::Sequence(templates) => {
                crate::disk_cache::DiskTemplateKind::Sequence(
                    templates
                        .iter()
                        .map(|template| template.to_disk())
                        .collect::<Option<Vec<_>>>()?,
                )
            }
        };
        Some(crate::disk_cache::DiskTemplate {
            name: self.name.clone(),
            dispatch_size: self.dispatch_size,
            kind,
        })
    }

    /// Rebuild a template from its serialized form; `None` (a cache miss)
    /// when the stored module no longer validates.
    pub(crate) fn from_disk(
        disk: crate::disk_cache::DiskTemplate,
        cache: &KernelCache,
    ) -> Option<Self> {
        let kind = match disk.kind {
            crate::disk_cache::DiskTemplateKind::Dynamic {
                module,
                subgroups,
                bindings,
            } => {
                let kernel = fusor_tile_ir::NagaKernel::from_module(module, subgroups).ok()?;
                DirectKernelTemplateKind::Dynamic {
                    cached: Arc::new(CachedKernel::new(Arc::new(kernel))),
                    bindings: bindings
                        .into_iter()
                        .map(|(binding, read_only)| DirectKernelTemplateBinding {
                            binding,
                            read_only,
                        })
                        .collect(),
                }
            }
            crate::disk_cache::DiskTemplateKind::Storage3 { module, subgroups } => {
                let kernel = fusor_tile_ir::NagaKernel::from_module(module, subgroups).ok()?;
                let cached = Arc::new(CachedKernel::new(Arc::new(kernel)));
                let pipeline =
                    crate::dispatch::prepare_three_buffer_pipeline(cache, &disk.name, &cached);
                DirectKernelTemplateKind::Storage3 {
                    pipeline,
                    cached: Some(cached),
                }
            }
            crate::disk_cache::DiskTemplateKind::Sequence(templates) => {
                DirectKernelTemplateKind::Sequence(
                    templates
                        .into_iter()
                        .map(|template| Self::from_disk(template, cache))
                        .collect::<Option<Vec<_>>>()?,
                )
            }
        };
        Some(Self {
            name: disk.name,
            dispatch_size: disk.dispatch_size,
            kind,
        })
    }
}

pub struct PreparedDirectDispatch {
    steps: Vec<PreparedDirectDispatchStep>,
    _buffers: Vec<Arc<wgpu::Buffer>>,
}

struct PreparedDirectDispatchStep {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
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
        cached: Option<Arc<CachedKernel>>,
        input: Arc<wgpu::Buffer>,
        weight: Arc<wgpu::Buffer>,
        output: Arc<wgpu::Buffer>,
        dispatch_size: [u32; 3],
    ) -> Self {
        Self {
            name: name.into(),
            dispatch_size,
            kind: DirectKernelKind::Storage3 {
                cached,
                pipeline,
                input,
                weight,
                output,
            },
        }
    }

    pub fn sequence(name: impl Into<String>, kernels: Vec<DirectKernel>) -> Self {
        Self {
            name: name.into(),
            dispatch_size: [1, 1, 1],
            kind: DirectKernelKind::Sequence(kernels),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Clone this kernel's dispatch metadata without retaining any currently
    /// bound buffers.
    pub fn to_template(&self) -> DirectKernelTemplate {
        let kind = match &self.kind {
            DirectKernelKind::Dynamic { cached, bindings } => DirectKernelTemplateKind::Dynamic {
                cached: cached.clone(),
                bindings: bindings
                    .iter()
                    .map(|binding| DirectKernelTemplateBinding {
                        binding: binding.binding,
                        read_only: binding.read_only,
                    })
                    .collect(),
            },
            DirectKernelKind::Storage3 {
                pipeline, cached, ..
            } => DirectKernelTemplateKind::Storage3 {
                pipeline: pipeline.clone(),
                cached: cached.clone(),
            },
            DirectKernelKind::Sequence(kernels) => DirectKernelTemplateKind::Sequence(
                kernels.iter().map(DirectKernel::to_template).collect(),
            ),
        };
        DirectKernelTemplate {
            name: self.name.clone(),
            dispatch_size: self.dispatch_size,
            kind,
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
        match &self.kind {
            DirectKernelKind::Sequence(kernels) => {
                let mut steps = Vec::new();
                let mut buffers = Vec::new();
                for kernel in kernels {
                    let dispatch = kernel.prepare_dispatch(cache)?;
                    steps.extend(dispatch.steps);
                    buffers.extend(dispatch._buffers);
                }
                (!steps.is_empty()).then_some(PreparedDirectDispatch {
                    steps,
                    _buffers: buffers,
                })
            }
            DirectKernelKind::Storage3 {
                pipeline,
                input,
                weight,
                output,
                cached: _,
            } => {
                let bind_group_layout = cache.direct_three_buffer_bind_group_layout();
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
                let bind_group = cache.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&self.name),
                    layout: &bind_group_layout,
                    entries: &bind_entries,
                });
                Some(PreparedDirectDispatch {
                    steps: vec![PreparedDirectDispatchStep {
                        pipeline: pipeline.clone(),
                        bind_group,
                        dispatch_size: self.dispatch_size,
                    }],
                    _buffers: vec![input.clone(), weight.clone(), output.clone()],
                })
            }
            DirectKernelKind::Dynamic { cached, bindings } => {
                let (bind_group_layout, pipeline) = self.dynamic_pipeline(cache, cached, bindings);

                let bind_entries = bindings
                    .iter()
                    .map(|b| wgpu::BindGroupEntry {
                        binding: b.binding,
                        resource: b.buffer.as_entire_binding(),
                    })
                    .collect::<Vec<_>>();
                let has_writable_binding = bindings.iter().any(|binding| !binding.read_only);
                let bind_group = if has_writable_binding {
                    cache.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&self.name),
                        layout: &bind_group_layout,
                        entries: &bind_entries,
                    })
                } else {
                    let bind_group_key = DirectDynamicBindGroupKey::new(
                        bindings
                            .iter()
                            .map(|b| (b.binding, b.read_only, b.buffer.clone())),
                    );
                    cache
                        .direct_dynamic_bind_group_cache
                        .write()
                        .get_or_insert(bind_group_key, || {
                            let bind_group =
                                cache.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                    label: Some(&self.name),
                                    layout: &bind_group_layout,
                                    entries: &bind_entries,
                                });
                            CachedDirectBindGroup::new(
                                bind_group,
                                bindings
                                    .iter()
                                    .map(|binding| binding.buffer.clone())
                                    .collect(),
                            )
                        })
                        .bind_group
                        .clone()
                };

                Some(PreparedDirectDispatch {
                    steps: vec![PreparedDirectDispatchStep {
                        pipeline,
                        bind_group,
                        dispatch_size: self.dispatch_size,
                    }],
                    _buffers: bindings
                        .iter()
                        .map(|binding| binding.buffer.clone())
                        .collect(),
                })
            }
        }
    }

    /// The buffer-independent compiled artifacts of a dynamic kernel:
    /// bind-group layout and compute pipeline (plus the shader module and
    /// pipeline layout behind them). Everything sits behind per-kernel
    /// once-cells, so this is thread-safe and idempotent.
    fn dynamic_pipeline(
        &self,
        cache: &KernelCache,
        cached: &Arc<CachedKernel>,
        bindings: &[DirectKernelBinding],
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let bind_group_layout = cached
            .dynamic_bind_group_layout
            .get_or_init(|| {
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
                cache
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some(&self.name),
                        entries: &layout_entries,
                    })
            })
            .clone();
        let pipeline_layout = cached
            .dynamic_pipeline_layout
            .get_or_init(|| {
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
                crate::note_compile(
                    cache.config(),
                    &format!(
                        "pipeline name={} dispatch={:?} bindings={}",
                        self.name,
                        self.dispatch_size,
                        bindings.len()
                    ),
                );
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
        (bind_group_layout, pipeline)
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
            DirectKernelKind::Sequence(kernels) => kernels
                .iter()
                .flat_map(|kernel| kernel.bindings_for_test())
                .collect(),
        }
    }

    /// The buffers this kernel binds, in the canonical order used internally by
    /// `prepare_dispatch` (and mirrored by `rebind_buffers`). Used by plan
    /// caches to record where each binding's buffer comes from so the kernel can
    /// be replayed with fresh per-resolve binding buffers.
    pub fn binding_buffers(&self) -> Vec<Arc<wgpu::Buffer>> {
        let mut out = Vec::new();
        self.collect_buffers(&mut out);
        out
    }

    fn collect_buffers(&self, out: &mut Vec<Arc<wgpu::Buffer>>) {
        match &self.kind {
            DirectKernelKind::Dynamic { bindings, .. } => {
                out.extend(bindings.iter().map(|binding| binding.buffer.clone()));
            }
            DirectKernelKind::Storage3 {
                input,
                weight,
                output,
                ..
            } => {
                out.push(input.clone());
                out.push(weight.clone());
                out.push(output.clone());
            }
            DirectKernelKind::Sequence(kernels) => {
                for kernel in kernels {
                    kernel.collect_buffers(out);
                }
            }
        }
    }

    /// Clone this kernel, replacing its bound buffers positionally with `new`
    /// (which must have exactly `binding_buffers().len()` entries in the same
    /// order). The compiled pipeline / cached analysis is preserved; only the
    /// buffers (i.e. the bind-group resources) change. This lets a plan cache
    /// reuse a kernel built during a previous resolve while swapping in the
    /// current replay buffers.
    pub fn rebind_buffers(&self, new: &[Arc<wgpu::Buffer>]) -> Self {
        let mut cursor = 0;
        let kernel = self.rebind_from(new, &mut cursor);
        debug_assert_eq!(
            cursor,
            new.len(),
            "rebind_buffers received {} buffers for a kernel binding {cursor}",
            new.len()
        );
        kernel
    }

    fn rebind_from(&self, new: &[Arc<wgpu::Buffer>], cursor: &mut usize) -> Self {
        let kind = match &self.kind {
            DirectKernelKind::Dynamic { cached, bindings } => {
                let bindings = bindings
                    .iter()
                    .map(|binding| {
                        let buffer = new[*cursor].clone();
                        *cursor += 1;
                        DirectKernelBinding {
                            binding: binding.binding,
                            buffer,
                            read_only: binding.read_only,
                        }
                    })
                    .collect();
                DirectKernelKind::Dynamic {
                    cached: cached.clone(),
                    bindings,
                }
            }
            DirectKernelKind::Storage3 {
                pipeline, cached, ..
            } => {
                let input = new[*cursor].clone();
                let weight = new[*cursor + 1].clone();
                let output = new[*cursor + 2].clone();
                *cursor += 3;
                DirectKernelKind::Storage3 {
                    pipeline: pipeline.clone(),
                    cached: cached.clone(),
                    input,
                    weight,
                    output,
                }
            }
            DirectKernelKind::Sequence(kernels) => DirectKernelKind::Sequence(
                kernels
                    .iter()
                    .map(|kernel| kernel.rebind_from(new, cursor))
                    .collect(),
            ),
        };
        Self {
            name: self.name.clone(),
            dispatch_size: self.dispatch_size,
            kind,
        }
    }
}

impl DirectKernelTemplate {
    /// Build a dispatchable [`DirectKernel`] by binding buffers positionally in
    /// the same order returned by [`DirectKernel::binding_buffers`].
    pub fn bind_buffers(&self, new: &[Arc<wgpu::Buffer>]) -> DirectKernel {
        let mut cursor = 0;
        let kernel = self.bind_from(new, &mut cursor);
        debug_assert_eq!(
            cursor,
            new.len(),
            "bind_buffers received {} buffers for a template binding {cursor}",
            new.len()
        );
        kernel
    }

    fn bind_from(&self, new: &[Arc<wgpu::Buffer>], cursor: &mut usize) -> DirectKernel {
        let kind = match &self.kind {
            DirectKernelTemplateKind::Dynamic { cached, bindings } => {
                let bindings = bindings
                    .iter()
                    .map(|binding| {
                        let buffer = new[*cursor].clone();
                        *cursor += 1;
                        DirectKernelBinding {
                            binding: binding.binding,
                            buffer,
                            read_only: binding.read_only,
                        }
                    })
                    .collect();
                DirectKernelKind::Dynamic {
                    cached: cached.clone(),
                    bindings,
                }
            }
            DirectKernelTemplateKind::Storage3 { pipeline, cached } => {
                let input = new[*cursor].clone();
                let weight = new[*cursor + 1].clone();
                let output = new[*cursor + 2].clone();
                *cursor += 3;
                DirectKernelKind::Storage3 {
                    pipeline: pipeline.clone(),
                    cached: cached.clone(),
                    input,
                    weight,
                    output,
                }
            }
            DirectKernelTemplateKind::Sequence(kernels) => DirectKernelKind::Sequence(
                kernels
                    .iter()
                    .map(|kernel| kernel.bind_from(new, cursor))
                    .collect(),
            ),
        };
        DirectKernel {
            name: self.name.clone(),
            dispatch_size: self.dispatch_size,
            kind,
        }
    }
}

impl PreparedDirectDispatch {
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    pub fn run_step<'a>(&'a self, pass: &mut ComputePass<'a>, step_index: usize) {
        let Some(step) = self.steps.get(step_index) else {
            return;
        };
        let [x, y, z] = step.dispatch_size;
        if step.dispatch_size.contains(&0) {
            return;
        }
        pass.set_pipeline(&step.pipeline);
        pass.set_bind_group(0, &step.bind_group, &[]);
        pass.dispatch_workgroups(x, y, z);
    }

    pub fn run<'a>(&'a self, pass: &mut ComputePass<'a>) {
        for step_index in 0..self.step_count() {
            self.run_step(pass, step_index);
        }
    }
}
