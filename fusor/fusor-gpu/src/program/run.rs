use super::{Plan, ProgramStats};
use crate::{
    pool::{GpuBuffer, TENSOR_USAGE},
    target::GpuTarget,
};
use fusor_ir::{Result, egraph::Id, error::Error, target::Buf};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

/// Owns the arena, compiled shader and binding of a fixed logical program.
/// Inputs and feedback state are independent of the graph used to compile it.
pub struct Program {
    target: Arc<GpuTarget>,
    plan: Plan,
    arena: Buf,
    kernels: Vec<(wgpu::ComputePipeline, wgpu::BindGroup)>,
    grids: Vec<u32>,
    submissions: u64,
    uniform_bytes: Vec<u8>,
    initialized: rustc_hash::FxHashSet<Id>,
    pending: AtomicUsize,
    acceleration: super::ProgramAcceleration,
    fallback_reason: Option<String>,
}
impl Program {
    pub async fn build(target: Arc<GpuTarget>, plan: Plan) -> Result<Self> {
        let device = target.device().device();
        if device.limits().max_compute_invocations_per_workgroup < super::plan::BLOCK
            || device.limits().max_compute_workgroup_size_x < super::plan::BLOCK
        {
            return Err(Error::Device(
                "fused programs require 256 workgroup invocations".into(),
            ));
        }
        let arena = target
            .pool()
            .alloc_with_usage(plan.stats().arena_bytes, TENSOR_USAGE)?;
        let raw = arena.downcast_ref::<GpuBuffer>().unwrap();
        let selected = super::ProgramAcceleration::select(target.device(), plan.options);
        let (acceleration, kernels, fallback_reason) =
            match Self::compile_kernels(device, &plan, raw, selected).await {
                Ok(kernels) => (
                    selected,
                    kernels,
                    target.device().matrix_fallback().map(str::to_owned),
                ),
                // Experimental browser shader dialects can change independently of
                // Rust dependencies. Retry the same plan with portable matrices;
                // storage, state, and subgroup collectives stay identical. Compiler
                // admission/validation failures still surface as errors.
                Err(Error::Device(reason))
                    if selected.matrices == super::MatrixInstructions::Browser =>
                {
                    let fallback = super::ProgramAcceleration {
                        matrices: super::MatrixInstructions::Portable,
                        ..selected
                    };
                    let kernels = Self::compile_kernels(device, &plan, raw, fallback).await?;
                    (fallback, kernels, Some(reason))
                }
                Err(error) => return Err(error),
            };
        let cooperative = acceleration.cooperative();
        let grids = (0..kernels.len())
            .map(|i| plan.groups(i, cooperative))
            .collect();
        Ok(Self {
            grids,
            target,
            plan,
            arena,
            kernels,
            submissions: 0,
            uniform_bytes: vec![],
            initialized: Default::default(),
            pending: AtomicUsize::new(0),
            acceleration,
            fallback_reason,
        })
    }
    async fn compile_kernels(
        device: &wgpu::Device,
        plan: &Plan,
        raw: &GpuBuffer,
        acceleration: super::ProgramAcceleration,
    ) -> Result<Vec<(wgpu::ComputePipeline, wgpu::BindGroup)>> {
        let cooperative = acceleration.cooperative();
        let mut kernels = vec![];
        for (region, source) in plan
            .shaders_with(acceleration.native_validation())?
            .into_iter()
            .enumerate()
        {
            let module = naga::front::wgsl::parse_str(&source)
                .map_err(|e| Error::Plan(e.emit_to_string(&source)))?;
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                if cooperative {
                    naga::valid::Capabilities::COOPERATIVE_MATRIX
                        | naga::valid::Capabilities::SUBGROUP
                } else if acceleration.subgroups {
                    naga::valid::Capabilities::SUBGROUP
                } else {
                    naga::valid::Capabilities::empty()
                },
            )
            .validate(&module)
            .map_err(|e| Error::Plan(e.emit_to_string(&source)))?;
            #[cfg(target_arch = "wasm32")]
            let shader_source = {
                let mut source = super::emit::shader(&plan, region, acceleration)?;
                // Naga recognizes subgroup operations without the enable line;
                // browser WGSL requires it explicitly.
                if acceleration.subgroups {
                    source.insert_str(0, "enable subgroups;\n");
                }
                wgpu::ShaderSource::Wgsl(std::borrow::Cow::Owned(source))
            };
            #[cfg(not(target_arch = "wasm32"))]
            let shader_source = {
                let _ = region;
                wgpu::ShaderSource::Naga(std::borrow::Cow::Owned(module))
            };
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            // SAFETY: emission contains only statically counted loops, and plan
            // admission excludes overflow of every counter (including its final
            // BLOCK-sized increment). Storage bounds checks remain enabled.
            let shader = unsafe {
                device.create_shader_module_trusted(
                    wgpu::ShaderModuleDescriptor {
                        label: Some("fusor fixed logical program"),
                        source: shader_source,
                    },
                    wgpu::ShaderRuntimeChecks {
                        force_loop_bounding: false,
                        ..wgpu::ShaderRuntimeChecks::checked()
                    },
                )
            };
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fusor fixed logical program"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                // All scratch reads are dominated by explicit writes and
                // barriers in emit.rs, including padded tiles and identity lanes.
                compilation_options: wgpu::PipelineCompilationOptions {
                    zero_initialize_workgroup_memory: false,
                    ..Default::default()
                },
                cache: None,
            });
            let binding = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: raw.buffer.as_entire_binding(),
                }],
            });
            if let Some(error) = scope.pop().await {
                return Err(Error::Device(format!("program pipeline: {error}")));
            }
            kernels.push((pipeline, binding));
        }
        Ok(kernels)
    }
    pub fn acceleration_fallback(&self) -> Option<&str> {
        self.fallback_reason.as_deref()
    }
    pub fn acceleration(&self) -> super::ProgramAcceleration {
        self.acceleration
    }
    pub fn stats(&self) -> &ProgramStats {
        self.plan.stats()
    }
    pub fn plan(&self) -> &Plan {
        &self.plan
    }
    pub fn uniforms(&mut self, values: &[(fusor_ir::shape::SymId, f32)]) -> Result<()> {
        self.target.device().lost().check()?;
        let values: rustc_hash::FxHashMap<_, _> = values.iter().copied().collect();
        let leaves = self
            .plan
            .inputs()
            .iter()
            .filter_map(|input| input.uniform.map(|sym| (input.id, input.dtype, sym)))
            .map(|(id, dtype, sym)| {
                let value = values
                    .get(&sym)
                    .ok_or_else(|| Error::Plan(format!("unbound program uniform {sym:?}")))?;
                let word = match dtype {
                    fusor_ir::dtype::Dtype::F32 => value.to_bits(),
                    fusor_ir::dtype::Dtype::I32 => (*value as i32) as u32,
                    fusor_ir::dtype::Dtype::U32 => *value as u32,
                    _ => return Err(Error::Dtype("unsupported program uniform type".into())),
                };
                Ok((id, word.to_le_bytes()))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut bytes = Vec::with_capacity(self.plan.uniforms.len() * 4);
        for uniform in &self.plan.uniforms {
            let value = values
                .get(&uniform.sym)
                .ok_or_else(|| Error::Plan(format!("unbound program uniform {:?}", uniform.sym)))?;
            let word = match uniform.dtype {
                fusor_ir::dtype::Dtype::F32 => value.to_bits(),
                fusor_ir::dtype::Dtype::U32 => *value as u32,
                fusor_ir::dtype::Dtype::I32 => (*value as i32) as u32,
                _ => return Err(Error::Dtype("unsupported program uniform type".into())),
            };
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        if bytes != self.uniform_bytes {
            if let Some(first) = self.plan.uniforms.first() {
                self.target.device().queue().write_buffer(
                    &self.arena.downcast_ref::<GpuBuffer>().unwrap().buffer,
                    u64::from(first.offset) * 4,
                    &bytes,
                );
            }
            self.uniform_bytes = bytes;
        }
        for (id, bytes) in leaves {
            self.write(id, &bytes)?;
        }
        Ok(())
    }
    pub fn write(&mut self, id: Id, bytes: &[u8]) -> Result<()> {
        self.target.device().lost().check()?;
        let id = self
            .plan
            .by_id
            .get(&id)
            .map(|i| self.plan.values[*i].id)
            .ok_or_else(|| Error::Plan("value is not a program input".into()))?;
        let input = self
            .plan
            .inputs()
            .iter()
            .find(|i| i.id == id)
            .ok_or_else(|| Error::Plan("value is not a program input".into()))?;
        if bytes.len() as u64 != u64::from(input.elements) * 4 {
            return Err(Error::Shape(
                "input byte count differs from its compiled shape".into(),
            ));
        }
        self.target.device().queue().write_buffer(
            &self.arena.downcast_ref::<GpuBuffer>().unwrap().buffer,
            u64::from(input.offset) * 4,
            bytes,
        );
        self.initialized.insert(id);
        Ok(())
    }
    pub fn run(&mut self) -> Result<()> {
        self.target.device().lost().check()?;
        if self
            .plan
            .inputs()
            .iter()
            .any(|i| !self.initialized.contains(&i.id))
            || self.uniform_bytes.len() != self.plan.uniforms.len() * 4
        {
            return Err(Error::Plan(
                "initialize every program input and uniform before running".into(),
            ));
        }
        if self.pending.load(Ordering::Relaxed) >= 32 {
            #[cfg(not(target_arch = "wasm32"))]
            {
                self.target.launcher().poll_wait()?;
                self.pending.store(0, Ordering::Relaxed);
            }
            #[cfg(target_arch = "wasm32")]
            return Err(Error::Device(
                "program submission window is full; use run_async or synchronize".into(),
            ));
        }
        let device = self.target.device();
        let mut encoder = device.device().create_command_encoder(&Default::default());
        #[cfg(not(target_arch = "wasm32"))]
        let queries = self
            .target
            .launcher()
            .timestamp_query_set(self.kernels.len());
        #[cfg(target_arch = "wasm32")]
        let queries: Option<wgpu::QuerySet> = None;
        if queries.is_some() {
            for (i, (pipeline, binding)) in self.kernels.iter().enumerate() {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    timestamp_writes: queries.as_ref().map(|set| {
                        wgpu::ComputePassTimestampWrites {
                            query_set: set,
                            beginning_of_pass_write_index: Some(i as u32 * 2),
                            end_of_pass_write_index: Some(i as u32 * 2 + 1),
                        }
                    }),
                    ..Default::default()
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, binding, &[]);
                pass.dispatch_workgroups(self.grids[i], 1, 1);
            }
        } else {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            for (i, (pipeline, binding)) in self.kernels.iter().enumerate() {
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, binding, &[]);
                pass.dispatch_workgroups(self.grids[i], 1, 1);
            }
        }
        device.queue().submit([encoder.finish()]);
        if let Some(set) = queries {
            self.target.launcher().poll_wait()?;
            let profile = self.target.launcher().read_timestamps(
                self.target.pool(),
                &set,
                self.kernels.len(),
            )?;
            self.target.launcher().set_last_profile(profile);
        }
        self.submissions += 1;
        self.pending.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    /// Queue a step, yielding to the browser/native queue when its bounded
    /// submission window fills. There is no fence while space remains.
    pub async fn run_async(&mut self) -> Result<()> {
        if self.pending.load(Ordering::Relaxed) >= 32 {
            self.synchronize().await?;
        }
        self.run()
    }
    pub async fn synchronize(&self) -> Result<()> {
        self.target.launcher().wait_async().await?;
        self.pending.store(0, Ordering::Relaxed);
        Ok(())
    }
    pub fn dispatch_count(&self) -> u64 {
        self.submissions * self.kernels.len() as u64
    }
    /// Export one retained output to a standalone buffer for ordinary graph
    /// execution or readback. This copy is only needed at an observation point.
    pub fn export(&self, id: Id) -> Result<Buf> {
        Ok(self.export_many(&[id])?.remove(0))
    }
    /// Copy all requested values in one submission. Validate every request
    /// before making any copy or exposing an output to the caller.
    pub fn export_many(&self, ids: &[Id]) -> Result<Vec<Buf>> {
        self.target.device().lost().check()?;
        let values = ids
            .iter()
            .map(|id| self.plan.output(*id))
            .collect::<Result<Vec<_>>>()?;
        if self.submissions == 0 && values.iter().any(|v| !self.initialized.contains(&v.id)) {
            return Err(Error::Plan(
                "computed outputs are unavailable before the first step".into(),
            ));
        }
        let buffers = values
            .iter()
            .map(|v| {
                self.target
                    .pool()
                    .alloc_with_usage(u64::from(v.len()) * 4, TENSOR_USAGE)
            })
            .collect::<Result<Vec<_>>>()?;
        if ids.is_empty() {
            return Ok(buffers);
        }
        let device = self.target.device();
        let mut encoder = device.device().create_command_encoder(&Default::default());
        for (v, dst) in values.iter().zip(&buffers) {
            encoder.copy_buffer_to_buffer(
                &self.arena.downcast_ref::<GpuBuffer>().unwrap().buffer,
                u64::from(v.offset.unwrap()) * 4,
                &dst.downcast_ref::<GpuBuffer>().unwrap().buffer,
                0,
                u64::from(v.len()) * 4,
            );
        }
        device.queue().submit([encoder.finish()]);
        Ok(buffers)
    }
    pub async fn read(&self, id: Id) -> Result<Vec<u8>> {
        let value = self.plan.output(id)?;
        let bytes = self
            .target
            .readback(&self.export(id)?, u64::from(value.len()) * 4)
            .await?;
        self.pending.store(0, Ordering::Relaxed);
        Ok(bytes)
    }
}
