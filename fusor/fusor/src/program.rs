//! Compiled fixed-shape training steps. State feedback is committed after the
//! whole step. Ordinary graph values are updated explicitly
//! at observation points with [`TrainingProgram::export`].
use crate::{Error, Result, graph::GraphRef, session::Backend, tensor::Dyn};
pub use fusor_gpu::program::{
    MatrixInstructions, ProgramAcceleration, ProgramOptions, ProgramStats,
};

/// An explicitly compiled GPU step with private, persistent inputs and state.
pub struct TrainingProgram {
    graph: GraphRef,
    gpu: fusor_gpu::program::Program,
}
impl TrainingProgram {
    /// Compile logical outputs and simultaneous `(state, next_state)` feedback.
    /// This snapshots current input values, but performs no training step.
    pub async fn compile(roots: &[Dyn], feedback: &[(Dyn, Dyn)]) -> Result<Self> {
        Self::compile_with_options(roots, feedback, ProgramOptions::default()).await
    }
    /// Compile with explicit fusion/portability controls.
    pub async fn compile_with_options(
        roots: &[Dyn],
        feedback: &[(Dyn, Dyn)],
        options: ProgramOptions,
    ) -> Result<Self> {
        let graph = roots
            .first()
            .ok_or_else(|| Error::Plan("training program requires an output".into()))?
            .graph()
            .clone();
        for t in roots
            .iter()
            .chain(feedback.iter().flat_map(|(a, b)| [a, b]))
        {
            if !GraphRef::ptr_eq(&graph, t.graph()) {
                return Err(Error::Device(
                    "program operands belong to different graphs".into(),
                ));
            }
        }
        let backend = graph.session().backend();
        let target = match backend {
            Backend::Gpu(target) => target,
            #[cfg(feature = "cpu")]
            _ => return Err(Error::Device("training programs require a GPU".into())),
        };
        let limits = target.device().device().limits();
        let max_bytes = limits
            .max_buffer_size
            .min(u64::from(limits.max_storage_buffer_binding_size));
        let ids = roots.iter().map(Dyn::id).collect::<Vec<_>>();
        let state = feedback
            .iter()
            .map(|(a, b)| (a.id(), b.id()))
            .collect::<Vec<_>>();
        let plan = graph.with_egraph(|g| {
            fusor_gpu::program::Plan::compile_with_options(g, &ids, &state, max_bytes, options)
        })?;
        let mut initial = vec![];
        for input in plan.inputs() {
            let bytes = if input.uniform.is_some() {
                // The uniform upload below initializes scalar leaves and
                // expression uniforms with their declared dtype together.
                continue;
            } else if graph.device_buf(input.id).is_none() {
                graph
                    .leaf_bytes(input.id)
                    .ok_or_else(|| Error::Plan("program input has no initial data".into()))?
            } else {
                graph.tensor(input.id).to_bytes_async().await?
            };
            initial.push((input.id, bytes));
        }
        let mut gpu = fusor_gpu::program::Program::build(target, plan).await?;
        for (id, bytes) in initial {
            gpu.write(id, &bytes)?;
        }
        gpu.uniforms(&graph.uniform_scalars())?;
        Ok(Self { graph, gpu })
    }
    fn check(&self, value: &Dyn) -> Result<()> {
        if !GraphRef::ptr_eq(&self.graph, value.graph()) {
            return Err(Error::Device("value belongs to another graph".into()));
        }
        Ok(())
    }
    /// Instructions actually selected after probing the device.
    pub fn acceleration(&self) -> ProgramAcceleration {
        self.gpu.acceleration()
    }
    /// Why an advertised browser matrix extension fell back during compilation.
    pub fn acceleration_fallback(&self) -> Option<&str> {
        self.gpu.acceleration_fallback()
    }
    /// Compiled stage count and memory requirements.
    pub fn stats(&self) -> &ProgramStats {
        self.gpu.stats()
    }
    /// Update a program input; the original graph leaf is not mutated.
    pub fn write(&mut self, input: &Dyn, bytes: &[u8]) -> Result<()> {
        self.check(input)?;
        self.gpu.write(input.id(), bytes)
    }
    /// Queue one complete step, including simultaneous state feedback.
    pub fn run(&mut self) -> Result<()> {
        self.gpu.uniforms(&self.graph.uniform_scalars())?;
        self.gpu.run()
    }
    /// Queue a step with bounded submissions and an asynchronous GPU fence
    /// only when the queue window fills. Use this entry point in browsers.
    pub async fn run_async(&mut self) -> Result<()> {
        self.gpu.uniforms(&self.graph.uniform_scalars())?;
        self.gpu.run_async().await
    }
    /// Wait for all submitted work without reading a tensor back.
    pub async fn synchronize(&self) -> Result<()> {
        self.gpu.synchronize().await
    }
    /// Number of GPU dispatches submitted by this program.
    pub fn dispatch_count(&self) -> u64 {
        self.gpu.dispatch_count()
    }
    /// Read a retained output or the current state as packed tensor bytes.
    pub async fn read(&self, output: &Dyn) -> Result<Vec<u8>> {
        self.check(output)?;
        self.gpu.read(output.id()).await
    }
    /// Make a retained output/state available to normal graph execution,
    /// using a GPU copy with no host round trip.
    pub fn export(&self, values: &[Dyn]) -> Result<()> {
        for value in values {
            self.check(value)?;
        }
        let ids = values.iter().map(Dyn::id).collect::<Vec<_>>();
        let buffers = self
            .gpu
            .export_many(&ids)?
            .into_iter()
            .zip(ids)
            .map(|(buf, id)| (id, buf, None))
            .collect::<Vec<_>>();
        self.graph.bind_classes(&buffers);
        Ok(())
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::{Device, Tensor, tensor::Scalar};
    use fusor_ir::{
        dtype::Dtype,
        ir::logical::{LeafKind, Logical},
    };

    #[test]
    fn changing_expression_and_integer_leaf_uniforms_reuses_the_program() {
        let device = Device::gpu_blocking().unwrap();
        let graph = device.handle();
        let scale = graph.fresh_sym();
        let integer = graph.fresh_sym();
        graph.set_uniform(scale, 1.);
        graph.set_uniform(integer, 1.);
        let leaf = graph
            .add_logical(Logical::Leaf(LeafKind::Uniform {
                sym: integer,
                dtype: Dtype::I32,
            }))
            .unwrap();
        let leaf = graph.tensor(leaf);
        let x = Tensor::<1, f32>::from_slice(&device, [2], &[1., 3.]);
        let output = x.as_dyn().mul_scalar(Scalar::Uniform(scale)).unwrap();
        let integers = Tensor::<1, i32>::from_slice(&device, [2], &[3, 5]);
        let integer_output = integers
            .as_dyn()
            .mul_scalar(Scalar::Uniform(scale))
            .unwrap();
        let mut program = pollster::block_on(TrainingProgram::compile(
            &[output.clone(), leaf.clone(), integer_output.clone()],
            &[],
        ))
        .unwrap();
        for step in 1..5 {
            graph.set_uniform(scale, step as f32 * 0.5);
            graph.set_uniform(integer, step as f32 + 0.75);
            program.run().unwrap();
            let bytes = pollster::block_on(program.read(&output)).unwrap();
            let got: Vec<_> = bytes
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            assert_eq!(got, [step as f32 * 0.5, step as f32 * 1.5]);
            let bytes = pollster::block_on(program.read(&leaf)).unwrap();
            assert_eq!(i32::from_le_bytes(bytes.try_into().unwrap()), step);
            let bytes = pollster::block_on(program.read(&integer_output)).unwrap();
            let got: Vec<_> = bytes
                .chunks_exact(4)
                .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            assert_eq!(got, [3 * (step / 2), 5 * (step / 2)]);
        }
    }
}
