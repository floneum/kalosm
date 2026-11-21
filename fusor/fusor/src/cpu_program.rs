//! Compiled CPU programs. A value is resolved once with every kernel launch
//! its resolve ran recorded; re-running the recording recomputes it from the
//! bytes then in its input leaves, with no graph work in between. This is the
//! per-call evaluator of a search or a sampler, where a resolve per call
//! would cost more than the kernels.
use crate::{Error, Result, graph::GraphRef, session::Backend, tensor::Dyn};
use fusor_cpu::{AlignedBuf, Bound, CpuKernel, CpuTarget};
use fusor_ir::target::{Artifact, Buf, Uniforms};
use std::sync::Arc;

/// One launch a resolve ran, with the buffers it was bound to.
pub(crate) struct Recorded {
    pub(crate) artifact: Artifact,
    pub(crate) grid: [u32; 3],
    pub(crate) binds: Vec<Buf>,
    pub(crate) uniforms: Uniforms,
}

/// The buffer of one value of a [`CpuProgram`].
pub struct CpuValue(Buf);

impl CpuValue {
    fn buffer(&self) -> &AlignedBuf {
        self.0
            .downcast_ref::<AlignedBuf>()
            .expect("checked at construction")
    }
    /// Replace the value's leading bytes.
    pub fn write<T: bytemuck::Pod>(&self, data: &[T]) -> Result<()> {
        let aligned = self.buffer();
        let bytes: &[u8] = bytemuck::cast_slice(data);
        if bytes.len() > aligned.len() {
            return Err(Error::Shape(format!(
                "{} bytes written to a {}-byte value",
                bytes.len(),
                aligned.len()
            )));
        }
        // SAFETY: a program runs on one thread, and `write` is the buffer's
        // only writer between runs; the length was checked above.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), aligned.as_mut_ptr(), bytes.len()) };
        Ok(())
    }
    /// The value's elements, to edit in place between runs.
    pub fn edit<T: bytemuck::Pod, R>(&self, edit: impl FnOnce(&mut [T]) -> R) -> R {
        let aligned = self.buffer();
        let len = aligned.len() / std::mem::size_of::<T>();
        // SAFETY: as `write`: this is the buffer's only access between runs,
        // and a CPU buffer is aligned for every element type.
        edit(unsafe { std::slice::from_raw_parts_mut(aligned.as_mut_ptr() as *mut T, len) })
    }
    /// Replace element `index`.
    pub fn set<T: bytemuck::Pod>(&self, index: usize, value: T) -> Result<()> {
        let aligned = self.buffer();
        let size = std::mem::size_of::<T>();
        if (index + 1) * size > aligned.len() {
            return Err(Error::Shape(format!("element {index} is past the value")));
        }
        // SAFETY: as `write`; the element was bounds-checked above.
        unsafe { (aligned.as_mut_ptr().add(index * size) as *mut T).write_unaligned(value) };
        Ok(())
    }
    /// Copy the value's leading bytes out, as many as `out` holds.
    pub fn read_into<T: bytemuck::Pod>(&self, out: &mut [T]) -> Result<()> {
        let aligned = self.buffer();
        let bytes: &mut [u8] = bytemuck::cast_slice_mut(out);
        if bytes.len() > aligned.len() {
            return Err(Error::Shape(format!(
                "{} bytes read from a {}-byte value",
                bytes.len(),
                aligned.len()
            )));
        }
        bytes.copy_from_slice(&aligned.as_slice()[..bytes.len()]);
        Ok(())
    }
    /// The value's first element.
    pub fn read_scalar<T: bytemuck::Pod>(&self) -> Result<T> {
        let mut out = [T::zeroed()];
        self.read_into(&mut out)?;
        Ok(out[0])
    }
}

/// A value's resolve, replayable.
pub struct CpuProgram {
    _target: Arc<CpuTarget>,
    graph: GraphRef,
    /// Each launch with its buffers already resolved.
    launches: Vec<(CpuKernel, [u32; 3], Bound)>,
}

impl CpuProgram {
    /// Resolve `output` on its CPU graph, recording the launches. The graph's
    /// `from_slice` leaves keep their buffers, so [`Self::write`] on one
    /// changes what the next [`Self::run`] computes. Holding the recording
    /// pins every buffer it binds.
    pub fn compile(output: &Dyn) -> Result<Self> {
        let graph = output.graph().clone();
        let session = graph.session();
        let target = match session.backend() {
            Backend::Cpu(target) => target,
            #[cfg(feature = "gpu")]
            Backend::Gpu(_) => return Err(Error::Device("CPU programs need a CPU graph".into())),
        };
        // Resolve only: a CPU resolve is synchronous everywhere, a readback is
        // not on the web.
        let launches = session
            .record_cpu(|| session.resolve(std::slice::from_ref(&graph.tensor(output.id()))))?;
        let launches = launches
            .into_iter()
            .map(|launch| {
                let kernel = launch
                    .artifact
                    .downcast_ref::<CpuKernel>()
                    .ok_or_else(|| {
                        Error::Device("a launch was not built by the CPU target".into())
                    })?
                    .clone();
                let bound = kernel.bind(&launch.binds, &launch.uniforms)?;
                Ok((kernel, launch.grid, bound))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            _target: target,
            graph,
            launches,
        })
    }
    /// Kernel launches one run makes.
    pub fn launch_count(&self) -> usize {
        self.launches.len()
    }
    /// A bound value's buffer, resolved once: an input leaf to write before a
    /// run, or the output to read after it.
    pub fn value(&self, value: &Dyn) -> Result<CpuValue> {
        let buf = self
            .graph
            .device_buf(value.id())
            .ok_or_else(|| Error::Plan("the value has no buffer in this program".into()))?;
        if buf.downcast_ref::<AlignedBuf>().is_none() {
            return Err(Error::Device("the buffer is not a CPU buffer".into()));
        }
        Ok(CpuValue(buf))
    }
    /// Replace the bytes of an input leaf (or any bound value).
    pub fn write<T: bytemuck::Pod>(&self, value: &Dyn, data: &[T]) -> Result<()> {
        self.value(value)?.write(data)
    }
    /// Copy a bound value's bytes out, as many as `out` holds.
    pub fn read_into<T: bytemuck::Pod>(&self, value: &Dyn, out: &mut [T]) -> Result<()> {
        self.value(value)?.read_into(out)
    }
    /// A bound value's first element.
    pub fn read_scalar<T: bytemuck::Pod>(&self, value: &Dyn) -> Result<T> {
        self.value(value)?.read_scalar()
    }
    /// Run every recorded launch, in order, on the calling thread: a program's
    /// kernels are too small to amortize waking the worker pool.
    pub fn run(&self) -> Result<()> {
        for (kernel, grid, bound) in &self.launches {
            kernel.run_bound(*grid, bound)?;
        }
        Ok(())
    }
    /// Diagnostic: one run with each launch timed, as `(kernel, grid, microseconds)`.
    pub fn profile(&self) -> Result<Vec<(&'static str, [u32; 3], f64)>> {
        let mut out = Vec::with_capacity(self.launches.len());
        for (kernel, grid, bound) in &self.launches {
            let start = web_time::Instant::now();
            kernel.run_bound(*grid, bound)?;
            out.push((kernel.name, *grid, start.elapsed().as_secs_f64() * 1e6));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Device, Tensor};

    /// Reference for the graph below.
    fn expected(w0: &[f32], wv: &[f32], d: usize, n: usize, idx: &[u32], val: &[f32]) -> f32 {
        let mut total = 0f32;
        for j in 0..d {
            let acc: f32 = idx
                .iter()
                .zip(val)
                .map(|(&i, &v)| w0[j * n + i as usize] * v)
                .sum();
            total += wv[j] * acc.max(0.);
        }
        total.tanh()
    }

    #[test]
    fn replays_a_gather_mlp_on_new_inputs() {
        let cpu = Device::cpu();
        let (d, n, k) = (16usize, 40usize, 8usize);
        let w: Vec<f32> = (0..d * n)
            .map(|i| ((i * 7) % 13) as f32 / 13. - 0.5)
            .collect();
        let v: Vec<f32> = (0..d).map(|i| i as f32 / d as f32 - 0.3).collect();
        let w0 = Tensor::<2, f32>::from_slice(&cpu, [d, n], &w);
        let wv = Tensor::<2, f32>::from_slice(&cpu, [1, d], &v);
        let idx = Tensor::<1, u32>::from_slice(&cpu, [k], &vec![0u32; k]);
        let val = Tensor::<2, f32>::from_slice(&cpu, [k, 1], &vec![0f32; k]);
        let out = wv
            .matmul(&w0.index_select(1, &idx).matmul(&val).relu())
            .tanh();
        let program = CpuProgram::compile(out.as_dyn()).unwrap();
        assert!(program.launch_count() > 0);
        for trial in 1..4u32 {
            let indices: Vec<u32> = (0..k as u32)
                .map(|i| (i * 5 + trial * 3) % n as u32)
                .collect();
            let values: Vec<f32> = (0..k)
                .map(|i| (i as f32 - 2.) * 0.25 * trial as f32)
                .collect();
            program.write(idx.as_dyn(), &indices).unwrap();
            program.write(val.as_dyn(), &values).unwrap();
            program.run().unwrap();
            let got: f32 = program.read_scalar(out.as_dyn()).unwrap();
            let want = expected(&w, &v, d, n, &indices, &values);
            assert!((got - want).abs() < 1e-5, "trial {trial}: {got} vs {want}");
        }
    }
}
