//! [`CpuTarget`] — the [`Target`] implementation.

use fusor2_ir::cost::DeviceFacts;
use fusor2_ir::device::Caps;
use fusor2_ir::dtype::Persistence;
use fusor2_ir::egraph::{Id, Rule};
use fusor2_ir::error::Error;
use fusor2_ir::ir::launch::SchedPoint;
use fusor2_ir::ir::kernel::KernelIr;
use fusor2_ir::ir::Node;
use fusor2_ir::target::{Artifact, Buf, EmitError, LowerCtx, Target, Uniforms};
use fusor2_ir::Result;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::alloc::AlignedBuf;
use crate::emit::{CpuArtifact, CpuKernel};

/// The SIMD backend.
pub struct CpuTarget {
    caps: Caps,
    facts: DeviceFacts,
    /// `(size)`-keyed free list; a buffer is reusable once its `Arc` is unique.
    pool: Mutex<FxHashMap<u64, Vec<Buf>>>,
}

impl CpuTarget {
    pub fn new() -> Result<Self> {
        let caps = crate::caps::cpu_caps().clone();
        let facts = seed_facts(&caps);
        Ok(Self {
            caps,
            facts,
            pool: Mutex::new(FxHashMap::default()),
        })
    }

    /// Compile without going through the opaque [`Artifact`] wrapper.
    ///
    /// No arena planner is attached: the emitter's sequential packing is
    /// always legal here because thread-local scratch aliases freely.
    pub fn compile(&self, ir: &KernelIr) -> std::result::Result<CpuArtifact, EmitError> {
        crate::emit::compile(ir, &self.caps, None)
    }
}

/// The shipped rate table for a CPU, derived from [`Caps`].
fn seed_facts(caps: &Caps) -> DeviceFacts {
    let threads = caps.threads.max(1) as u64;
    let lanes = *caps.simd_widths.last().unwrap_or(&4) as u64;
    // ~3 GHz x lanes x 2 (fma) per core.
    let fma = 3_000 * lanes * 2 * threads;
    DeviceFacts {
        launch_ps: 1_000_000,
        dram_bytes_per_us: 30_000,
        llc_bytes: crate::caps::CpuCaps::llc_bytes(),
        wg_bytes_per_us: 400_000,
        mac_per_us: [
            [fma, fma, fma, fma / 2, fma / 2],
            [1, 1, 1, 1, 1],
            [fma * 4, fma * 4, fma * 4, fma * 4, fma * 4],
        ],
        trans_ps: 2_000,
        store_ps_per_element: 300,
        saturation_lanes: (threads * lanes * 4) as u32,
        single_buffered_traffic_pct: 100,
        compile_ps_per_kernel: 200_000_000,
        // Measured order of magnitude for waking a parked worker and joining.
        thread_wake_ps: 2_000_000,
        caps: caps.clone(),
    }
}

impl Target for CpuTarget {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn caps(&self) -> &Caps {
        &self.caps
    }

    fn facts(&self) -> &DeviceFacts {
        &self.facts
    }

    fn rules(&self) -> &'static [Rule] {
        crate::rules::CPU_RULES
    }

    fn lower(&self, node: &Node, id: Id, theta: SchedPoint, cx: &LowerCtx<'_>) -> Result<KernelIr> {
        crate::lower::lower(&self.caps, node, id, theta, cx)
    }

    fn emit(&self, ir: &KernelIr) -> std::result::Result<Artifact, EmitError> {
        let artifact = self.compile(ir)?;
        Ok(Artifact::new(CpuKernel {
            name: artifact.name,
            block: artifact.block,
            vector_width: artifact.prog.width,
            artifact,
        }))
    }

    fn launch(
        &self,
        artifact: &Artifact,
        grid: [u32; 3],
        binds: &[Buf],
        uniforms: &Uniforms,
    ) -> Result<()> {
        let kernel = artifact
            .downcast_ref::<CpuKernel>()
            .ok_or_else(|| Error::Device("artifact was not built by the CPU target".into()))?;
        crate::launch::run(kernel, grid, binds, uniforms)
    }

    fn alloc(&self, bytes: u64, _persistence: Persistence) -> Result<Buf> {
        // Every load reads a `u32` word, so a buffer whose length is not a
        // whole number of words has an unreadable tail. Native quantized
        // blocks are 18, 22, 34 and 210 bytes, so this is not a corner case.
        let bytes = bytes.next_multiple_of(4);
        let mut pool = self.pool.lock();
        if let Some(free) = pool.get_mut(&bytes) {
            if let Some(pos) = free.iter().position(|b| b.refcount() == 1) {
                return Ok(free.swap_remove(pos));
            }
        }
        drop(pool);
        let buf = Buf::new(AlignedBuf::zeroed(bytes as usize)?);
        self.pool.lock().entry(bytes).or_default().push(buf.clone());
        Ok(buf)
    }

    /// No-op: [`crate::pool::WorkerPool::parallel_for`] joins synchronously, so
    /// every dispatch has already retired when `launch` returns.
    fn wait(&self) -> Result<()> {
        Ok(())
    }
}
