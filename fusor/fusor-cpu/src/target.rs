//! [`CpuTarget`] — the [`Target`] implementation.

use fusor_ir::Result;
use fusor_ir::cost::DeviceFacts;
use fusor_ir::device::Caps;
use fusor_ir::dtype::Persistence;
use fusor_ir::egraph::{Id, Rule};
use fusor_ir::error::Error;
use fusor_ir::ir::Node;
use fusor_ir::ir::kernel::KernelIr;
use fusor_ir::ir::launch::SchedPoint;
use fusor_ir::target::{Artifact, Buf, EmitError, LowerCtx, Target, Uniforms};
use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::alloc::AlignedBuf;
use crate::emit::CpuKernel;

/// The native CPU backend.
pub struct CpuTarget {
    caps: Caps,
    facts: DeviceFacts,
    /// `(size)`-keyed free list; a buffer is reusable once its `Arc` is unique.
    pool: Mutex<FxHashMap<u64, PoolBucket>>,
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
}

/// The shipped rate table for a CPU, derived from [`Caps`].
fn seed_facts(caps: &Caps) -> DeviceFacts {
    let threads = caps.threads.max(1) as u64;
    let lanes = *caps.simd_widths.last().unwrap_or(&4) as u64;
    // ~3 GHz x lanes x 2 (fma) per core.
    let fma = 3_000 * lanes * 2 * threads;
    DeviceFacts {
        coop_step_ps: 0,
        lane_step_ps: 0,
        lane_launch_ps: 0,
        // One-workgroup maps measure 10--20 us through the generic runner; pricing
        // them lower materializes hundreds of avoidable micro-kernels.
        launch_ps: 20_000_000,
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

    fn lower(&self, node: &Node, _: Id, theta: SchedPoint, cx: &LowerCtx<'_>) -> Result<KernelIr> {
        crate::lower::lower(&self.caps, node, theta, cx)
    }

    fn emit(&self, ir: &KernelIr) -> std::result::Result<Artifact, EmitError> {
        Ok(Artifact::new(crate::emit::emit(ir, &self.caps)?))
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
        // Loads read `u32` words, so round up: quantized blocks are 18/22/34/210 bytes.
        let bytes = bytes.next_multiple_of(4);
        let mut pool = self.pool.lock();
        if let Some(bucket) = pool.get_mut(&bytes)
            && let Some(buffer) = bucket.reusable()
        {
            return Ok(buffer);
        }
        drop(pool);
        let buf = Buf::new(AlignedBuf::zeroed(bytes as usize)?);
        self.pool
            .lock()
            .entry(bytes)
            .or_default()
            .buffers
            .push(buf.clone());
        Ok(buf)
    }

    fn copy(&self, src: &Buf) -> Result<Buf> {
        let source = src
            .downcast_ref::<AlignedBuf>()
            .ok_or_else(|| Error::Device("copy source is not an AlignedBuf".into()))?;
        let dst = self.alloc(source.len() as u64, Persistence::Persistent)?;
        let target = dst
            .downcast_ref::<AlignedBuf>()
            .ok_or_else(|| Error::Device("copy target is not an AlignedBuf".into()))?;
        // SAFETY: `target` is a distinct, fresh allocation of at least `source.len()`
        // bytes, written through `as_mut_ptr`.
        unsafe {
            std::ptr::copy_nonoverlapping(source.as_ptr(), target.as_mut_ptr(), source.len());
        }
        Ok(dst)
    }

    /// No-op: `parallel_for` joins, so every dispatch has retired.
    fn wait(&self) -> Result<()> {
        Ok(())
    }
}

#[derive(Default)]
struct PoolBucket {
    buffers: Vec<Buf>,
    cursor: usize,
}

impl PoolBucket {
    fn reusable(&mut self) -> Option<Buf> {
        const MAX_SCAN: usize = 16;
        let len = self.buffers.len();
        for _ in 0..len.min(MAX_SCAN) {
            let index = self.cursor % len;
            self.cursor = (index + 1) % len;
            if self.buffers[index].refcount() == 1 {
                // The pool keeps a handle; strong count back to one means reusable.
                return Some(self.buffers[index].clone());
            }
        }
        None
    }
}
