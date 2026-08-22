//! Dispatching a compiled [`CpuKernel`](crate::emit::CpuKernel) over the
//! worker pool. Every production kernel must have either a native Cranelift
//! artifact or a platform GEMM contract.

use fusor_ir::Result;
use fusor_ir::error::Error;
use fusor_ir::target::{Buf, Uniforms};
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock};

use crate::alloc::AlignedBuf;
use crate::emit::{CpuKernel, Program, RawBuf};
use crate::pool::{DISPATCH_COUNT, WorkerPool};

/// Run one dispatch, parallelizing the grid when the pool is worth waking.
pub(crate) fn run(
    kernel: &CpuKernel,
    grid: [u32; 3],
    binds: &[Buf],
    uniforms: &Uniforms,
) -> Result<()> {
    let prog = &kernel.artifact.prog;
    let total = grid[0] as u64 * grid[1] as u64 * grid[2] as u64;
    if total == 0 {
        return Ok(());
    }

    let bound = bind_buffers(prog, binds, uniforms)?;
    let bufs = bound.raw.as_slice();

    if let Some(contract) = &kernel.artifact.contract {
        return crate::gemm::run(contract, bufs);
    }

    let jit = kernel.artifact.jit.ok_or_else(|| {
        Error::Device(format!(
            "CPU kernel {} has neither a Cranelift artifact nor a GEMM contract",
            kernel.name
        ))
    })?;

    let pool = WorkerPool::global();
    // A kernel that accumulates atomically runs on one worker, which keeps the
    // accumulation order fixed and therefore the result bit-reproducible.
    let grain = if prog.has_atomic || total <= pool.num_threads() as u64 {
        total
    } else {
        grain_for(total, pool.num_threads())
    };
    let arena = kernel.artifact.arena_bytes.max(64) as usize;

    let bufs_ref: &[RawBuf] = bufs;
    // Dispatches attributable to *this* launch, so the count is grid
    // independent and a concurrent launch on another host thread cannot
    // inflate it.
    let dispatches = std::sync::atomic::AtomicU64::new(0);
    let body = |span: std::ops::Range<u64>| {
        pool.with_scratch(arena, |scratch| {
            let _scratch = scratch;
            dispatches.fetch_add(1, Ordering::Relaxed);
            for linear in span {
                let gid = unlinearize(linear, grid);
                jit.run(bufs_ref, gid, grid);
            }
        });
    };

    pool.parallel_for(0..total, grain, &body);
    DISPATCH_COUNT.store(dispatches.load(Ordering::Relaxed), Ordering::Relaxed);
    Ok(())
}

struct BoundBuffers {
    // Retain the immutable cached uniform allocation for as long as `raw` can
    // be used.
    _uniform: Option<Arc<AlignedBuf>>,
    raw: SmallVec<[RawBuf; 8]>,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct UniformKey {
    dims: Vec<u32>,
    scalars: Vec<u32>,
}

fn uniform_buffer(uniforms: &Uniforms) -> Result<Arc<AlignedBuf>> {
    static CACHE: OnceLock<Mutex<FxHashMap<UniformKey, Arc<AlignedBuf>>>> = OnceLock::new();
    let key = UniformKey {
        dims: uniforms.dims.clone(),
        scalars: uniforms
            .scalars
            .iter()
            .map(|value| value.to_bits())
            .collect(),
    };
    let cache = CACHE.get_or_init(|| Mutex::new(FxHashMap::default()));
    if let Some(buffer) = cache
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&key)
        .cloned()
    {
        return Ok(buffer);
    }

    let bytes = uniforms.to_bytes();
    let mut buffer = AlignedBuf::zeroed(bytes.len().max(4))?;
    buffer.as_mut_slice()[..bytes.len()].copy_from_slice(&bytes);
    let buffer = Arc::new(buffer);
    cache
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .insert(key, Arc::clone(&buffer));
    Ok(buffer)
}

fn bind_buffers(prog: &Program, binds: &[Buf], uniforms: &Uniforms) -> Result<BoundBuffers> {
    let mut uniform = None;
    let mut raw = SmallVec::with_capacity(prog.buffer_elements.len());
    if binds.len() + 1 == prog.buffer_elements.len() {
        let buf = uniform_buffer(uniforms)?;
        raw.push(RawBuf {
            ptr: buf.as_mut_ptr(),
            bytes: buf.len(),
        });
        uniform = Some(buf);
    } else if binds.len() != prog.buffer_elements.len() {
        return Err(Error::Device(format!(
            "{} buffers bound but the kernel declares {}",
            binds.len(),
            prog.buffer_elements.len()
        )));
    }
    for bind in binds {
        let buf = bind
            .downcast_ref::<AlignedBuf>()
            .ok_or_else(|| Error::Device("a bound buffer did not come from this target".into()))?;
        raw.push(RawBuf {
            ptr: buf.as_mut_ptr(),
            bytes: buf.len(),
        });
    }
    Ok(BoundBuffers {
        _uniform: uniform,
        raw,
    })
}

/// Recover the 3-D workgroup id from a linear grid index.
#[inline(always)]
fn unlinearize(linear: u64, grid: [u32; 3]) -> [u32; 3] {
    let x = grid[0].max(1) as u64;
    let y = grid[1].max(1) as u64;
    [
        (linear % x) as u32,
        ((linear / x) % y) as u32,
        (linear / (x * y)) as u32,
    ]
}

/// Chunk size handed to `parallel_for`, chosen so one chunk amortizes
/// `thread_wake_ps`.
///
/// This is the whole of the "should we parallelize?" question on CPU, and it
/// is a *cost* question, which is why `PARALLEL_THRESHOLD = 16_777_216` does
/// not appear anywhere in this crate: the extractor prices an outer tile loop
/// marked parallel against the measured pool-wake cost, and the launcher only
/// has to pick a grain that keeps every worker fed.
pub(crate) fn grain_for(total: u64, threads: u32) -> u64 {
    let threads = threads.max(1) as u64;
    if threads == 1 {
        return total.max(1);
    }
    // Four chunks per worker leave enough slack for uneven workgroups.
    (total.div_ceil(threads * 4)).max(1)
}

/// Parallel chunks used by the most recent launch.
#[cfg(test)]
pub(crate) fn dispatch_count() -> u64 {
    DISPATCH_COUNT.load(Ordering::Relaxed)
}

/// Reset the dispatch counter.
#[cfg(test)]
pub(crate) fn reset_dispatch_count() {
    DISPATCH_COUNT.store(0, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_unlinearizes_row_major() {
        let grid = [3, 4, 2];
        let mut seen = std::collections::HashSet::new();
        for i in 0..24u64 {
            let g = unlinearize(i, grid);
            assert!(g[0] < 3 && g[1] < 4 && g[2] < 2);
            assert!(seen.insert(g));
        }
        assert_eq!(seen.len(), 24);
    }

    #[test]
    fn grain_keeps_every_worker_fed() {
        assert_eq!(grain_for(1000, 1), 1000);
        let g = grain_for(1000, 8);
        assert!(g >= 1 && g * 8 * 4 >= 1000 - 8 * 4);
    }

    #[test]
    fn missing_native_artifact_is_an_error() {
        let prog = std::sync::Arc::new(Program {
            tape: Vec::new(),
            segments: Vec::new(),
            regs: 0,
            locals: 0,
            tiles: Vec::new(),
            maps: Vec::new(),
            buffer_elements: Vec::new(),
            arena_bytes: 0,
            block: 1,
            width: 4,
            has_atomic: false,
        });
        let kernel = CpuKernel {
            name: "missing_native_test",
            block: 1,
            vector_width: 4,
            artifact: crate::emit::CpuArtifact {
                prog,
                contract: None,
                jit: None,
                grid: [1, 1, 1],
                block: 1,
                name: "missing_native_test",
                arena_bytes: 0,
            },
        };

        let error = run(&kernel, [1, 1, 1], &[], &Uniforms::default()).unwrap_err();
        assert!(
            error.to_string().contains("neither a Cranelift artifact"),
            "{error}"
        );
    }
}
