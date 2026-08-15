//! Dispatching a compiled [`CpuKernel`](crate::emit::CpuKernel) over the
//! worker pool. The ISA level is dispatched **once per kernel launch**, not
//! per row: `dispatch!` establishes the target features around the whole grid
//! traversal, and the lane width `W` is a const generic inside it, so an `MxN`
//! register accumulator tile survives all the way into the innermost body.

use fusor2_ir::error::Error;
use fusor2_ir::target::{Buf, Uniforms};
use fusor2_ir::Result;
use std::sync::atomic::Ordering;

use crate::alloc::AlignedBuf;
use crate::emit::{run_workgroup, CpuKernel, Program, RawBuf};
use crate::pool::{WorkerPool, DISPATCH_COUNT};

/// Run one dispatch, parallelizing the grid when the pool is worth waking.
pub fn run(kernel: &CpuKernel, grid: [u32; 3], binds: &[Buf], uniforms: &Uniforms) -> Result<()> {
    let prog = &kernel.artifact.prog;
    let total = grid[0] as u64 * grid[1] as u64 * grid[2] as u64;
    if total == 0 {
        return Ok(());
    }

    // Binding 0 is always the uniform block. A caller may supply it, or leave
    // it out and let the launcher materialize it.
    let uni_buf;
    let mut bufs: Vec<RawBuf> = Vec::with_capacity(prog.buffer_elements.len());
    if binds.len() + 1 == prog.buffer_elements.len() {
        let bytes = uniforms.to_bytes();
        let mut b = AlignedBuf::zeroed(bytes.len().max(4))?;
        b.as_mut_slice()[..bytes.len()].copy_from_slice(&bytes);
        uni_buf = b;
        bufs.push(RawBuf {
            ptr: uni_buf.as_mut_ptr(),
            bytes: uni_buf.len(),
        });
    } else if binds.len() != prog.buffer_elements.len() {
        return Err(Error::Device(format!(
            "{} buffers bound but the kernel declares {}",
            binds.len(),
            prog.buffer_elements.len()
        )));
    }
    for b in binds {
        let buf = b.downcast_ref::<AlignedBuf>().ok_or_else(|| {
            Error::Device("a bound buffer did not come from this target".into())
        })?;
        bufs.push(RawBuf {
            ptr: buf.as_mut_ptr(),
            bytes: buf.len(),
        });
    }

    let pool = WorkerPool::global();
    // A kernel that accumulates atomically runs on one worker, which keeps the
    // accumulation order fixed and therefore the result bit-reproducible.
    let grain = if prog.has_atomic {
        total
    } else {
        grain_for(total, pool.num_threads())
    };
    let arena = kernel.artifact.arena_bytes.max(64) as usize;
    let width = prog.width;

    let bufs_ref: &[RawBuf] = &bufs;
    // Dispatches attributable to *this* launch, so the count is grid
    // independent and a concurrent launch on another host thread cannot
    // inflate it.
    let dispatches = std::sync::atomic::AtomicU64::new(0);
    let body = |span: std::ops::Range<u64>| {
        pool.with_scratch(arena, |scratch| {
            let ptr = scratch.as_mut_ptr();
            // One `Level` dispatch per parallel chunk; never per workgroup and
            // never per row. `Level::new()` itself ran once for the process.
            dispatches.fetch_add(1, Ordering::Relaxed);
            let level = crate::caps::level();
            fearless_simd::dispatch!(level, _simd => {
                for linear in span.clone() {
                    let gid = unlinearize(linear, grid);
                    match width {
                        16 => run_workgroup::<16>(prog, gid, grid, bufs_ref, ptr),
                        8 => run_workgroup::<8>(prog, gid, grid, bufs_ref, ptr),
                        _ => run_workgroup::<4>(prog, gid, grid, bufs_ref, ptr),
                    }
                }
            });
        });
    };

    pool.parallel_for(0..total, grain, &body);
    DISPATCH_COUNT.store(dispatches.load(Ordering::Relaxed), Ordering::Relaxed);
    Ok(())
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
pub fn grain_for(total: u64, threads: u32) -> u64 {
    let threads = threads.max(1) as u64;
    if threads == 1 {
        return total.max(1);
    }
    // Four chunks per worker: enough slack for uneven workgroups without
    // paying a wake per grid point.
    (total.div_ceil(threads * 4)).max(1)
}

/// `Level` dispatches the most recent launch performed.
pub fn dispatch_count() -> u64 {
    DISPATCH_COUNT.load(Ordering::Relaxed)
}

/// Reset the dispatch counter.
pub fn reset_dispatch_count() {
    DISPATCH_COUNT.store(0, Ordering::Relaxed);
}

/// Bytes one program's thread-local scratch needs.
pub fn arena_bytes(prog: &Program) -> u32 {
    prog.arena_bytes
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
}
