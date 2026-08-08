//! One persistent worker pool. Parallelism is a scheduling attribute on an
//! outer L1 tile loop, priced against the real pool-wake cost
//! (`DeviceFacts::thread_wake_ps`), so there is no element-count threshold.
//!
//! Threads are created once at pool init and never per call, which puts the
//! break-even for parallelizing at a few microseconds of work.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::ops::Range;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use crate::alloc::AlignedBuf;

/// A closure handed to the workers for the duration of one `parallel_for`.
///
/// The caller blocks until every chunk has retired, so the pointee outlives
/// every dereference. That is the whole safety argument; it is the same one a
/// scoped thread pool makes, minus the per-call spawn.
#[derive(Copy, Clone)]
struct JobPtr(*const (dyn Fn(Range<u64>) + Send + Sync));

// SAFETY: the referent is `Send + Sync` and `parallel_for` joins before it
// returns, so no worker can observe it after it is dropped.
unsafe impl Send for JobPtr {}
// SAFETY: as above.
unsafe impl Sync for JobPtr {}

#[derive(Default)]
struct Queue {
    chunks: VecDeque<Range<u64>>,
    job: Option<JobPtr>,
    /// Chunks popped but not yet finished.
    active: usize,
    shutdown: bool,
}

struct Shared {
    q: Mutex<Queue>,
    work: Condvar,
    done: Condvar,
    /// One submission at a time. Two host threads launching concurrently take
    /// turns rather than trampling each other's chunk queue; a *nested* call
    /// never reaches here, because `IN_POOL` sends it down the serial path.
    submit: Mutex<()>,
}

thread_local! {
    /// Re-entrancy guard: a `parallel_for` nested inside a worker runs serially
    /// rather than deadlocking on the pool it is already inside.
    static IN_POOL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };

    /// Per-thread 64-byte-aligned workgroup scratch, grown to
    /// `ArenaPlan::total_bytes` and reused across launches.
    static SCRATCH: RefCell<AlignedBuf> = RefCell::new(AlignedBuf::zeroed(0).expect("empty"));
}

/// Counts `Level` dispatches, so `level_dispatched_once` can assert exactly one
/// per launch regardless of grid size.
pub(crate) static DISPATCH_COUNT: AtomicU64 = AtomicU64::new(0);

/// The process-wide worker pool.
pub struct WorkerPool {
    threads: u32,
    shared: Arc<Shared>,
}

static POOL: OnceLock<WorkerPool> = OnceLock::new();

impl WorkerPool {
    /// The shared pool, started on first use.
    pub fn global() -> &'static WorkerPool {
        POOL.get_or_init(|| WorkerPool::start(crate::caps::CpuCaps::threads()))
    }

    fn start(threads: u32) -> WorkerPool {
        let shared = Arc::new(Shared {
            q: Mutex::new(Queue::default()),
            work: Condvar::new(),
            done: Condvar::new(),
            submit: Mutex::new(()),
        });
        // The calling thread participates, so only `threads - 1` are spawned.
        for i in 1..threads {
            let shared = Arc::clone(&shared);
            let _ = std::thread::Builder::new()
                .name(format!("fusor2-cpu-{i}"))
                .spawn(move || worker(&shared));
        }
        WorkerPool { threads, shared }
    }

    pub fn num_threads(&self) -> u32 {
        self.threads
    }

    /// Run `body` over `range` in chunks of at least `grain`.
    pub fn parallel_for(
        &self,
        range: Range<u64>,
        grain: u64,
        body: &(dyn Fn(Range<u64>) + Send + Sync),
    ) {
        if range.start >= range.end {
            return;
        }
        let grain = grain.max(1);
        let total = range.end - range.start;
        let nested = IN_POOL.with(|f| f.get());
        if self.threads <= 1 || nested || total <= grain {
            body(range);
            return;
        }

        let mut chunks = VecDeque::new();
        let mut at = range.start;
        while at < range.end {
            let hi = (at + grain).min(range.end);
            chunks.push_back(at..hi);
            at = hi;
        }

        let _turn = self
            .shared
            .submit
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        {
            let mut q = self.shared.q.lock().unwrap_or_else(|e| e.into_inner());
            debug_assert!(q.job.is_none(), "the submit lock serializes launches");
            q.chunks = chunks;
            // SAFETY: only the lifetime is erased. `parallel_for` blocks below
            // until `job` is cleared, which happens after the last chunk has
            // retired, so no worker can dereference this after `body` dies.
            let erased: *const (dyn Fn(Range<u64>) + Send + Sync + 'static) =
                unsafe { std::mem::transmute(body as *const (dyn Fn(Range<u64>) + Send + Sync)) };
            q.job = Some(JobPtr(erased));
            drop(q);
        }
        self.shared.work.notify_all();

        // The caller is a worker too: it keeps latency low at small grid sizes
        // and makes `threads == 1` a straight-line call.
        IN_POOL.with(|f| f.set(true));
        drain(&self.shared);
        IN_POOL.with(|f| f.set(false));

        let mut q = self.shared.q.lock().unwrap_or_else(|e| e.into_inner());
        while q.job.is_some() {
            q = self
                .shared
                .done
                .wait(q)
                .unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Per-thread workgroup scratch, grown to `bytes` and reused across
    /// launches. The closure form is what keeps the buffer thread-local
    /// without handing out a `'static` alias to it.
    pub fn with_scratch<R>(&self, bytes: usize, f: impl FnOnce(&mut [u8]) -> R) -> R {
        SCRATCH.with(|s| {
            let mut s = s.borrow_mut();
            if s.len() < bytes {
                *s = AlignedBuf::zeroed(bytes.next_power_of_two().max(4096))
                    .expect("workgroup scratch");
            }
            f(&mut s.as_mut_slice()[..bytes])
        })
    }
}

/// Free-function form, as the public API lists it.
pub fn parallel_for(range: Range<u64>, grain: u64, body: &(dyn Fn(Range<u64>) + Send + Sync)) {
    WorkerPool::global().parallel_for(range, grain, body);
}

fn worker(shared: &Shared) {
    IN_POOL.with(|f| f.set(true));
    loop {
        let mut q = shared.q.lock().unwrap_or_else(|e| e.into_inner());
        while !q.shutdown && (q.job.is_none() || q.chunks.is_empty()) {
            q = shared.work.wait(q).unwrap_or_else(|e| e.into_inner());
        }
        if q.shutdown {
            return;
        }
        drop(q);
        drain(shared);
    }
}

/// Pop and run chunks until the queue drains, then release the job.
fn drain(shared: &Shared) {
    loop {
        let (chunk, job) = {
            let mut q = shared.q.lock().unwrap_or_else(|e| e.into_inner());
            let Some(job) = q.job else { return };
            let Some(chunk) = q.chunks.pop_front() else {
                return;
            };
            q.active += 1;
            (chunk, job)
        };
        // SAFETY: `parallel_for` blocks until `active` returns to zero and
        // `chunks` is empty before dropping the closure it pointed at.
        let f = unsafe { &*job.0 };
        f(chunk);
        let mut q = shared.q.lock().unwrap_or_else(|e| e.into_inner());
        q.active -= 1;
        if q.active == 0 && q.chunks.is_empty() {
            q.job = None;
            drop(q);
            shared.done.notify_all();
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn parallel_for_covers_the_range_exactly_once() {
        let hits: Vec<AtomicUsize> = (0..10_000).map(|_| AtomicUsize::new(0)).collect();
        let f = |r: Range<u64>| {
            for i in r {
                hits[i as usize].fetch_add(1, Ordering::Relaxed);
            }
        };
        WorkerPool::global().parallel_for(0..10_000, 64, &f);
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1));
    }

    #[test]
    fn nested_parallel_for_does_not_deadlock() {
        let n = AtomicUsize::new(0);
        let outer = |r: Range<u64>| {
            let inner = |ri: Range<u64>| {
                n.fetch_add((ri.end - ri.start) as usize, Ordering::Relaxed);
            };
            WorkerPool::global().parallel_for(r, 8, &inner);
        };
        WorkerPool::global().parallel_for(0..1024, 32, &outer);
        assert_eq!(n.load(Ordering::Relaxed), 1024);
    }

    #[test]
    fn scratch_is_reused_and_grows() {
        let pool = WorkerPool::global();
        pool.with_scratch(128, |s| {
            assert_eq!(s.len(), 128);
            s[0] = 7;
        });
        pool.with_scratch(4096, |s| assert_eq!(s.len(), 4096));
    }
}
