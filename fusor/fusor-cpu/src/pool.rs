//! One persistent worker pool, created once; parallelism is priced against
//! `DeviceFacts::thread_wake_ps`.

use std::collections::VecDeque;
use std::ops::Range;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// A closure lent to the workers for one `parallel_for`; the caller blocks
/// until every chunk retires, so the pointee outlives every dereference.
#[derive(Copy, Clone)]
struct JobPtr(*const (dyn Fn(Range<u64>) + Send + Sync));

// SAFETY: `Send + Sync` referent, and `parallel_for` joins before returning.
unsafe impl Send for JobPtr {}
// SAFETY: as above.
unsafe impl Sync for JobPtr {}

#[derive(Default)]
struct Queue {
    chunks: VecDeque<Range<u64>>,
    job: Option<JobPtr>,
    /// Chunks popped but not yet finished.
    active: usize,
}

struct Shared {
    q: Mutex<Queue>,
    work: Condvar,
    done: Condvar,
    /// One submission at a time; nested calls take the serial path via `IN_POOL`.
    submit: Mutex<()>,
}

thread_local! {
    /// Re-entrancy guard: a nested `parallel_for` runs serially.
    static IN_POOL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Counts `Level` dispatches, for `level_dispatched_once`.
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
                .name(format!("fusor-cpu-{i}"))
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

        let _turn = self.shared.submit.lock().unwrap_or_else(|e| e.into_inner());
        {
            let mut q = self.shared.q.lock().unwrap_or_else(|e| e.into_inner());
            debug_assert!(q.job.is_none(), "the submit lock serializes launches");
            q.chunks = chunks;
            // SAFETY: only the lifetime is erased; `job` is cleared after the last chunk
            // retires, before `body` dies.
            let erased: *const (dyn Fn(Range<u64>) + Send + Sync + 'static) =
                unsafe { std::mem::transmute(body as *const (dyn Fn(Range<u64>) + Send + Sync)) };
            q.job = Some(JobPtr(erased));
            drop(q);
        }
        self.shared.work.notify_all();

        // The caller works too: low latency at small grids.
        IN_POOL.with(|f| f.set(true));
        drain(&self.shared);
        IN_POOL.with(|f| f.set(false));

        let mut q = self.shared.q.lock().unwrap_or_else(|e| e.into_inner());
        while q.job.is_some() {
            q = self.shared.done.wait(q).unwrap_or_else(|e| e.into_inner());
        }
    }
}

/// Free-function form, as the public API lists it.
fn worker(shared: &Shared) {
    IN_POOL.with(|f| f.set(true));
    loop {
        let mut q = shared.q.lock().unwrap_or_else(|e| e.into_inner());
        while q.job.is_none() || q.chunks.is_empty() {
            q = shared.work.wait(q).unwrap_or_else(|e| e.into_inner());
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
        // SAFETY: `parallel_for` waits for `active == 0` before dropping the closure.
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
