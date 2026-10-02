//! Encoding, submission and telemetry. The only host syncs are readback,
//! [`Target::wait`](fusor_ir::target::Target::wait) and the allocator's cap
//! retry; in-flight back-pressure is [`GpuConfig::max_in_flight_submits`].

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use web_time::{Duration, Instant};

use fusor_ir::Result;
use fusor_ir::error::Error;
use fusor_ir::target::{Artifact, Buf, Uniforms};
use parking_lot::Mutex;

use crate::pool::{BufferPool, GpuBuffer, READBACK_USAGE};
use crate::target::GpuConfig;

/// Past this many dispatches, passes are chunked to [`PASS_CHUNK`] dispatches.
pub const PASS_CHUNK_THRESHOLD: usize = 1024;
/// Dispatches per pass once a plan crosses [`PASS_CHUNK_THRESHOLD`].
pub const PASS_CHUNK: usize = 512;
/// Metal's per-submit dispatch chunk past the threshold.
pub const METAL_SUBMIT_CHUNK: usize = 256;
/// Chunk submits in flight before the encoder waits for the oldest: bounds
/// transients without draining the queue mid-plan.
pub const METAL_INFLIGHT_CHUNKS: usize = 2;
/// `poll_wait` spins in `Poll` mode for this long before blocking.
pub const POLL_SPIN: Duration = Duration::from_millis(2);

pub static CHUNK_WAIT_US: AtomicU64 = AtomicU64::new(0);
pub static POLL_WAIT_US: AtomicU64 = AtomicU64::new(0);

/// Adds the microseconds it was alive to a telemetry counter.
pub(crate) struct Stopwatch(&'static AtomicU64, Instant);

impl Stopwatch {
    pub(crate) fn start(sink: &'static AtomicU64) -> Self {
        Self(sink, Instant::now())
    }
}

impl Drop for Stopwatch {
    fn drop(&mut self) {
        self.0
            .fetch_add(self.1.elapsed().as_micros() as u64, Ordering::Relaxed);
    }
}

/// Pops every entry `keep` rejects; whether any went.
pub(crate) fn lru_retain<K: std::hash::Hash + Eq + Clone, V>(
    cache: &mut lru::LruCache<K, V>,
    keep: impl Fn(&K, &V) -> bool,
) -> bool {
    let dead: Vec<K> = cache
        .iter()
        .filter(|(k, v)| !keep(k, v))
        .map(|(k, _)| k.clone())
        .collect();
    for k in &dead {
        cache.pop(k);
    }
    !dead.is_empty()
}

/// Dispatches packed into one compute pass: chunked past the threshold,
/// never one pass per dispatch (a Metal pass boundary costs a small kernel).
pub fn dispatches_per_pass(total: usize) -> usize {
    if let Some(n) = crate::flags().pass_size {
        return n;
    }
    if total >= PASS_CHUNK_THRESHOLD {
        PASS_CHUNK
    } else {
        usize::MAX
    }
}

/// Dispatches per submit: Metal bounds in-flight memory on giant graphs.
pub fn dispatches_per_submit(total: usize, backend: wgpu::Backend) -> usize {
    if backend == wgpu::Backend::Metal && total >= PASS_CHUNK_THRESHOLD {
        METAL_SUBMIT_CHUNK
    } else {
        usize::MAX
    }
}

/// The completion of one `map_async` as a runtime-free future: the map
/// result, or `Err(())` if wgpu drops the callback uncalled.
#[derive(Clone, Default)]
struct MapDone(Arc<Mutex<MapDoneState>>);

#[derive(Default)]
struct MapDoneState {
    result: Option<std::result::Result<(), wgpu::BufferAsyncError>>,
    waker: Option<std::task::Waker>,
}

impl MapDone {
    fn complete(self, result: std::result::Result<(), wgpu::BufferAsyncError>) {
        let waker = {
            let mut state = self.0.lock();
            state.result = Some(result);
            state.waker.take()
        };
        if let Some(w) = waker {
            w.wake();
        }
    }
}

impl std::future::Future for MapDone {
    type Output = std::result::Result<std::result::Result<(), wgpu::BufferAsyncError>, ()>;
    fn poll(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        let mut state = self.0.lock();
        if let Some(result) = state.result.take() {
            return std::task::Poll::Ready(Ok(result));
        }
        // Only the callback's clone remains: gone uncompleted means rejected.
        if Arc::strong_count(&self.0) == 1 {
            return std::task::Poll::Ready(Err(()));
        }
        state.waker = Some(cx.waker().clone());
        std::task::Poll::Pending
    }
}

/// Under `FUSOR_TRACE_DISPATCH`, every binding's buffer and whether two alias:
/// WARP answers a resource bound as both input and output with device removal.
pub(crate) fn trace_binds(name: &str, grid: [u32; 3], binds: &[Buf]) {
    if !crate::flags().trace_dispatch {
        return;
    }
    let mut seen: Vec<usize> = Vec::new();
    let mut aliased = false;
    let desc: Vec<String> = binds
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let addr = b.addr();
            if seen.contains(&addr) {
                aliased = true;
            }
            seen.push(addr);
            let size = b.downcast_ref::<GpuBuffer>().map_or(0, |g| g.size);
            format!("{i}:{size}B@{addr:x}")
        })
        .collect();
    eprintln!(
        "[trace] binds {name} grid={grid:?} [{}]{}",
        desc.join(" "),
        if aliased { " ALIASED" } else { "" }
    );
}

/// One kernel's aggregated timing across a resolve.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelProfileRow {
    pub name: String,
    pub count: u32,
    pub total_ms: f64,
    pub average_us: f64,
    pub max_us: f64,
}

/// One resolve's timing.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelProfile {
    pub span_ms: f64,
    pub kernels: usize,
    pub top_names: Vec<KernelProfileRow>,
}

impl KernelProfile {
    /// Fold per-dispatch samples into rows, hottest first.
    pub fn from_samples(span_ms: f64, samples: &[(String, f64)]) -> Self {
        let mut rows: Vec<KernelProfileRow> = Vec::new();
        for (name, us) in samples {
            match rows.iter_mut().find(|r| r.name == *name) {
                Some(row) => {
                    row.count += 1;
                    row.total_ms += us / 1000.0;
                    row.max_us = row.max_us.max(*us);
                }
                None => rows.push(KernelProfileRow {
                    name: name.clone(),
                    count: 1,
                    total_ms: us / 1000.0,
                    average_us: 0.0,
                    max_us: *us,
                }),
            }
        }
        for row in &mut rows {
            row.average_us = row.total_ms * 1000.0 / f64::from(row.count.max(1));
        }
        rows.sort_by(|a, b| {
            b.total_ms
                .partial_cmp(&a.total_ms)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.name.cmp(&b.name))
        });
        Self {
            span_ms,
            kernels: samples.len(),
            top_names: rows,
        }
    }
}

/// One recorded command, in exact plan order.
#[derive(Clone)]
pub enum CommandRecord {
    Dispatch {
        name: &'static str,
        pipeline: Arc<wgpu::ComputePipeline>,
        bind_group: Arc<wgpu::BindGroup>,
        grid: [u32; 3],
    },
    /// `bytes` from the start of `src` to the start of `dst`.
    CopyBuffer { src: Buf, dst: Buf, bytes: u64 },
}

impl CommandRecord {
    /// A dispatch whose grid contains a zero launches nothing and is skipped.
    pub fn is_empty_dispatch(&self) -> bool {
        matches!(self, Self::Dispatch { grid, .. } if grid.contains(&0))
    }
}

/// Which dispatches of a traced resolve get timestamp boundary pairs. A
/// query set holds at most [`wgpu::QUERY_SET_MAX_QUERIES`] slots.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TimingMode<'a> {
    /// Every live dispatch owns slot pair `(2i, 2i+1)`.
    All,
    /// Only live dispatch `i` is timed, into slots `(0, 1)`.
    Focus(usize),
    /// The `k`-th named live dispatch (ascending) writes `(2k, 2k+1)`.
    Sparse(&'a [usize]),
    /// Live dispatches `[start, start+n)` own pairs from `(0, 1)` on.
    Range { start: usize, n: usize },
    /// One pair around the whole submission: the plan's GPU span.
    Whole,
}

impl TimingMode<'_> {
    /// Whether live dispatch `ix` must sit alone in its pass so its boundary
    /// pair brackets it alone (without in-pass timestamp writes).
    fn isolates(&self, ix: usize) -> bool {
        match self {
            TimingMode::Focus(f) => *f == ix,
            TimingMode::Sparse(ixs) => ixs.binary_search(&ix).is_ok(),
            _ => false,
        }
    }
}

/// How one resolve is timed: the query set and which dispatches write it.
pub(crate) struct TimingPlan {
    set: Option<wgpu::QuerySet>,
    kind: TimingKind,
    /// `(plan index, live index)` of each focused launch, ascending.
    focus: Vec<(usize, usize)>,
    live: Vec<usize>,
}

enum TimingKind {
    All,
    Focus,
    Whole,
    Range { start: usize, n: usize },
}

impl TimingPlan {
    pub(crate) fn set(&self) -> Option<&wgpu::QuerySet> {
        self.set.as_ref()
    }

    pub(crate) fn mode(&self) -> TimingMode<'_> {
        match self.kind {
            TimingKind::All => TimingMode::All,
            TimingKind::Focus if self.live.len() == 1 => TimingMode::Focus(self.live[0]),
            TimingKind::Focus => TimingMode::Sparse(&self.live),
            TimingKind::Whole => TimingMode::Whole,
            TimingKind::Range { start, n } => TimingMode::Range { start, n },
        }
    }
}

/// A compiled GPU artifact: the pipeline plus its derived binding list.
pub struct GpuArtifact {
    /// Process-unique and never reused: the bind group cache keys on it, and
    /// an address would be recycled with its bind groups.
    pub id: u64,
    pub name: &'static str,
    pub pipeline: Arc<wgpu::ComputePipeline>,
    pub layout: Arc<wgpu::BindGroupLayout>,
    /// `(binding, read_only)` in binding order, from the module's globals.
    pub bindings: Vec<(u32, bool)>,
    pub block: u32,
}

/// Owns the encoder, the in-flight submission window and the profile buffer.
pub struct Launcher {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    backend: wgpu::Backend,
    config: GpuConfig,
    in_flight: AtomicUsize,
    poll_waits: AtomicU64,
    dispatches: AtomicU64,
    pipeline_compiles: AtomicU64,
    profiles: Mutex<Vec<KernelProfile>>,
    /// Set during a tuning pass; turns on the per-launch timestamp path.
    tuning: AtomicBool,
    /// The last traced resolve's per-launch microseconds, in plan order.
    last_profile: Mutex<Option<Vec<f64>>>,
    /// Plan launch indices the next traced resolve times; taken by it.
    tuning_focus: Mutex<Option<Vec<usize>>>,
    /// Bind groups by `(artifact id, bound buffer addresses)`. An address
    /// names a buffer only while it lives, so an entry is served only while
    /// every weak witness beside it is alive.
    bind_groups: Mutex<lru::LruCache<BindGroupKey, BindGroupEntry>>,
    /// Set by the driver's device-lost callback; every poll checks it.
    lost: crate::device::LostFlag,
}

/// What a cached bind group was built from.
#[derive(Clone, PartialEq, Eq, Hash)]
struct BindGroupKey {
    artifact: u64,
    buffers: smallvec::SmallVec<[usize; 8]>,
}

struct BindGroupEntry {
    /// One per key address, in the same order.
    witnesses: smallvec::SmallVec<[fusor_ir::target::WeakBuf; 8]>,
    group: Arc<wgpu::BindGroup>,
}

/// Bind groups retained; above any one plan's launch count, or a plan evicts
/// its own entries every resolve.
const BIND_GROUP_CAPACITY: usize = 16_384;

/// Unmaps (cancelling the map of) a staging buffer if the readback awaiting
/// it is dropped, so the pool can hand it out again.
struct MapGuard<'a> {
    staging: Option<&'a Buf>,
}

impl MapGuard<'_> {
    /// The map resolved; there is nothing to cancel.
    fn disarm(mut self) {
        self.staging = None;
    }
}

impl Drop for MapGuard<'_> {
    fn drop(&mut self) {
        if let Some(staging) = self.staging.take()
            && let Some(gpu) = staging.downcast_ref::<GpuBuffer>()
        {
            gpu.buffer.unmap();
        }
    }
}

impl Launcher {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        backend: wgpu::Backend,
        config: GpuConfig,
        lost: crate::device::LostFlag,
    ) -> Self {
        Self {
            device,
            queue,
            backend,
            config,
            lost,
            in_flight: AtomicUsize::new(0),
            poll_waits: AtomicU64::new(0),
            dispatches: AtomicU64::new(0),
            pipeline_compiles: AtomicU64::new(0),
            profiles: Mutex::new(Vec::new()),
            tuning: AtomicBool::new(false),
            last_profile: Mutex::new(None),
            tuning_focus: Mutex::new(None),
            bind_groups: Mutex::new(lru::LruCache::new(
                std::num::NonZeroUsize::new(BIND_GROUP_CAPACITY).expect("nonzero"),
            )),
        }
    }

    pub fn backend(&self) -> wgpu::Backend {
        self.backend
    }

    pub fn config(&self) -> &GpuConfig {
        &self.config
    }

    /// Dispatches encoded since construction (not submissions).
    pub fn dispatch_count(&self) -> u64 {
        self.dispatches.load(Ordering::Relaxed)
    }

    /// Times the runtime blocked the host.
    pub fn poll_wait_count(&self) -> u64 {
        self.poll_waits.load(Ordering::Relaxed)
    }

    pub fn pipeline_compiles(&self) -> u64 {
        self.pipeline_compiles.load(Ordering::Relaxed)
    }

    pub fn note_pipeline_compile(&self) {
        self.pipeline_compiles.fetch_add(1, Ordering::Relaxed);
    }

    /// Encode and submit one dispatch: the `Target` trait's single-kernel entry.
    pub fn encode(
        &self,
        artifact: &Artifact,
        grid: [u32; 3],
        binds: &[Buf],
        uniforms: &Uniforms,
    ) -> Result<()> {
        let gpu = artifact
            .downcast_ref::<GpuArtifact>()
            .ok_or_else(|| Error::Device("artifact is not a gpu pipeline".into()))?;
        if binds.is_empty() {
            return Err(Error::Device(
                "binding 0 (uniforms) is always present and was not supplied".into(),
            ));
        }
        self.write_uniforms(&binds[0], uniforms)?;
        trace_binds(gpu.name, grid, binds);
        let bind_group = self.bind_group(gpu, binds)?;
        let record = CommandRecord::Dispatch {
            name: gpu.name,
            pipeline: gpu.pipeline.clone(),
            bind_group,
            grid,
        };
        self.encode_command_records(&[record], None, TimingMode::All)
    }

    /// Upload binding 0: runtime scalars live here, outside kernel identity.
    pub fn write_uniforms(&self, slot0: &Buf, uniforms: &Uniforms) -> Result<()> {
        let gpu = GpuBuffer::of(slot0, "binding 0")?;
        let mut bytes = uniforms.to_bytes();
        if bytes.is_empty() {
            bytes.extend_from_slice(&0u32.to_le_bytes());
        }
        while !(bytes.len() as u64).is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT) {
            bytes.push(0);
        }
        if bytes.len() as u64 > gpu.size {
            return Err(Error::Device(format!(
                "uniform block is {} bytes but binding 0 is {}",
                bytes.len(),
                gpu.size
            )));
        }
        crate::pool::UPLOAD_UNIFORM
            .fetch_add(bytes.len() as u64, std::sync::atomic::Ordering::Relaxed);
        self.queue.write_buffer(&gpu.buffer, 0, &bytes);
        Ok(())
    }

    /// The bind group, positional against the derived binding list.
    pub fn bind_group(
        &self,
        artifact: &GpuArtifact,
        binds: &[Buf],
    ) -> Result<Arc<wgpu::BindGroup>> {
        if binds.len() != artifact.bindings.len() {
            return Err(Error::Device(format!(
                "kernel {} wants {} bindings, the caller presented {}",
                artifact.name,
                artifact.bindings.len(),
                binds.len()
            )));
        }
        let key = BindGroupKey {
            artifact: artifact.id,
            buffers: binds.iter().map(Buf::addr).collect(),
        };
        {
            let mut cache = self.bind_groups.lock();
            match cache.get(&key) {
                // Every witness alive: the addresses still name the buffers.
                Some(entry) if entry.witnesses.iter().all(|w| w.alive()) => {
                    return Ok(Arc::clone(&entry.group));
                }
                // A dead witness: an address was reused, drop the entry.
                Some(_) => {
                    cache.pop(&key);
                }
                None => {}
            }
        }
        let mut entries = Vec::with_capacity(binds.len());
        for ((binding, _read_only), buf) in artifact.bindings.iter().zip(binds) {
            let gpu = GpuBuffer::of(buf, "bound value")?;
            entries.push(wgpu::BindGroupEntry {
                binding: *binding,
                resource: gpu.buffer.as_entire_binding(),
            });
        }
        let group = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(artifact.name),
            layout: &artifact.layout,
            entries: &entries,
        }));
        self.bind_groups.lock().put(
            key,
            BindGroupEntry {
                witnesses: binds.iter().map(Buf::downgrade).collect(),
                group: Arc::clone(&group),
            },
        );
        Ok(group)
    }

    /// One encoder per resolve, dispatches packed into few passes. The query
    /// set is resolved later, after [`Self::poll_wait`]: Metal's writeback of
    /// the final boundary samples races a resolve encoded behind it.
    pub fn encode_command_records(
        &self,
        records: &[CommandRecord],
        timestamps: Option<&wgpu::QuerySet>,
        mode: TimingMode,
    ) -> Result<()> {
        // A zero grid launches nothing but would cost a pass boundary.
        let live: Vec<&CommandRecord> = records.iter().filter(|r| !r.is_empty_dispatch()).collect();
        let total = live
            .iter()
            .filter(|r| matches!(r, CommandRecord::Dispatch { .. }))
            .count();
        // `FUSOR_TRACE_DISPATCH`: one dispatch per submit, each waited on, to
        // attribute an asynchronous device loss to a kernel.
        let trace = crate::flags().trace_dispatch;
        let per_submit = if trace {
            1
        } else {
            dispatches_per_submit(total, self.backend)
        };

        let mut dispatch_ix = 0usize;
        let mut chunk: Vec<&CommandRecord> = Vec::new();
        let mut dispatches_in_chunk = 0usize;
        let mut submits = 0usize;
        // Metal: keep at most [`METAL_INFLIGHT_CHUNKS`] chunks outstanding.
        let mut pending: std::collections::VecDeque<wgpu::SubmissionIndex> =
            std::collections::VecDeque::new();

        for record in live {
            let is_dispatch = matches!(record, CommandRecord::Dispatch { .. });
            chunk.push(record);
            if is_dispatch {
                dispatches_in_chunk += 1;
            }
            if dispatches_in_chunk >= per_submit {
                // `Instant` doesn't exist on wasm; tracing is native-only.
                #[cfg(not(target_arch = "wasm32"))]
                let started = trace.then(std::time::Instant::now);
                let (ix, submitted) =
                    self.encode_one_submit(&chunk, timestamps, mode, dispatch_ix, total)?;
                if trace {
                    let poll = self.device.poll(wgpu::PollType::wait_indefinitely());
                    #[cfg(not(target_arch = "wasm32"))]
                    let __us = started.map_or(0, |t| t.elapsed().as_micros());
                    #[cfg(target_arch = "wasm32")]
                    let __us = 0u128;
                    let state = match (
                        &poll,
                        self.lost.reason(),
                        crate::device::removed_reason(&self.device),
                    ) {
                        (_, Some(reason), _) => format!("LOST ({reason})"),
                        // Removed D3D12 fences complete instantly; the removal
                        // reason names the dispatch.
                        (_, None, Some(hr)) => format!("REMOVED ({hr})"),
                        (Err(e), None, None) => format!("poll error: {e}"),
                        (Ok(_), None, None) => "ok".to_string(),
                    };
                    if let CommandRecord::Dispatch { name, grid, .. } = record {
                        eprintln!("[trace] dispatch {name} grid={grid:?} -> {state} {__us}us");
                    }
                }
                dispatch_ix = ix;
                chunk.clear();
                dispatches_in_chunk = 0;
                submits += 1;
                pending.push_back(submitted);
                if pending.len() > METAL_INFLIGHT_CHUNKS
                    && let Some(oldest) = pending.pop_front()
                {
                    let _w = Stopwatch::start(&CHUNK_WAIT_US);
                    self.device
                        .poll(wgpu::PollType::Wait {
                            submission_index: Some(oldest),
                            timeout: None,
                        })
                        .map_err(|e| Error::Device(format!("device wait failed: {e}")))?;
                }
            }
        }
        if !chunk.is_empty() || submits == 0 {
            self.encode_one_submit(&chunk, timestamps, mode, dispatch_ix, total)?;
        }
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        self.apply_back_pressure()
    }

    /// One encoder's passes; returns the next query index and submission.
    fn encode_one_submit(
        &self,
        records: &[&CommandRecord],
        timestamps: Option<&wgpu::QuerySet>,
        mode: TimingMode,
        mut dispatch_ix: usize,
        total: usize,
    ) -> Result<(usize, wgpu::SubmissionIndex)> {
        let inside_passes = self
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
        // The slot pair one live dispatch writes, if any.
        let slots = |ix: usize| -> Option<u32> {
            match mode {
                TimingMode::All => u32::try_from(ix * 2).ok(),
                TimingMode::Focus(f) if ix == f => Some(0),
                TimingMode::Focus(_) => None,
                TimingMode::Sparse(ixs) => ixs
                    .binary_search(&ix)
                    .ok()
                    .and_then(|k| u32::try_from(k * 2).ok()),
                TimingMode::Range { start, n } if ix >= start && ix < start + n => {
                    u32::try_from((ix - start) * 2).ok()
                }
                TimingMode::Range { .. } | TimingMode::Whole => None,
            }
        };
        // Without in-pass writes a timed dispatch needs its own pass; only
        // timed dispatches pay this.
        let per_pass = if timestamps.is_some()
            && !inside_passes
            && matches!(mode, TimingMode::All | TimingMode::Range { .. })
        {
            1
        } else {
            dispatches_per_pass(total)
        };
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Resolver Encoder"),
            });

        // A copy breaks the current pass; a pass closes after `per_pass`.
        let mut at = 0usize;
        while at < records.len() {
            match records[at] {
                CommandRecord::CopyBuffer { src, dst, bytes } => {
                    let s = GpuBuffer::of(src, "copy source")?;
                    let d = GpuBuffer::of(dst, "copy destination")?;
                    encoder.copy_buffer_to_buffer(&s.buffer, 0, &d.buffer, 0, *bytes);
                    at += 1;
                }
                CommandRecord::Dispatch { .. } => {
                    let run_start = at;
                    let mut run_end = at;
                    // Without in-pass writes, cut the run around a timed
                    // dispatch so its boundary pair brackets it alone.
                    let cut_at_focus = timestamps.is_some() && !inside_passes;
                    while run_end < records.len()
                        && matches!(records[run_end], CommandRecord::Dispatch { .. })
                        && run_end - run_start < per_pass
                    {
                        let this = dispatch_ix + (run_end - run_start);
                        if cut_at_focus && mode.isolates(this) && run_end > run_start {
                            break;
                        }
                        run_end += 1;
                        if cut_at_focus && mode.isolates(this) {
                            break;
                        }
                    }
                    // The boundary pair, when this run is the one being timed.
                    let pass_slot = timestamps
                        .filter(|_| !inside_passes)
                        .and_then(|_| slots(dispatch_ix))
                        .filter(|_| {
                            matches!(mode, TimingMode::All | TimingMode::Range { .. })
                                || (mode.isolates(dispatch_ix) && run_end - run_start == 1)
                        });
                    let writes = if matches!(mode, TimingMode::Whole) {
                        timestamps.map(|set| wgpu::ComputePassTimestampWrites {
                            query_set: set,
                            beginning_of_pass_write_index: (dispatch_ix == 0).then_some(0),
                            end_of_pass_write_index: Some(1),
                        })
                    } else {
                        pass_slot.and_then(|q| {
                            timestamps.map(|set| wgpu::ComputePassTimestampWrites {
                                query_set: set,
                                beginning_of_pass_write_index: Some(q),
                                end_of_pass_write_index: Some(q + 1),
                            })
                        })
                    };
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("fusor resolve"),
                        timestamp_writes: writes,
                    });
                    for record in &records[run_start..run_end] {
                        let CommandRecord::Dispatch {
                            name,
                            pipeline,
                            bind_group,
                            grid,
                        } = record
                        else {
                            unreachable!("the run is all dispatches");
                        };
                        pass.push_debug_group(name);
                        let in_pass_slot = timestamps
                            .filter(|_| inside_passes)
                            .and_then(|_| slots(dispatch_ix));
                        if let (Some(set), Some(q)) = (timestamps, in_pass_slot)
                            && inside_passes
                        {
                            pass.write_timestamp(set, q);
                        }
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, bind_group.as_ref(), &[]);
                        pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
                        if let (Some(set), Some(q)) = (timestamps, in_pass_slot)
                            && inside_passes
                        {
                            pass.write_timestamp(set, q + 1);
                        }
                        pass.pop_debug_group();
                        dispatch_ix += 1;
                        self.dispatches.fetch_add(1, Ordering::Relaxed);
                    }
                    drop(pass);
                    at = run_end;
                }
            }
        }
        let submitted = self.queue.submit([encoder.finish()]);
        Ok((dispatch_ix, submitted))
    }

    /// Block only when in-flight submissions exceed the policy window.
    pub fn apply_back_pressure(&self) -> Result<()> {
        if self.in_flight.load(Ordering::Relaxed) > self.config.max_in_flight_submits {
            self.poll_wait()?;
        }
        Ok(())
    }

    /// Cached bind groups currently retained.
    pub fn bind_group_count(&self) -> usize {
        self.bind_groups.lock().len()
    }

    /// Drop cached bind groups whose artifact is not in `live`: they pin
    /// buffers for kernels nobody will dispatch.
    pub fn retain_bind_groups(&self, live: &rustc_hash::FxHashSet<u64>) {
        self.retain_groups(|key| live.contains(&key.artifact));
    }

    pub(crate) fn prune_dead_bind_groups(&self) {
        self.retain_groups(|_| true);
    }

    fn retain_groups(&self, keep: impl Fn(&BindGroupKey) -> bool) {
        lru_retain(&mut self.bind_groups.lock(), |k, e| {
            keep(k) && e.witnesses.iter().all(|w| w.alive())
        });
    }

    /// Resolve once every submission so far has retired, via an awaited
    /// `on_submitted_work_done` (polled to completion natively).
    pub async fn wait_async(&self) -> Result<()> {
        self.lost.check()?;
        let done = MapDone::default();
        let signal = done.clone();
        self.queue
            .on_submitted_work_done(move || signal.complete(Ok(())));
        #[cfg(not(target_arch = "wasm32"))]
        self.poll_wait()?;
        done.await
            .map_err(|_| Error::Device("the submitted-work callback never fired".into()))?
            .map_err(|e| Error::Device(format!("submitted-work callback failed: {e}")))?;
        self.in_flight.store(0, Ordering::Relaxed);
        self.lost.check()
    }

    /// Spin in `Poll` mode for [`POLL_SPIN`], then block. A lost device is
    /// reported by name before and after: wgpu itself only panics.
    pub fn poll_wait(&self) -> Result<()> {
        self.lost.check()?;
        self.poll_wait_inner()?;
        self.lost.check()
    }

    fn poll_wait_inner(&self) -> Result<()> {
        self.poll_waits.fetch_add(1, Ordering::Relaxed);
        let _w = Stopwatch::start(&POLL_WAIT_US);
        let deadline = Instant::now() + POLL_SPIN;
        while Instant::now() < deadline {
            match self.device.poll(wgpu::PollType::Poll) {
                Ok(wgpu::PollStatus::QueueEmpty) => {
                    self.in_flight.store(0, Ordering::Relaxed);
                    return Ok(());
                }
                Ok(_) => std::hint::spin_loop(),
                Err(e) => return Err(Error::Device(format!("device poll failed: {e}"))),
            }
        }
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Device(format!("device wait failed: {e}")))?;
        self.in_flight.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// Copy to a staging buffer, map it and return the bytes; one of the three
    /// host syncs. Async because a web map completes only on the event loop;
    /// natively it is already resolved, so `pollster` costs nothing.
    pub async fn readback(&self, pool: &BufferPool, src: &Buf, bytes: u64) -> Result<Vec<u8>> {
        let bytes = crate::pool::padded_copy_size(bytes);
        let staging = pool.alloc_with_usage(bytes, READBACK_USAGE)?;
        match self.readback_into(src, &staging, bytes).await {
            Ok(out) => {
                pool.recycle(staging);
                Ok(out)
            }
            Err(e) => {
                // The map state is unknown after a failure; the buffer
                // leaves the pool.
                pool.discard(staging);
                Err(e)
            }
        }
    }

    /// Queue a device copy of `bytes` from `src` to `dst`, ordered after
    /// every earlier submission.
    pub fn copy_buffer(&self, src: &Buf, dst: &Buf, bytes: u64) -> Result<()> {
        self.lost.check()?;
        crate::pool::COPY_BYTES.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        self.encode_copy(src, dst, bytes)
    }

    fn encode_copy(&self, src: &Buf, dst: &Buf, bytes: u64) -> Result<()> {
        let record = CommandRecord::CopyBuffer {
            src: src.clone(),
            dst: dst.clone(),
            bytes,
        };
        self.encode_command_records(&[record], None, TimingMode::All)
    }

    /// Copy `src` into `staging`, map it and return the bytes. Nothing of
    /// wgpu's lives across the `await` (a held `BufferSlice` overflows callers'
    /// `Send` checks), so the map is two sync halves around [`MapDone`].
    async fn readback_into(&self, src: &Buf, staging: &Buf, bytes: u64) -> Result<Vec<u8>> {
        let trace = crate::flags().trace_dispatch;
        let state = |what: &str| {
            if trace {
                let src_size = src.downcast_ref::<GpuBuffer>().map_or(0, |b| b.size);
                let lost = self.lost.reason().unwrap_or_else(|| "ok".into());
                eprintln!("[trace] readback {what}: {bytes}B of a {src_size}B buffer -> {lost}");
            }
        };
        self.encode_copy(src, staging, bytes)?;
        if trace {
            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            state("copy");
        }
        let done = self.begin_map(staging, bytes)?;
        // A dropped future must not leave the pooled buffer mapped.
        let pending = MapGuard {
            staging: Some(staging),
        };
        // Natively drives the map to completion; on the web returns at once.
        self.poll_wait()?;
        state("map");
        let mapped = done.await;
        pending.disarm();
        mapped
            .map_err(|_| {
                // wgpu drops the callback unfired when it rejects the map.
                match self.lost.reason() {
                    Some(reason) => Error::Device(format!(
                        "readback map rejected: the wgpu device was lost: {reason}"
                    )),
                    None => Error::Device("readback callback never fired".into()),
                }
            })?
            .map_err(|e| Error::Device(format!("buffer map failed: {e}")))?;
        Self::finish_map(staging, bytes)
    }

    /// Issue the map; the signal completes when the callback runs.
    fn begin_map(&self, staging: &Buf, bytes: u64) -> Result<MapDone> {
        let gpu = GpuBuffer::of(staging, "staging buffer")?;
        let done = MapDone::default();
        let signal = done.clone();
        gpu.buffer
            .slice(..bytes)
            .map_async(wgpu::MapMode::Read, move |r| signal.complete(r));
        Ok(done)
    }

    /// Copy the mapped bytes out and unmap.
    fn finish_map(staging: &Buf, bytes: u64) -> Result<Vec<u8>> {
        let gpu = GpuBuffer::of(staging, "staging buffer")?;
        let out = gpu.buffer.slice(..bytes).get_mapped_range().to_vec();
        gpu.buffer.unmap();
        Ok(out)
    }

    /// The query set for a traced resolve, or `None` (wall-clock fallback)
    /// without the feature, without a request, or past the slot limit.
    pub fn timestamp_query_set(&self, total_kernels: usize) -> Option<wgpu::QuerySet> {
        if !self.profiling()
            || !self
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return None;
        }
        // A write past the set's count is a validation error: don't time.
        let count = u32::try_from(total_kernels.saturating_mul(2)).ok()?;
        if count == 0 || count > wgpu::QUERY_SET_MAX_QUERIES {
            return None;
        }
        Some(self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("fusor kernel timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count,
        }))
    }

    /// Resolve `set` into microseconds per dispatch. Call after `poll_wait`.
    pub fn read_timestamps(
        &self,
        pool: &BufferPool,
        set: &wgpu::QuerySet,
        dispatches: usize,
    ) -> Result<Vec<f64>> {
        let slots = dispatches.saturating_mul(2);
        if slots == 0 {
            return Ok(Vec::new());
        }
        // 8 bytes per query into a 256-aligned destination.
        let bytes = ((slots as u64) * 8).div_ceil(256).max(1) * 256;
        let resolved = pool.alloc_with_usage(
            bytes,
            wgpu::BufferUsages::QUERY_RESOLVE.union(wgpu::BufferUsages::COPY_SRC),
        )?;
        let staging = pool.alloc_with_usage(bytes, READBACK_USAGE)?;
        let dst = GpuBuffer::of(&resolved, "query resolve target")?;
        let host = GpuBuffer::of(&staging, "query staging buffer")?
            .buffer
            .clone();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fusor timestamp resolve"),
            });
        encoder.resolve_query_set(set, 0..slots as u32, &dst.buffer, 0);
        encoder.copy_buffer_to_buffer(&dst.buffer, 0, &host, 0, bytes);
        self.queue.submit([encoder.finish()]);

        let done = self.begin_map(&staging, bytes)?;
        self.poll_wait()?;
        // `poll_wait` drove the map to completion; nothing is left to await.
        let mapped = std::pin::pin!(done)
            .poll(&mut std::task::Context::from_waker(std::task::Waker::noop()));
        match mapped {
            std::task::Poll::Ready(Ok(result)) => {
                result.map_err(|e| Error::Device(format!("timestamp map failed: {e}")))?
            }
            _ => return Err(Error::Device("timestamp callback never fired".into())),
        }
        let raw = Self::finish_map(&staging, bytes)?;
        pool.recycle(staging);
        pool.recycle(resolved);

        // A zero period reads every span as zero, i.e. "not timed".
        let period = f64::from(self.queue.get_timestamp_period());
        let tick = |i: usize| {
            raw.get(i * 8..i * 8 + 8)
                .and_then(|b| <[u8; 8]>::try_from(b).ok())
                .map_or(0u64, u64::from_le_bytes)
        };
        Ok((0..dispatches)
            .map(|d| tick(d * 2 + 1).saturating_sub(tick(d * 2)) as f64 * period / 1000.0)
            .collect())
    }

    /// Choose how a resolve is timed. A named `focus` wins over whole-plan
    /// timing, so a caller gets exactly the launches it will read.
    pub(crate) fn timing_plan(
        &self,
        records: &[CommandRecord],
        focus: Option<Vec<usize>>,
    ) -> TimingPlan {
        // Focus is restated in live-dispatch indices, sorted and deduped for
        // the binary-searched slot map.
        let mut focus = focus.unwrap_or_default();
        focus.sort_unstable();
        focus.dedup();
        let live_before = |ix: usize| {
            records[..ix]
                .iter()
                .filter(|r| !r.is_empty_dispatch())
                .count()
        };
        let focus: Vec<(usize, usize)> = focus
            .into_iter()
            .filter(|&ix| records.get(ix).is_some_and(|r| !r.is_empty_dispatch()))
            .map(|ix| (ix, live_before(ix)))
            .collect();
        let live: Vec<usize> = focus.iter().map(|&(_, l)| l).collect();
        let flags = crate::flags();
        let (set, kind) = if !focus.is_empty() {
            (self.timestamp_query_set(focus.len()), TimingKind::Focus)
        } else if flags.time_plan {
            // `TPLAN <us>`: the plan's whole GPU span.
            self.set_tuning(true);
            (self.timestamp_query_set(1), TimingKind::Whole)
        } else if let Some(start) = flags.time_range {
            // `TSPAN <index> <kernel> <us>` for live dispatches from `start`.
            let live = live_before(records.len());
            let cap = (wgpu::QUERY_SET_MAX_QUERIES as usize / 2).min(live.saturating_sub(start));
            if cap == 0 {
                (None, TimingKind::All)
            } else {
                self.set_tuning(true);
                (
                    self.timestamp_query_set(cap),
                    TimingKind::Range { start, n: cap },
                )
            }
        } else if self.can_time_whole(records.len()) {
            (self.timestamp_query_set(records.len()), TimingKind::All)
        } else {
            (None, TimingKind::All)
        };
        TimingPlan {
            set,
            kind,
            focus,
            live,
        }
    }

    /// Read a timed resolve back: the per-launch profile in plan order, or
    /// the `TPLAN`/`TSPAN` lines.
    pub(crate) fn publish_timing(
        &self,
        pool: &BufferPool,
        timing: &TimingPlan,
        records: &[CommandRecord],
        start: Instant,
    ) -> Result<()> {
        let Some(set) = timing.set() else {
            return Ok(());
        };
        self.poll_wait()?;
        let name = |r: &CommandRecord| match r {
            CommandRecord::Dispatch { name, .. } => *name,
            CommandRecord::CopyBuffer { .. } => "?",
        };
        match timing.kind {
            // Only focused dispatches were timed; other slots read zero.
            TimingKind::Focus => {
                let samples = self.read_timestamps(pool, set, timing.focus.len())?;
                if samples.iter().any(|s| *s > 0.0) {
                    let mut per_launch = vec![0.0; records.len()];
                    for (&(plan_ix, _), us) in timing.focus.iter().zip(samples) {
                        per_launch[plan_ix] = us;
                    }
                    self.set_last_profile(per_launch);
                }
            }
            TimingKind::Whole => {
                let samples = self.read_timestamps(pool, set, 1)?;
                if let Some(us) = samples.first() {
                    eprintln!("TPLAN {us:.1} n={}", records.len());
                }
            }
            // `TSPAN <live index> <kernel> <us> L<plan launch> grid=[x,y,z]`.
            TimingKind::Range { start, n } => {
                let samples = self.read_timestamps(pool, set, n)?;
                let live: Vec<(usize, &CommandRecord)> = records
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| !r.is_empty_dispatch())
                    .collect();
                for (j, us) in samples.iter().enumerate() {
                    let ix = start + j;
                    let (lix, record) = live.get(ix).copied().unzip();
                    let grid = match record {
                        Some(CommandRecord::Dispatch { grid, .. }) => *grid,
                        _ => [0; 3],
                    };
                    eprintln!(
                        "TSPAN {ix} {} {us:.1} L{} grid={grid:?}",
                        record.map_or("?", name),
                        lix.unwrap_or(usize::MAX)
                    );
                }
            }
            TimingKind::All => {
                let live = records.iter().filter(|r| !r.is_empty_dispatch()).count();
                let mut samples = self.read_timestamps(pool, set, live)?.into_iter();
                // Back to plan order: zero-grid launches own no sample.
                let per_launch: Vec<f64> = records
                    .iter()
                    .map(|r| match r.is_empty_dispatch() {
                        true => 0.0,
                        false => samples.next().unwrap_or(0.0),
                    })
                    .collect();
                if self.config.trace_gpu_kernels {
                    let named: Vec<(String, f64)> = records
                        .iter()
                        .zip(&per_launch)
                        .map(|(r, us)| (name(r).to_string(), *us))
                        .collect();
                    self.push_profile(KernelProfile::from_samples(
                        start.elapsed().as_secs_f64() * 1000.0,
                        &named,
                    ));
                }
                // An all-zero read means the device didn't write the slots;
                // the tuner falls back to the wall clock.
                if per_launch.iter().any(|us| *us > 0.0) {
                    self.set_last_profile(per_launch);
                }
            }
        }
        Ok(())
    }

    /// Turn the per-dispatch timestamp path on for a tuning pass.
    pub fn set_tuning(&self, on: bool) {
        self.tuning.store(on, Ordering::Relaxed);
        if !on {
            *self.tuning_focus.lock() = None;
        }
    }

    /// Whether a plan of `dispatches` launches fits a full query set.
    pub fn can_time_whole(&self, dispatches: usize) -> bool {
        u32::try_from(dispatches.saturating_mul(2))
            .is_ok_and(|count| count > 0 && count <= wgpu::QUERY_SET_MAX_QUERIES)
    }

    /// Plan indices (ascending) the next traced resolve times when the plan
    /// is too large to time whole. Take-semantics.
    pub fn set_tuning_focus(&self, launch_ixs: Option<Vec<usize>>) {
        *self.tuning_focus.lock() = launch_ixs;
    }

    pub fn take_tuning_focus(&self) -> Option<Vec<usize>> {
        self.tuning_focus.lock().take()
    }

    /// Whether this resolve must carry timestamps.
    pub fn profiling(&self) -> bool {
        self.config.trace_gpu_kernels || self.tuning.load(Ordering::Relaxed)
    }

    pub fn set_last_profile(&self, per_launch: Vec<f64>) {
        *self.last_profile.lock() = Some(per_launch);
    }

    /// The last traced resolve's per-launch microseconds, in plan order.
    pub fn take_last_profile(&self) -> Option<Vec<f64>> {
        self.last_profile.lock().take()
    }

    pub fn push_profile(&self, profile: KernelProfile) {
        self.profiles.lock().push(profile);
    }

    pub fn take_kernel_profiles(&self) -> Vec<KernelProfile> {
        std::mem::take(&mut *self.profiles.lock())
    }
}

/// A shared cursor a build cohort drains; racing workers can only duplicate
/// work, never see a half-built pipeline.
#[derive(Default)]
pub struct BuildCursor {
    next: AtomicUsize,
}

impl BuildCursor {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn take(&self, len: usize) -> Option<usize> {
        let i = self.next.fetch_add(1, Ordering::Relaxed);
        (i < len).then_some(i)
    }
}

// Explicit auto-trait impls, as on `GpuTarget` (E0275 in `Send` futures).
//
// SAFETY: `launcher_fields_are_send_sync` asserts `Send + Sync` for every
// field type, exactly what the auto impls would require.
unsafe impl Send for Launcher {}
unsafe impl Sync for Launcher {}

#[allow(dead_code)]
fn launcher_fields_are_send_sync() {
    fn assert<T: Send + Sync>() {}
    assert::<Arc<wgpu::Device>>();
    assert::<Arc<wgpu::Queue>>();
    assert::<wgpu::Backend>();
    assert::<GpuConfig>();
    assert::<crate::device::LostFlag>();
    assert::<AtomicUsize>();
    assert::<AtomicU64>();
    assert::<AtomicBool>();
    assert::<Mutex<Vec<KernelProfile>>>();
    assert::<Mutex<Option<Vec<f64>>>>();
    assert::<Mutex<Option<usize>>>();
    assert::<Mutex<lru::LruCache<BindGroupKey, BindGroupEntry>>>();
}

#[cfg(test)]
mod cancel_tests {
    /// `unmap` cancels an unresolved map: the premise of [`MapGuard`], only
    /// observable in a browser through readbacks.
    #[test]
    fn unmap_cancels_a_pending_map() {
        let instance = wgpu::Instance::default();
        let Some(adapter) =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .ok()
        else {
            eprintln!("no adapter; skipping");
            return;
        };
        let Ok((device, _queue)) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
        else {
            eprintln!("no device; skipping");
            return;
        };
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 256,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Map, abandon it as a dropped future does, and map again.
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        buffer.unmap();
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        buffer.unmap();
    }
}
