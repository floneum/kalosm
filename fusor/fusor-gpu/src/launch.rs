//! Encoding, submission and telemetry.
//!
//! Host syncs are exactly three: explicit readback, explicit
//! [`Target::wait`](fusor_ir::target::Target::wait), and the allocator's cap
//! retry. Back-pressure on in-flight submissions is a runtime policy
//! ([`GpuConfig::max_in_flight_submits`]).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use fusor_ir::Result;
use fusor_ir::error::Error;
use fusor_ir::extract::{Plan, PlanHash};
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
/// Chunk submits allowed in flight before the encoder waits for the oldest.
/// Bounds the working set to `METAL_INFLIGHT_CHUNKS * METAL_SUBMIT_CHUNK`
/// dispatches' transients without ever draining the queue mid-plan.
pub const METAL_INFLIGHT_CHUNKS: usize = 2;
/// `poll_wait` spins in `Poll` mode for this long before blocking.
pub const POLL_SPIN: Duration = Duration::from_millis(2);

pub static CHUNK_WAIT_US: AtomicU64 = AtomicU64::new(0);
pub static POLL_WAIT_US: AtomicU64 = AtomicU64::new(0);
struct ScopeGuard<F: FnMut()>(F);
impl<F: FnMut()> Drop for ScopeGuard<F> {
    fn drop(&mut self) {
        (self.0)();
    }
}
fn scopeguard<F: FnMut()>(f: F) -> ScopeGuard<F> {
    ScopeGuard(f)
}

/// Dispatches packed into one compute pass.
///
/// Consecutive dispatches share a pass; past the threshold the pass is
/// chunked, never dropped to a pass per dispatch — a Metal pass boundary
/// costs on the order of a small kernel.
pub const fn dispatches_per_pass(total: usize) -> usize {
    if total >= PASS_CHUNK_THRESHOLD {
        PASS_CHUNK
    } else {
        usize::MAX
    }
}

/// Dispatches per submit. Metal needs the in-flight memory bound on giant
/// training graphs; every other backend submits once.
pub fn dispatches_per_submit(total: usize, backend: wgpu::Backend) -> usize {
    if backend == wgpu::Backend::Metal && total >= PASS_CHUNK_THRESHOLD {
        METAL_SUBMIT_CHUNK
    } else {
        usize::MAX
    }
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
pub enum CommandRecord {
    Dispatch {
        name: &'static str,
        pipeline: Arc<wgpu::ComputePipeline>,
        bind_group: Arc<wgpu::BindGroup>,
        grid: [u32; 3],
    },
    CopyBuffer {
        src: Buf,
        src_offset: u64,
        dst: Buf,
        dst_offset: u64,
        bytes: u64,
    },
}

impl CommandRecord {
    /// A dispatch whose grid contains a zero launches nothing and is skipped.
    pub fn is_empty_dispatch(&self) -> bool {
        matches!(self, Self::Dispatch { grid, .. } if grid.contains(&0))
    }
}

/// Which dispatches of a traced resolve get timestamp boundary pairs.
///
/// A query set holds at most [`wgpu::QUERY_SET_MAX_QUERIES`] slots — 2048
/// dispatch pairs — so a plan that cannot be timed whole is timed at one
/// dispatch: two slots around the live dispatch the tuner asked about.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TimingMode<'a> {
    /// Every live dispatch owns slot pair `(2i, 2i+1)`.
    All,
    /// Only live dispatch `i` is timed, into slots `(0, 1)`.
    Focus(usize),
    /// The live dispatches named (ascending) own slot pairs in list order:
    /// the `k`-th named dispatch writes `(2k, 2k+1)`.
    Sparse(&'a [usize]),
    /// Live dispatches
    /// `[start, start+n)` own slot pairs `(2(i-start), 2(i-start)+1)`, so a
    /// plan too large for a full query set can be timed in two halves.
    Range { start: usize, n: usize },
}

impl TimingMode<'_> {
    /// Whether live dispatch `ix` must sit alone in its pass so its boundary
    /// pair brackets that kernel and nothing else (only meaningful without
    /// in-pass timestamp writes).
    fn isolates(&self, ix: usize) -> bool {
        match self {
            TimingMode::Focus(f) => *f == ix,
            TimingMode::Sparse(ixs) => ixs.binary_search(&ix).is_ok(),
            _ => false,
        }
    }
}

/// A compiled GPU artifact: the pipeline plus its derived binding list.
pub struct GpuArtifact {
    /// Process-unique, minted at construction and never reused. The bind
    /// group cache keys on it: an address would be recycled by the allocator
    /// the moment an artifact is evicted, and the next artifact at that
    /// address would inherit its bind groups.
    pub id: u64,
    pub name: &'static str,
    pub pipeline: Arc<wgpu::ComputePipeline>,
    pub layout: Arc<wgpu::BindGroupLayout>,
    /// `(binding, read_only)` in binding order, derived from the emitted
    /// module's storage globals.
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
    /// Set for the duration of a tuning pass; turns on the per-launch
    /// timestamp path.
    tuning: AtomicBool,
    /// The most recent traced resolve's per-launch microseconds, in plan
    /// order. Overwritten rather than queued.
    last_profile: Mutex<Option<Vec<f64>>>,
    /// The plan launch index the next traced resolve should time when the
    /// plan is too large for a full query set. Take-semantics: consumed by
    /// the next `dispatch_plan`, cleared with the tuning flag.
    tuning_focus: Mutex<Option<Vec<usize>>>,
    /// Bind groups by `(artifact id, bound buffer addresses)`.
    ///
    /// Correctness rests on the [`WeakBuf`]s stored beside the entry: an
    /// address identifies a buffer only while that buffer is alive, so an
    /// entry whose weaks have all survived was built from exactly these
    /// `Buf`s, and one that has lost any is dropped rather than served. Weak
    /// handles do not hold the buffer, so the pool's `strong_count == 1`
    /// recycling is unaffected.
    bind_groups: Mutex<lru::LruCache<BindGroupKey, BindGroupEntry>>,
}

/// What a cached bind group was built from.
#[derive(Clone, PartialEq, Eq, Hash)]
struct BindGroupKey {
    artifact: u64,
    buffers: smallvec::SmallVec<[usize; 8]>,
}

struct BindGroupEntry {
    /// One per key address, in the same order. Alive means the address still
    /// names the buffer the group was built from.
    witnesses: smallvec::SmallVec<[fusor_ir::target::WeakBuf; 8]>,
    group: Arc<wgpu::BindGroup>,
}

/// Bind groups retained. Sized above any one plan's launch count for the same
/// reason [`crate::target::ARTIFACT_CAPACITY`] is: a plan larger than the
/// cache evicts its own entries every resolve and never hits.
const BIND_GROUP_CAPACITY: usize = 16_384;

impl Launcher {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        backend: wgpu::Backend,
        config: GpuConfig,
    ) -> Self {
        Self {
            device,
            queue,
            backend,
            config,
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

    /// Dispatches encoded since construction. `Session::launch_count` reads
    /// this, so it counts *dispatches*, never encoder submissions.
    pub fn dispatch_count(&self) -> u64 {
        self.dispatches.load(Ordering::Relaxed)
    }

    /// Times the runtime blocked the host. A training step with no readback
    /// must leave this at zero below the in-flight threshold.
    pub fn poll_wait_count(&self) -> u64 {
        self.poll_waits.load(Ordering::Relaxed)
    }

    pub fn pipeline_compiles(&self) -> u64 {
        self.pipeline_compiles.load(Ordering::Relaxed)
    }

    pub fn note_pipeline_compile(&self) {
        self.pipeline_compiles.fetch_add(1, Ordering::Relaxed);
    }

    /// Encode and submit one dispatch. The whole-plan path is
    /// [`Self::encode_command_records`]; this is the single-kernel entry the
    /// `Target` trait exposes.
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
        let bind_group = self.bind_group(gpu, binds)?;
        let record = CommandRecord::Dispatch {
            name: gpu.name,
            pipeline: gpu.pipeline.clone(),
            bind_group,
            grid,
        };
        self.encode_command_records(&[record], None, TimingMode::All)
    }

    /// Upload binding 0. Scalars like the learning rate and sequence length
    /// are uniform words here, so they never enter a kernel's identity.
    pub fn write_uniforms(&self, slot0: &Buf, uniforms: &Uniforms) -> Result<()> {
        let gpu = slot0
            .downcast_ref::<GpuBuffer>()
            .ok_or_else(|| Error::Device("binding 0 is not a pooled buffer".into()))?;
        let mut bytes = uniforms.to_bytes();
        if bytes.is_empty() {
            bytes.extend_from_slice(&0u32.to_le_bytes());
        }
        while bytes.len() as u64 % wgpu::COPY_BUFFER_ALIGNMENT != 0 {
            bytes.push(0);
        }
        if bytes.len() as u64 > gpu.size {
            return Err(Error::Device(format!(
                "uniform block is {} bytes but binding 0 is {}",
                bytes.len(),
                gpu.size
            )));
        }
        self.queue.write_buffer(&gpu.buffer, 0, &bytes);
        Ok(())
    }

    /// Build the one bind group. Entries are positional against the derived
    /// binding list, so binding order and codegen cannot drift.
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
                // Every witness alive means every address still names the
                // buffer this group was built from.
                Some(entry) if entry.witnesses.iter().all(|w| w.alive()) => {
                    return Ok(Arc::clone(&entry.group));
                }
                // A dead witness means an address was reused: drop the entry
                // rather than serve a group over a buffer that no longer
                // exists.
                Some(_) => {
                    cache.pop(&key);
                }
                None => {}
            }
        }
        let mut entries = Vec::with_capacity(binds.len());
        for ((binding, _read_only), buf) in artifact.bindings.iter().zip(binds) {
            let gpu = buf
                .downcast_ref::<GpuBuffer>()
                .ok_or_else(|| Error::Device("bound value is not a pooled buffer".into()))?;
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

    /// One `wgpu::CommandEncoder` per resolve, consecutive dispatches packed
    /// into as few compute passes as the policy allows.
    ///
    /// When `timestamps` is present, every dispatch's boundary samples are
    /// written into it; the *resolve* of that query set is deliberately
    /// submitted after a [`Self::poll_wait`], because Metal's writeback of the
    /// final encoder's boundary samples races a resolve encoded behind it.
    pub fn encode_command_records(
        &self,
        records: &[CommandRecord],
        timestamps: Option<&wgpu::QuerySet>,
        mode: TimingMode,
    ) -> Result<()> {
        // A dispatch whose grid contains a zero launches nothing and still
        // costs a pass boundary, so it never reaches the encoder.
        let live: Vec<&CommandRecord> = records.iter().filter(|r| !r.is_empty_dispatch()).collect();
        let total = live
            .iter()
            .filter(|r| matches!(r, CommandRecord::Dispatch { .. }))
            .count();
        let per_submit = dispatches_per_submit(total, self.backend);

        let mut dispatch_ix = 0usize;
        let mut chunk: Vec<&CommandRecord> = Vec::new();
        let mut dispatches_in_chunk = 0usize;
        let mut submits = 0usize;
        // Metal only: a sliding window over submission indices keeps at most
        // [`METAL_INFLIGHT_CHUNKS`] chunks outstanding while the host keeps
        // encoding.
        let mut pending: std::collections::VecDeque<wgpu::SubmissionIndex> =
            std::collections::VecDeque::new();

        for record in live {
            let is_dispatch = matches!(record, CommandRecord::Dispatch { .. });
            chunk.push(record);
            if is_dispatch {
                dispatches_in_chunk += 1;
            }
            if dispatches_in_chunk >= per_submit {
                let (ix, submitted) =
                    self.encode_one_submit(&chunk, timestamps, mode, dispatch_ix, total)?;
                dispatch_ix = ix;
                chunk.clear();
                dispatches_in_chunk = 0;
                submits += 1;
                pending.push_back(submitted);
                if pending.len() > METAL_INFLIGHT_CHUNKS
                    && let Some(oldest) = pending.pop_front()
                {
                    let __w = Instant::now();
                    self.device
                        .poll(wgpu::PollType::Wait {
                            submission_index: Some(oldest),
                            timeout: None,
                        })
                        .map_err(|e| Error::Device(format!("device wait failed: {e}")))?;
                    CHUNK_WAIT_US.fetch_add(__w.elapsed().as_micros() as u64, Ordering::Relaxed);
                }
            }
        }
        if !chunk.is_empty() || submits == 0 {
            self.encode_one_submit(&chunk, timestamps, mode, dispatch_ix, total)?;
        }
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        self.apply_back_pressure()
    }

    /// One `wgpu::CommandEncoder`, consecutive dispatches packed into as few
    /// compute passes as [`dispatches_per_pass`] allows. Returns the next
    /// timestamp query index and the submission's index.
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
                TimingMode::Range { .. } => None,
            }
        };
        // A pass writes exactly one boundary pair, so without in-pass writes
        // a timed dispatch needs its own pass. Only the dispatches actually
        // being timed pay this: under `Focus` every other dispatch batches as
        // if untraced.
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

        // Split into runs: a copy breaks the current pass, and a pass closes
        // after `per_pass` dispatches.
        let mut at = 0usize;
        while at < records.len() {
            match records[at] {
                CommandRecord::CopyBuffer {
                    src,
                    src_offset,
                    dst,
                    dst_offset,
                    bytes,
                } => {
                    let s = src
                        .downcast_ref::<GpuBuffer>()
                        .ok_or_else(|| Error::Device("copy source is not pooled".into()))?;
                    let d = dst
                        .downcast_ref::<GpuBuffer>()
                        .ok_or_else(|| Error::Device("copy destination is not pooled".into()))?;
                    encoder.copy_buffer_to_buffer(
                        &s.buffer,
                        *src_offset,
                        &d.buffer,
                        *dst_offset,
                        *bytes,
                    );
                    at += 1;
                }
                CommandRecord::Dispatch { .. } => {
                    let run_start = at;
                    let mut run_end = at;
                    // Under `Focus`/`Sparse` without in-pass writes a timed
                    // dispatch must sit alone in its pass, so its boundary
                    // pair brackets that kernel and nothing else: the run is
                    // cut just before it and closed right after it.
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
                    // The pass boundary pair, when this run is the one being
                    // timed: under `All` with per_pass == 1 the run is a
                    // single dispatch; under `Focus`/`Sparse` only a run
                    // holding a timed dispatch (alone, by the cut above)
                    // writes.
                    let pass_slot = timestamps
                        .filter(|_| !inside_passes)
                        .and_then(|_| slots(dispatch_ix))
                        .filter(|_| {
                            matches!(mode, TimingMode::All | TimingMode::Range { .. })
                                || (mode.isolates(dispatch_ix) && run_end - run_start == 1)
                        });
                    let writes = pass_slot.and_then(|q| {
                        timestamps.map(|set| wgpu::ComputePassTimestampWrites {
                            query_set: set,
                            beginning_of_pass_write_index: Some(q),
                            end_of_pass_write_index: Some(q + 1),
                        })
                    });
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

    /// Block only when the in-flight submission count exceeds the library's
    /// policy. A step that reads nothing back and stays under the window never
    /// reaches [`Self::poll_wait`].
    pub fn apply_back_pressure(&self) -> Result<()> {
        if self.in_flight.load(Ordering::Relaxed) > self.config.max_in_flight_submits {
            self.poll_wait()?;
        }
        Ok(())
    }

    /// Spin in `Poll` mode for [`POLL_SPIN`], then block.
    pub fn poll_wait(&self) -> Result<()> {
        self.poll_waits.fetch_add(1, Ordering::Relaxed);
        let __w = Instant::now();
        let _g = scopeguard(move || {
            POLL_WAIT_US.fetch_add(__w.elapsed().as_micros() as u64, Ordering::Relaxed);
        });
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

    /// Copy a device buffer into a `COPY_DST | MAP_READ` staging buffer, map
    /// it, and return the bytes. **This is one of the three host syncs.**
    pub fn readback(&self, pool: &BufferPool, src: &Buf, bytes: u64) -> Result<Vec<u8>> {
        let bytes = crate::pool::padded_copy_size(bytes);
        let staging = pool.alloc_with_usage(bytes, READBACK_USAGE)?;
        {
            let record = CommandRecord::CopyBuffer {
                src: src.clone(),
                src_offset: 0,
                dst: staging.clone(),
                dst_offset: 0,
                bytes,
            };
            self.encode_command_records(&[record], None, TimingMode::All)?;
        }
        let gpu = staging
            .downcast_ref::<GpuBuffer>()
            .ok_or_else(|| Error::Device("staging buffer is not pooled".into()))?;
        let slice = gpu.buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.poll_wait()?;
        rx.recv()
            .map_err(|_| Error::Device("readback callback never fired".into()))?
            .map_err(|e| Error::Device(format!("buffer map failed: {e}")))?;
        let out = slice.get_mapped_range().to_vec();
        gpu.buffer.unmap();
        pool.recycle(staging);
        Ok(out)
    }

    /// Allocate the query set for a traced resolve.
    ///
    /// `None` — no timestamps, and the caller falls back to a wall clock — when
    /// the feature is absent, when nothing asked to be traced, or when the plan
    /// is too big to give every dispatch its own slot pair.
    pub fn timestamp_query_set(&self, total_kernels: usize) -> Option<wgpu::QuerySet> {
        if !self.profiling()
            || !self
                .device
                .features()
                .contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return None;
        }
        // Two slots per dispatch and every slot must exist: a write past the
        // set's count is a validation error, so a plan too big to time is not
        // timed at all rather than timed wrongly.
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

    /// Resolve `set` and return one microsecond figure per dispatch, in encoded
    /// order. Call only *after* [`Self::poll_wait`].
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
        // `resolve_query_set` writes 8 bytes per query into a 256-aligned
        // destination.
        let bytes = ((slots as u64) * 8).div_ceil(256).max(1) * 256;
        let resolved = pool.alloc_with_usage(
            bytes,
            wgpu::BufferUsages::QUERY_RESOLVE.union(wgpu::BufferUsages::COPY_SRC),
        )?;
        let staging = pool.alloc_with_usage(bytes, READBACK_USAGE)?;
        let dst = resolved
            .downcast_ref::<GpuBuffer>()
            .ok_or_else(|| Error::Device("query resolve target is not pooled".into()))?;
        let host = staging
            .downcast_ref::<GpuBuffer>()
            .ok_or_else(|| Error::Device("query staging buffer is not pooled".into()))?;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fusor timestamp resolve"),
            });
        encoder.resolve_query_set(set, 0..slots as u32, &dst.buffer, 0);
        encoder.copy_buffer_to_buffer(&dst.buffer, 0, &host.buffer, 0, bytes);
        self.queue.submit([encoder.finish()]);

        let slice = host.buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.poll_wait()?;
        rx.recv()
            .map_err(|_| Error::Device("timestamp callback never fired".into()))?
            .map_err(|e| Error::Device(format!("timestamp map failed: {e}")))?;
        let raw = slice.get_mapped_range().to_vec();
        host.buffer.unmap();
        pool.recycle(staging);
        pool.recycle(resolved);

        // Nanoseconds per tick. A device that reports zero has no usable clock,
        // and every span below then reads as zero — which the caller treats as
        // "not timed", never as "took no time".
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

    /// Turn the per-dispatch timestamp path on for a tuning pass.
    pub fn set_tuning(&self, on: bool) {
        self.tuning.store(on, Ordering::Relaxed);
        if !on {
            *self.tuning_focus.lock() = None;
        }
    }

    /// Whether a plan of `dispatches` launches can carry a full per-dispatch
    /// query set. Past this, only [`TimingMode::Focus`] can time anything.
    pub fn can_time_whole(&self, dispatches: usize) -> bool {
        u32::try_from(dispatches.saturating_mul(2))
            .is_ok_and(|count| count > 0 && count <= wgpu::QUERY_SET_MAX_QUERIES)
    }

    /// Ask the next traced resolve to time the launches at these **plan
    /// indices** (ascending) when the plan is too large to time whole.
    /// Take-semantics. One index is the classic focused launch; several is a
    /// restructuring candidate's changed window, timed together.
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

/// A shared cursor a build cohort drains. Every compiled artifact lives behind
/// a `OnceLock` on the cached kernel, so racing workers can only duplicate
/// work, never observe a half-built pipeline.
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

/// The plan's cache key: the plan is the key.
pub const fn plan_key(plan: &Plan) -> PlanHash {
    plan.hash
}
