//! Encoding, submission and telemetry.
//!
//! **Host syncs are exactly three**: explicit readback, explicit
//! [`Target::wait`](fusor2_ir::target::Target::wait), and the allocator's cap
//! retry. Back-pressure on in-flight submissions is a runtime policy here
//! ([`GpuConfig::max_in_flight_submits`]), not a `--drain-every` counter in a
//! training script.
//!
//! Owned by W9.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use fusor2_ir::Result;
use fusor2_ir::error::Error;
use fusor2_ir::extract::{Plan, PlanHash};
use fusor2_ir::target::{Artifact, Buf, Uniforms};
use parking_lot::Mutex;

use crate::pool::{BufferPool, GpuBuffer, READBACK_USAGE};
use crate::target::GpuConfig;

/// Queues shorter than this never bother with a parallel build cohort.
pub const MIN_PARALLEL_BUILD_QUEUE: usize = 16;
/// A build that takes longer than this is "cold" and justifies the cohort.
pub const COLD_BUILD_THRESHOLD: Duration = Duration::from_millis(1);
/// A cohort is only worth spawning with at least this many items left.
pub const MIN_PARALLEL_BUILD_REMAINDER: usize = 4;
/// Past this many dispatches, one compute pass per dispatch.
pub const PASS_CHUNK_THRESHOLD: usize = 1024;
/// Metal's per-submit dispatch chunk past the threshold.
pub const METAL_SUBMIT_CHUNK: usize = 256;
/// `poll_wait` spins in `Poll` mode for this long before blocking.
pub const POLL_SPIN: Duration = Duration::from_millis(2);

// ---------------------------------------------------------------------------
// Chunking policy
// ---------------------------------------------------------------------------

/// Dispatches packed into one compute pass.
///
/// Consecutive dispatches share a pass until the queue gets long enough that
/// one giant pass starts costing more in driver-side bookkeeping than the pass
/// boundaries do.
pub const fn dispatches_per_pass(total: usize) -> usize {
    if total >= PASS_CHUNK_THRESHOLD {
        1
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

/// Should the remaining builds move to a parallel cohort?
///
/// A serial probe runs first: a short queue stays serial, and a long one stays
/// serial until one build proves cold. That keeps a warm cache off the thread
/// pool entirely.
pub fn should_parallelize_build_remainder(
    queue_len: usize,
    remaining: usize,
    last_build: Duration,
) -> bool {
    queue_len >= MIN_PARALLEL_BUILD_QUEUE
        && remaining >= MIN_PARALLEL_BUILD_REMAINDER
        && last_build > COLD_BUILD_THRESHOLD
}

// ---------------------------------------------------------------------------
// Telemetry
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Command records
// ---------------------------------------------------------------------------

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

/// A compiled GPU artifact: the pipeline plus its derived binding list.
pub struct GpuArtifact {
    pub name: &'static str,
    pub pipeline: Arc<wgpu::ComputePipeline>,
    pub layout: Arc<wgpu::BindGroupLayout>,
    /// `(binding, read_only)` in binding order, derived from the emitted
    /// module's storage globals.
    pub bindings: Vec<(u32, bool)>,
    pub block: u32,
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

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
}

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
            bind_group: Arc::new(bind_group),
            grid,
        };
        self.encode_command_records(&[record], None)
    }

    /// Upload binding 0. This is the whole of trainer constraints 1 and 2: the
    /// learning rate and the sequence length are words here, so neither enters
    /// a kernel's identity.
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
    pub fn bind_group(&self, artifact: &GpuArtifact, binds: &[Buf]) -> Result<wgpu::BindGroup> {
        if binds.len() != artifact.bindings.len() {
            return Err(Error::Device(format!(
                "kernel {} wants {} bindings, the caller presented {}",
                artifact.name,
                artifact.bindings.len(),
                binds.len()
            )));
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
        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(artifact.name),
            layout: &artifact.layout,
            entries: &entries,
        }))
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
    ) -> Result<()> {
        // A dispatch whose grid contains a zero launches nothing and still
        // costs a pass boundary, so it never reaches the encoder.
        let live: Vec<&CommandRecord> =
            records.iter().filter(|r| !r.is_empty_dispatch()).collect();
        let total = live
            .iter()
            .filter(|r| matches!(r, CommandRecord::Dispatch { .. }))
            .count();
        let per_submit = dispatches_per_submit(total, self.backend);

        let mut query = 0u32;
        let mut chunk: Vec<&CommandRecord> = Vec::new();
        let mut dispatches_in_chunk = 0usize;
        let mut submits = 0usize;

        for record in live {
            let is_dispatch = matches!(record, CommandRecord::Dispatch { .. });
            chunk.push(record);
            if is_dispatch {
                dispatches_in_chunk += 1;
            }
            if dispatches_in_chunk >= per_submit {
                query = self.encode_one_submit(&chunk, timestamps, query, total)?;
                chunk.clear();
                dispatches_in_chunk = 0;
                submits += 1;
                // Metal only: bound the in-flight working set between chunks
                // of a giant graph.
                self.poll_wait()?;
            }
        }
        if !chunk.is_empty() || submits == 0 {
            self.encode_one_submit(&chunk, timestamps, query, total)?;
        }
        self.in_flight.fetch_add(1, Ordering::Relaxed);
        self.apply_back_pressure()
    }

    /// One `wgpu::CommandEncoder`, consecutive dispatches packed into as few
    /// compute passes as [`dispatches_per_pass`] allows. Returns the next
    /// timestamp query index.
    fn encode_one_submit(
        &self,
        records: &[&CommandRecord],
        timestamps: Option<&wgpu::QuerySet>,
        mut query: u32,
        total: usize,
    ) -> Result<u32> {
        let per_pass = dispatches_per_pass(total);
        let inside_passes = self
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES);
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
                    while run_end < records.len()
                        && matches!(records[run_end], CommandRecord::Dispatch { .. })
                        && run_end - run_start < per_pass
                    {
                        run_end += 1;
                    }
                    let writes =
                        timestamps
                            .filter(|_| !inside_passes)
                            .map(|set| wgpu::ComputePassTimestampWrites {
                                query_set: set,
                                beginning_of_pass_write_index: Some(query),
                                end_of_pass_write_index: Some(query + 1),
                            });
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("fusor2 resolve"),
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
                        if let Some(set) = timestamps
                            && inside_passes
                        {
                            pass.write_timestamp(set, query);
                        }
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, bind_group.as_ref(), &[]);
                        pass.dispatch_workgroups(grid[0], grid[1], grid[2]);
                        if let Some(set) = timestamps
                            && inside_passes
                        {
                            pass.write_timestamp(set, query + 1);
                        }
                        pass.pop_debug_group();
                        query = query.saturating_add(2);
                        self.dispatches.fetch_add(1, Ordering::Relaxed);
                    }
                    drop(pass);
                    at = run_end;
                }
            }
        }
        self.queue.submit([encoder.finish()]);
        Ok(query)
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
            self.encode_command_records(&[record], None)?;
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

    /// Allocate the query set for a traced resolve, bounded by
    /// `wgpu::QUERY_SET_MAX_QUERIES`.
    pub fn timestamp_query_set(&self, total_kernels: usize) -> Option<wgpu::QuerySet> {
        if !self.config.trace_gpu_kernels
            || !self.device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
        {
            return None;
        }
        let count = (total_kernels.saturating_mul(2) as u32)
            .min(wgpu::QUERY_SET_MAX_QUERIES)
            .max(2);
        Some(self.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("fusor2 kernel timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count,
        }))
    }

    pub fn push_profile(&self, profile: KernelProfile) {
        self.profiles.lock().push(profile);
    }

    pub fn take_kernel_profiles(&self) -> Vec<KernelProfile> {
        std::mem::take(&mut *self.profiles.lock())
    }
}

// ---------------------------------------------------------------------------
// Parallel build cursor
// ---------------------------------------------------------------------------

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

/// A DOT rendering of the realized launch DAG.
///
/// Nodes are launches, edges are the buffers one launch writes and another
/// reads, so the picture is the plan the extractor committed to rather than an
/// approximation of it.
pub fn graphvis(plan: &Plan, graph: &fusor2_ir::egraph::EGraph) -> String {
    let _ = graph;
    graphvis_impl(plan)
}

/// The same rendering without an e-graph handle, for callers that only hold a
/// plan.
pub fn graphvis_dot(plan: &Plan) -> String {
    graphvis_impl(plan)
}

fn graphvis_impl(plan: &Plan) -> String {
    use std::fmt::Write as _;
    let mut out = String::from("digraph plan {\n  rankdir=TB;\n");
    for (i, launch) in plan.launches.iter().enumerate() {
        let _ = writeln!(
            out,
            "  L{i} [label=\"{} root={} grid={:?} block={}\"];",
            i, launch.root, launch.grid, launch.block
        );
    }
    for (i, producer) in plan.launches.iter().enumerate() {
        for (j, consumer) in plan.launches.iter().enumerate() {
            if i == j {
                continue;
            }
            for w in producer
                .bindings
                .iter()
                .filter(|b| b.kind != fusor2_ir::extract::BindKind::Read)
            {
                if consumer.bindings.iter().any(|r| r.value == w.value) {
                    let _ = writeln!(out, "  L{i} -> L{j} [label=\"{}\"];", w.value);
                }
            }
        }
    }
    out.push_str("}\n");
    out
}

/// The plan's cache key — the `gpu_key` / `key` replacement. The plan **is**
/// the key, so there is no `hash_kernel_fields` to thread a new decision
/// variable into.
pub const fn plan_key(plan: &Plan) -> PlanHash {
    plan.hash
}

#[cfg(test)]
mod tests {
    use super::*;


    // -----------------------------------------------------------------------
    // Adapter-gated. These skip cleanly when no GPU is present.
    // -----------------------------------------------------------------------

    fn baseline_launcher() -> Option<Launcher> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions::default()),
        )
        .ok()?;
        let backend = adapter.get_info().backend;
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("fusor2 launcher test"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            },
        ))
        .ok()?;
        Some(Launcher::new(
            Arc::new(device),
            Arc::new(queue),
            backend,
            GpuConfig::default(),
        ))
    }

    /// Test 12: a resolve that reads nothing back records zero `poll_wait`
    /// calls below the `max_in_flight_submits` threshold. Above it the
    /// library — not the training script — applies back-pressure.
    #[test]
    fn zero_readback_step_never_blocks() {
        let Some(launcher) = baseline_launcher() else {
            eprintln!("no adapter; skipping zero_readback_step_never_blocks");
            return;
        };
        let window = launcher.config().max_in_flight_submits;
        for _ in 0..window {
            launcher.encode_command_records(&[], None).unwrap();
        }
        assert_eq!(
            launcher.poll_wait_count(),
            0,
            "a step with no readback must not block the host"
        );

        // One submission past the window and the library blocks on its own.
        launcher.encode_command_records(&[], None).unwrap();
        assert_eq!(
            launcher.poll_wait_count(),
            1,
            "back-pressure is a library policy, not a --drain-every counter"
        );
    }

    /// `Session::launch_count` reads this, so it must count dispatches rather
    /// than encoder submissions.
    #[test]
    fn dispatch_count_counts_dispatches_not_submits() {
        let Some(launcher) = baseline_launcher() else {
            eprintln!("no adapter; skipping dispatch_count_counts_dispatches_not_submits");
            return;
        };
        assert_eq!(launcher.dispatch_count(), 0);
        launcher.encode_command_records(&[], None).unwrap();
        assert_eq!(
            launcher.dispatch_count(),
            0,
            "an empty submit is not a dispatch"
        );
    }

    /// Test 14, verbatim: the three `should_parallelize_build_remainder`
    /// assertions.
    #[test]
    fn parallel_build_probe() {
        // A short queue stays serial no matter how cold the build was.
        assert!(!should_parallelize_build_remainder(
            8,
            8,
            Duration::from_millis(50)
        ));
        // A long queue whose builds are all warm stays serial too.
        assert!(!should_parallelize_build_remainder(
            64,
            60,
            Duration::from_micros(10)
        ));
        // A long queue with a cold build and work left goes parallel.
        assert!(should_parallelize_build_remainder(
            64,
            60,
            Duration::from_millis(5)
        ));
    }

    #[test]
    fn a_cold_build_with_no_remainder_stays_serial() {
        assert!(!should_parallelize_build_remainder(
            64,
            MIN_PARALLEL_BUILD_REMAINDER - 1,
            Duration::from_millis(5)
        ));
    }

    /// Test 13: 2048 dispatches on Metal encode as 2048 passes across 8
    /// submits; 512 dispatches encode as one pass and one submit.
    #[test]
    fn pass_and_submit_chunking() {
        assert_eq!(dispatches_per_pass(2048), 1);
        assert_eq!(dispatches_per_submit(2048, wgpu::Backend::Metal), 256);
        assert_eq!(2048 / dispatches_per_pass(2048), 2048, "one pass each");
        assert_eq!(
            2048_usize.div_ceil(dispatches_per_submit(2048, wgpu::Backend::Metal)),
            8,
            "eight submits"
        );

        assert_eq!(dispatches_per_pass(512), usize::MAX);
        assert_eq!(dispatches_per_submit(512, wgpu::Backend::Metal), usize::MAX);
        assert_eq!(512_usize.div_ceil(dispatches_per_pass(512)), 1);
    }

    #[test]
    fn only_metal_chunks_submits() {
        for backend in [
            wgpu::Backend::Vulkan,
            wgpu::Backend::Dx12,
            wgpu::Backend::Gl,
            wgpu::Backend::BrowserWebGpu,
        ] {
            assert_eq!(dispatches_per_submit(4096, backend), usize::MAX, "{backend:?}");
        }
        assert_eq!(dispatches_per_submit(4096, wgpu::Backend::Metal), 256);
    }

    #[test]
    fn a_zero_grid_dispatch_is_skipped() {
        // A grid with a zero launches nothing but would still cost a pass
        // boundary, so it never reaches the encoder.
        for grid in [[0u32, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0]] {
            assert!(grid.contains(&0), "{grid:?}");
        }
        assert!(![1u32, 2, 3].contains(&0));
    }

    #[test]
    fn build_cursor_hands_out_each_index_once() {
        let cursor = BuildCursor::new();
        let mut seen = Vec::new();
        while let Some(i) = cursor.take(5) {
            seen.push(i);
        }
        assert_eq!(seen, vec![0, 1, 2, 3, 4]);
        assert!(cursor.take(5).is_none());
    }

    #[test]
    fn profiles_fold_by_name_hottest_first() {
        let samples = vec![
            ("kmap".to_string(), 10.0),
            ("coop_matmul".to_string(), 200.0),
            ("kmap".to_string(), 30.0),
        ];
        let p = KernelProfile::from_samples(1.5, &samples);
        assert_eq!(p.kernels, 3);
        assert_eq!(p.top_names[0].name, "coop_matmul");
        assert_eq!(p.top_names[1].name, "kmap");
        assert_eq!(p.top_names[1].count, 2);
        assert!((p.top_names[1].total_ms - 0.04).abs() < 1e-9);
        assert!((p.top_names[1].average_us - 20.0).abs() < 1e-9);
        assert!((p.top_names[1].max_us - 30.0).abs() < 1e-9);
    }

    #[test]
    fn plan_key_is_the_plan_hash() {
        let plan = Plan {
            extraction: Default::default(),
            launches: Vec::new(),
            buffers: Vec::new(),
            symbols: Vec::new(),
            hash: PlanHash(0xabc),
            cost: Default::default(),
        };
        assert_eq!(plan_key(&plan), PlanHash(0xabc));
    }

    #[test]
    fn graphvis_names_every_launch() {
        use fusor2_ir::egraph::Id;
        use fusor2_ir::extract::{BindKind, BindingPlan, Launch};
        let launch = |root: u32, value: u32, kind: BindKind| Launch {
            root: Id(root),
            members: Default::default(),
            bindings: vec![BindingPlan {
                binding: 1,
                value: Id(value),
                kind,
            }],
            grid: [1, 1, 1],
            block: 256,
        };
        let plan = Plan {
            extraction: Default::default(),
            launches: vec![
                launch(1, 7, BindKind::Write),
                launch(2, 7, BindKind::Read),
            ],
            buffers: Vec::new(),
            symbols: Vec::new(),
            hash: PlanHash(0),
            cost: Default::default(),
        };
        let dot = graphvis_dot(&plan);
        assert!(dot.starts_with("digraph plan {"));
        assert!(dot.contains("L0"));
        assert!(dot.contains("L1"));
        assert!(dot.contains("L0 -> L1"), "the write/read edge must appear");
    }
}
