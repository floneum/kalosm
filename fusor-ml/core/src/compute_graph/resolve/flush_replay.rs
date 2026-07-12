//! Reusable materialization-plan replay for dense (QMatMul-free) graphs.
//!
//! Training loops flush an isomorphic tape every step, while iterative dense
//! inference repeatedly resolves one isomorphic output graph. Both present
//! the same operations, shapes, and expressions over fresh nodes and buffers.
//! The full resolver pipeline (execution-graph build, recognition + fusion,
//! toposort, lowering, consumer counting, per-op input gathering, workgroup
//! solving, and kernel building) is fully deterministic given that structure,
//! so its outcome can be recorded once and replayed on later steps.
//!
//! A replayed materialization skips every deterministic pass and only re-runs
//! the intrinsically per-step work: output-buffer allocation, positional buffer
//! rebinding ([`DirectKernelTemplate::bind_buffers`]), bind-group creation,
//! command encoding, and liveness bookkeeping.
//!
//! Safety properties (see the module tests and `verify_integrity`):
//! - Plans are strictly bufferless and `NodeIndex`-free: templates hold
//!   compile artifacts only, and all cross-step references are fingerprint
//!   slot positions remapped through a fresh DFS each step. Stable-graph
//!   index recycling can never alias, and no `Arc<wgpu::Buffer>` is retained
//!   across steps (the buffer pool's `strong_count == 1` recycling is never
//!   starved).
//! - Replay mutates the inner graph exclusively through the blessed APIs:
//!   `set_cached_result`, `add_dependency_edge` (re-adding the optimizer's
//!   recorded physical edges), and the exact release predicate used by
//!   `release_dead_intermediates` — evaluated live, never recorded, so
//!   reference-count drift (e.g. a user cloning a mid-graph handle) is
//!   handled identically to a full resolve.
//! - Decode graphs are untouched: the replay path is gated on the O(1)
//!   `qmatrix_node_count == 0` check before any fingerprint work, the
//!   fingerprint DFS aborts on any `QMatrix` variant, and the recorder
//!   poisons on the `QMatMul` lowering arm.
//! - Buffer identity between slots is deliberately absent from the
//!   fingerprint (buffers change every step), so the recorder must never
//!   produce a plan whose bindings depend on two slots incidentally sharing
//!   one buffer: `attribute_buffer` poisons the recording when distinct
//!   slots resolve to the same pointer unless the plan itself re-creates the
//!   sharing (view aliases, in-place outputs), and pins every tracked `Arc`
//!   so pool recycling cannot alias pointers mid-recording.

use std::hash::{Hash, Hasher};
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use lru::LruCache;
use parking_lot::Mutex;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHasher};
use web_time::Instant;

use super::{
    CommandRecord, ComputeGraphInner, ComputeGraphNodeVariant, DispatchRecord, NodeIndex,
    QueuedOperation, Resolver,
};
use crate::mir::kernel_backend::{DirectKernel, DirectKernelTemplate};
use crate::mir::operation::Operation;
use crate::tensor::TensorData;
use crate::{DataTypeEnum, Device, Layout};

/// Bump when anything about the recorded plan layout or the fingerprint
/// recipe changes, so stale entries can never be replayed.
/// v7: equality-saturation optimizer — recorded physical-edge sets shrink to
/// final-inputs-only and the optimizer output is produced by extraction.
const REPLAY_RECIPE_VERSION: u64 = 7;

const FLUSH_PLAN_CACHE_SIZE: usize = 8;

/// Structural fingerprint of one flush's pending subgraph.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct FlushPlanKey([u64; 2]);

#[derive(Clone)]
pub(crate) enum FlushPlanEntry {
    /// The fingerprint has been seen once. Recording is armed only on the
    /// second occurrence so one-shot workloads never pay plan construction.
    Seen,
    Recorded(Arc<FlushPlan>),
}

/// The reusable-plan decision for one materialization boundary. Both batched
/// flushes and ordinary single-target resolves enter through this gate so an
/// isomorphic graph has one cache lifecycle: see it, record it, then replay.
pub(crate) enum ReplayAction {
    Resolve,
    Record {
        key: FlushPlanKey,
        fingerprint: FlushFingerprint,
    },
    Replay {
        plan: Arc<FlushPlan>,
        fingerprint: FlushFingerprint,
    },
}

/// Per-device two-touch LRU of flush plans. Lives on `DeviceInner` beside the
/// kernel cache so it is reachable under the compute-graph write lock.
pub(crate) struct FlushPlanCache {
    plans: Mutex<LruCache<FlushPlanKey, FlushPlanEntry, FxBuildHasher>>,
    replays: AtomicU64,
    records: AtomicU64,
}

impl Default for FlushPlanCache {
    fn default() -> Self {
        Self {
            plans: Mutex::new(LruCache::with_hasher(
                NonZeroUsize::new(FLUSH_PLAN_CACHE_SIZE).expect("cache size must be non-zero"),
                FxBuildHasher,
            )),
            replays: AtomicU64::new(0),
            records: AtomicU64::new(0),
        }
    }
}

impl FlushPlanCache {
    pub(crate) fn get(&self, key: &FlushPlanKey) -> Option<FlushPlanEntry> {
        self.plans.lock().get(key).cloned()
    }

    pub(crate) fn insert(&self, key: FlushPlanKey, entry: FlushPlanEntry) {
        if matches!(entry, FlushPlanEntry::Recorded(_)) {
            self.records.fetch_add(1, Ordering::Relaxed);
        }
        self.plans.lock().put(key, entry);
    }

    pub(crate) fn note_replay(&self) {
        self.replays.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(crate) fn replay_count(&self) -> u64 {
        self.replays.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(crate) fn record_count(&self) -> u64 {
        self.records.load(Ordering::Relaxed)
    }
}

/// Whether the flush-plan replay path may run at all. Trace modes that need
/// the full resolver pipeline to produce their output disable it.
pub(crate) fn replay_enabled() -> bool {
    const DISABLING_ENVS: &[&str] = &[
        "FUSOR_DISABLE_RESOLVE_PLAN_CACHE",
        "FUSOR_TRACE_GPU_KERNELS",
        "FUSOR_TRACE_DECODE",
        "FUSOR_TRACE_RESOLVE",
        "FUSOR_TRACE_DECODE_NAMES",
        "FUSOR_TRACE_RESOLVE_HOST_CATEGORIES",
        "FUSOR_TRACE_ROW_FUSION",
    ];
    DISABLING_ENVS
        .iter()
        .all(|var| std::env::var_os(var).is_none())
}

/// Choose whether `targets` need a full resolve, should record, or can replay.
///
/// This performs the QMatrix gate before fingerprinting so quantized decode
/// retains its historical direct/general resolve paths. A failed replay
/// validation is non-mutating and falls back to a normal resolve.
pub(crate) fn prepare_replay(graph: &mut ComputeGraphInner, targets: &[NodeIndex]) -> ReplayAction {
    if graph.qmatrix_node_count != 0 || !replay_enabled() {
        return ReplayAction::Resolve;
    }
    let Some(fingerprint) = fingerprint_pending(graph, targets) else {
        return ReplayAction::Resolve;
    };
    // A single-target resolve can run while another live output depends on
    // part of the same graph. That outside descendant affects allocation
    // claims but is intentionally absent from this target's fingerprint.
    // Replay only closed materialization boundaries: every live uncached
    // handle must belong to the fingerprinted dependency closure.
    if graph.pending_sinks.iter().any(|(&node, _)| {
        graph
            .nodes
            .nodes
            .node_weight(node)
            .is_some_and(|data| data.reference_count > 0 && data.cached.is_none())
            && !fingerprint.pos_of.contains_key(&node)
    }) {
        return ReplayAction::Resolve;
    }

    let device = graph.device();
    let cache = device.flush_plan_cache();
    match cache.get(&fingerprint.key) {
        Some(FlushPlanEntry::Recorded(plan)) => {
            if validate_replay(graph, &device, &plan, &fingerprint) {
                ReplayAction::Replay { plan, fingerprint }
            } else {
                ReplayAction::Resolve
            }
        }
        Some(FlushPlanEntry::Seen) => ReplayAction::Record {
            key: fingerprint.key,
            fingerprint,
        },
        None => {
            cache.insert(fingerprint.key, FlushPlanEntry::Seen);
            ReplayAction::Resolve
        }
    }
}

/// Env vars that change what the resolver produces. Their values join the
/// fingerprint so plans recorded under different settings can never collide.
fn hash_env_snapshot(state: &mut FxHasher) {
    const KEYED_ENVS: &[&str] = &[
        "FUSOR_RESOLVE_SKIP_OPTIMIZE",
        "FUSOR_RESOLVE_SKIP_ASSIGN_CHAINS",
        "FUSOR_RESOLVE_OPTIMIZE_MAX_NODES",
        "FUSOR_RESOLVE_QMATMUL_ELEMENTWISE_FUSION",
        "FUSOR_DISABLE_DECODE_PLAN_CACHE",
        "FUSOR_RESOLVE_DISABLE_DENSE_REDUCE_FUSION",
        "FUSOR_DISABLE_HORIZONTAL_FUSION",
        "FUSOR_DISABLE_REGION_FUSION",
        "FUSOR_DISABLE_ALLOC_REUSE",
        "FUSOR_LAST_LEVEL_CACHE_BYTES",
    ];
    for var in KEYED_ENVS {
        match std::env::var(var) {
            Ok(value) => {
                1u8.hash(state);
                value.hash(state);
            }
            Err(_) => 0u8.hash(state),
        }
    }
}

/// Two accumulating hashers producing the 128-bit plan key. Lane `b` is fed
/// a deterministic mix of the same 64-bit words as lane `a`, so per-write
/// entropy stays 64 bits (one FxHash); the second lane widens the
/// *accumulator* state to make cross-node cancellation collisions harder,
/// not the per-node hash. Replay correctness does not rest on this key alone:
/// upfront validation re-checks step kinds and boundary caching, and the
/// recorder refuses plans whose buffer provenance is ambiguous.
struct FingerprintHasher {
    a: FxHasher,
    b: FxHasher,
}

impl FingerprintHasher {
    fn new() -> Self {
        let mut a = FxHasher::default();
        0u64.hash(&mut a);
        let mut b = FxHasher::default();
        1u64.hash(&mut b);
        Self { a, b }
    }

    fn write_u64(&mut self, value: u64) {
        value.hash(&mut self.a);
        (value.rotate_left(32) ^ 0x9E37_79B9_7F4A_7C15).hash(&mut self.b);
    }

    fn finish(self) -> FlushPlanKey {
        FlushPlanKey([self.a.finish(), self.b.finish()])
    }
}

fn local_hash(f: impl FnOnce(&mut FxHasher)) -> u64 {
    let mut hasher = FxHasher::default();
    f(&mut hasher);
    hasher.finish()
}

fn hash_layout(state: &mut FxHasher, layout: &Layout) {
    layout.offset().hash(state);
    layout.shape().hash(state);
    layout.strides().hash(state);
}

/// The structural fingerprint of one flush: a 128-bit key plus the slot
/// assignment (DFS discovery order) mapping fingerprint positions to the
/// current step's `NodeIndex`es. Recomputed fresh every flush; positions —
/// never `NodeIndex` values — are what plans store.
pub(crate) struct FlushFingerprint {
    pub(crate) key: FlushPlanKey,
    slots: Vec<NodeIndex>,
    pos_of: FxHashMap<NodeIndex, u32>,
    /// True for slots that were already cached when the fingerprint was
    /// taken. Boundaries are hashed opaquely (dtype/layout/allocation size)
    /// and never descended past — the cached-view-collapse rule.
    boundary: Vec<bool>,
}

struct FingerprintState {
    hasher: FingerprintHasher,
    slots: Vec<NodeIndex>,
    pos_of: FxHashMap<NodeIndex, u32>,
    boundary: Vec<bool>,
}

/// Fingerprint the pending subgraph, mirroring `build_execution_graph`'s
/// traversal exactly: DFS from the pending sinks in order, visiting
/// dependencies in `visit_dependencies` order, stopping at cached nodes.
/// Returns `None` if any reachable node is a `QMatrix` (decode graphs are
/// never fingerprinted).
pub(crate) fn fingerprint_pending(
    graph: &ComputeGraphInner,
    pending: &[NodeIndex],
) -> Option<FlushFingerprint> {
    let mut state = FingerprintState {
        hasher: FingerprintHasher::new(),
        slots: Vec::new(),
        pos_of: FxHashMap::default(),
        boundary: Vec::new(),
    };

    for &sink in pending {
        let pos = fingerprint_visit(graph, sink, &mut state)?;
        state.hasher.write_u64(0xE0);
        state.hasher.write_u64(pos as u64);
    }
    state.hasher.write_u64(REPLAY_RECIPE_VERSION);
    state.hasher.write_u64(local_hash(hash_env_snapshot));

    Some(FlushFingerprint {
        key: state.hasher.finish(),
        slots: state.slots,
        pos_of: state.pos_of,
        boundary: state.boundary,
    })
}

fn fingerprint_visit(
    graph: &ComputeGraphInner,
    node: NodeIndex,
    state: &mut FingerprintState,
) -> Option<u32> {
    if let Some(&pos) = state.pos_of.get(&node) {
        return Some(pos);
    }
    let node_data = graph.nodes.nodes.node_weight(node)?;
    let pos = u32::try_from(state.slots.len()).ok()?;
    state.slots.push(node);
    state.pos_of.insert(node, pos);

    if let Some(cached) = &node_data.cached {
        // Cached boundary leaf: opaque. The buffer contents are per-step
        // data; only the layout contract the recorded kernels were compiled
        // against must match. The liveness bit is part of the structure too:
        // allocation-reuse claims depend on whether the user still holds a
        // handle to the boundary tensor, so "user kept a handle" must never
        // collide with a plan whose claims assumed the buffer was dead.
        state.boundary.push(true);
        state.hasher.write_u64(0xB0);
        state
            .hasher
            .write_u64((node_data.reference_count > 0) as u64);
        state.hasher.write_u64(local_hash(|h| {
            cached.datatype().hash(h);
            hash_layout(h, cached.layout());
            cached.buffer().size().hash(h);
        }));
        return Some(pos);
    }
    state.boundary.push(false);

    // Recognition and fusion consult live references (`has_live_reference`),
    // so the liveness bit is part of the structure.
    let live_ref = node_data.reference_count > 0;
    match &node_data.variant {
        ComputeGraphNodeVariant::QMatrix(_) => return None,
        ComputeGraphNodeVariant::Tensor(data) => {
            state.hasher.write_u64(0xA1);
            state.hasher.write_u64(live_ref as u64);
            state.hasher.write_u64(local_hash(|h| {
                data.datatype().hash(h);
                hash_layout(h, data.layout());
                data.buffer().size().hash(h);
            }));
        }
        variant => {
            let (tag, op): (u64, &dyn Operation) = match variant {
                ComputeGraphNodeVariant::Elementwise(op) => (0xA2, op),
                ComputeGraphNodeVariant::Reduce(op) => (0xA3, op),
                ComputeGraphNodeVariant::View(op) => (0xA4, op),
                ComputeGraphNodeVariant::Assign(op) => (0xA5, op),
                ComputeGraphNodeVariant::Tensor(_) | ComputeGraphNodeVariant::QMatrix(_) => {
                    unreachable!("handled above")
                }
            };
            state.hasher.write_u64(tag);
            state.hasher.write_u64(live_ref as u64);
            state
                .hasher
                .write_u64(local_hash(|h| op.hash_kernel_fields(h)));
        }
    }

    let mut dependencies = Vec::new();
    node_data.variant.visit_dependencies(&mut |dep| {
        dependencies.push(dep);
    });
    state.hasher.write_u64(dependencies.len() as u64);
    for dep in dependencies {
        let dep_pos = fingerprint_visit(graph, dep, state)?;
        state.hasher.write_u64(dep_pos as u64);
    }
    Some(pos)
}

/// The recorded outcome of one materialization. Strictly bufferless and
/// `NodeIndex`-free; every cross-step reference is a fingerprint slot
/// position.
pub(crate) struct FlushPlan {
    node_count: u32,
    /// Positions of the flush targets. Targets keep their cached results;
    /// everything else may be released once its last consumer ran.
    target_positions: Box<[u32]>,
    /// Inner-graph dependency edges the optimizer added during the recording
    /// resolve (`add_physical_dependencies`). Replay re-adds them so liveness
    /// accounting sees post-fusion dependencies.
    physical_edges: Box<[(u32, u32)]>,
    steps: Box<[PlanStep]>,
}

enum PlanStep {
    /// Cache an input leaf's own tensor data (current step's contents).
    TensorLeaf { pos: u32 },
    /// Zero-cost view alias re-derived via `try_map_tensor`.
    ViewAlias { pos: u32, consumed: Box<[u32]> },
    /// In-place slice-assign buffer copies re-derived from the current graph.
    CopyAssign { pos: u32, consumed: Box<[u32]> },
    /// One lowered operation's kernel dispatches.
    Dispatch {
        pos: u32,
        kernels: Box<[PlanKernel]>,
        output: OutputSpec,
        /// Dependency positions (with multiplicity, in `visit_dependencies`
        /// order) to feed the release accounting after this step.
        consumed: Box<[u32]>,
    },
    /// One horizontally merged dispatch caching several segments' outputs.
    /// Every output is a freshly allocated buffer (never an alias), so slot
    /// attribution stays 1 buffer <-> 1 slot.
    MergedDispatch {
        outputs: Box<[(u32, OutputSpec)]>,
        kernels: Box<[PlanKernel]>,
        consumed: Box<[u32]>,
    },
}

struct PlanKernel {
    template: DirectKernelTemplate,
    /// Slot position per binding buffer, in `binding_buffers()` order.
    bindings: Box<[u32]>,
}

struct OutputSpec {
    layout: Layout,
    datatype: DataTypeEnum,
    source: OutputSource,
}

enum OutputSource {
    /// Freshly allocated output buffer (the common case: `Operation::inputs`
    /// allocates the output and appends it as the last MIR value).
    Fresh {
        buffer_size: u64,
        usage: wgpu::BufferUsages,
    },
    /// Output aliases an existing slot's buffer (in-place operations).
    Alias { slot: u32 },
}

/// Records one materialization into a [`FlushPlan`]. Armed on the second
/// sighting of a fingerprint; every hook is a no-op once poisoned. Any
/// structure the plan format cannot express
/// (unknown scratch buffers, quantized matmuls) poisons the recording — the
/// resolve completes normally and no plan is stored.
pub(crate) struct PlanRecorder {
    pos_of: FxHashMap<NodeIndex, u32>,
    node_count: u32,
    target_positions: Box<[u32]>,
    /// Buffer provenance: raw buffer pointer -> the slot whose caching event
    /// most recently produced that pointer. Seeded with boundary and tensor
    /// leaf buffers; updated as steps record. Sound only because `pinned`
    /// keeps every tracked `Arc` alive for the recording's duration (a
    /// pointer can never be pool-recycled into a different buffer
    /// mid-recording) and because incidental sharing poisons the recording
    /// (see [`Self::attribute_buffer`]).
    provenance: FxHashMap<usize, u32>,
    /// Strong clones of every buffer entered into `provenance`. Dropped with
    /// the recorder at the end of the flush, so no buffer outlives the step.
    pinned: Vec<Arc<wgpu::Buffer>>,
    physical_edges: Vec<(u32, u32)>,
    steps: Vec<PlanStep>,
    poisoned: bool,
}

impl PlanRecorder {
    pub(crate) fn new(
        graph: &ComputeGraphInner,
        targets: &[NodeIndex],
        fingerprint: FlushFingerprint,
    ) -> Self {
        let mut poisoned = false;
        let target_positions = targets
            .iter()
            .map(|target| match fingerprint.pos_of.get(target) {
                Some(&pos) => pos,
                None => {
                    poisoned = true;
                    0
                }
            })
            .collect();
        let mut recorder = Self {
            node_count: fingerprint.slots.len() as u32,
            pos_of: fingerprint.pos_of,
            target_positions,
            provenance: FxHashMap::default(),
            pinned: Vec::new(),
            physical_edges: Vec::new(),
            steps: Vec::new(),
            poisoned,
        };
        for (i, &node) in fingerprint.slots.iter().enumerate() {
            let Some(node_data) = graph.nodes.nodes.node_weight(node) else {
                continue;
            };
            if let Some(cached) = &node_data.cached {
                // Boundary sharing is incidental (e.g. a cached view aliasing
                // its cached input): never reproduced by the plan, so it must
                // poison rather than conflate the slots.
                recorder.attribute_buffer(cached.buffer(), i as u32, false);
            } else if let ComputeGraphNodeVariant::Tensor(data) = &node_data.variant {
                recorder.attribute_buffer(data.buffer(), i as u32, false);
            }
        }
        recorder
    }

    /// Attribute `buffer` to slot `pos`, pinning the `Arc` so its raw pointer
    /// stays unique for the rest of the recording.
    ///
    /// Kernel bindings are recorded by looking buffer pointers up in
    /// `provenance`, so two distinct slots sharing one buffer are
    /// indistinguishable at record time. That is only sound when the plan
    /// itself re-creates the sharing every step (`structural`: view aliases,
    /// in-place outputs — either slot resolves to the same buffer at replay).
    /// Incidental sharing (two cached boundaries or tensor leaves that happen
    /// to alias this step) is NOT part of the fingerprint — an isomorphic
    /// later step may present distinct buffers, and a plan recorded here
    /// would silently bind the wrong tensor. Poison instead.
    fn attribute_buffer(&mut self, buffer: &Arc<wgpu::Buffer>, pos: u32, structural: bool) {
        self.pinned.push(buffer.clone());
        if let Some(prev) = self.provenance.insert(Arc::as_ptr(buffer) as usize, pos)
            && prev != pos
            && !structural
        {
            self.poison();
        }
    }

    /// Classify one output buffer as Fresh or a resolver-chosen Alias.
    /// Returns `None` (and poisons) on any disagreement between pointer
    /// provenance and the resolver's claim ledger.
    /// Classify one output buffer as Fresh or a resolver-chosen Alias.
    /// The provenance slot (the buffer's last structural writer) is the
    /// replay-faithful alias target: several fingerprint slots can share one
    /// buffer (a producer and its views), so the claim ledger's source node
    /// attests that the reuse was deliberately chosen while provenance names
    /// the slot whose replay buffer is this buffer. A provenance hit without
    /// a chosen claim is accidental buffer sharing between distinct slots
    /// (replaying it would bind one slot's data where another slot's is
    /// expected), and a claim without a hit is an accounting inconsistency —
    /// both poison the plan.
    fn classify_output_source(
        &mut self,
        output_ptr: usize,
        output: &TensorData,
        claimed_from: Option<NodeIndex>,
    ) -> Option<OutputSource> {
        match (self.provenance.get(&output_ptr).copied(), claimed_from) {
            (Some(slot), Some(_)) => Some(OutputSource::Alias { slot }),
            (None, None) => Some(OutputSource::Fresh {
                buffer_size: output.buffer().size(),
                usage: output.buffer().usage(),
            }),
            _ => {
                self.poison();
                None
            }
        }
    }

    /// Raw pointers of every buffer this recorder holds a strong clone of.
    pub(super) fn pinned_ptrs(&self) -> impl Iterator<Item = usize> + '_ {
        self.pinned
            .iter()
            .map(|buffer| Arc::as_ptr(buffer) as usize)
    }

    fn pos(&mut self, node: NodeIndex) -> Option<u32> {
        match self.pos_of.get(&node) {
            Some(&pos) => Some(pos),
            None => {
                self.poison();
                None
            }
        }
    }

    fn consumed_positions(&mut self, deps: &[NodeIndex]) -> Option<Box<[u32]>> {
        let mut consumed = Vec::with_capacity(deps.len());
        for &dep in deps {
            consumed.push(self.pos(dep)?);
        }
        Some(consumed.into())
    }

    pub(super) fn poison(&mut self) {
        self.poisoned = true;
    }

    pub(super) fn record_physical_edge(&mut self, from: NodeIndex, to: NodeIndex) {
        if self.poisoned {
            return;
        }
        let Some(from) = self.pos(from) else { return };
        let Some(to) = self.pos(to) else { return };
        self.physical_edges.push((from, to));
    }

    pub(super) fn record_tensor_leaf(&mut self, node: NodeIndex, data: &TensorData) {
        if self.poisoned {
            return;
        }
        let Some(pos) = self.pos(node) else { return };
        // A leaf sharing a buffer with another slot (boundary or leaf) is
        // incidental: poison rather than conflate.
        self.attribute_buffer(data.buffer(), pos, false);
        self.steps.push(PlanStep::TensorLeaf { pos });
    }

    pub(super) fn record_view_alias(
        &mut self,
        node: NodeIndex,
        result: &TensorData,
        deps: &[NodeIndex],
    ) {
        if self.poisoned {
            return;
        }
        let Some(pos) = self.pos(node) else { return };
        let Some(consumed) = self.consumed_positions(deps) else {
            return;
        };
        // Structural: replay re-derives this alias from the input's current
        // buffer via `try_map_tensor`, so the sharing holds every step.
        self.attribute_buffer(result.buffer(), pos, true);
        self.steps.push(PlanStep::ViewAlias { pos, consumed });
    }

    pub(super) fn record_copy_assign(
        &mut self,
        node: NodeIndex,
        output: &TensorData,
        op: &QueuedOperation,
    ) {
        if self.poisoned {
            return;
        }
        let Some(pos) = self.pos(node) else { return };
        let mut deps = Vec::new();
        op.visit_dependencies(&mut |dep| deps.push(dep));
        let Some(consumed) = self.consumed_positions(&deps) else {
            return;
        };
        // Structural: replay re-derives the in-place output from the current
        // graph via `try_prepare_in_place_slice_assign_copy`.
        self.attribute_buffer(output.buffer(), pos, true);
        self.steps.push(PlanStep::CopyAssign { pos, consumed });
    }

    pub(super) fn record_dispatch(
        &mut self,
        node: NodeIndex,
        kernels: &[DirectKernel],
        output: &TensorData,
        op: &QueuedOperation,
        claimed_from: Option<NodeIndex>,
    ) {
        if self.poisoned {
            return;
        }
        let Some(pos) = self.pos(node) else { return };

        // Alias-vs-Fresh classification requires BOTH the pointer-provenance
        // hit AND the resolver's explicit claim to agree on the source slot:
        // a provenance hit without a chosen claim is accidental buffer
        // sharing between distinct slots (replaying it would bind one slot's
        // data where another slot's is expected — poison), and a claim
        // without a hit is an accounting inconsistency (poison). `pinned` keeps every
        // tracked pointer alive, so hits are never pool-recycled coincidences.
        let output_ptr = Arc::as_ptr(output.buffer()) as usize;
        let Some(source) = self.classify_output_source(output_ptr, output, claimed_from) else {
            return;
        };
        let output_spec = OutputSpec {
            layout: output.layout().clone(),
            datatype: output.datatype(),
            source,
        };
        // Register the output before classifying bindings: the output buffer
        // is itself a binding slot (appended by `Operation::inputs`).
        // Structural: replay reproduces the alias through `OutputSource`.
        self.attribute_buffer(output.buffer(), pos, true);

        let mut plan_kernels = Vec::with_capacity(kernels.len());
        for kernel in kernels {
            let buffers = kernel.binding_buffers();
            let mut bindings = Vec::with_capacity(buffers.len());
            for buffer in &buffers {
                let Some(&slot) = self.provenance.get(&(Arc::as_ptr(buffer) as usize)) else {
                    // Build-internal scratch buffer (e.g. split row-program
                    // sequences): the plan cannot re-create it.
                    self.poison();
                    return;
                };
                bindings.push(slot);
            }
            plan_kernels.push(PlanKernel {
                template: kernel.to_template(),
                bindings: bindings.into(),
            });
        }

        let mut deps = Vec::new();
        op.visit_dependencies(&mut |dep| deps.push(dep));
        let Some(consumed) = self.consumed_positions(&deps) else {
            return;
        };
        self.steps.push(PlanStep::Dispatch {
            pos,
            kernels: plan_kernels.into(),
            output: output_spec,
            consumed,
        });
    }

    /// Record one horizontally merged dispatch. Every segment output must be
    /// a fresh buffer: a provenance hit means the output aliases an existing
    /// slot, which the merged replay arm cannot re-create — poison.
    pub(super) fn record_merged_dispatch(
        &mut self,
        segment_outputs: &[(NodeIndex, &TensorData, Option<NodeIndex>)],
        kernels: &[DirectKernel],
        op: &super::merge_horizontal::MergedSegments,
    ) {
        if self.poisoned {
            return;
        }
        let mut outputs = Vec::with_capacity(segment_outputs.len());
        for (node, output, claimed_from) in segment_outputs {
            let Some(pos) = self.pos(*node) else { return };
            let output_ptr = Arc::as_ptr(output.buffer()) as usize;
            let Some(source) = self.classify_output_source(output_ptr, output, *claimed_from)
            else {
                return;
            };
            outputs.push((
                pos,
                OutputSpec {
                    layout: output.layout().clone(),
                    datatype: output.datatype(),
                    source,
                },
            ));
            // Structural: the merged replay arm re-creates this slot (fresh
            // allocation or the recorded alias).
            self.attribute_buffer(output.buffer(), pos, true);
        }
        let mut plan_kernels = Vec::with_capacity(kernels.len());
        for kernel in kernels {
            let buffers = kernel.binding_buffers();
            let mut bindings = Vec::with_capacity(buffers.len());
            for buffer in &buffers {
                let Some(&slot) = self.provenance.get(&(Arc::as_ptr(buffer) as usize)) else {
                    self.poison();
                    return;
                };
                bindings.push(slot);
            }
            plan_kernels.push(PlanKernel {
                template: kernel.to_template(),
                bindings: bindings.into(),
            });
        }

        let mut deps = Vec::new();
        op.visit_dependencies(&mut |dep| deps.push(dep));
        let Some(consumed) = self.consumed_positions(&deps) else {
            return;
        };
        self.steps.push(PlanStep::MergedDispatch {
            outputs: outputs.into(),
            kernels: plan_kernels.into(),
            consumed,
        });
    }

    pub(crate) fn finish(self) -> Option<FlushPlan> {
        if self.poisoned {
            return None;
        }
        Some(FlushPlan {
            node_count: self.node_count,
            target_positions: self.target_positions,
            physical_edges: self.physical_edges.into(),
            steps: self.steps.into(),
        })
    }
}

/// Replay a recorded plan against the current step's isomorphic graph.
/// Returns `false` (without having mutated anything) if upfront validation
/// fails; the caller then falls back to a full resolve.
fn validate_replay(
    graph: &ComputeGraphInner,
    device: &Device,
    plan: &FlushPlan,
    fingerprint: &FlushFingerprint,
) -> bool {
    let slots = &fingerprint.slots;
    if plan.node_count as usize != slots.len() {
        return false;
    }
    // Upfront validation, before any mutation: every step's slot must hold
    // the node kind the plan expects and must not be cached yet, and every
    // boundary must be cached. After this point the plan cannot fail.
    let check_slot = |pos: u32, kind: u8| -> bool {
        let Some(node) = graph.nodes.nodes.node_weight(slots[pos as usize]) else {
            return false;
        };
        if node.cached.is_some() {
            return false;
        }
        matches!(
            (&node.variant, kind),
            (ComputeGraphNodeVariant::Tensor(_), 0)
                | (ComputeGraphNodeVariant::View(_), 1)
                | (ComputeGraphNodeVariant::Assign(_), 2)
                | (_, 3)
        )
    };
    // An alias output overwrites its source slot's buffer. A source
    // produced by an earlier plan step is exclusively owned by this replay.
    // A boundary source (cached before the flush) is legal only when its
    // liveness matches the recording: the fingerprint pins the node-level
    // bits (reference count, descendants), and the strong count check below
    // catches non-graph holders (an in-flight download, a raw clone) — any
    // mismatch falls back to a full resolve, which re-decides the claims.
    let mut produced = vec![false; slots.len()];
    for step in &plan.steps {
        let alias_ok = |spec: &OutputSpec, produced: &[bool]| match spec.source {
            OutputSource::Alias { slot } => {
                if produced[slot as usize] {
                    return true;
                }
                let Some(node) = graph.nodes.nodes.node_weight(slots[slot as usize]) else {
                    return false;
                };
                let Some(cached) = &node.cached else {
                    return false;
                };
                let buffer = cached.buffer();
                let expected = 1 + u32::from(device.buffer_pool_is_tracked(
                    buffer.size(),
                    buffer.usage(),
                    buffer,
                ));
                Arc::strong_count(buffer) as u32 == expected
            }
            OutputSource::Fresh { .. } => true,
        };
        let ok = match step {
            PlanStep::TensorLeaf { pos } => {
                produced[*pos as usize] = true;
                check_slot(*pos, 0)
            }
            PlanStep::ViewAlias { pos, .. } => {
                produced[*pos as usize] = true;
                check_slot(*pos, 1)
            }
            PlanStep::CopyAssign { pos, .. } => {
                produced[*pos as usize] = true;
                check_slot(*pos, 2)
            }
            PlanStep::Dispatch { pos, output, .. } => {
                let ok = check_slot(*pos, 3) && alias_ok(output, &produced);
                produced[*pos as usize] = true;
                ok
            }
            PlanStep::MergedDispatch { outputs, .. } => {
                let ok = outputs
                    .iter()
                    .all(|(pos, spec)| check_slot(*pos, 3) && alias_ok(spec, &produced));
                for (pos, _) in outputs.iter() {
                    produced[*pos as usize] = true;
                }
                ok
            }
        };
        if !ok {
            return false;
        }
    }
    for (i, &node) in slots.iter().enumerate() {
        if fingerprint.boundary[i] && graph.get_cached_result(node).is_none() {
            return false;
        }
    }
    true
}

/// Execute an already validated plan and append `tail` to its final command
/// encoder before submission. Validation and execution are split so callers
/// retain ownership of an `FnOnce` tail when a plan miss must fall back to the
/// full resolver.
pub(crate) fn execute_replay_with_tail<T>(
    graph: &mut ComputeGraphInner,
    plan: &FlushPlan,
    fingerprint: &FlushFingerprint,
    tail: impl FnOnce(&TensorData, &mut wgpu::CommandEncoder) -> T,
) -> (super::ResolverResult, T) {
    let slots = &fingerprint.slots;
    let device = graph.device();
    let host_trace = std::env::var_os("FUSOR_TRACE_RESOLVE_HOST").is_some();
    let start = host_trace.then(Instant::now);

    // Re-add the optimizer's persistent physical dependency edges through the
    // liveness-maintaining API.
    for &(from, to) in plan.physical_edges.iter() {
        graph.add_dependency_edge(slots[from as usize], slots[to as usize]);
    }

    // Structural consumer counts, identical to the recording resolve's
    // `remaining_consumers` map (keyed by slot position instead of node).
    let mut counts = vec![0u32; slots.len()];
    for step in &plan.steps {
        for &c in step_consumed(step) {
            counts[c as usize] += 1;
        }
    }
    let mut is_target = vec![false; slots.len()];
    for &t in plan.target_positions.iter() {
        is_target[t as usize] = true;
    }

    // Buffers per slot, captured at each slot's caching event. Kept locally
    // (not read back through `cached`) so mid-replay releases can't drop a
    // buffer a later dispatch still binds — mirroring how the full resolve
    // pins bound buffers in its command records.
    let mut slot_buffers: Vec<Option<Arc<wgpu::Buffer>>> = vec![None; slots.len()];
    for (i, &node) in slots.iter().enumerate() {
        if fingerprint.boundary[i] {
            slot_buffers[i] = graph.get_cached_result(node).map(|d| d.buffer().clone());
        }
    }

    let kernel_cache = device.kernel_cache();
    let mut commands = Vec::<CommandRecord>::with_capacity(plan.steps.len());
    for step in &plan.steps {
        match step {
            PlanStep::TensorLeaf { pos } => {
                let idx = slots[*pos as usize];
                let ComputeGraphNodeVariant::Tensor(data) = &graph
                    .nodes
                    .nodes
                    .node_weight(idx)
                    .expect("flush replay: validated slot disappeared")
                    .variant
                else {
                    unreachable!("flush replay: validated tensor leaf changed kind");
                };
                let data = data.clone();
                slot_buffers[*pos as usize] = Some(data.buffer().clone());
                graph.set_cached_result(idx, data);
            }
            PlanStep::ViewAlias { pos, consumed } => {
                let idx = slots[*pos as usize];
                let result = {
                    let node = graph
                        .nodes
                        .nodes
                        .node_weight(idx)
                        .expect("flush replay: validated slot disappeared");
                    let ComputeGraphNodeVariant::View(view) = &node.variant else {
                        unreachable!("flush replay: validated view alias changed kind");
                    };
                    let input = graph
                        .get_cached_result(view.input)
                        .expect("flush replay: view alias input must be cached");
                    view.try_map_tensor(input)
                        .expect("flush replay: recorded view alias must still map")
                };
                slot_buffers[*pos as usize] = Some(result.buffer().clone());
                graph.set_cached_result(idx, result);
                release_consumed(graph, slots, &mut counts, &is_target, consumed);
            }
            PlanStep::CopyAssign { pos, consumed } => {
                let idx = slots[*pos as usize];
                let (output, copies) = {
                    let node = graph
                        .nodes
                        .nodes
                        .node_weight(idx)
                        .expect("flush replay: validated slot disappeared");
                    let ComputeGraphNodeVariant::Assign(op) = &node.variant else {
                        unreachable!("flush replay: validated slice assign changed kind");
                    };
                    Resolver::try_prepare_in_place_slice_assign_copy(graph, op)
                        .expect("flush replay: recorded slice-assign copy must still apply")
                };
                slot_buffers[*pos as usize] = Some(output.buffer().clone());
                graph.set_cached_result(idx, output);
                commands.extend(copies.into_iter().map(CommandRecord::CopyBuffer));
                release_consumed(graph, slots, &mut counts, &is_target, consumed);
            }
            PlanStep::Dispatch {
                pos,
                kernels,
                output,
                consumed,
            } => {
                let idx = slots[*pos as usize];
                let output_data = match &output.source {
                    OutputSource::Fresh { buffer_size, usage } => {
                        let buffer = device.create_buffer(*buffer_size, *usage);
                        TensorData::new_from_parts(
                            &device,
                            buffer,
                            output.layout.clone(),
                            output.datatype,
                        )
                    }
                    OutputSource::Alias { slot } => {
                        let buffer = slot_buffers[*slot as usize]
                            .clone()
                            .expect("flush replay: alias output slot has no buffer");
                        TensorData::new_from_parts(
                            &device,
                            buffer,
                            output.layout.clone(),
                            output.datatype,
                        )
                    }
                };
                slot_buffers[*pos as usize] = Some(output_data.buffer().clone());
                for kernel in kernels.iter() {
                    let buffers = kernel
                        .bindings
                        .iter()
                        .map(|&slot| {
                            slot_buffers[slot as usize]
                                .clone()
                                .expect("flush replay: binding slot has no buffer")
                        })
                        .collect::<Vec<_>>();
                    let bound = kernel.template.bind_buffers(&buffers);
                    if let Some(dispatch) = bound.prepare_dispatch(kernel_cache) {
                        commands.push(CommandRecord::Dispatch(DispatchRecord {
                            dispatch,
                            name: bound.name().to_string(),
                            category: None,
                        }));
                    }
                }
                graph.set_cached_result(idx, output_data);
                release_consumed(graph, slots, &mut counts, &is_target, consumed);
            }
            PlanStep::MergedDispatch {
                outputs,
                kernels,
                consumed,
            } => {
                for (pos, output) in outputs.iter() {
                    let buffer = match &output.source {
                        OutputSource::Fresh { buffer_size, usage } => {
                            device.create_buffer(*buffer_size, *usage)
                        }
                        OutputSource::Alias { slot } => slot_buffers[*slot as usize]
                            .clone()
                            .expect("flush replay: alias output slot has no buffer"),
                    };
                    let output_data = TensorData::new_from_parts(
                        &device,
                        buffer,
                        output.layout.clone(),
                        output.datatype,
                    );
                    slot_buffers[*pos as usize] = Some(output_data.buffer().clone());
                    graph.set_cached_result(slots[*pos as usize], output_data);
                }
                for kernel in kernels.iter() {
                    let buffers = kernel
                        .bindings
                        .iter()
                        .map(|&slot| {
                            slot_buffers[slot as usize]
                                .clone()
                                .expect("flush replay: binding slot has no buffer")
                        })
                        .collect::<Vec<_>>();
                    let bound = kernel.template.bind_buffers(&buffers);
                    if let Some(dispatch) = bound.prepare_dispatch(kernel_cache) {
                        commands.push(CommandRecord::Dispatch(DispatchRecord {
                            dispatch,
                            name: bound.name().to_string(),
                            category: None,
                        }));
                    }
                }
                release_consumed(graph, slots, &mut counts, &is_target, consumed);
            }
        }
    }

    let total_kernels = commands
        .iter()
        .filter(|command| matches!(command, CommandRecord::Dispatch(_)))
        .count();
    let mut command_encoder = encode_commands(&device, &commands, total_kernels);
    let target = slots[plan.target_positions[0] as usize];
    let data = graph
        .get_result(target)
        .expect("flush replay: target result not cached");
    let tail_result = tail(&data, &mut command_encoder);
    device.wgpu_queue().submit(Some(command_encoder.finish()));
    device.reset_initialized_buffers();
    device.flush_plan_cache().note_replay();

    if let Some(start) = start {
        tracing::info!(
            "resolve_host_profile queued_ops={} kernels={total_kernels} total={:?} replayed=true",
            plan.steps.len(),
            start.elapsed(),
        );
    }
    (
        super::ResolverResult {
            data,
            total_kernels,
        },
        tail_result,
    )
}

fn step_consumed(step: &PlanStep) -> &[u32] {
    match step {
        PlanStep::TensorLeaf { .. } => &[],
        PlanStep::ViewAlias { consumed, .. }
        | PlanStep::CopyAssign { consumed, .. }
        | PlanStep::Dispatch { consumed, .. }
        | PlanStep::MergedDispatch { consumed, .. } => consumed,
    }
}

/// The exact release predicate of `Resolver::release_dead_intermediates`,
/// evaluated live against the current graph: clear `cached` only when the
/// last recorded consumer ran AND the node is not a flush target AND no
/// user-held lazy tensor still transitively depends on it. Because
/// `has_live_lazy_descendant` is consulted at replay time, reference-count
/// drift invisible to the structural fingerprint is handled exactly as a
/// full resolve would handle it.
fn release_consumed(
    graph: &mut ComputeGraphInner,
    slots: &[NodeIndex],
    counts: &mut [u32],
    is_target: &[bool],
    consumed: &[u32],
) {
    for &c in consumed {
        let c = c as usize;
        let count = &mut counts[c];
        *count = count.saturating_sub(1);
        if *count == 0
            && !is_target[c]
            && !graph.has_live_lazy_descendant(slots[c])
            && let Some(node) = graph.nodes.nodes.node_weight_mut(slots[c])
        {
            node.cached = None;
        }
    }
}

/// Encode and submit the replayed command stream with the same pass/submit
/// chunking policy as the full resolver (single pass + single submit below
/// 1024 kernels; chunked with Metal waits above).
fn encode_commands(
    device: &Device,
    commands: &[CommandRecord],
    total_kernels: usize,
) -> wgpu::CommandEncoder {
    let command_encoder =
        device
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Resolver Encoder"),
            });
    super::queue_executor::encode_command_records(
        device,
        commands,
        total_kernels,
        command_encoder,
        |encoder, wait| {
            device.wgpu_queue().submit(Some(encoder.finish()));
            if wait {
                device.poll_wait();
            }
        },
    )
}
