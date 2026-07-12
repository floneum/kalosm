//! Liveness-based allocation reuse: an operation output claims the buffer
//! of a dead intermediate instead of allocating fresh.
//!
//! Claims are decided serially in queue order while the compute-graph
//! write lock is held for the whole resolve (no user thread can clone or
//! drop handles), so `Arc::strong_count` reads are stable and the claim
//! decision is a pure function of graph structure. A replayed plan
//! re-creates the same claims through `OutputSource::Alias`.
//!
//! Claim sources are restricted to buffers allocated during this resolve
//! (intermediates), and an operation never claims a buffer it reads:
//! in-place read-write claims need the kernel to fold the pair into one
//! read-write binding (wgpu rejects one buffer bound read-only and
//! read-write in the same dispatch), so they are declined here.
//!
//! The accounting is fail-safe by construction: every strong-reference
//! holder the resolver knows about is enumerated, and any *unaccounted*
//! holder makes the observed `Arc::strong_count` exceed the expectation, so
//! the claim is declined and a fresh buffer is allocated instead — a missed
//! registration can cause a missed optimization, never a wrong result.

use std::collections::VecDeque;

use super::*;

pub(super) struct BufferLedger {
    enabled: bool,
    /// Set while claims may still be made: release events feed the free
    /// list only until the claim window closes.
    accepting: bool,
    device: crate::Device,
    /// Shadow of the remaining-consumer accounting, advanced at gather time
    /// when releases run later than allocations (the batched dense queue,
    /// where every allocation happens before any release). Empty when the
    /// real release runs in step with allocation and feeds the free list
    /// directly via [`Self::note_released`].
    shadow: FxHashMap<NodeIndex, usize>,
    /// Dead intermediate buffers by (size, usage), in death order. One entry
    /// per buffer pointer (the entry holds its own strong clone).
    free: FxHashMap<(u64, wgpu::BufferUsages), VecDeque<(NodeIndex, Arc<wgpu::Buffer>)>>,
    /// Buffer pointers currently queued in `free`.
    queued: FxHashSet<usize>,
    /// ptr -> lingering `cached` clones of dead graph nodes (phase 3 has not
    /// released them yet).
    dead_cached: FxHashMap<usize, u32>,
    /// ptr -> phase-1 resolver clones registered so far (work-item inputs and
    /// outputs, view results, copy records).
    transient: FxHashMap<usize, u32>,
    /// Buffers allocated during this resolve's phase 1.
    allocated_here: FxHashSet<usize>,
    /// Claimer node -> source node: the recorder's sole authority for
    /// classifying an output-provenance hit as a chosen alias.
    chosen: FxHashMap<NodeIndex, NodeIndex>,
    /// ptr -> strong clones held by the flush-plan recorder (boundary pins
    /// taken at recorder construction).
    recorder_pins: FxHashMap<usize, u32>,
    pub(super) claims: usize,
}

impl BufferLedger {
    pub(super) fn new(
        device: &crate::Device,
        shadow_consumers: Option<&FxHashMap<NodeIndex, usize>>,
    ) -> Self {
        let enabled = !device.poisons_allocations();
        Self {
            enabled,
            accepting: enabled,
            device: device.clone(),
            shadow: match shadow_consumers {
                Some(consumers) if enabled => consumers.clone(),
                _ => FxHashMap::default(),
            },
            free: FxHashMap::default(),
            queued: FxHashSet::default(),
            dead_cached: FxHashMap::default(),
            transient: FxHashMap::default(),
            allocated_here: FxHashSet::default(),
            chosen: FxHashMap::default(),
            recorder_pins: FxHashMap::default(),
            claims: 0,
        }
    }

    /// Register strong clones an armed flush-plan recorder holds, so claimed
    /// buffers it pinned still account exactly.
    pub(super) fn register_recorder_pins(&mut self, pins: impl Iterator<Item = usize>) {
        if !self.enabled {
            return;
        }
        for ptr in pins {
            *self.recorder_pins.entry(ptr).or_insert(0) += 1;
        }
    }

    pub(super) fn enabled(&self) -> bool {
        self.enabled
    }

    /// Close the claim window: later release events no longer feed the free
    /// list (their buffers can no longer be claimed by anything).
    pub(super) fn freeze(&mut self) {
        self.accepting = false;
    }

    /// A dead node's cached buffer is about to be released: make it
    /// claimable if it was allocated during this resolve.
    pub(super) fn note_released(&mut self, source: NodeIndex, cached: &TensorData) {
        if !self.enabled || !self.accepting {
            return;
        }
        let buffer = cached.buffer();
        let ptr = Arc::as_ptr(buffer) as usize;
        if !self.allocated_here.contains(&ptr) || !self.queued.insert(ptr) {
            return;
        }
        self.free
            .entry((buffer.size(), buffer.usage()))
            .or_default()
            .push_back((source, buffer.clone()));
    }

    /// The source node this claimer's output aliases, if any.
    pub(super) fn chosen_source(&self, claimer: NodeIndex) -> Option<NodeIndex> {
        self.chosen.get(&claimer).copied()
    }

    /// Register a buffer allocated during this phase 1.
    pub(super) fn note_alloc(&mut self, data: &TensorData) {
        if self.enabled {
            self.allocated_here
                .insert(Arc::as_ptr(data.buffer()) as usize);
        }
    }

    /// Register one resolver-held clone (work-item input/output, view
    /// result, copy record) of `buffer`.
    pub(super) fn note_transient(&mut self, buffer: &Arc<wgpu::Buffer>) {
        if self.enabled {
            *self
                .transient
                .entry(Arc::as_ptr(buffer) as usize)
                .or_insert(0) += 1;
        }
    }

    /// Advance the shadow release accounting for one produced node's
    /// dependencies (mirror of `release_dead_intermediates`, without the
    /// release): nodes whose last consumer this is enter the free list when
    /// their buffer was allocated this resolve.
    pub(super) fn consume(
        &mut self,
        graph: &ComputeGraphInner,
        deps: &[NodeIndex],
        targets: &FxHashSet<NodeIndex>,
    ) {
        if !self.enabled || !self.accepting {
            return;
        }
        for &dep in deps {
            let Some(count) = self.shadow.get_mut(&dep) else {
                continue;
            };
            *count = count.saturating_sub(1);
            if *count != 0 || targets.contains(&dep) || graph.has_live_lazy_descendant(dep) {
                continue;
            }
            let Some(cached) = graph.get_cached_result(dep) else {
                continue;
            };
            let buffer = cached.buffer();
            let ptr = Arc::as_ptr(buffer) as usize;
            // Every dead node's lingering `cached` clone counts, even when
            // several dead nodes (a view and its base) share one buffer.
            *self.dead_cached.entry(ptr).or_insert(0) += 1;
            if !self.allocated_here.contains(&ptr) || !self.queued.insert(ptr) {
                continue;
            }
            self.free
                .entry((buffer.size(), buffer.usage()))
                .or_default()
                .push_back((dep, buffer.clone()));
        }
    }

    /// Claim a dead intermediate's buffer for `claimer`'s output, or `None`
    /// to allocate fresh. `forbidden` is the set of buffer pointers the
    /// claiming operation reads (in-place claims need binding folding in
    /// the kernel, so they are declined here).
    pub(super) fn try_claim(
        &mut self,
        claimer: NodeIndex,
        output: &TensorData,
        forbidden: &FxHashSet<usize>,
    ) -> Option<TensorData> {
        if !self.enabled {
            return None;
        }
        let key = (output.buffer().size(), output.buffer().usage());
        let candidates = self.free.get_mut(&key)?;
        let mut picked = None;
        for (index, (source, buffer)) in candidates.iter().enumerate() {
            let ptr = Arc::as_ptr(buffer) as usize;
            if forbidden.contains(&ptr) {
                continue;
            }
            // Expected holders: dead graph clones + this free-list entry +
            // registered phase-1 transients + the pool's tracked clone.
            let expected = self.dead_cached.get(&ptr).copied().unwrap_or(0)
                + 1
                + self.transient.get(&ptr).copied().unwrap_or(0)
                + self.recorder_pins.get(&ptr).copied().unwrap_or(0)
                + u32::from(self.device.buffer_pool_is_tracked(key.0, key.1, buffer));
            if Arc::strong_count(buffer) as u32 != expected {
                continue;
            }
            picked = Some((index, *source));
            break;
        }
        let (index, source) = picked?;
        let (_, buffer) = candidates.remove(index).expect("index in range");
        self.queued.remove(&(Arc::as_ptr(&buffer) as usize));
        self.chosen.insert(claimer, source);
        self.claims += 1;
        Some(TensorData::new_from_parts(
            &self.device,
            buffer,
            output.layout().clone(),
            output.datatype(),
        ))
    }

    /// Claim a specific dead node's buffer for an output that will be
    /// written by the same dispatch that reads it. The caller guarantees
    /// the kernel-level safety conditions (identity-indexed reads, one
    /// read-write binding, no later reader of the source within the
    /// dispatch); this checks liveness and exact reference accounting.
    pub(super) fn try_claim_in_place(
        &mut self,
        claimer: NodeIndex,
        output: &TensorData,
        source: NodeIndex,
        graph: &ComputeGraphInner,
        targets: &FxHashSet<NodeIndex>,
    ) -> Option<TensorData> {
        if !self.enabled {
            return None;
        }
        // The source must be dead at this queue position: the shadow
        // accounting has already consumed the claiming operation's reads.
        if self.shadow.get(&source).copied().unwrap_or(usize::MAX) != 0
            || targets.contains(&source)
            || graph.has_live_lazy_descendant(source)
        {
            return None;
        }
        let cached = graph.get_cached_result(source)?;
        if cached.datatype() != output.datatype()
            || cached.layout() != output.layout()
            || cached.buffer().size() != output.buffer().size()
            || cached.buffer().usage() != output.buffer().usage()
        {
            return None;
        }
        let buffer = cached.buffer().clone();
        let ptr = Arc::as_ptr(&buffer) as usize;
        // One extra holder for the `buffer` clone taken just above; free-list
        // membership adds another.
        let expected = self.dead_cached.get(&ptr).copied().unwrap_or(0)
            + 1
            + u32::from(self.queued.contains(&ptr))
            + self.transient.get(&ptr).copied().unwrap_or(0)
            + self.recorder_pins.get(&ptr).copied().unwrap_or(0)
            + u32::from(
                self.device
                    .buffer_pool_is_tracked(buffer.size(), buffer.usage(), &buffer),
            );
        if Arc::strong_count(&buffer) as u32 != expected {
            return None;
        }
        // Retire any free-list entry so nothing else claims this buffer.
        if self.queued.remove(&ptr) {
            if let Some(entries) = self.free.get_mut(&(buffer.size(), buffer.usage())) {
                entries.retain(|(_, entry)| Arc::as_ptr(entry) as usize != ptr);
            }
        }
        self.chosen.insert(claimer, source);
        self.claims += 1;
        Some(TensorData::new_from_parts(
            &self.device,
            buffer,
            output.layout().clone(),
            output.datatype(),
        ))
    }
}
