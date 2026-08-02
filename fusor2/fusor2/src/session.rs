//! `Session` and `Device`. The session owns the target, the cost model, the
//! extractor and the plan cache; `resolve` is the one place saturation,
//! extraction and dispatch happen.
//!
//! Owned by W13.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use fusor2_cost::{LocalSearch, ReplayMemo, Roofline};
use fusor2_cpu::CpuTarget;
use fusor2_gpu::GpuTarget;
use fusor2_ir::CORE_RULES;
use fusor2_ir::cost::CostModel;
use fusor2_ir::device::Caps;
use fusor2_ir::dtype::Persistence;
use fusor2_ir::egraph::{EGraph, Id, Rule, Saturate, SaturationBudget};
use fusor2_ir::extract::{ExtractBudget, Extractor, Plan, ReplayKey};
use fusor2_ir::ir::level0::{L0, LeafKind};
use fusor2_ir::ir::level2::ArenaPlanner;
use fusor2_ir::ir::{Op, OpDefRegistry, Semantics};
use fusor2_ir::saturate::Driver;
use fusor2_ir::shape::Dim;
use fusor2_ir::target::{Buf, LowerCtx, Target, Uniforms};
use fusor2_tile::{Planner, SCHED_RULES};
use rustc_hash::FxHashMap;

use crate::composite::register_macro_ops;
use crate::graph::GraphRef;
use crate::tensor::Tensor;
use crate::{Error, Result};

/// Submitted-but-unretired plans the session will let pile up before it
/// blocks in `resolve`. Back-pressure is a **runtime policy the library
/// owns** — the trainer's `--drain-every` counter disappears.
pub const MAX_INFLIGHT_PLANS: u32 = 8;

/// Proof that the holder owns a graph's `resolve_lock`.
///
/// Passed rather than taken by the `_locked` entry points, so "the caller
/// already holds it" is checked by the borrow checker instead of by comment.
pub(crate) type ResolveGuard<'a> = parking_lot::MutexGuard<'a, ()>;

/// Which backend a session runs on.
#[derive(Clone)]
pub enum Device {
    Cpu(Arc<CpuTarget>),
    Gpu(Arc<GpuTarget>),
}

impl Device {
    pub fn cpu() -> Result<Self> {
        Ok(Self::Cpu(Arc::new(CpuTarget::new()?)))
    }

    pub async fn gpu() -> Result<Self> {
        Ok(Self::Gpu(Arc::new(GpuTarget::new().await?)))
    }

    pub fn gpu_blocking() -> Result<Self> {
        Ok(Self::Gpu(Arc::new(GpuTarget::new_blocking()?)))
    }

    pub fn target(&self) -> Arc<dyn Target> {
        match self {
            Self::Cpu(t) => Arc::clone(t) as Arc<dyn Target>,
            Self::Gpu(t) => Arc::clone(t) as Arc<dyn Target>,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu(_) => "cpu",
            Self::Gpu(_) => "gpu",
        }
    }

    pub fn caps(&self) -> Caps {
        match self {
            Self::Cpu(t) => t.caps().clone(),
            Self::Gpu(t) => t.caps().clone(),
        }
    }

    /// Upload host bytes into a fresh device buffer.
    ///
    /// `Target` deliberately exposes only `alloc`: a write path would have to
    /// name a byte layout, and only the backend knows one. This is the one
    /// place the session distinguishes the two backends, and it distinguishes
    /// them on *upload* — never on lowering, cost or selection.
    fn upload(&self, bytes: &[u8], persistence: Persistence) -> Result<Buf> {
        match self {
            Self::Gpu(t) => t
                .pool()
                .create_buffer_init(bytes, fusor2_gpu::pool::TENSOR_USAGE),
            Self::Cpu(t) => {
                let buf = t.alloc(bytes.len().max(4) as u64, persistence)?;
                let aligned = buf
                    .downcast_ref::<fusor2_cpu::AlignedBuf>()
                    .ok_or_else(|| Error::Device("cpu buffer was not an AlignedBuf".into()))?;
                // `AlignedBuf::as_mut_ptr` is the backend's own interior-mutable
                // handle; the buffer was just allocated, so nothing else reads it.
                let dst =
                    unsafe { std::slice::from_raw_parts_mut(aligned.as_mut_ptr(), aligned.len()) };
                dst[..bytes.len()].copy_from_slice(bytes);
                Ok(buf)
            }
        }
    }

    /// Copy a device buffer back to the host. One of exactly three host syncs.
    fn download(&self, buf: &Buf, bytes: u64) -> Result<Vec<u8>> {
        match self {
            Self::Gpu(t) => pollster::block_on(t.readback(buf, bytes)),
            Self::Cpu(_) => {
                let aligned = buf
                    .downcast_ref::<fusor2_cpu::AlignedBuf>()
                    .ok_or_else(|| Error::Device("cpu buffer was not an AlignedBuf".into()))?;
                let len = (bytes as usize).min(aligned.len());
                Ok(aligned.as_slice()[..len].to_vec())
            }
        }
    }
}

/// One device, one cost model, one extractor, one plan cache.
#[derive(Clone)]
pub struct Session {
    inner: Arc<SessionInner>,
}

pub struct SessionInner {
    pub device: Device,
    pub cost: Arc<dyn CostModel>,
    pub extractor: Arc<dyn Extractor>,
    pub planner: Arc<dyn ArenaPlanner>,
    pub registry: OpDefRegistry,
    semantics: Arc<dyn Semantics>,
    rules: Vec<Rule>,
    replay: ReplayMemo,
    launches: AtomicU64,
    in_flight: AtomicU32,
}

impl Session {
    pub fn new(device: Device) -> Result<Self> {
        let planner = Planner::shared();

        // The one registration point. Ids follow table order because
        // `PlanHash` reads registration order.
        let mut registry = OpDefRegistry::new();
        register_macro_ops(&mut registry);

        let semantics =
            fusor2_ir::CoreSemantics::with_registry(Arc::clone(&planner), registry.clone());
        let target = device.target();
        let cost: Arc<dyn CostModel> = Arc::new(Roofline::new(target.facts().clone()));
        let extractor: Arc<dyn Extractor> = Arc::new(
            LocalSearch::new(Arc::clone(&planner), target.caps().clone())
                .with_registry(registry.clone()),
        );

        // Rule order carries no semantics; the fixed order exists only for
        // reproducibility.
        let mut rules: Vec<Rule> = Vec::new();
        rules.extend_from_slice(CORE_RULES);
        rules.extend_from_slice(SCHED_RULES);
        rules.extend_from_slice(fusor2_autograd::ADJOINT_RULES);
        rules.extend_from_slice(target.rules());

        Ok(Self {
            inner: Arc::new(SessionInner {
                device,
                cost,
                extractor,
                planner,
                registry,
                semantics,
                rules,
                replay: ReplayMemo::new(),
                launches: AtomicU64::new(0),
                in_flight: AtomicU32::new(0),
            }),
        })
    }

    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    pub fn caps(&self) -> Caps {
        self.inner.device.caps()
    }

    pub fn semantics(&self) -> Arc<dyn Semantics> {
        Arc::clone(&self.inner.semantics)
    }

    pub fn registry(&self) -> &OpDefRegistry {
        &self.inner.registry
    }

    pub fn rules(&self) -> &[Rule] {
        &self.inner.rules
    }

    /// Saturate, extract, lower, emit and dispatch everything `values` needs.
    ///
    /// Atomic against every other resolve and readback on the same graph — see
    /// [`crate::graph::GraphInner::resolve_lock`] for why the e-graph's own
    /// mutex cannot do that job.
    pub fn resolve(&self, values: &[Tensor]) -> Result<()> {
        let Some(first) = values.first() else {
            return Ok(());
        };
        let graph = first.graph.clone();
        let resolving = graph.resolve_lock.lock();
        self.resolve_locked(&resolving, values)
    }

    /// [`Self::resolve`]'s body, for a caller already holding the graph's
    /// `resolve_lock` — `read_back` does, so that a resolve and the readback
    /// that follows it are one section.
    pub(crate) fn resolve_locked(
        &self,
        _resolving: &ResolveGuard<'_>,
        values: &[Tensor],
    ) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        let graph = values[0].graph.clone();
        for v in values {
            if !Arc::ptr_eq(&v.graph, &graph) {
                return Err(Error::Device(
                    "operands come from two different graphs".into(),
                ));
            }
        }

        // Own the back-pressure: the trainer's `--drain-every` is a runtime
        // policy, not a counter in a training script.
        if self.inner.in_flight.load(Ordering::Relaxed) >= MAX_INFLIGHT_PLANS {
            self.wait()?;
        }

        let caps = self.caps();
        let binding: Vec<Dim> = graph
            .dim_bindings()
            .into_iter()
            .map(|(_, v)| Dim::Const(v))
            .collect();

        let plan = {
            let mut g = graph.egraph.lock();
            for v in values {
                g.add_root(v.id);
            }
            Driver::new().saturate(&mut g, &caps, &self.inner.rules, SaturationBudget::default())?;

            let roots: Vec<Id> = g.roots().to_vec();
            let key = ReplayKey {
                l0_term: fusor2_cost::replay::l0_term_hash(&g, &roots),
                device: self.inner.cost.facts().fingerprint(),
                binding: fusor2_cost::replay::binding_hash(&binding),
            };
            let graph_ref: &EGraph = &g;
            let (plan, _unchanged) = self.inner.replay.get_or_extract(key, || {
                self.inner.extractor.extract(
                    graph_ref,
                    &roots,
                    self.inner.cost.as_ref(),
                    ExtractBudget::default(),
                )
            })?;
            self.inner.extractor.verify_plan(graph_ref, &plan)?;
            plan
        };

        self.run(&graph, &plan, values)?;
        self.inner
            .launches
            .fetch_add(plan.launches.len() as u64, Ordering::Relaxed);
        self.inner.in_flight.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Submit whatever is encoded without waiting.
    ///
    /// Every backend encodes and submits inside `launch`, so there is no
    /// deferred encoder to push. This exists so a caller need not know that.
    pub fn flush(&self) -> Result<()> {
        Ok(())
    }

    /// Block until every submitted dispatch has retired.
    pub fn wait(&self) -> Result<()> {
        self.inner.device.target().wait()?;
        self.inner.in_flight.store(0, Ordering::Relaxed);
        Ok(())
    }

    /// Dispatches issued since construction. The conformance
    /// `resolves_in::<N>` asserts read this, so it counts **dispatches**, not
    /// encoder submissions.
    pub fn launch_count(&self) -> u64 {
        self.inner.launches.load(Ordering::Relaxed)
    }

    /// Resolve `values` and report whether it took exactly `N` dispatches.
    ///
    /// The two counter reads bracket the resolve *inside* the graph's
    /// resolve lock, so the difference is this call's own dispatches even
    /// when another thread is resolving the same graph. Counting outside it
    /// would attribute the other thread's launches to this assert.
    pub fn resolves_in<const N: u64>(&self, values: &[Tensor]) -> Result<bool> {
        let Some(first) = values.first() else {
            return Ok(N == 0);
        };
        let graph = first.graph.clone();
        let resolving = graph.resolve_lock.lock();
        let before = self.launch_count();
        self.resolve_locked(&resolving, values)?;
        Ok(self.launch_count() - before == N)
    }

    /// Bytes of an already-resolved value.
    ///
    /// `_resolving` is a witness that the caller holds the graph's
    /// `resolve_lock`, not a parameter this reads. Downloading a buffer is
    /// only meaningful while no other thread can be part-way through binding
    /// and dispatching a plan that writes it, and taking the guard here
    /// instead would leave a window open between a caller's `resolve` and its
    /// read. Passing the guard makes "I already hold it" the only way to
    /// call this.
    pub(crate) fn read_bytes_locked(
        &self,
        _resolving: &ResolveGuard<'_>,
        graph: &GraphRef,
        id: Id,
    ) -> Result<Vec<u8>> {
        let buf = graph
            .device_buf(id)
            .ok_or_else(|| Error::Plan(format!("{id} has no device buffer; resolve it first")))?;
        let facts = graph.facts(id);
        let elem = facts.dtype.byte_size();
        self.wait()?;

        // A selected `Coop`/`Sgemm` geometry pads the output buffer to its
        // tile multiple, so the bytes on the device are *not* the value's own
        // dense shape. Read the whole padded buffer and gather the value out
        // of it; a dense layout takes the straight path.
        let padded = graph
            .device_layout(id)
            .filter(|l| l.rank() == facts.shape.len() && l.shape() != &facts.shape[..]);
        let Some(layout) = padded else {
            let elements = resolve_elements(&facts.shape, graph)?;
            return self.inner.device.download(&buf, elements * elem);
        };

        let base = resolve_dim(layout.offset(), graph)?;
        let stored = resolve_elements(layout.shape(), graph)?;
        let raw = self.inner.device.download(&buf, (base + stored) * elem)?;
        let strides: Vec<u64> = layout
            .strides()
            .iter()
            .map(|d| resolve_dim(*d, graph))
            .collect::<Result<_>>()?;
        let extents: Vec<u64> = facts
            .shape
            .iter()
            .map(|d| resolve_dim(*d, graph))
            .collect::<Result<_>>()?;
        let count = extents.iter().product::<u64>() as usize;
        let mut out = Vec::with_capacity(count * elem as usize);
        let mut idx = vec![0u64; extents.len()];
        for _ in 0..count {
            let flat = base
                + idx
                    .iter()
                    .zip(&strides)
                    .map(|(i, s)| i * s)
                    .sum::<u64>();
            let start = (flat * elem) as usize;
            let end = start + elem as usize;
            match raw.get(start..end) {
                Some(slice) => out.extend_from_slice(slice),
                None => out.extend(std::iter::repeat_n(0u8, elem as usize)),
            }
            for axis in (0..extents.len()).rev() {
                idx[axis] += 1;
                if idx[axis] < extents[axis] {
                    break;
                }
                idx[axis] = 0;
            }
        }
        Ok(out)
    }

    // -----------------------------------------------------------------
    // Running a plan
    // -----------------------------------------------------------------

    /// The class member this plan selected for `id`.
    ///
    /// `Extraction::sigma` is `ClassId -> Id`, so it only runs one way: from
    /// the class a value belongs to, to the member the extractor picked. The
    /// facade holds the `L0` id the user built; the plan names the selected
    /// member. Those diverge the moment any rewrite fires, which is always.
    fn selected(&self, graph: &GraphRef, plan: &Plan, id: Id) -> Id {
        let class = graph.egraph.lock().class_of(id);
        plan.extraction.selected(class).unwrap_or(id)
    }

    /// Register `buf` under **every** id in `id`'s e-class, `Union` spine
    /// included.
    ///
    /// Registering only the selected member is what left `read_bytes` looking
    /// in a map its key was never inserted into. The class — not the member —
    /// is the stable identity of a value: `sigma` is keyed by it, and which
    /// member wins is an artifact of one extraction that a later resolve may
    /// change. Writing the whole class also means a later extraction that
    /// selects a different member overwrites every stale entry instead of
    /// leaving one to shadow.
    ///
    /// `EGraph::members` is the *selectable* set and drops the `Union` nodes,
    /// but `macro_op` hands the caller the id `union(defn, sugar)` returned —
    /// so every sugared spelling (`softmax`, `rms_norm`, `rope`, `attention`,
    /// every windowed view) names its value by a spine node. Binding only the
    /// members left all of those unreadable.
    fn bind_class(
        &self,
        graph: &GraphRef,
        id: Id,
        buf: &Buf,
        layout: Option<&fusor2_ir::shape::Layout>,
    ) {
        let members = {
            let g = graph.egraph.lock();
            g.class_ids(g.class_of(id))
        };
        for m in members {
            match layout {
                Some(l) => graph.set_device_buf_with_layout(m, buf.clone(), l.clone()),
                None => graph.set_device_buf(m, buf.clone()),
            }
        }
    }

    fn run(&self, graph: &GraphRef, plan: &Plan, values: &[Tensor]) -> Result<()> {
        // What the extractor selected for each requested value. `sigma` is
        // keyed by `ClassId`, so this is the only direction it runs in, and
        // the id the user's `Tensor` holds is generally *not* it.
        let wanted: Vec<Id> = values
            .iter()
            .map(|v| self.selected(graph, plan, v.id))
            .collect();
        // Every external leaf the plan reads, uploaded once. A `Persistent`
        // leaf keeps its buffer across resolves, which is what makes an
        // optimizer's state stay on device with no host round trip.
        let mut supplied: FxHashMap<Id, Buf> = FxHashMap::default();
        for launch in &plan.launches {
            for binding in &launch.bindings {
                if supplied.contains_key(&binding.value) {
                    continue;
                }
                if let Some(buf) = self.leaf_buffer(graph, binding.value)? {
                    supplied.insert(binding.value, buf);
                }
            }
        }
        // Root outputs are allocated here rather than inside the backend so
        // the handle survives for readback.
        for buffer in &plan.buffers {
            if supplied.contains_key(&buffer.value) {
                continue;
            }
            let is_launch_root = plan.launches.iter().any(|l| l.root == buffer.value);
            if !is_launch_root && !wanted.contains(&buffer.value) {
                continue;
            }
            let elements = resolve_dim(buffer.elements, graph)?;
            let bytes = (elements * buffer.dtype.byte_size()).max(4);
            let buf = self.inner.device.target().alloc(bytes, buffer.persistence)?;
            self.bind_class(graph, buffer.value, &buf, Some(&buffer.layout));
            supplied.insert(buffer.value, buf);
        }
        // A value the caller asked for that the plan never had to allocate: a
        // bare leaf, or a leaf that only ever appears as an operand. Reading
        // one back is still a legal request, and a graph of nothing but a leaf
        // has no launches at all.
        for (value, selected) in values.iter().zip(&wanted) {
            if graph.device_buf(value.id).is_some() {
                continue;
            }
            let buf = match supplied.get(selected) {
                Some(buf) => buf.clone(),
                None => match self.free_leaf_buffer(graph, *selected)? {
                    Some(buf) => {
                        supplied.insert(*selected, buf.clone());
                        buf
                    }
                    None => continue,
                },
            };
            self.bind_class(graph, *selected, &buf, None);
        }

        match &self.inner.device {
            Device::Gpu(target) => {
                let mut env = fusor2_gpu::target::BindingEnv::new();
                for (sym, value) in graph.dim_bindings() {
                    env = env.with_dim(sym, value);
                }
                for (sym, value) in graph.uniform_scalars() {
                    env = env.with_scalar(sym, value);
                }
                for (id, buf) in &supplied {
                    env = env.with_buffer(*id, buf.clone());
                }
                let g = graph.egraph.lock();
                target.resolve(plan, &g, &env)
            }
            Device::Cpu(target) => {
                let target = Arc::clone(target);
                self.run_cpu(target.as_ref(), graph, plan, &mut supplied)
            }
        }
    }

    /// The generic runner: one `lower -> emit -> launch` per plan launch, in
    /// plan order. The GPU takes `GpuTarget::resolve` instead, which adds the
    /// plan cache, the parallel build cohort and one encoder per resolve.
    fn run_cpu(
        &self,
        target: &CpuTarget,
        graph: &GraphRef,
        plan: &Plan,
        supplied: &mut FxHashMap<Id, Buf>,
    ) -> Result<()> {
        let uniforms = self.uniforms_for(plan, graph)?;

        for buffer in &plan.buffers {
            if supplied.contains_key(&buffer.value) {
                continue;
            }
            let elements = resolve_dim(buffer.elements, graph)?;
            let bytes = (elements * buffer.dtype.byte_size()).max(4);
            supplied.insert(buffer.value, target.alloc(bytes, buffer.persistence)?);
        }

        let g = graph.egraph.lock();
        for launch in &plan.launches {
            let theta = plan
                .extraction
                .theta
                .get(&launch.root)
                .copied()
                .unwrap_or(fusor2_ir::ir::level1::SchedPoint::Point);
            let cx = LowerCtx {
                plan,
                launch,
                graph: &g,
                symbols: &plan.symbols,
            };
            let ir = target.lower(g.node(launch.root), launch.root, theta, &cx)?;
            let artifact = target.emit(&ir)?;

            let mut ordered: Vec<_> = launch.bindings.iter().collect();
            ordered.sort_by_key(|b| b.binding);
            let mut binds = Vec::with_capacity(ordered.len());
            for b in ordered {
                let buf = supplied.get(&b.value).cloned().ok_or_else(|| {
                    Error::Plan(format!("launch binds {} which nothing allocates", b.value))
                })?;
                binds.push(buf);
            }
            // **The kernel's own grid, not the plan's.** `Launch::grid` is the
            // cost model's workgroup count, derived from the schedule point;
            // `KernelIr::grid` is what the lowering actually indexed the body
            // against. When they disagree the kernel silently computes a
            // prefix of its output.
            target.launch(&artifact, ir.grid, &binds, &uniforms)?;
        }
        Ok(())
    }

    /// Binding 0 for the CPU launcher, indexed by raw `SymId`: a dim symbol
    /// contributes its extent, a scalar symbol its `f32` bits, and the
    /// emitter bitcasts on read. Neither ever enters a kernel's identity.
    fn uniforms_for(&self, plan: &Plan, graph: &GraphRef) -> Result<Uniforms> {
        let dims = graph.dim_bindings();
        let scalars = graph.uniform_scalars();
        let highest = plan
            .symbols
            .iter()
            .map(|s| s.0)
            .chain(dims.iter().map(|(s, _)| s.0))
            .chain(scalars.iter().map(|(s, _)| s.0))
            .filter(|s| *s != u32::MAX)
            .max()
            .map(|m| m as usize + 1)
            .unwrap_or(1);
        let mut words = vec![0u32; highest.max(1)];
        for (sym, value) in dims {
            words[sym.0 as usize] = u32::try_from(value)
                .map_err(|_| Error::Plan(format!("extent {value} exceeds a u32")))?;
        }
        for (sym, value) in scalars {
            words[sym.0 as usize] = value.to_bits();
        }
        Ok(Uniforms {
            dims: words,
            scalars: Vec::new(),
        })
    }

    /// The buffer backing a leaf the caller asked to read back.
    ///
    /// An external leaf uploads its host bytes. A `Const` splat and a
    /// `Uniform` scalar have **no buffer by design** — a constant is folded
    /// into the kernel and a uniform is a word in binding 0 — but reading one
    /// back is still a legal request (a comparison's adjoint *is* a splat of
    /// zero), so their bytes are materialized here rather than reported as a
    /// missing buffer.
    fn free_leaf_buffer(&self, graph: &GraphRef, id: Id) -> Result<Option<Buf>> {
        if let Some(buf) = self.leaf_buffer(graph, id)? {
            return Ok(Some(buf));
        }
        let leaf = {
            let g = graph.egraph.lock();
            match &g.node(id).op {
                Op::L0(L0::Leaf(k @ (LeafKind::Const { .. } | LeafKind::Uniform { .. }))) => {
                    k.clone()
                }
                _ => return Ok(None),
            }
        };
        let facts = graph.facts(id);
        let unit = match &leaf {
            LeafKind::Const { value, .. } => splat_bytes(*value),
            LeafKind::Uniform { sym, .. } => {
                splat_bytes(fusor2_autograd::tape::splat_of(
                    facts.dtype,
                    graph.uniform_value(*sym).unwrap_or(0.0),
                )?)
            }
            _ => return Ok(None),
        };
        let elements = resolve_elements(&facts.shape, graph)? as usize;
        let mut bytes = Vec::with_capacity(elements * unit.len());
        for _ in 0..elements {
            bytes.extend_from_slice(&unit);
        }
        let buf = self.inner.device.upload(&bytes, facts.persistence)?;
        graph.set_device_buf(id, buf.clone());
        Ok(Some(buf))
    }

    /// The device buffer of an external leaf, uploading it on first use.
    fn leaf_buffer(&self, graph: &GraphRef, id: Id) -> Result<Option<Buf>> {
        let is_external = {
            let g = graph.egraph.lock();
            matches!(
                &g.node(id).op,
                Op::L0(L0::Leaf(
                    LeafKind::Buffer { .. } | LeafKind::Param { .. } | LeafKind::Quantized { .. }
                ))
            )
        };
        if !is_external {
            return Ok(None);
        }
        if let Some(buf) = graph.device_buf(id) {
            return Ok(Some(buf));
        }
        let facts = graph.facts(id);
        let buf = match graph.leaf_bytes(id) {
            Some(bytes) => self.inner.device.upload(&bytes, facts.persistence)?,
            None => {
                let elements = resolve_elements(&facts.shape, graph)?;
                let bytes = (elements * facts.dtype.byte_size()).max(4);
                self.inner.device.target().alloc(bytes, facts.persistence)?
            }
        };
        graph.set_device_buf(id, buf.clone());
        Ok(Some(buf))
    }
}

/// One element's little-endian bytes, in the splat's own dtype.
fn splat_bytes(s: fusor2_ir::dtype::Splat) -> Vec<u8> {
    use fusor2_ir::dtype::Splat;
    match s {
        Splat::F32(v) => v.to_le_bytes().to_vec(),
        Splat::F16(v) | Splat::BF16(v) => v.to_le_bytes().to_vec(),
        Splat::U32(v) => v.to_le_bytes().to_vec(),
        Splat::I32(v) => v.to_le_bytes().to_vec(),
    }
}

fn resolve_dim(d: Dim, graph: &GraphRef) -> Result<u64> {
    match d {
        Dim::Const(v) => Ok(v),
        Dim::Sym(s) => graph
            .dim_binding(s)
            .ok_or_else(|| Error::Plan(format!("dim {s} is unbound at dispatch"))),
    }
}

fn resolve_elements(shape: &[Dim], graph: &GraphRef) -> Result<u64> {
    let mut acc = 1u64;
    for d in shape {
        acc = acc.saturating_mul(resolve_dim(*d, graph)?);
    }
    Ok(acc.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cpu_session_registers_every_macro_op_in_table_order() {
        let s = Session::new(Device::cpu().unwrap()).unwrap();
        for op in crate::composite::MacroOp::ALL {
            assert_eq!(s.registry().get(op.def_id()).unwrap().name, op.name());
        }
    }

    #[test]
    fn the_rule_table_is_the_union_of_every_contributor() {
        let s = Session::new(Device::cpu().unwrap()).unwrap();
        let expected = CORE_RULES.len()
            + SCHED_RULES.len()
            + fusor2_autograd::ADJOINT_RULES.len()
            + fusor2_cpu::CPU_RULES.len();
        assert_eq!(s.rules().len(), expected);
    }

    #[test]
    fn back_pressure_is_a_library_policy() {
        assert_eq!(MAX_INFLIGHT_PLANS, 8);
        let s = Session::new(Device::cpu().unwrap()).unwrap();
        assert_eq!(s.launch_count(), 0);
        s.wait().unwrap();
    }
}
