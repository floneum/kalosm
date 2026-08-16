//! [`Planner`] — the one [`ArenaPlanner`] implementation.
//!
//! `arena_plan` is a **pure memoized function** of the ordered tile
//! declaration list, the barrier and loop structure of the body, and
//! `caps.fingerprint()`. `workgroup_bytes` synthesizes a minimal body over a
//! candidate geometry's tiles and runs the *same* function, so `verify_launch`'s
//! admission test, the Launch occupancy term and the Kernel emitter's layout are
//! provably the same number.

use std::sync::{Arc, OnceLock};

use fusor2_ir::Result;
use fusor2_ir::device::Caps;
use fusor2_ir::ir::kernel::{
    Addr, ArenaMode, ArenaPlan, ArenaPlanner, BarrierSuggestion, Buffer, CoopSrc, KernelIr, Local,
    ElementType, MergeBody, QuantizedView, ReduceKind, ScalarElement, Source, Stmt, StorageView,
    Tile, TileExpr, TileExprKind, TileLiteral, Tiles,
};
use parking_lot::RwLock;
use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};

use crate::arena;
use crate::liveness::{LivenessInfo, analyze, for_each_addr_expr, for_each_child};

/// Memo key: everything `arena_plan`'s result depends on.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct PlanKey {
    /// The tile declaration list, the barrier/loop skeleton, and the body
    /// term itself. The body term is a superset of the first two, which keeps
    /// the key exact at the cost of a few memo misses.
    pub body_hash: u64,
    pub caps_fingerprint: u64,
}

/// Memo key for the geometry-only entry.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TilesKey {
    pub tiles_hash: u64,
    pub caps_fingerprint: u64,
}

/// The shared arena planner. The memo is behind an `RwLock` because kernel
/// building runs on worker threads.
#[derive(Default)]
pub struct Planner {
    memo: RwLock<FxHashMap<PlanKey, ArenaPlan>>,
    tiles_memo: RwLock<FxHashMap<TilesKey, u32>>,
}

static GLOBAL: OnceLock<Planner> = OnceLock::new();

impl Planner {
    pub fn new() -> Self {
        Self::default()
    }

    /// The handle `CoreSemantics::new` wants.
    pub fn shared() -> Arc<dyn ArenaPlanner> {
        Arc::new(Self::new())
    }

    /// The process-wide planner, so `verify_kernel` and the emitters share one
    /// memo instead of re-deriving every plan.
    pub fn global() -> &'static Self {
        GLOBAL.get_or_init(Self::new)
    }

    pub fn memo_len(&self) -> usize {
        self.memo.read().len()
    }

    /// True when `(ir, caps)` is already planned. Test-facing.
    pub fn is_memoized(&self, ir: &KernelIr, caps: &Caps) -> bool {
        let live = analyze(ir);
        self.memo.read().contains_key(&plan_key(ir, &live, caps))
    }
}

/// Everything the plan depends on, hashed. The liveness digest is the
/// architecture's stated key (tile declarations in order, plus the barrier and
/// loop structure); the raw body term is folded in as well so a memo hit can
/// never cross two bodies that merely happen to have the same skeleton.
fn plan_key(ir: &KernelIr, live: &LivenessInfo, caps: &Caps) -> PlanKey {
    let mut h = FxHasher::default();
    // Ordered tile declaration list: element type, layout, allocation extent.
    (live.order.len() as u64).hash(&mut h);
    for tile in live.iter() {
        tile.element.hash(&mut h);
        tile.tile.layout.hash(&mut h);
        tile.elements.hash(&mut h);
        for access in &tile.accesses {
            access.position.hash(&mut h);
            access.kind.hash(&mut h);
        }
    }
    // Barrier and loop structure.
    for barrier in &live.barriers {
        barrier.position.hash(&mut h);
        barrier.guaranteed.hash(&mut h);
    }
    for info in &live.loops {
        info.span.hash(&mut h);
        info.guaranteed_once().hash(&mut h);
    }
    // The body term itself — barrier insertion candidates depend on the root
    // statement list, not only on the skeleton above.
    //
    // Hashed structurally, not via `Stmt`'s derived `Hash`: `StorageView`
    // hashes the buffer's address and `LocalDecl`/`TileDecl` hash their `id`,
    // so two separately-built copies of the same kernel would never agree.
    // `BodyHasher` substitutes each declaration's first-use ordinal for its
    // address and is otherwise exact.
    BodyHasher::default().body(&ir.body, &mut h);
    PlanKey {
        body_hash: h.finish(),
        caps_fingerprint: caps.fingerprint(),
    }
}

/// A pointer-free identity for a whole kernel: name, block, arena token,
/// declared buffers (by binding and contents) and the body up to renaming of
/// its buffer, tile and local declarations.
///
/// `TileExpr`'s cached digest — and therefore any hash derived from `Stmt` —
/// folds in `Arc` addresses: two byte-identical lowerings never agree, and a
/// recycled allocation can make two different kernels agree.
pub fn kernel_identity(ir: &KernelIr) -> u128 {
    let mut lanes = [0u64; 2];
    for (seed, lane) in lanes.iter_mut().enumerate() {
        let mut h = FxHasher::default();
        h.write_u64(seed as u64);
        ir.name.hash(&mut h);
        ir.block.hash(&mut h);
        ir.byte_arena.hash(&mut h);
        let mut bh = BodyHasher {
            seed: seed as u64,
            ..Default::default()
        };
        (ir.buffers.len() as u64).hash(&mut h);
        for b in &ir.buffers {
            b.binding.hash(&mut h);
            bh.buffer(b, &mut h);
        }
        bh.body(&ir.body, &mut h);
        *lane = h.finish();
    }
    (u128::from(lanes[0]) << 64) | u128::from(lanes[1])
}

#[derive(Default)]
struct BodyHasher {
    buffers: FxHashMap<usize, u32>,
    tiles: FxHashMap<usize, u32>,
    locals: FxHashMap<usize, u32>,
    /// Per-node sub-hash, keyed by [`TileExpr::node_ptr`]. A body is a DAG,
    /// so expanding it as a tree is exponential in the sharing depth; the
    /// memo makes the identity a Merkle fold, one hash per distinct node.
    ///
    /// Exact: ordinals are assigned on first visit and stable afterwards, so
    /// recomputing at the second occurrence would reproduce the memoized
    /// value. Multiplicity survives because the parent folds the sub-hash in
    /// once per edge.
    memo: FxHashMap<usize, u64>,
    /// Lane seed, mixed into every sub-hash so the two lanes of
    /// [`kernel_identity`] stay independent 64-bit hashes rather than
    /// agreeing on every shared subtree.
    seed: u64,
}

fn ptr_of<T>(v: &Arc<T>) -> usize {
    Arc::as_ptr(v) as *const () as usize
}

impl BodyHasher {
    fn ordinal(map: &mut FxHashMap<usize, u32>, ptr: usize) -> u32 {
        let next = map.len() as u32;
        *map.entry(ptr).or_insert(next)
    }

    fn buffer(&mut self, b: &Buffer, h: &mut FxHasher) {
        Self::ordinal(&mut self.buffers, ptr_of(b)).hash(h);
        // The decl's own contents still matter: two buffers may be distinct
        // allocations of different element types.
        b.element.hash(h);
        b.layout.hash(h);
        b.access.hash(h);
    }

    fn tile(&mut self, t: &Tile, h: &mut FxHasher) {
        Self::ordinal(&mut self.tiles, ptr_of(t)).hash(h);
        t.element.hash(h);
        t.layout.hash(h);
        t.name.hash(h);
    }

    fn local(&mut self, l: &Local, h: &mut FxHasher) {
        Self::ordinal(&mut self.locals, ptr_of(l)).hash(h);
        l.element.hash(h);
    }

    fn view(&mut self, v: &StorageView, h: &mut FxHasher) {
        self.buffer(&v.buffer, h);
        v.offset.hash(h);
        v.layout.hash(h);
    }

    fn quantized(&mut self, q: &QuantizedView, h: &mut FxHasher) {
        self.view(&q.data, h);
        q.fmt.hash(h);
        q.layout.hash(h);
    }

    fn source(&mut self, s: &Source, h: &mut FxHasher) {
        std::mem::discriminant(s).hash(h);
        match s {
            Source::Storage(v) => self.view(v, h),
            Source::Quantized(q) => self.quantized(q, h),
        }
    }

    fn addr(&mut self, a: &Addr, h: &mut FxHasher) {
        std::mem::discriminant(a).hash(h);
        for_each_addr_expr(a, &mut |e| self.expr(e, h));
    }

    fn reduce_kind(&mut self, k: &ReduceKind, h: &mut FxHasher) {
        std::mem::discriminant(k).hash(h);
        match k {
            ReduceKind::Subgroup => {}
            ReduceKind::Workgroup {
                scratch,
                group_size,
            } => {
                self.tile(scratch, h);
                group_size.hash(h);
            }
            ReduceKind::Loop {
                iterations,
                index,
                scratch,
                group_size,
            } => {
                iterations.hash(h);
                self.local(index, h);
                self.tile(scratch, h);
                group_size.hash(h);
            }
        }
    }

    /// The per-node payload: everything that is neither a child expression
    /// (walked by `for_each_child`) nor an identity already folded in above.
    /// Fold `e`'s identity into `h`, computing it once per distinct node.
    fn expr(&mut self, e: &TileExpr, h: &mut FxHasher) {
        let ptr = e.node_ptr();
        if let Some(cached) = self.memo.get(&ptr) {
            cached.hash(h);
            return;
        }
        let mut sub = FxHasher::default();
        sub.write_u64(self.seed);
        self.expr_uncached(e, &mut sub);
        let value = sub.finish();
        self.memo.insert(ptr, value);
        value.hash(h);
    }

    fn expr_uncached(&mut self, e: &TileExpr, h: &mut FxHasher) {
        let kind = e.kind();
        std::mem::discriminant(kind).hash(h);
        e.element().hash(h);
        match kind {
            TileExprKind::Literal(l) => l.hash(h),
            TileExprKind::Builtin(b) => b.hash(h),
            TileExprKind::LoadLocal(l) => self.local(l, h),
            TileExprKind::Load { src, .. } => self.source(src, h),
            TileExprKind::LoadTile { tile, .. } => self.tile(tile, h),
            TileExprKind::Unary { op, numeric, .. } => {
                op.hash(h);
                numeric.hash(h);
            }
            TileExprKind::Binary { op, numeric, .. } => {
                op.hash(h);
                numeric.hash(h);
            }
            TileExprKind::Compare { op, .. } => op.hash(h),
            TileExprKind::Round { mode, .. } => mode.hash(h),
            TileExprKind::Cast { to, .. } | TileExprKind::Bitcast { to, .. } => to.hash(h),
            TileExprKind::Vec { scalar, lanes, .. } => {
                scalar.hash(h);
                lanes.hash(h);
            }
            TileExprKind::VecComponent { component, .. } => component.hash(h),
            TileExprKind::Reduce { op, kind, .. } => {
                op.hash(h);
                self.reduce_kind(kind, h);
            }
            TileExprKind::CoopZero {
                role,
                scalar,
                rows,
                cols,
            } => {
                role.hash(h);
                scalar.hash(h);
                rows.hash(h);
                cols.hash(h);
            }
            TileExprKind::CoopLoad {
                role,
                scalar,
                rows,
                cols,
                src,
            } => {
                role.hash(h);
                scalar.hash(h);
                rows.hash(h);
                cols.hash(h);
                std::mem::discriminant(src.as_ref()).hash(h);
                match src.as_ref() {
                    CoopSrc::TileRegion {
                        tile, transposed, ..
                    } => {
                        self.tile(tile, h);
                        transposed.hash(h);
                    }
                    CoopSrc::BroadcastCol { src, .. } => self.view(src, h),
                }
            }
            // No payload beyond the children.
            TileExprKind::Select { .. } | TileExprKind::Dot { .. } | TileExprKind::CoopMma { .. } => {}
        }
        for_each_child(kind, &mut |c| self.expr(c, h));
    }

    fn merge(&mut self, m: &MergeBody, h: &mut FxHasher) {
        for l in m.lhs.iter().chain(m.rhs.iter()) {
            self.local(l, h);
        }
        for e in &m.body {
            self.expr(e, h);
        }
    }

    fn body(&mut self, stmts: &[Stmt], h: &mut FxHasher) {
        (stmts.len() as u64).hash(h);
        for s in stmts {
            self.stmt(s, h);
        }
    }

    fn stmt(&mut self, s: &Stmt, h: &mut FxHasher) {
        std::mem::discriminant(s).hash(h);
        match s {
            Stmt::Store {
                dst,
                addr,
                value,
                mask,
            }
            | Stmt::AtomicAdd {
                dst,
                addr,
                value,
                mask,
            } => {
                self.view(dst, h);
                self.addr(addr, h);
                self.expr(value, h);
                self.expr(mask, h);
            }
            Stmt::StoreLocal { dst, value } => {
                self.local(dst, h);
                self.expr(value, h);
            }
            Stmt::StoreTile { dst, index, value } => {
                self.tile(dst, h);
                self.expr(index, h);
                self.expr(value, h);
            }
            Stmt::FillTile { dst, value, bounds } => {
                self.tile(dst, h);
                self.expr(value, h);
                for b in bounds {
                    b.is_some().hash(h);
                    if let Some(b) = b {
                        self.expr(b, h);
                    }
                }
            }
            Stmt::CoopStore { acc, dst, addr } => {
                self.expr(acc, h);
                self.view(dst, h);
                self.addr(addr, h);
            }
            Stmt::CoopStoreTile {
                acc,
                tile,
                row,
                col,
            } => {
                self.expr(acc, h);
                self.tile(tile, h);
                self.expr(row, h);
                self.expr(col, h);
            }
            Stmt::If {
                condition,
                accept,
                reject,
            } => {
                self.expr(condition, h);
                self.body(accept, h);
                self.body(reject, h);
            }
            Stmt::Loop {
                count,
                index,
                accumulators,
                body,
            } => {
                count.is_some().hash(h);
                if let Some(c) = count {
                    self.expr(c, h);
                }
                index.is_some().hash(h);
                if let Some(i) = index {
                    self.local(i, h);
                }
                (accumulators.len() as u64).hash(h);
                for a in accumulators {
                    self.local(&a.local, h);
                    self.expr(&a.init, h);
                    self.expr(&a.update, h);
                }
                self.body(body, h);
            }
            Stmt::Reduce {
                kind,
                values,
                merge,
                fast,
                outs,
                scratch,
            } => {
                self.reduce_kind(kind, h);
                (values.len() as u64).hash(h);
                for v in values {
                    self.expr(v, h);
                }
                self.merge(merge, h);
                fast.hash(h);
                for o in outs {
                    self.local(o, h);
                }
                for t in scratch {
                    self.tile(t, h);
                }
            }
            Stmt::Break | Stmt::Return | Stmt::Barrier | Stmt::StorageBarrier => {}
        }
    }
}

/// A memoized plan is stored with placements in liveness order and no tile
/// identity of its own; retrieval rebinds them onto the caller's tiles. Two
/// same-shaped tiles are distinct allocations, so identity must come from the
/// caller's IR, never from whichever IR first populated the memo.
fn rebind(template: &ArenaPlan, live: &LivenessInfo) -> ArenaPlan {
    let mut plan = template.clone();
    for (placement, tile) in plan.placements.iter_mut().zip(live.iter()) {
        placement.tile = tile.tile.clone();
    }
    plan
}

/// Ordering key for the argmin. Fully deterministic: fewest bytes, then
/// `Regions`, then fewest insertions, then the lowest suggestion index.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Rank {
    bytes: u32,
    mode_rank: u8,
    insertions: u8,
    suggestion_index: u32,
}

impl Planner {
    fn plan_uncached(&self, ir: &KernelIr, caps: &Caps) -> Result<ArenaPlan> {
        let base_live = analyze(ir);
        let mut options: Vec<(SmallVec<[u32; 4]>, u32)> = vec![(SmallVec::new(), u32::MAX)];
        for suggestion in crate::barrier::suggestions(ir, &base_live)
            .into_iter()
            .take(3)
        {
            let mut one = SmallVec::new();
            one.push(suggestion.index);
            options.push((one, suggestion.index));
        }

        let mut best: Option<(Rank, ArenaPlan)> = None;
        for (insertions, suggestion_index) in options {
            let candidate = if insertions.is_empty() {
                ir.clone()
            } else {
                crate::barrier::insert(ir, &insertions)?
            };
            let live = analyze(&candidate);
            for mode in [ArenaMode::Regions, ArenaMode::ByteArena] {
                let plan = match mode {
                    ArenaMode::Regions => arena::regions(&live),
                    ArenaMode::ByteArena => {
                        // The arena only wins when cross-stride reuse actually
                        // fires; without it 16-byte rounding is a strict loss.
                        if !caps.workgroup_alias
                            || !arena::all_packable(&live)
                            || !arena::mixes_stride_widths(&live)
                        {
                            continue;
                        }
                        match arena::byte_arena(&live) {
                            Some(plan) => plan,
                            None => continue,
                        }
                    }
                };
                let rank = Rank {
                    bytes: plan.total_bytes,
                    mode_rank: match mode {
                        ArenaMode::Regions => 0,
                        ArenaMode::ByteArena => 1,
                    },
                    insertions: insertions.len() as u8,
                    suggestion_index,
                };
                let better = match &best {
                    Some((best_rank, _)) => rank < *best_rank,
                    None => true,
                };
                if better {
                    let mut plan = plan;
                    plan.barriers_inserted = insertions.clone();
                    best = Some((rank, plan));
                }
            }
        }

        let (_, plan) = best.expect("Regions is always a candidate");
        arena::check_budget(&plan, ir, caps)?;
        Ok(plan)
    }
}

impl ArenaPlanner for Planner {
    fn arena_plan(&self, ir: &KernelIr, caps: &Caps) -> Result<ArenaPlan> {
        let live = analyze(ir);
        let key = plan_key(ir, &live, caps);
        if let Some(template) = self.memo.read().get(&key) {
            return Ok(rebind(template, &live));
        }
        let plan = self.plan_uncached(ir, caps)?;
        self.memo.write().insert(key, plan.clone());
        Ok(rebind(&plan, &live))
    }

    fn workgroup_bytes(&self, tiles: &Tiles, caps: &Caps) -> Result<u32> {
        let mut h = FxHasher::default();
        tiles.hash(&mut h);
        let key = TilesKey {
            tiles_hash: h.finish(),
            caps_fingerprint: caps.fingerprint(),
        };
        if let Some(bytes) = self.tiles_memo.read().get(&key) {
            return Ok(*bytes);
        }
        let ir = synth_ir(tiles);
        let bytes = self.arena_plan(&ir, caps)?.total_bytes;
        self.tiles_memo.write().insert(key, bytes);
        Ok(bytes)
    }

    fn barrier_suggestions(&self, ir: &KernelIr) -> Vec<BarrierSuggestion> {
        crate::barrier::barrier_suggestions(ir)
    }

    fn verify_arena(&self, ir: &KernelIr, plan: &ArenaPlan) -> Result<()> {
        crate::verify_arena::verify_arena(ir, plan)
    }

    fn verify_uniformity(&self, ir: &KernelIr) -> Result<()> {
        crate::uniformity::verify_uniformity(ir)
    }
}

/// The minimal body a declared tile set implies: every tile written, one
/// barrier, every tile written again. Each tile's live range then spans the
/// barrier, so no two can share and the footprint is the geometry's
/// simultaneous demand — which is exactly what `verify_launch` must admit against.
pub fn synth_ir(tiles: &Tiles) -> KernelIr {
    let mut body: Vec<Stmt> = Vec::with_capacity(tiles.decls.len() * 2 + 1);
    let mut writes: Vec<Stmt> = Vec::with_capacity(tiles.decls.len());
    for tile in &tiles.decls {
        writes.push(Stmt::FillTile {
            dst: tile.clone(),
            value: zero(tile.element),
            bounds: [None, None],
        });
    }
    body.extend(writes.iter().cloned());
    body.push(Stmt::Barrier);
    body.extend(writes);
    KernelIr {
        buffers: Vec::new(),
        grid: [1, 1, 1],
        block: 1,
        body,
        byte_arena: None,
        name: "workgroup_bytes",
    }
}

fn zero(element: ElementType) -> TileExpr {
    let scalar = match element {
        ElementType::Scalar(scalar)
        | ElementType::Vector { scalar, .. }
        | ElementType::CoopMatrix { scalar, .. } => scalar,
    };
    let part = TileExpr::new(
        TileExprKind::Literal(match scalar {
            ScalarElement::F32 => TileLiteral::F32(0),
            ScalarElement::F16 => TileLiteral::F16(0),
            ScalarElement::BF16 => TileLiteral::BF16(0),
            ScalarElement::U32 => TileLiteral::U32(0),
            ScalarElement::I32 => TileLiteral::I32(0),
            ScalarElement::Bool => TileLiteral::Bool(false),
        }),
        ElementType::Scalar(scalar),
    );
    match element {
        ElementType::Vector { lanes, .. } => TileExpr::new(
            TileExprKind::Vec {
                scalar,
                lanes,
                parts: vec![part; lanes as usize],
            },
            element,
        ),
        ElementType::Scalar(_) | ElementType::CoopMatrix { .. } => part,
    }
}
