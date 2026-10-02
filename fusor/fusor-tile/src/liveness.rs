//! Workgroup-tile liveness over a Kernel statement list, feeding arena packing
//! and the barrier argmin. Two tiles may share bytes when their ranges are
//! disjoint and a uniform barrier orders every thread's last touch of one
//! before any first touch of the other. Ranges widen over each loop they touch
//! (the back edge is a hazard); [`TileLiveness::scoped`] recovers in-loop
//! sharing with barriers on both the forward edge and the wrap. Barriers under
//! `If` are never recorded; ones in skippable loops are not `guaranteed`.

use std::sync::Arc;

use fusor_ir::ir::kernel::{
    Accumulator, Addr, ElementType, KernelIr, MemoryLevel, ReduceKind, Stmt, Tile, TileExpr,
    TileExprKind, TileLiteral,
};
use rustc_hash::{FxHashMap, FxHashSet};

/// Identity of one tile declaration (`Arc::as_ptr` as `usize`, keeping
/// [`LivenessInfo`] `Send`/`Sync`).
pub(crate) type TileKeyPtr = usize;

/// The identity key of a tile declaration.
pub(crate) fn tile_key(tile: &Tile) -> TileKeyPtr {
    Arc::as_ptr(tile) as *const () as usize
}

/// The statement-position range over which one tile is live, in the flattened
/// pre-order walk of the body. Inclusive on both ends.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LiveRange {
    pub first: u32,
    pub last: u32,
}

impl LiveRange {
    pub(crate) const fn point(position: u32) -> Self {
        Self {
            first: position,
            last: position,
        }
    }
}

/// How a statement touches a tile.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum AccessKind {
    Read,
    Write,
    /// Collective read-modify-write (reduction scratch).
    ReadWrite,
}

/// How an expression consumes a tile.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) enum TileUse {
    Read,
    /// Read as a raw cooperative-matrix fragment pointer.
    CoopRead,
    ReadWrite,
}

/// One touch of a tile at a raw walk position.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct TileAccess {
    pub position: u32,
    pub kind: AccessKind,
}

/// Everything the packer and the verifier know about one tile.
#[derive(Clone, Debug)]
pub(crate) struct TileLiveness {
    /// The declaration itself.
    pub tile: Tile,
    /// Live range after loop expansion.
    pub range: LiveRange,
    pub element: ElementType,
    /// Allocation extent in elements of `element`.
    pub elements: u32,
    /// Every touch in walk order, at raw (pre-expansion) positions.
    pub accesses: Vec<TileAccess>,
    /// When every access lies inside one innermost loop: that loop and the
    /// tile's per-iteration phase, for barrier-separated in-loop sharing.
    pub scoped: Option<(u32, LiveRange)>,
    /// Consumed as a raw cooperative-matrix pointer, so its region keeps the
    /// tile's own element type.
    pub coop: bool,
}

/// One loop's span and early-exit facts.
#[derive(Clone, Debug)]
pub(crate) struct LoopInfo {
    /// Positions spanned: the `Loop` statement through the synthetic
    /// position after the body.
    pub span: LiveRange,
    /// A `Break` statement is attributed to this loop (innermost frame).
    pub has_break: bool,
    /// A `Return` occurs anywhere in the body (it exits every enclosing loop).
    pub has_return: bool,
    /// The loop count when it is a static literal.
    pub static_count: Option<u32>,
}

impl LoopInfo {
    /// Every execution runs the full body at least once: a positive literal
    /// count and no early exit.
    pub(crate) fn guaranteed_once(&self) -> bool {
        self.static_count.is_some_and(|count| count > 0) && !self.has_break && !self.has_return
    }
}

/// One recorded uniform barrier.
#[derive(Clone, Debug)]
pub(crate) struct BarrierInfo {
    pub position: u32,
    /// Enclosing loop indices, outermost first.
    pub enclosing_loops: Vec<u32>,
    /// Every enclosing loop is [`LoopInfo::guaranteed_once`].
    pub guaranteed: bool,
}

/// Tile liveness, barriers and loop spans for one kernel body.
#[derive(Debug, Default)]
pub(crate) struct LivenessInfo {
    pub tiles: FxHashMap<TileKeyPtr, TileLiveness>,
    /// First-touch order of workgroup tiles. Iterate this, never the map:
    /// pointer keys are not stable across runs.
    pub order: Vec<TileKeyPtr>,
    /// Uniform workgroup barriers, in position order.
    pub barriers: Vec<BarrierInfo>,
    /// Completed loop spans, indexed stably from frame push.
    pub loops: Vec<LoopInfo>,
}

impl LivenessInfo {
    /// One walk over the body, then loop expansion, guaranteed flags and phases.
    pub(crate) fn compute(ir: &KernelIr) -> Self {
        let mut walk = Walk::default();
        walk.visit_stmts(&ir.body);
        walk.expand_ranges_over_loops();
        for barrier in &mut walk.barriers {
            barrier.guaranteed = barrier
                .enclosing_loops
                .iter()
                .all(|&index| walk.loops[index as usize].guaranteed_once());
        }
        let mut info = Self {
            tiles: walk.tiles,
            order: walk.order,
            barriers: walk.barriers,
            loops: walk.loops,
        };
        info.compute_scoped_phases();
        info
    }

    /// Tiles in first-touch order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = &TileLiveness> {
        self.order.iter().map(|key| &self.tiles[key])
    }

    /// The innermost loop whose span strictly contains `[x, y]`.
    pub(crate) fn innermost_common_loop(&self, x: u32, y: u32) -> Option<u32> {
        let mut best: Option<u32> = None;
        for (index, info) in self.loops.iter().enumerate() {
            if info.span.first < x && y < info.span.last {
                let tighter = match best {
                    None => true,
                    Some(previous) => {
                        let previous = self.loops[previous as usize].span;
                        info.span.first >= previous.first && info.span.last <= previous.last
                    }
                };
                if tighter {
                    best = Some(index as u32);
                }
            }
        }
        best
    }

    /// Every loop enclosing `barrier` strictly below `scope` completes every
    /// pass. A `Break` in `scope` itself is fine: past it the tiles are dead.
    pub(crate) fn guaranteed_below(&self, barrier: &BarrierInfo, scope: u32) -> bool {
        match barrier
            .enclosing_loops
            .iter()
            .position(|&index| index == scope)
        {
            None => false,
            Some(position) => barrier.enclosing_loops[position + 1..]
                .iter()
                .all(|&index| self.loops[index as usize].guaranteed_once()),
        }
    }

    fn compute_scoped_phases(&mut self) {
        let mut scoped: Vec<(TileKeyPtr, Option<(u32, LiveRange)>)> = Vec::new();
        for &key in &self.order {
            let tile = &self.tiles[&key];
            let first = tile.accesses.iter().map(|access| access.position).min();
            let last = tile.accesses.iter().map(|access| access.position).max();
            let (Some(first), Some(last)) = (first, last) else {
                scoped.push((key, None));
                continue;
            };
            let Some(home) = self.innermost_common_loop(first, last) else {
                scoped.push((key, None));
                continue;
            };
            // Expand the phase over nested loops to fixpoint.
            let home_span = self.loops[home as usize].span;
            let mut phase = LiveRange { first, last };
            loop {
                let mut changed = false;
                for info in &self.loops {
                    let span = info.span;
                    let nested = span.first > home_span.first && span.last < home_span.last;
                    let intersects = phase.first < span.last && phase.last > span.first;
                    if nested && intersects && (phase.first > span.first || phase.last < span.last)
                    {
                        phase.first = phase.first.min(span.first);
                        phase.last = phase.last.max(span.last);
                        changed = true;
                    }
                }
                if !changed {
                    break;
                }
            }
            scoped.push((key, Some((home, phase))));
        }
        for (key, value) in scoped {
            self.tiles.get_mut(&key).expect("walk-recorded tile").scoped = value;
        }
    }

    /// A barrier inside loop `scope` at a position satisfying `in_interval`,
    /// executing on every full pass of the body.
    fn scoped_barrier(&self, scope: u32, in_interval: impl Fn(u32) -> bool) -> bool {
        let span = self.loops[scope as usize].span;
        self.barriers.iter().any(|barrier| {
            span.first < barrier.position
                && barrier.position < span.last
                && in_interval(barrier.position)
                && self.guaranteed_below(barrier, scope)
        })
    }

    /// A guaranteed uniform barrier in `(after, at]`; barriers in skippable
    /// loops never separate.
    pub(crate) fn separating_barrier(&self, after: u32, at: u32) -> bool {
        self.barriers
            .iter()
            .any(|barrier| barrier.guaranteed && barrier.position > after && barrier.position <= at)
    }

    /// Whether `later` may reuse `earlier`'s memory: disjoint expanded ranges
    /// with a uniform barrier between them.
    pub(crate) fn can_follow(&self, earlier: LiveRange, later: LiveRange) -> bool {
        earlier.last < later.first && self.separating_barrier(earlier.last, later.first)
    }

    /// Both arms of the reuse predicate: plain interval, and loop phase.
    pub(crate) fn can_follow_tiles(&self, earlier: &TileLiveness, later: &TileLiveness) -> bool {
        if self.can_follow(earlier.range, later.range) {
            return true;
        }
        // Phase arm: disjoint phases in one common loop, a barrier between them
        // and one covering the wrap.
        let (Some((home_a, phase_a)), Some((home_b, phase_b))) = (earlier.scoped, later.scoped)
        else {
            return false;
        };
        if home_a != home_b {
            return false;
        }
        let (first, second) = if phase_a.first <= phase_b.first {
            (phase_a, phase_b)
        } else {
            (phase_b, phase_a)
        };
        first.last < second.first
            && self.scoped_barrier(home_a, |p| p > first.last && p <= second.first)
            && self.scoped_barrier(home_a, |p| p > second.last || p <= first.first)
    }
}

/// Compute tile liveness over `ir`'s body.
pub(crate) fn analyze(ir: &KernelIr) -> LivenessInfo {
    LivenessInfo::compute(ir)
}

/// Every tile an expression node touches directly, and how.
pub(crate) fn for_each_tile(kind: &TileExprKind, f: &mut dyn FnMut(&Tile, TileUse)) {
    match kind {
        TileExprKind::LoadTile { tile, .. } => f(tile, TileUse::Read),
        TileExprKind::Reduce { kind, .. } => match kind.as_ref() {
            ReduceKind::Subgroup => {}
            ReduceKind::Workgroup { scratch, .. } => f(scratch, TileUse::ReadWrite),
        },
        TileExprKind::CoopLoad { src, .. } => f(&src.tile, TileUse::CoopRead),
        _ => {}
    }
}

struct Walk {
    position: u32,
    tiles: FxHashMap<TileKeyPtr, TileLiveness>,
    order: Vec<TileKeyPtr>,
    barriers: Vec<BarrierInfo>,
    loops: Vec<LoopInfo>,
    /// Open loop frames as indices into `loops`.
    loop_stack: Vec<u32>,
    /// Kind attributed to the next `touch`.
    access_kind: AccessKind,
    /// `If` nesting depth: barriers below a conditional are not recorded.
    conditional_depth: u32,
    /// Nodes visited in the current root expression, cleared per root so a
    /// node shared by two statements records at both positions.
    seen: FxHashSet<usize>,
}

impl Default for Walk {
    fn default() -> Self {
        Self {
            position: 0,
            tiles: FxHashMap::default(),
            order: Vec::new(),
            barriers: Vec::new(),
            loops: Vec::new(),
            loop_stack: Vec::new(),
            access_kind: AccessKind::Read,
            conditional_depth: 0,
            seen: FxHashSet::default(),
        }
    }
}

impl Walk {
    fn touch(&mut self, tile: &Tile, coop: bool) {
        if tile.layout.level != MemoryLevel::Workgroup {
            return;
        }
        let key = tile_key(tile);
        let position = self.position;
        if !self.tiles.contains_key(&key) {
            self.order.push(key);
            self.tiles.insert(
                key,
                TileLiveness {
                    tile: tile.clone(),
                    range: LiveRange::point(position),
                    element: tile.element,
                    elements: tile.layout.element_count().min(u32::MAX as u64) as u32,
                    accesses: Vec::new(),
                    scoped: None,
                    coop: false,
                },
            );
        }
        let liveness = self.tiles.get_mut(&key).expect("inserted above");
        liveness.range.last = position;
        liveness.coop |= coop;
        liveness.accesses.push(TileAccess {
            position,
            kind: self.access_kind,
        });
    }

    /// Record every tile one operand expression touches at the current
    /// position, visiting each DAG node once (duplicates inform nothing).
    fn visit_expr(&mut self, expr: &TileExpr) {
        self.seen.clear();
        self.visit_expr_once(expr);
    }

    fn visit_expr_once(&mut self, expr: &TileExpr) {
        if !self.seen.insert(expr.node_ptr()) {
            return;
        }
        for_each_tile(expr.kind(), &mut |tile, tile_use| {
            self.access_kind = match tile_use {
                TileUse::Read | TileUse::CoopRead => AccessKind::Read,
                TileUse::ReadWrite => AccessKind::ReadWrite,
            };
            self.touch(tile, matches!(tile_use, TileUse::CoopRead));
        });
        self.access_kind = AccessKind::Read;
        expr.kind()
            .visit_children(&mut |child| self.visit_expr_once(child));
    }

    fn visit_addr(&mut self, addr: &Addr) {
        match addr {
            Addr::Linear(index) => self.visit_expr(index),
            Addr::Rc2 { row, col } => {
                self.visit_expr(row);
                self.visit_expr(col);
            }
        }
    }

    fn visit_stmts(&mut self, stmts: &[Stmt]) {
        for stmt in stmts {
            self.position += 1;
            match stmt {
                Stmt::Store {
                    addr, value, mask, ..
                }
                | Stmt::AtomicAdd {
                    addr, value, mask, ..
                } => {
                    self.visit_addr(addr);
                    self.visit_expr(value);
                    self.visit_expr(mask);
                }
                Stmt::StoreLocal { value, .. } => self.visit_expr(value),
                Stmt::StoreTile { dst, index, value } => {
                    self.access_kind = AccessKind::Write;
                    self.touch(dst, false);
                    self.access_kind = AccessKind::Read;
                    self.visit_expr(index);
                    self.visit_expr(value);
                }
                Stmt::FillTile { dst, value, bounds } => {
                    self.access_kind = AccessKind::Write;
                    self.touch(dst, false);
                    self.access_kind = AccessKind::Read;
                    self.visit_expr(value);
                    for bound in bounds.iter().flatten() {
                        self.visit_expr(bound);
                    }
                }
                Stmt::CoopStore { acc, addr, .. } => {
                    self.visit_expr(acc);
                    self.visit_addr(addr);
                }
                Stmt::CoopStoreTile {
                    acc,
                    tile,
                    row,
                    col,
                } => {
                    self.access_kind = AccessKind::Write;
                    self.touch(tile, true);
                    self.access_kind = AccessKind::Read;
                    self.visit_expr(acc);
                    self.visit_expr(row);
                    self.visit_expr(col);
                }
                Stmt::If {
                    condition,
                    accept,
                    reject,
                } => {
                    self.visit_expr(condition);
                    self.conditional_depth += 1;
                    self.visit_stmts(accept);
                    self.visit_stmts(reject);
                    self.conditional_depth -= 1;
                }
                Stmt::Loop {
                    count,
                    accumulators,
                    body,
                    ..
                } => {
                    // Count and inits run once, before the loop.
                    if let Some(count) = count {
                        self.visit_expr(count);
                    }
                    for Accumulator { init, .. } in accumulators {
                        self.visit_expr(init);
                    }
                    let loop_index = self.loops.len() as u32;
                    self.loops.push(LoopInfo {
                        span: LiveRange::point(self.position),
                        has_break: false,
                        has_return: false,
                        static_count: count.as_ref().and_then(literal_u32),
                    });
                    self.loop_stack.push(loop_index);
                    self.visit_stmts(body);
                    // Updates run at the end of every iteration, so they count inside the span.
                    if !accumulators.is_empty() {
                        self.position += 1;
                        for Accumulator { update, .. } in accumulators {
                            self.visit_expr(update);
                        }
                    }
                    self.position += 1;
                    self.loop_stack.pop().expect("loop frame pushed above");
                    self.loops[loop_index as usize].span.last = self.position;
                }
                // One scratch tile per lane, all read-modify-written by one tree.
                Stmt::Reduce {
                    values, scratch, ..
                } => {
                    self.access_kind = AccessKind::ReadWrite;
                    for tile in scratch {
                        self.touch(tile, false);
                    }
                    self.access_kind = AccessKind::Read;
                    for value in values {
                        self.visit_expr(value);
                    }
                }
                Stmt::Break => {
                    if let Some(&frame) = self.loop_stack.last() {
                        self.loops[frame as usize].has_break = true;
                    }
                }
                Stmt::Return => {
                    for &frame in &self.loop_stack {
                        self.loops[frame as usize].has_return = true;
                    }
                }
                Stmt::Barrier => {
                    if self.conditional_depth == 0 {
                        self.barriers.push(BarrierInfo {
                            position: self.position,
                            enclosing_loops: self.loop_stack.clone(),
                            // Finalized after the walk, once loop facts are complete.
                            guaranteed: false,
                        });
                    }
                }
                Stmt::StorageBarrier => {}
            }
        }
    }

    /// Expand every tile's range over each loop body it intersects, to fixpoint:
    /// a touch in a loop recurs every iteration, back edge included.
    fn expand_ranges_over_loops(&mut self) {
        loop {
            let mut changed = false;
            for liveness in self.tiles.values_mut() {
                let range = &mut liveness.range;
                for info in &self.loops {
                    let span = info.span;
                    let intersects = range.first < span.last && range.last > span.first;
                    if intersects && (range.first > span.first || range.last < span.last) {
                        range.first = range.first.min(span.first);
                        range.last = range.last.max(span.last);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }
}

fn literal_u32(expr: &TileExpr) -> Option<u32> {
    match expr.kind() {
        TileExprKind::Literal(TileLiteral::U32(value)) => Some(*value),
        TileExprKind::Literal(TileLiteral::I32(value)) if *value >= 0 => Some(*value as u32),
        _ => None,
    }
}
