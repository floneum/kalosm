//! Workgroup-tile liveness over an Kernel statement list. Feeds arena packing (two
//! tiles whose ranges do not overlap may share bytes) and the barrier argmin.
//!
//! Two workgroup tiles may share one allocation when their live ranges are
//! disjoint *and* a uniform workgroup barrier orders every thread's last touch
//! of the earlier tile before any thread's first touch of the later one.
//! Threads of a workgroup are not in lockstep, so plain program-order
//! disjointness is not enough.
//!
//! Loops add a wrap-around hazard: when both tiles live inside a common loop,
//! the later tile's last touch of iteration `i` races the earlier tile's first
//! touch of iteration `i + 1`. [`LivenessInfo`] folds that in by widening every
//! range to cover each loop body it intersects, so two tiles sharing a loop
//! always overlap and plain interval disjointness plus one forward barrier is
//! sound. [`TileLiveness::scoped`] recovers the in-loop sharing case
//! separately, requiring barriers on both the forward edge and the wrap.
//!
//! Barriers inside `If` blocks are never recorded — uniformity is established
//! by [`crate::uniformity`], and a conditional barrier is not uniform by
//! construction here. Barriers inside loops that may break, return, or run a
//! dynamic number of iterations are recorded but not `guaranteed`.

use std::sync::Arc;

use fusor_ir::ir::kernel::{
    Accumulator, Addr, CoopSrc, ElementType, KernelIr, MemoryLevel, ReduceKind, Stmt, Tile,
    TileExpr, TileExprKind, TileLiteral,
};
use rustc_hash::{FxHashMap, FxHashSet};

/// Identity of one tile declaration. Declarations are never interned, so two
/// same-shaped tiles stay distinct and the arena knows they are two
/// allocations; `Arc::as_ptr` is that identity. Stored as `usize` so
/// [`LivenessInfo`] stays `Send`/`Sync`.
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
    /// The declaration itself, so a [`fusor_ir::ir::kernel::Placement`] can
    /// name it without a second lookup.
    pub tile: Tile,
    /// Live range after loop expansion.
    pub range: LiveRange,
    pub element: ElementType,
    /// Allocation extent in elements of `element`.
    pub elements: u32,
    /// Every touch in walk order, at raw (pre-expansion) positions.
    pub accesses: Vec<TileAccess>,
    /// When every access lies inside one innermost loop: that loop and the
    /// tile's per-iteration phase (raw positions expanded over loops nested
    /// inside it). Enables sharing between in-loop tiles whose phases are
    /// barrier-separated both forward and across the back edge.
    pub scoped: Option<(u32, LiveRange)>,
    /// Consumed as a raw cooperative-matrix pointer (`CoopLoad` /
    /// `CoopStoreTile`): the emitted array type must equal the tile's element,
    /// so its region never widens to a canonical type.
    pub coop: bool,
}

/// One loop's span and early-exit facts.
#[derive(Clone, Debug)]
pub(crate) struct LoopInfo {
    /// Positions spanned by the loop: `first` is the `Loop` statement itself,
    /// `last` the synthetic position after the body.
    pub span: LiveRange,
    /// A `Break` statement is attributed to this loop (innermost frame).
    pub has_break: bool,
    /// A `Return` occurs anywhere in the body (it exits every enclosing loop).
    pub has_return: bool,
    /// The loop count when it is a static literal.
    pub static_count: Option<u32>,
}

impl LoopInfo {
    /// Whether every dynamic execution of this loop runs the full body at
    /// least once: a positive static-literal count with no early exit. A
    /// dynamic count may be zero at runtime, and a `Break`/`Return` can skip
    /// the tail of the body — either way a barrier inside the loop is not
    /// guaranteed to execute.
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
    /// Every enclosing loop is [`LoopInfo::guaranteed_once`], so every thread
    /// passes this barrier on every full pass of the enclosing body.
    pub guaranteed: bool,
}

/// Tile liveness, barriers and loop spans for one kernel body.
#[derive(Debug, Default)]
pub(crate) struct LivenessInfo {
    pub tiles: FxHashMap<TileKeyPtr, TileLiveness>,
    /// First-touch order of workgroup tiles. **Always iterate this, never the
    /// map**: pointer keys are not stable across runs, `order` is.
    pub order: Vec<TileKeyPtr>,
    /// Uniform workgroup barriers, in position order.
    pub barriers: Vec<BarrierInfo>,
    /// Completed loop spans, indexed stably from frame push.
    pub loops: Vec<LoopInfo>,
}

impl LivenessInfo {
    /// One walk over `ir`'s body, then loop expansion, then the guaranteed
    /// flags, then the per-iteration phases.
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
    /// pass, so the barrier executes on every full pass of `scope`'s body.
    /// `Break` in `scope` itself does not disqualify: taking the back edge
    /// means the full body executed, and after an exit the loop's tiles are
    /// touched no more.
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
            // Expand the phase over loops nested inside the home loop, to
            // fixpoint: a touch inside a nested loop recurs every nested
            // iteration.
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

    /// A guaranteed uniform barrier strictly after `after` and at or before
    /// `at`. Barriers inside loops that may break, return, or run zero
    /// iterations are skippable at runtime and never separate.
    pub(crate) fn separating_barrier(&self, after: u32, at: u32) -> bool {
        self.barriers
            .iter()
            .any(|barrier| barrier.guaranteed && barrier.position > after && barrier.position <= at)
    }

    /// Whether `later` may reuse memory whose previous occupant was `earlier`:
    /// disjoint expanded ranges with a uniform barrier ordering every thread's
    /// last touch of `earlier` before any first touch of `later`.
    pub(crate) fn can_follow(&self, earlier: LiveRange, later: LiveRange) -> bool {
        earlier.last < later.first && self.separating_barrier(earlier.last, later.first)
    }

    /// Both arms of the reuse predicate: the plain interval arm, and the
    /// loop-phase arm for two tiles living only inside one common loop.
    pub(crate) fn can_follow_tiles(&self, earlier: &TileLiveness, later: &TileLiveness) -> bool {
        if self.can_follow(earlier.range, later.range) {
            return true;
        }
        // Phase arm: both tiles live only inside one common loop, with
        // disjoint per-iteration phases, a barrier between the phases, and a
        // barrier covering the wrap back to the earlier phase.
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

/// Every tile an expression node touches directly, with how it touches it.
/// Shared with uniformity and the Kernel verifier.
pub(crate) fn for_each_tile(kind: &TileExprKind, f: &mut dyn FnMut(&Tile, TileUse)) {
    match kind {
        TileExprKind::LoadTile { tile, .. } => f(tile, TileUse::Read),
        TileExprKind::Reduce { kind, .. } => match kind.as_ref() {
            ReduceKind::Subgroup => {}
            ReduceKind::Workgroup { scratch, .. } | ReduceKind::Loop { scratch, .. } => {
                f(scratch, TileUse::ReadWrite)
            }
        },
        TileExprKind::CoopLoad { src, .. } => match src.as_ref() {
            CoopSrc::TileRegion { tile, .. } => f(tile, TileUse::CoopRead),
            CoopSrc::BroadcastCol { .. } => {}
        },
        _ => {}
    }
}

/// Every direct child expression of a node, in a fixed order.
pub(crate) fn for_each_child(kind: &TileExprKind, f: &mut dyn FnMut(&TileExpr)) {
    match kind {
        TileExprKind::Literal(_)
        | TileExprKind::Builtin(_)
        | TileExprKind::LoadLocal(_)
        | TileExprKind::CoopZero { .. } => {}
        TileExprKind::Load {
            addr, mask, fill, ..
        } => {
            for_each_addr_expr(addr, f);
            f(mask);
            f(fill);
        }
        TileExprKind::LoadTile { index, .. } => f(index),
        TileExprKind::Unary { value, .. } => f(value),
        TileExprKind::Binary { left, right, .. } | TileExprKind::Compare { left, right, .. } => {
            f(left);
            f(right);
        }
        TileExprKind::Round { value, .. } => f(value),
        TileExprKind::Cast { value, .. } | TileExprKind::Bitcast { value, .. } => f(value),
        TileExprKind::Select {
            condition,
            accept,
            reject,
        } => {
            f(condition);
            f(accept);
            f(reject);
        }
        TileExprKind::Vec { parts, .. } => {
            for part in parts {
                f(part);
            }
        }
        TileExprKind::VecComponent { vector, .. } => f(vector),
        TileExprKind::Dot { left, right } => {
            f(left);
            f(right);
        }
        TileExprKind::Reduce { value, .. } => f(value),
        TileExprKind::CoopLoad { src, .. } => match src.as_ref() {
            CoopSrc::TileRegion { row, col, .. } => {
                f(row);
                f(col);
            }
            CoopSrc::BroadcastCol { col, .. } => f(col),
        },
        TileExprKind::CoopMma { a, b, c } => {
            f(a);
            f(b);
            f(c);
        }
    }
}

/// The expressions inside one address.
pub(crate) fn for_each_addr_expr(addr: &Addr, f: &mut dyn FnMut(&TileExpr)) {
    match addr {
        Addr::Linear(index) => f(index),
        Addr::Rc2 { row, col } => {
            f(row);
            f(col);
        }
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
    /// Nodes already visited in the operand expression in progress. Cleared
    /// per root expression, so a node shared by two statements is recorded
    /// at both positions. See [`Walk::visit_expr`].
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
    /// position.
    ///
    /// The expression is a DAG, so a naive walk is exponential in the sharing
    /// depth. Every visit of a node at one position records the same
    /// `(tile, position, kind)` and duplicate accesses inform nothing
    /// downstream, so visiting each node once is the same analysis.
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
        for_each_child(expr.kind(), &mut |child| self.visit_expr_once(child));
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
                    // Count and accumulator inits run once, before the loop:
                    // header position, outside the span.
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
                    // Accumulator updates run at the end of EVERY iteration —
                    // after any in-loop barrier — so their tile touches are
                    // attributed inside the span and expand over the loop.
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
                // One scratch tile per accumulator lane, all read-modify-written
                // by the same tree. `verify_arena`'s all-pairs recheck therefore
                // sees N tiles per reduction and separates them with the same
                // guaranteed-uniform barrier rule as one.
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
                            // Finalized after the walk, once every enclosing
                            // loop's break/return/count facts are complete.
                            guaranteed: false,
                        });
                    }
                }
                Stmt::StorageBarrier => {}
            }
        }
    }

    /// Expand every tile's range to cover each loop body it intersects, to
    /// fixpoint across nesting. A touch inside a loop recurs every iteration,
    /// so for hazard purposes the tile is live across the whole body —
    /// including the back edge.
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
