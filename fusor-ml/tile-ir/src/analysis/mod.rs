//! Workgroup-tile liveness analysis.
//!
//! One walk over a [`KernelIr`] body produces [`LivenessInfo`]: per-tile live
//! ranges, uniform-barrier positions, and loop spans. The lowering arena
//! consumes it to share workgroup allocations.
//!
//! Two workgroup tiles may share one allocation when their live ranges are
//! disjoint *and* a uniform workgroup barrier orders every thread's last
//! touch of the earlier tile before any thread's first touch of the later
//! one. Threads of a workgroup are not in lockstep, so plain program-order
//! disjointness is not enough — without the barrier a fast thread could
//! write the later tile while a slow thread still reads the earlier one.
//!
//! Loops add a wrap-around hazard: when both tiles live inside a common
//! loop, the later tile's last touch of iteration `i` races the earlier
//! tile's first touch of iteration `i + 1`. [`expand_ranges_over_loops`]
//! folds that in by widening every range to cover each loop body it
//! intersects, so two tiles sharing a loop always overlap and plain
//! interval disjointness plus one forward barrier is sound.
//!
//! Barriers inside `If` blocks are not uniform and never count. Barriers
//! inside loops that may break, return, or run zero dynamic iterations may
//! be skipped at runtime, so they never separate tiles either (they could
//! only ever separate tiles living wholly outside the loop — see
//! [`BarrierInfo::guaranteed`]).

use std::sync::atomic::{AtomicU8, Ordering};

use rustc_hash::FxHashMap;

use crate::ir::{Accumulator, Expr, KernelIr, Stmt, Tile, TileUse};
use crate::{ElementType, MemoryLevel};

mod elide;
pub use elide::BarrierSuggestion;
pub(crate) use elide::{apply_barrier_suggestion, barrier_suggestions, elide_barriers};

mod verify;
pub(crate) use verify::verify_arena;

mod trace {
    use super::*;

    const UNSET: u8 = 2;
    static TRACE: AtomicU8 = AtomicU8::new(UNSET);

    /// Enable or disable liveness/arena tracing at runtime. Called by the
    /// runtime crate when `FusorConfig` is materialized; until then the
    /// `FUSOR_TRACE_ARENA` env var is the fallback so standalone tile-ir
    /// tests keep working.
    pub fn set_liveness_trace(enabled: bool) {
        TRACE.store(enabled as u8, Ordering::Relaxed);
    }

    pub(crate) fn enabled() -> bool {
        match TRACE.load(Ordering::Relaxed) {
            UNSET => {
                let on = std::env::var_os("FUSOR_TRACE_ARENA").is_some();
                TRACE.store(on as u8, Ordering::Relaxed);
                on
            }
            value => value == 1,
        }
    }
}

pub use trace::set_liveness_trace;
pub(crate) use trace::enabled as trace_enabled;

#[derive(Clone, Copy)]
pub(crate) struct LiveRange {
    pub first: u32,
    pub last: u32,
}

/// How a statement touches a tile.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AccessKind {
    Read,
    Write,
    /// Collective read-modify-write (reduction scratch).
    ReadWrite,
}

impl AccessKind {
    pub(crate) fn writes(self) -> bool {
        !matches!(self, Self::Read)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct TileAccess {
    /// Position of the enclosing statement.
    pub position: u32,
    pub kind: AccessKind,
}

pub(crate) struct TileLiveness {
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
    /// `CoopStoreTile`): the emitted array type must equal the tile's
    /// element, so its region never widens to a canonical type.
    pub coop: bool,
}

pub(crate) struct LoopInfo {
    /// Positions spanned by the loop: `start` is the `Loop` statement itself,
    /// `end` the synthetic position after the body.
    pub span: LiveRange,
    /// A `Break` statement is attributed to this loop (innermost frame).
    pub has_break: bool,
    /// A `Return` statement occurs anywhere in the body (exits every loop).
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

pub(crate) struct BarrierInfo {
    pub position: u32,
    /// Statement indices from the body root to the barrier, descending only
    /// through `Loop` bodies (uniform barriers are never inside `If`).
    pub path: Vec<u32>,
    /// Enclosing loop indices, outermost first.
    pub enclosing_loops: Vec<u32>,
    /// Every enclosing loop is [`LoopInfo::guaranteed_once`], so every
    /// thread passes this barrier on every full pass of the enclosing body.
    /// Only guaranteed barriers separate tile live ranges: an in-loop
    /// barrier can only ever separate tiles living wholly outside the loop
    /// (range expansion pins intersecting tiles to the span boundary), which
    /// is exactly the case a zero-trip, `Break`, or `Return` can skip.
    pub guaranteed: bool,
}

pub(crate) struct LivenessInfo {
    pub tiles: FxHashMap<*const (), TileLiveness>,
    /// First-touch order of workgroup tiles (deterministic assignment order).
    pub order: Vec<*const ()>,
    /// Uniform workgroup barriers, in position order.
    pub barriers: Vec<BarrierInfo>,
    /// Completed loop spans, indexed stably from frame push.
    pub loops: Vec<LoopInfo>,
}

impl LivenessInfo {
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
        if trace_enabled() {
            for (index, key) in info.order.iter().enumerate() {
                let tile = &info.tiles[key];
                eprintln!(
                    "arena-tile {index}: {:?} x{} range=({},{})",
                    tile.element, tile.elements, tile.range.first, tile.range.last
                );
            }
            eprintln!(
                "arena-barriers {:?}",
                info.barriers
                    .iter()
                    .map(|barrier| barrier.position)
                    .collect::<Vec<_>>()
            );
            for info in &info.loops {
                eprintln!("arena-loop ({},{})", info.span.first, info.span.last);
            }
        }
        info
    }

    /// The innermost loop whose span strictly contains `[x, y]`.
    pub(crate) fn innermost_common_loop(&self, x: u32, y: u32) -> Option<u32> {
        let mut best: Option<u32> = None;
        for (index, info) in self.loops.iter().enumerate() {
            if info.span.first < x && y < info.span.last {
                let tighter = match best {
                    None => true,
                    Some(previous) => {
                        let previous = &self.loops[previous as usize].span;
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
        let mut scoped: Vec<(*const (), Option<(u32, LiveRange)>)> = Vec::new();
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
                    if nested
                        && intersects
                        && (phase.first > span.first || phase.last < span.last)
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

    /// Whether `later` may reuse memory whose previous occupant is
    /// `earlier`, considering both the plain interval arm and the loop
    /// phase arm.
    pub(crate) fn can_follow_tiles(&self, earlier: &TileLiveness, later: &TileLiveness) -> bool {
        if self.can_follow(earlier.range, later.range) {
            return true;
        }
        // Phase arm: both tiles live only inside one common loop, with
        // disjoint per-iteration phases, a barrier between the phases, and
        // a barrier covering the wrap back to the earlier phase.
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

    /// A guaranteed uniform barrier strictly after `after` and at or before
    /// `at`. Barriers inside loops that may break, return, or run zero
    /// iterations are skippable at runtime and never separate.
    pub(crate) fn separating_barrier(&self, after: u32, at: u32) -> bool {
        self.barriers
            .iter()
            .any(|barrier| barrier.guaranteed && barrier.position > after && barrier.position <= at)
    }

    /// Whether `later` may reuse memory whose previous occupant was
    /// `earlier`: disjoint expanded ranges with a uniform barrier ordering
    /// every thread's last touch of `earlier` before any first touch of
    /// `later`.
    pub(crate) fn can_follow(&self, earlier: LiveRange, later: LiveRange) -> bool {
        earlier.last < later.first && self.separating_barrier(earlier.last, later.first)
    }
}

#[derive(Default)]
struct Walk {
    position: u32,
    tiles: FxHashMap<*const (), TileLiveness>,
    order: Vec<*const ()>,
    barriers: Vec<BarrierInfo>,
    loops: Vec<LoopInfo>,
    /// Open loop frames as indices into `loops`.
    loop_stack: Vec<u32>,
    /// Statement indices from the body root, descending through `Loop`
    /// bodies only.
    path: Vec<u32>,
    /// Kind attributed to the next `touch` (writes are statement-level, so
    /// the statement arm sets this before visiting).
    access_kind: AccessKind,
    /// `If` nesting depth: barriers below a conditional are not uniform.
    conditional_depth: u32,
}

impl Default for AccessKind {
    fn default() -> Self {
        Self::Read
    }
}

impl Walk {
    fn touch(&mut self, tile: &Tile, coop: bool) {
        if tile.layout.memory_level() != MemoryLevel::Workgroup {
            return;
        }
        let key = std::rc::Rc::as_ptr(tile) as *const ();
        let position = self.position;
        if !self.tiles.contains_key(&key) {
            self.order.push(key);
            self.tiles.insert(
                key,
                TileLiveness {
                    range: LiveRange {
                        first: position,
                        last: position,
                    },
                    element: tile.element,
                    elements: tile.layout.allocation_element_count().get(),
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

    fn visit_expr(&mut self, expr: &Expr) {
        let mut touched: Vec<(Tile, TileUse)> = Vec::new();
        expr.kind().for_each_tile(&mut |tile, tile_use| {
            touched.push((tile.clone(), tile_use));
        });
        for (tile, tile_use) in &touched {
            self.access_kind = match tile_use {
                TileUse::Read | TileUse::CoopRead => AccessKind::Read,
                TileUse::ReadWrite => AccessKind::ReadWrite,
            };
            self.touch(tile, matches!(tile_use, TileUse::CoopRead));
        }
        self.access_kind = AccessKind::Read;
        expr.kind().for_each_child(&mut |child| self.visit_expr(child));
    }

    fn visit_stmts(&mut self, stmts: &[Stmt]) {
        for (index, stmt) in stmts.iter().enumerate() {
            self.position += 1;
            match stmt {
                Stmt::Store {
                    addr, value, mask, ..
                } => {
                    match addr {
                        crate::ir::Addr::Rc2 { row, col } => {
                            self.visit_expr(row);
                            self.visit_expr(col);
                        }
                        crate::ir::Addr::Linear(index) => self.visit_expr(index),
                    }
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
                Stmt::CoopStore { addr, .. } => match addr {
                    crate::ir::Addr::Rc2 { row, col } => {
                        self.visit_expr(row);
                        self.visit_expr(col);
                    }
                    crate::ir::Addr::Linear(index) => self.visit_expr(index),
                },
                Stmt::CoopStoreTile { tile, row, col, .. } => {
                    self.access_kind = AccessKind::Write;
                    self.touch(tile, true);
                    self.access_kind = AccessKind::Read;
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
                        span: LiveRange {
                            first: self.position,
                            last: self.position,
                        },
                        has_break: false,
                        has_return: false,
                        static_count: count.as_ref().and_then(literal_u32),
                    });
                    self.loop_stack.push(loop_index);
                    if self.conditional_depth == 0 {
                        self.path.push(index as u32);
                    }
                    self.visit_stmts(body);
                    if self.conditional_depth == 0 {
                        self.path.pop();
                    }
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
                        let mut path = self.path.clone();
                        path.push(index as u32);
                        self.barriers.push(BarrierInfo {
                            position: self.position,
                            path,
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

    /// Expand every tile's range to cover each loop body it intersects,
    /// to fixpoint across nesting. A touch inside a loop recurs every
    /// iteration, so for hazard purposes the tile is live across the whole
    /// body — including the back edge. After expansion, two tiles sharing a
    /// loop always overlap (never merge), and plain interval disjointness
    /// plus one forward barrier is sound; `Break` only shortens executions
    /// of ranges the expansion already covers.
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

fn literal_u32(expr: &Expr) -> Option<u32> {
    use crate::ir::{ExprKind, TileLiteral};
    match expr.kind() {
        ExprKind::Literal(TileLiteral::U32(value)) => Some(*value),
        _ => None,
    }
}
