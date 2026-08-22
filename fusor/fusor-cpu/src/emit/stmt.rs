//! Kernel statements -> the loop nest.
//!
//! `Barrier` splits the lane loop into two loops over the lane range, ensuring
//! iteration 0 does not read tile slots that a later iteration has not written.
//!
//! The split is done here, after statement compilation, by [`block`]: a
//! statement list containing a barrier at any depth is cut into consecutive
//! [`LaneLoop`]s and the lane loop is pushed into each piece; a list containing
//! none is wrapped in exactly one lane loop over the whole lane range. A barrier
//! inside a uniform `If`/`Loop` splits that body's list, with the lane loops
//! nested inside the `If`/`Loop`.

use fusor_ir::ir::kernel::{ScalarElement, TileReduceOp};
use fusor_ir::target::EmitError;

use super::expr::Slot;

/// A half-open range of tape instructions to evaluate before a statement runs.
pub(crate) type TapeRange = std::ops::Range<u32>;

/// One loop-carried accumulator, held in a register across iterations and
/// never reloaded.
#[derive(Clone, Debug)]
pub struct CAcc {
    pub local: u16,
    pub init_prep: TapeRange,
    pub init: Slot,
    pub update_prep: TapeRange,
    pub update: Slot,
}

/// A compiled statement. Every variant carries the tape range it must evaluate
/// immediately before executing, so a value inside a loop body is recomputed
/// per iteration while a value hoisted above the loop is not.
#[derive(Clone, Debug)]
pub enum CStmt {
    Store {
        prep: TapeRange,
        buf: u16,
        elem: ScalarElement,
        index: Slot,
        value: Slot,
        mask: Slot,
    },
    /// Per-thread accumulation is guaranteed by running an atomic-carrying
    /// program on one worker (see [`crate::launch`]); the add itself is an
    /// ordinary read-modify-write.
    AtomicAdd {
        prep: TapeRange,
        buf: u16,
        elem: ScalarElement,
        index: Slot,
        value: Slot,
        mask: Slot,
    },
    StoreLocal {
        prep: TapeRange,
        local: u16,
        value: Slot,
    },
    StoreTile {
        prep: TapeRange,
        tile: u16,
        elem: ScalarElement,
        index: Slot,
        value: Slot,
    },
    /// Collective, and therefore uniform: it completes for the whole tile
    /// before the next statement starts.
    FillTile {
        prep: TapeRange,
        tile: u16,
        elem: ScalarElement,
        value: Slot,
        extents: [u32; 2],
        lo: Option<Slot>,
        hi: Option<Slot>,
    },
    If {
        prep: TapeRange,
        cond: Slot,
        /// A uniform predicate is a real branch; a divergent one is a lane-mask
        /// select, with both arms' stores merged under `mask` / `!mask`.
        uniform: bool,
        accept: Vec<CStmt>,
        reject: Vec<CStmt>,
    },
    Loop {
        prep: TapeRange,
        count: Option<Slot>,
        index: Option<u16>,
        accs: Vec<CAcc>,
        body: Vec<CStmt>,
    },
    Break,
    Return,
    /// Cross-lane staging: run `prep` for every lane chunk, write `value` into
    /// `tile[lane]`, then tree-reduce each `group` of lanes and broadcast the
    /// group result back over the group.
    StageTree {
        prep: TapeRange,
        tile: u16,
        value: Slot,
        op: TileReduceOp,
        group: u32,
    },
    /// As [`CStmt::StageTree`], but each lane first accumulates `iterations`
    /// evaluations of `prep` while `index` walks `0..iterations`.
    LoopTree {
        prep: TapeRange,
        tile: u16,
        value: Slot,
        op: TileReduceOp,
        group: u32,
        iterations: u32,
        index: u16,
    },
    /// The **N-ary** cross-lane reduction: one scratch tile per accumulator
    /// lane, staged per lane chunk, then a log-tree over each group applying
    /// `merge` at every level and broadcasting the group result back.
    ///
    /// `merge` is a tape range evaluated once per tree step with `lhs`/`rhs`
    /// holding the two partials — `W` independent pairs at a time, since a merge
    /// reads only its formals and therefore vectorizes across pairs. `fast` is
    /// the single-lane hardware operator, which skips the tape entirely.
    CarrierTree {
        prep: TapeRange,
        tiles: Vec<u16>,
        values: Vec<Slot>,
        lhs: Vec<u16>,
        rhs: Vec<u16>,
        merge_prep: TapeRange,
        merged: Vec<Slot>,
        outs: Vec<u16>,
        group: u32,
        fast: Option<TileReduceOp>,
    },
    /// A marker consumed by [`block`]. Never reaches the runner.
    Barrier,
    /// One lane loop over the block's lane range. Contains no barrier.
    Lanes(Vec<CStmt>),
}

impl CStmt {
    /// True when the statement runs once for the whole workgroup rather than
    /// once per lane chunk, and therefore ends the enclosing lane loop.
    pub fn is_collective(&self) -> bool {
        match self {
            CStmt::Barrier
            | CStmt::StageTree { .. }
            | CStmt::LoopTree { .. }
            | CStmt::CarrierTree { .. }
            | CStmt::FillTile { .. } => true,
            CStmt::If { accept, reject, .. } => {
                accept.iter().any(CStmt::is_collective) || reject.iter().any(CStmt::is_collective)
            }
            CStmt::Loop { body, .. } => body.iter().any(CStmt::is_collective),
            _ => false,
        }
    }
}

/// One loop over the lane range, containing no barrier.
#[derive(Clone, Debug)]
pub struct LaneLoop {
    pub lanes: u32,
    pub width: u32,
    pub stmts: Vec<CStmt>,
}

/// Partition a compiled statement list at every barrier, pushing the lane loop
/// into each piece.
///
/// A list with no barrier at any depth yields exactly one [`LaneLoop`].
pub(crate) fn block(body: &[CStmt], lanes: u32, width: u32) -> Result<Vec<LaneLoop>, EmitError> {
    let mut out: Vec<LaneLoop> = Vec::new();
    let mut run: Vec<CStmt> = Vec::new();
    let flush = |run: &mut Vec<CStmt>, out: &mut Vec<LaneLoop>| {
        if !run.is_empty() {
            out.push(LaneLoop {
                lanes,
                width,
                stmts: std::mem::take(run),
            });
        }
    };
    for s in body {
        if !s.is_collective() {
            run.push(s.clone());
            continue;
        }
        flush(&mut run, &mut out);
        match s {
            CStmt::Barrier => {} // the split itself; nothing to emit
            CStmt::If {
                prep,
                cond,
                uniform,
                accept,
                reject,
            } => {
                if !*uniform {
                    // `verify_uniformity` guarantees a barrier only appears
                    // under a uniform predicate; reaching here means the Kernel
                    // verifier was skipped.
                    return Err(EmitError::Validation(
                        "barrier under a divergent `If`".into(),
                    ));
                }
                out.push(LaneLoop {
                    lanes,
                    width,
                    stmts: vec![CStmt::If {
                        prep: prep.clone(),
                        cond: *cond,
                        uniform: true,
                        accept: nest(accept, lanes, width)?,
                        reject: nest(reject, lanes, width)?,
                    }],
                });
            }
            CStmt::Loop {
                prep,
                count,
                index,
                accs,
                body,
            } => {
                out.push(LaneLoop {
                    lanes,
                    width,
                    stmts: vec![CStmt::Loop {
                        prep: prep.clone(),
                        count: *count,
                        index: *index,
                        accs: accs.clone(),
                        body: nest(body, lanes, width)?,
                    }],
                });
            }
            other => out.push(LaneLoop {
                lanes,
                width,
                stmts: vec![other.clone()],
            }),
        }
    }
    flush(&mut run, &mut out);
    Ok(out)
}

/// Recursively split a nested body, wrapping each barrier-free run in a
/// [`CStmt::Lanes`] so the runner re-enters the lane loop inside the control
/// flow rather than around it.
fn nest(body: &[CStmt], lanes: u32, width: u32) -> Result<Vec<CStmt>, EmitError> {
    if !body.iter().any(CStmt::is_collective) {
        return Ok(vec![CStmt::Lanes(body.to_vec())]);
    }
    Ok(block(body, lanes, width)?
        .into_iter()
        .map(|l| CStmt::Lanes(l.stmts))
        .collect())
}
