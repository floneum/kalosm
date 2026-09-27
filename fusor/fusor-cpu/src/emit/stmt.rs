//! Kernel statements -> the loop nest. [`block`] cuts a statement list at every
//! barrier into consecutive lane loops so iteration 0 never reads a tile slot
//! a later one has not written; a barrier inside a uniform `If`/`Loop` splits
//! that body, with the lane loops nested inside.

use fusor_ir::ir::kernel::{ScalarElement, TileReduceOp};
use fusor_ir::target::EmitError;

use super::expr::Slot;

/// A half-open range of tape instructions to evaluate before a statement runs.
pub(crate) type TapeRange = std::ops::Range<u32>;

/// One loop-carried accumulator, held in a register across iterations.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CAcc {
    pub local: u16,
    pub init_prep: TapeRange,
    pub init: Slot,
    pub update_prep: TapeRange,
    pub update: Slot,
}

/// A compiled statement, with the tape range to evaluate right before it, so a
/// loop body recomputes per iteration and a hoisted value does not.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CStmt {
    Store {
        prep: TapeRange,
        buf: u16,
        elem: ScalarElement,
        index: Slot,
        value: Slot,
        mask: Slot,
    },
    /// An ordinary read-modify-write: atomic programs run on one worker.
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
    /// Collective, hence uniform: completes for the whole tile first.
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
        /// Uniform predicate: a branch. Divergent: a lane-mask select of both arms.
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
    /// Cross-lane staging: `tile[lane] = value` per chunk, tree-reduce each
    /// `group`, broadcast back.
    StageTree {
        prep: TapeRange,
        tile: u16,
        value: Slot,
        op: TileReduceOp,
        group: u32,
    },
    /// N-ary cross-lane reduction: one scratch tile per accumulator lane, then a
    /// log-tree per group applying `merge` (a tape range over `lhs`/`rhs`, `W`
    /// pairs at a time); `fast` is the single-lane hardware operator.
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
    /// Runs once per workgroup, so it ends the enclosing lane loop.
    pub fn is_collective(&self) -> bool {
        match self {
            CStmt::Barrier
            | CStmt::StageTree { .. }
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

/// Partition a compiled statement list at every barrier into lane loops
/// (each barrier-free); no barrier yields exactly one.
pub(crate) fn block(body: &[CStmt]) -> Result<Vec<Vec<CStmt>>, EmitError> {
    let mut out: Vec<Vec<CStmt>> = Vec::new();
    let mut run: Vec<CStmt> = Vec::new();
    for s in body {
        if !s.is_collective() {
            run.push(s.clone());
            continue;
        }
        if !run.is_empty() {
            out.push(std::mem::take(&mut run));
        }
        out.push(vec![match s {
            CStmt::Barrier => continue, // the split itself; nothing to emit
            CStmt::If {
                prep,
                cond,
                uniform,
                accept,
                reject,
            } => {
                if !*uniform {
                    // Barriers only sit under uniform predicates unless verification was skipped.
                    return Err(EmitError::Validation(
                        "barrier under a divergent `If`".into(),
                    ));
                }
                CStmt::If {
                    prep: prep.clone(),
                    cond: *cond,
                    uniform: true,
                    accept: nest(accept)?,
                    reject: nest(reject)?,
                }
            }
            CStmt::Loop {
                prep,
                count,
                index,
                accs,
                body,
            } => CStmt::Loop {
                prep: prep.clone(),
                count: *count,
                index: *index,
                accs: accs.clone(),
                body: nest(body)?,
            },
            other => other.clone(),
        }]);
    }
    if !run.is_empty() {
        out.push(run);
    }
    Ok(out)
}

/// Split a nested body, wrapping each barrier-free run in [`CStmt::Lanes`].
fn nest(body: &[CStmt]) -> Result<Vec<CStmt>, EmitError> {
    if !body.iter().any(CStmt::is_collective) {
        return Ok(vec![CStmt::Lanes(body.to_vec())]);
    }
    Ok(block(body)?.into_iter().map(CStmt::Lanes).collect())
}
