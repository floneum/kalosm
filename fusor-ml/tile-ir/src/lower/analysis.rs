use super::*;

/// Which subgroup builtins are referenced anywhere in the IR. Aggregated by
/// `Lowerer::subgroup_index_usage` so that one tree walk fills all four flags
/// instead of four separate single-flag walks.
#[derive(Copy, Clone, Default, PartialEq, Eq)]
pub(super) struct SubgroupIndexUsage {
    pub subgroup_id: bool,
    pub subgroup_lane: bool,
    pub subgroup_size: bool,
    pub num_subgroups: bool,
}

impl<'a> Lowerer<'a> {
    pub(super) fn max_tile_program_block(ir: &KernelIr) -> u32 {
        ir.body().block
    }

    pub(super) fn live_tiles(ir: &KernelIr) -> Vec<bool> {
        let mut live = vec![false; ir.tiles().len()];
        for stmt in &ir.body().body {
            Self::mark_tile_stmt_live(stmt, &mut live);
        }
        live
    }

    fn mark_tile_stmt_live(stmt: &TileStmt, live: &mut [bool]) {
        match stmt {
            TileStmt::Store(store) => Self::mark_tile_expr_live(&store.value, live),
            TileStmt::StoreIndexed(store) => Self::mark_tile_expr_live(&store.value, live),
            TileStmt::StoreLocal { value, .. } => Self::mark_tile_expr_live(value, live),
            TileStmt::StoreWorkgroup { dst, value, .. } => {
                Self::mark_tile_live(*dst, live);
                Self::mark_tile_expr_live(value, live);
            }
            TileStmt::CopyToWorkgroupTile { dst, .. } => Self::mark_tile_live(*dst, live),
            TileStmt::LoadCoop { tile, .. } => Self::mark_tile_live(*tile, live),
            TileStmt::ZeroCoopAcc { .. }
            | TileStmt::Barrier
            | TileStmt::Mma { .. }
            | TileStmt::StoreCoopAcc { .. } => {}
            TileStmt::If {
                condition,
                accept,
                reject,
            } => {
                Self::mark_tile_expr_live(condition, live);
                for s in accept.iter().chain(reject.iter()) {
                    Self::mark_tile_stmt_live(s, live);
                }
            }
            TileStmt::Loop { body } => {
                for s in body {
                    Self::mark_tile_stmt_live(s, live);
                }
            }
            TileStmt::Fold {
                count,
                body: fold_body,
                accumulators,
                ..
            } => {
                Self::mark_tile_expr_live(count, live);
                for stmt in fold_body {
                    Self::mark_tile_stmt_live(stmt, live);
                }
                for acc in accumulators {
                    Self::mark_tile_expr_live(&acc.init, live);
                    Self::mark_tile_expr_live(&acc.update, live);
                }
            }
            TileStmt::Break | TileStmt::Return => {}
        }
    }

    pub(super) fn uses_cooperative_matrix(ir: &KernelIr) -> bool {
        ir.body()
            .body
            .iter()
            .any(Self::tile_stmt_uses_cooperative_matrix)
    }

    pub(super) fn uses_subgroup_reduce(ir: &KernelIr) -> bool {
        Self::tile_programs_expr_any(ir, |expr| matches!(expr, Expr::SubgroupReduce { .. }))
    }

    pub(super) fn subgroup_index_usage(ir: &KernelIr) -> SubgroupIndexUsage {
        let mut usage = SubgroupIndexUsage::default();
        Self::tile_programs_expr_any(ir, |expr| {
            if let Expr::Builtin(builtin) = expr {
                use crate::ir::Builtin;
                match builtin {
                    Builtin::SubgroupId => usage.subgroup_id = true,
                    Builtin::SubgroupLane => usage.subgroup_lane = true,
                    Builtin::SubgroupSize => usage.subgroup_size = true,
                    Builtin::NumSubgroups => usage.num_subgroups = true,
                    _ => {}
                }
            }
            // Always continue: we want to collect every flag.
            false
        });
        usage
    }

    pub(super) fn tile_programs_expr_any<F>(ir: &KernelIr, mut pred: F) -> bool
    where
        F: FnMut(&Expr) -> bool,
    {
        ir.body()
            .body
            .iter()
            .any(|stmt| Self::tile_stmt_expr_any(stmt, &mut pred))
    }

    pub(super) fn tile_stmt_expr_any<F>(stmt: &TileStmt, pred: &mut F) -> bool
    where
        F: FnMut(&Expr) -> bool,
    {
        match stmt {
            TileStmt::Store(store) => Self::tile_expr_any(&store.value, pred),
            TileStmt::StoreIndexed(store) => Self::tile_expr_any(&store.value, pred),
            TileStmt::StoreLocal { value, .. } | TileStmt::StoreWorkgroup { value, .. } => {
                Self::tile_expr_any(value, pred)
            }
            TileStmt::If {
                condition,
                accept,
                reject,
            } => {
                Self::tile_expr_any(condition, pred)
                    || accept
                        .iter()
                        .chain(reject.iter())
                        .any(|stmt| Self::tile_stmt_expr_any(stmt, pred))
            }
            TileStmt::Loop { body } => body.iter().any(|stmt| Self::tile_stmt_expr_any(stmt, pred)),
            TileStmt::Fold {
                count,
                body: fold_body,
                accumulators,
                ..
            } => {
                Self::tile_expr_any(count, pred)
                    || fold_body.iter().any(|s| Self::tile_stmt_expr_any(s, pred))
                    || accumulators.iter().any(|acc| {
                        Self::tile_expr_any(&acc.init, pred)
                            || Self::tile_expr_any(&acc.update, pred)
                    })
            }
            TileStmt::StoreCoopAcc { row, col, .. } => {
                Self::tile_expr_any(row, pred) || Self::tile_expr_any(col, pred)
            }
            TileStmt::CopyToWorkgroupTile {
                row_offset,
                col_offset,
                ..
            } => Self::tile_expr_any(row_offset, pred) || Self::tile_expr_any(col_offset, pred),
            TileStmt::LoadCoop { row, col, .. } => {
                Self::tile_expr_any(row, pred) || Self::tile_expr_any(col, pred)
            }
            TileStmt::ZeroCoopAcc { .. }
            | TileStmt::Barrier
            | TileStmt::Mma { .. }
            | TileStmt::Break
            | TileStmt::Return => false,
        }
    }

    pub(super) fn tile_expr_any<F>(expr: &Expr, pred: &mut F) -> bool
    where
        F: FnMut(&Expr) -> bool,
    {
        pred(expr) || Self::tile_expr_children_any(expr, |child| Self::tile_expr_any(child, pred))
    }

    pub(super) fn tile_expr_children_any<F>(expr: &Expr, mut pred: F) -> bool
    where
        F: FnMut(&Expr) -> bool,
    {
        match expr {
            Expr::Reduce { value, .. } => pred(value),
            Expr::LoadLocal(_) | Expr::Literal(_) | Expr::Builtin(_) => false,
            Expr::Load(load) => {
                pred(&load.row) || pred(&load.col) || pred(&load.mask) || pred(&load.fill)
            }
            Expr::LoadLinear(load) => pred(&load.index) || pred(&load.mask) || pred(&load.fill),
            Expr::LoadWorkgroup { index, .. } => pred(index),
            Expr::QuantizedBlockLane {
                k_base,
                col,
                mask,
                fill,
                ..
            } => pred(k_base) || pred(col) || pred(mask) || pred(fill),
            Expr::Unary { value, .. }
            | Expr::Cast { value, .. }
            | Expr::Bitcast { value, .. }
            | Expr::SubgroupReduce { value, .. } => pred(value),
            Expr::Binary { left, right, .. }
            | Expr::Compare { left, right, .. }
            | Expr::VectorDot { left, right, .. } => pred(left) || pred(right),
            Expr::Select {
                condition,
                accept,
                reject,
            } => pred(condition) || pred(accept) || pred(reject),
            Expr::ComposeVector { values, .. } => values.iter().any(pred),
            Expr::QuantizedDot {
                activations,
                k,
                col,
                mask,
                fill,
                ..
            } => {
                let activations_match = match activations {
                    PackedActivations::F32(a) | PackedActivations::Q8(a) => a.iter().any(&mut pred),
                    PackedActivations::Q4KGgml { low, high, sums } => low
                        .iter()
                        .chain(high.iter())
                        .chain(sums.iter())
                        .any(&mut pred),
                };
                let k_match = match k {
                    DotK::Base(k_base) => pred(k_base),
                    DotK::Block { block, c0, c1 } => pred(block) || pred(c0) || pred(c1),
                };
                activations_match || k_match || pred(col) || pred(mask) || pred(fill)
            }
        }
    }

    fn tile_stmt_uses_cooperative_matrix(stmt: &TileStmt) -> bool {
        Self::tile_stmt_tree_any(stmt, &mut |s| {
            matches!(
                s,
                TileStmt::ZeroCoopAcc { .. }
                    | TileStmt::CopyToWorkgroupTile { .. }
                    | TileStmt::LoadCoop { .. }
                    | TileStmt::Mma { .. }
                    | TileStmt::StoreCoopAcc { .. }
            )
        })
    }

    /// Pre-order tree-any over a `TileStmt`: returns true if `pred` matches
    /// the statement itself or any nested child. Visitors that test a
    /// statement-shape predicate (no expr walking) compose on top of this.
    fn tile_stmt_tree_any(stmt: &TileStmt, pred: &mut impl FnMut(&TileStmt) -> bool) -> bool {
        if pred(stmt) {
            return true;
        }
        match stmt {
            TileStmt::If { accept, reject, .. } => accept
                .iter()
                .chain(reject.iter())
                .any(|s| Self::tile_stmt_tree_any(s, pred)),
            TileStmt::Loop { body } | TileStmt::Fold { body, .. } => {
                body.iter().any(|s| Self::tile_stmt_tree_any(s, pred))
            }
            _ => false,
        }
    }

    fn mark_tile_expr_live(expr: &Expr, live: &mut [bool]) {
        match expr {
            Expr::LoadWorkgroup { src, .. } => Self::mark_tile_live(*src, live),
            Expr::Reduce { scratch, .. } => {
                Self::mark_tile_live(*scratch, live);
            }
            _ => {}
        }
        Self::tile_expr_children_any(expr, |child| {
            Self::mark_tile_expr_live(child, live);
            false
        });
    }

    pub(super) fn mark_tile_live(tile: TileRef, live: &mut [bool]) {
        if let Some(slot) = live.get_mut(tile.id.index()) {
            *slot = true;
        }
    }
}
