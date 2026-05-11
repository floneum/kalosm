use super::*;

#[derive(Copy, Clone, PartialEq, Eq)]
pub(super) enum SubgroupIndexKind {
    SubgroupId,
    SubgroupLane,
    SubgroupSize,
    NumSubgroups,
}

impl<'a> Lowerer<'a> {
    pub(super) fn max_tile_program_block(ir: &KernelIr) -> u32 {
        ir.body().block
    }

    pub(super) fn live_tiles(ir: &KernelIr) -> Vec<bool> {
        let mut live = vec![false; ir.tiles().len()];
        for stmt in &ir.body().body {
            Self::mark_tile_stmt_live(ir, stmt, &mut live);
        }
        live
    }

    fn mark_tile_stmt_live(ir: &KernelIr, stmt: &TileStmt, live: &mut [bool]) {
        match stmt {
            TileStmt::Store(store) => Self::mark_tile_expr_live(ir, &store.value, live),
            TileStmt::StoreIndexed(store) => Self::mark_tile_expr_live(ir, &store.value, live),
            TileStmt::StoreLocal { value, .. } => Self::mark_tile_expr_live(ir, value, live),
            TileStmt::StoreWorkgroup { dst, value, .. } => {
                Self::mark_tile_live(ir, *dst, live);
                Self::mark_tile_expr_live(ir, value, live);
            }
            TileStmt::CopyToWorkgroupTile { dst, .. } => {
                if let Some(slot) = live.get_mut(dst.id.index()) {
                    *slot = true;
                }
            }
            TileStmt::LoadCoop { tile, .. } => {
                if let Some(slot) = live.get_mut(tile.id.index()) {
                    *slot = true;
                }
            }
            TileStmt::ZeroCoopAcc { .. }
            | TileStmt::Barrier
            | TileStmt::Mma { .. }
            | TileStmt::StoreCoopAcc { .. } => {}
            TileStmt::If {
                condition,
                accept,
                reject,
            } => {
                Self::mark_tile_expr_live(ir, condition, live);
                for s in accept.iter().chain(reject.iter()) {
                    Self::mark_tile_stmt_live(ir, s, live);
                }
            }
            TileStmt::Loop { body } => {
                for s in body {
                    Self::mark_tile_stmt_live(ir, s, live);
                }
            }
            TileStmt::Fold {
                count,
                body: fold_body,
                accumulators,
                ..
            } => {
                Self::mark_tile_expr_live(ir, count, live);
                for stmt in fold_body {
                    Self::mark_tile_stmt_live(ir, stmt, live);
                }
                for acc in accumulators {
                    Self::mark_tile_expr_live(ir, &acc.init, live);
                    Self::mark_tile_expr_live(ir, &acc.update, live);
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

    pub(super) fn uses_index_kind(ir: &KernelIr, kind: SubgroupIndexKind) -> bool {
        Self::tile_programs_expr_any(ir, |expr| match expr {
            Expr::Builtin(builtin) => Self::builtin_is_kind(*builtin, kind),
            _ => false,
        })
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
            TileStmt::StoreLocal { value, .. }
            | TileStmt::StoreWorkgroup { value, .. } => Self::tile_expr_any(value, pred),
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
            TileStmt::Loop { body } => {
                body.iter().any(|stmt| Self::tile_stmt_expr_any(stmt, pred))
            }
            TileStmt::Fold {
                count,
                body: fold_body,
                accumulators,
                ..
            } => {
                Self::tile_expr_any(count, pred)
                    || fold_body
                        .iter()
                        .any(|s| Self::tile_stmt_expr_any(s, pred))
                    || accumulators.iter().any(|acc| {
                        Self::tile_expr_any(&acc.init, pred)
                            || Self::tile_expr_any(&acc.update, pred)
                    })
            }
            TileStmt::StoreCoopAcc { .. }
            | TileStmt::CopyToWorkgroupTile { .. }
            | TileStmt::ZeroCoopAcc { .. }
            | TileStmt::Barrier
            | TileStmt::LoadCoop { .. }
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
            Expr::LoadLocal(_)
            | Expr::Literal(_)
            | Expr::Builtin(_) => false,
            Expr::Load(load) => {
                pred(&load.row) || pred(&load.col) || pred(&load.mask) || pred(&load.fill)
            }
            Expr::LoadLinear(load) => {
                pred(&load.index) || pred(&load.mask) || pred(&load.fill)
            }
            Expr::LoadWorkgroup { index, .. } => pred(index),
            Expr::QuantizedLoad(load) => {
                pred(&load.row) || pred(&load.col) || pred(&load.mask) || pred(&load.fill)
            }
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
            | Expr::Vec4Dot { left, right } => pred(left) || pred(right),
            Expr::Select {
                condition,
                accept,
                reject,
            } => pred(condition) || pred(accept) || pred(reject),
            Expr::Compose4 { values } => values.iter().any(|expr| pred(expr)),
            Expr::QuantizedDot {
                activations,
                k,
                col,
                mask,
                fill,
                ..
            } => {
                let activations_match = match activations {
                    PackedActivations::F32(a) | PackedActivations::Q8(a) => {
                        a.iter().any(|expr| pred(expr))
                    }
                    PackedActivations::Q4KGgml { low, high, sums } => low
                        .iter()
                        .chain(high.iter())
                        .chain(sums.iter())
                        .any(|expr| pred(expr)),
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
        match stmt {
            TileStmt::Store(_)
            | TileStmt::StoreIndexed(_)
            | TileStmt::StoreLocal { .. }
            | TileStmt::StoreWorkgroup { .. }
            | TileStmt::Barrier
            | TileStmt::Break
            | TileStmt::Return => false,
            TileStmt::ZeroCoopAcc { .. }
            | TileStmt::CopyToWorkgroupTile { .. }
            | TileStmt::LoadCoop { .. }
            | TileStmt::Mma { .. }
            | TileStmt::StoreCoopAcc { .. } => true,
            TileStmt::If { accept, reject, .. } => accept
                .iter()
                .chain(reject.iter())
                .any(Self::tile_stmt_uses_cooperative_matrix),
            TileStmt::Loop { body } => body.iter().any(Self::tile_stmt_uses_cooperative_matrix),
            TileStmt::Fold { body, .. } => {
                body.iter().any(Self::tile_stmt_uses_cooperative_matrix)
            }
        }
    }

    fn builtin_is_kind(builtin: crate::ir::Builtin, kind: SubgroupIndexKind) -> bool {
        use crate::ir::Builtin;
        match builtin {
            Builtin::SubgroupId => kind == SubgroupIndexKind::SubgroupId,
            Builtin::SubgroupLane => kind == SubgroupIndexKind::SubgroupLane,
            Builtin::SubgroupSize => kind == SubgroupIndexKind::SubgroupSize,
            Builtin::NumSubgroups => kind == SubgroupIndexKind::NumSubgroups,
            _ => false,
        }
    }

    fn mark_tile_expr_live(ir: &KernelIr, expr: &Expr, live: &mut [bool]) {
        match expr {
            Expr::LoadWorkgroup { src, .. } => Self::mark_tile_live(ir, *src, live),
            Expr::Reduce { scratch, .. } => {
                Self::mark_tile_live(ir, *scratch, live);
            }
            _ => {}
        }
        Self::tile_expr_children_any(expr, |child| {
            Self::mark_tile_expr_live(ir, child, live);
            false
        });
    }

    pub(super) fn mark_tile_live(ir: &KernelIr, tile: TileRef, live: &mut [bool]) {
        if ir.tiles().get(tile.id.index()).is_none() {
            return;
        }
        live[tile.id.index()] = true;
    }
}
