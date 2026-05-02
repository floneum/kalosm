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
        ir.body()
            .ops()
            .iter()
            .map(|op| match op {
                Op::TileProgram(op) => op.block,
            })
            .max()
            .unwrap_or(0)
    }

    pub(super) fn live_tiles(ir: &KernelIr) -> Vec<bool> {
        let mut live = vec![false; ir.tiles().len()];
        for op in ir.body().ops() {
            let Op::TileProgram(op) = op;
            for stmt in &op.body {
                Self::mark_tile_stmt_live(ir, stmt, &mut live);
            }
        }
        live
    }

    fn mark_tile_stmt_live(ir: &KernelIr, stmt: &TileStmt, live: &mut [bool]) {
        match stmt {
            TileStmt::Store(store) => Self::mark_tile_expr_live(ir, &store.value, live),
            TileStmt::CopyToWorkgroupTile { dst, .. }
            | TileStmt::CopyQuantToWorkgroupTile { dst, .. } => {
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
            TileStmt::WhileTrue { body, .. } => {
                for s in body {
                    Self::mark_tile_stmt_live(ir, s, live);
                }
            }
        }
    }

    pub(super) fn uses_cooperative_matrix(ir: &KernelIr) -> bool {
        ir.body().ops().iter().any(|op| {
            let Op::TileProgram(op) = op;
            op.body.iter().any(Self::tile_stmt_uses_cooperative_matrix)
        })
    }

    pub(super) fn uses_subgroup_reduce(ir: &KernelIr) -> bool {
        ir.body().ops().iter().any(|op| {
            let Op::TileProgram(op) = op;
            op.body.iter().any(Self::tile_stmt_uses_subgroup_reduce)
        })
    }

    pub(super) fn uses_index_kind(ir: &KernelIr, kind: SubgroupIndexKind) -> bool {
        let in_programs = ir.body().ops().iter().any(|op| {
            let Op::TileProgram(op) = op;
            op.body
                .iter()
                .any(|stmt| Self::tile_stmt_uses_index_kind(stmt, kind))
        });
        in_programs
            || ir
                .pinned_values
                .iter()
                .any(|v| Self::tile_expr_uses_index_kind(v, kind))
            || ir.loop_fold_groups.iter().any(|g| {
                g.bodies
                    .iter()
                    .any(|b| Self::tile_expr_uses_index_kind(b, kind))
            })
    }

    fn tile_stmt_uses_cooperative_matrix(stmt: &TileStmt) -> bool {
        match stmt {
            TileStmt::Store(_) | TileStmt::Barrier => false,
            TileStmt::ZeroCoopAcc { .. }
            | TileStmt::CopyToWorkgroupTile { .. }
            | TileStmt::CopyQuantToWorkgroupTile { .. }
            | TileStmt::LoadCoop { .. }
            | TileStmt::Mma { .. }
            | TileStmt::StoreCoopAcc { .. } => true,
            TileStmt::WhileTrue { body, .. } => {
                body.iter().any(Self::tile_stmt_uses_cooperative_matrix)
            }
        }
    }

    fn tile_stmt_uses_subgroup_reduce(stmt: &TileStmt) -> bool {
        match stmt {
            TileStmt::Store(store) => Self::tile_expr_uses_subgroup_reduce(&store.value),
            TileStmt::WhileTrue { body, .. } => {
                body.iter().any(Self::tile_stmt_uses_subgroup_reduce)
            }
            TileStmt::ZeroCoopAcc { .. }
            | TileStmt::CopyToWorkgroupTile { .. }
            | TileStmt::CopyQuantToWorkgroupTile { .. }
            | TileStmt::Barrier
            | TileStmt::LoadCoop { .. }
            | TileStmt::Mma { .. }
            | TileStmt::StoreCoopAcc { .. } => false,
        }
    }

    fn tile_stmt_uses_index_kind(stmt: &TileStmt, kind: SubgroupIndexKind) -> bool {
        match stmt {
            TileStmt::Store(store) => {
                Self::tile_index_expr_uses_kind(&store.row, kind)
                    || Self::tile_index_expr_uses_kind(&store.col, kind)
                    || Self::tile_mask_expr_uses_kind(&store.mask, kind)
                    || Self::tile_expr_uses_index_kind(&store.value, kind)
            }
            TileStmt::CopyToWorkgroupTile {
                row_offset,
                col_offset,
                ..
            }
            | TileStmt::CopyQuantToWorkgroupTile {
                row_offset,
                col_offset,
                ..
            } => {
                Self::tile_index_expr_uses_kind(row_offset, kind)
                    || Self::tile_index_expr_uses_kind(col_offset, kind)
            }
            TileStmt::StoreCoopAcc { row, col, .. } => {
                Self::tile_index_expr_uses_kind(row, kind)
                    || Self::tile_index_expr_uses_kind(col, kind)
            }
            TileStmt::WhileTrue { body, .. } => body
                .iter()
                .any(|stmt| Self::tile_stmt_uses_index_kind(stmt, kind)),
            TileStmt::ZeroCoopAcc { .. }
            | TileStmt::Barrier
            | TileStmt::LoadCoop { .. }
            | TileStmt::Mma { .. } => false,
        }
    }

    fn tile_expr_uses_index_kind(expr: &TileExpr, kind: SubgroupIndexKind) -> bool {
        match expr {
            TileExpr::Load(load) => {
                Self::tile_index_expr_uses_kind(&load.row, kind)
                    || Self::tile_index_expr_uses_kind(&load.col, kind)
                    || Self::tile_mask_expr_uses_kind(&load.mask, kind)
            }
            TileExpr::QuantizedLoad(load) => {
                Self::tile_index_expr_uses_kind(&load.row, kind)
                    || Self::tile_index_expr_uses_kind(&load.col, kind)
                    || Self::tile_mask_expr_uses_kind(&load.mask, kind)
            }
            TileExpr::QuantizedBlockLane {
                k_base, col, mask, ..
            } => {
                Self::tile_index_expr_uses_kind(k_base, kind)
                    || Self::tile_index_expr_uses_kind(col, kind)
                    || Self::tile_mask_expr_uses_kind(mask, kind)
            }
            TileExpr::Dot4 { a, b } => a
                .iter()
                .chain(b.iter())
                .any(|expr| Self::tile_expr_uses_index_kind(expr, kind)),
            TileExpr::QuantizedQ8_0Dot8 {
                a,
                k_base,
                col,
                mask,
                ..
            } => {
                a.iter()
                    .any(|expr| Self::tile_expr_uses_index_kind(expr, kind))
                    || Self::tile_index_expr_uses_kind(k_base, kind)
                    || Self::tile_index_expr_uses_kind(col, kind)
                    || Self::tile_mask_expr_uses_kind(mask, kind)
            }
            TileExpr::PinnedRef { .. } | TileExpr::LoopFoldGroupOutput { .. } => false,
            TileExpr::Index(idx) => Self::tile_index_expr_uses_kind(idx, kind),
            TileExpr::Full(_) | TileExpr::Literal(_) => false,
            TileExpr::Scalar(expr) => match expr {
                TileScalarExpr::Reduce { value, .. } | TileScalarExpr::LoopReduce { value, .. } => {
                    Self::tile_expr_uses_index_kind(value, kind)
                }
                TileScalarExpr::Literal(_) => false,
            },
            TileExpr::Unary { value, .. }
            | TileExpr::Cast { value, .. }
            | TileExpr::LoopFold { value, .. }
            | TileExpr::GroupReduce { value, .. }
            | TileExpr::SubgroupReduce { value, .. } => {
                Self::tile_expr_uses_index_kind(value, kind)
            }
            TileExpr::Binary { left, right, .. } | TileExpr::Compare { left, right, .. } => {
                Self::tile_expr_uses_index_kind(left, kind)
                    || Self::tile_expr_uses_index_kind(right, kind)
            }
            TileExpr::Select {
                condition,
                accept,
                reject,
            } => {
                Self::tile_expr_uses_index_kind(condition, kind)
                    || Self::tile_expr_uses_index_kind(accept, kind)
                    || Self::tile_expr_uses_index_kind(reject, kind)
            }
        }
    }

    fn tile_index_expr_uses_kind(expr: &TileIndexExpr, kind: SubgroupIndexKind) -> bool {
        match expr {
            TileIndexExpr::SubgroupId => kind == SubgroupIndexKind::SubgroupId,
            TileIndexExpr::SubgroupLane => kind == SubgroupIndexKind::SubgroupLane,
            TileIndexExpr::SubgroupSize => kind == SubgroupIndexKind::SubgroupSize,
            TileIndexExpr::NumSubgroups => kind == SubgroupIndexKind::NumSubgroups,
            TileIndexExpr::Lane
            | TileIndexExpr::LoopIndex
            | TileIndexExpr::ProgramId(_)
            | TileIndexExpr::Literal(_) => false,
            TileIndexExpr::Add(left, right) => {
                Self::tile_index_expr_uses_kind(left, kind)
                    || Self::tile_index_expr_uses_kind(right, kind)
            }
            TileIndexExpr::Mul(value, _)
            | TileIndexExpr::Div(value, _)
            | TileIndexExpr::Mod(value, _) => Self::tile_index_expr_uses_kind(value, kind),
            TileIndexExpr::Value(value) => Self::tile_expr_uses_index_kind(value, kind),
        }
    }

    fn tile_mask_expr_uses_kind(expr: &TileMaskExpr, kind: SubgroupIndexKind) -> bool {
        match expr {
            TileMaskExpr::True => false,
            TileMaskExpr::Compare { left, right, .. } => {
                Self::tile_index_expr_uses_kind(left, kind)
                    || Self::tile_index_expr_uses_kind(right, kind)
            }
            TileMaskExpr::And(left, right) => {
                Self::tile_mask_expr_uses_kind(left, kind)
                    || Self::tile_mask_expr_uses_kind(right, kind)
            }
        }
    }

    fn tile_expr_uses_subgroup_reduce(expr: &TileExpr) -> bool {
        match expr {
            TileExpr::SubgroupReduce { .. } => true,
            TileExpr::Load(_)
            | TileExpr::QuantizedLoad(_)
            | TileExpr::QuantizedBlockLane { .. }
            | TileExpr::Full(_)
            | TileExpr::Literal(_)
            | TileExpr::Index(_) => false,
            TileExpr::Dot4 { a, b } => a
                .iter()
                .chain(b.iter())
                .any(|expr| Self::tile_expr_uses_subgroup_reduce(expr)),
            TileExpr::QuantizedQ8_0Dot8 { a, .. } => a
                .iter()
                .any(|expr| Self::tile_expr_uses_subgroup_reduce(expr)),
            TileExpr::PinnedRef { .. } | TileExpr::LoopFoldGroupOutput { .. } => false,
            TileExpr::Scalar(expr) => match expr {
                TileScalarExpr::Reduce { value, .. } | TileScalarExpr::LoopReduce { value, .. } => {
                    Self::tile_expr_uses_subgroup_reduce(value)
                }
                TileScalarExpr::Literal(_) => false,
            },
            TileExpr::Unary { value, .. }
            | TileExpr::Cast { value, .. }
            | TileExpr::LoopFold { value, .. }
            | TileExpr::GroupReduce { value, .. } => Self::tile_expr_uses_subgroup_reduce(value),
            TileExpr::Binary { left, right, .. } | TileExpr::Compare { left, right, .. } => {
                Self::tile_expr_uses_subgroup_reduce(left)
                    || Self::tile_expr_uses_subgroup_reduce(right)
            }
            TileExpr::Select {
                condition,
                accept,
                reject,
            } => {
                Self::tile_expr_uses_subgroup_reduce(condition)
                    || Self::tile_expr_uses_subgroup_reduce(accept)
                    || Self::tile_expr_uses_subgroup_reduce(reject)
            }
        }
    }

    fn mark_tile_expr_live(ir: &KernelIr, expr: &TileExpr, live: &mut [bool]) {
        match expr {
            TileExpr::Load(_)
            | TileExpr::QuantizedLoad(_)
            | TileExpr::Full(_)
            | TileExpr::Literal(_)
            | TileExpr::Index(_) => {}
            TileExpr::Scalar(expr) => Self::mark_tile_scalar_expr_live(ir, expr, live),
            TileExpr::Unary { value, .. }
            | TileExpr::Cast { value, .. }
            | TileExpr::LoopFold { value, .. } => Self::mark_tile_expr_live(ir, value, live),
            TileExpr::GroupReduce { value, scratch, .. } => {
                Self::mark_tile_live(ir, *scratch, live);
                Self::mark_tile_expr_live(ir, value, live);
            }
            TileExpr::SubgroupReduce { value, .. } => {
                Self::mark_tile_expr_live(ir, value, live);
            }
            TileExpr::QuantizedBlockLane { .. } => {}
            TileExpr::Dot4 { a, b } => {
                for expr in a.iter().chain(b.iter()) {
                    Self::mark_tile_expr_live(ir, expr, live);
                }
            }
            TileExpr::QuantizedQ8_0Dot8 { a, .. } => {
                for expr in a {
                    Self::mark_tile_expr_live(ir, expr, live);
                }
            }
            TileExpr::PinnedRef { id } => {
                if let Some(value) = ir.pinned_values.get(id.index()) {
                    Self::mark_tile_expr_live(ir, value, live);
                }
            }
            TileExpr::LoopFoldGroupOutput { group, .. } => {
                if let Some(g) = ir.loop_fold_groups.get(group.index()) {
                    for body in &g.bodies {
                        Self::mark_tile_expr_live(ir, body, live);
                    }
                }
            }
            TileExpr::Binary { left, right, .. } => {
                Self::mark_tile_expr_live(ir, left, live);
                Self::mark_tile_expr_live(ir, right, live);
            }
            TileExpr::Select {
                condition,
                accept,
                reject,
            } => {
                Self::mark_tile_expr_live(ir, condition, live);
                Self::mark_tile_expr_live(ir, accept, live);
                Self::mark_tile_expr_live(ir, reject, live);
            }
            TileExpr::Compare { left, right, .. } => {
                Self::mark_tile_expr_live(ir, left, live);
                Self::mark_tile_expr_live(ir, right, live);
            }
        }
    }

    fn mark_tile_scalar_expr_live(ir: &KernelIr, expr: &TileScalarExpr, live: &mut [bool]) {
        match expr {
            TileScalarExpr::Reduce { value, scratch, .. }
            | TileScalarExpr::LoopReduce { value, scratch, .. } => {
                Self::mark_tile_live(ir, *scratch, live);
                Self::mark_tile_expr_live(ir, value, live);
            }
            TileScalarExpr::Literal(_) => {}
        }
    }

    pub(super) fn mark_tile_live(ir: &KernelIr, tile: TileRef, live: &mut [bool]) {
        if ir.tiles().get(tile.id.index()).is_none() {
            return;
        }
        live[tile.id.index()] = true;
    }
}
