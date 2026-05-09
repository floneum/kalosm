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
            TileStmt::StoreIndexed(store) => Self::mark_tile_expr_live(ir, &store.value, live),
            TileStmt::StoreLocal { value, .. } => Self::mark_tile_expr_live(ir, value, live),
            TileStmt::Emit { value } => Self::mark_tile_expr_live(ir, value, live),
            TileStmt::StoreWorkgroup { dst, value, .. } => {
                Self::mark_tile_live(ir, *dst, live);
                Self::mark_tile_expr_live(ir, value, live);
            }
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
            TileStmt::Break | TileStmt::Return => {}
        }
    }

    pub(super) fn uses_cooperative_matrix(ir: &KernelIr) -> bool {
        ir.body().ops().iter().any(|op| {
            let Op::TileProgram(op) = op;
            op.body.iter().any(Self::tile_stmt_uses_cooperative_matrix)
        })
    }

    pub(super) fn uses_subgroup_reduce(ir: &KernelIr) -> bool {
        Self::tile_programs_expr_any(ir, |expr| matches!(expr, TileExpr::SubgroupReduce { .. }))
            || ir
                .pinned_values
                .iter()
                .any(Self::tile_expr_uses_subgroup_reduce)
            || ir.loop_fold_groups.iter().any(|g| {
                g.bodies
                    .iter()
                    .any(|b| Self::tile_expr_uses_subgroup_reduce(b))
            })
    }

    pub(super) fn uses_index_kind(ir: &KernelIr, kind: SubgroupIndexKind) -> bool {
        Self::tile_programs_index_any(ir, |expr| Self::index_expr_is_kind(expr, kind))
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

    pub(super) fn tile_programs_index_any<F>(ir: &KernelIr, mut pred: F) -> bool
    where
        F: FnMut(&TileIndexExpr) -> bool,
    {
        ir.body().ops().iter().any(|op| {
            let Op::TileProgram(op) = op;
            op.body
                .iter()
                .any(|stmt| Self::tile_stmt_index_any(stmt, &mut pred))
        })
    }

    pub(super) fn tile_programs_expr_any<F>(ir: &KernelIr, mut pred: F) -> bool
    where
        F: FnMut(&TileExpr) -> bool,
    {
        ir.body().ops().iter().any(|op| {
            let Op::TileProgram(op) = op;
            op.body
                .iter()
                .any(|stmt| Self::tile_stmt_expr_any(stmt, &mut pred))
        })
    }

    pub(super) fn tile_stmt_expr_any<F>(stmt: &TileStmt, pred: &mut F) -> bool
    where
        F: FnMut(&TileExpr) -> bool,
    {
        match stmt {
            TileStmt::Store(store) => Self::tile_expr_any(&store.value, pred),
            TileStmt::StoreIndexed(store) => Self::tile_expr_any(&store.value, pred),
            TileStmt::StoreLocal { value, .. }
            | TileStmt::Emit { value }
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
            TileStmt::Loop { body } | TileStmt::WhileTrue { body, .. } => {
                body.iter().any(|stmt| Self::tile_stmt_expr_any(stmt, pred))
            }
            TileStmt::StoreCoopAcc { .. }
            | TileStmt::CopyToWorkgroupTile { .. }
            | TileStmt::CopyQuantToWorkgroupTile { .. }
            | TileStmt::ZeroCoopAcc { .. }
            | TileStmt::Barrier
            | TileStmt::LoadCoop { .. }
            | TileStmt::Mma { .. }
            | TileStmt::Break
            | TileStmt::Return => false,
        }
    }

    pub(super) fn tile_stmt_index_any<F>(stmt: &TileStmt, pred: &mut F) -> bool
    where
        F: FnMut(&TileIndexExpr) -> bool,
    {
        match stmt {
            TileStmt::Store(store) => {
                Self::tile_index_expr_any(&store.row, pred)
                    || Self::tile_index_expr_any(&store.col, pred)
                    || Self::tile_mask_expr_index_any(&store.mask, pred)
                    || Self::tile_expr_index_any(&store.value, pred)
            }
            TileStmt::StoreIndexed(store) => {
                Self::tile_index_expr_any(&store.index, pred)
                    || Self::tile_mask_expr_index_any(&store.mask, pred)
                    || Self::tile_expr_index_any(&store.value, pred)
            }
            TileStmt::StoreLocal { value, .. } | TileStmt::Emit { value } => {
                Self::tile_expr_index_any(value, pred)
            }
            TileStmt::StoreWorkgroup { index, value, .. } => {
                Self::tile_index_expr_any(index, pred) || Self::tile_expr_index_any(value, pred)
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
            }
            | TileStmt::StoreCoopAcc {
                row: row_offset,
                col: col_offset,
                ..
            }
            | TileStmt::LoadCoop {
                row: row_offset,
                col: col_offset,
                ..
            } => {
                Self::tile_index_expr_any(row_offset, pred)
                    || Self::tile_index_expr_any(col_offset, pred)
            }
            TileStmt::If {
                condition,
                accept,
                reject,
            } => {
                Self::tile_expr_index_any(condition, pred)
                    || accept
                        .iter()
                        .chain(reject.iter())
                        .any(|stmt| Self::tile_stmt_index_any(stmt, pred))
            }
            TileStmt::Loop { body } | TileStmt::WhileTrue { body, .. } => body
                .iter()
                .any(|stmt| Self::tile_stmt_index_any(stmt, pred)),
            TileStmt::ZeroCoopAcc { .. }
            | TileStmt::Barrier
            | TileStmt::Mma { .. }
            | TileStmt::Break
            | TileStmt::Return => false,
        }
    }

    pub(super) fn tile_expr_any<F>(expr: &TileExpr, pred: &mut F) -> bool
    where
        F: FnMut(&TileExpr) -> bool,
    {
        pred(expr) || Self::tile_expr_children_any(expr, |child| Self::tile_expr_any(child, pred))
    }

    pub(super) fn tile_expr_children_any<F>(expr: &TileExpr, mut pred: F) -> bool
    where
        F: FnMut(&TileExpr) -> bool,
    {
        match expr {
            TileExpr::Scalar(TileScalarExpr::Reduce { value, .. })
            | TileExpr::Scalar(TileScalarExpr::LoopReduce { value, .. }) => pred(value),
            TileExpr::Scalar(TileScalarExpr::Literal(_))
            | TileExpr::Load(_)
            | TileExpr::LoadLinear(_)
            | TileExpr::LoadVec4(_)
            | TileExpr::LoadWorkgroup { .. }
            | TileExpr::LoadLocal(_)
            | TileExpr::QuantizedLoad(_)
            | TileExpr::Full(_)
            | TileExpr::Literal(_)
            | TileExpr::Index(_)
            | TileExpr::QuantizedBlockLane { .. }
            | TileExpr::PinnedRef { .. }
            | TileExpr::LoopFoldGroupOutput { .. } => false,
            TileExpr::Unary { value, .. }
            | TileExpr::Cast { value, .. }
            | TileExpr::Bitcast { value, .. }
            | TileExpr::LoopFold { value, .. }
            | TileExpr::GroupReduce { value, .. }
            | TileExpr::SubgroupReduce { value, .. }
            | TileExpr::Vec4Splat { value } => pred(value),
            TileExpr::Binary { left, right, .. }
            | TileExpr::Compare { left, right, .. }
            | TileExpr::Vec4Dot { left, right } => pred(left) || pred(right),
            TileExpr::Select {
                condition,
                accept,
                reject,
            } => pred(condition) || pred(accept) || pred(reject),
            TileExpr::Sum { values } => values.iter().any(|expr| pred(expr)),
            TileExpr::Dot4 { a, b } => a.iter().chain(b.iter()).any(|expr| pred(expr)),
            TileExpr::QuantizedQ8_0Dot8 { a, .. } => a.iter().any(|expr| pred(expr)),
            TileExpr::QuantizedVecDot { a, .. }
            | TileExpr::QuantizedQ6KGgmlDot { a, .. } => a.iter().any(|expr| pred(expr)),
            TileExpr::QuantizedQ4KGgmlDot {
                a_low,
                a_high,
                sums,
                ..
            } => a_low
                .iter()
                .chain(a_high.iter())
                .chain(sums.iter())
                .any(|expr| pred(expr)),
        }
    }

    pub(super) fn tile_expr_index_any<F>(expr: &TileExpr, pred: &mut F) -> bool
    where
        F: FnMut(&TileIndexExpr) -> bool,
    {
        match expr {
            TileExpr::Load(load) => {
                Self::tile_index_expr_any(&load.row, pred)
                    || Self::tile_index_expr_any(&load.col, pred)
                    || Self::tile_mask_expr_index_any(&load.mask, pred)
            }
            TileExpr::LoadLinear(load) => {
                Self::tile_index_expr_any(&load.index, pred)
                    || Self::tile_mask_expr_index_any(&load.mask, pred)
            }
            TileExpr::LoadVec4(load) => {
                Self::tile_index_expr_any(&load.index, pred)
                    || Self::tile_mask_expr_index_any(&load.mask, pred)
            }
            TileExpr::LoadWorkgroup { index, .. } => Self::tile_index_expr_any(index, pred),
            TileExpr::QuantizedLoad(load) => {
                Self::tile_index_expr_any(&load.row, pred)
                    || Self::tile_index_expr_any(&load.col, pred)
                    || Self::tile_mask_expr_index_any(&load.mask, pred)
            }
            TileExpr::QuantizedBlockLane {
                k_base, col, mask, ..
            } => {
                Self::tile_index_expr_any(k_base, pred)
                    || Self::tile_index_expr_any(col, pred)
                    || Self::tile_mask_expr_index_any(mask, pred)
            }
            TileExpr::QuantizedQ8_0Dot8 {
                k_base, col, mask, ..
            }
            | TileExpr::QuantizedVecDot {
                k_base, col, mask, ..
            } => {
                Self::tile_index_expr_any(k_base, pred)
                    || Self::tile_index_expr_any(col, pred)
                    || Self::tile_mask_expr_index_any(mask, pred)
                    || Self::tile_expr_children_any(expr, |child| {
                        Self::tile_expr_index_any(child, pred)
                    })
            }
            TileExpr::QuantizedQ4KGgmlDot {
                block,
                iq,
                ir,
                col,
                mask,
                ..
            } => {
                Self::tile_index_expr_any(block, pred)
                    || Self::tile_index_expr_any(iq, pred)
                    || Self::tile_index_expr_any(ir, pred)
                    || Self::tile_index_expr_any(col, pred)
                    || Self::tile_mask_expr_index_any(mask, pred)
                    || Self::tile_expr_children_any(expr, |child| {
                        Self::tile_expr_index_any(child, pred)
                    })
            }
            TileExpr::QuantizedQ6KGgmlDot {
                block,
                ip,
                il,
                col,
                mask,
                ..
            } => {
                Self::tile_index_expr_any(block, pred)
                    || Self::tile_index_expr_any(ip, pred)
                    || Self::tile_index_expr_any(il, pred)
                    || Self::tile_index_expr_any(col, pred)
                    || Self::tile_mask_expr_index_any(mask, pred)
                    || Self::tile_expr_children_any(expr, |child| {
                        Self::tile_expr_index_any(child, pred)
                    })
            }
            TileExpr::Index(index) => Self::tile_index_expr_any(index, pred),
            TileExpr::LoadLocal(_)
            | TileExpr::Full(_)
            | TileExpr::Literal(_)
            | TileExpr::PinnedRef { .. }
            | TileExpr::LoopFoldGroupOutput { .. } => false,
            _ => Self::tile_expr_children_any(expr, |child| Self::tile_expr_index_any(child, pred)),
        }
    }

    pub(super) fn tile_index_expr_any<F>(expr: &TileIndexExpr, pred: &mut F) -> bool
    where
        F: FnMut(&TileIndexExpr) -> bool,
    {
        pred(expr)
            || match expr {
                TileIndexExpr::Add(left, right) => {
                    Self::tile_index_expr_any(left, pred) || Self::tile_index_expr_any(right, pred)
                }
                TileIndexExpr::Mul(value, _)
                | TileIndexExpr::Div(value, _)
                | TileIndexExpr::Mod(value, _) => Self::tile_index_expr_any(value, pred),
                TileIndexExpr::Value(value) => Self::tile_expr_index_any(value, pred),
                TileIndexExpr::Lane
                | TileIndexExpr::LoopIndex
                | TileIndexExpr::ProgramId(_)
                | TileIndexExpr::SubgroupId
                | TileIndexExpr::SubgroupLane
                | TileIndexExpr::SubgroupSize
                | TileIndexExpr::NumSubgroups
                | TileIndexExpr::Literal(_) => false,
            }
    }

    pub(super) fn tile_mask_expr_index_any<F>(expr: &TileMaskExpr, pred: &mut F) -> bool
    where
        F: FnMut(&TileIndexExpr) -> bool,
    {
        match expr {
            TileMaskExpr::True => false,
            TileMaskExpr::Compare { left, right, .. } => {
                Self::tile_index_expr_any(left, pred) || Self::tile_index_expr_any(right, pred)
            }
            TileMaskExpr::And(left, right) => {
                Self::tile_mask_expr_index_any(left, pred)
                    || Self::tile_mask_expr_index_any(right, pred)
            }
        }
    }

    fn tile_stmt_uses_cooperative_matrix(stmt: &TileStmt) -> bool {
        match stmt {
            TileStmt::Store(_)
            | TileStmt::StoreIndexed(_)
            | TileStmt::StoreLocal { .. }
            | TileStmt::Emit { .. }
            | TileStmt::StoreWorkgroup { .. }
            | TileStmt::Barrier
            | TileStmt::Break
            | TileStmt::Return => false,
            TileStmt::ZeroCoopAcc { .. }
            | TileStmt::CopyToWorkgroupTile { .. }
            | TileStmt::CopyQuantToWorkgroupTile { .. }
            | TileStmt::LoadCoop { .. }
            | TileStmt::Mma { .. }
            | TileStmt::StoreCoopAcc { .. } => true,
            TileStmt::WhileTrue { body, .. } => {
                body.iter().any(Self::tile_stmt_uses_cooperative_matrix)
            }
            TileStmt::If { accept, reject, .. } => accept
                .iter()
                .chain(reject.iter())
                .any(Self::tile_stmt_uses_cooperative_matrix),
            TileStmt::Loop { body } => body.iter().any(Self::tile_stmt_uses_cooperative_matrix),
        }
    }

    fn tile_expr_uses_index_kind(expr: &TileExpr, kind: SubgroupIndexKind) -> bool {
        Self::tile_expr_index_any(expr, &mut |expr| Self::index_expr_is_kind(expr, kind))
    }

    fn index_expr_is_kind(expr: &TileIndexExpr, kind: SubgroupIndexKind) -> bool {
        match expr {
            TileIndexExpr::SubgroupId => kind == SubgroupIndexKind::SubgroupId,
            TileIndexExpr::SubgroupLane => kind == SubgroupIndexKind::SubgroupLane,
            TileIndexExpr::SubgroupSize => kind == SubgroupIndexKind::SubgroupSize,
            TileIndexExpr::NumSubgroups => kind == SubgroupIndexKind::NumSubgroups,
            _ => false,
        }
    }

    fn tile_expr_uses_subgroup_reduce(expr: &TileExpr) -> bool {
        Self::tile_expr_any(expr, &mut |expr| {
            matches!(expr, TileExpr::SubgroupReduce { .. })
        })
    }

    fn mark_tile_expr_live(ir: &KernelIr, expr: &TileExpr, live: &mut [bool]) {
        match expr {
            TileExpr::LoadWorkgroup { src, .. } => Self::mark_tile_live(ir, *src, live),
            TileExpr::Scalar(
                TileScalarExpr::Reduce { scratch, .. } | TileScalarExpr::LoopReduce { scratch, .. },
            )
            | TileExpr::GroupReduce { scratch, .. } => {
                Self::mark_tile_live(ir, *scratch, live);
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
