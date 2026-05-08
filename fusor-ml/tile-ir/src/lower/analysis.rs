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
            TileStmt::StoreSwiGlu(store) => {
                Self::mark_tile_expr_live(ir, &store.gate, live);
                Self::mark_tile_expr_live(ir, &store.up, live);
            }
            TileStmt::StoreVec4(store) => Self::mark_tile_expr_live(ir, &store.value, live),
            TileStmt::StoreLinear(store) => Self::mark_tile_expr_live(ir, &store.value, live),
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
        ir.body().ops().iter().any(|op| {
            let Op::TileProgram(op) = op;
            op.body.iter().any(Self::tile_stmt_uses_subgroup_reduce)
        }) || ir
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
            TileStmt::Store(_)
            | TileStmt::StoreSwiGlu(_)
            | TileStmt::StoreVec4(_)
            | TileStmt::StoreLinear(_)
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

    fn tile_stmt_uses_subgroup_reduce(stmt: &TileStmt) -> bool {
        match stmt {
            TileStmt::Store(store) => Self::tile_expr_uses_subgroup_reduce(&store.value),
            TileStmt::StoreSwiGlu(store) => {
                Self::tile_expr_uses_subgroup_reduce(&store.gate)
                    || Self::tile_expr_uses_subgroup_reduce(&store.up)
            }
            TileStmt::StoreVec4(store) => Self::tile_expr_uses_subgroup_reduce(&store.value),
            TileStmt::StoreLinear(store) => Self::tile_expr_uses_subgroup_reduce(&store.value),
            TileStmt::StoreLocal { value, .. }
            | TileStmt::Emit { value }
            | TileStmt::StoreWorkgroup { value, .. } => Self::tile_expr_uses_subgroup_reduce(value),
            TileStmt::WhileTrue { body, .. } => {
                body.iter().any(Self::tile_stmt_uses_subgroup_reduce)
            }
            TileStmt::If {
                condition,
                accept,
                reject,
            } => {
                Self::tile_expr_uses_subgroup_reduce(condition)
                    || accept
                        .iter()
                        .chain(reject.iter())
                        .any(Self::tile_stmt_uses_subgroup_reduce)
            }
            TileStmt::Loop { body } => body.iter().any(Self::tile_stmt_uses_subgroup_reduce),
            TileStmt::ZeroCoopAcc { .. }
            | TileStmt::CopyToWorkgroupTile { .. }
            | TileStmt::CopyQuantToWorkgroupTile { .. }
            | TileStmt::Barrier
            | TileStmt::LoadCoop { .. }
            | TileStmt::Mma { .. }
            | TileStmt::StoreCoopAcc { .. }
            | TileStmt::Break
            | TileStmt::Return => false,
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
            TileStmt::StoreSwiGlu(store) => {
                Self::tile_index_expr_uses_kind(&store.row, kind)
                    || Self::tile_index_expr_uses_kind(&store.col, kind)
                    || Self::tile_mask_expr_uses_kind(&store.mask, kind)
                    || Self::tile_expr_uses_index_kind(&store.gate, kind)
                    || Self::tile_expr_uses_index_kind(&store.up, kind)
            }
            TileStmt::StoreVec4(store) => {
                Self::tile_index_expr_uses_kind(&store.index, kind)
                    || Self::tile_mask_expr_uses_kind(&store.mask, kind)
                    || Self::tile_expr_uses_index_kind(&store.value, kind)
            }
            TileStmt::StoreLinear(store) => {
                Self::tile_index_expr_uses_kind(&store.index, kind)
                    || Self::tile_mask_expr_uses_kind(&store.mask, kind)
                    || Self::tile_expr_uses_index_kind(&store.value, kind)
            }
            TileStmt::StoreLocal { value, .. } => Self::tile_expr_uses_index_kind(value, kind),
            TileStmt::Emit { value } => Self::tile_expr_uses_index_kind(value, kind),
            TileStmt::StoreWorkgroup { index, value, .. } => {
                Self::tile_index_expr_uses_kind(index, kind)
                    || Self::tile_expr_uses_index_kind(value, kind)
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
            TileStmt::If {
                condition,
                accept,
                reject,
            } => {
                Self::tile_expr_uses_index_kind(condition, kind)
                    || accept
                        .iter()
                        .chain(reject.iter())
                        .any(|stmt| Self::tile_stmt_uses_index_kind(stmt, kind))
            }
            TileStmt::Loop { body } => body
                .iter()
                .any(|stmt| Self::tile_stmt_uses_index_kind(stmt, kind)),
            TileStmt::ZeroCoopAcc { .. }
            | TileStmt::Barrier
            | TileStmt::LoadCoop { .. }
            | TileStmt::Mma { .. }
            | TileStmt::Break
            | TileStmt::Return => false,
        }
    }

    fn tile_expr_uses_index_kind(expr: &TileExpr, kind: SubgroupIndexKind) -> bool {
        match expr {
            TileExpr::Load(load) => {
                Self::tile_index_expr_uses_kind(&load.row, kind)
                    || Self::tile_index_expr_uses_kind(&load.col, kind)
                    || Self::tile_mask_expr_uses_kind(&load.mask, kind)
            }
            TileExpr::LoadLinear(load) => {
                Self::tile_index_expr_uses_kind(&load.index, kind)
                    || Self::tile_mask_expr_uses_kind(&load.mask, kind)
            }
            TileExpr::LoadVec4(load) => {
                Self::tile_index_expr_uses_kind(&load.index, kind)
                    || Self::tile_mask_expr_uses_kind(&load.mask, kind)
            }
            TileExpr::LoadWorkgroup { index, .. } => Self::tile_index_expr_uses_kind(index, kind),
            TileExpr::LoadLocal(_) => false,
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
            TileExpr::Vec4Dot { left, right } => {
                Self::tile_expr_uses_index_kind(left, kind)
                    || Self::tile_expr_uses_index_kind(right, kind)
            }
            TileExpr::Vec4Splat { value } => Self::tile_expr_uses_index_kind(value, kind),
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
            TileExpr::QuantizedQ8ActivationDot {
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
            TileExpr::QuantizedQ4KF32Dot {
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
            TileExpr::QuantizedQ4KGgmlDot {
                a_low,
                a_high,
                sums,
                block,
                iq,
                ir,
                col,
                mask,
                ..
            } => {
                a_low
                    .iter()
                    .chain(a_high.iter())
                    .chain(sums.iter())
                    .any(|expr| Self::tile_expr_uses_index_kind(expr, kind))
                    || Self::tile_index_expr_uses_kind(block, kind)
                    || Self::tile_index_expr_uses_kind(iq, kind)
                    || Self::tile_index_expr_uses_kind(ir, kind)
                    || Self::tile_index_expr_uses_kind(col, kind)
                    || Self::tile_mask_expr_uses_kind(mask, kind)
            }
            TileExpr::QuantizedQ6KGgmlDot {
                a,
                block,
                ip,
                il,
                col,
                mask,
                ..
            } => {
                a.iter()
                    .any(|expr| Self::tile_expr_uses_index_kind(expr, kind))
                    || Self::tile_index_expr_uses_kind(block, kind)
                    || Self::tile_index_expr_uses_kind(ip, kind)
                    || Self::tile_index_expr_uses_kind(il, kind)
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
            | TileExpr::Bitcast { value, .. }
            | TileExpr::LoopFold { value, .. }
            | TileExpr::GroupReduce { value, .. }
            | TileExpr::SubgroupReduce { value, .. } => {
                Self::tile_expr_uses_index_kind(value, kind)
            }
            TileExpr::Binary { left, right, .. } | TileExpr::Compare { left, right, .. } => {
                Self::tile_expr_uses_index_kind(left, kind)
                    || Self::tile_expr_uses_index_kind(right, kind)
            }
            TileExpr::Sum { values } => values
                .iter()
                .any(|expr| Self::tile_expr_uses_index_kind(expr, kind)),
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
            | TileExpr::LoadLinear(_)
            | TileExpr::LoadVec4(_)
            | TileExpr::LoadLocal(_)
            | TileExpr::QuantizedLoad(_)
            | TileExpr::QuantizedBlockLane { .. }
            | TileExpr::Full(_)
            | TileExpr::Literal(_)
            | TileExpr::Index(_) => false,
            TileExpr::LoadWorkgroup { .. } => false,
            TileExpr::Dot4 { a, b } => a
                .iter()
                .chain(b.iter())
                .any(|expr| Self::tile_expr_uses_subgroup_reduce(expr)),
            TileExpr::Vec4Dot { left, right } => {
                Self::tile_expr_uses_subgroup_reduce(left)
                    || Self::tile_expr_uses_subgroup_reduce(right)
            }
            TileExpr::Vec4Splat { value } => Self::tile_expr_uses_subgroup_reduce(value),
            TileExpr::QuantizedQ8_0Dot8 { a, .. } => a
                .iter()
                .any(|expr| Self::tile_expr_uses_subgroup_reduce(expr)),
            TileExpr::QuantizedQ8ActivationDot { a, .. } => a
                .iter()
                .any(|expr| Self::tile_expr_uses_subgroup_reduce(expr)),
            TileExpr::QuantizedQ4KF32Dot { a, .. } => a
                .iter()
                .any(|expr| Self::tile_expr_uses_subgroup_reduce(expr)),
            TileExpr::QuantizedQ4KGgmlDot {
                a_low,
                a_high,
                sums,
                ..
            } => a_low
                .iter()
                .chain(a_high.iter())
                .chain(sums.iter())
                .any(|expr| Self::tile_expr_uses_subgroup_reduce(expr)),
            TileExpr::QuantizedQ6KGgmlDot { a, .. } => a
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
            | TileExpr::Bitcast { value, .. }
            | TileExpr::Cast { value, .. }
            | TileExpr::LoopFold { value, .. }
            | TileExpr::GroupReduce { value, .. } => Self::tile_expr_uses_subgroup_reduce(value),
            TileExpr::Binary { left, right, .. } | TileExpr::Compare { left, right, .. } => {
                Self::tile_expr_uses_subgroup_reduce(left)
                    || Self::tile_expr_uses_subgroup_reduce(right)
            }
            TileExpr::Sum { values } => values
                .iter()
                .any(|expr| Self::tile_expr_uses_subgroup_reduce(expr)),
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
            | TileExpr::LoadLinear(_)
            | TileExpr::LoadVec4(_)
            | TileExpr::LoadLocal(_)
            | TileExpr::QuantizedLoad(_)
            | TileExpr::Full(_)
            | TileExpr::Literal(_)
            | TileExpr::Index(_) => {}
            TileExpr::LoadWorkgroup { src, .. } => Self::mark_tile_live(ir, *src, live),
            TileExpr::Scalar(expr) => Self::mark_tile_scalar_expr_live(ir, expr, live),
            TileExpr::Unary { value, .. }
            | TileExpr::Bitcast { value, .. }
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
            TileExpr::Vec4Dot { left, right } => {
                Self::mark_tile_expr_live(ir, left, live);
                Self::mark_tile_expr_live(ir, right, live);
            }
            TileExpr::Vec4Splat { value } => Self::mark_tile_expr_live(ir, value, live),
            TileExpr::QuantizedQ8_0Dot8 { a, .. } => {
                for expr in a {
                    Self::mark_tile_expr_live(ir, expr, live);
                }
            }
            TileExpr::QuantizedQ8ActivationDot { a, .. } => {
                for expr in a {
                    Self::mark_tile_expr_live(ir, expr, live);
                }
            }
            TileExpr::QuantizedQ4KF32Dot { a, .. } => {
                for expr in a {
                    Self::mark_tile_expr_live(ir, expr, live);
                }
            }
            TileExpr::QuantizedQ4KGgmlDot {
                a_low,
                a_high,
                sums,
                ..
            } => {
                for expr in a_low.iter().chain(a_high.iter()).chain(sums.iter()) {
                    Self::mark_tile_expr_live(ir, expr, live);
                }
            }
            TileExpr::QuantizedQ6KGgmlDot { a, .. } => {
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
            TileExpr::Sum { values } => {
                for expr in values {
                    Self::mark_tile_expr_live(ir, expr, live);
                }
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
