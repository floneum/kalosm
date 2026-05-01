use super::*;

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
            for store in &op.stores {
                Self::mark_tile_expr_live(ir, &store.value, &mut live);
            }
        }
        live
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
