use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn matrix_shape(layout: &Layout) -> Result<[u32; 2], LowerError> {
        if layout.shape().rank() != 2 {
            return Err(LowerError::UnsupportedOperation(
                "expected rank-2 matrix layout",
            ));
        }
        Ok([
            layout.shape().dims()[0].get(),
            layout.shape().dims()[1].get(),
        ])
    }

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
            if let Some(TileProgramAccelerator::QMatmul(qmatmul)) = &op.accelerator {
                Self::mark_tile_live(ir, qmatmul.a_tile, &mut live);
                Self::mark_tile_live(ir, qmatmul.b_tile, &mut live);
            }
        }
        live
    }

    pub(super) fn uses_tile_qgemv(ir: &KernelIr) -> bool {
        ir.body().ops().iter().any(|op| {
            let Op::TileProgram(op) = op;
            matches!(&op.accelerator, Some(TileProgramAccelerator::QGemv(_)))
        })
    }

    pub(super) fn max_tile_program_coop_subgroups(ir: &KernelIr) -> u32 {
        ir.body()
            .ops()
            .iter()
            .filter_map(|op| match op {
                Op::TileProgram(op) => match &op.accelerator {
                    Some(TileProgramAccelerator::QMatmul(qmatmul)) => {
                        Self::coop8_subgroups_for_tile_shape(qmatmul.tile_m, qmatmul.tile_n)
                    }
                    Some(TileProgramAccelerator::QGemv(_)) => None,
                    None => None,
                },
            })
            .max()
            .unwrap_or(0)
    }

    pub(super) fn coop8_subgroups_for_tile_shape(m: u32, n: u32) -> Option<u32> {
        if m == 0
            || n == 0
            || !m.is_multiple_of(COOP_MATRIX_DIM)
            || !n.is_multiple_of(COOP_MATRIX_DIM)
        {
            return None;
        }
        let tile_rows = m / COOP_MATRIX_DIM;
        let tile_cols = n / COOP_MATRIX_DIM;
        if tile_rows * tile_cols <= 16 {
            return Some(1);
        }
        if m == 32 && n.is_multiple_of(32) {
            let subgroups = n / 32;
            if (2..=8).contains(&subgroups) {
                return Some(subgroups);
            }
        }
        if n >= m && m <= 64 && n.is_multiple_of(16) {
            let subgroups = n / 16;
            if (2..=8).contains(&subgroups) {
                return Some(subgroups);
            }
        }
        if n <= 64 && m.is_multiple_of(16) {
            let subgroups = m / 16;
            if (2..=8).contains(&subgroups) {
                return Some(subgroups);
            }
        }
        if m == 128 && n == 128 {
            return Some(16);
        }
        if m.is_multiple_of(32) && n.is_multiple_of(32) {
            let subgroups = (m / 32).checked_mul(n / 32)?;
            if (2..=16).contains(&subgroups) {
                return Some(subgroups);
            }
        }
        None
    }

    pub(super) fn is_row_major_storage_matrix(layout: &Layout) -> bool {
        layout.shape().rank() == 2
            && layout.strides().rank() == 2
            && layout.strides().values()[1] == 1
    }

    pub(super) fn row_major_matrix_leading_stride(layout: &Layout) -> Result<u32, LowerError> {
        if !Self::is_row_major_storage_matrix(layout) {
            return Err(LowerError::UnsupportedOperation(
                "cooperative matrix lowering requires row-major matrix tiles",
            ));
        }
        Ok(layout.strides().values()[0])
    }

    pub(super) fn cooperative_matrix_store_layout(
        layout: &Layout,
    ) -> Result<(u32, bool), LowerError> {
        if layout.shape().rank() != 2 || layout.strides().rank() != 2 {
            return Err(LowerError::UnsupportedOperation(
                "cooperative store requires a rank-2 output view",
            ));
        }

        let strides = layout.strides().values();
        if strides[1] == 1 {
            Ok((strides[0], false))
        } else if strides[0] == 1 {
            Ok((strides[1], true))
        } else {
            Err(LowerError::UnsupportedOperation(
                "cooperative store requires row-major or column-major output strides",
            ))
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
