use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn max_gemv_rows(block: &crate::Block) -> u32 {
        block
            .ops()
            .iter()
            .map(|op| match op {
                Op::Gemv(op) => op.rows_per_workgroup,
                Op::Block(op) => Self::max_gemv_rows(&op.body),
                Op::Loop(op) => Self::max_gemv_rows(&op.body),
                Op::Partition(op) => Self::max_gemv_rows(&op.body),
                _ => 0,
            })
            .max()
            .unwrap_or(0)
    }

    pub(super) fn is_single_direct_coop_gemm(ir: &KernelIr) -> bool {
        let ops = ir.body().ops();
        if ops.len() != 3 {
            return false;
        }
        let Some((_, loop_op, store, gemm, a_load, b_load)) = Self::fused_gemm_parts(ops, 0) else {
            return false;
        };
        let crate::LoopKind::RangeStep { iterations, .. } = loop_op.kind;
        let Some(acc_layout) = Self::tile_layout_in_ir(ir, gemm.acc) else {
            return false;
        };
        Self::storage_gemm_coop8_subgroups(
            &a_load.src.layout,
            &b_load.src.layout,
            acc_layout,
            &store.dst.layout,
            iterations,
        )
        .is_some()
    }

    pub(super) fn live_tiles(ir: &KernelIr, workgroup_invocations: u32) -> Vec<bool> {
        let mut live = vec![false; ir.tiles().len()];
        Self::collect_live_tiles(ir, ir.body(), &mut live, workgroup_invocations);
        live
    }

    pub(super) fn collect_live_tiles(
        ir: &KernelIr,
        block: &crate::Block,
        live: &mut [bool],
        workgroup_invocations: u32,
    ) {
        let ops = block.ops();
        let mut index = 0;
        while index < ops.len() {
            if Self::mark_shared_fused_gemm_tiles(ir, ops, index, live, workgroup_invocations) {
                index += 3;
                continue;
            }
            if Self::is_direct_fused_gemm_pattern(ops, index) {
                index += 3;
                continue;
            }

            match &ops[index] {
                Op::Block(op) => {
                    Self::collect_live_tiles(ir, &op.body, live, workgroup_invocations)
                }
                Op::FillTile(op) => Self::mark_tile_live(ir, op.dst, live),
                Op::CooperativeLoad(op) => Self::mark_tile_live(ir, op.dst, live),
                Op::Partition(op) => {
                    for binding in &op.bindings {
                        Self::mark_tile_live(ir, binding.source, live);
                        Self::mark_tile_live(ir, binding.view, live);
                    }
                    Self::collect_live_tiles(ir, &op.body, live, workgroup_invocations);
                }
                Op::Barrier(_) => {}
                Op::Gemm(op) => {
                    Self::mark_tile_live(ir, op.a, live);
                    Self::mark_tile_live(ir, op.b, live);
                    Self::mark_tile_live(ir, op.acc, live);
                }
                Op::Gemv(op) => Self::mark_tile_live(ir, op.partials, live),
                Op::Mma(op) => {
                    Self::mark_tile_live(ir, op.a, live);
                    Self::mark_tile_live(ir, op.b, live);
                    Self::mark_tile_live(ir, op.acc, live);
                }
                Op::StoreTile(op) => Self::mark_tile_live(ir, op.src, live),
                Op::Loop(op) => Self::collect_live_tiles(ir, &op.body, live, workgroup_invocations),
            }
            index += 1;
        }
    }

    pub(super) fn is_direct_fused_gemm_pattern(ops: &[Op], index: usize) -> bool {
        Self::fused_gemm_parts(ops, index).is_some()
    }

    pub(super) fn mark_shared_fused_gemm_tiles(
        ir: &KernelIr,
        ops: &[Op],
        index: usize,
        live: &mut [bool],
        workgroup_invocations: u32,
    ) -> bool {
        let Some((_, loop_op, store, gemm, a_load, b_load)) = Self::fused_gemm_parts(ops, index)
        else {
            return false;
        };
        let crate::LoopKind::RangeStep { iterations, .. } = loop_op.kind;

        let can_lower_coop = if PREFER_COOP_MATRIX_GEMM && PREFER_SHARED_COOP_GEMM {
            match (
                Self::tile_layout_in_ir(ir, gemm.a),
                Self::tile_layout_in_ir(ir, gemm.b),
                Self::tile_layout_in_ir(ir, gemm.acc),
            ) {
                (Some(a_layout), Some(b_layout), Some(acc_layout)) => {
                    Self::can_lower_shared_gemm_coop8(
                        a_layout,
                        b_layout,
                        acc_layout,
                        &store.dst.layout,
                        iterations,
                    )
                }
                _ => false,
            }
        } else {
            false
        };

        let can_lower_scalar = PREFER_SHARED_GEMM
            && Self::can_lower_shared_gemm_4col(ir, gemm, iterations, workgroup_invocations);
        if !can_lower_coop && !can_lower_scalar {
            return false;
        }

        Self::mark_tile_live(ir, a_load.dst, live);
        Self::mark_tile_live(ir, b_load.dst, live);
        true
    }

    pub(super) fn fused_gemm_parts<'ops>(
        ops: &'ops [Op],
        index: usize,
    ) -> Option<(
        &'ops crate::FillTileOp,
        &'ops crate::LoopOp,
        &'ops crate::StoreTileOp,
        &'ops GemmOp,
        &'ops crate::CooperativeLoadOp,
        &'ops crate::CooperativeLoadOp,
    )> {
        let Some(Op::FillTile(fill)) = ops.get(index) else {
            return None;
        };
        let Some(Op::Loop(loop_op)) = ops.get(index + 1) else {
            return None;
        };
        let Some(Op::StoreTile(store)) = ops.get(index + 2) else {
            return None;
        };
        if fill.value != crate::FillValue::Zero || store.src != fill.dst {
            return None;
        }
        let mut gemm = None;
        let mut loads = Vec::new();
        for op in loop_op.body.ops() {
            match op {
                Op::CooperativeLoad(op) => loads.push(op),
                Op::Barrier(_) => {}
                Op::Gemm(op) if op.acc == fill.dst && gemm.is_none() => gemm = Some(op),
                _ => return None,
            }
        }
        let gemm = gemm?;
        let a_load = loads.iter().copied().find(|load| load.dst == gemm.a)?;
        let b_load = loads.iter().copied().find(|load| load.dst == gemm.b)?;
        Some((fill, loop_op, store, gemm, a_load, b_load))
    }

    pub(super) fn can_lower_shared_gemm_4col(
        ir: &KernelIr,
        gemm: &GemmOp,
        outer_iterations: u32,
        workgroup_invocations: u32,
    ) -> bool {
        if outer_iterations == 0 {
            return false;
        }
        let Some(a_layout) = Self::tile_layout_in_ir(ir, gemm.a) else {
            return false;
        };
        let Some(b_layout) = Self::tile_layout_in_ir(ir, gemm.b) else {
            return false;
        };
        let Some(acc_layout) = Self::tile_layout_in_ir(ir, gemm.acc) else {
            return false;
        };
        if a_layout.memory_level() != MemoryLevel::Workgroup
            || b_layout.memory_level() != MemoryLevel::Workgroup
            || acc_layout.memory_level() != MemoryLevel::Private
        {
            return false;
        }
        let Ok([m, k_a]) = Self::matrix_shape(a_layout) else {
            return false;
        };
        let Ok([k_b, n]) = Self::matrix_shape(b_layout) else {
            return false;
        };
        let Ok([m_acc, n_acc]) = Self::matrix_shape(acc_layout) else {
            return false;
        };
        if k_a != k_b || m != m_acc || n != n_acc || n % 4 != 0 || k_a % 4 != 0 {
            return false;
        }
        m.checked_mul(n / 4) == Some(workgroup_invocations)
    }

    pub(super) fn max_coop_gemm_subgroups(ir: &KernelIr) -> u32 {
        Self::block_max_coop_gemm_subgroups(ir, ir.body())
    }

    pub(super) fn block_max_coop_gemm_subgroups(ir: &KernelIr, block: &crate::Block) -> u32 {
        let ops = block.ops();
        let mut index = 0;
        let mut max_subgroups = 0;
        while index < ops.len() {
            if let Some((_, loop_op, store, gemm, a_load, b_load)) =
                Self::fused_gemm_parts(ops, index)
            {
                let crate::LoopKind::RangeStep { iterations, .. } = loop_op.kind;
                if PREFER_SHARED_COOP_GEMM {
                    if let (Some(a_layout), Some(b_layout), Some(acc_layout)) = (
                        Self::tile_layout_in_ir(ir, gemm.a),
                        Self::tile_layout_in_ir(ir, gemm.b),
                        Self::tile_layout_in_ir(ir, gemm.acc),
                    ) {
                        if let Some(subgroups) = Self::shared_gemm_coop8_subgroups(
                            a_layout,
                            b_layout,
                            acc_layout,
                            &store.dst.layout,
                            iterations,
                        ) {
                            max_subgroups = max_subgroups.max(subgroups);
                        }
                    }
                }
                if let Some(acc_layout) = Self::tile_layout_in_ir(ir, gemm.acc) {
                    if let Some(subgroups) = Self::storage_gemm_coop8_subgroups(
                        &a_load.src.layout,
                        &b_load.src.layout,
                        acc_layout,
                        &store.dst.layout,
                        iterations,
                    ) {
                        max_subgroups = max_subgroups.max(subgroups);
                    }
                }
                index += 3;
                continue;
            }

            let nested = match &ops[index] {
                Op::Block(op) => Self::block_max_coop_gemm_subgroups(ir, &op.body),
                Op::Loop(op) => Self::block_max_coop_gemm_subgroups(ir, &op.body),
                Op::Partition(op) => Self::block_max_coop_gemm_subgroups(ir, &op.body),
                _ => 0,
            };
            max_subgroups = max_subgroups.max(nested);
            index += 1;
        }
        max_subgroups
    }

    pub(super) fn can_lower_storage_gemm_coop8(
        a_layout: &Layout,
        b_layout: &Layout,
        acc_layout: &Layout,
        dst_layout: &Layout,
        outer_iterations: u32,
    ) -> bool {
        Self::storage_gemm_coop8_subgroups(
            a_layout,
            b_layout,
            acc_layout,
            dst_layout,
            outer_iterations,
        )
        .is_some()
    }

    pub(super) fn can_lower_shared_gemm_coop8(
        a_layout: &Layout,
        b_layout: &Layout,
        acc_layout: &Layout,
        dst_layout: &Layout,
        outer_iterations: u32,
    ) -> bool {
        Self::shared_gemm_coop8_subgroups(
            a_layout,
            b_layout,
            acc_layout,
            dst_layout,
            outer_iterations,
        )
        .is_some()
    }

    pub(super) fn shared_gemm_coop8_subgroups(
        a_layout: &Layout,
        b_layout: &Layout,
        acc_layout: &Layout,
        dst_layout: &Layout,
        outer_iterations: u32,
    ) -> Option<u32> {
        if outer_iterations == 0
            || a_layout.memory_level() != MemoryLevel::Workgroup
            || b_layout.memory_level() != MemoryLevel::Workgroup
            || acc_layout.memory_level() != MemoryLevel::Private
            || dst_layout.memory_level() != MemoryLevel::Storage
            || !Self::is_row_major_storage_matrix(a_layout)
            || !Self::is_row_major_storage_matrix(b_layout)
            || !Self::is_row_major_storage_matrix(dst_layout)
        {
            return None;
        }

        Self::gemm_coop8_subgroups_for_shapes(a_layout, b_layout, acc_layout, dst_layout)
    }

    pub(super) fn storage_gemm_coop8_subgroups(
        a_layout: &Layout,
        b_layout: &Layout,
        acc_layout: &Layout,
        dst_layout: &Layout,
        outer_iterations: u32,
    ) -> Option<u32> {
        if outer_iterations == 0
            || acc_layout.memory_level() != MemoryLevel::Private
            || !Self::is_row_major_storage_matrix(a_layout)
            || !Self::is_row_major_storage_matrix(b_layout)
            || !Self::is_row_major_storage_matrix(dst_layout)
        {
            return None;
        }

        Self::gemm_coop8_subgroups_for_shapes(a_layout, b_layout, acc_layout, dst_layout)
    }

    pub(super) fn gemm_coop8_subgroups_for_shapes(
        a_layout: &Layout,
        b_layout: &Layout,
        acc_layout: &Layout,
        dst_layout: &Layout,
    ) -> Option<u32> {
        let Ok([m, k_a]) = Self::matrix_shape(a_layout) else {
            return None;
        };
        let Ok([k_b, n]) = Self::matrix_shape(b_layout) else {
            return None;
        };
        let Ok([m_acc, n_acc]) = Self::matrix_shape(acc_layout) else {
            return None;
        };
        let Ok([m_dst, n_dst]) = Self::matrix_shape(dst_layout) else {
            return None;
        };
        if k_a != k_b
            || k_a % COOP_MATRIX_DIM != 0
            || m_acc != m
            || n_acc != n
            || m_dst != m
            || n_dst != n
            || m % COOP_MATRIX_DIM != 0
            || n % COOP_MATRIX_DIM != 0
        {
            return None;
        }

        let tile_rows = m / COOP_MATRIX_DIM;
        let tile_cols = n / COOP_MATRIX_DIM;
        if tile_rows == 0 || tile_cols == 0 {
            return None;
        }
        if tile_rows * tile_cols <= 16 {
            return Some(1);
        }
        if m == 32 && n % 32 == 0 {
            let subgroups = n / 32;
            if (2..=8).contains(&subgroups) {
                return Some(subgroups);
            }
        }
        if n >= m && m <= 64 && n % 16 == 0 {
            let subgroups = n / 16;
            if (2..=8).contains(&subgroups) {
                return Some(subgroups);
            }
        }
        if n <= 64 && m % 16 == 0 {
            let subgroups = m / 16;
            if (2..=8).contains(&subgroups) {
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
                "cooperative matrix lowering currently requires row-major matrix views",
            ));
        }
        Ok(layout.strides().values()[0])
    }

    pub(super) fn tile_layout_in_ir(ir: &KernelIr, tile: TileRef) -> Option<&Layout> {
        let decl = ir.tiles().get(tile.id.index())?;
        (decl.element == tile.element).then_some(&decl.layout)
    }

    pub(super) fn mark_tile_live(ir: &KernelIr, tile: TileRef, live: &mut [bool]) {
        let Some(decl) = ir.tiles().get(tile.id.index()) else {
            return;
        };
        live[tile.id.index()] = true;
        if let TileOrigin::View { source, .. } = decl.origin {
            Self::mark_tile_live(ir, source, live);
        }
    }

    pub(super) fn max_gemm_sums(block: &crate::Block) -> u32 {
        block
            .ops()
            .iter()
            .map(|op| match op {
                Op::Gemm(_) => 8,
                Op::Block(op) => Self::max_gemm_sums(&op.body),
                Op::Loop(op) => Self::max_gemm_sums(&op.body),
                Op::Partition(op) => Self::max_gemm_sums(&op.body),
                _ => 0,
            })
            .max()
            .unwrap_or(0)
    }
}
