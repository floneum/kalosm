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

    pub(super) fn uses_qgemv(ir: &KernelIr) -> bool {
        Self::block_uses_qgemv(ir.body())
    }

    pub(super) fn qgemv_workgroup_invocations(ir: &KernelIr) -> Option<u32> {
        Self::block_qgemv_workgroup_invocations(ir.body())
    }

    fn block_qgemv_workgroup_invocations(block: &crate::Block) -> Option<u32> {
        block
            .ops()
            .iter()
            .filter_map(|op| match op {
                Op::QMatMul(op) => {
                    if op.use_qgemv
                        && Self::matrix_shape(&op.a.layout)
                            .map(|[m, _]| m == 1)
                            .unwrap_or(false)
                    {
                        Some(op.b.format.qgemv_subgroups_per_workgroup() * 32)
                    } else {
                        None
                    }
                }
                Op::Block(op) => Self::block_qgemv_workgroup_invocations(&op.body),
                Op::Loop(op) => Self::block_qgemv_workgroup_invocations(&op.body),
                Op::Partition(op) => Self::block_qgemv_workgroup_invocations(&op.body),
                _ => None,
            })
            .max()
    }

    fn block_uses_qgemv(block: &crate::Block) -> bool {
        block.ops().iter().any(|op| match op {
            Op::QMatMul(op) => {
                op.use_qgemv
                    && Self::matrix_shape(&op.a.layout)
                        .map(|[m, _]| m == 1)
                        .unwrap_or(false)
            }
            Op::Block(op) => Self::block_uses_qgemv(&op.body),
            Op::Loop(op) => Self::block_uses_qgemv(&op.body),
            Op::Partition(op) => Self::block_uses_qgemv(&op.body),
            _ => false,
        })
    }

    pub(super) fn is_single_direct_coop_gemm(ir: &KernelIr) -> bool {
        let ops = ir.body().ops();
        if ops.len() != 3 {
            return false;
        }
        let Some(parts) = Self::fused_gemm_parts(ir, ops, 0) else {
            return false;
        };
        let crate::LoopKind::RangeStep { iterations, .. } = parts.loop_op.kind;
        let Some(acc_layout) = Self::tile_layout_in_ir(ir, parts.gemm.acc) else {
            return false;
        };
        Self::storage_gemm_coop8_subgroups(
            &parts.a_load.src.layout,
            &parts.b_load.src.layout,
            acc_layout,
            &parts.store.dst.layout,
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
            if Self::is_direct_fused_gemm_pattern(ir, ops, index) {
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
                Op::QMatMul(op) => {
                    Self::mark_tile_live(ir, op.a_tile, live);
                    Self::mark_tile_live(ir, op.b_tile, live);
                }
                Op::QDequantize(_) => {}
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

    pub(super) fn is_direct_fused_gemm_pattern(ir: &KernelIr, ops: &[Op], index: usize) -> bool {
        Self::fused_gemm_parts(ir, ops, index).is_some()
    }

    pub(super) fn mark_shared_fused_gemm_tiles(
        ir: &KernelIr,
        ops: &[Op],
        index: usize,
        live: &mut [bool],
        workgroup_invocations: u32,
    ) -> bool {
        let Some(parts) = Self::fused_gemm_parts(ir, ops, index) else {
            return false;
        };
        let crate::LoopKind::RangeStep { iterations, .. } = parts.loop_op.kind;

        let can_lower_coop = if PREFER_COOP_MATRIX_GEMM && PREFER_SHARED_COOP_GEMM {
            match (
                Self::tile_layout_in_ir(ir, parts.gemm.a),
                Self::tile_layout_in_ir(ir, parts.gemm.b),
                Self::tile_layout_in_ir(ir, parts.gemm.acc),
            ) {
                (Some(a_layout), Some(b_layout), Some(acc_layout)) => {
                    Self::can_lower_shared_gemm_coop8(
                        a_layout,
                        b_layout,
                        acc_layout,
                        &parts.store.dst.layout,
                        iterations,
                    )
                }
                _ => false,
            }
        } else {
            false
        };

        let can_lower_scalar = PREFER_SHARED_GEMM
            && Self::can_lower_shared_gemm_4col(ir, &parts.gemm, iterations, workgroup_invocations);
        if !can_lower_coop && !can_lower_scalar {
            return false;
        }

        Self::mark_tile_live(ir, parts.a_load.dst, live);
        Self::mark_tile_live(ir, parts.b_load.dst, live);
        true
    }

    pub(super) fn fused_gemm_parts<'ops>(
        ir: &KernelIr,
        ops: &'ops [Op],
        index: usize,
    ) -> Option<FusedGemmParts<'ops>> {
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
        let mut loads = Vec::new();
        let mut legacy_gemm = None;
        let mut mmas = Vec::new();
        for op in loop_op.body.ops() {
            match op {
                Op::CooperativeLoad(op) => loads.push(op),
                Op::Barrier(_) => {}
                Op::Gemm(op) if op.acc == fill.dst && legacy_gemm.is_none() => {
                    legacy_gemm = Some(GemmDescriptor::from(op));
                }
                Op::Block(op) => {
                    if !Self::collect_mmas(&op.body, &mut mmas) {
                        return None;
                    }
                }
                Op::Partition(op) => {
                    if !Self::collect_mmas(&op.body, &mut mmas) {
                        return None;
                    }
                }
                Op::Mma(op) => mmas.push(op),
                _ => return None,
            }
        }
        let gemm = match (legacy_gemm, mmas.is_empty()) {
            (Some(gemm), true) => gemm,
            (None, false) => Self::primitive_gemm_descriptor(ir, fill.dst, &mmas)?,
            _ => return None,
        };
        let a_load = loads.iter().copied().find(|load| load.dst == gemm.a)?;
        let b_load = loads.iter().copied().find(|load| load.dst == gemm.b)?;
        Some(FusedGemmParts {
            fill,
            loop_op,
            store,
            gemm,
            a_load,
            b_load,
        })
    }

    fn collect_mmas<'ops>(block: &'ops crate::Block, mmas: &mut Vec<&'ops MmaOp>) -> bool {
        for op in block.ops() {
            match op {
                Op::Block(op) => {
                    if !Self::collect_mmas(&op.body, mmas) {
                        return false;
                    }
                }
                Op::Partition(op) => {
                    if !Self::collect_mmas(&op.body, mmas) {
                        return false;
                    }
                }
                Op::Mma(op) => mmas.push(op),
                _ => return false,
            }
        }
        true
    }

    fn primitive_gemm_descriptor(
        ir: &KernelIr,
        acc: TileRef,
        mmas: &[&MmaOp],
    ) -> Option<GemmDescriptor> {
        let first = mmas.first()?;
        let (a_root, _, _) = Self::tile_root_rect(ir, first.a)?;
        let (b_root, _, _) = Self::tile_root_rect(ir, first.b)?;
        let (acc_root, _, _) = Self::tile_root_rect(ir, first.acc)?;
        if acc_root != acc {
            return None;
        }

        let a_layout = Self::tile_layout_in_ir(ir, a_root)?;
        let b_layout = Self::tile_layout_in_ir(ir, b_root)?;
        let acc_layout = Self::tile_layout_in_ir(ir, acc_root)?;
        let [m, k_a] = Self::matrix_shape(a_layout).ok()?;
        let [k_b, n] = Self::matrix_shape(b_layout).ok()?;
        let [m_acc, n_acc] = Self::matrix_shape(acc_layout).ok()?;
        if k_a != k_b || m != m_acc || n != n_acc {
            return None;
        }

        for mma in mmas.iter().copied() {
            let (mma_a_root, a_origin, a_shape) = Self::tile_root_rect(ir, mma.a)?;
            let (mma_b_root, b_origin, b_shape) = Self::tile_root_rect(ir, mma.b)?;
            let (mma_acc_root, acc_origin, acc_shape) = Self::tile_root_rect(ir, mma.acc)?;
            if mma_a_root != a_root || mma_b_root != b_root || mma_acc_root != acc_root {
                return None;
            }
            if a_origin != [acc_origin[0], 0]
                || b_origin != [0, acc_origin[1]]
                || a_shape != [acc_shape[0], k_a]
                || b_shape != [k_a, acc_shape[1]]
            {
                return None;
            }
        }

        Self::mmas_cover_acc(ir, acc_root, mmas).then_some(GemmDescriptor {
            a: a_root,
            b: b_root,
            acc: acc_root,
        })
    }

    fn mmas_cover_acc(ir: &KernelIr, acc: TileRef, mmas: &[&MmaOp]) -> bool {
        let Some(layout) = Self::tile_layout_in_ir(ir, acc) else {
            return false;
        };
        let Ok([rows, cols]) = Self::matrix_shape(layout) else {
            return false;
        };
        let Some(total) = rows.checked_mul(cols) else {
            return false;
        };
        let mut covered = vec![false; total as usize];
        for mma in mmas.iter().copied() {
            let Some((root, origin, shape)) = Self::tile_root_rect(ir, mma.acc) else {
                return false;
            };
            if root != acc {
                return false;
            }
            let [row_origin, col_origin] = origin;
            let [tile_rows, tile_cols] = shape;
            if row_origin
                .checked_add(tile_rows)
                .is_none_or(|end| end > rows)
                || col_origin
                    .checked_add(tile_cols)
                    .is_none_or(|end| end > cols)
            {
                return false;
            }
            for row in row_origin..row_origin + tile_rows {
                for col in col_origin..col_origin + tile_cols {
                    let index = (row * cols + col) as usize;
                    if covered[index] {
                        return false;
                    }
                    covered[index] = true;
                }
            }
        }
        covered.into_iter().all(|cell| cell)
    }

    fn tile_root_rect(ir: &KernelIr, tile: TileRef) -> Option<(TileRef, [u32; 2], [u32; 2])> {
        let decl = ir.tiles().get(tile.id.index())?;
        if decl.element != tile.element {
            return None;
        }
        let shape = Self::matrix_shape(&decl.layout).ok()?;
        match decl.origin {
            TileOrigin::Allocation => Some((tile, [0, 0], shape)),
            TileOrigin::View { source, mapping } => {
                let (root, parent_origin, _) = Self::tile_root_rect(ir, source)?;
                let ViewMapping::Partition { origin, .. } = mapping;
                Some((
                    root,
                    [
                        parent_origin[0].checked_add(origin[0])?,
                        parent_origin[1].checked_add(origin[1])?,
                    ],
                    shape,
                ))
            }
        }
    }

    pub(super) fn can_lower_shared_gemm_4col(
        ir: &KernelIr,
        gemm: &GemmDescriptor,
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
            if let Some(parts) = Self::fused_gemm_parts(ir, ops, index) {
                let crate::LoopKind::RangeStep { iterations, .. } = parts.loop_op.kind;
                if PREFER_SHARED_COOP_GEMM {
                    if let (Some(a_layout), Some(b_layout), Some(acc_layout)) = (
                        Self::tile_layout_in_ir(ir, parts.gemm.a),
                        Self::tile_layout_in_ir(ir, parts.gemm.b),
                        Self::tile_layout_in_ir(ir, parts.gemm.acc),
                    ) {
                        if let Some(subgroups) = Self::shared_gemm_coop8_subgroups(
                            a_layout,
                            b_layout,
                            acc_layout,
                            &parts.store.dst.layout,
                            iterations,
                        ) {
                            max_subgroups = max_subgroups.max(subgroups);
                        }
                    }
                }
                if let Some(acc_layout) = Self::tile_layout_in_ir(ir, parts.gemm.acc) {
                    if let Some(subgroups) = Self::storage_gemm_coop8_subgroups(
                        &parts.a_load.src.layout,
                        &parts.b_load.src.layout,
                        acc_layout,
                        &parts.store.dst.layout,
                        iterations,
                    ) {
                        max_subgroups = max_subgroups.max(subgroups);
                    }
                }
                index += 3;
                continue;
            }

            let nested = match &ops[index] {
                Op::QMatMul(op) => Self::qmatmul_coop8_subgroups(ir, op).unwrap_or(0),
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

    pub(super) fn qmatmul_coop8_subgroups(ir: &KernelIr, op: &QMatMulOp) -> Option<u32> {
        let a_tile = Self::tile_layout_in_ir(ir, op.a_tile)?;
        let b_tile = Self::tile_layout_in_ir(ir, op.b_tile)?;
        let [m, k] = Self::matrix_shape(&op.a.layout).ok()?;
        let [y_m, n] = Self::matrix_shape(&op.y.layout).ok()?;
        let [a_tile_m, a_tile_k] = Self::matrix_shape(a_tile).ok()?;
        let [b_tile_k, b_tile_n] = Self::matrix_shape(b_tile).ok()?;
        if m == 1
            || m != y_m
            || k != op.b.rows
            || n != op.b.cols
            || a_tile.memory_level() != MemoryLevel::Workgroup
            || b_tile.memory_level() != MemoryLevel::Workgroup
            || a_tile_m != op.tile_m
            || a_tile_k != op.tile_k
            || b_tile_k != op.tile_k
            || b_tile_n != op.tile_n
            || !op.tile_k.is_multiple_of(COOP_MATRIX_DIM)
            || !op.tile_m.is_multiple_of(COOP_MATRIX_DIM)
            || !op.tile_n.is_multiple_of(COOP_MATRIX_DIM)
            || k % op.tile_k != 0
            || m % op.tile_m != 0
            || n % op.tile_n != 0
        {
            return None;
        }

        Self::coop8_subgroups_for_tile_shape(op.tile_m, op.tile_n)
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

    pub(super) fn max_gemm_sums(ir: &KernelIr, block: &crate::Block) -> u32 {
        let ops = block.ops();
        let mut index = 0;
        let mut max_sums = 0;
        while index < ops.len() {
            if Self::fused_gemm_parts(ir, ops, index).is_some() {
                max_sums = max_sums.max(8);
                index += 3;
                continue;
            }
            let nested = match &ops[index] {
                Op::Gemm(_) => 8,
                Op::Block(op) => Self::max_gemm_sums(ir, &op.body),
                Op::Loop(op) => Self::max_gemm_sums(ir, &op.body),
                Op::Partition(op) => Self::max_gemm_sums(ir, &op.body),
                _ => 0,
            };
            max_sums = max_sums.max(nested);
            index += 1;
        }
        max_sums
    }

    pub(super) fn max_qmatmul_sums(block: &crate::Block) -> u32 {
        block
            .ops()
            .iter()
            .map(|op| match op {
                Op::QMatMul(_) => 8,
                Op::Block(op) => Self::max_qmatmul_sums(&op.body),
                Op::Loop(op) => Self::max_qmatmul_sums(&op.body),
                Op::Partition(op) => Self::max_qmatmul_sums(&op.body),
                _ => 0,
            })
            .max()
            .unwrap_or(0)
    }
}
