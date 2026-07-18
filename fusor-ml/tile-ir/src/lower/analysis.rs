use super::*;

/// Everything one fused analysis walk over the IR tree discovers: the
/// [`Capabilities`] and the deduplicated, first-use-ordered declaration lists
/// the lowerer emits as the global/local arenas. Filled by [`Analysis::run`].
#[derive(Default)]
pub(super) struct Analysis {
    pub caps: Capabilities,
    pub buffers: Vec<Buffer>,
    pub tiles: Vec<Tile>,
    pub locals: Vec<Local>,
    buffer_seen: FxHashMap<*const (), ()>,
    tile_seen: FxHashMap<*const (), ()>,
    local_seen: FxHashMap<*const (), ()>,
}

/// Capability flags aggregated up front (not lazy first-use). `uses_f16` is
/// decided here so an f16 handle on a non-f16 adapter still yields
/// `UnsupportedOperation`.
#[derive(Copy, Clone, Default)]
pub(super) struct Capabilities {
    pub uses_f16: bool,
    pub native_f16_scales: bool,
    /// `unpack2x16float` reads two f16 lanes out of a `u32` into `vec2<f32>`,
    /// which naga gates behind `SHADER_FLOAT16_IN_FLOAT32` just like native
    /// f16 scales. A kernel can reach it without any quantized source (e.g. the
    /// Q4K paired ggml path decodes the f16 `d`/`dmin` header from raw words),
    /// so the flag is raised from the op itself rather than the matrix.
    pub unpacks_f16: bool,
    pub uses_subgroup_reduce: bool,
    pub uses_coop: bool,
    pub subgroup_id: bool,
    pub subgroup_lane: bool,
    pub subgroup_size: bool,
    pub num_subgroups: bool,
}

impl Capabilities {
    pub(super) fn uses_subgroups(self) -> bool {
        self.uses_subgroup_reduce
            || self.subgroup_id
            || self.subgroup_lane
            || self.subgroup_size
            || self.num_subgroups
    }
}

impl Analysis {
    pub(super) fn run(ir: &KernelIr) -> Self {
        let mut analysis = Analysis::default();
        for buffer in &ir.buffers {
            analysis.note_buffer(buffer);
        }
        for stmt in &ir.body {
            analysis.visit_stmt(stmt);
        }
        analysis
    }

    fn note_element(&mut self, element: ElementType) {
        if element.uses_f16() {
            self.caps.uses_f16 = true;
        }
    }

    fn note_buffer(&mut self, buffer: &Buffer) {
        self.note_element(buffer.element);
        if self.buffer_seen.insert(buffer_key(buffer), ()).is_none() {
            self.buffers.push(buffer.clone());
        }
    }

    fn note_view(&mut self, view: &StorageView) {
        self.note_buffer(&view.buffer);
    }

    fn note_quant(&mut self, matrix: &QuantizedMatrix) {
        self.note_view(&matrix.data);
        if matrix.format.has_native_f16_scales() {
            self.caps.native_f16_scales = true;
        }
    }

    fn note_tile(&mut self, tile: &Tile) {
        self.note_element(tile.element);
        if self.tile_seen.insert(tile_key(tile), ()).is_none() {
            self.tiles.push(tile.clone());
        }
    }

    fn note_local(&mut self, local: &Local) {
        self.note_element(local.element);
        if self.local_seen.insert(local_key(local), ()).is_none() {
            self.locals.push(local.clone());
        }
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Store {
                dst,
                addr,
                value,
                mask,
            } => {
                self.note_view(dst);
                self.visit_addr(addr);
                self.visit_expr(value);
                self.visit_expr(mask);
            }
            Stmt::StoreLocal { dst, value } => {
                self.note_local(dst);
                self.visit_expr(value);
            }
            Stmt::StoreTile { dst, index, value } => {
                self.note_tile(dst);
                self.visit_expr(index);
                self.visit_expr(value);
            }
            Stmt::FillTile { dst, value, bounds } => {
                self.note_tile(dst);
                self.visit_expr(value);
                for bound in bounds.iter().flatten() {
                    self.visit_expr(bound);
                }
            }
            Stmt::CoopStore { acc, dst, addr } => {
                self.caps.uses_coop = true;
                self.note_local(acc);
                self.note_view(dst);
                self.visit_addr(addr);
            }
            Stmt::CoopStoreTile {
                acc,
                tile,
                row,
                col,
            } => {
                self.caps.uses_coop = true;
                self.note_local(acc);
                self.note_tile(tile);
                self.visit_expr(row);
                self.visit_expr(col);
            }
            Stmt::If {
                condition,
                accept,
                reject,
            } => {
                self.visit_expr(condition);
                for s in accept.iter().chain(reject.iter()) {
                    self.visit_stmt(s);
                }
            }
            Stmt::Loop {
                count,
                index,
                accumulators,
                body,
            } => {
                if let Some(count) = count {
                    self.visit_expr(count);
                }
                if let Some(index) = index {
                    self.note_local(index);
                }
                for acc in accumulators {
                    self.note_local(&acc.local);
                    self.visit_expr(&acc.init);
                    self.visit_expr(&acc.update);
                }
                for s in body {
                    self.visit_stmt(s);
                }
            }
            Stmt::Break | Stmt::Return | Stmt::Barrier | Stmt::StorageBarrier => {}
        }
    }

    fn visit_addr(&mut self, addr: &Addr) {
        match addr {
            Addr::Linear(index) => self.visit_expr(index),
            Addr::Rc2 { row, col } => {
                self.visit_expr(row);
                self.visit_expr(col);
            }
        }
    }

    fn visit_source(&mut self, src: &Source) {
        match src {
            Source::Storage(view) => self.note_view(view),
            Source::Quantized(matrix) => self.note_quant(matrix),
        }
    }

    fn visit_coop_src(&mut self, src: &CoopSrc) {
        match src {
            CoopSrc::TileRegion { tile, row, col } => {
                self.note_tile(tile);
                self.visit_expr(row);
                self.visit_expr(col);
            }
            CoopSrc::BroadcastCol { src, col } => {
                self.note_view(src);
                self.visit_expr(col);
            }
        }
    }

    fn visit_reduce_kind(&mut self, kind: &ReduceKind) {
        match kind {
            ReduceKind::Subgroup => self.caps.uses_subgroup_reduce = true,
            ReduceKind::Workgroup { scratch, .. } => self.note_tile(scratch),
            ReduceKind::Loop { index, scratch, .. } => {
                self.note_local(index);
                self.note_tile(scratch);
            }
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        self.note_element(expr.element());
        match expr.kind() {
            ExprKind::Literal(_) => {}
            ExprKind::VecComponent { vector, .. } => self.visit_expr(vector),
            ExprKind::Builtin(builtin) => match builtin {
                Builtin::SubgroupId => self.caps.subgroup_id = true,
                Builtin::SubgroupLane => self.caps.subgroup_lane = true,
                Builtin::SubgroupSize => self.caps.subgroup_size = true,
                Builtin::NumSubgroups => self.caps.num_subgroups = true,
                Builtin::Lane | Builtin::ProgramId(_) => {}
            },
            ExprKind::LoadLocal(local) => self.note_local(local),
            ExprKind::Load {
                src,
                addr,
                mask,
                fill,
            } => {
                self.visit_source(src);
                self.visit_addr(addr);
                self.visit_expr(mask);
                self.visit_expr(fill);
            }
            ExprKind::LoadTile { tile, index } => {
                self.note_tile(tile);
                self.visit_expr(index);
            }
            ExprKind::Unary { op, value } => {
                if *op == TileUnaryOp::Unpack2x16Float {
                    self.caps.unpacks_f16 = true;
                }
                self.visit_expr(value);
            }
            ExprKind::Binary { left, right, .. } | ExprKind::Compare { left, right, .. } => {
                self.visit_expr(left);
                self.visit_expr(right);
            }
            ExprKind::Cast { value, to } | ExprKind::Bitcast { value, to } => {
                self.note_element(*to);
                self.visit_expr(value);
            }
            ExprKind::Select {
                condition,
                accept,
                reject,
            } => {
                self.visit_expr(condition);
                self.visit_expr(accept);
                self.visit_expr(reject);
            }
            ExprKind::Vec { parts, .. } => {
                for part in parts {
                    self.visit_expr(part);
                }
            }
            ExprKind::Dot { left, right } => {
                self.visit_expr(left);
                self.visit_expr(right);
            }
            ExprKind::Reduce { kind, value, .. } => {
                self.visit_reduce_kind(kind);
                self.visit_expr(value);
            }
            ExprKind::CoopLoad { src, .. } => {
                self.caps.uses_coop = true;
                self.visit_coop_src(src);
            }
            ExprKind::CoopMma { a, b, c } => {
                self.caps.uses_coop = true;
                self.visit_expr(a);
                self.visit_expr(b);
                self.visit_expr(c);
            }
            ExprKind::Dequantize {
                src,
                k_base,
                col,
                mask,
                fill,
                ..
            } => {
                self.note_quant(src);
                self.visit_expr(k_base);
                self.visit_expr(col);
                self.visit_expr(mask);
                self.visit_expr(fill);
            }
            ExprKind::QuantizedDot {
                src,
                activations,
                k_base,
                col,
                mask,
                fill,
                packing: _,
            } => {
                // Visit the activations (buffer `a`) before the quantized
                // weights (`src`, buffer `b`) so first-use buffer order matches
                // the dequant+dot path and the builder's creation order.
                for activation in activations {
                    self.visit_expr(activation);
                }
                self.note_quant(src);
                self.visit_expr(k_base);
                self.visit_expr(col);
                self.visit_expr(mask);
                self.visit_expr(fill);
            }
            ExprKind::LaneOf { block, .. } => self.visit_expr(block),
            ExprKind::Shared(inner) => self.visit_expr(inner),
        }
    }
}

impl<'a> Lowerer<'a> {
    pub(super) fn collect_buffers(&self) -> Vec<Buffer> {
        self.buffer_decls.clone()
    }

    pub(super) fn collect_tiles(&self) -> Vec<Tile> {
        self.tile_decls.clone()
    }

    pub(super) fn collect_locals(&self) -> Vec<Local> {
        self.local_decls.clone()
    }
}
