use super::*;

impl<'a> Lowerer<'a> {
    pub(in crate::lower) fn lower_tile_load_expr(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        load: &TileLoadExpr,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        match &load.src {
            LoadSource::Storage(view) => {
                self.lower_storage_load(expressions, scratch, body, load, view, spill_depth)
            }
            LoadSource::Quantized(matrix) => self.lower_masked_f32_value(
                expressions,
                scratch,
                body,
                &load.mask,
                spill_depth,
                &load.fill,
                |expressions, block| {
                    let row = self.lower_tile_expr_lane(
                        expressions,
                        scratch,
                        block,
                        &load.row,
                        spill_depth,
                    )?;
                    let col = self.lower_tile_expr_lane(
                        expressions,
                        scratch,
                        block,
                        &load.col,
                        spill_depth,
                    )?;
                    self.dequantize_qvalue(expressions, matrix, row, col, block)
                },
            ),
        }
    }

    fn lower_storage_load(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        load: &TileLoadExpr,
        view: &StorageView,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        self.lower_storage_load_with(
            expressions,
            scratch,
            body,
            view,
            &load.mask,
            &load.fill,
            spill_depth,
            |lowerer, expressions, accept| {
                let row = lowerer.lower_tile_expr_lane(
                    expressions,
                    scratch,
                    accept,
                    &load.row,
                    spill_depth,
                )?;
                let col = lowerer.lower_tile_expr_lane(
                    expressions,
                    scratch,
                    accept,
                    &load.col,
                    spill_depth,
                )?;
                lowerer.storage_index_from_coords(expressions, view, &[row, col], accept)
            },
        )
    }

    pub(in crate::lower) fn lower_tile_linear_load_expr(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        load: &TileLinearLoadExpr,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        self.lower_storage_load_with(
            expressions,
            scratch,
            body,
            &load.src,
            &load.mask,
            &load.fill,
            spill_depth,
            |lowerer, expressions, accept| {
                lowerer.lower_tile_expr_lane(expressions, scratch, accept, &load.index, spill_depth)
            },
        )
    }

    /// Shared masked-load skeleton. The `index` callback resolves the storage
    /// index expression each time it's called: once when the mask is constant
    /// true (directly into `body`) and once inside the masked-load accept
    /// block when not. `fill_expr` is the masked-out value, lowered eagerly
    /// only when the mask is not constant true.
    #[allow(clippy::too_many_arguments)]
    fn lower_storage_load_with(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        src: &StorageView,
        mask: &Expr,
        fill_expr: &Expr,
        spill_depth: usize,
        index: impl Fn(
            &Self,
            &mut Arena<Expression>,
            &mut Block,
        ) -> Result<Handle<Expression>, LowerError>,
    ) -> Result<Handle<Expression>, LowerError> {
        if mask.is_constant_true() {
            let src_index = index(self, expressions, body)?;
            let src_ptr = self.storage_dynamic_pointer(expressions, src, src_index, body)?;
            return Ok(Self::emit_load(expressions, body, src_ptr));
        }

        let element = src.buffer.element;
        let fill_source = fill_expr.element();
        let fill = self.lower_tile_expr_lane(expressions, scratch, body, fill_expr, spill_depth)?;
        let fill = self.cast_tile_value(expressions, body, fill, fill_source, element);
        self.lower_masked_value_to_local(
            expressions,
            scratch,
            body,
            mask,
            spill_depth,
            element,
            fill,
            |expressions, accept| {
                let src_index = index(self, expressions, accept)?;
                let src_ptr = self.storage_dynamic_pointer(expressions, src, src_index, accept)?;
                Ok(Self::emit_load(expressions, accept, src_ptr))
            },
        )
    }

}
