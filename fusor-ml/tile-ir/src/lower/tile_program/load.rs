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
        let element = view.buffer.element;
        if load.mask.is_constant_true() {
            let row =
                self.lower_tile_expr_lane(expressions, scratch, body, &load.row, spill_depth)?;
            let col =
                self.lower_tile_expr_lane(expressions, scratch, body, &load.col, spill_depth)?;
            let src_index =
                self.storage_index_from_coords(expressions, view, &[row, col], body)?;
            let src_ptr = self.storage_dynamic_pointer(expressions, view, src_index, body)?;
            return Ok(Self::emit_load(expressions, body, src_ptr));
        }

        let fill_source = self.tile_expr_element(&load.fill)?;
        let fill = self.lower_tile_expr_lane(expressions, scratch, body, &load.fill, spill_depth)?;
        let fill = self.cast_tile_value(expressions, body, fill, fill_source, element);
        self.lower_masked_value_to_local(
            expressions,
            scratch,
            body,
            &load.mask,
            spill_depth,
            element,
            fill,
            |expressions, accept| {
                let row = self.lower_tile_expr_lane(
                    expressions,
                    scratch,
                    accept,
                    &load.row,
                    spill_depth,
                )?;
                let col = self.lower_tile_expr_lane(
                    expressions,
                    scratch,
                    accept,
                    &load.col,
                    spill_depth,
                )?;
                let src_index =
                    self.storage_index_from_coords(expressions, view, &[row, col], accept)?;
                let src_ptr = self.storage_dynamic_pointer(expressions, view, src_index, accept)?;
                Ok(Self::emit_load(expressions, accept, src_ptr))
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
        let element = load.src.buffer.element;
        self.lower_indexed_storage_load(
            expressions,
            scratch,
            body,
            &load.src,
            &load.index,
            &load.mask,
            spill_depth,
            element,
            |lowerer, expressions, body| {
                let fill_source = lowerer.tile_expr_element(&load.fill)?;
                let fill = lowerer.lower_tile_expr_lane(
                    expressions,
                    scratch,
                    body,
                    &load.fill,
                    spill_depth,
                )?;
                Ok(lowerer.cast_tile_value(expressions, body, fill, fill_source, element))
            },
        )
    }

    pub(in crate::lower) fn lower_indexed_storage_load(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        src: &StorageView,
        index: &Expr,
        mask: &Expr,
        spill_depth: usize,
        element: ElementType,
        fill: impl FnOnce(
            &Self,
            &mut Arena<Expression>,
            &mut Block,
        ) -> Result<Handle<Expression>, LowerError>,
    ) -> Result<Handle<Expression>, LowerError> {
        if mask.is_constant_true() {
            let index =
                self.lower_tile_expr_lane(expressions, scratch, body, index, spill_depth)?;
            let src_ptr = self.storage_dynamic_pointer(expressions, src, index, body)?;
            return Ok(Self::emit_load(expressions, body, src_ptr));
        }

        let fill = fill(self, expressions, body)?;
        self.lower_masked_value_to_local(
            expressions,
            scratch,
            body,
            mask,
            spill_depth,
            element,
            fill,
            |expressions, accept| {
                let index =
                    self.lower_tile_expr_lane(expressions, scratch, accept, index, spill_depth)?;
                let src_ptr = self.storage_dynamic_pointer(expressions, src, index, accept)?;
                Ok(Self::emit_load(expressions, accept, src_ptr))
            },
        )
    }

}
