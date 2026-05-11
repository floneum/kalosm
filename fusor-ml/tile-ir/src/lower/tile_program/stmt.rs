use super::*;

impl<'a> Lowerer<'a> {
    pub(in crate::lower) fn lower_tile_program(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &TileProgramOp,
    ) -> Result<Statement, LowerError> {
        if op.block == 0 || op.block != self.workgroup_invocations {
            return Err(LowerError::UnsupportedOperation(
                "tile program block must match workgroup size",
            ));
        }

        let mut body = Block::new();
        for stmt in &op.body {
            match stmt {
                TileStmt::Store(store) => {
                    self.lower_tile_store_stmt(expressions, scratch, &mut body, store)?;
                }
                TileStmt::StoreIndexed(store) => {
                    self.lower_tile_indexed_store_stmt(
                        expressions,
                        scratch,
                        &mut body,
                        &store.dst,
                        &store.index,
                        &store.value,
                        &store.mask,
                    )?;
                }
                _ => self.lower_tile_stmt(expressions, scratch, &mut body, stmt)?,
            }
        }
        Ok(Statement::Block(body))
    }

    pub(in crate::lower) fn lower_tile_store_stmt(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        store: &TileStoreStmt,
    ) -> Result<(), LowerError> {
        self.clear_store_caches(true);
        let value = self.lower_tile_expr(expressions, scratch, body, &store.value)?;
        let mask = self.lower_tile_expr(expressions, scratch, body, &store.mask)?;
        let mut accept = Block::new();
        let row = self.lower_tile_expr(expressions, scratch, &mut accept, &store.row)?;
        let col = self.lower_tile_expr(expressions, scratch, &mut accept, &store.col)?;
        let dst_index =
            self.storage_index_from_coords(expressions, &store.dst, &[row, col], &mut accept)?;
        let dst_ptr =
            self.storage_dynamic_pointer(expressions, &store.dst, dst_index, &mut accept)?;
        Self::push_masked_store(body, mask, accept, dst_ptr, value);
        Ok(())
    }

    pub(in crate::lower) fn lower_tile_indexed_store_stmt(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        dst: &StorageView,
        index: &Expr,
        value: &Expr,
        mask: &Expr,
    ) -> Result<(), LowerError> {
        self.clear_store_caches(false);
        let value = self.lower_tile_expr(expressions, scratch, body, value)?;
        let mask = self.lower_tile_expr(expressions, scratch, body, mask)?;
        let mut accept = Block::new();
        let index = self.lower_tile_expr(expressions, scratch, &mut accept, index)?;
        let dst_ptr = self.storage_dynamic_pointer(expressions, dst, index, &mut accept)?;
        Self::push_masked_store(body, mask, accept, dst_ptr, value);
        Ok(())
    }

    pub(in crate::lower) fn clear_store_caches(&self, _clear_pins: bool) {
        self.block_dequant_cache.borrow_mut().clear();
        self.q8_activation_pack_cache.borrow_mut().clear();
    }

    pub(in crate::lower) fn push_masked_store(
        body: &mut Block,
        mask: Handle<Expression>,
        mut accept: Block,
        pointer: Handle<Expression>,
        value: Handle<Expression>,
    ) {
        accept.push(Statement::Store { pointer, value }, Span::default());
        body.push(
            Statement::If {
                condition: mask,
                accept,
                reject: Block::new(),
            },
            Span::default(),
        );
    }

    pub(in crate::lower) fn emit_load(
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        pointer: Handle<Expression>,
    ) -> Handle<Expression> {
        let value = expressions.append(Expression::Load { pointer }, Span::default());
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, value)),
            Span::default(),
        );
        value
    }
}
