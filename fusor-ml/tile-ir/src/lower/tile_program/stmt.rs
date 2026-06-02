use super::*;
use crate::ir::Addr;

impl<'a> Lowerer<'a> {
    /// Lower a per-lane `Stmt::Store`. `Addr::Rc2` is a dense rank-2 store
    /// (coords flattened through the view layout); `Addr::Linear` is an indexed
    /// rank-1 store.
    pub(in crate::lower) fn lower_store_stmt(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        dst: &StorageView,
        addr: &Addr,
        value: &Expr,
        mask: &Expr,
    ) -> Result<(), LowerError> {
        self.clear_store_caches();
        let value = self.lower_expr(expressions, body, value)?;
        let mut accept = Block::new();
        let dst_index = match addr {
            Addr::Rc2 { row, col } => {
                let row = self.lower_expr(expressions, &mut accept, row)?;
                let col = self.lower_expr(expressions, &mut accept, col)?;
                self.storage_index_from_coords(expressions, dst, &[row, col], &mut accept)?
            }
            Addr::Linear(index) => self.lower_expr(expressions, &mut accept, index)?,
        };
        let dst_ptr = self.storage_dynamic_pointer(expressions, dst, dst_index, &mut accept)?;
        if mask.is_constant_true() {
            accept.push(
                Statement::Store {
                    pointer: dst_ptr,
                    value,
                },
                Span::default(),
            );
            body.extend_block(accept);
        } else {
            let mask = self.lower_expr(expressions, body, mask)?;
            Self::push_masked_store(body, mask, accept, dst_ptr, value);
        }
        Ok(())
    }

    pub(in crate::lower) fn clear_store_caches(&self) {
        self.dequant_memo.borrow_mut().clear();
        self.expr_memo.borrow_mut().clear();
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
