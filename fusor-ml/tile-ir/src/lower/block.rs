use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn lower_body(
        &self,
        expressions: &mut Arena<Expression>,
    ) -> Result<Block, LowerError> {
        if self.ir.block == 0 || self.ir.block != self.workgroup_invocations {
            return Err(LowerError::UnsupportedOperation(
                "tile program block must match workgroup size",
            ));
        }
        let mut body = Block::new();
        let mut inner = Block::new();
        for stmt in &self.ir.body {
            self.lower_stmt(expressions, &mut inner, stmt)?;
        }
        body.push(Statement::Block(inner), Span::default());
        Ok(body)
    }
}
