use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn lower_body(
        &self,
        ir_body: &TileProgramOp,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
    ) -> Result<Block, LowerError> {
        let mut body = Block::new();
        body.push(
            self.lower_tile_program(expressions, scratch, ir_body)?,
            Span::default(),
        );
        Ok(body)
    }
}
