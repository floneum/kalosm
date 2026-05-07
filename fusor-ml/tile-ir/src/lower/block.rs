use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn lower_block(
        &self,
        ir_block: &crate::Block,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
    ) -> Result<Block, LowerError> {
        let mut body = Block::new();
        for op in ir_block.ops() {
            let Op::TileProgram(op) = op;
            body.push(
                self.lower_tile_program(expressions, scratch, op)?,
                Span::default(),
            );
        }
        Ok(body)
    }
}
