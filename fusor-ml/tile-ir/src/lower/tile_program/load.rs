use super::*;
use crate::ir::{Addr, Source};

impl<'a> Lowerer<'a> {
    /// Lower an `ExprKind::Load`. Dispatches on the source (dense storage vs
    /// quantized) and the address (linear vs rank-2).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::lower) fn lower_load_expr(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        src: &Source,
        addr: &Addr,
        mask: &Expr,
        fill: &Expr,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        match src {
            Source::Storage(view) => self.lower_storage_load_with(
                expressions,
                body,
                StorageLoadLowering {
                    src: view,
                    mask,
                    fill,
                    spill_depth,
                },
                |lowerer, expressions, accept| {
                    lowerer.lower_addr_index(expressions, accept, view, addr, spill_depth)
                },
            ),
            Source::Quantized(matrix) => self.lower_masked_f32_value(
                expressions,
                body,
                MaskedF32Value {
                    mask,
                    fill,
                    spill_depth,
                },
                |expressions, block| {
                    let (row, col) = self.lower_addr_rc2(expressions, block, addr, spill_depth)?;
                    self.dequantize_qvalue(expressions, matrix, row, col, block)
                },
            ),
        }
    }

    /// Resolve an `Addr` to the flat storage index for `view`.
    fn lower_addr_index(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        view: &StorageView,
        addr: &Addr,
        spill_depth: usize,
    ) -> Result<Handle<Expression>, LowerError> {
        match addr {
            Addr::Linear(index) => self.lower_expr_lane(expressions, body, index, spill_depth),
            Addr::Rc2 { row, col } => {
                let row = self.lower_expr_lane(expressions, body, row, spill_depth)?;
                let col = self.lower_expr_lane(expressions, body, col, spill_depth)?;
                self.storage_index_from_coords(expressions, view, &[row, col], body)
            }
        }
    }

    /// Resolve an `Addr` to a `(row, col)` pair (used by quantized loads, which
    /// dequantize against raw coordinates rather than a flat index).
    fn lower_addr_rc2(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        addr: &Addr,
        spill_depth: usize,
    ) -> Result<(Handle<Expression>, Handle<Expression>), LowerError> {
        match addr {
            Addr::Rc2 { row, col } => {
                let row = self.lower_expr_lane(expressions, body, row, spill_depth)?;
                let col = self.lower_expr_lane(expressions, body, col, spill_depth)?;
                Ok((row, col))
            }
            Addr::Linear(_) => Err(LowerError::UnsupportedOperation(
                "quantized load requires a rank-2 address",
            )),
        }
    }

    /// Shared masked-load skeleton. The `index` callback resolves the storage
    /// index expression each time it's called: once when the mask is constant
    /// true (directly into `body`) and once inside the masked-load accept block
    /// when not. `fill` is the masked-out value, lowered eagerly only when the
    /// mask is not constant true.
    fn lower_storage_load_with(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        request: StorageLoadLowering<'_>,
        index: impl Fn(
            &Self,
            &mut Arena<Expression>,
            &mut Block,
        ) -> Result<Handle<Expression>, LowerError>,
    ) -> Result<Handle<Expression>, LowerError> {
        if request.mask.is_constant_true() {
            let src_index = index(self, expressions, body)?;
            let src_ptr =
                self.storage_dynamic_pointer(expressions, request.src, src_index, body)?;
            return Ok(Self::emit_load(expressions, body, src_ptr));
        }

        let element = request.src.buffer.element;
        let fill_source = request.fill.element();
        let fill = self.lower_expr_lane(expressions, body, request.fill, request.spill_depth)?;
        let fill = self.cast_tile_value(expressions, body, fill, fill_source, element);
        self.lower_masked_value_to_local(
            expressions,
            body,
            MaskedLocalValue {
                mask: request.mask,
                element,
                fill,
                spill_depth: request.spill_depth,
            },
            |expressions, accept| {
                let src_index = index(self, expressions, accept)?;
                let src_ptr =
                    self.storage_dynamic_pointer(expressions, request.src, src_index, accept)?;
                Ok(Self::emit_load(expressions, accept, src_ptr))
            },
        )
    }
}
