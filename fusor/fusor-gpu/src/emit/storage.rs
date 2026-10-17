//! Typed accesses through a single physical storage binding.
//!
//! Homogeneous bindings retain their native element type. Mixed views share
//! u32 words; packed f16 stores update one half atomically so neighboring
//! lanes cannot overwrite each other's values.

use fusor_ir::ir::kernel::{
    Buffer, ElementType, ScalarElement, StorageView, TileBinaryOp, TileExpr, TileExprKind,
    TileLiteral,
};
use fusor_ir::target::EmitError;
use naga::{BinaryOperator, Block, Expression, Handle, MathFunction, ScalarKind, Statement};

use super::expr::{element_scalar, push};
use super::{Emitter, key};

/// Conservative bound for unsigned address arithmetic. Unknown expressions
/// and possible wrapping return the full range, so normalization is retained.
pub(super) fn unsigned_index_max(value: &TileExpr) -> u32 {
    if value.element() != ScalarElement::U32.element() {
        return u32::MAX;
    }
    match value.kind() {
        TileExprKind::Literal(TileLiteral::U32(n)) => *n,
        TileExprKind::Binary {
            op, left, right, ..
        } => {
            let a = unsigned_index_max(left);
            let literal = match right.kind() {
                TileExprKind::Literal(TileLiteral::U32(n)) => Some(*n),
                _ => None,
            };
            match op {
                TileBinaryOp::Add => a.saturating_add(unsigned_index_max(right)),
                TileBinaryOp::Mul => a.saturating_mul(unsigned_index_max(right)),
                TileBinaryOp::Div => literal.filter(|n| *n > 0).map_or(u32::MAX, |n| a / n),
                TileBinaryOp::Rem => literal
                    .filter(|n| *n > 0)
                    .map_or(u32::MAX, |n| a.min(n - 1)),
                TileBinaryOp::BitAnd => a.min(unsigned_index_max(right)),
                _ => u32::MAX,
            }
        }
        _ => u32::MAX,
    }
}

impl Emitter<'_> {
    pub(crate) fn buffer_element(&self, buffer: &Buffer) -> ElementType {
        self.buffer_elements[&key(buffer)]
    }

    pub(crate) fn packed_half(&self, buffer: &Buffer) -> bool {
        buffer.element == ElementType::Scalar(ScalarElement::F16)
            && self.buffer_element(buffer) == ElementType::Scalar(ScalarElement::U32)
    }

    /// Pointer at an absolute index in the view's element units.
    pub(crate) fn storage_pointer_absolute(
        &mut self,
        body: &mut Block,
        view: &StorageView,
        index: Handle<Expression>,
    ) -> Result<Handle<Expression>, EmitError> {
        let global = self.buffer_global(&view.buffer)?;
        let base = self.global_var(global);
        let index = if self.packed_half(&view.buffer) {
            self.div_literal_u32(body, index, 2)
        } else {
            index
        };
        if let Some(&width) = self.buffer_block_widths.get(&key(&view.buffer)) {
            let row = self.div_literal_u32(body, index, width);
            let col = self.mod_literal_u32(body, index, width);
            let base = self.emit_expr(body, Expression::Access { base, index: row });
            return Ok(self.emit_expr(body, Expression::Access { base, index: col }));
        }
        Ok(self.emit_expr(body, Expression::Access { base, index }))
    }

    /// Direct row/word load for an explicitly contiguous block view. Naga's
    /// ordinary runtime checks still guard both accesses.
    pub(crate) fn load_storage_block(
        &mut self,
        body: &mut Block,
        view: &StorageView,
        row: Handle<Expression>,
        col: Handle<Expression>,
        bounded_column: bool,
    ) -> Result<Handle<Expression>, EmitError> {
        // Rc2 is an affine address. Carry an oversized column into its row
        // before indexing a fixed-size inner array, including masked constant
        // addresses that Naga would otherwise reject before runtime masking.
        let width = self.buffer_block_widths[&key(&view.buffer)];
        let (row, col) = if bounded_column {
            (row, col)
        } else {
            let carry = self.div_literal_u32(body, col, width);
            let row = self.bin(body, BinaryOperator::Add, row, carry);
            let col = self.mod_literal_u32(body, col, width);
            (row, col)
        };
        let global = self.buffer_global(&view.buffer)?;
        let base = self.global_var(global);
        let base = self.emit_expr(body, Expression::Access { base, index: row });
        let pointer = self.emit_expr(body, Expression::Access { base, index: col });
        Ok(self.emit_load(body, pointer))
    }

    pub(crate) fn load_storage_absolute(
        &mut self,
        body: &mut Block,
        view: &StorageView,
        index: Handle<Expression>,
    ) -> Result<Handle<Expression>, EmitError> {
        let pointer = self.storage_pointer_absolute(body, view, index)?;
        let value = self.emit_load(body, pointer);
        let physical = self.buffer_element(&view.buffer);
        if physical == view.buffer.element {
            return Ok(value);
        }
        if self.packed_half(&view.buffer) {
            let halves = self.math(body, MathFunction::Unpack2x16float, &[value]);
            let half = self.mod_literal_u32(body, index, 2);
            let value = self.emit_expr(
                body,
                Expression::Access {
                    base: halves,
                    index: half,
                },
            );
            return Ok(self.cast_as(body, value, ScalarKind::Float, Some(2)));
        }
        let scalar = element_scalar(view.buffer.element)?;
        Ok(self.cast_as(body, value, scalar.kind, None))
    }

    pub(crate) fn load_storage_value(
        &mut self,
        body: &mut Block,
        view: &StorageView,
        index: Handle<Expression>,
    ) -> Result<Handle<Expression>, EmitError> {
        let index = self.add_literal_u32(body, index, view.offset);
        self.load_storage_absolute(body, view, index)
    }

    pub(crate) fn store_storage_value(
        &mut self,
        body: &mut Block,
        view: &StorageView,
        index: Handle<Expression>,
        value: Handle<Expression>,
    ) -> Result<(), EmitError> {
        let index = self.add_literal_u32(body, index, view.offset);
        let pointer = self.storage_pointer_absolute(body, view, index)?;
        if self.packed_half(&view.buffer) {
            return self.store_half(body, pointer, index, value);
        }
        let physical = self.buffer_element(&view.buffer);
        let value = if physical == view.buffer.element {
            value
        } else {
            self.cast_as(body, value, element_scalar(physical)?.kind, None)
        };
        push(body, Statement::Store { pointer, value });
        Ok(())
    }

    fn store_half(
        &mut self,
        out: &mut Block,
        pointer: Handle<Expression>,
        index: Handle<Expression>,
        value: Handle<Expression>,
    ) -> Result<(), EmitError> {
        let value = self.cast_as(out, value, ScalarKind::Float, Some(4));
        let zero = self.f32_lit(0.);
        let ty = self.element_type(ElementType::Vector {
            scalar: ScalarElement::F32,
            lanes: 2,
        })?;
        let pair = self.emit_expr(
            out,
            Expression::Compose {
                ty,
                components: vec![value, zero],
            },
        );
        let bits = self.math(out, MathFunction::Pack2x16float, &[pair]);
        let low = self.u32_lit(0xffff);
        let bits = self.bin(out, BinaryOperator::And, bits, low);
        let half = self.mod_literal_u32(out, index, 2);
        let shift = self.mul_literal_u32(out, half, 16);
        let mask = self.bin(out, BinaryOperator::ShiftLeft, low, shift);
        let bits = self.bin(out, BinaryOperator::ShiftLeft, bits, shift);
        let full = self.u32_lit(u32::MAX);
        let keep = self.bin(out, BinaryOperator::ExclusiveOr, mask, full);
        self.cas_loop(out, pointer, |em, body, old| {
            let kept = em.bin(body, BinaryOperator::And, old, keep);
            em.bin(body, BinaryOperator::InclusiveOr, kept, bits)
        });
        Ok(())
    }
}
