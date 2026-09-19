//! Typed accesses through a single physical storage binding.
//!
//! Homogeneous bindings retain their native element type. Mixed views share
//! u32 words; packed f16 stores update one half atomically so neighboring
//! lanes cannot overwrite each other's values.

use fusor_ir::ir::kernel::{Buffer, ElementType, ScalarElement, StorageView};
use fusor_ir::target::EmitError;
use naga::{
    AtomicFunction, BinaryOperator, Block, Expression, Handle, MathFunction, Scalar, ScalarKind,
    Span, Statement,
};

use super::expr::element_scalar;
use super::{Emitter, key};

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
        Ok(self.emit_expr(body, Expression::Access { base, index }))
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
            let halves = self.math1(body, MathFunction::Unpack2x16float, value);
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
        body.push(Statement::Store { pointer, value }, Span::default());
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
        let bits = self.math1(out, MathFunction::Pack2x16float, pair);
        let low = self.u32_lit(0xffff);
        let bits = self.bin(out, BinaryOperator::And, bits, low);
        let half = self.mod_literal_u32(out, index, 2);
        let shift = self.mul_literal_u32(out, half, 16);
        let mask = self.bin(out, BinaryOperator::ShiftLeft, low, shift);
        let bits = self.bin(out, BinaryOperator::ShiftLeft, bits, shift);
        let full = self.u32_lit(u32::MAX);
        let keep = self.bin(out, BinaryOperator::ExclusiveOr, mask, full);
        let cas_ty = self.module.generate_predeclared_type(
            naga::PredeclaredType::AtomicCompareExchangeWeakResult(Scalar::U32),
        );
        let mut body = Block::new();
        let old = self.emit_load(&mut body, pointer);
        let kept = self.bin(&mut body, BinaryOperator::And, old, keep);
        let new = self.bin(&mut body, BinaryOperator::InclusiveOr, kept, bits);
        let result = self.append(Expression::AtomicResult {
            ty: cas_ty,
            comparison: true,
        });
        body.push(
            Statement::Atomic {
                pointer,
                fun: AtomicFunction::Exchange { compare: Some(old) },
                value: new,
                result: Some(result),
            },
            Span::default(),
        );
        let exchanged = self.emit_expr(
            &mut body,
            Expression::AccessIndex {
                base: result,
                index: 1,
            },
        );
        body.push(
            Statement::If {
                condition: exchanged,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );
        out.push(
            Statement::Loop {
                body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
        Ok(())
    }
}
