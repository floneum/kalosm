//! Per-instruction emission helpers — wrappers over Naga's expression arena.

use egg::Id;
use naga::{
    BinaryOperator, CollectiveOperation, Expression, GatherMode, Handle, Literal, MathFunction,
    Span, Statement, SubgroupOperation, UnaryOperator,
};

use crate::types::ReduceOp;

use super::CodegenCtx;

impl CodegenCtx<'_> {
    pub(super) fn lower_shuffle(&mut self, src_id: Id, lane_id: Id) -> Handle<Expression> {
        let src = self.lower_expr(src_id);
        let lane = self.lower_expr(lane_id);

        // Allocate result expression first.
        let ty = self.infer_expr_type(src_id);
        let result = self
            .expressions
            .append(Expression::SubgroupOperationResult { ty }, Span::UNDEFINED);

        self.body.push(
            Statement::SubgroupGather {
                mode: GatherMode::ShuffleDown(lane),
                argument: src,
                result,
            },
            Span::UNDEFINED,
        );

        result
    }

    pub(super) fn lower_reduce_simd(&mut self, op: ReduceOp, src_id: Id) -> Handle<Expression> {
        let src = self.lower_expr(src_id);

        let sg_op = match op {
            ReduceOp::Add => SubgroupOperation::Add,
            ReduceOp::Mul => SubgroupOperation::Mul,
            ReduceOp::Max => SubgroupOperation::Max,
            ReduceOp::Min => SubgroupOperation::Min,
        };

        let ty = self.infer_expr_type(src_id);
        let result = self
            .expressions
            .append(Expression::SubgroupOperationResult { ty }, Span::UNDEFINED);

        self.body.push(
            Statement::SubgroupCollectiveOperation {
                op: sg_op,
                collective_op: CollectiveOperation::Reduce,
                argument: src,
                result,
            },
            Span::UNDEFINED,
        );

        result
    }

    // ═══════════════════════════════════════════════════════════════
    // Expression emission helpers
    // ═══════════════════════════════════════════════════════════════

    pub(super) fn emit_literal(&mut self, lit: Literal) -> Handle<Expression> {
        let h = self
            .expressions
            .append(Expression::Literal(lit), Span::UNDEFINED);
        self.emit_range(h, h);
        h
    }

    pub(super) fn emit_binary(
        &mut self,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        let h = self
            .expressions
            .append(Expression::Binary { op, left, right }, Span::UNDEFINED);
        self.emit_range(h, h);
        h
    }

    pub(super) fn emit_unary(
        &mut self,
        op: UnaryOperator,
        expr: Handle<Expression>,
    ) -> Handle<Expression> {
        let h = self
            .expressions
            .append(Expression::Unary { op, expr }, Span::UNDEFINED);
        self.emit_range(h, h);
        h
    }

    pub(super) fn emit_cast(
        &mut self,
        expr: Handle<Expression>,
        kind: naga::ScalarKind,
        width: naga::Bytes,
    ) -> Handle<Expression> {
        let h = self.expressions.append(
            Expression::As {
                expr,
                kind,
                convert: Some(width),
            },
            Span::UNDEFINED,
        );
        self.emit_range(h, h);
        h
    }

    pub(super) fn emit_math1(
        &mut self,
        fun: MathFunction,
        arg: Handle<Expression>,
    ) -> Handle<Expression> {
        let h = self.expressions.append(
            Expression::Math {
                fun,
                arg,
                arg1: None,
                arg2: None,
                arg3: None,
            },
            Span::UNDEFINED,
        );
        self.emit_range(h, h);
        h
    }

    pub(super) fn emit_math2(
        &mut self,
        fun: MathFunction,
        arg: Handle<Expression>,
        arg1: Handle<Expression>,
    ) -> Handle<Expression> {
        let h = self.expressions.append(
            Expression::Math {
                fun,
                arg,
                arg1: Some(arg1),
                arg2: None,
                arg3: None,
            },
            Span::UNDEFINED,
        );
        self.emit_range(h, h);
        h
    }

    pub(super) fn emit_math3(
        &mut self,
        fun: MathFunction,
        arg: Handle<Expression>,
        arg1: Handle<Expression>,
        arg2: Handle<Expression>,
    ) -> Handle<Expression> {
        let h = self.expressions.append(
            Expression::Math {
                fun,
                arg,
                arg1: Some(arg1),
                arg2: Some(arg2),
                arg3: None,
            },
            Span::UNDEFINED,
        );
        self.emit_range(h, h);
        h
    }

    pub(super) fn emit_select(
        &mut self,
        condition: Handle<Expression>,
        accept: Handle<Expression>,
        reject: Handle<Expression>,
    ) -> Handle<Expression> {
        let h = self.expressions.append(
            Expression::Select {
                condition,
                accept,
                reject,
            },
            Span::UNDEFINED,
        );
        self.emit_range(h, h);
        h
    }

    pub(super) fn emit_access(
        &mut self,
        base: Handle<Expression>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let h = self
            .expressions
            .append(Expression::Access { base, index }, Span::UNDEFINED);
        self.emit_range(h, h);
        h
    }

    pub(super) fn emit_access_index(
        &mut self,
        base: Handle<Expression>,
        index: u32,
    ) -> Handle<Expression> {
        let h = self
            .expressions
            .append(Expression::AccessIndex { base, index }, Span::UNDEFINED);
        self.emit_range(h, h);
        h
    }

    pub(super) fn emit_load(&mut self, pointer: Handle<Expression>) -> Handle<Expression> {
        let h = self
            .expressions
            .append(Expression::Load { pointer }, Span::UNDEFINED);
        self.emit_range(h, h);
        h
    }

    /// Push `Statement::Emit(range)` for the given expression, avoiding double-emit.
    ///
    /// Skips expressions that are "pre-emitted" (always in scope) or "result"
    /// expressions (made available by their producing statement).
    pub(super) fn emit_range(&mut self, start: Handle<Expression>, end: Handle<Expression>) {
        let expr = &self.expressions[start];
        // Pre-emit expressions: always in scope, never need Emit.
        if expr.needs_pre_emit() {
            return;
        }
        // Result expressions: produced by their statement, cannot be in Emit.
        if matches!(
            expr,
            Expression::SubgroupOperationResult { .. } | Expression::AtomicResult { .. }
        ) {
            return;
        }
        // Avoid double-emit: each handle can only appear in one Emit.
        if self.emitted.insert(start) {
            self.body.push(
                Statement::Emit(naga::Range::new_from_bounds(start, end)),
                Span::UNDEFINED,
            );
        }
    }
}
