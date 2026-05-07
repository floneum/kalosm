use wgpu::naga::{
    AddressSpace, Arena, BinaryOperator, Block, Expression, Function, GlobalVariable, Handle,
    Literal, LocalVariable, MathFunction, Module, Range, ResourceBinding, Span, Statement,
    StorageAccess, Type,
};

use super::SampleMirostat2Globals;

impl super::SampleMirostat2ModuleBuilder {
    pub(super) fn top_weight(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: SampleMirostat2Globals,
        max_value: Handle<Expression>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let value = self.top_value(expressions, body, globals, index);
        let delta = self.bin(
            expressions,
            body,
            BinaryOperator::Subtract,
            value,
            max_value,
        );
        self.exp_f32(expressions, body, delta)
    }

    pub(super) fn top_value(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: SampleMirostat2Globals,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let index = self.index1(
            expressions,
            body,
            self.meta.values_offset,
            self.meta.values_stride,
            index,
        );
        self.load_storage(expressions, body, globals.values, index)
    }

    pub(super) fn top_id(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: SampleMirostat2Globals,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let index = self.index1(
            expressions,
            body,
            self.meta.ids_offset,
            self.meta.ids_stride,
            index,
        );
        self.load_storage(expressions, body, globals.ids, index)
    }

    pub(super) fn store_sample_result(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        output: Handle<GlobalVariable>,
        status: u32,
        token: u32,
    ) {
        let token = self.u32_lit(expressions, token);
        self.store_sample_result_handle(expressions, body, output, status, token);
    }

    pub(super) fn store_sample_result_handle(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        output: Handle<GlobalVariable>,
        status: u32,
        token: Handle<Expression>,
    ) {
        let zero = self.u32_lit(expressions, 0);
        let one = self.u32_lit(expressions, 1);
        let status = self.u32_lit(expressions, status);
        self.store_storage(expressions, body, output, zero, status);
        self.store_storage(expressions, body, output, one, token);
    }

    pub(super) fn storage_global(
        module: &mut Module,
        name: &str,
        binding: u32,
        ty: Handle<Type>,
        read_only: bool,
    ) -> Handle<GlobalVariable> {
        module.global_variables.append(
            GlobalVariable {
                name: Some(name.into()),
                space: AddressSpace::Storage {
                    access: if read_only {
                        StorageAccess::LOAD
                    } else {
                        StorageAccess::LOAD | StorageAccess::STORE
                    },
                },
                binding: Some(ResourceBinding { group: 0, binding }),
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    pub(super) fn workgroup_global(
        module: &mut Module,
        name: &str,
        ty: Handle<Type>,
    ) -> Handle<GlobalVariable> {
        module.global_variables.append(
            GlobalVariable {
                name: Some(name.into()),
                space: AddressSpace::WorkGroup,
                binding: None,
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    pub(super) fn local(
        function: &mut Function,
        name: &str,
        ty: Handle<Type>,
    ) -> Handle<LocalVariable> {
        function.local_variables.append(
            LocalVariable {
                name: Some(name.into()),
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    pub(super) fn index1(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        offset: u32,
        stride: u32,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let scaled = if stride == 1 {
            index
        } else {
            self.mul_lit(expressions, body, index, stride)
        };
        self.add_lit(expressions, body, scaled, offset)
    }

    pub(super) fn load_storage(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let ptr = self.storage_ptr(expressions, body, global, index);
        self.emit(expressions, body, Expression::Load { pointer: ptr })
    }

    pub(super) fn store_storage(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
        value: Handle<Expression>,
    ) {
        let pointer = self.storage_ptr(expressions, body, global, index);
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    pub(super) fn storage_ptr(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let base = expressions.append(Expression::GlobalVariable(global), Span::default());
        self.emit(expressions, body, Expression::Access { base, index })
    }

    pub(super) fn load_local(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        local: Handle<LocalVariable>,
    ) -> Handle<Expression> {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        self.emit(expressions, body, Expression::Load { pointer })
    }

    pub(super) fn store_local(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        local: Handle<LocalVariable>,
        value: Handle<Expression>,
    ) {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    pub(super) fn load_param_f32(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: u32,
    ) -> Handle<Expression> {
        let index = self.u32_lit(expressions, index);
        self.load_storage(expressions, body, global, index)
    }

    pub(super) fn add_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        if literal == 0 {
            value
        } else {
            let rhs = self.u32_lit(expressions, literal);
            self.bin(expressions, body, BinaryOperator::Add, value, rhs)
        }
    }

    pub(super) fn mul_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Multiply, value, rhs)
    }

    pub(super) fn ge_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::GreaterEqual, value, rhs)
    }

    pub(super) fn eq_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Equal, value, rhs)
    }

    pub(super) fn exp_f32(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            expressions,
            body,
            Expression::Math {
                fun: MathFunction::Exp,
                arg: value,
                arg1: None,
                arg2: None,
                arg3: None,
            },
        )
    }

    pub(super) fn log2_f32(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            expressions,
            body,
            Expression::Math {
                fun: MathFunction::Log2,
                arg: value,
                arg1: None,
                arg2: None,
                arg3: None,
            },
        )
    }

    pub(super) fn max_f32(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            expressions,
            body,
            Expression::Math {
                fun: MathFunction::Max,
                arg: left,
                arg1: Some(right),
                arg2: None,
                arg3: None,
            },
        )
    }

    pub(super) fn bin(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(expressions, body, Expression::Binary { op, left, right })
    }

    pub(super) fn emit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        expression: Expression,
    ) -> Handle<Expression> {
        let handle = expressions.append(expression, Span::default());
        body.push(
            Statement::Emit(Range::new_from_bounds(handle, handle)),
            Span::default(),
        );
        handle
    }

    pub(super) fn f32_lit(
        &self,
        expressions: &mut Arena<Expression>,
        value: f32,
    ) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::F32(value)), Span::default())
    }

    pub(super) fn u32_lit(
        &self,
        expressions: &mut Arena<Expression>,
        value: u32,
    ) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::U32(value)), Span::default())
    }
}
