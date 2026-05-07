use std::num::NonZeroU32;

use wgpu::naga::{
    Arena, ArraySize, Barrier, BinaryOperator, Binding, Block, BuiltIn, EntryPoint, Expression,
    Function, FunctionArgument, GlobalVariable, Handle, LocalVariable, Module, Scalar, ShaderStage,
    Span, Statement, Type, TypeInner, VectorSize,
};

use super::{DECODE_HEAD_DIM, FLOAT_MIN, FlashDecodeSmallMeta};

#[path = "flash_decode_small_helpers.rs"]
mod helpers;

#[derive(Clone, Copy)]
struct FlashDecodeSmallGlobals {
    q: Handle<GlobalVariable>,
    k: Handle<GlobalVariable>,
    v: Handle<GlobalVariable>,
    output: Handle<GlobalVariable>,
    params: Handle<GlobalVariable>,
    scores: Handle<GlobalVariable>,
    probs: Handle<GlobalVariable>,
    reduce: Handle<GlobalVariable>,
}

#[derive(Clone, Copy)]
struct FlashDecodeSmallLocals {
    acc: Handle<LocalVariable>,
    kv: Handle<LocalVariable>,
    item: Handle<LocalVariable>,
}

#[derive(Clone, Copy)]
struct FlashDecodeRowIndices {
    batch_idx: Handle<Expression>,
    head_idx: Handle<Expression>,
    kv_head_idx: Handle<Expression>,
}

struct FlashDecodeSmallNagaBuilder {
    meta: FlashDecodeSmallMeta,
}

impl FlashDecodeSmallNagaBuilder {
    fn new(meta: FlashDecodeSmallMeta) -> Self {
        Self { meta }
    }

    fn build(self) -> Option<Module> {
        let mut module = Module::default();
        let f32_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::F32),
            },
            Span::default(),
        );
        let u32_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::default(),
        );
        let u32_vec3_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Vector {
                    size: VectorSize::Tri,
                    scalar: Scalar::U32,
                },
            },
            Span::default(),
        );
        let storage_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Array {
                    base: f32_ty,
                    size: ArraySize::Dynamic,
                    stride: 4,
                },
            },
            Span::default(),
        );
        let u32_storage_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Dynamic,
                    stride: 4,
                },
            },
            Span::default(),
        );
        let scratch_ty = module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Array {
                    base: f32_ty,
                    size: ArraySize::Constant(NonZeroU32::new(self.meta.decode_block)?),
                    stride: 4,
                },
            },
            Span::default(),
        );

        let q = Self::storage_global(&mut module, 0, storage_ty, true);
        let k = Self::storage_global(&mut module, 1, storage_ty, true);
        let v = Self::storage_global(&mut module, 2, storage_ty, true);
        let output = Self::storage_global(&mut module, 3, storage_ty, false);
        let params = Self::storage_global(&mut module, 4, u32_storage_ty, true);
        let scores = Self::workgroup_global(&mut module, scratch_ty);
        let probs = Self::workgroup_global(&mut module, scratch_ty);
        let reduce = Self::workgroup_global(&mut module, scratch_ty);
        let globals = FlashDecodeSmallGlobals {
            q,
            k,
            v,
            output,
            params,
            scores,
            probs,
            reduce,
        };

        let mut function = Function {
            name: None,
            arguments: vec![
                FunctionArgument {
                    name: None,
                    ty: u32_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)),
                },
                FunctionArgument {
                    name: None,
                    ty: u32_vec3_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::WorkGroupId)),
                },
            ],
            ..Function::default()
        };
        let locals = FlashDecodeSmallLocals {
            acc: Self::local(&mut function, f32_ty),
            kv: Self::local(&mut function, u32_ty),
            item: Self::local(&mut function, u32_ty),
        };

        function.body = self.entry_body(&mut function.expressions, globals, locals);
        function
            .body
            .push(Statement::Return { value: None }, Span::default());
        module.entry_points.push(EntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [self.meta.decode_block, 1, 1],
            workgroup_size_overrides: None,
            function,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });

        Some(module)
    }

    fn entry_body(
        &self,
        expressions: &mut Arena<Expression>,
        globals: FlashDecodeSmallGlobals,
        locals: FlashDecodeSmallLocals,
    ) -> Block {
        if self.meta.tiled {
            return self.entry_body_tiled(expressions, globals, locals);
        }

        let mut body = Block::new();
        let local = expressions.append(Expression::FunctionArgument(0), Span::default());
        let workgroup_id = expressions.append(Expression::FunctionArgument(1), Span::default());
        let zero_param_index = self.u32_lit(expressions, 0);
        let active_kv_len =
            self.load_storage(expressions, &mut body, globals.params, zero_param_index);
        let row = self.emit(
            expressions,
            &mut body,
            Expression::AccessIndex {
                base: workgroup_id,
                index: 0,
            },
        );
        let head_idx = self.rem_lit(expressions, &mut body, row, self.meta.dims.num_heads);
        let batch_idx = self.div_lit(expressions, &mut body, row, self.meta.dims.num_heads);
        let kv_head_idx = self.div_lit(expressions, &mut body, head_idx, self.meta.groups);
        let kv_valid = self.bin(
            expressions,
            &mut body,
            BinaryOperator::Less,
            local,
            active_kv_len,
        );

        let min_score = self.f32_lit(expressions, FLOAT_MIN);
        self.store_workgroup(expressions, &mut body, globals.scores, local, min_score);
        self.store_workgroup(expressions, &mut body, globals.reduce, local, min_score);

        let mut score_accept = Block::new();
        let zero = self.f32_lit(expressions, 0.0);
        let mut score = zero;
        for dim in 0..DECODE_HEAD_DIM {
            let q_index = self.index4_const_last(
                expressions,
                &mut score_accept,
                self.meta.q_offset,
                self.meta.q_strides,
                batch_idx,
                head_idx,
                0,
                dim,
            );
            let k_index = self.index4_const_last_dyn_i2(
                expressions,
                &mut score_accept,
                self.meta.k_offset,
                self.meta.k_strides,
                batch_idx,
                kv_head_idx,
                local,
                dim,
            );
            let q_value = self.load_storage(expressions, &mut score_accept, globals.q, q_index);
            let k_value = self.load_storage(expressions, &mut score_accept, globals.k, k_index);
            let product = self.bin(
                expressions,
                &mut score_accept,
                BinaryOperator::Multiply,
                q_value,
                k_value,
            );
            score = self.bin(
                expressions,
                &mut score_accept,
                BinaryOperator::Add,
                score,
                product,
            );
        }
        let scale = self.f32_lit(expressions, self.meta.scale);
        score = self.bin(
            expressions,
            &mut score_accept,
            BinaryOperator::Multiply,
            score,
            scale,
        );
        self.store_workgroup(expressions, &mut score_accept, globals.scores, local, score);
        self.store_workgroup(expressions, &mut score_accept, globals.reduce, local, score);
        body.push(
            Statement::If {
                condition: kv_valid,
                accept: score_accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        self.reduce_workgroup(
            expressions,
            &mut body,
            globals.reduce,
            local,
            FlashReduceOp::Max,
        );
        let zero_index = self.u32_lit(expressions, 0);
        let max_score = self.load_workgroup(expressions, &mut body, globals.reduce, zero_index);
        let score_value = self.load_workgroup(expressions, &mut body, globals.scores, local);
        let shifted = self.bin(
            expressions,
            &mut body,
            BinaryOperator::Subtract,
            score_value,
            max_score,
        );
        let raw_prob = self.exp_f32(expressions, &mut body, shifted);
        let prob = self.select(expressions, &mut body, kv_valid, raw_prob, zero);
        self.store_workgroup(expressions, &mut body, globals.probs, local, prob);
        self.store_workgroup(expressions, &mut body, globals.reduce, local, prob);
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        self.reduce_workgroup(
            expressions,
            &mut body,
            globals.reduce,
            local,
            FlashReduceOp::Sum,
        );
        let denom = self.load_workgroup(expressions, &mut body, globals.reduce, zero_index);
        let mut normalize_accept = Block::new();
        let prob = self.load_workgroup(expressions, &mut normalize_accept, globals.probs, local);
        let prob = self.bin(
            expressions,
            &mut normalize_accept,
            BinaryOperator::Divide,
            prob,
            denom,
        );
        self.store_workgroup(
            expressions,
            &mut normalize_accept,
            globals.probs,
            local,
            prob,
        );
        body.push(
            Statement::If {
                condition: kv_valid,
                accept: normalize_accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let out_valid = self.lt_lit(expressions, &mut body, local, DECODE_HEAD_DIM);
        let mut store_accept = Block::new();
        self.store_local(expressions, &mut store_accept, locals.acc, zero);
        let zero_u32 = self.u32_lit(expressions, 0);
        self.store_local(expressions, &mut store_accept, locals.kv, zero_u32);
        self.append_output_loop(
            expressions,
            &mut store_accept,
            globals,
            locals,
            batch_idx,
            head_idx,
            kv_head_idx,
            local,
            active_kv_len,
        );
        body.push(
            Statement::If {
                condition: out_valid,
                accept: store_accept,
                reject: Block::new(),
            },
            Span::default(),
        );

        body
    }

    fn score_for_kv(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: FlashDecodeSmallGlobals,
        indices: FlashDecodeRowIndices,
        kv: Handle<Expression>,
    ) -> Handle<Expression> {
        let zero = self.f32_lit(expressions, 0.0);
        let mut score = zero;
        for dim in 0..DECODE_HEAD_DIM {
            let q_index = self.index4_const_last(
                expressions,
                body,
                self.meta.q_offset,
                self.meta.q_strides,
                indices.batch_idx,
                indices.head_idx,
                0,
                dim,
            );
            let k_index = self.index4_const_last_dyn_i2(
                expressions,
                body,
                self.meta.k_offset,
                self.meta.k_strides,
                indices.batch_idx,
                indices.kv_head_idx,
                kv,
                dim,
            );
            let q_value = self.load_storage(expressions, body, globals.q, q_index);
            let k_value = self.load_storage(expressions, body, globals.k, k_index);
            let product = self.bin(
                expressions,
                body,
                BinaryOperator::Multiply,
                q_value,
                k_value,
            );
            score = self.bin(expressions, body, BinaryOperator::Add, score, product);
        }
        let scale = self.f32_lit(expressions, self.meta.scale);
        self.bin(expressions, body, BinaryOperator::Multiply, score, scale)
    }

    fn entry_body_tiled(
        &self,
        expressions: &mut Arena<Expression>,
        globals: FlashDecodeSmallGlobals,
        locals: FlashDecodeSmallLocals,
    ) -> Block {
        let mut body = Block::new();
        let local = expressions.append(Expression::FunctionArgument(0), Span::default());
        let workgroup_id = expressions.append(Expression::FunctionArgument(1), Span::default());
        let zero_param_index = self.u32_lit(expressions, 0);
        let active_kv_len =
            self.load_storage(expressions, &mut body, globals.params, zero_param_index);
        let row = self.emit(
            expressions,
            &mut body,
            Expression::AccessIndex {
                base: workgroup_id,
                index: 0,
            },
        );
        let head_idx = self.rem_lit(expressions, &mut body, row, self.meta.dims.num_heads);
        let batch_idx = self.div_lit(expressions, &mut body, row, self.meta.dims.num_heads);
        let kv_head_idx = self.div_lit(expressions, &mut body, head_idx, self.meta.groups);

        let min_score = self.f32_lit(expressions, FLOAT_MIN);
        self.store_workgroup(expressions, &mut body, globals.reduce, local, min_score);
        self.store_local(expressions, &mut body, locals.kv, local);
        self.append_tiled_max_loop(
            expressions,
            &mut body,
            globals,
            locals,
            FlashDecodeRowIndices {
                batch_idx,
                head_idx,
                kv_head_idx,
            },
            local,
            active_kv_len,
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        self.reduce_workgroup(
            expressions,
            &mut body,
            globals.reduce,
            local,
            FlashReduceOp::Max,
        );
        let zero_index = self.u32_lit(expressions, 0);
        let max_score = self.load_workgroup(expressions, &mut body, globals.reduce, zero_index);

        let zero = self.f32_lit(expressions, 0.0);
        self.store_workgroup(expressions, &mut body, globals.reduce, local, zero);
        self.store_local(expressions, &mut body, locals.kv, local);
        self.append_tiled_sum_loop(
            expressions,
            &mut body,
            globals,
            locals,
            FlashDecodeRowIndices {
                batch_idx,
                head_idx,
                kv_head_idx,
            },
            local,
            active_kv_len,
            max_score,
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        self.reduce_workgroup(
            expressions,
            &mut body,
            globals.reduce,
            local,
            FlashReduceOp::Sum,
        );
        let denom = self.load_workgroup(expressions, &mut body, globals.reduce, zero_index);

        self.store_local(expressions, &mut body, locals.acc, zero);
        let zero_u32 = self.u32_lit(expressions, 0);
        self.store_local(expressions, &mut body, locals.kv, zero_u32);
        self.append_tiled_output_loop(
            expressions,
            &mut body,
            globals,
            locals,
            FlashDecodeRowIndices {
                batch_idx,
                head_idx,
                kv_head_idx,
            },
            local,
            active_kv_len,
            max_score,
            denom,
        );

        let output_value = self.load_local(expressions, &mut body, locals.acc);
        let q_idx = self.u32_lit(expressions, 0);
        let output_index = self.index4_dyn_last(
            expressions,
            &mut body,
            self.meta.output_offset,
            self.meta.output_strides,
            batch_idx,
            head_idx,
            q_idx,
            local,
        );
        self.store_storage(
            expressions,
            &mut body,
            globals.output,
            output_index,
            output_value,
        );

        body
    }
}

#[derive(Clone, Copy)]
enum FlashReduceOp {
    Sum,
    Max,
}

pub(super) fn build_flash_decode_small_naga_module(meta: FlashDecodeSmallMeta) -> Option<Module> {
    FlashDecodeSmallNagaBuilder::new(meta).build()
}
