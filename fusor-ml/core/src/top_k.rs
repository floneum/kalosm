use std::num::NonZeroU32;

use crate::{
    mir::direct_kernel::{DirectKernel, DirectKernelBinding},
    tensor::{DataTypeEnum, TensorData},
};
use wgpu::naga::{
    AddressSpace, Arena, ArraySize, Barrier, BinaryOperator, Binding, Block, BuiltIn, EntryPoint,
    Expression, Function, FunctionArgument, GlobalVariable, Handle, Literal, LocalVariable,
    MathFunction, Module, Range, ResourceBinding, Scalar, ShaderStage, Span, Statement,
    StorageAccess, Type, TypeInner, VectorSize,
};

const TOP_K_BLOCK: u32 = 256;
pub(crate) const TOP_K_CHUNK: usize = TOP_K_BLOCK as usize;
pub(crate) const MIN_TOP_K_CANDIDATES_PER_CHUNK: usize = 64;
const MAX_F32: f32 = 3.4028234663852886e38;
const NEG_MAX_F32: f32 = -3.4028234663852886e38;

pub(crate) fn chunk_top_k_pair_data(
    input: &TensorData,
    candidate_count: usize,
    output_per_chunk: usize,
) -> Option<(TensorData, TensorData)> {
    if input.datatype() != DataTypeEnum::F32 || input.layout().rank() != 1 {
        return None;
    }

    let input_len = input.layout().shape()[0];
    let chunks = input_len.div_ceil(TOP_K_CHUNK);
    let output_len = chunks.checked_mul(output_per_chunk)?;
    let device = input.device();
    let ids = TensorData::new_for_shape(device, &[output_len], DataTypeEnum::U32);
    let values = TensorData::new_for_shape(device, &[output_len], DataTypeEnum::F32);
    if candidate_count == 0 || output_per_chunk == 0 || input_len == 0 {
        return Some((ids, values));
    }

    let input_offset = input.layout().offset();
    let input_stride = input.layout().strides()[0];
    let cache_key = format!(
        "chunk_top_k_pairs_f32:block={TOP_K_BLOCK}:chunk={TOP_K_CHUNK}:len={input_len}:candidate_count={candidate_count}:output_per_chunk={output_per_chunk}:offset={input_offset}:stride={input_stride}"
    );
    let module = if let Some(module) = device.naga_module_cache().write().get(&cache_key) {
        module.clone()
    } else {
        let module = TopKModuleBuilder::new(
            input_len.try_into().ok()?,
            output_per_chunk.try_into().ok()?,
            input_offset.try_into().ok()?,
            input_stride.try_into().ok()?,
        )
        .build()?;
        device
            .naga_module_cache()
            .write()
            .get_or_insert(cache_key.clone(), || module.clone())
            .clone()
    };

    let kernel = DirectKernel::new_with_cache_key(
        "chunk_top_k_pairs_f32",
        cache_key,
        module,
        vec![
            DirectKernelBinding::Storage {
                binding: 0,
                buffer: input.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 1,
                buffer: ids.buffer().clone(),
                read_only: false,
            },
            DirectKernelBinding::Storage {
                binding: 2,
                buffer: values.buffer().clone(),
                read_only: false,
            },
        ],
        [chunks.try_into().ok()?, 1, 1],
    );

    let mut encoder =
        device
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("chunk_top_k_pairs_f32 encoder"),
            });
    kernel.run(device, &mut encoder);
    device.wgpu_queue().submit(Some(encoder.finish()));

    Some((ids, values))
}

pub(crate) fn merge_sorted_chunk_top_k_pair_data(
    input_ids: &TensorData,
    input_values: &TensorData,
    chunks: usize,
    chunk_len: usize,
    chunk_stride: usize,
    input_len: usize,
    k: usize,
) -> Option<(TensorData, TensorData)> {
    if input_ids.datatype() != DataTypeEnum::U32 || input_values.datatype() != DataTypeEnum::F32 {
        return None;
    }
    if input_ids.layout().rank() != 1 || input_values.layout().rank() != 1 {
        return None;
    }
    let input_ids_len = input_ids.layout().shape()[0];
    let input_values_len = input_values.layout().shape()[0];
    let expected_len = if chunks == 0 {
        0
    } else {
        (chunks - 1)
            .checked_mul(chunk_stride)?
            .checked_add(chunk_len)?
    };
    if input_ids_len < expected_len || input_values_len < expected_len {
        return None;
    }

    let device = input_values.device();
    let output_len = k.min(input_len);
    let ids = TensorData::new_for_shape(device, &[output_len], DataTypeEnum::U32);
    let values = TensorData::new_for_shape(device, &[output_len], DataTypeEnum::F32);
    if chunks == 0 || chunk_len == 0 || output_len == 0 {
        return Some((ids, values));
    }

    let cache_key = format!(
        "merge_sorted_chunk_top_k_pairs_f32:block={TOP_K_BLOCK}:chunks={chunks}:chunk_len={chunk_len}:chunk_stride={chunk_stride}:input_len={input_len}:k={output_len}:ids={:?}:values={:?}",
        input_ids.layout(),
        input_values.layout()
    );
    let module = if let Some(module) = device.naga_module_cache().write().get(&cache_key) {
        module.clone()
    } else {
        let module = MergeTopKModuleBuilder::new(
            chunks.try_into().ok()?,
            chunk_len.try_into().ok()?,
            chunk_stride.try_into().ok()?,
            input_len.try_into().ok()?,
            output_len.try_into().ok()?,
        )
        .build()?;
        device
            .naga_module_cache()
            .write()
            .get_or_insert(cache_key.clone(), || module.clone())
            .clone()
    };

    let kernel = DirectKernel::new_with_cache_key(
        "merge_sorted_chunk_top_k_pairs_f32",
        cache_key,
        module,
        vec![
            DirectKernelBinding::Storage {
                binding: 0,
                buffer: input_ids.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 1,
                buffer: input_values.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 2,
                buffer: ids.buffer().clone(),
                read_only: false,
            },
            DirectKernelBinding::Storage {
                binding: 3,
                buffer: values.buffer().clone(),
                read_only: false,
            },
        ],
        [1, 1, 1],
    );

    let mut encoder =
        device
            .wgpu_device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("merge_sorted_chunk_top_k_pairs_f32 encoder"),
            });
    kernel.run(device, &mut encoder);
    device.wgpu_queue().submit(Some(encoder.finish()));

    Some((ids, values))
}

struct TopKModuleBuilder {
    input_len: u32,
    output_per_chunk: u32,
    input_offset: u32,
    input_stride: u32,
}

struct TopKGlobals {
    input: Handle<GlobalVariable>,
    output_ids: Handle<GlobalVariable>,
    output_values: Handle<GlobalVariable>,
    scratch_values: Handle<GlobalVariable>,
    scratch_ids: Handle<GlobalVariable>,
}

struct TopKLocals {
    current_value: Handle<LocalVariable>,
    current_id: Handle<LocalVariable>,
}

struct MergeTopKModuleBuilder {
    chunks: u32,
    chunk_len: u32,
    chunk_stride: u32,
    input_len: u32,
    k: u32,
}

struct MergeTopKGlobals {
    input_ids: Handle<GlobalVariable>,
    input_values: Handle<GlobalVariable>,
    output_ids: Handle<GlobalVariable>,
    output_values: Handle<GlobalVariable>,
    chunk_positions: Handle<GlobalVariable>,
    scratch_values: Handle<GlobalVariable>,
    scratch_ids: Handle<GlobalVariable>,
    scratch_chunks: Handle<GlobalVariable>,
}

struct MergeTopKLocals {
    rank: Handle<LocalVariable>,
    scan_chunk: Handle<LocalVariable>,
    local_best_value: Handle<LocalVariable>,
    local_best_id: Handle<LocalVariable>,
    local_best_chunk: Handle<LocalVariable>,
    reduce_step: Handle<LocalVariable>,
}

impl MergeTopKModuleBuilder {
    fn new(chunks: u32, chunk_len: u32, chunk_stride: u32, input_len: u32, k: u32) -> Self {
        Self {
            chunks,
            chunk_len,
            chunk_stride,
            input_len,
            k,
        }
    }

    fn build(self) -> Option<Module> {
        let mut module = Module::default();
        let f32_ty = module.types.insert(
            Type {
                name: Some("MergeTopKF32".into()),
                inner: TypeInner::Scalar(Scalar::F32),
            },
            Span::default(),
        );
        let u32_ty = module.types.insert(
            Type {
                name: Some("MergeTopKU32".into()),
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::default(),
        );
        let f32_storage_ty = module.types.insert(
            Type {
                name: Some("MergeTopKF32Buffer".into()),
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
                name: Some("MergeTopKU32Buffer".into()),
                inner: TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Dynamic,
                    stride: 4,
                },
            },
            Span::default(),
        );
        let chunk_positions_ty = module.types.insert(
            Type {
                name: Some("MergeTopKChunkPositions".into()),
                inner: TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Constant(NonZeroU32::new(self.chunks)?),
                    stride: 4,
                },
            },
            Span::default(),
        );
        let scratch_f32_ty = module.types.insert(
            Type {
                name: Some("MergeTopKScratchF32".into()),
                inner: TypeInner::Array {
                    base: f32_ty,
                    size: ArraySize::Constant(NonZeroU32::new(TOP_K_BLOCK)?),
                    stride: 4,
                },
            },
            Span::default(),
        );
        let scratch_u32_ty = module.types.insert(
            Type {
                name: Some("MergeTopKScratchU32".into()),
                inner: TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Constant(NonZeroU32::new(TOP_K_BLOCK)?),
                    stride: 4,
                },
            },
            Span::default(),
        );

        let globals = MergeTopKGlobals {
            input_ids: Self::storage_global(&mut module, "input_ids", 0, u32_storage_ty, true),
            input_values: Self::storage_global(
                &mut module,
                "input_values",
                1,
                f32_storage_ty,
                true,
            ),
            output_ids: Self::storage_global(&mut module, "output_ids", 2, u32_storage_ty, false),
            output_values: Self::storage_global(
                &mut module,
                "output_values",
                3,
                f32_storage_ty,
                false,
            ),
            chunk_positions: Self::workgroup_global(
                &mut module,
                "chunk_positions",
                chunk_positions_ty,
            ),
            scratch_values: Self::workgroup_global(&mut module, "scratch_values", scratch_f32_ty),
            scratch_ids: Self::workgroup_global(&mut module, "scratch_ids", scratch_u32_ty),
            scratch_chunks: Self::workgroup_global(&mut module, "scratch_chunks", scratch_u32_ty),
        };

        let mut function = Function {
            name: Some("main".into()),
            arguments: vec![FunctionArgument {
                name: Some("local_invocation_index".into()),
                ty: u32_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)),
            }],
            ..Function::default()
        };
        let locals = MergeTopKLocals {
            rank: Self::local(&mut function, "rank", u32_ty),
            scan_chunk: Self::local(&mut function, "scan_chunk", u32_ty),
            local_best_value: Self::local(&mut function, "local_best_value", f32_ty),
            local_best_id: Self::local(&mut function, "local_best_id", u32_ty),
            local_best_chunk: Self::local(&mut function, "local_best_chunk", u32_ty),
            reduce_step: Self::local(&mut function, "reduce_step", u32_ty),
        };

        function.body = self.entry_body(&mut function.expressions, globals, locals);
        function
            .body
            .push(Statement::Return { value: None }, Span::default());
        module.entry_points.push(EntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [TOP_K_BLOCK, 1, 1],
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
        globals: MergeTopKGlobals,
        locals: MergeTopKLocals,
    ) -> Block {
        let mut body = Block::new();
        let lane = expressions.append(Expression::FunctionArgument(0), Span::default());

        self.store_local(expressions, &mut body, locals.scan_chunk, lane);
        let mut init_body = Block::new();
        let chunk = self.load_local(expressions, &mut init_body, locals.scan_chunk);
        let done = self.ge_lit(expressions, &mut init_body, chunk, self.chunks);
        init_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );
        let zero = self.u32_lit(expressions, 0);
        self.store_storage(
            expressions,
            &mut init_body,
            globals.chunk_positions,
            chunk,
            zero,
        );
        let chunk = self.load_local(expressions, &mut init_body, locals.scan_chunk);
        let next = self.add_lit(expressions, &mut init_body, chunk, TOP_K_BLOCK);
        self.store_local(expressions, &mut init_body, locals.scan_chunk, next);
        body.push(
            Statement::Loop {
                body: init_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let zero = self.u32_lit(expressions, 0);
        self.store_local(expressions, &mut body, locals.rank, zero);
        let mut rank_body = Block::new();
        let rank = self.load_local(expressions, &mut rank_body, locals.rank);
        let done = self.ge_lit(expressions, &mut rank_body, rank, self.k);
        rank_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let neg_max = self.f32_lit(expressions, NEG_MAX_F32);
        let invalid = self.u32_lit(expressions, u32::MAX);
        self.store_local(
            expressions,
            &mut rank_body,
            locals.local_best_value,
            neg_max,
        );
        self.store_local(expressions, &mut rank_body, locals.local_best_id, invalid);
        self.store_local(
            expressions,
            &mut rank_body,
            locals.local_best_chunk,
            invalid,
        );
        self.store_local(expressions, &mut rank_body, locals.scan_chunk, lane);

        self.append_scan_chunks_loop(expressions, &mut rank_body, &globals, &locals);
        self.store_local_best_to_scratch(expressions, &mut rank_body, &globals, &locals, lane);
        self.append_reduce_loop(expressions, &mut rank_body, &globals, &locals, lane);
        self.store_rank_output(expressions, &mut rank_body, &globals, &locals, lane);

        let rank = self.load_local(expressions, &mut rank_body, locals.rank);
        let next_rank = self.add_lit(expressions, &mut rank_body, rank, 1);
        self.store_local(expressions, &mut rank_body, locals.rank, next_rank);
        body.push(
            Statement::Loop {
                body: rank_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );

        body
    }

    fn append_scan_chunks_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: &MergeTopKGlobals,
        locals: &MergeTopKLocals,
    ) {
        let mut scan_body = Block::new();
        let chunk = self.load_local(expressions, &mut scan_body, locals.scan_chunk);
        let done = self.ge_lit(expressions, &mut scan_body, chunk, self.chunks);
        scan_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let position =
            self.load_storage(expressions, &mut scan_body, globals.chunk_positions, chunk);
        let chunk_len = self.u32_lit(expressions, self.chunk_len);
        let in_chunk = self.bin(
            expressions,
            &mut scan_body,
            BinaryOperator::Less,
            position,
            chunk_len,
        );
        let mut candidate_accept = Block::new();
        let chunk_offset =
            self.mul_lit(expressions, &mut candidate_accept, chunk, self.chunk_stride);
        let index = self.bin(
            expressions,
            &mut candidate_accept,
            BinaryOperator::Add,
            chunk_offset,
            position,
        );
        let id = self.load_storage(expressions, &mut candidate_accept, globals.input_ids, index);
        let input_len = self.u32_lit(expressions, self.input_len);
        let valid_id = self.bin(
            expressions,
            &mut candidate_accept,
            BinaryOperator::Less,
            id,
            input_len,
        );
        let value = self.load_storage(
            expressions,
            &mut candidate_accept,
            globals.input_values,
            index,
        );
        let finite = self.is_finite(expressions, &mut candidate_accept, value);
        let valid = self.and(expressions, &mut candidate_accept, valid_id, finite);
        let best_value =
            self.load_local(expressions, &mut candidate_accept, locals.local_best_value);
        let best_id = self.load_local(expressions, &mut candidate_accept, locals.local_best_id);
        let better = self.better_candidate(
            expressions,
            &mut candidate_accept,
            value,
            id,
            best_value,
            best_id,
        );
        let should_update = self.and(expressions, &mut candidate_accept, valid, better);
        let mut update = Block::new();
        self.store_local(expressions, &mut update, locals.local_best_value, value);
        self.store_local(expressions, &mut update, locals.local_best_id, id);
        self.store_local(expressions, &mut update, locals.local_best_chunk, chunk);
        candidate_accept.push(
            Statement::If {
                condition: should_update,
                accept: update,
                reject: Block::new(),
            },
            Span::default(),
        );
        scan_body.push(
            Statement::If {
                condition: in_chunk,
                accept: candidate_accept,
                reject: Block::new(),
            },
            Span::default(),
        );

        let chunk = self.load_local(expressions, &mut scan_body, locals.scan_chunk);
        let next = self.add_lit(expressions, &mut scan_body, chunk, TOP_K_BLOCK);
        self.store_local(expressions, &mut scan_body, locals.scan_chunk, next);
        body.push(
            Statement::Loop {
                body: scan_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }

    fn store_local_best_to_scratch(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: &MergeTopKGlobals,
        locals: &MergeTopKLocals,
        lane: Handle<Expression>,
    ) {
        let value = self.load_local(expressions, body, locals.local_best_value);
        let id = self.load_local(expressions, body, locals.local_best_id);
        let chunk = self.load_local(expressions, body, locals.local_best_chunk);
        self.store_storage(expressions, body, globals.scratch_values, lane, value);
        self.store_storage(expressions, body, globals.scratch_ids, lane, id);
        self.store_storage(expressions, body, globals.scratch_chunks, lane, chunk);
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
    }

    fn append_reduce_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: &MergeTopKGlobals,
        locals: &MergeTopKLocals,
        lane: Handle<Expression>,
    ) {
        let half_block = self.u32_lit(expressions, TOP_K_BLOCK / 2);
        self.store_local(expressions, body, locals.reduce_step, half_block);

        let mut reduce_body = Block::new();
        let step = self.load_local(expressions, &mut reduce_body, locals.reduce_step);
        let done = self.eq_lit(expressions, &mut reduce_body, step, 0);
        reduce_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let participates = self.bin(
            expressions,
            &mut reduce_body,
            BinaryOperator::Less,
            lane,
            step,
        );
        let mut accept = Block::new();
        let other_index = self.bin(expressions, &mut accept, BinaryOperator::Add, lane, step);
        let other_value = self.load_storage(
            expressions,
            &mut accept,
            globals.scratch_values,
            other_index,
        );
        let other_id =
            self.load_storage(expressions, &mut accept, globals.scratch_ids, other_index);
        let other_chunk = self.load_storage(
            expressions,
            &mut accept,
            globals.scratch_chunks,
            other_index,
        );
        let current_value =
            self.load_storage(expressions, &mut accept, globals.scratch_values, lane);
        let current_id = self.load_storage(expressions, &mut accept, globals.scratch_ids, lane);
        let better = self.better_candidate(
            expressions,
            &mut accept,
            other_value,
            other_id,
            current_value,
            current_id,
        );
        let mut better_accept = Block::new();
        self.store_storage(
            expressions,
            &mut better_accept,
            globals.scratch_values,
            lane,
            other_value,
        );
        self.store_storage(
            expressions,
            &mut better_accept,
            globals.scratch_ids,
            lane,
            other_id,
        );
        self.store_storage(
            expressions,
            &mut better_accept,
            globals.scratch_chunks,
            lane,
            other_chunk,
        );
        accept.push(
            Statement::If {
                condition: better,
                accept: better_accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        reduce_body.push(
            Statement::If {
                condition: participates,
                accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        reduce_body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        let step = self.load_local(expressions, &mut reduce_body, locals.reduce_step);
        let two = self.u32_lit(expressions, 2);
        let next_step = self.bin(
            expressions,
            &mut reduce_body,
            BinaryOperator::Divide,
            step,
            two,
        );
        self.store_local(expressions, &mut reduce_body, locals.reduce_step, next_step);
        body.push(
            Statement::Loop {
                body: reduce_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }

    fn store_rank_output(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: &MergeTopKGlobals,
        locals: &MergeTopKLocals,
        lane: Handle<Expression>,
    ) {
        let lane_zero = self.eq_lit(expressions, body, lane, 0);
        let mut accept = Block::new();
        let zero = self.u32_lit(expressions, 0);
        let selected_value =
            self.load_storage(expressions, &mut accept, globals.scratch_values, zero);
        let zero = self.u32_lit(expressions, 0);
        let selected_id = self.load_storage(expressions, &mut accept, globals.scratch_ids, zero);
        let zero = self.u32_lit(expressions, 0);
        let selected_chunk =
            self.load_storage(expressions, &mut accept, globals.scratch_chunks, zero);
        let rank = self.load_local(expressions, &mut accept, locals.rank);
        self.store_storage(
            expressions,
            &mut accept,
            globals.output_values,
            rank,
            selected_value,
        );
        self.store_storage(
            expressions,
            &mut accept,
            globals.output_ids,
            rank,
            selected_id,
        );

        let chunks = self.u32_lit(expressions, self.chunks);
        let valid_chunk = self.bin(
            expressions,
            &mut accept,
            BinaryOperator::Less,
            selected_chunk,
            chunks,
        );
        let mut advance = Block::new();
        let position = self.load_storage(
            expressions,
            &mut advance,
            globals.chunk_positions,
            selected_chunk,
        );
        let next_position = self.add_lit(expressions, &mut advance, position, 1);
        self.store_storage(
            expressions,
            &mut advance,
            globals.chunk_positions,
            selected_chunk,
            next_position,
        );
        accept.push(
            Statement::If {
                condition: valid_chunk,
                accept: advance,
                reject: Block::new(),
            },
            Span::default(),
        );
        body.push(
            Statement::If {
                condition: lane_zero,
                accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
    }

    fn storage_global(
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

    fn workgroup_global(
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

    fn local(function: &mut Function, name: &str, ty: Handle<Type>) -> Handle<LocalVariable> {
        function.local_variables.append(
            LocalVariable {
                name: Some(name.into()),
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    fn is_finite(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
    ) -> Handle<Expression> {
        let self_equal = self.bin(expressions, body, BinaryOperator::Equal, value, value);
        let abs = self.emit(
            expressions,
            body,
            Expression::Math {
                fun: MathFunction::Abs,
                arg: value,
                arg1: None,
                arg2: None,
                arg3: None,
            },
        );
        let max = self.f32_lit(expressions, MAX_F32);
        let finite_magnitude = self.bin(expressions, body, BinaryOperator::LessEqual, abs, max);
        self.and(expressions, body, self_equal, finite_magnitude)
    }

    fn better_candidate(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        id: Handle<Expression>,
        best_value: Handle<Expression>,
        best_id: Handle<Expression>,
    ) -> Handle<Expression> {
        let value_greater = self.bin(
            expressions,
            body,
            BinaryOperator::Greater,
            value,
            best_value,
        );
        let value_equal = self.bin(expressions, body, BinaryOperator::Equal, value, best_value);
        let id_greater = self.bin(expressions, body, BinaryOperator::Greater, id, best_id);
        let equal_and_id = self.and(expressions, body, value_equal, id_greater);
        self.or(expressions, body, value_greater, equal_and_id)
    }

    fn load_storage(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let ptr = self.storage_ptr(expressions, body, global, index);
        self.emit(expressions, body, Expression::Load { pointer: ptr })
    }

    fn store_storage(
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

    fn storage_ptr(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let base = expressions.append(Expression::GlobalVariable(global), Span::default());
        self.emit(expressions, body, Expression::Access { base, index })
    }

    fn load_local(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        local: Handle<LocalVariable>,
    ) -> Handle<Expression> {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        self.emit(expressions, body, Expression::Load { pointer })
    }

    fn store_local(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        local: Handle<LocalVariable>,
        value: Handle<Expression>,
    ) {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    fn add_lit(
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

    fn mul_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Multiply, value, rhs)
    }

    fn ge_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::GreaterEqual, value, rhs)
    }

    fn eq_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Equal, value, rhs)
    }

    fn and(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(expressions, body, BinaryOperator::LogicalAnd, left, right)
    }

    fn or(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(expressions, body, BinaryOperator::LogicalOr, left, right)
    }

    fn bin(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(expressions, body, Expression::Binary { op, left, right })
    }

    fn emit(
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

    fn f32_lit(&self, expressions: &mut Arena<Expression>, value: f32) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::F32(value)), Span::default())
    }

    fn u32_lit(&self, expressions: &mut Arena<Expression>, value: u32) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::U32(value)), Span::default())
    }
}

impl TopKModuleBuilder {
    fn new(input_len: u32, output_per_chunk: u32, input_offset: u32, input_stride: u32) -> Self {
        Self {
            input_len,
            output_per_chunk,
            input_offset,
            input_stride,
        }
    }

    fn build(self) -> Option<Module> {
        let mut module = Module::default();
        let f32_ty = module.types.insert(
            Type {
                name: Some("TopKF32".into()),
                inner: TypeInner::Scalar(Scalar::F32),
            },
            Span::default(),
        );
        let u32_ty = module.types.insert(
            Type {
                name: Some("TopKU32".into()),
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::default(),
        );
        let u32_vec3_ty = module.types.insert(
            Type {
                name: Some("TopKWorkgroupId".into()),
                inner: TypeInner::Vector {
                    size: VectorSize::Tri,
                    scalar: Scalar::U32,
                },
            },
            Span::default(),
        );
        let f32_storage_ty = module.types.insert(
            Type {
                name: Some("TopKF32Buffer".into()),
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
                name: Some("TopKU32Buffer".into()),
                inner: TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Dynamic,
                    stride: 4,
                },
            },
            Span::default(),
        );
        let scratch_f32_ty = module.types.insert(
            Type {
                name: Some("TopKScratchF32".into()),
                inner: TypeInner::Array {
                    base: f32_ty,
                    size: ArraySize::Constant(NonZeroU32::new(TOP_K_BLOCK)?),
                    stride: 4,
                },
            },
            Span::default(),
        );
        let scratch_u32_ty = module.types.insert(
            Type {
                name: Some("TopKScratchU32".into()),
                inner: TypeInner::Array {
                    base: u32_ty,
                    size: ArraySize::Constant(NonZeroU32::new(TOP_K_BLOCK)?),
                    stride: 4,
                },
            },
            Span::default(),
        );

        let globals = TopKGlobals {
            input: Self::storage_global(&mut module, "input", 0, f32_storage_ty, true),
            output_ids: Self::storage_global(&mut module, "output_ids", 1, u32_storage_ty, false),
            output_values: Self::storage_global(
                &mut module,
                "output_values",
                2,
                f32_storage_ty,
                false,
            ),
            scratch_values: Self::workgroup_global(&mut module, "scratch_values", scratch_f32_ty),
            scratch_ids: Self::workgroup_global(&mut module, "scratch_ids", scratch_u32_ty),
        };

        let mut function = Function {
            name: Some("main".into()),
            arguments: vec![
                FunctionArgument {
                    name: Some("local_invocation_index".into()),
                    ty: u32_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)),
                },
                FunctionArgument {
                    name: Some("workgroup_id".into()),
                    ty: u32_vec3_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::WorkGroupId)),
                },
            ],
            ..Function::default()
        };
        let locals = TopKLocals {
            current_value: Self::local(&mut function, "current_value", f32_ty),
            current_id: Self::local(&mut function, "current_id", u32_ty),
        };

        function.body = self.entry_body(&mut function.expressions, globals, locals);
        function
            .body
            .push(Statement::Return { value: None }, Span::default());
        module.entry_points.push(EntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [TOP_K_BLOCK, 1, 1],
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
        globals: TopKGlobals,
        locals: TopKLocals,
    ) -> Block {
        let mut body = Block::new();
        let lane = expressions.append(Expression::FunctionArgument(0), Span::default());
        let workgroup_id = expressions.append(Expression::FunctionArgument(1), Span::default());
        let chunk = self.emit(
            expressions,
            &mut body,
            Expression::AccessIndex {
                base: workgroup_id,
                index: 0,
            },
        );
        let neg_max = self.f32_lit(expressions, NEG_MAX_F32);
        let invalid_id = self.u32_lit(expressions, u32::MAX);
        self.store_local(expressions, &mut body, locals.current_value, neg_max);
        self.store_local(expressions, &mut body, locals.current_id, invalid_id);

        let chunk_base = self.mul_lit(expressions, &mut body, chunk, TOP_K_CHUNK as u32);
        let token_id = self.bin(
            expressions,
            &mut body,
            BinaryOperator::Add,
            chunk_base,
            lane,
        );
        let input_len = self.u32_lit(expressions, self.input_len);
        let token_valid = self.bin(
            expressions,
            &mut body,
            BinaryOperator::Less,
            token_id,
            input_len,
        );
        let mut load_accept = Block::new();
        let input_index = if self.input_stride == 1 {
            self.add_lit(expressions, &mut load_accept, token_id, self.input_offset)
        } else {
            let scaled = self.mul_lit(expressions, &mut load_accept, token_id, self.input_stride);
            self.add_lit(expressions, &mut load_accept, scaled, self.input_offset)
        };
        let value = self.load_storage(expressions, &mut load_accept, globals.input, input_index);
        let finite = self.is_finite(expressions, &mut load_accept, value);
        let mut finite_accept = Block::new();
        self.store_local(expressions, &mut finite_accept, locals.current_value, value);
        self.store_local(expressions, &mut finite_accept, locals.current_id, token_id);
        load_accept.push(
            Statement::If {
                condition: finite,
                accept: finite_accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        body.push(
            Statement::If {
                condition: token_valid,
                accept: load_accept,
                reject: Block::new(),
            },
            Span::default(),
        );

        let current_value = self.load_local(expressions, &mut body, locals.current_value);
        let current_id = self.load_local(expressions, &mut body, locals.current_id);
        self.store_storage(
            expressions,
            &mut body,
            globals.scratch_values,
            lane,
            current_value,
        );
        self.store_storage(
            expressions,
            &mut body,
            globals.scratch_ids,
            lane,
            current_id,
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let mut size = 2;
        while size <= TOP_K_BLOCK {
            let mut stride = size / 2;
            while stride > 0 {
                self.append_bitonic_stage(expressions, &mut body, &globals, lane, size, stride);
                stride /= 2;
            }
            size *= 2;
        }

        let output_per_chunk = self.u32_lit(expressions, self.output_per_chunk);
        let writes_output = self.bin(
            expressions,
            &mut body,
            BinaryOperator::Less,
            lane,
            output_per_chunk,
        );
        let mut write_accept = Block::new();
        let chunk_base = self.mul_lit(expressions, &mut write_accept, chunk, self.output_per_chunk);
        let output_index = self.bin(
            expressions,
            &mut write_accept,
            BinaryOperator::Add,
            chunk_base,
            lane,
        );
        let selected_value =
            self.load_storage(expressions, &mut write_accept, globals.scratch_values, lane);
        let selected_id =
            self.load_storage(expressions, &mut write_accept, globals.scratch_ids, lane);
        self.store_storage(
            expressions,
            &mut write_accept,
            globals.output_values,
            output_index,
            selected_value,
        );
        self.store_storage(
            expressions,
            &mut write_accept,
            globals.output_ids,
            output_index,
            selected_id,
        );
        body.push(
            Statement::If {
                condition: writes_output,
                accept: write_accept,
                reject: Block::new(),
            },
            Span::default(),
        );

        body
    }

    fn append_bitonic_stage(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: &TopKGlobals,
        lane: Handle<Expression>,
        size: u32,
        stride: u32,
    ) {
        let stride_lit = self.u32_lit(expressions, stride);
        let partner = self.bin(
            expressions,
            body,
            BinaryOperator::ExclusiveOr,
            lane,
            stride_lit,
        );
        let current_value = self.load_storage(expressions, body, globals.scratch_values, lane);
        let current_id = self.load_storage(expressions, body, globals.scratch_ids, lane);
        let partner_value = self.load_storage(expressions, body, globals.scratch_values, partner);
        let partner_id = self.load_storage(expressions, body, globals.scratch_ids, partner);

        let stride_lit = self.u32_lit(expressions, stride);
        let lane_stride_bits = self.bin(expressions, body, BinaryOperator::And, lane, stride_lit);
        let size_lit = self.u32_lit(expressions, size);
        let lane_size_bits = self.bin(expressions, body, BinaryOperator::And, lane, size_lit);
        let zero = self.u32_lit(expressions, 0);
        let lower_lane = self.bin(
            expressions,
            body,
            BinaryOperator::Equal,
            lane_stride_bits,
            zero,
        );
        let descending = self.bin(
            expressions,
            body,
            BinaryOperator::Equal,
            lane_size_bits,
            zero,
        );
        let want_better = self.bin(
            expressions,
            body,
            BinaryOperator::Equal,
            lower_lane,
            descending,
        );

        let partner_better = self.better_candidate(
            expressions,
            body,
            partner_value,
            partner_id,
            current_value,
            current_id,
        );
        let current_better = self.better_candidate(
            expressions,
            body,
            current_value,
            current_id,
            partner_value,
            partner_id,
        );
        let false_lit = self.bool_lit(expressions, false);
        let want_worse = self.bin(
            expressions,
            body,
            BinaryOperator::Equal,
            want_better,
            false_lit,
        );
        let choose_better_partner = self.and(expressions, body, want_better, partner_better);
        let choose_worse_partner = self.and(expressions, body, want_worse, current_better);
        let choose_partner = self.or(
            expressions,
            body,
            choose_better_partner,
            choose_worse_partner,
        );

        let mut accept = Block::new();
        self.store_storage(
            expressions,
            &mut accept,
            globals.scratch_values,
            lane,
            partner_value,
        );
        self.store_storage(
            expressions,
            &mut accept,
            globals.scratch_ids,
            lane,
            partner_id,
        );
        body.push(
            Statement::If {
                condition: choose_partner,
                accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
    }

    fn storage_global(
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

    fn workgroup_global(
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

    fn local(function: &mut Function, name: &str, ty: Handle<Type>) -> Handle<LocalVariable> {
        function.local_variables.append(
            LocalVariable {
                name: Some(name.into()),
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    fn is_finite(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
    ) -> Handle<Expression> {
        let self_equal = self.bin(expressions, body, BinaryOperator::Equal, value, value);
        let abs = self.emit(
            expressions,
            body,
            Expression::Math {
                fun: MathFunction::Abs,
                arg: value,
                arg1: None,
                arg2: None,
                arg3: None,
            },
        );
        let max = self.f32_lit(expressions, MAX_F32);
        let finite_magnitude = self.bin(expressions, body, BinaryOperator::LessEqual, abs, max);
        self.and(expressions, body, self_equal, finite_magnitude)
    }

    fn better_candidate(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        id: Handle<Expression>,
        best_value: Handle<Expression>,
        best_id: Handle<Expression>,
    ) -> Handle<Expression> {
        let value_greater = self.bin(
            expressions,
            body,
            BinaryOperator::Greater,
            value,
            best_value,
        );
        let value_equal = self.bin(expressions, body, BinaryOperator::Equal, value, best_value);
        let id_greater = self.bin(expressions, body, BinaryOperator::Greater, id, best_id);
        let equal_and_id = self.and(expressions, body, value_equal, id_greater);
        self.or(expressions, body, value_greater, equal_and_id)
    }

    fn load_storage(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let ptr = self.storage_ptr(expressions, body, global, index);
        self.emit(expressions, body, Expression::Load { pointer: ptr })
    }

    fn store_storage(
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

    fn storage_ptr(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let base = expressions.append(Expression::GlobalVariable(global), Span::default());
        self.emit(expressions, body, Expression::Access { base, index })
    }

    fn load_local(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        local: Handle<LocalVariable>,
    ) -> Handle<Expression> {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        self.emit(expressions, body, Expression::Load { pointer })
    }

    fn store_local(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        local: Handle<LocalVariable>,
        value: Handle<Expression>,
    ) {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    fn add_lit(
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

    fn mul_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Multiply, value, rhs)
    }

    fn and(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(expressions, body, BinaryOperator::LogicalAnd, left, right)
    }

    fn or(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.bin(expressions, body, BinaryOperator::LogicalOr, left, right)
    }

    fn bin(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(expressions, body, Expression::Binary { op, left, right })
    }

    fn emit(
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

    fn f32_lit(&self, expressions: &mut Arena<Expression>, value: f32) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::F32(value)), Span::default())
    }

    fn u32_lit(&self, expressions: &mut Arena<Expression>, value: u32) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::U32(value)), Span::default())
    }

    fn bool_lit(&self, expressions: &mut Arena<Expression>, value: bool) -> Handle<Expression> {
        expressions.append(Expression::Literal(Literal::Bool(value)), Span::default())
    }
}

#[cfg(test)]
mod tests {
    use crate::{Device, Tensor};

    #[tokio::test]
    async fn top_k_pairs_match_cpu_sorted_order() {
        let device = Device::new().await.unwrap();
        let values = [
            0.25,
            f32::NAN,
            7.0,
            -3.0,
            f32::INFINITY,
            2.5,
            9.0,
            f32::NEG_INFINITY,
            8.5,
            9.0,
            6.0,
            -1.0,
        ];
        let tensor = Tensor::new(&device, values.as_slice());
        let (ids, logits) = tensor.top_k_pairs(5).await.unwrap();

        let mut expected = values
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| right.0.cmp(&left.0))
        });
        expected.truncate(5);

        let actual = ids
            .into_iter()
            .zip(logits)
            .map(|(id, value)| (id as usize, value))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn top_k_pairs_merge_path_match_cpu_sorted_order() {
        let device = Device::new().await.unwrap();
        let values = (0..4096)
            .map(|index| {
                if index % 997 == 0 {
                    f32::NAN
                } else if index % 991 == 0 {
                    f32::INFINITY
                } else if index % 983 == 0 {
                    f32::NEG_INFINITY
                } else {
                    let coarse = ((index * 37) % 251) as f32;
                    let tied = (index % 17) as f32 * 0.001;
                    coarse - tied
                }
            })
            .collect::<Vec<_>>();
        let tensor = Tensor::new(&device, values.as_slice());
        let (ids, logits) = tensor.top_k_pairs(16).await.unwrap();

        let mut expected = values
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| right.0.cmp(&left.0))
        });
        expected.truncate(16);

        let actual = ids
            .into_iter()
            .zip(logits)
            .map(|(id, value)| (id as usize, value))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[tokio::test]
    async fn top_k_pairs_large_vocab_merge_path_matches_cpu_sorted_order() {
        let device = Device::new().await.unwrap();
        let values = (0..128_256)
            .map(|index| {
                if index % 65_521 == 0 {
                    f32::NAN
                } else if index % 32_749 == 0 {
                    f32::INFINITY
                } else if index % 32_719 == 0 {
                    f32::NEG_INFINITY
                } else {
                    let coarse = ((index * 97) % 4093) as f32;
                    let tied = (index % 31) as f32 * 0.0001;
                    coarse - tied
                }
            })
            .collect::<Vec<_>>();
        let tensor = Tensor::new(&device, values.as_slice());
        let (ids, logits) = tensor.top_k_pairs(512).await.unwrap();

        let mut expected = values
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .collect::<Vec<_>>();
        expected.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| right.0.cmp(&left.0))
        });
        expected.truncate(512);

        let actual = ids
            .into_iter()
            .zip(logits)
            .map(|(id, value)| (id as usize, value))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }
}
