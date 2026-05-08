use std::num::NonZeroU32;

use crate::{
    Device,
    mir::{direct_kernel::DirectKernelBinding, kernel_backend},
    sampling::{
        GPU_SAMPLE_RESULT_WORDS, GPU_SAMPLE_STATUS_INVALID, GPU_SAMPLE_STATUS_RETRY_NEEDED,
        GPU_SAMPLE_STATUS_SAMPLED, GpuMirostat2Sampler, GpuMirostat2SamplerParams, TOP_K_BLOCK,
    },
    tensor::{DataTypeEnum, TensorData},
};
use wgpu::{
    CommandEncoder,
    naga::{
        Arena, ArraySize, Barrier, BinaryOperator, Binding, Block, BuiltIn, EntryPoint, Expression,
        Function, FunctionArgument, GlobalVariable, Handle, LocalVariable, Module, Scalar,
        ShaderStage, Span, Statement, Type, TypeInner,
    },
};

#[path = "mirostat_helpers.rs"]
mod helpers;

use crate::mir::kernel_backend::naga_helpers::{
    NagaBuilderExt, local, storage_global, workgroup_global,
};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Mirostat2Params {
    tau: f32,
    eta: f32,
    random: f32,
    _padding: f32,
}

fn mirostat2_params_data(device: &Device, params: GpuMirostat2SamplerParams) -> TensorData {
    let params = Mirostat2Params {
        tau: params.tau,
        eta: params.eta,
        random: params.random.clamp(0.0, 0.999_999_94),
        _padding: 0.0,
    };
    let buffer = device.create_buffer_init(
        bytemuck::bytes_of(&params),
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    );
    TensorData::new_from_buffer(device, buffer, &[1], DataTypeEnum::U32)
}

pub(crate) fn sample_from_sorted_top_k_data_with_encoder(
    ids: &TensorData,
    values: &TensorData,
    sampler: &mut GpuMirostat2Sampler,
    params: GpuMirostat2SamplerParams,
    exactness_flag: Option<&TensorData>,
    encoder: Option<&mut CommandEncoder>,
) -> Option<TensorData> {
    if ids.datatype() != DataTypeEnum::U32 || values.datatype() != DataTypeEnum::F32 {
        return None;
    }
    if ids.layout().rank() != 1 || values.layout().rank() != 1 {
        return None;
    }
    if let Some(flag) = exactness_flag
        && (flag.datatype() != DataTypeEnum::U32
            || flag.layout().rank() != 1
            || flag.layout().shape()[0] == 0
            || !values.device().is_same_device(flag.device()))
    {
        return None;
    }

    let top_k = params
        .top_k
        .min(ids.layout().shape()[0])
        .min(values.layout().shape()[0]);
    if top_k == 0 {
        return None;
    }
    let device = values.device();
    let params = mirostat2_params_data(device, params);
    let has_exactness_flag = exactness_flag.is_some();
    let output = TensorData::new_for_shape(device, &[GPU_SAMPLE_RESULT_WORDS], DataTypeEnum::U32);
    let meta = SampleMirostat2Meta {
        top_k: top_k.try_into().ok()?,
        ids_offset: ids.layout().offset().try_into().ok()?,
        ids_stride: ids.layout().strides()[0].try_into().ok()?,
        values_offset: values.layout().offset().try_into().ok()?,
        values_stride: values.layout().strides()[0].try_into().ok()?,
        has_exactness_flag,
    };
    let cache_key = format!(
        "sample_mirostat2_sorted_top_k_f32:backend-lowered:block={TOP_K_BLOCK}:top_k={top_k}:ids={:?}:values={:?}:exact={has_exactness_flag}",
        ids.layout(),
        values.layout()
    );
    let mut bindings = vec![
        DirectKernelBinding::Storage {
            binding: 0,
            buffer: ids.buffer().clone(),
            read_only: true,
        },
        DirectKernelBinding::Storage {
            binding: 1,
            buffer: values.buffer().clone(),
            read_only: true,
        },
        DirectKernelBinding::Storage {
            binding: 2,
            buffer: sampler.state.buffer().clone(),
            read_only: false,
        },
        DirectKernelBinding::Storage {
            binding: 3,
            buffer: params.buffer().clone(),
            read_only: true,
        },
        DirectKernelBinding::Storage {
            binding: 4,
            buffer: output.buffer().clone(),
            read_only: false,
        },
    ];
    if let Some(flag) = exactness_flag {
        bindings.push(DirectKernelBinding::Storage {
            binding: 5,
            buffer: flag.buffer().clone(),
            read_only: true,
        });
    }

    let kernel = kernel_backend::dynamic_kernel_from_backend_naga_module(
        device,
        "sample_mirostat2_sorted_top_k_f32",
        cache_key,
        || SampleMirostat2ModuleBuilder::new(meta).build(),
        bindings,
        [1, 1, 1],
    )?;

    if let Some(encoder) = encoder {
        kernel.run(device, encoder);
    } else {
        let mut encoder =
            device
                .wgpu_device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("sample_mirostat2_sorted_top_k_f32 encoder"),
                });
        kernel.run(device, &mut encoder);
        device.wgpu_queue().submit(Some(encoder.finish()));
    }

    Some(output)
}

#[derive(Clone, Copy)]
struct SampleMirostat2Meta {
    top_k: u32,
    ids_offset: u32,
    ids_stride: u32,
    values_offset: u32,
    values_stride: u32,
    has_exactness_flag: bool,
}

struct SampleMirostat2ModuleBuilder {
    meta: SampleMirostat2Meta,
}

#[derive(Clone, Copy)]
struct SampleMirostat2Globals {
    ids: Handle<GlobalVariable>,
    values: Handle<GlobalVariable>,
    state: Handle<GlobalVariable>,
    params: Handle<GlobalVariable>,
    output: Handle<GlobalVariable>,
    exactness_flag: Option<Handle<GlobalVariable>>,
    scratch: Handle<GlobalVariable>,
}

struct SampleMirostat2Locals {
    index: Handle<LocalVariable>,
    local_sum: Handle<LocalVariable>,
    reduce_step: Handle<LocalVariable>,
    cutoff: Handle<LocalVariable>,
    scan: Handle<LocalVariable>,
    cutoff_sum: Handle<LocalVariable>,
    cumulative: Handle<LocalVariable>,
    selected: Handle<LocalVariable>,
    selected_probability: Handle<LocalVariable>,
}

impl SampleMirostat2ModuleBuilder {
    fn new(meta: SampleMirostat2Meta) -> Self {
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
        let f32_storage_ty = module.types.insert(
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
                    size: ArraySize::Constant(NonZeroU32::new(TOP_K_BLOCK)?),
                    stride: 4,
                },
            },
            Span::default(),
        );

        let globals = SampleMirostat2Globals {
            ids: storage_global(&mut module, 0, u32_storage_ty, true),
            values: storage_global(&mut module, 1, f32_storage_ty, true),
            state: storage_global(&mut module, 2, f32_storage_ty, false),
            params: storage_global(&mut module, 3, f32_storage_ty, true),
            output: storage_global(&mut module, 4, u32_storage_ty, false),
            exactness_flag: self
                .meta
                .has_exactness_flag
                .then(|| storage_global(&mut module, 5, u32_storage_ty, true)),
            scratch: workgroup_global(&mut module, scratch_ty),
        };

        let mut function = Function {
            name: None,
            arguments: vec![FunctionArgument {
                name: None,
                ty: u32_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)),
            }],
            ..Function::default()
        };
        let locals = SampleMirostat2Locals {
            index: local(&mut function, u32_ty),
            local_sum: local(&mut function, f32_ty),
            reduce_step: local(&mut function, u32_ty),
            cutoff: local(&mut function, u32_ty),
            scan: local(&mut function, u32_ty),
            cutoff_sum: local(&mut function, f32_ty),
            cumulative: local(&mut function, f32_ty),
            selected: local(&mut function, u32_ty),
            selected_probability: local(&mut function, f32_ty),
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
        globals: SampleMirostat2Globals,
        locals: SampleMirostat2Locals,
    ) -> Block {
        let mut body = Block::new();
        let lane = expressions.append(Expression::FunctionArgument(0), Span::default());

        self.append_exactness_retry_guard(expressions, &mut body, globals, lane);
        self.append_invalid_top_guard(expressions, &mut body, globals, lane);

        let zero = self.u32_lit(expressions, 0);
        let max_value = self.top_value(expressions, &mut body, globals, zero);
        let zero_f32 = self.f32_lit(expressions, 0.0);
        self.store_local(expressions, &mut body, locals.local_sum, zero_f32);
        self.store_local(expressions, &mut body, locals.index, lane);
        self.append_sum_loop(expressions, &mut body, globals, &locals, max_value);

        let local_sum = self.load_local(expressions, &mut body, locals.local_sum);
        self.store_storage(expressions, &mut body, globals.scratch, lane, local_sum);
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let half_block = self.u32_lit(expressions, TOP_K_BLOCK / 2);
        self.store_local(expressions, &mut body, locals.reduce_step, half_block);
        self.append_reduce_loop(expressions, &mut body, globals, &locals, lane);

        let zero_u32 = self.u32_lit(expressions, 0);
        let lane_zero = self.bin(
            expressions,
            &mut body,
            BinaryOperator::Equal,
            lane,
            zero_u32,
        );
        let mut lane_zero_body = Block::new();
        self.append_lane_zero_sample(
            expressions,
            &mut lane_zero_body,
            globals,
            &locals,
            max_value,
        );
        body.push(
            Statement::If {
                condition: lane_zero,
                accept: lane_zero_body,
                reject: Block::from_vec(vec![Statement::Return { value: None }]),
            },
            Span::default(),
        );

        body
    }

    fn append_exactness_retry_guard(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: SampleMirostat2Globals,
        lane: Handle<Expression>,
    ) {
        let Some(exactness_flag) = globals.exactness_flag else {
            return;
        };
        let zero = self.u32_lit(expressions, 0);
        let flag = self.load_storage(expressions, body, exactness_flag, zero);
        let retry = self.bin(expressions, body, BinaryOperator::Equal, flag, zero);
        let mut retry_body = Block::new();
        let lane_zero = self.bin(
            expressions,
            &mut retry_body,
            BinaryOperator::Equal,
            lane,
            zero,
        );
        let mut store_body = Block::new();
        self.store_sample_result(
            expressions,
            &mut store_body,
            globals.output,
            GPU_SAMPLE_STATUS_RETRY_NEEDED,
            0,
        );
        retry_body.push(
            Statement::If {
                condition: lane_zero,
                accept: store_body,
                reject: Block::new(),
            },
            Span::default(),
        );
        retry_body.push(Statement::Return { value: None }, Span::default());
        body.push(
            Statement::If {
                condition: retry,
                accept: retry_body,
                reject: Block::new(),
            },
            Span::default(),
        );
    }

    fn append_invalid_top_guard(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: SampleMirostat2Globals,
        lane: Handle<Expression>,
    ) {
        let zero = self.u32_lit(expressions, 0);
        let top_id = self.top_id(expressions, body, globals, zero);
        let invalid_id = self.u32_lit(expressions, u32::MAX);
        let invalid = self.bin(expressions, body, BinaryOperator::Equal, top_id, invalid_id);
        let mut invalid_body = Block::new();
        let lane_zero = self.bin(
            expressions,
            &mut invalid_body,
            BinaryOperator::Equal,
            lane,
            zero,
        );
        let mut store_body = Block::new();
        self.store_sample_result(
            expressions,
            &mut store_body,
            globals.output,
            GPU_SAMPLE_STATUS_INVALID,
            0,
        );
        invalid_body.push(
            Statement::If {
                condition: lane_zero,
                accept: store_body,
                reject: Block::new(),
            },
            Span::default(),
        );
        invalid_body.push(Statement::Return { value: None }, Span::default());
        body.push(
            Statement::If {
                condition: invalid,
                accept: invalid_body,
                reject: Block::new(),
            },
            Span::default(),
        );
    }

    fn append_sum_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: SampleMirostat2Globals,
        locals: &SampleMirostat2Locals,
        max_value: Handle<Expression>,
    ) {
        let mut loop_body = Block::new();
        let index = self.load_local(expressions, &mut loop_body, locals.index);
        let done = self.ge_lit(expressions, &mut loop_body, index, self.meta.top_k);
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );
        let value = self.top_value(expressions, &mut loop_body, globals, index);
        let delta = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Subtract,
            value,
            max_value,
        );
        let weight = self.exp_f32(expressions, &mut loop_body, delta);
        let current = self.load_local(expressions, &mut loop_body, locals.local_sum);
        let next_sum = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Add,
            current,
            weight,
        );
        self.store_local(expressions, &mut loop_body, locals.local_sum, next_sum);
        let index = self.load_local(expressions, &mut loop_body, locals.index);
        let next_index = self.add_lit(expressions, &mut loop_body, index, TOP_K_BLOCK);
        self.store_local(expressions, &mut loop_body, locals.index, next_index);
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }

    fn append_reduce_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: SampleMirostat2Globals,
        locals: &SampleMirostat2Locals,
        lane: Handle<Expression>,
    ) {
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
        let rhs_index = self.bin(expressions, &mut accept, BinaryOperator::Add, lane, step);
        let lhs = self.load_storage(expressions, &mut accept, globals.scratch, lane);
        let rhs = self.load_storage(expressions, &mut accept, globals.scratch, rhs_index);
        let sum = self.bin(expressions, &mut accept, BinaryOperator::Add, lhs, rhs);
        self.store_storage(expressions, &mut accept, globals.scratch, lane, sum);
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

    fn append_lane_zero_sample(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: SampleMirostat2Globals,
        locals: &SampleMirostat2Locals,
        max_value: Handle<Expression>,
    ) {
        let zero_u32 = self.u32_lit(expressions, 0);
        let epsilon = self.f32_lit(expressions, 1.0e-20);
        let total = self.load_storage(expressions, body, globals.scratch, zero_u32);
        let total = self.max_f32(expressions, body, total, epsilon);
        let mu = self.load_storage(expressions, body, globals.state, zero_u32);
        self.store_local(expressions, body, locals.cutoff, zero_u32);
        self.store_local(expressions, body, locals.scan, zero_u32);
        self.append_cutoff_loop(expressions, body, globals, locals, max_value, total, mu);

        let zero_f32 = self.f32_lit(expressions, 0.0);
        self.store_local(expressions, body, locals.cutoff_sum, zero_f32);
        self.store_local(expressions, body, locals.scan, zero_u32);
        self.append_cutoff_sum_loop(expressions, body, globals, locals, max_value);
        let cutoff_sum = self.load_local(expressions, body, locals.cutoff_sum);
        let cutoff_sum = self.max_f32(expressions, body, cutoff_sum, epsilon);
        self.store_local(expressions, body, locals.cutoff_sum, cutoff_sum);

        let random = self.load_param_f32(expressions, body, globals.params, 2);
        let threshold = self.bin(
            expressions,
            body,
            BinaryOperator::Multiply,
            random,
            cutoff_sum,
        );
        self.store_local(expressions, body, locals.cumulative, zero_f32);
        let selected = self.top_id(expressions, body, globals, zero_u32);
        self.store_local(expressions, body, locals.selected, selected);
        let selected_weight = self.top_weight(expressions, body, globals, max_value, zero_u32);
        let selected_probability = self.bin(
            expressions,
            body,
            BinaryOperator::Divide,
            selected_weight,
            cutoff_sum,
        );
        self.store_local(
            expressions,
            body,
            locals.selected_probability,
            selected_probability,
        );
        self.store_local(expressions, body, locals.scan, zero_u32);
        self.append_sample_loop(
            expressions,
            body,
            globals,
            locals,
            max_value,
            cutoff_sum,
            threshold,
        );

        let selected_probability = self.load_local(expressions, body, locals.selected_probability);
        let selected_probability = self.max_f32(expressions, body, selected_probability, epsilon);
        let surprise = self.log2_f32(expressions, body, selected_probability);
        let neg_one = self.f32_lit(expressions, -1.0);
        let surprise = self.bin(
            expressions,
            body,
            BinaryOperator::Multiply,
            surprise,
            neg_one,
        );
        let tau = self.load_param_f32(expressions, body, globals.params, 0);
        let eta = self.load_param_f32(expressions, body, globals.params, 1);
        let error = self.bin(expressions, body, BinaryOperator::Subtract, surprise, tau);
        let correction = self.bin(expressions, body, BinaryOperator::Multiply, eta, error);
        let next_mu = self.bin(expressions, body, BinaryOperator::Subtract, mu, correction);
        self.store_storage(expressions, body, globals.state, zero_u32, next_mu);

        let selected = self.load_local(expressions, body, locals.selected);
        self.store_sample_result_handle(
            expressions,
            body,
            globals.output,
            GPU_SAMPLE_STATUS_SAMPLED,
            selected,
        );
    }

    fn append_cutoff_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: SampleMirostat2Globals,
        locals: &SampleMirostat2Locals,
        max_value: Handle<Expression>,
        total: Handle<Expression>,
        mu: Handle<Expression>,
    ) {
        let mut loop_body = Block::new();
        let scan = self.load_local(expressions, &mut loop_body, locals.scan);
        let done = self.ge_lit(expressions, &mut loop_body, scan, self.meta.top_k);
        let mut done_body = Block::new();
        let one = self.u32_lit(expressions, 1);
        self.store_local(expressions, &mut done_body, locals.cutoff, one);
        done_body.push(Statement::Break, Span::default());
        loop_body.push(
            Statement::If {
                condition: done,
                accept: done_body,
                reject: Block::new(),
            },
            Span::default(),
        );

        let scan = self.load_local(expressions, &mut loop_body, locals.scan);
        let weight = self.top_weight(expressions, &mut loop_body, globals, max_value, scan);
        let probability = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Divide,
            weight,
            total,
        );
        let epsilon = self.f32_lit(expressions, 1.0e-20);
        let probability = self.max_f32(expressions, &mut loop_body, probability, epsilon);
        let surprise = self.log2_f32(expressions, &mut loop_body, probability);
        let neg_one = self.f32_lit(expressions, -1.0);
        let surprise = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Multiply,
            surprise,
            neg_one,
        );
        let too_surprising = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Greater,
            surprise,
            mu,
        );
        let mut cutoff_body = Block::new();
        let scan = self.load_local(expressions, &mut cutoff_body, locals.scan);
        let one = self.u32_lit(expressions, 1);
        let scan_gt_one = self.bin(
            expressions,
            &mut cutoff_body,
            BinaryOperator::Greater,
            scan,
            one,
        );
        let mut scan_body = Block::new();
        self.store_local(expressions, &mut scan_body, locals.cutoff, scan);
        let mut one_body = Block::new();
        self.store_local(expressions, &mut one_body, locals.cutoff, one);
        cutoff_body.push(
            Statement::If {
                condition: scan_gt_one,
                accept: scan_body,
                reject: one_body,
            },
            Span::default(),
        );
        cutoff_body.push(Statement::Break, Span::default());
        loop_body.push(
            Statement::If {
                condition: too_surprising,
                accept: cutoff_body,
                reject: Block::new(),
            },
            Span::default(),
        );
        let scan = self.load_local(expressions, &mut loop_body, locals.scan);
        let next_scan = self.add_lit(expressions, &mut loop_body, scan, 1);
        self.store_local(expressions, &mut loop_body, locals.scan, next_scan);
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }

    fn append_cutoff_sum_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: SampleMirostat2Globals,
        locals: &SampleMirostat2Locals,
        max_value: Handle<Expression>,
    ) {
        let mut loop_body = Block::new();
        let scan = self.load_local(expressions, &mut loop_body, locals.scan);
        let cutoff = self.load_local(expressions, &mut loop_body, locals.cutoff);
        let done = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::GreaterEqual,
            scan,
            cutoff,
        );
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );
        let scan = self.load_local(expressions, &mut loop_body, locals.scan);
        let weight = self.top_weight(expressions, &mut loop_body, globals, max_value, scan);
        let current = self.load_local(expressions, &mut loop_body, locals.cutoff_sum);
        let next = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Add,
            current,
            weight,
        );
        self.store_local(expressions, &mut loop_body, locals.cutoff_sum, next);
        let scan = self.load_local(expressions, &mut loop_body, locals.scan);
        let next_scan = self.add_lit(expressions, &mut loop_body, scan, 1);
        self.store_local(expressions, &mut loop_body, locals.scan, next_scan);
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }

    fn append_sample_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: SampleMirostat2Globals,
        locals: &SampleMirostat2Locals,
        max_value: Handle<Expression>,
        cutoff_sum: Handle<Expression>,
        threshold: Handle<Expression>,
    ) {
        let mut loop_body = Block::new();
        let scan = self.load_local(expressions, &mut loop_body, locals.scan);
        let cutoff = self.load_local(expressions, &mut loop_body, locals.cutoff);
        let done = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::GreaterEqual,
            scan,
            cutoff,
        );
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );
        let scan = self.load_local(expressions, &mut loop_body, locals.scan);
        let weight = self.top_weight(expressions, &mut loop_body, globals, max_value, scan);
        let cumulative = self.load_local(expressions, &mut loop_body, locals.cumulative);
        let cumulative = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Add,
            cumulative,
            weight,
        );
        self.store_local(expressions, &mut loop_body, locals.cumulative, cumulative);
        let selected = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::GreaterEqual,
            cumulative,
            threshold,
        );
        let mut selected_body = Block::new();
        let scan = self.load_local(expressions, &mut selected_body, locals.scan);
        let token = self.top_id(expressions, &mut selected_body, globals, scan);
        self.store_local(expressions, &mut selected_body, locals.selected, token);
        let probability = self.bin(
            expressions,
            &mut selected_body,
            BinaryOperator::Divide,
            weight,
            cutoff_sum,
        );
        self.store_local(
            expressions,
            &mut selected_body,
            locals.selected_probability,
            probability,
        );
        selected_body.push(Statement::Break, Span::default());
        loop_body.push(
            Statement::If {
                condition: selected,
                accept: selected_body,
                reject: Block::new(),
            },
            Span::default(),
        );
        let scan = self.load_local(expressions, &mut loop_body, locals.scan);
        let next_scan = self.add_lit(expressions, &mut loop_body, scan, 1);
        self.store_local(expressions, &mut loop_body, locals.scan, next_scan);
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }
}
