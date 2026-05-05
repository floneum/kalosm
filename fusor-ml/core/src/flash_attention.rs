use std::{
    hash::{Hash, Hasher},
    num::{NonZeroU32, NonZeroUsize},
    sync::OnceLock,
};

use crate::{
    DataTypeEnum,
    compute_graph::{ComputeGraphInner, NodeIndex},
    mir::{
        direct_kernel::{DirectKernel, DirectKernelBinding},
        inputs::MirValue,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
    tensor::TensorData,
};
use lru::LruCache;
use parking_lot::RwLock;
use rustc_hash::{FxBuildHasher, FxHasher};
use wgpu::naga::{
    AddressSpace, Arena, ArraySize, Barrier, BinaryOperator, Binding, Block, BuiltIn, EntryPoint,
    Expression, Function, FunctionArgument, GlobalVariable, Handle, Literal, LocalVariable,
    MathFunction, Module, Range, ResourceBinding, Scalar, ShaderStage, Span, Statement,
    StorageAccess, Type, TypeInner, VectorSize,
};

const BLOCK: usize = 256;
const SIMD_WIDTH: usize = 32;
const OUTPUTS_PER_WORKGROUP: usize = BLOCK / SIMD_WIDTH;
const FLOAT_MIN: f32 = -1.0e30;
const FLASH_ATTENTION_MODULE_CACHE_SIZE: usize = 128;

fn flash_attention_module_cache() -> &'static RwLock<LruCache<[u64; 2], Module, FxBuildHasher>> {
    static CACHE: OnceLock<RwLock<LruCache<[u64; 2], Module, FxBuildHasher>>> = OnceLock::new();
    CACHE.get_or_init(|| {
        RwLock::new(LruCache::with_hasher(
            NonZeroUsize::new(FLASH_ATTENTION_MODULE_CACHE_SIZE).unwrap(),
            Default::default(),
        ))
    })
}

fn hash_layout<H: Hasher>(state: &mut H, layout: &crate::Layout) {
    layout.offset().hash(state);
    layout.shape().hash(state);
    layout.strides().hash(state);
}

fn flash_attention_module_key(
    dims: FlashAttentionDims,
    scale_bits: u32,
    dispatch_size: [u32; 3],
    q: &TensorData,
    k: &TensorData,
    v: &TensorData,
    mask: Option<&TensorData>,
    output: &TensorData,
) -> [u64; 2] {
    std::array::from_fn(|salt| {
        let mut hasher = FxHasher::default();
        (salt as u64).hash(&mut hasher);
        dims.batch.hash(&mut hasher);
        dims.num_heads.hash(&mut hasher);
        dims.num_kv_heads.hash(&mut hasher);
        dims.q_seq_len.hash(&mut hasher);
        dims.kv_seq_len.hash(&mut hasher);
        dims.head_dim.hash(&mut hasher);
        scale_bits.hash(&mut hasher);
        dispatch_size.hash(&mut hasher);
        hash_layout(&mut hasher, q.layout());
        hash_layout(&mut hasher, k.layout());
        hash_layout(&mut hasher, v.layout());
        mask.map(|mask| hash_layout(&mut hasher, mask.layout()))
            .hash(&mut hasher);
        hash_layout(&mut hasher, output.layout());
        hasher.finish()
    })
}

#[derive(Clone, Debug)]
pub(crate) struct FlashAttentionOperation {
    pub(crate) q: NodeIndex,
    pub(crate) k: NodeIndex,
    pub(crate) v: NodeIndex,
    pub(crate) mask: Option<NodeIndex>,
    pub(crate) out_shape: Box<[usize]>,
    q_shape: Box<[usize]>,
    k_shape: Box<[usize]>,
    scale: f32,
}

impl FlashAttentionOperation {
    pub(crate) fn new(
        q: NodeIndex,
        k: NodeIndex,
        v: NodeIndex,
        mask: Option<NodeIndex>,
        q_shape: &[usize],
        k_shape: &[usize],
        v_shape: &[usize],
        scale: f32,
    ) -> Self {
        assert_eq!(q_shape.len(), 4, "Q must be rank-4");
        assert_eq!(k_shape.len(), 4, "K must be rank-4");
        assert_eq!(v_shape.len(), 4, "V must be rank-4");
        assert_eq!(q_shape[0], k_shape[0], "Q and K batch dimensions differ");
        assert_eq!(q_shape[0], v_shape[0], "Q and V batch dimensions differ");
        assert_eq!(k_shape[1], v_shape[1], "K and V head dimensions differ");
        assert_eq!(k_shape[2], v_shape[2], "K and V sequence dimensions differ");
        assert_eq!(q_shape[3], k_shape[3], "Q and K head dimensions differ");
        assert_eq!(q_shape[3], v_shape[3], "Q and V head dimensions differ");
        assert!(
            q_shape[1].is_multiple_of(k_shape[1]),
            "Number of Q heads ({}) must be divisible by number of K/V heads ({})",
            q_shape[1],
            k_shape[1]
        );

        Self {
            q,
            k,
            v,
            mask,
            out_shape: q_shape.into(),
            q_shape: q_shape.into(),
            k_shape: k_shape.into(),
            scale,
        }
    }

    fn dims(&self) -> Option<FlashAttentionDims> {
        Some(FlashAttentionDims {
            batch: self.q_shape[0].try_into().ok()?,
            num_heads: self.q_shape[1].try_into().ok()?,
            num_kv_heads: self.k_shape[1].try_into().ok()?,
            q_seq_len: self.q_shape[2].try_into().ok()?,
            kv_seq_len: self.k_shape[2].try_into().ok()?,
            head_dim: self.q_shape[3].try_into().ok()?,
        })
    }
}

impl Operation for FlashAttentionOperation {
    fn workgroup_shape_constraints(&self, _device: &crate::Device) -> WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        constraints.add_constraint(0, Constraint::Equals(1));
        constraints.add_constraint(1, Constraint::Equals(1));
        constraints.add_constraint(2, Constraint::Equals(1));
        constraints
    }

    fn dispatch_size(&self, _workgroup_shape: &WorkgroupShape, _inputs: &[MirValue]) -> [u32; 3] {
        let dims = self.dims().expect("flash attention dimensions fit in u32");
        [
            dims.head_dim.div_ceil(OUTPUTS_PER_WORKGROUP as u32),
            dims.batch
                .checked_mul(dims.num_heads)
                .and_then(|value| value.checked_mul(dims.q_seq_len))
                .expect("flash attention row dispatch overflow"),
            1,
        ]
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.q);
        f(self.k);
        f(self.v);
        if let Some(mask) = self.mask {
            f(mask);
        }
    }

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<MirValue> {
        let q = nodes.get_result(self.q).unwrap();
        let k = nodes.get_result(self.k).unwrap();
        let v = nodes.get_result(self.v).unwrap();
        let output = TensorData::new_for_shape(q.device(), &self.out_shape, DataTypeEnum::F32);

        let mut inputs = vec![q.into(), k.into(), v.into()];
        if let Some(mask) = self.mask {
            inputs.push(nodes.get_result(mask).unwrap().into());
        }
        inputs.push(output.into());
        inputs
    }

    fn build_direct_kernel(
        &self,
        graph: &ComputeGraphInner,
        _workgroup_shape: &WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        let q = inputs.first()?.as_tensor()?.clone();
        let k = inputs.get(1)?.as_tensor()?.clone();
        let v = inputs.get(2)?.as_tensor()?.clone();
        let (mask, output_index) = if self.mask.is_some() {
            (Some(inputs.get(3)?.as_tensor()?.clone()), 4)
        } else {
            (None, 3)
        };
        let output = inputs.get(output_index)?.as_tensor()?.clone();

        if q.datatype() != DataTypeEnum::F32
            || k.datatype() != DataTypeEnum::F32
            || v.datatype() != DataTypeEnum::F32
            || output.datatype() != DataTypeEnum::F32
            || mask
                .as_ref()
                .is_some_and(|mask| mask.datatype() != DataTypeEnum::F32)
        {
            return None;
        }

        let dims = self.dims()?;
        if dims.batch == 0
            || dims.num_heads == 0
            || dims.num_kv_heads == 0
            || dims.q_seq_len == 0
            || dims.kv_seq_len == 0
            || dims.head_dim == 0
        {
            return None;
        }

        let dispatch_size = self.dispatch_size(&WorkgroupShape::new(1, 1, 1), inputs);
        if dispatch_size
            .iter()
            .any(|dim| *dim > graph.device().limits().max_compute_workgroups_per_dimension)
        {
            return None;
        }

        let q_meta = TensorMeta::new(&q)?;
        let k_meta = TensorMeta::new(&k)?;
        let v_meta = TensorMeta::new(&v)?;
        let mask_meta = if let Some(mask) = mask.as_ref() {
            Some(TensorMeta::new(mask)?)
        } else {
            None
        };
        let output_meta = TensorMeta::new(&output)?;

        if q_meta.datatype != DataTypeEnum::F32
            || k_meta.datatype != DataTypeEnum::F32
            || v_meta.datatype != DataTypeEnum::F32
            || output_meta.datatype != DataTypeEnum::F32
            || mask_meta
                .as_ref()
                .is_some_and(|mask| mask.datatype != DataTypeEnum::F32)
        {
            return None;
        }

        let module_key = flash_attention_module_key(
            dims,
            self.scale.to_bits(),
            dispatch_size,
            &q,
            &k,
            &v,
            mask.as_ref(),
            &output,
        );
        let cache_key = format!(
            "flash_attention:{:016x}{:016x}",
            module_key[0], module_key[1]
        );
        let module = if let Some(module) = flash_attention_module_cache().write().get(&module_key) {
            module.clone()
        } else {
            let verbose_cache_key = format!(
                "{}:naga:block={BLOCK}:simd={SIMD_WIDTH}:outputs={OUTPUTS_PER_WORKGROUP}:dispatch={dispatch_size:?}:scale={:?}:q={:?}:k={:?}:v={:?}:mask={:?}:out={:?}",
                self.name(),
                self.scale.to_bits(),
                q.layout(),
                k.layout(),
                v.layout(),
                mask.as_ref().map(|mask| mask.layout()),
                output.layout()
            );
            let module = if let Some(module) = graph
                .device()
                .naga_module_cache()
                .write()
                .get(&verbose_cache_key)
            {
                module.clone()
            } else {
                let module = build_flash_attention_naga_module(
                    dims,
                    self.scale,
                    q_meta,
                    k_meta,
                    v_meta,
                    mask_meta,
                    output_meta,
                    dispatch_size,
                )?;
                graph
                    .device()
                    .naga_module_cache()
                    .write()
                    .get_or_insert(verbose_cache_key, || module.clone())
                    .clone()
            };
            flash_attention_module_cache()
                .write()
                .get_or_insert(module_key, || module.clone())
                .clone()
        };

        let mut bindings = vec![
            DirectKernelBinding::Storage {
                binding: 0,
                buffer: q.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 1,
                buffer: k.buffer().clone(),
                read_only: true,
            },
            DirectKernelBinding::Storage {
                binding: 2,
                buffer: v.buffer().clone(),
                read_only: true,
            },
        ];
        if let Some(mask) = mask {
            bindings.push(DirectKernelBinding::Storage {
                binding: 3,
                buffer: mask.buffer().clone(),
                read_only: true,
            });
        }
        bindings.push(DirectKernelBinding::Storage {
            binding: output_index as u32,
            buffer: output.buffer().clone(),
            read_only: false,
        });

        Some(DirectKernel::new_with_cache_key(
            "flash_attention",
            cache_key,
            module,
            bindings,
            dispatch_size,
        ))
    }

    fn output(&self, _nodes: &ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        inputs.last().unwrap().clone()
    }

    fn name(&self) -> String {
        format!(
            "flash_attention_f32_{}x{}x{}x{}_by_{}x{}",
            self.q_shape[0],
            self.q_shape[1],
            self.q_shape[2],
            self.q_shape[3],
            self.k_shape[1],
            self.k_shape[2],
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct FlashAttentionDims {
    batch: u32,
    num_heads: u32,
    num_kv_heads: u32,
    q_seq_len: u32,
    kv_seq_len: u32,
    head_dim: u32,
}

#[derive(Clone)]
struct TensorMeta {
    datatype: DataTypeEnum,
    strides: Vec<u32>,
    offset: u32,
}

impl TensorMeta {
    fn new(tensor: &TensorData) -> Option<Self> {
        Some(Self {
            datatype: tensor.datatype(),
            strides: tensor
                .layout()
                .strides()
                .iter()
                .copied()
                .map(u32::try_from)
                .collect::<Result<Vec<_>, _>>()
                .ok()?,
            offset: tensor.layout().offset().try_into().ok()?,
        })
    }

    fn stride4(&self) -> Option<[u32; 4]> {
        self.strides.as_slice().try_into().ok()
    }

    fn stride2(&self) -> Option<[u32; 2]> {
        self.strides.as_slice().try_into().ok()
    }
}

fn build_flash_attention_naga_module(
    dims: FlashAttentionDims,
    scale: f32,
    q_meta: TensorMeta,
    k_meta: TensorMeta,
    v_meta: TensorMeta,
    mask_meta: Option<TensorMeta>,
    output_meta: TensorMeta,
    _dispatch_size: [u32; 3],
) -> Option<Module> {
    let q_strides = q_meta.stride4()?;
    let k_strides = k_meta.stride4()?;
    let v_strides = v_meta.stride4()?;
    let output_strides = output_meta.stride4()?;
    let mask_strides = if let Some(mask_meta) = mask_meta.as_ref() {
        Some(mask_meta.stride2()?)
    } else {
        None
    };
    let groups = dims.num_heads.checked_div(dims.num_kv_heads)?;
    if groups == 0 {
        return None;
    }

    let meta = FlashAttentionNagaMeta {
        dims,
        scale,
        groups,
        q_offset: q_meta.offset,
        k_offset: k_meta.offset,
        v_offset: v_meta.offset,
        mask_offset: mask_meta.as_ref().map(|mask| mask.offset),
        output_offset: output_meta.offset,
        q_strides,
        k_strides,
        v_strides,
        mask_strides,
        output_strides,
    };

    FlashAttentionNagaBuilder::new(meta, mask_meta.is_some()).build()
}

#[derive(Clone, Copy)]
struct FlashAttentionNagaMeta {
    dims: FlashAttentionDims,
    scale: f32,
    groups: u32,
    q_offset: u32,
    k_offset: u32,
    v_offset: u32,
    mask_offset: Option<u32>,
    output_offset: u32,
    q_strides: [u32; 4],
    k_strides: [u32; 4],
    v_strides: [u32; 4],
    mask_strides: Option<[u32; 2]>,
    output_strides: [u32; 4],
}

struct FlashAttentionNagaBuilder {
    meta: FlashAttentionNagaMeta,
    has_mask: bool,
}

#[derive(Clone, Copy)]
struct FlashAttentionGlobals {
    q: Handle<GlobalVariable>,
    k: Handle<GlobalVariable>,
    v: Handle<GlobalVariable>,
    mask: Option<Handle<GlobalVariable>>,
    output: Handle<GlobalVariable>,
    scratch: Handle<GlobalVariable>,
}

#[derive(Clone, Copy)]
struct FlashAttentionLocals {
    loop_idx: Handle<LocalVariable>,
    score: Handle<LocalVariable>,
    weighted: Handle<LocalVariable>,
    m: Handle<LocalVariable>,
    s: Handle<LocalVariable>,
    o: Handle<LocalVariable>,
}

#[derive(Clone, Copy)]
enum FlashReduceOp {
    Sum,
    Max,
}

impl FlashAttentionNagaBuilder {
    fn new(meta: FlashAttentionNagaMeta, has_mask: bool) -> Self {
        Self { meta, has_mask }
    }

    fn build(self) -> Option<Module> {
        let mut module = Module::default();
        let f32_ty = module.types.insert(
            Type {
                name: Some("FlashAttentionF32".into()),
                inner: TypeInner::Scalar(Scalar::F32),
            },
            Span::default(),
        );
        let u32_ty = module.types.insert(
            Type {
                name: Some("FlashAttentionU32".into()),
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::default(),
        );
        let u32_vec3_ty = module.types.insert(
            Type {
                name: Some("FlashAttentionWorkgroupId".into()),
                inner: TypeInner::Vector {
                    size: VectorSize::Tri,
                    scalar: Scalar::U32,
                },
            },
            Span::default(),
        );
        let storage_ty = module.types.insert(
            Type {
                name: Some("FlashAttentionBuffer".into()),
                inner: TypeInner::Array {
                    base: f32_ty,
                    size: ArraySize::Dynamic,
                    stride: 4,
                },
            },
            Span::default(),
        );
        let scratch_ty = module.types.insert(
            Type {
                name: Some("FlashAttentionScratch".into()),
                inner: TypeInner::Array {
                    base: f32_ty,
                    size: ArraySize::Constant(NonZeroU32::new(BLOCK as u32)?),
                    stride: 4,
                },
            },
            Span::default(),
        );

        let q = Self::storage_global(&mut module, "q", 0, storage_ty, true);
        let k = Self::storage_global(&mut module, "k", 1, storage_ty, true);
        let v = Self::storage_global(&mut module, "v", 2, storage_ty, true);
        let mask = self
            .has_mask
            .then(|| Self::storage_global(&mut module, "mask", 3, storage_ty, true));
        let output_binding = if self.has_mask { 4 } else { 3 };
        let output = Self::storage_global(&mut module, "output", output_binding, storage_ty, false);
        let scratch = module.global_variables.append(
            GlobalVariable {
                name: Some("flash_attention_scratch".into()),
                space: AddressSpace::WorkGroup,
                binding: None,
                ty: scratch_ty,
                init: None,
            },
            Span::default(),
        );
        let globals = FlashAttentionGlobals {
            q,
            k,
            v,
            mask,
            output,
            scratch,
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
        let locals = FlashAttentionLocals {
            loop_idx: Self::local(&mut function, "kv_chunk", u32_ty),
            score: Self::local(&mut function, "score", f32_ty),
            weighted: Self::local(&mut function, "weighted_value", f32_ty),
            m: Self::local(&mut function, "m", f32_ty),
            s: Self::local(&mut function, "s", f32_ty),
            o: Self::local(&mut function, "o", f32_ty),
        };

        function.body = self.entry_body(&mut function.expressions, globals, locals);
        function
            .body
            .push(Statement::Return { value: None }, Span::default());
        module.entry_points.push(EntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [BLOCK as u32, 1, 1],
            workgroup_size_overrides: None,
            function,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });

        Some(module)
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

    fn entry_body(
        &self,
        expressions: &mut Arena<Expression>,
        globals: FlashAttentionGlobals,
        locals: FlashAttentionLocals,
    ) -> Block {
        let mut body = Block::new();
        let lane = expressions.append(Expression::FunctionArgument(0), Span::default());
        let workgroup_id = expressions.append(Expression::FunctionArgument(1), Span::default());
        let workgroup_x = self.emit(
            expressions,
            &mut body,
            Expression::AccessIndex {
                base: workgroup_id,
                index: 0,
            },
        );
        let row = self.emit(
            expressions,
            &mut body,
            Expression::AccessIndex {
                base: workgroup_id,
                index: 1,
            },
        );

        let q_idx = self.rem_lit(expressions, &mut body, row, self.meta.dims.q_seq_len);
        let row_over_q = self.div_lit(expressions, &mut body, row, self.meta.dims.q_seq_len);
        let head_idx = self.rem_lit(expressions, &mut body, row_over_q, self.meta.dims.num_heads);
        let batch_idx = self.div_lit(
            expressions,
            &mut body,
            row,
            self.meta.dims.q_seq_len * self.meta.dims.num_heads,
        );
        let kv_head_idx = self.div_lit(expressions, &mut body, head_idx, self.meta.groups);
        let kv_lane = self.rem_lit(expressions, &mut body, lane, SIMD_WIDTH as u32);
        let out_slot = self.div_lit(expressions, &mut body, lane, SIMD_WIDTH as u32);
        let out_base = self.mul_lit(
            expressions,
            &mut body,
            workgroup_x,
            OUTPUTS_PER_WORKGROUP as u32,
        );
        let out_dim = self.bin(
            expressions,
            &mut body,
            BinaryOperator::Add,
            out_base,
            out_slot,
        );
        let out_valid = self.lt_lit(expressions, &mut body, out_dim, self.meta.dims.head_dim);

        let initial_m = self.f32_lit(expressions, FLOAT_MIN);
        let zero_f32 = self.f32_lit(expressions, 0.0);
        let zero_u32 = self.u32_lit(expressions, 0);
        self.store_local(expressions, &mut body, locals.m, initial_m);
        self.store_local(expressions, &mut body, locals.s, zero_f32);
        self.store_local(expressions, &mut body, locals.o, zero_f32);
        self.store_local(expressions, &mut body, locals.loop_idx, zero_u32);

        self.append_kv_loop(
            expressions,
            &mut body,
            globals,
            locals,
            FlashAttentionIndices {
                lane,
                kv_lane,
                out_dim,
                out_valid,
                batch_idx,
                head_idx,
                kv_head_idx,
                q_idx,
            },
        );

        let kv_lane_zero = self.eq_lit(expressions, &mut body, kv_lane, 0);
        let store_valid = self.bin(
            expressions,
            &mut body,
            BinaryOperator::LogicalAnd,
            kv_lane_zero,
            out_valid,
        );
        let numerator = self.load_local(expressions, &mut body, locals.o);
        let denominator = self.load_local(expressions, &mut body, locals.s);
        let output_value = self.bin(
            expressions,
            &mut body,
            BinaryOperator::Divide,
            numerator,
            denominator,
        );
        let mut accept = Block::new();
        let output_index = self.index4(
            expressions,
            &mut accept,
            self.meta.output_offset,
            self.meta.output_strides,
            batch_idx,
            head_idx,
            q_idx,
            out_dim,
        );
        self.store_storage(
            expressions,
            &mut accept,
            globals.output,
            output_index,
            output_value,
        );
        body.push(
            Statement::If {
                condition: store_valid,
                accept,
                reject: Block::new(),
            },
            Span::default(),
        );

        body
    }

    fn append_kv_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: FlashAttentionGlobals,
        locals: FlashAttentionLocals,
        indices: FlashAttentionIndices,
    ) {
        let kv_chunks = self.meta.dims.kv_seq_len.div_ceil(SIMD_WIDTH as u32);
        let mut loop_body = Block::new();
        let chunk = self.load_local(expressions, &mut loop_body, locals.loop_idx);
        let done = self.ge_lit(expressions, &mut loop_body, chunk, kv_chunks);
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let kv_base = self.mul_lit(expressions, &mut loop_body, chunk, SIMD_WIDTH as u32);
        let kv_idx = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Add,
            kv_base,
            indices.kv_lane,
        );
        let kv_valid = self.lt_lit(
            expressions,
            &mut loop_body,
            kv_idx,
            self.meta.dims.kv_seq_len,
        );
        let invalid_score = self.f32_lit(expressions, FLOAT_MIN);
        self.store_local(expressions, &mut loop_body, locals.score, invalid_score);

        let mut score_accept = Block::new();
        let mut score = self.f32_lit(expressions, 0.0);
        for dim in 0..self.meta.dims.head_dim {
            let q_index = self.index4_const_last(
                expressions,
                &mut score_accept,
                self.meta.q_offset,
                self.meta.q_strides,
                indices.batch_idx,
                indices.head_idx,
                indices.q_idx,
                dim,
            );
            let k_index = self.index4_const_last(
                expressions,
                &mut score_accept,
                self.meta.k_offset,
                self.meta.k_strides,
                indices.batch_idx,
                indices.kv_head_idx,
                kv_idx,
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
        if let (Some(mask), Some(mask_offset), Some(mask_strides)) =
            (globals.mask, self.meta.mask_offset, self.meta.mask_strides)
        {
            let mask_index = self.index2(
                expressions,
                &mut score_accept,
                mask_offset,
                mask_strides,
                indices.q_idx,
                kv_idx,
            );
            let mask_value = self.load_storage(expressions, &mut score_accept, mask, mask_index);
            score = self.bin(
                expressions,
                &mut score_accept,
                BinaryOperator::Add,
                score,
                mask_value,
            );
        }
        self.store_local(expressions, &mut score_accept, locals.score, score);
        loop_body.push(
            Statement::If {
                condition: kv_valid,
                accept: score_accept,
                reject: Block::new(),
            },
            Span::default(),
        );

        let score = self.load_local(expressions, &mut loop_body, locals.score);
        let block_max = self.reduce_group(
            expressions,
            &mut loop_body,
            globals.scratch,
            indices.lane,
            score,
            FlashReduceOp::Max,
        );
        let old_m = self.load_local(expressions, &mut loop_body, locals.m);
        let new_m = self.max_f32(expressions, &mut loop_body, old_m, block_max);
        let shifted_score = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Subtract,
            score,
            new_m,
        );
        let raw_exp = self.exp_f32(expressions, &mut loop_body, shifted_score);
        let zero_exp = self.f32_lit(expressions, 0.0);
        let exp_score = self.select(expressions, &mut loop_body, kv_valid, raw_exp, zero_exp);
        let block_sum = self.reduce_group(
            expressions,
            &mut loop_body,
            globals.scratch,
            indices.lane,
            exp_score,
            FlashReduceOp::Sum,
        );

        let zero_weighted = self.f32_lit(expressions, 0.0);
        self.store_local(expressions, &mut loop_body, locals.weighted, zero_weighted);
        let valid_value = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::LogicalAnd,
            kv_valid,
            indices.out_valid,
        );
        let mut weighted_accept = Block::new();
        let v_index = self.index4(
            expressions,
            &mut weighted_accept,
            self.meta.v_offset,
            self.meta.v_strides,
            indices.batch_idx,
            indices.kv_head_idx,
            kv_idx,
            indices.out_dim,
        );
        let v_value = self.load_storage(expressions, &mut weighted_accept, globals.v, v_index);
        let weighted = self.bin(
            expressions,
            &mut weighted_accept,
            BinaryOperator::Multiply,
            exp_score,
            v_value,
        );
        self.store_local(expressions, &mut weighted_accept, locals.weighted, weighted);
        loop_body.push(
            Statement::If {
                condition: valid_value,
                accept: weighted_accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        let weighted = self.load_local(expressions, &mut loop_body, locals.weighted);
        let block_out = self.reduce_group(
            expressions,
            &mut loop_body,
            globals.scratch,
            indices.lane,
            weighted,
            FlashReduceOp::Sum,
        );

        let m_shift = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Subtract,
            old_m,
            new_m,
        );
        let old_m_scale = self.exp_f32(expressions, &mut loop_body, m_shift);
        let old_s = self.load_local(expressions, &mut loop_body, locals.s);
        let scaled_s = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Multiply,
            old_s,
            old_m_scale,
        );
        let new_s = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Add,
            scaled_s,
            block_sum,
        );
        let old_o = self.load_local(expressions, &mut loop_body, locals.o);
        let scaled_o = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Multiply,
            old_o,
            old_m_scale,
        );
        let new_o = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Add,
            scaled_o,
            block_out,
        );
        self.store_local(expressions, &mut loop_body, locals.m, new_m);
        self.store_local(expressions, &mut loop_body, locals.s, new_s);
        self.store_local(expressions, &mut loop_body, locals.o, new_o);

        let one = self.u32_lit(expressions, 1);
        let next_chunk = self.bin(expressions, &mut loop_body, BinaryOperator::Add, chunk, one);
        self.store_local(expressions, &mut loop_body, locals.loop_idx, next_chunk);
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }

    fn reduce_group(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        scratch: Handle<GlobalVariable>,
        lane: Handle<Expression>,
        value: Handle<Expression>,
        op: FlashReduceOp,
    ) -> Handle<Expression> {
        self.store_storage(expressions, body, scratch, lane, value);
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        let group_offset = self.rem_lit(expressions, body, lane, SIMD_WIDTH as u32);
        let group_base = self.bin(
            expressions,
            body,
            BinaryOperator::Subtract,
            lane,
            group_offset,
        );

        let mut stride = SIMD_WIDTH as u32 / 2;
        while stride > 0 {
            let participates = self.lt_lit(expressions, body, group_offset, stride);
            let mut accept = Block::new();
            let lhs = self.load_storage(expressions, &mut accept, scratch, lane);
            let stride_lit = self.u32_lit(expressions, stride);
            let rhs_index = self.bin(
                expressions,
                &mut accept,
                BinaryOperator::Add,
                lane,
                stride_lit,
            );
            let rhs = self.load_storage(expressions, &mut accept, scratch, rhs_index);
            let reduced = match op {
                FlashReduceOp::Sum => {
                    self.bin(expressions, &mut accept, BinaryOperator::Add, lhs, rhs)
                }
                FlashReduceOp::Max => self.max_f32(expressions, &mut accept, lhs, rhs),
            };
            self.store_storage(expressions, &mut accept, scratch, lane, reduced);
            body.push(
                Statement::If {
                    condition: participates,
                    accept,
                    reject: Block::new(),
                },
                Span::default(),
            );
            body.push(
                Statement::ControlBarrier(Barrier::WORK_GROUP),
                Span::default(),
            );
            stride /= 2;
        }

        let result = self.load_storage(expressions, body, scratch, group_base);
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );
        result
    }

    fn index4_const_last(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        offset: u32,
        strides: [u32; 4],
        i0: Handle<Expression>,
        i1: Handle<Expression>,
        i2: Handle<Expression>,
        i3: u32,
    ) -> Handle<Expression> {
        let base = offset + i3 * strides[3];
        self.index3_with_base(
            expressions,
            body,
            base,
            [strides[0], strides[1], strides[2]],
            i0,
            i1,
            i2,
        )
    }

    fn index4(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        offset: u32,
        strides: [u32; 4],
        i0: Handle<Expression>,
        i1: Handle<Expression>,
        i2: Handle<Expression>,
        i3: Handle<Expression>,
    ) -> Handle<Expression> {
        let index = self.index3_with_base(
            expressions,
            body,
            offset,
            [strides[0], strides[1], strides[2]],
            i0,
            i1,
            i2,
        );
        self.add_scaled_index(expressions, body, index, i3, strides[3])
    }

    fn index2(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        offset: u32,
        strides: [u32; 2],
        i0: Handle<Expression>,
        i1: Handle<Expression>,
    ) -> Handle<Expression> {
        let base = self.u32_lit(expressions, offset);
        let index = self.add_scaled_index(expressions, body, base, i0, strides[0]);
        self.add_scaled_index(expressions, body, index, i1, strides[1])
    }

    fn index3_with_base(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        base: u32,
        strides: [u32; 3],
        i0: Handle<Expression>,
        i1: Handle<Expression>,
        i2: Handle<Expression>,
    ) -> Handle<Expression> {
        let base = self.u32_lit(expressions, base);
        let index = self.add_scaled_index(expressions, body, base, i0, strides[0]);
        let index = self.add_scaled_index(expressions, body, index, i1, strides[1]);
        self.add_scaled_index(expressions, body, index, i2, strides[2])
    }

    fn add_scaled_index(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        index: Handle<Expression>,
        component: Handle<Expression>,
        stride: u32,
    ) -> Handle<Expression> {
        if stride == 0 {
            return index;
        }
        let term = self.mul_lit(expressions, body, component, stride);
        self.bin(expressions, body, BinaryOperator::Add, index, term)
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

    fn exp_f32(
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

    fn max_f32(
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

    fn select(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        condition: Handle<Expression>,
        accept: Handle<Expression>,
        reject: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            expressions,
            body,
            Expression::Select {
                condition,
                accept,
                reject,
            },
        )
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

    fn lt_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Less, value, rhs)
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

    fn div_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Divide, value, rhs)
    }

    fn rem_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Modulo, value, rhs)
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

#[derive(Clone, Copy)]
struct FlashAttentionIndices {
    lane: Handle<Expression>,
    kv_lane: Handle<Expression>,
    out_dim: Handle<Expression>,
    out_valid: Handle<Expression>,
    batch_idx: Handle<Expression>,
    head_idx: Handle<Expression>,
    kv_head_idx: Handle<Expression>,
    q_idx: Handle<Expression>,
}
