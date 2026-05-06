use std::{
    hash::{Hash, Hasher},
    num::{NonZeroU32, NonZeroUsize},
    sync::{Arc, OnceLock},
};

use lru::LruCache;
use parking_lot::RwLock;
use phase_token_prototype as tile_ir;
use rustc_hash::{FxBuildHasher, FxHasher};
use wgpu::naga::{
    AddressSpace, Arena, ArraySize, Barrier, BinaryOperator, Binding, Block, BuiltIn,
    CollectiveOperation, EntryPoint, Expression, Function, FunctionArgument, GlobalVariable,
    Handle, Literal, LocalVariable, MathFunction, Module, Range, ResourceBinding, Scalar,
    ShaderStage, Span, Statement, StorageAccess, SubgroupOperation, Type, TypeInner, VectorSize,
};

use crate::{
    DataTypeEnum,
    compute_graph::{ComputeGraphInner, NodeIndex},
    mir::{
        direct_kernel::{DirectKernel, DirectKernelBinding},
        inputs::MirValue,
        operation::Operation,
        tile_direct::{
            flatten_matrix_layout, tile_storage_read_with_direct_layout,
            tile_storage_write_with_direct_layout,
        },
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
    tensor::TensorData,
};

const BLOCK: usize = 1024;
const VEC4_BLOCK: u32 = 128;
const VEC4_SUBGROUP_WIDTH: u32 = 32;
const RMS_NORM_MODULE_CACHE_SIZE: usize = 128;

fn rms_norm_module_cache()
-> &'static RwLock<LruCache<[u64; 2], Arc<wgpu::naga::Module>, FxBuildHasher>> {
    static CACHE: OnceLock<RwLock<LruCache<[u64; 2], Arc<wgpu::naga::Module>, FxBuildHasher>>> =
        OnceLock::new();
    CACHE.get_or_init(|| {
        RwLock::new(LruCache::with_hasher(
            NonZeroUsize::new(RMS_NORM_MODULE_CACHE_SIZE).unwrap(),
            Default::default(),
        ))
    })
}

fn hash_layout<H: Hasher>(state: &mut H, layout: &crate::Layout) {
    layout.offset().hash(state);
    layout.shape().hash(state);
    layout.strides().hash(state);
}

fn rms_norm_module_key(
    variant: RmsNormKernelVariant,
    rows: u32,
    cols: u32,
    eps_bits: u32,
    has_bias: bool,
    has_residual: bool,
    dispatch_size: [u32; 3],
    input: &TensorData,
    residual: Option<&TensorData>,
    weight: &TensorData,
    bias: Option<&TensorData>,
    output: &TensorData,
) -> [u64; 2] {
    std::array::from_fn(|salt| {
        let mut hasher = FxHasher::default();
        (salt as u64).hash(&mut hasher);
        variant.hash(&mut hasher);
        rows.hash(&mut hasher);
        cols.hash(&mut hasher);
        eps_bits.hash(&mut hasher);
        has_bias.hash(&mut hasher);
        has_residual.hash(&mut hasher);
        dispatch_size.hash(&mut hasher);
        hash_layout(&mut hasher, input.layout());
        residual
            .map(|residual| hash_layout(&mut hasher, residual.layout()))
            .hash(&mut hasher);
        hash_layout(&mut hasher, weight.layout());
        bias.map(|bias| hash_layout(&mut hasher, bias.layout()))
            .hash(&mut hasher);
        hash_layout(&mut hasher, output.layout());
        hasher.finish()
    })
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum RmsNormKernelVariant {
    Tile,
    Vec4,
}

#[derive(Clone, Debug)]
pub(crate) struct RmsNormOperation {
    pub(crate) input: NodeIndex,
    pub(crate) residual: Option<NodeIndex>,
    pub(crate) weight: NodeIndex,
    pub(crate) bias: Option<NodeIndex>,
    shape: Box<[usize]>,
    eps: f32,
}

impl RmsNormOperation {
    pub(crate) fn new(
        input: NodeIndex,
        weight: NodeIndex,
        bias: Option<NodeIndex>,
        shape: &[usize],
        eps: f32,
    ) -> Self {
        Self {
            input,
            residual: None,
            weight,
            bias,
            shape: shape.into(),
            eps,
        }
    }

    pub(crate) fn new_with_residual(
        input: NodeIndex,
        residual: NodeIndex,
        weight: NodeIndex,
        bias: Option<NodeIndex>,
        shape: &[usize],
        eps: f32,
    ) -> Self {
        Self {
            input,
            residual: Some(residual),
            weight,
            bias,
            shape: shape.into(),
            eps,
        }
    }

    fn rows_cols(&self) -> Option<(u32, u32)> {
        let cols = *self.shape.last()?;
        let rows = self.shape[..self.shape.len().saturating_sub(1)]
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))?;
        Some((rows.try_into().ok()?, cols.try_into().ok()?))
    }
}

impl Operation for RmsNormOperation {
    fn workgroup_shape_constraints(&self, _device: &crate::Device) -> WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        constraints.add_constraint(0, Constraint::Equals(1));
        constraints.add_constraint(1, Constraint::Equals(1));
        constraints.add_constraint(2, Constraint::Equals(1));
        constraints
    }

    fn dispatch_size(&self, _workgroup_shape: &WorkgroupShape, _inputs: &[MirValue]) -> [u32; 3] {
        let (rows, _) = self
            .rows_cols()
            .expect("rms norm requires a non-empty shape");
        [rows, 1, 1]
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.input);
        if let Some(residual) = self.residual {
            f(residual);
        }
        f(self.weight);
        if let Some(bias) = self.bias {
            f(bias);
        }
    }

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<MirValue> {
        let input = nodes.get_cached_result(self.input).unwrap();
        let residual = self
            .residual
            .map(|residual| nodes.get_cached_result(residual).unwrap());
        let weight = nodes.get_cached_result(self.weight).unwrap();
        let output =
            TensorData::new_for_shape(input.device(), input.layout().shape(), input.datatype());

        let mut inputs = vec![input.clone().into()];
        if let Some(residual) = residual {
            inputs.push(residual.clone().into());
        }
        inputs.push(weight.clone().into());
        if let Some(bias) = self.bias {
            inputs.push(nodes.get_cached_result(bias).unwrap().clone().into());
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
        let input = inputs.first()?.as_tensor()?;
        let (residual, weight_index) = if self.residual.is_some() {
            (Some(inputs.get(1)?.as_tensor()?), 2)
        } else {
            (None, 1)
        };
        let weight = inputs.get(weight_index)?.as_tensor()?;
        let (bias, output_index) = if self.bias.is_some() {
            (
                Some(inputs.get(weight_index + 1)?.as_tensor()?),
                weight_index + 2,
            )
        } else {
            (None, weight_index + 1)
        };
        let output = inputs.get(output_index)?.as_tensor()?;

        if input.datatype() != DataTypeEnum::F32
            || residual.is_some_and(|residual| residual.datatype() != DataTypeEnum::F32)
            || weight.datatype() != DataTypeEnum::F32
            || output.datatype() != DataTypeEnum::F32
            || bias.is_some_and(|bias| bias.datatype() != DataTypeEnum::F32)
        {
            return None;
        }

        let input_view = flatten_matrix_layout(input.layout())?;
        let residual_view = match residual {
            Some(residual) => Some(flatten_matrix_layout(residual.layout())?),
            None => None,
        };
        let output_view = flatten_matrix_layout(output.layout())?;
        let rows = input_view.rows;
        let cols = input_view.cols;
        if rows != output_view.rows || cols != output_view.cols {
            return None;
        }
        if let Some(residual_view) = residual_view.as_ref()
            && (rows != residual_view.rows || cols != residual_view.cols)
        {
            return None;
        }
        if weight.layout().shape() != [cols as usize] {
            return None;
        }
        if let Some(bias) = bias
            && bias.layout().shape() != [cols as usize]
        {
            return None;
        }

        let has_bias = bias.is_some();
        let has_residual = residual.is_some();
        let vec4_meta = graph
            .device()
            .subgroups_supported()
            .then(|| {
                build_vec4_rms_norm_meta(
                    input_view.clone(),
                    residual_view.clone(),
                    weight,
                    bias,
                    output_view.clone(),
                    self.eps,
                )
            })
            .flatten();
        let variant = if vec4_meta.is_some() {
            RmsNormKernelVariant::Vec4
        } else {
            RmsNormKernelVariant::Tile
        };
        let dispatch_size = [rows, 1, 1];
        let module_key = rms_norm_module_key(
            variant,
            rows,
            cols,
            self.eps.to_bits(),
            has_bias,
            has_residual,
            dispatch_size,
            input,
            residual,
            weight,
            bias,
            output,
        );
        let kernel_label = match variant {
            RmsNormKernelVariant::Tile => "rms_norm",
            RmsNormKernelVariant::Vec4 => "rms_norm_vec4",
        };
        let cache_key = format!(
            "{kernel_label}:{:016x}{:016x}",
            module_key[0], module_key[1]
        );
        let module = if let Some(module) = rms_norm_module_cache().write().get(&module_key) {
            module.clone()
        } else {
            let verbose_cache_key = format!(
                "{}:tile-program:rows={rows}:cols={cols}:eps={:?}:bias={has_bias}:residual={has_residual}:dispatch={dispatch_size:?}:{:?}:{:?}:{:?}:{:?}:{:?}",
                self.name(),
                self.eps.to_bits(),
                input.layout(),
                residual.map(|residual| residual.layout()),
                weight.layout(),
                bias.map(|bias| bias.layout()),
                output.layout()
            );
            let module = if let Some(module) = graph
                .device()
                .naga_module_cache()
                .write()
                .get(&verbose_cache_key)
            {
                Arc::new(module.clone())
            } else {
                let module = if let Some(meta) = vec4_meta {
                    build_rms_norm_vec4_naga_module(meta)?
                } else {
                    let ir = build_rms_norm_tile_ir(
                        input_view,
                        residual_view,
                        weight,
                        bias,
                        output_view,
                        self.eps,
                    )?;
                    ir.lower_to_naga().ok()?.module().clone()
                };
                let _ = graph
                    .device()
                    .naga_module_cache()
                    .write()
                    .get_or_insert(verbose_cache_key, || module.clone());
                Arc::new(module)
            };
            rms_norm_module_cache()
                .write()
                .get_or_insert(module_key, || module.clone())
                .clone()
        };

        let mut bindings = vec![DirectKernelBinding::Storage {
            binding: 0,
            buffer: input.buffer().clone(),
            read_only: true,
        }];
        let mut binding = 1;
        if let Some(residual) = residual {
            bindings.push(DirectKernelBinding::Storage {
                binding,
                buffer: residual.buffer().clone(),
                read_only: true,
            });
            binding += 1;
        }
        bindings.push(DirectKernelBinding::Storage {
            binding,
            buffer: weight.buffer().clone(),
            read_only: true,
        });
        binding += 1;
        if let Some(bias) = bias {
            bindings.push(DirectKernelBinding::Storage {
                binding,
                buffer: bias.buffer().clone(),
                read_only: true,
            });
            binding += 1;
        }
        bindings.push(DirectKernelBinding::Storage {
            binding,
            buffer: output.buffer().clone(),
            read_only: false,
        });

        Some(DirectKernel::new_with_arc_module(
            kernel_label,
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
        let op = if self.residual.is_some() {
            "rms_norm_residual"
        } else {
            "rms_norm"
        };
        format!(
            "{op}_f32_{}",
            self.shape
                .iter()
                .map(|dim| dim.to_string())
                .collect::<Vec<_>>()
                .join("x")
        )
    }
}

#[derive(Clone, Copy)]
struct RmsNormVec4Meta {
    cols: u32,
    cols_vec: u32,
    eps: f32,
    input_offset_vec: u32,
    input_row_stride_vec: u32,
    residual_offset_vec: Option<u32>,
    residual_row_stride_vec: u32,
    weight_offset_vec: u32,
    bias_offset_vec: Option<u32>,
    output_offset_vec: u32,
    output_row_stride_vec: u32,
}

fn build_vec4_rms_norm_meta(
    input_view: crate::mir::tile_direct::DirectMatrixLayout,
    residual_view: Option<crate::mir::tile_direct::DirectMatrixLayout>,
    weight: &TensorData,
    bias: Option<&TensorData>,
    output_view: crate::mir::tile_direct::DirectMatrixLayout,
    eps: f32,
) -> Option<RmsNormVec4Meta> {
    if input_view.index_map.is_some()
        || output_view.index_map.is_some()
        || residual_view
            .as_ref()
            .is_some_and(|residual| residual.index_map.is_some())
        || !input_view.cols.is_multiple_of(4)
    {
        return None;
    }

    let [input_row_stride, input_col_stride] = matrix_strides(input_view.layout.strides())?;
    let [output_row_stride, output_col_stride] = matrix_strides(output_view.layout.strides())?;
    if input_col_stride != 1
        || output_col_stride != 1
        || !input_view.offset.is_multiple_of(4)
        || !output_view.offset.is_multiple_of(4)
        || !input_row_stride.is_multiple_of(4)
        || !output_row_stride.is_multiple_of(4)
    {
        return None;
    }

    let (residual_offset_vec, residual_row_stride_vec) = if let Some(residual_view) = residual_view
    {
        let [residual_row_stride, residual_col_stride] =
            matrix_strides(residual_view.layout.strides())?;
        if residual_col_stride != 1
            || !residual_view.offset.is_multiple_of(4)
            || !residual_row_stride.is_multiple_of(4)
        {
            return None;
        }
        (Some(residual_view.offset / 4), residual_row_stride / 4)
    } else {
        (None, 0)
    };

    let weight_stride = *weight.layout().strides().first()?;
    if weight.layout().shape() != [input_view.cols as usize]
        || weight_stride != 1
        || !weight.layout().offset().is_multiple_of(4)
    {
        return None;
    }
    let bias_offset_vec = if let Some(bias) = bias {
        let bias_stride = *bias.layout().strides().first()?;
        if bias.layout().shape() != [input_view.cols as usize]
            || bias_stride != 1
            || !bias.layout().offset().is_multiple_of(4)
        {
            return None;
        }
        Some((bias.layout().offset() / 4).try_into().ok()?)
    } else {
        None
    };

    Some(RmsNormVec4Meta {
        cols: input_view.cols,
        cols_vec: input_view.cols / 4,
        eps,
        input_offset_vec: input_view.offset / 4,
        input_row_stride_vec: input_row_stride / 4,
        residual_offset_vec,
        residual_row_stride_vec,
        weight_offset_vec: (weight.layout().offset() / 4).try_into().ok()?,
        bias_offset_vec,
        output_offset_vec: output_view.offset / 4,
        output_row_stride_vec: output_row_stride / 4,
    })
}

fn matrix_strides(strides: &tile_ir::Strides) -> Option<[u32; 2]> {
    strides.values().try_into().ok()
}

#[derive(Clone, Copy)]
struct RmsNormVec4Globals {
    input: Handle<GlobalVariable>,
    residual: Option<Handle<GlobalVariable>>,
    weight: Handle<GlobalVariable>,
    bias: Option<Handle<GlobalVariable>>,
    output: Handle<GlobalVariable>,
    scratch: Handle<GlobalVariable>,
}

#[derive(Clone, Copy)]
struct RmsNormVec4Locals {
    col: Handle<LocalVariable>,
    sum: Handle<LocalVariable>,
}

struct RmsNormVec4NagaBuilder {
    meta: RmsNormVec4Meta,
    has_residual: bool,
    has_bias: bool,
}

impl RmsNormVec4NagaBuilder {
    fn new(meta: RmsNormVec4Meta) -> Self {
        Self {
            meta,
            has_residual: meta.residual_offset_vec.is_some(),
            has_bias: meta.bias_offset_vec.is_some(),
        }
    }

    fn build(self) -> Option<Module> {
        let mut module = Module::default();
        let f32_ty = module.types.insert(
            Type {
                name: Some("RmsNormF32".into()),
                inner: TypeInner::Scalar(Scalar::F32),
            },
            Span::default(),
        );
        let u32_ty = module.types.insert(
            Type {
                name: Some("RmsNormU32".into()),
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::default(),
        );
        let f32_vec4_ty = module.types.insert(
            Type {
                name: Some("RmsNormF32Vec4".into()),
                inner: TypeInner::Vector {
                    size: VectorSize::Quad,
                    scalar: Scalar::F32,
                },
            },
            Span::default(),
        );
        let u32_vec3_ty = module.types.insert(
            Type {
                name: Some("RmsNormWorkgroupId".into()),
                inner: TypeInner::Vector {
                    size: VectorSize::Tri,
                    scalar: Scalar::U32,
                },
            },
            Span::default(),
        );
        let storage_ty = module.types.insert(
            Type {
                name: Some("RmsNormVec4Buffer".into()),
                inner: TypeInner::Array {
                    base: f32_vec4_ty,
                    size: ArraySize::Dynamic,
                    stride: 16,
                },
            },
            Span::default(),
        );
        let scratch_ty = module.types.insert(
            Type {
                name: Some("RmsNormScratch".into()),
                inner: TypeInner::Array {
                    base: f32_ty,
                    size: ArraySize::Constant(NonZeroU32::new(VEC4_SUBGROUP_WIDTH)?),
                    stride: 4,
                },
            },
            Span::default(),
        );

        let input = Self::storage_global(&mut module, "input", 0, storage_ty, true);
        let mut binding = 1;
        let residual = if self.has_residual {
            let residual = Self::storage_global(&mut module, "residual", binding, storage_ty, true);
            binding += 1;
            Some(residual)
        } else {
            None
        };
        let weight = Self::storage_global(&mut module, "weight", binding, storage_ty, true);
        binding += 1;
        let bias = if self.has_bias {
            let bias = Self::storage_global(&mut module, "bias", binding, storage_ty, true);
            binding += 1;
            Some(bias)
        } else {
            None
        };
        let output = Self::storage_global(&mut module, "output", binding, storage_ty, false);
        let scratch = module.global_variables.append(
            GlobalVariable {
                name: Some("rms_norm_scratch".into()),
                space: AddressSpace::WorkGroup,
                binding: None,
                ty: scratch_ty,
                init: None,
            },
            Span::default(),
        );
        let globals = RmsNormVec4Globals {
            input,
            residual,
            weight,
            bias,
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
                FunctionArgument {
                    name: Some("subgroup_id".into()),
                    ty: u32_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::SubgroupId)),
                },
                FunctionArgument {
                    name: Some("subgroup_invocation_id".into()),
                    ty: u32_ty,
                    binding: Some(Binding::BuiltIn(BuiltIn::SubgroupInvocationId)),
                },
            ],
            ..Function::default()
        };
        let locals = RmsNormVec4Locals {
            col: Self::local(&mut function, "col_vec", u32_ty),
            sum: Self::local(&mut function, "sum", f32_ty),
        };

        function.body = self.entry_body(
            &mut function.expressions,
            globals,
            locals,
            f32_ty,
            f32_vec4_ty,
        );
        function
            .body
            .push(Statement::Return { value: None }, Span::default());
        module.entry_points.push(EntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [VEC4_BLOCK, 1, 1],
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
        globals: RmsNormVec4Globals,
        locals: RmsNormVec4Locals,
        f32_ty: Handle<Type>,
        f32_vec4_ty: Handle<Type>,
    ) -> Block {
        let mut body = Block::new();
        let local_index = expressions.append(Expression::FunctionArgument(0), Span::default());
        let workgroup_id = expressions.append(Expression::FunctionArgument(1), Span::default());
        let subgroup_id = expressions.append(Expression::FunctionArgument(2), Span::default());
        let subgroup_lane = expressions.append(Expression::FunctionArgument(3), Span::default());
        let row = self.emit(
            expressions,
            &mut body,
            Expression::AccessIndex {
                base: workgroup_id,
                index: 0,
            },
        );

        let first_subgroup = self.eq_lit(expressions, &mut body, subgroup_id, 0);
        let mut init_scratch = Block::new();
        let zero = self.f32_lit(expressions, 0.0);
        self.store_workgroup(
            expressions,
            &mut init_scratch,
            globals.scratch,
            subgroup_lane,
            zero,
        );
        body.push(
            Statement::If {
                condition: first_subgroup,
                accept: init_scratch,
                reject: Block::new(),
            },
            Span::default(),
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        self.store_local(expressions, &mut body, locals.sum, zero);
        self.store_local(expressions, &mut body, locals.col, local_index);
        self.append_sum_loop(expressions, &mut body, globals, locals, row);

        let sum = self.load_local(expressions, &mut body, locals.sum);
        let subgroup_sum = self.subgroup_sum(expressions, &mut body, sum, f32_ty);
        let subgroup_lane_zero = self.eq_lit(expressions, &mut body, subgroup_lane, 0);
        let mut store_subgroup_sum = Block::new();
        self.store_workgroup(
            expressions,
            &mut store_subgroup_sum,
            globals.scratch,
            subgroup_id,
            subgroup_sum,
        );
        body.push(
            Statement::If {
                condition: subgroup_lane_zero,
                accept: store_subgroup_sum,
                reject: Block::new(),
            },
            Span::default(),
        );
        body.push(
            Statement::ControlBarrier(Barrier::WORK_GROUP),
            Span::default(),
        );

        let scratch_sum =
            self.load_workgroup(expressions, &mut body, globals.scratch, subgroup_lane);
        let total_sum = self.subgroup_sum(expressions, &mut body, scratch_sum, f32_ty);
        let cols = self.f32_lit(expressions, self.meta.cols as f32);
        let mean = self.bin(
            expressions,
            &mut body,
            BinaryOperator::Divide,
            total_sum,
            cols,
        );
        let eps = self.f32_lit(expressions, self.meta.eps);
        let mean_eps = self.bin(expressions, &mut body, BinaryOperator::Add, mean, eps);
        let scale = self.emit(
            expressions,
            &mut body,
            Expression::Math {
                fun: MathFunction::InverseSqrt,
                arg: mean_eps,
                arg1: None,
                arg2: None,
                arg3: None,
            },
        );

        self.store_local(expressions, &mut body, locals.col, local_index);
        self.append_store_loop(
            expressions,
            &mut body,
            globals,
            locals,
            row,
            scale,
            f32_vec4_ty,
        );

        body
    }

    fn append_sum_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: RmsNormVec4Globals,
        locals: RmsNormVec4Locals,
        row: Handle<Expression>,
    ) {
        let mut loop_body = Block::new();
        let col = self.load_local(expressions, &mut loop_body, locals.col);
        let done = self.ge_lit(expressions, &mut loop_body, col, self.meta.cols_vec);
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let value = self.load_input_vec4(expressions, &mut loop_body, globals, row, col);
        let dot = self.emit(
            expressions,
            &mut loop_body,
            Expression::Math {
                fun: MathFunction::Dot,
                arg: value,
                arg1: Some(value),
                arg2: None,
                arg3: None,
            },
        );
        let sum = self.load_local(expressions, &mut loop_body, locals.sum);
        let sum = self.bin(expressions, &mut loop_body, BinaryOperator::Add, sum, dot);
        self.store_local(expressions, &mut loop_body, locals.sum, sum);
        let next_col = self.add_lit(expressions, &mut loop_body, col, VEC4_BLOCK);
        self.store_local(expressions, &mut loop_body, locals.col, next_col);

        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }

    fn append_store_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: RmsNormVec4Globals,
        locals: RmsNormVec4Locals,
        row: Handle<Expression>,
        scale: Handle<Expression>,
        f32_vec4_ty: Handle<Type>,
    ) {
        let mut loop_body = Block::new();
        let col = self.load_local(expressions, &mut loop_body, locals.col);
        let done = self.ge_lit(expressions, &mut loop_body, col, self.meta.cols_vec);
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let value = self.load_input_vec4(expressions, &mut loop_body, globals, row, col);
        let scale_vec = self.splat_vec4(expressions, &mut loop_body, f32_vec4_ty, scale);
        let normalized = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Multiply,
            value,
            scale_vec,
        );
        let weight_index = self.add_lit(
            expressions,
            &mut loop_body,
            col,
            self.meta.weight_offset_vec,
        );
        let weight = self.load_storage(expressions, &mut loop_body, globals.weight, weight_index);
        let mut output = self.bin(
            expressions,
            &mut loop_body,
            BinaryOperator::Multiply,
            normalized,
            weight,
        );
        if let (Some(bias), Some(bias_offset_vec)) = (globals.bias, self.meta.bias_offset_vec) {
            let bias_index = self.add_lit(expressions, &mut loop_body, col, bias_offset_vec);
            let bias_value = self.load_storage(expressions, &mut loop_body, bias, bias_index);
            output = self.bin(
                expressions,
                &mut loop_body,
                BinaryOperator::Add,
                output,
                bias_value,
            );
        }
        let output_index = self.matrix_index(
            expressions,
            &mut loop_body,
            self.meta.output_offset_vec,
            self.meta.output_row_stride_vec,
            row,
            col,
        );
        self.store_storage(
            expressions,
            &mut loop_body,
            globals.output,
            output_index,
            output,
        );
        let next_col = self.add_lit(expressions, &mut loop_body, col, VEC4_BLOCK);
        self.store_local(expressions, &mut loop_body, locals.col, next_col);

        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
    }

    fn load_input_vec4(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        globals: RmsNormVec4Globals,
        row: Handle<Expression>,
        col: Handle<Expression>,
    ) -> Handle<Expression> {
        let input_index = self.matrix_index(
            expressions,
            body,
            self.meta.input_offset_vec,
            self.meta.input_row_stride_vec,
            row,
            col,
        );
        let mut value = self.load_storage(expressions, body, globals.input, input_index);
        if let (Some(residual), Some(residual_offset_vec)) =
            (globals.residual, self.meta.residual_offset_vec)
        {
            let residual_index = self.matrix_index(
                expressions,
                body,
                residual_offset_vec,
                self.meta.residual_row_stride_vec,
                row,
                col,
            );
            let residual_value = self.load_storage(expressions, body, residual, residual_index);
            value = self.bin(
                expressions,
                body,
                BinaryOperator::Add,
                value,
                residual_value,
            );
        }
        value
    }

    fn matrix_index(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        offset: u32,
        row_stride: u32,
        row: Handle<Expression>,
        col: Handle<Expression>,
    ) -> Handle<Expression> {
        let base = self.u32_lit(expressions, offset);
        let index = self.add_scaled_index(expressions, body, base, row, row_stride);
        self.bin(expressions, body, BinaryOperator::Add, index, col)
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

    fn subgroup_sum(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        result_ty: Handle<Type>,
    ) -> Handle<Expression> {
        let result = expressions.append(
            Expression::SubgroupOperationResult { ty: result_ty },
            Span::default(),
        );
        body.push(
            Statement::SubgroupCollectiveOperation {
                op: SubgroupOperation::Add,
                collective_op: CollectiveOperation::Reduce,
                argument: value,
                result,
            },
            Span::default(),
        );
        result
    }

    fn splat_vec4(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        ty: Handle<Type>,
        value: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit(
            expressions,
            body,
            Expression::Compose {
                ty,
                components: vec![value, value, value, value],
            },
        )
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

    fn load_workgroup(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
    ) -> Handle<Expression> {
        let ptr = self.workgroup_ptr(expressions, body, global, index);
        self.emit(expressions, body, Expression::Load { pointer: ptr })
    }

    fn store_workgroup(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        global: Handle<GlobalVariable>,
        index: Handle<Expression>,
        value: Handle<Expression>,
    ) {
        let pointer = self.workgroup_ptr(expressions, body, global, index);
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

    fn workgroup_ptr(
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

    fn add_lit(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        let rhs = self.u32_lit(expressions, literal);
        self.bin(expressions, body, BinaryOperator::Add, value, rhs)
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
}

fn build_rms_norm_vec4_naga_module(meta: RmsNormVec4Meta) -> Option<Module> {
    RmsNormVec4NagaBuilder::new(meta).build()
}

fn build_rms_norm_tile_ir(
    input_view: crate::mir::tile_direct::DirectMatrixLayout,
    residual_view: Option<crate::mir::tile_direct::DirectMatrixLayout>,
    weight: &TensorData,
    bias: Option<&TensorData>,
    output_view: crate::mir::tile_direct::DirectMatrixLayout,
    eps: f32,
) -> Option<tile_ir::KernelIr> {
    let rows = input_view.rows;
    let cols = input_view.cols;
    let input_storage_layout = input_view.layout.clone();
    let residual_storage_layout = residual_view
        .as_ref()
        .map(|residual_view| residual_view.layout.clone());
    let residual_offset = residual_view.as_ref().map(|residual| residual.offset);
    let residual_index_map = residual_view
        .as_ref()
        .and_then(|residual_view| residual_view.index_map.clone());
    let output_storage_layout = output_view.layout.clone();
    let weight_layout = vector_as_row_layout(weight.layout())?;
    let bias_layout = match bias {
        Some(bias) => Some(vector_as_row_layout(bias.layout())?),
        None => None,
    };
    let weight_offset = weight.layout().offset().try_into().ok()?;
    let bias_offset = match bias {
        Some(bias) => Some(bias.layout().offset().try_into().ok()?),
        None => None,
    };

    Some(tile_ir::tile::build(move |phase| {
        let input = tile_storage_read_with_direct_layout(
            phase,
            crate::mir::tile_direct::DirectMatrixLayout {
                rows,
                cols,
                offset: input_view.offset,
                layout: input_storage_layout,
                index_map: input_view.index_map,
            },
        );
        let residual = residual_storage_layout.map(|layout| {
            tile_storage_read_with_direct_layout(
                phase,
                crate::mir::tile_direct::DirectMatrixLayout {
                    rows,
                    cols,
                    offset: residual_offset.expect("residual offset exists with layout"),
                    layout,
                    index_map: residual_index_map,
                },
            )
        });
        let weight =
            phase.storage_read_with_layout_offset::<tile_ir::F32, 2>(weight_layout, weight_offset);
        let bias = bias_layout.map(|layout| {
            phase.storage_read_with_layout_offset::<tile_ir::F32, 2>(
                layout,
                bias_offset.expect("bias offset exists when bias layout exists"),
            )
        });
        let output = tile_storage_write_with_direct_layout(
            phase,
            crate::mir::tile_direct::DirectMatrixLayout {
                rows,
                cols,
                offset: output_view.offset,
                layout: output_storage_layout,
                index_map: output_view.index_map,
            },
        );

        let chunks = cols.div_ceil(BLOCK as u32);
        phase.program_grid::<BLOCK>([rows, 1, 1], |program| {
            let row = program.program_id(tile_ir::WorkgroupAxis::X);
            let lane = program.arange();
            let reduce_col = program.loop_index() * BLOCK as u32 + lane.clone();
            let reduce_mask = reduce_col.lt(cols);
            let mut value = program.load(input.at(&row, &reduce_col), reduce_mask.clone(), 0.0);
            if let Some(residual) = &residual {
                value = value + program.load(residual.at(&row, &reduce_col), reduce_mask, 0.0);
            }
            let sum_square = program.loop_reduce_sum(chunks, value.clone() * value);
            let rms = (tile_ir::tile::Tile::<BLOCK>::from(sum_square)
                / tile_ir::tile::Scalar::literal(cols as f32)
                + tile_ir::tile::Scalar::literal(eps))
            .unary(tile_ir::TileUnaryOp::Sqrt);
            for chunk in 0..chunks {
                let col = lane.clone() + chunk * BLOCK as u32;
                let mask = col.lt(cols);
                let mut value = program.load(input.at(&row, &col), mask.clone(), 0.0);
                if let Some(residual) = &residual {
                    value = value + program.load(residual.at(&row, &col), mask.clone(), 0.0);
                }
                let weight = program.load(weight.at(0, &col), mask.clone(), 0.0);
                let mut normalized = value / rms.clone() * weight;
                if let Some(bias) = &bias {
                    let bias_value = program.load(bias.at(0, &col), mask.clone(), 0.0);
                    normalized = normalized + bias_value;
                }
                program.store(output.at(&row, col), normalized, mask);
            }
        });
    }))
}

fn vector_as_row_layout(layout: &crate::Layout) -> Option<tile_ir::Layout> {
    let shape = layout.shape();
    let strides = layout.strides();
    if shape.len() != 1 {
        return None;
    }
    Some(tile_ir::Layout::strided(
        tile_ir::MemoryLevel::Storage,
        tile_ir::Shape::new([1, (*shape.first()?).try_into().ok()?]),
        tile_ir::Strides::new([0, (*strides.first()?).try_into().ok()?]),
    ))
}

#[cfg(test)]
mod tests {
    use crate::{Device, Tensor};

    #[tokio::test]
    async fn rms_norm_direct_matches_reference() {
        let Ok(device) = Device::new().await else {
            return;
        };

        let input = Tensor::new(&device, &vec![vec![1.0f32, 2.0, 3.0, 4.0]]);
        let weight = Tensor::new(&device, &vec![0.5f32, 1.0, 1.5, 2.0]);
        let output = input.try_rms_norm_direct(&weight, None, 1e-5).unwrap();
        let output = output.as_slice().await.unwrap();

        let mean_square = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
        let rms = f32::sqrt(mean_square + 1e-5);
        let expected = [1.0 / rms * 0.5, 2.0 / rms, 3.0 / rms * 1.5, 4.0 / rms * 2.0];

        for (i, expected) in expected.into_iter().enumerate() {
            let actual = output[[0, i]];
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }
}
