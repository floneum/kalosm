use std::fmt;

use naga::{
    AddressSpace, Arena, ArraySize, Barrier, BinaryOperator, Binding, Block, BuiltIn, EntryPoint,
    Expression, Function, FunctionArgument, GlobalVariable, Handle, Literal, LocalVariable,
    MemoryDecorations, Module, Range, ResourceBinding, Scalar, ShaderStage, Span, Statement,
    StorageAccess, Type, TypeInner,
};

use crate::{
    BarrierScope, BufferAccess, BufferId, ElementType, KernelIr, Layout, MemoryLevel, MmaOp, Op,
    StorageView, TileId, TileOrigin, TileRef, ViewMapping,
};

const LOCAL_INVOCATION_INDEX_ARG: u32 = 0;
const WORKGROUP_SIZE_X: u32 = 256;

pub(crate) fn lower_to_naga(ir: &KernelIr) -> Result<NagaKernel, LowerError> {
    Lowerer::new(ir).lower()
}

/// A validated Naga lowering result.
pub struct NagaKernel {
    module: Module,
    info: naga::valid::ModuleInfo,
}

impl NagaKernel {
    /// The generated Naga module.
    pub fn module(&self) -> &Module {
        &self.module
    }

    /// Naga validation metadata for the generated module.
    pub fn info(&self) -> &naga::valid::ModuleInfo {
        &self.info
    }
}

/// Errors produced by the Naga lowering pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerError {
    /// An event referenced a tile that was never allocated.
    UnknownTile(TileId),
    /// An operation referenced a storage buffer that was never declared.
    UnknownBuffer(BufferId),
    /// The Naga lowerer cannot emit this memory level.
    UnsupportedMemoryLevel(MemoryLevel),
    /// The typed IR operation is outside the supported lowering subset.
    UnsupportedOperation(&'static str),
    /// An operation used a tile as a different element type than its declaration.
    TileElementMismatch {
        tile: TileId,
        declared: ElementType,
        used: ElementType,
    },
    /// Naga rejected the generated module.
    Validation(String),
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownTile(tile) => write!(f, "unknown tile {:?}", tile),
            Self::UnknownBuffer(buffer) => write!(f, "unknown buffer {:?}", buffer),
            Self::UnsupportedMemoryLevel(memory) => {
                write!(f, "unsupported memory level {:?}", memory)
            }
            Self::UnsupportedOperation(op) => write!(f, "unsupported operation {op}"),
            Self::TileElementMismatch {
                tile,
                declared,
                used,
            } => write!(
                f,
                "tile {:?} declared as {:?} but used as {:?}",
                tile, declared, used
            ),
            Self::Validation(error) => write!(f, "naga validation failed: {error}"),
        }
    }
}

impl std::error::Error for LowerError {}

struct Lowerer<'a> {
    ir: &'a KernelIr,
    module: Module,
    f32_ty: Handle<Type>,
    u32_ty: Handle<Type>,
    buffer_globals: Vec<Option<Handle<GlobalVariable>>>,
    tile_globals: Vec<Option<Handle<GlobalVariable>>>,
    tile_locals: Vec<Option<Handle<LocalVariable>>>,
}

#[derive(Copy, Clone)]
struct ScratchLocals {
    tile_index: Handle<LocalVariable>,
    linear_index: Handle<LocalVariable>,
    store_index: Handle<LocalVariable>,
    mma_i: Handle<LocalVariable>,
    mma_j: Handle<LocalVariable>,
    mma_k: Handle<LocalVariable>,
}

impl<'a> Lowerer<'a> {
    fn new(ir: &'a KernelIr) -> Self {
        let mut module = Module::default();
        let f32_ty = module.types.insert(
            Type {
                name: Some("TileElement".into()),
                inner: TypeInner::Scalar(Scalar::F32),
            },
            Span::default(),
        );
        let u32_ty = module.types.insert(
            Type {
                name: Some("SubgroupIndex".into()),
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::default(),
        );

        Self {
            ir,
            module,
            f32_ty,
            u32_ty,
            buffer_globals: Vec::new(),
            tile_globals: Vec::new(),
            tile_locals: Vec::new(),
        }
    }

    fn lower(mut self) -> Result<NagaKernel, LowerError> {
        self.create_storage_globals()?;
        self.create_workgroup_globals()?;

        let mut function = Function {
            name: Some("main".into()),
            arguments: vec![FunctionArgument {
                name: Some("local_invocation_index".into()),
                ty: self.u32_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)),
            }],
            ..Function::default()
        };
        let scratch = self.create_scratch_locals(&mut function);
        self.create_private_locals(&mut function)?;

        function.body = self.lower_block(self.ir.body(), &mut function.expressions, scratch)?;
        function
            .body
            .push(Statement::Return { value: None }, Span::default());

        self.module.entry_points.push(EntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [WORKGROUP_SIZE_X, 1, 1],
            workgroup_size_overrides: None,
            function,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });

        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&self.module)
        .map_err(|error| LowerError::Validation(format!("{error:#?}")))?;

        Ok(NagaKernel {
            module: self.module,
            info,
        })
    }

    fn create_storage_globals(&mut self) -> Result<(), LowerError> {
        self.buffer_globals = vec![None; self.ir.buffers().len()];
        for buffer in self.ir.buffers() {
            let ty = self.storage_type(buffer.id.index(), buffer.element, &buffer.layout);
            let access = match buffer.access {
                BufferAccess::Read => StorageAccess::LOAD,
                BufferAccess::ReadWrite => StorageAccess::LOAD | StorageAccess::STORE,
            };
            let global = self.module.global_variables.append(
                GlobalVariable {
                    name: Some(format!("buffer_{}", buffer.id.index())),
                    space: AddressSpace::Storage { access },
                    binding: Some(ResourceBinding {
                        group: 0,
                        binding: buffer.id.index() as u32,
                    }),
                    ty,
                    init: None,
                    memory_decorations: MemoryDecorations::empty(),
                },
                Span::default(),
            );
            self.buffer_globals[buffer.id.index()] = Some(global);
        }
        Ok(())
    }

    fn create_workgroup_globals(&mut self) -> Result<(), LowerError> {
        self.tile_globals = vec![None; self.ir.tiles().len()];
        for tile in self.ir.tiles() {
            if tile.layout.memory_level() != MemoryLevel::Workgroup
                || tile.origin != TileOrigin::Allocation
            {
                continue;
            }
            let ty = self.tile_type(tile.id.index(), tile.element, &tile.layout);
            let global = self.module.global_variables.append(
                GlobalVariable {
                    name: Some(format!("tile_{}", tile.id.index())),
                    space: AddressSpace::WorkGroup,
                    binding: None,
                    ty,
                    init: None,
                    memory_decorations: MemoryDecorations::empty(),
                },
                Span::default(),
            );
            self.tile_globals[tile.id.index()] = Some(global);
        }
        Ok(())
    }

    fn create_private_locals(&mut self, function: &mut Function) -> Result<(), LowerError> {
        self.tile_locals = vec![None; self.ir.tiles().len()];
        for tile in self.ir.tiles() {
            if tile.layout.memory_level() != MemoryLevel::Private
                || tile.origin != TileOrigin::Allocation
            {
                continue;
            }
            let ty = self.tile_type(tile.id.index(), tile.element, &tile.layout);
            let local = function.local_variables.append(
                LocalVariable {
                    name: Some(format!("tile_{}", tile.id.index())),
                    ty,
                    init: None,
                },
                Span::default(),
            );
            self.tile_locals[tile.id.index()] = Some(local);
        }
        Ok(())
    }

    fn create_scratch_locals(&self, function: &mut Function) -> ScratchLocals {
        ScratchLocals {
            tile_index: self.create_u32_local(function, "tile_index"),
            linear_index: self.create_u32_local(function, "linear_index"),
            store_index: self.create_u32_local(function, "store_index"),
            mma_i: self.create_u32_local(function, "mma_i"),
            mma_j: self.create_u32_local(function, "mma_j"),
            mma_k: self.create_u32_local(function, "mma_k"),
        }
    }

    fn create_u32_local(&self, function: &mut Function, name: &str) -> Handle<LocalVariable> {
        function.local_variables.append(
            LocalVariable {
                name: Some(name.into()),
                ty: self.u32_ty,
                init: None,
            },
            Span::default(),
        )
    }

    fn tile_type(&mut self, tile: usize, element: ElementType, layout: &Layout) -> Handle<Type> {
        self.array_type(format!("Tile{tile}"), element, layout)
    }

    fn storage_type(
        &mut self,
        buffer: usize,
        element: ElementType,
        layout: &Layout,
    ) -> Handle<Type> {
        self.array_type(format!("Buffer{buffer}"), element, layout)
    }

    fn array_type(&mut self, name: String, element: ElementType, layout: &Layout) -> Handle<Type> {
        let base = match element {
            ElementType::F32 => self.f32_ty,
        };

        self.module.types.insert(
            Type {
                name: Some(name),
                inner: TypeInner::Array {
                    base,
                    size: ArraySize::Constant(layout.element_count()),
                    stride: 4,
                },
            },
            Span::default(),
        )
    }

    fn lower_block(
        &self,
        ir_block: &crate::Block,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
    ) -> Result<Block, LowerError> {
        let mut body = Block::new();

        for op in ir_block.ops() {
            match op {
                Op::Block(op) => {
                    body.push(
                        Statement::Block(self.lower_block(&op.body, expressions, scratch)?),
                        Span::default(),
                    );
                }
                Op::FillTile(op) => match self.tile_layout(op.dst)?.memory_level() {
                    MemoryLevel::Workgroup => {
                        body.push(
                            self.store_zero_to_tile(expressions, scratch.tile_index, op.dst)?,
                            Span::default(),
                        );
                    }
                    MemoryLevel::Private => {
                        body.push(
                            self.fill_private_tile(expressions, scratch.linear_index, op.dst)?,
                            Span::default(),
                        );
                    }
                    memory => return Err(LowerError::UnsupportedMemoryLevel(memory)),
                },
                Op::CooperativeLoad(op) => {
                    body.push(
                        self.lower_cooperative_load(
                            expressions,
                            scratch.tile_index,
                            op.dst,
                            op.src,
                        )?,
                        Span::default(),
                    );
                }
                Op::Barrier(op) => {
                    let barrier = match op.scope {
                        BarrierScope::Workgroup => Barrier::WORK_GROUP,
                    };
                    body.push(Statement::ControlBarrier(barrier), Span::default());
                }
                Op::Partition(op) => {
                    for binding in &op.bindings {
                        self.tile_layout(binding.source)?;
                        self.tile_layout(binding.view)?;
                    }
                    body.push(
                        Statement::Block(self.lower_block(&op.body, expressions, scratch)?),
                        Span::default(),
                    );
                }
                Op::Gemm(op) => {
                    let _ = op;
                    return Err(LowerError::UnsupportedOperation("gemm before expansion"));
                }
                Op::Mma(op) => {
                    body.push(self.lower_mma(expressions, scratch, op)?, Span::default());
                }
                Op::StoreTile(op) => {
                    body.push(
                        self.lower_store_tile(expressions, scratch.store_index, op.src, op.dst)?,
                        Span::default(),
                    );
                }
                Op::Loop(op) => {
                    let mut loop_body = self.lower_block(&op.body, expressions, scratch)?;
                    loop_body.push(Statement::Break, Span::default());
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
        }

        Ok(body)
    }

    fn fill_private_tile(
        &self,
        expressions: &mut Arena<Expression>,
        index_local: Handle<LocalVariable>,
        tile: TileRef,
    ) -> Result<Statement, LowerError> {
        let layout = self.tile_layout(tile)?;
        let mut body = Block::new();
        let (index, index_emit) = self.load_u32_local(expressions, index_local);
        let (pointer, pointer_emits) = self.tile_dynamic_pointer(expressions, tile, index)?;
        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        body.push(Statement::Emit(index_emit), Span::default());
        for emit in pointer_emits {
            body.push(Statement::Emit(emit), Span::default());
        }
        body.push(
            Statement::Store {
                pointer,
                value: zero,
            },
            Span::default(),
        );

        Ok(self.counted_loop(expressions, index_local, layout.element_count().get(), body))
    }

    fn lower_mma(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        op: &MmaOp,
    ) -> Result<Statement, LowerError> {
        let a_layout = self.tile_layout(op.a)?;
        let b_layout = self.tile_layout(op.b)?;
        let acc_layout = self.tile_layout(op.acc)?;
        let [m, k_a] = Self::matrix_shape(a_layout)?;
        let [k_b, n] = Self::matrix_shape(b_layout)?;
        let [m_acc, n_acc] = Self::matrix_shape(acc_layout)?;

        if k_a != k_b || m != m_acc || n != n_acc {
            return Err(LowerError::UnsupportedOperation("mma shape mismatch"));
        }
        if acc_layout.memory_level() != MemoryLevel::Private {
            return Err(LowerError::UnsupportedMemoryLevel(
                acc_layout.memory_level(),
            ));
        }

        let mut k_body = Block::new();
        let (i, i_emit) = self.load_u32_local(expressions, scratch.mma_i);
        let (j, j_emit) = self.load_u32_local(expressions, scratch.mma_j);
        let (k, k_emit) = self.load_u32_local(expressions, scratch.mma_k);
        k_body.push(Statement::Emit(i_emit), Span::default());
        k_body.push(Statement::Emit(j_emit), Span::default());
        k_body.push(Statement::Emit(k_emit), Span::default());

        let (acc_index, acc_index_emits) =
            self.layout_index_expr(expressions, acc_layout, &[i, j])?;
        let (a_index, a_index_emits) = self.layout_index_expr(expressions, a_layout, &[i, k])?;
        let (b_index, b_index_emits) = self.layout_index_expr(expressions, b_layout, &[k, j])?;
        Self::push_emits(&mut k_body, acc_index_emits);
        Self::push_emits(&mut k_body, a_index_emits);
        Self::push_emits(&mut k_body, b_index_emits);

        let (acc_pointer, acc_pointer_emits) =
            self.tile_dynamic_pointer(expressions, op.acc, acc_index)?;
        let (a_pointer, a_pointer_emits) = self.tile_dynamic_pointer(expressions, op.a, a_index)?;
        let (b_pointer, b_pointer_emits) = self.tile_dynamic_pointer(expressions, op.b, b_index)?;
        Self::push_emits(&mut k_body, acc_pointer_emits);
        Self::push_emits(&mut k_body, a_pointer_emits);
        Self::push_emits(&mut k_body, b_pointer_emits);

        let acc_value = expressions.append(
            Expression::Load {
                pointer: acc_pointer,
            },
            Span::default(),
        );
        let a_value = expressions.append(Expression::Load { pointer: a_pointer }, Span::default());
        let b_value = expressions.append(Expression::Load { pointer: b_pointer }, Span::default());
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, acc_value)),
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, a_value)),
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, b_value)),
            Span::default(),
        );
        let product = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Multiply,
                left: a_value,
                right: b_value,
            },
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, product)),
            Span::default(),
        );
        let value = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: acc_value,
                right: product,
            },
            Span::default(),
        );
        k_body.push(
            Statement::Emit(Self::single_expression_range(expressions, value)),
            Span::default(),
        );
        k_body.push(
            Statement::Store {
                pointer: acc_pointer,
                value,
            },
            Span::default(),
        );

        let k_loop = self.counted_loop(expressions, scratch.mma_k, k_a, k_body);
        let j_loop =
            self.counted_loop(expressions, scratch.mma_j, n, Block::from_vec(vec![k_loop]));
        let i_loop =
            self.counted_loop(expressions, scratch.mma_i, m, Block::from_vec(vec![j_loop]));

        Ok(i_loop)
    }

    fn store_zero_to_tile(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        tile: TileRef,
    ) -> Result<Statement, LowerError> {
        self.lower_workgroup_tile_op(expressions, tile_index, tile, |this, expressions, index| {
            let (pointer, emit) = this.tile_index_pointer(expressions, index, tile)?;
            let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
            Ok(Block::from_vec(vec![
                Statement::Emit(emit),
                Statement::Store {
                    pointer,
                    value: zero,
                },
            ]))
        })
    }

    fn lower_cooperative_load(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        dst: TileRef,
        src: StorageView,
    ) -> Result<Statement, LowerError> {
        self.lower_workgroup_tile_op(expressions, tile_index, dst, |this, expressions, index| {
            let src_base = this.storage_base_expression(expressions, src)?;
            let dst_layout = this.tile_layout(dst)?;
            let (dst_pointer, dst_emit) = this.tile_index_pointer(expressions, index, dst)?;
            let (src_pointer, src_emits) = this.storage_index_pointer_from_tile_index_with_base(
                expressions,
                index,
                dst_layout,
                src,
                src_base,
            )?;
            let value = expressions.append(
                Expression::Load {
                    pointer: src_pointer,
                },
                Span::default(),
            );

            let mut body = Block::from_vec(vec![Statement::Emit(dst_emit)]);
            for emit in src_emits {
                body.push(Statement::Emit(emit), Span::default());
            }
            body.push(
                Statement::Emit(Self::single_expression_range(expressions, value)),
                Span::default(),
            );
            body.push(
                Statement::Store {
                    pointer: dst_pointer,
                    value,
                },
                Span::default(),
            );
            Ok(body)
        })
    }

    fn lower_store_tile(
        &self,
        expressions: &mut Arena<Expression>,
        index_local: Handle<LocalVariable>,
        src: TileRef,
        dst: StorageView,
    ) -> Result<Statement, LowerError> {
        let src_layout = self.tile_layout(src)?;
        let dst_layout = self.storage_layout(dst)?;
        if src_layout.shape() != dst_layout.shape() {
            return Err(LowerError::UnsupportedOperation("store shape mismatch"));
        }

        let mut body = Block::new();
        let (flat, flat_emit) = self.load_u32_local(expressions, index_local);
        body.push(Statement::Emit(flat_emit), Span::default());

        let (src_index, src_index_emits) =
            self.index_from_flat(expressions, flat, src_layout, src_layout)?;
        let (dst_index, dst_index_emits) =
            self.index_from_flat(expressions, flat, src_layout, dst_layout)?;
        Self::push_emits(&mut body, src_index_emits);
        Self::push_emits(&mut body, dst_index_emits);

        let (src_pointer, src_pointer_emits) =
            self.tile_dynamic_pointer(expressions, src, src_index)?;
        let (dst_pointer, dst_pointer_emits) =
            self.storage_dynamic_pointer(expressions, dst, dst_index)?;
        Self::push_emits(&mut body, src_pointer_emits);
        Self::push_emits(&mut body, dst_pointer_emits);

        let value = expressions.append(
            Expression::Load {
                pointer: src_pointer,
            },
            Span::default(),
        );
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, value)),
            Span::default(),
        );
        body.push(
            Statement::Store {
                pointer: dst_pointer,
                value,
            },
            Span::default(),
        );

        Ok(self.distributed_index_loop(expressions, index_local, src_layout.element_count(), body))
    }

    fn counted_loop(
        &self,
        expressions: &mut Arena<Expression>,
        index_local: Handle<LocalVariable>,
        end: u32,
        body: Block,
    ) -> Statement {
        let init = self.store_u32_literal(expressions, index_local, 0);
        let (done, done_emit) = Self::u32_done_condition(expressions, index_local, end);
        let mut loop_body = Block::new();
        loop_body.push(Statement::Emit(done_emit), Span::default());
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );
        loop_body.push(Statement::Block(body), Span::default());

        Statement::Block(Block::from_vec(vec![
            init,
            Statement::Loop {
                body: loop_body,
                continuing: Block::from_vec(vec![self.increment_u32_local(
                    expressions,
                    index_local,
                    1,
                )]),
                break_if: None,
            },
        ]))
    }

    fn store_u32_literal(
        &self,
        expressions: &mut Arena<Expression>,
        local: Handle<LocalVariable>,
        value: u32,
    ) -> Statement {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        let value = expressions.append(Expression::Literal(Literal::U32(value)), Span::default());
        Statement::Store { pointer, value }
    }

    fn increment_u32_local(
        &self,
        expressions: &mut Arena<Expression>,
        local: Handle<LocalVariable>,
        amount: u32,
    ) -> Statement {
        let amount = expressions.append(Expression::Literal(Literal::U32(amount)), Span::default());
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        let current = expressions.append(Expression::Load { pointer }, Span::default());
        let next = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: current,
                right: amount,
            },
            Span::default(),
        );
        Statement::Block(Block::from_vec(vec![
            Statement::Emit(Self::range_from(expressions, current, next)),
            Statement::Store {
                pointer,
                value: next,
            },
        ]))
    }

    fn load_u32_local(
        &self,
        expressions: &mut Arena<Expression>,
        local: Handle<LocalVariable>,
    ) -> (Handle<Expression>, Range<Expression>) {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        let value = expressions.append(Expression::Load { pointer }, Span::default());
        (value, Self::single_expression_range(expressions, value))
    }

    fn u32_done_condition(
        expressions: &mut Arena<Expression>,
        index_local: Handle<LocalVariable>,
        end: u32,
    ) -> (Handle<Expression>, Range<Expression>) {
        let end = expressions.append(Expression::Literal(Literal::U32(end)), Span::default());
        let pointer = expressions.append(Expression::LocalVariable(index_local), Span::default());
        let current = expressions.append(Expression::Load { pointer }, Span::default());
        let condition = expressions.append(
            Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left: current,
                right: end,
            },
            Span::default(),
        );

        (condition, Self::range_from(expressions, current, condition))
    }

    fn push_emits(body: &mut Block, emits: Vec<Range<Expression>>) {
        for emit in emits {
            body.push(Statement::Emit(emit), Span::default());
        }
    }

    fn lower_workgroup_tile_op(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        tile: TileRef,
        tile_body: impl FnOnce(
            &Self,
            &mut Arena<Expression>,
            Handle<LocalVariable>,
        ) -> Result<Block, LowerError>,
    ) -> Result<Statement, LowerError> {
        let layout = self.tile_layout(tile)?;
        let body = tile_body(self, expressions, tile_index)?;
        Ok(self.distributed_index_loop(expressions, tile_index, layout.element_count(), body))
    }

    fn distributed_index_loop(
        &self,
        expressions: &mut Arena<Expression>,
        index_local: Handle<LocalVariable>,
        end: std::num::NonZeroU32,
        body: Block,
    ) -> Statement {
        let init_index = self.init_tile_index(expressions, index_local);
        let mut loop_body = Block::new();
        let (done, done_emit) = Self::tile_done_condition(expressions, index_local, end);
        loop_body.push(Statement::Emit(done_emit), Span::default());
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );
        loop_body.push(Statement::Block(body), Span::default());

        let continuing = self.advance_tile_index(expressions, index_local);

        Statement::Block(Block::from_vec(vec![
            init_index,
            Statement::Loop {
                body: loop_body,
                continuing: Block::from_vec(vec![continuing]),
                break_if: None,
            },
        ]))
    }

    fn init_tile_index(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
    ) -> Statement {
        let pointer = expressions.append(Expression::LocalVariable(tile_index), Span::default());
        let lane = expressions.append(
            Expression::FunctionArgument(LOCAL_INVOCATION_INDEX_ARG),
            Span::default(),
        );
        Statement::Store {
            pointer,
            value: lane,
        }
    }

    fn advance_tile_index(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
    ) -> Statement {
        let workgroup_size = expressions.append(
            Expression::Literal(Literal::U32(WORKGROUP_SIZE_X)),
            Span::default(),
        );
        let pointer = expressions.append(Expression::LocalVariable(tile_index), Span::default());
        let current = expressions.append(Expression::Load { pointer }, Span::default());
        let next = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: current,
                right: workgroup_size,
            },
            Span::default(),
        );

        Statement::Block(Block::from_vec(vec![
            Statement::Emit(Self::range_from(expressions, current, next)),
            Statement::Store {
                pointer,
                value: next,
            },
        ]))
    }

    fn tile_done_condition(
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        element_count: std::num::NonZeroU32,
    ) -> (Handle<Expression>, Range<Expression>) {
        let element_count = expressions.append(
            Expression::Literal(Literal::U32(element_count.get())),
            Span::default(),
        );
        let pointer = expressions.append(Expression::LocalVariable(tile_index), Span::default());
        let current = expressions.append(Expression::Load { pointer }, Span::default());
        let condition = expressions.append(
            Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left: current,
                right: element_count,
            },
            Span::default(),
        );

        (condition, Self::range_from(expressions, current, condition))
    }

    fn tile_layout(&self, tile: TileRef) -> Result<&Layout, LowerError> {
        let decl = self
            .ir
            .tiles()
            .get(tile.id.index())
            .ok_or(LowerError::UnknownTile(tile.id))?;
        if decl.element != tile.element {
            return Err(LowerError::TileElementMismatch {
                tile: tile.id,
                declared: decl.element,
                used: tile.element,
            });
        }

        Ok(&decl.layout)
    }

    fn tile_index_pointer(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        tile: TileRef,
    ) -> Result<(Handle<Expression>, Range<Expression>), LowerError> {
        self.tile_layout(tile)?;

        let (storage_tile, offset) = self.storage_tile_and_offset(tile)?;
        let global = self
            .tile_globals
            .get(storage_tile.id.index())
            .copied()
            .flatten()
            .ok_or_else(|| {
                self.tile_layout(storage_tile)
                    .map(|layout| LowerError::UnsupportedMemoryLevel(layout.memory_level()))
                    .unwrap_or(LowerError::UnknownTile(storage_tile.id))
            })?;
        let base = expressions.append(Expression::GlobalVariable(global), Span::default());
        let index_pointer =
            expressions.append(Expression::LocalVariable(tile_index), Span::default());
        let index = expressions.append(
            Expression::Load {
                pointer: index_pointer,
            },
            Span::default(),
        );
        let index = self.add_literal_u32(expressions, index, offset);
        let pointer = expressions.append(Expression::Access { base, index }, Span::default());
        Ok((pointer, Self::range_from(expressions, index, pointer)))
    }

    fn tile_dynamic_pointer(
        &self,
        expressions: &mut Arena<Expression>,
        tile: TileRef,
        index: Handle<Expression>,
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        self.tile_layout(tile)?;

        let base = self.tile_base_expression(expressions, tile)?;
        let (_, offset) = self.storage_tile_and_offset(tile)?;
        let mut emits = Vec::new();
        let index = self.add_literal_u32_emitted(expressions, index, offset, &mut emits);
        let pointer = expressions.append(Expression::Access { base, index }, Span::default());
        emits.push(Self::single_expression_range(expressions, pointer));
        Ok((pointer, emits))
    }

    fn tile_base_expression(
        &self,
        expressions: &mut Arena<Expression>,
        tile: TileRef,
    ) -> Result<Handle<Expression>, LowerError> {
        let (storage_tile, _) = self.storage_tile_and_offset(tile)?;
        let layout = self.tile_layout(storage_tile)?;

        match layout.memory_level() {
            MemoryLevel::Workgroup => {
                let global = self
                    .tile_globals
                    .get(storage_tile.id.index())
                    .copied()
                    .flatten()
                    .ok_or(LowerError::UnknownTile(storage_tile.id))?;
                Ok(expressions.append(Expression::GlobalVariable(global), Span::default()))
            }
            MemoryLevel::Private => {
                let local = self
                    .tile_locals
                    .get(storage_tile.id.index())
                    .copied()
                    .flatten()
                    .ok_or(LowerError::UnknownTile(storage_tile.id))?;
                Ok(expressions.append(Expression::LocalVariable(local), Span::default()))
            }
            memory => Err(LowerError::UnsupportedMemoryLevel(memory)),
        }
    }

    fn storage_index_pointer_from_tile_index_with_base(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        dst_layout: &Layout,
        view: StorageView,
        base: Handle<Expression>,
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        let src_layout = self.storage_layout(view)?;
        if dst_layout.shape() != src_layout.shape() {
            return Err(LowerError::UnsupportedOperation("load shape mismatch"));
        }
        let mut emits = Vec::new();
        let index_pointer =
            expressions.append(Expression::LocalVariable(tile_index), Span::default());
        let flat = expressions.append(
            Expression::Load {
                pointer: index_pointer,
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, flat));
        let logical_index =
            self.storage_index_from_flat(expressions, flat, dst_layout, src_layout, &mut emits)?;
        let index =
            self.add_literal_u32_emitted(expressions, logical_index, view.offset, &mut emits);
        let pointer = expressions.append(Expression::Access { base, index }, Span::default());
        emits.push(Self::single_expression_range(expressions, pointer));
        Ok((pointer, emits))
    }

    fn storage_dynamic_pointer(
        &self,
        expressions: &mut Arena<Expression>,
        view: StorageView,
        index: Handle<Expression>,
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        let base = self.storage_base_expression(expressions, view)?;
        let mut emits = Vec::new();
        let index = self.add_literal_u32_emitted(expressions, index, view.offset, &mut emits);
        let pointer = expressions.append(Expression::Access { base, index }, Span::default());
        emits.push(Self::single_expression_range(expressions, pointer));
        Ok((pointer, emits))
    }

    fn storage_base_expression(
        &self,
        expressions: &mut Arena<Expression>,
        view: StorageView,
    ) -> Result<Handle<Expression>, LowerError> {
        self.storage_layout(view)?;
        let global = self
            .buffer_globals
            .get(view.buffer.id.index())
            .copied()
            .flatten()
            .ok_or(LowerError::UnknownBuffer(view.buffer.id))?;
        Ok(expressions.append(Expression::GlobalVariable(global), Span::default()))
    }

    fn storage_layout(&self, view: StorageView) -> Result<&Layout, LowerError> {
        let decl = self
            .ir
            .buffers()
            .get(view.buffer.id.index())
            .ok_or(LowerError::UnknownBuffer(view.buffer.id))?;
        if decl.element != view.buffer.element {
            return Err(LowerError::UnsupportedOperation("buffer element mismatch"));
        }
        Ok(&decl.layout)
    }

    fn index_from_flat(
        &self,
        expressions: &mut Arena<Expression>,
        flat: Handle<Expression>,
        logical_layout: &Layout,
        target_layout: &Layout,
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        let mut emits = Vec::new();
        let index = self.storage_index_from_flat(
            expressions,
            flat,
            logical_layout,
            target_layout,
            &mut emits,
        )?;
        Ok((index, emits))
    }

    fn layout_index_expr(
        &self,
        expressions: &mut Arena<Expression>,
        layout: &Layout,
        coords: &[Handle<Expression>],
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        if layout.strides().rank() != coords.len() {
            return Err(LowerError::UnsupportedOperation("layout rank mismatch"));
        }
        let mut emits = Vec::new();
        let mut terms = Vec::with_capacity(coords.len());
        for (coord, stride) in coords.iter().zip(layout.strides().values()) {
            terms.push(self.mul_literal_u32_emitted(expressions, *coord, *stride, &mut emits));
        }
        let mut terms = terms.into_iter();
        let Some(mut index) = terms.next() else {
            return Err(LowerError::UnsupportedOperation("zero-rank layout"));
        };
        for term in terms {
            index = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: index,
                    right: term,
                },
                Span::default(),
            );
            emits.push(Self::single_expression_range(expressions, index));
        }
        Ok((index, emits))
    }

    fn storage_index_from_flat(
        &self,
        expressions: &mut Arena<Expression>,
        flat: Handle<Expression>,
        dst_layout: &Layout,
        src_layout: &Layout,
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        match dst_layout.shape().rank() {
            1 => Ok(self.mul_literal_u32_emitted(
                expressions,
                flat,
                src_layout.strides().values()[0],
                emits,
            )),
            2 => {
                let cols = expressions.append(
                    Expression::Literal(Literal::U32(dst_layout.shape().dims()[1].get())),
                    Span::default(),
                );
                let row = expressions.append(
                    Expression::Binary {
                        op: BinaryOperator::Divide,
                        left: flat,
                        right: cols,
                    },
                    Span::default(),
                );
                emits.push(Self::single_expression_range(expressions, row));
                let col = expressions.append(
                    Expression::Binary {
                        op: BinaryOperator::Modulo,
                        left: flat,
                        right: cols,
                    },
                    Span::default(),
                );
                emits.push(Self::single_expression_range(expressions, col));
                let row = self.mul_literal_u32_emitted(
                    expressions,
                    row,
                    src_layout.strides().values()[0],
                    emits,
                );
                let col = self.mul_literal_u32_emitted(
                    expressions,
                    col,
                    src_layout.strides().values()[1],
                    emits,
                );
                let index = expressions.append(
                    Expression::Binary {
                        op: BinaryOperator::Add,
                        left: row,
                        right: col,
                    },
                    Span::default(),
                );
                emits.push(Self::single_expression_range(expressions, index));
                Ok(index)
            }
            _ => Err(LowerError::UnsupportedOperation("rank > 2 storage view")),
        }
    }

    fn storage_tile_and_offset(&self, tile: TileRef) -> Result<(TileRef, u32), LowerError> {
        let decl = self
            .ir
            .tiles()
            .get(tile.id.index())
            .ok_or(LowerError::UnknownTile(tile.id))?;
        if decl.element != tile.element {
            return Err(LowerError::TileElementMismatch {
                tile: tile.id,
                declared: decl.element,
                used: tile.element,
            });
        }

        match decl.origin {
            TileOrigin::Allocation => Ok((tile, 0)),
            TileOrigin::View { source, mapping } => {
                let (root, base_offset) = self.storage_tile_and_offset(source)?;
                let source_layout = self.tile_layout(source)?;
                let local_offset = match mapping {
                    ViewMapping::Partition { origin, .. } => {
                        Self::linear_index_prefix(source_layout, &origin)?
                    }
                };
                Ok((
                    root,
                    base_offset.checked_add(local_offset).ok_or(
                        LowerError::UnsupportedOperation("tile view offset overflow"),
                    )?,
                ))
            }
        }
    }

    fn matrix_shape(layout: &Layout) -> Result<[u32; 2], LowerError> {
        if layout.shape().rank() != 2 {
            return Err(LowerError::UnsupportedOperation("non-matrix mma"));
        }
        Ok([
            layout.shape().dims()[0].get(),
            layout.shape().dims()[1].get(),
        ])
    }

    fn linear_index_prefix(layout: &Layout, coords: &[u32]) -> Result<u32, LowerError> {
        let rank = layout.strides().rank();
        if coords.len() > rank && coords[rank..].iter().any(|coord| *coord != 0) {
            return Err(LowerError::UnsupportedOperation("layout rank mismatch"));
        }
        Ok(coords
            .iter()
            .take(rank)
            .zip(layout.strides().values())
            .map(|(coord, stride)| coord * stride)
            .sum())
    }

    fn add_literal_u32(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        if literal == 0 {
            return value;
        }
        let literal =
            expressions.append(Expression::Literal(Literal::U32(literal)), Span::default());
        expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: value,
                right: literal,
            },
            Span::default(),
        )
    }

    fn add_literal_u32_emitted(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        literal: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        if literal == 0 {
            return value;
        }
        let literal =
            expressions.append(Expression::Literal(Literal::U32(literal)), Span::default());
        let value = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: value,
                right: literal,
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, value));
        value
    }

    fn mul_literal_u32_emitted(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        literal: u32,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        if literal == 1 {
            return value;
        }
        let literal =
            expressions.append(Expression::Literal(Literal::U32(literal)), Span::default());
        let value = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Multiply,
                left: value,
                right: literal,
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, value));
        value
    }

    fn single_expression_range(
        expressions: &Arena<Expression>,
        handle: Handle<Expression>,
    ) -> Range<Expression> {
        Self::range_from(expressions, handle, handle)
    }

    fn range_from(
        expressions: &Arena<Expression>,
        first: Handle<Expression>,
        last: Handle<Expression>,
    ) -> Range<Expression> {
        Range::from_index_range(first.index() as u32..last.index() as u32 + 1, expressions)
    }
}
