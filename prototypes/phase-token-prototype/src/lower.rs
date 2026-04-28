use std::fmt;

use naga::{
    AddressSpace, Arena, Barrier, Block, EntryPoint, Expression, Function, GlobalVariable, Handle,
    Literal, LocalVariable, MemoryDecorations, Module, Range, Scalar, ShaderStage, Span, Statement,
    Type, TypeInner,
};

use crate::{Event, KernelIr, TileId};

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

/// Errors produced by the toy Naga lowering pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LowerError {
    /// The flat event stream contained a loop end with no matching start.
    UnmatchedRangeEnd,
    /// The flat event stream ended with an unclosed loop body.
    UnclosedRange,
    /// An event referenced a tile that was never allocated.
    UnknownTile(TileId),
    /// Naga rejected the generated module.
    Validation(String),
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnmatchedRangeEnd => f.write_str("range end without matching range start"),
            Self::UnclosedRange => f.write_str("range start without matching range end"),
            Self::UnknownTile(tile) => write!(f, "unknown tile {:?}", tile),
            Self::Validation(error) => write!(f, "naga validation failed: {error}"),
        }
    }
}

impl std::error::Error for LowerError {}

struct Lowerer<'a> {
    ir: &'a KernelIr,
    module: Module,
    f32_ty: Handle<Type>,
    tile_globals: Vec<Handle<GlobalVariable>>,
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

        Self {
            ir,
            module,
            f32_ty,
            tile_globals: Vec::new(),
        }
    }

    fn lower(mut self) -> Result<NagaKernel, LowerError> {
        self.create_workgroup_globals();

        let mut function = Function {
            name: Some("main".into()),
            ..Function::default()
        };
        let sink = function.local_variables.append(
            LocalVariable {
                name: Some("read_sink".into()),
                ty: self.f32_ty,
                init: None,
            },
            Span::default(),
        );

        function.body = self.lower_events(&mut function.expressions, sink)?;
        function
            .body
            .push(Statement::Return { value: None }, Span::default());

        self.module.entry_points.push(EntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [1, 1, 1],
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

    fn create_workgroup_globals(&mut self) {
        for tile in 0..self.ir.next_tile {
            let global = self.module.global_variables.append(
                GlobalVariable {
                    name: Some(format!("tile_{tile}")),
                    space: AddressSpace::WorkGroup,
                    binding: None,
                    ty: self.f32_ty,
                    init: None,
                    memory_decorations: MemoryDecorations::empty(),
                },
                Span::default(),
            );
            self.tile_globals.push(global);
        }
    }

    fn lower_events(
        &self,
        expressions: &mut Arena<Expression>,
        sink: Handle<LocalVariable>,
    ) -> Result<Block, LowerError> {
        let mut stack = vec![Block::new()];

        for event in &self.ir.events {
            match *event {
                Event::AllocWorkgroup { .. } | Event::Finish => {}
                Event::CooperativeLoad { tile } => {
                    Self::current_block(&mut stack)
                        .push(self.store_zero_to_tile(expressions, tile)?, Span::default());
                }
                Event::WorkgroupBarrier => {
                    Self::current_block(&mut stack).push(
                        Statement::ControlBarrier(Barrier::WORK_GROUP),
                        Span::default(),
                    );
                }
                Event::ReadReady { tile } => {
                    Self::current_block(&mut stack).push(
                        self.read_tile_to_sink(expressions, sink, tile)?,
                        Span::default(),
                    );
                }
                Event::RangeStepStart => stack.push(Block::new()),
                Event::RangeStepEnd => {
                    if stack.len() == 1 {
                        return Err(LowerError::UnmatchedRangeEnd);
                    }

                    let mut body = stack.pop().expect("checked stack length");
                    body.push(Statement::Break, Span::default());
                    Self::current_block(&mut stack).push(
                        Statement::Loop {
                            body,
                            continuing: Block::new(),
                            break_if: None,
                        },
                        Span::default(),
                    );
                }
            }
        }

        if stack.len() != 1 {
            return Err(LowerError::UnclosedRange);
        }

        Ok(stack.pop().expect("checked stack length"))
    }

    fn store_zero_to_tile(
        &self,
        expressions: &mut Arena<Expression>,
        tile: TileId,
    ) -> Result<Statement, LowerError> {
        let pointer = self.tile_pointer(expressions, tile)?;
        let zero = expressions.append(Expression::Literal(Literal::F32(0.0)), Span::default());
        Ok(Statement::Store {
            pointer,
            value: zero,
        })
    }

    fn read_tile_to_sink(
        &self,
        expressions: &mut Arena<Expression>,
        sink: Handle<LocalVariable>,
        tile: TileId,
    ) -> Result<Statement, LowerError> {
        let tile_pointer = self.tile_pointer(expressions, tile)?;
        let value = expressions.append(
            Expression::Load {
                pointer: tile_pointer,
            },
            Span::default(),
        );
        let sink_pointer = expressions.append(Expression::LocalVariable(sink), Span::default());

        Ok(Statement::Block(Block::from_vec(vec![
            Statement::Emit(Self::single_expression_range(expressions, value)),
            Statement::Store {
                pointer: sink_pointer,
                value,
            },
        ])))
    }

    fn tile_pointer(
        &self,
        expressions: &mut Arena<Expression>,
        tile: TileId,
    ) -> Result<Handle<Expression>, LowerError> {
        let global = self
            .tile_globals
            .get(tile.0 as usize)
            .copied()
            .ok_or(LowerError::UnknownTile(tile))?;
        Ok(expressions.append(Expression::GlobalVariable(global), Span::default()))
    }

    fn single_expression_range(
        expressions: &Arena<Expression>,
        handle: Handle<Expression>,
    ) -> Range<Expression> {
        let start = handle.index() as u32;
        Range::from_index_range(start..start + 1, expressions)
    }

    fn current_block(stack: &mut [Block]) -> &mut Block {
        stack.last_mut().expect("lowering stack is never empty")
    }
}
