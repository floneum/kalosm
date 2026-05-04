use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn new(ir: &'a KernelIr) -> Self {
        let mut module = Module::default();
        let f32_ty = module.types.insert(
            Type {
                name: Some("TileElement".into()),
                inner: TypeInner::Scalar(Scalar::F32),
            },
            Span::default(),
        );
        let f32_vec4_ty = module.types.insert(
            Type {
                name: Some("Dot4".into()),
                inner: TypeInner::Vector {
                    size: VectorSize::Quad,
                    scalar: Scalar::F32,
                },
            },
            Span::default(),
        );
        let i32_ty = module.types.insert(
            Type {
                name: Some("Signed".into()),
                inner: TypeInner::Scalar(Scalar {
                    kind: ScalarKind::Sint,
                    width: 4,
                }),
            },
            Span::default(),
        );
        let i32_vec4_ty = module.types.insert(
            Type {
                name: Some("PackedI8".into()),
                inner: TypeInner::Vector {
                    size: VectorSize::Quad,
                    scalar: Scalar {
                        kind: ScalarKind::Sint,
                        width: 4,
                    },
                },
            },
            Span::default(),
        );
        let uses_f16 = ir
            .buffers()
            .iter()
            .any(|buffer| buffer.element == ElementType::F16)
            || ir
                .tiles()
                .iter()
                .any(|tile| tile.element == ElementType::F16)
            || Self::tile_programs_use_f16(ir);
        let f16_ty = uses_f16.then(|| {
            module.types.insert(
                Type {
                    name: Some("TileElementF16".into()),
                    inner: TypeInner::Scalar(Scalar {
                        kind: ScalarKind::Float,
                        width: 2,
                    }),
                },
                Span::default(),
            )
        });
        let u32_ty = module.types.insert(
            Type {
                name: Some("Index".into()),
                inner: TypeInner::Scalar(Scalar::U32),
            },
            Span::default(),
        );
        let u32_vec3_ty = module.types.insert(
            Type {
                name: Some("WorkgroupId".into()),
                inner: TypeInner::Vector {
                    size: VectorSize::Tri,
                    scalar: Scalar::U32,
                },
            },
            Span::default(),
        );
        let uses_subgroup_reduce = Self::uses_subgroup_reduce(ir);
        let uses_cooperative_matrix = Self::uses_cooperative_matrix(ir);
        let uses_subgroup_id_idx =
            Self::uses_index_kind(ir, super::analysis::SubgroupIndexKind::SubgroupId);
        let uses_subgroup_lane_idx =
            Self::uses_index_kind(ir, super::analysis::SubgroupIndexKind::SubgroupLane);
        let uses_subgroup_size_idx =
            Self::uses_index_kind(ir, super::analysis::SubgroupIndexKind::SubgroupSize);
        let uses_num_subgroups_idx =
            Self::uses_index_kind(ir, super::analysis::SubgroupIndexKind::NumSubgroups);
        let uses_subgroup_id =
            uses_subgroup_reduce || uses_subgroup_id_idx || uses_cooperative_matrix;
        let uses_subgroup_invocation_id = uses_subgroup_lane_idx;
        let uses_subgroup_size = uses_subgroup_size_idx;
        let uses_num_subgroups = uses_num_subgroups_idx;

        let coop_c_ty = uses_cooperative_matrix.then(|| {
            module.types.insert(
                Type {
                    name: Some("CoopC8x8F32".into()),
                    inner: TypeInner::CooperativeMatrix {
                        columns: naga::CooperativeSize::Eight,
                        rows: naga::CooperativeSize::Eight,
                        scalar: Scalar::F32,
                        role: naga::CooperativeRole::C,
                    },
                },
                Span::default(),
            )
        });

        let tile_program_block = Self::max_tile_program_block(ir);
        let (workgroup_invocations, workgroup_size) = if tile_program_block > 0 {
            (tile_program_block, [tile_program_block, 1, 1])
        } else {
            (DEFAULT_WORKGROUP_INVOCATIONS, DEFAULT_WORKGROUP_SIZE)
        };
        let live_tiles = Self::live_tiles(ir);

        Self {
            ir,
            module,
            f32_ty,
            f32_vec4_ty,
            i32_ty,
            i32_vec4_ty,
            f16_ty,
            u32_ty,
            u32_vec3_ty,
            buffer_globals: Vec::new(),
            tile_globals: Vec::new(),
            tile_locals: Vec::new(),
            live_tiles,
            loop_index_local: None,
            workgroup_invocations,
            workgroup_size,
            uses_subgroup_id,
            uses_subgroup_invocation_id,
            uses_subgroup_size,
            uses_num_subgroups,
            block_dequant_cache: std::cell::RefCell::new(std::collections::HashMap::new()),
            pin_cache: std::cell::RefCell::new(std::collections::HashMap::new()),
            q8_activation_pack_cache: std::cell::RefCell::new(std::collections::HashMap::new()),
            loop_fold_group_cache: std::cell::RefCell::new(std::collections::HashMap::new()),
            fold_accumulator_locals: Vec::new(),
            fold_group_offsets: Vec::new(),
            coop_c_ty,
            coop_acc_locals: Vec::new(),
            coop_fragment_cache: std::cell::RefCell::new(std::collections::HashMap::new()),
            coop_acc_value_cache: std::cell::RefCell::new(std::collections::HashMap::new()),
            uses_cooperative_matrix,
        }
    }

    pub(super) fn lower(mut self) -> Result<NagaKernel, LowerError> {
        self.create_storage_globals();
        self.create_workgroup_globals()?;

        let mut arguments = vec![
            FunctionArgument {
                name: Some("local_invocation_index".into()),
                ty: self.u32_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)),
            },
            FunctionArgument {
                name: Some("workgroup_id".into()),
                ty: self.u32_vec3_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::WorkGroupId)),
            },
        ];
        if self.uses_subgroup_id {
            arguments.push(FunctionArgument {
                name: Some("subgroup_id".into()),
                ty: self.u32_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::SubgroupId)),
            });
        }
        if self.uses_subgroup_invocation_id {
            arguments.push(FunctionArgument {
                name: Some("subgroup_invocation_id".into()),
                ty: self.u32_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::SubgroupInvocationId)),
            });
        }
        if self.uses_subgroup_size {
            arguments.push(FunctionArgument {
                name: Some("subgroup_size".into()),
                ty: self.u32_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::SubgroupSize)),
            });
        }
        if self.uses_num_subgroups {
            arguments.push(FunctionArgument {
                name: Some("num_subgroups".into()),
                ty: self.u32_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::NumSubgroups)),
            });
        }

        let mut function = Function {
            name: Some("main".into()),
            arguments,
            ..Function::default()
        };
        let scratch = self.create_scratch_locals(&mut function);
        self.loop_index_local = Some(scratch.loop_index);
        self.create_private_locals(&mut function)?;
        self.create_fold_group_locals(&mut function);
        self.create_coop_acc_locals(&mut function);

        function.body = self.lower_block(self.ir.body(), &mut function.expressions, scratch)?;
        function
            .body
            .push(Statement::Return { value: None }, Span::default());

        self.module.entry_points.push(EntryPoint {
            name: "main".into(),
            stage: ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: self.workgroup_size,
            workgroup_size_overrides: None,
            function,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });

        let mut capabilities = naga::valid::Capabilities::empty();
        if self.f16_ty.is_some() {
            capabilities |= naga::valid::Capabilities::SHADER_FLOAT16;
        }
        if self.uses_subgroup_id {
            capabilities |= naga::valid::Capabilities::SUBGROUP;
        }
        if self.uses_cooperative_matrix {
            capabilities |= naga::valid::Capabilities::COOPERATIVE_MATRIX;
        }
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
            .validate(&self.module)
            .map_err(|error| LowerError::Validation(format!("{error:#?}")))?;

        Ok(NagaKernel {
            module: self.module,
            info,
        })
    }

    pub(super) fn create_storage_globals(&mut self) {
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
                },
                Span::default(),
            );
            self.buffer_globals[buffer.id.index()] = Some(global);
        }
    }

    pub(super) fn create_workgroup_globals(&mut self) -> Result<(), LowerError> {
        self.tile_globals = vec![None; self.ir.tiles().len()];
        for tile in self.ir.tiles() {
            if !self
                .live_tiles
                .get(tile.id.index())
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
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
                },
                Span::default(),
            );
            self.tile_globals[tile.id.index()] = Some(global);
        }
        Ok(())
    }

    pub(super) fn create_private_locals(
        &mut self,
        function: &mut Function,
    ) -> Result<(), LowerError> {
        self.tile_locals = vec![None; self.ir.tiles().len()];
        for tile in self.ir.tiles() {
            if !self
                .live_tiles
                .get(tile.id.index())
                .copied()
                .unwrap_or(false)
            {
                continue;
            }
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

    pub(super) fn create_scratch_locals(&self, function: &mut Function) -> ScratchLocals {
        ScratchLocals {
            loop_index: self.create_u32_local(function, "loop_index"),
            values: [
                self.create_f32_local(function, "tile_value_f32"),
                self.create_f16_local(function, "tile_value_f16"),
                self.create_u32_local(function, "tile_value_u32"),
            ],
            spills: [
                std::array::from_fn(|index| {
                    self.create_f32_local(function, &format!("tile_spill_f32_{index}"))
                }),
                std::array::from_fn(|index| {
                    self.create_f16_local(function, &format!("tile_spill_f16_{index}"))
                }),
                std::array::from_fn(|index| {
                    self.create_u32_local(function, &format!("tile_spill_u32_{index}"))
                }),
            ],
            block_dequant: std::array::from_fn(|index| {
                self.create_f32_local(function, &format!("tile_block_dequant_{index}"))
            }),
            q8_activation_scales: std::array::from_fn(|index| {
                self.create_f32_local(function, &format!("tile_q8_activation_scale_{index}"))
            }),
            q8_activation_packs: std::array::from_fn(|index| {
                self.create_u32_local(function, &format!("tile_q8_activation_pack_{index}"))
            }),
            q8_activation_sums_i32: std::array::from_fn(|index| {
                self.create_i32_local(function, &format!("tile_q8_activation_sum_{index}"))
            }),
        }
    }

    pub(super) fn create_coop_acc_locals(&mut self, function: &mut Function) {
        if let Some(ty) = self.coop_c_ty {
            self.coop_acc_locals = self
                .ir
                .coop_accs
                .iter()
                .map(|decl| {
                    self.create_local(function, &format!("coop_acc_{}", decl.id.index()), ty)
                })
                .collect();
        }
    }

    pub(super) fn create_fold_group_locals(&mut self, function: &mut Function) {
        let mut offsets = Vec::with_capacity(self.ir.loop_fold_groups.len());
        let mut total = 0usize;
        for group in &self.ir.loop_fold_groups {
            offsets.push(total);
            total += group.initials.len();
        }
        self.fold_group_offsets = offsets;
        self.fold_accumulator_locals = (0..total)
            .map(|i| self.create_f32_local(function, &format!("fold_acc_{i}")))
            .collect();
    }

    pub(super) fn create_u32_local(
        &self,
        function: &mut Function,
        name: &str,
    ) -> Handle<LocalVariable> {
        function.local_variables.append(
            LocalVariable {
                name: Some(name.into()),
                ty: self.u32_ty,
                init: None,
            },
            Span::default(),
        )
    }

    pub(super) fn create_f32_local(
        &self,
        function: &mut Function,
        name: &str,
    ) -> Handle<LocalVariable> {
        self.create_local(function, name, self.f32_ty)
    }

    pub(super) fn create_i32_local(
        &self,
        function: &mut Function,
        name: &str,
    ) -> Handle<LocalVariable> {
        self.create_local(function, name, self.i32_ty)
    }

    pub(super) fn create_f16_local(
        &self,
        function: &mut Function,
        name: &str,
    ) -> Handle<LocalVariable> {
        self.create_local(function, name, self.f16_ty.unwrap_or(self.f32_ty))
    }

    pub(super) fn create_local(
        &self,
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

    pub(super) fn tile_type(
        &mut self,
        tile: usize,
        element: ElementType,
        layout: &Layout,
    ) -> Handle<Type> {
        self.array_type(format!("Tile{tile}"), element, layout)
    }

    pub(super) fn storage_type(
        &mut self,
        buffer: usize,
        element: ElementType,
        _layout: &Layout,
    ) -> Handle<Type> {
        self.array_type_with_size(format!("Buffer{buffer}"), element, ArraySize::Dynamic)
    }

    pub(super) fn array_type(
        &mut self,
        name: String,
        element: ElementType,
        layout: &Layout,
    ) -> Handle<Type> {
        self.array_type_with_size(
            name,
            element,
            ArraySize::Constant(layout.allocation_element_count()),
        )
    }

    pub(super) fn array_type_with_size(
        &mut self,
        name: String,
        element: ElementType,
        size: ArraySize,
    ) -> Handle<Type> {
        let base = match element {
            ElementType::F32 => self.f32_ty,
            ElementType::F16 => self
                .f16_ty
                .expect("f16 buffer or tile requested without f16 type"),
            ElementType::U32 => self.u32_ty,
        };
        let stride = match element {
            ElementType::F16 => 2,
            ElementType::F32 | ElementType::U32 => 4,
        };

        self.module.types.insert(
            Type {
                name: Some(name),
                inner: TypeInner::Array { base, size, stride },
            },
            Span::default(),
        )
    }

    fn tile_programs_use_f16(ir: &KernelIr) -> bool {
        ir.body().ops().iter().any(|op| {
            let Op::TileProgram(op) = op;
            op.body.iter().any(Self::tile_stmt_uses_f16)
        })
    }

    fn tile_stmt_uses_f16(stmt: &TileStmt) -> bool {
        match stmt {
            TileStmt::Store(store) => {
                store.dst.buffer.element == ElementType::F16
                    || Self::tile_index_expr_uses_f16(&store.row)
                    || Self::tile_index_expr_uses_f16(&store.col)
                    || Self::tile_mask_expr_uses_f16(&store.mask)
                    || Self::tile_expr_uses_f16(&store.value)
            }
            TileStmt::CopyToWorkgroupTile {
                dst,
                src,
                row_offset,
                col_offset,
            } => {
                dst.element == ElementType::F16
                    || src.buffer.element == ElementType::F16
                    || Self::tile_index_expr_uses_f16(row_offset)
                    || Self::tile_index_expr_uses_f16(col_offset)
            }
            TileStmt::CopyQuantToWorkgroupTile {
                row_offset,
                col_offset,
                ..
            } => {
                Self::tile_index_expr_uses_f16(row_offset)
                    || Self::tile_index_expr_uses_f16(col_offset)
            }
            TileStmt::LoadCoop { row, col, .. } => {
                Self::tile_index_expr_uses_f16(row) || Self::tile_index_expr_uses_f16(col)
            }
            TileStmt::StoreCoopAcc { dst, row, col, .. } => {
                dst.buffer.element == ElementType::F16
                    || Self::tile_index_expr_uses_f16(row)
                    || Self::tile_index_expr_uses_f16(col)
            }
            TileStmt::WhileTrue { body, .. } => body.iter().any(Self::tile_stmt_uses_f16),
            TileStmt::ZeroCoopAcc { .. } | TileStmt::Barrier | TileStmt::Mma { .. } => false,
        }
    }

    fn tile_expr_uses_f16(expr: &TileExpr) -> bool {
        match expr {
            TileExpr::Load(load) => {
                load.src.buffer.element == ElementType::F16
                    || load.fill.element() == ElementType::F16
                    || Self::tile_index_expr_uses_f16(&load.row)
                    || Self::tile_index_expr_uses_f16(&load.col)
                    || Self::tile_mask_expr_uses_f16(&load.mask)
            }
            TileExpr::QuantizedLoad(load) => {
                Self::tile_index_expr_uses_f16(&load.row)
                    || Self::tile_index_expr_uses_f16(&load.col)
                    || Self::tile_mask_expr_uses_f16(&load.mask)
            }
            TileExpr::Full(_) | TileExpr::Index(_) => false,
            TileExpr::Literal(value) => value.element() == ElementType::F16,
            TileExpr::Scalar(expr) => Self::tile_scalar_expr_uses_f16(expr),
            TileExpr::Unary { value, .. } => Self::tile_expr_uses_f16(value),
            TileExpr::Cast { value, to } => {
                *to == ElementType::F16 || Self::tile_expr_uses_f16(value)
            }
            TileExpr::Binary { left, right, .. } => {
                Self::tile_expr_uses_f16(left) || Self::tile_expr_uses_f16(right)
            }
            TileExpr::Compare {
                left,
                right,
                output,
                ..
            } => {
                *output == ElementType::F16
                    || Self::tile_expr_uses_f16(left)
                    || Self::tile_expr_uses_f16(right)
            }
            TileExpr::Select {
                condition,
                accept,
                reject,
            } => {
                Self::tile_expr_uses_f16(condition)
                    || Self::tile_expr_uses_f16(accept)
                    || Self::tile_expr_uses_f16(reject)
            }
            TileExpr::LoopFold { value, initial, .. } => {
                initial.element() == ElementType::F16 || Self::tile_expr_uses_f16(value)
            }
            TileExpr::GroupReduce { value, scratch, .. } => {
                scratch.element == ElementType::F16 || Self::tile_expr_uses_f16(value)
            }
            TileExpr::SubgroupReduce { value, .. } => Self::tile_expr_uses_f16(value),
            TileExpr::QuantizedBlockLane {
                k_base, col, mask, ..
            } => {
                Self::tile_index_expr_uses_f16(k_base)
                    || Self::tile_index_expr_uses_f16(col)
                    || Self::tile_mask_expr_uses_f16(mask)
            }
            TileExpr::Dot4 { a, b } => a
                .iter()
                .chain(b.iter())
                .any(|expr| Self::tile_expr_uses_f16(expr)),
            TileExpr::QuantizedQ8_0Dot8 {
                a,
                k_base,
                col,
                mask,
                ..
            } => {
                a.iter().any(|expr| Self::tile_expr_uses_f16(expr))
                    || Self::tile_index_expr_uses_f16(k_base)
                    || Self::tile_index_expr_uses_f16(col)
                    || Self::tile_mask_expr_uses_f16(mask)
            }
            TileExpr::QuantizedQ8ActivationDot {
                a,
                k_base,
                col,
                mask,
                ..
            } => {
                a.iter().any(|expr| Self::tile_expr_uses_f16(expr))
                    || Self::tile_index_expr_uses_f16(k_base)
                    || Self::tile_index_expr_uses_f16(col)
                    || Self::tile_mask_expr_uses_f16(mask)
            }
            TileExpr::PinnedRef { .. } | TileExpr::LoopFoldGroupOutput { .. } => false,
        }
    }

    fn tile_scalar_expr_uses_f16(expr: &TileScalarExpr) -> bool {
        match expr {
            TileScalarExpr::Reduce { value, scratch, .. }
            | TileScalarExpr::LoopReduce { value, scratch, .. } => {
                scratch.element == ElementType::F16 || Self::tile_expr_uses_f16(value)
            }
            TileScalarExpr::Literal(value) => value.element() == ElementType::F16,
        }
    }

    fn tile_index_expr_uses_f16(expr: &TileIndexExpr) -> bool {
        match expr {
            TileIndexExpr::Lane
            | TileIndexExpr::LoopIndex
            | TileIndexExpr::ProgramId(_)
            | TileIndexExpr::SubgroupId
            | TileIndexExpr::SubgroupLane
            | TileIndexExpr::SubgroupSize
            | TileIndexExpr::NumSubgroups
            | TileIndexExpr::Literal(_) => false,
            TileIndexExpr::Add(left, right) => {
                Self::tile_index_expr_uses_f16(left) || Self::tile_index_expr_uses_f16(right)
            }
            TileIndexExpr::Mul(value, _)
            | TileIndexExpr::Div(value, _)
            | TileIndexExpr::Mod(value, _) => Self::tile_index_expr_uses_f16(value),
            TileIndexExpr::Value(value) => Self::tile_expr_uses_f16(value),
        }
    }

    fn tile_mask_expr_uses_f16(expr: &TileMaskExpr) -> bool {
        match expr {
            TileMaskExpr::True => false,
            TileMaskExpr::Compare { left, right, .. } => {
                Self::tile_index_expr_uses_f16(left) || Self::tile_index_expr_uses_f16(right)
            }
            TileMaskExpr::And(left, right) => {
                Self::tile_mask_expr_uses_f16(left) || Self::tile_mask_expr_uses_f16(right)
            }
        }
    }
}
