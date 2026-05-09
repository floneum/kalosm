use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn new(ir: &'a KernelIr) -> Self {
        let mut module = Module::default();
        let i32_scalar = Scalar {
            kind: ScalarKind::Sint,
            width: 4,
        };
        let f32_ty = Self::scalar_type(&mut module, Scalar::F32);
        let f32_vec4_ty = Self::vector_type(&mut module, VectorSize::Quad, Scalar::F32);
        let i32_ty = Self::scalar_type(&mut module, i32_scalar);
        let i32_vec4_ty = Self::vector_type(&mut module, VectorSize::Quad, i32_scalar);
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
            Self::scalar_type(
                &mut module,
                Scalar {
                    kind: ScalarKind::Float,
                    width: 2,
                },
            )
        });
        let u32_ty = Self::scalar_type(&mut module, Scalar::U32);
        let bool_ty = Self::scalar_type(&mut module, Scalar::BOOL);
        let u32_vec3_ty = Self::vector_type(&mut module, VectorSize::Tri, Scalar::U32);
        let uses_cooperative_matrix = Self::uses_cooperative_matrix(ir);
        let uses_subgroup_id_idx =
            Self::uses_index_kind(ir, super::analysis::SubgroupIndexKind::SubgroupId);
        let uses_subgroup_lane_idx =
            Self::uses_index_kind(ir, super::analysis::SubgroupIndexKind::SubgroupLane);
        let uses_subgroup_size_idx =
            Self::uses_index_kind(ir, super::analysis::SubgroupIndexKind::SubgroupSize);
        let uses_num_subgroups_idx =
            Self::uses_index_kind(ir, super::analysis::SubgroupIndexKind::NumSubgroups);
        let uses_subgroup_id = uses_subgroup_id_idx || uses_cooperative_matrix;
        let uses_subgroup_invocation_id = uses_subgroup_lane_idx;
        let uses_subgroup_size = uses_subgroup_size_idx;
        let uses_num_subgroups = uses_num_subgroups_idx;

        let coop_c_ty = uses_cooperative_matrix.then(|| {
            Self::type_with_inner(
                &mut module,
                TypeInner::CooperativeMatrix {
                    columns: naga::CooperativeSize::Eight,
                    rows: naga::CooperativeSize::Eight,
                    scalar: Scalar::F32,
                    role: naga::CooperativeRole::C,
                },
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
            bool_ty,
            u32_vec3_ty,
            buffer_globals: Vec::new(),
            tile_globals: Vec::new(),
            tile_locals: Vec::new(),
            private_locals: Vec::new(),
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

    fn scalar_type(module: &mut Module, scalar: Scalar) -> Handle<Type> {
        Self::type_with_inner(module, TypeInner::Scalar(scalar))
    }

    fn vector_type(module: &mut Module, size: VectorSize, scalar: Scalar) -> Handle<Type> {
        Self::type_with_inner(module, TypeInner::Vector { size, scalar })
    }

    fn type_with_inner(module: &mut Module, inner: TypeInner) -> Handle<Type> {
        module
            .types
            .insert(Type { name: None, inner }, Span::default())
    }

    pub(super) fn lower(mut self) -> Result<NagaKernel, LowerError> {
        self.create_storage_globals();
        self.create_workgroup_globals()?;

        let mut arguments = vec![
            FunctionArgument {
                name: None,
                ty: self.u32_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::LocalInvocationIndex)),
            },
            FunctionArgument {
                name: None,
                ty: self.u32_vec3_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::WorkGroupId)),
            },
        ];
        if self.uses_subgroup_id {
            arguments.push(FunctionArgument {
                name: None,
                ty: self.u32_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::SubgroupId)),
            });
        }
        if self.uses_subgroup_invocation_id {
            arguments.push(FunctionArgument {
                name: None,
                ty: self.u32_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::SubgroupInvocationId)),
            });
        }
        if self.uses_subgroup_size {
            arguments.push(FunctionArgument {
                name: None,
                ty: self.u32_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::SubgroupSize)),
            });
        }
        if self.uses_num_subgroups {
            arguments.push(FunctionArgument {
                name: None,
                ty: self.u32_ty,
                binding: Some(Binding::BuiltIn(BuiltIn::NumSubgroups)),
            });
        }

        let mut function = Function {
            name: None,
            arguments,
            ..Function::default()
        };
        let scratch = self.create_scratch_locals(&mut function);
        self.loop_index_local = Some(scratch.loop_index);
        self.create_private_locals(&mut function)?;
        self.create_program_private_locals(&mut function);
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
        if Self::uses_subgroup_reduce(self.ir) || self.uses_subgroup_id {
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
                    name: None,
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
                    name: None,
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
                    name: None,
                    ty,
                    init: None,
                },
                Span::default(),
            );
            self.tile_locals[tile.id.index()] = Some(local);
        }
        Ok(())
    }

    pub(super) fn create_program_private_locals(&mut self, function: &mut Function) {
        self.private_locals = vec![None; self.ir.locals().len()];
        for local in self.ir.locals() {
            let ty = self.element_type(local.element);
            let handle = function.local_variables.append(
                LocalVariable {
                    name: None,
                    ty,
                    init: None,
                },
                Span::default(),
            );
            self.private_locals[local.id.index()] = Some(handle);
        }
    }

    pub(super) fn create_scratch_locals(&self, function: &mut Function) -> ScratchLocals {
        let full_scratch = Self::tile_programs_need_full_scratch(self.ir);
        if !full_scratch && !Self::tile_programs_need_value_scratch(self.ir) {
            let dummy = self.create_f32_local(function);
            let loop_index = self.create_u32_local(function);
            return ScratchLocals {
                loop_index,
                values: [dummy; 5],
                spills: [[dummy; 32]; 5],
                block_dequant: [dummy; 16],
                q8_activation_scales: [dummy; 4],
                q8_activation_packs: [dummy; 4],
                q8_activation_sums_i32: [dummy; 4],
            };
        }
        let f32_value = self.create_f32_local(function);
        let f16_value = self.create_f16_local(function);
        let u32_value = self.create_u32_local(function);
        let f32_vec4_value = self.create_local(function, self.f32_vec4_ty);
        let bool_value = self.create_local(function, self.bool_ty);
        ScratchLocals {
            loop_index: self.create_u32_local(function),
            values: [f32_value, f16_value, u32_value, f32_vec4_value, bool_value],
            spills: if full_scratch {
                [
                    std::array::from_fn(|_| self.create_f32_local(function)),
                    std::array::from_fn(|_| self.create_f16_local(function)),
                    std::array::from_fn(|_| self.create_u32_local(function)),
                    std::array::from_fn(|_| self.create_local(function, self.f32_vec4_ty)),
                    std::array::from_fn(|_| self.create_local(function, self.bool_ty)),
                ]
            } else {
                [
                    [f32_value; 32],
                    [f16_value; 32],
                    [u32_value; 32],
                    [f32_vec4_value; 32],
                    [bool_value; 32],
                ]
            },
            block_dequant: if full_scratch {
                std::array::from_fn(|_| self.create_f32_local(function))
            } else {
                [f32_value; 16]
            },
            q8_activation_scales: if full_scratch {
                std::array::from_fn(|_| self.create_f32_local(function))
            } else {
                [f32_value; 4]
            },
            q8_activation_packs: if full_scratch {
                std::array::from_fn(|_| self.create_u32_local(function))
            } else {
                [u32_value; 4]
            },
            q8_activation_sums_i32: if full_scratch {
                std::array::from_fn(|_| self.create_i32_local(function))
            } else {
                [self.create_i32_local(function); 4]
            },
        }
    }

    fn tile_programs_need_full_scratch(ir: &KernelIr) -> bool {
        Self::tile_programs_expr_any(ir, Self::tile_expr_needs_full_scratch)
    }

    fn tile_programs_need_value_scratch(ir: &KernelIr) -> bool {
        Self::tile_programs_expr_any(ir, Self::tile_expr_needs_value_scratch)
    }

    fn tile_expr_needs_value_scratch(expr: &TileExpr) -> bool {
        match expr {
            TileExpr::Load(load) => !matches!(load.mask, TileMaskExpr::True),
            TileExpr::LoadLinear(load) => !matches!(load.mask, TileMaskExpr::True),
            TileExpr::LoadVec4(load) => !matches!(load.mask, TileMaskExpr::True),
            TileExpr::QuantizedLoad(load) => !matches!(load.mask, TileMaskExpr::True),
            _ => Self::tile_expr_children_any(expr, Self::tile_expr_needs_value_scratch),
        }
    }

    fn tile_expr_needs_full_scratch(expr: &TileExpr) -> bool {
        match expr {
            TileExpr::QuantizedBlockLane { .. }
            | TileExpr::QuantizedQ8_0Dot8 { .. }
            | TileExpr::QuantizedVecDot { .. }
            | TileExpr::QuantizedQ4KGgmlDot { .. }
            | TileExpr::QuantizedQ6KGgmlDot { .. }
            | TileExpr::LoopFold { .. } => true,
            TileExpr::Scalar(TileScalarExpr::LoopReduce { .. }) => true,
            _ => Self::tile_expr_children_any(expr, Self::tile_expr_needs_full_scratch),
        }
    }

    pub(super) fn create_coop_acc_locals(&mut self, function: &mut Function) {
        if let Some(ty) = self.coop_c_ty {
            self.coop_acc_locals = self
                .ir
                .coop_accs
                .iter()
                .map(|_| self.create_local(function, ty))
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
            .map(|_| self.create_f32_local(function))
            .collect();
    }

    pub(super) fn create_u32_local(&self, function: &mut Function) -> Handle<LocalVariable> {
        function.local_variables.append(
            LocalVariable {
                name: None,
                ty: self.u32_ty,
                init: None,
            },
            Span::default(),
        )
    }

    pub(super) fn create_f32_local(&self, function: &mut Function) -> Handle<LocalVariable> {
        self.create_local(function, self.f32_ty)
    }

    pub(super) fn create_i32_local(&self, function: &mut Function) -> Handle<LocalVariable> {
        self.create_local(function, self.i32_ty)
    }

    pub(super) fn create_f16_local(&self, function: &mut Function) -> Handle<LocalVariable> {
        self.create_local(function, self.f16_ty.unwrap_or(self.f32_ty))
    }

    pub(super) fn element_type(&self, element: ElementType) -> Handle<Type> {
        match element {
            ElementType::F32 => self.f32_ty,
            ElementType::F16 => self
                .f16_ty
                .expect("f16 buffer or tile requested without f16 type"),
            ElementType::U32 => self.u32_ty,
            ElementType::F32Vec4 => self.f32_vec4_ty,
            ElementType::Bool => self.bool_ty,
        }
    }

    pub(super) fn create_local(
        &self,
        function: &mut Function,
        ty: Handle<Type>,
    ) -> Handle<LocalVariable> {
        function.local_variables.append(
            LocalVariable {
                name: None,
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    pub(super) fn tile_type(
        &mut self,
        _tile: usize,
        element: ElementType,
        layout: &Layout,
    ) -> Handle<Type> {
        self.array_type(element, layout)
    }

    pub(super) fn storage_type(
        &mut self,
        _buffer: usize,
        element: ElementType,
        _layout: &Layout,
    ) -> Handle<Type> {
        self.array_type_with_size(element, ArraySize::Dynamic)
    }

    pub(super) fn array_type(&mut self, element: ElementType, layout: &Layout) -> Handle<Type> {
        self.array_type_with_size(
            element,
            ArraySize::Constant(layout.allocation_element_count()),
        )
    }

    pub(super) fn array_type_with_size(
        &mut self,
        element: ElementType,
        size: ArraySize,
    ) -> Handle<Type> {
        let base = match element {
            ElementType::Bool => panic!("bool arrays are not supported"),
            _ => self.element_type(element),
        };
        let stride = match element {
            ElementType::F16 => 2,
            ElementType::F32 | ElementType::U32 => 4,
            ElementType::F32Vec4 => 16,
            ElementType::Bool => panic!("bool arrays are not supported"),
        };

        self.module.types.insert(
            Type {
                name: None,
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
        if Self::tile_stmt_f16_payload(stmt) {
            return true;
        }
        let mut index_uses_f16 = |expr: &TileIndexExpr| Self::tile_index_expr_uses_f16(expr);
        if Self::tile_stmt_index_any(stmt, &mut index_uses_f16) {
            return true;
        }
        let mut expr_uses_f16 = |expr: &TileExpr| Self::tile_expr_uses_f16(expr);
        Self::tile_stmt_expr_any(stmt, &mut expr_uses_f16)
    }

    fn tile_stmt_f16_payload(stmt: &TileStmt) -> bool {
        match stmt {
            TileStmt::Store(store) => store.dst.buffer.element == ElementType::F16,
            TileStmt::StoreIndexed(store) => store.dst.buffer.element == ElementType::F16,
            TileStmt::StoreLocal { dst, value } => {
                dst.element == ElementType::F16 || Self::tile_expr_uses_f16(value)
            }
            TileStmt::Emit { .. } => false,
            TileStmt::StoreWorkgroup { dst, .. } => dst.element == ElementType::F16,
            TileStmt::CopyToWorkgroupTile { dst, src, .. } => {
                dst.element == ElementType::F16 || src.buffer.element == ElementType::F16
            }
            TileStmt::StoreCoopAcc { dst, .. } => dst.buffer.element == ElementType::F16,
            TileStmt::If { .. }
            | TileStmt::Loop { .. }
            | TileStmt::CopyQuantToWorkgroupTile { .. }
            | TileStmt::LoadCoop { .. }
            | TileStmt::WhileTrue { .. }
            | TileStmt::ZeroCoopAcc { .. }
            | TileStmt::Barrier
            | TileStmt::Mma { .. }
            | TileStmt::Break
            | TileStmt::Return => false,
        }
    }

    fn tile_expr_uses_f16(expr: &TileExpr) -> bool {
        if Self::tile_expr_f16_payload(expr) {
            return true;
        }
        let mut index_uses_f16 = |expr: &TileIndexExpr| Self::tile_index_expr_uses_f16(expr);
        Self::tile_expr_index_any(expr, &mut index_uses_f16)
            || Self::tile_expr_children_any(expr, Self::tile_expr_uses_f16)
    }

    fn tile_expr_f16_payload(expr: &TileExpr) -> bool {
        match expr {
            TileExpr::Load(load) => {
                load.src.buffer.element == ElementType::F16
                    || load.fill.element() == ElementType::F16
            }
            TileExpr::LoadLinear(load) => {
                load.src.buffer.element == ElementType::F16
                    || load.fill.element() == ElementType::F16
            }
            TileExpr::LoadVec4(_) => false,
            TileExpr::LoadWorkgroup { src, index } => {
                src.element == ElementType::F16 || Self::tile_index_expr_uses_f16(index)
            }
            TileExpr::LoadLocal(local) => local.element == ElementType::F16,
            TileExpr::QuantizedLoad(_) | TileExpr::Full(_) | TileExpr::Index(_) => false,
            TileExpr::Literal(value) => value.element() == ElementType::F16,
            TileExpr::Scalar(TileScalarExpr::Literal(value)) => value.element() == ElementType::F16,
            TileExpr::Scalar(
                TileScalarExpr::Reduce { scratch, .. } | TileScalarExpr::LoopReduce { scratch, .. },
            ) => scratch.element == ElementType::F16,
            TileExpr::Unary { .. } => false,
            TileExpr::Cast { value, to } => {
                *to == ElementType::F16 || Self::tile_expr_uses_f16(value)
            }
            TileExpr::Bitcast { value, to } => {
                *to == ElementType::F16 || Self::tile_expr_uses_f16(value)
            }
            TileExpr::Binary { .. } | TileExpr::Sum { .. } => false,
            TileExpr::Compare { output, .. } => *output == ElementType::F16,
            TileExpr::Select { .. } => false,
            TileExpr::LoopFold { initial, .. } => initial.element() == ElementType::F16,
            TileExpr::GroupReduce { value, scratch, .. } => {
                scratch.element == ElementType::F16 || Self::tile_expr_uses_f16(value)
            }
            TileExpr::SubgroupReduce { .. }
            | TileExpr::QuantizedBlockLane { .. }
            | TileExpr::Dot4 { .. }
            | TileExpr::Vec4Dot { .. }
            | TileExpr::Vec4Splat { .. }
            | TileExpr::QuantizedQ8_0Dot8 { .. }
            | TileExpr::QuantizedVecDot { .. }
            | TileExpr::QuantizedQ4KGgmlDot { .. }
            | TileExpr::QuantizedQ6KGgmlDot { .. }
            | TileExpr::PinnedRef { .. }
            | TileExpr::LoopFoldGroupOutput { .. } => false,
        }
    }

    fn tile_index_expr_uses_f16(expr: &TileIndexExpr) -> bool {
        Self::tile_index_expr_any(expr, &mut |expr| match expr {
            TileIndexExpr::Value(value) => Self::tile_expr_uses_f16(value),
            _ => false,
        })
    }
}
