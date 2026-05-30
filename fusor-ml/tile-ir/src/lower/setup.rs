use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn new(ir: &'a KernelIr) -> Result<Self, LowerError> {
        let mut module = Module::default();
        let i32_scalar = Scalar {
            kind: ScalarKind::Sint,
            width: 4,
        };
        let f32_ty = Self::scalar_type(&mut module, Scalar::F32);
        let f32_vec2_ty = Self::vector_type(&mut module, VectorSize::Bi, Scalar::F32);
        let f32_vec3_ty = Self::vector_type(&mut module, VectorSize::Tri, Scalar::F32);
        let f32_vec4_ty = Self::vector_type(&mut module, VectorSize::Quad, Scalar::F32);
        let i32_ty = Self::scalar_type(&mut module, i32_scalar);
        let i32_vec4_ty = Self::vector_type(&mut module, VectorSize::Quad, i32_scalar);
        let uses_f16 = ir.buffers().iter().any(|buffer| buffer.element.uses_f16())
            || ir.tiles().iter().any(|tile| tile.element.uses_f16())
            || ir.locals().iter().any(|local| local.element.uses_f16())
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
        let f16_vec2_ty = uses_f16.then(|| {
            Self::vector_type(
                &mut module,
                VectorSize::Bi,
                Scalar {
                    kind: ScalarKind::Float,
                    width: 2,
                },
            )
        });
        let f16_vec3_ty = uses_f16.then(|| {
            Self::vector_type(
                &mut module,
                VectorSize::Tri,
                Scalar {
                    kind: ScalarKind::Float,
                    width: 2,
                },
            )
        });
        let f16_vec4_ty = uses_f16.then(|| {
            Self::vector_type(
                &mut module,
                VectorSize::Quad,
                Scalar {
                    kind: ScalarKind::Float,
                    width: 2,
                },
            )
        });
        let u32_ty = Self::scalar_type(&mut module, Scalar::U32);
        let u32_vec2_ty = Self::vector_type(&mut module, VectorSize::Bi, Scalar::U32);
        let bool_ty = Self::scalar_type(&mut module, Scalar::BOOL);
        let u32_vec3_ty = Self::vector_type(&mut module, VectorSize::Tri, Scalar::U32);
        let u32_vec4_ty = Self::vector_type(&mut module, VectorSize::Quad, Scalar::U32);
        let bool_vec2_ty = Self::vector_type(&mut module, VectorSize::Bi, Scalar::BOOL);
        let bool_vec3_ty = Self::vector_type(&mut module, VectorSize::Tri, Scalar::BOOL);
        let bool_vec4_ty = Self::vector_type(&mut module, VectorSize::Quad, Scalar::BOOL);
        let uses_cooperative_matrix = Self::uses_cooperative_matrix(ir);
        let mut subgroup_usage = Self::subgroup_index_usage(ir);
        // Cooperative-matrix lowering needs a subgroup id even if the kernel
        // never asks for one explicitly.
        subgroup_usage.subgroup_id |= uses_cooperative_matrix;
        let coop_matrix_types = Self::create_coop_matrix_types(ir, &mut module)?;

        let tile_program_block = Self::max_tile_program_block(ir);
        let (workgroup_invocations, workgroup_size) = if tile_program_block > 0 {
            (tile_program_block, [tile_program_block, 1, 1])
        } else {
            (DEFAULT_WORKGROUP_INVOCATIONS, DEFAULT_WORKGROUP_SIZE)
        };
        let live_tiles = Self::live_tiles(ir);

        Ok(Self {
            ir,
            module,
            f32_ty,
            f32_vec2_ty,
            f32_vec3_ty,
            f32_vec4_ty,
            i32_ty,
            i32_vec4_ty,
            f16_ty,
            f16_vec2_ty,
            f16_vec3_ty,
            f16_vec4_ty,
            u32_ty,
            u32_vec2_ty,
            u32_vec3_ty,
            u32_vec4_ty,
            bool_ty,
            bool_vec2_ty,
            bool_vec3_ty,
            bool_vec4_ty,
            buffer_globals: Vec::new(),
            tile_globals: Vec::new(),
            tile_locals: Vec::new(),
            private_locals: Vec::new(),
            live_tiles,
            workgroup_invocations,
            workgroup_size,
            subgroup_usage,
            block_dequant_cache: Default::default(),
            q8_activation_pack_cache: Default::default(),
            coop_matrix_types,
            coop_fragment_cache: Default::default(),
            coop_acc_value_cache: Default::default(),
            uses_cooperative_matrix,
        })
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

    fn create_coop_matrix_types(
        ir: &KernelIr,
        module: &mut Module,
    ) -> Result<HashMap<ElementType, Handle<Type>>, LowerError> {
        let mut types = HashMap::new();
        for element in ir.locals().iter().map(|local| local.element) {
            if matches!(element, ElementType::CoopMatrix { .. }) && !types.contains_key(&element) {
                let inner = Self::coop_matrix_type_inner(element)?;
                let ty = Self::type_with_inner(module, inner);
                types.insert(element, ty);
            }
        }
        Ok(types)
    }

    fn coop_matrix_type_inner(element: ElementType) -> Result<TypeInner, LowerError> {
        let ElementType::CoopMatrix {
            scalar,
            role,
            rows,
            cols,
        } = element
        else {
            return Err(LowerError::UnsupportedOperation(
                "cooperative-matrix type requested for non-cooperative element",
            ));
        };
        Ok(TypeInner::CooperativeMatrix {
            columns: Self::cooperative_size(cols)?,
            rows: Self::cooperative_size(rows)?,
            scalar: Self::scalar_type_inner(scalar)?,
            role: Self::cooperative_role(role),
        })
    }

    pub(super) fn cooperative_size(size: u32) -> Result<naga::CooperativeSize, LowerError> {
        match size {
            8 => Ok(naga::CooperativeSize::Eight),
            16 => Ok(naga::CooperativeSize::Sixteen),
            _ => Err(LowerError::UnsupportedOperation(
                "cooperative-matrix size must be 8 or 16",
            )),
        }
    }

    fn cooperative_role(role: CoopMatrixRole) -> naga::CooperativeRole {
        match role {
            CoopMatrixRole::A => naga::CooperativeRole::A,
            CoopMatrixRole::B => naga::CooperativeRole::B,
            CoopMatrixRole::C => naga::CooperativeRole::C,
        }
    }

    pub(in crate::lower) fn scalar_type_inner(scalar: ScalarElement) -> Result<Scalar, LowerError> {
        match scalar {
            ScalarElement::F32 => Ok(Scalar::F32),
            ScalarElement::F16 => Ok(Scalar {
                kind: ScalarKind::Float,
                width: 2,
            }),
            ScalarElement::U32 => Ok(Scalar::U32),
            ScalarElement::Bool => Ok(Scalar::BOOL),
        }
    }

    pub(super) fn lower(mut self) -> Result<NagaKernel, LowerError> {
        self.create_storage_globals()?;
        self.create_workgroup_globals()?;

        let mut arguments = vec![
            builtin_arg(self.u32_ty, BuiltIn::LocalInvocationIndex),
            builtin_arg(self.u32_vec3_ty, BuiltIn::WorkGroupId),
        ];
        let optional_subgroup_args = [
            (self.subgroup_usage.subgroup_id, BuiltIn::SubgroupId),
            (
                self.subgroup_usage.subgroup_lane,
                BuiltIn::SubgroupInvocationId,
            ),
            (self.subgroup_usage.subgroup_size, BuiltIn::SubgroupSize),
            (self.subgroup_usage.num_subgroups, BuiltIn::NumSubgroups),
        ];
        for (used, builtin) in optional_subgroup_args {
            if used {
                arguments.push(builtin_arg(self.u32_ty, builtin));
            }
        }

        let mut function = Function {
            name: None,
            arguments,
            ..Function::default()
        };
        let scratch = self.create_scratch_locals(&mut function)?;
        self.create_private_locals(&mut function)?;
        self.create_program_private_locals(&mut function)?;

        function.body = self.lower_body(self.ir.body(), &mut function.expressions, scratch)?;
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
        if Self::uses_shader_float16_in_float32(self.ir) {
            capabilities |= naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32;
        }
        if Self::uses_subgroup_reduce(self.ir) || self.subgroup_usage.subgroup_id {
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

    pub(super) fn create_storage_globals(&mut self) -> Result<(), LowerError> {
        self.buffer_globals = vec![None; self.ir.buffers().len()];
        for buffer in self.ir.buffers() {
            let ty = self.storage_type(buffer.element)?;
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
        Ok(())
    }

    pub(super) fn create_workgroup_globals(&mut self) -> Result<(), LowerError> {
        self.tile_globals = vec![None; self.ir.tiles().len()];
        for (index, ty) in self.live_tile_types(MemoryLevel::Workgroup)? {
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
            self.tile_globals[index] = Some(global);
        }
        Ok(())
    }

    pub(super) fn create_private_locals(
        &mut self,
        function: &mut Function,
    ) -> Result<(), LowerError> {
        self.tile_locals = vec![None; self.ir.tiles().len()];
        for (index, ty) in self.live_tile_types(MemoryLevel::Private)? {
            let local = function.local_variables.append(
                LocalVariable {
                    name: None,
                    ty,
                    init: None,
                },
                Span::default(),
            );
            self.tile_locals[index] = Some(local);
        }
        Ok(())
    }

    /// `(tile_index, naga_type)` for every live tile whose layout is at
    /// `level`. Materialised eagerly so the caller can mutate `self.module`
    /// and `function.local_variables` while iterating.
    fn live_tile_types(
        &mut self,
        level: MemoryLevel,
    ) -> Result<Vec<(usize, Handle<Type>)>, LowerError> {
        let pending: Vec<_> = self
            .ir
            .tiles()
            .iter()
            .filter(|tile| {
                self.live_tiles
                    .get(tile.id.index())
                    .copied()
                    .unwrap_or(false)
                    && tile.layout.memory_level() == level
            })
            .map(|tile| (tile.id.index(), tile.element, tile.layout.clone()))
            .collect();
        pending
            .into_iter()
            .map(|(index, element, layout)| Ok((index, self.tile_type(element, &layout)?)))
            .collect()
    }

    pub(super) fn create_program_private_locals(
        &mut self,
        function: &mut Function,
    ) -> Result<(), LowerError> {
        self.private_locals = vec![None; self.ir.locals().len()];
        for local in self.ir.locals() {
            let ty = self.element_type(local.element)?;
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
        Ok(())
    }

    pub(super) fn create_scratch_locals(
        &self,
        function: &mut Function,
    ) -> Result<ScratchLocals, LowerError> {
        let full_scratch = Self::tile_programs_need_full_scratch(self.ir);
        if !full_scratch && !Self::tile_programs_need_value_scratch(self.ir) {
            let dummy = self.create_f32_local(function);
            let loop_index = self.create_u32_local(function);
            return Ok(ScratchLocals {
                loop_index,
                values: [dummy; SCRATCH_ELEMENT_COUNT],
                spills: [[dummy; 32]; SCRATCH_ELEMENT_COUNT],
                block_dequant: [dummy; 16],
                q8_activation_scales: [dummy; 4],
                q8_activation_packs: [dummy; 4],
                q8_activation_sums_i32: [dummy; 4],
            });
        }
        let values = self.create_scratch_value_locals(function)?;
        let f32_value = values[Self::element_scratch_index(ElementType::F32)?];
        let u32_value = values[Self::element_scratch_index(ElementType::U32)?];
        let spills = if full_scratch {
            self.create_scratch_spill_locals(function)?
        } else {
            values.map(|value| [value; 32])
        };
        Ok(ScratchLocals {
            loop_index: self.create_u32_local(function),
            values,
            spills,
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
        })
    }

    fn create_scratch_value_locals(
        &self,
        function: &mut Function,
    ) -> Result<[Handle<LocalVariable>; SCRATCH_ELEMENT_COUNT], LowerError> {
        SCRATCH_ELEMENTS
            .map(|element| Ok(self.create_local(function, self.scratch_element_type(element)?)))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| LowerError::UnsupportedOperation("scratch element count mismatch"))
    }

    fn create_scratch_spill_locals(
        &self,
        function: &mut Function,
    ) -> Result<[[Handle<LocalVariable>; 32]; SCRATCH_ELEMENT_COUNT], LowerError> {
        SCRATCH_ELEMENTS
            .map(|element| {
                let ty = self.scratch_element_type(element)?;
                Ok(std::array::from_fn(|_| self.create_local(function, ty)))
            })
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?
            .try_into()
            .map_err(|_| LowerError::UnsupportedOperation("scratch element count mismatch"))
    }

    fn scratch_element_type(&self, element: ElementType) -> Result<Handle<Type>, LowerError> {
        if element.uses_f16() && self.f16_ty.is_none() {
            return Ok(self.f32_ty);
        }
        self.element_type(element)
    }

    fn tile_programs_need_full_scratch(ir: &KernelIr) -> bool {
        Self::tile_programs_expr_any(ir, Self::tile_expr_needs_full_scratch)
    }

    fn tile_programs_need_value_scratch(ir: &KernelIr) -> bool {
        Self::tile_programs_expr_any(ir, Self::tile_expr_needs_value_scratch)
    }

    fn tile_expr_needs_value_scratch(expr: &Expr) -> bool {
        match expr {
            Expr::Load(load) => !load.mask.is_constant_true(),
            Expr::LoadLinear(load) => !load.mask.is_constant_true(),
            _ => Self::tile_expr_children_any(expr, Self::tile_expr_needs_value_scratch),
        }
    }

    fn tile_expr_needs_full_scratch(expr: &Expr) -> bool {
        match expr {
            Expr::QuantizedBlockLane { .. } | Expr::QuantizedDot { .. } => true,
            Expr::Reduce { iterations, .. } if *iterations > 1 => true,
            _ => Self::tile_expr_children_any(expr, Self::tile_expr_needs_full_scratch),
        }
    }

    pub(super) fn create_u32_local(&self, function: &mut Function) -> Handle<LocalVariable> {
        self.create_local(function, self.u32_ty)
    }

    pub(super) fn create_f32_local(&self, function: &mut Function) -> Handle<LocalVariable> {
        self.create_local(function, self.f32_ty)
    }

    pub(super) fn create_i32_local(&self, function: &mut Function) -> Handle<LocalVariable> {
        self.create_local(function, self.i32_ty)
    }

    pub(super) fn element_type(&self, element: ElementType) -> Result<Handle<Type>, LowerError> {
        match element {
            ElementType::F32 => Ok(self.f32_ty),
            ElementType::F16 => Ok(self.f16_ty.ok_or(LowerError::UnsupportedOperation(
                "f16 type requested without f16 capability",
            ))?),
            ElementType::U32 => Ok(self.u32_ty),
            ElementType::Bool => Ok(self.bool_ty),
            ElementType::Vector { scalar, lanes } => self.vector_type_handle(scalar, lanes),
            ElementType::CoopMatrix { .. } => self.coop_matrix_types.get(&element).copied().ok_or(
                LowerError::UnsupportedOperation("unsupported cooperative-matrix type"),
            ),
        }
    }

    pub(super) fn vector_type_handle(
        &self,
        scalar: ScalarElement,
        lanes: u32,
    ) -> Result<Handle<Type>, LowerError> {
        match (scalar, lanes) {
            (ScalarElement::F32, 2) => Ok(self.f32_vec2_ty),
            (ScalarElement::F32, 3) => Ok(self.f32_vec3_ty),
            (ScalarElement::F32, 4) => Ok(self.f32_vec4_ty),
            (ScalarElement::F16, 2) => self.f16_vec2_ty.ok_or(LowerError::UnsupportedOperation(
                "f16 vector requested without f16 capability",
            )),
            (ScalarElement::F16, 3) => self.f16_vec3_ty.ok_or(LowerError::UnsupportedOperation(
                "f16 vector requested without f16 capability",
            )),
            (ScalarElement::F16, 4) => self.f16_vec4_ty.ok_or(LowerError::UnsupportedOperation(
                "f16 vector requested without f16 capability",
            )),
            (ScalarElement::U32, 2) => Ok(self.u32_vec2_ty),
            (ScalarElement::U32, 3) => Ok(self.u32_vec3_ty),
            (ScalarElement::U32, 4) => Ok(self.u32_vec4_ty),
            (ScalarElement::Bool, 2) => Ok(self.bool_vec2_ty),
            (ScalarElement::Bool, 3) => Ok(self.bool_vec3_ty),
            (ScalarElement::Bool, 4) => Ok(self.bool_vec4_ty),
            _ => Err(LowerError::UnsupportedOperation(
                "vectors must have 2, 3, or 4 lanes",
            )),
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
        element: ElementType,
        layout: &Layout,
    ) -> Result<Handle<Type>, LowerError> {
        self.array_type(element, layout)
    }

    pub(super) fn storage_type(
        &mut self,
        element: ElementType,
    ) -> Result<Handle<Type>, LowerError> {
        self.array_type_with_size(element, ArraySize::Dynamic)
    }

    pub(super) fn array_type(
        &mut self,
        element: ElementType,
        layout: &Layout,
    ) -> Result<Handle<Type>, LowerError> {
        self.array_type_with_size(
            element,
            ArraySize::Constant(layout.allocation_element_count()),
        )
    }

    pub(super) fn array_type_with_size(
        &mut self,
        element: ElementType,
        size: ArraySize,
    ) -> Result<Handle<Type>, LowerError> {
        let stride = Self::element_array_stride(element)?;
        let base = self.element_type(element)?;

        Ok(self.module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Array { base, size, stride },
            },
            Span::default(),
        ))
    }

    fn element_array_stride(element: ElementType) -> Result<u32, LowerError> {
        match element {
            ElementType::F32 | ElementType::U32 => Ok(4),
            ElementType::F16 => Ok(2),
            ElementType::Vector { scalar, lanes } => Self::vector_array_stride(scalar, lanes),
            ElementType::Bool => Err(LowerError::UnsupportedOperation(
                "bool arrays are not supported",
            )),
            ElementType::CoopMatrix { .. } => Err(LowerError::UnsupportedOperation(
                "cooperative-matrix arrays are not supported",
            )),
        }
    }

    fn vector_array_stride(scalar: ScalarElement, lanes: u32) -> Result<u32, LowerError> {
        let scalar_size = match scalar {
            ScalarElement::F32 | ScalarElement::U32 => 4,
            ScalarElement::F16 => 2,
            ScalarElement::Bool => {
                return Err(LowerError::UnsupportedOperation(
                    "bool vector arrays are not supported",
                ));
            }
        };
        match lanes {
            2 => Ok(2 * scalar_size),
            3 | 4 => Ok(4 * scalar_size),
            _ => Err(LowerError::UnsupportedOperation(
                "vectors must have 2, 3, or 4 lanes",
            )),
        }
    }

    fn tile_programs_use_f16(ir: &KernelIr) -> bool {
        ir.body().body.iter().any(Self::tile_stmt_uses_f16)
    }

    fn tile_stmt_uses_f16(stmt: &TileStmt) -> bool {
        if Self::tile_stmt_f16_payload(stmt) {
            return true;
        }
        let mut expr_uses_f16 = |expr: &Expr| Self::tile_expr_uses_f16(expr);
        Self::tile_stmt_expr_any(stmt, &mut expr_uses_f16)
    }

    /// Per-statement F16 check. Expr trees and nested statement bodies are
    /// recursed into by the caller via `tile_stmt_expr_any`, so this method
    /// only inspects the element types attached directly to the current
    /// statement (storage views, locals, accumulators, workgroup tiles).
    fn tile_stmt_f16_payload(stmt: &TileStmt) -> bool {
        match stmt {
            TileStmt::Store(store) => store.dst.buffer.element.uses_f16(),
            TileStmt::StoreIndexed(store) => store.dst.buffer.element.uses_f16(),
            TileStmt::StoreLocal { dst, .. } => dst.element.uses_f16(),
            TileStmt::StoreWorkgroup { dst, .. } => dst.element.uses_f16(),
            TileStmt::CopyToWorkgroupTile { dst, src, .. } => {
                let src_uses_f16 = match src {
                    crate::ir::CopySource::Storage(view) => view.buffer.element.uses_f16(),
                    crate::ir::CopySource::Quantized(_) => false,
                };
                dst.element.uses_f16() || src_uses_f16
            }
            TileStmt::StoreCoopAcc { dst, .. } => dst.buffer.element.uses_f16(),
            TileStmt::LoadCoopBroadcast { scalar, .. } => *scalar == ScalarElement::F16,
            TileStmt::Fold { accumulators, .. } => {
                accumulators.iter().any(|acc| acc.element.uses_f16())
            }
            TileStmt::If { .. }
            | TileStmt::Loop { .. }
            | TileStmt::ZeroCoopAcc { .. }
            | TileStmt::Barrier
            | TileStmt::Mma { .. }
            | TileStmt::SetCoopAcc { .. }
            | TileStmt::Break
            | TileStmt::Return => false,
            TileStmt::LoadCoop { scalar, .. } => *scalar == ScalarElement::F16,
        }
    }

    fn tile_expr_uses_f16(expr: &Expr) -> bool {
        if Self::tile_expr_f16_payload(expr) {
            return true;
        }
        Self::tile_expr_children_any(expr, Self::tile_expr_uses_f16)
    }

    /// Per-node F16 check. Tree recursion happens in the caller via
    /// `tile_expr_children_any`, so this method only inspects the element
    /// types attached to the current node (storage views, locals, scratch,
    /// cast destinations).
    fn tile_expr_f16_payload(expr: &Expr) -> bool {
        match expr {
            Expr::Load(load) => match &load.src {
                LoadSource::Storage(view) => view.buffer.element.uses_f16(),
                LoadSource::Quantized(_) => false,
            },
            Expr::LoadLinear(load) => load.src.buffer.element.uses_f16(),
            Expr::LoadWorkgroup { src, .. } => src.element.uses_f16(),
            Expr::LoadLocal(local) => local.element.uses_f16(),
            Expr::Literal(value) => value.element().uses_f16(),
            Expr::Reduce { scratch, .. } => scratch.element.uses_f16(),
            Expr::Cast { to, .. } | Expr::Bitcast { to, .. } => to.uses_f16(),
            Expr::VectorDot { scalar, .. } | Expr::ComposeVector { scalar, .. } => {
                *scalar == ScalarElement::F16
            }
            Expr::Unary { .. }
            | Expr::Binary { .. }
            | Expr::Compare { .. }
            | Expr::Select { .. }
            | Expr::SubgroupReduce { .. }
            | Expr::QuantizedBlockLane { .. }
            | Expr::QuantizedDot { .. }
            | Expr::Builtin(_) => false,
        }
    }
}

fn builtin_arg(ty: Handle<Type>, builtin: BuiltIn) -> FunctionArgument {
    FunctionArgument {
        name: None,
        ty,
        binding: Some(Binding::BuiltIn(builtin)),
    }
}
