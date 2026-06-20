use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn new(ir: &'a KernelIr) -> Result<Self, LowerError> {
        let analysis = analysis::Analysis::run(ir);
        let mut caps = analysis.caps;
        // Cooperative-matrix lowering needs a subgroup id even if the kernel
        // never asks for one explicitly.
        caps.subgroup_id |= caps.uses_coop;
        let uses_f16 = caps.uses_f16;

        let mut module = Module::default();
        let i32_scalar = Scalar {
            kind: ScalarKind::Sint,
            width: 4,
        };
        let f16_scalar = Scalar {
            kind: ScalarKind::Float,
            width: 2,
        };

        // The prelude types are created in a fixed order so the module's type
        // arena is deterministic. The interner is then pre-populated with them;
        // coop-matrix and array types are added on demand below the
        // `Expr -> Handle` boundary.
        let mut types: FxHashMap<ElementType, Handle<Type>> = FxHashMap::default();

        let f32_ty = Self::scalar_type(&mut module, Scalar::F32);
        let f32_vec2_ty = Self::vector_type(&mut module, VectorSize::Bi, Scalar::F32);
        let f32_vec3_ty = Self::vector_type(&mut module, VectorSize::Tri, Scalar::F32);
        let f32_vec4_ty = Self::vector_type(&mut module, VectorSize::Quad, Scalar::F32);
        let i32_ty = Self::scalar_type(&mut module, i32_scalar);
        let i32_vec4_ty = Self::vector_type(&mut module, VectorSize::Quad, i32_scalar);
        let f16_handles = uses_f16.then(|| {
            let f16_ty = Self::scalar_type(&mut module, f16_scalar);
            let f16_vec2_ty = Self::vector_type(&mut module, VectorSize::Bi, f16_scalar);
            let f16_vec3_ty = Self::vector_type(&mut module, VectorSize::Tri, f16_scalar);
            let f16_vec4_ty = Self::vector_type(&mut module, VectorSize::Quad, f16_scalar);
            (f16_ty, f16_vec2_ty, f16_vec3_ty, f16_vec4_ty)
        });
        let u32_ty = Self::scalar_type(&mut module, Scalar::U32);
        let u32_vec2_ty = Self::vector_type(&mut module, VectorSize::Bi, Scalar::U32);
        let bool_ty = Self::scalar_type(&mut module, Scalar::BOOL);
        let u32_vec3_ty = Self::vector_type(&mut module, VectorSize::Tri, Scalar::U32);
        let u32_vec4_ty = Self::vector_type(&mut module, VectorSize::Quad, Scalar::U32);
        let bool_vec2_ty = Self::vector_type(&mut module, VectorSize::Bi, Scalar::BOOL);
        let bool_vec3_ty = Self::vector_type(&mut module, VectorSize::Tri, Scalar::BOOL);
        let bool_vec4_ty = Self::vector_type(&mut module, VectorSize::Quad, Scalar::BOOL);

        types.insert(ElementType::F32, f32_ty);
        types.insert(ElementType::vector(ScalarElement::F32, 2), f32_vec2_ty);
        types.insert(ElementType::vector(ScalarElement::F32, 3), f32_vec3_ty);
        types.insert(ElementType::vector(ScalarElement::F32, 4), f32_vec4_ty);
        types.insert(ElementType::U32, u32_ty);
        types.insert(ElementType::vector(ScalarElement::U32, 2), u32_vec2_ty);
        types.insert(ElementType::vector(ScalarElement::U32, 3), u32_vec3_ty);
        types.insert(ElementType::vector(ScalarElement::U32, 4), u32_vec4_ty);
        types.insert(ElementType::Bool, bool_ty);
        types.insert(ElementType::vector(ScalarElement::Bool, 2), bool_vec2_ty);
        types.insert(ElementType::vector(ScalarElement::Bool, 3), bool_vec3_ty);
        types.insert(ElementType::vector(ScalarElement::Bool, 4), bool_vec4_ty);
        if let Some((f16_ty, f16_vec2_ty, f16_vec3_ty, f16_vec4_ty)) = f16_handles {
            types.insert(ElementType::F16, f16_ty);
            types.insert(ElementType::vector(ScalarElement::F16, 2), f16_vec2_ty);
            types.insert(ElementType::vector(ScalarElement::F16, 3), f16_vec3_ty);
            types.insert(ElementType::vector(ScalarElement::F16, 4), f16_vec4_ty);
        }

        // Cooperative-matrix types are created up front: walk the program
        // locals in first-use order and intern each distinct coop element
        // exactly once.
        for local in &analysis.locals {
            let element = local.element;
            if matches!(element, ElementType::CoopMatrix { .. }) && !types.contains_key(&element) {
                let inner = Self::coop_matrix_type_inner(element)?;
                let handle = Self::type_with_inner(&mut module, inner);
                types.insert(element, handle);
            }
        }

        let tile_program_block = ir.block;
        let (workgroup_invocations, workgroup_size) = if tile_program_block > 0 {
            (tile_program_block, [tile_program_block, 1, 1])
        } else {
            (DEFAULT_WORKGROUP_INVOCATIONS, DEFAULT_WORKGROUP_SIZE)
        };

        Ok(Self {
            ir,
            module,
            types: RefCell::new(types),
            f32_ty,
            f32_vec4_ty,
            i32_ty,
            i32_vec4_ty,
            u32_ty,
            u32_vec3_ty,
            uses_f16,
            globals: Default::default(),
            locals: Default::default(),
            scratch: Default::default(),
            func_locals: RefCell::new(Arena::new()),
            q8_activation_pack_cache: Default::default(),
            coop_acc_value_cache: Default::default(),
            dequant_memo: Default::default(),
            expr_memo: Default::default(),
            workgroup_invocations,
            workgroup_size,
            caps,
            subgroup_id_arg: None,
            subgroup_invocation_id_arg: None,
            subgroup_size_arg: None,
            num_subgroups_arg: None,
            buffer_decls: analysis.buffers,
            tile_decls: analysis.tiles,
            local_decls: analysis.locals,
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
            (self.caps.subgroup_id, BuiltIn::SubgroupId),
            (self.caps.subgroup_lane, BuiltIn::SubgroupInvocationId),
            (self.caps.subgroup_size, BuiltIn::SubgroupSize),
            (self.caps.num_subgroups, BuiltIn::NumSubgroups),
        ];
        for (index, (used, builtin)) in optional_subgroup_args.into_iter().enumerate() {
            if used {
                let arg = arguments.len() as u32;
                match index {
                    0 => self.subgroup_id_arg = Some(arg),
                    1 => self.subgroup_invocation_id_arg = Some(arg),
                    2 => self.subgroup_size_arg = Some(arg),
                    3 => self.num_subgroups_arg = Some(arg),
                    _ => unreachable!(),
                }
                arguments.push(builtin_arg(self.u32_ty, builtin));
            }
        }

        let mut function = Function {
            name: None,
            arguments,
            ..Function::default()
        };
        // Private tiles and program locals are appended before scratch, which
        // is demand-allocated into the same arena during body lowering. The
        // arena is moved into the function once lowering is done.
        self.create_private_locals()?;
        self.create_program_private_locals()?;

        let mut body = self.lower_body(&mut function.expressions)?;
        body.push(Statement::Return { value: None }, Span::default());
        function.body = body;
        function.local_variables = self.func_locals.take();

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
        if self.uses_f16 {
            capabilities |= naga::valid::Capabilities::SHADER_FLOAT16;
        }
        if self.caps.native_f16_scales || self.caps.unpacks_f16 {
            capabilities |= naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32;
        }
        let uses_subgroups = self.caps.uses_subgroups();
        if uses_subgroups {
            capabilities |= naga::valid::Capabilities::SUBGROUP;
        }
        if self.caps.uses_coop {
            capabilities |= naga::valid::Capabilities::COOPERATIVE_MATRIX;
        }
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
            .validate(&self.module)
            .map_err(|error| LowerError::Validation(format!("{error:#?}")))?;

        Ok(NagaKernel {
            module: self.module,
            info,
            wgsl_extensions: WgslExtensions::new(uses_subgroups),
        })
    }

    fn create_storage_globals(&mut self) -> Result<(), LowerError> {
        // Buffers are emitted in declaration order. The builder assigns
        // `binding` incrementally at creation time, so sorting by binding keeps
        // the global-variable arena independent of which kernel touches which
        // buffer first.
        let mut buffers = self.collect_buffers();
        buffers.sort_by_key(|buffer| buffer.binding);
        for buffer in &buffers {
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
                        binding: buffer.binding,
                    }),
                    ty,
                    init: None,
                    memory_decorations: naga::MemoryDecorations::empty(),
                },
                Span::default(),
            );
            self.globals
                .borrow_mut()
                .insert(std::rc::Rc::as_ptr(buffer) as *const (), global);
        }
        Ok(())
    }

    fn create_workgroup_globals(&mut self) -> Result<(), LowerError> {
        let tiles = self.collect_tiles();
        for tile in &tiles {
            if tile.layout.memory_level() != MemoryLevel::Workgroup {
                continue;
            }
            let ty = self.tile_type(tile.element, &tile.layout)?;
            let global = self.module.global_variables.append(
                GlobalVariable {
                    name: None,
                    space: AddressSpace::WorkGroup,
                    binding: None,
                    ty,
                    init: None,
                    memory_decorations: naga::MemoryDecorations::empty(),
                },
                Span::default(),
            );
            self.globals
                .borrow_mut()
                .insert(std::rc::Rc::as_ptr(tile) as *const (), global);
        }
        Ok(())
    }

    fn create_private_locals(&mut self) -> Result<(), LowerError> {
        let tiles = self.collect_tiles();
        for tile in &tiles {
            if tile.layout.memory_level() != MemoryLevel::Private {
                continue;
            }
            let ty = self.tile_type(tile.element, &tile.layout)?;
            let local = self.append_func_local(ty);
            self.locals
                .borrow_mut()
                .insert(std::rc::Rc::as_ptr(tile) as *const (), local);
        }
        Ok(())
    }

    fn create_program_private_locals(&mut self) -> Result<(), LowerError> {
        let locals = self.collect_locals();
        for local in &locals {
            let ty = self.element_type(local.element)?;
            let handle = self.append_func_local(ty);
            self.locals
                .borrow_mut()
                .insert(std::rc::Rc::as_ptr(local) as *const (), handle);
        }
        Ok(())
    }

    fn append_func_local(&self, ty: Handle<Type>) -> Handle<LocalVariable> {
        self.func_locals.borrow_mut().append(
            LocalVariable {
                name: None,
                ty,
                init: None,
            },
            Span::default(),
        )
    }

    /// Look up the `Handle<Type>` for an `ElementType`. All scalar, vector, and
    /// coop-matrix element types the program uses are interned up front in
    /// `new()`; this is a pure lookup. f16 without the f16 capability and
    /// unsupported vector arities surface as `UnsupportedOperation`.
    pub(super) fn element_type(&self, element: ElementType) -> Result<Handle<Type>, LowerError> {
        if let Some(handle) = self.types.borrow().get(&element).copied() {
            return Ok(handle);
        }
        match element {
            ElementType::F16
            | ElementType::Vector {
                scalar: ScalarElement::F16,
                ..
            } => Err(LowerError::UnsupportedOperation(
                "f16 type requested without f16 capability",
            )),
            ElementType::Vector { .. } => Err(LowerError::UnsupportedOperation(
                "vectors must have 2, 3, or 4 lanes",
            )),
            ElementType::CoopMatrix { .. } => Err(LowerError::UnsupportedOperation(
                "unsupported cooperative-matrix type",
            )),
            ElementType::F32 | ElementType::U32 | ElementType::Bool => {
                unreachable!("prelude scalar types are interned up front")
            }
        }
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

    pub(super) fn vector_type_handle(
        &self,
        scalar: ScalarElement,
        lanes: u32,
    ) -> Result<Handle<Type>, LowerError> {
        self.element_type(ElementType::Vector { scalar, lanes })
    }

    pub(super) fn create_local(&self, ty: Handle<Type>) -> Handle<LocalVariable> {
        self.append_func_local(ty)
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
}

fn builtin_arg(ty: Handle<Type>, builtin: BuiltIn) -> FunctionArgument {
    FunctionArgument {
        name: None,
        ty,
        binding: Some(Binding::BuiltIn(builtin)),
    }
}
