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
        let u32_ty = module.types.insert(
            Type {
                name: Some("SubgroupIndex".into()),
                inner: TypeInner::Scalar(Scalar::U32),
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

        let coop_subgroups = if PREFER_COOP_MATRIX_GEMM {
            Self::max_coop_gemm_subgroups(ir)
        } else {
            0
        };
        let uses_coop_gemm = coop_subgroups > 0;
        let uses_subgroup_id = coop_subgroups > 1;
        let (coop_f32_a_ty, coop_f32_b_ty, coop_f32_c_ty) = if uses_coop_gemm {
            let coop_f32_a_ty = module.types.insert(
                Type {
                    name: Some("CoopA8x8F32".into()),
                    inner: TypeInner::CooperativeMatrix {
                        columns: COOP_MATRIX_SIZE,
                        rows: COOP_MATRIX_SIZE,
                        scalar: Scalar::F32,
                        role: CooperativeRole::A,
                    },
                },
                Span::default(),
            );
            let coop_f32_b_ty = module.types.insert(
                Type {
                    name: Some("CoopB8x8F32".into()),
                    inner: TypeInner::CooperativeMatrix {
                        columns: COOP_MATRIX_SIZE,
                        rows: COOP_MATRIX_SIZE,
                        scalar: Scalar::F32,
                        role: CooperativeRole::B,
                    },
                },
                Span::default(),
            );
            let coop_f32_c_ty = module.types.insert(
                Type {
                    name: Some("CoopC8x8F32".into()),
                    inner: TypeInner::CooperativeMatrix {
                        columns: COOP_MATRIX_SIZE,
                        rows: COOP_MATRIX_SIZE,
                        scalar: Scalar::F32,
                        role: CooperativeRole::C,
                    },
                },
                Span::default(),
            );
            (
                Some(coop_f32_a_ty),
                Some(coop_f32_b_ty),
                Some(coop_f32_c_ty),
            )
        } else {
            (None, None, None)
        };

        let max_gemv_rows = Self::max_gemv_rows(ir.body());
        let max_scratch_sums = max_gemv_rows.max(Self::max_gemm_sums(ir, ir.body()));
        let (workgroup_invocations, workgroup_size) = if max_gemv_rows > 0 {
            (GEMV_WORKGROUP_INVOCATIONS, GEMV_WORKGROUP_SIZE)
        } else if uses_coop_gemm {
            (
                COOP_MATRIX_WORKGROUP_INVOCATIONS * coop_subgroups,
                [COOP_MATRIX_WORKGROUP_SIZE[0] * coop_subgroups, 1, 1],
            )
        } else {
            (DEFAULT_WORKGROUP_INVOCATIONS, DEFAULT_WORKGROUP_SIZE)
        };
        let live_tiles = Self::live_tiles(ir, workgroup_invocations);

        Self {
            ir,
            module,
            f32_ty,
            f32_vec4_ty,
            u32_ty,
            u32_vec3_ty,
            coop_f32_a_ty,
            coop_f32_b_ty,
            coop_f32_c_ty,
            buffer_globals: Vec::new(),
            tile_globals: Vec::new(),
            tile_locals: Vec::new(),
            live_tiles,
            loop_index_local: None,
            workgroup_invocations,
            workgroup_size,
            max_gemv_rows: max_scratch_sums,
            uses_coop_gemm,
            coop_subgroups,
            uses_subgroup_id,
        }
    }

    pub(super) fn lower(mut self) -> Result<NagaKernel, LowerError> {
        self.create_storage_globals()?;
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

        let mut function = Function {
            name: Some("main".into()),
            arguments,
            ..Function::default()
        };
        let scratch = self.create_scratch_locals(&mut function);
        self.loop_index_local = Some(scratch.loop_index);
        self.create_private_locals(&mut function)?;

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
        if self.uses_coop_gemm {
            capabilities |= naga::valid::Capabilities::COOPERATIVE_MATRIX;
        }
        if self.uses_subgroup_id {
            capabilities |= naga::valid::Capabilities::SUBGROUP;
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
                    memory_decorations: MemoryDecorations::empty(),
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
        let mut coop_accs = [None; 16];
        if let Some(ty) = self.coop_f32_c_ty {
            for (index, local) in coop_accs.iter_mut().enumerate() {
                *local = Some(self.create_local(function, &format!("coop_acc_{index}"), ty));
            }
        }

        if self.uses_coop_gemm
            && !PREFER_SHARED_COOP_GEMM
            && Self::is_single_direct_coop_gemm(self.ir)
        {
            let loop_index = self.create_u32_local(function, "loop_index");
            let mma_k = self.create_u32_local(function, "mma_k");
            let mma_sum = self.create_f32_local(function, "mma_sum");
            let mma_sum_1 = self.create_f32_local(function, "mma_sum_1");
            return ScratchLocals {
                tile_index: loop_index,
                linear_index: loop_index,
                store_index: loop_index,
                loop_index,
                mma_i: mma_k,
                mma_j: mma_k,
                mma_k,
                mma_sum,
                mma_sum_1,
                mma_sum_2: None,
                mma_sum_3: None,
                mma_sum_4: None,
                mma_sum_5: None,
                mma_sum_6: None,
                mma_sum_7: None,
                coop_accs,
            };
        }

        ScratchLocals {
            tile_index: self.create_u32_local(function, "tile_index"),
            linear_index: self.create_u32_local(function, "linear_index"),
            store_index: self.create_u32_local(function, "store_index"),
            loop_index: self.create_u32_local(function, "loop_index"),
            mma_i: self.create_u32_local(function, "mma_i"),
            mma_j: self.create_u32_local(function, "mma_j"),
            mma_k: self.create_u32_local(function, "mma_k"),
            mma_sum: self.create_f32_local(function, "mma_sum"),
            mma_sum_1: self.create_f32_local(function, "mma_sum_1"),
            mma_sum_2: (self.max_gemv_rows > 2)
                .then(|| self.create_f32_local(function, "mma_sum_2")),
            mma_sum_3: (self.max_gemv_rows > 3)
                .then(|| self.create_f32_local(function, "mma_sum_3")),
            mma_sum_4: (self.max_gemv_rows > 4)
                .then(|| self.create_f32_local(function, "mma_sum_4")),
            mma_sum_5: (self.max_gemv_rows > 5)
                .then(|| self.create_f32_local(function, "mma_sum_5")),
            mma_sum_6: (self.max_gemv_rows > 6)
                .then(|| self.create_f32_local(function, "mma_sum_6")),
            mma_sum_7: (self.max_gemv_rows > 7)
                .then(|| self.create_f32_local(function, "mma_sum_7")),
            coop_accs,
        }
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
        layout: &Layout,
    ) -> Handle<Type> {
        self.array_type(format!("Buffer{buffer}"), element, layout)
    }

    pub(super) fn array_type(
        &mut self,
        name: String,
        element: ElementType,
        layout: &Layout,
    ) -> Handle<Type> {
        let base = match element {
            ElementType::F32 => self.f32_ty,
        };

        self.module.types.insert(
            Type {
                name: Some(name),
                inner: TypeInner::Array {
                    base,
                    size: ArraySize::Constant(layout.allocation_element_count()),
                    stride: 4,
                },
            },
            Span::default(),
        )
    }
}
