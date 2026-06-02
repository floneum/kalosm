use super::*;

impl<'a> Lowerer<'a> {
    // ---- decl resolution (pointer-keyed maps, no Vec side tables) ----

    pub(super) fn buffer_global(
        &self,
        buffer: &Buffer,
    ) -> Result<Handle<GlobalVariable>, LowerError> {
        self.globals
            .borrow()
            .get(&buffer_key(buffer))
            .copied()
            .ok_or(LowerError::UnsupportedOperation("buffer not declared"))
    }

    pub(super) fn tile_global(&self, tile: &Tile) -> Option<Handle<GlobalVariable>> {
        self.globals.borrow().get(&tile_key(tile)).copied()
    }

    pub(super) fn tile_local(&self, tile: &Tile) -> Option<Handle<LocalVariable>> {
        self.locals.borrow().get(&tile_key(tile)).copied()
    }

    pub(super) fn private_local(&self, local: &Local) -> Result<Handle<LocalVariable>, LowerError> {
        self.locals
            .borrow()
            .get(&local_key(local))
            .copied()
            .ok_or(LowerError::UnsupportedOperation("local not declared"))
    }

    // ---- demand-allocated scratch ----

    /// Intern (or allocate) the scratch local for `(kind, element, depth)`.
    /// Allocated lazily into the function-local arena; a non-f16 adapter falls
    /// back to f32 storage so an f16 scratch slot still validates.
    pub(super) fn scratch_local(
        &self,
        kind: ScratchKind,
        element: ElementType,
        depth: u32,
    ) -> Result<Handle<LocalVariable>, LowerError> {
        let key = (kind, element, depth);
        if let Some(handle) = self.scratch.borrow().get(&key).copied() {
            return Ok(handle);
        }
        let stored = if element.uses_f16() && !self.uses_f16 {
            ElementType::F32
        } else {
            element
        };
        let ty = self.element_type(stored)?;
        let handle = self.create_local(ty);
        self.scratch.borrow_mut().insert(key, handle);
        Ok(handle)
    }

    pub(super) fn scratch_u32(&self, kind: ScratchKind, depth: u32) -> Handle<LocalVariable> {
        self.scratch_local(kind, ElementType::U32, depth)
            .expect("u32 scratch always resolves")
    }

    pub(super) fn scratch_f32(&self, kind: ScratchKind, depth: u32) -> Handle<LocalVariable> {
        self.scratch_local(kind, ElementType::F32, depth)
            .expect("f32 scratch always resolves")
    }

    /// Demand-allocate an `i32`-typed scratch local. Keyed under the `U32`
    /// element slot (the only i32 scratch is the Q8 activation sum), but
    /// materialised with the signed-32 type.
    pub(super) fn scratch_i32(&self, kind: ScratchKind, depth: u32) -> Handle<LocalVariable> {
        let key = (kind, ElementType::U32, depth);
        if let Some(handle) = self.scratch.borrow().get(&key).copied() {
            return handle;
        }
        let handle = self.create_local(self.i32_ty);
        self.scratch.borrow_mut().insert(key, handle);
        handle
    }

    // ---- tile pointer + layout resolution ----

    pub(super) fn tile_layout<'t>(&self, tile: &'t Tile) -> &'t Layout {
        &tile.layout
    }

    pub(super) fn tile_dynamic_pointer(
        &self,
        expressions: &mut Arena<Expression>,
        tile: &Tile,
        index: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        let base = self.tile_base_expression(expressions, tile)?;
        Ok(self.access_offset_pointer(expressions, body, base, index, 0))
    }

    pub(super) fn tile_base_expression(
        &self,
        expressions: &mut Arena<Expression>,
        tile: &Tile,
    ) -> Result<Handle<Expression>, LowerError> {
        match tile.layout.memory_level() {
            MemoryLevel::Workgroup => {
                let global = self
                    .tile_global(tile)
                    .ok_or(LowerError::UnsupportedOperation("tile not declared"))?;
                Ok(self.global_var(expressions, global))
            }
            MemoryLevel::Private => {
                let local = self
                    .tile_local(tile)
                    .ok_or(LowerError::UnsupportedOperation("tile not declared"))?;
                Ok(self.local_var(expressions, local))
            }
            memory => Err(LowerError::UnsupportedMemoryLevel(memory)),
        }
    }

    pub(super) fn storage_dynamic_pointer(
        &self,
        expressions: &mut Arena<Expression>,
        view: &StorageView,
        index: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        let base = self.storage_base_expression(expressions, view)?;
        Ok(self.access_offset_pointer(expressions, body, base, index, view.offset))
    }

    /// `&base[index + offset]`.
    pub(super) fn access_offset_pointer(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        base: Handle<Expression>,
        index: Handle<Expression>,
        offset: u32,
    ) -> Handle<Expression> {
        let index = self.add_literal_u32_emitted(expressions, index, offset, body);
        self.emit(expressions, body, Expression::Access { base, index })
    }

    pub(super) fn storage_base_expression(
        &self,
        expressions: &mut Arena<Expression>,
        view: &StorageView,
    ) -> Result<Handle<Expression>, LowerError> {
        let global = self.buffer_global(&view.buffer)?;
        Ok(self.global_var(expressions, global))
    }

    pub(super) fn storage_layout<'view>(&self, view: &'view StorageView) -> &'view Layout {
        &view.layout
    }

    pub(super) fn is_u32_literal(
        expressions: &Arena<Expression>,
        value: Handle<Expression>,
        expected: u32,
    ) -> bool {
        Self::u32_literal(expressions, value) == Some(expected)
    }

    pub(super) fn u32_literal(
        expressions: &Arena<Expression>,
        value: Handle<Expression>,
    ) -> Option<u32> {
        match expressions[value] {
            Expression::Literal(Literal::U32(value)) => Some(value),
            _ => None,
        }
    }

    pub(super) fn storage_index_from_coords(
        &self,
        expressions: &mut Arena<Expression>,
        view: &StorageView,
        coords: &[Handle<Expression>],
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        let layout = self.storage_layout(view);
        self.storage_index_from_multi_flatten(expressions, layout.indexing(), coords, body)
    }

    fn storage_index_from_multi_flatten(
        &self,
        expressions: &mut Arena<Expression>,
        map: &MultiFlattenMap,
        coords: &[Handle<Expression>],
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        if map.groups.len() != coords.len() {
            return Err(LowerError::UnsupportedOperation(
                "multi-flatten map rank does not match coord count",
            ));
        }
        let mut acc: Option<Handle<Expression>> = None;
        for (group, &coord) in map.groups.iter().zip(coords.iter()) {
            let Some(term) = self.lower_axis_group(expressions, group, coord, body)? else {
                continue;
            };
            acc = Some(match acc {
                Some(a) => self.add_u32_expr(expressions, a, term, body),
                None => term,
            });
        }
        Ok(acc.unwrap_or_else(|| self.u32(expressions, 0)))
    }

    fn lower_axis_group(
        &self,
        expressions: &mut Arena<Expression>,
        group: &AxisGroup,
        coord: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Option<Handle<Expression>>, LowerError> {
        let sub = &group.sub_axes;
        if sub.is_empty() {
            return Err(LowerError::UnsupportedOperation("empty axis group"));
        }
        let mut remaining = coord;
        let mut terms = Vec::with_capacity(sub.len());
        for axis in (0..sub.len()).rev() {
            let sub_coord = if axis == 0 {
                remaining
            } else {
                let extent = sub[axis].extent;
                let c = self.mod_literal_u32_emitted(expressions, remaining, extent, body);
                remaining = self.div_literal_u32_emitted(expressions, remaining, extent, body);
                c
            };
            let stride = sub[axis].stride;
            if stride == 0 {
                continue;
            }
            if Self::is_u32_literal(expressions, sub_coord, 0) {
                continue;
            }
            terms.push(self.mul_literal_u32_emitted(expressions, sub_coord, stride, body));
        }
        let mut iter = terms.into_iter();
        let Some(mut sum) = iter.next() else {
            return Ok(None);
        };
        for t in iter {
            sum = self.add_u32_expr(expressions, sum, t, body);
        }
        Ok(Some(sum))
    }

    fn add_u32_expr(
        &self,
        expressions: &mut Arena<Expression>,
        left: Handle<Expression>,
        right: Handle<Expression>,
        body: &mut Block,
    ) -> Handle<Expression> {
        if Self::is_u32_literal(expressions, left, 0) {
            return right;
        }
        if Self::is_u32_literal(expressions, right, 0) {
            return left;
        }
        self.emit(
            expressions,
            body,
            Expression::Binary {
                op: BinaryOperator::Add,
                left,
                right,
            },
        )
    }
}
