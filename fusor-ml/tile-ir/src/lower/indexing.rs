use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn tile_layout(&self, tile: TileRef) -> Result<&Layout, LowerError> {
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

    pub(super) fn tile_dynamic_pointer(
        &self,
        expressions: &mut Arena<Expression>,
        tile: TileRef,
        index: Handle<Expression>,
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        self.tile_layout(tile)?;

        let base = self.tile_base_expression(expressions, tile)?;
        let (_, offset) = self.storage_tile_and_offset(tile)?;
        Ok(self.access_offset_pointer(expressions, body, base, index, offset))
    }

    pub(super) fn tile_base_expression(
        &self,
        expressions: &mut Arena<Expression>,
        tile: TileRef,
    ) -> Result<Handle<Expression>, LowerError> {
        let (storage_tile, _) = self.storage_tile_and_offset(tile)?;
        let layout = self.tile_layout(storage_tile)?;

        let id = storage_tile.id;
        let unknown = || LowerError::UnknownTile(id);
        match layout.memory_level() {
            MemoryLevel::Workgroup => {
                let global = lookup_handle(&self.tile_globals, id.index(), unknown)?;
                Ok(self.global_var(expressions, global))
            }
            MemoryLevel::Private => {
                let local = lookup_handle(&self.tile_locals, id.index(), unknown)?;
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

    /// `&base[index + offset]`. Threads through the same emit dance both
    /// `tile_dynamic_pointer` and `storage_dynamic_pointer` need: bias the
    /// index by a constant, then `Expression::Access`.
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
        self.storage_layout(view)?;
        let global = lookup_handle(&self.buffer_globals, view.buffer.id.index(), || {
            LowerError::UnknownBuffer(view.buffer.id)
        })?;
        Ok(self.global_var(expressions, global))
    }

    pub(super) fn storage_layout<'view>(
        &self,
        view: &'view StorageView,
    ) -> Result<&'view Layout, LowerError> {
        let decl = self
            .ir
            .buffers()
            .get(view.buffer.id.index())
            .ok_or(LowerError::UnknownBuffer(view.buffer.id))?;
        if decl.element != view.buffer.element {
            return Err(LowerError::UnsupportedOperation("buffer element mismatch"));
        }
        Ok(&view.layout)
    }

    pub(super) fn private_local(
        &self,
        local: LocalRef,
    ) -> Result<Handle<LocalVariable>, LowerError> {
        let decl = self
            .ir
            .locals()
            .get(local.id.index())
            .ok_or(LowerError::UnknownLocal(local.id))?;
        if decl.element != local.element {
            return Err(LowerError::LocalElementMismatch {
                local: local.id,
                declared: decl.element,
                used: local.element,
            });
        }
        lookup_handle(&self.private_locals, local.id.index(), || {
            LowerError::UnknownLocal(local.id)
        })
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
        let layout = self.storage_layout(view)?;
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

/// Resolve a side-table slot of `Option<Handle<H>>` indexed by an IR id.
/// Returns the handle if both the slot exists and was filled in, otherwise
/// produces the caller's "unknown id" error.
fn lookup_handle<H, E>(
    table: &[Option<Handle<H>>],
    index: usize,
    err: impl FnOnce() -> E,
) -> Result<Handle<H>, E> {
    table.get(index).copied().flatten().ok_or_else(err)
}
