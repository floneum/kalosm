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
                Ok(expressions.append(Expression::GlobalVariable(global), Span::default()))
            }
            MemoryLevel::Private => {
                let local = lookup_handle(&self.tile_locals, id.index(), unknown)?;
                Ok(expressions.append(Expression::LocalVariable(local), Span::default()))
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
        Ok(expressions.append(Expression::GlobalVariable(global), Span::default()))
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
        if let Some(index_map) = &view.index_map {
            return self.storage_index_from_index_map(expressions, index_map, coords, body);
        }

        let layout = self.storage_layout(view)?;
        if layout.strides().rank() != coords.len() {
            return Err(LowerError::UnsupportedOperation("layout rank mismatch"));
        }

        let mut terms = Vec::with_capacity(coords.len());
        for (coord, stride) in coords.iter().zip(layout.strides().values()) {
            terms.push(self.mul_literal_u32_emitted(expressions, *coord, *stride, body));
        }
        let mut terms = terms.into_iter();
        let Some(mut index) = terms.next() else {
            return Err(LowerError::UnsupportedOperation("zero-rank layout"));
        };
        for term in terms {
            index = self.emit(
                expressions,
                body,
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: index,
                    right: term,
                },
            );
        }

        Ok(index)
    }

    fn storage_index_from_index_map(
        &self,
        expressions: &mut Arena<Expression>,
        index_map: &StorageIndexMap,
        coords: &[Handle<Expression>],
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        match index_map {
            StorageIndexMap::Im2ColNhwc(map) => {
                self.storage_index_from_im2col_nhwc(expressions, *map, coords, body)
            }
            StorageIndexMap::FlattenedMatrix(map) => {
                self.storage_index_from_flattened_matrix(expressions, map, coords, body)
            }
        }
    }

    fn storage_index_from_flattened_matrix(
        &self,
        expressions: &mut Arena<Expression>,
        map: &FlattenedMatrixMap,
        coords: &[Handle<Expression>],
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        if coords.len() != 2 {
            return Err(LowerError::UnsupportedOperation(
                "flattened matrix storage views require matrix coordinates",
            ));
        }
        if map.prefix_shape.is_empty() || map.prefix_shape.len() != map.prefix_strides.len() {
            return Err(LowerError::UnsupportedOperation(
                "flattened matrix prefix metadata mismatch",
            ));
        }

        let row = coords[0];
        let col = coords[1];
        let mut remaining = row;
        let mut terms = Vec::with_capacity(map.prefix_shape.len() + 1);

        for axis in (0..map.prefix_shape.len()).rev() {
            let dim = map.prefix_shape[axis];
            let coord = if axis == 0 {
                remaining
            } else {
                let coord = self.mod_literal_u32_emitted(expressions, remaining, dim, body);
                remaining = self.div_literal_u32_emitted(expressions, remaining, dim, body);
                coord
            };
            let stride = map.prefix_strides[axis];
            if stride != 0 {
                terms.push(self.mul_literal_u32_emitted(expressions, coord, stride, body));
            }
        }

        if map.column_stride != 0 {
            terms.push(self.mul_literal_u32_emitted(
                expressions,
                col,
                map.column_stride,
                body,
            ));
        }

        let mut terms = terms.into_iter();
        let Some(mut index) = terms.next() else {
            let zero = expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
            return Ok(zero);
        };
        for term in terms {
            index = self.add_u32_expr(expressions, index, term, body);
        }
        Ok(index)
    }

    fn storage_index_from_im2col_nhwc(
        &self,
        expressions: &mut Arena<Expression>,
        map: Im2ColNhwcMap,
        coords: &[Handle<Expression>],
        body: &mut Block,
    ) -> Result<Handle<Expression>, LowerError> {
        if coords.len() != 2 {
            return Err(LowerError::UnsupportedOperation(
                "im2col storage views require matrix coordinates",
            ));
        }

        let row = coords[0];
        let k = coords[1];
        let output_pixels =
            map.out_h
                .checked_mul(map.out_w)
                .ok_or(LowerError::UnsupportedOperation(
                    "im2col output shape overflow",
                ))?;
        let kernel_w_channels =
            map.kernel_w
                .checked_mul(map.channels)
                .ok_or(LowerError::UnsupportedOperation(
                    "im2col kernel shape overflow",
                ))?;

        let batch = self.div_literal_u32_emitted(expressions, row, output_pixels, body);
        let output_index =
            self.mod_literal_u32_emitted(expressions, row, output_pixels, body);
        let out_y = self.div_literal_u32_emitted(expressions, output_index, map.out_w, body);
        let out_x = self.mod_literal_u32_emitted(expressions, output_index, map.out_w, body);

        let kernel_y = self.div_literal_u32_emitted(expressions, k, kernel_w_channels, body);
        let kernel_xc = self.mod_literal_u32_emitted(expressions, k, kernel_w_channels, body);
        let kernel_x =
            self.div_literal_u32_emitted(expressions, kernel_xc, map.channels, body);
        let channel =
            self.mod_literal_u32_emitted(expressions, kernel_xc, map.channels, body);

        let out_y = self.mul_literal_u32_emitted(expressions, out_y, map.stride_h, body);
        let kernel_y =
            self.mul_literal_u32_emitted(expressions, kernel_y, map.dilation_h, body);
        let in_y = self.add_u32_expr(expressions, out_y, kernel_y, body);
        let out_x = self.mul_literal_u32_emitted(expressions, out_x, map.stride_w, body);
        let kernel_x =
            self.mul_literal_u32_emitted(expressions, kernel_x, map.dilation_w, body);
        let in_x = self.add_u32_expr(expressions, out_x, kernel_x, body);

        let terms = [
            (batch, map.batch_stride),
            (in_y, map.row_stride),
            (in_x, map.col_stride),
            (channel, map.channel_stride),
        ];
        let mut index = None;
        for (coord, stride) in terms {
            if Self::is_u32_literal(expressions, coord, 0) || stride == 0 {
                continue;
            }
            let term = self.mul_literal_u32_emitted(expressions, coord, stride, body);
            index = Some(match index {
                Some(index) => self.add_u32_expr(expressions, index, term, body),
                None => term,
            });
        }
        let index = index.unwrap_or_else(|| {
            expressions.append(Expression::Literal(Literal::U32(0)), Span::default())
        });
        Ok(index)
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
