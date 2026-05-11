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

    pub(super) fn tile_base_expression(
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

    pub(super) fn storage_dynamic_pointer(
        &self,
        expressions: &mut Arena<Expression>,
        view: &StorageView,
        index: Handle<Expression>,
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        let base = self.storage_base_expression(expressions, view)?;
        let mut emits = Vec::new();
        let index = self.add_literal_u32_emitted(expressions, index, view.offset, &mut emits);
        let pointer = expressions.append(Expression::Access { base, index }, Span::default());
        emits.push(Self::single_expression_range(expressions, pointer));
        Ok((pointer, emits))
    }

    pub(super) fn storage_base_expression(
        &self,
        expressions: &mut Arena<Expression>,
        view: &StorageView,
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
        self.private_locals
            .get(local.id.index())
            .copied()
            .flatten()
            .ok_or(LowerError::UnknownLocal(local.id))
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
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        if let Some(index_map) = &view.index_map {
            return self.storage_index_from_index_map(expressions, index_map, coords);
        }

        let layout = self.storage_layout(view)?;
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

    fn storage_index_from_index_map(
        &self,
        expressions: &mut Arena<Expression>,
        index_map: &StorageIndexMap,
        coords: &[Handle<Expression>],
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        match index_map {
            StorageIndexMap::Im2ColNhwc(map) => {
                self.storage_index_from_im2col_nhwc(expressions, *map, coords)
            }
            StorageIndexMap::FlattenedMatrix(map) => {
                self.storage_index_from_flattened_matrix(expressions, map, coords)
            }
        }
    }

    fn storage_index_from_flattened_matrix(
        &self,
        expressions: &mut Arena<Expression>,
        map: &FlattenedMatrixMap,
        coords: &[Handle<Expression>],
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
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

        let mut emits = Vec::new();
        let row = coords[0];
        let col = coords[1];
        let mut remaining = row;
        let mut terms = Vec::with_capacity(map.prefix_shape.len() + 1);

        for axis in (0..map.prefix_shape.len()).rev() {
            let dim = map.prefix_shape[axis];
            let coord = if axis == 0 {
                remaining
            } else {
                let coord = self.mod_literal_u32_emitted(expressions, remaining, dim, &mut emits);
                remaining = self.div_literal_u32_emitted(expressions, remaining, dim, &mut emits);
                coord
            };
            let stride = map.prefix_strides[axis];
            if stride != 0 {
                terms.push(self.mul_literal_u32_emitted(expressions, coord, stride, &mut emits));
            }
        }

        if map.column_stride != 0 {
            terms.push(self.mul_literal_u32_emitted(
                expressions,
                col,
                map.column_stride,
                &mut emits,
            ));
        }

        let mut terms = terms.into_iter();
        let Some(mut index) = terms.next() else {
            let zero = expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
            return Ok((zero, emits));
        };
        for term in terms {
            index = Self::add_u32_expr(expressions, index, term, &mut emits);
        }
        Ok((index, emits))
    }

    fn storage_index_from_im2col_nhwc(
        &self,
        expressions: &mut Arena<Expression>,
        map: Im2ColNhwcMap,
        coords: &[Handle<Expression>],
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        if coords.len() != 2 {
            return Err(LowerError::UnsupportedOperation(
                "im2col storage views require matrix coordinates",
            ));
        }

        let mut emits = Vec::new();
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

        let batch = self.div_literal_u32_emitted(expressions, row, output_pixels, &mut emits);
        let output_index =
            self.mod_literal_u32_emitted(expressions, row, output_pixels, &mut emits);
        let out_y = self.div_literal_u32_emitted(expressions, output_index, map.out_w, &mut emits);
        let out_x = self.mod_literal_u32_emitted(expressions, output_index, map.out_w, &mut emits);

        let kernel_y = self.div_literal_u32_emitted(expressions, k, kernel_w_channels, &mut emits);
        let kernel_xc = self.mod_literal_u32_emitted(expressions, k, kernel_w_channels, &mut emits);
        let kernel_x =
            self.div_literal_u32_emitted(expressions, kernel_xc, map.channels, &mut emits);
        let channel =
            self.mod_literal_u32_emitted(expressions, kernel_xc, map.channels, &mut emits);

        let out_y = self.mul_literal_u32_emitted(expressions, out_y, map.stride_h, &mut emits);
        let kernel_y =
            self.mul_literal_u32_emitted(expressions, kernel_y, map.dilation_h, &mut emits);
        let in_y = Self::add_u32_expr(expressions, out_y, kernel_y, &mut emits);
        let out_x = self.mul_literal_u32_emitted(expressions, out_x, map.stride_w, &mut emits);
        let kernel_x =
            self.mul_literal_u32_emitted(expressions, kernel_x, map.dilation_w, &mut emits);
        let in_x = Self::add_u32_expr(expressions, out_x, kernel_x, &mut emits);

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
            let term = self.mul_literal_u32_emitted(expressions, coord, stride, &mut emits);
            index = Some(match index {
                Some(index) => Self::add_u32_expr(expressions, index, term, &mut emits),
                None => term,
            });
        }
        let index = index.unwrap_or_else(|| {
            expressions.append(Expression::Literal(Literal::U32(0)), Span::default())
        });
        Ok((index, emits))
    }

    fn add_u32_expr(
        expressions: &mut Arena<Expression>,
        left: Handle<Expression>,
        right: Handle<Expression>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        if Self::is_u32_literal(expressions, left, 0) {
            return right;
        }
        if Self::is_u32_literal(expressions, right, 0) {
            return left;
        }
        let value = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left,
                right,
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, value));
        value
    }

}
