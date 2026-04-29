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

    pub(super) fn tile_index_pointer(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        tile: TileRef,
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        self.tile_layout(tile)?;

        let (storage_tile, offset) = self.storage_tile_and_offset(tile)?;
        let global = self
            .tile_globals
            .get(storage_tile.id.index())
            .copied()
            .flatten()
            .ok_or_else(|| {
                self.tile_layout(storage_tile)
                    .map(|layout| LowerError::UnsupportedMemoryLevel(layout.memory_level()))
                    .unwrap_or(LowerError::UnknownTile(storage_tile.id))
            })?;
        let base = expressions.append(Expression::GlobalVariable(global), Span::default());
        let index_pointer =
            expressions.append(Expression::LocalVariable(tile_index), Span::default());
        let flat = expressions.append(
            Expression::Load {
                pointer: index_pointer,
            },
            Span::default(),
        );
        let mut emits = Vec::new();
        emits.push(Self::single_expression_range(expressions, flat));
        let layout = self.tile_layout(tile)?;
        let index = if layout.is_row_major() {
            flat
        } else {
            self.storage_index_from_flat(expressions, flat, layout, layout, &[], &mut emits)?
        };
        let index = self.add_literal_u32_emitted(expressions, index, offset, &mut emits);
        let pointer = expressions.append(Expression::Access { base, index }, Span::default());
        emits.push(Self::single_expression_range(expressions, pointer));
        Ok((pointer, emits))
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

    pub(super) fn storage_index_pointer_from_tile_index_with_base(
        &self,
        expressions: &mut Arena<Expression>,
        tile_index: Handle<LocalVariable>,
        dst_layout: &Layout,
        view: &StorageView,
        base: Handle<Expression>,
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        let src_layout = self.storage_layout(view)?;
        if dst_layout.shape() != src_layout.shape() {
            return Err(LowerError::UnsupportedOperation("load shape mismatch"));
        }
        let mut emits = Vec::new();
        let index_pointer =
            expressions.append(Expression::LocalVariable(tile_index), Span::default());
        let flat = expressions.append(
            Expression::Load {
                pointer: index_pointer,
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, flat));
        let logical_index = self.storage_index_from_flat(
            expressions,
            flat,
            dst_layout,
            src_layout,
            &view.dynamic_offsets,
            &mut emits,
        )?;
        let index =
            self.add_literal_u32_emitted(expressions, logical_index, view.offset, &mut emits);
        let pointer = expressions.append(Expression::Access { base, index }, Span::default());
        emits.push(Self::single_expression_range(expressions, pointer));
        Ok((pointer, emits))
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

    pub(super) fn index_from_flat(
        &self,
        expressions: &mut Arena<Expression>,
        flat: Handle<Expression>,
        logical_layout: &Layout,
        target_layout: &Layout,
        dynamic_offsets: &[Option<DynamicOffset>],
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        let mut emits = Vec::new();
        let index = self.storage_index_from_flat(
            expressions,
            flat,
            logical_layout,
            target_layout,
            dynamic_offsets,
            &mut emits,
        )?;
        Ok((index, emits))
    }

    pub(super) fn layout_index_expr(
        &self,
        expressions: &mut Arena<Expression>,
        layout: &Layout,
        coords: &[Handle<Expression>],
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        if layout.strides().rank() != coords.len() {
            return Err(LowerError::UnsupportedOperation("layout rank mismatch"));
        }
        let mut emits = Vec::new();
        let mut terms = Vec::with_capacity(coords.len());
        for (coord, stride) in coords.iter().zip(layout.strides().values()) {
            if Self::is_u32_literal(expressions, *coord, 0) || *stride == 0 {
                continue;
            }
            terms.push(self.mul_literal_u32_emitted(expressions, *coord, *stride, &mut emits));
        }
        let mut terms = terms.into_iter();
        let Some(mut index) = terms.next() else {
            let zero = expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
            return Ok((zero, emits));
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
        self.storage_index_from_coords_filtered(expressions, view, coords, false)
    }

    pub(super) fn storage_index_from_coords_without_loop_offsets(
        &self,
        expressions: &mut Arena<Expression>,
        view: &StorageView,
        coords: &[Handle<Expression>],
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        self.storage_index_from_coords_filtered(expressions, view, coords, true)
    }

    pub(super) fn storage_linear_base_without_loop_offsets(
        &self,
        expressions: &mut Arena<Expression>,
        view: &StorageView,
    ) -> Result<(Option<Handle<Expression>>, Vec<Range<Expression>>), LowerError> {
        let layout = self.storage_layout(view)?;
        let mut emits = Vec::new();
        let mut terms = Vec::new();
        for (axis_index, stride) in layout.strides().values().iter().copied().enumerate() {
            if let Some(DynamicOffset::Workgroup(offset)) =
                view.dynamic_offsets.get(axis_index).copied().flatten()
            {
                let workgroup_id = expressions.append(
                    Expression::FunctionArgument(WORKGROUP_ID_ARG),
                    Span::default(),
                );
                let axis = expressions.append(
                    Expression::AccessIndex {
                        base: workgroup_id,
                        index: offset.axis.index(),
                    },
                    Span::default(),
                );
                emits.push(Self::single_expression_range(expressions, axis));
                let scaled =
                    self.mul_literal_u32_emitted(expressions, axis, offset.scale, &mut emits);
                terms.push(self.mul_literal_u32_emitted(expressions, scaled, stride, &mut emits));
            }
        }

        let mut terms = terms.into_iter();
        let Some(mut base) = terms.next() else {
            return Ok((None, emits));
        };
        for term in terms {
            base = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: base,
                    right: term,
                },
                Span::default(),
            );
            emits.push(Self::single_expression_range(expressions, base));
        }
        Ok((Some(base), emits))
    }

    pub(super) fn storage_dynamic_base_index(
        &self,
        expressions: &mut Arena<Expression>,
        view: &StorageView,
    ) -> Result<(Option<Handle<Expression>>, Vec<Range<Expression>>), LowerError> {
        let layout = self.storage_layout(view)?;
        let mut emits = Vec::new();
        let mut terms = Vec::new();
        if view.offset != 0 {
            terms.push(expressions.append(
                Expression::Literal(Literal::U32(view.offset)),
                Span::default(),
            ));
        }
        for (axis_index, stride) in layout.strides().values().iter().copied().enumerate() {
            let Some(offset) = view.dynamic_offsets.get(axis_index).copied().flatten() else {
                continue;
            };
            let scaled = self.dynamic_offset_scaled(expressions, offset, &mut emits);
            terms.push(self.mul_literal_u32_emitted(expressions, scaled, stride, &mut emits));
        }

        let mut terms = terms.into_iter();
        let Some(mut base) = terms.next() else {
            return Ok((None, emits));
        };
        for term in terms {
            base = expressions.append(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: base,
                    right: term,
                },
                Span::default(),
            );
            emits.push(Self::single_expression_range(expressions, base));
        }
        Ok((Some(base), emits))
    }

    pub(super) fn add_optional_base_u32_emitted(
        &self,
        expressions: &mut Arena<Expression>,
        value: Handle<Expression>,
        base: Option<Handle<Expression>>,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        let Some(base) = base else {
            return value;
        };
        let value = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: value,
                right: base,
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, value));
        value
    }

    pub(super) fn storage_index_from_coords_filtered(
        &self,
        expressions: &mut Arena<Expression>,
        view: &StorageView,
        coords: &[Handle<Expression>],
        skip_loop_offsets: bool,
    ) -> Result<(Handle<Expression>, Vec<Range<Expression>>), LowerError> {
        let layout = self.storage_layout(view)?;
        if layout.strides().rank() != coords.len() {
            return Err(LowerError::UnsupportedOperation("layout rank mismatch"));
        }

        let mut emits = Vec::new();
        let mut terms = Vec::with_capacity(coords.len());
        for (axis, (coord, stride)) in coords.iter().zip(layout.strides().values()).enumerate() {
            let coord = self.apply_dynamic_offset_filtered(
                expressions,
                *coord,
                &view.dynamic_offsets,
                axis,
                skip_loop_offsets,
                &mut emits,
            );
            terms.push(self.mul_literal_u32_emitted(expressions, coord, *stride, &mut emits));
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

    pub(super) fn storage_index_from_flat(
        &self,
        expressions: &mut Arena<Expression>,
        flat: Handle<Expression>,
        dst_layout: &Layout,
        src_layout: &Layout,
        dynamic_offsets: &[Option<DynamicOffset>],
        emits: &mut Vec<Range<Expression>>,
    ) -> Result<Handle<Expression>, LowerError> {
        match dst_layout.shape().rank() {
            1 => {
                let coord = self.apply_dynamic_offset(expressions, flat, dynamic_offsets, 0, emits);
                Ok(self.mul_literal_u32_emitted(
                    expressions,
                    coord,
                    src_layout.strides().values()[0],
                    emits,
                ))
            }
            2 => {
                let cols = expressions.append(
                    Expression::Literal(Literal::U32(dst_layout.shape().dims()[1].get())),
                    Span::default(),
                );
                let row = expressions.append(
                    Expression::Binary {
                        op: BinaryOperator::Divide,
                        left: flat,
                        right: cols,
                    },
                    Span::default(),
                );
                emits.push(Self::single_expression_range(expressions, row));
                let col = expressions.append(
                    Expression::Binary {
                        op: BinaryOperator::Modulo,
                        left: flat,
                        right: cols,
                    },
                    Span::default(),
                );
                emits.push(Self::single_expression_range(expressions, col));
                let row = self.apply_dynamic_offset(expressions, row, dynamic_offsets, 0, emits);
                let col = self.apply_dynamic_offset(expressions, col, dynamic_offsets, 1, emits);
                let row = self.mul_literal_u32_emitted(
                    expressions,
                    row,
                    src_layout.strides().values()[0],
                    emits,
                );
                let col = self.mul_literal_u32_emitted(
                    expressions,
                    col,
                    src_layout.strides().values()[1],
                    emits,
                );
                let index = expressions.append(
                    Expression::Binary {
                        op: BinaryOperator::Add,
                        left: row,
                        right: col,
                    },
                    Span::default(),
                );
                emits.push(Self::single_expression_range(expressions, index));
                Ok(index)
            }
            _ => Err(LowerError::UnsupportedOperation("rank > 2 storage view")),
        }
    }

    pub(super) fn apply_dynamic_offset(
        &self,
        expressions: &mut Arena<Expression>,
        coord: Handle<Expression>,
        dynamic_offsets: &[Option<DynamicOffset>],
        axis_index: usize,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        self.apply_dynamic_offset_filtered(
            expressions,
            coord,
            dynamic_offsets,
            axis_index,
            false,
            emits,
        )
    }

    pub(super) fn apply_dynamic_offset_filtered(
        &self,
        expressions: &mut Arena<Expression>,
        coord: Handle<Expression>,
        dynamic_offsets: &[Option<DynamicOffset>],
        axis_index: usize,
        skip_loop_offsets: bool,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        let Some(Some(offset)) = dynamic_offsets.get(axis_index) else {
            return coord;
        };
        if skip_loop_offsets && matches!(offset, DynamicOffset::Loop(_)) {
            return coord;
        }
        let scaled = self.dynamic_offset_scaled(expressions, *offset, emits);
        let coord = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: coord,
                right: scaled,
            },
            Span::default(),
        );
        emits.push(Self::single_expression_range(expressions, coord));
        coord
    }

    pub(super) fn dynamic_offset_scaled(
        &self,
        expressions: &mut Arena<Expression>,
        offset: DynamicOffset,
        emits: &mut Vec<Range<Expression>>,
    ) -> Handle<Expression> {
        match offset {
            DynamicOffset::Workgroup(offset) => {
                let workgroup_id = expressions.append(
                    Expression::FunctionArgument(WORKGROUP_ID_ARG),
                    Span::default(),
                );
                let axis = expressions.append(
                    Expression::AccessIndex {
                        base: workgroup_id,
                        index: offset.axis.index(),
                    },
                    Span::default(),
                );
                emits.push(Self::single_expression_range(expressions, axis));
                self.mul_literal_u32_emitted(expressions, axis, offset.scale, emits)
            }
            DynamicOffset::Loop(offset) => {
                let (loop_index, loop_emit) =
                    self.load_u32_local(expressions, self.current_loop_index());
                emits.push(loop_emit);
                self.mul_literal_u32_emitted(expressions, loop_index, offset.scale, emits)
            }
        }
    }
}
