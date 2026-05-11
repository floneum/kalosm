use super::*;

impl<'a> Lowerer<'a> {
    pub(super) fn push_emits(body: &mut Block, emits: Vec<Range<Expression>>) {
        for emit in emits {
            body.push(Statement::Emit(emit), Span::default());
        }
    }

    pub(super) fn push_guarded_or_full_block(
        body: &mut Block,
        guard_emits: Vec<Range<Expression>>,
        condition: Option<Handle<Expression>>,
        accept: Block,
    ) {
        if let Some(condition) = condition {
            Self::push_emits(body, guard_emits);
            body.push(
                Statement::If {
                    condition,
                    accept,
                    reject: Block::new(),
                },
                Span::default(),
            );
        } else if guard_emits.is_empty() {
            body.push(Statement::Block(accept), Span::default());
        } else {
            let mut block = Block::new();
            Self::push_emits(&mut block, guard_emits);
            block.push(Statement::Block(accept), Span::default());
            body.push(Statement::Block(block), Span::default());
        }
    }

    pub(super) fn single_expression_range(
        _expressions: &Arena<Expression>,
        value: Handle<Expression>,
    ) -> Range<Expression> {
        Range::new_from_bounds(value, value)
    }

    pub(super) fn range_from(
        _expressions: &Arena<Expression>,
        first: Handle<Expression>,
        second: Handle<Expression>,
    ) -> Range<Expression> {
        Range::new_from_bounds(first, second)
    }

    pub(super) fn increment_u32_local(
        &self,
        expressions: &mut Arena<Expression>,
        local: Handle<LocalVariable>,
        amount: u32,
    ) -> Statement {
        let amount = expressions.append(Expression::Literal(Literal::U32(amount)), Span::default());
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        let current = expressions.append(Expression::Load { pointer }, Span::default());
        let next = expressions.append(
            Expression::Binary {
                op: BinaryOperator::Add,
                left: current,
                right: amount,
            },
            Span::default(),
        );
        Statement::Block(Block::from_vec(vec![
            Statement::Emit(Self::range_from(expressions, current, next)),
            Statement::Store {
                pointer,
                value: next,
            },
        ]))
    }

    /// Same shape as `emit_counted_loop` but takes a dynamic `iterations`
    /// expression. Compares `loop_index >= iterations_expr` at the top of
    /// each iteration and breaks when true.
    pub(super) fn emit_dynamic_counted_loop<T>(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        iterations: Handle<Expression>,
        build_body: impl FnOnce(
            &mut Arena<Expression>,
            &mut Block,
            Handle<Expression>,
        ) -> Result<T, LowerError>,
    ) -> Result<T, LowerError> {
        let loop_ptr = expressions.append(
            Expression::LocalVariable(scratch.loop_index),
            Span::default(),
        );
        let zero = expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        body.push(
            Statement::Store {
                pointer: loop_ptr,
                value: zero,
            },
            Span::default(),
        );

        let mut loop_body = Block::new();
        let loop_index =
            expressions.append(Expression::Load { pointer: loop_ptr }, Span::default());
        loop_body.push(
            Statement::Emit(Self::single_expression_range(expressions, loop_index)),
            Span::default(),
        );
        let done = expressions.append(
            Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left: loop_index,
                right: iterations,
            },
            Span::default(),
        );
        loop_body.push(
            Statement::Emit(Self::single_expression_range(expressions, done)),
            Span::default(),
        );
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let result = build_body(expressions, &mut loop_body, loop_index)?;

        loop_body.push(
            self.increment_u32_local(expressions, scratch.loop_index, 1),
            Span::default(),
        );
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
        Ok(result)
    }

    pub(super) fn emit_counted_loop<T>(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        iterations: u32,
        build_body: impl FnOnce(
            &mut Arena<Expression>,
            &mut Block,
            Handle<Expression>,
        ) -> Result<T, LowerError>,
    ) -> Result<T, LowerError> {
        let loop_ptr = expressions.append(
            Expression::LocalVariable(scratch.loop_index),
            Span::default(),
        );
        let zero = expressions.append(Expression::Literal(Literal::U32(0)), Span::default());
        body.push(
            Statement::Store {
                pointer: loop_ptr,
                value: zero,
            },
            Span::default(),
        );

        let mut loop_body = Block::new();
        let loop_index =
            expressions.append(Expression::Load { pointer: loop_ptr }, Span::default());
        loop_body.push(
            Statement::Emit(Self::single_expression_range(expressions, loop_index)),
            Span::default(),
        );
        let done = self.bin_lit_u32(
            expressions,
            &mut loop_body,
            BinaryOperator::GreaterEqual,
            loop_index,
            iterations,
        );
        loop_body.push(
            Statement::If {
                condition: done,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );

        let result = build_body(expressions, &mut loop_body, loop_index)?;

        loop_body.push(
            self.increment_u32_local(expressions, scratch.loop_index, 1),
            Span::default(),
        );
        body.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
        Ok(result)
    }

    pub(super) fn snapshot_tile_loop_caches(&self) -> TileLoopCacheSnapshot {
        let snapshot = TileLoopCacheSnapshot {
            block_dequant: self.block_dequant_cache.borrow_mut().drain().collect(),
        };
        self.q8_activation_pack_cache.borrow_mut().clear();
        snapshot
    }

    pub(super) fn restore_tile_loop_caches(&self, snapshot: TileLoopCacheSnapshot) {
        {
            let mut cache = self.block_dequant_cache.borrow_mut();
            cache.clear();
            for (key, value) in snapshot.block_dequant {
                cache.insert(key, value);
            }
        }
        self.q8_activation_pack_cache.borrow_mut().clear();
    }

    pub(super) fn snapshot_coop_loop_caches(&self) -> CoopLoopCacheSnapshot {
        CoopLoopCacheSnapshot {
            fragments: self.coop_fragment_cache.borrow_mut().drain().collect(),
            acc_values: self.coop_acc_value_cache.borrow_mut().drain().collect(),
        }
    }

    pub(super) fn restore_coop_loop_caches(&self, snapshot: CoopLoopCacheSnapshot) {
        {
            let mut cache = self.coop_fragment_cache.borrow_mut();
            cache.clear();
            for (key, value) in snapshot.fragments {
                cache.insert(key, value);
            }
        }
        {
            let mut cache = self.coop_acc_value_cache.borrow_mut();
            cache.clear();
            for (key, value) in snapshot.acc_values {
                cache.insert(key, value);
            }
        }
    }

    pub(super) fn load_u32_local(
        &self,
        expressions: &mut Arena<Expression>,
        local: Handle<LocalVariable>,
    ) -> (Handle<Expression>, Range<Expression>) {
        let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
        let value = expressions.append(Expression::Load { pointer }, Span::default());
        (value, Self::single_expression_range(expressions, value))
    }

    pub(super) fn current_loop_index(&self) -> Handle<LocalVariable> {
        self.loop_index_local
            .expect("scratch locals must be created before lowering storage offsets")
    }

    /// Lower a `TileStmt::Fold`. Initializes each accumulator local from its
    /// `init` expression in the surrounding scope, then emits a counted loop
    /// over `iter`. Inside the loop body, the iterator value is stored into
    /// `iter_var`'s local, the body statements run, and each accumulator's
    /// `update` expression is evaluated and stored back into its local. After
    /// the loop, the locals hold the final values and are read by subsequent
    /// `LoadLocal`s in the surrounding scope.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn lower_tile_fold_stmt(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        iter: &TileIter,
        iter_var: LocalId,
        fold_body: &[TileStmt],
        accumulators: &[crate::ir::FoldAccumulator],
    ) -> Result<(), LowerError> {
        // 1. Initialize each accumulator local from its init expression.
        for acc in accumulators {
            let init_value =
                self.lower_tile_expr_lane(expressions, scratch, body, &acc.init, 0)?;
            let local = self.private_local(LocalRef::new(acc.name, acc.element))?;
            let pointer = expressions.append(Expression::LocalVariable(local), Span::default());
            body.push(
                Statement::Store {
                    pointer,
                    value: init_value,
                },
                Span::default(),
            );
        }

        // 2. Lower the iterator's count expression in the surrounding scope.
        let count_handle = match iter {
            TileIter::Range { count } => {
                self.lower_tile_expr_lane(expressions, scratch, body, count, 0)?
            }
        };

        // 3. Emit the counted loop. Inside the body, store the loop index into
        //    iter_var's local so subsequent LoadLocal(iter_var) reads see it.
        let iter_var_local =
            self.private_local(LocalRef::new(iter_var, ElementType::U32))?;
        self.emit_dynamic_counted_loop(
            expressions,
            scratch,
            body,
            count_handle,
            |expressions, loop_body, loop_index| {
                // Store the loop index into iter_var's local.
                let iter_ptr = expressions
                    .append(Expression::LocalVariable(iter_var_local), Span::default());
                loop_body.push(
                    Statement::Store {
                        pointer: iter_ptr,
                        value: loop_index,
                    },
                    Span::default(),
                );

                // Snapshot caches whose SSA handles are scoped to the outer
                // block so the body's lowering can repopulate them inside the
                // loop, then restore on exit. Coop fragments and acc-value SSA
                // chains live within one iteration only; flush at the
                // iteration boundary.
                let tile_saved = self.snapshot_tile_loop_caches();
                let coop_saved = self.snapshot_coop_loop_caches();

                for stmt in fold_body {
                    self.lower_tile_stmt(expressions, scratch, loop_body, stmt)?;
                }

                // Lower each accumulator's update expression and store it back
                // into the accumulator's local.
                for acc in accumulators {
                    let value = self.lower_tile_expr_lane(
                        expressions,
                        scratch,
                        loop_body,
                        &acc.update,
                        0,
                    )?;
                    let acc_local =
                        self.private_local(LocalRef::new(acc.name, acc.element))?;
                    let pointer = expressions
                        .append(Expression::LocalVariable(acc_local), Span::default());
                    loop_body.push(
                        Statement::Store {
                            pointer,
                            value,
                        },
                        Span::default(),
                    );
                }

                self.flush_coop_acc_cache(expressions, loop_body);
                self.restore_coop_loop_caches(coop_saved);
                self.restore_tile_loop_caches(tile_saved);
                Ok(())
            },
        )
    }

    pub(super) fn bin_lit_u32(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: u32,
    ) -> Handle<Expression> {
        let right = expressions.append(Expression::Literal(Literal::U32(right)), Span::default());
        let value = expressions.append(Expression::Binary { op, left, right }, Span::default());
        body.push(
            Statement::Emit(Self::single_expression_range(expressions, value)),
            Span::default(),
        );
        value
    }
}
