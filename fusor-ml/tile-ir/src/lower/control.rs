use super::*;
use crate::lower::tile_program::TileFoldLowering;

impl<'a> Lowerer<'a> {
    pub(super) fn push_guarded_or_full_block(
        body: &mut Block,
        mut guard_block: Block,
        condition: Option<Handle<Expression>>,
        accept: Block,
    ) {
        if let Some(condition) = condition {
            body.append(&mut guard_block);
            body.push(
                Statement::If {
                    condition,
                    accept,
                    reject: Block::new(),
                },
                Span::default(),
            );
        } else if guard_block.is_empty() {
            body.push(Statement::Block(accept), Span::default());
        } else {
            guard_block.push(Statement::Block(accept), Span::default());
            body.push(Statement::Block(guard_block), Span::default());
        }
    }

    pub(super) fn single_expression_range(
        _expressions: &Arena<Expression>,
        value: Handle<Expression>,
    ) -> Range<Expression> {
        Range::new_from_bounds(value, value)
    }

    pub(super) fn increment_u32_local(
        &self,
        expressions: &mut Arena<Expression>,
        local: Handle<LocalVariable>,
        amount: u32,
    ) -> Statement {
        let amount = self.u32(expressions, amount);
        let pointer = self.local_var(expressions, local);
        let mut block = Block::new();
        let current = Self::emit_load(expressions, &mut block, pointer);
        let next = self.emit(
            expressions,
            &mut block,
            Expression::Binary {
                op: BinaryOperator::Add,
                left: current,
                right: amount,
            },
        );
        block.push(
            Statement::Store {
                pointer,
                value: next,
            },
            Span::default(),
        );
        Statement::Block(block)
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
        let loop_ptr = self.local_var(expressions, scratch.loop_index);
        let zero = self.u32(expressions, 0);
        body.push(
            Statement::Store {
                pointer: loop_ptr,
                value: zero,
            },
            Span::default(),
        );

        let mut loop_body = Block::new();
        let loop_index = Self::emit_load(expressions, &mut loop_body, loop_ptr);
        let done = self.emit(
            expressions,
            &mut loop_body,
            Expression::Binary {
                op: BinaryOperator::GreaterEqual,
                left: loop_index,
                right: iterations,
            },
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
        let count = self.u32(expressions, iterations);
        self.emit_dynamic_counted_loop(expressions, scratch, body, count, build_body)
    }

    pub(super) fn snapshot_tile_loop_caches(&self) -> TileLoopCacheSnapshot {
        let snapshot = TileLoopCacheSnapshot {
            block_dequant: self.block_dequant_cache.borrow_mut().drain().collect(),
        };
        self.q8_activation_pack_cache.borrow_mut().clear();
        snapshot
    }

    pub(super) fn restore_tile_loop_caches(&self, snapshot: TileLoopCacheSnapshot) {
        Self::replace_cache(&self.block_dequant_cache, snapshot.block_dequant);
        self.q8_activation_pack_cache.borrow_mut().clear();
    }

    pub(super) fn snapshot_coop_loop_caches(&self) -> CoopLoopCacheSnapshot {
        CoopLoopCacheSnapshot {
            fragments: self.coop_fragment_cache.borrow_mut().drain().collect(),
            acc_values: self.coop_acc_value_cache.borrow_mut().drain().collect(),
        }
    }

    pub(super) fn restore_coop_loop_caches(&self, snapshot: CoopLoopCacheSnapshot) {
        Self::replace_cache(&self.coop_fragment_cache, snapshot.fragments);
        Self::replace_cache(&self.coop_acc_value_cache, snapshot.acc_values);
    }

    /// Drain `cache` and refill it with `entries`. Snapshot/restore helpers
    /// use this to atomically reset a cache to a previously-recorded set.
    fn replace_cache<K: std::hash::Hash + Eq, V>(
        cache: &RefCell<HashMap<K, V>>,
        entries: Vec<(K, V)>,
    ) {
        let mut cache = cache.borrow_mut();
        cache.clear();
        cache.extend(entries);
    }

    /// Lower a `TileStmt::Fold`. Initializes each accumulator local from its
    /// `init` expression in the surrounding scope, then emits a counted loop
    /// over `0..count`. Inside the loop body, the iterator value is stored
    /// into `iter_var`'s local, the body statements run, and each
    /// accumulator's `update` expression is evaluated and stored back into its
    /// local. After the loop, the locals hold the final values and are read by
    /// subsequent `LoadLocal`s in the surrounding scope.
    pub(super) fn lower_tile_fold_stmt(
        &self,
        expressions: &mut Arena<Expression>,
        scratch: ScratchLocals,
        body: &mut Block,
        request: TileFoldLowering<'_>,
    ) -> Result<(), LowerError> {
        let TileFoldLowering {
            count,
            iter_var,
            body: fold_body,
            accumulators,
        } = request;
        // 1. Initialize each accumulator local from its init expression.
        for acc in accumulators {
            let init_value = self.lower_tile_expr(expressions, scratch, body, &acc.init)?;
            let local = self.private_local(LocalRef::new(acc.name, acc.element))?;
            self.store_local(expressions, body, local, init_value);
        }

        // 2. Lower the iterator's count expression in the surrounding scope.
        let count_handle = self.lower_tile_expr(expressions, scratch, body, count)?;

        // 3. Emit the counted loop. Inside the body, store the loop index into
        //    iter_var's local so subsequent LoadLocal(iter_var) reads see it.
        let iter_var_local = self.private_local(LocalRef::new(iter_var, ElementType::U32))?;
        self.emit_dynamic_counted_loop(
            expressions,
            scratch,
            body,
            count_handle,
            |expressions, loop_body, loop_index| {
                // Store the loop index into iter_var's local.
                self.store_local(expressions, loop_body, iter_var_local, loop_index);

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
                    let value =
                        self.lower_tile_expr(expressions, scratch, loop_body, &acc.update)?;
                    let acc_local = self.private_local(LocalRef::new(acc.name, acc.element))?;
                    self.store_local(expressions, loop_body, acc_local, value);
                }

                self.flush_coop_acc_cache(expressions, loop_body);
                self.restore_coop_loop_caches(coop_saved);
                self.restore_tile_loop_caches(tile_saved);
                Ok(())
            },
        )
    }
}
