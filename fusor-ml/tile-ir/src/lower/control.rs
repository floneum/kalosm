use super::*;

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
    /// expression. Compares `loop_index >= iterations_expr` at the top of each
    /// iteration and breaks when true.
    pub(super) fn emit_dynamic_counted_loop<T>(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        iterations: Handle<Expression>,
        build_body: impl FnOnce(
            &mut Arena<Expression>,
            &mut Block,
            Handle<Expression>,
        ) -> Result<T, LowerError>,
    ) -> Result<T, LowerError> {
        let loop_local = self.scratch_u32(ScratchKind::LoopIndex, 0);
        let loop_ptr = self.local_var(expressions, loop_local);
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
            self.increment_u32_local(expressions, loop_local, 1),
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
        body: &mut Block,
        iterations: u32,
        build_body: impl FnOnce(
            &mut Arena<Expression>,
            &mut Block,
            Handle<Expression>,
        ) -> Result<T, LowerError>,
    ) -> Result<T, LowerError> {
        let count = self.u32(expressions, iterations);
        self.emit_dynamic_counted_loop(expressions, body, count, build_body)
    }

    // ---- per-iteration cache snapshot / restore ----

    pub(super) fn snapshot_loop_caches(&self) -> LoopCacheSnapshot {
        let snapshot = LoopCacheSnapshot {
            dequant_memo: self.dequant_memo.borrow_mut().drain().collect(),
            expr_memo: self.expr_memo.borrow_mut().drain().collect(),
        };
        self.q8_activation_pack_cache.borrow_mut().clear();
        snapshot
    }

    pub(super) fn restore_loop_caches(&self, snapshot: LoopCacheSnapshot) {
        Self::replace_cache(&self.dequant_memo, snapshot.dequant_memo);
        Self::replace_cache(&self.expr_memo, snapshot.expr_memo);
        self.q8_activation_pack_cache.borrow_mut().clear();
    }

    pub(super) fn lower_branch_block(
        &self,
        expressions: &mut Arena<Expression>,
        stmts: &[Stmt],
    ) -> Result<Block, LowerError> {
        let saved = self.snapshot_loop_caches();
        let mut block = Block::new();
        let result = self.lower_stmt_body(expressions, &mut block, stmts);
        self.restore_loop_caches(saved);
        result.map(|()| block)
    }

    pub(super) fn snapshot_coop_loop_caches(&self) -> CoopLoopCacheSnapshot {
        CoopLoopCacheSnapshot {
            acc_values: self.coop_acc_value_cache.borrow_mut().drain().collect(),
        }
    }

    pub(super) fn restore_coop_loop_caches(&self, snapshot: CoopLoopCacheSnapshot) {
        Self::replace_cache(&self.coop_acc_value_cache, snapshot.acc_values);
    }

    /// Drain `cache` and refill it with `entries`. Snapshot/restore helpers use
    /// this to atomically reset a cache to a previously-recorded set.
    fn replace_cache<K: std::hash::Hash + Eq, V>(
        cache: &RefCell<FxHashMap<K, V>>,
        entries: Vec<(K, V)>,
    ) {
        let mut cache = cache.borrow_mut();
        cache.clear();
        cache.extend(entries);
    }

    /// Lower a counted `Stmt::Loop` (`count: Some(..)`). Initializes each
    /// accumulator local from its `init` expression in the surrounding scope,
    /// then emits a counted loop over `0..count`. Inside the loop body, the
    /// iterator value is stored into `index`'s local, the body statements run,
    /// and each accumulator's `update` expression is evaluated and stored back.
    pub(super) fn lower_counted_loop(
        &self,
        expressions: &mut Arena<Expression>,
        body: &mut Block,
        count: &Expr,
        index: Option<&Local>,
        accumulators: &[Accumulator],
        loop_body: &[Stmt],
    ) -> Result<(), LowerError> {
        // 1. Initialize each accumulator local from its init expression.
        for acc in accumulators {
            let init_value = self.lower_expr(expressions, body, &acc.init)?;
            let local = self.private_local(&acc.local)?;
            self.store_local(expressions, body, local, init_value);
        }

        // 2. Lower the iterator's count expression in the surrounding scope.
        let count_handle = self.lower_expr(expressions, body, count)?;

        // 3. Emit the counted loop, storing the loop index into `index`'s local.
        let iter_var_local = index.map(|i| self.private_local(i)).transpose()?;
        self.emit_dynamic_counted_loop(
            expressions,
            body,
            count_handle,
            |expressions, loop_block, loop_index| {
                if let Some(iter_var_local) = iter_var_local {
                    self.store_local(expressions, loop_block, iter_var_local, loop_index);
                }

                // Snapshot caches whose SSA handles are scoped to the outer
                // block so the body's lowering can repopulate them inside the
                // loop, then restore on exit. Coop fragments and acc-value SSA
                // chains live within one iteration only; flush at the iteration
                // boundary.
                let saved = self.snapshot_loop_caches();
                let coop_saved = self.snapshot_coop_loop_caches();

                for stmt in loop_body {
                    self.lower_stmt(expressions, loop_block, stmt)?;
                }

                for acc in accumulators {
                    let value = self.lower_expr(expressions, loop_block, &acc.update)?;
                    let acc_local = self.private_local(&acc.local)?;
                    self.store_local(expressions, loop_block, acc_local, value);
                }

                self.flush_coop_acc_cache(expressions, loop_block);
                self.restore_coop_loop_caches(coop_saved);
                self.restore_loop_caches(saved);
                Ok(())
            },
        )
    }
}
