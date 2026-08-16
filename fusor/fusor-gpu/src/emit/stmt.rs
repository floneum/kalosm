//! Kernel statements -> naga blocks. `Barrier` becomes `controlBarrier`;
//! `AtomicAdd` becomes `atomicAdd`, or a bitcast compare-exchange loop on f32.

use fusor_ir::ir::kernel::{
    Accumulator, Addr, ElementType, ScalarElement, Source, Stmt, StorageView, Tile, TileExpr,
};
use fusor_ir::target::EmitError;
use naga::{
    AtomicFunction, Barrier, BinaryOperator, Block, Expression, Handle, Scalar, ScalarKind, Span,
    Statement,
};

use super::coop::{row_major_tile_stride, tile_shape};
use super::{Emitter, ScratchKind, key};

impl Emitter<'_> {
    pub(crate) fn block(&mut self, body: &[Stmt]) -> Result<Block, EmitError> {
        let (block, ()) = self.nested(|em, block| {
            for stmt in body {
                em.stmt(stmt, block)?;
            }
            em.flush_coop_acc(block);
            Ok(())
        })?;
        Ok(block)
    }

    /// Emit one statement, then retire the memoized reads it invalidated.
    ///
    /// The invalidation runs *after* the body so the statement's own operands
    /// still see the memo state they were built against — a store reads its
    /// value expression before it writes anything.
    pub(crate) fn stmt(&mut self, stmt: &Stmt, out: &mut Block) -> Result<(), EmitError> {
        let result = self.stmt_inner(stmt, out);
        let written = stmt.writes();
        if !written.is_empty() {
            self.invalidate_mem(written);
        }
        result
    }

    fn stmt_inner(&mut self, stmt: &Stmt, out: &mut Block) -> Result<(), EmitError> {
        match stmt {
            Stmt::Store {
                dst,
                addr,
                value,
                mask,
            } => self.store(out, dst, addr, value, mask),
            Stmt::AtomicAdd {
                dst,
                addr,
                value,
                mask,
            } => self.atomic_add(out, dst, addr, value, mask),
            Stmt::StoreLocal { dst, value } => {
                // A coop accumulator fed by an MMA stays in SSA: the store is
                // deferred to the next flush, giving one Load, N MMAs and one
                // Store per scope. Any other coop-valued store (zero-init,
                // reset) writes through and drops the memo entry.
                let is_coop = matches!(dst.element, ElementType::CoopMatrix { .. });
                if is_coop
                    && matches!(
                        value.kind(),
                        fusor_ir::ir::kernel::TileExprKind::CoopMma { .. }
                    )
                {
                    let next = self.expr(value, out)?;
                    self.coop_acc.insert(key(dst), next);
                    return Ok(());
                }
                let v = self.expr(value, out)?;
                let local = self.private_local(dst)?;
                if is_coop {
                    self.coop_acc.remove(&key(dst));
                }
                self.store_local(out, local, v);
                Ok(())
            }
            Stmt::StoreTile { dst, index, value } => {
                let source = value.element();
                let v = self.expr(value, out)?;
                // Mixed-precision staging: a per-lane f32 stored into an f16
                // tile (or the reverse) converts on the way in.
                let v = match (source, dst.element) {
                    (a, b) if a == b => v,
                    (
                        ElementType::Scalar(ScalarElement::F32),
                        ElementType::Scalar(ScalarElement::F16),
                    )
                    | (
                        ElementType::Scalar(ScalarElement::F16),
                        ElementType::Scalar(ScalarElement::F32),
                    ) => self.cast_tile_value(out, v, source, dst.element)?,
                    _ => v,
                };
                let index = self.expr(index, out)?;
                let ptr = self.tile_dynamic_pointer(out, dst, index)?;
                self.store_tile_value(out, dst, ptr, v)
            }
            Stmt::FillTile { dst, value, bounds } => self.fill_tile(out, dst, value, bounds),
            Stmt::CoopStore { acc, dst, addr } => self.coop_store(acc, dst, addr, out),
            Stmt::CoopStoreTile {
                acc,
                tile,
                row,
                col,
            } => self.coop_store_tile(acc, tile, row, col, out),
            Stmt::If {
                condition,
                accept,
                reject,
            } => {
                let cond_ty = condition.element();
                let c = self.expr(condition, out)?;
                let c = self.condition_value(out, c, cond_ty)?;
                let accept_block = self.block(accept)?;
                let reject_block = self.block(reject)?;
                out.push(
                    Statement::If {
                        condition: c,
                        accept: accept_block,
                        reject: reject_block,
                    },
                    Span::default(),
                );
                Ok(())
            }
            Stmt::Loop {
                count: Some(count),
                index,
                accumulators,
                body,
            } => self.counted_loop(out, count, index.as_ref(), accumulators, body),
            Stmt::Loop {
                count: None, body, ..
            } => {
                self.flush_coop_acc(out);
                let loop_body = self.block(body)?;
                out.push(
                    Statement::Loop {
                        body: loop_body,
                        continuing: Block::new(),
                        break_if: None,
                    },
                    Span::default(),
                );
                Ok(())
            }
            // The N-ary reduction. One lane with a hardware operator takes the
            // *existing* expression path, so a single-slot fold that reaches
            // here emits `subgroupAdd` / the shared-memory tree unchanged.
            Stmt::Reduce {
                kind,
                values,
                merge,
                fast,
                outs,
                scratch,
            } => {
                if let (1, Some(op)) = (values.len(), *fast) {
                    let v = self.reduce(op, kind, &values[0], out)?;
                    let local = self.private_local(&outs[0])?;
                    self.store_local(out, local, v);
                    return Ok(());
                }
                self.reduce_n(kind, values, merge, scratch, outs, out)
            }
            Stmt::Break => {
                out.push(Statement::Break, Span::default());
                Ok(())
            }
            Stmt::Return => {
                out.push(Statement::Return { value: None }, Span::default());
                Ok(())
            }
            Stmt::Barrier => {
                out.push(
                    Statement::ControlBarrier(Barrier::WORK_GROUP),
                    Span::default(),
                );
                Ok(())
            }
            Stmt::StorageBarrier => {
                out.push(Statement::ControlBarrier(Barrier::STORAGE), Span::default());
                Ok(())
            }
        }
    }

    fn store(
        &mut self,
        out: &mut Block,
        dst: &StorageView,
        addr: &Addr,
        value: &TileExpr,
        mask: &TileExpr,
    ) -> Result<(), EmitError> {
        // A store closes a value scope: nothing computed for the previous
        // store may be reused, because the next one may run under a different
        // predicate.
        let v = self.expr(value, out)?;
        if mask.is_constant_true() {
            let index = self.addr_index(out, dst, addr)?;
            let ptr = self.storage_dynamic_pointer(out, dst, index)?;
            out.push(
                Statement::Store {
                    pointer: ptr,
                    value: v,
                },
                Span::default(),
            );
            return Ok(());
        }
        let mask_ty = mask.element();
        let m = self.expr(mask, out)?;
        let m = self.condition_value(out, m, mask_ty)?;
        let dst = dst.clone();
        let addr = addr.clone();
        let (accept, ()) = self.nested(|em, accept| {
            let index = em.addr_index(accept, &dst, &addr)?;
            let ptr = em.storage_dynamic_pointer(accept, &dst, index)?;
            accept.push(
                Statement::Store {
                    pointer: ptr,
                    value: v,
                },
                Span::default(),
            );
            Ok(())
        })?;
        out.push(
            Statement::If {
                condition: m,
                accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        Ok(())
    }

    /// `AtomicAdd`. U32/I32 are one `atomicAdd`; f32 is a bitcast
    /// compare-exchange loop.
    fn atomic_add(
        &mut self,
        out: &mut Block,
        dst: &StorageView,
        addr: &Addr,
        value: &TileExpr,
        mask: &TileExpr,
    ) -> Result<(), EmitError> {
        let element = dst.buffer.element;
        let v = self.expr(value, out)?;
        let masked = !mask.is_constant_true();
        let condition = if masked {
            let mask_ty = mask.element();
            let m = self.expr(mask, out)?;
            Some(self.condition_value(out, m, mask_ty)?)
        } else {
            None
        };

        let dst_c = dst.clone();
        let addr_c = addr.clone();
        let (body, ()) = self.nested(|em, block| {
            let index = em.addr_index(block, &dst_c, &addr_c)?;
            let ptr = em.storage_dynamic_pointer(block, &dst_c, index)?;
            match element {
                ElementType::Scalar(ScalarElement::U32 | ScalarElement::I32) => {
                    block.push(
                        Statement::Atomic {
                            pointer: ptr,
                            fun: AtomicFunction::Add,
                            value: v,
                            result: None,
                        },
                        Span::default(),
                    );
                    Ok(())
                }
                ElementType::Scalar(ScalarElement::F32) => em.f32_atomic_add(block, ptr, v),
                other => Err(EmitError::Unsupported(format!(
                    "atomic add on {other:?} is undefined"
                ))),
            }
        })?;

        match condition {
            Some(c) => out.push(
                Statement::If {
                    condition: c,
                    accept: body,
                    reject: Block::new(),
                },
                Span::default(),
            ),
            None => out.push(Statement::Block(body), Span::default()),
        }
        Ok(())
    }

    /// `loop { old = *p; new = bitcast(bitcast(old) + v); cas(p, old, new);
    /// if exchanged { break } }`.
    fn f32_atomic_add(
        &mut self,
        out: &mut Block,
        pointer: Handle<Expression>,
        value: Handle<Expression>,
    ) -> Result<(), EmitError> {
        let cas_ty = self.module.generate_predeclared_type(
            naga::PredeclaredType::AtomicCompareExchangeWeakResult(Scalar::U32),
        );
        let mut body = Block::new();
        let old = self.emit_load(&mut body, pointer);
        let old_f = self.cast_as(&mut body, old, ScalarKind::Float, None);
        let sum = self.bin(&mut body, BinaryOperator::Add, old_f, value);
        let new_u = self.cast_as(&mut body, sum, ScalarKind::Uint, None);
        let result = self.append(Expression::AtomicResult {
            ty: cas_ty,
            comparison: true,
        });
        body.push(
            Statement::Atomic {
                pointer,
                fun: AtomicFunction::Exchange { compare: Some(old) },
                value: new_u,
                result: Some(result),
            },
            Span::default(),
        );
        let exchanged = self.emit_expr(
            &mut body,
            Expression::AccessIndex {
                base: result,
                index: 1,
            },
        );
        body.push(
            Statement::If {
                condition: exchanged,
                accept: Block::from_vec(vec![Statement::Break]),
                reject: Block::new(),
            },
            Span::default(),
        );
        out.push(
            Statement::Loop {
                body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
        Ok(())
    }

    /// A counted loop with SSA-carried accumulators: each accumulator local is
    /// initialised in the surrounding scope, updated at the end of every
    /// iteration, and readable after the loop.
    fn counted_loop(
        &mut self,
        out: &mut Block,
        count: &TileExpr,
        index: Option<&fusor_ir::ir::kernel::Local>,
        accumulators: &[Accumulator],
        body: &[Stmt],
    ) -> Result<(), EmitError> {
        // Every accumulator is read at the value it had entering the step,
        // then all are written: a loop carrying `(n, mean, m2)` has `mean`'s
        // update read `n`, and writing `n` first would make it read the new
        // count.
        let inits: Vec<Handle<Expression>> = accumulators
            .iter()
            .map(|acc| self.expr(&acc.init, out))
            .collect::<Result<_, _>>()?;
        for (acc, init) in accumulators.iter().zip(inits) {
            let local = self.private_local(&acc.local)?;
            self.store_local(out, local, init);
        }
        let count_h = self.expr(count, out)?;
        let iter_local = index.map(|i| self.private_local(i)).transpose()?;
        let accumulators = accumulators.to_vec();
        let body = body.to_vec();
        self.dynamic_loop(out, count_h, move |em, loop_body, loop_index| {
            if let Some(local) = iter_local {
                em.store_local(loop_body, local, loop_index);
            }
            for stmt in &body {
                em.stmt(stmt, loop_body)?;
            }
            let values: Vec<Handle<Expression>> = accumulators
                .iter()
                .map(|acc| em.expr(&acc.update, loop_body))
                .collect::<Result<_, _>>()?;
            for (acc, value) in accumulators.iter().zip(values) {
                let local = em.private_local(&acc.local)?;
                em.store_local(loop_body, local, value);
            }
            em.flush_coop_acc(loop_body);
            Ok(())
        })
    }

    /// `i = 0; loop { if i >= n { break } ...; i += 1 }`. Coop and expression
    /// memos are scoped to one iteration: a value cached in iteration `i` is
    /// not reused in `i + 1`.
    pub(crate) fn dynamic_loop(
        &mut self,
        out: &mut Block,
        iterations: Handle<Expression>,
        build: impl FnOnce(&mut Self, &mut Block, Handle<Expression>) -> Result<(), EmitError>,
    ) -> Result<(), EmitError> {
        self.flush_coop_acc(out);
        let depth = self.depth;
        let loop_local = self.scratch_local(
            ScratchKind::LoopIndex,
            ElementType::Scalar(ScalarElement::U32),
            depth,
        )?;
        let zero = self.u32_lit(0);
        self.store_local(out, loop_local, zero);

        self.depth += 1;
        let saved_coop = std::mem::take(&mut self.coop_acc);
        let (mut loop_body, ()) = self.nested(|em, loop_body| {
            let loop_index = em.load_local_handle(loop_body, loop_local);
            let done = em.bin(
                loop_body,
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
            build(em, loop_body, loop_index)?;
            em.flush_coop_acc(loop_body);
            Ok(())
        })?;
        self.coop_acc = saved_coop;
        self.depth -= 1;

        // `i += 1` closes the iteration.
        let one = self.u32_lit(1);
        let ptr = self.local_var(loop_local);
        let current = self.emit_load(&mut loop_body, ptr);
        let next = self.bin(&mut loop_body, BinaryOperator::Add, current, one);
        loop_body.push(
            Statement::Store {
                pointer: ptr,
                value: next,
            },
            Span::default(),
        );

        out.push(
            Statement::Loop {
                body: loop_body,
                continuing: Block::new(),
                break_if: None,
            },
            Span::default(),
        );
        Ok(())
    }

    /// `FillTile` is collective: it is the only form whose vectorized and
    /// guard-free variants the emitter can select. Lane
    /// enumeration order comes from
    /// [`fusor_ir::shape::MultiFlattenMap::axis_unit_run`] — lanes advance
    /// along the axis whose unit-stride runs they can actually follow.
    fn fill_tile(
        &mut self,
        out: &mut Block,
        dst: &Tile,
        value: &TileExpr,
        bounds: &[Option<TileExpr>; 2],
    ) -> Result<(), EmitError> {
        let fusor_ir::ir::kernel::TileExprKind::Load { src, addr, .. } = value.kind() else {
            return Err(EmitError::Unsupported(
                "FillTile's value must be a Load".into(),
            ));
        };
        let Addr::Rc2 { row, col } = &**addr else {
            return Err(EmitError::Unsupported(
                "FillTile's value must be a rank-2 Load".into(),
            ));
        };

        let [rows, cols] = tile_shape(dst)?;
        let stride = row_major_tile_stride(dst)?;
        let row_base = self.expr(row, out)?;
        let col_base = self.expr(col, out)?;
        let bounded = bounds.iter().any(Option::is_some);
        let row_limit = match &bounds[0] {
            Some(b) => Some(self.expr(b, out)?),
            None => None,
        };
        let col_limit = match &bounds[1] {
            Some(b) => Some(self.expr(b, out)?),
            None => None,
        };

        match src {
            Source::Storage(view) => {
                let indexing = &view.layout.indexing;
                let cols_fastest = !indexing.is_affine()
                    || indexing.rank() != 2
                    || indexing.axis_unit_run(1) >= indexing.axis_unit_run(0);
                const VEC: u32 = 4;
                if !bounded && cols_fastest && cols.is_multiple_of(VEC) && unit_stride(view, 1) {
                    return self
                        .fill_vec4(out, dst, view, rows, cols, stride, row_base, col_base, true);
                }
                if !bounded && !cols_fastest && rows.is_multiple_of(VEC) && unit_stride(view, 0) {
                    return self.fill_vec4(
                        out, dst, view, rows, cols, stride, row_base, col_base, false,
                    );
                }
                let view = view.clone();
                let total = rows
                    .checked_mul(cols)
                    .ok_or_else(|| EmitError::Unsupported("workgroup tile size overflow".into()))?;
                let dst = dst.clone();
                self.copy_passes(out, total, move |em, accept, flat| {
                    let (local_row, local_col) =
                        em.lane_coords(accept, flat, rows, cols, cols_fastest);
                    let global_row = em.add_u32(accept, row_base, local_row);
                    let global_col = em.add_u32(accept, col_base, local_col);
                    let tile_index = em.tile_matrix_index(accept, local_row, local_col, stride);
                    let tile_ptr = em.tile_dynamic_pointer(accept, &dst, tile_index)?;
                    let in_bounds =
                        em.bounds_check(accept, global_row, global_col, row_limit, col_limit);
                    let load = |em: &mut Self, block: &mut Block| -> Result<_, EmitError> {
                        let index =
                            em.storage_index_from_coords(block, &view, &[global_row, global_col])?;
                        let ptr = em.storage_dynamic_pointer(block, &view, index)?;
                        let v = em.emit_load(block, ptr);
                        em.cast_tile_value(block, v, view.buffer.element, dst.element)
                    };
                    em.guarded_tile_store(accept, &dst, tile_ptr, in_bounds, load)
                })
            }
            // A block-quantized operand never reaches the collective fill:
            // `stage_operand_tile` stages it one lane at a time as a
            // `Load` + `StoreTile`, because `pre` has to run per element on
            // the way in.
            Source::Quantized(_) => Err(EmitError::Unsupported(
                "FillTile's source must be dense storage: a quantized operand \
                 stages through per-lane Load + StoreTile"
                    .into(),
            )),
        }
    }

    /// Split `total` element slots across `workgroup_invocations` lanes,
    /// guarding only the ragged final pass.
    fn copy_passes(
        &mut self,
        out: &mut Block,
        total: u32,
        mut build: impl FnMut(&mut Self, &mut Block, Handle<Expression>) -> Result<(), EmitError>,
    ) -> Result<(), EmitError> {
        let lanes = self.workgroup_invocations;
        let passes = total.div_ceil(lanes.max(1));
        for pass in 0..passes {
            let full = (pass + 1) * lanes <= total;
            let lane = self.lane();
            let flat = self.add_literal_u32(out, lane, pass * lanes);
            let condition = if full {
                None
            } else {
                let limit = self.u32_lit(total);
                Some(self.bin(out, BinaryOperator::Less, flat, limit))
            };
            let (accept, ()) = self.nested(|em, accept| build(em, accept, flat))?;
            match condition {
                Some(c) => out.push(
                    Statement::If {
                        condition: c,
                        accept,
                        reject: Block::new(),
                    },
                    Span::default(),
                ),
                None => out.push(Statement::Block(accept), Span::default()),
            }
        }
        Ok(())
    }

    /// The vec4 fast path: four consecutive unit-stride elements per lane.
    #[allow(clippy::too_many_arguments)]
    fn fill_vec4(
        &mut self,
        out: &mut Block,
        dst: &Tile,
        view: &StorageView,
        rows: u32,
        cols: u32,
        stride: u32,
        row_base: Handle<Expression>,
        col_base: Handle<Expression>,
        cols_fastest: bool,
    ) -> Result<(), EmitError> {
        const VEC: u32 = 4;
        let (groups_per_line, lines) = if cols_fastest {
            (cols / VEC, rows)
        } else {
            (rows / VEC, cols)
        };
        let total = lines
            .checked_mul(groups_per_line)
            .ok_or_else(|| EmitError::Unsupported("workgroup tile size overflow".into()))?;
        let view = view.clone();
        let dst = dst.clone();
        self.copy_passes(out, total, move |em, accept, flat| {
            let line = em.div_literal_u32(accept, flat, groups_per_line);
            let group = em.mod_literal_u32(accept, flat, groups_per_line);
            let group_base = em.mul_literal_u32(accept, group, VEC);
            let (local_row, local_col) = if cols_fastest {
                (line, group_base)
            } else {
                (group_base, line)
            };
            let global_row = em.add_u32(accept, row_base, local_row);
            let global_col = em.add_u32(accept, col_base, local_col);
            let storage_base =
                em.storage_index_from_coords(accept, &view, &[global_row, global_col])?;
            let tile_base = em.tile_matrix_index(accept, local_row, local_col, stride);
            let mut values = Vec::with_capacity(VEC as usize);
            for i in 0..VEC {
                let index = em.add_literal_u32(accept, storage_base, i);
                let ptr = em.storage_dynamic_pointer(accept, &view, index)?;
                let loaded = em.emit_load(accept, ptr);
                values.push(em.cast_tile_value(
                    accept,
                    loaded,
                    view.buffer.element,
                    dst.element,
                )?);
            }
            for (i, value) in values.into_iter().enumerate() {
                let step = if cols_fastest { 1 } else { stride };
                let tile_index = em.add_literal_u32(accept, tile_base, i as u32 * step);
                let ptr = em.tile_dynamic_pointer(accept, &dst, tile_index)?;
                em.store_tile_value(accept, &dst, ptr, value)?;
            }
            Ok(())
        })
    }

    fn lane_coords(
        &mut self,
        body: &mut Block,
        flat: Handle<Expression>,
        rows: u32,
        cols: u32,
        cols_fastest: bool,
    ) -> (Handle<Expression>, Handle<Expression>) {
        if cols_fastest {
            let r = self.div_literal_u32(body, flat, cols.max(1));
            let c = self.mod_literal_u32(body, flat, cols.max(1));
            (r, c)
        } else {
            let r = self.mod_literal_u32(body, flat, rows.max(1));
            let c = self.div_literal_u32(body, flat, rows.max(1));
            (r, c)
        }
    }

    pub(crate) fn tile_matrix_index(
        &mut self,
        body: &mut Block,
        row: Handle<Expression>,
        col: Handle<Expression>,
        stride: u32,
    ) -> Handle<Expression> {
        let offset = self.mul_literal_u32(body, row, stride);
        self.add_u32(body, offset, col)
    }

    fn bounds_check(
        &mut self,
        body: &mut Block,
        row: Handle<Expression>,
        col: Handle<Expression>,
        row_limit: Option<Handle<Expression>>,
        col_limit: Option<Handle<Expression>>,
    ) -> Option<Handle<Expression>> {
        let mut acc: Option<Handle<Expression>> = None;
        for (coord, limit) in [(row, row_limit), (col, col_limit)] {
            let Some(limit) = limit else { continue };
            let check = self.bin(body, BinaryOperator::Less, coord, limit);
            acc = Some(match acc {
                None => check,
                Some(prev) => self.bin(body, BinaryOperator::LogicalAnd, prev, check),
            });
        }
        acc
    }

    /// Store `load(..)` into the tile, or zero when out of bounds. Edge tiles
    /// stay garbage-free so a partial M/N/K tail is safe for an MMA.
    fn guarded_tile_store(
        &mut self,
        out: &mut Block,
        dst: &Tile,
        tile_ptr: Handle<Expression>,
        in_bounds: Option<Handle<Expression>>,
        load: impl FnOnce(&mut Self, &mut Block) -> Result<Handle<Expression>, EmitError>,
    ) -> Result<(), EmitError> {
        match in_bounds {
            None => {
                let value = load(self, out)?;
                self.store_tile_value(out, dst, tile_ptr, value)
            }
            Some(condition) => {
                let dst_ok = dst.clone();
                let (accept, ()) = self.nested(move |em, block| {
                    let value = load(em, block)?;
                    em.store_tile_value(block, &dst_ok, tile_ptr, value)
                })?;
                let zero = self.zero_literal(dst.element)?;
                let dst_zero = dst.clone();
                let (reject, ()) = self.nested(move |em, block| {
                    em.store_tile_value(block, &dst_zero, tile_ptr, zero)
                })?;
                out.push(
                    Statement::If {
                        condition,
                        accept,
                        reject,
                    },
                    Span::default(),
                );
                Ok(())
            }
        }
    }
}

/// True when the view is affine with a unit stride along `axis`.
fn unit_stride(view: &StorageView, axis: usize) -> bool {
    let indexing = &view.layout.indexing;
    indexing.is_affine()
        && indexing
            .groups
            .get(axis)
            .and_then(|g| g.sub_axes.first())
            .map(|s| s.stride)
            == Some(1)
}
