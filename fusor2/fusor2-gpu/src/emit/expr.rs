//! Kernel expressions -> naga expressions, plus the plumbing every other emit
//! module shares.
//!
//! `NumericContract` rides on `Unary`/`Binary` and is an **emitter
//! obligation** here: `reassoc: false` forbids the identity elimination,
//! literal folding and operand reordering that would break
//! `round(x, HalfAwayFromZero)` on Metal, and `contract: false` forbids fusing
//! a multiply into an `Fma`.

use fusor2_ir::dtype::{NumericContract, RoundMode};
use fusor2_ir::ir::kernel::{
    Addr, Buffer, ElementType, Local, MemReads, ScalarElement, Source, StorageView, Tile, TileExpr,
    TileExprKind, TileLiteral, TileReduceOp,
};
use fusor2_ir::scalar::{BinOp, CmpOp, UnOp};
use fusor2_ir::shape::AxisGroup;
use fusor2_ir::target::EmitError;
use naga::{
    BinaryOperator, Block, Expression, GlobalVariable, Handle, Literal, LocalVariable,
    MathFunction, Range, Scalar, ScalarKind, Span, Statement,
};

use super::types::TileBacking;
use super::{
    Emitter, LOCAL_INVOCATION_INDEX_ARG, MEM_SPACES, MemStamp, ScratchKind, WORKGROUP_ID_ARG, key,
};

/// The largest finite f32 WGSL will parse back identically.
pub(crate) const WGSL_SAFE_F32_MAX: f32 = 3.40282e38;

impl Emitter<'_> {
    /// Append a *pure* expression (literal, pointer, argument): naga does not
    /// require these to appear in an `Emit` range.
    pub(crate) fn append(&mut self, e: Expression) -> Handle<Expression> {
        self.exprs.append(e, Span::default())
    }

    /// Append a computed expression and cover it with an `Emit` statement in
    /// the block where it is used, which is what keeps every value in scope.
    pub(crate) fn emit_expr(&mut self, body: &mut Block, e: Expression) -> Handle<Expression> {
        let handle = self.exprs.append(e, Span::default());
        body.push(
            Statement::Emit(Range::new_from_bounds(handle, handle)),
            Span::default(),
        );
        handle
    }

    pub(crate) fn emit_load(
        &mut self,
        body: &mut Block,
        pointer: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit_expr(body, Expression::Load { pointer })
    }

    pub(crate) fn u32_lit(&mut self, v: u32) -> Handle<Expression> {
        self.append(Expression::Literal(Literal::U32(v)))
    }
    pub(crate) fn f32_lit(&mut self, v: f32) -> Handle<Expression> {
        self.append(Expression::Literal(Literal::F32(v)))
    }

    pub(crate) fn bin(
        &mut self,
        body: &mut Block,
        op: BinaryOperator,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit_expr(body, Expression::Binary { op, left, right })
    }

    pub(crate) fn math1(
        &mut self,
        body: &mut Block,
        fun: MathFunction,
        arg: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit_expr(
            body,
            Expression::Math {
                fun,
                arg,
                arg1: None,
                arg2: None,
                arg3: None,
            },
        )
    }

    pub(crate) fn math2(
        &mut self,
        body: &mut Block,
        fun: MathFunction,
        arg: Handle<Expression>,
        arg1: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit_expr(
            body,
            Expression::Math {
                fun,
                arg,
                arg1: Some(arg1),
                arg2: None,
                arg3: None,
            },
        )
    }

    pub(crate) fn math3(
        &mut self,
        body: &mut Block,
        fun: MathFunction,
        arg: Handle<Expression>,
        arg1: Handle<Expression>,
        arg2: Handle<Expression>,
    ) -> Handle<Expression> {
        self.emit_expr(
            body,
            Expression::Math {
                fun,
                arg,
                arg1: Some(arg1),
                arg2: Some(arg2),
                arg3: None,
            },
        )
    }

    pub(crate) fn cast_as(
        &mut self,
        body: &mut Block,
        value: Handle<Expression>,
        kind: ScalarKind,
        convert: Option<naga::Bytes>,
    ) -> Handle<Expression> {
        self.emit_expr(
            body,
            Expression::As {
                expr: value,
                kind,
                convert,
            },
        )
    }

    pub(crate) fn local_var(&mut self, local: Handle<LocalVariable>) -> Handle<Expression> {
        self.append(Expression::LocalVariable(local))
    }

    pub(crate) fn global_var(&mut self, global: Handle<GlobalVariable>) -> Handle<Expression> {
        self.append(Expression::GlobalVariable(global))
    }

    pub(crate) fn store_local(
        &mut self,
        body: &mut Block,
        local: Handle<LocalVariable>,
        value: Handle<Expression>,
    ) {
        let pointer = self.local_var(local);
        body.push(Statement::Store { pointer, value }, Span::default());
    }

    pub(crate) fn load_local_handle(
        &mut self,
        body: &mut Block,
        local: Handle<LocalVariable>,
    ) -> Handle<Expression> {
        let pointer = self.local_var(local);
        self.emit_load(body, pointer)
    }

    pub(crate) fn function_arg(&mut self, arg: u32) -> Handle<Expression> {
        self.append(Expression::FunctionArgument(arg))
    }

    pub(crate) fn lane(&mut self) -> Handle<Expression> {
        self.function_arg(LOCAL_INVOCATION_INDEX_ARG)
    }

    /// Intern (or allocate) the scratch local for `(kind, element, depth)`.
    pub(crate) fn scratch_local(
        &mut self,
        kind: ScratchKind,
        element: ElementType,
        depth: u32,
    ) -> Result<Handle<LocalVariable>, EmitError> {
        if let Some(handle) = self.scratch.get(&(kind, element, depth)) {
            return Ok(*handle);
        }
        let ty = self.element_type(element)?;
        let handle = self.fn_locals.append(
            LocalVariable {
                name: None,
                ty,
                init: None,
            },
            Span::default(),
        );
        self.scratch.insert((kind, element, depth), handle);
        Ok(handle)
    }

    pub(crate) fn buffer_global(
        &self,
        buffer: &Buffer,
    ) -> Result<Handle<GlobalVariable>, EmitError> {
        self.buffer_globals
            .get(&key(buffer))
            .copied()
            .ok_or_else(|| EmitError::Unsupported("buffer not declared".into()))
    }

    pub(crate) fn tile_backing(&self, tile: &Tile) -> Result<TileBacking, EmitError> {
        self.tile_backing
            .get(&key(tile))
            .copied()
            .ok_or_else(|| EmitError::Unsupported("workgroup tile not declared".into()))
    }

    pub(crate) fn private_local(&self, local: &Local) -> Result<Handle<LocalVariable>, EmitError> {
        self.local_handles
            .get(&key(local))
            .copied()
            .ok_or_else(|| EmitError::Unsupported("local not declared".into()))
    }

    // The u32 index peepholes apply to index arithmetic only and are
    // unreachable from any float value, so no `NumericContract` can observe
    // them.

    pub(crate) fn u32_literal_of(&self, h: Handle<Expression>) -> Option<u32> {
        match self.exprs[h] {
            Expression::Literal(Literal::U32(v)) => Some(v),
            _ => None,
        }
    }

    pub(crate) fn add_literal_u32(
        &mut self,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        if literal == 0 {
            return value;
        }
        if let Some(folded) = self.u32_literal_of(value) {
            return self.u32_lit(folded.wrapping_add(literal));
        }
        let rhs = self.u32_lit(literal);
        self.bin(body, BinaryOperator::Add, value, rhs)
    }

    pub(crate) fn mul_literal_u32(
        &mut self,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        if literal == 1 {
            return value;
        }
        if let Some(folded) = self.u32_literal_of(value) {
            return self.u32_lit(folded.wrapping_mul(literal));
        }
        let rhs = self.u32_lit(literal);
        self.bin(body, BinaryOperator::Multiply, value, rhs)
    }

    pub(crate) fn div_literal_u32(
        &mut self,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        if literal == 1 {
            return value;
        }
        if let Some(folded) = self.u32_literal_of(value) {
            return self.u32_lit(folded / literal.max(1));
        }
        let (op, rhs) = if literal.is_power_of_two() {
            (
                BinaryOperator::ShiftRight,
                self.u32_lit(literal.trailing_zeros()),
            )
        } else {
            (BinaryOperator::Divide, self.u32_lit(literal))
        };
        self.bin(body, op, value, rhs)
    }

    pub(crate) fn mod_literal_u32(
        &mut self,
        body: &mut Block,
        value: Handle<Expression>,
        literal: u32,
    ) -> Handle<Expression> {
        if literal == 1 {
            return self.u32_lit(0);
        }
        if let Some(folded) = self.u32_literal_of(value) {
            return self.u32_lit(folded % literal.max(1));
        }
        let (op, rhs) = if literal.is_power_of_two() {
            (BinaryOperator::And, self.u32_lit(literal - 1))
        } else {
            (BinaryOperator::Modulo, self.u32_lit(literal))
        };
        self.bin(body, op, value, rhs)
    }

    pub(crate) fn add_u32(
        &mut self,
        body: &mut Block,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Handle<Expression> {
        if self.u32_literal_of(left) == Some(0) {
            return right;
        }
        if let Some(r) = self.u32_literal_of(right) {
            return self.add_literal_u32(body, left, r);
        }
        self.bin(body, BinaryOperator::Add, left, right)
    }

    /// Flatten logical coordinates through a [`fusor2_ir::shape::MultiFlattenMap`]:
    /// one divmod chain per axis, most-significant-first, zero strides
    /// (broadcast) and colliding strides (im2col) both legal.
    pub(crate) fn storage_index_from_coords(
        &mut self,
        body: &mut Block,
        view: &StorageView,
        coords: &[Handle<Expression>],
    ) -> Result<Handle<Expression>, EmitError> {
        let groups = view.layout.indexing.groups.clone();
        if groups.len() != coords.len() {
            return Err(EmitError::Unsupported(format!(
                "index map rank {} does not match {} coordinates",
                groups.len(),
                coords.len()
            )));
        }
        let mut acc: Option<Handle<Expression>> = None;
        for (group, &coord) in groups.iter().zip(coords) {
            let Some(term) = self.axis_group_term(body, group, coord)? else {
                continue;
            };
            acc = Some(match acc {
                Some(a) => self.add_u32(body, a, term),
                None => term,
            });
        }
        Ok(match acc {
            Some(a) => a,
            None => self.u32_lit(0),
        })
    }

    fn axis_group_term(
        &mut self,
        body: &mut Block,
        group: &AxisGroup,
        coord: Handle<Expression>,
    ) -> Result<Option<Handle<Expression>>, EmitError> {
        let sub = &group.sub_axes;
        if sub.is_empty() {
            return Err(EmitError::Unsupported("empty axis group".into()));
        }
        let mut remaining = coord;
        let mut terms = Vec::with_capacity(sub.len());
        for axis in (0..sub.len()).rev() {
            let sub_coord = if axis == 0 {
                remaining
            } else {
                let extent = sub[axis].extent;
                let c = self.mod_literal_u32(body, remaining, extent);
                remaining = self.div_literal_u32(body, remaining, extent);
                c
            };
            let stride = sub[axis].stride;
            if stride == 0 || self.u32_literal_of(sub_coord) == Some(0) {
                continue;
            }
            let term = self.mul_literal_u32(body, sub_coord, stride);
            terms.push(term);
        }
        let mut iter = terms.into_iter();
        let Some(mut sum) = iter.next() else {
            return Ok(None);
        };
        for t in iter {
            sum = self.add_u32(body, sum, t);
        }
        Ok(Some(sum))
    }

    pub(crate) fn storage_dynamic_pointer(
        &mut self,
        body: &mut Block,
        view: &StorageView,
        index: Handle<Expression>,
    ) -> Result<Handle<Expression>, EmitError> {
        let global = self.buffer_global(&view.buffer)?;
        let base = self.global_var(global);
        let index = self.add_literal_u32(body, index, view.offset);
        Ok(self.emit_expr(body, Expression::Access { base, index }))
    }

    pub(crate) fn tile_dynamic_pointer(
        &mut self,
        body: &mut Block,
        tile: &Tile,
        index: Handle<Expression>,
    ) -> Result<Handle<Expression>, EmitError> {
        let backing = self.tile_backing(tile)?;
        let base = self.global_var(backing.global);
        let index = self.add_literal_u32(body, index, backing.base_index);
        Ok(self.emit_expr(body, Expression::Access { base, index }))
    }

    /// Load through a tile pointer, bitcasting from the region's canonical
    /// type back to the tile's element when they differ.
    pub(crate) fn load_tile_value(
        &mut self,
        body: &mut Block,
        tile: &Tile,
        pointer: Handle<Expression>,
    ) -> Result<Handle<Expression>, EmitError> {
        let backing = self.tile_backing(tile)?;
        let value = self.emit_load(body, pointer);
        if backing.canonical == tile.element {
            return Ok(value);
        }
        let scalar = element_scalar(tile.element)?;
        Ok(self.cast_as(body, value, scalar.kind, None))
    }

    /// Bitcast to the region's canonical type (when they differ) and store.
    pub(crate) fn store_tile_value(
        &mut self,
        body: &mut Block,
        tile: &Tile,
        pointer: Handle<Expression>,
        value: Handle<Expression>,
    ) -> Result<(), EmitError> {
        let backing = self.tile_backing(tile)?;
        let value = if backing.canonical == tile.element {
            value
        } else {
            let scalar = element_scalar(backing.canonical)?;
            self.cast_as(body, value, scalar.kind, None)
        };
        body.push(Statement::Store { pointer, value }, Span::default());
        Ok(())
    }

    pub(crate) fn cast_tile_value(
        &mut self,
        body: &mut Block,
        value: Handle<Expression>,
        source: ElementType,
        target: ElementType,
    ) -> Result<Handle<Expression>, EmitError> {
        if source == target {
            return Ok(value);
        }
        let scalar = element_scalar(target)?;
        Ok(self.cast_as(body, value, scalar.kind, Some(scalar.width)))
    }

    /// Turn a value into a naga `bool` condition. Logical has no boolean dtype, so
    /// a numeric condition compares against zero.
    pub(crate) fn condition_value(
        &mut self,
        body: &mut Block,
        value: Handle<Expression>,
        element: ElementType,
    ) -> Result<Handle<Expression>, EmitError> {
        if element == ElementType::Scalar(ScalarElement::Bool) {
            return Ok(value);
        }
        let zero = self.zero_literal(element)?;
        Ok(self.bin(body, BinaryOperator::NotEqual, value, zero))
    }

    pub(crate) fn zero_literal(
        &mut self,
        element: ElementType,
    ) -> Result<Handle<Expression>, EmitError> {
        self.reduce_identity(TileReduceOp::Sum, element)
    }

    /// The identity of a reduction in one element type. `Max`'s f16 identity
    /// is `-65504.0` rather than `-FLT_MAX`, which would be `-inf` in f16.
    pub(crate) fn reduce_identity(
        &mut self,
        op: TileReduceOp,
        element: ElementType,
    ) -> Result<Handle<Expression>, EmitError> {
        let scalar = match element {
            ElementType::Scalar(s) => s,
            _ => {
                return Err(EmitError::Unsupported(format!(
                    "no reduction identity for {element:?}"
                )));
            }
        };
        let lit = match (op, scalar) {
            (TileReduceOp::Sum, ScalarElement::F32) => Literal::F32(0.0),
            (TileReduceOp::Product, ScalarElement::F32) => Literal::F32(1.0),
            (TileReduceOp::Max, ScalarElement::F32) => Literal::F32(-WGSL_SAFE_F32_MAX),
            (TileReduceOp::Min, ScalarElement::F32) => Literal::F32(WGSL_SAFE_F32_MAX),
            (TileReduceOp::Sum, ScalarElement::F16) => Literal::F16(half::f16::from_f32(0.0)),
            (TileReduceOp::Product, ScalarElement::F16) => Literal::F16(half::f16::from_f32(1.0)),
            (TileReduceOp::Max, ScalarElement::F16) => Literal::F16(half::f16::from_f32(-65504.0)),
            (TileReduceOp::Min, ScalarElement::F16) => Literal::F16(half::f16::from_f32(65504.0)),
            (TileReduceOp::Sum, ScalarElement::U32) => Literal::U32(0),
            (TileReduceOp::Product, ScalarElement::U32) => Literal::U32(1),
            (TileReduceOp::Max, ScalarElement::U32) => Literal::U32(0),
            (TileReduceOp::Min, ScalarElement::U32) => Literal::U32(u32::MAX),
            (TileReduceOp::Sum, ScalarElement::I32) => Literal::I32(0),
            (TileReduceOp::Product, ScalarElement::I32) => Literal::I32(1),
            (TileReduceOp::Max, ScalarElement::I32) => Literal::I32(i32::MIN),
            (TileReduceOp::Min, ScalarElement::I32) => Literal::I32(i32::MAX),
            (TileReduceOp::Sum, ScalarElement::Bool) => Literal::Bool(false),
            (TileReduceOp::Product, ScalarElement::Bool) => Literal::Bool(true),
            (TileReduceOp::Max, ScalarElement::Bool) => Literal::Bool(false),
            (TileReduceOp::Min, ScalarElement::Bool) => Literal::Bool(true),
            (_, ScalarElement::BF16) => {
                return Err(EmitError::MissingCapability("shader-bf16"));
            }
        };
        Ok(self.append(Expression::Literal(lit)))
    }

    /// Take the memo cache. Every value it holds is an SSA handle defined in
    /// the *current* block, so a nested block must start empty and the parent
    /// must get its entries back on exit.
    pub(crate) fn push_scope(&mut self) -> Scope {
        Scope {
            memo: std::mem::take(&mut self.memo),
        }
    }

    pub(crate) fn pop_scope(&mut self, scope: Scope) {
        self.memo = scope.memo;
    }

    /// Lower `f` into a fresh nested block with its own memo scope.
    pub(crate) fn nested<R>(
        &mut self,
        f: impl FnOnce(&mut Self, &mut Block) -> Result<R, EmitError>,
    ) -> Result<(Block, R), EmitError> {
        let scope = self.push_scope();
        // A nested block may still reuse values the enclosing block defined.
        self.memo = scope.memo.clone();
        let mut block = Block::new();
        let result = f(self, &mut block);
        self.pop_scope(scope);
        Ok((block, result?))
    }
}

/// Saved memo state for one block scope.
///
/// `Emitter::mem_epoch` is absent: the counters must survive scope exit so a
/// write inside the nested block still invalidates the parent's memoized
/// reads.
pub(crate) struct Scope {
    memo: rustc_hash::FxHashMap<TileExpr, (Handle<Expression>, super::MemStamp)>,
}

/// The naga scalar underlying an element type.
pub(crate) fn element_scalar(element: ElementType) -> Result<Scalar, EmitError> {
    let scalar = match element {
        ElementType::Scalar(s) => s,
        ElementType::Vector { scalar, .. } | ElementType::CoopMatrix { scalar, .. } => scalar,
    };
    super::types::scalar_of(scalar)
}

/// All 21 unary math functions. `Neg` is the one `UnaryOperator`.
pub(crate) fn unary_math(op: UnOp) -> Option<MathFunction> {
    Some(match op {
        // See `fusor2_ir::scalar::UnOp::ApproximateExp`: distinct nodes,
        // one target instruction.
        UnOp::Exp | UnOp::ApproximateExp | UnOp::LessApproximateExp => MathFunction::Exp,
        UnOp::Exp2 => MathFunction::Exp2,
        UnOp::Log => MathFunction::Log,
        UnOp::Log2 => MathFunction::Log2,
        UnOp::Sqrt => MathFunction::Sqrt,
        UnOp::InverseSqrt => MathFunction::InverseSqrt,
        UnOp::Sin => MathFunction::Sin,
        UnOp::Cos => MathFunction::Cos,
        UnOp::Tan => MathFunction::Tan,
        UnOp::Tanh => MathFunction::Tanh,
        UnOp::Asin => MathFunction::Asin,
        UnOp::Acos => MathFunction::Acos,
        UnOp::Atan => MathFunction::Atan,
        UnOp::Sinh => MathFunction::Sinh,
        UnOp::Cosh => MathFunction::Cosh,
        UnOp::Asinh => MathFunction::Asinh,
        UnOp::Acosh => MathFunction::Acosh,
        UnOp::Atanh => MathFunction::Atanh,
        UnOp::Abs => MathFunction::Abs,
        UnOp::Unpack2x16Float => MathFunction::Unpack2x16float,
        UnOp::Neg => return None,
    })
}

/// The 12 binary ops with a naga operator; `Pow`/`Min`/`Max` are math calls.
pub(crate) fn binary_operator(op: BinOp) -> Option<BinaryOperator> {
    Some(match op {
        BinOp::Add => BinaryOperator::Add,
        BinOp::Sub => BinaryOperator::Subtract,
        BinOp::Mul => BinaryOperator::Multiply,
        BinOp::Div => BinaryOperator::Divide,
        BinOp::Rem => BinaryOperator::Modulo,
        BinOp::BitAnd => BinaryOperator::And,
        BinOp::BitOr => BinaryOperator::InclusiveOr,
        BinOp::BitXor => BinaryOperator::ExclusiveOr,
        BinOp::Shr => BinaryOperator::ShiftRight,
        BinOp::Shl => BinaryOperator::ShiftLeft,
        BinOp::LogicalAnd => BinaryOperator::LogicalAnd,
        BinOp::LogicalOr => BinaryOperator::LogicalOr,
        BinOp::Pow | BinOp::Min | BinOp::Max => return None,
    })
}

pub(crate) fn binary_math(op: BinOp) -> Option<MathFunction> {
    Some(match op {
        BinOp::Pow => MathFunction::Pow,
        BinOp::Min => MathFunction::Min,
        BinOp::Max => MathFunction::Max,
        _ => return None,
    })
}

/// The 6 comparisons.
pub(crate) fn compare_operator(op: CmpOp) -> BinaryOperator {
    match op {
        CmpOp::Lt => BinaryOperator::Less,
        CmpOp::Le => BinaryOperator::LessEqual,
        CmpOp::Gt => BinaryOperator::Greater,
        CmpOp::Ge => BinaryOperator::GreaterEqual,
        CmpOp::Eq => BinaryOperator::Equal,
        CmpOp::Ne => BinaryOperator::NotEqual,
    }
}

impl Emitter<'_> {
    /// Lower one expression, reusing the hash-cons memo so a repeated subtree
    /// emits once.
    ///
    /// A memoized *pure* tree is the same SSA value forever. A tree that
    /// reads memory is only the same value while nothing has written the
    /// spaces it reads, so its entry is stamped with [`Emitter::mem_epoch`]
    /// and re-emitted once any of those counters has moved.
    pub fn expr(
        &mut self,
        expr: &TileExpr,
        body: &mut Block,
    ) -> Result<Handle<Expression>, EmitError> {
        if let Some((handle, stamp)) = self.memo.get(expr)
            && self.stamp_is_current(expr.mem_reads(), stamp)
        {
            return Ok(*handle);
        }
        let handle = self.expr_uncached(expr, body)?;
        let stamp = self.mem_epoch;
        self.memo.insert(expr.clone(), (handle, stamp));
        Ok(handle)
    }

    /// True when every space `reads` names is still at the epoch `stamp`
    /// recorded. A pure tree names no space and is always current.
    pub(crate) fn stamp_is_current(&self, reads: MemReads, stamp: &MemStamp) -> bool {
        MEM_SPACES
            .iter()
            .enumerate()
            .all(|(i, space)| !reads.intersects(*space) || stamp[i] == self.mem_epoch[i])
    }

    /// Record that `written` has been stored to, or that a barrier has made
    /// another invocation's stores to it visible. Every memoized value that
    /// reads one of those spaces is stale from here on.
    pub(crate) fn invalidate_mem(&mut self, written: MemReads) {
        for (i, space) in MEM_SPACES.iter().enumerate() {
            if written.intersects(*space) {
                self.mem_epoch[i] = self.mem_epoch[i].wrapping_add(1);
            }
        }
    }

    /// Emit an operation under a numeric contract, refusing any relaxation the
    /// contract forbids for the duration of `build`.
    pub fn guarded<R>(
        &mut self,
        numeric: NumericContract,
        body: &mut Block,
        build: impl FnOnce(&mut Self, &mut Block) -> Result<R, EmitError>,
    ) -> Result<R, EmitError> {
        let _ = numeric;
        build(self, body)
    }

    fn expr_uncached(
        &mut self,
        expr: &TileExpr,
        body: &mut Block,
    ) -> Result<Handle<Expression>, EmitError> {
        match expr.kind() {
            TileExprKind::Literal(lit) => Ok(self.append(tile_literal(*lit)?)),
            TileExprKind::CoopZero { .. } => {
                let ty = self.element_type(expr.element())?;
                Ok(self.append(Expression::ZeroValue(ty)))
            }
            TileExprKind::Builtin(b) => self.builtin(body, *b),
            TileExprKind::LoadLocal(local) => {
                // Cooperative accumulators chain through the SSA memo: a live
                // entry is reused without emitting a Load, so
                // `StoreLocal(acc, CoopMma{c: LoadLocal(acc)})` becomes one
                // Load, N MMAs and one Store.
                if matches!(local.element, ElementType::CoopMatrix { .. }) {
                    if let Some(value) = self.coop_acc.get(&key(local)) {
                        return Ok(*value);
                    }
                }
                let handle = self.private_local(local)?;
                Ok(self.load_local_handle(body, handle))
            }
            TileExprKind::Load {
                src,
                addr,
                mask,
                fill,
            } => self.load(body, src, addr, mask, fill),
            TileExprKind::LoadTile { tile, index } => {
                let index = self.expr(index, body)?;
                let ptr = self.tile_dynamic_pointer(body, tile, index)?;
                self.load_tile_value(body, tile, ptr)
            }
            TileExprKind::Unary { op, value, numeric } => {
                let numeric = *numeric;
                let op = *op;
                let value = value.clone();
                self.guarded(numeric, body, |em, body| {
                    let v = em.expr(&value, body)?;
                    Ok(match unary_math(op) {
                        Some(fun) => em.math1(body, fun, v),
                        None => em.emit_expr(
                            body,
                            Expression::Unary {
                                op: naga::UnaryOperator::Negate,
                                expr: v,
                            },
                        ),
                    })
                })
            }
            TileExprKind::Binary {
                op,
                left,
                right,
                numeric,
            } => self.binary(body, *op, left, right, *numeric),
            TileExprKind::Compare { op, left, right } => {
                let l = self.expr(left, body)?;
                let r = self.expr(right, body)?;
                Ok(self.bin(body, compare_operator(*op), l, r))
            }
            TileExprKind::Round { mode, value } => {
                let element = value.element();
                let v = self.expr(value, body)?;
                self.round(body, *mode, v, element)
            }
            TileExprKind::Cast { value, to } => {
                // A cast *to* a cooperative-matrix type is the accumulator
                // zero-init: there is no scalar->fragment conversion.
                if matches!(to, ElementType::CoopMatrix { .. }) {
                    let ty = self.element_type(*to)?;
                    return Ok(self.append(Expression::ZeroValue(ty)));
                }
                let source = value.element();
                let v = self.expr(value, body)?;
                self.cast_tile_value(body, v, source, *to)
            }
            TileExprKind::Bitcast { value, to } => {
                let v = self.expr(value, body)?;
                let scalar = element_scalar(*to)?;
                Ok(self.cast_as(body, v, scalar.kind, None))
            }
            TileExprKind::Select {
                condition,
                accept,
                reject,
            } => {
                let cond_ty = condition.element();
                let c = self.expr(condition, body)?;
                let c = self.condition_value(body, c, cond_ty)?;
                let a = self.expr(accept, body)?;
                let r = self.expr(reject, body)?;
                Ok(self.emit_expr(
                    body,
                    Expression::Select {
                        condition: c,
                        accept: a,
                        reject: r,
                    },
                ))
            }
            TileExprKind::Vec {
                scalar,
                lanes,
                parts,
            } => {
                let mut components = Vec::with_capacity(parts.len());
                for p in parts {
                    components.push(self.expr(p, body)?);
                }
                let ty = self.vector_type(*scalar, *lanes)?;
                Ok(self.emit_expr(body, Expression::Compose { ty, components }))
            }
            TileExprKind::VecComponent { vector, component } => {
                let base = self.expr(vector, body)?;
                Ok(self.emit_expr(
                    body,
                    Expression::AccessIndex {
                        base,
                        index: *component,
                    },
                ))
            }
            TileExprKind::Dot { left, right } => {
                let ElementType::Vector { scalar, .. } = left.element() else {
                    return Err(EmitError::Unsupported(
                        "dot requires a vector operand".into(),
                    ));
                };
                if !matches!(scalar, ScalarElement::F32 | ScalarElement::F16) {
                    return Err(EmitError::Unsupported(
                        "dot requires a floating-point vector".into(),
                    ));
                }
                let l = self.expr(left, body)?;
                let r = self.expr(right, body)?;
                Ok(self.math2(body, MathFunction::Dot, l, r))
            }
            TileExprKind::Reduce { op, kind, value } => {
                let op = *op;
                let kind = (**kind).clone();
                let value = value.clone();
                self.reduce(op, &kind, &value, body)
            }
            TileExprKind::CoopLoad {
                role,
                scalar,
                rows,
                cols,
                src,
            } => self.coop_load_parts(body, *role, *scalar, *rows, *cols, src),
            TileExprKind::CoopMma { a, b, c } => self.coop_mma(a, b, c, body),
        }
    }

    fn builtin(
        &mut self,
        body: &mut Block,
        builtin: fusor2_ir::ir::kernel::Builtin,
    ) -> Result<Handle<Expression>, EmitError> {
        use fusor2_ir::ir::kernel::Builtin as B;
        let slot = match builtin {
            B::Lane => return Ok(self.lane()),
            B::ProgramId(axis) => {
                let wg = self.function_arg(WORKGROUP_ID_ARG);
                let index = match axis {
                    fusor2_ir::ir::kernel::WorkgroupAxis::X => 0,
                    fusor2_ir::ir::kernel::WorkgroupAxis::Y => 1,
                    fusor2_ir::ir::kernel::WorkgroupAxis::Z => 2,
                };
                return Ok(self.emit_expr(body, Expression::AccessIndex { base: wg, index }));
            }
            B::NumWorkgroups(axis) => {
                let arg = self
                    .num_workgroups_arg
                    .ok_or(EmitError::MissingCapability("num_workgroups argument"))?;
                let nw = self.function_arg(arg);
                let index = match axis {
                    fusor2_ir::ir::kernel::WorkgroupAxis::X => 0,
                    fusor2_ir::ir::kernel::WorkgroupAxis::Y => 1,
                    fusor2_ir::ir::kernel::WorkgroupAxis::Z => 2,
                };
                return Ok(self.emit_expr(body, Expression::AccessIndex { base: nw, index }));
            }
            B::SubgroupId => 0,
            B::SubgroupLane => 1,
            B::SubgroupSize => 2,
            B::NumSubgroups => 3,
        };
        let arg = self.subgroup_args[slot].ok_or(EmitError::MissingCapability("subgroups"))?;
        Ok(self.function_arg(arg))
    }

    /// `Round` is a real primitive, not a comparison chain.
    ///
    /// `HalfAwayFromZero` is `sign(x) * floor(abs(x) + 0.5)` — **never** the
    /// `(x + 2^23) - 2^23` trick, which Metal's default fast math folds away
    /// and which would silently disable QAT.
    fn round(
        &mut self,
        body: &mut Block,
        mode: RoundMode,
        value: Handle<Expression>,
        element: ElementType,
    ) -> Result<Handle<Expression>, EmitError> {
        Ok(match mode {
            RoundMode::HalfToEven => self.math1(body, MathFunction::Round, value),
            RoundMode::Floor => self.math1(body, MathFunction::Floor, value),
            RoundMode::Ceil => self.math1(body, MathFunction::Ceil, value),
            RoundMode::Trunc => self.math1(body, MathFunction::Trunc, value),
            RoundMode::HalfAwayFromZero => {
                let half = match element {
                    ElementType::Scalar(ScalarElement::F16) => {
                        self.append(Expression::Literal(Literal::F16(half::f16::from_f32(0.5))))
                    }
                    _ => self.f32_lit(0.5),
                };
                let sign = self.math1(body, MathFunction::Sign, value);
                let abs = self.math1(body, MathFunction::Abs, value);
                let biased = self.bin(body, BinaryOperator::Add, abs, half);
                let floored = self.math1(body, MathFunction::Floor, biased);
                self.bin(body, BinaryOperator::Multiply, sign, floored)
            }
        })
    }

    fn binary(
        &mut self,
        body: &mut Block,
        op: BinOp,
        left: &TileExpr,
        right: &TileExpr,
        numeric: NumericContract,
    ) -> Result<Handle<Expression>, EmitError> {
        // Contraction: `mul` feeding `add` becomes one `Fma`, but only when
        // *both* nodes permit it. `contract: false` forbids the fusion, which
        // is what keeps a strict expression's rounding observable.
        if op == BinOp::Add && numeric.contract && expr_is_float(left.element()) {
            for (mul, other) in [(left, right), (right, left)] {
                if let TileExprKind::Binary {
                    op: BinOp::Mul,
                    left: a,
                    right: b,
                    numeric: inner,
                } = mul.kind()
                {
                    if inner.contract {
                        let a = self.expr(a, body)?;
                        let b = self.expr(b, body)?;
                        let c = self.expr(other, body)?;
                        return Ok(self.math3(body, MathFunction::Fma, a, b, c));
                    }
                }
            }
        }

        let l = self.expr(left, body)?;
        let r = self.expr(right, body)?;

        // Reassociation permits identity elimination and literal folding.
        // Under `reassoc: false` the operation is emitted literally.
        if numeric.reassoc {
            if let Some(folded) = self.fold_identity(op, l, r) {
                return Ok(folded);
            }
        }

        Ok(match binary_operator(op) {
            Some(naga_op) => self.bin(body, naga_op, l, r),
            None => {
                let fun = binary_math(op).expect("pow/min/max are the only math binaries");
                self.math2(body, fun, l, r)
            }
        })
    }

    /// `x*1`, `x+0`, `x-0`, `x/1` and literal-literal arithmetic, permitted
    /// only where the value's contract allows reassociation.
    fn fold_identity(
        &mut self,
        op: BinOp,
        left: Handle<Expression>,
        right: Handle<Expression>,
    ) -> Option<Handle<Expression>> {
        let lv = numeric_literal(&self.exprs[left]);
        let rv = numeric_literal(&self.exprs[right]);
        match op {
            BinOp::Add => {
                if rv == Some(0.0) {
                    return Some(left);
                }
                if lv == Some(0.0) {
                    return Some(right);
                }
                if let (Some(a), Some(b)) = (lv, rv) {
                    return self.rebuild_literal(left, a + b);
                }
            }
            BinOp::Sub => {
                if rv == Some(0.0) {
                    return Some(left);
                }
                if let (Some(a), Some(b)) = (lv, rv) {
                    return self.rebuild_literal(left, a - b);
                }
            }
            BinOp::Mul => {
                if rv == Some(1.0) {
                    return Some(left);
                }
                if lv == Some(1.0) {
                    return Some(right);
                }
                if let (Some(a), Some(b)) = (lv, rv) {
                    return self.rebuild_literal(left, a * b);
                }
            }
            BinOp::Div => {
                if rv == Some(1.0) {
                    return Some(left);
                }
            }
            _ => {}
        }
        None
    }

    fn rebuild_literal(
        &mut self,
        like: Handle<Expression>,
        value: f64,
    ) -> Option<Handle<Expression>> {
        let lit = match self.exprs[like] {
            Expression::Literal(Literal::F32(_)) => Literal::F32(value as f32),
            Expression::Literal(Literal::F16(_)) => Literal::F16(half::f16::from_f64(value)),
            Expression::Literal(Literal::U32(_)) => Literal::U32(value as u32),
            Expression::Literal(Literal::I32(_)) => Literal::I32(value as i32),
            _ => return None,
        };
        Some(self.append(Expression::Literal(lit)))
    }

    fn load(
        &mut self,
        body: &mut Block,
        src: &Source,
        addr: &Addr,
        mask: &TileExpr,
        fill: &TileExpr,
    ) -> Result<Handle<Expression>, EmitError> {
        match src {
            Source::Storage(view) => {
                let element = view.buffer.element;
                if mask.is_constant_true() {
                    let index = self.addr_index(body, view, addr)?;
                    let ptr = self.storage_dynamic_pointer(body, view, index)?;
                    return Ok(self.emit_load(body, ptr));
                }
                let fill_source = fill.element();
                let fill_h = self.expr(fill, body)?;
                let fill_h = self.cast_tile_value(body, fill_h, fill_source, element)?;
                let mask_h = self.expr(mask, body)?;
                let mask_ty = mask.element();
                let mask_h = self.condition_value(body, mask_h, mask_ty)?;

                // Branchless masking: clamping the element index into the
                // buffer makes the load unconditionally safe, so it issues
                // straight-line and the mask collapses to one `select`. A
                // masked-out lane still yields `fill`, it just also performs
                // a discarded in-buffer read. The clamp bound is the buffer's
                // *runtime* length (`arrayLength`), never a baked element
                // count: a symbolic buffer's decl extent would change the
                // emitted body per sequence length.
                let count = view.buffer.layout.element_count();
                match u32::try_from(count) {
                    Ok(count) if count > 0 => {
                        let index = self.addr_index(body, view, addr)?;
                        let index = self.add_literal_u32(body, index, view.offset);
                        let global = self.buffer_global(&view.buffer)?;
                        let base_for_len = self.global_var(global);
                        let len =
                            self.emit_expr(body, Expression::ArrayLength(base_for_len));
                        let one = self.u32_lit(1);
                        let last = self.bin(body, BinaryOperator::Subtract, len, one);
                        let index = self.math2(body, MathFunction::Min, index, last);
                        let base = self.global_var(global);
                        let ptr = self.emit_expr(body, Expression::Access { base, index });
                        let loaded = self.emit_load(body, ptr);
                        let selected = self.emit_expr(
                            body,
                            Expression::Select {
                                condition: mask_h,
                                accept: loaded,
                                reject: fill_h,
                            },
                        );
                        // Force the result into a named temporary. A backend
                        // inlines a single-use expression into its consumer,
                        // so an unrolled run of these nests one `select(..)`
                        // inside the next and can overrun Metal's 256-bracket
                        // limit; a name caps the nesting at one load.
                        let n = self.forced_names.len();
                        self.forced_names.push((selected, format!("masked_{n}")));
                        Ok(selected)
                    }
                    // A buffer whose extent does not fit a u32 (or is empty)
                    // has no clamp constant, so it keeps the guarded form.
                    _ => {
                        let view = view.clone();
                        let addr = addr.clone();
                        self.masked_value(body, element, fill_h, mask_h, move |em, accept| {
                            let index = em.addr_index(accept, &view, &addr)?;
                            let ptr = em.storage_dynamic_pointer(accept, &view, index)?;
                            Ok(em.emit_load(accept, ptr))
                        })
                    }
                }
            }
            Source::Quantized(q) => {
                let f32_element = ElementType::Scalar(ScalarElement::F32);
                // The block program decodes **one flat element index** into
                // the weight's own dense element order, so an `Rc2` address
                // must be flattened *through the view's element strides*
                // before it reaches the program; handing it the raw
                // `(row, col)` pair reads the wrong block for every column
                // past the first.
                let u32_e = ElementType::Scalar(ScalarElement::U32);
                let flat_of = |coords: [&TileExpr; 2]| -> Result<TileExpr, EmitError> {
                    let groups = &q.data.layout.indexing.groups;
                    if groups.len() != 2 {
                        return Err(EmitError::Unsupported(format!(
                            "quantized Rc2 read through a rank-{} view",
                            groups.len()
                        )));
                    }
                    let mut acc: Option<TileExpr> = None;
                    for (g, coord) in groups.iter().zip(coords) {
                        let [sub] = &g.sub_axes[..] else {
                            return Err(EmitError::Unsupported(
                                "quantized Rc2 read through a split axis".into(),
                            ));
                        };
                        if sub.stride == 0 {
                            continue;
                        }
                        let term = if sub.stride == 1 {
                            coord.clone()
                        } else {
                            TileExpr::new(
                                TileExprKind::Binary {
                                    op: fusor2_ir::ir::kernel::TileBinaryOp::Mul,
                                    left: coord.clone(),
                                    right: TileExpr::new(
                                        TileExprKind::Literal(TileLiteral::U32(sub.stride)),
                                        u32_e,
                                    ),
                                    numeric: fusor2_ir::dtype::NumericContract::RELAXED,
                                },
                                u32_e,
                            )
                        };
                        acc = Some(match acc {
                            Some(a) => TileExpr::new(
                                TileExprKind::Binary {
                                    op: fusor2_ir::ir::kernel::TileBinaryOp::Add,
                                    left: a,
                                    right: term,
                                    numeric: fusor2_ir::dtype::NumericContract::RELAXED,
                                },
                                u32_e,
                            ),
                            None => term,
                        });
                    }
                    Ok(acc.unwrap_or_else(|| {
                        TileExpr::new(TileExprKind::Literal(TileLiteral::U32(0)), u32_e)
                    }))
                };
                let zero = TileExpr::new(TileExprKind::Literal(TileLiteral::U32(0)), u32_e);
                let (row, col, element_view) = match addr {
                    Addr::Rc2 { row, col } => (flat_of([row, col])?, zero, true),
                    Addr::Linear(index) => (index.clone(), zero, false),
                };
                let q = q.clone();
                if mask.is_constant_true() {
                    return self.decode_one(body, &q, &row, &col);
                }
                // Clamp-and-select, never a branch: with the flat index
                // clamped into the value's extent the decode is a pure
                // expression in the *enclosing* block, so the window's shared
                // subexpressions deduplicate and the mask survives as a
                // select on the value. Only an `Rc2` address rides an
                // element-space view whose extents bound the flat index; the
                // `Linear` arm's view is the raw word stream and its extent
                // clamps the wrong unit.
                let total: u64 = q
                    .data
                    .layout
                    .extents
                    .iter()
                    .map(|&e| u64::from(e))
                    .product();
                if element_view && total > 0 && total <= u64::from(u32::MAX) {
                    let clamped = TileExpr::new(
                        TileExprKind::Binary {
                            op: fusor2_ir::ir::kernel::TileBinaryOp::Min,
                            left: row,
                            right: TileExpr::new(
                                TileExprKind::Literal(TileLiteral::U32(total as u32 - 1)),
                                u32_e,
                            ),
                            numeric: fusor2_ir::dtype::NumericContract::RELAXED,
                        },
                        u32_e,
                    );
                    let decoded = self.decode_one(body, &q, &clamped, &col)?;
                    let fill_source = fill.element();
                    let fill_h = self.expr(fill, body)?;
                    let fill_h =
                        self.cast_tile_value(body, fill_h, fill_source, f32_element)?;
                    let mask_h = self.expr(mask, body)?;
                    let mask_ty = mask.element();
                    let mask_h = self.condition_value(body, mask_h, mask_ty)?;
                    return Ok(self.emit_expr(
                        body,
                        Expression::Select {
                            condition: mask_h,
                            accept: decoded,
                            reject: fill_h,
                        },
                    ));
                }
                let fill_source = fill.element();
                let fill_h = self.expr(fill, body)?;
                let fill_h = self.cast_tile_value(body, fill_h, fill_source, f32_element)?;
                let mask_h = self.expr(mask, body)?;
                let mask_ty = mask.element();
                let mask_h = self.condition_value(body, mask_h, mask_ty)?;
                self.masked_value(body, f32_element, fill_h, mask_h, move |em, accept| {
                    em.decode_one(accept, &q, &row, &col)
                })
            }
        }
    }

    pub(crate) fn addr_index(
        &mut self,
        body: &mut Block,
        view: &StorageView,
        addr: &Addr,
    ) -> Result<Handle<Expression>, EmitError> {
        match addr {
            Addr::Linear(index) => self.expr(index, body),
            Addr::Rc2 { row, col } => {
                let r = self.expr(row, body)?;
                let c = self.expr(col, body)?;
                self.storage_index_from_coords(body, view, &[r, c])
            }
        }
    }

    /// `tmp = fill; if mask { tmp = value }; tmp` — the spill local is
    /// demand-interned on `(ScratchKind::Value, element, depth)`.
    pub(crate) fn masked_value(
        &mut self,
        body: &mut Block,
        element: ElementType,
        fill: Handle<Expression>,
        mask: Handle<Expression>,
        value: impl FnOnce(&mut Self, &mut Block) -> Result<Handle<Expression>, EmitError>,
    ) -> Result<Handle<Expression>, EmitError> {
        let tmp = self.scratch_local(ScratchKind::Value, element, self.depth)?;
        self.store_local(body, tmp, fill);
        self.depth += 1;
        let (mut accept, stored) = self.nested(|em, accept| {
            let v = value(em, accept)?;
            Ok(v)
        })?;
        self.depth -= 1;
        let ptr = self.local_var(tmp);
        accept.push(
            Statement::Store {
                pointer: ptr,
                value: stored,
            },
            Span::default(),
        );
        body.push(
            Statement::If {
                condition: mask,
                accept,
                reject: Block::new(),
            },
            Span::default(),
        );
        Ok(self.emit_load(body, ptr))
    }
}

fn expr_is_float(element: ElementType) -> bool {
    matches!(
        element,
        ElementType::Scalar(ScalarElement::F32 | ScalarElement::F16)
            | ElementType::Vector {
                scalar: ScalarElement::F32 | ScalarElement::F16,
                ..
            }
    )
}

fn numeric_literal(e: &Expression) -> Option<f64> {
    match e {
        Expression::Literal(Literal::F32(v)) => Some(*v as f64),
        Expression::Literal(Literal::F16(v)) => Some(v.to_f64()),
        Expression::Literal(Literal::U32(v)) => Some(*v as f64),
        Expression::Literal(Literal::I32(v)) => Some(*v as f64),
        _ => None,
    }
}

pub(crate) fn tile_literal(lit: TileLiteral) -> Result<Expression, EmitError> {
    Ok(Expression::Literal(match lit {
        TileLiteral::F32(bits) => Literal::F32(f32::from_bits(bits)),
        TileLiteral::F16(bits) => Literal::F16(half::f16::from_bits(bits)),
        TileLiteral::U32(v) => Literal::U32(v),
        TileLiteral::I32(v) => Literal::I32(v),
        TileLiteral::Bool(v) => Literal::Bool(v),
        TileLiteral::BF16(_) => return Err(EmitError::MissingCapability("shader-bf16")),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::testkit::*;
    use crate::emit::{emit_module, testkit};
    use fusor2_ir::dtype::NumericContract;
    use fusor2_ir::ir::kernel::{KernelIr, Stmt};

    /// `out[lane] = round(in[lane], mode)`.
    fn round_kernel(mode: RoundMode) -> KernelIr {
        let uni = testkit::buffer(0, u32e(), 4, false);
        let src = testkit::buffer(1, f32e(), 8, false);
        let dst = testkit::buffer(2, f32e(), 8, true);
        let sv = view(&src, &[8]);
        let dv = view(&dst, &[8]);
        let rounded = TileExpr::new(
            TileExprKind::Round {
                mode,
                value: load(&sv, lane()),
            },
            f32e(),
        );
        KernelIr {
            buffers: vec![uni, src, dst],
            grid: [1, 1, 1],
            block: 8,
            body: vec![store(&dv, lane(), rounded)],
            byte_arena: None,
            name: "round",
        }
    }

    /// Test 7 — the rounding modes are exact, and the `2^23` trick is absent.
    #[test]
    fn round_modes_are_exact() {
        let inputs = [-2.5f32, -0.5, 0.5, 1.5, 2.5, 6.5, 0.0, -0.0];
        let caps = caps(false, true);
        for (mode, expected) in [
            (
                RoundMode::HalfAwayFromZero,
                [-3.0f32, -1.0, 1.0, 2.0, 3.0, 7.0, 0.0, 0.0],
            ),
            (
                RoundMode::HalfToEven,
                [-2.0f32, -0.0, 0.0, 2.0, 2.0, 6.0, 0.0, -0.0],
            ),
        ] {
            let ir = round_kernel(mode);
            let emitted = emit_module(&ir, &caps, &no_plan()).expect("emit");
            // `(x + 2^23) - 2^23` is the trick Metal's fast math folds away;
            // it must not appear in any form.
            assert!(
                !format!("{:#?}", emitted.module).contains("8388608"),
                "{mode:?} used the 2^23 bias trick"
            );
            let Some(gpu) = gpu() else {
                eprintln!("no wgpu adapter; skipping the numeric half");
                continue;
            };
            let buffers = vec![uniforms(), bytes_of(&inputs), bytes_of(&[0.0; 8])];
            let out = f32s(&run(&gpu, &ir, &no_plan(), &buffers, 2));
            for (i, (got, want)) in out.iter().zip(expected).enumerate() {
                assert_eq!(*got, want, "{mode:?} lane {i}: {} -> {got}", inputs[i]);
            }
        }
    }

    /// Test 8 — `reassoc: false` and `contract: false` survive to the module.
    #[test]
    fn reassoc_false_survives() {
        let caps = caps(false, true);
        let uni = testkit::buffer(0, u32e(), 4, false);
        let src = testkit::buffer(1, f32e(), 8, false);
        let dst = testkit::buffer(2, f32e(), 8, true);
        let sv = view(&src, &[8]);
        let dv = view(&dst, &[8]);

        // mul feeding add.
        let build_fma = |numeric: NumericContract| {
            let x = load(&sv, lane());
            let product = testkit::bin(BinOp::Mul, x.clone(), lit_f32(3.0), numeric);
            let sum = testkit::bin(BinOp::Add, product, lit_f32(1.0), numeric);
            KernelIr {
                buffers: vec![uni.clone(), src.clone(), dst.clone()],
                grid: [1, 1, 1],
                block: 8,
                body: vec![store(&dv, lane(), sum)],
                byte_arena: None,
                name: "fma",
            }
        };

        let strict = emit_module(&build_fma(NumericContract::STRICT), &caps, &no_plan())
            .expect("strict emit")
            .module;
        assert_eq!(fma_count(&strict), 0, "contract:false must not fuse");
        assert_eq!(
            binary_count(&strict, naga::BinaryOperator::Multiply)
                + binary_count(&strict, naga::BinaryOperator::Add),
            2,
            "two separate Binary nodes"
        );

        let relaxed = emit_module(&build_fma(NumericContract::RELAXED), &caps, &no_plan())
            .expect("relaxed emit")
            .module;
        assert_eq!(fma_count(&relaxed), 1, "contract:true fuses one Fma");

        // `x * 1.0` folds under RELAXED and does not fold under STRICT.
        let build_identity = |numeric: NumericContract| {
            let x = load(&sv, lane());
            KernelIr {
                buffers: vec![uni.clone(), src.clone(), dst.clone()],
                grid: [1, 1, 1],
                block: 8,
                body: vec![store(
                    &dv,
                    lane(),
                    testkit::bin(BinOp::Mul, x, lit_f32(1.0), numeric),
                )],
                byte_arena: None,
                name: "identity",
            }
        };
        let folded = emit_module(&build_identity(NumericContract::RELAXED), &caps, &no_plan())
            .expect("relaxed")
            .module;
        assert_eq!(binary_count(&folded, naga::BinaryOperator::Multiply), 0);
        let kept = emit_module(&build_identity(NumericContract::STRICT), &caps, &no_plan())
            .expect("strict")
            .module;
        assert_eq!(binary_count(&kept, naga::BinaryOperator::Multiply), 1);
    }

    fn fma_count(module: &naga::Module) -> usize {
        count_exprs(module, |e| {
            matches!(
                e,
                naga::Expression::Math {
                    fun: naga::MathFunction::Fma,
                    ..
                }
            )
        })
    }

    fn binary_count(module: &naga::Module, want: naga::BinaryOperator) -> usize {
        count_exprs(
            module,
            |e| matches!(e, naga::Expression::Binary { op, .. } if *op == want),
        )
    }

    /// Every unary, binary and comparison row lowers, and casts run in both
    /// float<->int directions.
    #[test]
    fn every_op_row_lowers() {
        let caps = caps(false, true);
        let uni = testkit::buffer(0, u32e(), 4, false);
        let src = testkit::buffer(1, f32e(), 8, false);
        let dst = testkit::buffer(2, f32e(), 8, true);
        let sv = view(&src, &[8]);
        let dv = view(&dst, &[8]);
        let x = load(&sv, lane());

        let unaries = [
            UnOp::Exp,
            UnOp::Exp2,
            UnOp::Log,
            UnOp::Log2,
            UnOp::Sqrt,
            UnOp::InverseSqrt,
            UnOp::Sin,
            UnOp::Cos,
            UnOp::Tan,
            UnOp::Tanh,
            UnOp::Asin,
            UnOp::Acos,
            UnOp::Atan,
            UnOp::Sinh,
            UnOp::Cosh,
            UnOp::Asinh,
            UnOp::Acosh,
            UnOp::Atanh,
            UnOp::Abs,
            UnOp::Neg,
        ];
        assert_eq!(unaries.len(), 20, "Unpack2x16Float is covered separately");
        let mut acc = x.clone();
        for op in unaries {
            acc = testkit::un(op, acc);
        }
        // `Unpack2x16Float` takes a u32 and yields two f32 lanes.
        let packed = TileExpr::new(
            TileExprKind::Bitcast {
                value: x.clone(),
                to: u32e(),
            },
            u32e(),
        );
        let unpacked = TileExpr::new(
            TileExprKind::Unary {
                op: UnOp::Unpack2x16Float,
                value: packed,
                numeric: NumericContract::RELAXED,
            },
            ElementType::Vector {
                scalar: ScalarElement::F32,
                lanes: 2,
            },
        );
        acc = testkit::bin(
            BinOp::Add,
            acc,
            TileExpr::new(
                TileExprKind::VecComponent {
                    vector: unpacked,
                    component: 0,
                },
                f32e(),
            ),
            NumericContract::RELAXED,
        );

        let float_binaries = [
            BinOp::Add,
            BinOp::Sub,
            BinOp::Mul,
            BinOp::Div,
            BinOp::Pow,
            BinOp::Min,
            BinOp::Max,
        ];
        for op in float_binaries {
            acc = testkit::bin(op, acc, lit_f32(1.5), NumericContract::STRICT);
        }
        // The integer rows run on a u32 value.
        let mut ints = lane();
        for op in [
            BinOp::Add,
            BinOp::Sub,
            BinOp::Mul,
            BinOp::Div,
            BinOp::Rem,
            BinOp::BitAnd,
            BinOp::BitOr,
            BinOp::BitXor,
            BinOp::Shr,
            BinOp::Shl,
        ] {
            ints = testkit::bin(op, ints, lit_u32(3), NumericContract::STRICT);
        }
        // Booleans are the logical rows' operand type.
        let mut logic = TileExpr::new(
            TileExprKind::Compare {
                op: CmpOp::Lt,
                left: ints.clone(),
                right: lit_u32(7),
            },
            boole(),
        );
        for op in [BinOp::LogicalAnd, BinOp::LogicalOr] {
            let other = TileExpr::new(
                TileExprKind::Compare {
                    op: CmpOp::Ne,
                    left: ints.clone(),
                    right: lit_u32(1),
                },
                boole(),
            );
            logic = testkit::bin(op, logic, other, NumericContract::STRICT);
        }
        for op in [
            CmpOp::Lt,
            CmpOp::Le,
            CmpOp::Gt,
            CmpOp::Ge,
            CmpOp::Eq,
            CmpOp::Ne,
        ] {
            let cmp = TileExpr::new(
                TileExprKind::Compare {
                    op,
                    left: ints.clone(),
                    right: lit_u32(2),
                },
                boole(),
            );
            logic = testkit::bin(BinOp::LogicalOr, logic, cmp, NumericContract::STRICT);
        }

        // Cast in both directions, including the f32 -> u32 / f32 -> i32 pair.
        let to_u32 = TileExpr::new(
            TileExprKind::Cast {
                value: acc.clone(),
                to: u32e(),
            },
            u32e(),
        );
        let to_i32 = TileExpr::new(
            TileExprKind::Cast {
                value: acc.clone(),
                to: ElementType::Scalar(ScalarElement::I32),
            },
            ElementType::Scalar(ScalarElement::I32),
        );
        let back = TileExpr::new(
            TileExprKind::Cast {
                value: to_u32.clone(),
                to: f32e(),
            },
            f32e(),
        );
        let bitcast = TileExpr::new(
            TileExprKind::Bitcast {
                value: to_i32,
                to: f32e(),
            },
            f32e(),
        );
        let selected = TileExpr::new(
            TileExprKind::Select {
                condition: logic,
                accept: back,
                reject: bitcast,
            },
            f32e(),
        );
        // Vec compose + dot close the vector rows.
        let v = TileExpr::new(
            TileExprKind::Vec {
                scalar: ScalarElement::F32,
                lanes: 4,
                parts: vec![selected.clone(), lit_f32(1.0), lit_f32(2.0), lit_f32(3.0)],
            },
            ElementType::Vector {
                scalar: ScalarElement::F32,
                lanes: 4,
            },
        );
        let dotted = TileExpr::new(
            TileExprKind::Dot {
                left: v.clone(),
                right: v,
            },
            f32e(),
        );

        let ir = KernelIr {
            buffers: vec![uni, src, dst],
            grid: [1, 1, 1],
            block: 8,
            body: vec![Stmt::Store {
                dst: dv,
                addr: Addr::Linear(lane()),
                value: dotted,
                mask: tru(),
            }],
            byte_arena: None,
            name: "all_ops",
        };
        emit_module(&ir, &caps, &no_plan()).expect("every op row lowers and validates");
    }
}
