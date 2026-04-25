//! Pure-expression lowering — turns extracted IR nodes into Naga expressions.

use egg::{Id, Language};
use naga::{
    BinaryOperator, Block, Expression, Handle, Literal, LocalVariable, MathFunction, Scalar,
    ScalarKind, Span, Statement, Type, UnaryOperator,
};

use crate::language::{DispatchNode, SimdNode, TensorIr, extract_list};
use crate::types::{
    BinaryOp, BinderKind, DType, IndexLevel, MemTier, ScalarValue, TernaryOp, UnaryOp, VarRef,
    slots,
};

use super::{
    BinderFrame, CodegenCtx, MAX_DISPATCH_WORKGROUPS_PER_DIMENSION, ThetaFrame, binary_args,
};

impl CodegenCtx<'_> {
    /// Find the `depth`-th frame of the given `kind` on the binder stack,
    /// counting from the top (0 = innermost of that kind). Returns None when
    /// the stack doesn't have that many frames of the kind.
    fn kinded_frame(&self, kind: BinderKind, depth: u32) -> Option<&BinderFrame> {
        self.binder_stack
            .iter()
            .rev()
            .filter(|f| f.kind() == kind)
            .nth(depth as usize)
    }

    fn theta_frame(&self, depth: u32) -> Option<&ThetaFrame> {
        match self.kinded_frame(BinderKind::Theta, depth)? {
            BinderFrame::Theta(tf) => Some(tf),
            _ => None,
        }
    }

    /// Look up the scalar handle bound to `var` in the current scope. For
    /// `Var(Acc)`, returns slot 0 when the accumulator has arity 1 (bare
    /// `Var(Acc)` === `Extract(0, Var(Acc))` for scalar Thetas).
    pub(super) fn lookup_var(&self, var: &VarRef) -> Option<Handle<Expression>> {
        match var {
            VarRef::Bound {
                kind: BinderKind::Theta,
                slot,
                depth,
            } => {
                let frame = self.theta_frame(*depth)?;
                match *slot {
                    slots::THETA_ITER => Some(frame.iter),
                    slots::THETA_ACC => {
                        (frame.acc_handles.len() == 1).then_some(frame.acc_handles[0])
                    }
                    _ => None,
                }
            }
            VarRef::Bound {
                kind: BinderKind::Dispatch,
                slot,
                depth,
            } => match self.kinded_frame(BinderKind::Dispatch, *depth)? {
                BinderFrame::Dispatch(df) => match *slot {
                    slots::DISPATCH_LANE => Some(df.lane),
                    slots::DISPATCH_SIMDGROUP => Some(df.simdgroup),
                    slots::DISPATCH_WORKGROUP => Some(df.workgroup),
                    _ => None,
                },
                _ => None,
            },
        }
    }

    /// Handles for all accumulator slots when `var` names a Theta acc.
    pub(super) fn lookup_var_tuple(&self, var: &VarRef) -> Option<&Vec<Handle<Expression>>> {
        match var {
            VarRef::Bound {
                kind: BinderKind::Theta,
                slot: slots::THETA_ACC,
                depth,
            } => Some(&self.theta_frame(*depth)?.acc_handles),
            _ => None,
        }
    }

    /// Slot types for all accumulator slots when `var` names a Theta acc.
    pub(super) fn lookup_var_tuple_types(&self, var: &VarRef) -> Option<&Vec<Handle<Type>>> {
        match var {
            VarRef::Bound {
                kind: BinderKind::Theta,
                slot: slots::THETA_ACC,
                depth,
            } => Some(&self.theta_frame(*depth)?.acc_types),
            _ => None,
        }
    }

    /// Type of a bare-`Var(Acc)` reference: only valid when the accumulator
    /// has arity 1 (scalar Theta). Returns `None` for tuple accs, where the
    /// caller should use `Extract` and `lookup_var_tuple_types`.
    pub(super) fn lookup_var_scalar_type(&self, var: &VarRef) -> Option<Handle<Type>> {
        match var {
            VarRef::Bound {
                kind: BinderKind::Theta,
                slot: slots::THETA_ACC,
                depth,
            } => {
                let frame = self.theta_frame(*depth)?;
                (frame.acc_types.len() == 1).then_some(frame.acc_types[0])
            }
            _ => None,
        }
    }

    /// True iff `var` has any binding in scope (scalar or tuple).
    pub(super) fn var_is_bound(&self, var: &VarRef) -> bool {
        self.lookup_var(var).is_some() || self.lookup_var_tuple(var).is_some()
    }

    pub(super) fn select_lowering_node(&self, canonical: Id) -> TensorIr {
        let chosen = self
            .chosen_nodes
            .get(&canonical)
            .cloned()
            .unwrap_or_else(|| self.egraph[canonical].iter().next().unwrap().clone());

        if let TensorIr::Simd(SimdNode::Var(var)) = chosen {
            if !self.var_is_bound(&var)
                && let Some(alt) = self.egraph[canonical]
                    .iter()
                    .find(|node| {
                        !matches!(
                            node,
                            TensorIr::Simd(SimdNode::Var(_) | SimdNode::Theta { .. })
                        ) && !node
                            .children()
                            .iter()
                            .any(|child| self.egraph.find(*child) == canonical)
                    })
                    .cloned()
            {
                return alt;
            }
            return TensorIr::Simd(SimdNode::Var(var));
        }

        let has_self_ref = chosen
            .children()
            .iter()
            .any(|child| self.egraph.find(*child) == canonical);
        if !has_self_ref || matches!(chosen, TensorIr::Simd(SimdNode::Theta { .. })) {
            return chosen;
        }

        self.egraph[canonical]
            .iter()
            .find(|node| {
                !matches!(node, TensorIr::Simd(SimdNode::Theta { .. }))
                    && !node
                        .children()
                        .iter()
                        .any(|child| self.egraph.find(*child) == canonical)
            })
            .cloned()
            .unwrap_or(chosen)
    }

    /// Lower an e-graph expression to a Naga expression handle.
    /// Memoized by canonical Id.
    pub(super) fn lower_expr(&mut self, id: Id) -> Handle<Expression> {
        let canonical = self.egraph.find(id);
        if let Some(&h) = self.id_cache.get(&canonical) {
            return h;
        }

        // If this Theta was already lowered (e.g., inside a loop body), emit a
        // fresh load from its accumulator local variable instead of creating a
        // second loop. The acc_ptr is a LocalVariable expression, which is
        // always in scope at any point in the function. Only applies to
        // arity-1 Thetas — tuple Thetas can only be reused via Extract, so
        // falling through to lower_extract reads the correct slot.
        // Only reuse a lowered Theta when its value is independent of any
        // enclosing Theta iter/acc binding. Softmax-style nested reductions
        // reuse the same canonical shape under different outer iter values.
        if self.egraph[canonical].data.free_var_dep.is_empty()
            && let Some(ptrs) = self.theta_acc_ptrs.get(&canonical)
            && ptrs.len() == 1
        {
            let ptr = ptrs[0];
            let h = self.emit_load(ptr);
            self.id_cache.insert(canonical, h);
            return h;
        }

        // Invariant: `select_lowering_node` filters out non-Theta nodes that
        // self-reference their canonical id (lower.rs select_lowering_node:92-110)
        // and Theta short-circuits via `theta_acc_ptrs` above. So a transitive
        // cycle here would mean the chosen extraction is malformed (extractor
        // bug). `unreachable!` documents the structural invariant.
        if !self.lowering_set.insert(canonical) {
            unreachable!(
                "lower_expr re-entered canonical {canonical:?}: extractor produced a cyclic chosen extraction"
            );
        }
        self.lowering_stack.push(canonical);

        // Use the extractor's chosen node if available, otherwise fall back
        // to the first e-node in the e-class.
        let node = self.select_lowering_node(canonical);

        // For Theta nodes, pass the canonical Id so we can record the
        // accumulator pointer for later re-use at outer scopes.
        let handle = if let TensorIr::Simd(SimdNode::Theta {
            children: [init, count, update],
            ..
        }) = &node
        {
            self.lower_theta(canonical, *init, *count, *update)
        } else {
            self.lower_node(&node)
        };

        self.lowering_stack.pop();
        self.lowering_set.remove(&canonical);
        self.id_cache.insert(canonical, handle);
        handle
    }

    pub(super) fn lower_node(&mut self, node: &TensorIr) -> Handle<Expression> {
        match node {
            TensorIr::Dispatch(DispatchNode::Token) => self.token_handle(),
            TensorIr::Dispatch(DispatchNode::Pack {
                children_list: list_id,
            }) => {
                let mut last = self.token_handle();
                for id in extract_list(self.egraph, *list_id) {
                    last = self.lower_expr(id);
                }
                last
            }
            TensorIr::Dispatch(DispatchNode::Extract { index, tuple }) => {
                self.lower_extract(*tuple, *index as usize)
            }

            // ── Literals ──
            TensorIr::Const(v) => self.lower_scalar_literal(v),

            // ── Variables ──
            TensorIr::Simd(SimdNode::Var(var)) => {
                if let Some(handle) = self.lookup_var(var) {
                    handle
                } else {
                    // Invariant: pre-codegen `verify::verify` checks every
                    // `Var(Bound { depth, .. })` against its enclosing
                    // binder count. `lower_dispatch_program` accepts only
                    // `Verified<'_>`, so an unbound `Var` here would
                    // indicate either a verifier bug or an out-of-scope
                    // construction. Encode via `unreachable!` to make the
                    // structural guarantee explicit.
                    unreachable!(
                        "unbound variable {var:?} reached codegen; verifier should have rejected"
                    )
                }
            }

            // ── Arithmetic / logic ──
            TensorIr::BinOp(name, args) => self.lower_op(*name, args),
            TensorIr::UnOp(name, arg) => self.lower_unop(*name, *arg),
            TensorIr::TernOp(TernaryOp::Fma, args) => {
                let a = self.lower_expr(args[0]);
                let b = self.lower_expr(args[1]);
                let c = self.lower_expr(args[2]);
                self.emit_math3(MathFunction::Fma, a, b, c)
            }
            TensorIr::TernOp(TernaryOp::Select, args) => {
                let cond = self.lower_expr(args[0]);
                let accept = self.lower_expr(args[1]);
                let reject = self.lower_expr(args[2]);
                self.emit_select(cond, accept, reject)
            }

            // ── Memory loads ──
            TensorIr::Simd(SimdNode::Load { tier, children }) => {
                self.lower_expr(children[1]);
                self.lower_load(tier, children[0])
            }

            // ── Theta (accumulation loop) ──
            // Invariant: `lower_expr` dispatches Theta to `lower_theta` directly
            // (lower.rs:154). `lower_node` is only reached for non-Theta nodes.
            TensorIr::Simd(SimdNode::Theta { .. }) => {
                unreachable!("Theta is dispatched in lower_expr; never reaches lower_node")
            }

            TensorIr::Simd(SimdNode::Store { tier, children }) => {
                self.lower_expr(children[2]);
                self.emit_store(tier, children[0], children[1]);
                self.token_handle()
            }
            TensorIr::Simd(SimdNode::StoreIf { tier, children }) => {
                self.lower_expr(children[3]);
                self.emit_store_if(tier, children[0], children[1], children[2]);
                self.token_handle()
            }
            TensorIr::Simd(SimdNode::Barrier { state, .. }) => {
                self.lower_expr(*state);
                self.body.push(
                    Statement::ControlBarrier(naga::Barrier::WORK_GROUP),
                    Span::UNDEFINED,
                );
                self.token_handle()
            }

            // ── SIMD operations ──
            TensorIr::Simd(SimdNode::Shuffle([src, lane])) => self.lower_shuffle(*src, *lane),
            TensorIr::Simd(SimdNode::ReduceSimd { op, src }) => self.lower_reduce_simd(*op, *src),

            other => panic!("naga_codegen: unsupported node: {other:?}"),
        }
    }

    // ── Leaf lowering ──

    pub(super) fn lower_scalar_literal(&mut self, v: &ScalarValue) -> Handle<Expression> {
        let lit = match v {
            ScalarValue::F16(f) => Literal::F16(half::f16::from_f32(f.0)),
            ScalarValue::F32(f) => Literal::F32(f.0),
            ScalarValue::I32(i) => Literal::I32(*i),
            ScalarValue::U32(u) => Literal::U32(*u),
            ScalarValue::Bool(b) => Literal::Bool(*b),
        };
        self.emit_literal(lit)
    }

    pub(super) fn lower_index(&mut self, level: IndexLevel) -> Handle<Expression> {
        match level {
            IndexLevel::Lane => {
                let lid_x = self.emit_access_index(self.local_invocation_id, 0);
                if self.simdgroups > 1 {
                    // local_invocation_id.x % SIMD_WIDTH
                    let sw = self.emit_literal(Literal::U32(self.simd_width));
                    self.emit_binary(BinaryOperator::Modulo, lid_x, sw)
                } else {
                    // local_invocation_id.x (when only 1 simdgroup, lid.x IS the lane)
                    lid_x
                }
            }
            IndexLevel::Simdgroup => {
                // local_invocation_id.x / SIMD_WIDTH
                // Gives the simdgroup index within the workgroup (0..simdgroups-1).
                let lid_x = self.emit_access_index(self.local_invocation_id, 0);
                let sw = self.emit_literal(Literal::U32(self.simd_width));
                self.emit_binary(BinaryOperator::Divide, lid_x, sw)
            }
            IndexLevel::Workgroup => {
                let wg_x = self.emit_access_index(self.workgroup_id, 0);
                let wg_y = self.emit_access_index(self.workgroup_id, 1);
                let stride = self.emit_literal(Literal::U32(MAX_DISPATCH_WORKGROUPS_PER_DIMENSION));
                let wg_y_offset = self.emit_binary(BinaryOperator::Multiply, wg_y, stride);
                let physical_wg = self.emit_binary(BinaryOperator::Add, wg_y_offset, wg_x);
                if self.simdgroups > 1 {
                    // Virtual workgroup index: physical_workgroup * simdgroups + simdgroup_in_wg.
                    // The physical workgroup index is linearized from x/y so runtimes can split
                    // large launches across WebGPU's per-dimension dispatch limit.
                    let sg_lit = self.emit_literal(Literal::U32(self.simdgroups));
                    let wg_times_sg =
                        self.emit_binary(BinaryOperator::Multiply, physical_wg, sg_lit);
                    let lid_x = self.emit_access_index(self.local_invocation_id, 0);
                    let sw = self.emit_literal(Literal::U32(self.simd_width));
                    let sg_in_wg = self.emit_binary(BinaryOperator::Divide, lid_x, sw);
                    self.emit_binary(BinaryOperator::Add, wg_times_sg, sg_in_wg)
                } else {
                    physical_wg
                }
            }
        }
    }

    // ── Arithmetic ops ──

    pub(super) fn lower_unop(&mut self, op: UnaryOp, arg_id: Id) -> Handle<Expression> {
        let arg = self.lower_expr(arg_id);
        match op {
            UnaryOp::Neg => self.emit_unary(UnaryOperator::Negate, arg),
            UnaryOp::Not => self.emit_unary(UnaryOperator::LogicalNot, arg),
            UnaryOp::Exp => self.emit_math1(MathFunction::Exp, arg),
            UnaryOp::Exp2 => self.emit_math1(MathFunction::Exp2, arg),
            UnaryOp::Log => self.emit_math1(MathFunction::Log, arg),
            UnaryOp::Log2 => self.emit_math1(MathFunction::Log2, arg),
            UnaryOp::Sin => self.emit_math1(MathFunction::Sin, arg),
            UnaryOp::Cos => self.emit_math1(MathFunction::Cos, arg),
            UnaryOp::Tan => self.emit_math1(MathFunction::Tan, arg),
            UnaryOp::Tanh => self.emit_math1(MathFunction::Tanh, arg),
            UnaryOp::Asin => self.emit_math1(MathFunction::Asin, arg),
            UnaryOp::Acos => self.emit_math1(MathFunction::Acos, arg),
            UnaryOp::Atan => self.emit_math1(MathFunction::Atan, arg),
            UnaryOp::Sinh => self.emit_math1(MathFunction::Sinh, arg),
            UnaryOp::Cosh => self.emit_math1(MathFunction::Cosh, arg),
            UnaryOp::Asinh => self.emit_math1(MathFunction::Asinh, arg),
            UnaryOp::Acosh => self.emit_math1(MathFunction::Acosh, arg),
            UnaryOp::Atanh => self.emit_math1(MathFunction::Atanh, arg),
            UnaryOp::Abs => self.emit_math1(MathFunction::Abs, arg),
            UnaryOp::Sqrt => self.emit_math1(MathFunction::Sqrt, arg),
            UnaryOp::CastF32 => self.emit_cast(arg, ScalarKind::Float, 4),
            UnaryOp::CastF16 => self.emit_cast(arg, ScalarKind::Float, 2),
            UnaryOp::CastI32 => self.emit_cast(arg, ScalarKind::Sint, 4),
            UnaryOp::CastU32 => self.emit_cast(arg, ScalarKind::Uint, 4),
            UnaryOp::CastBool => self.emit_cast(arg, ScalarKind::Bool, 1),
        }
    }

    pub(super) fn lower_op(&mut self, op: BinaryOp, args: &[Id]) -> Handle<Expression> {
        let [lhs, rhs] = binary_args(args)
            .unwrap_or_else(|| panic!("naga_codegen: BinaryOp expects 2 args, got {}", args.len()));
        let lhs = self.lower_expr(lhs);
        let rhs = self.lower_expr(rhs);
        match op {
            BinaryOp::Add => self.emit_binary(BinaryOperator::Add, lhs, rhs),
            BinaryOp::Sub => self.emit_binary(BinaryOperator::Subtract, lhs, rhs),
            BinaryOp::Mul => self.emit_binary(BinaryOperator::Multiply, lhs, rhs),
            BinaryOp::Div => self.emit_binary(BinaryOperator::Divide, lhs, rhs),
            BinaryOp::Mod => self.emit_binary(BinaryOperator::Modulo, lhs, rhs),
            BinaryOp::Lt => self.emit_binary(BinaryOperator::Less, lhs, rhs),
            BinaryOp::Le => self.emit_binary(BinaryOperator::LessEqual, lhs, rhs),
            BinaryOp::Gt => self.emit_binary(BinaryOperator::Greater, lhs, rhs),
            BinaryOp::Ge => self.emit_binary(BinaryOperator::GreaterEqual, lhs, rhs),
            BinaryOp::Eq => self.emit_binary(BinaryOperator::Equal, lhs, rhs),
            BinaryOp::Neq => self.emit_binary(BinaryOperator::NotEqual, lhs, rhs),
            BinaryOp::And => self.emit_binary(BinaryOperator::And, lhs, rhs),
            BinaryOp::Or => self.emit_binary(BinaryOperator::InclusiveOr, lhs, rhs),
            BinaryOp::Xor => self.emit_binary(BinaryOperator::ExclusiveOr, lhs, rhs),
            BinaryOp::Shl => self.emit_binary(BinaryOperator::ShiftLeft, lhs, rhs),
            BinaryOp::Shr => self.emit_binary(BinaryOperator::ShiftRight, lhs, rhs),
            BinaryOp::Max => self.emit_math2(MathFunction::Max, lhs, rhs),
            BinaryOp::Min => self.emit_math2(MathFunction::Min, lhs, rhs),
            BinaryOp::Pow => self.emit_math2(MathFunction::Pow, lhs, rhs),
        }
    }

    // ── Memory loads ──

    pub(super) fn lower_load(&mut self, tier: &MemTier, addr_id: Id) -> Handle<Expression> {
        let buf_name = tier.buffer();
        let mut addr = self.lower_expr(addr_id);

        // For scaled tg buffers, add simdgroup offset: addr += sg_idx * stride
        if let MemTier::Threadgroup(name) = tier
            && let Some(&stride) = self.tg_sg_read_strides.get(name)
        {
            let sg_idx = self.lower_index(IndexLevel::Simdgroup);
            let stride_lit = self.emit_literal(Literal::U32(stride));
            let offset = self.emit_binary(BinaryOperator::Multiply, sg_idx, stride_lit);
            addr = self.emit_binary(BinaryOperator::Add, addr, offset);
        }

        let (gv, _is_wg) = *self
            .buffer_map
            .get(tier)
            .unwrap_or_else(|| panic!("unknown buffer: {tier}"));

        let gv_expr = self
            .expressions
            .append(Expression::GlobalVariable(gv), Span::UNDEFINED);
        let access = self.emit_access(gv_expr, addr);
        let load = self.emit_load(access);
        self.named_expressions
            .insert(load, format!("val_{buf_name}"));
        load
    }

    pub(super) fn token_handle(&mut self) -> Handle<Expression> {
        self.emit_literal(Literal::U32(0))
    }

    pub(super) fn lower_extract(&mut self, tuple_id: Id, index: usize) -> Handle<Expression> {
        let canonical = self.egraph.find(tuple_id);

        // Invariants below: `verify::pack_arity` follows `Theta { init, .. }`
        // chains to a `Pack` (or scalar leaf) and rejects out-of-bounds
        // extracts at verification time. Codegen only sees verified programs,
        // so an out-of-bounds index here would indicate verifier drift.
        if let Some(ptrs) = self.theta_acc_ptrs.get(&canonical) {
            let ptr = ptrs.get(index).copied().unwrap_or_else(|| {
                unreachable!(
                    "extract index {index} of Theta {canonical:?} (arity {}): verifier should have rejected",
                    ptrs.len()
                )
            });
            return self.emit_load(ptr);
        }

        if self.egraph[canonical]
            .iter()
            .any(|node| matches!(node, TensorIr::Simd(SimdNode::Theta { .. })))
            && !self.lowering_set.contains(&canonical)
        {
            let _ = self.lower_expr(tuple_id);
            if let Some(ptrs) = self.theta_acc_ptrs.get(&canonical) {
                let ptr = ptrs.get(index).copied().unwrap_or_else(|| {
                    unreachable!(
                        "extract index {index} of recovered Theta {canonical:?} (arity {}): verifier should have rejected",
                        ptrs.len()
                    )
                });
                return self.emit_load(ptr);
            }
        }

        let tuple_node = self.select_lowering_node(canonical);
        if let TensorIr::Simd(SimdNode::Var(var)) = tuple_node {
            if let Some(handles) = self.lookup_var_tuple(&var) {
                return *handles.get(index).unwrap_or_else(|| {
                    panic!("tuple var extract {index} out of bounds for {var:?}")
                });
            }
            if let Some(handle) = self.lookup_var(&var) {
                if index == 0 {
                    return handle;
                }
                panic!("tuple extract {index} out of bounds for scalar var {var:?}");
            }
        }

        let tuple_node = self.select_lowering_node(canonical);
        if let TensorIr::Dispatch(DispatchNode::Pack {
            children_list: list_id,
        }) = tuple_node
        {
            let mut selected = None;
            for (i, child) in extract_list(self.egraph, list_id).iter().enumerate() {
                let handle = self.lower_expr(*child);
                if i == index {
                    selected = Some(handle);
                }
            }
            return selected.unwrap_or_else(|| panic!("tuple extract {index} out of bounds"));
        }

        panic!("extract from non-pack node: {tuple_node:?}");
    }

    pub(super) fn emit_store(&mut self, tier: &MemTier, addr_id: Id, value_id: Id) {
        let buf_name = tier.buffer();
        let addr_h = self.lower_expr(addr_id);
        let val_h = self.lower_expr(value_id);

        let (gv, _) = *self
            .buffer_map
            .get(tier)
            .unwrap_or_else(|| panic!("unknown buffer for store: {tier}"));
        let _ = buf_name;

        let gv_expr = self
            .expressions
            .append(Expression::GlobalVariable(gv), Span::UNDEFINED);
        let ptr = self.emit_access(gv_expr, addr_h);

        self.body.push(
            Statement::Store {
                pointer: ptr,
                value: val_h,
            },
            Span::UNDEFINED,
        );
    }

    pub(super) fn emit_store_if(&mut self, tier: &MemTier, cond_id: Id, addr_id: Id, value_id: Id) {
        let cond_handle = self.lower_expr(cond_id);

        let saved_body = std::mem::replace(&mut self.body, Block::new());
        self.emit_store(tier, addr_id, value_id);
        let accept = std::mem::replace(&mut self.body, saved_body);

        self.body.push(
            Statement::If {
                condition: cond_handle,
                accept,
                reject: Block::new(),
            },
            Span::UNDEFINED,
        );
    }

    // ── Theta (accumulation loop) ──

    pub(super) fn lower_theta(
        &mut self,
        theta_id: Id,
        init_id: Id,
        count_id: Id,
        update_id: Id,
    ) -> Handle<Expression> {
        let init_parts = self.theta_parts(init_id);
        let update_parts = self.theta_parts(update_id);
        assert_eq!(
            init_parts.len(),
            update_parts.len(),
            "theta init/update arity mismatch: {} vs {}",
            init_parts.len(),
            update_parts.len()
        );

        // Allocate the accumulator local(s) and seed them with the init
        // expression's value, then set up the iteration counter. Each of these
        // runs OUTSIDE the Theta's scope, so Var lookups fall through to the
        // surrounding frame.
        let (acc_ptrs, acc_types) = self.init_theta_accumulators(&init_parts);
        if self.egraph[theta_id].data.free_var_dep.is_empty() {
            self.theta_acc_ptrs.insert(theta_id, acc_ptrs.clone());
        }
        let ctr_ptr = self.create_theta_counter();
        let count_val = self.lower_expr(count_id);

        // Snapshot the outer body and id cache so we can restore them after
        // the Theta's body finishes emitting.
        let outer_body = std::mem::replace(&mut self.body, Block::new());
        let outer_cache = self.id_cache.clone();

        // Push a new Theta frame and populate its iter + acc slots. After this
        // point, any `Var(Bound { kind: Theta, .., 0 })` in the body resolves
        // through the new frame.
        let frame = self.open_theta_frame(ctr_ptr, &acc_ptrs, &acc_types);
        self.binder_stack.push(BinderFrame::Theta(frame));
        self.id_cache.clear();

        self.store_theta_updates(&acc_ptrs, &update_parts);
        self.emit_theta_loop(ctr_ptr, count_val, outer_body);

        // Pop the Theta frame and restore the outer id cache before returning.
        self.binder_stack.pop();
        self.id_cache = outer_cache;
        self.emit_load(acc_ptrs[0])
    }

    /// Build the binding frame for a new Theta — loads counter and
    /// accumulator values, populates the `ThetaFrame` slots. Scalar Thetas
    /// are represented as one-slot tuples; the arity-1 load is named `acc`
    /// for WGSL readability.
    pub(super) fn open_theta_frame(
        &mut self,
        ctr_ptr: Handle<Expression>,
        acc_ptrs: &[Handle<Expression>],
        acc_types: &[Handle<Type>],
    ) -> ThetaFrame {
        let iter = self.emit_load(ctr_ptr);
        self.named_expressions.insert(iter, "k".into());

        let acc_handles: Vec<_> = acc_ptrs.iter().map(|ptr| self.emit_load(*ptr)).collect();
        if acc_handles.len() == 1 {
            self.named_expressions.insert(acc_handles[0], "acc".into());
        }
        ThetaFrame {
            iter,
            acc_handles,
            acc_types: acc_types.to_vec(),
        }
    }

    pub(super) fn init_theta_accumulators(
        &mut self,
        init_parts: &[Id],
    ) -> (Vec<Handle<Expression>>, Vec<Handle<Type>>) {
        let mut acc_ptrs = Vec::with_capacity(init_parts.len());
        let mut acc_types = Vec::with_capacity(init_parts.len());
        for (index, init_part) in init_parts.iter().enumerate() {
            let ty = self.infer_expr_type(*init_part);
            let init_val = self.lower_expr(*init_part);
            let local = self.local_variables.append(
                LocalVariable {
                    name: Some(format!("acc_{index}")),
                    ty,
                    init: None,
                },
                Span::UNDEFINED,
            );
            let ptr = self
                .expressions
                .append(Expression::LocalVariable(local), Span::UNDEFINED);
            self.body.push(
                Statement::Store {
                    pointer: ptr,
                    value: init_val,
                },
                Span::UNDEFINED,
            );
            acc_ptrs.push(ptr);
            acc_types.push(ty);
        }
        (acc_ptrs, acc_types)
    }

    pub(super) fn create_theta_counter(&mut self) -> Handle<Expression> {
        let ctr_var = self.local_variables.append(
            LocalVariable {
                name: Some("k".into()),
                ty: self.types.scalar(self.module, Scalar::U32),
                init: None,
            },
            Span::UNDEFINED,
        );
        let ctr_ptr = self
            .expressions
            .append(Expression::LocalVariable(ctr_var), Span::UNDEFINED);
        let zero = self.emit_literal(Literal::U32(0));
        self.body.push(
            Statement::Store {
                pointer: ctr_ptr,
                value: zero,
            },
            Span::UNDEFINED,
        );
        ctr_ptr
    }

    pub(super) fn store_theta_updates(
        &mut self,
        acc_ptrs: &[Handle<Expression>],
        update_parts: &[Id],
    ) {
        for (ptr, update_part) in acc_ptrs.iter().zip(update_parts.iter()) {
            let update_val = self.lower_expr(*update_part);
            self.body.push(
                Statement::Store {
                    pointer: *ptr,
                    value: update_val,
                },
                Span::UNDEFINED,
            );
        }
    }

    pub(super) fn emit_theta_loop(
        &mut self,
        ctr_ptr: Handle<Expression>,
        count_val: Handle<Expression>,
        outer_body: Block,
    ) {
        let one = self.emit_literal(Literal::U32(1));
        let ctr_load = self.emit_load(ctr_ptr);
        let ctr_inc = self.emit_binary(BinaryOperator::Add, ctr_load, one);
        self.body.push(
            Statement::Store {
                pointer: ctr_ptr,
                value: ctr_inc,
            },
            Span::UNDEFINED,
        );

        let loop_body_block = std::mem::replace(&mut self.body, Block::new());
        let ctr_load = self.emit_load(ctr_ptr);
        let break_cond = self.emit_binary(BinaryOperator::GreaterEqual, ctr_load, count_val);
        let continuing = std::mem::replace(&mut self.body, outer_body);
        self.body.push(
            Statement::Loop {
                body: loop_body_block,
                continuing,
                break_if: Some(break_cond),
            },
            Span::UNDEFINED,
        );
    }

    pub(super) fn theta_parts(&self, id: Id) -> Vec<Id> {
        let canonical = self.egraph.find(id);
        match self.select_lowering_node(canonical) {
            TensorIr::Dispatch(DispatchNode::Pack {
                children_list: list_id,
            }) => extract_list(self.egraph, list_id),
            _ => vec![id],
        }
    }

    fn type_for_dtype(&mut self, dtype: DType) -> Handle<Type> {
        self.types.scalar_for_dtype(self.module, dtype)
    }

    fn dtype_for_tier(&self, tier: &MemTier) -> DType {
        self.buffer_dtypes.get(tier).copied().unwrap_or(DType::F32)
    }

    pub(super) fn infer_expr_type(&mut self, id: Id) -> Handle<Type> {
        let canonical = self.egraph.find(id);
        if let Some(dtype) = self.egraph[canonical].data.dtype {
            return self.type_for_dtype(dtype);
        }
        match self.select_lowering_node(canonical) {
            TensorIr::Const(value) => match value {
                ScalarValue::F16(_) => self.types.scalar(self.module, Scalar::F16),
                ScalarValue::F32(_) => self.types.scalar(self.module, Scalar::F32),
                ScalarValue::I32(_) => self.types.scalar(self.module, Scalar::I32),
                ScalarValue::U32(_) => self.types.scalar(self.module, Scalar::U32),
                ScalarValue::Bool(_) => self.types.scalar(self.module, Scalar::BOOL),
            },
            TensorIr::Simd(SimdNode::Var(var)) => match var {
                VarRef::Bound {
                    kind: BinderKind::Theta,
                    slot: slots::THETA_ACC,
                    ..
                } => {
                    if let Some(ty) = self.lookup_var_scalar_type(&var) {
                        ty
                    } else {
                        self.types.scalar(self.module, Scalar::F32)
                    }
                }
                // Iter index (Theta) and thread indices (Dispatch) are u32.
                _ => self.types.scalar(self.module, Scalar::U32),
            },
            TensorIr::Simd(SimdNode::Load { tier, .. }) => {
                self.type_for_dtype(self.dtype_for_tier(&tier))
            }
            TensorIr::Simd(SimdNode::ReduceSimd { src, .. }) => self.infer_expr_type(src),
            TensorIr::Simd(SimdNode::Shuffle([src, _])) => self.infer_expr_type(src),
            TensorIr::UnOp(op, arg) => match op {
                UnaryOp::CastF16 => self.types.scalar(self.module, Scalar::F16),
                UnaryOp::CastF32 => self.types.scalar(self.module, Scalar::F32),
                UnaryOp::CastI32 => self.types.scalar(self.module, Scalar::I32),
                UnaryOp::CastU32 => self.types.scalar(self.module, Scalar::U32),
                UnaryOp::CastBool | UnaryOp::Not => self.types.scalar(self.module, Scalar::BOOL),
                _ => self.infer_expr_type(arg),
            },
            TensorIr::TernOp(op, args) => match op {
                TernaryOp::Fma => self.infer_expr_type(args[0]),
                TernaryOp::Select => self.infer_expr_type(args[1]),
            },
            TensorIr::BinOp(name, args) => match name {
                BinaryOp::Lt
                | BinaryOp::Le
                | BinaryOp::Gt
                | BinaryOp::Ge
                | BinaryOp::Eq
                | BinaryOp::Neq => self.types.scalar(self.module, Scalar::BOOL),
                _ if args.is_empty() => self.types.scalar(self.module, Scalar::U32),
                _ => self.infer_expr_type(args[0]),
            },
            TensorIr::Dispatch(DispatchNode::Extract { index, tuple }) => {
                let tuple_canonical = self.egraph.find(tuple);
                match self.select_lowering_node(tuple_canonical) {
                    TensorIr::Dispatch(DispatchNode::Pack {
                        children_list: list_id,
                    }) => self.infer_expr_type(extract_list(self.egraph, list_id)[index as usize]),
                    TensorIr::Simd(SimdNode::Var(var)) => self
                        .lookup_var_tuple_types(&var)
                        .and_then(|types| types.get(index as usize).copied())
                        .unwrap_or_else(|| self.types.scalar(self.module, Scalar::U32)),
                    TensorIr::Simd(SimdNode::Theta {
                        children: [init, ..],
                        ..
                    }) => {
                        let init_parts = self.theta_parts(init);
                        if let Some(part) = init_parts.get(index as usize) {
                            self.infer_expr_type(*part)
                        } else {
                            self.types.scalar(self.module, Scalar::U32)
                        }
                    }
                    _ => self.types.scalar(self.module, Scalar::U32),
                }
            }
            TensorIr::Simd(SimdNode::Theta {
                children: [init, ..],
                ..
            }) => self.infer_expr_type(init),
            _ => self.types.scalar(self.module, Scalar::U32),
        }
    }

    // ── SIMD operations ──
}
