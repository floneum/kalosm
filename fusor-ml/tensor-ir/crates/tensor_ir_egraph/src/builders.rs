use std::cmp::Reverse;

use egg::{Id, RecExpr};

use crate::language::{DispatchNode, EffectNode, HighLevelNode, SimdNode, TensorIr};
use crate::types::{
    BinaryOp, DType, IndexLevel, MemTier, ReduceOp, ScalarValue, Shape, Strides, TernaryOp,
    UnaryOp, VarRef,
};

/// Builder for constructing tensor IR expressions using `RecExpr`.
pub struct IrBuilder {
    pub expr: RecExpr<TensorIr>,
}

impl IrBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self {
            expr: RecExpr::default(),
        }
    }

    fn add(&mut self, node: TensorIr) -> Id {
        self.expr.add(node)
    }

    /// Construct a linked list of Ids from a slice.
    pub fn list(&mut self, items: &[Id]) -> Id {
        let mut curr = self.add(TensorIr::Nil);
        for &item in items.iter().rev() {
            curr = self.add(TensorIr::Cons([item, curr]));
        }
        curr
    }

    // ═══════════════════════════════════════════════════════
    // HIGH-LEVEL TENSOR BUILDERS
    // ═══════════════════════════════════════════════════════

    pub fn input(&mut self, id: u32, shape: Shape, dtype: DType) -> Id {
        self.add(TensorIr::HighLevel(HighLevelNode::Input {
            id,
            shape,
            dtype,
        }))
    }

    pub fn restride(&mut self, expr: Id, new_shape: Shape, strides: Strides) -> Id {
        self.restride_with_offset(expr, new_shape, strides, 0)
    }

    pub fn restride_with_offset(
        &mut self,
        expr: Id,
        new_shape: Shape,
        strides: Strides,
        offset: i64,
    ) -> Id {
        self.add(TensorIr::HighLevel(HighLevelNode::Restride {
            new_shape,
            strides,
            offset,
            expr,
        }))
    }

    /// # Panics
    ///
    /// Panics if `inputs.len()` does not fit in `u32`.
    pub fn elementwise(&mut self, index_space: Shape, inputs: &[Id], body: Id) -> Id {
        let num_inputs = u32::try_from(inputs.len()).expect("elementwise inputs must fit in u32");
        let mut children = inputs.to_vec();
        children.push(body);
        let children_list = self.list(&children);
        self.add(TensorIr::HighLevel(HighLevelNode::Elementwise {
            index_space,
            num_inputs,
            children_list,
        }))
    }

    pub fn slice_assign(
        &mut self,
        input: Id,
        value: Id,
        output_shape: Shape,
        slices: Vec<(u32, u32)>,
    ) -> Id {
        assert_eq!(
            output_shape.rank(),
            slices.len(),
            "slice_assign slices must match output rank"
        );

        let mut in_slice = self.scalar_lit(ScalarValue::Bool(true));
        let mut relative_indices = Vec::with_capacity(slices.len());
        let mut axis_in_slice = Vec::with_capacity(slices.len());
        for (axis, (start, end)) in slices.iter().copied().enumerate() {
            let index = self.scalar_index(axis as u32);
            let full_axis =
                matches!(output_shape.0[axis].as_const(), Some(dim) if start == 0 && end == dim);
            if full_axis {
                relative_indices.push(index);
                axis_in_slice.push(self.scalar_lit(ScalarValue::Bool(true)));
                continue;
            }

            let start_lit = self.scalar_u32(start);
            let end_lit = self.scalar_u32(end);
            let ge_start = self.bin_op(BinaryOp::Ge, index, start_lit);
            let lt_end = self.bin_op(BinaryOp::Lt, index, end_lit);
            let axis_in_range = self.bin_op(BinaryOp::And, ge_start, lt_end);
            in_slice = self.bin_op(BinaryOp::And, in_slice, axis_in_range);
            axis_in_slice.push(axis_in_range);

            let relative = if start == 0 {
                index
            } else {
                self.bin_op(BinaryOp::Sub, index, start_lit)
            };
            relative_indices.push(relative);
        }

        let zero = self.scalar_u32(0);
        let safe_indices = relative_indices
            .into_iter()
            .zip(axis_in_slice)
            .map(|(index, axis_in_range)| {
                if matches!(
                    self.expr.as_ref()[usize::from(axis_in_range)],
                    TensorIr::Const(ScalarValue::Bool(true))
                ) {
                    index
                } else {
                    self.tern_op(TernaryOp::Select, axis_in_range, index, zero)
                }
            })
            .collect::<Vec<_>>();
        let replacement = self.indexed_arg(1, &safe_indices);
        let original = self.scalar_arg(0);
        let body = self.tern_op(TernaryOp::Select, in_slice, replacement, original);
        self.elementwise(output_shape, &[input, value], body)
    }

    pub fn index_select(&mut self, input: Id, indices: Id, output_shape: Shape, axis: u32) -> Id {
        assert!(
            (axis as usize) < output_shape.rank(),
            "index_select axis must be in bounds"
        );
        let source_indices = (0..output_shape.rank())
            .map(|dim| {
                if dim == axis as usize {
                    let index = self.scalar_index(axis);
                    self.indexed_arg(1, &[index])
                } else {
                    self.scalar_index(dim as u32)
                }
            })
            .collect::<Vec<_>>();
        let body = self.indexed_arg(0, &source_indices);
        self.elementwise(output_shape, &[input, indices], body)
    }

    pub fn resize(&mut self, input: Id, input_shape: Shape, output_shape: Shape) -> Id {
        if input_shape.static_numel() == output_shape.static_numel() {
            let strides = Strides::row_major_for_shape(&output_shape);
            return self.restride_with_offset(input, output_shape, strides, 0);
        }

        assert_eq!(
            input_shape.rank(),
            output_shape.rank(),
            "size-changing resize must preserve rank"
        );

        let mut in_bounds = self.scalar_lit(ScalarValue::Bool(true));
        let mut safe_indices = Vec::with_capacity(output_shape.rank());
        for (axis, dim) in input_shape.0.iter().enumerate() {
            let Some(limit) = dim.as_const() else {
                panic!("resize currently requires literal input dimensions");
            };
            let index = self.scalar_index(axis as u32);
            let limit = self.scalar_u32(limit);
            let axis_in_bounds = self.bin_op(BinaryOp::Lt, index, limit);
            in_bounds = self.bin_op(BinaryOp::And, in_bounds, axis_in_bounds);
            let zero = self.scalar_u32(0);
            safe_indices.push(self.tern_op(TernaryOp::Select, axis_in_bounds, index, zero));
        }

        let value = self.indexed_arg(0, &safe_indices);
        let dtype = self.infer_expr_dtype(input, None).unwrap_or(DType::F32);
        let zero = self.zero_for_dtype(dtype);
        let body = self.tern_op(TernaryOp::Select, in_bounds, value, zero);
        self.elementwise(output_shape, &[input], body)
    }

    pub fn reduce(&mut self, expr: Id, axis: u32, op: ReduceOp) -> Id {
        self.add(TensorIr::HighLevel(HighLevelNode::Reduce {
            axis,
            op,
            expr,
        }))
    }

    // ═══════════════════════════════════════════════════════
    // SCALAR/SIMD EXPRESSION BUILDERS
    // ═══════════════════════════════════════════════════════

    pub fn scalar_lit(&mut self, v: ScalarValue) -> Id {
        self.add(TensorIr::Const(v))
    }

    pub fn low_lit(&mut self, v: ScalarValue) -> Id {
        self.scalar_lit(v)
    }

    pub fn scalar_f32(&mut self, v: f32) -> Id {
        self.scalar_lit(ScalarValue::F32(ordered_float::OrderedFloat(v)))
    }
    pub fn low_f32(&mut self, v: f32) -> Id {
        self.scalar_f32(v)
    }

    pub fn scalar_i32(&mut self, v: i32) -> Id {
        self.scalar_lit(ScalarValue::I32(v))
    }
    pub fn low_i32(&mut self, v: i32) -> Id {
        self.scalar_i32(v)
    }

    pub fn scalar_u32(&mut self, v: u32) -> Id {
        self.scalar_lit(ScalarValue::U32(v))
    }
    pub fn low_u32(&mut self, v: u32) -> Id {
        self.scalar_u32(v)
    }

    pub fn scalar_arg(&mut self, i: u32) -> Id {
        self.add(TensorIr::HighLevel(HighLevelNode::Param(i)))
    }

    pub fn scalar_index(&mut self, i: u32) -> Id {
        self.add(TensorIr::HighLevel(HighLevelNode::Index(i)))
    }

    pub fn indexed_arg(&mut self, i: u32, indices: &[Id]) -> Id {
        let children_list = self.list(indices);
        self.add(TensorIr::HighLevel(HighLevelNode::IndexedParam {
            index: i,
            children_list,
        }))
    }

    pub fn bin_op(&mut self, op: BinaryOp, lhs: Id, rhs: Id) -> Id {
        self.add(TensorIr::BinOp(op, [lhs, rhs]))
    }

    pub fn un_op(&mut self, op: UnaryOp, arg: Id) -> Id {
        self.add(TensorIr::UnOp(op, arg))
    }

    pub fn tern_op(&mut self, op: TernaryOp, a: Id, b: Id, c: Id) -> Id {
        self.add(TensorIr::TernOp(op, [a, b, c]))
    }

    fn zero_for_dtype(&mut self, dtype: DType) -> Id {
        let value = match dtype {
            DType::F16 => ScalarValue::F16(ordered_float::OrderedFloat(0.0)),
            DType::F32 => ScalarValue::F32(ordered_float::OrderedFloat(0.0)),
            DType::U32 => ScalarValue::U32(0),
            DType::I32 => ScalarValue::I32(0),
            DType::Bool => ScalarValue::Bool(false),
        };
        self.scalar_lit(value)
    }

    fn infer_expr_dtype(&self, id: Id, params: Option<&[DType]>) -> Option<DType> {
        fn unary_dtype(op: UnaryOp, input: Option<DType>) -> Option<DType> {
            match op {
                UnaryOp::CastF16 => Some(DType::F16),
                UnaryOp::CastF32 => Some(DType::F32),
                UnaryOp::CastI32 => Some(DType::I32),
                UnaryOp::CastU32 => Some(DType::U32),
                UnaryOp::CastBool | UnaryOp::Not => Some(DType::Bool),
                _ => input,
            }
        }

        let node = self.expr.as_ref().get(usize::from(id))?;
        match node {
            TensorIr::HighLevel(HighLevelNode::Input { dtype, .. }) => Some(*dtype),
            TensorIr::HighLevel(HighLevelNode::Restride { expr, .. })
            | TensorIr::HighLevel(HighLevelNode::Reduce { expr, .. }) => {
                self.infer_expr_dtype(*expr, params)
            }
            TensorIr::HighLevel(HighLevelNode::Elementwise {
                children_list,
                num_inputs,
                ..
            }) => {
                let children =
                    crate::language::extract_recexpr_list(self.expr.as_ref(), *children_list);
                let input_dtypes = children[..(*num_inputs as usize).min(children.len())]
                    .iter()
                    .map(|input| self.infer_expr_dtype(*input, params))
                    .collect::<Option<Vec<_>>>()?;
                children
                    .last()
                    .and_then(|body| self.infer_expr_dtype(*body, Some(&input_dtypes)))
            }
            TensorIr::BinOp(op, children) => match op {
                BinaryOp::Lt
                | BinaryOp::Le
                | BinaryOp::Gt
                | BinaryOp::Ge
                | BinaryOp::Eq
                | BinaryOp::Neq => Some(DType::Bool),
                _ => self.infer_expr_dtype(children[0], params),
            },
            TensorIr::UnOp(op, child) => unary_dtype(*op, self.infer_expr_dtype(*child, params)),
            TensorIr::TernOp(op, children) => match op {
                TernaryOp::Fma => self.infer_expr_dtype(children[0], params),
                TernaryOp::Select => self.infer_expr_dtype(children[1], params),
            },
            TensorIr::Const(value) => Some(match value {
                ScalarValue::F16(_) => DType::F16,
                ScalarValue::F32(_) => DType::F32,
                ScalarValue::I32(_) => DType::I32,
                ScalarValue::U32(_) => DType::U32,
                ScalarValue::Bool(_) => DType::Bool,
            }),
            TensorIr::ShapeParam(_) => Some(DType::U32),
            TensorIr::HighLevel(HighLevelNode::Param(index))
            | TensorIr::HighLevel(HighLevelNode::IndexedParam { index, .. }) => {
                params.and_then(|params| params.get(*index as usize).copied())
            }
            TensorIr::HighLevel(HighLevelNode::Index(_)) => Some(DType::U32),
            TensorIr::Dispatch(_)
            | TensorIr::Effect(_)
            | TensorIr::Simd(_)
            | TensorIr::Nil
            | TensorIr::Cons(_) => None,
        }
    }

    // ═══════════════════════════════════════════════════════
    // LOW-LEVEL BUILDERS
    // ═══════════════════════════════════════════════════════

    /// Add a typed `Var` reference (Bound, BlockedAcc, etc.).
    pub fn var(&mut self, var: VarRef) -> Id {
        self.add(TensorIr::Simd(SimdNode::Var(var)))
    }

    pub fn index(&mut self, level: IndexLevel) -> Id {
        self.add(TensorIr::Simd(SimdNode::Var(VarRef::thread(level))))
    }

    pub fn token(&mut self) -> Id {
        self.add(TensorIr::Dispatch(DispatchNode::Token))
    }

    pub fn pack(&mut self, values: &[Id]) -> Id {
        let children_list = self.list(values);
        self.add(TensorIr::Dispatch(DispatchNode::Pack { children_list }))
    }

    pub fn extract(&mut self, tuple: Id, index: u32) -> Id {
        self.add(TensorIr::Dispatch(DispatchNode::Extract { index, tuple }))
    }

    pub fn load_at(&mut self, tier: MemTier, addr: Id, state: Id) -> Id {
        self.add(TensorIr::Simd(SimdNode::Load {
            tier,
            children: [addr, state],
        }))
    }

    pub fn load(&mut self, tier: MemTier, addr: Id) -> Id {
        let state = self.token();
        self.load_at(tier, addr, state)
    }

    pub fn shuffle(&mut self, src: Id, lane: Id) -> Id {
        self.add(TensorIr::Simd(SimdNode::Shuffle([src, lane])))
    }

    pub fn reduce_simd(&mut self, src: Id, op: ReduceOp) -> Id {
        self.add(TensorIr::Simd(SimdNode::ReduceSimd { op, src }))
    }

    pub fn theta(&mut self, init: Id, count: Id, update: Id) -> Id {
        self.add(TensorIr::Simd(SimdNode::Theta {
            children: [init, count, update],
        }))
    }

    /// # Panics
    ///
    /// Panics if `inputs.len()` does not fit in `u32`.
    pub fn dispatch(
        &mut self,
        inputs: &[Id],
        workgroups: impl Into<crate::types::Dim>,
        body: Id,
    ) -> Id {
        let num_inputs = u32::try_from(inputs.len()).expect("dispatch inputs must fit in u32");
        let mut children = inputs.to_vec();
        children.push(body);
        let children_list = self.list(&children);
        self.add(TensorIr::Dispatch(DispatchNode::Dispatch {
            workgroups: workgroups.into(),
            num_inputs,
            children_list,
        }))
    }

    pub fn effect_token(&mut self) -> Id {
        self.add(TensorIr::Effect(EffectNode::Token))
    }

    pub fn effect_store(&mut self, tier: MemTier, addr: Id, value: Id, state: Id) -> Id {
        self.add(TensorIr::Effect(EffectNode::Store {
            tier,
            children: [addr, value, state],
        }))
    }

    pub fn effect_store_if(
        &mut self,
        tier: MemTier,
        cond: Id,
        addr: Id,
        value: Id,
        state: Id,
    ) -> Id {
        self.add(TensorIr::Effect(EffectNode::StoreIf {
            tier,
            children: [cond, addr, value, state],
        }))
    }

    pub fn effect_barrier(&mut self, regions: Vec<crate::types::BufferRef>, state: Id) -> Id {
        self.add(TensorIr::Effect(EffectNode::Barrier { regions, state }))
    }

    pub fn effect_dispatch(
        &mut self,
        workgroups: impl Into<crate::types::Dim>,
        simdgroups: u32,
        state: Id,
        body: Id,
    ) -> Id {
        self.add(TensorIr::Effect(EffectNode::Dispatch {
            workgroups: workgroups.into(),
            simdgroups,
            children: [state, body],
        }))
    }

    pub fn effect_seq(&mut self, steps: &[Id]) -> Id {
        let list = self.list(steps);
        self.add(TensorIr::Effect(EffectNode::Seq(list)))
    }

    pub fn effect_program(&mut self, buffers: &[Id], body: Id, outputs: &[Id]) -> Id {
        let buffers = self.list(buffers);
        let outputs = self.list(outputs);
        self.add(TensorIr::Effect(EffectNode::Program {
            children: [buffers, body, outputs],
        }))
    }

    pub fn seq(&mut self, dispatches: &[Id]) -> Id {
        let list = self.list(dispatches);
        self.add(TensorIr::Dispatch(DispatchNode::Seq(list)))
    }

    pub fn pipeline(&mut self, dispatches: &[Id]) -> Id {
        let list = self.list(dispatches);
        self.add(TensorIr::Dispatch(DispatchNode::Pipeline(list)))
    }

    // ═══════════════════════════════════════════════════════
    // COMPOSITE HIGH-LEVEL PATTERNS
    // ═══════════════════════════════════════════════════════

    /// Build an arbitrary literal-rank contraction from restrided inputs, a
    /// scalar body, and a set of reductions expressed in the original
    /// `index_space`.
    ///
    /// `inputs` supplies `(expr, strides)` pairs in scalar-argument order. The
    /// `body` should reference them via `scalar_arg(0..inputs.len())`.
    ///
    /// Reductions are applied in descending-axis order so callers can specify
    /// axis ids against the original `index_space` without manually adjusting
    /// for earlier axis removal.
    pub fn contraction(
        &mut self,
        index_space: Shape,
        inputs: &[(Id, Strides)],
        body: Id,
        reductions: &[(u32, ReduceOp)],
    ) -> Id {
        let restrided_inputs: Vec<Id> = inputs
            .iter()
            .map(|(expr, strides)| self.restride(*expr, index_space.clone(), strides.clone()))
            .collect();
        let mut expr = self.elementwise(index_space, &restrided_inputs, body);

        let mut ordered_reductions = reductions.to_vec();
        ordered_reductions.sort_unstable_by_key(|reduction| Reverse(reduction.0));
        for (axis, op) in ordered_reductions {
            expr = self.reduce(expr, axis, op);
        }

        expr
    }

    pub fn softmax(&mut self, x: Id, shape: Shape, axis: u32) -> Id {
        let max_val = self.reduce(x, axis, ReduceOp::Max);

        let arg0 = self.scalar_arg(0);
        let arg1 = self.scalar_arg(1);
        let sub_body = self.bin_op(BinaryOp::Sub, arg0, arg1);
        let reduced_shape = shape.remove_axis(axis as usize);
        let mut broadcast_strides = Strides::row_major_for_shape(&reduced_shape).0;
        broadcast_strides.insert(axis as usize, crate::types::Dim::Const(0));
        let max_broadcast =
            self.restride(max_val, shape.clone(), Strides(broadcast_strides.clone()));
        let shifted = self.elementwise(shape.clone(), &[x, max_broadcast], sub_body);

        let arg0 = self.scalar_arg(0);
        let exp_body = self.un_op(UnaryOp::Exp, arg0);
        let exp_val = self.elementwise(shape.clone(), &[shifted], exp_body);

        let sum_val = self.reduce(exp_val, axis, ReduceOp::Add);

        let arg0 = self.scalar_arg(0);
        let arg1 = self.scalar_arg(1);
        let div_body = self.bin_op(BinaryOp::Div, arg0, arg1);
        let sum_broadcast = self.restride(sum_val, shape.clone(), Strides(broadcast_strides));
        self.elementwise(shape, &[exp_val, sum_broadcast], div_body)
    }
}

impl Default for IrBuilder {
    fn default() -> Self {
        Self::new()
    }
}
