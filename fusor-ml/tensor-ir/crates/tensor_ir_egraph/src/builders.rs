use std::cmp::Reverse;

use egg::{Id, RecExpr};

use crate::language::{DispatchNode, HighLevelNode, SimdNode, TensorIr};
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
        self.add(TensorIr::HighLevel(HighLevelNode::SliceAssign {
            output_shape,
            slices,
            children: [input, value],
        }))
    }

    pub fn index_select(&mut self, input: Id, indices: Id, output_shape: Shape, axis: u32) -> Id {
        self.add(TensorIr::HighLevel(HighLevelNode::IndexSelect {
            output_shape,
            axis,
            children: [input, indices],
        }))
    }

    pub fn resize(&mut self, input: Id, input_shape: Shape, output_shape: Shape) -> Id {
        self.add(TensorIr::HighLevel(HighLevelNode::Resize {
            input_shape,
            output_shape,
            expr: input,
        }))
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
    pub fn dispatch(&mut self, inputs: &[Id], workgroups: u32, body: Id) -> Id {
        let num_inputs = u32::try_from(inputs.len()).expect("dispatch inputs must fit in u32");
        let mut children = inputs.to_vec();
        children.push(body);
        let children_list = self.list(&children);
        self.add(TensorIr::Dispatch(DispatchNode::Dispatch {
            workgroups,
            num_inputs,
            children_list,
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
        let mut broadcast_strides = Strides::row_major_for_shape(&reduced_shape)
            .map(|strides| strides.0)
            .unwrap_or_else(|| vec![1i64; reduced_shape.rank()]);
        broadcast_strides.insert(axis as usize, 0);
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
