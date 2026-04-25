use std::{cmp::Reverse, collections::HashMap};

use crate::types::{
    BinaryOp, DType, Dim, ReduceOp, ScalarValue, Shape, Strides, TernaryOp, UnaryOp,
};

/// Stable identifier for a node in a `TensorExprProgram`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExprId(pub usize);

impl From<ExprId> for usize {
    fn from(value: ExprId) -> Self {
        value.0
    }
}

/// Pure tensor-expression IR.
///
/// This stage is intentionally free of dispatch, token, store, barrier,
/// and SIMD execution concepts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorExprNode {
    Input {
        id: u32,
        shape: Shape,
        dtype: DType,
    },
    Restride {
        expr: ExprId,
        new_shape: Shape,
        strides: Strides,
        offset: i64,
    },
    Elementwise {
        index_space: Shape,
        inputs: Vec<ExprId>,
        body: ExprId,
    },
    SliceAssign {
        input: ExprId,
        value: ExprId,
        output_shape: Shape,
        slices: Vec<(u32, u32)>,
    },
    IndexSelect {
        input: ExprId,
        indices: ExprId,
        output_shape: Shape,
        axis: u32,
    },
    Resize {
        input: ExprId,
        input_shape: Shape,
        output_shape: Shape,
    },
    Reduce {
        expr: ExprId,
        axis: u32,
        op: ReduceOp,
    },
    UnOp(UnaryOp, ExprId),
    BinOp(BinaryOp, [ExprId; 2]),
    TernOp(TernaryOp, [ExprId; 3]),
    Const(ScalarValue),
    Arg(u32),
    Index(u32),
    IndexedArg {
        index: u32,
        indices: Vec<ExprId>,
    },
}

impl TensorExprNode {
    #[must_use]
    pub fn children(&self) -> Vec<ExprId> {
        match self {
            Self::Input { .. } | Self::Const(_) | Self::Arg(_) | Self::Index(_) => Vec::new(),
            Self::IndexedArg { indices, .. } => indices.clone(),
            Self::Restride { expr, .. } | Self::UnOp(_, expr) | Self::Reduce { expr, .. } => {
                vec![*expr]
            }
            Self::Elementwise { inputs, body, .. } => {
                let mut children = inputs.clone();
                children.push(*body);
                children
            }
            Self::SliceAssign { input, value, .. } => vec![*input, *value],
            Self::IndexSelect { input, indices, .. } => vec![*input, *indices],
            Self::Resize { input, .. } => vec![*input],
            Self::BinOp(_, children) => children.to_vec(),
            Self::TernOp(_, children) => children.to_vec(),
        }
    }
}

/// A closed tensor expression DAG with an explicit root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorExprProgram {
    nodes: Vec<TensorExprNode>,
    root: ExprId,
}

impl TensorExprProgram {
    #[must_use]
    pub const fn new(nodes: Vec<TensorExprNode>, root: ExprId) -> Self {
        Self { nodes, root }
    }

    /// # Errors
    ///
    /// Returns an error if the tensor expression is empty, contains invalid child references,
    /// or uses an out-of-bounds reduction axis.
    pub fn validate(&self) -> Result<(), String> {
        if self.nodes.is_empty() {
            return Err("tensor expression cannot be empty".into());
        }
        if self.root.0 >= self.nodes.len() {
            return Err(format!(
                "tensor expression root {} is out of bounds for {} nodes",
                self.root.0,
                self.nodes.len()
            ));
        }

        for (idx, node) in self.nodes.iter().enumerate() {
            for child in node.children() {
                if child.0 >= self.nodes.len() {
                    return Err(format!(
                        "node {idx} references child {} outside the program",
                        child.0
                    ));
                }
                if child.0 >= idx {
                    return Err(format!(
                        "node {idx} references child {} that is not topologically earlier",
                        child.0
                    ));
                }
            }

            if let TensorExprNode::Reduce { expr, axis, .. } = node {
                let Some(shape) = self.infer_node_shape(*expr) else {
                    return Err(format!(
                        "reduce node {idx} does not have an inferable input shape"
                    ));
                };
                if (*axis as usize) >= shape.rank() {
                    return Err(format!(
                        "reduce node {idx} uses axis {axis} on rank-{} input",
                        shape.rank()
                    ));
                }
            }

            if let TensorExprNode::SliceAssign {
                value,
                output_shape,
                slices,
                ..
            } = node
            {
                if slices.len() != output_shape.rank() {
                    return Err(format!(
                        "slice assign node {idx} has {} slices for rank-{} output",
                        slices.len(),
                        output_shape.rank()
                    ));
                }
                for (axis, ((start, end), dim)) in
                    slices.iter().zip(output_shape.0.iter()).enumerate()
                {
                    let Dim::Lit(dim) = dim else {
                        continue;
                    };
                    if start > end || end > dim {
                        return Err(format!(
                            "slice assign node {idx} has invalid slice {start}..{end} on axis {axis} with dim {dim}"
                        ));
                    }
                }
                if let Some(value_shape) = self.infer_node_shape(*value) {
                    if value_shape.rank() != slices.len() {
                        return Err(format!(
                            "slice assign node {idx} value rank {} does not match slice rank {}",
                            value_shape.rank(),
                            slices.len()
                        ));
                    }
                    for (axis, ((start, end), dim)) in
                        slices.iter().zip(value_shape.0.iter()).enumerate()
                    {
                        let Dim::Lit(dim) = dim else {
                            continue;
                        };
                        let expected = end - start;
                        if *dim != expected {
                            return Err(format!(
                                "slice assign node {idx} value axis {axis} has dim {dim}, expected {expected}"
                            ));
                        }
                    }
                }
            }

            if let TensorExprNode::Resize {
                input,
                input_shape,
                output_shape,
            } = node
            {
                if input_shape.static_numel() != output_shape.static_numel()
                    && input_shape.rank() != output_shape.rank()
                {
                    return Err(format!(
                        "size-changing resize node {idx} changes rank from {} to {}",
                        input_shape.rank(),
                        output_shape.rank()
                    ));
                }
                if let Some(inferred) = self.infer_node_shape(*input)
                    && inferred != *input_shape
                {
                    return Err(format!(
                        "resize node {idx} input shape {:?} does not match declared {:?}",
                        inferred, input_shape
                    ));
                }
            }

            if let TensorExprNode::IndexSelect {
                input,
                indices,
                output_shape,
                axis,
            } = node
            {
                let axis = *axis as usize;
                if axis >= output_shape.rank() {
                    return Err(format!(
                        "index select node {idx} uses axis {axis} on rank-{} output",
                        output_shape.rank()
                    ));
                }
                if let Some(input_shape) = self.infer_node_shape(*input) {
                    if input_shape.rank() != output_shape.rank() {
                        return Err(format!(
                            "index select node {idx} input rank {} does not match output rank {}",
                            input_shape.rank(),
                            output_shape.rank()
                        ));
                    }
                    for dim in 0..output_shape.rank() {
                        if dim != axis && input_shape.0[dim] != output_shape.0[dim] {
                            return Err(format!(
                                "index select node {idx} output dim {dim} {:?} does not match input dim {:?}",
                                output_shape.0[dim], input_shape.0[dim]
                            ));
                        }
                    }
                }
                if let Some(indices_shape) = self.infer_node_shape(*indices) {
                    if indices_shape.rank() != 1 {
                        return Err(format!(
                            "index select node {idx} index tensor must be rank 1, got rank {}",
                            indices_shape.rank()
                        ));
                    }
                    if indices_shape.0[0] != output_shape.0[axis] {
                        return Err(format!(
                            "index select node {idx} output axis {axis} {:?} does not match index len {:?}",
                            output_shape.0[axis], indices_shape.0[0]
                        ));
                    }
                }
            }
        }

        Ok(())
    }

    #[must_use]
    pub fn nodes(&self) -> &[TensorExprNode] {
        &self.nodes
    }

    #[must_use]
    pub const fn root(&self) -> ExprId {
        self.root
    }

    #[must_use]
    pub fn node(&self, id: ExprId) -> &TensorExprNode {
        &self.nodes[id.0]
    }

    /// # Errors
    ///
    /// Returns an error if the tensor expression fails validation.
    pub fn summary(&self) -> Result<TensorExprSummary, String> {
        self.validate()?;

        let input_count = self
            .nodes
            .iter()
            .filter(|node| matches!(node, TensorExprNode::Input { .. }))
            .count();

        let output_shape = self.infer_node_shape(self.root);
        let dtype = self.infer_node_dtype(self.root);
        let has_reduce = self
            .nodes
            .iter()
            .any(|node| matches!(node, TensorExprNode::Reduce { .. }));
        let has_elementwise = self
            .nodes
            .iter()
            .any(|node| matches!(node, TensorExprNode::Elementwise { .. }));

        Ok(TensorExprSummary {
            input_count,
            output_shape,
            dtype,
            has_reduce,
            has_elementwise,
        })
    }

    fn infer_node_shape(&self, id: ExprId) -> Option<Shape> {
        fn infer(
            program: &TensorExprProgram,
            id: ExprId,
            cache: &mut HashMap<ExprId, Option<Shape>>,
        ) -> Option<Shape> {
            if let Some(shape) = cache.get(&id) {
                return shape.clone();
            }

            let shape = match program.node(id) {
                TensorExprNode::Input { shape, .. } => Some(shape.clone()),
                TensorExprNode::Restride { new_shape, .. } => Some(new_shape.clone()),
                TensorExprNode::Elementwise { index_space, .. } => Some(index_space.clone()),
                TensorExprNode::SliceAssign { output_shape, .. } => Some(output_shape.clone()),
                TensorExprNode::IndexSelect { output_shape, .. } => Some(output_shape.clone()),
                TensorExprNode::Resize { output_shape, .. } => Some(output_shape.clone()),
                TensorExprNode::Reduce { expr, axis, .. } => {
                    infer(program, *expr, cache).map(|shape| shape.remove_axis(*axis as usize))
                }
                TensorExprNode::BinOp(..)
                | TensorExprNode::UnOp(..)
                | TensorExprNode::TernOp(..)
                | TensorExprNode::Const(_)
                | TensorExprNode::Arg(_)
                | TensorExprNode::Index(_)
                | TensorExprNode::IndexedArg { .. } => None,
            };

            cache.insert(id, shape.clone());
            shape
        }

        let mut cache = HashMap::new();
        infer(self, id, &mut cache)
    }

    fn infer_node_dtype(&self, id: ExprId) -> Option<DType> {
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

        fn infer(
            program: &TensorExprProgram,
            id: ExprId,
            cache: &mut HashMap<ExprId, Option<DType>>,
            params: Option<&[DType]>,
        ) -> Option<DType> {
            if params.is_none()
                && let Some(dtype) = cache.get(&id)
            {
                return *dtype;
            }

            let dtype = match program.node(id) {
                TensorExprNode::Input { dtype, .. } => Some(*dtype),
                TensorExprNode::Restride { expr, .. } | TensorExprNode::Reduce { expr, .. } => {
                    infer(program, *expr, cache, params)
                }
                TensorExprNode::SliceAssign { input, .. } => infer(program, *input, cache, params),
                TensorExprNode::IndexSelect { input, .. } => infer(program, *input, cache, params),
                TensorExprNode::Resize { input, .. } => infer(program, *input, cache, params),
                TensorExprNode::Elementwise { inputs, body, .. } => {
                    let input_dtypes = inputs
                        .iter()
                        .map(|input| infer(program, *input, cache, params))
                        .collect::<Option<Vec<_>>>();
                    input_dtypes
                        .as_deref()
                        .and_then(|input_dtypes| infer(program, *body, cache, Some(input_dtypes)))
                }
                TensorExprNode::BinOp(op, children) => match op {
                    BinaryOp::Lt
                    | BinaryOp::Le
                    | BinaryOp::Gt
                    | BinaryOp::Ge
                    | BinaryOp::Eq
                    | BinaryOp::Neq => Some(DType::Bool),
                    _ => infer(program, children[0], cache, params),
                },
                TensorExprNode::UnOp(op, child) => {
                    unary_dtype(*op, infer(program, *child, cache, params))
                }
                TensorExprNode::TernOp(op, children) => match op {
                    TernaryOp::Fma => infer(program, children[0], cache, params),
                    TernaryOp::Select => infer(program, children[1], cache, params),
                },
                TensorExprNode::Arg(index) => {
                    params.and_then(|params| params.get(*index as usize).copied())
                }
                TensorExprNode::Index(_) => Some(DType::U32),
                TensorExprNode::IndexedArg { index, .. } => {
                    params.and_then(|params| params.get(*index as usize).copied())
                }
                TensorExprNode::Const(value) => Some(match value {
                    ScalarValue::F16(_) => DType::F16,
                    ScalarValue::F32(_) => DType::F32,
                    ScalarValue::I32(_) => DType::I32,
                    ScalarValue::U32(_) => DType::U32,
                    ScalarValue::Bool(_) => DType::Bool,
                }),
            };

            if params.is_none() {
                cache.insert(id, dtype);
            }
            dtype
        }

        let mut cache = HashMap::new();
        infer(self, id, &mut cache, None)
    }
}

/// Builder for the pure tensor-expression stage.
#[derive(Debug, Default)]
pub struct TensorExprBuilder {
    nodes: Vec<TensorExprNode>,
}

impl TensorExprBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn add(&mut self, node: TensorExprNode) -> ExprId {
        let id = ExprId(self.nodes.len());
        self.nodes.push(node);
        id
    }

    pub fn input(&mut self, id: u32, shape: Shape, dtype: DType) -> ExprId {
        self.add(TensorExprNode::Input { id, shape, dtype })
    }

    pub fn restride(&mut self, expr: ExprId, new_shape: Shape, strides: Strides) -> ExprId {
        self.restride_with_offset(expr, new_shape, strides, 0)
    }

    pub fn restride_with_offset(
        &mut self,
        expr: ExprId,
        new_shape: Shape,
        strides: Strides,
        offset: i64,
    ) -> ExprId {
        self.add(TensorExprNode::Restride {
            expr,
            new_shape,
            strides,
            offset,
        })
    }

    pub fn elementwise(&mut self, index_space: Shape, inputs: &[ExprId], body: ExprId) -> ExprId {
        self.add(TensorExprNode::Elementwise {
            index_space,
            inputs: inputs.to_vec(),
            body,
        })
    }

    pub fn slice_assign(
        &mut self,
        input: ExprId,
        value: ExprId,
        output_shape: Shape,
        slices: Vec<(u32, u32)>,
    ) -> ExprId {
        self.add(TensorExprNode::SliceAssign {
            input,
            value,
            output_shape,
            slices,
        })
    }

    pub fn index_select(
        &mut self,
        input: ExprId,
        indices: ExprId,
        output_shape: Shape,
        axis: u32,
    ) -> ExprId {
        self.add(TensorExprNode::IndexSelect {
            input,
            indices,
            output_shape,
            axis,
        })
    }

    pub fn resize(&mut self, input: ExprId, input_shape: Shape, output_shape: Shape) -> ExprId {
        self.add(TensorExprNode::Resize {
            input,
            input_shape,
            output_shape,
        })
    }

    pub fn reduce(&mut self, expr: ExprId, axis: u32, op: ReduceOp) -> ExprId {
        self.add(TensorExprNode::Reduce { expr, axis, op })
    }

    pub fn scalar_lit(&mut self, value: ScalarValue) -> ExprId {
        self.add(TensorExprNode::Const(value))
    }

    pub fn scalar_f32(&mut self, value: f32) -> ExprId {
        self.scalar_lit(ScalarValue::F32(ordered_float::OrderedFloat(value)))
    }

    pub fn scalar_i32(&mut self, value: i32) -> ExprId {
        self.scalar_lit(ScalarValue::I32(value))
    }

    pub fn scalar_u32(&mut self, value: u32) -> ExprId {
        self.scalar_lit(ScalarValue::U32(value))
    }

    pub fn scalar_arg(&mut self, index: u32) -> ExprId {
        self.add(TensorExprNode::Arg(index))
    }

    pub fn scalar_index(&mut self, index: u32) -> ExprId {
        self.add(TensorExprNode::Index(index))
    }

    pub fn indexed_arg(&mut self, index: u32, indices: Vec<ExprId>) -> ExprId {
        self.add(TensorExprNode::IndexedArg { index, indices })
    }

    pub fn scalar_binop(&mut self, op: BinaryOp, args: [ExprId; 2]) -> ExprId {
        self.add(TensorExprNode::BinOp(op, args))
    }

    pub fn scalar_unop(&mut self, op: UnaryOp, arg: ExprId) -> ExprId {
        self.add(TensorExprNode::UnOp(op, arg))
    }

    pub fn scalar_ternop(&mut self, op: TernaryOp, args: [ExprId; 3]) -> ExprId {
        self.add(TensorExprNode::TernOp(op, args))
    }

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
        inputs: &[(ExprId, Strides)],
        body: ExprId,
        reductions: &[(u32, ReduceOp)],
    ) -> ExprId {
        let restrided_inputs: Vec<ExprId> = inputs
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

    pub fn softmax(&mut self, x: ExprId, shape: Shape, axis: u32) -> ExprId {
        let max_value = self.reduce(x, axis, ReduceOp::Max);

        let arg0 = self.scalar_arg(0);
        let arg1 = self.scalar_arg(1);
        let sub_body = self.scalar_binop(BinaryOp::Sub, [arg0, arg1]);
        let reduced_shape = shape.remove_axis(axis as usize);
        let mut broadcast_strides = Strides::row_major_for_shape(&reduced_shape)
            .map(|strides| strides.0)
            .unwrap_or_else(|| vec![1i64; reduced_shape.rank()]);
        broadcast_strides.insert(axis as usize, 0);
        let max_broadcast =
            self.restride(max_value, shape.clone(), Strides(broadcast_strides.clone()));
        let shifted = self.elementwise(shape.clone(), &[x, max_broadcast], sub_body);

        let arg0 = self.scalar_arg(0);
        let exp_body = self.scalar_unop(UnaryOp::Exp, arg0);
        let exp_value = self.elementwise(shape.clone(), &[shifted], exp_body);

        let sum_value = self.reduce(exp_value, axis, ReduceOp::Add);

        let arg0 = self.scalar_arg(0);
        let arg1 = self.scalar_arg(1);
        let div_body = self.scalar_binop(BinaryOp::Div, [arg0, arg1]);
        let sum_broadcast = self.restride(sum_value, shape.clone(), Strides(broadcast_strides));
        self.elementwise(shape, &[exp_value, sum_broadcast], div_body)
    }

    /// # Errors
    ///
    /// Returns an error if the built tensor expression fails validation.
    pub fn build(self, root: ExprId) -> Result<TensorExprProgram, String> {
        let program = TensorExprProgram::new(self.nodes, root);
        program.validate()?;
        Ok(program)
    }
}

/// Summary facts for the pure tensor-expression stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorExprSummary {
    pub input_count: usize,
    pub output_shape: Option<Shape>,
    pub dtype: Option<DType>,
    pub has_reduce: bool,
    pub has_elementwise: bool,
}

/// Named phases of optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Lower high-level tensor ops to naive per-lane Dispatches.
    Lowering,
    /// Unified late-phase dispatch-shape saturation. Runs tiling, TG
    /// promotion, cooperative split, reduce-simd, shuffle-tree,
    /// register-blocking, and dispatch fusion together so the extractor's
    /// shape-aware cost model can compare all dispatch variants in one
    /// e-graph rather than piece-by-piece across earlier phases.
    LateDispatch,
    /// State-threading rewrite now owned by the dispatch lowering layer.
    StateThreading,
}

impl Phase {
    /// All phases in recommended execution order.
    #[must_use]
    pub const fn all() -> &'static [Self] {
        &[Self::Lowering, Self::LateDispatch, Self::StateThreading]
    }
}
