use std::{cmp::Reverse, collections::HashMap};

use crate::types::{BinaryOp, DType, ReduceOp, ScalarValue, Shape, Strides, TernaryOp, UnaryOp};

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
    Reduce {
        expr: ExprId,
        axis: u32,
        op: ReduceOp,
    },
    BinOp(BinaryOp, [ExprId; 2]),
    UnOp(UnaryOp, ExprId),
    TernOp(TernaryOp, [ExprId; 3]),
    Const(ScalarValue),
    Arg(u32),
}

impl TensorExprNode {
    #[must_use]
    pub fn children(&self) -> Vec<ExprId> {
        match self {
            Self::Input { .. } | Self::Const(_) | Self::Arg(_) => Vec::new(),
            Self::Restride { expr, .. } | Self::UnOp(_, expr) | Self::Reduce { expr, .. } => {
                vec![*expr]
            }
            Self::Elementwise { inputs, body, .. } => {
                let mut children = inputs.clone();
                children.push(*body);
                children
            }
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
                TensorExprNode::Reduce { expr, axis, .. } => {
                    infer(program, *expr, cache).map(|shape| shape.remove_axis(*axis as usize))
                }
                TensorExprNode::BinOp(..)
                | TensorExprNode::UnOp(..)
                | TensorExprNode::TernOp(..)
                | TensorExprNode::Const(_)
                | TensorExprNode::Arg(_) => None,
            };

            cache.insert(id, shape.clone());
            shape
        }

        let mut cache = HashMap::new();
        infer(self, id, &mut cache)
    }

    fn infer_node_dtype(&self, id: ExprId) -> Option<DType> {
        fn infer(
            program: &TensorExprProgram,
            id: ExprId,
            cache: &mut HashMap<ExprId, Option<DType>>,
        ) -> Option<DType> {
            if let Some(dtype) = cache.get(&id) {
                return *dtype;
            }

            let dtype = match program.node(id) {
                TensorExprNode::Input { dtype, .. } => Some(*dtype),
                TensorExprNode::Restride { expr, .. } | TensorExprNode::Reduce { expr, .. } => {
                    infer(program, *expr, cache)
                }
                TensorExprNode::Elementwise { inputs, body, .. } => inputs
                    .first()
                    .and_then(|input| infer(program, *input, cache))
                    .or_else(|| infer(program, *body, cache)),
                TensorExprNode::BinOp(..)
                | TensorExprNode::UnOp(..)
                | TensorExprNode::TernOp(..)
                | TensorExprNode::Arg(_) => None,
                TensorExprNode::Const(value) => Some(match value {
                    ScalarValue::F32(_) => DType::F32,
                    ScalarValue::I32(_) => DType::I32,
                    ScalarValue::U32(_) => DType::U32,
                    ScalarValue::Bool(_) => DType::Bool,
                }),
            };

            cache.insert(id, dtype);
            dtype
        }

        let mut cache = HashMap::new();
        infer(self, id, &mut cache)
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
        let mut broadcast_strides = vec![1i64; shape.rank()];
        broadcast_strides[axis as usize] = 0;
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
