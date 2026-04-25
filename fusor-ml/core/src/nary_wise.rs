use crate::{
    compute_graph::{ComputeGraphInner, NodeIndex},
    mir::{inputs::MirValue, operation::Operation},
    tensor::{DataTypeEnum, TensorData},
    visit_tiled::MaybeQData,
};
use tensor_ir::{BinaryOp, UnaryOp};

#[derive(Clone, Debug)]
pub(crate) enum NaryFunctionKind {
    Unary(UnaryOp),
    Binary(BinaryOp),
    Select {
        condition_type: DataTypeEnum,
    },
    BinaryConst {
        op: BinaryOp,
        constant: String,
        input_first: bool,
    },
    CompareConst {
        op: BinaryOp,
        constant: String,
    },
    Cast(DataTypeEnum),
    Unsupported,
}

#[derive(Clone, Debug)]
pub(crate) struct NaryFunction {
    pub(crate) name: Option<String>,
    pub(crate) kind: NaryFunctionKind,
    pub(crate) output_type: DataTypeEnum,
}

impl NaryFunction {
    pub fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("op")
    }

    pub fn unary(
        name: Option<String>,
        op: UnaryOp,
        _input_type: DataTypeEnum,
        output_type: DataTypeEnum,
    ) -> Self {
        Self {
            name,
            kind: NaryFunctionKind::Unary(op),
            output_type,
        }
    }

    pub fn binary(
        name: Option<String>,
        op: BinaryOp,
        _input_a_type: DataTypeEnum,
        _input_b_type: DataTypeEnum,
        output_type: DataTypeEnum,
    ) -> Self {
        Self {
            name,
            kind: NaryFunctionKind::Binary(op),
            output_type,
        }
    }

    pub fn select(condition_type: DataTypeEnum, output_type: DataTypeEnum) -> Self {
        Self {
            name: Some("select".to_string()),
            kind: NaryFunctionKind::Select { condition_type },
            output_type,
        }
    }

    pub fn binary_const(
        name: Option<String>,
        op: BinaryOp,
        constant: impl ToString,
        input_first: bool,
        _input_type: DataTypeEnum,
        output_type: DataTypeEnum,
    ) -> Self {
        Self {
            name,
            kind: NaryFunctionKind::BinaryConst {
                op,
                constant: constant.to_string(),
                input_first,
            },
            output_type,
        }
    }

    pub fn compare_const(
        name: Option<String>,
        op: BinaryOp,
        constant: impl ToString,
        _input_type: DataTypeEnum,
        output_type: DataTypeEnum,
    ) -> Self {
        Self {
            name,
            kind: NaryFunctionKind::CompareConst {
                op,
                constant: constant.to_string(),
            },
            output_type,
        }
    }

    pub fn cast(_input_type: DataTypeEnum, output_type: DataTypeEnum) -> Self {
        Self {
            name: Some("cast".to_string()),
            kind: NaryFunctionKind::Cast(output_type),
            output_type,
        }
    }

    pub fn unsupported_unary(
        name: Option<String>,
        _input_type: DataTypeEnum,
        output_type: DataTypeEnum,
    ) -> Self {
        Self {
            name,
            kind: NaryFunctionKind::Unsupported,
            output_type,
        }
    }
}

/// A chain of unary functions used for pre/post processing in reduce/matmul/dequantize.
/// Each function takes a single input and produces a single output; the chain is applied sequentially.
#[derive(Clone, Debug)]
pub(crate) struct UnaryFunctionChain {
    input_datatype: DataTypeEnum,
    pub(crate) functions: Vec<NaryFunction>,
}

impl UnaryFunctionChain {
    pub fn empty(input_datatype: DataTypeEnum) -> Self {
        Self {
            input_datatype,
            functions: Vec::new(),
        }
    }

    pub fn input_datatype(&self) -> DataTypeEnum {
        self.input_datatype
    }

    pub fn out_datatype(&self) -> DataTypeEnum {
        if let Some(first) = self.functions.first() {
            first.output_type
        } else {
            self.input_datatype
        }
    }
}

/// Expression tree node supporting any arity operations
#[derive(Clone, Debug)]
pub(crate) enum NaryExpr {
    /// Operation with N children (supports unary, binary, or more)
    Op {
        children: Vec<NaryExpr>,
        function: NaryFunction,
    },
    /// Index into input tensor using computed index expressions
    IndexedInput {
        /// Which input to access (index into inputs array)
        input_idx: usize,
        /// Index expressions, one per dimension of the input tensor.
        /// Each element evaluates to a u32 index.
        /// For element-wise access, use `vec![DimIndex(0), DimIndex(1), ..., DimIndex(rank-1)]`.
        indices: Vec<NaryExpr>,
    },
    /// Get current output dimension index
    DimIndex(usize),
}

#[derive(Clone, Debug)]
pub(crate) enum NaryInputReplacement {
    Leaf(usize),
    Expr(NaryExpr),
}

impl NaryExpr {
    /// Create an input expression that accesses at the current dimension indices (element-wise)
    pub fn input(input_idx: usize, rank: usize) -> Self {
        NaryExpr::IndexedInput {
            input_idx,
            indices: (0..rank).map(NaryExpr::DimIndex).collect(),
        }
    }

    /// Create an input expression with custom index expressions
    pub fn indexed_input(input_idx: usize, indices: Vec<NaryExpr>) -> Self {
        NaryExpr::IndexedInput { input_idx, indices }
    }

    /// Check if indices represent element-wise access (just DimIndex(0), DimIndex(1), ..., DimIndex(rank-1))
    pub(crate) fn is_elementwise_indices(indices: &[NaryExpr]) -> bool {
        indices
            .iter()
            .enumerate()
            .all(|(i, idx)| matches!(idx, NaryExpr::DimIndex(d) if *d == i))
    }

    /// Create a select expression (ternary operator)
    /// Semantics: condition != 0 ? on_true : on_false
    pub fn select(
        condition: NaryExpr,
        on_true: NaryExpr,
        on_false: NaryExpr,
        condition_type: DataTypeEnum,
        output_type: DataTypeEnum,
    ) -> NaryExpr {
        NaryExpr::Op {
            children: vec![condition, on_true, on_false],
            function: NaryFunction::select(condition_type, output_type),
        }
    }

    /// Create a multiplication expression: a * b
    pub fn mul(a: NaryExpr, b: NaryExpr, datatype: DataTypeEnum) -> NaryExpr {
        NaryExpr::Op {
            children: vec![a, b],
            function: NaryFunction::binary(
                Some("mul".to_string()),
                BinaryOp::Mul,
                datatype,
                datatype,
                datatype,
            ),
        }
    }

    /// Create an addition expression: a + b
    pub fn add(a: NaryExpr, b: NaryExpr, datatype: DataTypeEnum) -> NaryExpr {
        NaryExpr::Op {
            children: vec![a, b],
            function: NaryFunction::binary(
                Some("add".to_string()),
                BinaryOp::Add,
                datatype,
                datatype,
                datatype,
            ),
        }
    }

    /// Create a negation expression: -a
    pub fn neg(a: NaryExpr, datatype: DataTypeEnum) -> NaryExpr {
        NaryExpr::Op {
            children: vec![a],
            function: NaryFunction::unary(
                Some("neg".to_string()),
                UnaryOp::Neg,
                datatype,
                datatype,
            ),
        }
    }

    /// Create a unary expression that must be taught to tensor_ir before execution.
    pub fn unsupported_unary(
        a: NaryExpr,
        name: &str,
        input_type: DataTypeEnum,
        output_type: DataTypeEnum,
    ) -> NaryExpr {
        NaryExpr::Op {
            children: vec![a],
            function: NaryFunction::unsupported_unary(
                Some(name.to_string()),
                input_type,
                output_type,
            ),
        }
    }

    /// Create an index_select expression
    ///
    /// This creates an expression that:
    /// - Accesses the index tensor (input 1) at the select dimension to get the index value
    /// - Uses that index value to access the main tensor (input 0) along the select dimension
    /// - Uses normal output dimensions for all other dimensions
    ///
    /// For a tensor with rank R, selecting along dimension D:
    /// - Input 0: main tensor (rank R)
    /// - Input 1: index tensor (rank 1, u32)
    /// - Output: tensor with shape where dimension D is replaced with index tensor length
    pub fn index_select(rank: usize, select_dimension: usize) -> NaryExpr {
        // Build the index components for the main tensor access
        let index_components: Vec<NaryExpr> = (0..rank)
            .map(|dim| {
                if dim == select_dimension {
                    // For the select dimension, look up the index in the index tensor
                    // The index tensor is 1D, accessed at the current output's select_dimension position
                    NaryExpr::indexed_input(1, vec![NaryExpr::DimIndex(select_dimension)])
                } else {
                    // For other dimensions, use the current output dimension index directly
                    NaryExpr::DimIndex(dim)
                }
            })
            .collect();

        // Access the main tensor with the computed index
        NaryExpr::indexed_input(0, index_components)
    }

    /// Check if an expression uses custom indexing (not element-wise) for a specific input
    /// Returns true if the input is accessed with custom indexing, meaning buffer reuse is unsafe
    pub fn uses_custom_indexing_for_input(&self, target_input_idx: usize) -> bool {
        match self {
            NaryExpr::Op { children, .. } => children
                .iter()
                .any(|c| c.uses_custom_indexing_for_input(target_input_idx)),
            NaryExpr::IndexedInput { input_idx, indices } => {
                if *input_idx == target_input_idx {
                    // Custom indexing if indices is NOT the simple element-wise pattern
                    !Self::is_elementwise_indices(indices)
                } else {
                    // Recurse into the index expressions
                    indices
                        .iter()
                        .any(|c| c.uses_custom_indexing_for_input(target_input_idx))
                }
            }
            NaryExpr::DimIndex(_) => false,
        }
    }

    pub(crate) fn uses_any_custom_indexing(&self) -> bool {
        match self {
            NaryExpr::Op { children, .. } => {
                children.iter().any(NaryExpr::uses_any_custom_indexing)
            }
            NaryExpr::IndexedInput { indices, .. } => {
                !Self::is_elementwise_indices(indices)
                    || indices.iter().any(NaryExpr::uses_any_custom_indexing)
            }
            NaryExpr::DimIndex(_) => false,
        }
    }

    pub(crate) fn substitute_inputs(&self, replacements: &[NaryInputReplacement]) -> Self {
        match self {
            NaryExpr::Op { children, function } => NaryExpr::Op {
                children: children
                    .iter()
                    .map(|child| child.substitute_inputs(replacements))
                    .collect(),
                function: function.clone(),
            },
            NaryExpr::IndexedInput { input_idx, indices } => {
                let mapped_indices = indices
                    .iter()
                    .map(|index| index.substitute_inputs(replacements))
                    .collect::<Vec<_>>();
                match replacements.get(*input_idx) {
                    Some(NaryInputReplacement::Leaf(mapped_input)) => NaryExpr::IndexedInput {
                        input_idx: *mapped_input,
                        indices: mapped_indices,
                    },
                    Some(NaryInputReplacement::Expr(expr)) => {
                        debug_assert!(Self::is_elementwise_indices(indices));
                        expr.clone()
                    }
                    None => NaryExpr::IndexedInput {
                        input_idx: *input_idx,
                        indices: mapped_indices,
                    },
                }
            }
            NaryExpr::DimIndex(dim) => NaryExpr::DimIndex(*dim),
        }
    }

    /// Get the name of the expression for debugging
    pub fn name(&self) -> String {
        match self {
            NaryExpr::Op { children, function } => {
                let child_names: Vec<_> = children.iter().map(|c| c.name()).collect();
                format!("{}({})", function.name(), child_names.join(","))
            }
            NaryExpr::IndexedInput { input_idx, indices } => {
                if Self::is_elementwise_indices(indices) {
                    format!("input_{}", input_idx)
                } else {
                    let idx_names: Vec<_> = indices.iter().map(|c| c.name()).collect();
                    format!("input_{}[{}]", input_idx, idx_names.join(","))
                }
            }
            NaryExpr::DimIndex(dim) => format!("dim_{}", dim),
        }
    }
}

/// N-ary operation combining multiple inputs with arbitrary operations.
/// Can fuse chains of element-wise and pair-wise operations into a single kernel.
#[derive(Clone, Debug)]
pub(crate) struct NaryOperation {
    /// Input tensors (leaves of expression tree)
    pub(crate) inputs: Vec<NodeIndex>,
    /// Expression tree describing computation (includes all operations)
    pub(crate) expression: NaryExpr,
    pub(crate) shape: Box<[usize]>,
    pub(crate) output_datatype: DataTypeEnum,
}

impl NaryOperation {}

impl Operation for NaryOperation {
    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        for input in &self.inputs {
            f(*input);
        }
    }

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<MirValue> {
        let mut mir_inputs: Vec<MirValue> = self
            .inputs
            .iter()
            .enumerate()
            .map(|(i, idx)| {
                if self.expression.uses_custom_indexing_for_input(i) {
                    if let Some(cached) = nodes.get_result(*idx) {
                        return cached.into();
                    }
                }
                nodes.get_result_or_tensor(*idx).unwrap().into()
            })
            .collect();

        // Check if we can reuse an input allocation for output
        // We can only reuse if:
        // 1. The input matches datatype, is owned, and doesn't overlap
        // 2. The input is NOT accessed with custom indexing (which would cause read/write races)
        let reuse_index = mir_inputs.iter().enumerate().find_map(|(i, input)| {
            // Don't reuse if this input is accessed with custom indexing
            if self.expression.uses_custom_indexing_for_input(i) {
                return None;
            }
            if let Ok(data) = std::convert::TryInto::<MaybeQData>::try_into(input.clone())
                && data.datatype() == self.output_datatype
                && data.owned()
                && !data.layout().allocation_overlaps()
            {
                return Some(i);
            }
            None
        });

        if reuse_index.is_none() {
            // Need to allocate a new output tensor
            let first_input: MaybeQData = mir_inputs[0].clone().try_into().unwrap();
            let output_tensor =
                TensorData::new_for_shape(first_input.device(), &self.shape, self.output_datatype);
            mir_inputs.push(output_tensor.into());
        }

        mir_inputs
    }

    fn name(&self) -> String {
        format!(
            "nary_{}_{}",
            self.expression.name(),
            self.shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("x")
        )
    }

    fn build_tensor_ir(
        &self,
        _nodes: &ComputeGraphInner,
        inputs: &[MirValue],
    ) -> Result<crate::mir::operation::TensorIrLowering, String> {
        crate::tensor_ir_lowering::nary(self, inputs)
    }
}
