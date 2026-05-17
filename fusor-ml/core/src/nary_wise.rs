use std::hash::{Hash, Hasher};

use rustc_hash::FxHasher;

use crate::{
    TILE_SIZE,
    compute_graph::{ComputeGraphInner, NodeIndex},
    mir::{inputs::MirValue, kernel_backend::DirectKernel, operation::Operation},
    tensor::{DataTypeEnum, TensorData},
    visit_tiled::{MaybeQData, titled_map_dispatch_size, titled_map_workgroup_size_constraints},
};

#[derive(Clone, Copy, Debug)]
pub(crate) enum NaryScalar {
    F32(f32),
    F16(half::f16),
    U32(u32),
}

impl PartialEq for NaryScalar {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::F32(a), Self::F32(b)) => a.to_bits() == b.to_bits(),
            (Self::F16(a), Self::F16(b)) => a.to_bits() == b.to_bits(),
            (Self::U32(a), Self::U32(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for NaryScalar {}

impl Hash for NaryScalar {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::F32(value) => value.to_bits().hash(state),
            Self::F16(value) => value.to_bits().hash(state),
            Self::U32(value) => value.hash(state),
        }
    }
}

impl NaryScalar {
    pub(crate) fn datatype(self) -> DataTypeEnum {
        match self {
            Self::F32(_) => DataTypeEnum::F32,
            Self::F16(_) => DataTypeEnum::F16,
            Self::U32(_) => DataTypeEnum::U32,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum NaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    Min,
    Max,
    Equal,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Neg,
    Cast,
    Select,
    Exp,
    Exp2,
    Log,
    Log2,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Tanh,
    TanhExact,
    Asin,
    Acos,
    Atan,
    Sinh,
    Cosh,
    Asinh,
    Acosh,
    Atanh,
    Abs,
    ApproximateExp,
    LessApproximateExp,
    AddConst(NaryScalar),
    SubConst(NaryScalar),
    RSubConst(NaryScalar),
    MulConst(NaryScalar),
    DivConst(NaryScalar),
    RDivConst(NaryScalar),
    RemConst(NaryScalar),
    RRemConst(NaryScalar),
    PowConst(NaryScalar),
    MinConst(NaryScalar),
    MaxConst(NaryScalar),
    EqualConst(NaryScalar),
    LessConst(NaryScalar),
    LessEqualConst(NaryScalar),
    GreaterConst(NaryScalar),
    GreaterEqualConst(NaryScalar),
}

/// A function that can be applied in the expression tree.
/// Supports any arity (unary, binary, etc.)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct NaryFunction {
    pub(crate) name: Option<String>,
    pub(crate) op: NaryOp,
    pub(crate) input_types: Vec<DataTypeEnum>,
    pub(crate) output_type: DataTypeEnum,
}

impl NaryFunction {
    pub fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("op")
    }

    pub fn new(
        name: Option<String>,
        op: NaryOp,
        input_types: Vec<DataTypeEnum>,
        output_type: DataTypeEnum,
    ) -> Self {
        Self {
            name,
            op,
            input_types,
            output_type,
        }
    }

    pub fn unary(
        name: Option<String>,
        op: NaryOp,
        input_type: DataTypeEnum,
        output_type: DataTypeEnum,
    ) -> Self {
        Self {
            name,
            op,
            input_types: vec![input_type],
            output_type,
        }
    }

    pub fn binary(
        name: Option<String>,
        op: NaryOp,
        input_a_type: DataTypeEnum,
        input_b_type: DataTypeEnum,
        output_type: DataTypeEnum,
    ) -> Self {
        Self {
            name,
            op,
            input_types: vec![input_a_type, input_b_type],
            output_type,
        }
    }
}

/// A chain of unary functions used for pre/post processing in reduce/matmul/dequantize.
/// Each function takes a single input and produces a single output; the chain is applied sequentially.
#[derive(Clone, Debug, Hash)]
pub(crate) struct UnaryFunctionChain {
    input_datatype: DataTypeEnum,
    pub(crate) functions: Vec<NaryFunction>,
}

impl UnaryFunctionChain {
    pub fn new(functions: Vec<NaryFunction>, input_datatype: DataTypeEnum) -> Self {
        Self {
            input_datatype,
            functions,
        }
    }

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
        if let Some(last) = self.functions.last() {
            last.output_type
        } else {
            self.input_datatype
        }
    }
}

/// Result of extracting a unary function chain from an NaryOperation.
/// Used by the resolver to fuse unary ops into reduce/matmul/dequantize.
pub(crate) struct ExtractedUnaryChain {
    pub(crate) value: crate::compute_graph::NodeIndex,
    pub(crate) functions: UnaryFunctionChain,
}

/// Expression tree node supporting any arity operations
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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
    Scalar(NaryScalar),
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

    pub fn scalar(value: NaryScalar) -> Self {
        NaryExpr::Scalar(value)
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
            function: NaryFunction::new(
                Some("select".to_string()),
                NaryOp::Select,
                vec![condition_type, output_type, output_type],
                output_type,
            ),
        }
    }

    /// Create a multiplication expression: a * b
    pub fn mul(a: NaryExpr, b: NaryExpr, datatype: DataTypeEnum) -> NaryExpr {
        NaryExpr::Op {
            children: vec![a, b],
            function: NaryFunction::binary(
                Some("mul".to_string()),
                NaryOp::Mul,
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
                NaryOp::Add,
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
            function: NaryFunction::unary(Some("neg".to_string()), NaryOp::Neg, datatype, datatype),
        }
    }

    /// Create a custom unary operation
    pub fn unary_op(
        a: NaryExpr,
        name: &str,
        op: NaryOp,
        input_type: DataTypeEnum,
        output_type: DataTypeEnum,
    ) -> NaryExpr {
        NaryExpr::Op {
            children: vec![a],
            function: NaryFunction::unary(Some(name.to_string()), op, input_type, output_type),
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
            NaryExpr::Scalar(_) => false,
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
            NaryExpr::Scalar(value) => format!("{value:?}"),
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

impl NaryOperation {
    /// Attempt to extract a unary function chain from this NaryOperation.
    /// This will only succeed if there is only a single input to the operation.
    pub(crate) fn try_extract_unary_chain(&self) -> Option<ExtractedUnaryChain> {
        if self.inputs.len() != 1 {
            return None;
        }

        let mut functions = Vec::new();
        if !Self::collect_unary_chain(&self.expression, &mut functions)? {
            return None;
        }
        if functions.is_empty() {
            return None;
        }

        let input_datatype = functions.first()?.input_types.first().copied()?;
        let mut current = input_datatype;
        for function in &functions {
            if function.input_types.as_slice() != [current] {
                return None;
            }
            current = function.output_type;
        }
        if current != self.output_datatype {
            return None;
        }

        Some(ExtractedUnaryChain {
            value: self.inputs[0],
            functions: UnaryFunctionChain::new(functions, input_datatype),
        })
    }

    fn collect_unary_chain(expr: &NaryExpr, functions: &mut Vec<NaryFunction>) -> Option<bool> {
        match expr {
            NaryExpr::IndexedInput { input_idx, indices } => {
                Some(*input_idx == 0 && NaryExpr::is_elementwise_indices(indices))
            }
            NaryExpr::Op { children, function } => {
                if children.len() != 1 || function.input_types.len() != 1 {
                    return None;
                }
                if !Self::collect_unary_chain(&children[0], functions)? {
                    return Some(false);
                }
                functions.push(function.clone());
                Some(true)
            }
            NaryExpr::DimIndex(_) | NaryExpr::Scalar(_) => None,
        }
    }

    /// Recognize a tile-IR-evaluatable expression over a paired-split pattern:
    /// two of the inputs are the gate/up halves of a `q_mat_mul` output, and
    /// any remaining inputs are per-column broadcast tensors (e.g. bias
    /// vectors) accessed as `IndexedInput(k, [DimIndex(last_axis)])`. The
    /// resolver uses this to auto-detect both the no-bias and biased FFN
    /// patterns and synthesize a `PairedEpilogue` that captures the whole
    /// expression.
    pub(crate) fn try_extract_paired_split(&self) -> Option<ExtractedPairedSplit> {
        if self.inputs.len() < 2 {
            return None;
        }
        let output_rank = self.shape.len();
        if !Self::expr_is_paired_evaluatable(&self.expression, output_rank) {
            return None;
        }
        let mut input_seen = vec![false; self.inputs.len()];
        Self::collect_input_usage_n(&self.expression, &mut input_seen, output_rank)?;
        Some(ExtractedPairedSplit {
            inputs: self.inputs.clone(),
            inputs_seen: input_seen,
            expression: self.expression.clone(),
        })
    }

    /// An expression is paired-evaluatable when every input access is either
    /// fully element-wise (matmul-view inputs: gate/up) or a single
    /// `DimIndex(output_rank - 1)` access into a 1D broadcast tensor
    /// (per-column extras: bias vectors). Other structures (DimIndex outside
    /// IndexedInput leaves, non-trivial index arithmetic) block fusion.
    fn expr_is_paired_evaluatable(expr: &NaryExpr, output_rank: usize) -> bool {
        match expr {
            NaryExpr::Op { children, .. } => children
                .iter()
                .all(|c| Self::expr_is_paired_evaluatable(c, output_rank)),
            NaryExpr::IndexedInput { indices, .. } => {
                NaryExpr::is_elementwise_indices(indices)
                    || Self::is_last_dim_broadcast(indices, output_rank)
            }
            NaryExpr::Scalar(_) => true,
            NaryExpr::DimIndex(_) => false,
        }
    }

    /// `indices` describes a 1D-broadcast access whose only index is the
    /// output's last-dim coordinate (e.g. `bias[col]` where `col` is the
    /// kernel's output column). Bias vectors hit this branch.
    fn is_last_dim_broadcast(indices: &[NaryExpr], output_rank: usize) -> bool {
        if output_rank == 0 || indices.len() != 1 {
            return false;
        }
        matches!(&indices[0], NaryExpr::DimIndex(d) if *d == output_rank - 1)
    }

    fn collect_input_usage_n(expr: &NaryExpr, seen: &mut [bool], output_rank: usize) -> Option<()> {
        match expr {
            NaryExpr::Op { children, .. } => {
                for child in children {
                    Self::collect_input_usage_n(child, seen, output_rank)?;
                }
                Some(())
            }
            NaryExpr::IndexedInput { input_idx, indices } => {
                if *input_idx >= seen.len() {
                    return None;
                }
                if !NaryExpr::is_elementwise_indices(indices)
                    && !Self::is_last_dim_broadcast(indices, output_rank)
                {
                    return None;
                }
                seen[*input_idx] = true;
                Some(())
            }
            NaryExpr::Scalar(_) => Some(()),
            NaryExpr::DimIndex(_) => None,
        }
    }
}

/// Result of extracting a paired-split FFN pattern from an NaryOperation. The
/// expression is preserved verbatim — the resolver re-emits it at the tile-IR
/// level inside the qgemv kernel, substituting the actual `gate` / `up` /
/// `extras...` tile values for each `IndexedInput` leaf.
pub(crate) struct ExtractedPairedSplit {
    pub(crate) inputs: Vec<NodeIndex>,
    /// `inputs_seen[i]` is `true` if the captured expression references
    /// `IndexedInput(i, ...)` anywhere. The resolver requires all inputs to
    /// be used (unused inputs would point at dead graph nodes).
    pub(crate) inputs_seen: Vec<bool>,
    pub(crate) expression: NaryExpr,
}

impl Operation for NaryOperation {
    fn hash_kernel_fields(&self, state: &mut FxHasher) {
        self.expression.hash(state);
        self.shape.hash(state);
        self.output_datatype.hash(state);
    }

    fn workgroup_shape_constraints(
        &self,
        device: &crate::Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        titled_map_workgroup_size_constraints(&self.shape, device)
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
    ) -> [u32; 3] {
        let first_input: MaybeQData = inputs[0].clone().try_into().unwrap();
        let max_per_dim = first_input
            .device()
            .limits()
            .max_compute_workgroups_per_dimension;
        titled_map_dispatch_size(TILE_SIZE, *workgroup_shape, &self.shape, max_per_dim)
    }

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
                // If this input uses custom indexing, we need the dequantized tensor,
                // not the raw QMatrix (custom indexing doesn't work on quantized data)
                if self.expression.uses_custom_indexing_for_input(i) {
                    // Try to get the cached (dequantized) result first
                    if let Some(cached) = nodes.get_result(*idx) {
                        return cached.into();
                    }
                }
                // Otherwise use the normal path which may return QMatrix for Dequantize nodes
                nodes.get_result_or_qmatrix(*idx).unwrap().into()
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
                && data.datatype() == self.output_datatype.into()
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

    fn output(&self, _nodes: &ComputeGraphInner, inputs: &[MirValue]) -> MirValue {
        // Check if we reused an input allocation
        let reuse_index = inputs[..self.inputs.len()]
            .iter()
            .enumerate()
            .find_map(|(i, input)| {
                // Don't reuse if this input is accessed with custom indexing
                if self.expression.uses_custom_indexing_for_input(i) {
                    return None;
                }
                if let Ok(data) = std::convert::TryInto::<MaybeQData>::try_into(input.clone())
                    && data.datatype() == self.output_datatype.into()
                    && data.owned()
                    && !data.layout().allocation_overlaps()
                {
                    return Some(i);
                }
                None
            });

        if let Some(idx) = reuse_index {
            inputs[idx].clone()
        } else {
            // Output is the last input (newly allocated)
            inputs.last().unwrap().clone()
        }
    }

    fn build_direct_kernel(
        &self,
        nodes: &ComputeGraphInner,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        crate::nary_direct::build_nary_direct_kernel(self, nodes, workgroup_shape, inputs)
    }

    fn requires_single_kernel_batch(&self) -> bool {
        true
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
}
