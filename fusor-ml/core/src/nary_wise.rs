use std::fmt::Write;

use crate::{
    ElementWiseFunction, ElementWiseOperation, TILE_SIZE,
    compute_graph::{ComputeGraphInner, NodeIndex},
    mir::{function::Function, inputs::MirValue, kernel::GenericKernel, operation::Operation},
    tensor::{DataTypeEnum, TensorData},
    visit_tiled::{
        MaybeQData, build_visit_tiled_kernel, titled_map_dispatch_size,
        titled_map_workgroup_size_constraints,
    },
};

/// A function that can be applied in the expression tree.
/// Supports any arity (unary, binary, etc.)
#[derive(Clone, Debug)]
pub(crate) struct NaryFunction {
    pub(crate) name: Option<String>,
    /// WGSL code, e.g. "let output = a + b;" or "let output = sin(input);"
    pub(crate) operation: String,
    /// Input parameter names, e.g. ["a", "b"] for binary, ["input"] for unary
    pub(crate) input_names: Vec<String>,
    pub(crate) input_types: Vec<DataTypeEnum>,
    pub(crate) output_type: DataTypeEnum,
}

impl NaryFunction {
    pub fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("op")
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
            function: NaryFunction {
                name: Some("select".to_string()),
                operation: format!(
                    "let output = select(on_false, on_true, condition != {}(0));",
                    condition_type
                ),
                input_names: vec![
                    "condition".to_string(),
                    "on_true".to_string(),
                    "on_false".to_string(),
                ],
                input_types: vec![condition_type, output_type, output_type],
                output_type,
            },
        }
    }

    /// Create a multiplication expression: a * b
    pub fn mul(a: NaryExpr, b: NaryExpr, datatype: DataTypeEnum) -> NaryExpr {
        NaryExpr::Op {
            children: vec![a, b],
            function: NaryFunction {
                name: Some("mul".to_string()),
                operation: "let output = a * b;".to_string(),
                input_names: vec!["a".to_string(), "b".to_string()],
                input_types: vec![datatype, datatype],
                output_type: datatype,
            },
        }
    }

    /// Create an addition expression: a + b
    pub fn add(a: NaryExpr, b: NaryExpr, datatype: DataTypeEnum) -> NaryExpr {
        NaryExpr::Op {
            children: vec![a, b],
            function: NaryFunction {
                name: Some("add".to_string()),
                operation: "let output = a + b;".to_string(),
                input_names: vec!["a".to_string(), "b".to_string()],
                input_types: vec![datatype, datatype],
                output_type: datatype,
            },
        }
    }

    /// Create a negation expression: -a
    pub fn neg(a: NaryExpr, datatype: DataTypeEnum) -> NaryExpr {
        NaryExpr::Op {
            children: vec![a],
            function: NaryFunction {
                name: Some("neg".to_string()),
                operation: "let output = -input;".to_string(),
                input_names: vec!["input".to_string()],
                input_types: vec![datatype],
                output_type: datatype,
            },
        }
    }

    /// Create a custom unary operation
    pub fn unary_op(
        a: NaryExpr,
        name: &str,
        operation: impl Into<String>,
        input_type: DataTypeEnum,
        output_type: DataTypeEnum,
    ) -> NaryExpr {
        NaryExpr::Op {
            children: vec![a],
            function: NaryFunction {
                name: Some(name.to_string()),
                operation: operation.into(),
                input_names: vec!["input".to_string()],
                input_types: vec![input_type],
                output_type,
            },
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

impl NaryOperation {
    /// Generate WGSL code for evaluating the expression tree.
    /// Returns (value_string, actual_datatype) where actual_datatype is the type of the returned value.
    #[allow(clippy::too_many_arguments)]
    fn generate_expr_code(
        &self,
        expr: &NaryExpr,
        kernel: &mut GenericKernel,
        input_values: &[String],
        input_tensors: &[crate::visit_tiled::MaybeQTensorInput],
        input_datatypes: &[DataTypeEnum],
        current_dims: &[String],
        temp_counter: &mut usize,
        functions_cache: &mut Vec<(String, Vec<DataTypeEnum>, Function)>,
    ) -> (String, DataTypeEnum) {
        match expr {
            NaryExpr::Op { children, function } => {
                // Recursively evaluate all children
                let child_results: Vec<(String, DataTypeEnum)> = children
                    .iter()
                    .map(|child| {
                        self.generate_expr_code(
                            child,
                            kernel,
                            input_values,
                            input_tensors,
                            input_datatypes,
                            current_dims,
                            temp_counter,
                            functions_cache,
                        )
                    })
                    .collect();

                // Cast child values to expected types if needed
                let child_values: Vec<String> = child_results
                    .iter()
                    .zip(&function.input_types)
                    .map(|((value, actual_type), expected_type)| {
                        if actual_type == expected_type {
                            value.clone()
                        } else {
                            // Insert type cast
                            format!("{}({})", expected_type, value)
                        }
                    })
                    .collect();

                // Check if we already have this function cached (by operation AND types)
                let func = if let Some((_, _, cached_func)) =
                    functions_cache.iter().find(|(op, types, _)| {
                        *op == function.operation && *types == function.input_types
                    }) {
                    cached_func.clone()
                } else {
                    // Create the function with proper input names and types
                    let func = kernel.add_function(
                        function.output_type,
                        function.operation.clone(),
                        function
                            .input_names
                            .iter()
                            .zip(&function.input_types)
                            .map(|(name, ty)| (name.clone(), ty.to_string())),
                    );
                    functions_cache.push((
                        function.operation.clone(),
                        function.input_types.clone(),
                        func.clone(),
                    ));
                    func
                };

                // Generate temp variable for result
                let temp_name = format!("tmp_{}", *temp_counter);
                *temp_counter += 1;

                // Call function with child values
                writeln!(kernel, "let {temp_name} = {};", func.call(child_values)).unwrap();

                (temp_name, function.output_type)
            }
            NaryExpr::IndexedInput { input_idx, indices } => {
                use crate::visit_tiled::MaybeQTensorInput;

                let actual_type = input_datatypes[*input_idx];

                // Check if this is element-wise access (can use pre-computed value)
                if NaryExpr::is_elementwise_indices(indices) {
                    (input_values[*input_idx].clone(), actual_type)
                } else {
                    // Custom indexing - evaluate each index expression
                    let dims: Vec<String> = indices
                        .iter()
                        .map(|idx_expr| {
                            let (value, _) = self.generate_expr_code(
                                idx_expr,
                                kernel,
                                input_values,
                                input_tensors,
                                input_datatypes,
                                current_dims,
                                temp_counter,
                                functions_cache,
                            );
                            value
                        })
                        .collect();

                    let custom_idx_var = format!("custom_idx_{}", *temp_counter);
                    *temp_counter += 1;

                    write!(kernel, "let {} = ", custom_idx_var).unwrap();
                    match &input_tensors[*input_idx] {
                        MaybeQTensorInput::Tensor(t) => {
                            t.strided_index(kernel, dims);
                        }
                        MaybeQTensorInput::QTensor(_) => {
                            panic!("Custom indexing not supported for quantized tensors");
                        }
                    }
                    writeln!(kernel, ";").unwrap();

                    (
                        format!("{}[{}]", input_tensors[*input_idx], custom_idx_var),
                        actual_type,
                    )
                }
            }
            NaryExpr::DimIndex(dim) => {
                // Return the current dimension variable directly
                (current_dims[*dim].clone(), DataTypeEnum::U32)
            }
        }
    }

    /// Attempt to convert this NaryOperation into an ElementwiseOperation. This will only succeed
    /// if there is only a single input to the operation
    pub(crate) fn try_into_elementwise_op(&self) -> Option<ElementWiseOperation> {
        if self.inputs.len() == 1 {
            let output_datatype = self.output_datatype;
            let value = self.inputs[0];
            let input_datatype = match &self.expression {
                NaryExpr::Op { function, .. } => function.input_types[0],
                NaryExpr::IndexedInput { .. } => {
                    // For ElementWise, we need the datatype from the first input
                    // This is a limitation - we can't easily get it without graph access
                    output_datatype
                }
                NaryExpr::DimIndex(_) => {
                    panic!("DimIndex cannot be the root expression for ElementWise conversion");
                }
            };

            fn collect_functions(
                expr: &NaryExpr,
                function_body: &mut String,
                out_id: &mut usize,
            ) -> std::fmt::Result {
                let this_output = *out_id;
                match expr {
                    NaryExpr::Op { children, function } => {
                        let mut inputs = Vec::new();
                        for child in children {
                            *out_id += 1;
                            inputs.push(*out_id);
                            collect_functions(child, function_body, out_id)?;
                        }
                        // TODO: this isn't a great way to handle this. Refactor once we get rid of elementwise and pairwise ops
                        let default_value = match function.output_type {
                            DataTypeEnum::F32 => "0.0",
                            DataTypeEnum::F16 => "f16(0.0)",
                            DataTypeEnum::U32 => "0u",
                        };
                        writeln!(function_body, "var output_{this_output} = {default_value};",)?;
                        writeln!(function_body, "{{",)?;
                        for (i, input_id) in inputs.iter().enumerate() {
                            writeln!(
                                function_body,
                                "    let {} = output_{};",
                                function.input_names[i], input_id
                            )?;
                        }
                        writeln!(function_body, "{}", function.operation)?;
                        writeln!(function_body, "    output_{} = output;", this_output,)?;
                        writeln!(function_body, "}}",)?;
                        Ok(())
                    }
                    NaryExpr::IndexedInput { indices, .. } => {
                        // Only element-wise access can be converted
                        if NaryExpr::is_elementwise_indices(indices) {
                            writeln!(function_body, "let output_{this_output} = input;")
                        } else {
                            panic!(
                                "IndexedInput with custom indices cannot be converted to ElementWise operation"
                            );
                        }
                    }
                    NaryExpr::DimIndex(_) => {
                        panic!("DimIndex cannot be converted to ElementWise operation");
                    }
                }
            }

            let mut function_body = String::new();
            let mut out_id = 0;
            collect_functions(&self.expression, &mut function_body, &mut out_id).unwrap();
            writeln!(function_body, "let output = output_0;").unwrap();
            let functions = ElementWiseFunction::new(function_body, output_datatype);

            let shape = self.shape.clone();
            Some(ElementWiseOperation::new(
                input_datatype,
                value,
                functions,
                shape,
            ))
        } else {
            None
        }
    }
}

impl Operation for NaryOperation {
    fn workgroup_shape_constraints(
        &self,
        device: &crate::Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        titled_map_workgroup_size_constraints(&self.shape, device)
    }

    fn dispatch_size(
        &self,
        workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        _inputs: &[MirValue],
    ) -> [u32; 3] {
        titled_map_dispatch_size(TILE_SIZE, *workgroup_shape, &self.shape)
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

    fn build_kernel(
        &self,
        graph: &ComputeGraphInner,
        _workgroup_shape: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[MirValue],
        kernel: &mut GenericKernel,
    ) {
        // Determine output tensor index
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

        let output_tensor_index = reuse_index.unwrap_or(self.inputs.len());

        // Collect inputs with datatypes and ranks for all inputs
        let tiled_inputs: Vec<_> = inputs
            .iter()
            .filter_map(|input| {
                let result: Result<MaybeQData, _> = input.clone().try_into();
                result.ok()
            })
            .map(|data| {
                let datatype = data.datatype();
                let input_rank = data.layout().shape().len() as u32;
                crate::visit_tiled::VisitTiledInput::new(datatype, input_rank)
            })
            .collect();

        // Extract DataTypeEnum for each input for type checking during code generation
        let input_datatypes: Vec<DataTypeEnum> = tiled_inputs
            .iter()
            .take(self.inputs.len())
            .map(|input| match input.datatype {
                crate::visit_tiled::VisitTiledInputType::Quantized(_) => DataTypeEnum::F32, // Quantized dequantizes to f32
                crate::visit_tiled::VisitTiledInputType::Dequantized(d) => d,
            })
            .collect();

        let mut functions_cache: Vec<(String, Vec<DataTypeEnum>, Function)> = Vec::new();

        let device = graph.device();
        build_visit_tiled_kernel(
            &device,
            &self.shape,
            TILE_SIZE,
            tiled_inputs,
            output_tensor_index,
            |kernel, indexes, tensors, values| {
                let input_values: Vec<_> = values[..self.inputs.len()].to_vec();
                let input_tensors = &tensors[..self.inputs.len()];
                let output_index = &indexes[output_tensor_index];
                let output_tensor = &tensors[output_tensor_index];

                let mut temp_counter = 0;

                // Extract dimension variables (dim_0, dim_1, ..., dim_N)
                let rank = self.shape.len();
                let current_dims: Vec<String> = (0..rank).map(|i| format!("dim_{}", i)).collect();

                // Generate expression tree evaluation
                let (result, _result_type) = self.generate_expr_code(
                    &self.expression,
                    kernel,
                    &input_values,
                    input_tensors,
                    &input_datatypes,
                    &current_dims,
                    &mut temp_counter,
                    &mut functions_cache,
                );

                format!("{output_tensor}[{output_index}] = {result};")
            },
            kernel,
        );
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
