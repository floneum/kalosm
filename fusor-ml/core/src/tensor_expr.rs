use std::{fmt, sync::Arc};

use rustc_hash::FxHashMap;

use crate::{
    Layout, MatMulOperation, ReduceOperation, TensorLayoutInfo,
    compute_graph::{ComputeGraphInner, NodeIndex},
    map_layout::MapLayoutOperation,
    mir::{
        inputs::MirValue,
        operation::{Operation, TensorIrLowering},
    },
    nary_wise::NaryOperation,
    resize::ResizeOperation,
    slice_assign::SliceAssignOperation,
};

type InputBuilder = Arc<dyn Fn(&ComputeGraphInner) -> Vec<MirValue> + Send + Sync>;
type LowerBuilder =
    Arc<dyn Fn(&ComputeGraphInner, &[MirValue]) -> Result<TensorIrLowering, String> + Send + Sync>;
type LayoutBuilder =
    Arc<dyn Fn(&FxHashMap<NodeIndex, TensorLayoutInfo>) -> TensorLayoutInfo + Send + Sync>;
type MetadataLower = Arc<dyn Fn(&ComputeGraphInner) -> Option<MapLayoutOperation> + Send + Sync>;

#[derive(Clone)]
pub(crate) struct TensorExprOperation {
    dependencies: Arc<[NodeIndex]>,
    name: Arc<str>,
    inputs: InputBuilder,
    lower: LowerBuilder,
    output_layout: LayoutBuilder,
    metadata_lower: Option<MetadataLower>,
    fusable_nary: Option<NaryOperation>,
}

impl fmt::Debug for TensorExprOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorExprOperation")
            .field("name", &self.name)
            .field("dependencies", &self.dependencies)
            .finish_non_exhaustive()
    }
}

impl TensorExprOperation {
    fn new(
        dependencies: impl Into<Arc<[NodeIndex]>>,
        name: impl Into<Arc<str>>,
        inputs: InputBuilder,
        lower: LowerBuilder,
        output_layout: LayoutBuilder,
        metadata_lower: Option<MetadataLower>,
        fusable_nary: Option<NaryOperation>,
    ) -> Self {
        Self {
            dependencies: dependencies.into(),
            name: name.into(),
            inputs,
            lower,
            output_layout,
            metadata_lower,
            fusable_nary,
        }
    }

    pub(crate) fn try_metadata_lower(
        &self,
        graph: &ComputeGraphInner,
    ) -> Option<MapLayoutOperation> {
        self.metadata_lower.as_ref().and_then(|lower| lower(graph))
    }

    pub(crate) fn as_fusable_nary(&self) -> Option<&NaryOperation> {
        self.fusable_nary.as_ref()
    }

    pub(crate) fn from_nary(op: NaryOperation) -> Self {
        let dependencies = op.inputs.clone();
        let shape = op.shape.clone();
        let datatype = op.output_datatype;
        let name = op.name();
        let input_op = op.clone();
        let lower_op = op.clone();
        Self::new(
            dependencies,
            name,
            Arc::new(move |graph| input_op.inputs(graph)),
            Arc::new(move |graph, inputs| lower_op.build_tensor_ir(graph, inputs)),
            Arc::new(move |_| TensorLayoutInfo::new(Layout::contiguous(&shape), datatype)),
            None,
            Some(op),
        )
    }

    pub(crate) fn from_matmul(op: MatMulOperation) -> Self {
        let dependencies = vec![op.first, op.second];
        let out_shape = op.out_shape.clone();
        let name = op.name();
        let datatype = op.datatype;
        let input_op = op.clone();
        let lower_op = op;
        let first = input_op.first;
        Self::new(
            dependencies,
            name,
            Arc::new(move |graph| input_op.inputs(graph)),
            Arc::new(move |graph, inputs| lower_op.build_tensor_ir(graph, inputs)),
            Arc::new(move |layouts| {
                let datatype = layouts
                    .get(&first)
                    .map(TensorLayoutInfo::datatype)
                    .unwrap_or(datatype);
                TensorLayoutInfo::new(Layout::contiguous(&out_shape), datatype)
            }),
            None,
            None,
        )
    }

    pub(crate) fn from_reduce(op: ReduceOperation) -> Self {
        let dependencies = vec![op.value];
        let value = op.value;
        let axis = op.axis;
        let output_datatype = op.out_datatype();
        let name = op.name();
        let input_op = op.clone();
        let lower_op = op;
        Self::new(
            dependencies,
            name,
            Arc::new(move |graph| input_op.inputs(graph)),
            Arc::new(move |graph, inputs| lower_op.build_tensor_ir(graph, inputs)),
            Arc::new(move |layouts| {
                let input = layouts
                    .get(&value)
                    .expect("reduce input layout is available");
                let new_shape = input
                    .layout()
                    .shape()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, dim)| (i != axis).then_some(*dim))
                    .collect::<Vec<_>>();
                TensorLayoutInfo::new(Layout::contiguous(&new_shape), output_datatype)
            }),
            None,
            None,
        )
    }

    pub(crate) fn from_resize(op: ResizeOperation) -> Self {
        let dependencies = vec![op.input];
        let input = op.input;
        let new_shape = op.new_shape.clone();
        let name = op.name();
        let input_op = op.clone();
        let lower_op = op.clone();
        let metadata_op = op;
        Self::new(
            dependencies,
            name,
            Arc::new(move |graph| input_op.inputs(graph)),
            Arc::new(move |graph, inputs| lower_op.build_tensor_ir(graph, inputs)),
            Arc::new(move |layouts| {
                let datatype = layouts
                    .get(&input)
                    .expect("resize input layout is available")
                    .datatype();
                TensorLayoutInfo::new(Layout::contiguous(&new_shape), datatype)
            }),
            Some(Arc::new(move |graph| metadata_op.lower(graph))),
            None,
        )
    }

    pub(crate) fn from_slice_assign(op: SliceAssignOperation) -> Self {
        let dependencies = vec![op.input, op.value];
        let input = op.input;
        let name = op.name();
        let input_op = op.clone();
        let lower_op = op;
        Self::new(
            dependencies,
            name,
            Arc::new(move |graph| input_op.inputs(graph)),
            Arc::new(move |graph, inputs| lower_op.build_tensor_ir(graph, inputs)),
            Arc::new(move |layouts| {
                layouts
                    .get(&input)
                    .expect("slice_assign input layout is available")
                    .clone()
            }),
            None,
            None,
        )
    }
}

impl Operation for TensorExprOperation {
    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        for dependency in self.dependencies.iter().copied() {
            f(dependency);
        }
    }

    fn inputs(&self, nodes: &ComputeGraphInner) -> Vec<MirValue> {
        (self.inputs)(nodes)
    }

    fn name(&self) -> String {
        self.name.to_string()
    }

    fn output_layout(&self, map: &FxHashMap<NodeIndex, TensorLayoutInfo>) -> TensorLayoutInfo {
        (self.output_layout)(map)
    }

    fn build_tensor_ir(
        &self,
        nodes: &ComputeGraphInner,
        inputs: &[MirValue],
    ) -> Result<TensorIrLowering, String> {
        (self.lower)(nodes, inputs)
    }
}
