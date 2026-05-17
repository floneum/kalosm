use std::{fmt::Debug, sync::Arc};

use crate::{
    DataType, Layout, MaxRank, Tensor, TensorData, compute_graph::NodeIndex,
    mir::operation::Operation,
};

type MapLayout = Arc<dyn Fn(&Layout) -> Layout + Send + Sync>;

#[derive(Clone)]
pub(crate) struct MapLayoutOperation {
    pub(crate) input: NodeIndex,
    pub(crate) map_layout_fn: MapLayout,
}

impl Debug for MapLayoutOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MapLayoutOperation")
            .field("input", &self.input)
            .finish()
    }
}

impl MapLayoutOperation {
    pub fn new(
        input: NodeIndex,
        map_layout_fn: impl Fn(&Layout) -> Layout + Send + Sync + 'static,
    ) -> Self {
        Self {
            input,
            map_layout_fn: Arc::new(map_layout_fn),
        }
    }

    pub fn map_tensor(&self, tensor: &TensorData) -> TensorData {
        TensorData::new_from_parts(
            tensor.device(),
            tensor.buffer().clone(),
            self.map_layout(tensor.layout()),
            tensor.datatype(),
        )
    }

    pub fn map_layout(&self, layout: &Layout) -> Layout {
        (self.map_layout_fn)(layout)
    }

    pub fn run(&self, graph: &mut crate::compute_graph::ComputeGraphInner) -> TensorData {
        let input = graph.get_result(self.input).unwrap();
        self.map_tensor(&input)
    }
}

impl Operation for MapLayoutOperation {
    fn workgroup_shape_constraints(
        &self,
        _: &crate::Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        Default::default()
    }

    fn dispatch_size(
        &self,
        _: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[crate::mir::inputs::MirValue],
    ) -> [u32; 3] {
        [1, 1, 1]
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.input);
    }

    fn inputs(
        &self,
        nodes: &crate::compute_graph::ComputeGraphInner,
    ) -> Vec<crate::mir::inputs::MirValue> {
        vec![nodes.get_result(self.input).unwrap().into()]
    }

    fn output(
        &self,
        _: &crate::compute_graph::ComputeGraphInner,
        inputs: &[crate::mir::inputs::MirValue],
    ) -> crate::mir::inputs::MirValue {
        let input = inputs[0].as_tensor().unwrap();
        self.map_tensor(input).into()
    }

    fn build_direct_kernel(
        &self,
        _: &crate::compute_graph::ComputeGraphInner,
        _: &crate::mir::workgroup_shape::WorkgroupShape,
        _: &[crate::mir::inputs::MirValue],
    ) -> Option<crate::mir::kernel_backend::DirectKernel> {
        None
    }

    fn name(&self) -> String {
        "map_layout".to_string()
    }
}

impl<const R: usize, T: DataType> Tensor<R, T> {
    pub fn restride<const R2: usize>(&self, specs: [crate::StrideSpec; R2]) -> Tensor<R2, T> {
        self.add_map_layout(MapLayoutOperation::new(self.key(), move |layout| {
            layout.restride(&specs)
        }))
    }

    /// Replace the tensor's layout with `new_layout`, treating the underlying
    /// buffer as a flat blob. The user-supplied offset and strides are absolute
    /// (in buffer elements), so the input must itself be a contiguous view with
    /// offset 0 — otherwise the user's strides would compose nonsensically with
    /// the input's own strides and silently read the wrong elements.
    ///
    /// Callers that need to re-layout a non-contiguous view should materialize
    /// it first (e.g. with `to_concrete()`), or use [`restride`] which composes
    /// stride specs relative to the input's current strides.
    pub fn restride_layout<const R2: usize>(&self, new_layout: Layout) -> Tensor<R2, T> {
        self.add_map_layout(MapLayoutOperation::new(self.key(), move |input_layout| {
            assert!(
                input_layout.is_contiguous() && input_layout.offset() == 0,
                "restride_layout requires a contiguous, offset-0 input — got \
                 offset={} strides={:?} shape={:?}. Call `.to_concrete()` first, \
                 or use `restride` for stride composition.",
                input_layout.offset(),
                input_layout.strides(),
                input_layout.shape()
            );
            new_layout.clone()
        }))
    }

    pub(crate) fn broadcast_as<const R2: usize>(&self, out_shape: [usize; R2]) -> Tensor<R2, T> {
        const {
            assert!(
                R2 >= R,
                "The output dimension must be more than the input dimension"
            )
        };

        let shape = self.shape();
        let specs: [crate::StrideSpec; R2] = std::array::from_fn(|out_i| {
            let in_i = out_i as isize - (R2 as isize - R as isize);
            if in_i < 0 {
                crate::StrideSpec::dim_with(0, out_shape[out_i], 0)
            } else {
                let in_i = in_i as usize;
                if shape[in_i] == 1 && out_shape[out_i] > 1 {
                    crate::StrideSpec::dim_with(in_i, out_shape[out_i], 0)
                } else {
                    crate::StrideSpec::dim(in_i, out_shape[out_i])
                }
            }
        });
        self.restride(specs)
    }

    pub(crate) fn broadcast_together<const R2: usize, const R3: usize>(
        first: &Tensor<R, T>,
        second: &Tensor<R2, T>,
    ) -> (Tensor<R3, T>, Tensor<R3, T>)
    where
        (Tensor<R, T>, Tensor<R2, T>): MaxRank<R3, T>,
    {
        const {
            assert!(
                R3 == if R > R2 { R } else { R2 },
                "The output dimension must be the maximum of the two input dimensions"
            )
        };

        let shape = if first.rank() > second.rank() {
            std::array::from_fn(|i| first.shape()[i])
        } else if first.rank() < second.rank() {
            std::array::from_fn(|i| second.shape()[i])
        } else {
            std::array::from_fn(|i| first.shape()[i].max(second.shape()[i]))
        };
        (first.broadcast_as(shape), second.broadcast_as(shape))
    }

    pub(crate) fn broadcast_then_elementwise_op<const R2: usize, const R3: usize>(
        first: &Tensor<R, T>,
        second: &Tensor<R2, T>,
        op: impl Fn(Tensor<R3, T>, Tensor<R3, T>) -> Tensor<R3, T>,
    ) -> Tensor<R3, T>
    where
        (Tensor<R, T>, Tensor<R2, T>): MaxRank<R3, T>,
    {
        let (b1, b2) = Self::broadcast_together(first, second);
        assert_eq!(b1.shape(), b2.shape());
        op(b1, b2)
    }
}
