use crate::{
    DataTypeEnum, Layout, TILE_SIZE, Tensor, TensorData,
    compute_graph::NodeIndex,
    map_layout::MapLayoutOperation,
    mir::{
        inputs::MirValue,
        kernel_backend::DirectKernel,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShape, WorkgroupShapeConstraints},
    },
    nary_wise::{NaryExpr, NaryOp, NaryOperation, NaryScalar},
    visit_tiled::distribute_workgroups,
};

const BLOCKSIZE: u32 = 256;

#[derive(Debug, Clone)]
pub(crate) struct ResizeOperation {
    pub(crate) input: NodeIndex,
    pub(crate) current_shape: Box<[usize]>,
    pub(crate) new_shape: Box<[usize]>,
    pub(crate) fill_shape: Box<[usize]>,
}

impl ResizeOperation {
    pub fn new(
        input: NodeIndex,
        current_shape: Box<[usize]>,
        new_shape: Box<[usize]>,
        fill_shape: Box<[usize]>,
    ) -> Self {
        Self {
            input,
            current_shape,
            new_shape,
            fill_shape,
        }
    }
}

impl ResizeOperation {
    pub(crate) fn lower(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
    ) -> Option<MapLayoutOperation> {
        let full_fill = self.fill_shape == self.new_shape;
        let matching_size = self.current_shape.iter().product::<usize>()
            == self.new_shape.iter().product::<usize>();
        if !full_fill || !matching_size {
            return None;
        }

        let input = graph.get_cached_result(self.input)?;
        let input_layout = input.layout();
        if !is_row_major_contiguous(input_layout) {
            return None;
        }

        // Find the chunks of strides that are contiguous in the input
        let mut contiguous_stride_chunks = Vec::new();
        for (stride, len) in input_layout
            .strides()
            .iter()
            .rev()
            .zip(input_layout.shape().iter().rev())
        {
            let Some((last_stride, last_len)) = contiguous_stride_chunks.last_mut() else {
                contiguous_stride_chunks.push((*stride, *len));
                continue;
            };
            let contiguous_stride = *last_stride * *last_len;
            if *stride == contiguous_stride {
                *last_len *= len;
            } else {
                contiguous_stride_chunks.push((*stride, *len));
            }
        }

        // Check if the new shape can be formed by combining contiguous chunks
        // of the input shape
        let new_shape = self.new_shape.clone();
        let mut new_strides = Vec::new();
        let offset = input_layout.offset();
        let mut contiguous_stride_chunks_iter = contiguous_stride_chunks.iter_mut().peekable();
        for shape in new_shape.iter().rev() {
            // If we've used up this chunk and the current shape dimension is more than 1,
            // move to the next chunk
            while contiguous_stride_chunks_iter
                .next_if(|(_, len)| *len == 1 && *shape > 1)
                .is_some()
            {}
            let (stride, len) = contiguous_stride_chunks_iter.peek_mut()?;
            // Make sure the current chunk can be divided to form the new shape
            if *len % *shape != 0 {
                return None;
            }
            *len /= *shape;
            new_strides.push(*stride);
            *stride *= *shape;
        }
        new_strides.reverse();

        Some(MapLayoutOperation::new(self.input, move |_layout| {
            Layout::from_parts(offset, new_shape.clone(), new_strides.as_slice().into())
        }))
    }

    fn copy_expression(&self) -> Option<NaryExpr> {
        let flat = row_major_flat_expr(&self.fill_shape)?;
        let input_indices = row_major_indices_from_flat(flat, &self.current_shape)?;
        Some(NaryExpr::indexed_input(0, input_indices))
    }

    fn in_fill_bounds_expression(&self) -> NaryExpr {
        let mut condition = NaryExpr::scalar(NaryScalar::U32(1));
        for (dim, &fill) in self.fill_shape.iter().enumerate() {
            let lt_fill = NaryExpr::unary_op(
                NaryExpr::DimIndex(dim),
                "lt_fill",
                NaryOp::LessConst(NaryScalar::U32(fill as u32)),
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            );
            condition = NaryExpr::mul(condition, lt_fill, DataTypeEnum::U32);
        }
        condition
    }

    fn zero_expression(datatype: DataTypeEnum) -> NaryExpr {
        let zero = match datatype {
            DataTypeEnum::F32 => NaryScalar::F32(0.0),
            DataTypeEnum::F16 => NaryScalar::F16(half::f16::from_f32(0.0)),
            DataTypeEnum::U32 => NaryScalar::U32(0),
        };
        NaryExpr::scalar(zero)
    }

    fn expression(&self, datatype: DataTypeEnum) -> Option<NaryExpr> {
        let copied = self.copy_expression()?;
        if self.fill_shape == self.new_shape {
            return Some(copied);
        }
        Some(NaryExpr::select(
            self.in_fill_bounds_expression(),
            copied,
            Self::zero_expression(datatype),
            DataTypeEnum::U32,
            datatype,
        ))
    }
}

fn is_row_major_contiguous(layout: &Layout) -> bool {
    let mut expected_stride = 1usize;
    for (dim, stride) in layout.shape().iter().zip(layout.strides()).rev() {
        if *dim > 1 && *stride != expected_stride {
            return false;
        }
        expected_stride = expected_stride.saturating_mul(*dim);
    }
    true
}

impl Operation for ResizeOperation {
    fn workgroup_shape_constraints(
        &self,
        _: &crate::Device,
    ) -> crate::mir::workgroup_shape::WorkgroupShapeConstraints {
        let mut constraints = WorkgroupShapeConstraints::new();
        constraints.add_constraint(0, Constraint::equals(BLOCKSIZE));
        constraints.add_constraint(1, Constraint::equals(1));
        constraints.add_constraint(2, Constraint::equals(1));
        constraints
    }

    fn dispatch_size(
        &self,
        _: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[crate::mir::inputs::MirValue],
    ) -> [u32; 3] {
        let output = inputs[1].as_tensor().unwrap();
        let total_workgroups = (output.layout().shape().iter().product::<usize>() as u32)
            .div_ceil(TILE_SIZE * BLOCKSIZE);
        distribute_workgroups(
            total_workgroups,
            output
                .device()
                .limits()
                .max_compute_workgroups_per_dimension,
        )
    }

    fn visit_dependencies(&self, f: &mut dyn FnMut(NodeIndex)) {
        f(self.input);
    }

    fn inputs(
        &self,
        nodes: &crate::compute_graph::ComputeGraphInner,
    ) -> Vec<crate::mir::inputs::MirValue> {
        let input = nodes.get_cached_result(self.input).unwrap().clone();
        let output = TensorData::new_for_shape(input.device(), &self.new_shape, input.datatype());
        vec![input.into(), output.into()]
    }

    fn build_direct_kernel(
        &self,
        graph: &crate::compute_graph::ComputeGraphInner,
        workgroup_shape: &WorkgroupShape,
        inputs: &[MirValue],
    ) -> Option<DirectKernel> {
        let input = inputs[0].as_tensor()?;
        let operation = NaryOperation {
            inputs: vec![self.input],
            expression: self.expression(input.datatype())?,
            shape: self.new_shape.clone(),
            output_datatype: input.datatype(),
        };
        crate::nary_direct::build_nary_direct_kernel_to_output(
            &operation,
            graph,
            workgroup_shape,
            inputs,
            1,
        )
    }

    fn output(
        &self,
        _: &crate::compute_graph::ComputeGraphInner,
        inputs: &[crate::mir::inputs::MirValue],
    ) -> crate::mir::inputs::MirValue {
        let output = inputs[1].as_tensor().unwrap();
        TensorData::new_from_buffer(
            output.device(),
            output.buffer().clone(),
            &self.new_shape,
            output.datatype(),
        )
        .into()
    }

    fn name(&self) -> String {
        format!(
            "resize_from_{}_to_{}",
            self.current_shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("x"),
            self.new_shape
                .iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join("x")
        )
    }
}

fn row_major_flat_expr(shape: &[usize]) -> Option<NaryExpr> {
    let mut flat = NaryExpr::scalar(NaryScalar::U32(0));
    for axis in 0..shape.len() {
        let stride = shape[axis + 1..]
            .iter()
            .try_fold(1u32, |acc, dim| acc.checked_mul((*dim).try_into().ok()?))?;
        let dim = NaryExpr::DimIndex(axis);
        let term = if stride == 1 {
            dim
        } else {
            NaryExpr::unary_op(
                dim,
                "mul_const",
                NaryOp::MulConst(NaryScalar::U32(stride)),
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            )
        };
        flat = NaryExpr::add(flat, term, DataTypeEnum::U32);
    }
    Some(flat)
}

fn row_major_indices_from_flat(flat: NaryExpr, shape: &[usize]) -> Option<Vec<NaryExpr>> {
    let mut indices = Vec::with_capacity(shape.len());
    for axis in 0..shape.len() {
        let divisor = shape[axis + 1..]
            .iter()
            .try_fold(1u32, |acc, dim| acc.checked_mul((*dim).try_into().ok()?))?;
        let dim = u32::try_from(shape[axis]).ok()?;
        let quotient = if divisor == 1 {
            flat.clone()
        } else {
            NaryExpr::unary_op(
                flat.clone(),
                "div_const",
                NaryOp::DivConst(NaryScalar::U32(divisor)),
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            )
        };
        indices.push(if dim == 1 {
            NaryExpr::scalar(NaryScalar::U32(0))
        } else {
            NaryExpr::unary_op(
                quotient,
                "rem_const",
                NaryOp::RemConst(NaryScalar::U32(dim)),
                DataTypeEnum::U32,
                DataTypeEnum::U32,
            )
        });
    }
    Some(indices)
}

impl Tensor {
    pub fn resize(&self, new_shape: impl AsRef<[usize]>) -> Tensor {
        let new_shape: Box<[usize]> = new_shape.as_ref().into();
        let input = self.key();
        self.add_resize(ResizeOperation::new(
            input,
            self.shape().into(),
            new_shape,
            self.shape().into(),
        ))
    }

    pub fn reshape(&self, new_shape: impl AsRef<[usize]>) -> Tensor {
        let new_shape = new_shape.as_ref();
        assert_eq!(
            new_shape.iter().product::<usize>(),
            self.shape().iter().product::<usize>(),
            "Reshape requires the number of elements to be the same. \
            Current shape: {:?}, target shape: {:?}",
            self.shape(),
            new_shape
        );
        let new_shape: Box<[usize]> = new_shape.into();
        let input = self.key();
        self.add_resize(ResizeOperation::new(
            input,
            self.shape().into(),
            new_shape.clone(),
            new_shape.clone(),
        ))
    }

    pub fn flatten_last_n(&self, from_end: usize) -> Tensor {
        assert!(
            from_end < self.rank(),
            "flatten_last_n FROM_END must be less than input rank"
        );
        let out_rank = self.rank() - from_end;
        let new_shape: Vec<usize> = (0..out_rank)
            .map(|i| {
                if i < self.rank() - 1 - from_end {
                    self.shape()[i]
                } else if i == self.rank() - 1 - from_end {
                    self.shape()[i..].iter().product()
                } else {
                    1
                }
            })
            .collect();
        self.reshape(new_shape)
    }

    pub fn flatten_first_n(&self, from_start: usize) -> Tensor {
        assert!(
            from_start < self.rank(),
            "flatten_first_n FROM_START must be less than input rank"
        );
        let out_rank = self.rank() - from_start;
        let new_shape: Vec<usize> = (0..out_rank)
            .map(|i| {
                if i == 0 {
                    self.shape()[..=from_start].iter().product()
                } else {
                    self.shape()[i + from_start]
                }
            })
            .collect();
        self.reshape(new_shape)
    }

    pub fn flatten_all(&self) -> Tensor {
        let size = self.shape().iter().product();
        self.reshape([size])
    }
}

pub use fusor_types::ShapeWithOneHole;
