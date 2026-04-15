use std::fmt::Write;

use crate::{
    DataTypeEnum, Layout, SmallerRank, TILE_SIZE, Tensor, TensorData,
    compute_graph::NodeIndex,
    map_layout::MapLayoutOperation,
    mir::{
        kernel::GenericKernel,
        operation::Operation,
        workgroup_shape::{Constraint, WorkgroupShapeConstraints},
    },
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

    fn kernel(
        &self,
        input_rank: u32,
        datatype: DataTypeEnum,
        tile_size: u32,
        kernel: &mut GenericKernel,
    ) {
        let global_id = kernel.global_id();
        let input = kernel.add_tensor_input(input_rank, true, datatype);
        let output = kernel.add_tensor_input(self.new_shape.len() as u32, true, datatype);

        for local_index in 0..tile_size {
            writeln!(kernel, "{{").unwrap();
            for (prefix, tensor) in [("input", &input), ("output", &output)] {
                writeln!(
                    kernel,
                    "var {prefix}_remaining_index = {global_id}.x * {tile_size} + {local_index};"
                )
                .unwrap();
                for i in (0..tensor.rank()).rev() {
                    let shape_i = tensor.shape_binding(i);
                    writeln!(
                        kernel,
                        "let {prefix}_index_{i} = {prefix}_remaining_index % {shape_i};",
                    )
                    .unwrap();
                    writeln!(kernel, "{prefix}_remaining_index /= {shape_i};",).unwrap();
                }
            }
            write!(kernel, "let input_index = ").unwrap();
            input.strided_index(kernel, (0..).map(|i| format!("input_index_{i}")));
            writeln!(kernel, ";").unwrap();
            write!(kernel, "let output_index = ").unwrap();
            output.strided_index(kernel, (0..).map(|i| format!("output_index_{i}")));
            writeln!(kernel, ";").unwrap();
            writeln!(kernel, "{output}[output_index] = {input}[input_index];").unwrap();

            writeln!(kernel, "}}").unwrap();
        }
    }
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
        let input = inputs[1].as_tensor().unwrap();
        let total_workgroups = (input.layout().shape().iter().product::<usize>() as u32)
            .div_ceil(TILE_SIZE * BLOCKSIZE);
        distribute_workgroups(total_workgroups)
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
        let output_sliced =
            output.slice(&self.fill_shape.iter().map(|x| 0..*x).collect::<Vec<_>>());
        vec![input.into(), output_sliced.into()]
    }

    fn build_kernel(
        &self,
        _: &crate::compute_graph::ComputeGraphInner,
        _: &crate::mir::workgroup_shape::WorkgroupShape,
        inputs: &[crate::mir::inputs::MirValue],
        kernel: &mut GenericKernel,
    ) {
        let input = inputs[0].as_tensor().unwrap();
        let rank = input.layout().rank() as u32;
        let datatype = input.datatype();
        self.kernel(rank, datatype, TILE_SIZE, kernel);
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

impl<const R: usize, T: crate::DataType> Tensor<R, T> {
    pub fn resize(&self, new_shape: [usize; R]) -> Tensor<R, T> {
        let new_shape = new_shape.into();
        let input = self.key();
        self.add_resize(ResizeOperation::new(
            input,
            (*self.shape()).into(),
            new_shape,
            (*self.shape()).into(),
        ))
    }

    pub fn reshape<const R2: usize>(&self, new_shape: impl ShapeWithOneHole<R2>) -> Tensor<R2, T> {
        let new_shape = new_shape.resolve_shape(self.shape());
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
            (*self.shape()).into(),
            new_shape.clone(),
            new_shape.clone(),
        ))
    }

    pub fn flatten_last_n<const FROM_END: usize, const O: usize>(&self) -> Tensor<O, T>
    where
        Self: SmallerRank<FROM_END, O, T>,
    {
        let new_shape = std::array::from_fn(|i| {
            if i < self.rank() - 1 - FROM_END {
                self.shape()[i]
            } else if i == self.rank() - 1 - FROM_END {
                self.shape()[i..].iter().product()
            } else {
                1
            }
        });
        self.reshape(new_shape)
    }

    pub fn flatten_first_n<const FROM_START: usize, const O: usize>(&self) -> Tensor<O, T>
    where
        Self: SmallerRank<FROM_START, O, T>,
    {
        let new_shape = std::array::from_fn(|i| {
            if i == 0 {
                self.shape()[..=FROM_START].iter().product()
            } else {
                self.shape()[i - FROM_START]
            }
        });
        self.reshape(new_shape)
    }

    pub fn flatten_all(&self) -> Tensor<1, T> {
        let size = self.shape().iter().product();
        self.reshape([size])
    }
}

pub trait ShapeWithOneHole<const R: usize> {
    fn resolve_shape(&self, original_shape: &[usize]) -> [usize; R];
}

impl<const R: usize> ShapeWithOneHole<R> for [usize; R] {
    fn resolve_shape(&self, _original_shape: &[usize]) -> [usize; R] {
        *self
    }
}

impl ShapeWithOneHole<1> for ((),) {
    fn resolve_shape(&self, original_shape: &[usize]) -> [usize; 1] {
        [original_shape.iter().product()]
    }
}

pub(crate) trait IndexTuple<const INDEX: usize> {
    type Output;
    fn const_index(&self) -> &Self::Output;
}

macro_rules! impl_index_tuple {
    // Internal: generate a single impl
    (@impl [$($T:ident),+] $idx:tt $Ti:ident) => {
        impl<$($T),+> IndexTuple<$idx> for ($($T,)+) {
            type Output = $Ti;
            fn const_index(&self) -> &Self::Output {
                &self.$idx
            }
        }
    };

    // Internal: recursively process parallel lists of indices and types
    (@step [$($T:ident),+] [$idx:tt $(, $rest_idx:tt)*] [$curr:ident $(, $rest:ident)*]) => {
        impl_index_tuple!(@impl [$($T),+] $idx $curr);
        impl_index_tuple!(@step [$($T),+] [$($rest_idx),*] [$($rest),*]);
    };

    // Base case: both lists exhausted
    (@step [$($T:ident),+] [] []) => {};

    // Entry point: [indices] followed by types
    ([$($idx:tt),+] $($T:ident),+ $(,)?) => {
        impl_index_tuple!(@step [$($T),+] [$($idx),+] [$($T),+]);
    };
}

impl_index_tuple!([0] T);
impl_index_tuple!([0, 1] T1, T2);
impl_index_tuple!([0, 1, 2] T1, T2, T3);
impl_index_tuple!([0, 1, 2, 3] T1, T2, T3, T4);
impl_index_tuple!([0, 1, 2, 3, 4] T1, T2, T3, T4, T5);
impl_index_tuple!([0, 1, 2, 3, 4, 5] T1, T2, T3, T4, T5, T6);
impl_index_tuple!([0, 1, 2, 3, 4, 5, 6] T1, T2, T3, T4, T5, T6, T7);
impl_index_tuple!([0, 1, 2, 3, 4, 5, 6, 7] T1, T2, T3, T4, T5, T6, T7, T8);
impl_index_tuple!([0, 1, 2, 3, 4, 5, 6, 7, 8] T1, T2, T3, T4, T5, T6, T7, T8, T9);
impl_index_tuple!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] T1, T2, T3, T4, T5, T6, T7, T8, T9, T10);
impl_index_tuple!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] T1, T2, T3, T4, T5, T6, T7, T8, T9, T10, T11);

macro_rules! impl_shape_with_one_hole {
    ($($name:ident),+) => {
        impl_shape_with_one_hole!(@push_forward (), $($name,)+);
    };
    (@push_forward $($before:ident,)* (), $next:ident, $($after:ident,)*) => {
        impl_shape_with_one_hole!(@impl_tuple $($before,)* (), $next, $($after,)*);
        impl_shape_with_one_hole!(@push_forward $($before,)* $next, (), $($after,)*);
    };
    (@push_forward $($before:ident,)* (),) => {
        impl_shape_with_one_hole!(@impl_tuple $($before,)* (),);
    };
    (@usize $($t:tt)*) => {
        usize
    };
    (@one $($t:ident)*) => {
        1
    };
    (@tuple_size $($before:ident,)* (), $($after:ident,)*) => {
        $(impl_shape_with_one_hole!(@one $before) + )* $(impl_shape_with_one_hole!(@one $after) + )* 1
    };
    (@known_size $first:ident, $($before:ident,)* (), $($after:ident,)* = $sum:expr) => {
        const $first: usize = $sum;
        impl_shape_with_one_hole!(@known_size $($before,)* (), $($after,)* = $sum + 1);
    };
    (@known_size (), $first:ident, $($after:ident,)* = $sum:expr) => {
        const $first: usize = $sum + 1;
        impl_shape_with_one_hole!(@known_size (), $($after,)* = $sum + 1);
    };
    (@known_size (), = $sum:expr) => {};
    (@impl_tuple $($before:ident,)* (), $($after:ident,)*) => {
        #[allow(non_snake_case)]
        impl ShapeWithOneHole<{impl_shape_with_one_hole!(@tuple_size $($before,)* (), $($after,)*)}> for ($(impl_shape_with_one_hole!(@usize $before),)* (), $(impl_shape_with_one_hole!(@usize $after),)*) {
            fn resolve_shape(&self, original_shape: &[usize]) -> [usize; impl_shape_with_one_hole!(@tuple_size $($before,)* (), $($after,)*)] {
                let total_size = original_shape.iter().product::<usize>();
                impl_shape_with_one_hole!(@known_size $($before,)* (), $($after,)* = 0);
                let known_size = {
                    let mut size = 1;
                    $(
                        size *= *IndexTuple::<{$before}>::const_index(self);
                    )*
                    $(
                        size *= *IndexTuple::<{$after}>::const_index(self);
                    )*
                    size
                };
                let hole_size = total_size / known_size;
                [
                    $(
                        *IndexTuple::<{$before}>::const_index(self),
                    )*
                    hole_size,
                    $(
                        *IndexTuple::<{$after}>::const_index(self),
                    )*
                ]
            }
        }
    };
}

impl_shape_with_one_hole!(A);
impl_shape_with_one_hole!(A, B);
impl_shape_with_one_hole!(A, B, C);
impl_shape_with_one_hole!(A, B, C, D);
impl_shape_with_one_hole!(A, B, C, D, E);
impl_shape_with_one_hole!(A, B, C, D, E, F);
impl_shape_with_one_hole!(A, B, C, D, E, F, G);
impl_shape_with_one_hole!(A, B, C, D, E, F, G, H);
impl_shape_with_one_hole!(A, B, C, D, E, F, G, H, I);
impl_shape_with_one_hole!(A, B, C, D, E, F, G, H, I, J);

#[cfg(test)]
#[tokio::test]
async fn test_resize() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.resize([30, 20]);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 2.);
    assert_eq!(as_slice[[1, 0]], 3.);
    assert_eq!(as_slice[[1, 1]], 4.);
    assert_eq!(as_slice[[2, 0]], 5.);
    assert_eq!(as_slice[[2, 1]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_reshape() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.reshape([2, 3]);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 2.);
    assert_eq!(as_slice[[0, 2]], 3.);
    assert_eq!(as_slice[[1, 0]], 4.);
    assert_eq!(as_slice[[1, 1]], 5.);
    assert_eq!(as_slice[[1, 2]], 6.);

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.reshape([6]);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0]], 1.);
    assert_eq!(as_slice[[1]], 2.);
    assert_eq!(as_slice[[2]], 3.);
    assert_eq!(as_slice[[3]], 4.);
    assert_eq!(as_slice[[4]], 5.);
    assert_eq!(as_slice[[5]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_transposed_reshape() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.t();
    let tensor = tensor.reshape([2, 3]);
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 3.);
    assert_eq!(as_slice[[0, 2]], 5.);
    assert_eq!(as_slice[[1, 0]], 2.);
    assert_eq!(as_slice[[1, 1]], 4.);
    assert_eq!(as_slice[[1, 2]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_flatten_last_n() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.flatten_last_n::<1, _>();
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0]], 1.);
    assert_eq!(as_slice[[1]], 2.);
    assert_eq!(as_slice[[2]], 3.);
    assert_eq!(as_slice[[3]], 4.);
    assert_eq!(as_slice[[4]], 5.);
    assert_eq!(as_slice[[5]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_flatten_first_n() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.flatten_first_n::<1, _>();
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0]], 1.);
    assert_eq!(as_slice[[1]], 2.);
    assert_eq!(as_slice[[2]], 3.);
    assert_eq!(as_slice[[3]], 4.);
    assert_eq!(as_slice[[4]], 5.);
    assert_eq!(as_slice[[5]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_flatten_all() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let tensor = tensor.flatten_all();
    let as_slice = tensor.as_slice().await.unwrap();
    println!("{as_slice:?}");
    assert_eq!(as_slice[[0]], 1.);
    assert_eq!(as_slice[[1]], 2.);
    assert_eq!(as_slice[[2]], 3.);
    assert_eq!(as_slice[[3]], 4.);
    assert_eq!(as_slice[[4]], 5.);
    assert_eq!(as_slice[[5]], 6.);
}
