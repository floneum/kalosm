use std::{
    fmt::{Debug, Display},
    marker::PhantomData,
    num::NonZeroU64,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Range, Sub, SubAssign},
    sync::Arc,
};

use bytemuck::{AnyBitPattern, NoUninit};
use tabbycat::Graph;
use wgpu::COPY_BUFFER_ALIGNMENT;

use crate::{
    Device, Dim, ElementWiseOperation, Layout, MatMulOperation, MatMulParams, PairWiseFunction,
    PairWiseOperation, ReduceFunction, ReduceOperation,
    compute_graph::NodeIndex,
    index_select::IndexSelectOperation,
    map_layout::MapLayoutOperation,
    mir::operation::Operation,
    quantized::{QMatrix, matmul::QMatMulOperation},
    resize::ResizeOperation,
    slice_assign::SliceAssignOperation,
};

pub use fusor_types::TensorSlice;

pub trait DataType:
    Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + PartialOrd
    + NoUninit
    + AnyBitPattern
    + Debug
    + Display
{
    const WGSL_TYPE: DataTypeEnum;

    fn zero() -> Self;
    fn one() -> Self;
}

pub trait FloatDataType: DataType {
    fn from_f32(value: f32) -> Self;

    fn is_finite(&self) -> bool;
}

impl DataType for f32 {
    const WGSL_TYPE: DataTypeEnum = DataTypeEnum::F32;

    fn zero() -> Self {
        0.
    }

    fn one() -> Self {
        1.
    }
}

impl FloatDataType for f32 {
    fn from_f32(value: f32) -> Self {
        value
    }

    fn is_finite(&self) -> bool {
        f32::is_finite(*self)
    }
}

impl DataType for half::f16 {
    const WGSL_TYPE: DataTypeEnum = DataTypeEnum::F16;

    fn zero() -> Self {
        half::f16::from_f32(0.)
    }

    fn one() -> Self {
        half::f16::from_f32(1.)
    }
}

impl FloatDataType for half::f16 {
    fn from_f32(value: f32) -> Self {
        half::f16::from_f32(value)
    }

    fn is_finite(&self) -> bool {
        half::f16::is_finite(*self)
    }
}

impl DataType for u32 {
    const WGSL_TYPE: DataTypeEnum = DataTypeEnum::U32;

    fn zero() -> Self {
        0
    }

    fn one() -> Self {
        1
    }
}

#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DataTypeEnum {
    F32,
    F16,
    U32,
}

impl DataTypeEnum {
    pub fn as_str(&self) -> &'static str {
        match self {
            DataTypeEnum::F32 => "f32",
            DataTypeEnum::F16 => "f16",
            DataTypeEnum::U32 => "u32",
        }
    }

    pub fn element_size(&self) -> usize {
        match self {
            DataTypeEnum::F32 => size_of::<f32>(),
            DataTypeEnum::F16 => size_of::<half::f16>(),
            DataTypeEnum::U32 => size_of::<u32>(),
        }
    }
}

impl Display for DataTypeEnum {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TensorLayoutInfo {
    layout: Layout,
    datatype: DataTypeEnum,
}

impl Display for TensorLayoutInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} {}", self.layout.shape(), self.datatype)
    }
}

impl TensorLayoutInfo {
    pub(crate) fn new(layout: Layout, datatype: DataTypeEnum) -> Self {
        Self { layout, datatype }
    }

    pub(crate) fn layout(&self) -> &Layout {
        &self.layout
    }

    pub(crate) fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    pub(crate) fn datatype(&self) -> DataTypeEnum {
        self.datatype
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TensorInfo {
    shape: Box<[usize]>,
    datatype: DataTypeEnum,
}

impl Display for TensorInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?} {}", self.shape, self.datatype)
    }
}

impl TensorInfo {
    pub(crate) fn new(shape: Box<[usize]>, datatype: DataTypeEnum) -> Self {
        Self { shape, datatype }
    }

    pub(crate) fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub(crate) fn rank(&self) -> usize {
        self.shape.len()
    }

    pub(crate) fn datatype(&self) -> DataTypeEnum {
        self.datatype
    }
}

pub(crate) struct LazyTensorData {
    device: Device,
    info: TensorInfo,
    key: NodeIndex,
}

impl Clone for LazyTensorData {
    fn clone(&self) -> Self {
        self.device.compute_graph().add_reference(self.key);
        Self {
            device: self.device.clone(),
            info: self.info.clone(),
            key: self.key,
        }
    }
}

impl Drop for LazyTensorData {
    fn drop(&mut self) {
        self.device.compute_graph().remove_reference(self.key);
    }
}

impl LazyTensorData {
    pub(crate) fn new(data: TensorData) -> Self {
        let device = data.device.clone();
        let info = data.info.clone();
        let key = device.compute_graph().create_tensor(data);

        Self {
            device,
            info: TensorInfo::new(info.shape().into(), info.datatype()),
            key,
        }
    }

    pub(crate) fn from_parts(device: Device, info: TensorInfo, key: NodeIndex) -> Self {
        Self { device, info, key }
    }

    pub(crate) fn custom(&self, custom: Arc<dyn Operation + Send + Sync>) -> Self {
        let device = self.device.clone();
        let info = self.info.clone();
        let key = device.compute_graph().create_custom(custom);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn where_cond(
        &self,
        operation: crate::composite::where_cond::WhereCondOperation,
    ) -> Self {
        let device = self.device.clone();
        let info = self.info.clone();
        let key = device.compute_graph().create_where_cond(operation);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn element_wise(&self, function: ElementWiseOperation) -> Self {
        let device = self.device.clone();
        let mut info = self.info.clone();
        info.datatype = function.functions.out_datatype();
        let key = device.compute_graph().create_element_wise(function);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn pair_wise(&self, function: PairWiseOperation) -> Self {
        let device = self.device.clone();
        let info = self.info.clone();
        let key = device.compute_graph().create_pair_wise(function);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn mat_mul(&self, function: MatMulOperation) -> Self {
        let device = self.device.clone();
        let mut info = self.info.clone();
        info.shape = function.out_shape.clone();
        let key = device.compute_graph().create_mat_mul(function);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn q_mat_mul(&self, function: QMatMulOperation) -> Self {
        let device = self.device.clone();
        let mut info = self.info.clone();
        info.shape = function.out_shape.clone();
        let key = device.compute_graph().create_q_mat_mul(function);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn reduce(&self, function: ReduceOperation) -> Self {
        let device = self.device.clone();
        let mut info = self.info.clone();
        let dim = function.axis;
        let new_shape: Box<[usize]> = self
            .info
            .shape()
            .iter()
            .enumerate()
            .filter_map(|(i, x)| (i != dim).then_some(*x))
            .collect();
        info = TensorInfo::new(new_shape, info.datatype());
        let key = device.compute_graph().create_reduce(function);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn map_layout(&self, op: MapLayoutOperation) -> Self {
        let device = self.device.clone();
        // Compute output shape by applying the layout transformation to a temporary layout
        let temp_layout = Layout::contiguous(self.info.shape());
        let new_layout = op.map_layout(&temp_layout);
        let info = TensorInfo::new(new_layout.shape().into(), self.info.datatype());
        let key = device.compute_graph().create_map_layout(op);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn resize(&self, op: ResizeOperation) -> Self {
        let device = self.device.clone();
        let info = TensorInfo::new(op.new_shape.clone(), self.info.datatype());
        let key = device.compute_graph().create_resize(op);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn slice_assign(&self, op: SliceAssignOperation) -> Self {
        let device = self.device.clone();
        let info = self.info.clone();
        let key = device.compute_graph().create_slice_assign(op);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn index_select(&self, op: IndexSelectOperation) -> Self {
        let device = self.device.clone();
        let mut info = self.info.clone();
        info.shape = op.output_shape();
        let key = device.compute_graph().create_index_select(op);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn materialize(&self) -> (TensorData, usize) {
        let result = self.device.compute_graph().resolve(self.key, &self.device);
        (result.data, result.total_kernels)
    }

    pub fn graphvis(&self) -> Graph {
        self.device.compute_graph().graphvis(self.key)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct TensorData {
    device: Device,
    buffer: Arc<wgpu::Buffer>,
    info: TensorLayoutInfo,
}

impl PartialEq for TensorData {
    fn eq(&self, other: &Self) -> bool {
        self.info == other.info && self.buffer == other.buffer
    }
}

impl TensorData {
    pub(crate) fn new_from_buffer(
        device: &Device,
        buffer: impl Into<Arc<wgpu::Buffer>>,
        size: &[usize],
        datatype: DataTypeEnum,
    ) -> Self {
        let layout = Layout::contiguous(size);
        Self::new_from_parts(device, buffer, layout, datatype)
    }

    pub(crate) fn new_from_parts(
        device: &Device,
        buffer: impl Into<Arc<wgpu::Buffer>>,
        layout: Layout,
        datatype: DataTypeEnum,
    ) -> Self {
        let buffer = buffer.into();
        let buffer_len = buffer.size() / datatype.element_size() as u64;
        assert!(
            layout.offset()
                + layout
                    .strides()
                    .iter()
                    .zip(layout.shape().iter())
                    .map(|(s, dim)| s * dim.saturating_sub(1))
                    .sum::<usize>()
                < buffer_len as usize
        );
        Self {
            device: device.clone(),
            buffer,
            info: TensorLayoutInfo::new(layout, datatype),
        }
    }

    pub(crate) fn new_for_shape(device: &Device, shape: &[usize], datatype: DataTypeEnum) -> Self {
        let size =
            padded_tensor_size((datatype.element_size() * shape.iter().product::<usize>()) as u64);

        // Try to get a buffer from the cache first
        let buffer = device.create_buffer(
            size,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

        Self::new_from_buffer(device, buffer, shape, datatype)
    }

    pub(crate) fn new_splat<D: DataType>(device: &Device, shape: &[usize], data: D) -> Self {
        let datatype = D::WGSL_TYPE;
        let raw_data = bytemuck::bytes_of(&data);
        let unpadded_size = raw_data.len();
        let size = padded_tensor_size(unpadded_size as u64) as usize;
        let mut padded_data = vec![0u8; size];
        padded_data[..unpadded_size].copy_from_slice(raw_data);
        let buffer = device.create_buffer_init(
            &padded_data,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let strides = (0..shape.len()).map(|_| 0).collect();
        let layout = Layout::from_parts(0, shape.into(), strides);
        Self::new_from_parts(device, buffer, layout, datatype)
    }

    fn new_inner<'a, D: DataType, I: Iterator<Item = &'a D>>(
        device: &Device,
        data: I,
        shape: &[usize],
    ) -> Self {
        // MODIFIED from: https://github.com/gfx-rs/wgpu/blob/d8833d079833c62b4fd00325d0ba08ec0c8bc309/wgpu/src/util/device.rs#L38
        fn create_aligned_buffer(
            element_size: u64,
            shape: &[usize],
            device: &Device,
        ) -> (Arc<wgpu::Buffer>, u64) {
            let size = element_size * shape.iter().copied().product::<usize>() as u64;

            let padded_size = padded_tensor_size(size);

            let buffer = device.create_buffer(
                padded_size,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );
            (buffer, padded_size)
        }
        let (buffer, padded_size) = create_aligned_buffer(size_of::<D>() as u64, shape, device);

        if let Some(padded_size) = NonZeroU64::new(padded_size) {
            let write = device
                .wgpu_queue()
                .write_buffer_with(&buffer, 0, padded_size);
            if let Some(mut write) = write {
                write
                    .iter_mut()
                    .zip(data.flat_map(bytemuck::bytes_of))
                    .for_each(|(dst, src)| *dst = *src);
            }
        }

        Self::new_from_buffer(device, buffer, shape, D::WGSL_TYPE)
    }

    pub fn slice(&self, ranges: &[Range<usize>]) -> Self {
        let layout = self.info.layout.slice(ranges);
        Self {
            device: self.device.clone(),
            buffer: self.buffer.clone(),
            info: TensorLayoutInfo::new(layout, self.info.datatype),
        }
    }

    pub(crate) fn layout(&self) -> &Layout {
        &self.info.layout
    }

    pub(crate) fn datatype(&self) -> DataTypeEnum {
        self.info.datatype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn buffer(&self) -> &Arc<wgpu::Buffer> {
        &self.buffer
    }

    pub(crate) fn info(&self) -> &TensorLayoutInfo {
        &self.info
    }

    /// Check if this is the only reference to the buffer
    pub(crate) fn owned(&self) -> bool {
        std::sync::Arc::strong_count(&self.buffer) == 1
    }
}

pub struct Tensor<const R: usize, D> {
    data: LazyTensorData,
    datatype: PhantomData<D>,
}

impl<const R: usize, D: DataType> Display for Tensor<R, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} x {:?}", self.datatype(), self.shape())
    }
}

impl<const R: usize, D: DataType> Debug for Tensor<R, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor({} x {:?})", self.datatype(), self.shape())
    }
}

impl<const R: usize, D: DataType> From<TensorData> for Tensor<R, D> {
    fn from(value: TensorData) -> Self {
        Self {
            data: LazyTensorData::new(value),
            datatype: PhantomData,
        }
    }
}

impl<const R: usize, D> Clone for Tensor<R, D> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            datatype: PhantomData,
        }
    }
}

impl<D: DataType> fusor_types::FromArray<0, D, (), Device> for Tensor<0, D> {
    fn from_array(_data: (), device: &Device) -> Self {
        let iter = std::iter::empty();
        Tensor::new_inner(device, iter, [])
    }
}

impl<'a, I, D: DataType> fusor_types::FromArray<1, D, I, Device> for Tensor<1, D>
where
    I: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn from_array(data: I, device: &Device) -> Self {
        let iter = data.into_iter();
        let size = iter.len();
        Tensor::new_inner(device, iter, [size])
    }
}

impl<'a, I, I2, D: DataType> fusor_types::FromArray<2, D, I, Device> for Tensor<2, D>
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn from_array(data: I, device: &Device) -> Self {
        let mut iter = data.into_iter().map(IntoIterator::into_iter).peekable();
        let size = iter.len();
        let second_size = iter.peek().map(ExactSizeIterator::len).unwrap_or_default();
        let iter = iter.flat_map(|i| {
            let size = i.len();
            if size != second_size {
                panic!("expected a rectangular matrix. The first inner iterator size was {second_size}, but another inner iterator size was {size}");
            }
            i
        });
        Tensor::new_inner(device, iter, [size, second_size])
    }
}

impl<'a, I, I2, I3, D: DataType> fusor_types::FromArray<3, D, I, Device> for Tensor<3, D>
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = I3, IntoIter: ExactSizeIterator>,
    I3: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn from_array(data: I, device: &Device) -> Self {
        let mut iter = data
            .into_iter()
            .map(|i| i.into_iter().map(IntoIterator::into_iter).peekable())
            .peekable();
        let mut shape = [iter.len(), 0, 0];
        if let Some(iter) = iter.peek_mut() {
            let size = iter.len();
            shape[1] = size;
            if let Some(iter) = iter.peek() {
                let size = iter.len();
                shape[2] = size;
            }
        }

        let iter = iter.flat_map(|i| {
            let size = i.len();
            let required_size = shape[1];
            if size != required_size {
                panic!("expected a rectangular matrix. The first inner iterator size was {required_size}, but another inner iterator size was {size}");
            }
            i.flat_map(|i| {
                let size = i.len();
                let required_size = shape[2];
                if size != required_size {
                    panic!("expected a rectangular matrix. The first inner inner iterator size was {required_size}, but another inner inner iterator size was {size}");
                }
                i
            })
        });

        Tensor::new_inner(device, iter, shape)
    }
}

impl<'a, I, I2, I3, I4, D: DataType> fusor_types::FromArray<4, D, I, Device> for Tensor<4, D>
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = I3, IntoIter: ExactSizeIterator>,
    I3: IntoIterator<Item = I4, IntoIter: ExactSizeIterator>,
    I4: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn from_array(data: I, device: &Device) -> Self {
        let mut iter = data
            .into_iter()
            .map(|i| {
                i.into_iter()
                    .map(|i| i.into_iter().map(IntoIterator::into_iter).peekable())
                    .peekable()
            })
            .peekable();
        let mut shape = [iter.len(), 0, 0, 0];
        if let Some(iter) = iter.peek_mut() {
            let size = iter.len();
            shape[1] = size;
            if let Some(iter) = iter.peek_mut() {
                let size = iter.len();
                shape[2] = size;
                if let Some(iter) = iter.peek() {
                    let size = iter.len();
                    shape[3] = size;
                }
            }
        }

        let iter = iter.flat_map(|i| {
            let size = i.len();
            let required_size = shape[1];
            if size != required_size {
                panic!("expected a rectangular matrix. The first inner iterator size was {required_size}, but another inner iterator size was {size}");
            }
            i.flat_map(|i| {
                let size = i.len();
                let required_size = shape[2];
                if size != required_size {
                    panic!("expected a rectangular matrix. The first inner inner iterator size was {required_size}, but another inner inner iterator size was {size}");
                }
                i.flat_map(|i| {
                    let size = i.len();
                    let required_size = shape[3];
                    if size != required_size {
                        panic!("expected a rectangular matrix. The first inner inner inner iterator size was {required_size}, but another inner inner inner iterator size was {size}");
                    }
                    i
                })
            })
        });

        Tensor::new_inner(device, iter, shape)
    }
}

impl<'a, I, I2, I3, I4, I5, D: DataType> fusor_types::FromArray<5, D, I, Device> for Tensor<5, D>
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = I3, IntoIter: ExactSizeIterator>,
    I3: IntoIterator<Item = I4, IntoIter: ExactSizeIterator>,
    I4: IntoIterator<Item = I5, IntoIter: ExactSizeIterator>,
    I5: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn from_array(data: I, device: &Device) -> Self {
        let mut iter = data
            .into_iter()
            .map(|i| {
                i.into_iter()
                    .map(|i| {
                        i.into_iter()
                            .map(|i| i.into_iter().map(IntoIterator::into_iter).peekable())
                            .peekable()
                    })
                    .peekable()
            })
            .peekable();
        let mut shape = [iter.len(), 0, 0, 0, 0];
        if let Some(iter) = iter.peek_mut() {
            let size = iter.len();
            shape[1] = size;
            if let Some(iter) = iter.peek_mut() {
                let size = iter.len();
                shape[2] = size;
                if let Some(iter) = iter.peek_mut() {
                    let size = iter.len();
                    shape[3] = size;
                    if let Some(iter) = iter.peek() {
                        let size = iter.len();
                        shape[4] = size;
                    }
                }
            }
        }

        let iter = iter.flat_map(|i| {
            let size = i.len();
            let required_size = shape[1];
            if size != required_size {
                panic!("expected a rectangular matrix. The first inner iterator size was {required_size}, but another inner iterator size was {size}");
            }
            i.flat_map(|i| {
                let size = i.len();
                let required_size = shape[2];
                if size != required_size {
                    panic!("expected a rectangular matrix. The first inner inner iterator size was {required_size}, but another inner inner iterator size was {size}");
                }
                i.flat_map(|i| {
                    let size = i.len();
                    let required_size = shape[3];
                    if size != required_size {
                        panic!("expected a rectangular matrix. The first inner inner inner iterator size was {required_size}, but another inner inner inner iterator size was {size}");
                    }
                    i.flat_map(|i| {
                        let size = i.len();
                        let required_size = shape[4];
                        if size != required_size {
                            panic!("expected a rectangular matrix. The first inner inner inner inner iterator size was {required_size}, but another inner inner inner inner iterator size was {size}");
                        }
                        i
                    })
                })
            })
        });

        Tensor::new_inner(device, iter, shape)
    }
}

impl<D: DataType, const R: usize> Tensor<R, D> {
    pub fn new<T>(device: &Device, data: T) -> Self
    where
        Self: fusor_types::FromArray<R, D, T, Device>,
    {
        fusor_types::FromArray::from_array(data, device)
    }

    pub fn splat(device: &Device, value: D, shape: [usize; R]) -> Self {
        Self::from_parts(LazyTensorData::new(TensorData::new_splat(
            device, &shape, value,
        )))
    }

    /// Alias for [`Tensor::splat`]
    pub fn full(device: &Device, value: D, shape: [usize; R]) -> Self {
        Self::splat(device, value, shape)
    }

    pub(crate) fn from_parts(data: LazyTensorData) -> Self {
        debug_assert_eq!(D::WGSL_TYPE, data.info.datatype());
        Self {
            data,
            datatype: PhantomData,
        }
    }

    fn new_inner<'a, I: Iterator<Item = &'a D>>(
        device: &Device,
        data: I,
        shape: [usize; R],
    ) -> Self {
        Self::from_parts(LazyTensorData::new(TensorData::new_inner(
            device, data, &shape,
        )))
    }

    async fn as_slice_from_tensor_data(
        tensor: &TensorData,
    ) -> Result<TensorSlice<R, D, MappedBuffer>, wgpu::BufferAsyncError> {
        let buffer = tensor.buffer();
        let device = tensor.device.wgpu_device();
        let queue = tensor.device.wgpu_queue();
        let size = buffer.size();

        // Create a staging buffer for reading
        let download = device.create_buffer(&wgpu::BufferDescriptor {
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
            label: None,
        });

        // Copy data to staging buffer
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(buffer, 0, &download, 0, size);
        queue.submit(Some(encoder.finish()));

        // Map the staging buffer using map_async which correctly uses WasmNotSend
        let (sender, receiver) = futures_channel::oneshot::channel();
        download
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                _ = sender.send(result);
            });

        receiver.await.map_err(|_| wgpu::BufferAsyncError)??;

        // Get the mapped view
        let view = download.slice(..).get_mapped_range();
        Ok(TensorSlice::new(
            MappedBuffer { view },
            tensor.layout().clone(),
        ))
    }

    #[track_caller]
    pub fn materialize(&self) -> impl Future<Output = ()> + 'static {
        #[allow(unused)]
        let (data, _) = self.data.materialize();
        #[cfg(feature = "extra_assertions")]
        let caller = std::panic::Location::caller();
        let (sender, receiver) = futures_channel::oneshot::channel();
        self.device().wgpu_queue().on_submitted_work_done(|| {
            _ = sender.send(());
        });
        async move {
            let _ = receiver.await;
            #[cfg(feature = "extra_assertions")]
            {
                let mut contains_non_finite = false;
                if D::WGSL_TYPE == DataTypeEnum::F32 {
                    let data: TensorSlice<R, f32, MappedBuffer> =
                        Tensor::as_slice_from_tensor_data(&data).await.unwrap();
                    data.visit_items(|item| {
                        contains_non_finite |= !item.is_finite();
                    });
                } else if D::WGSL_TYPE == DataTypeEnum::F16 {
                    let data: TensorSlice<R, half::f16, MappedBuffer> =
                        Tensor::as_slice_from_tensor_data(&data).await.unwrap();
                    data.visit_items(|item| {
                        contains_non_finite |= !item.is_finite();
                    });
                }

                if contains_non_finite {
                    tracing::warn!(
                        "Tensor materialized at {} contains non-finite values. This may lead to unexpected behavior.",
                        caller
                    );
                }
            }
        }
    }

    /// How many kernel calls are needed to fully resolve this tensor
    pub fn count_kernels_to_resolve(&self) -> usize {
        let (_, count) = self.data.materialize();
        count
    }

    pub async fn as_slice(
        &self,
    ) -> Result<TensorSlice<R, D, MappedBuffer>, wgpu::BufferAsyncError> {
        #[cfg(not(target_arch = "wasm32"))]
        let start_time = std::time::Instant::now();
        let (tensor, _) = self.data.materialize();
        #[cfg(not(target_arch = "wasm32"))]
        tracing::trace!("Materialized tensor in {:?}", start_time.elapsed());
        #[cfg(not(target_arch = "wasm32"))]
        let start_time = std::time::Instant::now();
        let out = Self::as_slice_from_tensor_data(&tensor).await;
        #[cfg(not(target_arch = "wasm32"))]
        tracing::trace!("Downloaded tensor in {:?}", start_time.elapsed());
        out
    }

    pub async fn to_scalar(&self) -> Result<D, wgpu::BufferAsyncError> {
        let slice = self.as_slice().await?;
        Ok(slice.as_scalar())
    }

    pub fn debug_assert_real(self) -> Self
    where
        D: FloatDataType,
    {
        #[cfg(debug_assertions)]
        {
            use pollster::FutureExt as _;
            let as_slice = self.as_slice().block_on().unwrap();
            for item in as_slice.as_slice() {
                assert!(item.is_finite(), "Tensor contains non-finite value: {item}");
            }
        }
        self
    }

    pub(crate) fn element_wise<D2: DataType>(
        &self,
        function: ElementWiseOperation,
    ) -> Tensor<R, D2> {
        Tensor::from_parts(self.data.element_wise(function))
    }

    pub(crate) fn pair_wise(&self, other: &Self, function: PairWiseFunction) -> Self {
        // If the two tensors are the same, we can lower this to a cheaper element wise operation
        if self.data.key == other.data.key {
            return self.element_wise(ElementWiseOperation::new(
                self.datatype(),
                self.key(),
                function.lower_to_element_wise(),
                self.shape().as_slice(),
            ));
        }

        assert_eq!(self.shape(), other.shape());
        let operation =
            PairWiseOperation::new(function, self.data.key, other.data.key, self.shape());
        Self::from_parts(self.data.pair_wise(operation))
    }

    pub(crate) fn add_mat_mul(&self, other: &Self, parameters: Option<MatMulParams>) -> Self {
        let operation = MatMulOperation::new(
            self.datatype(),
            self.data.key,
            other.data.key,
            self.shape(),
            other.shape(),
            parameters,
        );

        Self::from_parts(self.data.mat_mul(operation))
    }

    pub(crate) fn add_q_mat_mul(&self, other: &QMatrix) -> Self {
        let operation =
            QMatMulOperation::new(self.datatype(), self.shape(), self.data.key, other.clone());

        Self::from_parts(self.data.q_mat_mul(operation))
    }

    pub(crate) fn add_resize<const R2: usize>(&self, op: ResizeOperation) -> Tensor<R2, D> {
        Tensor {
            data: self.data.resize(op),
            datatype: PhantomData,
        }
    }

    pub(crate) fn add_slice_assign(&self, other: &Self, slices: [Range<usize>; R]) -> Self {
        let input_shape: Box<[usize]> = self.shape().to_vec().into_boxed_slice();
        let op =
            SliceAssignOperation::new(self.data.key, other.data.key, slices.into(), input_shape);
        Self::from_parts(self.data.slice_assign(op))
    }

    pub(crate) fn add_index_select(&self, dimension: usize, indexes: &Tensor<1, u32>) -> Self {
        let op = IndexSelectOperation::new(
            self.data.key,
            indexes.data.key,
            self.datatype(),
            dimension,
            self.shape(),
            indexes.shape(),
        );
        Self::from_parts(self.data.index_select(op))
    }

    pub(crate) fn reduce<const OUT: usize>(
        &self,
        function: ReduceFunction,
        dim: impl Dim<R>,
    ) -> Tensor<OUT, D> {
        Tensor {
            data: self.data.reduce(ReduceOperation::new(
                self.data.key,
                function,
                dim.resolve(),
                self.shape(),
            )),
            datatype: PhantomData,
        }
    }

    pub(crate) fn add_map_layout<const R2: usize>(&self, op: MapLayoutOperation) -> Tensor<R2, D> {
        Tensor::from_parts(self.data.map_layout(op))
    }

    pub(crate) fn key(&self) -> NodeIndex {
        self.data.key
    }

    pub fn shape(&self) -> &[usize; R] {
        let shape = self.data.info.shape();
        match shape.try_into() {
            Ok(shape) => shape,
            Err(_) => {
                panic!("Internal error. Expected a tensor of rank {R}, found shape: {shape:?}")
            }
        }
    }

    pub fn rank(&self) -> usize {
        self.data.info.rank()
    }

    pub fn datatype(&self) -> DataTypeEnum {
        self.data.info.datatype()
    }

    pub fn device(&self) -> &Device {
        &self.data.device
    }

    pub fn graphvis(&self) -> Graph {
        self.data.graphvis()
    }

    pub(crate) fn data(&self) -> &LazyTensorData {
        &self.data
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_tensor_slice() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);

    let slice = tensor.slice([0..2, 0..1]);
    let as_slice = slice.as_slice().await.unwrap();
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice.get([0, 1]), None);
    assert_eq!(as_slice[[1, 0]], 3.);
    assert_eq!(as_slice.get([1, 1]), None);
    assert_eq!(as_slice.get([2, 0]), None);
    assert_eq!(as_slice.get([2, 1]), None);
}

/// A buffer that has been mapped for reading. Wraps a wgpu BufferView and provides
/// access to its mapped contents.
pub struct MappedBuffer {
    view: wgpu::BufferView,
}

impl std::ops::Deref for MappedBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.view.as_ref()
    }
}

pub(crate) fn padded_tensor_size(size: u64) -> u64 {
    // Valid vulkan usage is
    // 1. buffer size must be a multiple of COPY_BUFFER_ALIGNMENT.
    // 2. buffer size must be greater than 0.
    // Therefore we round the value up to the nearest multiple, and ensure it's at least COPY_BUFFER_ALIGNMENT.
    let align_mask = COPY_BUFFER_ALIGNMENT - 1;

    ((size + align_mask) & !align_mask).max(COPY_BUFFER_ALIGNMENT)
}

#[cfg(test)]
#[tokio::test]
async fn test_tensor_compare() {
    let device = Device::test_instance();

    let data = [
        [[1., 2.], [1., 2.]],
        [[3., 4.], [3., 4.]],
        [[5., 6.], [5., 6.]],
        [[7., 8.], [7., 8.]],
        [[9., 10.], [9., 10.]],
        [[11., 12.], [11., 12.]],
    ];
    let tensor = Tensor::new(&device, &data);

    let slice = tensor.slice([0..2, 0..1, 0..1]);
    let as_slice = slice.as_slice().await.unwrap();
    assert_eq!(as_slice, as_slice);

    let other_slice = tensor.slice([0..1, 0..1, 0..1]);
    let other_as_slice = other_slice.as_slice().await.unwrap();
    assert!(as_slice != other_as_slice);

    let other_slice = tensor.slice([1..3, 0..1, 0..1]);
    let other_as_slice = other_slice.as_slice().await.unwrap();
    assert!(as_slice != other_as_slice);
}

#[cfg(test)]
#[tokio::test]
async fn test_tensor() {
    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.], [5., 6.]];
    let tensor = Tensor::new(&device, &data);
    let as_slice = tensor.as_slice().await.unwrap();
    assert_eq!(as_slice[[0, 0]], 1.);
    assert_eq!(as_slice[[0, 1]], 2.);
    assert_eq!(as_slice[[1, 0]], 3.);
    assert_eq!(as_slice[[1, 1]], 4.);
    assert_eq!(as_slice[[2, 0]], 5.);
    assert_eq!(as_slice[[2, 1]], 6.);
}

#[cfg(test)]
#[tokio::test]
async fn test_zeros_f16() {
    let device = Device::test_instance();

    let tensor: Tensor<2, half::f16> = Tensor::zeros(&device, [2, 2]);

    let as_slice = tensor.as_slice().await.unwrap();
    assert_eq!(as_slice[[0, 0]], half::f16::ZERO);
    assert_eq!(as_slice[[0, 1]], half::f16::ZERO);
    assert_eq!(as_slice[[1, 0]], half::f16::ZERO);
    assert_eq!(as_slice[[1, 1]], half::f16::ZERO);
}
