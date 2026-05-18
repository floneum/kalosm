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
    Device, Dim, FlashAttentionInputs, FlashAttentionOperation, Layout, MatMulOperation,
    MatMulParams, ReduceFunction, ReduceOperation,
    compute_graph::NodeIndex,
    map_layout::MapLayoutOperation,
    nary_wise::{NaryExpr, NaryFunction, NaryOperation},
    quantized::QMatrix,
    quantized::matmul::QMatMulOperation,
    resize::ResizeOperation,
    rms_norm::RmsNormOperation,
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
    + Send
    + Sync
    + 'static
{
    const DATA_TYPE: DataTypeEnum;

    fn zero() -> Self;
    fn one() -> Self;
}

pub trait FloatDataType: DataType {
    fn from_f32(value: f32) -> Self;

    fn is_finite(&self) -> bool;
}

impl DataType for f32 {
    const DATA_TYPE: DataTypeEnum = DataTypeEnum::F32;

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
    const DATA_TYPE: DataTypeEnum = DataTypeEnum::F16;

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
    const DATA_TYPE: DataTypeEnum = DataTypeEnum::U32;

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

    pub(crate) fn nary(&self, nary: NaryOperation) -> Self {
        let device = self.device.clone();
        let info = self.info.clone();
        let key = device.compute_graph().create_nary(nary);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn unary_nary(&self, function: NaryFunction) -> Self {
        let device = self.device.clone();
        let mut info = self.info.clone();
        info.datatype = function.output_type;
        let rank = info.rank();
        let nary = NaryOperation {
            inputs: vec![self.key],
            expression: NaryExpr::Op {
                children: vec![NaryExpr::input(0, rank)],
                function,
            },
            shape: info.shape().into(),
            output_datatype: info.datatype,
        };
        let key = device.compute_graph().create_nary(nary);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn binary_nary(
        &self,
        other_key: NodeIndex,
        function: NaryFunction,
        shape: &[usize],
    ) -> Self {
        let device = self.device.clone();
        let mut info = self.info.clone();
        info.datatype = function.output_type;
        let rank = shape.len();
        let nary = NaryOperation {
            inputs: vec![self.key, other_key],
            expression: NaryExpr::Op {
                children: vec![NaryExpr::input(0, rank), NaryExpr::input(1, rank)],
                function,
            },
            shape: shape.into(),
            output_datatype: info.datatype,
        };
        let key = device.compute_graph().create_nary(nary);

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
        let input_shape = self.info.shape();
        let new_shape: Box<[usize]> = input_shape
            .iter()
            .enumerate()
            .filter_map(|(i, x)| (i != dim).then_some(*x))
            .collect();
        // Short-circuit empty inputs: any zero-sized dim makes the input
        // empty. Reducing along the zero axis yields the reduction's identity
        // value (e.g. 0 for sum, 1 for product); reducing along a different
        // axis preserves the zero in the output. Both cases are produced by
        // splatting the identity at `new_shape` — the kernel would otherwise
        // panic on `iterations > 0` in `tile-ir/.../reduce.rs`.
        if input_shape.contains(&0) {
            let data =
                TensorData::new_splat_scalar(&device, &new_shape, function.function.initial_value);
            return Self::new(data);
        }
        info = TensorInfo::new(new_shape, info.datatype());
        let key = device.compute_graph().create_reduce(function);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn rms_norm(&self, function: RmsNormOperation) -> Self {
        let device = self.device.clone();
        let info = self.info.clone();
        let key = device.compute_graph().create_rms_norm(function);

        Self::from_parts(device, info, key)
    }

    pub(crate) fn flash_attention(&self, function: FlashAttentionOperation) -> Self {
        let device = self.device.clone();
        let mut info = self.info.clone();
        info.shape = function.out_shape.clone();
        let key = device.compute_graph().create_flash_attention(function);

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

    pub(crate) fn materialize(&self) -> (TensorData, usize) {
        let result = self.device.compute_graph().resolve(self.key);
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
        // Empty tensors (any dim is 0) have no valid indices, so the bounds
        // check below would compare strides * (dim - 1) against a buffer sized
        // for zero elements and spuriously fail.
        let is_empty = layout.shape().contains(&0);
        assert!(
            is_empty
                || layout.offset()
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

    pub(crate) fn new_splat_scalar(
        device: &Device,
        shape: &[usize],
        scalar: crate::nary_wise::NaryScalar,
    ) -> Self {
        match scalar {
            crate::nary_wise::NaryScalar::F32(v) => Self::new_splat(device, shape, v),
            crate::nary_wise::NaryScalar::F16(v) => Self::new_splat(device, shape, v),
            crate::nary_wise::NaryScalar::U32(v) => Self::new_splat(device, shape, v),
        }
    }

    pub(crate) fn new_splat<D: DataType>(device: &Device, shape: &[usize], data: D) -> Self {
        let datatype = D::DATA_TYPE;
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

        Self::new_from_buffer(device, buffer, shape, D::DATA_TYPE)
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

    /// Build a `tile-ir` tensor ref over this tensor's buffer with the
    /// kernel-side rank-1 linear storage layout (the kernel's `Meta` struct
    /// already encodes any offset/stride).
    pub(crate) fn as_kernel_tensor_ref(&self) -> fusor_tile_ir::KernelTensorRef<Arc<wgpu::Buffer>> {
        fusor_tile_ir::KernelTensorRef::new(
            self.buffer.clone(),
            fusor_tile_ir_kernels::linear_storage_layout(),
        )
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

impl<const R: usize, D> Tensor<R, D> {
    /// Resolve the current tensor value on device and return a fresh leaf tensor
    /// that no longer carries the original compute graph history.
    pub fn detach(&self) -> Self {
        let (data, _) = self.data.materialize();
        Self {
            data: LazyTensorData::new(data),
            datatype: PhantomData,
        }
    }
}

impl<const R: usize, D, T> fusor_types::FromArray<R, D, T, Device> for Tensor<R, D>
where
    D: DataType,
    T: fusor_types::IntoFlatArray<D, R>,
{
    fn from_array(data: T, device: &Device) -> Self {
        let flat = data.into_flat_array();
        Tensor::new_inner(device, flat.data.iter(), flat.shape)
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
        debug_assert_eq!(D::DATA_TYPE, data.info.datatype());
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
        #[cfg(not(target_arch = "wasm32"))]
        tensor.device.poll_wait();

        receiver.await.map_err(|_| wgpu::BufferAsyncError)??;

        // Get the mapped view
        let view = download.slice(..).get_mapped_range();
        Ok(TensorSlice::new(
            MappedBuffer { view },
            tensor.layout().clone(),
        ))
    }

    /// Synchronously dispatch and wait for GPU completion using device.poll().
    /// More efficient than the async version for benchmarking since it avoids
    /// the on_submitted_work_done callback overhead.
    pub fn materialize_sync(&self) {
        self.data.materialize();
        self.device().poll_wait();
    }

    #[track_caller]
    pub fn materialize(&self) -> impl Future<Output = ()> + 'static {
        let data = self.data.clone();
        let device = self.device().clone();
        #[cfg(feature = "extra_assertions")]
        let caller = std::panic::Location::caller();
        async move {
            #[cfg_attr(not(feature = "extra_assertions"), allow(unused_variables))]
            let (data, _) = data.materialize();
            #[cfg(not(target_arch = "wasm32"))]
            device.poll_wait();
            #[cfg(target_arch = "wasm32")]
            {
                let (sender, receiver) = futures_channel::oneshot::channel();
                device.wgpu_queue().on_submitted_work_done(|| {
                    _ = sender.send(());
                });
                let _ = receiver.await;
            }
            #[cfg(feature = "extra_assertions")]
            {
                let mut contains_non_finite = false;
                if D::DATA_TYPE == DataTypeEnum::F32 {
                    let data: TensorSlice<R, f32, MappedBuffer> =
                        Tensor::as_slice_from_tensor_data(&data).await.unwrap();
                    data.visit_items(|item| {
                        contains_non_finite |= !item.is_finite();
                    });
                } else if D::DATA_TYPE == DataTypeEnum::F16 {
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

    pub(crate) fn unary_nary<D2: DataType>(&self, function: NaryFunction) -> Tensor<R, D2> {
        Tensor::from_parts(self.data.unary_nary(function))
    }

    pub(crate) fn binary_nary(&self, other: &Self, function: NaryFunction) -> Self {
        // Keep one storage input while preserving the binary expression.
        if self.data.key == other.data.key {
            let device = self.device().clone();
            let mut info = self.data.info.clone();
            info.datatype = function.output_type;
            let rank = self.shape().len();
            let nary = NaryOperation {
                inputs: vec![self.data.key],
                expression: NaryExpr::Op {
                    children: vec![NaryExpr::input(0, rank), NaryExpr::input(0, rank)],
                    function,
                },
                shape: self.shape().as_slice().into(),
                output_datatype: info.datatype,
            };
            let key = device.compute_graph().create_nary(nary);
            return Self::from_parts(LazyTensorData::from_parts(device, info, key));
        }

        assert_eq!(self.shape(), other.shape());
        Self::from_parts(
            self.data
                .binary_nary(other.data.key, function, self.shape()),
        )
    }

    pub(crate) fn add_mat_mul(&self, other: &Self, parameters: Option<MatMulParams>) -> Self {
        let operation = MatMulOperation::new(
            self.datatype(),
            self.data.key,
            other.data.key,
            self.shape(),
            other.shape(),
            parameters,
            &self.data.device,
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

    #[doc(hidden)]
    pub fn slice_assign_in_place(&self, slices: [Range<usize>; R], value: &Self) -> Self {
        let input_shape: Box<[usize]> = self.shape().to_vec().into_boxed_slice();
        let op = SliceAssignOperation::new_in_place(
            self.data.key,
            value.data.key,
            slices.into(),
            input_shape,
        );
        Self::from_parts(self.data.slice_assign(op))
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

    /// Return the compute-graph node index for this tensor.
    pub fn key(&self) -> NodeIndex {
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

    pub(crate) fn try_rms_norm_direct<const W: usize>(
        &self,
        weight: &Tensor<W, D>,
        bias: Option<&Tensor<W, D>>,
        eps: f32,
    ) -> Option<Self> {
        if D::DATA_TYPE != DataTypeEnum::F32 {
            return None;
        }
        let operation = RmsNormOperation::new(
            self.data.key,
            weight.data.key,
            bias.map(|bias| bias.data.key),
            self.shape(),
            eps,
        );
        Some(Self::from_parts(self.data.rms_norm(operation)))
    }

    pub(crate) fn try_rms_norm_residual_direct<const W: usize>(
        &self,
        residual: &Self,
        weight: &Tensor<W, D>,
        bias: Option<&Tensor<W, D>>,
        eps: f32,
    ) -> Option<Self> {
        if D::DATA_TYPE != DataTypeEnum::F32 || self.shape() != residual.shape() {
            return None;
        }
        let operation = RmsNormOperation::new_with_residual(
            self.data.key,
            residual.data.key,
            weight.data.key,
            bias.map(|bias| bias.data.key),
            self.shape(),
            eps,
        );
        Some(Self::from_parts(self.data.rms_norm(operation)))
    }

    pub(crate) fn try_flash_attention_direct(
        &self,
        k: &Self,
        v: &Self,
        scale: f32,
        mask: Option<&Tensor<2, D>>,
    ) -> Option<Self> {
        if R != 4 || !matches!(D::DATA_TYPE, DataTypeEnum::F32 | DataTypeEnum::F16) {
            return None;
        }
        // The streaming flash attention kernel emits a separate
        // monomorphization per hardware subgroup width and relies on
        // `subgroup_reduce_*`, so it can only target devices that expose a
        // fixed, supported subgroup size. wgpu doesn't surface
        // `requiredSubgroupSize`, so devices that report a variable subgroup
        // range (e.g. Mesa lavapipe) fall through to the composite path here.
        let device = &self.data.device;
        if !device.subgroups_supported()
            || device.min_subgroup_size() != device.max_subgroup_size()
            || !matches!(device.min_subgroup_size(), 4 | 8 | 16 | 32 | 64)
        {
            return None;
        }
        let q_shape = self.shape();
        let k_shape = k.shape();
        // The streaming and decode_small flash-attention kernels both
        // miscompile on at least one Windows wgpu backend (WARP / DX12)
        // when the input is small: the conformance regression at
        // `flash_attention_matches_cpu_reference_on_varied_shapes`
        // deterministically reports `actual=-0.34986264 expected=-0.5393032`,
        // which corresponds to a `subgroup_reduce_sum` that drops the
        // contributions from valid lanes past lane 0. The streaming kernel
        // pads lanes-past-`kv_seq_len` with NEG_MAX and `decode_small`
        // pads with zero, and both rely on the reduction to ignore them;
        // the Windows shader compiler does not.
        //
        // Route any small-`kv_seq_len` shape through the composite
        // `mat_mul + softmax` path which has no subgroup dependence. 32 is
        // a safe threshold (the largest hardware subgroup the streaming
        // kernel monomorphises against) — production-sized attention has
        // `kv_seq_len` well above that and still gets the fast path.
        const MIN_DIRECT_KV_SEQ: usize = 32;
        if k_shape[2] < MIN_DIRECT_KV_SEQ {
            return None;
        }
        let v_shape = v.shape();
        if q_shape[0] != k_shape[0]
            || q_shape[0] != v_shape[0]
            || k_shape[1] != v_shape[1]
            || k_shape[2] != v_shape[2]
            || q_shape[3] != k_shape[3]
            || q_shape[3] != v_shape[3]
            || q_shape[0] == 0
            || q_shape[1] == 0
            || q_shape[2] == 0
            || k_shape[1] == 0
            || !q_shape[1].is_multiple_of(k_shape[1])
            || q_shape[3] == 0
            || k_shape[2] == 0
        {
            return None;
        }
        if let Some(mask) = mask
            && *mask.shape() != [q_shape[2], k_shape[2]]
        {
            return None;
        }
        let batch = u32::try_from(q_shape[0]).ok()?;
        let num_heads = u32::try_from(q_shape[1]).ok()?;
        let q_seq_len = u32::try_from(q_shape[2]).ok()?;
        let head_dim = u32::try_from(q_shape[3]).ok()?;
        let row_dispatch = batch.checked_mul(num_heads)?.checked_mul(q_seq_len)?;
        let x_dispatch = head_dim.div_ceil(8);
        let max_dispatch = self
            .data
            .device
            .limits()
            .max_compute_workgroups_per_dimension;
        if x_dispatch > max_dispatch || row_dispatch > max_dispatch {
            return None;
        }

        let operation = FlashAttentionOperation::new(FlashAttentionInputs {
            q: self.data.key,
            k: k.data.key,
            v: v.data.key,
            mask: mask.map(|mask| mask.data.key),
            q_shape,
            k_shape,
            v_shape,
            scale,
            input_dtype: D::DATA_TYPE,
        });
        Some(Self::from_parts(self.data.flash_attention(operation)))
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

impl Tensor<1, f32> {
    pub async fn try_sample_mirostat2_token_q_mat(
        &self,
        matrix: &QMatrix,
        sampler: &mut crate::top_k::GpuMirostat2Sampler,
        previous_tokens: &[u32],
        params: crate::top_k::GpuMirostat2SamplerParams,
    ) -> Result<Option<u32>, wgpu::BufferAsyncError> {
        let (input, _) = self.data.materialize();
        crate::top_k::qmat_mirostat2_sample_token_to_host(
            &input,
            matrix,
            sampler,
            previous_tokens,
            params,
        )
        .await
    }

    pub async fn sample_mirostat2_token(
        &self,
        sampler: &mut crate::top_k::GpuMirostat2Sampler,
        previous_tokens: &[u32],
        params: crate::top_k::GpuMirostat2SamplerParams,
    ) -> Result<u32, wgpu::BufferAsyncError> {
        let (input, _) = self.data.materialize();
        if let Some(token) =
            crate::top_k::mirostat2_sample_token_to_host(&input, sampler, previous_tokens, params)
                .await?
        {
            return Ok(token);
        }

        let (ids, _) = self.top_k_pairs(params.top_k).await?;
        Ok(ids.first().copied().unwrap_or_default())
    }

    pub async fn top_k_pairs(
        &self,
        k: usize,
    ) -> Result<(Vec<u32>, Vec<f32>), wgpu::BufferAsyncError> {
        if k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let (input, _) = self.data.materialize();
        if input.datatype() != DataTypeEnum::F32 || input.layout().rank() != 1 {
            return cpu_top_k_pairs_from_tensor_data(&input, k).await;
        }

        let input_len = input.layout().shape()[0];
        let k = k.min(input_len);
        if k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let chunks = input_len.div_ceil(crate::top_k::TOP_K_CHUNK);
        let mut candidate_count = k
            .div_ceil(chunks)
            .max(crate::top_k::MIN_TOP_K_CANDIDATES_PER_CHUNK)
            .min(k)
            .min(crate::top_k::TOP_K_CHUNK);

        loop {
            let output_per_chunk = if candidate_count >= crate::top_k::TOP_K_CHUNK {
                crate::top_k::TOP_K_CHUNK
            } else {
                candidate_count + 1
            };
            let mut encoder = input.device().wgpu_device().create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("top_k_pairs encoder"),
                },
            );
            let Some((ids, values)) = crate::top_k::chunk_top_k_pair_data_with_encoder(
                &input,
                candidate_count,
                output_per_chunk,
                Some(&mut encoder),
            ) else {
                return cpu_top_k_pairs_from_tensor_data(&input, k).await;
            };
            if candidate_count >= crate::top_k::TOP_K_CHUNK {
                let Some((ids, values)) =
                    crate::top_k::merge_sorted_chunk_top_k_pair_data_with_encoder(
                        &ids,
                        &values,
                        crate::top_k::MergeSortedChunkTopKParams {
                            chunks,
                            chunk_len: crate::top_k::TOP_K_CHUNK,
                            chunk_stride: crate::top_k::TOP_K_CHUNK,
                            input_len,
                            k,
                        },
                        Some(&mut encoder),
                    )
                else {
                    return cpu_top_k_pairs_from_tensor_data(&input, k).await;
                };
                input.device().wgpu_queue().submit(Some(encoder.finish()));
                let ids = Tensor::<1, u32>::as_slice_from_tensor_data(&ids).await?;
                let values = Tensor::<1, f32>::as_slice_from_tensor_data(&values).await?;
                return Ok((ids.as_slice().to_vec(), values.as_slice().to_vec()));
            }
            let Some((merged_ids, merged_values)) =
                crate::top_k::merge_sorted_chunk_top_k_pair_data_with_encoder(
                    &ids,
                    &values,
                    crate::top_k::MergeSortedChunkTopKParams {
                        chunks,
                        chunk_len: candidate_count,
                        chunk_stride: output_per_chunk,
                        input_len,
                        k,
                    },
                    Some(&mut encoder),
                )
            else {
                return cpu_top_k_pairs_from_tensor_data(&input, k).await;
            };
            input.device().wgpu_queue().submit(Some(encoder.finish()));
            let merged_ids = Tensor::<1, u32>::as_slice_from_tensor_data(&merged_ids).await?;
            let merged_values = Tensor::<1, f32>::as_slice_from_tensor_data(&merged_values).await?;
            let chunk_values = Tensor::<1, f32>::as_slice_from_tensor_data(&values).await?;
            let exact = top_k_chunk_bounds_prove_exact(
                merged_values.as_slice(),
                chunk_values.as_slice(),
                k,
                chunks,
                candidate_count,
                output_per_chunk,
            );
            if exact {
                return Ok((
                    merged_ids.as_slice().to_vec(),
                    merged_values.as_slice().to_vec(),
                ));
            }

            let ids = Tensor::<1, u32>::as_slice_from_tensor_data(&ids).await?;
            if let Some(top) = top_k_from_chunk_candidates(
                ids.as_slice(),
                chunk_values.as_slice(),
                k,
                input_len,
                chunks,
                candidate_count,
                output_per_chunk,
            ) {
                return Ok(top.into_iter().unzip());
            }

            if candidate_count >= crate::top_k::TOP_K_CHUNK {
                return cpu_top_k_pairs_from_tensor_data(&input, k).await;
            }
            candidate_count = (candidate_count * 2).min(crate::top_k::TOP_K_CHUNK);
        }
    }
}

fn top_k_chunk_bounds_prove_exact(
    top_values: &[f32],
    chunk_values: &[f32],
    k: usize,
    chunks: usize,
    candidate_count: usize,
    output_per_chunk: usize,
) -> bool {
    let Some(&threshold) = top_values.get(k.saturating_sub(1)) else {
        return !chunk_bounds(chunk_values, chunks, candidate_count, output_per_chunk)
            .any(|bound| bound.is_finite());
    };
    if !threshold.is_finite() {
        return !chunk_bounds(chunk_values, chunks, candidate_count, output_per_chunk)
            .any(|bound| bound.is_finite());
    }
    !chunk_bounds(chunk_values, chunks, candidate_count, output_per_chunk)
        .any(|bound| bound.is_finite() && bound >= threshold)
}

fn chunk_bounds(
    values: &[f32],
    chunks: usize,
    candidate_count: usize,
    output_per_chunk: usize,
) -> impl Iterator<Item = f32> + '_ {
    (0..chunks).filter_map(move |chunk| {
        let index = chunk
            .checked_mul(output_per_chunk)?
            .checked_add(candidate_count)?;
        values.get(index).copied()
    })
}

fn top_k_from_chunk_candidates(
    ids: &[u32],
    values: &[f32],
    k: usize,
    input_len: usize,
    chunks: usize,
    candidate_count: usize,
    output_per_chunk: usize,
) -> Option<Vec<(u32, f32)>> {
    let mut candidates = Vec::with_capacity(chunks * candidate_count);
    let mut bounds = Vec::with_capacity(chunks);

    for chunk in 0..chunks {
        let base = chunk * output_per_chunk;
        for rank in 0..candidate_count.min(output_per_chunk) {
            let index = base + rank;
            let logit = values[index];
            if logit.is_finite() && (ids[index] as usize) < input_len {
                candidates.push((ids[index], logit));
            }
        }
        if candidate_count < crate::top_k::TOP_K_CHUNK {
            let index = base + candidate_count;
            let valid = (ids[index] as usize) < input_len;
            bounds.push(valid.then_some(values[index]));
        }
    }

    candidates.sort_unstable_by_key(|(token_id, _)| *token_id);
    let top = fold_top_k_pairs(candidates, k);
    let Some((_, threshold)) = top.get(k.saturating_sub(1)).copied() else {
        if bounds.iter().flatten().any(|bound| bound.is_finite()) {
            return None;
        }
        return Some(top);
    };

    if candidate_count < crate::top_k::TOP_K_CHUNK
        && bounds
            .iter()
            .flatten()
            .any(|bound| bound.is_finite() && *bound >= threshold)
    {
        return None;
    }

    Some(top)
}

fn fold_top_k_pairs(candidates: impl IntoIterator<Item = (u32, f32)>, k: usize) -> Vec<(u32, f32)> {
    let mut top = Vec::<(u32, f32)>::with_capacity(k);
    for (token_id, logit) in candidates {
        if !logit.is_finite() {
            continue;
        }
        if top.len() == k {
            let Some((last_token_id, last_logit)) = top.last().copied() else {
                continue;
            };
            if logit > last_logit || (logit == last_logit && token_id > last_token_id) {
                top.truncate(k - 1);
            } else {
                continue;
            }
        }
        let insert = top.partition_point(|(existing_id, value)| {
            *value > logit || (*value == logit && *existing_id > token_id)
        });
        top.insert(insert, (token_id, logit));
    }
    top
}

async fn cpu_top_k_pairs_from_tensor_data(
    input: &TensorData,
    k: usize,
) -> Result<(Vec<u32>, Vec<f32>), wgpu::BufferAsyncError> {
    if k == 0 {
        return Ok((Vec::new(), Vec::new()));
    }

    let values = Tensor::<1, f32>::as_slice_from_tensor_data(input).await?;
    let top = fold_top_k_pairs(
        values
            .as_slice()
            .iter()
            .copied()
            .enumerate()
            .map(|(token_id, logit)| (token_id as u32, logit)),
        k,
    );
    Ok(top.into_iter().unzip())
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
