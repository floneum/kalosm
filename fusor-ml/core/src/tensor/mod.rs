use std::{
    fmt::{Debug, Display},
    ops::Range,
};

#[cfg(feature = "graphvis")]
use tabbycat::Graph;
use wgpu::COPY_BUFFER_ALIGNMENT;

use crate::{
    Device, Layout, ReduceFunction, ReduceOperation,
    compute_graph::NodeIndex,
    nary_wise::{ElementwiseOperation, NaryExpr, NaryFunction},
    quantized::QMatrix,
    slice_assign::SliceAssignOperation,
};

pub use fusor_types::TensorSlice;

mod eager_data;
mod layout_info;
mod lazy_data;
mod sampling;
mod traits;

pub use traits::{DataType, DataTypeEnum, FloatDataType};

pub(crate) use eager_data::TensorData;
pub(crate) use layout_info::{TensorInfo, TensorLayoutInfo};
pub(crate) use lazy_data::LazyTensorData;

pub struct Tensor {
    pub(crate) data: LazyTensorData,
}

impl Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} x {:?}", self.datatype(), self.shape())
    }
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor({} x {:?})", self.datatype(), self.shape())
    }
}

impl From<TensorData> for Tensor {
    fn from(value: TensorData) -> Self {
        Self {
            data: LazyTensorData::new(value),
        }
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

impl Tensor {
    /// Resolve the current tensor value on device and return a fresh leaf tensor
    /// that no longer carries the original compute graph history.
    pub fn detach(&self) -> Self {
        let (data, _) = self.data.materialize();
        Self {
            data: LazyTensorData::new(data),
        }
    }
}

impl<const R: usize, D, T> fusor_types::FromArray<R, D, T, Device> for Tensor
where
    D: DataType,
    T: fusor_types::IntoFlatArray<D, R>,
{
    fn from_array(data: T, device: &Device) -> Self {
        let flat = data.into_flat_array();
        Tensor::new_inner(device, flat.data.iter(), flat.shape)
    }
}

impl Tensor {
    pub fn new<D: DataType, const R: usize, T>(device: &Device, data: T) -> Self
    where
        Self: fusor_types::FromArray<R, D, T, Device>,
    {
        fusor_types::FromArray::from_array(data, device)
    }

    pub fn from_slice<D: DataType>(
        device: &Device,
        shape: impl AsRef<[usize]>,
        data: &[D],
    ) -> Self {
        let shape = shape.as_ref();
        assert_eq!(
            data.len(),
            shape.iter().product::<usize>(),
            "Data length must match shape"
        );
        Self::from_parts(LazyTensorData::new(TensorData::new_from_slice(
            device, data, shape,
        )))
    }

    /// Allocate a concrete tensor backing for `shape` without uploading
    /// initialized host data.
    ///
    /// Callers must overwrite any region before reading it. This is intended
    /// for cache backing allocations where only assigned slices become visible.
    pub fn uninit<D: DataType>(device: &Device, shape: impl AsRef<[usize]>) -> Self {
        Self::from_parts(LazyTensorData::new(TensorData::new_for_shape(
            device,
            shape.as_ref(),
            D::DATA_TYPE,
        )))
    }

    pub fn splat<D: DataType>(device: &Device, value: D, shape: impl AsRef<[usize]>) -> Self {
        Self::from_parts(LazyTensorData::new(TensorData::new_splat(
            device,
            shape.as_ref(),
            value,
        )))
    }

    /// Alias for [`Tensor::splat`]
    pub fn full<D: DataType>(device: &Device, value: D, shape: impl AsRef<[usize]>) -> Self {
        Self::splat(device, value, shape)
    }

    pub(crate) fn from_parts(data: LazyTensorData) -> Self {
        Self { data }
    }

    fn new_inner<'a, D: DataType, I: Iterator<Item = &'a D>>(
        device: &Device,
        data: I,
        shape: impl AsRef<[usize]>,
    ) -> Self {
        Self::from_parts(LazyTensorData::new(TensorData::new_inner(
            device,
            data,
            shape.as_ref(),
        )))
    }

    pub(crate) async fn as_slice_from_tensor_data<const R: usize, D: DataType>(
        tensor: &TensorData,
    ) -> Result<TensorSlice<R, D, MappedBuffer>, wgpu::BufferAsyncError> {
        let device = tensor.device.wgpu_device();
        let queue = tensor.device.wgpu_queue();
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let download = Self::enqueue_download::<R, D>(tensor, &mut encoder);
        queue.submit(Some(encoder.finish()));
        Self::map_download(tensor, download).await
    }

    fn enqueue_download<const R: usize, D: DataType>(
        tensor: &TensorData,
        encoder: &mut wgpu::CommandEncoder,
    ) -> (wgpu::Buffer, Layout) {
        assert_eq!(tensor.datatype(), D::DATA_TYPE);
        assert_eq!(tensor.layout().shape().len(), R);
        let buffer = tensor.buffer();
        let device = tensor.device.wgpu_device();
        let layout = tensor.layout();
        let element_size = tensor.datatype().element_size() as u64;
        let source_offset = layout.offset() as u64 * element_size;
        let compact_size = padded_tensor_size(layout.num_elements() as u64 * element_size);
        let dense_strides = Layout::continuous_strides(layout.shape());
        let can_copy_compact = layout.strides() == dense_strides.as_ref()
            && source_offset.is_multiple_of(COPY_BUFFER_ALIGNMENT)
            && source_offset + compact_size <= buffer.size();
        let (source_offset, size, download_layout) = if can_copy_compact {
            (
                source_offset,
                compact_size,
                Layout::contiguous(layout.shape()),
            )
        } else {
            (0, buffer.size(), layout.clone())
        };
        let download = device.create_buffer(&wgpu::BufferDescriptor {
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
            label: None,
        });
        encoder.copy_buffer_to_buffer(buffer, source_offset, &download, 0, size);
        (download, download_layout)
    }

    async fn map_download<const R: usize, D: DataType>(
        tensor: &TensorData,
        download: (wgpu::Buffer, Layout),
    ) -> Result<TensorSlice<R, D, MappedBuffer>, wgpu::BufferAsyncError> {
        let (download, layout) = download;
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
        Ok(TensorSlice::new(MappedBuffer { view }, layout))
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
                if data.datatype() == DataTypeEnum::F32 && data.layout().rank() == 1 {
                    let data: TensorSlice<1, f32, MappedBuffer> =
                        Tensor::as_slice_from_tensor_data(&data).await.unwrap();
                    data.visit_items(|item| {
                        contains_non_finite |= !item.is_finite();
                    });
                } else if data.datatype() == DataTypeEnum::F16 && data.layout().rank() == 1 {
                    let data: TensorSlice<1, half::f16, MappedBuffer> =
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
        let (data, count) = self.data.materialize();
        #[cfg(not(target_arch = "wasm32"))]
        data.device().poll_wait();
        #[cfg(target_arch = "wasm32")]
        drop(data);
        count
    }

    /// Whether fully resolving this tensor takes exactly `N` kernel calls
    pub fn resolves_in<const N: usize>(&self) -> bool {
        self.count_kernels_to_resolve() == N
    }

    pub async fn as_slice<const R: usize, D: DataType>(
        &self,
    ) -> Result<TensorSlice<R, D, MappedBuffer>, wgpu::BufferAsyncError> {
        self.assert_rank::<R>();
        self.assert_datatype::<D>();
        #[cfg(not(target_arch = "wasm32"))]
        let start_time = std::time::Instant::now();
        let (tensor, _, download) = self
            .data
            .materialize_with_tail(Self::enqueue_download::<R, D>);
        #[cfg(not(target_arch = "wasm32"))]
        tracing::trace!("Materialized tensor in {:?}", start_time.elapsed());
        #[cfg(not(target_arch = "wasm32"))]
        let start_time = std::time::Instant::now();
        let out = Self::map_download(&tensor, download).await;
        #[cfg(not(target_arch = "wasm32"))]
        tracing::trace!("Downloaded tensor in {:?}", start_time.elapsed());
        out
    }

    pub async fn to_scalar<D: DataType>(&self) -> Result<D, wgpu::BufferAsyncError> {
        let slice = self.as_slice::<0, D>().await?;
        Ok(slice.as_scalar())
    }

    pub fn debug_assert_real(self) -> Self {
        #[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
        {
            use pollster::FutureExt as _;
            if self.rank() == 1 {
                match self.datatype() {
                    DataTypeEnum::F32 => {
                        let as_slice = self.as_slice::<1, f32>().block_on().unwrap();
                        for item in as_slice.as_slice() {
                            assert!(item.is_finite(), "Tensor contains non-finite value: {item}");
                        }
                    }
                    DataTypeEnum::F16 => {
                        let as_slice = self.as_slice::<1, half::f16>().block_on().unwrap();
                        for item in as_slice.as_slice() {
                            assert!(item.is_finite(), "Tensor contains non-finite value: {item}");
                        }
                    }
                    DataTypeEnum::U32 => {}
                }
            }
        }
        self
    }

    pub(crate) fn unary_nary(&self, function: NaryFunction) -> Tensor {
        Tensor::from_parts(self.data.unary_nary(function))
    }

    pub(crate) fn unary_nary_dtype(&self, function: NaryFunction) -> Tensor {
        Tensor::from_parts(self.data.unary_nary(function))
    }

    pub(crate) fn binary_nary(&self, other: &Self, function: NaryFunction) -> Self {
        // Keep one storage input while preserving the binary expression.
        if self.data.key == other.data.key {
            let device = self.device().clone();
            let mut info = self.data.info.clone();
            info.datatype = function.output_type;
            let rank = self.shape().len();
            let nary = ElementwiseOperation {
                inputs: vec![self.data.key],
                expression: NaryExpr::Op {
                    children: vec![NaryExpr::input(0, rank), NaryExpr::input(0, rank)],
                    function,
                },
                shape: self.shape().into(),
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

    /// Quantized matrix multiply in its composed form: the activation
    /// `[.., K]` and the dequantized matrix `[N, K]` multiply over the
    /// `[.., N, K]` index space and sum along `K`. The resolver recognizes
    /// the canonical cluster and routes it to the quantized matmul kernels.
    pub(crate) fn add_q_mat_mul(&self, other: &QMatrix) -> Self {
        let in_shape = self.shape();
        let rank = in_shape.len();
        assert!(rank >= 1, "q_mat_mul requires rank >= 1");
        assert_eq!(
            in_shape[rank - 1],
            other.shape()[1],
            "q_mat_mul contraction dimensions must match: {in_shape:?} x {:?}",
            other.shape()
        );

        let datatype = self.datatype();
        let device = self.device().clone();
        let matrix_key = device.compute_graph().dequantize(other.clone(), datatype);
        let matrix = Tensor::from_parts(LazyTensorData::from_parts(
            device,
            TensorInfo::new(other.shape().into(), datatype),
            matrix_key,
        ));

        let n = other.shape()[0];
        // Index space [.., N, K]: K stays last so the reduce axis is the
        // final dimension.
        let mut index_space = in_shape.to_vec();
        index_space.insert(rank - 1, n);
        let (n_dim, k_dim) = (rank - 1, rank);

        let activation_indices: Vec<NaryExpr> = (0..rank - 1)
            .chain(std::iter::once(k_dim))
            .map(NaryExpr::DimIndex)
            .collect();
        let matrix_indices: Vec<NaryExpr> = [n_dim, k_dim].map(NaryExpr::DimIndex).to_vec();

        let product = Tensor::from_parts(self.data.nary(ElementwiseOperation {
            inputs: vec![self.key(), matrix.key()],
            expression: NaryExpr::mul(
                NaryExpr::indexed_input(0, activation_indices),
                NaryExpr::indexed_input(1, matrix_indices),
                datatype,
            ),
            shape: index_space.into(),
            output_datatype: datatype,
        }));
        product.sum(k_dim)
    }

    /// Slice assignment in its composed form: per output coordinate, read
    /// the assigned value inside the slice region and this tensor outside
    /// it. A plain elementwise op — no specialized kernel.
    pub(crate) fn add_slice_assign(
        &self,
        other: &Self,
        slices: impl Into<Box<[Range<usize>]>>,
    ) -> Self {
        let slices: Box<[Range<usize>]> = slices.into();
        assert_eq!(
            slices.len(),
            self.rank(),
            "slice_assign requires one range per dimension"
        );
        let expression = crate::slice_assign::slice_assign_expression(&slices, self.datatype());
        Self::from_parts(self.data.nary(ElementwiseOperation {
            inputs: vec![self.data.key, other.data.key],
            expression,
            shape: self.shape().into(),
            output_datatype: self.datatype(),
        }))
    }

    #[doc(hidden)]
    pub fn slice_assign_in_place(
        &self,
        slices: impl Into<Box<[Range<usize>]>>,
        value: &Self,
    ) -> Self {
        let op = SliceAssignOperation::new_in_place(self.data.key, value.data.key, slices.into());
        Self::from_parts(self.data.slice_assign(op))
    }

    pub(crate) fn reduce(&self, function: ReduceFunction, dim: usize) -> Tensor {
        Tensor::from_parts(self.data.reduce(ReduceOperation::new(
            self.data.key,
            function,
            dim,
            self.shape(),
        )))
    }

    /// Return the compute-graph node index for this tensor.
    pub fn key(&self) -> NodeIndex {
        self.data.key
    }

    pub fn shape(&self) -> &[usize] {
        self.data.info.shape()
    }

    pub fn shape_array<const R: usize>(&self) -> &[usize; R] {
        self.shape().try_into().unwrap_or_else(|_| {
            panic!(
                "Expected a tensor of rank {R}, found shape: {:?}",
                self.shape()
            )
        })
    }

    pub fn assert_rank<const R: usize>(&self) {
        assert_eq!(self.rank(), R, "unexpected tensor rank");
    }

    pub fn assert_datatype<D: DataType>(&self) {
        assert_eq!(self.datatype(), D::DATA_TYPE, "unexpected tensor dtype");
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

    #[cfg(feature = "graphvis")]
    pub fn graphvis(&self) -> Graph {
        self.data.graphvis()
    }

    pub(crate) fn data(&self) -> &LazyTensorData {
        &self.data
    }
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
