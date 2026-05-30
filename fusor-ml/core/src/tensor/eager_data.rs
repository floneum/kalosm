use std::{ops::Range, sync::Arc};

use crate::{Device, Layout};

use super::{DataType, DataTypeEnum, TensorLayoutInfo, padded_tensor_size};

#[derive(Clone, Debug)]
pub(crate) struct TensorData {
    pub(crate) device: Device,
    pub(crate) buffer: Arc<wgpu::Buffer>,
    pub(crate) info: TensorLayoutInfo,
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

    pub(crate) fn new_inner<'a, D: DataType, I: Iterator<Item = &'a D>>(
        device: &Device,
        data: I,
        shape: &[usize],
    ) -> Self {
        let size = size_of::<D>() as u64 * shape.iter().copied().product::<usize>() as u64;
        let mut bytes = Vec::with_capacity(size as usize);
        for value in data {
            bytes.extend_from_slice(bytemuck::bytes_of(value));
        }
        let buffer = device.create_buffer_init(
            &bytes,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );

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
}
