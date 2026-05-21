//! Construction operations that work on both CPU and GPU backends.

use crate::{Device, SimdElement, Tensor};
use fusor_core::DataType;
use fusor_types::FromArray;

impl<const R: usize, D, T> FromArray<R, D, T, Device> for Tensor<R, D>
where
    D: DataType + SimdElement + Default,
    T: fusor_types::IntoFlatArray<D, R>,
{
    fn from_array(data: T, device: &Device) -> Self {
        match device {
            Device::Cpu => Tensor::Cpu(FromArray::from_array(data, &())),
            Device::Gpu(gpu_device) => Tensor::Gpu(FromArray::from_array(data, gpu_device)),
        }
    }
}

impl<const R: usize, D> Tensor<R, D>
where
    D: SimdElement + DataType + Default,
{
    /// Create a tensor from data on the specified device.
    ///
    /// This method accepts nested arrays/slices matching the tensor rank.
    /// For example:
    /// - Rank 1: `Tensor::new(&device, &[1.0, 2.0, 3.0])`
    /// - Rank 2: `Tensor::new(&device, &[[1.0, 2.0], [3.0, 4.0]])`
    pub fn new<T>(device: &Device, data: T) -> Self
    where
        Self: FromArray<R, D, T, Device>,
    {
        FromArray::from_array(data, device)
    }

    /// Create a tensor from a slice of data with the given shape.
    ///
    /// The data must have exactly as many elements as the shape specifies.
    pub fn from_slice(device: &Device, shape: [usize; R], data: &[D]) -> Self {
        let total_elements: usize = shape.iter().product();
        assert_eq!(data.len(), total_elements, "Data length must match shape");
        match device {
            Device::Cpu => Tensor::Cpu(fusor_cpu::Tensor::from_slice(shape, data)),
            Device::Gpu(gpu_device) => {
                Tensor::Gpu(fusor_core::Tensor::from_slice(gpu_device, shape, data))
            }
        }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(device: &Device, shape: [usize; R]) -> Self {
        Self::splat(device, D::default(), shape)
    }

    /// Create a tensor filled with zeros that has the same shape as this tensor.
    pub fn zeros_like(&self) -> Self {
        let shape = self.shape();
        Self::splat(&self.device(), D::default(), shape)
    }

    /// Create a tensor filled with a specific value.
    pub fn splat(device: &Device, value: D, shape: [usize; R]) -> Self {
        match device {
            Device::Cpu => {
                let data = vec![value; shape.iter().product()];
                Tensor::Cpu(fusor_cpu::Tensor::from_slice(shape, &data))
            }
            Device::Gpu(gpu_device) => {
                Tensor::Gpu(fusor_core::Tensor::splat(gpu_device, value, shape))
            }
        }
    }

    /// Create a tensor filled with a specific value (alias for splat).
    pub fn full(device: &Device, shape: [usize; R], value: D) -> Self {
        Self::splat(device, value, shape)
    }
}
