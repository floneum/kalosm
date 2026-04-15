//! Construction operations that work on both CPU and GPU backends.

use crate::{Device, SimdElement, Tensor};
use fusor_core::DataType;
use fusor_types::FromArray;

impl<D: DataType + SimdElement + Default> FromArray<0, D, (), Device> for Tensor<0, D> {
    fn from_array(data: (), device: &Device) -> Self {
        match device {
            Device::Cpu => Tensor::Cpu(FromArray::from_array(data, &())),
            Device::Gpu(gpu_device) => Tensor::Gpu(FromArray::from_array(data, gpu_device)),
        }
    }
}

impl<'a, I, D: DataType + SimdElement + Default + Copy> FromArray<1, D, I, Device> for Tensor<1, D>
where
    I: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn from_array(data: I, device: &Device) -> Self {
        match device {
            Device::Cpu => Tensor::Cpu(FromArray::from_array(data, &())),
            Device::Gpu(gpu_device) => Tensor::Gpu(FromArray::from_array(data, gpu_device)),
        }
    }
}

impl<'a, I, I2, D: DataType + SimdElement + Default + Copy> FromArray<2, D, I, Device>
    for Tensor<2, D>
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn from_array(data: I, device: &Device) -> Self {
        match device {
            Device::Cpu => Tensor::Cpu(FromArray::from_array(data, &())),
            Device::Gpu(gpu_device) => Tensor::Gpu(FromArray::from_array(data, gpu_device)),
        }
    }
}

impl<'a, I, I2, I3, D: DataType + SimdElement + Default + Copy> FromArray<3, D, I, Device>
    for Tensor<3, D>
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = I3, IntoIter: ExactSizeIterator>,
    I3: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn from_array(data: I, device: &Device) -> Self {
        match device {
            Device::Cpu => Tensor::Cpu(FromArray::from_array(data, &())),
            Device::Gpu(gpu_device) => Tensor::Gpu(FromArray::from_array(data, gpu_device)),
        }
    }
}

impl<'a, I, I2, I3, I4, D: DataType + SimdElement + Default + Copy> FromArray<4, D, I, Device>
    for Tensor<4, D>
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = I3, IntoIter: ExactSizeIterator>,
    I3: IntoIterator<Item = I4, IntoIter: ExactSizeIterator>,
    I4: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn from_array(data: I, device: &Device) -> Self {
        match device {
            Device::Cpu => Tensor::Cpu(FromArray::from_array(data, &())),
            Device::Gpu(gpu_device) => Tensor::Gpu(FromArray::from_array(data, gpu_device)),
        }
    }
}

impl<'a, I, I2, I3, I4, I5, D: DataType + SimdElement + Default + Copy> FromArray<5, D, I, Device>
    for Tensor<5, D>
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = I3, IntoIter: ExactSizeIterator>,
    I3: IntoIterator<Item = I4, IntoIter: ExactSizeIterator>,
    I4: IntoIterator<Item = I5, IntoIter: ExactSizeIterator>,
    I5: IntoIterator<Item = &'a D, IntoIter: ExactSizeIterator>,
{
    fn from_array(data: I, device: &Device) -> Self {
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
                // Create 1D tensor then reshape to desired shape
                let t1d: fusor_core::Tensor<1, D> = fusor_core::Tensor::new(gpu_device, data);
                Tensor::Gpu(t1d.reshape(shape))
            }
        }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(device: &Device, shape: [usize; R]) -> Self {
        match device {
            Device::Cpu => Tensor::Cpu(fusor_cpu::Tensor::zeros(shape)),
            Device::Gpu(gpu_device) => Tensor::Gpu(fusor_core::Tensor::zeros(gpu_device, shape)),
        }
    }

    /// Create a tensor filled with zeros that has the same shape as this tensor.
    pub fn zeros_like(&self) -> Self {
        match self {
            Tensor::Cpu(t) => {
                let shape: [usize; R] = t.shape();
                Tensor::Cpu(fusor_cpu::Tensor::zeros(shape))
            }
            Tensor::Gpu(t) => Tensor::Gpu(t.zeros_like()),
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_zeros_cpu() {
        let device = Device::Cpu;
        let t: Tensor<2, f32> = Tensor::zeros(&device, [2, 3]);
        let slice = t.as_slice().await.unwrap();

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(slice[[i, j]], 0.0);
            }
        }
    }

    #[tokio::test]
    async fn test_zeros_like_cpu() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t: Tensor<2, f32> = Tensor::Cpu(fusor_cpu::Tensor::from_slice([2, 3], &data));
        let zeros = t.zeros_like();
        let slice = zeros.as_slice().await.unwrap();

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(slice[[i, j]], 0.0);
            }
        }
    }

    #[tokio::test]
    async fn test_splat_cpu() {
        let device = Device::Cpu;
        let t: Tensor<2, f32> = Tensor::splat(&device, 5.0, [2, 3]);
        let slice = t.as_slice().await.unwrap();

        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(slice[[i, j]], 5.0);
            }
        }
    }

    #[tokio::test]
    async fn test_new_1d_cpu() {
        let device = Device::Cpu;
        let t: Tensor<1, f32> = Tensor::new(&device, &[1.0, 2.0, 3.0]);
        let slice = t.as_slice().await.unwrap();
        assert_eq!(slice[[0]], 1.0);
        assert_eq!(slice[[1]], 2.0);
        assert_eq!(slice[[2]], 3.0);
    }

    #[tokio::test]
    async fn test_new_2d_cpu() {
        let device = Device::Cpu;
        let t: Tensor<2, f32> = Tensor::new(&device, &[[1.0, 2.0], [3.0, 4.0]]);
        let slice = t.as_slice().await.unwrap();
        assert_eq!(slice[[0, 0]], 1.0);
        assert_eq!(slice[[0, 1]], 2.0);
        assert_eq!(slice[[1, 0]], 3.0);
        assert_eq!(slice[[1, 1]], 4.0);
    }

    #[tokio::test]
    async fn test_new_3d_cpu() {
        let device = Device::Cpu;
        let t: Tensor<3, f32> = Tensor::new(&device, &[[[1.0, 2.0], [3.0, 4.0]]]);
        let slice = t.as_slice().await.unwrap();
        assert_eq!(slice[[0, 0, 0]], 1.0);
        assert_eq!(slice[[0, 0, 1]], 2.0);
        assert_eq!(slice[[0, 1, 0]], 3.0);
        assert_eq!(slice[[0, 1, 1]], 4.0);
    }
}
