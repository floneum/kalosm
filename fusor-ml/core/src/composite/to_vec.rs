use std::ops::Deref;

use bytemuck::{AnyBitPattern, NoUninit};

use crate::{DataType, Tensor, TensorSlice};

impl<D: DataType> Tensor<1, D> {
    /// Convert a 1D tensor to a `Vec<D>`
    pub async fn to_vec1(&self) -> crate::Result<Vec<D>, crate::Error> {
        let slice = self.as_slice().await?;
        Ok(slice.to_vec1())
    }
}

impl<D: DataType> Tensor<2, D> {
    /// Convert a 2D tensor to a `Vec<Vec<D>>`
    pub async fn to_vec2(&self) -> crate::Result<Vec<Vec<D>>, crate::Error> {
        let slice = self.as_slice().await?;
        Ok(slice.to_vec2())
    }
}

impl<D: DataType> Tensor<3, D> {
    /// Convert a 3D tensor to a `Vec<Vec<Vec<D>>>`
    pub async fn to_vec3(&self) -> crate::Result<Vec<Vec<Vec<D>>>, crate::Error> {
        let slice = self.as_slice().await?;
        Ok(slice.to_vec3())
    }
}

/// Extension trait for TensorSlice to convert to Vec types
pub trait ToVec1<D> {
    fn to_vec1(&self) -> Vec<D>;
}

/// Extension trait for TensorSlice to convert to Vec types
pub trait ToVec2<D> {
    fn to_vec2(&self) -> Vec<Vec<D>>;
}

/// Extension trait for TensorSlice to convert to Vec types
pub trait ToVec3<D> {
    fn to_vec3(&self) -> Vec<Vec<Vec<D>>>;
}

impl<D: NoUninit + AnyBitPattern + Copy, Bytes: Deref<Target = [u8]>> ToVec1<D>
    for TensorSlice<1, D, Bytes>
{
    /// Convert a 1D tensor slice to a `Vec<D>`
    fn to_vec1(&self) -> Vec<D> {
        let shape = self.shape();
        let len = shape[0];

        let mut result = Vec::with_capacity(len);
        for i in 0..len {
            result.push(self[[i]]);
        }
        result
    }
}

impl<D: NoUninit + AnyBitPattern + Copy, Bytes: Deref<Target = [u8]>> ToVec2<D>
    for TensorSlice<2, D, Bytes>
{
    /// Convert a 2D tensor slice to a `Vec<Vec<D>>`
    fn to_vec2(&self) -> Vec<Vec<D>> {
        let shape = self.shape();
        let rows = shape[0];
        let cols = shape[1];

        let mut result = Vec::with_capacity(rows);
        for i in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for j in 0..cols {
                row.push(self[[i, j]]);
            }
            result.push(row);
        }
        result
    }
}

impl<D: NoUninit + AnyBitPattern + Copy, Bytes: Deref<Target = [u8]>> ToVec3<D>
    for TensorSlice<3, D, Bytes>
{
    /// Convert a 3D tensor slice to a `Vec<Vec<Vec<D>>>`
    fn to_vec3(&self) -> Vec<Vec<Vec<D>>> {
        let shape = self.shape();
        let dim0 = shape[0];
        let dim1 = shape[1];
        let dim2 = shape[2];

        let mut result = Vec::with_capacity(dim0);
        for i in 0..dim0 {
            let mut layer = Vec::with_capacity(dim1);
            for j in 0..dim1 {
                let mut row = Vec::with_capacity(dim2);
                for k in 0..dim2 {
                    row.push(self[[i, j, k]]);
                }
                layer.push(row);
            }
            result.push(layer);
        }
        result
    }
}

#[cfg(test)]
#[tokio::test]
async fn test_to_vec2() {
    use crate::Device;

    let device = Device::test_instance();

    let data = [[1., 2.], [3., 4.]];
    let tensor = Tensor::new(&device, &data);
    let vec2 = tensor.to_vec2().await.unwrap();

    assert_eq!(vec2.len(), 2);
    assert_eq!(vec2[0].len(), 2);
    assert_eq!(vec2[0][0], 1.);
    assert_eq!(vec2[0][1], 2.);
    assert_eq!(vec2[1][0], 3.);
    assert_eq!(vec2[1][1], 4.);
}
