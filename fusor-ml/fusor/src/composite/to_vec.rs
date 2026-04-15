//! Conversion operations to Vec types that work on both CPU and GPU backends.

use std::ops::Deref;

use bytemuck::{AnyBitPattern, NoUninit};

use fusor_core::TensorSlice;

/// Extension trait for TensorSlice to convert to Vec types
pub trait ToVec {
    type Output;
    fn to_vec(&self) -> Self::Output;
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

impl<D: NoUninit + AnyBitPattern + Copy, Bytes: Deref<Target = [u8]>> ToVec
    for TensorSlice<1, D, Bytes>
{
    type Output = Vec<D>;

    /// Convert a 1D tensor slice to a `Vec<D>`
    fn to_vec(&self) -> Self::Output {
        self.to_vec1()
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

impl<D: NoUninit + AnyBitPattern + Copy, Bytes: Deref<Target = [u8]>> ToVec
    for TensorSlice<2, D, Bytes>
{
    type Output = Vec<Vec<D>>;

    /// Convert a 2D tensor slice to a `Vec<Vec<D>>`
    fn to_vec(&self) -> Self::Output {
        self.to_vec2()
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

impl<D: NoUninit + AnyBitPattern + Copy, Bytes: Deref<Target = [u8]>> ToVec
    for TensorSlice<3, D, Bytes>
{
    type Output = Vec<Vec<Vec<D>>>;

    /// Convert a 3D tensor slice to a `Vec<Vec<Vec<D>>>`
    fn to_vec(&self) -> Self::Output {
        self.to_vec3()
    }
}

