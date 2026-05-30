use std::{
    fmt::Debug,
    marker::PhantomData,
    mem::size_of,
    ops::{Deref, Index},
};

use bytemuck::{AnyBitPattern, NoUninit};

use crate::Layout;

/// A slice of tensor data that has been resolved/materialized.
///
/// This struct provides access to tensor data with support for non-contiguous layouts
/// through stride-based indexing.
///
/// # Type Parameters
/// - `R`: The rank (number of dimensions) of the tensor
/// - `D`: The element data type (must be `NoUninit + AnyBitPattern`)
/// - `Bytes`: The buffer type holding the raw bytes (must deref to `[u8]`)
pub struct TensorSlice<const R: usize, D, Bytes> {
    buffer: Bytes,
    layout: Layout,
    datatype: PhantomData<D>,
}

impl<D: NoUninit + AnyBitPattern + Debug, Bytes> Debug for TensorSlice<0, D, Bytes>
where
    Bytes: Deref<Target = [u8]>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.get([]).fmt(f)
    }
}

impl<D: NoUninit + AnyBitPattern + Debug, Bytes> Debug for TensorSlice<1, D, Bytes>
where
    Bytes: Deref<Target = [u8]>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.layout.shape();
        let vec = (0..shape[0])
            .map(|i| self.get([i]).unwrap())
            .collect::<Vec<_>>();
        vec.fmt(f)
    }
}

impl<D: NoUninit + AnyBitPattern + Debug, Bytes> Debug for TensorSlice<2, D, Bytes>
where
    Bytes: Deref<Target = [u8]>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.layout.shape();
        let vec = (0..shape[0])
            .map(|i| {
                (0..shape[1])
                    .map(|j| self.get([i, j]).unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        vec.fmt(f)
    }
}

impl<D: NoUninit + AnyBitPattern + Debug, Bytes> Debug for TensorSlice<3, D, Bytes>
where
    Bytes: Deref<Target = [u8]>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.layout.shape();
        let vec = (0..shape[0])
            .map(|i| {
                (0..shape[1])
                    .map(|j| {
                        (0..shape[2])
                            .map(|k| self.get([i, j, k]).unwrap())
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        vec.fmt(f)
    }
}

impl<D: NoUninit + AnyBitPattern + Debug, Bytes> Debug for TensorSlice<4, D, Bytes>
where
    Bytes: Deref<Target = [u8]>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let shape = self.layout.shape();
        let vec = (0..shape[0])
            .map(|i| {
                (0..shape[1])
                    .map(|j| {
                        (0..shape[2])
                            .map(|k| {
                                (0..shape[3])
                                    .map(|l| self.get([i, j, k, l]).unwrap())
                                    .collect::<Vec<_>>()
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        vec.fmt(f)
    }
}

impl<const R: usize, D: NoUninit + AnyBitPattern + PartialEq, Bytes> PartialEq
    for TensorSlice<R, D, Bytes>
where
    Bytes: Deref<Target = [u8]>,
{
    fn eq(&self, other: &Self) -> bool {
        let self_shape = self.layout.shape();
        let other_shape = other.layout.shape();
        if self_shape != other_shape {
            return false;
        }
        if R == 0 {
            return true;
        }
        let mut matches = true;
        self.visit_indexes(|index| {
            matches &= self.get(index) == other.get(index);
        });
        matches
    }
}

impl<const R: usize, D: NoUninit + AnyBitPattern, Bytes> TensorSlice<R, D, Bytes>
where
    Bytes: Deref<Target = [u8]>,
{
    fn visit_indexes(&self, mut visitor: impl FnMut([usize; R])) {
        let self_shape = self.layout.shape();
        let mut index = [0; R];
        loop {
            index[0] += 1;
            for i in 0..R {
                if index[i] >= self_shape[i] {
                    index[i] = 0;
                    if i == R - 1 {
                        return;
                    }
                    index[i + 1] += 1;
                } else {
                    break;
                }
            }
            visitor(index);
        }
    }

    pub fn map_bytes<Bytes2>(
        self,
        mut mapper: impl FnMut(Bytes) -> Bytes2,
    ) -> TensorSlice<R, D, Bytes2> {
        TensorSlice {
            buffer: mapper(self.buffer),
            layout: self.layout.clone(),
            datatype: PhantomData,
        }
    }

    /// Get the scalar value from a rank-0 tensor.
    pub fn as_scalar(&self) -> D
    where
        D: Copy,
    {
        self.as_slice()[0]
    }

    /// Visit all items in the tensor with a callback.
    pub fn visit_items(&self, mut visitor: impl FnMut(&D)) {
        self.visit_indexes(|index| {
            visitor(self.get(index).unwrap());
        });
    }
}

impl<'a, D: NoUninit + AnyBitPattern + PartialEq, Bytes> PartialEq<&'a [D]>
    for TensorSlice<1, D, Bytes>
where
    Bytes: Deref<Target = [u8]>,
{
    fn eq(&self, other: &&'a [D]) -> bool {
        self.as_slice() == *other
    }
}

impl<const N: usize, D: NoUninit + AnyBitPattern + PartialEq, Bytes> PartialEq<[D; N]>
    for TensorSlice<1, D, Bytes>
where
    Bytes: Deref<Target = [u8]>,
{
    fn eq(&self, other: &[D; N]) -> bool {
        self.as_slice() == *other
    }
}

impl<D: NoUninit + AnyBitPattern + PartialEq, Bytes> PartialEq<TensorSlice<1, D, Bytes>> for &[D]
where
    Bytes: Deref<Target = [u8]>,
{
    fn eq(&self, other: &TensorSlice<1, D, Bytes>) -> bool {
        *self == other.as_slice()
    }
}

impl<const N: usize, D: NoUninit + AnyBitPattern + PartialEq, Bytes>
    PartialEq<TensorSlice<1, D, Bytes>> for &[D; N]
where
    Bytes: Deref<Target = [u8]>,
{
    fn eq(&self, other: &TensorSlice<1, D, Bytes>) -> bool {
        *self == other.as_slice()
    }
}

impl<D: NoUninit + AnyBitPattern, const R: usize, Bytes> TensorSlice<R, D, Bytes>
where
    Bytes: Deref<Target = [u8]>,
{
    /// Create a new TensorSlice from a buffer and layout.
    pub fn new(buffer: Bytes, layout: Layout) -> Self {
        Self {
            buffer,
            layout,
            datatype: PhantomData,
        }
    }

    /// Get the underlying buffer as a slice of the element type.
    pub fn as_slice(&self) -> &[D] {
        bytemuck::cast_slice(&self.buffer.deref()[self.layout.offset() * size_of::<D>()..])
    }
}

impl<D: NoUninit + AnyBitPattern, const R: usize, Bytes> TensorSlice<R, D, Bytes>
where
    Bytes: Deref<Target = [u8]>,
{
    /// Get the shape of this tensor.
    pub fn shape(&self) -> &[usize] {
        self.layout.shape()
    }

    /// Get the element at the given index, returning None if out of bounds.
    pub fn get(&self, index: [usize; R]) -> Option<&D> {
        let mut index_sum = 0;
        let layout = &self.layout;
        for ((index_component, &stride), &size) in
            index.into_iter().zip(layout.strides()).zip(layout.shape())
        {
            if index_component >= size {
                return None;
            }
            index_sum += stride * index_component;
        }

        self.as_slice().get(index_sum)
    }
}

impl<D: NoUninit + AnyBitPattern, const R: usize, Bytes> Index<[usize; R]>
    for TensorSlice<R, D, Bytes>
where
    Bytes: Deref<Target = [u8]>,
{
    type Output = D;

    fn index(&self, index: [usize; R]) -> &Self::Output {
        self.get(index).unwrap()
    }
}
