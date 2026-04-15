//! Tensor - the unified interface over different tensor backends

use std::ops::{
    Add as StdAdd, Div as StdDiv, Mul as StdMul, Neg as StdNeg, Range, Rem as StdRem, Sub as StdSub,
};

use fusor_types::{Layout, SlidingWindow};
use pulp::Simd;

use crate::cast::{CastTo, cast_tensor};
use crate::comparison::{self, EqOp, GtOp, GteOp, LtOp, LteOp, NeOp};
use crate::concrete_tensor::IndexIterator;
use crate::conditional::{IsNonZero, where_cond_ref};
use crate::elementwise::{
    AbsOp, AcosOp, AcoshOp, AsinOp, AsinhOp, AtanOp, AtanhOp, CosOp, CoshOp, Exp2Op, ExpOp, Log2Op,
    LogOp, NegOp, SimdUnaryOp, SinOp, SinhOp, SqrtOp, TanOp, TanhOp,
};
use crate::index::index_select_ref;
use crate::matmul::MatmulImpl;
use crate::pairwise::{AddOp, DivOp, MulOp, RemOp, SimdBinaryOp, SubOp};
use crate::reduce::{
    MaxOp, MinOp, ProdOp, SimdReduceOp, SumOp, reduce_tensor_axis_dyn, reduce_tensor_op,
};
use crate::slice_assign::slice_assign_ref;
use crate::{
    ConcreteTensor, CpuMappedBuffer, LastRank, MapLayout, MaxRank, ResolvedTensor, SimdElement,
    TensorBacking, TensorSlice, elementwise, pairwise, scalar,
};

/// A tensor wrapper that provides a unified interface over different tensor backends.
#[derive(Copy, Clone)]
pub struct Tensor<const R: usize, T: TensorBacking<R>> {
    inner: T,
}

impl<const R: usize, T: TensorBacking<R>> Tensor<R, T> {
    /// Create a new tensor from an inner backing type.
    pub fn new(inner: T) -> Self {
        Self { inner }
    }

    /// Get a reference to the inner backing type.
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Get a mutable reference to the inner backing type.
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Consume the tensor and return the inner backing type.
    pub fn into_inner(self) -> T {
        self.inner
    }

    /// Borrow the tensor's backing, returning a tensor that holds a reference.
    ///
    /// This enables sharing the backing between multiple tensors without cloning.
    pub fn as_ref(&self) -> Tensor<R, &T> {
        Tensor::new(&self.inner)
    }
}

// Constructors for Tensor that create ConcreteTensor backing
impl<const R: usize, E: SimdElement> Tensor<R, ConcreteTensor<E, R>> {
    /// Create a new tensor filled with zeros
    pub fn zeros(shape: [usize; R]) -> Self
    where
        E: Default,
    {
        Self::new(ConcreteTensor::zeros(shape))
    }

    /// Create a new tensor from existing data
    pub fn from_slice(shape: [usize; R], data: &[E]) -> Self {
        Self::new(ConcreteTensor::from_slice(shape, data))
    }

    /// Get element at logical indices
    pub fn get(&self, indices: [usize; R]) -> E {
        self.inner.get(indices)
    }

    /// Set element at logical indices
    pub fn set(&mut self, indices: [usize; R], value: E) {
        self.inner.set(indices, value)
    }
}

// Methods for Tensor with MapLayout backing
impl<const R: usize, T: crate::LazyBacking> Tensor<R, MapLayout<T, R>> {
    /// Get element at logical indices
    pub fn get(&self, indices: [usize; R]) -> T::Elem {
        self.inner.get(indices)
    }
}

// Methods available on any Tensor with TensorBacking inner
impl<const R: usize, E, T> Tensor<R, T>
where
    E: SimdElement,
    T: TensorBacking<R, Elem = E>,
{
    /// Returns the shape of the tensor as a fixed-size array
    pub fn shape(&self) -> [usize; R] {
        self.inner
            .layout()
            .shape()
            .try_into()
            .expect("Shape length mismatch")
    }

    /// Materialize the tensor to a ConcreteTensor
    pub fn to_concrete(&self) -> Tensor<R, ConcreteTensor<E, R>> {
        Tensor::new(self.inner.to_concrete())
    }

    /// Slice the tensor along all dimensions
    ///
    /// Returns a view into the tensor's data with updated layout.
    /// This operation is lazy and preserves laziness of the inner tensor.
    pub fn slice(self, slices: [Range<usize>; R]) -> Tensor<R, MapLayout<T, R>> {
        let current_layout = self.inner.layout();
        let new_layout = current_layout.slice(&slices);
        Tensor::new(MapLayout::new(self.inner, new_layout))
    }

    /// Permute the tensor dimensions according to the given axes order
    ///
    /// # Arguments
    /// * `axes` - A permutation of [0, 1, ..., R-1] specifying the new order
    ///
    /// This operation is lazy and preserves laziness of the inner tensor.
    pub fn permute(self, axes: [usize; R]) -> Tensor<R, MapLayout<T, R>> {
        let current_layout = self.inner.layout();
        let new_layout = current_layout.permute(&axes);
        Tensor::new(MapLayout::new(self.inner, new_layout))
    }

    /// Transpose two dimensions of the tensor
    ///
    /// # Arguments
    /// * `dim0` - First dimension to swap
    /// * `dim1` - Second dimension to swap
    ///
    /// This operation is lazy and preserves laziness of the inner tensor.
    pub fn transpose(self, dim0: usize, dim1: usize) -> Tensor<R, MapLayout<T, R>> {
        let current_layout = self.inner.layout();
        let new_layout = current_layout.transpose(dim0, dim1);
        Tensor::new(MapLayout::new(self.inner, new_layout))
    }

    /// Broadcast the tensor to a larger shape
    ///
    /// Broadcasting rules:
    /// - Dimensions are aligned from the right
    /// - A dimension can be broadcast if it's 1 or matches the target
    /// - New dimensions can be added on the left
    ///
    /// This operation is lazy and preserves laziness of the inner tensor.
    pub fn broadcast_as<const R2: usize>(
        self,
        out_shape: [usize; R2],
    ) -> Tensor<R2, MapLayout<T, R2>> {
        let current_layout = self.inner.layout();
        let new_layout = current_layout.broadcast_to(&out_shape);
        Tensor::new(MapLayout::new(self.inner, new_layout))
    }

    /// Reshape the tensor to a new shape
    ///
    /// The total number of elements must remain the same.
    /// This operation is lazy and preserves laziness of the inner tensor.
    pub fn reshape<const R2: usize>(self, new_shape: [usize; R2]) -> Tensor<R2, MapLayout<T, R2>> {
        let new_layout = Layout::contiguous(&new_shape);
        Tensor::new(MapLayout::new(self.inner, new_layout))
    }

    /// Flatten the tensor to 1D
    /// This operation is lazy and preserves laziness of the inner tensor.
    pub fn flatten_all(self) -> Tensor<1, MapLayout<T, 1>> {
        let total: usize = self.inner.layout().num_elements();
        self.reshape([total])
    }

    /// Make the tensor contiguous by copying data if necessary
    pub fn make_contiguous(self) -> Tensor<R, ConcreteTensor<E, R>> {
        let concrete = self.inner.to_concrete();
        if concrete.layout().is_contiguous() {
            return Tensor::new(concrete);
        }

        let shape: [usize; R] = concrete
            .layout()
            .shape()
            .try_into()
            .expect("Shape length mismatch");
        let mut output = ConcreteTensor::<E, R>::zeros(shape);

        for indices in IndexIterator::new(concrete.layout().shape()) {
            let indices_arr: [usize; R] = indices.try_into().expect("Indices length mismatch");
            let src_idx = concrete.layout().linear_index(&indices_arr);
            let dst_idx = output.layout().linear_index(&indices_arr);
            output.backing_mut()[dst_idx] = concrete.backing()[src_idx];
        }

        Tensor::new(output)
    }

    /// Narrow the tensor along a given dimension
    ///
    /// # Arguments
    /// * `dim` - The dimension to narrow
    /// * `start` - The starting index
    /// * `length` - The length of the slice
    ///
    /// This operation is lazy and preserves laziness of the inner tensor.
    pub fn narrow(self, dim: usize, start: usize, length: usize) -> Tensor<R, MapLayout<T, R>> {
        let current_layout = self.inner.layout();
        let new_layout = current_layout.narrow(dim, start, length);
        Tensor::new(MapLayout::new(self.inner, new_layout))
    }

    /// Split the tensor into chunks along a given dimension
    ///
    /// # Arguments
    /// * `chunks` - Number of chunks to split into
    /// * `dim` - The dimension to split along
    ///
    /// This operation is lazy and preserves laziness of the inner tensor.
    pub fn chunk(self, chunks: usize, dim: usize) -> Vec<Tensor<R, MapLayout<T, R>>>
    where
        T: Clone,
    {
        assert!(dim < R, "Dimension {} out of range for rank {}", dim, R);
        assert!(chunks > 0, "Number of chunks must be positive");

        let dim_size = self.inner.layout().shape()[dim];
        let chunk_size = dim_size.div_ceil(chunks);

        let mut result = Vec::with_capacity(chunks);
        let mut start = 0;

        while start < dim_size {
            let length = chunk_size.min(dim_size - start);
            result.push(Tensor::new(self.inner.clone()).narrow(dim, start, length));
            start += length;
        }

        result
    }

    /// Repeat the tensor along each dimension
    ///
    /// # Arguments
    /// * `repeats` - Number of times to repeat along each dimension
    pub fn repeat(self, repeats: [usize; R]) -> Tensor<R, ConcreteTensor<E, R>> {
        let concrete = self.inner.to_concrete();
        let old_shape = concrete.layout().shape();
        let mut new_shape = [0usize; R];
        for i in 0..R {
            new_shape[i] = old_shape[i] * repeats[i];
        }

        let mut output = ConcreteTensor::<E, R>::zeros(new_shape);

        for out_indices in IndexIterator::new(&new_shape) {
            let out_arr: [usize; R] = out_indices.try_into().expect("Indices length mismatch");

            let mut in_arr = [0usize; R];
            for i in 0..R {
                in_arr[i] = out_arr[i] % old_shape[i];
            }

            let src_idx = concrete.layout().linear_index(&in_arr);
            let dst_idx = output.layout().linear_index(&out_arr);
            output.backing_mut()[dst_idx] = concrete.backing()[src_idx];
        }

        Tensor::new(output)
    }

    /// Squeeze a dimension of size 1
    ///
    /// # Arguments
    /// * `dim` - The dimension to squeeze (must have size 1)
    ///
    /// This operation is lazy and preserves laziness of the inner tensor.
    pub fn squeeze<const R2: usize>(self, dim: usize) -> Tensor<R2, MapLayout<T, R2>> {
        assert!(R2 == R - 1, "Output rank must be R - 1");
        let current_layout = self.inner.layout();
        let new_layout = current_layout.squeeze(dim);
        Tensor::new(MapLayout::new(self.inner, new_layout))
    }

    /// Unsqueeze (add a dimension of size 1)
    ///
    /// # Arguments
    /// * `dim` - Where to insert the new dimension
    ///
    /// This operation is lazy and preserves laziness of the inner tensor.
    pub fn unsqueeze<const R2: usize>(self, dim: usize) -> Tensor<R2, MapLayout<T, R2>> {
        assert!(R2 == R + 1, "Output rank must be R + 1");
        let current_layout = self.inner.layout();
        let new_layout = current_layout.unsqueeze(dim);
        Tensor::new(MapLayout::new(self.inner, new_layout))
    }

    /// Expand the tensor to a larger shape (alias for broadcast_as)
    ///
    /// This is an alias for `broadcast_as` for compatibility with other tensor libraries.
    /// This operation is lazy and preserves laziness of the inner tensor.
    pub fn expand<const R2: usize>(self, out_shape: [usize; R2]) -> Tensor<R2, MapLayout<T, R2>> {
        self.broadcast_as(out_shape)
    }

    /// Flatten the last N dimensions into one
    ///
    /// # Type Parameters
    /// * `N` - Number of dimensions from the end to flatten (must be >= 1)
    /// * `R2` - Output rank (must be R - N + 1)
    ///
    /// # Example
    /// A tensor of shape [2, 3, 4] with N=2 becomes [2, 12]
    /// This operation is lazy and preserves laziness of the inner tensor.
    pub fn flatten_last_n<const N: usize, const R2: usize>(self) -> Tensor<R2, MapLayout<T, R2>> {
        assert!(R2 == R - N + 1, "Output rank must be R - N + 1");
        let current_layout = self.inner.layout();
        let new_layout = current_layout.flatten_last_n(N);
        Tensor::new(MapLayout::new(self.inner, new_layout))
    }

    /// Flatten the first N+1 dimensions into one
    ///
    /// # Type Parameters
    /// * `N` - Number indicating how many dimensions to include (flattens first N+1 dims)
    /// * `R2` - Output rank (must be R - N)
    ///
    /// # Example
    /// A tensor of shape [2, 3, 4] with N=1 becomes [6, 4]
    /// This operation is lazy and preserves laziness of the inner tensor.
    pub fn flatten_first_n<const N: usize, const R2: usize>(self) -> Tensor<R2, MapLayout<T, R2>> {
        assert!(R2 == R - N, "Output rank must be R - N");
        let current_layout = self.inner.layout();
        let new_layout = current_layout.flatten_first_n(N);
        Tensor::new(MapLayout::new(self.inner, new_layout))
    }

    /// Squeeze (remove) multiple dimensions of size 1 at once
    ///
    /// # Type Parameters
    /// * `DIFF` - Number of dimensions to remove
    /// * `R2` - Output rank (must be R - DIFF)
    ///
    /// # Arguments
    /// * `axes` - The dimensions to squeeze (must all have size 1)
    ///
    /// This operation is lazy and preserves laziness of the inner tensor.
    pub fn squeeze_dims<const DIFF: usize, const R2: usize>(
        self,
        axes: [usize; DIFF],
    ) -> Tensor<R2, MapLayout<T, R2>> {
        assert!(R2 == R - DIFF, "Output rank must be R - DIFF");
        let current_layout = self.inner.layout();
        let new_layout = current_layout.squeeze_dims(&axes);
        Tensor::new(MapLayout::new(self.inner, new_layout))
    }

    /// Unsqueeze (add) multiple dimensions of size 1 at specified positions
    ///
    /// # Type Parameters
    /// * `DIFF` - Number of dimensions to add
    /// * `R2` - Output rank (must be R + DIFF)
    ///
    /// # Arguments
    /// * `axes` - Where to insert the new dimensions (positions in the output tensor)
    ///
    /// This operation is lazy and preserves laziness of the inner tensor.
    pub fn unsqueeze_dims<const DIFF: usize, const R2: usize>(
        self,
        axes: [usize; DIFF],
    ) -> Tensor<R2, MapLayout<T, R2>> {
        assert!(R2 == R + DIFF, "Output rank must be R + DIFF");
        let current_layout = self.inner.layout();
        let new_layout = current_layout.unsqueeze_dims(&axes);
        Tensor::new(MapLayout::new(self.inner, new_layout))
    }

    /// Create a sliding window view of the tensor (zero-copy)
    ///
    /// This creates overlapping windows along specified dimensions without copying data.
    ///
    /// # Type Parameters
    /// * `DIFF` - Number of windows to create
    /// * `R2` - Output rank (must be R + DIFF)
    ///
    /// # Arguments
    /// * `windows` - Array of SlidingWindow configurations specifying axis, window size, and step
    ///
    /// # Example
    /// A 1D tensor [1, 2, 3, 4, 5, 6, 7] with window size 3 and step 2 becomes:
    /// [[1, 2, 3], [3, 4, 5], [5, 6, 7]]
    /// This operation is lazy and preserves laziness of the inner tensor.
    pub fn sliding_window_view<const DIFF: usize, const R2: usize>(
        self,
        windows: [SlidingWindow; DIFF],
    ) -> Tensor<R2, MapLayout<T, R2>> {
        assert!(R2 == R + DIFF, "Output rank must be R + DIFF");
        let current_layout = self.inner.layout();
        let new_layout = current_layout.sliding_window(&windows);
        Tensor::new(MapLayout::new(self.inner, new_layout))
    }

    /// Sum all elements in the tensor
    #[inline]
    pub fn sum(self) -> E
    where
        SumOp: SimdReduceOp<E>,
    {
        reduce_tensor_op::<E, R, SumOp>(&self.inner.to_concrete())
    }

    /// Find the maximum element in the tensor
    #[inline]
    pub fn max(self) -> E
    where
        MaxOp: SimdReduceOp<E>,
    {
        reduce_tensor_op::<E, R, MaxOp>(&self.inner.to_concrete())
    }

    /// Find the minimum element in the tensor
    #[inline]
    pub fn min(self) -> E
    where
        MinOp: SimdReduceOp<E>,
    {
        reduce_tensor_op::<E, R, MinOp>(&self.inner.to_concrete())
    }

    /// Multiply all elements in the tensor
    #[inline]
    pub fn prod(self) -> E
    where
        ProdOp: SimdReduceOp<E>,
    {
        reduce_tensor_op::<E, R, ProdOp>(&self.inner.to_concrete())
    }

    /// Element-wise equality comparison
    #[inline]
    pub fn eq<T2: TensorBacking<R, Elem = E>>(
        self,
        rhs: Tensor<R, T2>,
    ) -> Tensor<R, comparison::Eq<E, R, T, T2>>
    where
        E: Default,
        EqOp: SimdBinaryOp<E>,
    {
        Tensor::new(comparison::Eq::new(self.inner, rhs.inner))
    }

    /// Element-wise less than comparison
    #[inline]
    pub fn lt<T2: TensorBacking<R, Elem = E>>(
        self,
        rhs: Tensor<R, T2>,
    ) -> Tensor<R, comparison::Lt<E, R, T, T2>>
    where
        E: Default,
        LtOp: SimdBinaryOp<E>,
    {
        Tensor::new(comparison::Lt::new(self.inner, rhs.inner))
    }

    /// Element-wise greater than comparison
    #[inline]
    pub fn gt<T2: TensorBacking<R, Elem = E>>(
        self,
        rhs: Tensor<R, T2>,
    ) -> Tensor<R, comparison::Gt<E, R, T, T2>>
    where
        E: Default,
        GtOp: SimdBinaryOp<E>,
    {
        Tensor::new(comparison::Gt::new(self.inner, rhs.inner))
    }

    /// Element-wise not equal comparison
    #[inline]
    pub fn ne<T2: TensorBacking<R, Elem = E>>(
        self,
        rhs: Tensor<R, T2>,
    ) -> Tensor<R, comparison::Ne<E, R, T, T2>>
    where
        E: Default,
        NeOp: SimdBinaryOp<E>,
    {
        Tensor::new(comparison::Ne::new(self.inner, rhs.inner))
    }

    /// Element-wise less than or equal comparison
    #[inline]
    pub fn lte<T2: TensorBacking<R, Elem = E>>(
        self,
        rhs: Tensor<R, T2>,
    ) -> Tensor<R, comparison::Lte<E, R, T, T2>>
    where
        E: Default,
        LteOp: SimdBinaryOp<E>,
    {
        Tensor::new(comparison::Lte::new(self.inner, rhs.inner))
    }

    /// Element-wise greater than or equal comparison
    #[inline]
    pub fn gte<T2: TensorBacking<R, Elem = E>>(
        self,
        rhs: Tensor<R, T2>,
    ) -> Tensor<R, comparison::Gte<E, R, T, T2>>
    where
        E: Default,
        GteOp: SimdBinaryOp<E>,
    {
        Tensor::new(comparison::Gte::new(self.inner, rhs.inner))
    }

    /// Compare tensor elements with scalar for equality
    #[inline]
    pub fn eq_scalar(
        self,
        scalar_val: E,
    ) -> Tensor<R, comparison::Eq<E, R, T, scalar::Broadcast<E, R>>>
    where
        E: Default,
        EqOp: SimdBinaryOp<E>,
    {
        let shape: [usize; R] = self.layout().shape().try_into().unwrap();
        Tensor::new(comparison::Eq::new(
            self.inner,
            scalar::Broadcast::new(scalar_val, shape),
        ))
    }

    /// Compare tensor elements with scalar for inequality
    #[inline]
    pub fn ne_scalar(
        self,
        scalar_val: E,
    ) -> Tensor<R, comparison::Ne<E, R, T, scalar::Broadcast<E, R>>>
    where
        E: Default,
        NeOp: SimdBinaryOp<E>,
    {
        let shape: [usize; R] = self.layout().shape().try_into().unwrap();
        Tensor::new(comparison::Ne::new(
            self.inner,
            scalar::Broadcast::new(scalar_val, shape),
        ))
    }

    /// Compare tensor elements with scalar for less than
    #[inline]
    pub fn lt_scalar(
        self,
        scalar_val: E,
    ) -> Tensor<R, comparison::Lt<E, R, T, scalar::Broadcast<E, R>>>
    where
        E: Default,
        LtOp: SimdBinaryOp<E>,
    {
        let shape: [usize; R] = self.layout().shape().try_into().unwrap();
        Tensor::new(comparison::Lt::new(
            self.inner,
            scalar::Broadcast::new(scalar_val, shape),
        ))
    }

    /// Compare tensor elements with scalar for less than or equal
    #[inline]
    pub fn lte_scalar(
        self,
        scalar_val: E,
    ) -> Tensor<R, comparison::Lte<E, R, T, scalar::Broadcast<E, R>>>
    where
        E: Default,
        LteOp: SimdBinaryOp<E>,
    {
        let shape: [usize; R] = self.layout().shape().try_into().unwrap();
        Tensor::new(comparison::Lte::new(
            self.inner,
            scalar::Broadcast::new(scalar_val, shape),
        ))
    }

    /// Compare tensor elements with scalar for greater than
    #[inline]
    pub fn gt_scalar(
        self,
        scalar_val: E,
    ) -> Tensor<R, comparison::Gt<E, R, T, scalar::Broadcast<E, R>>>
    where
        E: Default,
        GtOp: SimdBinaryOp<E>,
    {
        let shape: [usize; R] = self.layout().shape().try_into().unwrap();
        Tensor::new(comparison::Gt::new(
            self.inner,
            scalar::Broadcast::new(scalar_val, shape),
        ))
    }

    /// Compare tensor elements with scalar for greater than or equal
    #[inline]
    pub fn gte_scalar(
        self,
        scalar_val: E,
    ) -> Tensor<R, comparison::Gte<E, R, T, scalar::Broadcast<E, R>>>
    where
        E: Default,
        GteOp: SimdBinaryOp<E>,
    {
        let shape: [usize; R] = self.layout().shape().try_into().unwrap();
        Tensor::new(comparison::Gte::new(
            self.inner,
            scalar::Broadcast::new(scalar_val, shape),
        ))
    }

    /// Conditional selection: where self != 0, select on_true, else on_false
    #[inline]
    pub fn where_cond(
        self,
        on_true: Tensor<R, impl TensorBacking<R, Elem = E>>,
        on_false: Tensor<R, impl TensorBacking<R, Elem = E>>,
    ) -> Tensor<R, ConcreteTensor<E, R>>
    where
        E: Default + IsNonZero,
    {
        Tensor::new(where_cond_ref(
            &self.inner.to_concrete(),
            &on_true.inner.to_concrete(),
            &on_false.inner.to_concrete(),
        ))
    }

    /// Cast tensor to another element type
    #[inline]
    pub fn cast<E2>(self) -> Tensor<R, ConcreteTensor<E2, R>>
    where
        E: CastTo<E2>,
        E2: SimdElement,
    {
        Tensor::new(cast_tensor(&self.inner.to_concrete()))
    }

    /// Get the tensor data as a TensorSlice for reading.
    ///
    /// This is the CPU equivalent of fusor-core's `as_slice()` method for GPU tensors.
    /// It materializes the tensor (if lazy) and returns a slice view of the data.
    pub fn as_slice(self) -> TensorSlice<R, E, CpuMappedBuffer> {
        let concrete = self.inner.to_concrete();
        let layout = concrete.layout().clone();
        // Convert the tensor data to raw bytes
        let bytes: Box<[u8]> = bytemuck::cast_slice(concrete.data().as_ref()).into();
        TensorSlice::new(CpuMappedBuffer::new(bytes), layout)
    }

    /// Select elements along a dimension using indices
    #[inline]
    pub fn index_select(
        self,
        dimension: usize,
        indices: Tensor<1, impl TensorBacking<1, Elem = u32>>,
    ) -> Tensor<R, ConcreteTensor<E, R>> {
        Tensor::new(index_select_ref(
            &self.inner.to_concrete(),
            dimension,
            &indices.inner.to_concrete(),
        ))
    }

    /// Returns a new tensor with the slice region replaced by values from the value tensor
    ///
    /// # Arguments
    /// * `slices` - Array of ranges specifying the slice region in each dimension
    /// * `value` - Tensor containing values to assign into the slice region
    ///
    /// # Panics
    /// * If slice bounds exceed input tensor dimensions
    /// * If value tensor shape doesn't match slice dimensions
    ///
    /// # Example
    /// ```ignore
    /// let tensor = Tensor::from_slice([3, 3], &[1.0; 9]);
    /// let value = Tensor::from_slice([2, 2], &[10.0; 4]);
    /// let result = tensor.slice_assign([0..2, 0..2], &value);
    /// // result[0..2, 0..2] = value, rest copied from tensor
    /// ```
    #[inline]
    pub fn slice_assign(
        self,
        slices: [Range<usize>; R],
        value: Tensor<R, impl TensorBacking<R, Elem = E>>,
    ) -> Tensor<R, ConcreteTensor<E, R>> {
        Tensor::new(slice_assign_ref(
            &self.inner.to_concrete(),
            slices,
            &value.inner.to_concrete(),
        ))
    }

    /// Sum along a specific axis, reducing the tensor rank by 1
    #[inline]
    pub fn sum_axis<const OUT_RANK: usize>(
        self,
        axis: usize,
    ) -> Tensor<OUT_RANK, ConcreteTensor<E, OUT_RANK>>
    where
        E: Default,
        ConcreteTensor<E, R>: LastRank<OUT_RANK, E>,
        SumOp: SimdReduceOp<E>,
    {
        Tensor::new(reduce_tensor_axis_dyn::<E, R, OUT_RANK, SumOp>(
            &self.inner.to_concrete(),
            axis,
        ))
    }

    /// Maximum along a specific axis, reducing the tensor rank by 1
    #[inline]
    pub fn max_axis<const OUT_RANK: usize>(
        self,
        axis: usize,
    ) -> Tensor<OUT_RANK, ConcreteTensor<E, OUT_RANK>>
    where
        E: Default,
        ConcreteTensor<E, R>: LastRank<OUT_RANK, E>,
        MaxOp: SimdReduceOp<E>,
    {
        Tensor::new(reduce_tensor_axis_dyn::<E, R, OUT_RANK, MaxOp>(
            &self.inner.to_concrete(),
            axis,
        ))
    }

    /// Minimum along a specific axis, reducing the tensor rank by 1
    #[inline]
    pub fn min_axis<const OUT_RANK: usize>(
        self,
        axis: usize,
    ) -> Tensor<OUT_RANK, ConcreteTensor<E, OUT_RANK>>
    where
        E: Default,
        ConcreteTensor<E, R>: LastRank<OUT_RANK, E>,
        MinOp: SimdReduceOp<E>,
    {
        Tensor::new(reduce_tensor_axis_dyn::<E, R, OUT_RANK, MinOp>(
            &self.inner.to_concrete(),
            axis,
        ))
    }

    /// Product along a specific axis, reducing the tensor rank by 1
    #[inline]
    pub fn prod_axis<const OUT_RANK: usize>(
        self,
        axis: usize,
    ) -> Tensor<OUT_RANK, ConcreteTensor<E, OUT_RANK>>
    where
        E: Default,
        ConcreteTensor<E, R>: LastRank<OUT_RANK, E>,
        ProdOp: SimdReduceOp<E>,
    {
        Tensor::new(reduce_tensor_axis_dyn::<E, R, OUT_RANK, ProdOp>(
            &self.inner.to_concrete(),
            axis,
        ))
    }
}

/// Trait for float types that support power, min, max, and clamp operations
pub trait FloatOps: SimdElement + PartialOrd {
    fn powf(self, exp: Self) -> Self;
    fn float_max(self, other: Self) -> Self;
    fn float_min(self, other: Self) -> Self;
}

impl FloatOps for f32 {
    #[inline(always)]
    fn powf(self, exp: Self) -> Self {
        self.powf(exp)
    }
    #[inline(always)]
    fn float_max(self, other: Self) -> Self {
        self.max(other)
    }
    #[inline(always)]
    fn float_min(self, other: Self) -> Self {
        self.min(other)
    }
}

impl FloatOps for f64 {
    #[inline(always)]
    fn powf(self, exp: Self) -> Self {
        self.powf(exp)
    }
    #[inline(always)]
    fn float_max(self, other: Self) -> Self {
        self.max(other)
    }
    #[inline(always)]
    fn float_min(self, other: Self) -> Self {
        self.min(other)
    }
}

// Lazy unary operations
impl<const R: usize, E, T> Tensor<R, T>
where
    E: SimdElement,
    T: TensorBacking<R, Elem = E>,
{
    /// Absolute value element-wise (lazy)
    #[inline]
    pub fn abs(self) -> Tensor<R, elementwise::Abs<E, R, T>>
    where
        AbsOp: SimdUnaryOp<E>,
    {
        Tensor::new(elementwise::Abs::new(self.inner))
    }

    /// Square root element-wise (lazy)
    #[inline]
    pub fn sqrt(self) -> Tensor<R, elementwise::Sqrt<E, R, T>>
    where
        SqrtOp: SimdUnaryOp<E>,
    {
        Tensor::new(elementwise::Sqrt::new(self.inner))
    }

    /// Exponential (e^x) element-wise (lazy)
    #[inline]
    pub fn exp(self) -> Tensor<R, elementwise::Exp<E, R, T>>
    where
        ExpOp: SimdUnaryOp<E>,
    {
        Tensor::new(elementwise::Exp::new(self.inner))
    }

    /// Natural logarithm element-wise (lazy)
    #[inline]
    pub fn log(self) -> Tensor<R, elementwise::Log<E, R, T>>
    where
        LogOp: SimdUnaryOp<E>,
    {
        Tensor::new(elementwise::Log::new(self.inner))
    }

    /// Sine element-wise (lazy)
    #[inline]
    pub fn sin(self) -> Tensor<R, elementwise::Sin<E, R, T>>
    where
        SinOp: SimdUnaryOp<E>,
    {
        Tensor::new(elementwise::Sin::new(self.inner))
    }

    /// Cosine element-wise (lazy)
    #[inline]
    pub fn cos(self) -> Tensor<R, elementwise::Cos<E, R, T>>
    where
        CosOp: SimdUnaryOp<E>,
    {
        Tensor::new(elementwise::Cos::new(self.inner))
    }

    /// Tangent element-wise (lazy)
    #[inline]
    pub fn tan(self) -> Tensor<R, elementwise::Tan<E, R, T>>
    where
        TanOp: SimdUnaryOp<E>,
    {
        Tensor::new(elementwise::Tan::new(self.inner))
    }

    /// Base-2 exponential (2^x) element-wise (lazy)
    #[inline]
    pub fn exp2(self) -> Tensor<R, elementwise::Exp2<E, R, T>>
    where
        Exp2Op: SimdUnaryOp<E>,
    {
        Tensor::new(elementwise::Exp2::new(self.inner))
    }

    /// Base-2 logarithm element-wise (lazy)
    #[inline]
    pub fn log2(self) -> Tensor<R, elementwise::Log2<E, R, T>>
    where
        Log2Op: SimdUnaryOp<E>,
    {
        Tensor::new(elementwise::Log2::new(self.inner))
    }

    /// Arc sine (inverse sin) element-wise (lazy)
    #[inline]
    pub fn asin(self) -> Tensor<R, elementwise::Asin<E, R, T>>
    where
        AsinOp: SimdUnaryOp<E>,
    {
        Tensor::new(elementwise::Asin::new(self.inner))
    }

    /// Arc cosine (inverse cos) element-wise (lazy)
    #[inline]
    pub fn acos(self) -> Tensor<R, elementwise::Acos<E, R, T>>
    where
        AcosOp: SimdUnaryOp<E>,
    {
        Tensor::new(elementwise::Acos::new(self.inner))
    }

    /// Arc tangent (inverse tan) element-wise (lazy)
    #[inline]
    pub fn atan(self) -> Tensor<R, elementwise::Atan<E, R, T>>
    where
        AtanOp: SimdUnaryOp<E>,
    {
        Tensor::new(elementwise::Atan::new(self.inner))
    }

    /// Hyperbolic sine element-wise (lazy)
    #[inline]
    pub fn sinh(self) -> Tensor<R, elementwise::Sinh<E, R, T>>
    where
        SinhOp: SimdUnaryOp<E>,
    {
        Tensor::new(elementwise::Sinh::new(self.inner))
    }

    /// Hyperbolic cosine element-wise (lazy)
    #[inline]
    pub fn cosh(self) -> Tensor<R, elementwise::Cosh<E, R, T>>
    where
        CoshOp: SimdUnaryOp<E>,
    {
        Tensor::new(elementwise::Cosh::new(self.inner))
    }

    /// Hyperbolic tangent element-wise (lazy)
    #[inline]
    pub fn tanh(self) -> Tensor<R, elementwise::Tanh<E, R, T>>
    where
        TanhOp: SimdUnaryOp<E>,
    {
        Tensor::new(elementwise::Tanh::new(self.inner))
    }

    /// Inverse hyperbolic sine element-wise (lazy)
    #[inline]
    pub fn asinh(self) -> Tensor<R, elementwise::Asinh<E, R, T>>
    where
        AsinhOp: SimdUnaryOp<E>,
    {
        Tensor::new(elementwise::Asinh::new(self.inner))
    }

    /// Inverse hyperbolic cosine element-wise (lazy)
    #[inline]
    pub fn acosh(self) -> Tensor<R, elementwise::Acosh<E, R, T>>
    where
        AcoshOp: SimdUnaryOp<E>,
    {
        Tensor::new(elementwise::Acosh::new(self.inner))
    }

    /// Inverse hyperbolic tangent element-wise (lazy)
    #[inline]
    pub fn atanh(self) -> Tensor<R, elementwise::Atanh<E, R, T>>
    where
        AtanhOp: SimdUnaryOp<E>,
    {
        Tensor::new(elementwise::Atanh::new(self.inner))
    }
}

// Specialized methods for float tensors (f32 and f64)
impl<const R: usize, E, T> Tensor<R, T>
where
    E: FloatOps,
    T: TensorBacking<R, Elem = E>,
{
    /// Raise each element to a power
    #[inline]
    pub fn pow_scalar(self, exponent: E) -> Tensor<R, ConcreteTensor<E, R>> {
        let concrete = self.inner.to_concrete();
        let shape: [usize; R] = concrete
            .layout()
            .shape()
            .try_into()
            .expect("Shape length mismatch");
        Tensor::new(ConcreteTensor::from_fn(shape, |i| {
            concrete.data()[i].powf(exponent)
        }))
    }

    /// Element-wise maximum with a scalar
    #[inline]
    pub fn max_scalar(self, scalar: E) -> Tensor<R, ConcreteTensor<E, R>> {
        let concrete = self.inner.to_concrete();
        let shape: [usize; R] = concrete
            .layout()
            .shape()
            .try_into()
            .expect("Shape length mismatch");
        Tensor::new(ConcreteTensor::from_fn(shape, |i| {
            concrete.data()[i].float_max(scalar)
        }))
    }

    /// Element-wise minimum with a scalar
    #[inline]
    pub fn min_scalar(self, scalar: E) -> Tensor<R, ConcreteTensor<E, R>> {
        let concrete = self.inner.to_concrete();
        let shape: [usize; R] = concrete
            .layout()
            .shape()
            .try_into()
            .expect("Shape length mismatch");
        Tensor::new(ConcreteTensor::from_fn(shape, |i| {
            concrete.data()[i].float_min(scalar)
        }))
    }

    /// Clamp each element to a range [min, max]
    #[inline]
    pub fn clamp(self, min: E, max: E) -> Tensor<R, ConcreteTensor<E, R>> {
        let concrete = self.inner.to_concrete();
        let shape: [usize; R] = concrete
            .layout()
            .shape()
            .try_into()
            .expect("Shape length mismatch");
        Tensor::new(ConcreteTensor::from_fn(shape, |i| {
            concrete.data()[i].float_max(min).float_min(max)
        }))
    }
}

// Matrix multiplication for N-dimensional tensors (N >= 2)
impl<const R: usize, E, T> Tensor<R, T>
where
    E: SimdElement + MatmulImpl,
    T: TensorBacking<R, Elem = E>,
{
    /// Matrix multiplication (batched for rank > 2)
    /// For 2D: [M, K] @ [K, N] -> [M, N]
    /// For ND: [...batch, M, K] @ [...batch, K, N] -> [...batch, M, N]
    /// Panics if R < 2
    #[inline]
    pub fn matmul(
        self,
        rhs: Tensor<R, impl TensorBacking<R, Elem = E>>,
    ) -> Tensor<R, ConcreteTensor<E, R>> {
        Tensor::new(
            self.inner
                .to_concrete()
                .matmul_ref(&rhs.inner.to_concrete()),
        )
    }
}

impl<const R: usize, T: TensorBacking<R>> crate::LazyBacking for Tensor<R, T> {
    type Elem = T::Elem;

    #[inline(always)]
    fn eval_scalar(&self, idx: usize) -> Self::Elem {
        self.inner.eval_scalar(idx)
    }

    #[inline(always)]
    fn eval_simd<S: Simd>(&self, simd: S, base_idx: usize) -> <Self::Elem as SimdElement>::Simd<S> {
        self.inner.eval_simd(simd, base_idx)
    }
}

impl<const R: usize, T: TensorBacking<R>> TensorBacking<R> for Tensor<R, T> {
    fn layout(&self) -> fusor_types::Layout {
        self.inner.layout()
    }

    fn to_concrete(&self) -> ConcreteTensor<T::Elem, R> {
        self.inner.to_concrete()
    }
}

/// Macro to implement pairwise operators for CPU Tensor.
///
/// Generates all four combinations of owned/reference implementations:
/// - `Tensor op Tensor` (owned + owned)
/// - `&Tensor op &Tensor` (ref + ref)
/// - `Tensor op &Tensor` (owned + ref)
/// - `&Tensor op Tensor` (ref + owned)
macro_rules! impl_cpu_pairwise_op {
    ($std_trait:ident, $method:ident, $pairwise_ty:ident, $simd_op:ident) => {
        // Owned + Owned
        impl<const R: usize, E, T1, T2> $std_trait<Tensor<R, T2>> for Tensor<R, T1>
        where
            E: SimdElement + $std_trait<Output = E> + Default,
            T1: TensorBacking<R, Elem = E>,
            T2: TensorBacking<R, Elem = E>,
            $simd_op: SimdBinaryOp<E>,
        {
            type Output = Tensor<R, pairwise::$pairwise_ty<E, R, T1, T2>>;

            fn $method(self, rhs: Tensor<R, T2>) -> Self::Output {
                Tensor::new(pairwise::$pairwise_ty::new(self.inner, rhs.inner))
            }
        }

        // Ref + Ref
        impl<'a, const R: usize, E, T1, T2> $std_trait<&'a Tensor<R, T2>> for &'a Tensor<R, T1>
        where
            E: SimdElement + $std_trait<Output = E> + Default,
            T1: TensorBacking<R, Elem = E>,
            T2: TensorBacking<R, Elem = E>,
            $simd_op: SimdBinaryOp<E>,
        {
            type Output = Tensor<R, pairwise::$pairwise_ty<E, R, &'a T1, &'a T2>>;

            fn $method(self, rhs: &'a Tensor<R, T2>) -> Self::Output {
                Tensor::new(pairwise::$pairwise_ty::new(&self.inner, &rhs.inner))
            }
        }

        // Owned + Ref
        impl<'a, const R: usize, E, T1, T2> $std_trait<&'a Tensor<R, T2>> for Tensor<R, T1>
        where
            E: SimdElement + $std_trait<Output = E> + Default,
            T1: TensorBacking<R, Elem = E>,
            T2: TensorBacking<R, Elem = E>,
            $simd_op: SimdBinaryOp<E>,
        {
            type Output = Tensor<R, pairwise::$pairwise_ty<E, R, T1, &'a T2>>;

            fn $method(self, rhs: &'a Tensor<R, T2>) -> Self::Output {
                Tensor::new(pairwise::$pairwise_ty::new(self.inner, &rhs.inner))
            }
        }

        // Ref + Owned
        impl<'a, const R: usize, E, T1, T2> $std_trait<Tensor<R, T2>> for &'a Tensor<R, T1>
        where
            E: SimdElement + $std_trait<Output = E> + Default,
            T1: TensorBacking<R, Elem = E>,
            T2: TensorBacking<R, Elem = E>,
            $simd_op: SimdBinaryOp<E>,
        {
            type Output = Tensor<R, pairwise::$pairwise_ty<E, R, &'a T1, T2>>;

            fn $method(self, rhs: Tensor<R, T2>) -> Self::Output {
                Tensor::new(pairwise::$pairwise_ty::new(&self.inner, rhs.inner))
            }
        }
    };
}

impl_cpu_pairwise_op!(StdAdd, add, Add, AddOp);
impl_cpu_pairwise_op!(StdSub, sub, Sub, SubOp);
impl_cpu_pairwise_op!(StdMul, mul, Mul, MulOp);
impl_cpu_pairwise_op!(StdDiv, div, Div, DivOp);
impl_cpu_pairwise_op!(StdRem, rem, Rem, RemOp);

// Neg is unary, so handle separately
impl<const R: usize, T> StdNeg for Tensor<R, T>
where
    T: TensorBacking<R>,
    <T as crate::LazyBacking>::Elem:
        SimdElement + StdNeg<Output = <T as crate::LazyBacking>::Elem> + Default,
    NegOp: SimdUnaryOp<<T as crate::LazyBacking>::Elem>,
{
    type Output = Tensor<R, elementwise::Neg<<T as crate::LazyBacking>::Elem, R, T>>;

    fn neg(self) -> Self::Output {
        Tensor::new(elementwise::Neg::new(self.inner))
    }
}

impl<'a, const R: usize, T> StdNeg for &'a Tensor<R, T>
where
    T: TensorBacking<R>,
    <T as crate::LazyBacking>::Elem:
        SimdElement + StdNeg<Output = <T as crate::LazyBacking>::Elem> + Default,
    NegOp: SimdUnaryOp<<T as crate::LazyBacking>::Elem>,
{
    type Output = Tensor<R, elementwise::Neg<<T as crate::LazyBacking>::Elem, R, &'a T>>;

    fn neg(self) -> Self::Output {
        Tensor::new(elementwise::Neg::new(&self.inner))
    }
}

/// Marker trait for scalar types (not tensors).
/// This is used to disambiguate `Tensor * scalar` from `Tensor * Tensor`.
pub trait Scalar: Copy {}

impl Scalar for f32 {}
impl Scalar for f64 {}
impl Scalar for i8 {}
impl Scalar for i16 {}
impl Scalar for i32 {}
impl Scalar for i64 {}
impl Scalar for u8 {}
impl Scalar for u16 {}
impl Scalar for u32 {}
impl Scalar for u64 {}

// Scalar multiplication: Tensor * scalar
impl<const R: usize, T, E> StdMul<E> for Tensor<R, T>
where
    T: TensorBacking<R, Elem = E>,
    E: SimdElement + StdMul<Output = E> + Default + Scalar,
    MulOp: SimdBinaryOp<E>,
{
    type Output = Tensor<R, scalar::MulScalar<E, R, T>>;

    fn mul(self, rhs: E) -> Self::Output {
        Tensor::new(scalar::MulScalar::new(self.inner, rhs))
    }
}

impl<'a, const R: usize, T, E> StdMul<E> for &'a Tensor<R, T>
where
    T: TensorBacking<R, Elem = E>,
    E: SimdElement + StdMul<Output = E> + Default + Scalar,
    MulOp: SimdBinaryOp<E>,
{
    type Output = Tensor<R, scalar::MulScalar<E, R, &'a T>>;

    fn mul(self, rhs: E) -> Self::Output {
        Tensor::new(scalar::MulScalar::new(&self.inner, rhs))
    }
}

// Scalar addition: Tensor + scalar
impl<const R: usize, T, E> StdAdd<E> for Tensor<R, T>
where
    T: TensorBacking<R, Elem = E>,
    E: SimdElement + StdAdd<Output = E> + Default + Scalar,
    AddOp: SimdBinaryOp<E>,
{
    type Output = Tensor<R, scalar::AddScalar<E, R, T>>;

    fn add(self, rhs: E) -> Self::Output {
        Tensor::new(scalar::AddScalar::new(self.inner, rhs))
    }
}

impl<'a, const R: usize, T, E> StdAdd<E> for &'a Tensor<R, T>
where
    T: TensorBacking<R, Elem = E>,
    E: SimdElement + StdAdd<Output = E> + Default + Scalar,
    AddOp: SimdBinaryOp<E>,
{
    type Output = Tensor<R, scalar::AddScalar<E, R, &'a T>>;

    fn add(self, rhs: E) -> Self::Output {
        Tensor::new(scalar::AddScalar::new(&self.inner, rhs))
    }
}

// Scalar arithmetic methods for Tensor
impl<const R: usize, E, T> Tensor<R, T>
where
    E: SimdElement + Default,
    T: TensorBacking<R, Elem = E>,
{
    /// Add a scalar to each element
    #[inline]
    pub fn add_scalar(self, scalar_val: E) -> Tensor<R, scalar::AddScalar<E, R, T>>
    where
        E: StdAdd<Output = E>,
        AddOp: SimdBinaryOp<E>,
    {
        Tensor::new(scalar::AddScalar::new(self.inner, scalar_val))
    }

    /// Subtract a scalar from each element
    #[inline]
    pub fn sub_scalar(self, scalar_val: E) -> Tensor<R, scalar::SubScalar<E, R, T>>
    where
        E: StdSub<Output = E>,
        SubOp: SimdBinaryOp<E>,
    {
        Tensor::new(scalar::SubScalar::new(self.inner, scalar_val))
    }

    /// Multiply each element by a scalar
    #[inline]
    pub fn mul_scalar(self, scalar_val: E) -> Tensor<R, scalar::MulScalar<E, R, T>>
    where
        E: StdMul<Output = E>,
        MulOp: SimdBinaryOp<E>,
    {
        Tensor::new(scalar::MulScalar::new(self.inner, scalar_val))
    }

    /// Divide each element by a scalar
    #[inline]
    pub fn div_scalar(self, scalar_val: E) -> Tensor<R, scalar::DivScalar<E, R, T>>
    where
        E: StdDiv<Output = E>,
        DivOp: SimdBinaryOp<E>,
    {
        Tensor::new(scalar::DivScalar::new(self.inner, scalar_val))
    }
}

/// Calculate the broadcasted shape for two tensors.
/// Returns the output shape where each dimension is the max of the corresponding input dimensions.
/// Dimensions are aligned from the right.
fn broadcast_shapes<const R1: usize, const R2: usize, const R3: usize>(
    shape1: &[usize; R1],
    shape2: &[usize; R2],
) -> [usize; R3] {
    let mut result = [1usize; R3];

    // Align shapes from the right
    for (i, &dim) in shape1.iter().enumerate().take(R1) {
        let idx = R3 - R1 + i;
        result[idx] = dim;
    }

    for i in 0..R2 {
        let idx = R3 - R2 + i;
        let d2 = shape2[i];
        let d1 = result[idx];
        if d1 == 1 {
            result[idx] = d2;
        } else if d2 != 1 && d1 != d2 {
            panic!(
                "Cannot broadcast shapes {:?} and {:?}: incompatible dimensions {} and {} at index {}",
                shape1, shape2, d1, d2, idx
            );
        }
    }

    result
}

// Broadcasting binary operations for multi-rank tensors
// These methods handle broadcasting when tensors have different ranks.
impl<const R: usize, E, T> Tensor<R, T>
where
    E: SimdElement + Default,
    T: TensorBacking<R, Elem = E>,
{
    /// Multiply with broadcasting support (multi-rank tensors).
    ///
    /// Both tensors are broadcast to a common shape before multiplication.
    /// Supports tensors of different ranks.
    pub fn mul_<const R2: usize, const R3: usize, T2>(
        self,
        other: Tensor<R2, T2>,
    ) -> Tensor<R3, ConcreteTensor<E, R3>>
    where
        E: StdMul<Output = E>,
        MulOp: SimdBinaryOp<E>,
        T2: TensorBacking<R2, Elem = E>,
        (ConcreteTensor<E, R>, ConcreteTensor<E, R2>): MaxRank<R3, E>,
    {
        let shape1: [usize; R] = self.layout().shape().try_into().unwrap();
        let shape2: [usize; R2] = other.layout().shape().try_into().unwrap();
        let out_shape: [usize; R3] = broadcast_shapes(&shape1, &shape2);

        let a = self.broadcast_as(out_shape);
        let b = other.broadcast_as(out_shape);

        (&a * &b).to_concrete()
    }

    /// Add with broadcasting support (multi-rank tensors).
    ///
    /// Both tensors are broadcast to a common shape before addition.
    /// Supports tensors of different ranks.
    pub fn add_<const R2: usize, const R3: usize, T2>(
        self,
        other: Tensor<R2, T2>,
    ) -> Tensor<R3, ConcreteTensor<E, R3>>
    where
        E: StdAdd<Output = E>,
        AddOp: SimdBinaryOp<E>,
        T2: TensorBacking<R2, Elem = E>,
        (ConcreteTensor<E, R>, ConcreteTensor<E, R2>): MaxRank<R3, E>,
    {
        let shape1: [usize; R] = self.layout().shape().try_into().unwrap();
        let shape2: [usize; R2] = other.layout().shape().try_into().unwrap();
        let out_shape: [usize; R3] = broadcast_shapes(&shape1, &shape2);

        let a = self.broadcast_as(out_shape);
        let b = other.broadcast_as(out_shape);

        (&a + &b).to_concrete()
    }

    /// Subtract with broadcasting support (multi-rank tensors).
    ///
    /// Both tensors are broadcast to a common shape before subtraction.
    /// Supports tensors of different ranks.
    pub fn sub_<const R2: usize, const R3: usize, T2>(
        self,
        other: Tensor<R2, T2>,
    ) -> Tensor<R3, ConcreteTensor<E, R3>>
    where
        E: StdSub<Output = E>,
        SubOp: SimdBinaryOp<E>,
        T2: TensorBacking<R2, Elem = E>,
        (ConcreteTensor<E, R>, ConcreteTensor<E, R2>): MaxRank<R3, E>,
    {
        let shape1: [usize; R] = self.layout().shape().try_into().unwrap();
        let shape2: [usize; R2] = other.layout().shape().try_into().unwrap();
        let out_shape: [usize; R3] = broadcast_shapes(&shape1, &shape2);

        let a = self.broadcast_as(out_shape);
        let b = other.broadcast_as(out_shape);

        (&a - &b).to_concrete()
    }

    /// Divide with broadcasting support (multi-rank tensors).
    ///
    /// Both tensors are broadcast to a common shape before division.
    /// Supports tensors of different ranks.
    pub fn div_<const R2: usize, const R3: usize, T2>(
        self,
        other: Tensor<R2, T2>,
    ) -> Tensor<R3, ConcreteTensor<E, R3>>
    where
        E: StdDiv<Output = E>,
        DivOp: SimdBinaryOp<E>,
        T2: TensorBacking<R2, Elem = E>,
        (ConcreteTensor<E, R>, ConcreteTensor<E, R2>): MaxRank<R3, E>,
    {
        let shape1: [usize; R] = self.layout().shape().try_into().unwrap();
        let shape2: [usize; R2] = other.layout().shape().try_into().unwrap();
        let out_shape: [usize; R3] = broadcast_shapes(&shape1, &shape2);

        let a = self.broadcast_as(out_shape);
        let b = other.broadcast_as(out_shape);

        (&a / &b).to_concrete()
    }

    /// Power with broadcasting support (multi-rank tensors).
    ///
    /// Both tensors are broadcast to a common shape before computing power.
    /// Supports tensors of different ranks.
    pub fn pow_<const R2: usize, const R3: usize, T2>(
        self,
        other: Tensor<R2, T2>,
    ) -> Tensor<R3, ConcreteTensor<E, R3>>
    where
        E: FloatOps,
        T2: TensorBacking<R2, Elem = E>,
        (ConcreteTensor<E, R>, ConcreteTensor<E, R2>): MaxRank<R3, E>,
    {
        let shape1: [usize; R] = self.layout().shape().try_into().unwrap();
        let shape2: [usize; R2] = other.layout().shape().try_into().unwrap();
        let out_shape: [usize; R3] = broadcast_shapes(&shape1, &shape2);

        let a: Tensor<R3, ConcreteTensor<E, R3>> = self.broadcast_as(out_shape).to_concrete();
        let b: Tensor<R3, ConcreteTensor<E, R3>> = other.broadcast_as(out_shape).to_concrete();

        // Compute power element-wise
        let a_data = ResolvedTensor::data(a.inner());
        let b_data = ResolvedTensor::data(b.inner());
        let result: Vec<E> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(x, y)| x.powf(*y))
            .collect();

        Tensor::new(ConcreteTensor::from_slice(out_shape, &result))
    }
}

// Matrix transpose for 2D tensors
impl<E, T> Tensor<2, T>
where
    E: SimdElement,
    T: TensorBacking<2, Elem = E>,
{
    /// Transpose a 2D matrix (swap dimensions 0 and 1)
    pub fn t(self) -> Tensor<2, MapLayout<T, 2>> {
        self.transpose(0, 1)
    }
}

// Matrix transpose for 3D tensors
impl<E, T> Tensor<3, T>
where
    E: SimdElement,
    T: TensorBacking<3, Elem = E>,
{
    /// Transpose last two dimensions
    pub fn t(self) -> Tensor<3, MapLayout<T, 3>> {
        self.transpose(1, 2)
    }
}

// Matrix transpose for 4D tensors
impl<E, T> Tensor<4, T>
where
    E: SimdElement,
    T: TensorBacking<4, Elem = E>,
{
    /// Transpose last two dimensions
    pub fn t(self) -> Tensor<4, MapLayout<T, 4>> {
        self.transpose(2, 3)
    }
}

// Static methods for concatenation and stacking
impl<const R: usize, T: TensorBacking<R>> Tensor<R, T> {
    /// Concatenate multiple tensors along a given dimension
    ///
    /// # Arguments
    /// * `tensors` - Iterator of tensors to concatenate
    /// * `dim` - The dimension to concatenate along
    pub fn cat(
        tensors: impl IntoIterator<Item = Self>,
        dim: usize,
    ) -> Tensor<R, ConcreteTensor<T::Elem, R>> {
        let tensors: Vec<_> = tensors.into_iter().map(|t| t.to_concrete()).collect();
        assert!(
            !tensors.is_empty(),
            "Cannot concatenate empty list of tensors"
        );
        assert!(dim < R, "Dimension {} out of range for rank {}", dim, R);

        let first_shape = tensors[0].inner.layout().shape();

        // Validate shapes and calculate output dimension size
        let mut cat_dim_size = 0;
        for tensor in &tensors {
            let shape = tensor.inner.layout().shape();
            for (i, (&s1, &s2)) in first_shape.iter().zip(shape.iter()).enumerate() {
                if i == dim {
                    cat_dim_size += s2;
                } else {
                    assert_eq!(
                        s1, s2,
                        "Shape mismatch at dimension {}: {} vs {}",
                        i, s1, s2
                    );
                }
            }
        }

        // Build output shape
        let mut out_shape = [0usize; R];
        for (i, &s) in first_shape.iter().enumerate() {
            out_shape[i] = if i == dim { cat_dim_size } else { s };
        }

        let mut output = ConcreteTensor::<T::Elem, R>::zeros(out_shape);
        let mut offset_in_dim = 0;

        for tensor in tensors {
            let tensor_shape = tensor.inner.layout().shape();
            let tensor_dim_size = tensor_shape[dim];

            for indices in IndexIterator::new(tensor_shape) {
                let src_arr: [usize; R] =
                    indices.clone().try_into().expect("Indices length mismatch");

                let mut dst_arr = src_arr;
                dst_arr[dim] += offset_in_dim;

                let src_idx = tensor.inner.layout().linear_index(&src_arr);
                let dst_idx = output.layout().linear_index(&dst_arr);
                output.backing_mut()[dst_idx] = tensor.inner.backing()[src_idx];
            }

            offset_in_dim += tensor_dim_size;
        }

        Tensor::new(output)
    }

    /// Stack tensors along a new dimension
    ///
    /// # Arguments
    /// * `tensors` - Iterator of tensors to stack
    /// * `dim` - Where to insert the new stacking dimension
    pub fn stack<const R2: usize>(
        tensors: impl IntoIterator<Item = Self>,
        dim: usize,
    ) -> Tensor<R2, ConcreteTensor<T::Elem, R2>> {
        assert!(R2 == R + 1, "Output rank must be R + 1");
        assert!(
            dim <= R,
            "Stack dimension {} out of range for rank {}",
            dim,
            R
        );

        let tensors: Vec<_> = tensors.into_iter().map(|t| t.to_concrete()).collect();
        assert!(!tensors.is_empty(), "Cannot stack empty list of tensors");

        let first_shape = tensors[0].inner.layout().shape();

        // Validate all tensors have same shape
        for tensor in &tensors {
            assert_eq!(
                tensor.inner.layout().shape(),
                first_shape,
                "All tensors must have same shape for stacking"
            );
        }

        // Unsqueeze each tensor and concatenate
        let unsqueezed: Vec<Tensor<R2, ConcreteTensor<T::Elem, R2>>> = tensors
            .into_iter()
            .map(|t| t.unsqueeze(dim).to_concrete())
            .collect();

        Tensor::cat(unsqueezed, dim)
    }
}

// Static methods for creating 1D tensors
impl<E: SimdElement> Tensor<1, ConcreteTensor<E, 1>> {
    /// Create a range tensor from start (inclusive) to end (exclusive)
    ///
    /// # Arguments
    /// * `start` - Starting value
    /// * `end` - Ending value (exclusive)
    pub fn arange(start: E, end: E) -> Self
    where
        E: std::ops::Add<Output = E> + PartialOrd + From<u8>,
    {
        Self::arange_step(start, end, E::from(1u8))
    }

    /// Create a range tensor with a custom step
    ///
    /// # Arguments
    /// * `start` - Starting value
    /// * `end` - Ending value (exclusive)
    /// * `step` - Step size between values
    pub fn arange_step(start: E, end: E, step: E) -> Self
    where
        E: std::ops::Add<Output = E> + PartialOrd,
    {
        let mut values = Vec::new();
        let mut current = start;
        while current < end {
            values.push(current);
            current = current + step;
        }

        let len = values.len();
        Tensor::from_slice([len], &values)
    }
}

// FromArray implementations for CPU tensors (using () as device type since CPU has no device)
impl<E: SimdElement + Default> fusor_types::FromArray<0, E, (), ()>
    for Tensor<0, ConcreteTensor<E, 0>>
{
    fn from_array(_data: (), _device: &()) -> Self {
        Tensor::from_slice([], &[])
    }
}

impl<'a, I, E: SimdElement + Default + Copy> fusor_types::FromArray<1, E, I, ()>
    for Tensor<1, ConcreteTensor<E, 1>>
where
    I: IntoIterator<Item = &'a E, IntoIter: ExactSizeIterator>,
{
    fn from_array(data: I, _device: &()) -> Self {
        let data_vec: Vec<E> = data.into_iter().copied().collect();
        let len = data_vec.len();
        Tensor::from_slice([len], &data_vec)
    }
}

impl<'a, I, I2, E: SimdElement + Default + Copy> fusor_types::FromArray<2, E, I, ()>
    for Tensor<2, ConcreteTensor<E, 2>>
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = &'a E, IntoIter: ExactSizeIterator>,
{
    fn from_array(data: I, _device: &()) -> Self {
        let mut iter = data.into_iter().map(IntoIterator::into_iter).peekable();
        let size = iter.len();
        let second_size = iter.peek().map(ExactSizeIterator::len).unwrap_or_default();
        let data_vec: Vec<E> = iter
            .flat_map(|i| {
                let size = i.len();
                if size != second_size {
                    panic!("expected a rectangular matrix. The first inner iterator size was {second_size}, but another inner iterator size was {size}");
                }
                i.copied()
            })
            .collect();
        Tensor::from_slice([size, second_size], &data_vec)
    }
}

impl<'a, I, I2, I3, E: SimdElement + Default + Copy> fusor_types::FromArray<3, E, I, ()>
    for Tensor<3, ConcreteTensor<E, 3>>
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = I3, IntoIter: ExactSizeIterator>,
    I3: IntoIterator<Item = &'a E, IntoIter: ExactSizeIterator>,
{
    fn from_array(data: I, _device: &()) -> Self {
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

        let data_vec: Vec<E> = iter
            .flat_map(|i| {
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
                    i.copied()
                })
            })
            .collect();

        Tensor::from_slice(shape, &data_vec)
    }
}

impl<'a, I, I2, I3, I4, E: SimdElement + Default + Copy> fusor_types::FromArray<4, E, I, ()>
    for Tensor<4, ConcreteTensor<E, 4>>
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = I3, IntoIter: ExactSizeIterator>,
    I3: IntoIterator<Item = I4, IntoIter: ExactSizeIterator>,
    I4: IntoIterator<Item = &'a E, IntoIter: ExactSizeIterator>,
{
    fn from_array(data: I, _device: &()) -> Self {
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

        let data_vec: Vec<E> = iter
            .flat_map(|i| {
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
                        i.copied()
                    })
                })
            })
            .collect();

        Tensor::from_slice(shape, &data_vec)
    }
}

impl<'a, I, I2, I3, I4, I5, E: SimdElement + Default + Copy> fusor_types::FromArray<5, E, I, ()>
    for Tensor<5, ConcreteTensor<E, 5>>
where
    I: IntoIterator<Item = I2, IntoIter: ExactSizeIterator>,
    I2: IntoIterator<Item = I3, IntoIter: ExactSizeIterator>,
    I3: IntoIterator<Item = I4, IntoIter: ExactSizeIterator>,
    I4: IntoIterator<Item = I5, IntoIter: ExactSizeIterator>,
    I5: IntoIterator<Item = &'a E, IntoIter: ExactSizeIterator>,
{
    fn from_array(data: I, _device: &()) -> Self {
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

        let data_vec: Vec<E> = iter
            .flat_map(|i| {
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
                            i.copied()
                        })
                    })
                })
            })
            .collect();

        Tensor::from_slice(shape, &data_vec)
    }
}
