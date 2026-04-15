//! Shape manipulation operations that work on both CPU and GPU backends.

use std::ops::Range;

use crate::{ConcreteTensor, Device, SimdElement, Tensor};
use fusor_core::{DataType, Dim, ShapeWithOneHole};
use fusor_cpu::{MapLayout, TensorBacking};
use fusor_types::{Layout, SlidingWindow, StrideSpec};

impl<const R: usize, D, B> Tensor<R, D, B>
where
    D: SimdElement + DataType + Default,
    B: TensorBacking<R, Elem = D>,
{
    /// Reshape the tensor to a new shape.
    ///
    /// The total number of elements must remain the same.
    pub fn reshape<const R2: usize>(
        &self,
        new_shape: impl ShapeWithOneHole<R2>,
    ) -> Tensor<R2, D, MapLayout<&B, R2>> {
        match self {
            Tensor::Cpu(t) => {
                let resolved_shape = new_shape.resolve_shape(&t.shape());
                Tensor::Cpu(t.as_ref().reshape(resolved_shape))
            }
            Tensor::Gpu(t) => Tensor::Gpu(t.reshape(new_shape)),
        }
    }

    /// Transpose two dimensions of the tensor.
    ///
    /// # Arguments
    /// * `dim0` - First dimension to swap
    /// * `dim1` - Second dimension to swap
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor<R, D, MapLayout<&B, R>> {
        let shape = self.shape();
        let specs: [StrideSpec; R] = std::array::from_fn(|i| {
            if i == dim0 {
                StrideSpec::dim(dim1, shape[dim1])
            } else if i == dim1 {
                StrideSpec::dim(dim0, shape[dim0])
            } else {
                StrideSpec::dim(i, shape[i])
            }
        });
        self.restride(specs)
    }

    /// Slice the tensor along all dimensions.
    ///
    /// Returns a view into the tensor's data with updated layout.
    pub fn slice(&self, slices: [Range<usize>; R]) -> Tensor<R, D, MapLayout<&B, R>> {
        let specs: [StrideSpec; R] = std::array::from_fn(|i| {
            StrideSpec::dim(i, slices[i].len()).with_offset(slices[i].start)
        });
        self.restride(specs)
    }

    /// Create a view with stride patterns specified per output dimension.
    ///
    /// Each [`StrideSpec`] maps an output dimension to an input dimension's stride
    /// with an optional multiplier. This is relative to the current strides, so it
    /// composes correctly when the GPU optimizer changes upstream strides.
    /// The output rank can differ from the input.
    pub fn restride<const R2: usize>(
        &self,
        specs: [StrideSpec; R2],
    ) -> Tensor<R2, D, MapLayout<&B, R2>> {
        match self {
            Tensor::Cpu(t) => {
                let new_layout = t.layout().restride(&specs);
                Tensor::Cpu(t.as_ref().restride_layout(new_layout))
            }
            Tensor::Gpu(t) => Tensor::Gpu(t.restride(specs)),
        }
    }

    /// Set the layout directly from a pre-computed [`Layout`].
    pub fn restride_layout<const R2: usize>(
        &self,
        new_layout: Layout,
    ) -> Tensor<R2, D, MapLayout<&B, R2>> {
        match self {
            Tensor::Cpu(t) => Tensor::Cpu(t.as_ref().restride_layout(new_layout.clone())),
            Tensor::Gpu(t) => Tensor::Gpu(t.restride_layout(new_layout)),
        }
    }

    /// Permute the tensor dimensions according to the given axes order.
    ///
    /// # Arguments
    /// * `axes` - A permutation of [0, 1, ..., R-1] specifying the new order
    pub fn permute(&self, axes: [usize; R]) -> Tensor<R, D, MapLayout<&B, R>> {
        let shape = self.shape();
        let specs: [StrideSpec; R] =
            std::array::from_fn(|i| StrideSpec::dim(axes[i], shape[axes[i]]));
        self.restride(specs)
    }

    /// Broadcast the tensor to a larger shape.
    ///
    /// Broadcasting rules:
    /// - Dimensions are aligned from the right
    /// - A dimension can be broadcast if it's 1 or matches the target
    /// - New dimensions can be added on the left
    pub fn broadcast_as<const R2: usize>(
        &self,
        out_shape: [usize; R2],
    ) -> Tensor<R2, D, MapLayout<&B, R2>> {
        let shape = self.shape();
        let specs: [StrideSpec; R2] = std::array::from_fn(|out_i| {
            let in_i = out_i as isize - (R2 as isize - R as isize);
            if in_i < 0 {
                StrideSpec::dim_with(0, out_shape[out_i], 0)
            } else {
                let in_i = in_i as usize;
                if shape[in_i] == 1 && out_shape[out_i] > 1 {
                    StrideSpec::dim_with(in_i, out_shape[out_i], 0)
                } else {
                    StrideSpec::dim(in_i, out_shape[out_i])
                }
            }
        });
        self.restride(specs)
    }

    /// Expand the tensor to a larger shape (alias for broadcast_as).
    pub fn expand<const R2: usize>(
        &self,
        out_shape: [usize; R2],
    ) -> Tensor<R2, D, MapLayout<&B, R2>> {
        self.broadcast_as(out_shape)
    }

    /// Flatten the tensor to 1D.
    pub fn flatten_all(&self) -> Tensor<1, D, MapLayout<&B, 1>> {
        let total = self.shape().iter().product();
        self.reshape([total])
    }

    /// Narrow the tensor along a given dimension.
    ///
    /// # Arguments
    /// * `dim` - The dimension to narrow (can be `usize` or `D::Minus1`, etc.)
    /// * `start` - The starting index
    /// * `length` - The length of the slice
    pub fn narrow(
        &self,
        dim: impl Dim<R>,
        start: usize,
        length: usize,
    ) -> Tensor<R, D, MapLayout<&B, R>> {
        let dim = dim.resolve();
        let shape = self.shape();
        let specs: [StrideSpec; R] = std::array::from_fn(|i| {
            if i == dim {
                StrideSpec::dim(i, length).with_offset(start)
            } else {
                StrideSpec::dim(i, shape[i])
            }
        });
        self.restride(specs)
    }

    /// Split the tensor into chunks along a given dimension.
    ///
    /// # Arguments
    /// * `chunks` - Number of chunks to split into
    /// * `dim` - The dimension to split along (can be `usize` or `D::Minus1`, etc.)
    pub fn chunk(&self, chunks: usize, dim: impl Dim<R>) -> Vec<Tensor<R, D, MapLayout<&B, R>>> {
        let dim = dim.resolve();
        let shape = self.shape();
        let dim_size = shape[dim];
        let chunk_size = dim_size.div_ceil(chunks);

        let mut result = Vec::with_capacity(chunks);
        let mut start = 0;

        while start < dim_size {
            let length = chunk_size.min(dim_size - start);
            result.push(self.narrow(dim, start, length));
            start += length;
        }

        result
    }

    /// Repeat the tensor along each dimension.
    ///
    /// # Arguments
    /// * `repeats` - Number of times to repeat along each dimension
    pub fn repeat(&self, repeats: [usize; R]) -> Tensor<R, D> {
        // Concatenate copies along each dimension
        let mut result: Tensor<R, D> = self.to_concrete();
        for dim in 0..R {
            if repeats[dim] > 1 {
                let copies: Vec<Tensor<R, D>> = (0..repeats[dim]).map(|_| result.clone()).collect();
                result = cat(copies, dim);
            }
        }
        result
    }

    /// Squeeze a dimension of size 1.
    ///
    /// # Arguments
    /// * `dim` - The dimension to squeeze (must have size 1)
    pub fn squeeze<const R2: usize>(&self, dim: usize) -> Tensor<R2, D, MapLayout<&B, R2>>
    where
        ConcreteTensor<D, R>: fusor_cpu::LastRank<R2, D>,
        fusor_core::Tensor<R, D>: fusor_core::LastRank<R2, D>,
    {
        let shape = self.shape();
        assert_eq!(shape[dim], 1, "Squeeze dimension must have size 1");
        let specs: [StrideSpec; R2] = std::array::from_fn(|out_i| {
            let in_i = if out_i < dim { out_i } else { out_i + 1 };
            StrideSpec::dim(in_i, shape[in_i])
        });
        self.restride(specs)
    }

    /// Unsqueeze (add a dimension of size 1).
    ///
    /// # Arguments
    /// * `dim` - Where to insert the new dimension
    pub fn unsqueeze<const R2: usize>(&self, dim: usize) -> Tensor<R2, D, MapLayout<&B, R2>>
    where
        ConcreteTensor<D, R>: fusor_cpu::NextRank<R2, D>,
        fusor_core::Tensor<R, D>: fusor_core::NextRank<R2, D>,
    {
        let shape = self.shape();
        let specs: [StrideSpec; R2] = std::array::from_fn(|out_i| {
            if out_i == dim {
                StrideSpec::dim_with(0, 1, 0)
            } else {
                let in_i = if out_i < dim { out_i } else { out_i - 1 };
                StrideSpec::dim(in_i, shape[in_i])
            }
        });
        self.restride(specs)
    }

    /// Squeeze multiple dimensions of size 1.
    ///
    /// # Type Parameters
    /// * `DIFF` - Number of dimensions to squeeze
    /// * `R2` - Output rank (must be R - DIFF)
    ///
    /// # Arguments
    /// * `axes` - Array of dimensions to squeeze (each must have size 1)
    pub fn squeeze_dims<const DIFF: usize, const R2: usize>(
        &self,
        axes: [usize; DIFF],
    ) -> Tensor<R2, D, MapLayout<&B, R2>>
    where
        ConcreteTensor<D, R>: fusor_cpu::SmallerRank<R2, DIFF, D>,
        fusor_core::Tensor<R, D>: fusor_core::SmallerRank<DIFF, R2, D>,
    {
        let shape = self.shape();
        for &ax in &axes {
            assert_eq!(shape[ax], 1, "Squeeze dimension {} must have size 1", ax);
        }
        let mut sorted_axes = axes;
        sorted_axes.sort_unstable();
        let mut in_i = 0;
        let mut axis_idx = 0;
        let specs: [StrideSpec; R2] = std::array::from_fn(|_| {
            while axis_idx < DIFF && in_i == sorted_axes[axis_idx] {
                in_i += 1;
                axis_idx += 1;
            }
            let spec = StrideSpec::dim(in_i, shape[in_i]);
            in_i += 1;
            spec
        });
        self.restride(specs)
    }

    /// Unsqueeze multiple dimensions (add dimensions of size 1).
    ///
    /// # Type Parameters
    /// * `DIFF` - Number of dimensions to add
    /// * `R2` - Output rank (must be R + DIFF)
    ///
    /// # Arguments
    /// * `axes` - Array of positions where to insert new dimensions
    pub fn unsqueeze_dims<const DIFF: usize, const R2: usize>(
        &self,
        axes: [usize; DIFF],
    ) -> Tensor<R2, D, MapLayout<&B, R2>>
    where
        ConcreteTensor<D, R>: fusor_cpu::LargerRank<R2, DIFF, D>,
        fusor_core::Tensor<R, D>: fusor_core::LargerRank<DIFF, R2, D>,
    {
        let shape = self.shape();
        let mut sorted_axes = axes;
        sorted_axes.sort_unstable();
        let mut old_idx = 0;
        let mut axis_idx = 0;
        let specs: [StrideSpec; R2] = std::array::from_fn(|out_i| {
            if axis_idx < DIFF && out_i == sorted_axes[axis_idx] {
                axis_idx += 1;
                StrideSpec::dim_with(0, 1, 0)
            } else {
                let spec = StrideSpec::dim(old_idx, shape[old_idx]);
                old_idx += 1;
                spec
            }
        });
        self.restride(specs)
    }

    /// Create a sliding window view of the tensor (zero-copy).
    ///
    /// This creates overlapping windows along specified dimensions without copying data.
    ///
    /// # Type Parameters
    /// * `DIFF` - Number of windows to create
    /// * `R2` - Output rank (must be R + DIFF)
    ///
    /// # Arguments
    /// * `windows` - Array of SlidingWindow configurations specifying axis, window size, and step
    pub fn sliding_window_view<const DIFF: usize, const R2: usize>(
        &self,
        windows: [SlidingWindow; DIFF],
    ) -> Tensor<R2, D, MapLayout<&B, R2>>
    where
        ConcreteTensor<D, R>: fusor_cpu::LargerRank<R2, DIFF, D>,
        fusor_core::Tensor<R, D>: fusor_core::LargerRank<DIFF, R2, D>,
    {
        let shape = self.shape();
        let mut sorted_windows = windows;
        sorted_windows.sort_by_key(|w| w.axis);
        let specs: [StrideSpec; R2] = std::array::from_fn(|out_i| {
            if out_i < R {
                if let Some(w) = sorted_windows.iter().find(|w| w.axis == out_i) {
                    let num_positions = (shape[out_i] - w.window_size) / w.step + 1;
                    StrideSpec::dim_with(out_i, num_positions, w.step)
                } else {
                    StrideSpec::dim(out_i, shape[out_i])
                }
            } else {
                let w = &sorted_windows[out_i - R];
                StrideSpec::dim(w.axis, w.window_size)
            }
        });
        self.restride(specs)
    }
}

impl<const R: usize, D, B> Tensor<R, D, B>
where
    D: SimdElement + DataType + Default,
    B: TensorBacking<R, Elem = D>,
{
    /// Stack tensors along a new dimension.
    ///
    /// This is an associated function version of the free `stack` function,
    /// matching fusor-core's API.
    ///
    /// # Arguments
    /// * `tensors` - Iterator of tensors to stack
    /// * `dim` - Where to insert the new stacking dimension
    pub fn stack<const R2: usize>(
        tensors: impl IntoIterator<Item = Self>,
        dim: usize,
    ) -> Tensor<R2, D>
    where
        ConcreteTensor<D, R>: fusor_cpu::NextRank<R2, D>,
        fusor_core::Tensor<R, D>: fusor_core::NextRank<R2, D>,
    {
        stack(tensors, dim)
    }

    /// Concatenate tensors along a given dimension.
    ///
    /// This is an associated function version of the free `cat` function,
    /// matching fusor-core's API.
    ///
    /// # Arguments
    /// * `tensors` - Iterator of tensors to concatenate
    /// * `dim` - The dimension to concatenate along
    pub fn cat(tensors: impl IntoIterator<Item = Self>, dim: usize) -> Tensor<R, D> {
        cat(tensors, dim)
    }
}

// Transpose for ND tensors (convenience method)
impl<const R: usize, D, B> Tensor<R, D, B>
where
    D: SimdElement + DataType + Default,
    B: TensorBacking<R, Elem = D>,
{
    /// Transpose a ND tensor (swap the last two dimensions).
    pub fn t(&self) -> Tensor<R, D, MapLayout<&B, R>> {
        self.transpose(R - 2, R - 1)
    }
}

/// Calculate the broadcasted shape for two tensors.
/// Returns the output shape where each dimension is the max of the corresponding input dimensions.
/// Dimensions are aligned from the right.
pub(crate) fn broadcast_shapes<const R1: usize, const R2: usize, const R3: usize>(
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

/// Concatenate multiple tensors along a given dimension.
///
/// # Arguments
/// * `tensors` - Iterator of tensors to concatenate
/// * `dim` - The dimension to concatenate along
pub fn cat<const R: usize, D, B>(
    tensors: impl IntoIterator<Item = Tensor<R, D, B>>,
    dim: usize,
) -> Tensor<R, D>
where
    D: SimdElement + DataType + Default,
    B: TensorBacking<R, Elem = D>,
{
    let tensors: Vec<Tensor<R, D>> = tensors.into_iter().map(|t| t.to_concrete()).collect();
    assert!(!tensors.is_empty(), "Cannot cat empty list of tensors");

    let first_shape = tensors[0].shape();
    let total_dim_size: usize = tensors.iter().map(|t| t.shape()[dim]).sum();
    let new_shape: [usize; R] = std::array::from_fn(|i| {
        if i == dim {
            total_dim_size
        } else {
            first_shape[i]
        }
    });

    // Create the output tensor with splat, then slice_assign each tensor into it
    let mut result = Tensor::splat(&tensors[0].device(), D::default(), new_shape);
    let mut offset = 0;
    for tensor in &tensors {
        let len = tensor.shape()[dim];
        let slice: [std::ops::Range<usize>; R] = std::array::from_fn(|i| {
            if i == dim {
                offset..(offset + len)
            } else {
                0..new_shape[i]
            }
        });
        result = result.slice_assign(slice, tensor);
        offset += len;
    }
    result
}

/// Stack tensors along a new dimension.
///
/// # Arguments
/// * `tensors` - Iterator of tensors to stack
/// * `dim` - Where to insert the new stacking dimension
pub fn stack<const R: usize, const R2: usize, D, B>(
    tensors: impl IntoIterator<Item = Tensor<R, D, B>>,
    dim: usize,
) -> Tensor<R2, D, ConcreteTensor<D, R2>>
where
    D: SimdElement + DataType + Default,
    ConcreteTensor<D, R>: fusor_cpu::NextRank<R2, D>,
    fusor_core::Tensor<R, D>: fusor_core::NextRank<R2, D>,
    B: TensorBacking<R, Elem = D>,
{
    // Unsqueeze each tensor at the target dim, then cat along that dim
    let unsqueezed: Vec<Tensor<R2, D>> = tensors
        .into_iter()
        .map(|t| t.to_concrete().unsqueeze::<R2>(dim).to_concrete())
        .collect();
    cat(unsqueezed, dim)
}

impl<D> Tensor<1, D>
where
    D: SimdElement + DataType + Default,
{
    /// Create a range tensor from start (inclusive) to end (exclusive).
    pub fn arange(device: &Device, start: D, end: D) -> Tensor<1, D, ConcreteTensor<D, 1>>
    where
        D: std::ops::Add<Output = D> + PartialOrd + From<u8>,
    {
        arange(device, start, end)
    }

    /// Create a range tensor with a custom step.
    pub fn arange_step(
        device: &Device,
        start: D,
        end: D,
        step: D,
    ) -> Tensor<1, D, ConcreteTensor<D, 1>>
    where
        D: std::ops::Add<Output = D> + PartialOrd + Copy,
    {
        arange_step(device, start, end, step)
    }
}

/// Create a range tensor from start (inclusive) to end (exclusive).
pub fn arange<D>(device: &Device, start: D, end: D) -> Tensor<1, D, ConcreteTensor<D, 1>>
where
    D: SimdElement + DataType + Default + std::ops::Add<Output = D> + PartialOrd + From<u8>,
{
    arange_step(device, start, end, D::from(1u8))
}

/// Create a range tensor with a custom step.
pub fn arange_step<D>(
    device: &Device,
    start: D,
    end: D,
    step: D,
) -> Tensor<1, D, ConcreteTensor<D, 1>>
where
    D: SimdElement + DataType + Default + std::ops::Add<Output = D> + PartialOrd + Copy,
{
    // Build the data on CPU, then transfer to the right device
    let mut data = Vec::new();
    let mut val = start;
    while val < end {
        data.push(val);
        val += step;
    }
    let len = data.len();
    match device {
        Device::Cpu => Tensor::Cpu(fusor_cpu::Tensor::from_slice([len], &data)),
        Device::Gpu(gpu_device) => {
            let t1d: fusor_core::Tensor<1, D> = fusor_core::Tensor::new(gpu_device, &data);
            Tensor::Gpu(t1d)
        }
    }
}

