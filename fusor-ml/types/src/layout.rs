use std::ops::Range;

pub const TILE_SIZE: u32 = 1;

pub fn slice_strides(
    slices: &[Range<usize>],
    offset: usize,
    strides: &[usize],
) -> (usize, Box<[usize]>) {
    let start_offset = slices
        .iter()
        .zip(strides.iter())
        .map(|(range, stride)| *stride * range.start)
        .sum::<usize>();

    (offset + start_offset, strides.into())
}

pub fn slice_shape(slices: &[Range<usize>], _shape: &[usize]) -> Box<[usize]> {
    slices.iter().map(|range| range.len()).collect()
}

/// Configuration for sliding window operations.
///
/// Specifies how to create overlapping windows along a tensor dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlidingWindow {
    /// The axis (dimension) to create windows along
    pub axis: usize,
    /// The size of each window
    pub window_size: usize,
    /// The step size between window starts
    pub step: usize,
}

impl SlidingWindow {
    /// Create a new SlidingWindow configuration
    pub fn new(axis: usize, window_size: usize, step: usize) -> Self {
        Self {
            axis,
            window_size,
            step,
        }
    }
}

impl From<(usize, usize)> for SlidingWindow {
    fn from(val: (usize, usize)) -> Self {
        SlidingWindow {
            axis: val.0,
            window_size: val.1,
            step: 1,
        }
    }
}

impl From<[usize; 2]> for SlidingWindow {
    fn from(val: [usize; 2]) -> Self {
        SlidingWindow {
            axis: val[0],
            window_size: val[1],
            step: 1,
        }
    }
}

impl From<(usize, usize, usize)> for SlidingWindow {
    fn from(val: (usize, usize, usize)) -> Self {
        SlidingWindow {
            axis: val.0,
            window_size: val.1,
            step: val.2,
        }
    }
}

impl From<[usize; 3]> for SlidingWindow {
    fn from(val: [usize; 3]) -> Self {
        SlidingWindow {
            axis: val[0],
            window_size: val[1],
            step: val[2],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    offset: usize,
    shape: Box<[usize]>,
    strides: Box<[usize]>,
    contiguous: bool,
}

fn check_contiguous(offset: usize, shape: &[usize], strides: &[usize]) -> bool {
    if offset != 0 {
        return false;
    }
    let mut expected_stride = 1;
    for i in (0..shape.len()).rev() {
        if strides[i] != expected_stride {
            return false;
        }
        expected_stride *= shape[i];
    }
    true
}

impl Layout {
    pub fn contiguous(shape: &[usize]) -> Self {
        let strides = Self::continuous_strides(shape);
        Self {
            offset: 0,
            shape: shape.into(),
            strides,
            contiguous: true,
        }
    }

    pub fn from_parts(offset: usize, shape: Box<[usize]>, strides: Box<[usize]>) -> Self {
        let contiguous = check_contiguous(offset, &shape, &strides);
        Self {
            offset,
            shape,
            strides,
            contiguous,
        }
    }

    #[inline(always)]
    pub fn is_contiguous(&self) -> bool {
        self.contiguous
    }

    /// Returns true if the innermost dimension has stride 1
    #[inline(always)]
    pub fn inner_dim_contiguous(&self) -> bool {
        !self.strides.is_empty() && self.strides[self.strides.len() - 1] == 1
    }

    /// Returns the size of the innermost dimension
    #[inline(always)]
    pub fn inner_dim_size(&self) -> usize {
        if self.shape.is_empty() {
            0
        } else {
            self.shape[self.shape.len() - 1]
        }
    }

    pub fn slice(&self, slices: &[Range<usize>]) -> Self {
        let (offset, strides) = slice_strides(slices, self.offset, &self.strides);
        let shape = slice_shape(slices, &self.strides);
        let contiguous = check_contiguous(offset, &shape, &strides);

        Self {
            offset,
            shape,
            strides,
            contiguous,
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Check if any items in this tensor point to the same allocation. This will be true if
    /// the tensor has a stride of 0.
    pub fn allocation_overlaps(&self) -> bool {
        self.strides.contains(&0)
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Calculate the linear index for a given set of logical indices
    #[inline(always)]
    pub fn linear_index(&self, indices: &[usize]) -> usize {
        self.offset
            + indices
                .iter()
                .zip(self.strides.iter())
                .map(|(idx, stride)| idx * stride)
                .sum::<usize>()
    }

    /// Calculate the linear index for a given set of logical indices
    ///
    /// # SAFETY
    /// the length of indices must be exactly equal to the length of the strides in the layout
    #[inline(always)]
    pub unsafe fn linear_index_unchecked<const R: usize>(&self, indices: &[usize; R]) -> usize {
        let mut linear_idx = self.offset;
        for i in 0..R {
            linear_idx += unsafe { indices.get_unchecked(i) * self.strides.get_unchecked(i) };
        }
        linear_idx
    }

    /// Get the total number of elements in the tensor
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn continuous_strides(shape: &[usize]) -> Box<[usize]> {
        let mut acc = 1;
        let mut strides = vec![0; shape.len()].into_boxed_slice();
        for i in (0..shape.len()).rev() {
            strides[i] = acc;
            acc *= shape[i];
        }
        strides
    }

    /// Permute the layout dimensions according to the given axes order.
    ///
    /// # Arguments
    /// * `axes` - A permutation specifying the new order of dimensions
    ///
    /// # Panics
    /// Panics if axes don't form a valid permutation
    pub fn permute(&self, axes: &[usize]) -> Self {
        let rank = self.shape.len();
        assert_eq!(axes.len(), rank, "Axes length must match rank");

        // Validate axes are a valid permutation
        let mut seen = vec![false; rank];
        for &axis in axes {
            assert!(axis < rank, "Axis {} out of range for rank {}", axis, rank);
            assert!(!seen[axis], "Duplicate axis {} in permutation", axis);
            seen[axis] = true;
        }

        let mut new_shape = vec![0usize; rank];
        let mut new_strides = vec![0usize; rank];
        for (new_idx, &old_idx) in axes.iter().enumerate() {
            new_shape[new_idx] = self.shape[old_idx];
            new_strides[new_idx] = self.strides[old_idx];
        }

        let contiguous = check_contiguous(self.offset, &new_shape, &new_strides);
        Self {
            offset: self.offset,
            shape: new_shape.into_boxed_slice(),
            strides: new_strides.into_boxed_slice(),
            contiguous,
        }
    }

    /// Transpose two dimensions of the layout.
    ///
    /// # Arguments
    /// * `dim0` - First dimension to swap
    /// * `dim1` - Second dimension to swap
    ///
    /// # Panics
    /// Panics if either dimension is out of range
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        let rank = self.shape.len();
        assert!(dim0 < rank, "dim0 {} out of range for rank {}", dim0, rank);
        assert!(dim1 < rank, "dim1 {} out of range for rank {}", dim1, rank);

        let mut new_shape: Vec<usize> = self.shape.to_vec();
        let mut new_strides: Vec<usize> = self.strides.to_vec();

        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);

        let contiguous = check_contiguous(self.offset, &new_shape, &new_strides);
        Self {
            offset: self.offset,
            shape: new_shape.into_boxed_slice(),
            strides: new_strides.into_boxed_slice(),
            contiguous,
        }
    }

    /// Create a broadcast layout to match the target shape.
    ///
    /// Broadcasting rules (flexible, iterating from right to left):
    /// - Source dimensions are consumed only when they match the target or are size 1
    /// - Matching dimensions keep their stride
    /// - Size-1 dimensions get stride 0 (broadcast)
    /// - Unmatched target dimensions get stride 0 (new dimensions)
    /// - New dimensions can be added anywhere, not just on the left
    ///
    /// # Panics
    /// Panics if not all source dimensions can be matched to target dimensions
    pub fn broadcast_to(&self, target_shape: &[usize]) -> Self {
        let new_rank = target_shape.len();
        assert!(
            new_rank >= self.shape.len(),
            "Cannot broadcast to smaller rank"
        );

        let mut new_strides = vec![0usize; new_rank];
        let mut shape_iter = self.shape.iter().rev().peekable();
        let mut strides_iter = self.strides.iter().rev();

        for (new_idx, &target_dim) in target_shape.iter().enumerate().rev() {
            let stride = if let Some(&src_dim) = shape_iter
                .next_if(|&&src_dim| src_dim == target_dim || (src_dim == 1 && target_dim > 1))
            {
                let stride = *strides_iter.next().unwrap();
                // Matching dim, use the same stride
                if src_dim == target_dim {
                    stride
                }
                // Broadcasted dim, set stride to 0
                else {
                    0
                }
            } else {
                // New dimension or non-matching - stride 0
                0
            };
            new_strides[new_idx] = stride;
        }

        assert_eq!(
            shape_iter.len(),
            0,
            "Failed to broadcast: not all source dimensions were matched. \
            Source shape {:?} is not compatible with target shape {:?}",
            self.shape,
            target_shape
        );

        let contiguous = check_contiguous(self.offset, target_shape, &new_strides);
        Self {
            offset: self.offset,
            shape: target_shape.into(),
            strides: new_strides.into_boxed_slice(),
            contiguous,
        }
    }

    /// Squeeze (remove) a dimension of size 1.
    ///
    /// # Arguments
    /// * `dim` - The dimension to squeeze (must have size 1)
    ///
    /// # Panics
    /// Panics if the dimension doesn't have size 1 or is out of range
    pub fn squeeze(&self, dim: usize) -> Self {
        let rank = self.shape.len();
        assert!(rank > 0, "Cannot squeeze a scalar layout");
        assert!(
            dim < rank,
            "Dimension {} out of range for rank {}",
            dim,
            rank
        );
        assert!(
            self.shape[dim] == 1,
            "Cannot squeeze dimension {} of size {} (must be 1)",
            dim,
            self.shape[dim]
        );

        let mut new_shape = Vec::with_capacity(rank - 1);
        let mut new_strides = Vec::with_capacity(rank - 1);

        for i in 0..rank {
            if i != dim {
                new_shape.push(self.shape[i]);
                new_strides.push(self.strides[i]);
            }
        }

        let contiguous = check_contiguous(self.offset, &new_shape, &new_strides);
        Self {
            offset: self.offset,
            shape: new_shape.into_boxed_slice(),
            strides: new_strides.into_boxed_slice(),
            contiguous,
        }
    }

    /// Unsqueeze (add) a dimension of size 1 at the specified position.
    ///
    /// # Arguments
    /// * `dim` - Where to insert the new dimension
    ///
    /// # Panics
    /// Panics if dim is out of range
    pub fn unsqueeze(&self, dim: usize) -> Self {
        let rank = self.shape.len();
        assert!(
            dim <= rank,
            "Dimension {} out of range for inserting into rank {}",
            dim,
            rank
        );

        let new_rank = rank + 1;
        let mut new_shape = Vec::with_capacity(new_rank);
        let mut new_strides = Vec::with_capacity(new_rank);

        for i in 0..new_rank {
            if i == dim {
                new_shape.push(1);
                // Stride for size-1 dim doesn't matter, but using 1 is conventional
                new_strides.push(1);
            } else {
                let old_idx = if i < dim { i } else { i - 1 };
                new_shape.push(self.shape[old_idx]);
                new_strides.push(self.strides[old_idx]);
            }
        }

        let contiguous = check_contiguous(self.offset, &new_shape, &new_strides);
        Self {
            offset: self.offset,
            shape: new_shape.into_boxed_slice(),
            strides: new_strides.into_boxed_slice(),
            contiguous,
        }
    }

    /// Reshape the layout to a new shape.
    ///
    /// This only works on contiguous layouts. For non-contiguous layouts,
    /// the caller must first make the data contiguous.
    ///
    /// # Arguments
    /// * `new_shape` - The target shape (must have same total element count)
    ///
    /// # Panics
    /// Panics if the layout is not contiguous or element count doesn't match
    pub fn reshape(&self, new_shape: &[usize]) -> Self {
        let old_elements: usize = self.shape.iter().product();
        let new_elements: usize = new_shape.iter().product();
        assert_eq!(
            old_elements, new_elements,
            "Cannot reshape: element count mismatch ({} vs {})",
            old_elements, new_elements
        );
        assert!(
            self.is_contiguous(),
            "Cannot reshape non-contiguous layout; make it contiguous first"
        );

        Self::contiguous(new_shape)
    }

    /// Narrow the layout along a given dimension.
    ///
    /// # Arguments
    /// * `dim` - The dimension to narrow
    /// * `start` - The starting index
    /// * `length` - The length of the slice
    ///
    /// # Panics
    /// Panics if the slice is out of bounds
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Self {
        let rank = self.shape.len();
        assert!(
            dim < rank,
            "Dimension {} out of range for rank {}",
            dim,
            rank
        );
        assert!(
            start + length <= self.shape[dim],
            "Narrow out of bounds: {}..{} for dimension of size {}",
            start,
            start + length,
            self.shape[dim]
        );

        // Build ranges for all dimensions
        let slices: Vec<Range<usize>> = (0..rank)
            .map(|i| {
                if i == dim {
                    start..start + length
                } else {
                    0..self.shape[i]
                }
            })
            .collect();

        self.slice(&slices)
    }

    /// Flatten the last `n` dimensions into one.
    ///
    /// The new shape will be `[..original[:-n], product(original[-n:])]`.
    ///
    /// # Arguments
    /// * `n` - Number of dimensions from the end to flatten (must be >= 1)
    ///
    /// # Panics
    /// * If `n` is 0 or greater than the rank
    /// * If the layout is not contiguous (use make_contiguous first)
    pub fn flatten_last_n(&self, n: usize) -> Self {
        let rank = self.shape.len();
        assert!(n >= 1, "n must be at least 1");
        assert!(
            n <= rank,
            "Cannot flatten {} dimensions from rank {}",
            n,
            rank
        );
        assert!(
            self.is_contiguous(),
            "Cannot flatten non-contiguous layout; make it contiguous first"
        );

        let new_rank = rank - n + 1;
        let mut new_shape = Vec::with_capacity(new_rank);

        // Copy dimensions before the flattened region
        for i in 0..(rank - n) {
            new_shape.push(self.shape[i]);
        }

        // Flatten the last n dimensions into one
        let flattened_size: usize = self.shape[(rank - n)..].iter().product();
        new_shape.push(flattened_size);

        Self::contiguous(&new_shape)
    }

    /// Flatten the first `n+1` dimensions into one.
    ///
    /// The new shape will be `[product(original[:n+1]), ..original[n+1:]]`.
    ///
    /// # Arguments
    /// * `n` - Number of dimensions from the start to include in flattening (0-indexed, so n=0 means flatten first dimension with second)
    ///
    /// # Panics
    /// * If `n+1` is greater than the rank
    /// * If the layout is not contiguous
    pub fn flatten_first_n(&self, n: usize) -> Self {
        let rank = self.shape.len();
        assert!(
            n < rank,
            "Cannot flatten first {} dimensions from rank {}",
            n + 1,
            rank
        );
        assert!(
            self.is_contiguous(),
            "Cannot flatten non-contiguous layout; make it contiguous first"
        );

        let new_rank = rank - n;
        let mut new_shape = Vec::with_capacity(new_rank);

        // Flatten the first n+1 dimensions into one
        let flattened_size: usize = self.shape[..=n].iter().product();
        new_shape.push(flattened_size);

        // Copy remaining dimensions
        for i in (n + 1)..rank {
            new_shape.push(self.shape[i]);
        }

        Self::contiguous(&new_shape)
    }

    /// Squeeze (remove) multiple dimensions of size 1 at once.
    ///
    /// # Arguments
    /// * `axes` - The dimensions to squeeze (must all have size 1)
    ///
    /// # Panics
    /// * If any dimension is out of range
    /// * If any dimension doesn't have size 1
    pub fn squeeze_dims(&self, axes: &[usize]) -> Self {
        let rank = self.shape.len();

        // Validate all axes
        for &axis in axes {
            assert!(axis < rank, "Axis {} out of range for rank {}", axis, rank);
            assert!(
                self.shape[axis] == 1,
                "Cannot squeeze dimension {} of size {} (must be 1)",
                axis,
                self.shape[axis]
            );
        }

        // Sort axes for efficient removal
        let mut sorted_axes: Vec<usize> = axes.to_vec();
        sorted_axes.sort_unstable();

        let new_rank = rank - sorted_axes.len();
        let mut new_shape = Vec::with_capacity(new_rank);
        let mut new_strides = Vec::with_capacity(new_rank);

        let mut axis_iter = sorted_axes.iter().peekable();
        for i in 0..rank {
            if axis_iter.peek() == Some(&&i) {
                axis_iter.next();
            } else {
                new_shape.push(self.shape[i]);
                new_strides.push(self.strides[i]);
            }
        }

        let contiguous = check_contiguous(self.offset, &new_shape, &new_strides);
        Self {
            offset: self.offset,
            shape: new_shape.into_boxed_slice(),
            strides: new_strides.into_boxed_slice(),
            contiguous,
        }
    }

    /// Unsqueeze (add) multiple dimensions of size 1 at specified positions.
    ///
    /// # Arguments
    /// * `axes` - Where to insert the new dimensions (positions in the output tensor)
    ///
    /// # Panics
    /// * If any axis is out of range for the new rank
    pub fn unsqueeze_dims(&self, axes: &[usize]) -> Self {
        let rank = self.shape.len();
        let new_rank = rank + axes.len();

        // Validate all axes
        for &axis in axes {
            assert!(
                axis < new_rank,
                "Axis {} out of range for new rank {}",
                axis,
                new_rank
            );
        }

        // Sort axes
        let mut sorted_axes: Vec<usize> = axes.to_vec();
        sorted_axes.sort_unstable();

        let mut new_shape = Vec::with_capacity(new_rank);
        let mut new_strides = Vec::with_capacity(new_rank);

        let mut axis_iter = sorted_axes.iter().peekable();
        let mut old_idx = 0;

        for new_idx in 0..new_rank {
            if axis_iter.peek() == Some(&&new_idx) {
                axis_iter.next();
                new_shape.push(1);
                new_strides.push(1); // Stride for size-1 dimension doesn't matter
            } else {
                new_shape.push(self.shape[old_idx]);
                new_strides.push(self.strides[old_idx]);
                old_idx += 1;
            }
        }

        let contiguous = check_contiguous(self.offset, &new_shape, &new_strides);
        Self {
            offset: self.offset,
            shape: new_shape.into_boxed_slice(),
            strides: new_strides.into_boxed_slice(),
            contiguous,
        }
    }

    /// Create a sliding window view of the layout.
    ///
    /// This transforms dimensions by creating overlapping windows. Each windowed dimension
    /// is replaced with `(dim - window_size) / step + 1` positions, and a new dimension
    /// of size `window_size` is appended for each window configuration.
    ///
    /// # Arguments
    /// * `windows` - Slice of SlidingWindow configurations
    ///
    /// # Panics
    /// * If any axis is out of bounds
    /// * If axes are not unique
    pub fn sliding_window(&self, windows: &[SlidingWindow]) -> Self {
        let old_rank = self.shape.len();
        let num_windows = windows.len();
        let new_rank = old_rank + num_windows;

        // Sort windows by axis
        let mut sorted_windows: Vec<SlidingWindow> = windows.to_vec();
        sorted_windows.sort_by_key(|w| w.axis);

        // Validate axes
        for window in &sorted_windows {
            assert!(
                window.axis < old_rank,
                "Sliding window axis {} out of bounds",
                window.axis
            );
        }
        for pair in sorted_windows.windows(2) {
            assert!(
                pair[0].axis != pair[1].axis,
                "Sliding window axes must be unique"
            );
        }

        // Transform shape: original dimensions get their window positions, new dimensions get window sizes
        let new_shape: Vec<usize> = (0..new_rank)
            .map(|i| {
                if i < old_rank {
                    // Original dimension
                    if let Ok(idx) = sorted_windows.binary_search_by_key(&i, |w| w.axis) {
                        // This dimension is being windowed
                        let window = &sorted_windows[idx];
                        (self.shape[i] - window.window_size) / window.step + 1
                    } else {
                        // Not windowed
                        self.shape[i]
                    }
                } else {
                    // New dimensions for windows
                    let index = i - old_rank;
                    sorted_windows[index].window_size
                }
            })
            .collect();

        // Transform strides: original dimensions get step-multiplied strides, new dimensions get original strides
        let new_strides: Vec<usize> = (0..new_rank)
            .map(|i| {
                if i < old_rank {
                    // Original dimension
                    if let Ok(idx) = sorted_windows.binary_search_by_key(&i, |w| w.axis) {
                        // This dimension is being windowed - multiply stride by step
                        let window = &sorted_windows[idx];
                        self.strides[i] * window.step
                    } else {
                        // Not windowed - keep original stride
                        self.strides[i]
                    }
                } else {
                    // New dimensions for windows - use original stride of windowed axis
                    let index = i - old_rank;
                    let axis = sorted_windows[index].axis;
                    self.strides[axis]
                }
            })
            .collect();

        let contiguous = check_contiguous(self.offset, &new_shape, &new_strides);
        Self {
            offset: self.offset,
            shape: new_shape.into_boxed_slice(),
            strides: new_strides.into_boxed_slice(),
            contiguous,
        }
    }
}

#[test]
fn test_contiguous() {
    let layout = Layout::contiguous(&[2, 3]);
    assert!(layout.is_contiguous());
    assert!(!layout.slice(&[0..1, 0..1]).is_contiguous());
    assert!(!layout.slice(&[1..2, 0..3]).is_contiguous());
}

#[test]
fn test_flatten_last_n() {
    // Flatten last 2 dimensions of a 3D tensor
    let layout = Layout::contiguous(&[2, 3, 4]);
    let flattened = layout.flatten_last_n(2);
    assert_eq!(flattened.shape(), &[2, 12]);
    assert!(flattened.is_contiguous());

    // Flatten all dimensions
    let layout = Layout::contiguous(&[2, 3, 4]);
    let flattened = layout.flatten_last_n(3);
    assert_eq!(flattened.shape(), &[24]);
    assert!(flattened.is_contiguous());

    // Flatten just the last dimension (no-op in terms of structure)
    let layout = Layout::contiguous(&[2, 3, 4]);
    let flattened = layout.flatten_last_n(1);
    assert_eq!(flattened.shape(), &[2, 3, 4]);
}

#[test]
fn test_flatten_first_n() {
    // Flatten first 2 dimensions of a 3D tensor
    let layout = Layout::contiguous(&[2, 3, 4]);
    let flattened = layout.flatten_first_n(1);
    assert_eq!(flattened.shape(), &[6, 4]);
    assert!(flattened.is_contiguous());

    // Flatten all dimensions
    let layout = Layout::contiguous(&[2, 3, 4]);
    let flattened = layout.flatten_first_n(2);
    assert_eq!(flattened.shape(), &[24]);
    assert!(flattened.is_contiguous());
}

#[test]
fn test_squeeze_dims() {
    // Squeeze two dimensions at once
    let layout = Layout::contiguous(&[1, 3, 1, 4]);
    let squeezed = layout.squeeze_dims(&[0, 2]);
    assert_eq!(squeezed.shape(), &[3, 4]);

    // Squeeze in different order
    let layout = Layout::contiguous(&[1, 3, 1, 4]);
    let squeezed = layout.squeeze_dims(&[2, 0]);
    assert_eq!(squeezed.shape(), &[3, 4]);

    // Single squeeze
    let layout = Layout::contiguous(&[1, 3, 4]);
    let squeezed = layout.squeeze_dims(&[0]);
    assert_eq!(squeezed.shape(), &[3, 4]);
}

#[test]
fn test_unsqueeze_dims() {
    // Unsqueeze two dimensions at once
    let layout = Layout::contiguous(&[3, 4]);
    let unsqueezed = layout.unsqueeze_dims(&[0, 2]);
    assert_eq!(unsqueezed.shape(), &[1, 3, 1, 4]);

    // Unsqueeze in different order
    let layout = Layout::contiguous(&[3, 4]);
    let unsqueezed = layout.unsqueeze_dims(&[2, 0]);
    assert_eq!(unsqueezed.shape(), &[1, 3, 1, 4]);

    // Single unsqueeze
    let layout = Layout::contiguous(&[3, 4]);
    let unsqueezed = layout.unsqueeze_dims(&[0]);
    assert_eq!(unsqueezed.shape(), &[1, 3, 4]);
}

#[test]
fn test_sliding_window() {
    // 1D sliding window
    let layout = Layout::contiguous(&[7]);
    let windowed = layout.sliding_window(&[SlidingWindow::new(0, 3, 2)]);
    // (7 - 3) / 2 + 1 = 3 positions, window size 3
    assert_eq!(windowed.shape(), &[3, 3]);
    assert_eq!(windowed.strides(), &[2, 1]);

    // 2D sliding window
    let layout = Layout::contiguous(&[6, 6]);
    let windowed =
        layout.sliding_window(&[SlidingWindow::new(0, 3, 3), SlidingWindow::new(1, 3, 3)]);
    // (6 - 3) / 3 + 1 = 2 positions in each dimension
    assert_eq!(windowed.shape(), &[2, 2, 3, 3]);
    assert_eq!(windowed.strides(), &[18, 3, 6, 1]);
}
