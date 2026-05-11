use std::num::NonZeroU32;

/// One sub-axis of a logical axis: extent and stride into the underlying
/// buffer. Strides may be zero (broadcast) or may collide with other
/// sub-axes (non-injective views, e.g. im2col).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SubAxis {
    pub extent: u32,
    pub stride: u32,
}

/// A logical axis that decomposes into one or more physical sub-axes via
/// divmod. Sub-axes are listed most-significant first: the logical coord
/// is divided by the product of the trailing sub-axis extents to recover
/// the head sub-coord, then the remainder is divmod-walked through the
/// rest. A group with a single sub-axis is the affine case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AxisGroup {
    pub sub_axes: Vec<SubAxis>,
}

impl AxisGroup {
    pub fn affine(extent: u32, stride: u32) -> Self {
        Self {
            sub_axes: vec![SubAxis { extent, stride }],
        }
    }
}

/// Logical-to-storage mapping. One [`AxisGroup`] per logical axis; affine
/// views are the all-single-sub-axis case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiFlattenMap {
    pub groups: Vec<AxisGroup>,
}

impl MultiFlattenMap {
    /// Build an affine map from per-axis strides.
    pub fn affine(shape: &Shape, strides: &[u32]) -> Self {
        assert_eq!(
            shape.rank(),
            strides.len(),
            "shape rank and strides rank must match",
        );
        let groups = shape
            .dims()
            .iter()
            .zip(strides.iter())
            .map(|(dim, stride)| AxisGroup::affine(dim.get(), *stride))
            .collect();
        Self { groups }
    }

    /// Build a row-major contiguous affine map for `shape`.
    pub fn row_major_for(shape: &Shape) -> Self {
        Self::affine(shape, &row_major_strides(shape))
    }

    /// Build a column-major contiguous affine map for `shape`.
    pub fn col_major_for(shape: &Shape) -> Self {
        Self::affine(shape, &col_major_strides(shape))
    }

    pub fn rank(&self) -> usize {
        self.groups.len()
    }

    /// True iff every logical axis decomposes into exactly one sub-axis.
    pub fn is_affine(&self) -> bool {
        self.groups.iter().all(|g| g.sub_axes.len() == 1)
    }

    /// Owned per-axis strides for an affine map. Panics if `!is_affine()`.
    pub fn affine_strides(&self) -> Vec<u32> {
        self.groups
            .iter()
            .map(|g| {
                assert_eq!(
                    g.sub_axes.len(),
                    1,
                    "affine_strides called on non-affine map",
                );
                g.sub_axes[0].stride
            })
            .collect()
    }

    pub fn is_row_major(&self, shape: &Shape) -> bool {
        self.is_affine() && self.affine_strides() == row_major_strides(shape)
    }

    pub fn is_col_major(&self, shape: &Shape) -> bool {
        self.is_affine() && self.affine_strides() == col_major_strides(shape)
    }

    pub fn is_contiguous(&self, shape: &Shape) -> bool {
        self.is_row_major(shape) || self.is_col_major(shape)
    }
}

fn row_major_strides(shape: &Shape) -> Vec<u32> {
    let mut values = vec![1u32; shape.rank()];
    let dims = shape.dims();
    for axis in (0..shape.rank().saturating_sub(1)).rev() {
        values[axis] = values[axis + 1] * dims[axis + 1].get();
    }
    values
}

fn col_major_strides(shape: &Shape) -> Vec<u32> {
    let mut values = vec![1u32; shape.rank()];
    let dims = shape.dims();
    for axis in 1..shape.rank() {
        values[axis] = values[axis - 1] * dims[axis - 1].get();
    }
    values
}

/// A concrete layout for a tile-like value. Holds shape, memory level, and
/// the logical-to-storage index map.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    indexing: MultiFlattenMap,
    memory_level: MemoryLevel,
}

impl Layout {
    /// Construct a row-major contiguous layout in a memory level.
    pub fn contiguous(memory_level: MemoryLevel, shape: Shape) -> Self {
        let indexing = MultiFlattenMap::row_major_for(&shape);
        Self {
            shape,
            indexing,
            memory_level,
        }
    }

    /// Construct an explicit affine strided layout in a memory level.
    pub fn strided(memory_level: MemoryLevel, shape: Shape, strides: &[u32]) -> Self {
        let indexing = MultiFlattenMap::affine(&shape, strides);
        Self {
            shape,
            indexing,
            memory_level,
        }
    }

    /// Construct a layout with an explicit (possibly non-affine) indexing.
    pub fn with_indexing(memory_level: MemoryLevel, shape: Shape, indexing: MultiFlattenMap) -> Self {
        assert_eq!(
            shape.rank(),
            indexing.rank(),
            "layout shape and indexing must have the same rank",
        );
        Self {
            shape,
            indexing,
            memory_level,
        }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn indexing(&self) -> &MultiFlattenMap {
        &self.indexing
    }

    pub const fn memory_level(&self) -> MemoryLevel {
        self.memory_level
    }

    /// Total number of logical elements addressed by this layout.
    pub fn element_count(&self) -> NonZeroU32 {
        self.shape.element_count()
    }

    /// Number of elements required to back this layout, including padding
    /// implied by non-contiguous strides. Computed over every sub-axis.
    pub fn allocation_element_count(&self) -> NonZeroU32 {
        let last_index = self
            .indexing
            .groups
            .iter()
            .flat_map(|g| g.sub_axes.iter())
            .try_fold(0u32, |acc, sub| {
                let extent = sub.extent.checked_sub(1)?;
                acc.checked_add(extent.checked_mul(sub.stride)?)
            })
            .and_then(|index| index.checked_add(1))
            .expect("layout allocation span overflow");
        NonZeroU32::new(last_index).expect("layout rank is non-zero")
    }

    /// True iff the indexing is affine (each axis = one sub-axis).
    pub fn is_affine(&self) -> bool {
        self.indexing.is_affine()
    }

    /// Owned per-axis strides. Panics if the layout is not affine.
    pub fn affine_strides(&self) -> Vec<u32> {
        self.indexing.affine_strides()
    }

    pub fn is_row_major(&self) -> bool {
        self.indexing.is_row_major(&self.shape)
    }

    pub fn is_col_major(&self) -> bool {
        self.indexing.is_col_major(&self.shape)
    }

    pub fn is_contiguous(&self) -> bool {
        self.indexing.is_contiguous(&self.shape)
    }
}

impl Default for Layout {
    fn default() -> Self {
        Self::contiguous(MemoryLevel::Workgroup, Shape::tile())
    }
}

/// The logical shape of a tile-level operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<NonZeroU32>,
}

impl Shape {
    /// Construct a tile shape from positive dimension sizes.
    pub fn new<const R: usize>(dims: [u32; R]) -> Self {
        assert!(R > 0, "tile shape must have at least one dimension");
        Self {
            dims: dims
                .into_iter()
                .map(|dim| NonZeroU32::new(dim).expect("tile dimensions must be non-zero"))
                .collect(),
        }
    }

    /// Construct the default one-dimensional tile shape.
    pub fn tile() -> Self {
        Self::new([32])
    }

    /// Rank of the logical shape.
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Dimension sizes.
    pub fn dims(&self) -> &[NonZeroU32] {
        &self.dims
    }

    /// Number of logical elements in the tile.
    pub fn element_count(&self) -> NonZeroU32 {
        let elements = self
            .dims
            .iter()
            .fold(1u32, |acc, dim| acc.checked_mul(dim.get()).unwrap());
        NonZeroU32::new(elements).expect("shape rank is non-zero")
    }
}

/// Where a layout lives in the GPU memory hierarchy.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum MemoryLevel {
    Storage,
    Uniform,
    Workgroup,
    Private,
}
