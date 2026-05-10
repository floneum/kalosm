use std::num::NonZeroU32;

/// A concrete layout for a tile-like value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    strides: Strides,
    memory_level: MemoryLevel,
}

impl Layout {
    /// Construct a row-major contiguous layout in a memory level.
    pub fn contiguous(memory_level: MemoryLevel, shape: Shape) -> Self {
        let strides = Strides::row_major_for(&shape);
        Self {
            shape,
            strides,
            memory_level,
        }
    }

    /// Construct an explicit strided layout in a memory level.
    pub fn strided(memory_level: MemoryLevel, shape: Shape, strides: Strides) -> Self {
        assert_eq!(
            shape.rank(),
            strides.rank(),
            "layout shape and strides must have the same rank"
        );
        Self {
            shape,
            strides,
            memory_level,
        }
    }

    /// Logical shape of the tile.
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Logical strides of the tile.
    pub fn strides(&self) -> &Strides {
        &self.strides
    }

    /// The memory level where this tile is represented.
    pub const fn memory_level(&self) -> MemoryLevel {
        self.memory_level
    }

    /// Total number of logical elements addressed by this layout.
    pub fn element_count(&self) -> NonZeroU32 {
        self.shape.element_count()
    }

    /// Number of elements required to back this layout, including padding
    /// implied by non-contiguous strides.
    pub fn allocation_element_count(&self) -> NonZeroU32 {
        let last_index = self
            .shape
            .dims()
            .iter()
            .zip(self.strides.values())
            .try_fold(0u32, |acc, (dim, stride)| {
                let extent = dim.get().checked_sub(1)?;
                acc.checked_add(extent.checked_mul(*stride)?)
            })
            .and_then(|index| index.checked_add(1))
            .expect("layout allocation span overflow");
        NonZeroU32::new(last_index).expect("layout rank is non-zero")
    }

    /// True when the strides match row-major contiguous order.
    pub fn is_row_major(&self) -> bool {
        self.strides == Strides::row_major_for(&self.shape)
    }

    /// True when the strides match column-major contiguous order.
    pub fn is_col_major(&self) -> bool {
        self.strides == Strides::col_major_for(&self.shape)
    }

    /// True when the strides are a standard contiguous row- or column-major layout.
    pub fn is_contiguous(&self) -> bool {
        self.is_row_major() || self.is_col_major()
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

/// Logical strides for a tile layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Strides {
    values: Vec<u32>,
}

impl Strides {
    /// Construct explicit strides.
    pub fn new<const R: usize>(values: [u32; R]) -> Self {
        assert!(R > 0, "strides must have at least one dimension");
        Self {
            values: values.into_iter().collect(),
        }
    }

    /// Construct row-major contiguous strides for a shape.
    pub fn row_major_for(shape: &Shape) -> Self {
        let mut values = vec![1; shape.rank()];
        let dims = shape.dims();
        for axis in (0..shape.rank() - 1).rev() {
            values[axis] = values[axis + 1] * dims[axis + 1].get();
        }
        Self { values }
    }

    /// Construct column-major contiguous strides for a shape.
    pub fn col_major_for(shape: &Shape) -> Self {
        let mut values = vec![1; shape.rank()];
        let dims = shape.dims();
        for axis in 1..shape.rank() {
            values[axis] = values[axis - 1] * dims[axis - 1].get();
        }
        Self { values }
    }

    /// Rank of the stride vector.
    pub fn rank(&self) -> usize {
        self.values.len()
    }

    /// Stride values.
    pub fn values(&self) -> &[u32] {
        &self.values
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

/// The execution hierarchy level that owns a tile.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TileLevel {
    Workgroup,
}

/// Whether a tile declaration owns storage.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TileOrigin {
    Allocation,
}
