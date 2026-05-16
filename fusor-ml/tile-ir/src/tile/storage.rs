use std::marker::PhantomData;

use super::*;
use crate::ir::{AxisGroup, Layout, MultiFlattenMap, Shape, StorageView, SubAxis};

/// Typed handle to a storage buffer view declared on a [`Program`].
///
/// `T` is the element type exposed to the tile API and `R` is the logical
/// rank of the view. Use [`Storage::view`] to inspect the underlying
/// [`StorageView`](crate::StorageView).
pub struct Storage<T, const R: usize> {
    pub(crate) view: StorageView,
    pub(super) _ty: PhantomData<T>,
}

/// Marker for storage whose element type is carried by the [`StorageView`]
/// instead of a compile-time [`Numeric`] marker.
pub struct RuntimeElement;

/// Converts rank-specific index arguments into a typed storage address.
pub trait StorageIndex<T, const R: usize> {
    /// Address type produced by this index.
    type Address;

    /// Build an address against `view`.
    fn address(self, view: StorageView) -> Self::Address;
}

impl<T, I> StorageIndex<T, 1> for I
where
    I: IntoIndex,
{
    type Address = LinearAddress<T>;

    fn address(self, view: StorageView) -> Self::Address {
        LinearAddress {
            view,
            index: self.into_index(),
            _ty: PhantomData,
        }
    }
}

impl<T, Row, Col> StorageIndex<T, 2> for (Row, Col)
where
    Row: IntoIndex,
    Col: IntoIndex,
{
    type Address = Address<T>;

    fn address(self, view: StorageView) -> Self::Address {
        let (row, col) = self;
        Address {
            view,
            row: row.into_index(),
            col: col.into_index(),
            _ty: PhantomData,
        }
    }
}

impl<T, const R: usize> Storage<T, R> {
    /// Underlying storage view.
    pub fn view(&self) -> &StorageView {
        &self.view
    }

    /// Address one element in this storage view.
    pub fn at<I>(&self, index: I) -> I::Address
    where
        I: StorageIndex<T, R>,
    {
        index.address(self.view.clone())
    }

    /// Construct a typed storage handle from an existing view. Caller is
    /// responsible for ensuring the view's element type matches `T` and its
    /// layout's rank matches `R`.
    pub fn from_view(view: StorageView) -> Self {
        Self {
            view,
            _ty: PhantomData,
        }
    }

    /// Re-view this storage as rank `R2` with arbitrary `(extent, stride)`
    /// per axis. Strides may overlap (non-injective views); the resulting
    /// view is affine — no divmod indexing.
    pub fn restride<const R2: usize>(
        &self,
        extents: [u32; R2],
        strides: [u32; R2],
    ) -> Storage<T, R2> {
        assert!(
            self.view.layout.is_affine(),
            "restride source must be an affine view",
        );
        let layout = Layout::strided(
            self.view.layout.memory_level(),
            Shape::new(extents),
            &strides,
        );
        Storage {
            view: StorageView {
                buffer: self.view.buffer,
                offset: self.view.offset,
                layout,
            },
            _ty: PhantomData,
        }
    }

    /// Fuse adjacent axes into groups, lowering rank from `R` to `R2` via
    /// divmod indexing. `groups[i]` lists the source axes (most-significant
    /// first) of output axis `i`.
    pub fn flatten_axes<const R2: usize>(&self, groups: [&[usize]; R2]) -> Storage<T, R2> {
        assert!(
            self.view.layout.is_affine(),
            "flatten_axes source must be an affine view",
        );
        let src_dims = self.view.layout.shape().dims();
        let src_strides = self.view.layout.affine_strides();

        let mut new_extents = [0u32; R2];
        let new_groups: Vec<AxisGroup> = groups
            .iter()
            .enumerate()
            .map(|(out_axis, axes)| {
                assert!(!axes.is_empty(), "axis group must be non-empty");
                let mut extent_product: u32 = 1;
                let sub_axes = axes
                    .iter()
                    .map(|&src_axis| {
                        let extent = src_dims[src_axis].get();
                        extent_product = extent_product
                            .checked_mul(extent)
                            .expect("flatten_axes extent overflow");
                        SubAxis {
                            extent,
                            stride: src_strides[src_axis],
                        }
                    })
                    .collect();
                new_extents[out_axis] = extent_product;
                AxisGroup { sub_axes }
            })
            .collect();

        let indexing = MultiFlattenMap { groups: new_groups };
        let layout = Layout::with_indexing(
            self.view.layout.memory_level(),
            Shape::new(new_extents),
            indexing,
        );
        Storage {
            view: StorageView {
                buffer: self.view.buffer,
                offset: self.view.offset,
                layout,
            },
            _ty: PhantomData,
        }
    }
}
