use std::marker::PhantomData;

use super::value::boxed_index;
use super::*;
use crate::ir::{AxisGroup, Layout, MultiFlattenMap, Shape, StorageView, SubAxis, U32};

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

/// Convert rank-specific index syntax into storage address components.
pub trait StorageIndex<const R: usize> {
    #[doc(hidden)]
    fn storage_index(self) -> [Box<crate::ir::Expr>; R];
}

impl<I> StorageIndex<1> for I
where
    I: Into<Tile<U32>>,
{
    fn storage_index(self) -> [Box<crate::ir::Expr>; 1] {
        [boxed_index(self)]
    }
}

impl<I> StorageIndex<1> for (I,)
where
    I: Into<Tile<U32>>,
{
    fn storage_index(self) -> [Box<crate::ir::Expr>; 1] {
        [boxed_index(self.0)]
    }
}

impl<I, const R: usize> StorageIndex<R> for [I; R]
where
    I: Into<Tile<U32>>,
{
    fn storage_index(self) -> [Box<crate::ir::Expr>; R] {
        self.map(boxed_index)
    }
}

macro_rules! impl_tuple_storage_index {
    ($rank:literal, $($name:ident),+ $(,)?) => {
        impl<$($name),+> StorageIndex<$rank> for ($($name,)+)
        where
            $($name: Into<Tile<U32>>,)+
        {
            #[allow(non_snake_case)]
            fn storage_index(self) -> [Box<crate::ir::Expr>; $rank] {
                let ($($name,)+) = self;
                [$(boxed_index($name),)+]
            }
        }
    };
}

impl_tuple_storage_index!(2, A, B);
impl_tuple_storage_index!(3, A, B, C);
impl_tuple_storage_index!(4, A, B, C, D);
impl_tuple_storage_index!(5, A, B, C, D, E);
impl_tuple_storage_index!(6, A, B, C, D, E, F);

impl<T, const R: usize> Storage<T, R> {
    /// Underlying storage view.
    pub fn view(&self) -> &StorageView {
        &self.view
    }

    /// Address one element in this storage view.
    pub fn at(&self, index: impl StorageIndex<R>) -> Address<T, R> {
        Address {
            view: self.view.clone(),
            indices: index.storage_index(),
            _ty: PhantomData,
        }
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
