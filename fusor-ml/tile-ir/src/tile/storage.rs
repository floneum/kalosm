use super::value::boxed_index;
use super::Tile;
use crate::ir::{Addr, ElementType, Layout, Shape, StorageView};

/// Runtime-typed handle to a storage buffer view declared on a
/// [`Program`](super::Program).
///
/// The element type and logical rank are runtime data carried by the view. Use
/// [`Storage::view`] to inspect the underlying [`StorageView`].
#[derive(Clone)]
pub struct Storage {
    pub(crate) view: StorageView,
}

/// Convert rank-specific index syntax into a memory [`Addr`].
///
/// Only rank-1 (`Addr::Linear`) and rank-2 (`Addr::Rc2`) addresses are
/// representable in the IR; higher-rank views are flattened at the frontend.
pub trait StorageIndex {
    /// Lower this index syntax into a memory address.
    fn storage_addr(self) -> Addr;
}

impl<I> StorageIndex for I
where
    I: Into<Tile>,
{
    fn storage_addr(self) -> Addr {
        Addr::Linear(boxed_index(self))
    }
}

impl<I> StorageIndex for (I,)
where
    I: Into<Tile>,
{
    fn storage_addr(self) -> Addr {
        Addr::Linear(boxed_index(self.0))
    }
}

impl<R, C> StorageIndex for (R, C)
where
    R: Into<Tile>,
    C: Into<Tile>,
{
    fn storage_addr(self) -> Addr {
        Addr::Rc2 {
            row: boxed_index(self.0),
            col: boxed_index(self.1),
        }
    }
}

impl Storage {
    /// The runtime element type of this storage view.
    pub fn element(&self) -> ElementType {
        self.view.buffer.element
    }

    /// The logical rank of this storage view.
    pub fn rank(&self) -> usize {
        self.view.layout.shape().rank()
    }

    /// Underlying storage view.
    pub fn view(&self) -> &StorageView {
        &self.view
    }

    /// The view's logical layout.
    pub fn layout(&self) -> &Layout {
        &self.view.layout
    }

    /// Address one element in this storage view (rank-1 linear or rank-2
    /// row/col).
    pub fn at(&self, index: impl StorageIndex) -> super::value::Address {
        super::value::Address {
            view: self.view.clone(),
            addr: index.storage_addr(),
        }
    }

    /// Construct a storage handle from an existing view.
    pub fn from_view(view: StorageView) -> Self {
        Self { view }
    }

    /// Re-view this storage with arbitrary `(extent, stride)` per axis. Strides
    /// may overlap (non-injective views); the resulting view is affine.
    pub fn restride<const R2: usize>(&self, extents: [u32; R2], strides: [u32; R2]) -> Storage {
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
                buffer: self.view.buffer.clone(),
                offset: self.view.offset,
                layout,
            },
        }
    }
}
