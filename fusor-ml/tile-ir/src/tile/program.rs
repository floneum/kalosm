use std::marker::PhantomData;

use super::*;
use crate::ir::{
    BlockDequantId, BufferAccess, BufferDecl, BufferRef, CoopFragmentId, KernelIr, Layout,
    LocalRef, MemoryLevel, Numeric, Shape, StorageView, TileDecl, TileProgramOp, TileRef, F32,
};

macro_rules! storage_accessors {
    ($read:ident, $write:ident($($arg:ident: $ty:ty),*) => ($layout:expr, $offset:expr)) => {
        /// Declare a read-only typed storage view.
        pub fn $read<T: Numeric, const R: usize>(&mut self, $($arg: $ty),*) -> Storage<T, R> {
            self.storage_with_layout_and_access($layout, $offset, BufferAccess::Read)
        }

        /// Declare a read-write typed storage view.
        pub fn $write<T: Numeric, const R: usize>(&mut self, $($arg: $ty),*) -> Storage<T, R> {
            self.storage_with_layout_and_access(
                $layout,
                $offset,
                BufferAccess::ReadWrite,
            )
        }
    };
}

macro_rules! element_storage_accessors {
    ($read:ident, $write:ident($($arg:ident: $ty:ty),*) => ($layout:expr, $offset:expr)) => {
        /// Declare a read-only storage view with an element type known at runtime.
        pub fn $read<const R: usize>(
            &mut self,
            element: crate::ElementType,
            $($arg: $ty),*
        ) -> Storage<RuntimeElement, R> {
            self.storage_element_with_layout_and_access(element, $layout, $offset, BufferAccess::Read)
        }

        /// Declare a read-write storage view with an element type known at runtime.
        pub fn $write<const R: usize>(
            &mut self,
            element: crate::ElementType,
            $($arg: $ty),*
        ) -> Storage<RuntimeElement, R> {
            self.storage_element_with_layout_and_access(
                element,
                $layout,
                $offset,
                BufferAccess::ReadWrite,
            )
        }
    };
}

/// Builder for one tile IR kernel.
///
/// A `Program` owns storage declarations, scratch allocations, and the single
/// tile program body. Most callers construct one through [`build`](crate::tile::build).
pub struct Program {
    pub(crate) ir: KernelIr,
    /// Builder-only counter for fresh `BufferId`s. Lives here (not on
    /// `KernelIr`) because the finished IR is immutable data — the counter
    /// is only needed during construction.
    pub(crate) next_buffer: u32,
    /// Builder-only counter for fresh `TileId`s. Same reasoning as
    /// `next_buffer`.
    pub(crate) next_tile: u32,
    /// Builder-only counter for fresh `LocalId`s. Same reasoning as
    /// `next_buffer`.
    pub(crate) next_local: u32,
    /// Builder-only counter for fresh `BlockDequantId`s. Lives here (not on
    /// `KernelIr`) because these ids are SSA-scoped names allocated by the
    /// builder and never observed off the finished IR.
    pub(crate) next_block_dequant: u32,
    /// Builder-only counter for fresh `CoopFragmentId`s. Same reasoning as
    /// `next_block_dequant`.
    pub(crate) next_coop_fragment: u32,
}

impl Program {
    /// Create an empty builder. Most callers should use [`build`] instead;
    /// this is for [`crate::kernel_builder::KernelBuilder`] which owns the
    /// program plus a parallel binding list.
    pub fn new() -> Self {
        Self {
            ir: KernelIr::default(),
            next_buffer: 0,
            next_tile: 0,
            next_local: 0,
            next_block_dequant: 0,
            next_coop_fragment: 0,
        }
    }

    /// Consume the builder and return the constructed [`KernelIr`].
    pub fn into_ir(self) -> KernelIr {
        self.ir
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

impl Program {
    storage_accessors!(
        storage_read,
        storage_write(shape: Shape) => (
            Layout::contiguous(MemoryLevel::Storage, shape),
            0
        )
    );
    storage_accessors!(
        storage_read_with_layout,
        storage_write_with_layout(layout: Layout) => (layout, 0)
    );
    storage_accessors!(
        storage_read_with_layout_offset,
        storage_write_with_layout_offset(layout: Layout, offset: u32) => (layout, offset)
    );

    fn storage_with_layout_and_access<T: Numeric, const R: usize>(
        &mut self,
        layout: Layout,
        offset: u32,
        access: BufferAccess,
    ) -> Storage<T, R> {
        let view =
            self.storage_view_with_layout_and_access::<R>(T::ELEMENT, layout, offset, access);
        Storage {
            view,
            _ty: PhantomData,
        }
    }

    fn storage_element_with_layout_and_access<const R: usize>(
        &mut self,
        element: crate::ElementType,
        layout: Layout,
        offset: u32,
        access: BufferAccess,
    ) -> Storage<RuntimeElement, R> {
        let view = self.storage_view_with_layout_and_access::<R>(element, layout, offset, access);
        Storage {
            view,
            _ty: PhantomData,
        }
    }

    element_storage_accessors!(
        storage_read_element_with_layout_offset,
        storage_write_element_with_layout_offset(layout: Layout, offset: u32) => (layout, offset)
    );

    fn storage_view_with_layout_and_access<const R: usize>(
        &mut self,
        element: crate::ElementType,
        layout: Layout,
        offset: u32,
        access: BufferAccess,
    ) -> StorageView {
        assert_eq!(
            layout.memory_level(),
            MemoryLevel::Storage,
            "storage tensors must use MemoryLevel::Storage"
        );
        assert_eq!(layout.shape().rank(), R, "storage rank mismatch");
        let buffer = self.alloc_buffer_element(element, layout.clone(), access);
        StorageView {
            buffer,
            offset,
            layout,
        }
    }

    /// Emit a tile-program body over a dispatch grid.
    pub fn program_grid<const BLOCK: usize>(
        &mut self,
        grid: [u32; 3],
        body: impl FnOnce(&mut TileBlock<'_>),
    ) {
        assert!(BLOCK > 0, "tile block size must be non-zero");
        assert!(
            BLOCK <= 1024 && BLOCK.is_power_of_two(),
            "tile block size must be a power of two at most 1024"
        );
        let mut block = TileBlock {
            program: self,
            grid,
            block: BLOCK,
            body: Vec::new(),
            stmt_stack: Vec::new(),
        };
        body(&mut block);
        block.program.ir.body = TileProgramOp {
            grid,
            block: BLOCK as u32,
            body: block.body,
        };
    }

    fn alloc_buffer_element(
        &mut self,
        element: crate::ElementType,
        layout: Layout,
        access: BufferAccess,
    ) -> BufferRef {
        let id = crate::ir::BufferId(post_inc(&mut self.next_buffer));
        let buffer = BufferRef::new(id, element);
        self.ir.buffers.push(BufferDecl {
            id,
            element,
            layout,
            access,
        });
        buffer
    }

    pub(super) fn next_block_dequant_id(&mut self) -> BlockDequantId {
        BlockDequantId(post_inc(&mut self.next_block_dequant))
    }

    pub(super) fn next_coop_fragment_id(&mut self) -> CoopFragmentId {
        CoopFragmentId(post_inc(&mut self.next_coop_fragment))
    }

    pub(super) fn alloc_local<T: Numeric>(&mut self) -> LocalRef {
        self.alloc_local_element(T::ELEMENT)
    }

    pub(super) fn alloc_local_element(&mut self, element: crate::ElementType) -> LocalRef {
        let id = crate::ir::LocalId(post_inc(&mut self.next_local));
        let local = LocalRef::new(id, element);
        self.ir.locals.push(local);
        local
    }

    /// Allocate a rank-2 workgroup-scope tile of shape `[rows, cols]`.
    pub fn alloc_workgroup_tile<T: Numeric>(&mut self, rows: u32, cols: u32) -> Workgroup<T> {
        self.alloc_workgroup_tile_padded::<T>(rows, cols, 0)
    }

    /// Allocate a rank-2 workgroup-scope tile of shape `[rows, cols]` with
    /// `inner_pad` extra elements of stride between consecutive rows. Used to
    /// pad away bank conflicts on the inner axis (e.g. on Apple Silicon).
    pub fn alloc_workgroup_tile_padded<T: Numeric>(
        &mut self,
        rows: u32,
        cols: u32,
        inner_pad: u32,
    ) -> Workgroup<T> {
        self.alloc_tile::<T>(Layout::row_major_padded(
            MemoryLevel::Workgroup,
            Shape::new([rows, cols]),
            inner_pad,
        ))
    }

    /// Allocate a workgroup-scope f32 tile of shape `[rows, cols]`.
    pub fn alloc_workgroup_tile_f32(&mut self, rows: u32, cols: u32) -> Workgroup<F32> {
        self.alloc_workgroup_tile::<F32>(rows, cols)
    }

    /// Allocate a rank-1 workgroup-scope scratch array.
    pub fn alloc_workgroup_array<T: Numeric>(&mut self, len: u32) -> Workgroup<T> {
        self.alloc_tile::<T>(Layout::contiguous(
            MemoryLevel::Workgroup,
            Shape::new([len]),
        ))
    }

    pub(super) fn alloc_tile<T: Numeric>(&mut self, layout: Layout) -> Workgroup<T> {
        let id = crate::ir::TileId(post_inc(&mut self.next_tile));
        let tile = TileRef::new(id, T::ELEMENT);
        self.ir.tiles.push(TileDecl {
            id,
            element: T::ELEMENT,
            layout,
        });
        Workgroup {
            tile,
            _ty: PhantomData,
        }
    }
}

/// Returns the current value and post-increments. The builder uses this for
/// every fresh `BufferId` / `TileId` / `LocalId` / SSA id.
fn post_inc(counter: &mut u32) -> u32 {
    let value = *counter;
    *counter += 1;
    value
}
