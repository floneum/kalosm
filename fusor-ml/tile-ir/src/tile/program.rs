use std::rc::Rc;

use super::value::{CoopAcc, PrivateLocal, WorkgroupTile};
use super::{Storage, TileBlock};
use crate::ir::{
    BufferAccess, BufferDecl, CoopMatrixRole, ElementType, KernelIr, Layout, LocalDecl,
    MemoryLevel, ScalarElement, Shape, StorageView, TileDecl,
};

/// Builder for one tile IR kernel.
///
/// A `Program` owns storage declarations, scratch allocations, and the single
/// tile program body. Most callers construct one through
/// [`build`](crate::tile::build).
///
/// Storage, tile, and local declarations carry an [`ElementType`] as data.
/// Tiles and locals are `Rc`-identified, so the only externally meaningful
/// name is the buffer binding slot.
pub struct Program {
    pub(crate) ir: KernelIr,
    /// Builder-only counter for fresh buffer binding slots. Lives here (not on
    /// `KernelIr`) because the finished IR is immutable data — the counter is
    /// only needed during construction. Tiles and locals need no counter: a
    /// declaration *is* its identity (`Rc::as_ptr`).
    pub(crate) next_binding: u32,
}

impl Program {
    /// Create an empty builder. Most callers should use
    /// [`build`](crate::tile::build) instead; this is for
    /// [`crate::KernelBuilder`] which owns the program plus a parallel binding
    /// list.
    pub fn new() -> Self {
        Self {
            ir: KernelIr::default(),
            next_binding: 0,
        }
    }

    /// Consume the builder and return the constructed [`KernelIr`].
    pub(crate) fn into_ir(self) -> KernelIr {
        self.ir
    }
}

impl Default for Program {
    fn default() -> Self {
        Self::new()
    }
}

impl Program {
    // ---- storage declarations -------------------------------------------

    /// Declare a read-only storage view of `element` with the given `shape`.
    pub fn storage_read(&mut self, element: ElementType, shape: Shape) -> Storage {
        self.storage_with_layout_and_access(
            element,
            Layout::contiguous(MemoryLevel::Storage, shape),
            0,
            BufferAccess::Read,
        )
    }

    /// Declare a read-write storage view of `element` with the given `shape`.
    pub fn storage_write(&mut self, element: ElementType, shape: Shape) -> Storage {
        self.storage_with_layout_and_access(
            element,
            Layout::contiguous(MemoryLevel::Storage, shape),
            0,
            BufferAccess::ReadWrite,
        )
    }

    /// Declare a read-only storage view with an explicit layout and element
    /// offset.
    pub fn storage_read_with_layout_offset(
        &mut self,
        element: ElementType,
        layout: Layout,
        offset: u32,
    ) -> Storage {
        self.storage_with_layout_and_access(element, layout, offset, BufferAccess::Read)
    }

    /// Declare a read-write storage view with an explicit layout and element
    /// offset.
    pub fn storage_write_with_layout_offset(
        &mut self,
        element: ElementType,
        layout: Layout,
        offset: u32,
    ) -> Storage {
        self.storage_with_layout_and_access(element, layout, offset, BufferAccess::ReadWrite)
    }

    fn storage_with_layout_and_access(
        &mut self,
        element: ElementType,
        layout: Layout,
        offset: u32,
        access: BufferAccess,
    ) -> Storage {
        assert_eq!(
            layout.memory_level(),
            MemoryLevel::Storage,
            "storage tensors must use MemoryLevel::Storage"
        );
        let buffer = self.alloc_buffer(element, layout.clone(), access);
        Storage {
            view: StorageView {
                buffer,
                offset,
                layout,
            },
        }
    }

    fn alloc_buffer(
        &mut self,
        element: ElementType,
        layout: Layout,
        access: BufferAccess,
    ) -> crate::ir::Buffer {
        let binding = self.next_binding;
        self.next_binding += 1;
        let buffer = Rc::new(BufferDecl {
            binding,
            element,
            layout,
            access,
        });
        self.ir.buffers.push(buffer.clone());
        buffer
    }

    // ---- grid ------------------------------------------------------------

    /// Emit a tile-program body over a dispatch grid with a runtime `block`
    /// (workgroup invocation) count.
    ///
    /// The lowerer bakes `block` as a shader-compile-time `@workgroup_size`,
    /// so it participates in the emitted Naga and kernel cache key.
    pub fn program_grid(
        &mut self,
        block: u32,
        grid: [u32; 3],
        body: impl FnOnce(&mut TileBlock<'_>),
    ) {
        assert!(block > 0, "tile block size must be non-zero");
        assert!(
            block <= 1024 && block.is_power_of_two(),
            "tile block size must be a power of two at most 1024"
        );
        let mut tile_block = TileBlock {
            program: self,
            grid,
            block,
            body: Vec::new(),
            stmt_stack: Vec::new(),
        };
        body(&mut tile_block);
        let statements = tile_block.body;
        tile_block.program.ir.grid = grid;
        tile_block.program.ir.block = block;
        tile_block.program.ir.body = statements;
    }

    // ---- tile / local allocation ----------------------------------------

    /// Allocate a rank-2 workgroup-scope tile of shape `[rows, cols]`.
    pub fn alloc_workgroup_tile(
        &mut self,
        element: ScalarElement,
        rows: u32,
        cols: u32,
    ) -> WorkgroupTile {
        self.alloc_workgroup_tile_padded(element, rows, cols, 0)
    }

    /// Allocate a rank-2 workgroup-scope tile with `inner_pad` extra stride
    /// elements between consecutive rows (to dodge bank conflicts).
    pub fn alloc_workgroup_tile_padded(
        &mut self,
        element: ScalarElement,
        rows: u32,
        cols: u32,
        inner_pad: u32,
    ) -> WorkgroupTile {
        self.alloc_tile(
            element.element(),
            Layout::row_major_padded(MemoryLevel::Workgroup, Shape::new([rows, cols]), inner_pad),
        )
    }

    /// Allocate a rank-1 workgroup-scope scratch array of arbitrary element
    /// type (vectors included — one load/store moves the whole element).
    pub fn alloc_workgroup_array_elements(
        &mut self,
        element: ElementType,
        len: u32,
    ) -> WorkgroupTile {
        self.alloc_tile(
            element,
            Layout::contiguous(MemoryLevel::Workgroup, Shape::new([len])),
        )
    }

    /// Allocate a rank-1 workgroup-scope scratch array.
    pub fn alloc_workgroup_array(&mut self, element: ScalarElement, len: u32) -> WorkgroupTile {
        self.alloc_tile(
            element.element(),
            Layout::contiguous(MemoryLevel::Workgroup, Shape::new([len])),
        )
    }

    pub(super) fn alloc_tile(&mut self, element: ElementType, layout: Layout) -> WorkgroupTile {
        WorkgroupTile {
            tile: Rc::new(TileDecl { element, layout }),
        }
    }

    pub(super) fn alloc_local(&mut self, element: ElementType) -> PrivateLocal {
        PrivateLocal {
            local: Rc::new(LocalDecl { element }),
        }
    }

    /// Allocate a cooperative-matrix accumulator local of the given scalar and
    /// shape (always `CoopMatrixRole::C`).
    pub(super) fn alloc_coop_acc(
        &mut self,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
    ) -> CoopAcc {
        CoopAcc {
            local: Rc::new(LocalDecl {
                element: ElementType::coop_matrix(scalar, CoopMatrixRole::C, rows, cols),
            }),
        }
    }
}
