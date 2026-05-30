use super::{BufferId, ElementType, Layout, LocalId, TileId};

/// A storage buffer declaration.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferDecl {
    /// Buffer id.
    pub id: BufferId,
    /// Buffer element type.
    pub element: ElementType,
    /// Buffer layout.
    pub layout: Layout,
    /// Required storage access.
    pub access: BufferAccess,
}

/// A storage buffer reference.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferRef {
    /// Buffer id.
    pub id: BufferId,
    /// Buffer element type.
    pub element: ElementType,
}

impl BufferRef {
    /// Create a typed reference to an existing buffer declaration.
    pub const fn new(id: BufferId, element: ElementType) -> Self {
        Self { id, element }
    }
}

/// Access required for a storage buffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BufferAccess {
    /// Read-only storage access.
    Read,
    /// Read-write storage access.
    ReadWrite,
}

/// A typed workgroup tile declaration. Tiles are always workgroup-level and
/// always own their storage — the IR has no other shape today.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TileDecl {
    /// Tile id.
    pub id: TileId,
    /// Tile element type.
    pub element: ElementType,
    /// Tile layout.
    pub layout: Layout,
}

/// A typed reference to a tile declaration.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TileRef {
    /// Tile id.
    pub id: TileId,
    /// Tile element type.
    pub element: ElementType,
}

impl TileRef {
    /// Create a typed reference to an existing tile declaration.
    pub const fn new(id: TileId, element: ElementType) -> Self {
        Self { id, element }
    }
}

/// A typed private per-invocation local. Used both as the declaration in
/// `KernelIr::locals` and as the reference embedded in `Expr::LoadLocal` /
/// `TileStmt::StoreLocal` — they carry the same fields.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct LocalRef {
    /// Local id.
    pub id: LocalId,
    /// Local element type.
    pub element: ElementType,
}

impl LocalRef {
    /// Create a typed reference to an existing private local.
    pub const fn new(id: LocalId, element: ElementType) -> Self {
        Self { id, element }
    }
}

/// A shaped view into a storage buffer.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct StorageView {
    /// Referenced storage buffer.
    pub buffer: BufferRef,
    /// Element offset into `buffer`.
    pub offset: u32,
    /// Logical view layout.
    pub layout: Layout,
}

impl StorageView {
    /// Construct a storage view directly over `buffer`.
    pub fn root(buffer: BufferRef, layout: Layout) -> Self {
        Self {
            buffer,
            offset: 0,
            layout,
        }
    }
}

/// Axis of `@builtin(workgroup_id)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum WorkgroupAxis {
    /// X workgroup-id axis.
    X,
    /// Y workgroup-id axis.
    Y,
    /// Z workgroup-id axis.
    Z,
}

impl WorkgroupAxis {
    pub(crate) const fn index(self) -> u32 {
        match self {
            Self::X => 0,
            Self::Y => 1,
            Self::Z => 2,
        }
    }
}
