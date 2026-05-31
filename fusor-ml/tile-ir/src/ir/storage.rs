use std::hash::{Hash, Hasher};
use std::rc::Rc;

use super::{ElementType, Layout};

/// Access required for a storage buffer.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BufferAccess {
    /// Read-only storage access.
    Read,
    /// Read-write storage access.
    ReadWrite,
}

/// A storage buffer declaration. Owned by the nodes that name it via an
/// [`Rc`]; `binding` is the single externally-meaningful name (see
/// ARBOR_DESIGN.md §3). A declaration *is* its identity — there is no
/// `BufferId`; sharing is `Rc` identity (`Rc::as_ptr`).
#[derive(Debug)]
pub struct BufferDecl {
    /// Binding slot — the one externally-meaningful name.
    pub binding: u32,
    /// Buffer element type.
    pub element: ElementType,
    /// Buffer layout.
    pub layout: Layout,
    /// Required storage access.
    pub access: BufferAccess,
}

/// A typed workgroup tile declaration. Tiles are always workgroup-level and
/// always own their storage — the IR has no other shape today. Owned by use
/// sites via an [`Rc`]; identity is `Rc::as_ptr`, not an id.
#[derive(Debug)]
pub struct TileDecl {
    /// Tile element type.
    pub element: ElementType,
    /// Tile layout.
    pub layout: Layout,
}

/// A typed private per-invocation local. Owned by use sites via an [`Rc`]: a
/// `Loop` owns its `index` and each `Accumulator.local`, so scoping is
/// structural and there is no `LocalId`.
#[derive(Debug)]
pub struct LocalDecl {
    /// Local element type.
    pub element: ElementType,
}

/// Shared, `Rc`-owned handle to a storage buffer declaration.
pub type Buffer = Rc<BufferDecl>;
/// Shared, `Rc`-owned handle to a workgroup tile declaration.
pub type Tile = Rc<TileDecl>;
/// Shared, `Rc`-owned handle to a private local declaration.
pub type Local = Rc<LocalDecl>;

/// A shaped view into a storage buffer.
///
/// `Hash`/`PartialEq`/`Eq` identify the `buffer` by `Rc::as_ptr`, so a view is
/// equal iff it names the *same* declaration with the same offset/layout. This
/// keeps `QuantizedMatrix` (which embeds a view) hashable for the kernel cache
/// key.
#[derive(Clone, Debug)]
pub struct StorageView {
    /// Referenced storage buffer.
    pub buffer: Buffer,
    /// Element offset into `buffer`.
    pub offset: u32,
    /// Logical view layout.
    pub layout: Layout,
}

impl StorageView {
    /// Construct a storage view directly over `buffer`.
    pub fn root(buffer: Buffer, layout: Layout) -> Self {
        Self {
            buffer,
            offset: 0,
            layout,
        }
    }
}

impl PartialEq for StorageView {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.buffer, &other.buffer)
            && self.offset == other.offset
            && self.layout == other.layout
    }
}

impl Eq for StorageView {}

impl Hash for StorageView {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Rc::as_ptr(&self.buffer) as usize).hash(state);
        self.offset.hash(state);
        self.layout.hash(state);
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
