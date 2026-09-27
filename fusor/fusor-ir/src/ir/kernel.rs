//! Kernel `tile` — one kernel body, hash-consed so identical subtrees merge.
//! Produced after extraction, outside the e-graph; element type is runtime data.

use crate::dtype::{NumericContract, QFmt, QLayout};
use crate::error::Result;
use crate::shape::MultiFlattenMap;
use rustc_hash::FxHasher;
use smallvec::SmallVec;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Element types
// ---------------------------------------------------------------------------

/// Scalar elements backing scalar, vector and cooperative-matrix values.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ScalarElement {
    F32,
    F16,
    BF16,
    U32,
    I32,
    /// Exists only at Kernel — Logical encodes booleans as 1.0/0.0.
    Bool,
}

impl ScalarElement {
    pub const fn byte_size(self) -> u64 {
        match self {
            Self::F32 | Self::U32 | Self::I32 | Self::Bool => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }
    pub const fn element(self) -> ElementType {
        ElementType::Scalar(self)
    }
}

/// Cooperative-matrix operand role. A data enum, not typestate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CoopMatrixRole {
    A,
    B,
    C,
}

/// Runtime element type of an Kernel value.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ElementType {
    Scalar(ScalarElement),
    Vector {
        scalar: ScalarElement,
        lanes: u32,
    },
    /// Fragment dims are runtime `u32`; there is no `CoopSize` generic.
    CoopMatrix {
        scalar: ScalarElement,
        role: CoopMatrixRole,
        rows: u32,
        cols: u32,
    },
}

impl ElementType {
    pub const fn byte_size(self) -> u64 {
        match self {
            Self::Scalar(s) => s.byte_size(),
            Self::Vector { scalar, lanes } => scalar.byte_size() * lanes as u64,
            Self::CoopMatrix { scalar, .. } => scalar.byte_size(),
        }
    }

    /// Array stride in a workgroup allocation, or `None` if it cannot back one.
    /// Arena packing and emission both read this, so they cannot disagree.
    pub const fn workgroup_array_stride(self) -> Option<u32> {
        match self {
            Self::Scalar(ScalarElement::Bool) | Self::CoopMatrix { .. } => None,
            Self::Scalar(s) => Some(s.byte_size() as u32),
            Self::Vector { scalar, lanes } => {
                if matches!(scalar, ScalarElement::Bool) {
                    return None;
                }
                let size = scalar.byte_size() as u32;
                match lanes {
                    2 => Some(2 * size),
                    3 | 4 => Some(4 * size),
                    _ => None,
                }
            }
        }
    }

    pub const fn uses_f16(self) -> bool {
        matches!(
            self,
            Self::Scalar(ScalarElement::F16)
                | Self::Vector {
                    scalar: ScalarElement::F16,
                    ..
                }
                | Self::CoopMatrix {
                    scalar: ScalarElement::F16,
                    ..
                }
        )
    }
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

/// Exactly two memory spaces; nothing fusor emits needs any other.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum MemoryLevel {
    Storage,
    Workgroup,
}

/// Access a storage buffer requires.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BufferAccess {
    Read,
    ReadWrite,
}

/// A concrete Kernel layout: extents plus a logical-to-storage index map.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct TileLayout {
    pub extents: SmallVec<[u32; 4]>,
    pub indexing: MultiFlattenMap,
    pub level: MemoryLevel,
}

impl TileLayout {
    pub fn contiguous(level: MemoryLevel, extents: &[u32]) -> Self {
        let mut strides = vec![1u32; extents.len()];
        for axis in (0..extents.len().saturating_sub(1)).rev() {
            strides[axis] = strides[axis + 1] * extents[axis + 1];
        }
        Self {
            extents: extents.iter().copied().collect(),
            indexing: MultiFlattenMap::affine(extents, &strides),
            level,
        }
    }

    pub fn element_count(&self) -> u64 {
        self.extents.iter().map(|e| *e as u64).product()
    }

    pub fn is_affine(&self) -> bool {
        self.indexing.is_affine()
    }
}

/// A storage buffer declaration. Declarations sharing a `binding` are typed
/// views of one buffer.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferDecl {
    pub binding: u32,
    pub element: ElementType,
    pub layout: TileLayout,
    pub access: BufferAccess,
}

/// A workgroup tile declaration. Identity-bearing: two same-shaped tiles are
/// two allocations (e.g. double buffering), so equality keys on `id`.
#[derive(Clone, Debug, Eq)]
pub struct TileDecl {
    pub element: ElementType,
    pub layout: TileLayout,
    pub name: &'static str,
    id: u64,
}

thread_local! {
    /// Decl ids, unique within one kernel build. Resettable so identical
    /// lowerings mint identical ids and the pipeline cache dedups them.
    static NEXT_DECL_ID: std::cell::Cell<u64> = const { std::cell::Cell::new(1) };
}

/// Restart decl numbering, so a rebuilt kernel matches its first build.
pub fn reset_decl_ids() {
    NEXT_DECL_ID.with(|c| c.set(1));
}

fn fresh_decl_id() -> u64 {
    NEXT_DECL_ID.with(|c| {
        let id = c.get();
        c.set(id + 1);
        id
    })
}

impl TileDecl {
    pub fn new(element: ElementType, layout: TileLayout, name: &'static str) -> Self {
        Self {
            element,
            layout,
            name,
            id: fresh_decl_id(),
        }
    }

    pub const fn id(&self) -> u64 {
        self.id
    }
}

impl PartialEq for TileDecl {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Hash for TileDecl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.id);
    }
}

/// A private per-invocation local. Identity-bearing: two same-typed locals
/// are two registers, so equality and hashing key on `id`.
#[derive(Clone, Debug, Eq)]
pub struct LocalDecl {
    pub element: ElementType,
    id: u64,
}

impl LocalDecl {
    pub fn new(element: ElementType) -> Self {
        Self {
            element,
            id: fresh_decl_id(),
        }
    }

    pub const fn id(&self) -> u64 {
        self.id
    }
}

impl PartialEq for LocalDecl {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Hash for LocalDecl {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.id);
    }
}

/// Shared handle to a storage buffer (`Arc`: kernels build on worker threads).
pub type Buffer = Arc<BufferDecl>;
/// Shared handle to a workgroup tile.
pub type Tile = Arc<TileDecl>;
/// Shared handle to a private local.
pub type Local = Arc<LocalDecl>;

/// A shaped view into a storage buffer.
#[derive(Clone, Debug)]
pub struct StorageView {
    pub buffer: Buffer,
    pub offset: u32,
    pub layout: TileLayout,
}

impl PartialEq for StorageView {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.buffer, &other.buffer)
            && self.offset == other.offset
            && self.layout == other.layout
    }
}
impl Eq for StorageView {}
impl Hash for StorageView {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (Arc::as_ptr(&self.buffer) as usize).hash(state);
        self.offset.hash(state);
        self.layout.hash(state);
    }
}

/// Axis of `@builtin(workgroup_id)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum WorkgroupAxis {
    X,
    Y,
    Z,
}

// ---------------------------------------------------------------------------
// Op tables
// ---------------------------------------------------------------------------

/// The 21 unary math functions.
pub type TileUnaryOp = crate::scalar::UnOp;
/// The 15 binary ops.
pub type TileBinaryOp = crate::scalar::BinOp;
/// The 6 comparisons.
pub type TileCompareOp = crate::scalar::CmpOp;

/// Cross-lane reduction operators with a direct hardware spelling. Anything
/// wider goes through [`Stmt::Reduce`]'s [`MergeBody`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TileReduceOp {
    Sum,
    Product,
    Max,
    Min,
}

impl TileReduceOp {
    /// The binary operator this folds with.
    pub const fn binary(self) -> TileBinaryOp {
        match self {
            Self::Sum => TileBinaryOp::Add,
            Self::Product => TileBinaryOp::Mul,
            Self::Max => TileBinaryOp::Max,
            Self::Min => TileBinaryOp::Min,
        }
    }

    /// The operator a binary merge folds with, or `None` when the hardware has
    /// no collective for it.
    pub const fn of_binary(op: TileBinaryOp) -> Option<Self> {
        Some(match op {
            TileBinaryOp::Add => Self::Sum,
            TileBinaryOp::Mul => Self::Product,
            TileBinaryOp::Max => Self::Max,
            TileBinaryOp::Min => Self::Min,
            _ => return None,
        })
    }
}

/// The hardware collective a carrier reduces with, or `None` for anything the
/// N-ary [`Stmt::Reduce`] has to carry. Both emitters read this so the fast
/// path cannot drift between them.
pub fn fast_reduce_op(c: &crate::carrier::Carrier) -> Option<TileReduceOp> {
    if !matches!(c.slots.as_slice(), [crate::carrier::SlotTy::Scalar]) {
        return None;
    }
    TileReduceOp::of_binary(c.kind()?)
}

/// Built-in u32 quantities appearing as leaves in index arithmetic.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Builtin {
    Lane,
    ProgramId(WorkgroupAxis),
    /// One extent of the dispatched grid (`@builtin(num_workgroups)`), so the
    /// body never bakes the grid in.
    NumWorkgroups(WorkgroupAxis),
    SubgroupId,
    SubgroupLane,
    SubgroupSize,
    NumSubgroups,
}

/// A typed Kernel literal.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TileLiteral {
    F32(u32),
    F16(u16),
    BF16(u16),
    U32(u32),
    I32(i32),
    Bool(bool),
}

/// Source of a [`TileExprKind::Load`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Source {
    Storage(StorageView),
    Quantized(QuantizedView),
}

/// A quantized matrix bound as a plain u32 storage buffer. Carries no
/// extents: the decode never reads them, so a bound is computed by its user.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct QuantizedView {
    pub data: StorageView,
    pub fmt: QFmt,
    pub layout: QLayout,
}

/// Address of a memory access.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Addr {
    Linear(TileExpr),
    Rc2 { row: TileExpr, col: TileExpr },
}

impl Addr {
    /// The expressions inside this address.
    pub fn for_each_expr(&self, f: &mut dyn FnMut(&TileExpr)) {
        match self {
            Addr::Linear(index) => f(index),
            Addr::Rc2 { row, col } => {
                f(row);
                f(col);
            }
        }
    }
}

/// Cross-lane reduction strategy, a late capability-driven choice.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReduceKind {
    Subgroup,
    Workgroup { scratch: Tile, group_size: u32 },
}

/// Source region of a cooperative fragment load.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CoopSrc {
    pub tile: Tile,
    pub row: TileExpr,
    pub col: TileExpr,
    pub transposed: bool,
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

/// A hash-consed Kernel value; `ty` and `hash` are cached at construction.
#[derive(Clone, Debug)]
pub struct TileExpr(Arc<TileNode>);

/// An Kernel node with its cached type, hash and memory-read set.
#[derive(Debug)]
pub struct TileNode {
    pub kind: TileExprKind,
    pub ty: ElementType,
    pub hash: u64,
    /// Which memory spaces this tree reads, folded up at construction.
    pub mem_reads: MemReads,
    /// Collective results depend on which invocations reach the expression.
    pub scope_dependent: bool,
}

/// The memory spaces a [`TileExpr`] reads: lets a hash-consing emitter drop
/// exactly the memo entries a write or barrier makes stale.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct MemReads(u8);

impl MemReads {
    /// A pure tree: the same value forever.
    pub const NONE: Self = Self(0);
    /// A storage buffer (including the u32 buffer behind a quantized view).
    pub const STORAGE: Self = Self(1 << 0);
    /// A workgroup tile.
    pub const TILE: Self = Self(1 << 1);
    /// A private per-invocation local.
    pub const LOCAL: Self = Self(1 << 2);
    /// Every space, for a caller that wants to invalidate wholesale.
    pub const ALL: Self = Self(0b111);

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
    /// True when the two sets share a space.
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
    pub const fn bits(self) -> u8 {
        self.0
    }
}

/// The Kernel value tree.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum TileExprKind {
    // leaves
    Literal(TileLiteral),
    Builtin(Builtin),
    LoadLocal(Local),
    // memory
    Load {
        src: Source,
        addr: Box<Addr>,
        mask: TileExpr,
        fill: TileExpr,
    },
    LoadTile {
        tile: Tile,
        index: TileExpr,
    },
    // ALU — `numeric` is the emitter obligation: `reassoc: false` forbids
    // fast-math folding.
    Unary {
        op: TileUnaryOp,
        value: TileExpr,
        numeric: NumericContract,
    },
    Binary {
        op: TileBinaryOp,
        left: TileExpr,
        right: TileExpr,
        numeric: NumericContract,
    },
    Compare {
        op: TileCompareOp,
        left: TileExpr,
        right: TileExpr,
    },
    Round {
        mode: crate::dtype::RoundMode,
        value: TileExpr,
    },
    Cast {
        value: TileExpr,
        to: ElementType,
    },
    Bitcast {
        value: TileExpr,
        to: ElementType,
    },
    Select {
        condition: TileExpr,
        accept: TileExpr,
        reject: TileExpr,
    },
    Vec {
        scalar: ScalarElement,
        lanes: u32,
        parts: Vec<TileExpr>,
    },
    VecComponent {
        vector: TileExpr,
        component: u32,
    },
    Dot {
        left: TileExpr,
        right: TileExpr,
    },
    // reductions
    Reduce {
        op: TileReduceOp,
        kind: Box<ReduceKind>,
        value: TileExpr,
    },
    // cooperative matrix
    /// An all-zero cooperative-matrix fragment: a coop accumulator's init must
    /// have the fragment type, and no arithmetic makes one from a scalar.
    CoopZero {
        role: CoopMatrixRole,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
    },
    CoopLoad {
        role: CoopMatrixRole,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
        src: Box<CoopSrc>,
    },
    CoopMma {
        a: TileExpr,
        b: TileExpr,
        c: TileExpr,
    },
}

impl TileExprKind {
    /// Every direct child expression of a node, in a fixed order.
    pub fn visit_children(&self, f: &mut dyn FnMut(&TileExpr)) {
        match self {
            TileExprKind::Literal(_)
            | TileExprKind::Builtin(_)
            | TileExprKind::LoadLocal(_)
            | TileExprKind::CoopZero { .. } => {}
            TileExprKind::Load {
                addr, mask, fill, ..
            } => {
                match addr.as_ref() {
                    Addr::Linear(index) => f(index),
                    Addr::Rc2 { row, col } => {
                        f(row);
                        f(col);
                    }
                }
                f(mask);
                f(fill);
            }
            TileExprKind::LoadTile { index, .. } => f(index),
            TileExprKind::Unary { value, .. } => f(value),
            TileExprKind::Binary { left, right, .. }
            | TileExprKind::Compare { left, right, .. } => {
                f(left);
                f(right);
            }
            TileExprKind::Round { value, .. } => f(value),
            TileExprKind::Cast { value, .. } | TileExprKind::Bitcast { value, .. } => f(value),
            TileExprKind::Select {
                condition,
                accept,
                reject,
            } => {
                f(condition);
                f(accept);
                f(reject);
            }
            TileExprKind::Vec { parts, .. } => {
                for part in parts {
                    f(part);
                }
            }
            TileExprKind::VecComponent { vector, .. } => f(vector),
            TileExprKind::Dot { left, right } => {
                f(left);
                f(right);
            }
            TileExprKind::Reduce { value, .. } => f(value),
            TileExprKind::CoopLoad { src, .. } => {
                f(&src.row);
                f(&src.col);
            }
            TileExprKind::CoopMma { a, b, c } => {
                f(a);
                f(b);
                f(c);
            }
        }
    }
}

impl TileExpr {
    pub fn new(kind: TileExprKind, ty: ElementType) -> Self {
        let mut h = FxHasher::default();
        kind.hash(&mut h);
        ty.hash(&mut h);
        let mem_reads = kind_mem_reads(&kind);
        let mut scope_dependent = matches!(&kind, TileExprKind::Reduce { kind, .. }
            if matches!(kind.as_ref(), ReduceKind::Subgroup));
        kind.visit_children(&mut |child| scope_dependent |= child.scope_dependent());
        Self(Arc::new(TileNode {
            kind,
            ty,
            hash: h.finish(),
            mem_reads,
            scope_dependent,
        }))
    }
    pub fn kind(&self) -> &TileExprKind {
        &self.0.kind
    }
    pub fn element(&self) -> ElementType {
        self.0.ty
    }
    pub fn structural_hash(&self) -> u64 {
        self.0.hash
    }

    /// This node's identity, as the address of its shared allocation. A body
    /// is a DAG; walks memoize on this to stay linear.
    pub fn node_ptr(&self) -> usize {
        Arc::as_ptr(&self.0) as *const () as usize
    }
    /// A statically-true mask, which the lowerer skips codegen for.
    pub fn is_constant_true(&self) -> bool {
        matches!(&self.0.kind, TileExprKind::Literal(TileLiteral::Bool(true)))
    }

    /// Which memory spaces this tree reads anywhere inside it.
    pub fn mem_reads(&self) -> MemReads {
        self.0.mem_reads
    }

    pub fn scope_dependent(&self) -> bool {
        self.0.scope_dependent
    }
}

/// Fold the memory-read set for one node from its children. Exhaustive so a
/// new kind must state what it reads.
fn kind_mem_reads(kind: &TileExprKind) -> MemReads {
    use TileExprKind as K;
    let direct = match kind {
        K::LoadLocal(_) => MemReads::LOCAL,
        K::Load { .. } => MemReads::STORAGE,
        K::LoadTile { .. } | K::CoopLoad { .. } => MemReads::TILE,
        K::Reduce { kind, .. } => match kind.as_ref() {
            ReduceKind::Subgroup => MemReads::NONE,
            ReduceKind::Workgroup { .. } => MemReads::TILE,
        },
        K::Literal(_)
        | K::Builtin(_)
        | K::CoopZero { .. }
        | K::Unary { .. }
        | K::Binary { .. }
        | K::Compare { .. }
        | K::Round { .. }
        | K::Cast { .. }
        | K::Bitcast { .. }
        | K::Select { .. }
        | K::Vec { .. }
        | K::VecComponent { .. }
        | K::Dot { .. }
        | K::CoopMma { .. } => MemReads::NONE,
    };
    let mut reads = direct;
    kind.visit_children(&mut |child| reads = reads.union(child.mem_reads()));
    reads
}

impl PartialEq for TileExpr {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || (self.0.hash == other.0.hash && self.0.kind == other.0.kind)
    }
}
impl Eq for TileExpr {}
impl Hash for TileExpr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0.hash);
    }
}

// ---------------------------------------------------------------------------
// Statements
// ---------------------------------------------------------------------------

/// One accumulator carried by a counted loop, so the lowerer emits
/// SSA-carried values rather than reloading per iteration.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Accumulator {
    pub local: Local,
    pub init: TileExpr,
    pub update: TileExpr,
}

/// The merge of two partial accumulators, one expression per scalar lane
/// (`body.len()` is the carrier's `lanes()`). Lanes may read each other's
/// `lhs`/`rhs` (flash reads the running max); nothing outside them.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MergeBody {
    /// Formal parameters for the left partial, one `Local` per lane.
    pub lhs: SmallVec<[Local; 4]>,
    /// Formal parameters for the right partial.
    pub rhs: SmallVec<[Local; 4]>,
    /// One expression per lane, reading only `lhs`/`rhs` locals and literals.
    pub body: SmallVec<[TileExpr; 4]>,
}

impl MergeBody {
    pub fn lanes(&self) -> usize {
        self.body.len()
    }
    /// Arity agreement across the three vectors.
    pub fn is_arity_consistent(&self) -> bool {
        self.lhs.len() == self.body.len() && self.rhs.len() == self.body.len()
    }
}

/// One ordered Kernel statement. `CoopStore` is subgroup-collective;
/// `CoopStoreTile` stages fragments for per-lane math (attention softmax).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Stmt {
    Store {
        dst: StorageView,
        addr: Addr,
        value: TileExpr,
        mask: TileExpr,
    },
    /// Added for `ScatterMode::Atomic`; carries `Effect::InPlace` at Launch.
    AtomicAdd {
        dst: StorageView,
        addr: Addr,
        value: TileExpr,
        mask: TileExpr,
    },
    StoreLocal {
        dst: Local,
        value: TileExpr,
    },
    StoreTile {
        dst: Tile,
        index: TileExpr,
        value: TileExpr,
    },
    FillTile {
        dst: Tile,
        value: TileExpr,
        bounds: [Option<TileExpr>; 2],
    },
    CoopStore {
        acc: TileExpr,
        dst: StorageView,
        addr: Addr,
    },
    CoopStoreTile {
        acc: TileExpr,
        tile: Tile,
        row: TileExpr,
        col: TileExpr,
    },
    If {
        condition: TileExpr,
        accept: Vec<Stmt>,
        reject: Vec<Stmt>,
    },
    Loop {
        count: Option<TileExpr>,
        index: Option<Local>,
        accumulators: Vec<Accumulator>,
        body: Vec<Stmt>,
    },
    /// The N-ary cross-lane reduction: one partial and one `outs` local per
    /// lane, folded by `merge`. `fast` is set iff it is one lane of a plain
    /// hardware op. `scratch` is one tile per lane (`Workgroup` only).
    Reduce {
        kind: Box<ReduceKind>,
        values: SmallVec<[TileExpr; 4]>,
        merge: Box<MergeBody>,
        fast: Option<TileReduceOp>,
        outs: SmallVec<[Local; 4]>,
        scratch: SmallVec<[Tile; 4]>,
    },
    Break,
    Return,
    Barrier,
    StorageBarrier,
}

impl Stmt {
    /// Every statement of `body` and of the bodies nested in it, pre-order.
    pub fn walk(body: &[Stmt], f: &mut dyn FnMut(&Stmt)) {
        for stmt in body {
            f(stmt);
            match stmt {
                Stmt::If { accept, reject, .. } => {
                    Stmt::walk(accept, f);
                    Stmt::walk(reject, f);
                }
                Stmt::Loop { body, .. } => Stmt::walk(body, f),
                _ => {}
            }
        }
    }

    /// [`Stmt::walk`], mutably.
    pub fn walk_mut(body: &mut [Stmt], f: &mut dyn FnMut(&mut Stmt)) {
        for stmt in body {
            f(stmt);
            match stmt {
                Stmt::If { accept, reject, .. } => {
                    Stmt::walk_mut(accept, f);
                    Stmt::walk_mut(reject, f);
                }
                Stmt::Loop { body, .. } => Stmt::walk_mut(body, f),
                _ => {}
            }
        }
    }

    /// The expressions of this statement, not of its nested bodies.
    pub fn for_each_expr(&self, f: &mut dyn FnMut(&TileExpr)) {
        match self {
            Stmt::Store {
                addr, value, mask, ..
            }
            | Stmt::AtomicAdd {
                addr, value, mask, ..
            } => {
                addr.for_each_expr(f);
                f(value);
                f(mask);
            }
            Stmt::StoreLocal { value, .. } => f(value),
            Stmt::StoreTile { index, value, .. } => {
                f(index);
                f(value);
            }
            Stmt::FillTile { value, bounds, .. } => {
                f(value);
                for bound in bounds.iter().flatten() {
                    f(bound);
                }
            }
            Stmt::CoopStore { acc, addr, .. } => {
                f(acc);
                addr.for_each_expr(f);
            }
            Stmt::CoopStoreTile { acc, row, col, .. } => {
                f(acc);
                f(row);
                f(col);
            }
            Stmt::If { condition, .. } => f(condition),
            Stmt::Loop {
                count,
                accumulators,
                ..
            } => {
                if let Some(count) = count {
                    f(count);
                }
                for Accumulator { init, update, .. } in accumulators {
                    f(init);
                    f(update);
                }
            }
            Stmt::Reduce { values, merge, .. } => {
                for value in values {
                    f(value);
                }
                for lane in &merge.body {
                    f(lane);
                }
            }
            Stmt::Break | Stmt::Return | Stmt::Barrier | Stmt::StorageBarrier => {}
        }
    }

    /// The memory spaces this statement makes stale: those it writes, or for
    /// a barrier those whose other-invocation writes it publishes. `If` and
    /// `Loop` name nothing; their bodies name their own.
    pub fn writes(&self) -> MemReads {
        match self {
            Self::Store { .. } | Self::AtomicAdd { .. } | Self::CoopStore { .. } => {
                MemReads::STORAGE
            }
            Self::StoreLocal { .. } => MemReads::LOCAL,
            Self::StoreTile { .. } | Self::FillTile { .. } | Self::CoopStoreTile { .. } => {
                MemReads::TILE
            }
            // Its scratch tiles and `outs` locals.
            Self::Reduce { .. } => MemReads::TILE.union(MemReads::LOCAL),
            // Conservatively both shared spaces.
            Self::Barrier | Self::StorageBarrier => MemReads::STORAGE.union(MemReads::TILE),
            Self::If { .. } | Self::Loop { .. } | Self::Break | Self::Return => MemReads::NONE,
        }
    }
}

// ---------------------------------------------------------------------------
// Capability tokens
// ---------------------------------------------------------------------------

/// Proof that the device supports workgroup byte-arena aliasing.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ByteArenaToken;

// ---------------------------------------------------------------------------
// Kernel IR
// ---------------------------------------------------------------------------

/// One kernel body. `buffers` is in binding order; binding 0 is always the
/// uniform block. `grid` is already folded against
/// `max_compute_workgroups_per_dimension`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct KernelIr {
    pub buffers: Vec<Buffer>,
    pub grid: [u32; 3],
    pub block: u32,
    pub body: Vec<Stmt>,
    pub byte_arena: Option<ByteArenaToken>,
    pub name: &'static str,
}

/// How workgroup tiles are packed.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ArenaMode {
    /// One allocation per stride class, every tile at offset 0.
    Regions,
    /// One byte arena, tiles at byte offsets. Needs [`ByteArenaToken`].
    ByteArena,
}

/// Where one tile lives in the packed arena.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Placement {
    pub tile: Tile,
    pub byte_offset: u32,
    pub byte_len: u32,
}

/// Tiles a candidate geometry declares, before packing.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Tiles {
    pub decls: SmallVec<[Tile; 8]>,
}

/// The result of workgroup-arena planning: a pure memoized function shared by
/// `verify_launch` admission and the emitter's layout.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ArenaPlan {
    pub mode: ArenaMode,
    pub total_bytes: u32,
    pub placements: SmallVec<[Placement; 8]>,
    /// Root-level statement indices where a barrier was inserted, best first.
    pub barriers_inserted: SmallVec<[u32; 4]>,
}

/// Barrier-insertion candidate with its measured saving.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BarrierSuggestion {
    pub index: u32,
    pub bytes_saved: u32,
}

/// Workgroup-memory planning, liveness and the arena verifier. Object-safe;
/// one implementation lives in `fusor-tile`.
pub trait ArenaPlanner: Send + Sync {
    /// Pack under `caps`, taking the argmin of `total_bytes` over
    /// `{Regions, ByteArena} x {no barrier, top-3 insertions}`. Memoized.
    fn arena_plan(&self, ir: &KernelIr, caps: &crate::device::Caps) -> Result<ArenaPlan>;

    /// Workgroup bytes a candidate geometry needs, without building the
    /// body — the exact value `verify_launch` admits against.
    fn workgroup_bytes(&self, tiles: &Tiles, caps: &crate::device::Caps) -> Result<u32>;

    fn barrier_suggestions(&self, ir: &KernelIr) -> Vec<BarrierSuggestion>;

    /// All-pairs recheck: every byte-overlapping tile pair must be separated
    /// by a guaranteed-uniform barrier.
    fn verify_arena(&self, ir: &KernelIr, plan: &ArenaPlan) -> Result<()>;

    /// A `Barrier` may not appear under an `If` whose predicate is
    /// non-uniform over the group.
    fn verify_uniformity(&self, ir: &KernelIr) -> Result<()>;
}

/// Why Kernel lowering failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LowerError {
    BarrierHazard(String),
    NonUniformBarrier(String),
    UnmaskedLoad(String),
    CoopStoreLayout(String),
    Validation(String),
}

impl fmt::Display for LowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BarrierHazard(e) => write!(f, "workgroup barrier hazard: {e}"),
            Self::NonUniformBarrier(e) => write!(f, "barrier under non-uniform control: {e}"),
            Self::UnmaskedLoad(e) => write!(f, "load not provably in range: {e}"),
            Self::CoopStoreLayout(e) => write!(f, "cooperative store layout: {e}"),
            Self::Validation(e) => write!(f, "validation failed: {e}"),
        }
    }
}
impl std::error::Error for LowerError {}

/// `CoopStore` requires an affine rank-2 destination with a unit stride on
/// one side; anything else falls back to a per-lane store path.
pub fn cooperative_store_layout_supported(layout: &TileLayout) -> bool {
    if !layout.is_affine() || layout.extents.len() != 2 {
        return false;
    }
    let strides: SmallVec<[u32; 2]> = layout
        .indexing
        .groups
        .iter()
        .map(|g| g.sub_axes[0].stride)
        .collect();
    strides[0] == 1 || strides[1] == 1
}

// ---------------------------------------------------------------------------
// Aligned-window index algebra
// ---------------------------------------------------------------------------

/// Largest `t` such that `e` is provably a multiple of `2^t` — the number of
/// low bits known to be zero. Conservative: `0` whenever nothing is known.
fn known_zero_low_bits(e: &TileExpr) -> u32 {
    match e.kind() {
        TileExprKind::Literal(TileLiteral::U32(v)) => {
            if *v == 0 {
                31
            } else {
                v.trailing_zeros().min(31)
            }
        }
        TileExprKind::Binary {
            op, left, right, ..
        } => match op {
            TileBinaryOp::Mul => (known_zero_low_bits(left) + known_zero_low_bits(right)).min(31),
            TileBinaryOp::Add | TileBinaryOp::Sub => {
                known_zero_low_bits(left).min(known_zero_low_bits(right))
            }
            TileBinaryOp::Shl => match lit_u32(right) {
                Some(s) => (known_zero_low_bits(left) + s).min(31),
                None => 0,
            },
            TileBinaryOp::Shr => match lit_u32(right) {
                Some(s) => known_zero_low_bits(left).saturating_sub(s),
                None => 0,
            },
            TileBinaryOp::BitAnd => {
                let from_mask = match (lit_u32(left), lit_u32(right)) {
                    (Some(m), _) | (_, Some(m)) => {
                        if m == 0 {
                            31
                        } else {
                            m.trailing_zeros().min(31)
                        }
                    }
                    _ => 0,
                };
                known_zero_low_bits(left)
                    .max(known_zero_low_bits(right))
                    .max(from_mask)
            }
            TileBinaryOp::BitOr => known_zero_low_bits(left).min(known_zero_low_bits(right)),
            _ => 0,
        },
        _ => 0,
    }
}

fn lit_u32(e: &TileExpr) -> Option<u32> {
    match e.kind() {
        TileExprKind::Literal(TileLiteral::U32(v)) => Some(*v),
        _ => None,
    }
}

/// Upper bound on `e mod 2^s` (`s <= 32`), computed structurally: proves a
/// split window's offsets carry-free where alignment alone cannot. Sound under
/// wrapping u32 since mod `2^s` is a ring homomorphism of Add/Mul/Shl.
fn low_max(e: &TileExpr, s: u32) -> u128 {
    let s = s.min(32);
    if s == 0 {
        return 0;
    }
    let cap = (1u128 << s) - 1;
    let bound = match e.kind() {
        TileExprKind::Literal(TileLiteral::U32(v)) => u128::from(*v) & cap,
        TileExprKind::Binary {
            op, left, right, ..
        } => match op {
            TileBinaryOp::Add => low_max(left, s) + low_max(right, s),
            TileBinaryOp::Mul => {
                let by_lit = |a: &TileExpr, l: u32| -> u128 {
                    if l == 0 {
                        return 0;
                    }
                    // `a * (m << t) mod 2^s = ((a * m) mod 2^(s-t)) << t`.
                    let t = l.trailing_zeros().min(s);
                    let m = u128::from(l >> t);
                    let inner_cap = (1u128 << (s - t)) - 1;
                    (low_max(a, s - t).saturating_mul(m)).min(inner_cap) << t
                };
                match (lit_u32(left), lit_u32(right)) {
                    (_, Some(l)) => by_lit(left, l),
                    (Some(l), _) => by_lit(right, l),
                    _ => low_max(left, 32).saturating_mul(low_max(right, 32)),
                }
            }
            TileBinaryOp::Div => match lit_u32(right) {
                Some(d) if d > 0 => low_max(left, 32) / u128::from(d),
                _ => cap,
            },
            TileBinaryOp::Rem => match lit_u32(right) {
                Some(d) if d > 0 => {
                    let global = u128::from(d) - 1;
                    let refined = if d.is_power_of_two() {
                        low_max(left, d.trailing_zeros().min(s))
                    } else {
                        cap
                    };
                    global.min(refined).min(low_max(left, 32))
                }
                _ => cap,
            },
            TileBinaryOp::BitAnd => match (lit_u32(left), lit_u32(right)) {
                (_, Some(m)) => u128::from(m).min(low_max(left, s)),
                (Some(m), _) => u128::from(m).min(low_max(right, s)),
                _ => low_max(left, s).min(low_max(right, s)),
            },
            // `a | b <= a + b`, and mod 2^s is bitwise for both sides.
            TileBinaryOp::BitOr | TileBinaryOp::BitXor => low_max(left, s) + low_max(right, s),
            TileBinaryOp::Shl => match lit_u32(right) {
                Some(k) if k < s => low_max(left, s - k) << k,
                Some(_) => 0,
                None => cap,
            },
            TileBinaryOp::Shr => match lit_u32(right) {
                Some(k) => low_max(left, (s + k).min(32)) >> k,
                None => cap,
            },
            _ => cap,
        },
        _ => cap,
    };
    bound.min(cap)
}

/// `Some((base, c))` when `e` is a top-level `base + c` with a literal `c`,
/// with no alignment claim.
fn top_literal_add(e: &TileExpr) -> Option<(TileExpr, u32)> {
    let TileExprKind::Binary {
        op: TileBinaryOp::Add,
        left,
        right,
        ..
    } = e.kind()
    else {
        return None;
    };
    match (lit_u32(left), lit_u32(right)) {
        (_, Some(c)) => Some((left.clone(), c)),
        (Some(c), _) => Some((right.clone(), c)),
        _ => None,
    }
}

/// `Some((aligned, c))` when `e` is `aligned + c` with the addition provably
/// carry-free: `c` is a literal strictly below `2^t` for `t` the aligned
/// side's known zero low bits. A straddling literal's aligned high part folds
/// into the base; peeling nothing returns `None`, which ends the recursion.
fn carry_free_add(e: &TileExpr) -> Option<(TileExpr, u32)> {
    let TileExprKind::Binary {
        op: TileBinaryOp::Add,
        left,
        right,
        numeric,
    } = e.kind()
    else {
        return None;
    };
    let (base, c) = match (lit_u32(left), lit_u32(right)) {
        (_, Some(c)) => (left, c),
        (Some(c), _) => (right, c),
        _ => return None,
    };
    let t = known_zero_low_bits(base);
    if t >= 32 || u64::from(c) < (1u64 << t) {
        return Some((base.clone(), c));
    }
    let mask = (1u32 << t) - 1;
    let (c_lo, c_hi) = (c & mask, c & !mask);
    if c_lo == 0 {
        return None;
    }
    let base = TileExpr::new(
        TileExprKind::Binary {
            op: TileBinaryOp::Add,
            left: base.clone(),
            right: TileExpr::new(TileExprKind::Literal(TileLiteral::U32(c_hi)), e.element()),
            numeric: *numeric,
        },
        e.element(),
    );
    Some((base, c_lo))
}

/// Rewrite index arithmetic under aligned-window algebra: when `a + c` cannot
/// carry, shifts, masks and power-of-two div/rem distribute over it. Makes
/// consecutive window elements' addresses structurally equal so loads share.
pub fn simplify_index(e: &TileExpr) -> TileExpr {
    simplify_index_cached(e, &mut rustc_hash::FxHashMap::default())
}

fn simplify_index_cached(
    e: &TileExpr,
    memo: &mut rustc_hash::FxHashMap<TileExpr, TileExpr>,
) -> TileExpr {
    if let Some(result) = memo.get(e) {
        return result.clone();
    }
    let result = rewrite_index(e, memo);
    memo.insert(e.clone(), result.clone());
    result
}

fn rewrite_index(e: &TileExpr, memo: &mut rustc_hash::FxHashMap<TileExpr, TileExpr>) -> TileExpr {
    let rebuilt = match e.kind() {
        TileExprKind::Binary {
            op,
            left,
            right,
            numeric,
        } => {
            let l = simplify_index_cached(left, memo);
            let r = simplify_index_cached(right, memo);
            TileExpr::new(
                TileExprKind::Binary {
                    op: *op,
                    left: l,
                    right: r,
                    numeric: *numeric,
                },
                e.element(),
            )
        }
        TileExprKind::Unary { op, value, numeric } => TileExpr::new(
            TileExprKind::Unary {
                op: *op,
                value: simplify_index_cached(value, memo),
                numeric: *numeric,
            },
            e.element(),
        ),
        TileExprKind::Compare { op, left, right } => TileExpr::new(
            TileExprKind::Compare {
                op: *op,
                left: simplify_index_cached(left, memo),
                right: simplify_index_cached(right, memo),
            },
            e.element(),
        ),
        TileExprKind::Cast { value, to } => TileExpr::new(
            TileExprKind::Cast {
                value: simplify_index_cached(value, memo),
                to: *to,
            },
            e.element(),
        ),
        TileExprKind::Bitcast { value, to } => TileExpr::new(
            TileExprKind::Bitcast {
                value: simplify_index_cached(value, memo),
                to: *to,
            },
            e.element(),
        ),
        TileExprKind::Select {
            condition,
            accept,
            reject,
        } => TileExpr::new(
            TileExprKind::Select {
                condition: simplify_index_cached(condition, memo),
                accept: simplify_index_cached(accept, memo),
                reject: simplify_index_cached(reject, memo),
            },
            e.element(),
        ),
        TileExprKind::Round { mode, value } => TileExpr::new(
            TileExprKind::Round {
                mode: *mode,
                value: simplify_index_cached(value, memo),
            },
            e.element(),
        ),
        // Container arms matter: a missing one fences the rewrite out of the
        // subtree (the f16 scale decode rides `VecComponent`).
        TileExprKind::Vec {
            scalar,
            lanes,
            parts,
        } => TileExpr::new(
            TileExprKind::Vec {
                scalar: *scalar,
                lanes: *lanes,
                parts: parts
                    .iter()
                    .map(|part| simplify_index_cached(part, memo))
                    .collect(),
            },
            e.element(),
        ),
        TileExprKind::VecComponent { vector, component } => TileExpr::new(
            TileExprKind::VecComponent {
                vector: simplify_index_cached(vector, memo),
                component: *component,
            },
            e.element(),
        ),
        TileExprKind::Dot { left, right } => TileExpr::new(
            TileExprKind::Dot {
                left: simplify_index_cached(left, memo),
                right: simplify_index_cached(right, memo),
            },
            e.element(),
        ),
        TileExprKind::Load {
            src,
            addr,
            mask,
            fill,
        } => {
            let addr = match addr.as_ref() {
                Addr::Linear(i) => Addr::Linear(simplify_index_cached(i, memo)),
                Addr::Rc2 { row, col } => Addr::Rc2 {
                    row: simplify_index_cached(row, memo),
                    col: simplify_index_cached(col, memo),
                },
            };
            TileExpr::new(
                TileExprKind::Load {
                    src: src.clone(),
                    addr: Box::new(addr),
                    mask: simplify_index_cached(mask, memo),
                    fill: simplify_index_cached(fill, memo),
                },
                e.element(),
            )
        }
        TileExprKind::LoadTile { tile, index } => TileExpr::new(
            TileExprKind::LoadTile {
                tile: tile.clone(),
                index: simplify_index_cached(index, memo),
            },
            e.element(),
        ),
        // Leaves and forms no index expression rides through.
        _ => return e.clone(),
    };

    let u32_e = ElementType::Scalar(ScalarElement::U32);
    let lit = |v: u32| TileExpr::new(TileExprKind::Literal(TileLiteral::U32(v)), u32_e);
    let TileExprKind::Binary {
        op,
        left,
        right,
        numeric,
    } = rebuilt.kind()
    else {
        return rebuilt;
    };
    if rebuilt.element() != u32_e {
        return rebuilt;
    }
    let with = |kind: TileExprKind| TileExpr::new(kind, u32_e);
    let add = |a: TileExpr, c: u32| {
        if c == 0 {
            a
        } else {
            with(TileExprKind::Binary {
                op: TileBinaryOp::Add,
                left: a,
                right: lit(c),
                numeric: *numeric,
            })
        }
    };
    match op {
        TileBinaryOp::Add => {
            // Hoist every literal of the Add chain to the top as one
            // constant, so `carry_free_add` can peel it (always sound).
            fn flatten(e: &TileExpr, terms: &mut Vec<TileExpr>, c: &mut u32) {
                match e.kind() {
                    TileExprKind::Binary {
                        op: TileBinaryOp::Add,
                        left,
                        right,
                        ..
                    } => {
                        flatten(left, terms, c);
                        flatten(right, terms, c);
                    }
                    TileExprKind::Literal(TileLiteral::U32(v)) => *c = c.wrapping_add(*v),
                    _ => terms.push(e.clone()),
                }
            }
            let mut terms = Vec::new();
            let mut c = 0u32;
            flatten(left, &mut terms, &mut c);
            flatten(right, &mut terms, &mut c);
            let base = terms.into_iter().reduce(|a, b| {
                with(TileExprKind::Binary {
                    op: TileBinaryOp::Add,
                    left: a,
                    right: b,
                    numeric: *numeric,
                })
            });
            let canonical = match base {
                Some(b) => add(b, c),
                None => lit(c),
            };
            if canonical != rebuilt {
                return canonical;
            }
        }
        TileBinaryOp::Shr => {
            if let (Some((a, c)), Some(s)) = (carry_free_add(left), lit_u32(right)) {
                let shifted = with(TileExprKind::Binary {
                    op: TileBinaryOp::Shr,
                    left: a,
                    right: lit(s),
                    numeric: *numeric,
                });
                return simplify_index_cached(&add(shifted, c >> s.min(31)), memo);
            }
            // Mod-interval second chance: no carry across bit `s`.
            if let (Some((a, c)), Some(s)) = (top_literal_add(left), lit_u32(right)) {
                let s = s.min(31);
                let cap = (1u128 << s) - 1;
                if low_max(&a, s) + (u128::from(c) & cap) <= cap {
                    let shifted = with(TileExprKind::Binary {
                        op: TileBinaryOp::Shr,
                        left: a,
                        right: lit(s),
                        numeric: *numeric,
                    });
                    return simplify_index_cached(&add(shifted, c >> s), memo);
                }
            }
        }
        TileBinaryOp::BitAnd => {
            let (masked, m) = match (lit_u32(right), lit_u32(left)) {
                (Some(m), _) => (left, m),
                (_, Some(m)) => (right, m),
                _ => return rebuilt,
            };
            // The mask lies inside the known-zero low bits.
            if m >> known_zero_low_bits(masked).min(31) == 0 {
                return lit(0);
            }
            if let Some((a, c)) = carry_free_add(masked) {
                let anded = with(TileExprKind::Binary {
                    op: TileBinaryOp::BitAnd,
                    left: a,
                    right: lit(m),
                    numeric: *numeric,
                });
                return simplify_index_cached(&add(anded, c & m), memo);
            }
            // Mod-interval second chance for a low mask.
            if m < u32::MAX && (m + 1).is_power_of_two() {
                let s = (m + 1).trailing_zeros();
                if let Some((a, c)) = top_literal_add(masked)
                    && low_max(&a, s) + u128::from(c & m) <= u128::from(m)
                {
                    let anded = with(TileExprKind::Binary {
                        op: TileBinaryOp::BitAnd,
                        left: a,
                        right: lit(m),
                        numeric: *numeric,
                    });
                    return simplify_index_cached(&add(anded, c & m), memo);
                }
            }
        }
        TileBinaryOp::Div => {
            if let (Some((a, c)), Some(d)) = (carry_free_add(left), lit_u32(right)) {
                // Splits only when `d` divides the alignment `2^t`.
                if d.is_power_of_two() && u64::from(d) <= (1u64 << known_zero_low_bits(&a)) {
                    let divided = with(TileExprKind::Binary {
                        op: TileBinaryOp::Div,
                        left: a,
                        right: lit(d),
                        numeric: *numeric,
                    });
                    return simplify_index_cached(&add(divided, c / d), memo);
                }
            }
            if let (Some((a, c)), Some(d)) = (top_literal_add(left), lit_u32(right))
                && d.is_power_of_two()
                && d > 1
            {
                let s = d.trailing_zeros();
                if low_max(&a, s) + u128::from(c % d) < u128::from(d) {
                    let divided = with(TileExprKind::Binary {
                        op: TileBinaryOp::Div,
                        left: a,
                        right: lit(d),
                        numeric: *numeric,
                    });
                    return simplify_index_cached(&add(divided, c / d), memo);
                }
            }
        }
        TileBinaryOp::Rem => {
            if let (Some((a, c)), Some(d)) = (carry_free_add(left), lit_u32(right))
                && d.is_power_of_two()
                && u64::from(d) <= (1u64 << known_zero_low_bits(&a))
            {
                // `a % d = 0` outright: the alignment covers `d`.
                return lit(c % d);
            }
            if let (Some((a, c)), Some(d)) = (top_literal_add(left), lit_u32(right))
                && d.is_power_of_two()
                && d > 1
            {
                let s = d.trailing_zeros();
                if low_max(&a, s) + u128::from(c % d) < u128::from(d) {
                    let reduced = with(TileExprKind::Binary {
                        op: TileBinaryOp::Rem,
                        left: a,
                        right: lit(d),
                        numeric: *numeric,
                    });
                    return simplify_index_cached(&add(reduced, c % d), memo);
                }
            }
        }
        _ => {}
    }
    rebuilt
}
