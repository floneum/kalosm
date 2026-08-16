//! Kernel `tile` — one kernel body. Structural sharing is the hash-cons, so two
//! identical subtrees built separately merge. [`Stmt::AtomicAdd`] is supported.
//! Element type is runtime data, never a marker type.
//!
//! Kernel is produced *after* extraction and is not part of the e-graph. Barrier
//! elision and arena packing stay closed-form argmins here with an independent
//! verifier — an honest exclusion, marked as such.

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

    /// Array stride in a workgroup allocation, or `None` for elements that
    /// cannot back one. The single source of stride truth: arena packing
    /// and module emission both read this, so they cannot disagree.
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

/// Exactly two memory spaces. Nothing fusor emits needs uniform buffers,
/// push constants, textures, samplers, or (outside [`Stmt::AtomicAdd`])
/// atomics.
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

/// A storage buffer declaration. `binding` is the one externally meaningful
/// name; read-only-ness is what the derived bind group reads back out of
/// the emitted module.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferDecl {
    pub binding: u32,
    pub element: ElementType,
    pub layout: TileLayout,
    pub access: BufferAccess,
}

/// A workgroup tile declaration.
///
/// **Identity-bearing**, for the same reason [`LocalDecl`] is. Two tiles of
/// the same element, shape and name are two *allocations*, which the arena
/// may place at two different offsets and which a barrier may separate. Under
/// structural equality they were one value to the Kernel term memo, so a
/// `LoadTile`/`CoopLoad` off the second folded into the first — a lowering
/// that staged into two same-shaped buffers (double buffering, `staging: 2`)
/// read one of them twice and never touched the other.
#[derive(Clone, Debug, Eq)]
pub struct TileDecl {
    pub element: ElementType,
    pub layout: TileLayout,
    pub name: &'static str,
    id: u64,
}

thread_local! {
    /// Decl ids are unique **within a kernel build** — that is the whole
    /// contract (the Kernel term memo is per builder, the arena per kernel).
    /// They are thread-local and resettable rather than process-global so
    /// that two identical lowerings mint identical ids: the pipeline cache
    /// deduplicates compiled kernels on body identity, and a globally-unique
    /// id would make every relower of the same kernel hash differently —
    /// measured as one Metal compile per launch per decode step.
    static NEXT_DECL_ID: std::cell::Cell<u64> = const { std::cell::Cell::new(1) };
}

/// Restart decl numbering. Called at each kernel-lowering entry point so a
/// rebuilt kernel is byte-identical to its first build.
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

/// A private per-invocation local.
///
/// **Identity-bearing.** Two locals of the same element type are two
/// registers, so `id` — not `element` — is what equality and hashing key on.
/// Without it the Kernel term memo folded `LoadLocal(a)` into `LoadLocal(b)`
/// whenever they had the same type, and every kernel carrying more than one
/// same-typed accumulator (a `tn`-wide register tile, a multi-slot fold
/// carrier, a coop accumulator pair) read one register `tn` times.
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

/// Shared handle to a storage buffer. `Arc`, not `Rc`: kernel building runs
/// on worker threads.
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

/// Cross-lane reduction operators.
///
/// This is the **hardware fast path**, not the general reduction algebra: the
/// four operators a subgroup collective and a shared-memory tree can be spelled
/// with directly. Everything wider goes through [`Stmt::Reduce`]'s
/// [`MergeBody`]. `TileReduceOp` survives because the single-slot path carries
/// every fold in the system and must keep emitting byte-identical code.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum TileReduceOp {
    Sum,
    Product,
    Max,
    Min,
}

impl TileReduceOp {
    /// The binary operator this folds with — used by both loop-fold
    /// desugaring and the cross-lane tree lowerer.
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
/// N-ary [`Stmt::Reduce`] has to carry.
///
/// **One decision, in one place.** Both emitters read this, so the fast path
/// cannot drift between them and a carrier can never be silently truncated to
/// its first slot on one backend and refused on the other.
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
    /// One extent of the dispatched grid (`@builtin(num_workgroups)`).
    /// This is how a kernel linearizes its workgroup coordinate without
    /// baking the grid into its body — the body stays one compiled pipeline
    /// across every sequence length.
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

/// A quantized matrix bound as a plain u32 storage buffer.
///
/// The decode program addresses the block stream from `data` and the
/// `(k_base, col)` the load supplies; the matrix extents are not among its
/// inputs (`BlockDecodeArgs` has no row or column count), so they are not
/// carried here. A reader that needs a bound computes it where the bound is
/// used, not from the view.
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

/// Cross-lane reduction strategy. One node with the strategy as a
/// parameter, so it stays a late capability-driven choice.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReduceKind {
    Subgroup,
    Workgroup {
        scratch: Tile,
        group_size: u32,
    },
    Loop {
        iterations: u32,
        index: Local,
        scratch: Tile,
        group_size: u32,
    },
}

/// Source region of a cooperative fragment load.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CoopSrc {
    TileRegion {
        tile: Tile,
        row: TileExpr,
        col: TileExpr,
        transposed: bool,
    },
    BroadcastCol {
        src: StorageView,
        col: TileExpr,
    },
}

// ---------------------------------------------------------------------------
// Expressions
// ---------------------------------------------------------------------------

/// A hash-consed Kernel value. Structural sharing *is* the hash-cons: two
/// identical subtrees built separately merge, which pointer-keyed
/// memoization cannot do. `ty` and `hash` are cached at construction.
#[derive(Clone, Debug)]
pub struct TileExpr(Arc<TileNode>);

/// An Kernel node with its cached type, hash and memory-read set.
#[derive(Debug)]
pub struct TileNode {
    pub kind: TileExprKind,
    pub ty: ElementType,
    pub hash: u64,
    /// Which memory spaces this tree reads. See [`TileExpr::mem_reads`]; the
    /// set is folded up from the children at construction so a consumer's
    /// memo invalidation is O(1) per entry rather than a re-walk.
    pub mem_reads: MemReads,
}

/// The memory spaces a [`TileExpr`] reads.
///
/// A backend that hash-conses expressions is only sound while the memory its
/// keys read is unchanged: `LoadTile(t, i)` before a write to `t` and after
/// it are two different values that compare equal. This set is what lets an
/// emitter drop *exactly* the affected entries — a private-local store does
/// not invalidate a workgroup tile read, and a workgroup barrier does not
/// invalidate a private local read.
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
    /// True when the two sets name at least one space in common — the test a
    /// memo invalidation makes against the spaces a statement writes.
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
    /// An all-zero cooperative-matrix fragment.
    ///
    /// A `CoopMatrix` accumulator has to start somewhere, and a scalar zero is
    /// not that somewhere: `Stmt::Loop` requires `init.element() ==
    /// local.element`, so `lower_coop` initializing its C fragment with an
    /// `f32` literal failed `verify_kernel` on every device that selected the
    /// cooperative family. There is no arithmetic that produces a zero
    /// fragment from a scalar, so it is a leaf.
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

impl TileExpr {
    pub fn new(kind: TileExprKind, ty: ElementType) -> Self {
        let mut h = FxHasher::default();
        kind.hash(&mut h);
        ty.hash(&mut h);
        let mem_reads = kind_mem_reads(&kind);
        Self(Arc::new(TileNode {
            kind,
            ty,
            hash: h.finish(),
            mem_reads,
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

    /// This node's identity, as the address of its shared allocation.
    ///
    /// A body is a **DAG**, not a tree: the builder hands the same
    /// `TileExpr` to every consumer of a value, so a node is reached once
    /// per edge into it and a naive recursive walk is exponential in the
    /// sharing depth. A walk memoizes on this — exactly, unlike
    /// [`Self::structural_hash`], and for free — and `PartialEq` is the
    /// same test, so two pointers agree iff the subtrees are the same
    /// object.
    pub fn node_ptr(&self) -> usize {
        Arc::as_ptr(&self.0) as *const () as usize
    }
    /// A statically-true mask, which the lowerer skips codegen for.
    pub fn is_constant_true(&self) -> bool {
        matches!(&self.0.kind, TileExprKind::Literal(TileLiteral::Bool(true)))
    }

    /// Which memory spaces this tree reads, anywhere inside it — i.e. what
    /// its value is a function of besides its own operands.
    ///
    /// A backend that hash-conses expressions must drop exactly the entries
    /// whose set intersects the spaces a statement writes (or a barrier makes
    /// another invocation's writes visible in). The set is folded up at
    /// construction, so the test is a field read.
    pub fn mem_reads(&self) -> MemReads {
        self.0.mem_reads
    }

}

/// Fold the memory-read set for one node from its children.
///
/// Exhaustive on purpose: a new `TileExprKind` must state which spaces it
/// reads rather than inherit [`MemReads::NONE`] from a wildcard and silently
/// join the pure half of a backend memo.
fn kind_mem_reads(kind: &TileExprKind) -> MemReads {
    use TileExprKind as K;
    let addr = |a: &Addr| match a {
        Addr::Linear(e) => e.mem_reads(),
        Addr::Rc2 { row, col } => row.mem_reads().union(col.mem_reads()),
    };
    match kind {
        // Pure leaves.
        K::Literal(_) | K::Builtin(_) | K::CoopZero { .. } => MemReads::NONE,
        // Reads, each unioned with whatever its address and predicate read.
        K::LoadLocal(_) => MemReads::LOCAL,
        K::Load {
            src,
            addr: a,
            mask,
            fill,
        } => {
            // Both `Source` arms are storage buffers; a quantized view is a
            // u32 buffer plus a decode program.
            let _ = src;
            MemReads::STORAGE
                .union(addr(a))
                .union(mask.mem_reads())
                .union(fill.mem_reads())
        }
        K::LoadTile { index, .. } => MemReads::TILE.union(index.mem_reads()),
        K::CoopLoad { src, .. } => match &**src {
            CoopSrc::TileRegion { row, col, .. } => MemReads::TILE
                .union(row.mem_reads())
                .union(col.mem_reads()),
            CoopSrc::BroadcastCol { col, .. } => MemReads::STORAGE.union(col.mem_reads()),
        },
        // Pure combinators: the union over the children.
        K::Unary { value, .. }
        | K::Round { value, .. }
        | K::Cast { value, .. }
        | K::Bitcast { value, .. }
        | K::VecComponent { vector: value, .. } => value.mem_reads(),
        // A cross-lane reduction stages through the scratch tile its
        // `ReduceKind` names, so it reads a workgroup tile on every strategy
        // but `Subgroup`.
        K::Reduce { kind, value, .. } => match &**kind {
            ReduceKind::Subgroup => value.mem_reads(),
            ReduceKind::Workgroup { .. } => value.mem_reads().union(MemReads::TILE),
            ReduceKind::Loop { .. } => value
                .mem_reads()
                .union(MemReads::TILE)
                .union(MemReads::LOCAL),
        },
        K::Binary { left, right, .. } | K::Compare { left, right, .. } | K::Dot { left, right } => {
            left.mem_reads().union(right.mem_reads())
        }
        K::Select {
            condition,
            accept,
            reject,
        } => condition
            .mem_reads()
            .union(accept.mem_reads())
            .union(reject.mem_reads()),
        K::Vec { parts, .. } => parts
            .iter()
            .fold(MemReads::NONE, |acc, p| acc.union(p.mem_reads())),
        K::CoopMma { a, b, c } => a.mem_reads().union(b.mem_reads()).union(c.mem_reads()),
    }
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

/// The merge of two partial accumulators, one expression per **lane**.
///
/// A lane is one scalar accumulator: a `SlotTy::Scalar` slot is one lane and a
/// `SlotTy::Vector(d)` slot is `d` of them, so `body.len()` is the carrier's
/// `lanes()`, never its `width()`.
///
/// **Cross-lane reads are required, not forbidden.** `body[1]` may read
/// `lhs[0]`: flash's running sum and its output accumulator both read the
/// running max. What `verify_kernel` rejects is a read of anything *outside*
/// `lhs`/`rhs` — a merge that reads a `Tile`, a `Builtin` or a lane id is not a
/// merge, and a per-lane-independent merge would be the wrong abstraction.
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
    /// Arity agreement across the three vectors — the clause that makes the
    /// `accs[0]` bug unrepresentable.
    pub fn is_arity_consistent(&self) -> bool {
        self.lhs.len() == self.body.len() && self.rhs.len() == self.body.len()
    }
}

/// One ordered Kernel statement. `FillTile` is not sugar: it is the only form
/// whose vectorized and guard-free variants the lowerer can select.
/// `CoopStore` is subgroup-collective, never a per-lane store;
/// `CoopStoreTile` is the staging step attention needs between fragment
/// math and a per-lane softmax over the same values.
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
    /// The **N-ary cross-lane reduction**, beside [`TileExprKind::Reduce`] and
    /// not in place of it.
    ///
    /// `values` is one partial per accumulator lane and `outs` one `Local` per
    /// lane; `merge` folds two partials. There is no single `TileReduceOp` to
    /// resolve for the whole fold, so `Fold{(max, sum)}` cannot compute `max(x)`
    /// and discard the sum: there is nowhere to discard it to.
    ///
    /// `fast` is set by the canonical constructor **iff** `values.len() == 1`
    /// and `merge.body[0]` is exactly `binary(op.binary(), load(lhs[0]),
    /// load(rhs[0]))`. It is computed, never author-supplied, so it cannot drift
    /// from `merge`; both emitters open their arm with it and take the existing
    /// collective path unchanged.
    ///
    /// `scratch` holds one workgroup tile per lane for the `Workgroup`/`Loop`
    /// kinds and is empty for `Subgroup`. `kind`'s own scratch is `scratch[0]`,
    /// so a one-lane reduction is exactly the node it is today.
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
    /// The memory spaces this statement makes stale for a reader.
    ///
    /// Either because it writes them, or — for the two barriers — because it
    /// makes *another invocation's* writes to them visible. A backend that
    /// hash-conses expressions must retire every memoized value whose
    /// [`TileExpr::mem_reads`] intersects this set once the statement is
    /// emitted; see `fusor2-gpu`'s `Emitter::invalidate_mem`.
    ///
    /// `If` and `Loop` name nothing themselves: their bodies are emitted
    /// statement by statement and each names its own.
    pub fn writes(&self) -> MemReads {
        match self {
            Self::Store { .. } | Self::AtomicAdd { .. } | Self::CoopStore { .. } => {
                MemReads::STORAGE
            }
            Self::StoreLocal { .. } => MemReads::LOCAL,
            Self::StoreTile { .. } | Self::FillTile { .. } | Self::CoopStoreTile { .. } => {
                MemReads::TILE
            }
            // The scratch tiles it stages through and the `outs` locals it
            // lands in. The one-lane `fast` path only writes the local, but
            // naming the tile as well costs a re-emit and never a wrong value.
            Self::Reduce { .. } => MemReads::TILE.union(MemReads::LOCAL),
            // A barrier publishes other invocations' writes. Conservatively
            // both shared spaces: `Stmt::Barrier` is emitted to order
            // workgroup staging, but nothing stops a lowering from using it
            // to order storage traffic inside one workgroup, and a private
            // local is never another invocation's to write.
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

/// The result of workgroup-arena planning. `arena_plan` is a **pure
/// memoized function** of `(geom, dtype, caps)` and the *same* function
/// `verify_launch` admits against and the Kernel emitter lays out with. `total_bytes`
/// feeds both the footprint check and the occupancy term.
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
/// one implementation lives in `fusor2-tile`.
pub trait ArenaPlanner: Send + Sync {
    /// Pack under `caps`, taking the argmin of `total_bytes` over
    /// `{Regions, ByteArena} x {no barrier, top-3 insertions}`. Memoized.
    fn arena_plan(&self, ir: &KernelIr, caps: &crate::device::Caps) -> Result<ArenaPlan>;

    /// Workgroup bytes a candidate geometry needs, without building the
    /// body — the exact value `verify_launch` admits against.
    fn workgroup_bytes(&self, tiles: &Tiles, caps: &crate::device::Caps) -> Result<u32>;

    fn barrier_suggestions(&self, ir: &KernelIr) -> Vec<BarrierSuggestion>;

    /// Independent all-pairs recheck: every byte-overlapping tile pair must
    /// be separated by a *guaranteed uniform* barrier. Fails lowering
    /// rather than racing.
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

/// Per-target lowering of one [`crate::ir::OpDef`] into Kernel.
pub type LowerFn = fn(&crate::ir::Node, &crate::ir::launch::SchedPoint) -> Result<KernelIr>;

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

#[cfg(test)]
mod local_identity_tests {
    use super::*;

    /// Two locals of the same element type are two registers. Before `id` they
    /// were `==`, so the Kernel term memo folded their `LoadLocal`s together and a
    /// kernel carrying `tn` same-typed accumulators read one register `tn`
    /// times — which is what made every register-tiled GEMM column identical.
    #[test]
    fn two_same_typed_locals_are_two_registers() {
        let f32e = ElementType::Scalar(ScalarElement::F32);
        let a: Local = Arc::new(LocalDecl::new(f32e));
        let b: Local = Arc::new(LocalDecl::new(f32e));
        assert_ne!(a, b);
        let ra = TileExpr::new(TileExprKind::LoadLocal(Arc::clone(&a)), f32e);
        let rb = TileExpr::new(TileExprKind::LoadLocal(Arc::clone(&b)), f32e);
        assert_ne!(ra, rb, "two accumulators must not hash-cons together");
        assert_ne!(ra.structural_hash(), rb.structural_hash());
        // Reading the *same* local twice still shares.
        let ra2 = TileExpr::new(TileExprKind::LoadLocal(a), f32e);
        assert_eq!(ra, ra2);
    }
}

// ---------------------------------------------------------------------------
// Aligned-window index algebra
// ---------------------------------------------------------------------------

/// Largest `t` such that `e` is provably a multiple of `2^t` — the number of
/// low bits known to be zero. Conservative: `0` whenever nothing is known.
fn known_zero_low_bits(e: &TileExpr) -> u32 {
    match e.kind() {
        TileExprKind::Literal(TileLiteral::U32(v)) => {
            if *v == 0 { 31 } else { v.trailing_zeros().min(31) }
        }
        TileExprKind::Binary {
            op, left, right, ..
        } => match op {
            TileBinaryOp::Mul => {
                (known_zero_low_bits(left) + known_zero_low_bits(right)).min(31)
            }
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
                        if m == 0 { 31 } else { m.trailing_zeros().min(31) }
                    }
                    _ => 0,
                };
                known_zero_low_bits(left)
                    .max(known_zero_low_bits(right))
                    .max(from_mask)
            }
            TileBinaryOp::BitOr => {
                known_zero_low_bits(left).min(known_zero_low_bits(right))
            }
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

/// Upper bound on `e mod 2^s` (`s <= 32`), computed structurally.
///
/// Where [`known_zero_low_bits`] only sees power-of-two alignment, this is a
/// mod-interval: `Rem(x, 8) * 4` has just two zero low bits, but its residue
/// mod 256 is at most 28, and `Div(x, 8) * 128` contributes at most 128 —
/// so a base built from both provably cannot carry into bit 8 under any
/// window offset below 100. That is exactly the fact a *split* lane window
/// needs: its runs add offsets that are far beyond the base's alignment yet
/// still provably carry-free at the cut the shift or mask cares about.
///
/// Sound under wrapping u32 arithmetic because reduction mod `2^s` is a ring
/// homomorphism of the wrapping ops (`Add`/`Mul`/`Shl`), and every other arm
/// either bounds the value globally (`Rem`, `Div`, `BitAnd` by a literal) or
/// gives up with the cap.
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
            TileBinaryOp::BitOr | TileBinaryOp::BitXor => {
                low_max(left, s) + low_max(right, s)
            }
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

/// `Some((base, c))` when `e` is a top-level `base + c` with a literal `c` —
/// no alignment claim, unlike [`carry_free_add`]; the mod-interval laws
/// establish carry-freedom themselves at whatever cut they need.
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
/// side's known zero low bits. The two halves then occupy disjoint bits, so
/// every shift, mask and exact division distributes over them.
///
/// A literal that straddles the alignment boundary still splits: its
/// `t`-aligned high part folds into the base (the sum of two `t`-aligned
/// values is `t`-aligned) and only the low part is peeled. `c_lo == 0`
/// peels nothing and returns `None`, which is also what keeps the callers'
/// rebuild-then-resimplify recursion terminating.
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

/// Rewrite the index arithmetic of `e` under aligned-window algebra.
///
/// The one law: when `a` has `t` known-zero low bits and `c < 2^t`, the sum
/// `a + c` is `a | c` — no carry anywhere — so
///
/// - `(a + c) >> s`  =  `(a >> s) + (c >> s)`
/// - `(a + c) &  m`  =  `(a & m) + (c & m)`
/// - `(a + c) /  d`  =  `a / d`   and   `(a + c) % d  =  (a % d) + c`
///   whenever additionally `2^t` is a multiple of `d`'s power-of-two part
///   covering `c` — kept to the `c < d`, `d | 2^t`-compatible case below.
///
/// This is not a peephole for any data format. It is what makes the address
/// expressions of *consecutive elements of an aligned window* structurally
/// equal wherever they agree — `(base + 0) >> 3` and `(base + 7) >> 3`
/// become the same node — so the emitter's structural memo shares the loads
/// between them. A packed-word decode (GGUF blocks, f16 pairs, bit fields)
/// collapses to one word load per window as a *consequence*; the rewrite
/// itself never heard of any of them.
pub fn simplify_index(e: &TileExpr) -> TileExpr {
    let rebuilt = match e.kind() {
        TileExprKind::Binary {
            op,
            left,
            right,
            numeric,
        } => {
            let l = simplify_index(left);
            let r = simplify_index(right);
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
                value: simplify_index(value),
                numeric: *numeric,
            },
            e.element(),
        ),
        TileExprKind::Compare { op, left, right } => TileExpr::new(
            TileExprKind::Compare {
                op: *op,
                left: simplify_index(left),
                right: simplify_index(right),
            },
            e.element(),
        ),
        TileExprKind::Cast { value, to } => TileExpr::new(
            TileExprKind::Cast {
                value: simplify_index(value),
                to: *to,
            },
            e.element(),
        ),
        TileExprKind::Bitcast { value, to } => TileExpr::new(
            TileExprKind::Bitcast {
                value: simplify_index(value),
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
                condition: simplify_index(condition),
                accept: simplify_index(accept),
                reject: simplify_index(reject),
            },
            e.element(),
        ),
        TileExprKind::Round { mode, value } => TileExpr::new(
            TileExprKind::Round {
                mode: *mode,
                value: simplify_index(value),
            },
            e.element(),
        ),
        // A missing container arm silently fences the rewrite out of the
        // whole subtree: the f16 scale decode rides `VecComponent(Unpack(..))`
        // and skipping it left every element of a window recomputing its own
        // block base — and re-loading the scale words the memo should share.
        TileExprKind::Vec {
            scalar,
            lanes,
            parts,
        } => TileExpr::new(
            TileExprKind::Vec {
                scalar: *scalar,
                lanes: *lanes,
                parts: parts.iter().map(simplify_index).collect(),
            },
            e.element(),
        ),
        TileExprKind::VecComponent { vector, component } => TileExpr::new(
            TileExprKind::VecComponent {
                vector: simplify_index(vector),
                component: *component,
            },
            e.element(),
        ),
        TileExprKind::Dot { left, right } => TileExpr::new(
            TileExprKind::Dot {
                left: simplify_index(left),
                right: simplify_index(right),
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
                Addr::Linear(i) => Addr::Linear(simplify_index(i)),
                Addr::Rc2 { row, col } => Addr::Rc2 {
                    row: simplify_index(row),
                    col: simplify_index(col),
                },
            };
            TileExpr::new(
                TileExprKind::Load {
                    src: src.clone(),
                    addr: Box::new(addr),
                    mask: simplify_index(mask),
                    fill: simplify_index(fill),
                },
                e.element(),
            )
        }
        TileExprKind::LoadTile { tile, index } => TileExpr::new(
            TileExprKind::LoadTile {
                tile: tile.clone(),
                index: simplify_index(index),
            },
            e.element(),
        ),
        // Leaves and forms no index expression rides through.
        _ => return e.clone(),
    };

    let u32_e = ElementType::Scalar(ScalarElement::U32);
    let lit = |v: u32| TileExpr::new(TileExprKind::Literal(TileLiteral::U32(v)), u32_e);
    let TileExprKind::Binary {
        op, left, right, numeric,
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
            // Literal-to-top normalization: flatten the whole Add chain,
            // fold every literal into one constant, and re-emit as
            // `(sum of non-literal terms) + c`. Wrapping u32 addition is
            // associative and commutative, so this is unconditionally
            // sound; its point is that `carry_free_add` only matches a
            // literal that is an immediate operand of the *top-level* Add,
            // and lowering builds sums in whatever order the address math
            // arrived. Without this, `(base + (k + v)) + col*stride` never
            // splits its window literal `v` off, and consecutive elements
            // of an aligned window never share their word loads.
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
                return simplify_index(&add(shifted, c >> s.min(31)));
            }
            // Mod-interval second chance: the literal is far beyond the
            // base's alignment (a split window's run offset), but the base's
            // residue below the cut is bounded and the sum still cannot
            // carry across bit `s` — so the shift distributes anyway.
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
                    return simplify_index(&add(shifted, c >> s));
                }
            }
        }
        TileBinaryOp::BitAnd => {
            let (masked, m) = match (lit_u32(right), lit_u32(left)) {
                (Some(m), _) => (left, m),
                (_, Some(m)) => (right, m),
                _ => return rebuilt,
            };
            // Every set bit of the mask sits below the value's known-zero
            // low bits: the AND is identically zero.
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
                return simplify_index(&add(anded, c & m));
            }
            // Mod-interval second chance for a low mask: no carry across
            // the mask's top, so the AND distributes over the sum.
            if m < u32::MAX && (m + 1).is_power_of_two() {
                let s = (m + 1).trailing_zeros();
                if let Some((a, c)) = top_literal_add(masked) {
                    if low_max(&a, s) + u128::from(c & m) <= u128::from(m) {
                        let anded = with(TileExprKind::Binary {
                            op: TileBinaryOp::BitAnd,
                            left: a,
                            right: lit(m),
                            numeric: *numeric,
                        });
                        return simplify_index(&add(anded, c & m));
                    }
                }
            }
        }
        TileBinaryOp::Div => {
            if let (Some((a, c)), Some(d)) = (carry_free_add(left), lit_u32(right)) {
                // Carry-freedom gives `a`'s low `t` bits zero and `c < 2^t`;
                // splitting a division needs `d` to divide the alignment,
                // i.e. `d` a power of two no larger than `2^t`.
                if d.is_power_of_two() && u64::from(d) <= (1u64 << known_zero_low_bits(&a)) {
                    let divided = with(TileExprKind::Binary {
                        op: TileBinaryOp::Div,
                        left: a,
                        right: lit(d),
                        numeric: *numeric,
                    });
                    return simplify_index(&add(divided, c / d));
                }
            }
            if let (Some((a, c)), Some(d)) = (top_literal_add(left), lit_u32(right)) {
                if d.is_power_of_two() && d > 1 {
                    let s = d.trailing_zeros();
                    if low_max(&a, s) + u128::from(c % d) < u128::from(d) {
                        let divided = with(TileExprKind::Binary {
                            op: TileBinaryOp::Div,
                            left: a,
                            right: lit(d),
                            numeric: *numeric,
                        });
                        return simplify_index(&add(divided, c / d));
                    }
                }
            }
        }
        TileBinaryOp::Rem => {
            if let (Some((a, c)), Some(d)) = (carry_free_add(left), lit_u32(right)) {
                if d.is_power_of_two() && u64::from(d) <= (1u64 << known_zero_low_bits(&a)) {
                    // `a % d = 0` outright: the alignment covers `d`.
                    return lit(c % d);
                }
            }
            if let (Some((a, c)), Some(d)) = (top_literal_add(left), lit_u32(right)) {
                if d.is_power_of_two() && d > 1 {
                    let s = d.trailing_zeros();
                    if low_max(&a, s) + u128::from(c % d) < u128::from(d) {
                        let reduced = with(TileExprKind::Binary {
                            op: TileBinaryOp::Rem,
                            left: a,
                            right: lit(d),
                            numeric: *numeric,
                        });
                        return simplify_index(&add(reduced, c % d));
                    }
                }
            }
        }
        _ => {}
    }
    rebuilt
}

#[cfg(test)]
mod mod_interval_tests {
    use super::*;

    const U32E: ElementType = ElementType::Scalar(ScalarElement::U32);

    fn lit(v: u32) -> TileExpr {
        TileExpr::new(TileExprKind::Literal(TileLiteral::U32(v)), U32E)
    }
    fn bin(op: TileBinaryOp, l: TileExpr, r: TileExpr) -> TileExpr {
        TileExpr::new(
            TileExprKind::Binary {
                op,
                left: l,
                right: r,
                numeric: NumericContract::RELAXED,
            },
            U32E,
        )
    }
    /// The split-window lane base: `(lane / 8) * 128 + (lane % 8) * 4`.
    /// Its residue mod 256 is at most 128 + 28 = 156 — a fact alignment
    /// alone cannot see (only two zero low bits).
    fn split_base() -> TileExpr {
        let lane = TileExpr::new(TileExprKind::Builtin(Builtin::SubgroupLane), U32E);
        let hi = bin(
            TileBinaryOp::Mul,
            bin(TileBinaryOp::Div, lane.clone(), lit(8)),
            lit(128),
        );
        let lo = bin(TileBinaryOp::Mul, bin(TileBinaryOp::Rem, lane, lit(8)), lit(4));
        bin(TileBinaryOp::Add, hi, lo)
    }

    #[test]
    fn run_offsets_share_the_block_index() {
        // `(X + c) >> 8` for every run offset of a (parts 4, gap 32) window
        // folds to `X >> 8`: the same quant block, so one metadata load.
        let x = split_base();
        let plain = simplify_index(&bin(TileBinaryOp::Shr, x.clone(), lit(8)));
        for c in [32u32, 64, 96, 99] {
            let shifted = bin(
                TileBinaryOp::Shr,
                bin(TileBinaryOp::Add, x.clone(), lit(c)),
                lit(8),
            );
            assert_eq!(simplify_index(&shifted), plain, "offset {c}");
        }
    }

    #[test]
    fn a_carrying_offset_does_not_fold() {
        // 156 + 200 crosses bit 8 for some lanes: folding would be a
        // wrong-answer rewrite, so it must not happen.
        let x = split_base();
        let plain = simplify_index(&bin(TileBinaryOp::Shr, x.clone(), lit(8)));
        let shifted = bin(
            TileBinaryOp::Shr,
            bin(TileBinaryOp::Add, x, lit(200)),
            lit(8),
        );
        assert_ne!(simplify_index(&shifted), plain);
    }

    #[test]
    fn paired_offsets_agree_under_mask_and_shift() {
        // `((X + 64) & 255) >> 6` and `((X + 96) & 255) >> 6` are the same
        // 64-element group of the same block — the fact that lets one data
        // word serve both nibble offsets of a split window.
        let x = split_base();
        let at = |c: u32| {
            let masked = bin(
                TileBinaryOp::BitAnd,
                bin(TileBinaryOp::Add, x.clone(), lit(c)),
                lit(255),
            );
            simplify_index(&bin(TileBinaryOp::Shr, masked, lit(6)))
        };
        assert_eq!(at(64), at(96));
        assert_ne!(at(0), at(64));
    }

    #[test]
    fn div_and_rem_split_like_the_shift() {
        let x = split_base();
        let plain = simplify_index(&bin(TileBinaryOp::Div, x.clone(), lit(256)));
        let divided = bin(
            TileBinaryOp::Div,
            bin(TileBinaryOp::Add, x.clone(), lit(96)),
            lit(256),
        );
        assert_eq!(simplify_index(&divided), plain);
        // `(X + 96) % 32 = X % 32`: the offset is a whole number of gaps.
        let rem_plain = simplify_index(&bin(TileBinaryOp::Rem, x.clone(), lit(32)));
        let rem = bin(
            TileBinaryOp::Rem,
            bin(TileBinaryOp::Add, x, lit(96)),
            lit(32),
        );
        assert_eq!(simplify_index(&rem), rem_plain);
    }
}
