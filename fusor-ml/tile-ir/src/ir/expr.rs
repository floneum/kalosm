use std::hash::{Hash, Hasher};
use std::rc::Rc;

use rustc_hash::FxHasher;

use crate::quantized::QuantizedMatrix;

use super::{
    ElementType, Local, ScalarElement, StorageView, Tile, TileBinaryOp, TileCompareOp, TileLiteral,
    TileReduceOp, TileUnaryOp, WorkgroupAxis,
};

/// Built-in u32 quantities that show up as leaves in index/address arithmetic.
/// Promoted to `ExprKind::Builtin` so a single expression type can host both
/// per-lane data and indexing math.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Builtin {
    /// `@builtin(local_invocation_index)` — flat lane within the workgroup.
    Lane,
    /// `@builtin(workgroup_id).{x|y|z}`.
    ProgramId(WorkgroupAxis),
    /// `@builtin(subgroup_id)`.
    SubgroupId,
    /// `@builtin(subgroup_invocation_id)` — lane within the subgroup.
    SubgroupLane,
    /// `@builtin(subgroup_size)` — runtime subgroup size.
    SubgroupSize,
    /// `@builtin(num_subgroups)` — number of subgroups per workgroup.
    NumSubgroups,
}

/// Source of an `ExprKind::Load`. The lowerer dispatches on the variant to
/// choose between a raw storage read and a dequantized read.
#[derive(Clone, Debug)]
pub enum Source {
    /// Dense storage source.
    Storage(StorageView),
    /// Quantized matrix source — dequantizes on the fly, result is `f32`.
    Quantized(QuantizedMatrix),
}

/// Address of a memory access. `Linear` is a flat rank-1 index; `Rc2` is a
/// rank-2 (row, col) coordinate.
#[derive(Clone, Debug)]
pub enum Addr {
    /// Flat rank-1 linear index.
    Linear(Box<Expr>),
    /// Rank-2 (row, col) coordinate.
    Rc2 {
        /// Row coordinate.
        row: Box<Expr>,
        /// Column coordinate.
        col: Box<Expr>,
    },
}

/// How an `ExprKind::QuantizedDot`'s f32 activations combine with the weights.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum QuantActivation {
    /// Decode the weights to f32 and dot against the f32 activations.
    F32,
    /// Pack the activations to int8 and DP4a-dot the still-quantized weights.
    Q8,
}

/// Source region for an `ExprKind::CoopLoad`.
#[derive(Clone, Debug)]
pub enum CoopSrc {
    /// A region of a workgroup tile, addressed by `(row, col)`.
    TileRegion {
        /// Source workgroup tile.
        tile: Tile,
        /// Row coordinate of the fragment origin.
        row: Box<Expr>,
        /// Column coordinate of the fragment origin.
        col: Box<Expr>,
    },
    /// A rank-1 storage vector broadcast across all fragment rows.
    BroadcastCol {
        /// Source storage view.
        src: StorageView,
        /// Column coordinate to broadcast.
        col: Box<Expr>,
    },
}

/// Selects the reduction strategy for an `ExprKind::Reduce`.
#[derive(Clone, Debug)]
pub enum ReduceKind {
    /// Reduction across the lanes of one subgroup
    /// (`subgroupAdd`/`subgroupMax`/...). No shared-memory tree, no
    /// workgroup-shape divisibility constraint.
    Subgroup,
    /// Cross-lane shared-memory tree over `scratch`. `group_size` is the
    /// contiguous-lane group reduced together.
    Workgroup {
        /// Shared-memory scratch tile.
        scratch: Tile,
        /// Contiguous-lane group reduced together.
        group_size: u32,
    },
    /// Per-lane accumulation across `iterations` loop iterations into `index`,
    /// then the cross-lane tree over `scratch`.
    Loop {
        /// Number of per-lane accumulation iterations.
        iterations: u32,
        /// U32 local carrying the current iteration.
        index: Local,
        /// Shared-memory scratch tile.
        scratch: Tile,
        /// Contiguous-lane group reduced together.
        group_size: u32,
    },
}

/// A cached value-tree node. `ty` AND `hash` are computed at construction
/// (`hash` bottom-up via `FxHasher`), so element queries and structural
/// hashing are O(1).
#[derive(Debug)]
pub struct Node {
    /// The node operator and operands.
    pub kind: ExprKind,
    /// Element type produced by this node.
    pub ty: ElementType,
    /// Cached bottom-up structural hash.
    pub hash: u64,
}

/// A rank-1-per-lane value, an `Rc`-shared handle into the value tree. `Clone`
/// is an `Rc` refcount bump.
#[derive(Clone, Debug)]
pub struct Expr(Rc<Node>);

/// The value tree. Every removed op (the coop op surface, the over-fused
/// `QuantizedDot`/`PackedActivations`/`DotK`, the marker/const-generic surface)
/// is recovered by composition.
#[derive(Clone, Debug)]
pub enum ExprKind {
    // ---- leaves ----
    /// A typed scalar literal.
    Literal(TileLiteral),
    /// A built-in u32 quantity (lane id, program id, subgroup builtins).
    Builtin(Builtin),
    /// Load from a private per-invocation local.
    LoadLocal(Local),
    // ---- memory ----
    /// Masked storage/quantized load (dense or quant, rank-1 or rank-2).
    Load {
        /// Dense or quantized source.
        src: Source,
        /// Linear or rank-2 address.
        addr: Addr,
        /// Per-lane mask.
        mask: Box<Expr>,
        /// Masked-out fill value.
        fill: Box<Expr>,
    },
    /// Load from a workgroup tile at a dynamic flat index.
    LoadTile {
        /// Source tile.
        tile: Tile,
        /// Flat index into the tile.
        index: Box<Expr>,
    },
    // ---- ALU ----
    /// Unary op.
    Unary {
        /// Operator.
        op: TileUnaryOp,
        /// Operand.
        value: Box<Expr>,
    },
    /// Binary op.
    Binary {
        /// Operator.
        op: TileBinaryOp,
        /// Left operand.
        left: Box<Expr>,
        /// Right operand.
        right: Box<Expr>,
    },
    /// Comparison op. Returns `Bool`.
    Compare {
        /// Operator.
        op: TileCompareOp,
        /// Left operand.
        left: Box<Expr>,
        /// Right operand.
        right: Box<Expr>,
    },
    /// Numeric cast.
    Cast {
        /// Value being cast.
        value: Box<Expr>,
        /// Target element type.
        to: ElementType,
    },
    /// Reinterpreting bitcast.
    Bitcast {
        /// Value being reinterpreted.
        value: Box<Expr>,
        /// Target element type.
        to: ElementType,
    },
    /// Per-lane select.
    Select {
        /// Bool condition.
        condition: Box<Expr>,
        /// Value when true.
        accept: Box<Expr>,
        /// Value when false.
        reject: Box<Expr>,
    },
    /// Compose scalar values into a vector value.
    Vec {
        /// Component scalar type.
        scalar: ScalarElement,
        /// Vector lane count.
        lanes: u32,
        /// Component expressions.
        parts: Vec<Expr>,
    },
    /// Dot product between two vector values.
    Dot {
        /// Left operand.
        left: Box<Expr>,
        /// Right operand.
        right: Box<Expr>,
    },
    // ---- reductions ----
    /// Cross-lane reduction; `kind` selects subgroup / tree / loop-then-tree.
    Reduce {
        /// Reduction operator.
        op: TileReduceOp,
        /// Reduction strategy.
        kind: ReduceKind,
        /// Reduced value.
        value: Box<Expr>,
    },
    // ---- cooperative matrix (value-producing) ----
    /// Cooperatively load a matrix fragment.
    CoopLoad {
        /// Operand role.
        role: super::CoopMatrixRole,
        /// Component scalar type.
        scalar: ScalarElement,
        /// Fragment rows.
        rows: u32,
        /// Fragment cols.
        cols: u32,
        /// Load source.
        src: CoopSrc,
    },
    /// `a * b + c` over cooperative fragments — value-producing.
    CoopMma {
        /// Left fragment.
        a: Box<Expr>,
        /// Right fragment.
        b: Box<Expr>,
        /// Accumulator fragment.
        c: Box<Expr>,
    },
    // ---- composable quant primitive ----
    /// Dequantize a block to `lanes` f32 values. `lanes` carries the caller's
    /// `values_per_lane` tiling choice as array width. Wrap in `Shared` and
    /// project per-lane with `LaneOf` to share one emission across all lanes.
    Dequantize {
        /// Quantized source matrix.
        src: QuantizedMatrix,
        /// Base K coordinate.
        k_base: Box<Expr>,
        /// Column coordinate.
        col: Box<Expr>,
        /// Per-lane mask.
        mask: Box<Expr>,
        /// Masked-out fill value.
        fill: Box<Expr>,
        /// Number of f32 lanes produced.
        lanes: u32,
    },
    /// Project lane `lane` out of a `Dequantize` (usually wrapped in `Shared`).
    LaneOf {
        /// The block being projected (a `Dequantize`, typically `Shared`).
        block: Box<Expr>,
        /// Lane index.
        lane: u32,
    },
    /// Per-column fused dot of f32 `activations` against a quantized-matrix
    /// block. The block scale is decoded **once** and the dot accumulated
    /// directly, so this is strictly more compact than `Dequantize` (decode N
    /// lanes to f32 tiles) + `dot4_sum` — which re-decodes the block scale per
    /// lane. The `packing` selects how activations meet the weights:
    /// [`QuantActivation::F32`] decodes the weights to f32 and dots; `Q8` keeps
    /// the weights quantized and DP4a-dots int8-packed activations (the path
    /// `Dequantize` + `Dot` cannot express). Both are irreducible format fast
    /// paths selected by the lowerer below the boundary. Value-producing (f32).
    QuantizedDot {
        /// Quantized source matrix; its format + `packing` select the helper.
        src: QuantizedMatrix,
        /// How the activations are combined with the weights.
        packing: QuantActivation,
        /// f32 activations (the lowerer packs them to int8 when `packing == Q8`).
        activations: Vec<Expr>,
        /// Base K coordinate.
        k_base: Box<Expr>,
        /// Column coordinate.
        col: Box<Expr>,
        /// Per-lane mask.
        mask: Box<Expr>,
        /// Masked-out fill value.
        fill: Box<Expr>,
    },
    /// Structural sharing => emit-once (covers coop fragments AND cross-lane
    /// dequant). The lowerer memoizes on `Rc::as_ptr` of the inner node.
    Shared(Expr),
}

impl Expr {
    /// Build a node, computing its `ty` and bottom-up `hash` once.
    pub(crate) fn new(kind: ExprKind, ty: ElementType) -> Self {
        let hash = hash_kind(&kind);
        Expr(Rc::new(Node { kind, ty, hash }))
    }

    /// Element type of this expression. O(1) — read from the cached `ty`.
    pub fn element(&self) -> ElementType {
        self.0.ty
    }

    /// Bottom-up structural hash. O(1) — no DAG re-walk.
    pub fn structural_hash(&self) -> u64 {
        self.0.hash
    }

    /// The node operator and operands.
    pub fn kind(&self) -> &ExprKind {
        &self.0.kind
    }

    /// Borrow the underlying `Node` (kind + cached ty + cached hash).
    pub fn node(&self) -> &Node {
        &self.0
    }

    /// Identity pointer of the underlying node — the key the lowerer uses to
    /// memoize `Shared`/`Dequantize` emissions.
    pub fn as_ptr(&self) -> *const Node {
        Rc::as_ptr(&self.0)
    }

    /// Recognize a Bool-typed expression that is statically `true`. Used by the
    /// lowerer to skip mask codegen entirely for unconditional masks.
    pub fn is_constant_true(&self) -> bool {
        match &self.0.kind {
            ExprKind::Literal(TileLiteral::Bool(true)) => true,
            ExprKind::Binary {
                op: TileBinaryOp::LogicalAnd,
                left,
                right,
            } => left.is_constant_true() && right.is_constant_true(),
            _ => false,
        }
    }
}

// ---- equality / hashing ----
//
// `PartialEq` compares the cached hash first and only deep-compares on a
// collision, keeping `Eq` semantics for the kernel cache key while staying
// cheap. `Hash` writes the cached `u64`.

impl PartialEq for Expr {
    fn eq(&self, other: &Self) -> bool {
        if Rc::ptr_eq(&self.0, &other.0) {
            return true;
        }
        self.0.hash == other.0.hash && kind_eq(&self.0.kind, &other.0.kind)
    }
}

impl Eq for Expr {}

impl Hash for Expr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.0.hash);
    }
}

/// Bottom-up `FxHasher` hash of a node's kind. Child `Expr`s contribute their
/// cached `structural_hash()` (so this is O(1) per node, O(n) per tree build);
/// decl handles contribute their `Rc::as_ptr` identity.
fn hash_kind(kind: &ExprKind) -> u64 {
    let mut h = FxHasher::default();
    hash_kind_into(kind, &mut h);
    h.finish()
}

fn hash_expr(expr: &Expr, h: &mut FxHasher) {
    h.write_u64(expr.0.hash);
}

fn hash_exprs(exprs: &[Expr], h: &mut FxHasher) {
    h.write_usize(exprs.len());
    for expr in exprs {
        hash_expr(expr, h);
    }
}

fn hash_ptr<T>(rc: &Rc<T>, h: &mut FxHasher) {
    h.write_usize(Rc::as_ptr(rc) as usize);
}

fn hash_source(src: &Source, h: &mut FxHasher) {
    match src {
        Source::Storage(view) => {
            h.write_u8(0);
            view.hash(h);
        }
        Source::Quantized(q) => {
            h.write_u8(1);
            q.hash(h);
        }
    }
}

fn hash_addr(addr: &Addr, h: &mut FxHasher) {
    match addr {
        Addr::Linear(index) => {
            h.write_u8(0);
            hash_expr(index, h);
        }
        Addr::Rc2 { row, col } => {
            h.write_u8(1);
            hash_expr(row, h);
            hash_expr(col, h);
        }
    }
}

fn hash_coop_src(src: &CoopSrc, h: &mut FxHasher) {
    match src {
        CoopSrc::TileRegion { tile, row, col } => {
            h.write_u8(0);
            hash_ptr(tile, h);
            hash_expr(row, h);
            hash_expr(col, h);
        }
        CoopSrc::BroadcastCol { src, col } => {
            h.write_u8(1);
            src.hash(h);
            hash_expr(col, h);
        }
    }
}

fn hash_reduce_kind(kind: &ReduceKind, h: &mut FxHasher) {
    match kind {
        ReduceKind::Subgroup => h.write_u8(0),
        ReduceKind::Workgroup {
            scratch,
            group_size,
        } => {
            h.write_u8(1);
            hash_ptr(scratch, h);
            group_size.hash(h);
        }
        ReduceKind::Loop {
            iterations,
            index,
            scratch,
            group_size,
        } => {
            h.write_u8(2);
            iterations.hash(h);
            hash_ptr(index, h);
            hash_ptr(scratch, h);
            group_size.hash(h);
        }
    }
}

fn hash_kind_into(kind: &ExprKind, h: &mut FxHasher) {
    std::mem::discriminant(kind).hash(h);
    match kind {
        ExprKind::Literal(lit) => lit.hash(h),
        ExprKind::Builtin(builtin) => builtin.hash(h),
        ExprKind::LoadLocal(local) => hash_ptr(local, h),
        ExprKind::Load {
            src,
            addr,
            mask,
            fill,
        } => {
            hash_source(src, h);
            hash_addr(addr, h);
            hash_expr(mask, h);
            hash_expr(fill, h);
        }
        ExprKind::LoadTile { tile, index } => {
            hash_ptr(tile, h);
            hash_expr(index, h);
        }
        ExprKind::Unary { op, value } => {
            op.hash(h);
            hash_expr(value, h);
        }
        ExprKind::Binary { op, left, right } => {
            op.hash(h);
            hash_expr(left, h);
            hash_expr(right, h);
        }
        ExprKind::Compare { op, left, right } => {
            op.hash(h);
            hash_expr(left, h);
            hash_expr(right, h);
        }
        ExprKind::Cast { value, to } => {
            hash_expr(value, h);
            to.hash(h);
        }
        ExprKind::Bitcast { value, to } => {
            hash_expr(value, h);
            to.hash(h);
        }
        ExprKind::Select {
            condition,
            accept,
            reject,
        } => {
            hash_expr(condition, h);
            hash_expr(accept, h);
            hash_expr(reject, h);
        }
        ExprKind::Vec {
            scalar,
            lanes,
            parts,
        } => {
            scalar.hash(h);
            lanes.hash(h);
            hash_exprs(parts, h);
        }
        ExprKind::Dot { left, right } => {
            hash_expr(left, h);
            hash_expr(right, h);
        }
        ExprKind::Reduce { op, kind, value } => {
            op.hash(h);
            hash_reduce_kind(kind, h);
            hash_expr(value, h);
        }
        ExprKind::CoopLoad {
            role,
            scalar,
            rows,
            cols,
            src,
        } => {
            role.hash(h);
            scalar.hash(h);
            rows.hash(h);
            cols.hash(h);
            hash_coop_src(src, h);
        }
        ExprKind::CoopMma { a, b, c } => {
            hash_expr(a, h);
            hash_expr(b, h);
            hash_expr(c, h);
        }
        ExprKind::Dequantize {
            src,
            k_base,
            col,
            mask,
            fill,
            lanes,
        } => {
            src.hash(h);
            hash_expr(k_base, h);
            hash_expr(col, h);
            hash_expr(mask, h);
            hash_expr(fill, h);
            lanes.hash(h);
        }
        ExprKind::LaneOf { block, lane } => {
            hash_expr(block, h);
            lane.hash(h);
        }
        ExprKind::QuantizedDot {
            src,
            packing,
            activations,
            k_base,
            col,
            mask,
            fill,
        } => {
            src.hash(h);
            packing.hash(h);
            hash_exprs(activations, h);
            hash_expr(k_base, h);
            hash_expr(col, h);
            hash_expr(mask, h);
            hash_expr(fill, h);
        }
        ExprKind::Shared(inner) => hash_expr(inner, h),
    }
}

// ---- deep structural equality (collision fallback for `PartialEq`) ----

fn expr_eq(a: &Expr, b: &Expr) -> bool {
    a == b
}

fn exprs_eq(a: &[Expr], b: &[Expr]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| expr_eq(x, y))
}

fn source_eq(a: &Source, b: &Source) -> bool {
    match (a, b) {
        (Source::Storage(x), Source::Storage(y)) => x == y,
        (Source::Quantized(x), Source::Quantized(y)) => x == y,
        _ => false,
    }
}

fn addr_eq(a: &Addr, b: &Addr) -> bool {
    match (a, b) {
        (Addr::Linear(x), Addr::Linear(y)) => expr_eq(x, y),
        (Addr::Rc2 { row: rx, col: cx }, Addr::Rc2 { row: ry, col: cy }) => {
            expr_eq(rx, ry) && expr_eq(cx, cy)
        }
        _ => false,
    }
}

fn coop_src_eq(a: &CoopSrc, b: &CoopSrc) -> bool {
    match (a, b) {
        (
            CoopSrc::TileRegion {
                tile: tx,
                row: rx,
                col: cx,
            },
            CoopSrc::TileRegion {
                tile: ty,
                row: ry,
                col: cy,
            },
        ) => Rc::ptr_eq(tx, ty) && expr_eq(rx, ry) && expr_eq(cx, cy),
        (
            CoopSrc::BroadcastCol { src: sx, col: cx },
            CoopSrc::BroadcastCol { src: sy, col: cy },
        ) => sx == sy && expr_eq(cx, cy),
        _ => false,
    }
}

fn reduce_kind_eq(a: &ReduceKind, b: &ReduceKind) -> bool {
    match (a, b) {
        (ReduceKind::Subgroup, ReduceKind::Subgroup) => true,
        (
            ReduceKind::Workgroup {
                scratch: sx,
                group_size: gx,
            },
            ReduceKind::Workgroup {
                scratch: sy,
                group_size: gy,
            },
        ) => Rc::ptr_eq(sx, sy) && gx == gy,
        (
            ReduceKind::Loop {
                iterations: ix,
                index: nx,
                scratch: sx,
                group_size: gx,
            },
            ReduceKind::Loop {
                iterations: iy,
                index: ny,
                scratch: sy,
                group_size: gy,
            },
        ) => ix == iy && Rc::ptr_eq(nx, ny) && Rc::ptr_eq(sx, sy) && gx == gy,
        _ => false,
    }
}

fn kind_eq(a: &ExprKind, b: &ExprKind) -> bool {
    match (a, b) {
        (ExprKind::Literal(x), ExprKind::Literal(y)) => x == y,
        (ExprKind::Builtin(x), ExprKind::Builtin(y)) => x == y,
        (ExprKind::LoadLocal(x), ExprKind::LoadLocal(y)) => Rc::ptr_eq(x, y),
        (
            ExprKind::Load {
                src: sx,
                addr: ax,
                mask: mx,
                fill: fx,
            },
            ExprKind::Load {
                src: sy,
                addr: ay,
                mask: my,
                fill: fy,
            },
        ) => source_eq(sx, sy) && addr_eq(ax, ay) && expr_eq(mx, my) && expr_eq(fx, fy),
        (
            ExprKind::LoadTile {
                tile: tx,
                index: ix,
            },
            ExprKind::LoadTile {
                tile: ty,
                index: iy,
            },
        ) => Rc::ptr_eq(tx, ty) && expr_eq(ix, iy),
        (ExprKind::Unary { op: ox, value: vx }, ExprKind::Unary { op: oy, value: vy }) => {
            ox == oy && expr_eq(vx, vy)
        }
        (
            ExprKind::Binary {
                op: ox,
                left: lx,
                right: rx,
            },
            ExprKind::Binary {
                op: oy,
                left: ly,
                right: ry,
            },
        ) => ox == oy && expr_eq(lx, ly) && expr_eq(rx, ry),
        (
            ExprKind::Compare {
                op: ox,
                left: lx,
                right: rx,
            },
            ExprKind::Compare {
                op: oy,
                left: ly,
                right: ry,
            },
        ) => ox == oy && expr_eq(lx, ly) && expr_eq(rx, ry),
        (ExprKind::Cast { value: vx, to: tx }, ExprKind::Cast { value: vy, to: ty }) => {
            expr_eq(vx, vy) && tx == ty
        }
        (ExprKind::Bitcast { value: vx, to: tx }, ExprKind::Bitcast { value: vy, to: ty }) => {
            expr_eq(vx, vy) && tx == ty
        }
        (
            ExprKind::Select {
                condition: cx,
                accept: ax,
                reject: rx,
            },
            ExprKind::Select {
                condition: cy,
                accept: ay,
                reject: ry,
            },
        ) => expr_eq(cx, cy) && expr_eq(ax, ay) && expr_eq(rx, ry),
        (
            ExprKind::Vec {
                scalar: sx,
                lanes: lx,
                parts: px,
            },
            ExprKind::Vec {
                scalar: sy,
                lanes: ly,
                parts: py,
            },
        ) => sx == sy && lx == ly && exprs_eq(px, py),
        (
            ExprKind::Dot {
                left: lx,
                right: rx,
            },
            ExprKind::Dot {
                left: ly,
                right: ry,
            },
        ) => expr_eq(lx, ly) && expr_eq(rx, ry),
        (
            ExprKind::Reduce {
                op: ox,
                kind: kx,
                value: vx,
            },
            ExprKind::Reduce {
                op: oy,
                kind: ky,
                value: vy,
            },
        ) => ox == oy && reduce_kind_eq(kx, ky) && expr_eq(vx, vy),
        (
            ExprKind::CoopLoad {
                role: rx,
                scalar: sx,
                rows: rwx,
                cols: cx,
                src: srx,
            },
            ExprKind::CoopLoad {
                role: ry,
                scalar: sy,
                rows: rwy,
                cols: cy,
                src: sry,
            },
        ) => rx == ry && sx == sy && rwx == rwy && cx == cy && coop_src_eq(srx, sry),
        (
            ExprKind::CoopMma {
                a: ax,
                b: bx,
                c: cx,
            },
            ExprKind::CoopMma {
                a: ay,
                b: by,
                c: cy,
            },
        ) => expr_eq(ax, ay) && expr_eq(bx, by) && expr_eq(cx, cy),
        (
            ExprKind::Dequantize {
                src: sx,
                k_base: kx,
                col: clx,
                mask: mx,
                fill: fx,
                lanes: lnx,
            },
            ExprKind::Dequantize {
                src: sy,
                k_base: ky,
                col: cly,
                mask: my,
                fill: fy,
                lanes: lny,
            },
        ) => {
            sx == sy
                && expr_eq(kx, ky)
                && expr_eq(clx, cly)
                && expr_eq(mx, my)
                && expr_eq(fx, fy)
                && lnx == lny
        }
        (
            ExprKind::LaneOf {
                block: bx,
                lane: lx,
            },
            ExprKind::LaneOf {
                block: by,
                lane: ly,
            },
        ) => expr_eq(bx, by) && lx == ly,
        (
            ExprKind::QuantizedDot {
                src: sx,
                packing: px,
                activations: ax,
                k_base: kx,
                col: clx,
                mask: mx,
                fill: fx,
            },
            ExprKind::QuantizedDot {
                src: sy,
                packing: py,
                activations: ay,
                k_base: ky,
                col: cly,
                mask: my,
                fill: fy,
            },
        ) => {
            sx == sy
                && px == py
                && exprs_eq(ax, ay)
                && expr_eq(kx, ky)
                && expr_eq(clx, cly)
                && expr_eq(mx, my)
                && expr_eq(fx, fy)
        }
        (ExprKind::Shared(x), ExprKind::Shared(y)) => expr_eq(x, y),
        _ => false,
    }
}
