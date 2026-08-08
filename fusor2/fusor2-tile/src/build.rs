//! Hash-consed L2 term builders, shared by both emitters. Structural sharing
//! comes from hash-consing the whole L2 term, so two identical subtrees built
//! separately merge. Hash-consing is scope-free: there is no loop-boundary
//! snapshot/restore.
//!
//! Declarations are not interned. Two same-shaped tiles stay distinct so the
//! arena knows they are two allocations, so `alloc_tile`/`alloc_local`/
//! `alloc_buffer` each mint a fresh `Arc` and push it to an ordered list.

use fusor2_ir::dtype::{NumericContract, RoundMode};
use fusor2_ir::ir::level2::{
    Accumulator, Addr, Buffer, BufferAccess, BufferDecl, Builtin, CoopMatrixRole,
    CoopSrc, ElementType, KernelIr, Local, LocalDecl, MemoryLevel, ReduceKind, ScalarElement,
    Source, Stmt, StorageView, Tile, TileBinaryOp, TileCompareOp, TileDecl, TileExpr, TileExprKind,
    TileLayout, TileLiteral, TileReduceOp, TileUnaryOp,
};
use rustc_hash::{FxHashMap, FxHasher};
use smallvec::SmallVec;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// The largest finite value WGSL can spell. WGSL has no infinite literal and
/// naga rejects a module holding one, so every `-inf` sentinel is spelled as
/// this instead; `exp(x - m)` underflows to zero against it just the same.
pub const WGSL_SAFE_F32_MAX: f32 = 3.40282e38;

type SmallVecTiles = SmallVec<[Tile; 4]>;
type SmallVecLocals = SmallVec<[Local; 4]>;

/// Variant tag of a [`TileExprKind`], so the memo never compares kinds across
/// variants on a hash collision.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DiscriminantTag(u16);

fn tag_of(kind: &TileExprKind) -> DiscriminantTag {
    DiscriminantTag(match kind {
        TileExprKind::Literal(_) => 0,
        TileExprKind::Builtin(_) => 1,
        TileExprKind::LoadLocal(_) => 2,
        TileExprKind::Load { .. } => 3,
        TileExprKind::LoadTile { .. } => 4,
        TileExprKind::Unary { .. } => 5,
        TileExprKind::Binary { .. } => 6,
        TileExprKind::Compare { .. } => 7,
        TileExprKind::Round { .. } => 8,
        TileExprKind::Cast { .. } => 9,
        TileExprKind::Bitcast { .. } => 10,
        TileExprKind::Select { .. } => 11,
        TileExprKind::Vec { .. } => 12,
        TileExprKind::VecComponent { .. } => 13,
        TileExprKind::Dot { .. } => 14,
        TileExprKind::Reduce { .. } => 15,
        TileExprKind::CoopLoad { .. } => 16,
        TileExprKind::CoopZero { .. } => 21,
        TileExprKind::CoopMma { .. } => 17,
    })
}

/// Memo key. The hash is bottom-up — [`TileExpr`]'s own `Hash` writes only its
/// cached `structural_hash`, so hashing a node is O(1) in its children.
pub type TileKey = (u64, ElementType, DiscriminantTag);

/// Builds L2 terms with a hash-cons memo, so two identical subtrees built by
/// separate call sites are one node.
#[derive(Default)]
pub struct TileBuilder {
    exprs: FxHashMap<TileKey, TileExpr>,
    tiles: Vec<Tile>,
    locals: Vec<Local>,
    buffers: Vec<Buffer>,
    body: Vec<Stmt>,
}

impl TileBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// A fresh workgroup/private tile. Two same-shaped tiles stay distinct.
    pub fn alloc_tile(&mut self, element: ElementType, layout: TileLayout) -> Tile {
        self.alloc_tile_named(element, layout, "tile")
    }

    /// [`Self::alloc_tile`] with a name the emitter can use for the global.
    pub fn alloc_tile_named(
        &mut self,
        element: ElementType,
        layout: TileLayout,
        name: &'static str,
    ) -> Tile {
        let tile = Arc::new(TileDecl::new(element, layout, name));
        self.tiles.push(tile.clone());
        tile
    }

    /// A fresh private per-invocation local.
    pub fn alloc_local(&mut self, element: ElementType) -> Local {
        let local = Arc::new(LocalDecl::new(element));
        self.locals.push(local.clone());
        local
    }

    /// A fresh storage buffer. Binding 0 is by convention the uniform block.
    pub fn alloc_buffer(
        &mut self,
        binding: u32,
        element: ElementType,
        layout: TileLayout,
        access: BufferAccess,
    ) -> Buffer {
        let buffer = Arc::new(BufferDecl {
            binding,
            element,
            layout,
            access,
        });
        self.buffers.push(buffer.clone());
        buffer
    }

    /// Declared buffers in binding order.
    pub fn buffers(&self) -> Vec<Buffer> {
        let mut out = self.buffers.clone();
        out.sort_by_key(|b| b.binding);
        out
    }

    pub fn tiles(&self) -> &[Tile] {
        &self.tiles
    }

    pub fn locals(&self) -> &[Local] {
        &self.locals
    }

    /// How many distinct expression nodes have been interned.
    pub fn interned_len(&self) -> usize {
        self.exprs.len()
    }

    /// Intern one node. Collisions are resolved by comparing the full
    /// [`TileExprKind`]; the hash alone is never trusted.
    pub fn expr(&mut self, kind: TileExprKind, ty: ElementType) -> TileExpr {
        let mut hasher = FxHasher::default();
        kind.hash(&mut hasher);
        ty.hash(&mut hasher);
        let key = (hasher.finish(), ty, tag_of(&kind));
        if let Some(existing) = self.exprs.get(&key)
            && existing.kind() == &kind
        {
            return existing.clone();
        }
        let node = TileExpr::new(kind, ty);
        self.exprs.insert(key, node.clone());
        node
    }

    /// Intern a node whose element type is derived by the L2 type rules. A
    /// malformed construction is not silently accepted: the fallback type
    /// makes `verify_l2` reject the term.
    fn infer_expr(&mut self, kind: TileExprKind) -> TileExpr {
        let ty = crate::verify_l2::infer_kind(&kind)
            .unwrap_or(ElementType::Scalar(ScalarElement::F32));
        self.expr(kind, ty)
    }

    pub fn lit(&mut self, value: TileLiteral) -> TileExpr {
        self.infer_expr(TileExprKind::Literal(value))
    }
    pub fn lit_f32(&mut self, value: f32) -> TileExpr {
        self.lit(TileLiteral::F32(value.to_bits()))
    }
    pub fn lit_u32(&mut self, value: u32) -> TileExpr {
        self.lit(TileLiteral::U32(value))
    }
    pub fn lit_i32(&mut self, value: i32) -> TileExpr {
        self.lit(TileLiteral::I32(value))
    }
    pub fn lit_bool(&mut self, value: bool) -> TileExpr {
        self.lit(TileLiteral::Bool(value))
    }
    /// The statically-true mask the lowerer skips codegen for.
    pub fn mask_true(&mut self) -> TileExpr {
        self.lit_bool(true)
    }

    /// A typed zero, used as a fill value and as a tile initializer.
    pub fn zero(&mut self, element: ElementType) -> TileExpr {
        match element {
            ElementType::Scalar(scalar) => self.zero_scalar(scalar),
            ElementType::Vector { scalar, lanes } => {
                let part = self.zero_scalar(scalar);
                let parts = vec![part; lanes as usize];
                self.vec(scalar, parts)
            }
            // A cooperative fragment has no literal form; the scalar zero is
            // the only sensible placeholder and `verify_l2` rejects it if a
            // caller actually stores one.
            ElementType::CoopMatrix { scalar, .. } => self.zero_scalar(scalar),
        }
    }

    /// The zero of a scalar element type, used as a load fill and an
    /// accumulator init.
    pub fn zero_scalar(&mut self, scalar: ScalarElement) -> TileExpr {
        match scalar {
            ScalarElement::F32 => self.lit(TileLiteral::F32(0)),
            ScalarElement::F16 => self.lit(TileLiteral::F16(0)),
            ScalarElement::BF16 => self.lit(TileLiteral::BF16(0)),
            ScalarElement::U32 => self.lit(TileLiteral::U32(0)),
            ScalarElement::I32 => self.lit(TileLiteral::I32(0)),
            ScalarElement::Bool => self.lit(TileLiteral::Bool(false)),
        }
    }

    /// The "smaller than anything real" sentinel a max carrier starts from.
    /// Finite, not `-inf`, since WGSL has no infinite literal. These are the
    /// same values the GPU emitter's reduce identities use, so a max started
    /// here and a max started by a `Reduce` agree bit for bit.
    pub fn neg_inf(&mut self, elem: ScalarElement) -> TileExpr {
        match elem {
            ScalarElement::F16 => {
                self.lit(TileLiteral::F16(half::f16::from_f32(-65504.0).to_bits()))
            }
            // bf16 rounds the f32 sentinel straight back to -inf, so take its
            // own finite extreme.
            ScalarElement::BF16 => self.lit(TileLiteral::BF16(half::bf16::MIN.to_bits())),
            _ => self.lit_f32(-WGSL_SAFE_F32_MAX),
        }
    }

    /// The `Min` identity, the mirror of [`Self::neg_inf`]: the largest
    /// finite magnitude, so a min started here and a min started by a
    /// `Reduce` agree bit for bit.
    pub fn pos_inf(&mut self, elem: ScalarElement) -> TileExpr {
        match elem {
            ScalarElement::F16 => {
                self.lit(TileLiteral::F16(half::f16::from_f32(65504.0).to_bits()))
            }
            ScalarElement::BF16 => self.lit(TileLiteral::BF16(half::bf16::MAX.to_bits())),
            _ => self.lit_f32(WGSL_SAFE_F32_MAX),
        }
    }

    pub fn builtin(&mut self, builtin: Builtin) -> TileExpr {
        self.infer_expr(TileExprKind::Builtin(builtin))
    }

    pub fn load_local(&mut self, local: Local) -> TileExpr {
        self.infer_expr(TileExprKind::LoadLocal(local))
    }

    pub fn load(&mut self, src: Source, addr: Addr, mask: TileExpr, fill: TileExpr) -> TileExpr {
        self.infer_expr(TileExprKind::Load {
            src,
            addr: Box::new(addr),
            mask,
            fill,
        })
    }

    pub fn load_tile(&mut self, tile: Tile, index: TileExpr) -> TileExpr {
        self.infer_expr(TileExprKind::LoadTile { tile, index })
    }

    pub fn unary(
        &mut self,
        op: TileUnaryOp,
        value: TileExpr,
        numeric: NumericContract,
    ) -> TileExpr {
        self.infer_expr(TileExprKind::Unary { op, value, numeric })
    }

    pub fn binary(
        &mut self,
        op: TileBinaryOp,
        left: TileExpr,
        right: TileExpr,
        numeric: NumericContract,
    ) -> TileExpr {
        self.infer_expr(TileExprKind::Binary {
            op,
            left,
            right,
            numeric,
        })
    }

    pub fn add(&mut self, a: TileExpr, b: TileExpr) -> TileExpr {
        self.binary(TileBinaryOp::Add, a, b, NumericContract::RELAXED)
    }
    pub fn mul(&mut self, a: TileExpr, b: TileExpr) -> TileExpr {
        self.binary(TileBinaryOp::Mul, a, b, NumericContract::RELAXED)
    }
    pub fn sub(&mut self, a: TileExpr, b: TileExpr) -> TileExpr {
        self.binary(TileBinaryOp::Sub, a, b, NumericContract::RELAXED)
    }
    /// `a * b + c` with contraction permitted, the fused-multiply-add the
    /// emitter is free to issue as one instruction.
    pub fn fma(&mut self, a: TileExpr, b: TileExpr, c: TileExpr) -> TileExpr {
        let p = self.mul(a, b);
        self.add(p, c)
    }
    pub fn and(&mut self, a: TileExpr, b: TileExpr) -> TileExpr {
        self.binary(TileBinaryOp::LogicalAnd, a, b, NumericContract::RELAXED)
    }

    pub fn compare(&mut self, op: TileCompareOp, left: TileExpr, right: TileExpr) -> TileExpr {
        self.infer_expr(TileExprKind::Compare { op, left, right })
    }

    pub fn round(&mut self, mode: RoundMode, value: TileExpr) -> TileExpr {
        self.infer_expr(TileExprKind::Round { mode, value })
    }

    /// An identity cast is elided rather than interned: the emitters spell a
    /// same-type cast out, so keeping the node would change the shader text.
    pub fn cast(&mut self, value: TileExpr, to: ElementType) -> TileExpr {
        if value.element() == to {
            return value;
        }
        self.infer_expr(TileExprKind::Cast { value, to })
    }

    pub fn bitcast(&mut self, value: TileExpr, to: ElementType) -> TileExpr {
        self.infer_expr(TileExprKind::Bitcast { value, to })
    }

    pub fn select(&mut self, condition: TileExpr, accept: TileExpr, reject: TileExpr) -> TileExpr {
        self.infer_expr(TileExprKind::Select {
            condition,
            accept,
            reject,
        })
    }

    pub fn vec(&mut self, scalar: ScalarElement, parts: Vec<TileExpr>) -> TileExpr {
        let lanes = parts.len() as u32;
        self.infer_expr(TileExprKind::Vec {
            scalar,
            lanes,
            parts,
        })
    }

    pub fn dot(&mut self, left: TileExpr, right: TileExpr) -> TileExpr {
        self.infer_expr(TileExprKind::Dot { left, right })
    }

    pub fn reduce(&mut self, op: TileReduceOp, kind: ReduceKind, value: TileExpr) -> TileExpr {
        self.infer_expr(TileExprKind::Reduce {
            op,
            kind: Box::new(kind),
            value,
        })
    }

    /// The N-ary reduction, expressed as a carrier. A carrier that is one
    /// scalar slot merged by a binop delegates to [`Self::reduce`] and returns
    /// a one-element vector. Otherwise it allocates one scratch tile, one pair
    /// of merge formals and one output local per accumulator lane, pushes a
    /// [`Stmt::Reduce`] into `out`, and returns the per-lane reads. `merge`
    /// builds lane `i`'s merged expression from the formals; `values` is one
    /// partial per lane and `scratch_extents` the tile shape one lane needs.
    pub fn reduce_carrier<E>(
        &mut self,
        kind: ReduceKind,
        carrier: &fusor2_ir::carrier::Carrier,
        values: &[TileExpr],
        scratch_extents: &[u32],
        out: &mut Vec<Stmt>,
        mut merge: impl FnMut(&mut Self, usize, &[TileExpr], &[TileExpr]) -> Result<TileExpr, E>,
    ) -> Result<Vec<TileExpr>, E>
    where
        E: From<String>,
    {
        if let Some(op) = fusor2_ir::ir::level2::fast_reduce_op(carrier)
            && values.len() == 1
        {
            return Ok(vec![self.reduce(op, kind, values[0].clone())]);
        }
        let n = values.len();
        let scratch: SmallVecTiles = match &kind {
            ReduceKind::Subgroup => SmallVecTiles::new(),
            ReduceKind::Workgroup { scratch, .. } | ReduceKind::Loop { scratch, .. } => {
                let head = scratch.clone();
                let mut tiles = SmallVecTiles::new();
                tiles.push(head);
                for _ in 1..n {
                    tiles.push(self.alloc_tile_named(
                        values[0].element(),
                        TileLayout::contiguous(
                            fusor2_ir::ir::level2::MemoryLevel::Workgroup,
                            scratch_extents,
                        ),
                        "fold_scratch",
                    ));
                }
                tiles
            }
        };
        let lhs: SmallVecLocals = (0..n).map(|i| self.alloc_local(values[i].element())).collect();
        let rhs: SmallVecLocals = (0..n).map(|i| self.alloc_local(values[i].element())).collect();
        let outs: SmallVecLocals = (0..n).map(|i| self.alloc_local(values[i].element())).collect();
        let lhs_reads: Vec<TileExpr> = lhs.iter().map(|l| self.load_local(l.clone())).collect();
        let rhs_reads: Vec<TileExpr> = rhs.iter().map(|l| self.load_local(l.clone())).collect();
        let mut body: SmallVec<[TileExpr; 4]> = SmallVec::new();
        for i in 0..n {
            body.push(merge(self, i, &lhs_reads, &rhs_reads)?);
        }
        out.push(Stmt::Reduce {
            kind: Box::new(kind),
            values: values.iter().cloned().collect(),
            merge: Box::new(fusor2_ir::ir::level2::MergeBody { lhs, rhs, body }),
            // Only reached when the carrier is not a single scalar binop slot.
            fast: None,
            outs: outs.clone(),
            scratch,
        });
        Ok(outs.into_iter().map(|l| self.load_local(l)).collect())
    }

    pub fn coop_load(
        &mut self,
        role: CoopMatrixRole,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
        src: CoopSrc,
    ) -> TileExpr {
        self.infer_expr(TileExprKind::CoopLoad {
            role,
            scalar,
            rows,
            cols,
            src: Box::new(src),
        })
    }

    /// An all-zero fragment of the same shape as a cooperative accumulator.
    pub fn coop_zero(
        &mut self,
        role: CoopMatrixRole,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
    ) -> TileExpr {
        self.infer_expr(TileExprKind::CoopZero {
            role,
            scalar,
            rows,
            cols,
        })
    }

    pub fn coop_mma(&mut self, a: TileExpr, b: TileExpr, c: TileExpr) -> TileExpr {
        self.infer_expr(TileExprKind::CoopMma { a, b, c })
    }

    /// A private per-invocation local minted without registration, for a
    /// lowerer that assembles its `KernelIr` itself. Identity-bearing, so not
    /// interned; [`Self::alloc_local`] is the recording form.
    pub fn local(&self, element: ElementType) -> Local {
        Arc::new(LocalDecl::new(element))
    }

    /// A workgroup tile minted without registration. Also identity-bearing:
    /// two tiles with the same shape are two allocations the arena may or may
    /// not overlap. [`Self::alloc_tile_named`] is the recording form.
    pub fn tile(&self, name: &'static str, element: ElementType, extents: &[u32]) -> Tile {
        Arc::new(TileDecl::new(
            element,
            TileLayout::contiguous(MemoryLevel::Workgroup, extents),
            name,
        ))
    }

    pub fn store(&self, dst: StorageView, addr: Addr, value: TileExpr, mask: TileExpr) -> Stmt {
        Stmt::Store {
            dst,
            addr,
            value,
            mask,
        }
    }

    /// The `ScatterMode::Atomic` verb. Carries `Effect::InPlace` at L1.
    pub fn atomic_add(
        &self,
        dst: StorageView,
        addr: Addr,
        value: TileExpr,
        mask: TileExpr,
    ) -> Stmt {
        Stmt::AtomicAdd {
            dst,
            addr,
            value,
            mask,
        }
    }

    pub fn store_local(&self, dst: Local, value: TileExpr) -> Stmt {
        Stmt::StoreLocal { dst, value }
    }

    pub fn store_tile(&self, dst: Tile, index: TileExpr, value: TileExpr) -> Stmt {
        Stmt::StoreTile { dst, index, value }
    }

    pub fn fill_tile(&self, dst: Tile, value: TileExpr, bounds: [Option<TileExpr>; 2]) -> Stmt {
        Stmt::FillTile { dst, value, bounds }
    }

    pub fn coop_store(&self, acc: TileExpr, dst: StorageView, addr: Addr) -> Stmt {
        Stmt::CoopStore { acc, dst, addr }
    }

    pub fn coop_store_tile(&self, acc: TileExpr, tile: Tile, row: TileExpr, col: TileExpr) -> Stmt {
        Stmt::CoopStoreTile {
            acc,
            tile,
            row,
            col,
        }
    }

    pub fn if_then_else(&self, condition: TileExpr, accept: Vec<Stmt>, reject: Vec<Stmt>) -> Stmt {
        Stmt::If {
            condition,
            accept,
            reject,
        }
    }

    /// A counted loop with SSA-carried accumulators, so the lowerer never
    /// reloads an accumulator per iteration.
    pub fn loop_counted(
        &self,
        count: Option<TileExpr>,
        index: Option<Local>,
        accumulators: Vec<Accumulator>,
        body: Vec<Stmt>,
    ) -> Stmt {
        Stmt::Loop {
            count,
            index,
            accumulators,
            body,
        }
    }

    /// An unstructured loop exited by [`Self::break_`].
    pub fn loop_forever(&self, body: Vec<Stmt>) -> Stmt {
        Stmt::Loop {
            count: None,
            index: None,
            accumulators: Vec::new(),
            body,
        }
    }

    pub fn barrier(&self) -> Stmt {
        Stmt::Barrier
    }

    pub fn push(&mut self, stmt: Stmt) {
        self.body.push(stmt);
    }

    pub fn extend(&mut self, stmts: impl IntoIterator<Item = Stmt>) {
        self.body.extend(stmts);
    }

    pub fn set_body(&mut self, body: Vec<Stmt>) {
        self.body = body;
    }

    pub fn body(&self) -> &[Stmt] {
        &self.body
    }

    /// Close the kernel. Buffers come out in binding order, so the derived
    /// bind group and the builder's buffer list cannot drift. `byte_arena` is
    /// always `None`: that token is a device capability only the backend
    /// lowerers, which hold `Caps`, can mint.
    pub fn finish(&mut self, grid: [u32; 3], block: u32, name: &'static str) -> KernelIr {
        KernelIr {
            buffers: self.buffers(),
            grid,
            block,
            body: std::mem::take(&mut self.body),
            byte_arena: None,
            name,
        }
    }
}

/// Shared test fixture: tile A 8x8 f32 (256 B) and tile B 4x8 f32 (128 B),
/// touched on either side of a caller-supplied `between`.
#[cfg(test)]
pub(crate) mod fixtures {
    use super::*;
    use fusor2_ir::device::{Caps, DeviceKind, Limits, SubgroupWidths};
    use fusor2_ir::ir::level2::MemoryLevel;

    /// Both tiles in one region: `max(256, 128)`.
    pub const SHARED: u32 = 256;
    /// Two regions: `256 + 128`.
    pub const UNSHARED: u32 = 384;

    pub fn base_caps() -> Caps {
        Caps {
            kind: DeviceKind::Gpu,
            name: "test".into(),
            limits: Limits::default(),
            subgroups: Some(SubgroupWidths { min: 32, max: 32 }),
            f16: true,
            bf16: true,
            coop: smallvec::SmallVec::new(),
            atomic_f32: true,
            workgroup_alias: false,
            mixed_precision_coop_store: false,
            pipeline_cache: false,
            timestamp_query: false,
            simd_widths: smallvec::smallvec![4, 8],
            threads: 1,
        }
    }

    pub fn caps_with(edit: impl FnOnce(&mut Caps)) -> Caps {
        let mut caps = base_caps();
        edit(&mut caps);
        caps
    }

    pub fn wg_tile(builder: &mut TileBuilder, element: ElementType, elements: u32) -> Tile {
        builder.alloc_tile(
            element,
            TileLayout::contiguous(MemoryLevel::Workgroup, &[elements]),
        )
    }

    pub fn two_f32_tiles(builder: &mut TileBuilder) -> (Tile, Tile) {
        let a = builder.alloc_tile_named(
            ScalarElement::F32.element(),
            TileLayout::contiguous(MemoryLevel::Workgroup, &[8, 8]),
            "a",
        );
        let b = builder.alloc_tile_named(
            ScalarElement::F32.element(),
            TileLayout::contiguous(MemoryLevel::Workgroup, &[4, 8]),
            "b",
        );
        (a, b)
    }

    /// `write A; <between>; write B`.
    pub fn pair_kernel(builder: &mut TileBuilder, between: Vec<Stmt>) -> KernelIr {
        let (a, b) = two_f32_tiles(builder);
        let zero = builder.lit_f32(0.0);
        let index = builder.lit_u32(0);
        let write_a = builder.store_tile(a, index.clone(), zero.clone());
        let write_b = builder.store_tile(b, index, zero);
        let mut body = vec![write_a];
        body.extend(between);
        body.push(write_b);
        builder.set_body(body);
        builder.finish([1, 1, 1], 64, "pair")
    }

    pub fn whole_buffer_view(buffer: &Buffer) -> StorageView {
        StorageView {
            buffer: buffer.clone(),
            offset: 0,
            layout: buffer.layout.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor2_ir::scalar::{BinOp, UnOp};

    fn sqrt_of_product(b: &mut TileBuilder, a: f32, c: f32) -> TileExpr {
        let x = b.lit_f32(a);
        let y = b.lit_f32(c);
        let product = b.binary(BinOp::Mul, x, y, NumericContract::RELAXED);
        b.unary(UnOp::Sqrt, product, NumericContract::RELAXED)
    }

    /// `TileExpr`'s `Arc` is private, but `kind()` borrows out of the
    /// allocation, so equal `kind()` addresses means `Arc::ptr_eq`.
    fn same_node(a: &TileExpr, b: &TileExpr) -> bool {
        std::ptr::eq(a.kind() as *const TileExprKind, b.kind() as *const TileExprKind)
    }

    #[test]
    fn hash_cons_merges_separately_built_subtrees() {
        let mut b = TileBuilder::new();
        // Two independent call paths building the same term.
        let first = sqrt_of_product(&mut b, 2.0, 3.0);
        let after_first = b.interned_len();
        let second = {
            let x = b.lit_f32(2.0);
            let y = b.lit_f32(3.0);
            let product = b.binary(BinOp::Mul, x, y, NumericContract::RELAXED);
            b.unary(UnOp::Sqrt, product, NumericContract::RELAXED)
        };
        // 2 literals + 1 binary + 1 unary, and the second path allocates none.
        assert_eq!(after_first, 4);
        assert_eq!(b.interned_len(), 4);
        assert!(same_node(&first, &second));
        assert_eq!(first, second);
        assert_eq!(first.structural_hash(), second.structural_hash());
    }

    #[test]
    fn a_numeric_contract_difference_does_not_merge() {
        let mut b = TileBuilder::new();
        let x = b.lit_f32(2.0);
        let y = b.lit_f32(3.0);
        let relaxed = b.binary(BinOp::Mul, x.clone(), y.clone(), NumericContract::RELAXED);
        let strict = b.binary(BinOp::Mul, x, y, NumericContract::STRICT);
        assert!(!same_node(&relaxed, &strict));
    }

    #[test]
    fn atomic_add_is_a_first_class_verb() {
        use fusor2_ir::ir::level2::{BufferAccess, MemoryLevel};
        let mut b = TileBuilder::new();
        let buffer = b.alloc_buffer(
            0,
            ScalarElement::F32.element(),
            TileLayout::contiguous(MemoryLevel::Storage, &[16]),
            BufferAccess::ReadWrite,
        );
        let view = fixtures::whole_buffer_view(&buffer);
        let index = b.lit_u32(3);
        let value = b.lit_f32(1.0);
        let mask = b.mask_true();
        let stmt = b.atomic_add(view, Addr::Linear(index), value, mask);
        b.push(stmt);
        let ir = b.finish([1, 1, 1], 64, "scatter-add");
        crate::verify_l2::verify_l2(&ir, &fixtures::base_caps()).unwrap();
    }

    #[test]
    fn distinct_literals_do_not_merge() {
        let mut b = TileBuilder::new();
        let _ = b.lit_f32(1.0);
        let _ = b.lit_f32(-1.0);
        let _ = b.lit_u32(1);
        assert_eq!(b.interned_len(), 3);
    }

    #[test]
    fn declarations_are_not_interned() {
        let mut b = TileBuilder::new();
        let layout = TileLayout::contiguous(
            fusor2_ir::ir::level2::MemoryLevel::Workgroup,
            &[8, 8],
        );
        let a = b.alloc_tile(ScalarElement::F32.element(), layout.clone());
        let c = b.alloc_tile(ScalarElement::F32.element(), layout);
        assert!(!Arc::ptr_eq(&a, &c));
        assert_eq!(b.tiles().len(), 2);
    }

    #[test]
    fn bf16_is_a_full_scalar_row() {
        assert_eq!(ScalarElement::BF16.byte_size(), 2);
        assert_eq!(
            ScalarElement::BF16.element().workgroup_array_stride(),
            Some(2)
        );
        let mut b = TileBuilder::new();
        let z = b.zero(ScalarElement::BF16.element());
        assert_eq!(z.element(), ScalarElement::BF16.element());
    }
}
