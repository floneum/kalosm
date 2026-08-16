//! Launch node + `SchedPoint` -> `KernelIr`, one module per node family.
//!
//! Everything shared by the six family lowerings lives here: the grid fold,
//! the 2-D matrix flattening of an N-D strided operand, the hash-consing Kernel
//! term builder, and the [`Ctx`] that turns `Plan`-carried buffer layouts into
//! Kernel storage views.
//!
//! **Operand layouts are never re-derived.** Every layout comes from
//! `Plan::buffers[..].layout`, which the extractor established; a mismatch
//! is a broken plan ([`Error::Plan`]).

pub mod contract;
pub mod gather_scatter;
pub mod map_fold;
pub mod region;

use fusor2_ir::Result;
use fusor2_ir::device::{Caps, Limits};
use fusor2_ir::dtype::{Dtype, NumericContract, QLayout};
use fusor2_ir::egraph::Id;
use fusor2_ir::error::Error;
use fusor2_ir::ir::launch::{ContractSide, IndexSpace, Launch, Operand, SchedPoint};
use fusor2_ir::ir::kernel::{
    Addr, Buffer, BufferAccess, BufferDecl, Builtin, CoopMatrixRole, CoopSrc, ElementType, KernelIr,
    Local, LocalDecl, MemoryLevel, ReduceKind, ScalarElement, Source, Stmt, Tile, TileBinaryOp,
    TileCompareOp, TileDecl, TileExpr, TileExprKind, TileLayout, TileLiteral, TileReduceOp,
    TileUnaryOp,
};
use fusor2_ir::ir::{Node, Op};
use fusor2_ir::shape::{AxisGroup, Dim, Layout, MultiFlattenMap, SubAxis, SymId};
use fusor2_ir::target::LowerCtx;
use fusor2_cost::realize::distribute_workgroups;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use std::sync::Arc;

use crate::uniforms::UniformPack;

/// Binding index of the always-present uniform block.
pub const UNIFORM_BINDING: u32 = 0;

/// One staged input of a contraction side: a memory source, or a `Const`
/// leaf already folded to its literal.
#[derive(Clone)]
pub enum StagedSource {
    Mem(Source),
    Const(TileExpr),
}

pub fn bound_layout(cx: &LowerCtx<'_>, value: Id) -> (Layout, Dtype) {
    let value = cx.selected(value);
    match cx.plan.buffers.iter().find(|b| b.value == value) {
        Some(b) => (b.layout.clone(), b.dtype),
        None => {
            let facts = cx.graph.facts(value);
            (Layout::contiguous(&facts.shape), facts.dtype)
        }
    }
}

/// The step-invariant decl extent: constants multiply, symbolic dims count
/// as 1. Storage globals are runtime-sized arrays, in-range masks are built
/// from the plan layout's `Dim`s, and the emitter's clamp reads
/// `arrayLength`, so nothing consumes this number for a symbolic buffer —
/// and resolving it would bake the sequence length into the kernel's
/// identity. An *unmasked* load through a symbolic view still fails
/// `verify_kernel` loudly, as it must.
fn decl_elements(layout: &Layout) -> u64 {
    // Padding lives in the strides: the extent of the plan's row-major
    // layouts is `shape[0] * strides[0]`, and the shape product undercounts
    // a padded buffer. A non-const stride slot 0 is the `row_major_strides`
    // placeholder, which implies no padding — the product of the remaining
    // extents is exactly what it derives to.
    let (Some(first), Some(stride0)) = (layout.shape().first(), layout.strides().first()) else {
        return 1;
    };
    let outer = first.as_const().unwrap_or(1);
    let stride0 = stride0.as_const().unwrap_or_else(|| {
        layout.shape()[1..]
            .iter()
            .map(|d| d.as_const().unwrap_or(1))
            .product()
    });
    outer.saturating_mul(stride0).max(1)
}

/// Runtime extents for the plan's symbols. A plan is compiled once for a whole
/// shape family, so the *grid* reads this and the *kernel body* reads binding 0
/// — never the other way round.
///
/// Every read is recorded: the set of symbols a lowering consulted is exactly
/// the set whose values its `KernelIr` (grid included) can depend on, so the
/// artifact cache keys a built kernel on those values alone. A kernel that
/// never reads a symbol is shared across every binding, which is what makes
/// a decode step's length change recompile nothing.
#[derive(Clone, Debug, Default)]
pub struct DimBinding {
    values: FxHashMap<SymId, u64>,
    consulted: std::sync::Arc<parking_lot::Mutex<rustc_hash::FxHashSet<SymId>>>,
    /// Symbols read *only* to fold the dispatch grid, and the
    /// `(space, block)` pairs those reads served. The grid is not the body:
    /// a symbol that moved only the workgroup count leaves the emitted
    /// module byte-identical, so it must not force a rebuild. Recording the
    /// derivation lets the artifact cache recompute the grid at the new
    /// binding instead — see [`DimBinding::grid_derivation`].
    grid: std::sync::Arc<parking_lot::Mutex<GridReads>>,
}

/// What a lowering read to fold its dispatch grid.
#[derive(Clone, Debug, Default)]
struct GridReads {
    symbols: rustc_hash::FxHashSet<SymId>,
    /// Every distinct `(space, block)` [`grid_for`] was called with. More
    /// than one and the lowering's committed grid is ambiguous from here, so
    /// nothing is replayable and the reads fall back to `consulted`.
    specs: Vec<GridSpec>,
}

/// The index space and workgroup width one [`grid_for`] call folded.
///
/// This is the whole of a dispatch grid's dependence on the binding: replaying
/// it at another binding is exactly what re-lowering would have computed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GridSpec {
    pub space: IndexSpace,
    pub block: u32,
}

impl DimBinding {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_pairs(pairs: impl IntoIterator<Item = (SymId, u64)>) -> Self {
        Self {
            values: pairs.into_iter().collect(),
            grid: Default::default(),
            consulted: Default::default(),
        }
    }

    pub fn bind(&mut self, sym: SymId, value: u64) {
        self.values.insert(sym, value);
    }

    pub fn get(&self, sym: SymId) -> Option<u64> {
        let hit = self.values.get(&sym).copied();
        if hit.is_some() {
            self.consulted.lock().insert(sym);
        }
        hit
    }

    /// Every symbol whose value has been handed out through this binding (or
    /// any clone of it), sorted. Shared across clones deliberately:
    /// `lower_node` takes the binding by value and the caller keeps a clone.
    pub fn consulted(&self) -> Vec<SymId> {
        self.body_consulted(false)
    }

    pub fn values(&self) -> &FxHashMap<SymId, u64> {
        &self.values
    }

    /// Concrete extent of a dim, or `None` when the symbol is unbound.
    pub fn resolve(&self, dim: Dim) -> Option<u64> {
        match dim {
            Dim::Const(v) => Some(v),
            Dim::Sym(s) => self.get(s),
        }
    }

    /// Concrete extent, or `Error::Plan`. Grid computation cannot proceed on
    /// an unbound symbol and must not guess one.
    pub fn require(&self, dim: Dim) -> Result<u64> {
        self.resolve(dim)
            .ok_or_else(|| Error::Plan(format!("dim {dim} is unbound at dispatch")))
    }

    /// Concrete extent for a *grid* fold. The read lands in the grid record,
    /// not in `consulted`: it cannot reach the emitted module, only the
    /// workgroup count.
    fn require_for_grid(&self, dim: Dim) -> Result<u64> {
        let value = match dim {
            Dim::Const(v) => Some(v),
            Dim::Sym(s) => {
                let hit = self.values.get(&s).copied();
                if hit.is_some() {
                    self.grid.lock().symbols.insert(s);
                }
                hit
            }
        };
        value.ok_or_else(|| Error::Plan(format!("dim {dim} is unbound at dispatch")))
    }

    /// The one grid derivation this lowering committed to, when it has one:
    /// a single [`grid_for`] call whose replay reproduces `grid`.
    ///
    /// `None` — several distinct folds, none at all, or a fold that does not
    /// reproduce the grid the lowering finished with — means the grid is not
    /// replayable from here, and [`Self::consulted`] then reports the grid's
    /// symbols too so the cache keys on them.
    pub fn grid_derivation(&self, grid: [u32; 3], limits: &Limits) -> Option<GridSpec> {
        let spec = {
            let g = self.grid.lock();
            let mut specs = g.specs.iter();
            let first = specs.next()?.clone();
            if specs.any(|s| *s != first) {
                return None;
            }
            first
        };
        (grid_from(&spec.space, spec.block, self, limits).ok()? == grid).then_some(spec)
    }

    /// Every symbol whose value the emitted module can depend on.
    ///
    /// Grid-only reads are excluded exactly when [`Self::grid_derivation`]
    /// yields a replay for them; otherwise they are folded back in, because a
    /// grid nobody can recompute must be rebuilt.
    pub fn body_consulted(&self, replayable: bool) -> Vec<SymId> {
        let mut out: rustc_hash::FxHashSet<SymId> = self.consulted.lock().clone();
        if !replayable {
            out.extend(self.grid.lock().symbols.iter().copied());
        }
        let mut out: Vec<SymId> = out.into_iter().collect();
        out.sort_unstable();
        out
    }
}

/// The dispatch grid for an index space at a given workgroup width.
pub fn grid_for(
    space: &IndexSpace,
    block: u32,
    binding: &DimBinding,
    limits: &Limits,
) -> Result<[u32; 3]> {
    binding.grid.lock().specs.push(GridSpec {
        space: space.clone(),
        block,
    });
    grid_from(space, block, binding, limits)
}

/// Fold a grid without recording the fold. [`grid_for`] is this plus the
/// record the artifact cache replays; a caller that already *holds* a
/// [`GridSpec`] is evaluating that record, not making a new one.
pub fn grid_from(
    space: &IndexSpace,
    block: u32,
    binding: &DimBinding,
    limits: &Limits,
) -> Result<[u32; 3]> {
    let mut elements: u64 = 1;
    for dim in &space.dims {
        elements = elements
            .checked_mul(binding.require_for_grid(*dim)?)
            .ok_or_else(|| Error::Plan("index space overflows a u64".into()))?;
    }
    let block = u64::from(block.max(1));
    let groups = elements.div_ceil(block);
    let groups = u32::try_from(groups)
        .map_err(|_| Error::Plan(format!("{groups} workgroups exceeds a u32")))?;
    Ok(distribute_workgroups(
        groups,
        limits.max_compute_workgroups_per_dimension,
    ))
}

/// An N-D strided operand seen as a 2-D matrix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatrixView {
    pub rows: u32,
    pub cols: u32,
    pub offset: u32,
    pub layout: TileLayout,
}

/// Flatten a strided layout into a 2-D matrix view: `shape[..row_dims]`
/// flattens to rows, `shape[row_dims..]` to columns.
///
/// Sides whose dims merge affinely use a plain strided layout; anything else
/// (a conv im2col window, a non-affine batch prefix) becomes a
/// [`MultiFlattenMap`] whose sub-axes divmod the flat coordinate back apart
/// per load. Extent-1 axes are dropped from the decomposition, saving a
/// divmod per load.
///
/// The plan guarantees these strides, so a failure is [`Error::Plan`].
///
/// `row_dims` may be `0` or `rank`: a contraction whose `n` (or `k`) extent
/// is 1 has *no* axes on that side, and its operand is a one-column (or
/// one-row) matrix. An empty side contributes a single index of 0, so its
/// stride never enters an address.
pub fn flatten_matrix_layout_split(
    layout: &Layout,
    row_dims: usize,
    binding: &DimBinding,
) -> Result<MatrixView> {
    let rank = layout.rank();
    if row_dims > rank {
        return Err(Error::Plan(format!(
            "matrix split at {row_dims} is outside rank {rank}"
        )));
    }

    let mut shape = SmallVec::<[u64; 6]>::new();
    for d in layout.shape() {
        shape.push(binding.require(*d)?);
    }
    let mut strides = SmallVec::<[u64; 6]>::new();
    for (axis, s) in layout.strides().iter().enumerate() {
        // A `row_major_strides` placeholder means the plan carried a stride it
        // never derived. Recompute it from the (now concrete) shape rather
        // than emitting the placeholder.
        let v = match s {
            Dim::Sym(sym) if *sym == crate::uniforms::DERIVED_STRIDE => {
                shape.iter().skip(axis + 1).product::<u64>()
            }
            other => binding.require(*other)?,
        };
        strides.push(v);
    }

    let rows: u64 = shape[..row_dims].iter().product();
    let cols: u64 = shape[row_dims..].iter().product();
    let rows_u32 = u32::try_from(rows)
        .map_err(|_| Error::Plan(format!("{rows} rows exceeds a u32 coordinate")))?;
    let cols_u32 = u32::try_from(cols)
        .map_err(|_| Error::Plan(format!("{cols} cols exceeds a u32 coordinate")))?;
    let offset = u32::try_from(binding.require(layout.offset())?)
        .map_err(|_| Error::Plan("layout offset exceeds a u32".into()))?;

    let side_is_affine = |lo: usize, hi: usize| -> bool {
        (lo..hi)
            .zip(lo + 1..hi)
            .all(|(axis, next)| strides[axis] == strides[next].saturating_mul(shape[next]))
    };

    // An empty side is a single index of 0; stride 0 keeps it out of the
    // address rather than reaching past the end of `strides`.
    let innermost = |lo: usize, hi: usize| -> u64 {
        if lo == hi { 0 } else { strides[hi - 1] }
    };
    let tile_layout = if side_is_affine(0, row_dims) && side_is_affine(row_dims, rank) {
        let row_stride = u32::try_from(innermost(0, row_dims))
            .map_err(|_| Error::Plan("row stride exceeds a u32".into()))?;
        let col_stride = u32::try_from(innermost(row_dims, rank))
            .map_err(|_| Error::Plan("col stride exceeds a u32".into()))?;
        TileLayout {
            extents: smallvec::smallvec![rows_u32, cols_u32],
            indexing: MultiFlattenMap::affine(&[rows_u32, cols_u32], &[row_stride, col_stride]),
            level: MemoryLevel::Storage,
        }
    } else {
        let group = |lo: usize, hi: usize| -> Result<AxisGroup> {
            let mut sub_axes: SmallVec<[SubAxis; 2]> = SmallVec::new();
            for axis in lo..hi {
                // Extent-1 axes contribute nothing to the flat coordinate
                // decomposition; dropping them saves a divmod per load.
                if shape[axis] == 1 {
                    continue;
                }
                sub_axes.push(SubAxis {
                    extent: u32::try_from(shape[axis])
                        .map_err(|_| Error::Plan("sub-axis extent exceeds a u32".into()))?,
                    stride: u32::try_from(strides[axis])
                        .map_err(|_| Error::Plan("sub-axis stride exceeds a u32".into()))?,
                });
            }
            if sub_axes.is_empty() {
                sub_axes.push(SubAxis {
                    extent: 1,
                    stride: 0,
                });
            }
            Ok(AxisGroup { sub_axes })
        };
        TileLayout {
            extents: smallvec::smallvec![rows_u32, cols_u32],
            indexing: MultiFlattenMap {
                groups: smallvec::smallvec![group(0, row_dims)?, group(row_dims, rank)?],
            },
            level: MemoryLevel::Storage,
        }
    };

    Ok(MatrixView {
        rows: rows_u32,
        cols: cols_u32,
        offset,
        layout: tile_layout,
    })
}

/// The axis split that presents `layout` as exactly `rows` by `cols`
/// elements.
///
/// [`Launch::Contract`](fusor2_ir::ir::launch::Launch::Contract) records four
/// *extents* — `m`, `n`, `k`, `batch` — and not the label partition they came
/// from, so the number of trailing `k` (resp. `n`) axes is not on the node.
/// It is recoverable, because `canonical_for_mnk` admits only
/// `a = [batch.., m.., k..]` and `b = [batch.., k.., n..]`: the split is the
/// position whose prefix multiplies to `rows` and whose suffix multiplies to
/// `cols`. The longest qualifying prefix is taken, which pins the choice when
/// an extent-1 axis makes two positions equivalent.
pub fn matrix_split_for(
    layout: &Layout,
    binding: &DimBinding,
    rows: u64,
    cols: u64,
) -> Result<usize> {
    let rank = layout.rank();
    let mut extents = SmallVec::<[u64; 6]>::new();
    for d in layout.shape() {
        extents.push(binding.require(*d)?);
    }
    (0..=rank)
        .rev()
        .find(|split| {
            extents[..*split].iter().product::<u64>() == rows
                && extents[*split..].iter().product::<u64>() == cols
        })
        .ok_or_else(|| {
            Error::Plan(format!(
                "no axis split of {extents:?} yields {rows} rows by {cols} columns"
            ))
        })
}

/// `u32` words a block-quantized value of `elements` elements occupies.
pub fn quantized_words(fmt: fusor2_ir::dtype::QFmt, layout: QLayout, elements: u64) -> u64 {
    let blocks = elements.div_ceil(u64::from(fmt.block_elements()).max(1));
    (blocks * u64::from(fmt.block_bytes(layout))).div_ceil(4)
}

/// The storage layout a quantized value carries, read off its `LeafKind`.
/// Layout is a priced operand attribute, never a device branch, so it is
/// recovered from the leaf rather than assumed.
pub fn qlayout_of(cx: &LowerCtx<'_>, value: Id) -> Option<QLayout> {
    let class = cx.graph.class_of(value);
    cx.graph
        .class_ids(class)
        .into_iter()
        .find_map(|m| match &cx.graph.node(m).op {
            fusor2_ir::ir::Op::Logical(fusor2_ir::ir::logical::Logical::Leaf(
                fusor2_ir::ir::logical::LeafKind::Quantized { layout, .. },
            )) => Some(*layout),
            _ => None,
        })
}

/// Logical/Launch dtype to Kernel element. Quantized weights bind as plain `u32` storage;
/// their decode is arithmetic over those words, never a buffer type.
pub const fn scalar_element(dtype: Dtype) -> ScalarElement {
    match dtype {
        Dtype::F32 => ScalarElement::F32,
        Dtype::F16 => ScalarElement::F16,
        Dtype::BF16 => ScalarElement::BF16,
        Dtype::I32 => ScalarElement::I32,
        Dtype::U32 | Dtype::Q(_) => ScalarElement::U32,
    }
}

/// Hash-consing Kernel term builder: two identical subtrees built separately
/// return the same `Arc`.
#[derive(Default)]
pub struct Kernel {
    memo: FxHashMap<u64, SmallVec<[TileExpr; 2]>>,
}

/// Clamp an infinite literal to the largest finite value WGSL can spell.
///
/// WGSL has no infinite literal, and naga rejects a module holding one; the
/// frontend hands one down on every causal or softmax path.
///
/// `-3.40282e38` is the same sentinel `emit::expr`'s reduce identities
/// already use, and `exp(x - m)` underflows to zero against it exactly as
/// it would against a real infinity.
pub fn finite_f32(v: f32) -> f32 {
    if v.is_infinite() {
        if v.is_sign_negative() {
            -crate::emit::expr::WGSL_SAFE_F32_MAX
        } else {
            crate::emit::expr::WGSL_SAFE_F32_MAX
        }
    } else if v.is_nan() {
        0.0
    } else {
        v
    }
}

/// [`finite_f32`] on the f16 bit pattern: 0x7C00 / 0xFC00 are the
/// infinities, 65504 the largest finite magnitude.
pub fn finite_f16(bits: u16) -> u16 {
    let v = half::f16::from_bits(bits);
    if v.is_infinite() || v.is_nan() {
        half::f16::from_f32(if v.is_sign_negative() && v.is_infinite() {
            -65504.0
        } else if v.is_infinite() {
            65504.0
        } else {
            0.0
        })
        .to_bits()
    } else {
        bits
    }
}

/// [`finite_f32`] on the bf16 bit pattern.
pub fn finite_bf16(bits: u16) -> u16 {
    let v = half::bf16::from_bits(bits);
    if v.is_nan() {
        return half::bf16::ZERO.to_bits();
    }
    if !v.is_infinite() {
        return bits;
    }
    // `from_f32(-3.40282e38)` rounds *back* to -inf in bf16, so take bf16's
    // own finite extreme rather than round-tripping the f32 sentinel.
    if v.is_sign_negative() {
        half::bf16::MIN.to_bits()
    } else {
        half::bf16::MAX.to_bits()
    }
}

impl Kernel {
    pub fn new() -> Self {
        Self::default()
    }

    fn intern(&mut self, kind: TileExprKind, ty: ElementType) -> TileExpr {
        let expr = TileExpr::new(kind, ty);
        let bucket = self.memo.entry(expr.structural_hash()).or_default();
        if let Some(hit) = bucket.iter().find(|e| **e == expr) {
            return hit.clone();
        }
        bucket.push(expr.clone());
        expr
    }

    pub fn lit(&mut self, value: TileLiteral) -> TileExpr {
        let ty = match value {
            TileLiteral::F32(_) => ScalarElement::F32,
            TileLiteral::F16(_) => ScalarElement::F16,
            TileLiteral::BF16(_) => ScalarElement::BF16,
            TileLiteral::U32(_) => ScalarElement::U32,
            TileLiteral::I32(_) => ScalarElement::I32,
            TileLiteral::Bool(_) => ScalarElement::Bool,
        };
        self.intern(TileExprKind::Literal(value), ty.element())
    }

    pub fn f32(&mut self, v: f32) -> TileExpr {
        self.lit(TileLiteral::F32(v.to_bits()))
    }
    pub fn u32(&mut self, v: u32) -> TileExpr {
        self.lit(TileLiteral::U32(v))
    }
    pub fn i32(&mut self, v: i32) -> TileExpr {
        self.lit(TileLiteral::I32(v))
    }
    pub fn bool(&mut self, v: bool) -> TileExpr {
        self.lit(TileLiteral::Bool(v))
    }
    /// The zero of an element type, used as a load fill and an accumulator
    /// init.
    pub fn zero(&mut self, elem: ScalarElement) -> TileExpr {
        match elem {
            ScalarElement::F32 => self.f32(0.0),
            ScalarElement::F16 => self.lit(TileLiteral::F16(0)),
            ScalarElement::BF16 => self.lit(TileLiteral::BF16(0)),
            ScalarElement::U32 => self.u32(0),
            ScalarElement::I32 => self.i32(0),
            ScalarElement::Bool => self.bool(false),
        }
    }
    /// The "smaller than anything real" sentinel a max carrier starts from.
    ///
    /// Finite, not `-inf`: WGSL has no infinite literal and naga rejects a
    /// module holding one. Same values as `emit::expr`'s reduce identities,
    /// so a max started here and a max started by a `Reduce` agree bit for
    /// bit, and `exp(x - m)` underflows to zero exactly as it would against
    /// a real infinity.
    pub fn neg_inf(&mut self, elem: ScalarElement) -> TileExpr {
        match elem {
            ScalarElement::F16 => {
                self.lit(TileLiteral::F16(half::f16::from_f32(-65504.0).to_bits()))
            }
            // bf16 rounds the f32 sentinel straight back to -inf, so take its
            // own finite extreme.
            ScalarElement::BF16 => self.lit(TileLiteral::BF16(half::bf16::MIN.to_bits())),
            _ => self.f32(-crate::emit::expr::WGSL_SAFE_F32_MAX),
        }
    }

    /// The `Min` identity, the mirror of [`Kernel::neg_inf`]: the largest
    /// finite magnitude, matching `emit::expr`'s reduce identities bit for
    /// bit.
    pub fn pos_inf(&mut self, elem: ScalarElement) -> TileExpr {
        match elem {
            ScalarElement::F16 => {
                self.lit(TileLiteral::F16(half::f16::from_f32(65504.0).to_bits()))
            }
            ScalarElement::BF16 => self.lit(TileLiteral::BF16(half::bf16::MAX.to_bits())),
            _ => self.f32(crate::emit::expr::WGSL_SAFE_F32_MAX),
        }
    }

    pub fn builtin(&mut self, b: Builtin) -> TileExpr {
        self.intern(
            TileExprKind::Builtin(b),
            ElementType::Scalar(ScalarElement::U32),
        )
    }

    pub fn load_local(&mut self, local: Local) -> TileExpr {
        let ty = local.element;
        self.intern(TileExprKind::LoadLocal(local), ty)
    }

    pub fn load(&mut self, src: Source, addr: Addr, mask: TileExpr, fill: TileExpr) -> TileExpr {
        let ty = match &src {
            Source::Storage(v) => v.buffer.element,
            // A quantized load decodes to f32 before it is ever a value.
            Source::Quantized(_) => ElementType::Scalar(ScalarElement::F32),
        };
        self.intern(
            TileExprKind::Load {
                src,
                addr: Box::new(addr),
                mask,
                fill,
            },
            ty,
        )
    }

    pub fn load_tile(&mut self, tile: Tile, index: TileExpr) -> TileExpr {
        let ty = tile.element;
        self.intern(TileExprKind::LoadTile { tile, index }, ty)
    }

    pub fn unary(&mut self, op: TileUnaryOp, value: TileExpr, numeric: NumericContract) -> TileExpr {
        let ty = if op == TileUnaryOp::Unpack2x16Float {
            ElementType::Vector {
                scalar: ScalarElement::F32,
                lanes: 2,
            }
        } else {
            value.element()
        };
        self.intern(
            TileExprKind::Unary {
                op,
                value,
                numeric,
            },
            ty,
        )
    }

    pub fn binary(
        &mut self,
        op: TileBinaryOp,
        left: TileExpr,
        right: TileExpr,
        numeric: NumericContract,
    ) -> TileExpr {
        let ty = left.element();
        self.intern(
            TileExprKind::Binary {
                op,
                left,
                right,
                numeric,
            },
            ty,
        )
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

    pub fn compare(&mut self, op: TileCompareOp, left: TileExpr, right: TileExpr) -> TileExpr {
        self.intern(
            TileExprKind::Compare { op, left, right },
            ElementType::Scalar(ScalarElement::Bool),
        )
    }

    pub fn and(&mut self, a: TileExpr, b: TileExpr) -> TileExpr {
        self.intern(
            TileExprKind::Binary {
                op: TileBinaryOp::LogicalAnd,
                left: a,
                right: b,
                numeric: NumericContract::RELAXED,
            },
            ElementType::Scalar(ScalarElement::Bool),
        )
    }

    pub fn cast(&mut self, value: TileExpr, to: ElementType) -> TileExpr {
        if value.element() == to {
            return value;
        }
        self.intern(TileExprKind::Cast { value, to }, to)
    }

    pub fn bitcast(&mut self, value: TileExpr, to: ElementType) -> TileExpr {
        self.intern(TileExprKind::Bitcast { value, to }, to)
    }

    pub fn select(&mut self, condition: TileExpr, accept: TileExpr, reject: TileExpr) -> TileExpr {
        let ty = accept.element();
        self.intern(
            TileExprKind::Select {
                condition,
                accept,
                reject,
            },
            ty,
        )
    }

    pub fn vector(&mut self, scalar: ScalarElement, parts: Vec<TileExpr>) -> TileExpr {
        let lanes = parts.len() as u32;
        self.intern(
            TileExprKind::Vec {
                scalar,
                lanes,
                parts,
            },
            ElementType::Vector { scalar, lanes },
        )
    }

    pub fn component(&mut self, vector: TileExpr, component: u32) -> TileExpr {
        let ty = match vector.element() {
            ElementType::Vector { scalar, .. } => ElementType::Scalar(scalar),
            other => other,
        };
        self.intern(
            TileExprKind::VecComponent { vector, component },
            ty,
        )
    }

    pub fn dot(&mut self, left: TileExpr, right: TileExpr) -> TileExpr {
        let ty = match left.element() {
            ElementType::Vector { scalar, .. } => ElementType::Scalar(scalar),
            other => other,
        };
        self.intern(TileExprKind::Dot { left, right }, ty)
    }

    pub fn round(&mut self, mode: fusor2_ir::dtype::RoundMode, value: TileExpr) -> TileExpr {
        let ty = value.element();
        self.intern(TileExprKind::Round { mode, value }, ty)
    }

    pub fn reduce(&mut self, op: TileReduceOp, kind: ReduceKind, value: TileExpr) -> TileExpr {
        let ty = value.element();
        self.intern(
            TileExprKind::Reduce {
                op,
                kind: Box::new(kind),
                value,
            },
            ty,
        )
    }

    pub fn coop_load(
        &mut self,
        role: CoopMatrixRole,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
        src: CoopSrc,
    ) -> TileExpr {
        self.intern(
            TileExprKind::CoopLoad {
                role,
                scalar,
                rows,
                cols,
                src: Box::new(src),
            },
            ElementType::CoopMatrix {
                scalar,
                role,
                rows,
                cols,
            },
        )
    }

    /// An all-zero fragment of the same shape as a cooperative accumulator.
    pub fn coop_zero(
        &mut self,
        role: CoopMatrixRole,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
    ) -> TileExpr {
        self.intern(
            TileExprKind::CoopZero {
                role,
                scalar,
                rows,
                cols,
            },
            ElementType::CoopMatrix {
                scalar,
                role,
                rows,
                cols,
            },
        )
    }

    pub fn coop_mma(&mut self, a: TileExpr, b: TileExpr, c: TileExpr) -> TileExpr {
        let ty = c.element();
        self.intern(TileExprKind::CoopMma { a, b, c }, ty)
    }

    /// A private per-invocation local. Locals are identity-bearing, so they
    /// are deliberately *not* interned.
    pub fn local(&self, element: ElementType) -> Local {
        Arc::new(LocalDecl::new(element))
    }

    /// A workgroup tile. Also identity-bearing: two tiles with the same shape
    /// are two allocations the arena may or may not overlap.
    pub fn tile(&self, name: &'static str, element: ElementType, extents: &[u32]) -> Tile {
        Arc::new(TileDecl::new(
            element,
            TileLayout::contiguous(MemoryLevel::Workgroup, extents),
            name,
        ))
    }
}

/// Per-kernel lowering state: the buffer table in binding order, the uniform
/// word layout, and the Kernel builder.
pub struct Ctx<'a> {
    pub caps: &'a Caps,
    pub cx: &'a LowerCtx<'a>,
    pub b: Kernel,
    pub binding: DimBinding,
    /// Binding order. Index 0 is always the uniform block.
    pub buffers: Vec<Buffer>,
    /// `Plan` value -> index into [`Self::buffers`].
    slot_of: FxHashMap<Id, usize>,
    pack: std::sync::Arc<UniformPack>,
}

impl<'a> Ctx<'a> {
    /// Build the buffer table for one launch.
    ///
    /// Binding 0 is the uniform block; every plan binding follows in
    /// `BindingPlan::binding` order at `1 + position`. Reserving slot 0 is
    /// what lets a kernel read a symbolic extent without the extent entering
    /// its identity.
    pub fn new(caps: &'a Caps, cx: &'a LowerCtx<'a>, binding: DimBinding) -> Result<Self> {
        Self::with_pack(
            caps,
            cx,
            binding,
            std::sync::Arc::new(UniformPack::new(cx.plan)),
        )
    }

    /// [`Self::new`] with the plan's binding-0 word layout supplied.
    ///
    /// The pack is a function of the plan alone, so a caller that lowers more
    /// than one launch of one plan derives it once and hands it down.
    pub fn with_pack(
        caps: &'a Caps,
        cx: &'a LowerCtx<'a>,
        binding: DimBinding,
        pack: std::sync::Arc<UniformPack>,
    ) -> Result<Self> {
        // Deterministic decl numbering per kernel build: a relower of the
        // same launch mints the same ids, so the pipeline cache's body-hash
        // dedup actually hits.
        fusor2_ir::ir::kernel::reset_decl_ids();
        let uniform_words = (pack.byte_len() / 4).max(1) as u32;
        let mut buffers: Vec<Buffer> = vec![Arc::new(BufferDecl {
            binding: UNIFORM_BINDING,
            element: ElementType::Scalar(ScalarElement::U32),
            layout: TileLayout::contiguous(MemoryLevel::Storage, &[uniform_words]),
            access: BufferAccess::Read,
        })];

        let mut ordered: Vec<_> = cx.launch.bindings.iter().collect();
        ordered.sort_by_key(|b| b.binding);

        let mut slot_of = FxHashMap::default();
        for (position, plan_binding) in ordered.iter().enumerate() {
            let (layout, dtype) = bound_layout(cx, plan_binding.value);
            let elements = decl_elements(&layout);
            // A quantized buffer holds blocks, not elements: it binds as the
            // `u32` word stream the decode program addresses.
            let elements = match dtype {
                Dtype::Q(fmt) => {
                    let qlayout = qlayout_of(cx, plan_binding.value).unwrap_or(QLayout::Native);
                    quantized_words(fmt, qlayout, elements)
                }
                _ => elements,
            };
            let extent = u32::try_from(elements)
                .map_err(|_| Error::Plan("buffer element count exceeds a u32".into()))?;
            let access = match plan_binding.kind {
                fusor2_ir::extract::BindKind::Read => BufferAccess::Read,
                _ => BufferAccess::ReadWrite,
            };
            // Keyed by every id in the value's class, not only by the
            // selected one: an `Operand::src` names whichever id the rule
            // author wrote, and they all denote the same buffer. `class_ids`
            // includes the `Union` spine, which macro ops hand their callers.
            let class = cx.graph.class_of(plan_binding.value);
            for member in cx.graph.class_ids(class) {
                slot_of.insert(member, buffers.len());
            }
            buffers.push(Arc::new(BufferDecl {
                binding: 1 + position as u32,
                element: ElementType::Scalar(scalar_element(dtype)),
                layout: TileLayout::contiguous(MemoryLevel::Storage, &[extent.max(1)]),
                access,
            }));
        }

        Ok(Self {
            caps,
            cx,
            b: Kernel::new(),
            binding,
            buffers,
            slot_of,
            pack,
        })
    }

    /// The bound buffer for a plan value.
    pub fn buffer(&self, value: Id) -> Result<Buffer> {
        let slot = self
            .slot_of
            .get(&value)
            .ok_or_else(|| Error::Plan(format!("value {value} is not bound by this launch")))?;
        Ok(self.buffers[*slot].clone())
    }

    /// The `BufferPlan` layout for a value. **Never re-derived** where the
    /// plan has one — that is the padded stride set the extractor committed
    /// to. See [`bound_layout`] for the leaf case.
    pub fn plan_layout(&self, value: Id) -> Result<Layout> {
        Ok(bound_layout(self.cx, value).0)
    }

    pub fn plan_dtype(&self, value: Id) -> Result<Dtype> {
        Ok(bound_layout(self.cx, value).1)
    }

    /// A flat rank-1 view of a value's buffer, for elementwise access.
    pub fn linear_view(&self, value: Id) -> Result<fusor2_ir::ir::kernel::StorageView> {
        let buffer = self.buffer(value)?;
        let layout = buffer.layout.clone();
        Ok(fusor2_ir::ir::kernel::StorageView {
            buffer,
            offset: 0,
            layout,
        })
    }

    /// A 2-D matrix view of an operand, split at `row_dims`, built from the
    /// plan's layout.
    pub fn matrix_view(
        &self,
        operand: &Operand,
        row_dims: usize,
    ) -> Result<fusor2_ir::ir::kernel::StorageView> {
        let layout = self.repad_operand_layout(operand)?;
        let view = flatten_matrix_layout_split(&layout, row_dims, &self.binding)?;
        let buffer = self.buffer(operand.src)?;
        Ok(fusor2_ir::ir::kernel::StorageView {
            buffer,
            offset: view.offset,
            layout: view.layout,
        })
    }

    /// An operand's layout restated over the producer's *plan* buffer.
    ///
    /// The operand's strides address the producer's logical dense element
    /// space; the buffer holds whatever the plan laid out, and those differ
    /// exactly when the producer's schedule point padded it. This is
    /// [`Ctx::repad_index`]'s statement for the contraction path, which loads
    /// through strided views rather than a flat index: every operand axis
    /// must walk exactly one producer axis — its stride is that axis's dense
    /// row-major stride — and the restatement substitutes the padded stride
    /// for the dense one, axis for axis. A transposed or batch-permuted edge
    /// (`permuted_alias`, an absorbed producer) satisfies that by
    /// construction; an operand whose stride is no producer axis's own is an
    /// error, never a silent dense read.
    fn repad_operand_layout(&self, operand: &Operand) -> Result<Layout> {
        let selected = self.cx.selected(operand.src);
        let Some(plan) = self
            .cx
            .plan
            .buffers
            .iter()
            .find(|b| b.value == selected)
        else {
            return Ok(operand.layout.clone());
        };
        let logical = self.cx.graph.facts(selected).shape.clone();
        if plan.layout.rank() != logical.len() || logical.is_empty() {
            return Ok(operand.layout.clone());
        }
        let dense = Layout::row_major_strides(&logical);
        let unpadded = plan.layout.offset().known_eq(Dim::Const(0))
            && plan
                .layout
                .shape()
                .iter()
                .zip(&logical)
                .all(|(p, l)| p.known_eq(*l))
            && plan
                .layout
                .strides()
                .iter()
                .zip(&dense)
                .all(|(s, w)| s.known_eq(*w));
        if unpadded {
            return Ok(operand.layout.clone());
        }
        if !plan.layout.offset().known_eq(Dim::Const(0)) {
            return Err(Error::Plan(format!(
                "operand {} reads a buffer at offset {}; the contraction path \
                 cannot restate an offset layout",
                operand.src,
                plan.layout.offset()
            )));
        }
        // The operand may be a *reshaped* spelling of the producer — a
        // `[2, 2, 3, 4]` read of a `[4, 3, 4]` contract — so an operand axis
        // walks `k` steps of one producer axis rather than exactly one: its
        // stride is `k * dense[i]`, and it stays inside that axis
        // (`k * (ext - 1) < logical[i]`). Substituting `k * padded[i]`
        // restates it, because a within-axis walk scales linearly with the
        // axis's own stride whatever the padding did to the axes outside it.
        let padded = plan.layout.strides();
        let remap = |ext: Dim, s: Dim| -> Result<Dim> {
            // Unobservable axes keep whatever they said.
            if ext.known_eq(Dim::Const(1)) || s.known_eq(Dim::Const(0)) {
                return Ok(s);
            }
            let (Some(sv), Some(ev)) = (s.as_const(), ext.as_const()) else {
                return Err(Error::Plan(format!(
                    "operand {} reads a padded buffer through symbolic stride {s}",
                    operand.src
                )));
            };
            for (i, d) in dense.iter().enumerate() {
                let (Some(dv), Some(lv)) = (d.as_const(), logical[i].as_const()) else {
                    continue;
                };
                if dv == 0 || sv % dv != 0 {
                    continue;
                }
                let k = sv / dv;
                if k >= 1 && k.saturating_mul(ev - 1) < lv {
                    let pv = padded[i].as_const().ok_or_else(|| {
                        Error::Plan(format!(
                            "operand {} reads a buffer with symbolic padded stride",
                            operand.src
                        ))
                    })?;
                    return Ok(Dim::Const(k * pv));
                }
            }
            Err(Error::Plan(format!(
                "operand {} reads a padded buffer through stride {s}, which walks \
                 no single axis of the producer's dense layout {dense:?}",
                operand.src
            )))
        };
        let strides: Vec<Dim> = operand
            .layout
            .shape()
            .iter()
            .zip(operand.layout.strides())
            .map(|(ext, s)| remap(*ext, *s))
            .collect::<Result<_>>()?;
        Layout::from_parts(operand.layout.offset(), operand.layout.shape(), &strides)
    }

    /// The [`Source`] a contraction stages one operand from.
    ///
    /// Dense operands read storage. A block-quantized operand reads
    /// [`Source::Quantized`], whose decode program the Kernel emitter runs at the
    /// `(row, col)` the staging fill already computes — so a quantized weight
    /// costs the decode math on the way into shared memory and nothing else.
    /// The staging tile, the fragments, the MMA and the arena footprint are the
    /// dense ones.
    pub fn contract_stage_source(
        &mut self,
        operand: &Operand,
        view: &fusor2_ir::ir::kernel::StorageView,
    ) -> Result<Source> {
        let Dtype::Q(fmt) = self.plan_dtype(operand.src)? else {
            return Ok(Source::Storage(view.clone()));
        };
        let qlayout = qlayout_of(self.cx, operand.src).unwrap_or(QLayout::Native);
        Ok(Source::Quantized(fusor2_ir::ir::kernel::QuantizedView {
            data: view.clone(),
            fmt,
            layout: qlayout,
        }))
    }

    /// Every buffer one contraction side reads, as a staging source apiece.
    ///
    /// A side is a list because an absorbed producer brings its own edges —
    /// the GGUF block decode arrives with the quant plane, the block scale,
    /// the block minimum and the group scales, each a `Restride` of the same
    /// block stream at its own offset. They share the side's `(rows, cols)`
    /// index and differ only in strides, so each gets its own view and all of
    /// them are loaded at the same coordinate before the side's `pre` runs
    /// over the results.
    pub fn contract_side_sources(
        &mut self,
        side: &ContractSide,
        rows: u32,
        cols: u32,
    ) -> Result<Vec<StagedSource>> {
        side.ops
            .iter()
            .map(|o| {
                // A `Const` leaf is folded into the kernel — no buffer, no
                // binding — exactly as `load_operand` treats it. Absorbed
                // producers bring these: a layer norm's `1/N`, an epsilon.
                if let Some(lit) = self.const_operand(o.src) {
                    return Ok(StagedSource::Const(lit));
                }
                let view = self.contract_operand_view(o, rows, cols)?;
                Ok(StagedSource::Mem(self.contract_stage_source(o, &view)?))
            })
            .collect()
    }

    pub fn contract_operand_view(
        &self,
        operand: &Operand,
        rows: u32,
        cols: u32,
    ) -> Result<fusor2_ir::ir::kernel::StorageView> {
        let split =
            matrix_split_for(&operand.layout, &self.binding, u64::from(rows), u64::from(cols))?;
        self.matrix_view(operand, split)
    }

    /// Read a `u32` word out of binding 0.
    pub fn uniform_word(&mut self, slot: u32) -> TileExpr {
        let view = fusor2_ir::ir::kernel::StorageView {
            buffer: self.buffers[0].clone(),
            offset: 0,
            layout: self.buffers[0].layout.clone(),
        };
        let index = self.b.u32(slot);
        let mask = self.b.bool(true);
        let fill = self.b.u32(0);
        self.b
            .load(Source::Storage(view), Addr::Linear(index), mask, fill)
    }

    /// A `u32` expression for a dim: a literal when constant, a binding-0 word
    /// when symbolic. A sequence length is a word, never a baked constant.
    pub fn dim_expr(&mut self, dim: Dim) -> Result<TileExpr> {
        match dim {
            Dim::Const(v) => {
                let v = u32::try_from(v)
                    .map_err(|_| Error::Plan(format!("extent {v} exceeds a u32")))?;
                Ok(self.b.u32(v))
            }
            Dim::Sym(s) => {
                let slot = self
                    .pack
                    .dim_slot(s)
                    .ok_or_else(|| Error::Plan(format!("symbol {s} has no uniform slot")))?;
                Ok(self.uniform_word(slot))
            }
        }
    }

    /// An `f32` expression for a runtime scalar: `m * lr` reads a word, so a
    /// learning-rate change recompiles nothing.
    pub fn scalar_expr(&mut self, sym: SymId) -> Result<TileExpr> {
        let slot = self
            .pack
            .scalar_slot(sym)
            .ok_or_else(|| Error::Plan(format!("scalar {sym} has no uniform slot")))?;
        let word = self.uniform_word(slot);
        Ok(self
            .b
            .bitcast(word, ElementType::Scalar(ScalarElement::F32)))
    }

    /// The global linear element index this invocation owns, linearized
    /// against **this launch's** grid.
    ///
    /// `grid` must be the same `[x, y, z]` the kernel is dispatched with — the
    /// one handed to [`Ctx::finish`], not `max_compute_workgroups_per_dimension`:
    /// `distribute_workgroups` does not saturate `x` before opening a second
    /// slab. It picks the slab count first and
    /// sizes `x` to the slab, so 122,880 groups dispatch as `[61440, 2, 1]`.
    /// Reading `gy * 65535` off that grid would put every workgroup past the
    /// first slab at a wildly out-of-range index, mask itself out, and leave
    /// the tail of the output untouched — silently, for any launch over the
    /// per-dimension limit.
    pub fn global_index(&mut self, block: u32, grid: [u32; 3]) -> TileExpr {
        use fusor2_ir::ir::kernel::WorkgroupAxis;
        let lane = self.b.builtin(Builtin::Lane);
        let gx = self.b.builtin(Builtin::ProgramId(WorkgroupAxis::X));
        let gy = self.b.builtin(Builtin::ProgramId(WorkgroupAxis::Y));
        let gz = self.b.builtin(Builtin::ProgramId(WorkgroupAxis::Z));
        // group = gx + gy*X + gz*X*Y, exactly as the grid fold laid it out —
        // with X and Y read from `@builtin(num_workgroups)`, never baked, so
        // the extents never enter the body. `grid` still names the dispatch
        // this lowering derived.
        let _ = grid;
        let x_e = self.b.builtin(Builtin::NumWorkgroups(WorkgroupAxis::X));
        let y_e = self.b.builtin(Builtin::NumWorkgroups(WorkgroupAxis::Y));
        let xy_e = self.b.mul(x_e.clone(), y_e);
        let yx = self.b.mul(gy, x_e);
        let zxy = self.b.mul(gz, xy_e);
        let group = self.b.add(gx, yx);
        let group = self.b.add(group, zxy);
        let block_e = self.b.u32(block);
        let base = self.b.mul(group, block_e);
        self.b.add(base, lane)
    }

    /// The value this launch writes: the launch root when it is bound for
    /// writing, else the first writable binding.
    pub fn output(&self) -> Result<Id> {
        let root = self.cx.launch.root;
        if self
            .cx
            .launch
            .bindings
            .iter()
            .any(|b| b.value == root && b.kind != fusor2_ir::extract::BindKind::Read)
        {
            return Ok(root);
        }
        self.cx
            .launch
            .bindings
            .iter()
            .find(|b| b.kind != fusor2_ir::extract::BindKind::Read)
            .map(|b| b.value)
            .ok_or_else(|| Error::Plan("launch binds nothing writable".into()))
    }

    /// Per-axis coordinates of a flat index over `space`, most-significant
    /// axis first. One divmod per axis past the innermost, exactly as the
    /// index-op cost term prices.
    pub fn coords_from_linear(
        &mut self,
        linear: TileExpr,
        space: &IndexSpace,
    ) -> Result<Vec<TileExpr>> {
        let rank = space.rank();
        let mut coords = vec![linear.clone(); rank];
        let mut rest = linear;
        for axis in (0..rank).rev() {
            let extent = self.dim_expr(space.dims[axis])?;
            if axis == 0 {
                coords[0] = rest;
                break;
            }
            coords[axis] =
                self.b
                    .binary(TileBinaryOp::Rem, rest.clone(), extent.clone(), NumericContract::RELAXED);
            rest = self
                .b
                .binary(TileBinaryOp::Div, rest, extent, NumericContract::RELAXED);
        }
        Ok(coords)
    }

    /// Translate a [`fusor2_ir::scalar::ScalarExpr`] body into Kernel.
    ///
    /// `args` are the already-loaded operand values; `coords` are the index
    /// space coordinates `IndexOf` reads. Comparisons return 1.0/0.0 in the
    /// operand's own dtype, matching Logical semantics — Kernel's `Bool` exists only
    /// between the compare and the select.
    pub fn eval_scalar(
        &mut self,
        expr: &fusor2_ir::scalar::ScalarExpr,
        args: &[TileExpr],
        coords: &[TileExpr],
    ) -> Result<TileExpr> {
        use fusor2_ir::scalar::ScalarKind as K;
        let relaxed = NumericContract::RELAXED;
        Ok(match expr.kind() {
            K::Arg(i) => args
                .get(*i as usize)
                .cloned()
                .ok_or_else(|| Error::Plan(format!("body reads Arg({i}) with {} operands", args.len())))?,
            K::Lit(l) => match l.0 {
                fusor2_ir::dtype::Splat::F32(v) => {
                    let v = finite_f32(v);
                    self.b.f32(v)
                }
                fusor2_ir::dtype::Splat::F16(v) => self.b.lit(TileLiteral::F16(finite_f16(v))),
                fusor2_ir::dtype::Splat::BF16(v) => self.b.lit(TileLiteral::BF16(finite_bf16(v))),
                fusor2_ir::dtype::Splat::U32(v) => self.b.u32(v),
                fusor2_ir::dtype::Splat::I32(v) => self.b.i32(v),
            },
            K::Uniform(sym) => self.scalar_expr(*sym)?,
            K::IndexOf(axis) => {
                let c = coords.get(*axis as usize).cloned().ok_or_else(|| {
                    Error::Plan(format!("body reads IndexOf({axis}) outside the index space"))
                })?;
                self.b.cast(c, ElementType::Scalar(ScalarElement::U32))
            }
            K::Un { op, x } => {
                let v = self.eval_scalar(x, args, coords)?;
                self.b.unary(*op, v, relaxed)
            }
            K::Bin { op, a, b } => {
                let l = self.eval_scalar(a, args, coords)?;
                let r = self.eval_scalar(b, args, coords)?;
                self.b.binary(*op, l, r, relaxed)
            }
            K::Cmp { op, a, b } => {
                let l = self.eval_scalar(a, args, coords)?;
                let r = self.eval_scalar(b, args, coords)?;
                let elem = l.element();
                let c = self.b.compare(*op, l, r);
                let one = self.one_of(elem);
                let zero = self.zero_of(elem);
                self.b.select(c, one, zero)
            }
            K::Select { c, t, f } => {
                let cv = self.eval_scalar(c, args, coords)?;
                let tv = self.eval_scalar(t, args, coords)?;
                let fv = self.eval_scalar(f, args, coords)?;
                let elem = cv.element();
                let zero = self.zero_of(elem);
                let nonzero = self.b.compare(TileCompareOp::Ne, cv, zero);
                self.b.select(nonzero, tv, fv)
            }
            K::Cast { to, x } => {
                let v = self.eval_scalar(x, args, coords)?;
                self.b.cast(v, ElementType::Scalar(scalar_element(*to)))
            }
            K::Bitcast { to, x } => {
                let v = self.eval_scalar(x, args, coords)?;
                self.b.bitcast(v, ElementType::Scalar(scalar_element(*to)))
            }
            // `Round` is its own Kernel node, so there is no arithmetic
            // identity for Metal's default fast math to fold away and QAT
            // cannot be silently disabled.
            K::Round { mode, x } => {
                let v = self.eval_scalar(x, args, coords)?;
                self.b.round(*mode, v)
            }
            K::Dot { a, b } => {
                let l = self.eval_scalar(a, args, coords)?;
                let r = self.eval_scalar(b, args, coords)?;
                self.b.dot(l, r)
            }
            K::Splat { lanes, x } => {
                let v = self.eval_scalar(x, args, coords)?;
                let scalar = match v.element() {
                    ElementType::Scalar(s) => s,
                    ElementType::Vector { scalar, .. } => scalar,
                    ElementType::CoopMatrix { scalar, .. } => scalar,
                };
                self.b.vector(scalar, vec![v; *lanes as usize])
            }
        })
    }

    fn zero_of(&mut self, elem: ElementType) -> TileExpr {
        match elem {
            ElementType::Scalar(s) => self.b.zero(s),
            ElementType::Vector { scalar, lanes } => {
                let z = self.b.zero(scalar);
                self.b.vector(scalar, vec![z; lanes as usize])
            }
            ElementType::CoopMatrix { scalar, .. } => self.b.zero(scalar),
        }
    }

    fn one_of(&mut self, elem: ElementType) -> TileExpr {
        let scalar = match elem {
            ElementType::Scalar(s) => s,
            ElementType::Vector { scalar, .. } | ElementType::CoopMatrix { scalar, .. } => scalar,
        };
        match scalar {
            ScalarElement::F32 => self.b.f32(1.0),
            ScalarElement::F16 => self.b.lit(TileLiteral::F16(half::f16::ONE.to_bits())),
            ScalarElement::BF16 => self.b.lit(TileLiteral::BF16(half::bf16::ONE.to_bits())),
            ScalarElement::U32 => self.b.u32(1),
            ScalarElement::I32 => self.b.i32(1),
            ScalarElement::Bool => self.b.bool(true),
        }
    }

    /// Load one operand at the reading kernel's **flat space index**, running
    /// it through the edge's [`fusor2_ir::ir::launch::AddressMap`] first.
    ///
    /// [`Ctx::load_operand`] is the raw form, for readers that have already
    /// computed a storage index themselves (gather, scatter, the contraction
    /// nests). Everything whose index *is* the space coordinate must come
    /// through here: a stride-0 broadcast axis, a transposed view, a narrowed
    /// slice and a conv window all disagree with the bare flat index.
    pub fn load_mapped(
        &mut self,
        operand: &Operand,
        flat: TileExpr,
        space_total: u64,
    ) -> Result<TileExpr> {
        let addr = self.operand_address(operand, flat, space_total)?;
        self.load_operand(operand, addr)
    }

    /// `flat` run through one operand's index map.
    pub fn operand_address(
        &mut self,
        operand: &Operand,
        flat: TileExpr,
        space_total: u64,
    ) -> Result<TileExpr> {
        let Some(map) = operand.address_map() else {
            // A symbolic extent (or a stride past one) has no compile-time
            // `AddressMap`; the address is computed with binding-0 words
            // instead of literals, so a length change recompiles nothing.
            return self.symbolic_operand_address(operand, flat);
        };
        if map.is_identity_over(space_total) {
            return Ok(flat);
        }
        let mut acc: Option<TileExpr> = if map.offset != 0 {
            Some(self.b.u32(map.offset))
        } else {
            None
        };
        for (i, t) in map.terms.iter().enumerate() {
            let mut e = flat.clone();
            if t.divisor > 1 {
                let d = self.b.u32(t.divisor);
                e = self
                    .b
                    .binary(TileBinaryOp::Div, e, d, NumericContract::RELAXED);
            }
            if map.needs_modulo(i, space_total) {
                let m = self.b.u32(t.modulus);
                e = self
                    .b
                    .binary(TileBinaryOp::Rem, e, m, NumericContract::RELAXED);
            }
            if t.stride != 1 {
                let s = self.b.u32(t.stride);
                e = self.b.mul(e, s);
            }
            acc = Some(match acc {
                Some(a) => self.b.add(a, e),
                None => e,
            });
        }
        Ok(match acc {
            Some(a) => a,
            None => self.b.u32(0),
        })
    }

    /// [`Ctx::operand_address`] for a layout no compile-time [`AddressMap`]
    /// can express: at least one extent (or a stride past one) is symbolic.
    ///
    /// Emits `offset + Σ_axis ((flat / Π extents-right-of-axis) % extent) *
    /// stride` with every symbolic quantity read from binding 0 via
    /// [`Ctx::dim_expr`]. The `row_major_strides` placeholder
    /// (`DERIVED_STRIDE`) is the running right-product itself, which the walk
    /// already carries. Axes with stride 0 (broadcast) or extent 1 contribute
    /// no term but still advance the divisor. The most significant axis skips
    /// its `%`: `flat` is masked below the space total by the caller, so the
    /// quotient is already in range.
    fn symbolic_operand_address(
        &mut self,
        operand: &Operand,
        flat: TileExpr,
    ) -> Result<TileExpr> {
        if matches!(operand.access, fusor2_ir::ir::launch::AccessPlan::Unflatten(_)) {
            return Err(Error::Plan(format!(
                "a symbolic Unflatten window is not lowerable; operand {} laid out {:?}",
                operand.src, operand.layout
            )));
        }
        let layout = operand.layout.clone();
        // A contiguous offset-0 layout is the identity over its own space —
        // the dense read every elementwise kernel does.
        if layout.is_contiguous() && layout.offset().known_eq(Dim::Const(0)) {
            return Ok(flat);
        }
        let shape: Vec<Dim> = layout.shape().to_vec();
        let strides: Vec<Dim> = layout.strides().to_vec();
        let mut acc: Option<TileExpr> = match layout.offset() {
            Dim::Const(0) => None,
            d => Some(self.dim_expr(d)?),
        };
        // Product of extents right of the current axis, as an expression;
        // `None` is 1.
        let mut div: Option<TileExpr> = None;
        for axis in (0..shape.len()).rev() {
            let extent = shape[axis];
            let stride = strides[axis];
            let contributes =
                !stride.known_eq(Dim::Const(0)) && !extent.known_eq(Dim::Const(1));
            if contributes {
                let mut e = flat.clone();
                if let Some(d) = &div {
                    e = self
                        .b
                        .binary(TileBinaryOp::Div, e, d.clone(), NumericContract::RELAXED);
                }
                if axis != 0 {
                    let m = self.dim_expr(extent)?;
                    e = self
                        .b
                        .binary(TileBinaryOp::Rem, e, m, NumericContract::RELAXED);
                }
                let is_derived =
                    matches!(stride, Dim::Sym(s) if s == crate::uniforms::DERIVED_STRIDE);
                if is_derived {
                    // Row-major placeholder: stride == the running product.
                    if let Some(d) = &div {
                        e = self.b.mul(e, d.clone());
                    }
                } else if !stride.known_eq(Dim::Const(1)) {
                    let s = self.dim_expr(stride)?;
                    e = self.b.mul(e, s);
                }
                acc = Some(match acc {
                    Some(a) => self.b.add(a, e),
                    None => e,
                });
            }
            if !extent.known_eq(Dim::Const(1)) {
                let m = self.dim_expr(extent)?;
                div = Some(match div {
                    Some(d) => self.b.mul(d, m),
                    None => m,
                });
            }
        }
        Ok(match acc {
            Some(a) => a,
            None => self.b.u32(0),
        })
    }

    /// Re-address a **logical** dense element index of `src` into the buffer
    /// the plan actually laid out for it.
    ///
    /// `Plan::buffers` is authoritative about storage, and
    /// `fusor2_cost::plan::buffer_layout_for` pads a `Coop` contraction's
    /// output to whole `bm x bn` blocks, while every other reader of that
    /// value names its elements densely over the logical shape. Without this
    /// step a `[16, 1]` contraction padded to `[16, 16]` is read as the first
    /// sixteen elements of row 0.
    ///
    /// Identity — and emitted as nothing — whenever the plan's layout is the
    /// logical dense one, which is every value the extractor did not pad.
    fn repad_index(&mut self, src: Id, index: TileExpr) -> Result<TileExpr> {
        let selected = self.cx.selected(src);
        let Some(plan) = self
            .cx
            .plan
            .buffers
            .iter()
            .find(|b| b.value == selected)
            .cloned()
        else {
            return Ok(index);
        };
        let logical = self.cx.graph.facts(selected).shape.clone();
        if plan.layout.rank() != logical.len() || logical.is_empty() {
            return Ok(index);
        }
        let strides = plan.layout.strides().to_vec();
        let shape = plan.layout.shape().to_vec();
        let dense = Layout::row_major_strides(&logical);
        let unpadded = plan.layout.offset().known_eq(Dim::Const(0))
            && shape.iter().zip(&logical).all(|(p, l)| p.known_eq(*l))
            && strides.iter().zip(&dense).all(|(s, w)| s.known_eq(*w));
        if unpadded {
            return Ok(index);
        }
        // Every extent has to be decidable to state the delinearize; when one
        // is not, the previous dense address is still what the rest of the
        // launch agreed on, so leave it alone rather than mint a wrong one.
        let Ok(extents) = logical
            .iter()
            .map(|d| self.binding.require(*d))
            .collect::<Result<Vec<u64>>>()
        else {
            return Ok(index);
        };
        let Ok(logical_strides) = dense
            .iter()
            .map(|d| self.binding.require(*d))
            .collect::<Result<Vec<u64>>>()
        else {
            return Ok(index);
        };
        let Ok(padded_strides) = strides
            .iter()
            .map(|d| self.binding.require(*d))
            .collect::<Result<Vec<u64>>>()
        else {
            return Ok(index);
        };
        let offset = plan.layout.offset().as_const().unwrap_or(0);

        let mut acc: Option<TileExpr> = (offset != 0).then(|| {
            let o = u32::try_from(offset).unwrap_or(u32::MAX);
            self.b.u32(o)
        });
        for axis in 0..logical.len() {
            let extent = extents[axis];
            let stride = padded_strides[axis];
            if extent <= 1 || stride == 0 {
                continue;
            }
            let mut e = index.clone();
            let div = logical_strides[axis];
            if div > 1 {
                let d = self.b.u32(u32::try_from(div).unwrap_or(u32::MAX));
                e = self
                    .b
                    .binary(TileBinaryOp::Div, e, d, NumericContract::RELAXED);
            }
            // The most significant axis needs no `%`: `flat` is already below
            // its bound for every live lane, and an overhang lane is masked.
            if axis > 0 {
                let m = self.b.u32(u32::try_from(extent).unwrap_or(u32::MAX));
                e = self
                    .b
                    .binary(TileBinaryOp::Rem, e, m, NumericContract::RELAXED);
            }
            if stride != 1 {
                let s = self.b.u32(u32::try_from(stride).unwrap_or(u32::MAX));
                e = self.b.mul(e, s);
            }
            acc = Some(match acc {
                Some(a) => self.b.add(a, e),
                None => e,
            });
        }
        Ok(match acc {
            Some(a) => a,
            None => self.b.u32(0),
        })
    }

    /// Load one operand at an already-computed **storage** element index. The
    /// mask is the plan's runtime bounds obligation; a load is never emitted
    /// unmasked unless the extent is a compile-time multiple of the block.
    pub fn load_operand(&mut self, operand: &Operand, index: TileExpr) -> Result<TileExpr> {
        // A `Leaf::Const` is folded into the kernel: no buffer, no binding,
        // no traffic. That is exactly what `LeafRole::Free` means in the
        // plan, so `derive_bindings` never emits one and loading it would
        // look up a binding that deliberately does not exist.
        if let Some(lit) = self.const_operand(operand.src) {
            return Ok(lit);
        }
        // An `Operand`'s index arithmetic is stated over the producer's
        // logical dense element space; the buffer it lands in is whatever
        // the plan laid out. Those differ exactly when the producer's
        // schedule point padded it.
        let index = self.repad_index(operand.src, index)?;
        let buffer = self.buffer(operand.src)?;
        // A block-quantized operand has no dense element to load: reading
        // element `i` runs the format's decode program at flat index `i`.
        // The dense table is never materialized.
        if let Dtype::Q(fmt) = self.plan_dtype(operand.src)? {
            let qlayout = qlayout_of(self.cx, operand.src).unwrap_or(QLayout::Native);
            let facts = self.cx.graph.facts(self.cx.selected(operand.src));
            let cols = facts
                .shape
                .last()
                .map(|d| self.binding.require(*d))
                .transpose()?
                .unwrap_or(0);
            let mut rows: u64 = 1;
            for d in &facts.shape[..facts.shape.len().saturating_sub(1)] {
                rows = rows.saturating_mul(self.binding.require(*d)?);
            }
            let bound = self.b.u32(u32::try_from(rows.saturating_mul(cols)).unwrap_or(u32::MAX));
            let mask = self.b.compare(TileCompareOp::Lt, index.clone(), bound);
            let fill = self.b.f32(0.0);
            let layout = buffer.layout.clone();
            let view = fusor2_ir::ir::kernel::QuantizedView {
                data: fusor2_ir::ir::kernel::StorageView {
                    buffer,
                    offset: 0,
                    layout,
                },
                fmt,
                layout: qlayout,
            };
            return Ok(self
                .b
                .load(Source::Quantized(view), Addr::Linear(index), mask, fill));
        }
        let elem = buffer.element;
        // `index` is a storage element index, so the bound is the buffer's
        // own extent, built from the plan layout's `Dim`s — never from the
        // resolved decl extents, which would bake this dispatch's sequence
        // length into the body.
        let layout = buffer.layout.clone();
        let view = fusor2_ir::ir::kernel::StorageView {
            buffer,
            offset: 0,
            layout,
        };
        // The buffer's extent is not the shape product: padding lives in the
        // strides, so the shape product undercounts a padded buffer. For the
        // row-major layouts the plan emits (offset 0), the extent is
        // `shape[0] * strides[0]`; a `DERIVED_STRIDE` placeholder implies no
        // padding and resolves as the product of the remaining logical
        // extents.
        let (plan_layout, _) = bound_layout(self.cx, operand.src);
        let bound = match (plan_layout.shape().first(), plan_layout.strides().first()) {
            (Some(&outer), Some(&stride0)) => {
                let outer_e = if outer.known_eq(Dim::Const(1)) {
                    None
                } else {
                    Some(self.dim_expr(outer)?)
                };
                let stride_e = match stride0 {
                    Dim::Sym(s) if s == crate::uniforms::DERIVED_STRIDE => {
                        let mut acc: Option<TileExpr> = None;
                        for d in plan_layout.shape().iter().skip(1).copied() {
                            if d.known_eq(Dim::Const(1)) {
                                continue;
                            }
                            let e = self.dim_expr(d)?;
                            acc = Some(match acc {
                                Some(a) => self.b.mul(a, e),
                                None => e,
                            });
                        }
                        acc
                    }
                    s if s.known_eq(Dim::Const(1)) || s.known_eq(Dim::Const(0)) => None,
                    s => Some(self.dim_expr(s)?),
                };
                match (outer_e, stride_e) {
                    (Some(o), Some(s)) => self.b.mul(o, s),
                    (Some(o), None) => o,
                    (None, Some(s)) => s,
                    (None, None) => self.b.u32(1),
                }
            }
            _ => self.b.u32(1),
        };
        let mask = self.b.compare(TileCompareOp::Lt, index.clone(), bound);
        let fill = self.zero_of(elem);
        Ok(self
            .b
            .load(Source::Storage(view), Addr::Linear(index), mask, fill))
    }

    /// The literal a `Leaf::Const` operand folds to, if it is one.
    pub(crate) fn const_operand(&mut self, src: Id) -> Option<TileExpr> {
        let selected = self.cx.selected(src);
        let fusor2_ir::ir::Op::Logical(fusor2_ir::ir::logical::Logical::Leaf(
            fusor2_ir::ir::logical::LeafKind::Const { value, .. },
        )) = &self.cx.graph.node(selected).op
        else {
            return None;
        };
        Some(match *value {
            fusor2_ir::dtype::Splat::F32(v) => self.b.f32(v),
            fusor2_ir::dtype::Splat::F16(v) => self.b.lit(TileLiteral::F16(v)),
            fusor2_ir::dtype::Splat::BF16(v) => self.b.lit(TileLiteral::BF16(v)),
            fusor2_ir::dtype::Splat::U32(v) => self.b.u32(v),
            fusor2_ir::dtype::Splat::I32(v) => self.b.i32(v),
        })
    }

    /// Finish a kernel body into a [`KernelIr`].
    pub fn finish(self, name: &'static str, grid: [u32; 3], block: u32, body: Vec<Stmt>) -> KernelIr {
        KernelIr {
            buffers: self.buffers,
            grid,
            block,
            body,
            byte_arena: if self.caps.workgroup_alias {
                Some(fusor2_ir::ir::kernel::ByteArenaToken)
            } else {
                None
            },
            name,
        }
    }
}

/// Lower one selected Launch node at one schedule point.
///
/// One match over `Launch` into the eight submodule entry points. Every arm gets a
/// real body: there is no "unsupported, fall back" path, because the extractor
/// already proved the node selectable on this target.
pub fn lower_node(
    caps: &Caps,
    node: &Node,
    theta: SchedPoint,
    cx: &LowerCtx<'_>,
    binding: DimBinding,
    pack: std::sync::Arc<UniformPack>,
) -> Result<Vec<KernelIr>> {
    let Op::Launch(op) = &node.op else {
        return Err(Error::Plan(format!(
            "lowering was handed a {:?} node, but only Launch nodes are selectable",
            node.level
        )));
    };
    let ctx = Ctx::with_pack(caps, cx, binding, pack)?;
    match op {
        Launch::Map { .. } => map_fold::lower_kmap(ctx, op, theta).map(|k| vec![k]),
        Launch::Fold { .. } => map_fold::lower_kfold(ctx, op, theta).map(|k| vec![k]),
        Launch::Contract { family, .. } => contract::lower_contract(ctx, op, *family, theta),
        Launch::Gather { .. } => gather_scatter::lower_kgather(ctx, op, theta).map(|k| vec![k]),
        Launch::Scatter { .. } => gather_scatter::lower_kscatter(ctx, op, theta),
        Launch::Region { .. } => region::lower_kregion(ctx, op, theta).map(|k| vec![k]),
        Launch::Ext { def, .. } => ext::lower(*def, node, theta).map(|k| vec![k]),
    }
}

/// `Launch::Ext` lowering: the one escape hatch out of the closed `Logical`/`Launch` enums.
pub mod ext {
    use super::*;
    use fusor2_ir::ir::{OpDefId, OpDefRegistry};
    use std::sync::RwLock;

    /// The registry `Launch::Ext` lowering resolves `OpDefId` against.
    ///
    /// [`LowerCtx`] does not carry the [`OpDefRegistry`] the graph was built
    /// with, so until it grows the field the embedder installs the same
    /// registry here that it installed on the e-graph's semantics.
    /// Registration order is id order and must match.
    static DEFS: RwLock<Option<OpDefRegistry>> = RwLock::new(None);

    /// Install the extension registry this process lowers against. Idempotent
    /// and last-write-wins; a second install with a differently ordered
    /// registry would silently rename every `OpDefId`, so callers pass the
    /// registry the graph was built with, unchanged.
    pub fn install(registry: OpDefRegistry) {
        *DEFS.write().expect("the OpDef registry lock is poisoned") = Some(registry);
    }

    /// The installed registry, if the embedder installed one.
    pub fn installed() -> Option<OpDefRegistry> {
        DEFS.read()
            .expect("the OpDef registry lock is poisoned")
            .clone()
    }

    /// Lower one registered extension op through its `"gpu"` row.
    pub fn lower(def: OpDefId, node: &Node, theta: SchedPoint) -> Result<KernelIr> {
        let registry = installed().ok_or_else(|| {
            Error::Plan(format!(
                "{def:?} is an extension op, but no OpDefRegistry is installed on the \
                 GPU target; call fusor2_gpu::lower::ext::install"
            ))
        })?;
        let entry = registry
            .get(def)
            .ok_or_else(|| Error::Plan(format!("no OpDef is registered as {def:?}")))?;
        let lower = entry
            .lower_per_target
            .iter()
            .find(|(target, _)| *target == "gpu")
            .map(|(_, f)| *f)
            .ok_or_else(|| {
                Error::Plan(format!(
                    "OpDef \"{}\" declares no \"gpu\" lowering; its \
                     lower_per_target names {:?}",
                    entry.name,
                    entry
                        .lower_per_target
                        .iter()
                        .map(|(t, _)| *t)
                        .collect::<Vec<_>>()
                ))
            })?;
        lower(node, &theta)
    }
}

/// Dispatch one selected Launch node to its family lowering.
///
/// The [`fusor2_ir::target::Target`] contract returns exactly one `KernelIr`
/// per node; families that need two launches (split-K, sort-then-segment) are
/// driven through [`lower_node`] by the executor, which binds their scratch.
pub fn lower(
    caps: &Caps,
    node: &Node,
    id: Id,
    theta: SchedPoint,
    cx: &LowerCtx<'_>,
) -> Result<KernelIr> {
    let _ = id;
    let binding = DimBinding::new();
    let pack = std::sync::Arc::new(UniformPack::new(cx.plan));
    let mut kernels = lower_node(caps, node, theta, cx, binding, pack)?;
    if kernels.len() == 1 {
        Ok(kernels.remove(0))
    } else {
        Err(Error::Plan(format!(
            "{} lowers to {} kernels; use lower_node",
            node.op.tag() as u32,
            kernels.len()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The linearization every lowering applies to `@builtin(workgroup_id)`
    /// must be `gx + gy*X + gz*X*Y` **of the dispatched grid**, and
    /// `max_compute_workgroups_per_dimension` is not `X`.
    ///
    /// `distribute_workgroups` picks the slab count first and sizes `x` to the
    /// slab, so `x < max_per_dim` for most launches that need a second slab.
    /// `Ctx::global_index` used the limit as the stride; every workgroup off
    /// the first slab then computed an index far past the end, masked itself
    /// out, and left that part of the output at its prior contents.
    #[test]
    fn the_workgroup_linearization_uses_the_grid_not_the_limit() {
        const MAX: u32 = 65535;
        for total in [1u32, 6, 65535, 65536, 122_880, 200_000, 4_000_000] {
            let [x, y, z] = distribute_workgroups(total, MAX);
            // Every group in `0..total` is hit exactly once by the grid walk.
            let mut seen = vec![false; total as usize];
            for gz in 0..z {
                for gy in 0..y {
                    for gx in 0..x {
                        let id = u64::from(gx)
                            + u64::from(gy) * u64::from(x)
                            + u64::from(gz) * u64::from(x) * u64::from(y);
                        if let Some(slot) = seen.get_mut(id as usize) {
                            assert!(!*slot, "{total}: group {id} visited twice");
                            *slot = true;
                        }
                    }
                }
            }
            assert!(seen.iter().all(|s| *s), "{total}: a group was never visited");
        }

        // ...and the limit really is the wrong stride: at 6 groups the grid is
        // [3, 2, 1], so `gy * MAX` sends the whole second row out of range.
        let grid = distribute_workgroups(6u32, 4);
        assert_eq!(grid, [3, 2, 1]);
        assert_ne!(grid[0], 4, "the slab width is not the per-dimension limit");
    }

    /// WGSL has no infinite literal and naga rejects a module holding one,
    /// so every `-inf` the frontend hands down — a causal mask's reject arm, a
    /// max fold's identity — has to arrive as the largest finite magnitude.
    #[test]
    fn an_infinite_literal_is_clamped_to_something_wgsl_can_spell() {
        assert!(finite_f32(f32::NEG_INFINITY).is_finite());
        assert!(finite_f32(f32::INFINITY).is_finite());
        assert!(finite_f32(f32::NEG_INFINITY) < -3.0e38);
        assert!(finite_f32(f32::INFINITY) > 3.0e38);
        assert_eq!(finite_f32(-2.5), -2.5);
        assert_eq!(finite_f32(f32::NAN), 0.0);

        let neg = half::f16::from_bits(finite_f16(half::f16::NEG_INFINITY.to_bits()));
        assert!(neg.is_finite() && neg.to_f32() <= -65504.0);
        let bneg = half::bf16::from_bits(finite_bf16(half::bf16::NEG_INFINITY.to_bits()));
        assert!(bneg.is_finite() && bneg.to_f32() < -3.0e38);

        // The max-carrier sentinel agrees with `emit::expr`'s reduce identity,
        // so a max started by hand and one started by a `Reduce` match.
        let mut b = Kernel::new();
        let sentinel = b.neg_inf(ScalarElement::F32);
        let expected = b.f32(-crate::emit::expr::WGSL_SAFE_F32_MAX);
        assert_eq!(sentinel, expected);
    }

    #[test]
    fn hash_consing_merges_separately_built_subtrees() {
        let mut b = Kernel::new();
        let a1 = {
            let x = b.f32(2.0);
            let y = b.f32(3.0);
            b.add(x, y)
        };
        let a2 = {
            let x = b.f32(2.0);
            let y = b.f32(3.0);
            b.add(x, y)
        };
        assert_eq!(a1, a2);
        assert_eq!(a1.structural_hash(), a2.structural_hash());
    }

    #[test]
    fn affine_flatten_uses_plain_strides() {
        let binding = DimBinding::new();
        let layout = Layout::contiguous(&[Dim::Const(4), Dim::Const(8), Dim::Const(16)]);
        let view = flatten_matrix_layout_split(&layout, 2, &binding).unwrap();
        assert_eq!(view.rows, 32);
        assert_eq!(view.cols, 16);
        assert!(view.layout.is_affine(), "an affine side must stay affine");
        assert_eq!(view.layout.indexing.groups[0].sub_axes[0].stride, 16);
        assert_eq!(view.layout.indexing.groups[1].sub_axes[0].stride, 1);
    }

    #[test]
    fn non_affine_flatten_emits_sub_axes_and_drops_extent_one() {
        let binding = DimBinding::new();
        // A transposed batch prefix: axis 0 does not merge with axis 1.
        let layout = Layout::from_parts(
            Dim::Const(0),
            &[Dim::Const(3), Dim::Const(1), Dim::Const(5), Dim::Const(7)],
            &[
                Dim::Const(1),
                Dim::Const(0),
                Dim::Const(21),
                Dim::Const(3),
            ],
        )
        .unwrap();
        let view = flatten_matrix_layout_split(&layout, 2, &binding).unwrap();
        assert_eq!(view.rows, 3);
        assert_eq!(view.cols, 35);
        // The extent-1 axis is dropped, saving a divmod per load.
        assert_eq!(view.layout.indexing.groups[0].sub_axes.len(), 1);
        assert_eq!(view.layout.indexing.groups[0].sub_axes[0].extent, 3);
    }

    #[test]
    fn a_split_past_the_rank_is_a_plan_error() {
        let binding = DimBinding::new();
        let layout = Layout::contiguous(&[Dim::Const(4), Dim::Const(8)]);
        assert!(matches!(
            flatten_matrix_layout_split(&layout, 3, &binding),
            Err(Error::Plan(_))
        ));
    }

    /// A contraction whose `n` extent is 1 has no `n` axes, so its B operand
    /// is a one-column matrix: the split lands *on* the rank, not inside it.
    /// The empty side must contribute nothing to the address.
    #[test]
    fn a_degenerate_split_is_a_one_row_or_one_column_matrix() {
        let binding = DimBinding::new();
        let layout = Layout::contiguous(&[Dim::Const(4), Dim::Const(8)]);

        let cols = flatten_matrix_layout_split(&layout, 2, &binding).unwrap();
        assert_eq!((cols.rows, cols.cols), (32, 1));
        assert!(cols.layout.is_affine());
        // Row `r` is element `r`; the empty column side has stride 0.
        assert_eq!(cols.layout.indexing.groups[0].sub_axes[0].stride, 1);
        assert_eq!(cols.layout.indexing.groups[1].sub_axes[0].stride, 0);

        let rows = flatten_matrix_layout_split(&layout, 0, &binding).unwrap();
        assert_eq!((rows.rows, rows.cols), (1, 32));
        assert_eq!(rows.layout.indexing.groups[0].sub_axes[0].stride, 0);
        assert_eq!(rows.layout.indexing.groups[1].sub_axes[0].stride, 1);
    }

    /// The split a contraction operand needs comes from its `(rows, cols)`
    /// element counts, never from its rank: a batched `[b, m, k]` A splits at
    /// 2, and a `[b, k]` B of an `n = 1` contraction splits at 2 as well —
    /// both of which `rank - 1` gets wrong.
    #[test]
    fn matrix_split_comes_from_the_extents_not_the_rank() {
        let binding = DimBinding::new();
        let a = Layout::contiguous(&[Dim::Const(3), Dim::Const(4), Dim::Const(5)]);
        assert_eq!(matrix_split_for(&a, &binding, 12, 5).unwrap(), 2);
        assert_eq!(matrix_split_for(&a, &binding, 3, 20).unwrap(), 1);
        assert_eq!(matrix_split_for(&a, &binding, 60, 1).unwrap(), 3);
        assert_eq!(matrix_split_for(&a, &binding, 1, 60).unwrap(), 0);

        // `sum(x * x, 1)` over `[3, 5]`: batch 3, m 1, n 1, k 5. A is
        // `[batch * m, k] = [3, 5]` and B is `[batch * k, n] = [15, 1]`.
        let x = Layout::contiguous(&[Dim::Const(3), Dim::Const(5)]);
        assert_eq!(matrix_split_for(&x, &binding, 3, 5).unwrap(), 1);
        assert_eq!(matrix_split_for(&x, &binding, 15, 1).unwrap(), 2);

        assert!(matches!(
            matrix_split_for(&x, &binding, 7, 2),
            Err(Error::Plan(_))
        ));
    }

    #[test]
    fn grid_for_respects_the_per_dimension_limit() {
        let limits = Limits::default();
        let space = IndexSpace::new([Dim::Const(1024), Dim::Const(1024)]);
        let grid = grid_for(&space, 256, &DimBinding::new(), &limits).unwrap();
        let launched = u64::from(grid[0]) * u64::from(grid[1]) * u64::from(grid[2]);
        assert!(launched >= (1024 * 1024) / 256);
        assert!(grid.iter().all(|d| *d <= limits.max_compute_workgroups_per_dimension));
    }

    #[test]
    fn grid_for_needs_every_symbol_bound() {
        let limits = Limits::default();
        let space = IndexSpace::new([Dim::Sym(SymId(0))]);
        assert!(grid_for(&space, 64, &DimBinding::new(), &limits).is_err());
        let bound = DimBinding::from_pairs([(SymId(0), 512)]);
        assert_eq!(grid_for(&space, 64, &bound, &limits).unwrap(), [8, 1, 1]);
    }

    /// The same plan at three sequence lengths produces three grids and one
    /// kernel identity, because the extent never enters the body.
    #[test]
    fn sequence_length_only_moves_the_grid() {
        let limits = Limits::default();
        let space = IndexSpace::new([Dim::Sym(SymId(0)), Dim::Const(64)]);
        let grids: Vec<_> = [256u64, 512, 768]
            .into_iter()
            .map(|n| {
                grid_for(
                    &space,
                    256,
                    &DimBinding::from_pairs([(SymId(0), n)]),
                    &limits,
                )
                .unwrap()
            })
            .collect();
        assert_eq!(grids, vec![[64, 1, 1], [128, 1, 1], [192, 1, 1]]);
    }
}


/// End-to-end cover for the extension seam: a registered `OpDef` lowering to
/// real WGSL, dispatching on the adapter, and returning the numbers it
/// computed.
#[cfg(test)]
mod ext_tests {
    use super::*;
    use fusor2_ir::dtype::Persistence;
    use fusor2_ir::egraph::EGraph;
    use fusor2_ir::extract::{
        BindKind, BindingPlan, Extraction, Dispatch, Plan, PlanHash,
    };
    use fusor2_ir::facts::{ValueFacts, Work};
    use fusor2_ir::ir::logical::{BufferId, Logical, LeafKind};
    use fusor2_ir::ir::launch::{AccessPlan, Effect};
    use fusor2_ir::ir::kernel::{
        Addr, BufferAccess, BufferDecl, MemoryLevel, Source, StorageView, TileLayout,
    };
    use fusor2_ir::ir::{AttrId, OpDef, OpDefId, OpDefRegistry, OpTag, VerifyCtx};
    use fusor2_ir::semantics::{CoreSemantics, SumArenaPlanner};
    use fusor2_ir::shape::Layout;
    use fusor2_ir::ir::kernel::WorkgroupAxis;
    use fusor2_ir::target::Target;
    use std::sync::Arc;

    const N: u32 = 8;

    /// The registered op's own `"gpu"` lowering: `y = 3 * x`. It builds the
    /// whole `KernelIr` from the node, which is all `lower_per_target` offers,
    /// and nothing in `fusor2-gpu` knows what "triple" means.
    ///
    /// `theta` is read rather than dropped. `Launch::schedule` returns `None` for
    /// `Ext` — fusor2 cannot enumerate geometries for an `OpDef`-supplied
    /// lowering — so extraction owes this seam `SchedPoint::Point` and nothing
    /// else. Matching it states that contract instead of assuming it: an `Ext`
    /// handed a `Fold` or `Sgemm` point means the schedule space grew a case
    /// `schedule()` does not describe, and the seam should say so rather than
    /// silently lower against a geometry it never read.
    fn lower_triple(node: &Node, theta: &SchedPoint) -> Result<KernelIr> {
        let SchedPoint::Point = theta else {
            return Err(Error::Plan(
                format!("Ext lowerings take SchedPoint::Point; got {theta:?}").into(),
            ));
        };
        let Op::Launch(Launch::Ext { ops, .. }) = &node.op else {
            return Err(Error::Plan("triple got a foreign node".into()));
        };
        let n: u32 = ops
            .first()
            .ok_or_else(|| Error::Plan("triple needs an operand".into()))?
            .layout
            .shape()
            .iter()
            .map(|d| d.as_const().unwrap_or(1) as u32)
            .product();
        let f32e = ElementType::Scalar(ScalarElement::F32);
        let decl = |binding, element, len: u32, access| {
            Arc::new(BufferDecl {
                binding,
                element,
                layout: TileLayout::contiguous(MemoryLevel::Storage, &[len]),
                access,
            })
        };
        let uniforms = decl(
            0,
            ElementType::Scalar(ScalarElement::U32),
            1,
            BufferAccess::Read,
        );
        let input = decl(1, f32e, n, BufferAccess::Read);
        let output = decl(2, f32e, n, BufferAccess::ReadWrite);

        let mut b = Kernel::new();
        let block = 64u32;
        let gid = b.builtin(Builtin::ProgramId(WorkgroupAxis::X));
        let lane = b.builtin(Builtin::Lane);
        let stride = b.u32(block);
        let base = b.mul(gid, stride);
        let index = b.add(base, lane);
        let bound = b.u32(n);
        let mask = b.compare(TileCompareOp::Lt, index.clone(), bound);
        let fill = b.zero(ScalarElement::F32);
        let view = |d: &Arc<BufferDecl>| StorageView {
            layout: d.layout.clone(),
            buffer: Arc::clone(d),
            offset: 0,
        };
        let x = b.load(
            Source::Storage(view(&input)),
            Addr::Linear(index.clone()),
            mask.clone(),
            fill,
        );
        let three = b.f32(3.0);
        let value = b.mul(x, three);
        Ok(KernelIr {
            buffers: vec![uniforms, input, output.clone()],
            grid: [n.div_ceil(block).max(1), 1, 1],
            block,
            body: vec![Stmt::Store {
                dst: view(&output),
                addr: Addr::Linear(index),
                value,
                mask,
            }],
            byte_arena: None,
            name: "triple",
        })
    }

    fn registry() -> OpDefRegistry {
        fn infer(ins: &[ValueFacts]) -> Result<ValueFacts> {
            ins.first()
                .cloned()
                .ok_or_else(|| Error::Shape("triple needs an operand".into()))
        }
        fn work(_: &[ValueFacts], out: &ValueFacts) -> Work {
            let n = out.elements().unwrap_or(1);
            Work {
                macs: n,
                transcendentals: 0,
                index_ops: n,
                wg_bytes: 0,
            }
        }
        fn verify(_: &VerifyCtx<'_>) -> Result<()> {
            Ok(())
        }
        let mut r = OpDefRegistry::new();
        r.register(OpDef {
            name: "triple",
            tag: OpTag::Ext,
            verify,
            infer,
            work,
            adjoint: None,
            lower_per_target: &[("gpu", lower_triple)],
            effect: Effect::Pure,
        });
        r
    }

    #[test]
    fn a_registered_op_def_lowers_emits_and_dispatches() {
        let reg = registry();
        ext::install(reg.clone());
        let mut g = EGraph::new(CoreSemantics::with_registry(
            Arc::new(SumArenaPlanner),
            reg,
        ));
        let x = g
            .add(Op::Logical(Logical::Leaf(LeafKind::Buffer {
                name: BufferId(0),
                dtype: fusor2_ir::dtype::Dtype::F32,
                shape: smallvec::smallvec![Dim::Const(N as u64)],
            })))
            .unwrap();
        let ops = vec![Operand {
            src: x,
            layout: Layout::contiguous(&g.facts(x).shape),
            access: AccessPlan::Alias,
        }];
        let e = g
            .add(Op::Launch(Launch::Ext {
                def: OpDefId(0),
                ops,
                attrs: AttrId(0),
            }))
            .unwrap();

        let plan = Plan {
            extraction: Extraction::default(),
            launches: vec![Dispatch {
                root: e,
                members: smallvec::smallvec![e],
                bindings: vec![
                    BindingPlan {
                        binding: 1,
                        value: x,
                        kind: BindKind::Read,
                    },
                    BindingPlan {
                        binding: 2,
                        value: e,
                        kind: BindKind::Write,
                    },
                ],
                grid: [1, 1, 1],
                block: 64,
            }],
            buffers: Vec::new(),
            symbols: Vec::new(),
            hash: PlanHash(0),
            cost: fusor2_ir::cost::Picoseconds(0),
        };
        let cx = LowerCtx {
            plan: &plan,
            launch: &plan.launches[0],
            graph: &g,
            symbols: &[],
        };

        let target = crate::target::GpuTarget::new_blocking().unwrap();
        let ir = lower(target.caps(), g.node(e), e, SchedPoint::Point, &cx).unwrap();
        assert_eq!(ir.name, "triple", "the OpDef's own lowering must run");

        let artifact = target.emit(&ir).unwrap();
        let data: Vec<f32> = (1..=N).map(|v| v as f32).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input = target
            .pool()
            .create_buffer_init(&bytes, crate::pool::TENSOR_USAGE)
            .unwrap();
        let out = target
            .alloc((N as u64) * 4, Persistence::Step)
            .unwrap();
        let uniform = target.alloc(4, Persistence::Step).unwrap();
        target
            .launch(
                &artifact,
                ir.grid,
                &[uniform, input, out.clone()],
                &Default::default(),
            )
            .unwrap();
        target.wait().unwrap();
        let back = target
            .launcher()
            .readback(target.pool(), &out, (N as u64) * 4)
            .unwrap();
        let got: Vec<f32> = back[..(N as usize) * 4]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(got, vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0]);

        // And an op with no `"gpu"` row is a plan answer naming the op, never
        // a panic and never a silent fallback.
        let mut empty = OpDefRegistry::new();
        empty.register(OpDef {
            name: "cpu_only",
            tag: OpTag::Ext,
            verify: |_| Ok(()),
            infer: |ins: &[ValueFacts]| {
                ins.first()
                    .cloned()
                    .ok_or_else(|| Error::Shape("no operand".into()))
            },
            work: |_, out: &ValueFacts| Work {
                macs: out.elements().unwrap_or(1),
                ..Work::default()
            },
            adjoint: None,
            lower_per_target: &[],
            effect: Effect::Pure,
        });
        ext::install(empty);
        let err = ext::lower(OpDefId(0), g.node(e), SchedPoint::Point).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("cpu_only") && msg.contains("gpu"), "{msg}");
    }
}
