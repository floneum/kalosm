//! The hash-consing Kernel term builder both backends lower through, and the
//! literal, operand and reduction helpers their lowerings share.

use std::cell::RefCell;
use std::sync::Arc;

use fusor_ir::Result;
use fusor_ir::carrier::{Carrier, SlotTy};
use fusor_ir::dtype::{Dtype, NumericContract, QFmt, QLayout, RoundMode, Splat};
use fusor_ir::egraph::Id;
use fusor_ir::error::Error;
use fusor_ir::ir::Op;
use fusor_ir::ir::kernel::{
    Addr, Builtin, CoopMatrixRole, CoopSrc, ElementType, Local, LocalDecl, MemoryLevel, MergeBody,
    ReduceKind, ScalarElement, Source, Stmt, Tile, TileBinaryOp, TileCompareOp, TileDecl, TileExpr,
    TileExprKind, TileLayout, TileLiteral, TileReduceOp, TileUnaryOp,
};
use fusor_ir::ir::launch::AddressMap;
use fusor_ir::ir::logical::{LeafKind, Logical};
use fusor_ir::scalar::ScalarExpr;
use fusor_ir::target::LowerCtx;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

/// The largest finite f32 WGSL parses back identically; stands in for ±inf.
pub const SAFE_F32_MAX: f32 = 3.40282e38;

/// Clamp an infinite literal finite (WGSL cannot spell one; `exp(x - m)`
/// underflows the same) and a NaN to zero.
pub fn finite_f32(v: f32) -> f32 {
    if v.is_infinite() {
        if v.is_sign_negative() {
            -SAFE_F32_MAX
        } else {
            SAFE_F32_MAX
        }
    } else if v.is_nan() {
        0.0
    } else {
        v
    }
}

/// [`finite_f32`] on the f16 bit pattern.
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

/// [`finite_f32`] on the bf16 bit pattern, at bf16's own finite extremes.
pub fn finite_bf16(bits: u16) -> u16 {
    let v = half::bf16::from_bits(bits);
    if v.is_nan() {
        return half::bf16::ZERO.to_bits();
    }
    if !v.is_infinite() {
        return bits;
    }
    if v.is_sign_negative() {
        half::bf16::MIN.to_bits()
    } else {
        half::bf16::MAX.to_bits()
    }
}

/// The identity of a reduction over `elem`, finite where the type has
/// infinities.
pub fn reduce_identity(op: TileReduceOp, elem: ScalarElement) -> TileLiteral {
    use TileReduceOp::{Max, Min, Product, Sum};
    let f16 = |v: f32| TileLiteral::F16(half::f16::from_f32(v).to_bits());
    let bf16 = |v: half::bf16| TileLiteral::BF16(v.to_bits());
    match (op, elem) {
        (Sum, ScalarElement::F32) => TileLiteral::F32(0f32.to_bits()),
        (Product, ScalarElement::F32) => TileLiteral::F32(1f32.to_bits()),
        (Max, ScalarElement::F32) => TileLiteral::F32((-SAFE_F32_MAX).to_bits()),
        (Min, ScalarElement::F32) => TileLiteral::F32(SAFE_F32_MAX.to_bits()),
        (Sum, ScalarElement::F16) => f16(0.0),
        (Product, ScalarElement::F16) => f16(1.0),
        (Max, ScalarElement::F16) => f16(-65504.0),
        (Min, ScalarElement::F16) => f16(65504.0),
        (Sum, ScalarElement::BF16) => bf16(half::bf16::ZERO),
        (Product, ScalarElement::BF16) => bf16(half::bf16::ONE),
        (Max, ScalarElement::BF16) => bf16(half::bf16::MIN),
        (Min, ScalarElement::BF16) => bf16(half::bf16::MAX),
        (Sum | Max, ScalarElement::U32) => TileLiteral::U32(0),
        (Product, ScalarElement::U32) => TileLiteral::U32(1),
        (Min, ScalarElement::U32) => TileLiteral::U32(u32::MAX),
        (Sum, ScalarElement::I32) => TileLiteral::I32(0),
        (Product, ScalarElement::I32) => TileLiteral::I32(1),
        (Max, ScalarElement::I32) => TileLiteral::I32(i32::MIN),
        (Min, ScalarElement::I32) => TileLiteral::I32(i32::MAX),
        (Sum | Max, ScalarElement::Bool) => TileLiteral::Bool(false),
        (Product | Min, ScalarElement::Bool) => TileLiteral::Bool(true),
    }
}

/// A splat as a typed literal, verbatim.
pub fn splat_literal(s: Splat) -> TileLiteral {
    match s {
        Splat::F32(v) => TileLiteral::F32(v.to_bits()),
        Splat::F16(v) => TileLiteral::F16(v),
        Splat::BF16(v) => TileLiteral::BF16(v),
        Splat::U32(v) => TileLiteral::U32(v),
        Splat::I32(v) => TileLiteral::I32(v),
    }
}

/// A splat as a literal a WGSL module can hold: infinities and NaN clamp.
pub fn finite_literal(s: Splat) -> TileLiteral {
    match s {
        Splat::F32(v) => TileLiteral::F32(finite_f32(v).to_bits()),
        Splat::F16(v) => TileLiteral::F16(finite_f16(v)),
        Splat::BF16(v) => TileLiteral::BF16(finite_bf16(v)),
        other => splat_literal(other),
    }
}

/// Logical dtype to Kernel element; quantized weights bind as `u32` words.
pub const fn scalar_element(dtype: Dtype) -> ScalarElement {
    match dtype {
        Dtype::F32 => ScalarElement::F32,
        Dtype::F16 => ScalarElement::F16,
        Dtype::BF16 => ScalarElement::BF16,
        Dtype::I32 => ScalarElement::I32,
        Dtype::U32 | Dtype::Q(_) => ScalarElement::U32,
    }
}

/// `u32` words a block-quantized value of `elements` elements occupies.
pub fn quantized_words(fmt: QFmt, layout: QLayout, elements: u64) -> u64 {
    let blocks = elements.div_ceil(u64::from(fmt.block_elements()).max(1));
    (blocks * u64::from(fmt.block_bytes(layout))).div_ceil(4)
}

/// The storage layout a quantized value carries, read off its `LeafKind`.
pub fn qlayout_of(cx: &LowerCtx<'_>, value: Id) -> Option<QLayout> {
    let class = cx.graph.class_of(value);
    cx.graph
        .class_ids(class)
        .into_iter()
        .find_map(|m| match &cx.graph.node(m).op {
            Op::Logical(Logical::Leaf(LeafKind::Quantized { layout, .. })) => Some(*layout),
            _ => None,
        })
}

/// The splat a `Leaf::Const` operand folds to, if it is one.
pub fn const_splat(cx: &LowerCtx<'_>, src: Id) -> Option<Splat> {
    match &cx.graph.node(cx.selected(src)).op {
        Op::Logical(Logical::Leaf(LeafKind::Const { value, .. })) => Some(*value),
        _ => None,
    }
}

/// Hash-consing Kernel term builder: identical subtrees share one `Arc`.
#[derive(Default)]
pub struct Kernel {
    memo: RefCell<FxHashMap<u64, SmallVec<[TileExpr; 2]>>>,
}

impl Kernel {
    pub fn new() -> Self {
        Self::default()
    }

    fn intern(&self, kind: TileExprKind, ty: ElementType) -> TileExpr {
        let expr = TileExpr::new(kind, ty);
        let mut memo = self.memo.borrow_mut();
        let bucket = memo.entry(expr.structural_hash()).or_default();
        if let Some(hit) = bucket.iter().find(|e| **e == expr) {
            return hit.clone();
        }
        bucket.push(expr.clone());
        expr
    }

    pub fn lit(&self, value: TileLiteral) -> TileExpr {
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

    pub fn f32(&self, v: f32) -> TileExpr {
        self.lit(TileLiteral::F32(v.to_bits()))
    }
    pub fn u32(&self, v: u32) -> TileExpr {
        self.lit(TileLiteral::U32(v))
    }
    pub fn i32(&self, v: i32) -> TileExpr {
        self.lit(TileLiteral::I32(v))
    }
    pub fn bool(&self, v: bool) -> TileExpr {
        self.lit(TileLiteral::Bool(v))
    }

    /// The zero of an element type.
    pub fn zero(&self, elem: ScalarElement) -> TileExpr {
        match elem {
            ScalarElement::F32 => self.f32(0.0),
            ScalarElement::F16 => self.lit(TileLiteral::F16(0)),
            ScalarElement::BF16 => self.lit(TileLiteral::BF16(0)),
            ScalarElement::U32 => self.u32(0),
            ScalarElement::I32 => self.i32(0),
            ScalarElement::Bool => self.bool(false),
        }
    }

    /// [`Self::zero`] of any element type; a vector is a vector of zeros.
    pub fn zero_of(&self, elem: ElementType) -> TileExpr {
        match elem {
            ElementType::Scalar(s) | ElementType::CoopMatrix { scalar: s, .. } => self.zero(s),
            ElementType::Vector { scalar, lanes } => {
                let z = self.zero(scalar);
                self.vector(scalar, vec![z; lanes as usize])
            }
        }
    }

    /// The finite `Max` identity, bit-equal to the emitted reduce identity.
    pub fn neg_inf(&self, elem: ScalarElement) -> TileExpr {
        self.extreme(TileReduceOp::Max, elem)
    }

    /// The `Min` identity, the mirror of [`Self::neg_inf`].
    pub fn pos_inf(&self, elem: ScalarElement) -> TileExpr {
        self.extreme(TileReduceOp::Min, elem)
    }

    /// `op`'s identity in a float type; every other element takes f32's.
    fn extreme(&self, op: TileReduceOp, elem: ScalarElement) -> TileExpr {
        match elem {
            ScalarElement::F16 | ScalarElement::BF16 => self.lit(reduce_identity(op, elem)),
            _ => self.lit(reduce_identity(op, ScalarElement::F32)),
        }
    }

    /// A carrier identity as a finite literal of `elem`.
    pub fn identity(&self, s: Splat, elem: ScalarElement) -> TileExpr {
        let f = match s {
            Splat::F32(v) => v,
            Splat::F16(b) => half::f16::from_bits(b).to_f32(),
            Splat::BF16(b) => half::bf16::from_bits(b).to_f32(),
            Splat::U32(0) | Splat::I32(0) => return self.zero(elem),
            Splat::U32(u32::MAX) | Splat::I32(i32::MAX) => return self.pos_inf(elem),
            Splat::I32(i32::MIN) => return self.neg_inf(elem),
            Splat::U32(v) => return self.u32(v),
            Splat::I32(v) => return self.i32(v),
        };
        if f == f32::NEG_INFINITY {
            self.neg_inf(elem)
        } else if f == f32::INFINITY {
            self.pos_inf(elem)
        } else if f == 0.0 {
            self.zero(elem)
        } else if f == 1.0 {
            match elem {
                ScalarElement::U32 => self.u32(1),
                ScalarElement::I32 => self.i32(1),
                _ => self.f32(1.0),
            }
        } else {
            self.f32(f)
        }
    }

    pub fn builtin(&self, b: Builtin) -> TileExpr {
        self.intern(TileExprKind::Builtin(b), ScalarElement::U32.element())
    }

    pub fn load_local(&self, local: Local) -> TileExpr {
        let ty = local.element;
        self.intern(TileExprKind::LoadLocal(local), ty)
    }

    pub fn load(&self, src: Source, addr: Addr, mask: TileExpr, fill: TileExpr) -> TileExpr {
        let ty = match &src {
            Source::Storage(v) => v.buffer.element,
            // A quantized load decodes to f32 before it is ever a value.
            Source::Quantized(_) => ScalarElement::F32.element(),
        };
        let addr = Box::new(addr);
        self.intern(
            TileExprKind::Load {
                src,
                addr,
                mask,
                fill,
            },
            ty,
        )
    }

    pub fn load_tile(&self, tile: Tile, index: TileExpr) -> TileExpr {
        let ty = tile.element;
        self.intern(TileExprKind::LoadTile { tile, index }, ty)
    }

    pub fn unary(&self, op: TileUnaryOp, value: TileExpr, numeric: NumericContract) -> TileExpr {
        let ty = if op == TileUnaryOp::Unpack2x16Float {
            ElementType::Vector {
                scalar: ScalarElement::F32,
                lanes: 2,
            }
        } else {
            value.element()
        };
        self.intern(TileExprKind::Unary { op, value, numeric }, ty)
    }

    pub fn binary(
        &self,
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

    fn relaxed(&self, op: TileBinaryOp, a: TileExpr, b: TileExpr) -> TileExpr {
        self.binary(op, a, b, NumericContract::RELAXED)
    }
    pub fn add(&self, a: TileExpr, b: TileExpr) -> TileExpr {
        self.relaxed(TileBinaryOp::Add, a, b)
    }
    pub fn mul(&self, a: TileExpr, b: TileExpr) -> TileExpr {
        self.relaxed(TileBinaryOp::Mul, a, b)
    }
    pub fn sub(&self, a: TileExpr, b: TileExpr) -> TileExpr {
        self.relaxed(TileBinaryOp::Sub, a, b)
    }
    pub fn div(&self, a: TileExpr, b: TileExpr) -> TileExpr {
        self.relaxed(TileBinaryOp::Div, a, b)
    }
    pub fn rem(&self, a: TileExpr, b: TileExpr) -> TileExpr {
        self.relaxed(TileBinaryOp::Rem, a, b)
    }
    pub fn min(&self, a: TileExpr, b: TileExpr) -> TileExpr {
        self.relaxed(TileBinaryOp::Min, a, b)
    }
    /// `(a / b, a % b)`.
    pub fn divrem(&self, a: TileExpr, b: TileExpr) -> (TileExpr, TileExpr) {
        (self.div(a.clone(), b.clone()), self.rem(a, b))
    }
    /// `a * b + c`, contractible to one fma.
    pub fn fma(&self, a: TileExpr, b: TileExpr, c: TileExpr) -> TileExpr {
        self.add(self.mul(a, b), c)
    }
    /// `base + index * stride`.
    pub fn at(&self, base: TileExpr, index: TileExpr, stride: TileExpr) -> TileExpr {
        self.add(base, self.mul(index, stride))
    }
    /// `(x / inner) * span + x % inner`: where row `x` starts when `inner`
    /// elements follow its reduced axis and one outer step spans `span`.
    pub fn row_base(&self, x: TileExpr, inner: TileExpr, span: TileExpr) -> TileExpr {
        self.add(
            self.mul(self.div(x.clone(), inner.clone()), span),
            self.rem(x, inner),
        )
    }
    /// The first of the `tm` elements lane `base` owns, `stride` apart.
    pub fn tile_origin(&self, base: TileExpr, stride: TileExpr, tm: u32) -> TileExpr {
        let span = self.mul(stride.clone(), self.u32(tm));
        self.row_base(base, stride, span)
    }

    pub fn compare(&self, op: TileCompareOp, left: TileExpr, right: TileExpr) -> TileExpr {
        self.intern(
            TileExprKind::Compare { op, left, right },
            ScalarElement::Bool.element(),
        )
    }
    pub fn lt(&self, a: TileExpr, b: TileExpr) -> TileExpr {
        self.compare(TileCompareOp::Lt, a, b)
    }
    pub fn eq(&self, a: TileExpr, b: TileExpr) -> TileExpr {
        self.compare(TileCompareOp::Eq, a, b)
    }

    pub fn and(&self, a: TileExpr, b: TileExpr) -> TileExpr {
        self.intern(
            TileExprKind::Binary {
                op: TileBinaryOp::LogicalAnd,
                left: a,
                right: b,
                numeric: NumericContract::RELAXED,
            },
            ScalarElement::Bool.element(),
        )
    }

    pub fn cast(&self, value: TileExpr, to: ElementType) -> TileExpr {
        if value.element() == to {
            return value;
        }
        self.intern(TileExprKind::Cast { value, to }, to)
    }

    pub fn bitcast(&self, value: TileExpr, to: ElementType) -> TileExpr {
        self.intern(TileExprKind::Bitcast { value, to }, to)
    }

    pub fn select(&self, condition: TileExpr, accept: TileExpr, reject: TileExpr) -> TileExpr {
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

    pub fn vector(&self, scalar: ScalarElement, parts: Vec<TileExpr>) -> TileExpr {
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

    pub fn dot(&self, left: TileExpr, right: TileExpr) -> TileExpr {
        let ty = match left.element() {
            ElementType::Vector { scalar, .. } => ElementType::Scalar(scalar),
            other => other,
        };
        self.intern(TileExprKind::Dot { left, right }, ty)
    }

    pub fn round(&self, mode: RoundMode, value: TileExpr) -> TileExpr {
        let ty = value.element();
        self.intern(TileExprKind::Round { mode, value }, ty)
    }

    pub fn reduce(&self, op: TileReduceOp, kind: ReduceKind, value: TileExpr) -> TileExpr {
        let ty = value.element();
        let kind = Box::new(kind);
        self.intern(TileExprKind::Reduce { op, kind, value }, ty)
    }

    pub fn coop_load(
        &self,
        role: CoopMatrixRole,
        scalar: ScalarElement,
        rows: u32,
        cols: u32,
        src: CoopSrc,
    ) -> TileExpr {
        let src = Box::new(src);
        self.intern(
            TileExprKind::CoopLoad {
                role,
                scalar,
                rows,
                cols,
                src,
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
        &self,
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

    pub fn coop_mma(&self, a: TileExpr, b: TileExpr, c: TileExpr) -> TileExpr {
        let ty = c.element();
        self.intern(TileExprKind::CoopMma { a, b, c }, ty)
    }

    /// A private per-invocation local. Identity-bearing, so never interned.
    pub fn local(&self, element: ElementType) -> Local {
        Arc::new(LocalDecl::new(element))
    }

    /// A workgroup tile; identity-bearing, like a local.
    pub fn tile(&self, name: &'static str, element: ElementType, extents: &[u32]) -> Tile {
        Arc::new(TileDecl::new(
            element,
            TileLayout::contiguous(MemoryLevel::Workgroup, extents),
            name,
        ))
    }

    /// `((x / div) % modulus) * stride`, each step skipped when absent: one
    /// axis's contribution to a strided address.
    pub fn term(
        &self,
        x: TileExpr,
        div: Option<TileExpr>,
        modulus: Option<TileExpr>,
        stride: Option<TileExpr>,
    ) -> TileExpr {
        let x = div.map_or(x.clone(), |d| self.div(x, d));
        let x = modulus.map_or(x.clone(), |m| self.rem(x, m));
        stride.map_or(x.clone(), |s| self.mul(x, s))
    }

    /// `acc += e`, starting the sum at `e`.
    pub fn accumulate(&self, acc: &mut Option<TileExpr>, e: TileExpr) {
        *acc = Some(match acc.take() {
            Some(a) => self.add(a, e),
            None => e,
        });
    }

    /// `flat` run through a compile-time [`AddressMap`] over a space of
    /// `space_total` elements.
    pub fn address(&self, map: &AddressMap, flat: TileExpr, space_total: u64) -> TileExpr {
        if map.is_identity_over(space_total) {
            return flat;
        }
        let mut acc = (map.offset != 0).then(|| self.u32(map.offset));
        for (i, t) in map.terms.iter().enumerate() {
            let term = self.term(
                flat.clone(),
                (t.divisor > 1).then(|| self.u32(t.divisor)),
                map.needs_modulo(i, space_total)
                    .then(|| self.u32(t.modulus)),
                (t.stride != 1).then(|| self.u32(t.stride)),
            );
            self.accumulate(&mut acc, term);
        }
        acc.unwrap_or_else(|| self.u32(0))
    }

    /// The N-ary cross-lane close of a carrier, `merge` building the body
    /// from `lhs ++ rhs` lane reads; returns the statement and output reads.
    pub fn merge_tree(
        &self,
        scratch: SmallVec<[Tile; 4]>,
        group_size: u32,
        partials: Vec<TileExpr>,
        fast: Option<TileReduceOp>,
        ty: ElementType,
        merge: impl FnOnce(&[TileExpr]) -> Result<SmallVec<[TileExpr; 4]>>,
    ) -> Result<(Stmt, Vec<TileExpr>)> {
        let lanes = scratch.len();
        let locals = || -> SmallVec<[Local; 4]> { (0..lanes).map(|_| self.local(ty)).collect() };
        let (lhs, rhs, outs) = (locals(), locals(), locals());
        let args: Vec<TileExpr> = lhs
            .iter()
            .chain(rhs.iter())
            .map(|l| self.load_local(l.clone()))
            .collect();
        let body = merge(&args)?;
        let reads = outs.iter().map(|l| self.load_local(l.clone())).collect();
        let stmt = Stmt::Reduce {
            kind: Box::new(ReduceKind::Workgroup {
                scratch: scratch[0].clone(),
                group_size,
            }),
            values: partials.into_iter().collect(),
            merge: Box::new(MergeBody { lhs, rhs, body }),
            fast,
            outs,
            scratch,
        };
        Ok((stmt, reads))
    }
}

/// A fold's carrier expanded to one expression per accumulator lane.
pub struct FoldLanes {
    pub merges: Vec<ScalarExpr>,
    pub posts: Vec<ScalarExpr>,
    /// `(slot, promoted position)` of each lane.
    pub slots: Vec<(usize, u64)>,
    pub identities: Vec<Splat>,
    /// Iteration axis `j` is space axis `iter_axes[j]`.
    pub iter_axes: Vec<usize>,
}

impl FoldLanes {
    /// `space` is `free.. ++ vec.. ++ [reduced]` for a promoted nest; refuses
    /// a promoted fold not reducing last and a `Vector` slot unpromoted.
    pub fn of(
        carrier: &Carrier,
        post: &[ScalarExpr],
        rank: usize,
        axis: usize,
        vec_axes: &[u32],
        error: fn(String) -> Error,
    ) -> Result<Self> {
        let merges = carrier.merge_lanes().ok_or_else(|| {
            error("this carrier's merge does not expand to one expression per lane".into())
        })?;
        let posts = carrier.expand_lanes(post).ok_or_else(|| {
            error(format!(
                "a {}-slot carrier carries {} post expressions, or a slot's post reads \
                 a sibling of a different width",
                carrier.width(),
                post.len()
            ))
        })?;
        let symbolic = || error("this carrier has a symbolic Vector extent".into());
        let slots = carrier.lane_slots().ok_or_else(symbolic)?;
        let identities = carrier.identity_lanes().ok_or_else(symbolic)?;
        if axis >= rank {
            return Err(error(format!(
                "fold axis {axis} is outside a rank-{rank} space"
            )));
        }
        if !vec_axes.is_empty() && axis + 1 != rank {
            return Err(error(
                "a promoted Fold whose reduced axis is not last is not lowered".into(),
            ));
        }
        if vec_axes.is_empty() && carrier.slots.iter().any(|s| *s != SlotTy::Scalar) {
            return Err(error(
                "a Vector carrier slot needs a promoted axis to read its positions from".into(),
            ));
        }
        let iter_axes = (0..rank)
            .filter(|i| !vec_axes.contains(&(*i as u32)))
            .collect();
        Ok(Self {
            merges,
            posts,
            slots,
            identities,
            iter_axes,
        })
    }

    pub fn lanes(&self) -> usize {
        self.merges.len()
    }
}
