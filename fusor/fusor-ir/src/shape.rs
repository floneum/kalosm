//! Symbolic dims, shapes, strides, layouts, the broadcast rule, and the
//! non-affine index map a conv window operand needs.

use crate::error::{Error, Result};
use smallvec::SmallVec;
use std::fmt;

/// A symbolic quantity bound at dispatch, never at compile. Sequence
/// lengths, batch sizes and tile counts are `SymId`s; they hash as symbols,
/// so one extracted plan serves a whole shape family.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SymId(pub u32);

impl fmt::Display for SymId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "s{}", self.0)
    }
}

/// One extent. Rank is runtime data; extents are either known or symbolic.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Dim {
    Const(u64),
    Sym(SymId),
}

impl Dim {
    pub const fn as_const(self) -> Option<u64> {
        match self {
            Self::Const(v) => Some(v),
            Self::Sym(_) => None,
        }
    }

    /// Decidably at least `n`. Symbolic dims answer `false`, so guards stay
    /// conservative under symbolic shapes by construction.
    pub const fn at_least(self, n: u64) -> bool {
        match self {
            Self::Const(v) => v >= n,
            Self::Sym(_) => false,
        }
    }

    /// Decidably equal. Two distinct `Sym`s are not decidably equal even if
    /// they happen to bind the same value.
    pub const fn known_eq(self, other: Self) -> bool {
        match (self, other) {
            (Self::Const(a), Self::Const(b)) => a == b,
            (Self::Sym(a), Self::Sym(b)) => a.0 == b.0,
            _ => false,
        }
    }

    pub const ONE: Dim = Dim::Const(1);
}

impl From<u64> for Dim {
    fn from(v: u64) -> Self {
        Dim::Const(v)
    }
}
impl From<usize> for Dim {
    fn from(v: usize) -> Self {
        Dim::Const(v as u64)
    }
}
impl From<SymId> for Dim {
    fn from(v: SymId) -> Self {
        Dim::Sym(v)
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Const(v) => write!(f, "{v}"),
            Self::Sym(s) => write!(f, "{s}"),
        }
    }
}

/// A shape: inline up to rank 6, heap past that. No rank ceiling and no
/// const-generic rank.
pub type Dims = SmallVec<[Dim; 6]>;

/// A per-axis view spec, composed **relative to the current strides**.
/// `out_stride[i] = if multiplier == 0 { 0 } else { in_stride[input_dim] *
/// multiplier }`, `out_shape[i] = size`, offset gains
/// `offset * in_stride[input_dim]`. Every reshape, transpose, permute,
/// slice, narrow, broadcast, squeeze, unsqueeze and flatten is a vector of
/// these.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct StrideSpec {
    pub input_dim: u32,
    pub multiplier: u32,
    pub size: Dim,
    pub offset: Dim,
}

impl StrideSpec {
    pub const fn dim(input_dim: u32, size: Dim) -> Self {
        Self {
            input_dim,
            multiplier: 1,
            size,
            offset: Dim::Const(0),
        }
    }

    pub const fn dim_with(input_dim: u32, size: Dim, multiplier: u32) -> Self {
        Self {
            input_dim,
            multiplier,
            size,
            offset: Dim::Const(0),
        }
    }

    /// A stride-0 axis. The frontend emits this instead of implicit
    /// broadcasting; `verify_l0` requires identical `Map` operand shapes.
    pub const fn broadcast(size: Dim) -> Self {
        Self {
            input_dim: 0,
            multiplier: 0,
            size,
            offset: Dim::Const(0),
        }
    }

    pub const fn with_offset(mut self, offset: Dim) -> Self {
        self.offset = offset;
        self
    }

    pub const fn is_broadcast(self) -> bool {
        self.multiplier == 0
    }
}

/// One windowed axis of an `Logical::Window`. Kept separate from [`StrideSpec`]
/// because the adjoint needs `window` and `step` as *integers*: under
/// `Dim::Sym`, injectivity of a relative stride composition is undecidable.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SlidingWindow {
    pub axis: u32,
    pub window: u32,
    pub step: u32,
}

impl SlidingWindow {
    pub const fn new(axis: u32, window: u32, step: u32) -> Self {
        Self { axis, window, step }
    }

    /// Non-overlapping. The verifier turns this into "the adjoint is an
    /// elementwise mask-and-broadcast, not a scatter".
    pub const fn is_non_overlapping(self) -> bool {
        self.step >= self.window
    }
}

/// A dense tensor layout: offset + shape + strides, all inline. Contiguity
/// is derived on construction; reshape/flatten refuse a non-contiguous one.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Layout {
    offset: Dim,
    shape: Dims,
    strides: SmallVec<[Dim; 6]>,
    contiguous: bool,
}

impl Layout {
    pub fn contiguous(shape: &[Dim]) -> Self {
        let strides = Self::row_major_strides(shape);
        Self {
            offset: Dim::Const(0),
            shape: shape.iter().copied().collect(),
            strides,
            contiguous: true,
        }
    }

    /// Explicit layout. Contiguity is derived, never asserted by the caller.
    pub fn from_parts(offset: Dim, shape: &[Dim], strides: &[Dim]) -> Result<Self> {
        if shape.len() != strides.len() {
            return Err(Error::Shape(format!(
                "layout rank mismatch: {} shape dims, {} strides",
                shape.len(),
                strides.len()
            )));
        }
        let contiguous =
            matches!(offset, Dim::Const(0)) && strides == &Self::row_major_strides(shape)[..];
        Ok(Self {
            offset,
            shape: shape.iter().copied().collect(),
            strides: strides.iter().copied().collect(),
            contiguous,
        })
    }

    /// Row-major strides. A stride past a `Sym` axis is itself symbolic and
    /// is materialized into the uniform block at dispatch.
    pub fn row_major_strides(shape: &[Dim]) -> SmallVec<[Dim; 6]> {
        let mut out: SmallVec<[Dim; 6]> = smallvec::smallvec![Dim::Const(1); shape.len()];
        let mut acc: Option<u64> = Some(1);
        for axis in (0..shape.len()).rev() {
            out[axis] = match acc {
                Some(v) => Dim::Const(v),
                None => Dim::Sym(SymId(u32::MAX)),
            };
            acc = match (acc, shape[axis]) {
                (Some(a), Dim::Const(d)) => a.checked_mul(d),
                _ => None,
            };
        }
        out
    }

    pub const fn offset(&self) -> Dim {
        self.offset
    }
    pub fn shape(&self) -> &[Dim] {
        &self.shape
    }
    pub fn strides(&self) -> &[Dim] {
        &self.strides
    }
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
    pub const fn is_contiguous(&self) -> bool {
        self.contiguous
    }
    /// True when any stride is zero, i.e. the layout aliases itself.
    pub fn overlaps(&self) -> bool {
        self.strides.iter().any(|s| s.known_eq(Dim::Const(0)))
    }
}

/// One sub-axis of a logical axis. Strides may be zero (broadcast) or
/// collide (im2col); plain per-axis strides cannot express a conv operand.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SubAxis {
    pub extent: u32,
    pub stride: u32,
}

/// One logical axis, decomposed most-significant-first by divmod.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct AxisGroup {
    pub sub_axes: SmallVec<[SubAxis; 2]>,
}

impl AxisGroup {
    pub fn affine(extent: u32, stride: u32) -> Self {
        Self {
            sub_axes: smallvec::smallvec![SubAxis { extent, stride }],
        }
    }
}

/// Logical-to-storage index map: one [`AxisGroup`] per logical axis.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct MultiFlattenMap {
    pub groups: SmallVec<[AxisGroup; 4]>,
}

impl MultiFlattenMap {
    pub fn affine(extents: &[u32], strides: &[u32]) -> Self {
        Self {
            groups: extents
                .iter()
                .zip(strides)
                .map(|(&e, &s)| AxisGroup::affine(e, s))
                .collect(),
        }
    }

    pub fn rank(&self) -> usize {
        self.groups.len()
    }

    pub fn is_affine(&self) -> bool {
        self.groups.iter().all(|g| g.sub_axes.len() == 1)
    }

    /// Contiguous unit-stride run length as this axis's coordinate
    /// increments — the coalescing metric for picking the lane axis.
    pub fn axis_unit_run(&self, axis: usize) -> u32 {
        let mut run = 1u32;
        for sub in self.groups[axis].sub_axes.iter().rev() {
            if sub.stride != run {
                break;
            }
            run = run.saturating_mul(sub.extent);
        }
        run
    }

    /// Divmods a load through this map costs — the term the
    /// view-fold-vs-gather tradeoff prices.
    pub fn divmod_ops(&self) -> u64 {
        self.groups
            .iter()
            .map(|g| g.sub_axes.len().saturating_sub(1) as u64)
            .sum()
    }
}

/// Row-major strides of `shape` as `u64`, or `None` under a symbolic extent.
pub fn const_row_major(shape: &[Dim]) -> Option<Vec<u64>> {
    let mut out = vec![1u64; shape.len()];
    let mut acc = 1u64;
    for axis in (0..shape.len()).rev() {
        out[axis] = acc;
        acc = acc.checked_mul(shape[axis].as_const()?)?;
    }
    Some(out)
}

/// The `StrideSpec` for an inserted size-1 axis: multiplier 1 against a
/// neighbouring input dim, or a stride-0 axis when the input is rank 0 and
/// there is no neighbour to name.
pub fn singleton_spec(in_rank: usize, next_src: usize) -> StrideSpec {
    if in_rank == 0 {
        StrideSpec::broadcast(Dim::Const(1))
    } else {
        let neighbour = next_src.min(in_rank - 1);
        StrideSpec::dim(neighbour as u32, Dim::Const(1))
    }
}

/// Derive the spec vector for a reshape from `in_shape` to `out_shape`.
///
/// The two shapes are walked in lockstep and split into minimal groups of
/// equal product. A one-to-one group is a plain `dim` spec, so a symbolic
/// extent that passes through unchanged costs nothing. A many-to-one (merge)
/// or one-to-many (split) group names the group's innermost input axis and
/// multiplies its stride by the output axis's stride *within the group* —
/// exactly `Layout::contiguous(new_shape)` when the group is contiguous.
pub fn reshape_specs(in_shape: &[Dim], out_shape: &[Dim]) -> Result<SmallVec<[StrideSpec; 6]>> {
    let mut specs: SmallVec<[StrideSpec; 6]> = SmallVec::with_capacity(out_shape.len());
    let (in_len, out_len) = (in_shape.len(), out_shape.len());
    let (mut i, mut j) = (0usize, 0usize);

    while i < in_len || j < out_len {
        if i < in_len && j < out_len && in_shape[i].known_eq(out_shape[j]) {
            specs.push(StrideSpec::dim(i as u32, out_shape[j]));
            i += 1;
            j += 1;
            continue;
        }
        if i < in_len && in_shape[i].known_eq(Dim::Const(1)) {
            i += 1;
            continue;
        }
        if j < out_len && out_shape[j].known_eq(Dim::Const(1)) {
            specs.push(singleton_spec(in_len, i));
            j += 1;
            continue;
        }
        if i >= in_len || j >= out_len {
            return Err(Error::Shape(format!(
                "reshape {in_shape:?} -> {out_shape:?}: element counts disagree"
            )));
        }

        // A real group. Grow both sides until the products agree.
        let j0 = j;
        let mut pi = in_shape[i]
            .as_const()
            .ok_or_else(|| reshape_symbolic(in_shape, out_shape))?;
        let mut pj = out_shape[j]
            .as_const()
            .ok_or_else(|| reshape_symbolic(in_shape, out_shape))?;
        i += 1;
        j += 1;
        while pi != pj {
            if pi < pj {
                let d = in_shape
                    .get(i)
                    .and_then(|d| d.as_const())
                    .ok_or_else(|| reshape_symbolic(in_shape, out_shape))?;
                pi = pi
                    .checked_mul(d)
                    .ok_or_else(|| Error::Shape("reshape group overflows u64".into()))?;
                i += 1;
            } else {
                let d = out_shape
                    .get(j)
                    .and_then(|d| d.as_const())
                    .ok_or_else(|| reshape_symbolic(in_shape, out_shape))?;
                pj = pj
                    .checked_mul(d)
                    .ok_or_else(|| Error::Shape("reshape group overflows u64".into()))?;
                j += 1;
            }
        }

        let inner = i - 1; // innermost input axis of the group, stride 1 within it
        let group = &out_shape[j0..j];
        let local = const_row_major(group).ok_or_else(|| reshape_symbolic(in_shape, out_shape))?;
        for (k, &d) in group.iter().enumerate() {
            let mult = u32::try_from(local[k]).map_err(|_| {
                Error::Shape(format!("reshape multiplier {} exceeds u32", local[k]))
            })?;
            specs.push(StrideSpec::dim_with(inner as u32, d, mult));
        }
    }
    Ok(specs)
}

fn reshape_symbolic(in_shape: &[Dim], out_shape: &[Dim]) -> Error {
    Error::Shape(format!(
        "reshape {in_shape:?} -> {out_shape:?} would split or merge a symbolic extent; \
         a merged size is not expressible as a Dim"
    ))
}

/// The one broadcast rule, applied by the frontend before ingestion.
/// Right-aligned: a source dim is consumed when it equals the target or is
/// 1 (stride 0); unmatched target dims are inserted with stride 0 at **any**
/// position; an unconsumed source dim is an error. There is no implicit
/// broadcasting inside the IR.
pub fn broadcast_specs(src: &[Dim], dst: &[Dim]) -> Result<SmallVec<[StrideSpec; 6]>> {
    if dst.len() < src.len() {
        return Err(Error::Shape(format!(
            "cannot broadcast rank {} into rank {}",
            src.len(),
            dst.len()
        )));
    }
    let mut specs: SmallVec<[StrideSpec; 6]> = SmallVec::with_capacity(dst.len());
    let mut remaining = src.len();
    for target in dst.iter().rev().copied() {
        if remaining > 0 {
            let source = src[remaining - 1];
            if source.known_eq(target) {
                remaining -= 1;
                specs.push(StrideSpec::dim(remaining as u32, target));
                continue;
            }
            if source.known_eq(Dim::ONE) {
                remaining -= 1;
                specs.push(StrideSpec::broadcast(target));
                continue;
            }
        }
        specs.push(StrideSpec::broadcast(target));
    }
    if remaining != 0 {
        return Err(Error::Shape(format!(
            "broadcast left {remaining} source dim(s) unconsumed: {src:?} -> {dst:?}"
        )));
    }
    specs.reverse();
    Ok(specs)
}

/// The shape two operands broadcast to, or an error when neither is 1 at a
/// mismatched axis.
pub fn broadcast_shapes(a: &[Dim], b: &[Dim]) -> Result<Dims> {
    let rank = a.len().max(b.len());
    let mut out: Dims = smallvec::smallvec![Dim::ONE; rank];
    for i in 0..rank {
        let da = a.len().checked_sub(rank - i).map(|k| a[k]);
        let db = b.len().checked_sub(rank - i).map(|k| b[k]);
        out[i] = match (da, db) {
            (None, None) => Dim::ONE,
            (Some(d), None) | (None, Some(d)) => d,
            (Some(x), Some(y)) if x.known_eq(y) => x,
            (Some(x), Some(y)) if x.known_eq(Dim::ONE) => y,
            (Some(x), Some(y)) if y.known_eq(Dim::ONE) => x,
            (Some(x), Some(y)) => {
                return Err(Error::Shape(format!(
                    "cannot broadcast axis {i}: {x} vs {y}"
                )));
            }
        };
    }
    Ok(out)
}

/// Whether a restride's bounds are decidable at compile time. `Const` dims
/// are checked statically; a `Sym` records a runtime mask obligation on the
/// node, discharged by codegen.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BoundsProof {
    Static,
    RuntimeMask,
}
