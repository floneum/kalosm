//! The ~22 view ops. Every one is a vector of `StrideSpec`s over a single
//! `L0::Restride` — there is no view-op zoo, and a shape is never padded to
//! make one of them legal. `sliding_window_view` is the one exception: it
//! mints `L0::Window`, because its adjoint is decided by two integers and
//! injectivity of a relative stride composition is undecidable under `Sym`.
//!
//! Owned by W12.

use std::ops::Range;

use fusor2_ir::ir::level0::L0;
use fusor2_ir::shape::{
    BoundsProof, Dim, Dims, Layout, SlidingWindow, StrideSpec, SymId, reshape_specs,
    singleton_spec,
};

use crate::tensor::Tensor;
use crate::{Error, Result};

/// One entry of a [`Tensor::reshape`] target: a concrete extent or the single
/// inferred hole.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Extent {
    Dim(Dim),
    /// Exactly one of these is permitted; its extent is the element count
    /// divided by the product of the rest.
    Hole,
}

impl From<Dim> for Extent {
    fn from(d: Dim) -> Self {
        Self::Dim(d)
    }
}
impl From<usize> for Extent {
    fn from(d: usize) -> Self {
        Self::Dim(Dim::Const(d as u64))
    }
}
impl From<u64> for Extent {
    fn from(d: u64) -> Self {
        Self::Dim(Dim::Const(d))
    }
}
impl From<SymId> for Extent {
    fn from(s: SymId) -> Self {
        Self::Dim(Dim::Sym(s))
    }
}
impl From<()> for Extent {
    fn from(_: ()) -> Self {
        Self::Hole
    }
}

// ---------------------------------------------------------------------------
// bounds
// ---------------------------------------------------------------------------

/// The [`BoundsProof`] a spec vector carries over `in_shape`.
///
/// This is [`fusor2_autograd::tape::bounds_proof`] verbatim, deliberately: the
/// frontend and the adjoint transform must agree on what a view proves, or a
/// `Restride` and its own adjoint would carry different obligations. `Static`
/// exactly when every extent, offset and stride is `Const` **and** the
/// composed reach stays inside the input's element count; anything else is a
/// runtime mask obligation, and there is no third case.
///
/// Note that the reach is composed over the *whole* input, not per axis, so an
/// axis-merging reshape (`[2,3] -> [6]` reads `dim_with(1, 6, 1)`, addressing
/// six elements past a dim of extent three) is `Static` rather than rejected.
pub fn bounds_for(specs: &[StrideSpec], in_shape: &[Dim]) -> BoundsProof {
    fusor2_autograd::tape::bounds_proof(specs, in_shape)
}

// ---------------------------------------------------------------------------
// the primitive
// ---------------------------------------------------------------------------

impl Tensor {
    /// The one view primitive: a vector of relative [`StrideSpec`]s over a
    /// single `L0::Restride`. Every other view op in this file builds a spec
    /// vector and calls this.
    pub fn restride(&self, specs: &[StrideSpec]) -> Result<Tensor> {
        let bounds = bounds_for(specs, &self.shape());
        self.emit_here(L0::Restride {
            specs: specs.iter().copied().collect(),
            bounds,
            x: self.id,
        })
    }

    // -- reshape ------------------------------------------------------------

    /// Rank-changing reshape with at most one inferred [`Extent::Hole`].
    ///
    /// Refuses a target whose element count disagrees, and refuses to merge
    /// axes across a symbolic extent (the merged size would not be a `Dim`).
    /// Like the reference's `Layout::reshape`, correctness relies on the value
    /// being contiguous over each merged group.
    pub fn reshape(&self, shape: &[Extent]) -> Result<Tensor> {
        let target = self.resolve_extents(shape)?;
        let specs = reshape_specs(&self.shape(), &target)?;
        self.restride(&specs)
    }

    /// Reshape to a fully specified shape.
    pub fn reshape_dims(&self, shape: &[Dim]) -> Result<Tensor> {
        let extents: Vec<Extent> = shape.iter().copied().map(Extent::Dim).collect();
        self.reshape(&extents)
    }

    /// Resolve the single `Hole` in a reshape target.
    fn resolve_extents(&self, shape: &[Extent]) -> Result<Dims> {
        let holes = shape.iter().filter(|e| **e == Extent::Hole).count();
        if holes > 1 {
            return Err(Error::Shape(format!(
                "reshape accepts at most one hole, got {holes}"
            )));
        }
        if holes == 0 {
            return Ok(shape
                .iter()
                .map(|e| match e {
                    Extent::Dim(d) => *d,
                    Extent::Hole => unreachable!(),
                })
                .collect());
        }
        let total = self.elem_count().ok_or_else(|| {
            Error::Shape("cannot infer a reshape hole under a symbolic extent".into())
        })?;
        let mut known = 1u64;
        for e in shape {
            if let Extent::Dim(d) = e {
                let v = d.as_const().ok_or_else(|| {
                    Error::Shape("cannot infer a reshape hole beside a symbolic extent".into())
                })?;
                known = known.checked_mul(v).ok_or_else(|| {
                    Error::Shape("reshape target overflows a u64 element count".into())
                })?;
            }
        }
        if known == 0 || total % known != 0 {
            return Err(Error::Shape(format!(
                "reshape hole does not divide: {total} elements over a known product of {known}"
            )));
        }
        let hole = Dim::Const(total / known);
        Ok(shape
            .iter()
            .map(|e| match e {
                Extent::Dim(d) => *d,
                Extent::Hole => hole,
            })
            .collect())
    }

    // -- axis permutation ---------------------------------------------------

    /// Swap two axes.
    pub fn transpose(&self, d0: usize, d1: usize) -> Result<Tensor> {
        self.check_axis(d0, "transpose")?;
        self.check_axis(d1, "transpose")?;
        let mut order: Vec<usize> = (0..self.rank()).collect();
        order.swap(d0, d1);
        self.permute(&order)
    }

    /// Swap the last two axes. Requires rank >= 2.
    pub fn t(&self) -> Result<Tensor> {
        if self.rank() < 2 {
            return Err(Error::Shape(format!(
                "t() needs rank >= 2, got rank {}",
                self.rank()
            )));
        }
        self.transpose(self.rank() - 2, self.rank() - 1)
    }

    /// Arbitrary axis permutation. `order` must be a true permutation of
    /// `0..rank`.
    pub fn permute(&self, order: &[usize]) -> Result<Tensor> {
        let rank = self.rank();
        if order.len() != rank {
            return Err(Error::Shape(format!(
                "permute needs {rank} axes, got {}",
                order.len()
            )));
        }
        let mut seen = vec![false; rank];
        for &a in order {
            if a >= rank {
                return Err(Error::Shape(format!(
                    "permute axis {a} out of range for rank {rank}"
                )));
            }
            if std::mem::replace(&mut seen[a], true) {
                return Err(Error::Shape(format!("permute axis {a} is repeated")));
            }
        }
        let specs: Vec<StrideSpec> = order
            .iter()
            .map(|&a| StrideSpec::dim(a as u32, self.dim(a)))
            .collect();
        self.restride(&specs)
    }

    // -- slicing ------------------------------------------------------------

    /// Rank-preserving sub-view, one range per axis.
    pub fn slice(&self, ranges: &[Range<usize>]) -> Result<Tensor> {
        if ranges.len() != self.rank() {
            return Err(Error::Shape(format!(
                "slice needs one range per axis: {} ranges for rank {}",
                ranges.len(),
                self.rank()
            )));
        }
        let mut specs: Vec<StrideSpec> = Vec::with_capacity(ranges.len());
        for (i, r) in ranges.iter().enumerate() {
            if r.end < r.start {
                return Err(Error::Shape(format!(
                    "slice range {i} is inverted: {}..{}",
                    r.start, r.end
                )));
            }
            if let Some(extent) = self.dim(i).as_const()
                && r.end as u64 > extent
            {
                return Err(Error::Shape(format!(
                    "slice range {i} is {}..{} but axis {i} has extent {extent}",
                    r.start, r.end
                )));
            }
            specs.push(
                StrideSpec::dim(i as u32, Dim::Const((r.end - r.start) as u64))
                    .with_offset(Dim::Const(r.start as u64)),
            );
        }
        self.restride(&specs)
    }

    /// Slice one axis. Unlike [`Tensor::slice`] this leaves every other axis —
    /// including a symbolic one — untouched.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Tensor> {
        self.check_axis(dim, "narrow")?;
        if let Some(extent) = self.dim(dim).as_const()
            && (start + len) as u64 > extent
        {
            return Err(Error::Shape(format!(
                "narrow {start}..{} exceeds axis {dim} of extent {extent}",
                start + len
            )));
        }
        let specs: Vec<StrideSpec> = (0..self.rank())
            .map(|i| {
                if i == dim {
                    StrideSpec::dim(i as u32, Dim::Const(len as u64))
                        .with_offset(Dim::Const(start as u64))
                } else {
                    StrideSpec::dim(i as u32, self.dim(i))
                }
            })
            .collect();
        self.restride(&specs)
    }

    /// Split `dim` into `chunks` narrow views of `ceil(extent / chunks)`; the
    /// last one may be shorter.
    pub fn chunk(&self, chunks: usize, dim: usize) -> Result<Vec<Tensor>> {
        self.check_axis(dim, "chunk")?;
        if chunks == 0 {
            return Err(Error::Shape("chunk count must be nonzero".into()));
        }
        let extent = self
            .dim(dim)
            .as_const()
            .ok_or_else(|| Error::Shape("chunk needs a constant extent".into()))?
            as usize;
        let size = extent.div_ceil(chunks).max(1);
        let mut out = Vec::new();
        let mut start = 0usize;
        while start < extent {
            let len = size.min(extent - start);
            out.push(self.narrow(dim, start, len)?);
            start += len;
        }
        Ok(out)
    }

    // -- flatten ------------------------------------------------------------

    /// Reshape to rank 1.
    pub fn flatten_all(&self) -> Result<Tensor> {
        let n = self
            .elem_count()
            .ok_or_else(|| Error::Shape("flatten_all needs a constant element count".into()))?;
        self.reshape(&[Extent::Dim(Dim::Const(n))])
    }

    /// Collapse the last `from_end + 1` axes into one. `from_end == 0` is a
    /// no-op reshape.
    pub fn flatten_last_n(&self, from_end: usize) -> Result<Tensor> {
        let rank = self.rank();
        if from_end + 1 > rank {
            return Err(Error::Shape(format!(
                "flatten_last_n({from_end}) needs rank >= {}, got {rank}",
                from_end + 1
            )));
        }
        let keep = rank - from_end - 1;
        let shape = self.shape();
        let merged = shape[keep..]
            .iter()
            .try_fold(1u64, |acc, d| acc.checked_mul(d.as_const()?))
            .ok_or_else(|| Error::Shape("cannot flatten across a symbolic extent".into()))?;
        let mut target: Vec<Extent> = shape[..keep].iter().copied().map(Extent::Dim).collect();
        target.push(Extent::Dim(Dim::Const(merged)));
        self.reshape(&target)
    }

    /// Collapse the first `from_start + 1` axes into one.
    pub fn flatten_first_n(&self, from_start: usize) -> Result<Tensor> {
        let rank = self.rank();
        if from_start + 1 > rank {
            return Err(Error::Shape(format!(
                "flatten_first_n({from_start}) needs rank >= {}, got {rank}",
                from_start + 1
            )));
        }
        let take = from_start + 1;
        let shape = self.shape();
        let merged = shape[..take]
            .iter()
            .try_fold(1u64, |acc, d| acc.checked_mul(d.as_const()?))
            .ok_or_else(|| Error::Shape("cannot flatten across a symbolic extent".into()))?;
        let mut target: Vec<Extent> = vec![Extent::Dim(Dim::Const(merged))];
        target.extend(shape[take..].iter().copied().map(Extent::Dim));
        self.reshape(&target)
    }

    /// Collapse axes `from..=to` into one.
    pub fn flatten(&self, from: usize, to: usize) -> Result<Tensor> {
        if from > to {
            return Err(Error::Shape(format!("flatten range {from}..={to} is empty")));
        }
        self.check_axis(to, "flatten")?;
        let shape = self.shape();
        let merged = shape[from..=to]
            .iter()
            .try_fold(1u64, |acc, d| acc.checked_mul(d.as_const()?))
            .ok_or_else(|| Error::Shape("cannot flatten across a symbolic extent".into()))?;
        let mut target: Vec<Extent> = shape[..from].iter().copied().map(Extent::Dim).collect();
        target.push(Extent::Dim(Dim::Const(merged)));
        target.extend(shape[to + 1..].iter().copied().map(Extent::Dim));
        self.reshape(&target)
    }

    // -- squeeze / unsqueeze -------------------------------------------------

    /// Remove one size-1 axis.
    pub fn squeeze(&self, dim: usize) -> Result<Tensor> {
        self.squeeze_dims(&[dim])
    }

    /// Remove several size-1 axes.
    pub fn squeeze_dims(&self, axes: &[usize]) -> Result<Tensor> {
        let rank = self.rank();
        let mut drop = vec![false; rank];
        for &a in axes {
            self.check_axis(a, "squeeze")?;
            if !self.dim(a).known_eq(Dim::Const(1)) {
                return Err(Error::Shape(format!(
                    "squeeze axis {a} has extent {} , not 1",
                    self.dim(a)
                )));
            }
            drop[a] = true;
        }
        let specs: Vec<StrideSpec> = (0..rank)
            .filter(|i| !drop[*i])
            .map(|i| StrideSpec::dim(i as u32, self.dim(i)))
            .collect();
        self.restride(&specs)
    }

    /// Insert one size-1 axis at `dim` of the *output*.
    pub fn unsqueeze(&self, dim: usize) -> Result<Tensor> {
        self.unsqueeze_dims(&[dim])
    }

    /// Insert several size-1 axes, at the given positions of the output.
    ///
    /// The inserted axis is an **ordinary size-1 axis** (`multiplier == 1`
    /// against a neighbouring input dim), not a stride-0 broadcast axis. That
    /// distinction is load-bearing for the restride adjoint: a stride-0 axis
    /// reduces on the way back, a size-1 axis does not.
    pub fn unsqueeze_dims(&self, axes: &[usize]) -> Result<Tensor> {
        let in_rank = self.rank();
        let out_rank = in_rank + axes.len();
        let mut insert = vec![false; out_rank];
        for &a in axes {
            if a >= out_rank {
                return Err(Error::Shape(format!(
                    "unsqueeze axis {a} out of range for output rank {out_rank}"
                )));
            }
            if std::mem::replace(&mut insert[a], true) {
                return Err(Error::Shape(format!("unsqueeze axis {a} is repeated")));
            }
        }
        let mut specs: Vec<StrideSpec> = Vec::with_capacity(out_rank);
        let mut src = 0usize;
        for slot in insert.iter().take(out_rank) {
            if *slot {
                specs.push(singleton_spec(in_rank, src));
            } else {
                specs.push(StrideSpec::dim(src as u32, self.dim(src)));
                src += 1;
            }
        }
        self.restride(&specs)
    }

    // -- windows -------------------------------------------------------------

    /// Zero-copy overlapping windows: one `L0::Window`.
    ///
    /// Each windowed axis `i` becomes `(extent - window) / step + 1`
    /// positions, and one new trailing axis of size `window` per spec.
    pub fn sliding_window_view(&self, specs: &[SlidingWindow]) -> Result<Tensor> {
        let mut seen: Vec<u32> = Vec::with_capacity(specs.len());
        for w in specs {
            if w.axis as usize >= self.rank() {
                return Err(Error::Shape(format!(
                    "window axis {} out of range for rank {}",
                    w.axis,
                    self.rank()
                )));
            }
            if seen.contains(&w.axis) {
                return Err(Error::Shape(format!(
                    "window axis {} appears twice",
                    w.axis
                )));
            }
            seen.push(w.axis);
            if w.step == 0 || w.window == 0 {
                return Err(Error::Shape(
                    "window size and step must both be nonzero".into(),
                ));
            }
            if let Some(extent) = self.dim(w.axis as usize).as_const()
                && (w.window as u64) > extent
            {
                return Err(Error::Shape(format!(
                    "window {} does not fit axis {} of extent {extent}",
                    w.window, w.axis
                )));
            }
        }
        self.emit_here(L0::Window {
            specs: specs.iter().copied().collect(),
            x: self.id,
        })
    }

    /// One windowed axis, the common case.
    pub fn windows(&self, axis: u32, window: u32, step: u32) -> Result<Tensor> {
        self.sliding_window_view(&[SlidingWindow::new(axis, window, step)])
    }

    // -- layout escape hatch --------------------------------------------------

    /// Set the view wholesale from a precomputed [`Layout`].
    ///
    /// This is `attention_grads`' dk/dv aliasing escape hatch. One
    /// [`StrideSpec`] is derived per output axis by finding an input axis
    /// whose stride divides the target stride (`multiplier = target /
    /// in_stride`); the offset delta is decomposed in the input's row-major
    /// basis and attached to specs naming the corresponding axes. The input
    /// is taken to be contiguous over its own shape — that is the only layout
    /// the frontend knows, since the real one is an extraction decision.
    pub fn restride_layout(&self, target: &Layout) -> Result<Tensor> {
        let input = Layout::contiguous(&self.shape());
        let in_strides: Vec<u64> = input
            .strides()
            .iter()
            .map(|d| d.as_const().unwrap_or(0))
            .collect();

        let mut specs: Vec<StrideSpec> = Vec::with_capacity(target.rank());
        for (a, (&size, &stride)) in target.shape().iter().zip(target.strides()).enumerate() {
            let want = stride.as_const().ok_or_else(|| {
                Error::Shape(format!("restride_layout: axis {a} has a symbolic stride"))
            })?;
            if want == 0 {
                specs.push(StrideSpec::broadcast(size));
                continue;
            }
            // Prefer the largest divisor, i.e. the smallest multiplier.
            let pick = (0..in_strides.len())
                .filter(|&d| in_strides[d] != 0 && want % in_strides[d] == 0)
                .max_by_key(|&d| in_strides[d]);
            let Some(d) = pick else {
                return Err(Error::Shape(format!(
                    "restride_layout: no input stride divides target stride {want} on axis {a}"
                )));
            };
            let mult = want / in_strides[d];
            let mult = u32::try_from(mult).map_err(|_| {
                Error::Shape(format!("restride_layout: multiplier {mult} exceeds u32"))
            })?;
            specs.push(StrideSpec::dim_with(d as u32, size, mult));
        }

        // Distribute the offset delta over the specs that name each axis.
        let mut delta = target
            .offset()
            .as_const()
            .ok_or_else(|| Error::Shape("restride_layout: symbolic offset".into()))?;
        for d in 0..in_strides.len() {
            if delta == 0 {
                break;
            }
            let s = in_strides[d];
            if s == 0 || delta < s {
                continue;
            }
            let digit = delta / s;
            let slot = specs.iter().position(|sp| {
                sp.multiplier != 0 && sp.input_dim as usize == d && sp.offset.known_eq(Dim::Const(0))
            });
            let Some(slot) = slot else {
                return Err(Error::Shape(format!(
                    "restride_layout: offset component {digit} on axis {d} has no spec to carry it"
                )));
            };
            specs[slot] = specs[slot].with_offset(Dim::Const(digit));
            delta -= digit * s;
        }
        if delta != 0 {
            return Err(Error::Shape(format!(
                "restride_layout: {delta} of the offset is not expressible"
            )));
        }
        self.restride(&specs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(v: &[u64]) -> Vec<Dim> {
        v.iter().map(|&d| Dim::Const(d)).collect()
    }

    #[test]
    fn reshape_pass_through_is_one_to_one() {
        let s = reshape_specs(&dims(&[2, 3]), &dims(&[2, 3])).unwrap();
        assert_eq!(
            &s[..],
            &[
                StrideSpec::dim(0, Dim::Const(2)),
                StrideSpec::dim(1, Dim::Const(3))
            ]
        );
    }

    #[test]
    fn reshape_merges_onto_the_innermost_axis() {
        let s = reshape_specs(&dims(&[2, 3]), &dims(&[6])).unwrap();
        assert_eq!(&s[..], &[StrideSpec::dim_with(1, Dim::Const(6), 1)]);
    }

    #[test]
    fn reshape_splits_with_local_row_major_multipliers() {
        let s = reshape_specs(&dims(&[6]), &dims(&[2, 3])).unwrap();
        assert_eq!(
            &s[..],
            &[
                StrideSpec::dim_with(0, Dim::Const(2), 3),
                StrideSpec::dim_with(0, Dim::Const(3), 1),
            ]
        );
    }

    #[test]
    fn reshape_mixed_group_then_pass_through() {
        // [2,3,4] -> [6,4]
        let s = reshape_specs(&dims(&[2, 3, 4]), &dims(&[6, 4])).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s[0], StrideSpec::dim_with(1, Dim::Const(6), 1));
        assert_eq!(s[1], StrideSpec::dim(2, Dim::Const(4)));
    }

    #[test]
    fn reshape_keeps_a_symbolic_axis_that_passes_through() {
        let sym = Dim::Sym(SymId(3));
        let s = reshape_specs(&[sym, Dim::Const(6)], &[sym, Dim::Const(2), Dim::Const(3)]).unwrap();
        assert_eq!(s[0], StrideSpec::dim(0, sym));
        assert_eq!(s[1], StrideSpec::dim_with(1, Dim::Const(2), 3));
        assert_eq!(s[2], StrideSpec::dim_with(1, Dim::Const(3), 1));
    }

    #[test]
    fn reshape_refuses_to_merge_a_symbolic_axis() {
        let sym = Dim::Sym(SymId(3));
        assert!(reshape_specs(&[sym, Dim::Const(6)], &[Dim::Const(12)]).is_err());
    }

    #[test]
    fn reshape_inserts_and_drops_ones() {
        let s = reshape_specs(&dims(&[6]), &dims(&[1, 6])).unwrap();
        assert_eq!(s[0].size, Dim::Const(1));
        assert_eq!(s[0].multiplier, 1);
        assert_eq!(s[1], StrideSpec::dim(0, Dim::Const(6)));

        let s = reshape_specs(&dims(&[1, 6]), &dims(&[6])).unwrap();
        assert_eq!(&s[..], &[StrideSpec::dim(1, Dim::Const(6))]);
    }

    #[test]
    fn reshape_rank_zero_round_trip() {
        let s = reshape_specs(&[], &dims(&[1])).unwrap();
        assert_eq!(s[0].multiplier, 0);
        let s = reshape_specs(&dims(&[1]), &[]).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn bounds_are_static_only_when_everything_is_const_and_in_range() {
        let shape = dims(&[4]);
        let ok = [StrideSpec::dim(0, Dim::Const(3)).with_offset(Dim::Const(1))];
        assert_eq!(bounds_for(&ok, &shape), BoundsProof::Static);

        // offset 2 + (3-1)*1 = 4, one past the end: not statically provable.
        let oob = [StrideSpec::dim(0, Dim::Const(3)).with_offset(Dim::Const(2))];
        assert_eq!(bounds_for(&oob, &shape), BoundsProof::RuntimeMask);

        let sym = [Dim::Sym(SymId(1))];
        let masked = [StrideSpec::dim(0, Dim::Sym(SymId(1)))];
        assert_eq!(bounds_for(&masked, &sym), BoundsProof::RuntimeMask);
    }

    #[test]
    fn a_merge_is_in_range_against_the_whole_input() {
        // [2,3] -> [6] reads six elements past a dim of extent three; that is
        // exactly what makes a flatten a view, and the reach is composed over
        // the whole input rather than per axis.
        let s = reshape_specs(&dims(&[2, 3]), &dims(&[6])).unwrap();
        assert_eq!(bounds_for(&s, &dims(&[2, 3])), BoundsProof::Static);
        // Seven would not be.
        let too_far = [StrideSpec::dim_with(1, Dim::Const(7), 1)];
        assert_eq!(
            bounds_for(&too_far, &dims(&[2, 3])),
            BoundsProof::RuntimeMask
        );
    }

    #[test]
    fn singleton_specs_are_ordinary_axes_not_broadcasts() {
        assert_eq!(singleton_spec(2, 1).multiplier, 1);
        assert_eq!(singleton_spec(0, 0).multiplier, 0);
    }
}
