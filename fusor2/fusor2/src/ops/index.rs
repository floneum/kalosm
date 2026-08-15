//! Indexing and scatter. `index_select`, `embedding` and `gather_last` are
//! all one `L0::Gather`; its adjoint is `Scatter{Add}`, which has four
//! coexisting lowerings the cost model chooses between.
//!
//! `Scatter{Set}` is the single write substrate: `slice_assign`, `cat`,
//! `stack`, `pad_axis`, `repeat` and `resize` are all a `Leaf(Const)` fill
//! plus one scatter per source, and `unique: true` is provable because the
//! written regions are disjoint by construction.

use std::ops::{Range, RangeFrom, RangeFull, RangeTo};

use fusor2_ir::dtype::Dtype;
use fusor2_ir::ir::level0::{L0, ScatterCombine};
use fusor2_ir::shape::Dim;

use crate::ops::view::Extent;
use crate::tensor::Tensor;
use crate::{Error, Result};

impl Tensor {
    // -- gather ---------------------------------------------------------------

    /// Gather rows along `dim` with a rank-1 `U32` index tensor.
    pub fn index_select(&self, dim: usize, idx: &Tensor) -> Result<Tensor> {
        self.check_axis(dim, "index_select")?;
        if idx.rank() != 1 {
            return Err(Error::Shape(format!(
                "index_select indices must be rank 1, not rank {}",
                idx.rank()
            )));
        }
        if !matches!(idx.dtype(), Dtype::U32 | Dtype::I32) {
            return Err(Error::Dtype(format!(
                "index_select indices must be U32 or I32, not {:?}",
                idx.dtype()
            )));
        }
        self.emit_here(L0::Gather {
            axis: dim as u32,
            x: self.id,
            idx: idx.id,
        })
    }

    /// Row lookup into a rank-2 embedding table with rank-N `u32` indices:
    /// `[..ids] -> [..ids, width]`.
    ///
    /// Its backward is a `Scatter{Add}`.
    pub fn embedding(&self, ids: &Tensor) -> Result<Tensor> {
        if self.rank() != 2 {
            return Err(Error::Shape(format!(
                "embedding needs a rank-2 table, got rank {}",
                self.rank()
            )));
        }
        let flat = ids.flatten_all()?;
        let rows = self.index_select(0, &flat)?;
        let mut target: Vec<Extent> = ids.shape().iter().copied().map(Extent::Dim).collect();
        target.push(Extent::Dim(self.dim(1)));
        rows.reshape(&target)
    }

    /// One element per row of a rank-2 value: `[rows, width]` picked by a
    /// rank-1 `[rows]` index into `[rows]`.
    ///
    /// Build the row offsets `0, width, 2*width, ...`, add the per-row column,
    /// and gather out of the flattened table.
    pub fn gather_last(&self, idx: &Tensor) -> Result<Tensor> {
        if self.rank() != 2 {
            return Err(Error::Shape(format!(
                "gather_last needs rank 2, got rank {}",
                self.rank()
            )));
        }
        let (rows, width) = row_offset_params(self.dim(0), self.dim(1))?;
        if !idx.dim(0).known_eq(self.dim(0)) {
            return Err(Error::Shape(format!(
                "gather_last needs one index per row: {} indices for {rows} rows",
                idx.dim(0)
            )));
        }
        let offsets = Tensor::arange_step(
            &self.graph,
            Dtype::U32,
            0.0,
            (rows * width) as f64,
            width as f64,
        )?;
        let linear = offsets.add(idx)?;
        self.flatten_all()?.index_select(0, &linear)
    }

    // -- scatter --------------------------------------------------------------

    /// `Scatter{Add}`; duplicate indices accumulate, which is normative — an
    /// embedding table receiving one token twice gets the summed gradient.
    pub fn scatter_add(&self, axis: usize, idx: &Tensor, updates: &Tensor) -> Result<Tensor> {
        self.scatter(axis, ScatterCombine::Add, idx, updates, false)
    }

    /// `Scatter{Set}`. `unique` is a caller-supplied proof; `verify_l0`
    /// rejects `Set` without it.
    pub fn scatter_set(
        &self,
        axis: usize,
        idx: &Tensor,
        updates: &Tensor,
        unique: bool,
    ) -> Result<Tensor> {
        self.scatter(axis, ScatterCombine::Set, idx, updates, unique)
    }

    fn scatter(
        &self,
        axis: usize,
        combine: ScatterCombine,
        idx: &Tensor,
        updates: &Tensor,
        unique: bool,
    ) -> Result<Tensor> {
        self.check_axis(axis, "scatter")?;
        // Enforce `Scatter{Set}` uniqueness here; the API-level proof is
        // supplied at this function.
        if matches!(combine, ScatterCombine::Set) && !unique {
            return Err(Error::Shape(
                "Scatter{Set} with possibly-duplicate indices; declare unique: true or use \
                 scatter_add"
                    .into(),
            ));
        }
        self.emit_here(L0::Scatter {
            axis: axis as u32,
            combine,
            base: self.id,
            idx: idx.id,
            upd: updates.id,
            unique,
        })
    }

    // -- slice_assign, the write substrate ------------------------------------

    /// A copy of `self` with the region named by `ranges` overwritten by
    /// `value`.
    ///
    /// When the region is full on every axis but one, this is a single
    /// `Scatter{Set}` along that axis with an index vector as long as the
    /// written extent — the `cat`/`pad`/`stack` case. When two or more axes
    /// are narrowed there is no single-axis form, so the value is flattened
    /// and scattered against an explicit index vector; that costs one
    /// host-built `u32` per written element and is the honest general
    /// lowering, not a fast path.
    pub fn slice_assign(&self, ranges: &[Range<usize>], value: &Tensor) -> Result<Tensor> {
        if ranges.len() != self.rank() {
            return Err(Error::Shape(format!(
                "slice_assign needs one range per axis: {} for rank {}",
                ranges.len(),
                self.rank()
            )));
        }
        if self.dtype() != value.dtype() {
            return Err(Error::Dtype(format!(
                "slice_assign dtype mismatch: {:?} vs {:?}",
                self.dtype(),
                value.dtype()
            )));
        }
        if value.rank() != self.rank() {
            return Err(Error::Shape(format!(
                "slice_assign value has rank {} but the base has rank {}",
                value.rank(),
                self.rank()
            )));
        }
        let mut narrowed: Vec<usize> = Vec::new();
        for (i, r) in ranges.iter().enumerate() {
            if r.end < r.start {
                return Err(Error::Shape(format!("slice_assign range {i} is inverted")));
            }
            let len = (r.end - r.start) as u64;
            if !value.dim(i).known_eq(Dim::Const(len)) {
                return Err(Error::Shape(format!(
                    "slice_assign value axis {i} is {} but the range is {len} wide",
                    value.dim(i)
                )));
            }
            let extent = self.dim(i).as_const().ok_or_else(|| {
                Error::Shape("slice_assign needs constant extents on the base".into())
            })?;
            if r.end as u64 > extent {
                return Err(Error::Shape(format!(
                    "slice_assign range {i} is {}..{} but the axis has extent {extent}",
                    r.start, r.end
                )));
            }
            if r.start != 0 || len != extent {
                narrowed.push(i);
            }
        }

        if narrowed.len() <= 1 {
            let axis = narrowed.first().copied().unwrap_or(0);
            let r = &ranges[axis];
            let idx = Tensor::arange(&self.graph, Dtype::U32, r.start as f64, r.end as f64)?;
            return self.scatter_set(axis, &idx, value, true);
        }

        // Two or more narrowed axes: flatten and scatter explicit positions.
        let shape = self.shape();
        let bytes = region_flat_indices(&shape, ranges)?;
        let count = bytes.len() as u64 / 4;
        let idx = Tensor::from_slice(&self.graph, Dtype::U32, &[Dim::Const(count)], &bytes)?;
        let base = self.flatten_all()?;
        let upd = value.flatten_all()?;
        let written = base.scatter_set(0, &idx, &upd, true)?;
        written.reshape_dims(&shape)
    }

    // -- cat / stack ----------------------------------------------------------

    /// Concatenate along `dim`: one `Leaf(Const)` fill plus one
    /// `Scatter{Set}` per part.
    pub fn cat(parts: &[Tensor], dim: usize) -> Result<Tensor> {
        cat(parts, dim)
    }

    /// Insert a new axis at `dim` and concatenate along it.
    pub fn stack(parts: &[Tensor], dim: usize) -> Result<Tensor> {
        stack(parts, dim)
    }

    // -- pad / repeat / resize --------------------------------------------------

    /// Zero-pad one axis.
    pub fn pad_axis(&self, axis: usize, padding: (usize, usize)) -> Result<Tensor> {
        self.pad_with_zeros(axis, padding.0, padding.1)
    }

    /// Zero-pad one axis by `left` before and `right` after.
    pub fn pad_with_zeros(&self, axis: usize, left: usize, right: usize) -> Result<Tensor> {
        self.check_axis(axis, "pad")?;
        if left == 0 && right == 0 {
            return Ok(self.clone());
        }
        let extent = self
            .dim(axis)
            .as_const()
            .ok_or_else(|| Error::Shape("pad needs a constant extent".into()))?
            as usize;
        let shape = self.shape();
        let mut out: Vec<Dim> = shape.to_vec();
        out[axis] = Dim::Const((left + extent + right) as u64);
        let base = Tensor::zeros(&self.graph, self.dtype(), &out)?;
        let ranges = full_ranges_with(&out, axis, left..left + extent)?;
        base.slice_assign(&ranges, self)
    }

    /// Tile the tensor `repeats[i]` times along each axis.
    ///
    /// A zero repeat short-circuits to a single `Leaf(Const)` of the
    /// degenerate shape — no scatter, no index upload.
    pub fn repeat(&self, repeats: &[usize]) -> Result<Tensor> {
        if repeats.len() != self.rank() {
            return Err(Error::Shape(format!(
                "repeat needs one count per axis: {} for rank {}",
                repeats.len(),
                self.rank()
            )));
        }
        if repeats.contains(&0) {
            let shape: Vec<Dim> = self
                .shape()
                .into_iter()
                .zip(repeats)
                .map(|(d, &r)| match d.as_const() {
                    Some(v) => Dim::Const(v * r as u64),
                    None if r == 0 => Dim::Const(0),
                    None => d,
                })
                .collect();
            return Tensor::zeros(&self.graph, self.dtype(), &shape);
        }
        let mut cur = self.clone();
        for (axis, &count) in repeats.iter().enumerate() {
            if count == 1 {
                continue;
            }
            let parts: Vec<Tensor> = std::iter::repeat_n(cur.clone(), count).collect();
            cur = cat(&parts, axis)?;
        }
        Ok(cur)
    }

    /// Pad-or-truncate each axis independently into a zero-filled result.
    pub fn resize(&self, new_shape: &[Dim]) -> Result<Tensor> {
        if new_shape.len() != self.rank() {
            return Err(Error::Shape(format!(
                "resize cannot change rank: {} vs {}",
                new_shape.len(),
                self.rank()
            )));
        }
        let out = Tensor::zeros(&self.graph, self.dtype(), new_shape)?;
        let mut overlap: Vec<Range<usize>> = Vec::with_capacity(self.rank());
        for i in 0..self.rank() {
            let old = self
                .dim(i)
                .as_const()
                .ok_or_else(|| Error::Shape("resize needs constant extents".into()))?;
            let new = new_shape[i]
                .as_const()
                .ok_or_else(|| Error::Shape("resize needs constant extents".into()))?;
            overlap.push(0..old.min(new) as usize);
        }
        if overlap.iter().any(|r| r.is_empty()) {
            return Ok(out);
        }
        let src = self.slice(&overlap)?;
        out.slice_assign(&overlap, &src)
    }

    // -- i() -------------------------------------------------------------------

    /// PyTorch-style indexing. Exactly one component must be a bare `usize`,
    /// which removes that axis; the rest are ranges.
    ///
    /// # Panics
    /// If the number of bare `usize` components is not exactly one. The
    /// message is `"i() needs exactly one bare usize index, got N"`.
    pub fn i<I: TensorIndex>(&self, index: I) -> Result<Tensor> {
        let ops = index.ops();
        if ops.len() != self.rank() {
            return Err(Error::Shape(format!(
                "i() needs one component per axis: {} for rank {}",
                ops.len(),
                self.rank()
            )));
        }
        let picks: Vec<usize> = ops
            .iter()
            .enumerate()
            .filter_map(|(i, o)| matches!(o, IndexOp::Index(_)).then_some(i))
            .collect();
        assert!(
            picks.len() == 1,
            "i() needs exactly one bare usize index, got {}",
            picks.len()
        );
        let removed = picks[0];

        let mut ranges: Vec<Range<usize>> = Vec::with_capacity(ops.len());
        for (i, op) in ops.iter().enumerate() {
            let extent = self
                .dim(i)
                .as_const()
                .ok_or_else(|| Error::Shape("i() needs constant extents".into()))?
                as usize;
            ranges.push(match op {
                IndexOp::Full => 0..extent,
                IndexOp::Range(r) => r.clone(),
                IndexOp::RangeTo(e) => 0..*e,
                IndexOp::RangeFrom(s) => *s..extent,
                IndexOp::Index(k) => *k..*k + 1,
            });
        }

        // When the removed axis is picked at 0 the dropped axis contributes no
        // offset, so the whole thing collapses into one `Restride`.
        // Otherwise it takes two, to preserve the relative-composition
        // property that `Restride` depends on.
        if ranges[removed].start == 0 {
            let specs: Vec<fusor2_ir::shape::StrideSpec> = ranges
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != removed)
                .map(|(i, r)| {
                    fusor2_ir::shape::StrideSpec::dim(i as u32, Dim::Const((r.end - r.start) as u64))
                        .with_offset(Dim::Const(r.start as u64))
                })
                .collect();
            return self.restride(&specs);
        }
        self.slice(&ranges)?.squeeze(removed)
    }
}

/// `rows` and `width` of a rank-2 value as constants.
fn row_offset_params(rows: Dim, width: Dim) -> Result<(u64, u64)> {
    match (rows.as_const(), width.as_const()) {
        (Some(r), Some(w)) => Ok((r, w)),
        _ => Err(Error::Shape(
            "gather_last needs constant row and column extents".into(),
        )),
    }
}

/// Full ranges on every axis but `axis`, which takes `r`.
fn full_ranges_with(shape: &[Dim], axis: usize, r: Range<usize>) -> Result<Vec<Range<usize>>> {
    let mut out = Vec::with_capacity(shape.len());
    for (i, d) in shape.iter().enumerate() {
        if i == axis {
            out.push(r.clone());
        } else {
            let e = d
                .as_const()
                .ok_or_else(|| Error::Shape("a symbolic extent has no explicit range".into()))?;
            out.push(0..e as usize);
        }
    }
    Ok(out)
}

/// Row-major flat positions of the region `ranges` inside `shape`, as
/// little-endian `u32` bytes ready to upload.
pub(crate) fn region_flat_indices(shape: &[Dim], ranges: &[Range<usize>]) -> Result<Vec<u8>> {
    let extents: Vec<u64> = shape
        .iter()
        .map(|d| {
            d.as_const()
                .ok_or_else(|| Error::Shape("a scatter region needs constant extents".into()))
        })
        .collect::<Result<_>>()?;
    let mut strides = vec![1u64; extents.len()];
    for i in (0..extents.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * extents[i + 1];
    }
    let count: usize = ranges.iter().map(|r| r.end - r.start).product();
    let mut out = Vec::with_capacity(count * 4);
    let mut cursor: Vec<usize> = ranges.iter().map(|r| r.start).collect();
    for _ in 0..count {
        let flat: u64 = cursor
            .iter()
            .enumerate()
            .map(|(i, &k)| k as u64 * strides[i])
            .sum();
        let flat = u32::try_from(flat)
            .map_err(|_| Error::Shape("scatter index exceeds u32".into()))?;
        out.extend_from_slice(&flat.to_le_bytes());
        for axis in (0..cursor.len()).rev() {
            cursor[axis] += 1;
            if cursor[axis] < ranges[axis].end {
                break;
            }
            cursor[axis] = ranges[axis].start;
        }
    }
    Ok(out)
}

/// Concatenate rank-R tensors along `dim`.
pub fn cat(parts: &[Tensor], dim: usize) -> Result<Tensor> {
    let Some(first) = parts.first() else {
        return Err(Error::Shape("cat needs at least one tensor".into()));
    };
    let rank = first.rank();
    if dim >= rank {
        return Err(Error::Shape(format!(
            "cat axis {dim} out of range for rank {rank}"
        )));
    }
    let mut total = 0u64;
    for p in parts {
        if p.rank() != rank {
            return Err(Error::Shape("cat operands differ in rank".into()));
        }
        if p.dtype() != first.dtype() {
            return Err(Error::Dtype("cat operands differ in dtype".into()));
        }
        for i in 0..rank {
            if i != dim && !p.dim(i).known_eq(first.dim(i)) {
                return Err(Error::Shape(format!(
                    "cat operands disagree on axis {i}: {} vs {}",
                    p.dim(i),
                    first.dim(i)
                )));
            }
        }
        total += p
            .dim(dim)
            .as_const()
            .ok_or_else(|| Error::Shape("cat needs a constant extent on the joined axis".into()))?;
    }
    if parts.len() == 1 {
        return Ok(first.clone());
    }

    let mut shape: Vec<Dim> = first.shape().to_vec();
    shape[dim] = Dim::Const(total);
    let mut out = Tensor::zeros(&first.graph, first.dtype(), &shape)?;
    let mut offset = 0usize;
    for p in parts {
        let len = p.dim(dim).as_const().unwrap_or(0) as usize;
        let ranges = full_ranges_with(&shape, dim, offset..offset + len)?;
        out = out.slice_assign(&ranges, p)?;
        offset += len;
    }
    Ok(out)
}

/// Insert a new axis at `dim` in every part and concatenate along it.
pub fn stack(parts: &[Tensor], dim: usize) -> Result<Tensor> {
    let lifted: Vec<Tensor> = parts
        .iter()
        .map(|p| p.unsqueeze(dim))
        .collect::<Result<_>>()?;
    cat(&lifted, dim)
}

// ---------------------------------------------------------------------------
// i() index descriptors
// ---------------------------------------------------------------------------

/// One component of an [`Tensor::i`] index tuple.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IndexOp {
    Full,
    Range(Range<usize>),
    RangeTo(usize),
    RangeFrom(usize),
    /// A bare `usize`; exactly one is permitted, and it removes the axis.
    Index(usize),
}

impl From<RangeFull> for IndexOp {
    fn from(_: RangeFull) -> Self {
        Self::Full
    }
}
impl From<Range<usize>> for IndexOp {
    fn from(r: Range<usize>) -> Self {
        Self::Range(r)
    }
}
impl From<RangeTo<usize>> for IndexOp {
    fn from(r: RangeTo<usize>) -> Self {
        Self::RangeTo(r.end)
    }
}
impl From<RangeFrom<usize>> for IndexOp {
    fn from(r: RangeFrom<usize>) -> Self {
        Self::RangeFrom(r.start)
    }
}
impl From<usize> for IndexOp {
    fn from(i: usize) -> Self {
        Self::Index(i)
    }
}

/// An [`Tensor::i`] argument: a tuple of components, one per axis.
pub trait TensorIndex {
    fn ops(self) -> Vec<IndexOp>;
}

impl TensorIndex for usize {
    fn ops(self) -> Vec<IndexOp> {
        vec![IndexOp::Index(self)]
    }
}

impl TensorIndex for Vec<IndexOp> {
    fn ops(self) -> Vec<IndexOp> {
        self
    }
}

macro_rules! index_tuple {
    ($(($($n:ident),*);)*) => {$(
        #[allow(non_snake_case)]
        impl<$($n: Into<IndexOp>),*> TensorIndex for ($($n,)*) {
            fn ops(self) -> Vec<IndexOp> {
                let ($($n,)*) = self;
                vec![$($n.into()),*]
            }
        }
    )*};
}

index_tuple! {
    (A);
    (A, B);
    (A, B, C);
    (A, B, C, D);
    (A, B, C, D, E);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(v: &[u64]) -> Vec<Dim> {
        v.iter().map(|&d| Dim::Const(d)).collect()
    }
    fn u32s(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn region_indices_are_row_major() {
        // Rows 1..3, cols 1..3 of a [4, 4] tensor.
        let b = region_flat_indices(&dims(&[4, 4]), &[1..3, 1..3]).unwrap();
        assert_eq!(u32s(&b), vec![5, 6, 9, 10]);
    }

    #[test]
    fn region_indices_of_a_full_slab() {
        let b = region_flat_indices(&dims(&[2, 3]), &[0..2, 1..2]).unwrap();
        assert_eq!(u32s(&b), vec![1, 4]);
    }

    #[test]
    fn region_indices_refuse_a_symbolic_extent() {
        let shape = [Dim::Sym(fusor2_ir::shape::SymId(0)), Dim::Const(3)];
        assert!(region_flat_indices(&shape, &[0..1, 0..3]).is_err());
    }

    #[test]
    fn index_op_conversions() {
        assert_eq!(IndexOp::from(..), IndexOp::Full);
        assert_eq!(IndexOp::from(1..3), IndexOp::Range(1..3));
        assert_eq!(IndexOp::from(..4), IndexOp::RangeTo(4));
        assert_eq!(IndexOp::from(2..), IndexOp::RangeFrom(2));
        assert_eq!(IndexOp::from(7usize), IndexOp::Index(7));
    }

    #[test]
    fn index_tuples_flatten_in_order() {
        let ops = (.., 2usize, 1..3).ops();
        assert_eq!(
            ops,
            vec![IndexOp::Full, IndexOp::Index(2), IndexOp::Range(1..3)]
        );
    }

    #[test]
    fn full_ranges_keep_every_other_axis_whole() {
        let r = full_ranges_with(&dims(&[2, 9]), 1, 3..6).unwrap();
        assert_eq!(r, vec![0..2, 3..6]);
    }

    #[test]
    fn gather_last_row_offsets() {
        // [3, 4] -> offsets 0, 4, 8; adding [1, 0, 3] gives [1, 4, 11].
        let (rows, width) = row_offset_params(Dim::Const(3), Dim::Const(4)).unwrap();
        let offsets = crate::tensor::construction::arange_bytes(
            Dtype::U32,
            0.0,
            (rows * width) as f64,
            width as f64,
        )
        .unwrap();
        assert_eq!(u32s(&offsets), vec![0, 4, 8]);
        let idx = [1u32, 0, 3];
        let linear: Vec<u32> = u32s(&offsets).iter().zip(idx).map(|(a, b)| a + b).collect();
        assert_eq!(linear, vec![1, 4, 11]);
    }
}
