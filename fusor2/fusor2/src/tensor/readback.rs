//! Host readback. One of exactly three host syncs in the runtime (the others
//! are an explicit wait and the allocator's memory-cap retry).
//!
//! A [`TensorSlice`] is bytes plus the [`Layout`] they were read under, so
//! indexing honours offset and strides rather than assuming contiguity.
//! Every accessor is total: a symbolic extent that never got bound is an
//! `Err`/`None`, never a panic.

use std::fmt;
use std::marker::PhantomData;
use std::ops::Index;

use fusor2_ir::dtype::Dtype;
use fusor2_ir::shape::{Dim, Layout};

use crate::tensor::typed::Element;
use crate::tensor::Tensor;
use crate::{Error, Result};

/// A host copy of one value, plus the layout it was read under.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSlice {
    bytes: Vec<u8>,
    layout: Layout,
    dtype: Dtype,
}

impl TensorSlice {
    /// Build a slice directly. Used by `Graph::read` and by tests.
    pub fn new(bytes: Vec<u8>, layout: Layout, dtype: Dtype) -> Self {
        Self {
            bytes,
            layout,
            dtype,
        }
    }

    pub fn shape(&self) -> &[Dim] {
        self.layout.shape()
    }
    pub fn strides(&self) -> &[Dim] {
        self.layout.strides()
    }
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
    pub fn rank(&self) -> usize {
        self.layout.rank()
    }
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The flat element index of `idx`, honouring offset and strides.
    /// `None` when the rank disagrees, an index is out of range, or any
    /// extent or stride is still symbolic.
    pub fn linear_index(&self, idx: &[usize]) -> Option<u64> {
        if idx.len() != self.rank() {
            return None;
        }
        let mut flat = self.layout.offset().as_const()?;
        for (i, &k) in idx.iter().enumerate() {
            let extent = self.layout.shape()[i].as_const()?;
            if k as u64 >= extent {
                return None;
            }
            let stride = self.layout.strides()[i].as_const()?;
            flat = flat.checked_add(k as u64 * stride)?;
        }
        Some(flat)
    }

    /// The raw bytes of one element.
    pub fn element_bytes(&self, idx: &[usize]) -> Option<&[u8]> {
        let size = self.dtype.byte_size() as usize;
        if size == 0 {
            return None;
        }
        let start = self.linear_index(idx)? as usize * size;
        self.bytes.get(start..start + size)
    }

    /// One element, read at `D`. `None` on a dtype mismatch or a bad index.
    pub fn get<D: Element>(&self, idx: &[usize]) -> Option<D> {
        if D::DTYPE != self.dtype {
            return None;
        }
        let raw = self.element_bytes(idx)?;
        Some(bytemuck::pod_read_unaligned(raw))
    }

    /// A rank- and dtype-checked view, which is what [`ToVec`] hangs off.
    pub fn ranked<const R: usize, D: Element>(&self) -> Result<Ranked<'_, R, D>> {
        if self.rank() != R {
            return Err(Error::Shape(format!(
                "TensorSlice has rank {}, not {R}",
                self.rank()
            )));
        }
        if self.dtype != D::DTYPE {
            return Err(Error::Dtype(format!(
                "TensorSlice has dtype {:?}, not {:?}",
                self.dtype,
                D::DTYPE
            )));
        }
        Ok(Ranked(self, PhantomData))
    }

    /// Element 0 of a rank-0 (or single-element) value.
    pub fn scalar<D: Element>(&self) -> Result<D> {
        if D::DTYPE != self.dtype {
            return Err(Error::Dtype(format!(
                "TensorSlice has dtype {:?}, not {:?}",
                self.dtype,
                D::DTYPE
            )));
        }
        let zeros = vec![0usize; self.rank()];
        self.get::<D>(&zeros)
            .ok_or_else(|| Error::Shape("TensorSlice is empty or has an unbound extent".into()))
    }

    /// Extents as `usize`, or an error when one is still symbolic.
    fn const_shape(&self) -> Result<Vec<usize>> {
        self.layout
            .shape()
            .iter()
            .map(|d| {
                d.as_const().map(|v| v as usize).ok_or_else(|| {
                    Error::Shape("cannot read a value whose extent is still symbolic".into())
                })
            })
            .collect()
    }

    /// Row-major copy of every element, ignoring the layout's own order.
    pub fn to_flat<D: Element>(&self) -> Result<Vec<D>> {
        if D::DTYPE != self.dtype {
            return Err(Error::Dtype(format!(
                "TensorSlice has dtype {:?}, not {:?}",
                self.dtype,
                D::DTYPE
            )));
        }
        self.gather(|s, idx| {
            s.get::<D>(idx)
                .ok_or_else(|| Error::Shape("readback index out of range".into()))
        })
    }

    /// Every element widened to `f32`, row-major.
    ///
    /// [`TensorSlice::to_flat`] is the typed accessor and refuses a dtype it
    /// was not asked for; this is the numeric one, so an f16 activation and a
    /// u32 token id both come back as the numbers they denote. A
    /// block-quantized value has no dense element and is refused.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        if self.dtype == Dtype::F32 {
            return self.to_flat::<f32>();
        }
        self.gather(Self::element_f32)
    }

    /// Row-major visit of every position, parameterized by the element
    /// decoder.
    fn gather<T>(&self, read: impl Fn(&Self, &[usize]) -> Result<T>) -> Result<Vec<T>> {
        let shape = self.const_shape()?;
        let ranges: Vec<std::ops::Range<usize>> = shape.iter().map(|&e| 0..e).collect();
        let mut out = Vec::with_capacity(shape.iter().product());
        for_each_position(&ranges, |idx| {
            out.push(read(self, idx)?);
            Ok(())
        })?;
        Ok(out)
    }

    /// One element as the number it denotes.
    fn element_f32(&self, idx: &[usize]) -> Result<f32> {
        let raw = self
            .element_bytes(idx)
            .ok_or_else(|| Error::Shape("readback index out of range".into()))?;
        Ok(match self.dtype {
            Dtype::F32 => f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
            Dtype::F16 => half::f16::from_le_bytes([raw[0], raw[1]]).to_f32(),
            Dtype::BF16 => half::bf16::from_le_bytes([raw[0], raw[1]]).to_f32(),
            Dtype::U32 => u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as f32,
            Dtype::I32 => i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as f32,
            Dtype::Q(fmt) => {
                return Err(Error::Dtype(format!(
                    "a {fmt:?} value has no dense element; dequantize it first"
                )));
            }
        })
    }
}

/// Row-major odometer over one `Range` per axis, last axis fastest: `f` runs
/// at every position, and a failure stops the walk. The empty product — rank
/// 0 — visits the single empty position once; an empty range visits nothing.
pub(crate) fn for_each_position(
    ranges: &[std::ops::Range<usize>],
    mut f: impl FnMut(&[usize]) -> Result<()>,
) -> Result<()> {
    let count: usize = ranges.iter().map(|r| r.len()).product();
    let mut cursor: Vec<usize> = ranges.iter().map(|r| r.start).collect();
    for _ in 0..count {
        f(&cursor)?;
        for axis in (0..cursor.len()).rev() {
            cursor[axis] += 1;
            if cursor[axis] < ranges[axis].end {
                break;
            }
            cursor[axis] = ranges[axis].start;
        }
    }
    Ok(())
}

impl<const N: usize> Index<[usize; N]> for TensorSlice {
    /// The element's raw bytes: a `TensorSlice` is not generic over its
    /// element type, so `Index` cannot hand back a typed reference. Use
    /// [`TensorSlice::get`] for that.
    type Output = [u8];
    fn index(&self, idx: [usize; N]) -> &[u8] {
        self.element_bytes(&idx)
            .expect("TensorSlice index out of range")
    }
}

/// A rank- and dtype-checked view of a [`TensorSlice`].
#[derive(Copy, Clone)]
pub struct Ranked<'a, const R: usize, D: Element>(&'a TensorSlice, PhantomData<D>);

impl<const R: usize, D: Element> Ranked<'_, R, D> {
    pub fn get(&self, idx: [usize; R]) -> Option<D> {
        self.0.get::<D>(&idx)
    }
    pub fn shape(&self) -> &[Dim] {
        self.0.shape()
    }
    fn extents(&self) -> Result<Vec<usize>> {
        self.0.const_shape()
    }
}

/// Convert a readback into the nested `Vec` matching its rank.
pub trait ToVec {
    type Output;
    fn to_vec(&self) -> Self::Output;
}

impl<D: Element> ToVec for Ranked<'_, 1, D> {
    type Output = Result<Vec<D>>;
    fn to_vec(&self) -> Result<Vec<D>> {
        let s = self.extents()?;
        (0..s[0])
            .map(|i| {
                self.get([i])
                    .ok_or_else(|| Error::Shape("readback index out of range".into()))
            })
            .collect()
    }
}

impl<D: Element> ToVec for Ranked<'_, 2, D> {
    type Output = Result<Vec<Vec<D>>>;
    fn to_vec(&self) -> Result<Vec<Vec<D>>> {
        let s = self.extents()?;
        (0..s[0])
            .map(|i| {
                (0..s[1])
                    .map(|j| {
                        self.get([i, j])
                            .ok_or_else(|| Error::Shape("readback index out of range".into()))
                    })
                    .collect()
            })
            .collect()
    }
}

impl<D: Element> ToVec for Ranked<'_, 3, D> {
    type Output = Result<Vec<Vec<Vec<D>>>>;
    fn to_vec(&self) -> Result<Vec<Vec<Vec<D>>>> {
        let s = self.extents()?;
        (0..s[0])
            .map(|i| {
                (0..s[1])
                    .map(|j| {
                        (0..s[2])
                            .map(|k| {
                                self.get([i, j, k]).ok_or_else(|| {
                                    Error::Shape("readback index out of range".into())
                                })
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }
}

impl<const R: usize, D: Element + fmt::Debug> fmt::Debug for Ranked<'_, R, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Ok(s) = self.extents() else {
            return write!(f, "Tensor(<unbound extent>)");
        };
        match R {
            0 => write!(f, "{:?}", self.0.get::<D>(&[])),
            1 => {
                let row: Vec<Option<D>> = (0..s[0]).map(|i| self.0.get::<D>(&[i])).collect();
                write!(f, "{row:?}")
            }
            2 => {
                let m: Vec<Vec<Option<D>>> = (0..s[0])
                    .map(|i| (0..s[1]).map(|j| self.0.get::<D>(&[i, j])).collect())
                    .collect();
                write!(f, "{m:?}")
            }
            3 => {
                let m: Vec<Vec<Vec<Option<D>>>> = (0..s[0])
                    .map(|i| {
                        (0..s[1])
                            .map(|j| (0..s[2]).map(|k| self.0.get::<D>(&[i, j, k])).collect())
                            .collect()
                    })
                    .collect();
                write!(f, "{m:?}")
            }
            _ => write!(f, "Tensor(rank {R})"),
        }
    }
}

impl Tensor {
    /// Resolve the graph up to this value and copy it back.
    ///
    /// `Session::read_bytes` hands back a contiguous copy, so the layout a
    /// [`TensorSlice`] carries is the value's own shape in row-major order.
    pub fn as_slice(&self) -> Result<TensorSlice> {
        let facts = self.facts();
        let bytes = self.graph.read_back(self.id)?;
        Ok(TensorSlice::new(
            bytes,
            Layout::contiguous(&facts.shape),
            facts.dtype,
        ))
    }

    /// Async spelling of [`Tensor::as_slice`], for callers driving a runtime.
    pub fn as_slice_async(&self) -> impl Future<Output = Result<TensorSlice>> + '_ {
        std::future::ready(self.as_slice())
    }

    /// The single element of a rank-0 value.
    pub fn to_scalar<D: Element>(&self) -> Result<D> {
        self.as_slice()?.scalar::<D>()
    }

    /// Row-major host copy.
    pub fn to_flat<D: Element>(&self) -> Result<Vec<D>> {
        self.as_slice()?.to_flat::<D>()
    }

    /// Every element as the number it denotes, whatever the value's dtype is.
    /// See [`TensorSlice::to_vec_f32`].
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        self.as_slice()?.to_vec_f32()
    }
    pub fn to_vec_u32(&self) -> Result<Vec<u32>> {
        self.to_flat::<u32>()
    }
    pub fn to_vec_i32(&self) -> Result<Vec<i32>> {
        self.to_flat::<i32>()
    }
    /// Raw bytes in the value's own dtype and layout.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(self.as_slice()?.bytes)
    }
    pub async fn to_vec_f32_async(&self) -> Result<Vec<f32>> {
        self.as_slice()?.to_vec_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dims(v: &[u64]) -> Vec<Dim> {
        v.iter().map(|&d| Dim::Const(d)).collect()
    }

    /// offset = 1, shape [2,2], strides [3,1] over 0..24 f32 picks
    /// [[1,2],[4,5]].
    fn strided() -> TensorSlice {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let layout =
            Layout::from_parts(Dim::Const(1), &dims(&[2, 2]), &dims(&[3, 1])).unwrap();
        TensorSlice::new(bytemuck::cast_slice(&data).to_vec(), layout, Dtype::F32)
    }

    #[test]
    fn get_honours_offset_and_strides() {
        let s = strided();
        assert_eq!(s.get::<f32>(&[0, 0]), Some(1.0));
        assert_eq!(s.get::<f32>(&[0, 1]), Some(2.0));
        assert_eq!(s.get::<f32>(&[1, 0]), Some(4.0));
        assert_eq!(s.get::<f32>(&[1, 1]), Some(5.0));
        // Out of range, wrong rank and wrong dtype are all `None`.
        assert_eq!(s.get::<f32>(&[2, 0]), None);
        assert_eq!(s.get::<f32>(&[0]), None);
        assert_eq!(s.get::<u32>(&[0, 0]), None);
    }

    #[test]
    fn to_vec_walks_the_strides() {
        let s = strided();
        let v = s.ranked::<2, f32>().unwrap().to_vec().unwrap();
        assert_eq!(v, vec![vec![1.0, 2.0], vec![4.0, 5.0]]);
    }

    #[test]
    fn index_returns_the_element_bytes() {
        let s = strided();
        assert_eq!(f32::from_le_bytes(s[[1, 1]].try_into().unwrap()), 5.0);
    }

    #[test]
    fn ranked_checks_rank_and_dtype() {
        let s = strided();
        assert!(s.ranked::<3, f32>().is_err());
        assert!(s.ranked::<2, u32>().is_err());
        assert!(s.ranked::<2, f32>().is_ok());
    }

    #[test]
    fn an_unbound_extent_is_an_error_not_a_panic() {
        let layout = Layout::from_parts(
            Dim::Const(0),
            &[Dim::Sym(fusor2_ir::shape::SymId(0))],
            &[Dim::Const(1)],
        )
        .unwrap();
        let s = TensorSlice::new(vec![0u8; 16], layout, Dtype::F32);
        assert!(s.to_flat::<f32>().is_err());
        assert_eq!(s.get::<f32>(&[0]), None);
    }

    #[test]
    fn rank_zero_scalar_reads() {
        let layout = Layout::contiguous(&[]);
        let s = TensorSlice::new(2.5f32.to_le_bytes().to_vec(), layout, Dtype::F32);
        assert_eq!(s.rank(), 0);
        assert_eq!(s.scalar::<f32>().unwrap(), 2.5);
    }

    #[test]
    fn debug_renders_ranks_zero_to_three() {
        let s = strided();
        let r = s.ranked::<2, f32>().unwrap();
        assert_eq!(format!("{r:?}"), "[[Some(1.0), Some(2.0)], [Some(4.0), Some(5.0)]]");
    }
}
