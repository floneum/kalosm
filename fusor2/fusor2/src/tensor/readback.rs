//! Host readback. One of exactly three host syncs in the runtime (the others
//! are an explicit wait and the allocator's memory-cap retry).
//!
//! A [`TensorSlice`] is bytes plus the [`Layout`] they were read under, so
//! indexing honours **offset and strides** rather than assuming contiguity.
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

    /// Logical extents.
    pub fn shape(&self) -> &[Dim] {
        self.layout.shape()
    }
    /// Element strides.
    pub fn strides(&self) -> &[Dim] {
        self.layout.strides()
    }
    /// The complete readback layout.
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
    /// Element dtype.
    pub fn dtype(&self) -> Dtype {
        self.dtype
    }
    /// Number of axes.
    pub fn rank(&self) -> usize {
        self.layout.rank()
    }
    /// Raw backing bytes.
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
        let shape = self.const_shape()?;
        if D::DTYPE != self.dtype {
            return Err(Error::Dtype(format!(
                "TensorSlice has dtype {:?}, not {:?}",
                self.dtype,
                D::DTYPE
            )));
        }
        let n: usize = shape.iter().product();
        let mut out = Vec::with_capacity(n);
        let mut idx = vec![0usize; shape.len()];
        for _ in 0..n {
            out.push(
                self.get::<D>(&idx)
                    .ok_or_else(|| Error::Shape("readback index out of range".into()))?,
            );
            for axis in (0..shape.len()).rev() {
                idx[axis] += 1;
                if idx[axis] < shape[axis] {
                    break;
                }
                idx[axis] = 0;
            }
        }
        Ok(out)
    }

    /// Every element widened to `f32`, row-major.
    ///
    /// [`TensorSlice::to_flat`] is the *typed* accessor and refuses a dtype it
    /// was not asked for; this is the *numeric* one. An f16 activation and a
    /// u32 token id both come back as the numbers they denote. A
    /// block-quantized value has no dense element and is still refused.
    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        if self.dtype == Dtype::F32 {
            return self.to_flat::<f32>();
        }
        let shape = self.const_shape()?;
        let n: usize = shape.iter().product();
        let mut out = Vec::with_capacity(n);
        let mut idx = vec![0usize; shape.len()];
        for _ in 0..n {
            out.push(self.element_f32(&idx)?);
            for axis in (0..shape.len()).rev() {
                idx[axis] += 1;
                if idx[axis] < shape[axis] {
                    break;
                }
                idx[axis] = 0;
            }
        }
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
    /// Read one element, or `None` for an out-of-bounds index.
    pub fn get(&self, idx: [usize; R]) -> Option<D> {
        self.0.get::<D>(&idx)
    }
    /// Logical extents.
    pub fn shape(&self) -> &[Dim] {
        self.0.shape()
    }
    fn extents(&self) -> Result<Vec<usize>> {
        self.0.const_shape()
    }
}

/// Convert a readback into the nested `Vec` matching its rank.
pub trait ToVec {
    /// Nested vector shape produced by this rank.
    type Output;
    /// Convert the readback to nested row-major vectors.
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
    /// The stride machinery above is still exercised by anything that builds
    /// a slice directly over a device layout.
    pub fn as_slice(&self) -> Result<TensorSlice> {
        let facts = self.facts();
        let bytes = self.graph.read_back(self.id)?;
        Ok(TensorSlice::new(
            bytes,
            Layout::contiguous(&facts.shape),
            facts.dtype,
        ))
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
    /// Flat `u32` copy.
    pub fn to_vec_u32(&self) -> Result<Vec<u32>> {
        self.to_flat::<u32>()
    }
    /// Flat `i32` copy.
    pub fn to_vec_i32(&self) -> Result<Vec<i32>> {
        self.to_flat::<i32>()
    }
    /// Raw bytes in the value's own dtype and layout.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(self.as_slice()?.bytes)
    }
    /// Asynchronously read every element as `f32`.
    pub async fn to_vec_f32_async(&self) -> Result<Vec<f32>> {
        self.as_slice()?.to_vec_f32()
    }
}
