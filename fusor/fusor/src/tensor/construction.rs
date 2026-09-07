//! Constructing leaves: parameters, buffers, constants, uniforms and the
//! shaped fills.
//!
//! `zeros`/`ones`/`splat`/`full` mint a `Logical::Leaf(LeafKind::Const)` with no
//! upload and no kernel. `arange` is built host-side and uploaded once.

use fusor_ir::dtype::{Dtype, Splat};
use fusor_ir::ir::logical::{LeafKind, Logical};
use fusor_ir::shape::{Dim, SymId};

use crate::graph::GraphRef;
use crate::tensor::typed::Element;
use crate::tensor::{Tensor, splat_one, splat_zero};
use crate::{Error, Result};

/// Mint a `Leaf::Buffer` with no host bytes and no device buffer.
pub(crate) fn leaf_buffer_node(graph: &GraphRef, dtype: Dtype, shape: &[Dim]) -> Result<Tensor> {
    Tensor::emit(
        graph,
        Logical::Leaf(LeafKind::Buffer {
            name: graph.fresh_buffer_id(),
            dtype,
            shape: shape.iter().copied().collect(),
        }),
    )
}

/// Mint a `Leaf::Buffer` and attach owned host bytes to it. The bytes stay on
/// the host until a resolve uploads them, and stay readable through
/// `leaf_bytes` afterwards — `arange` and `detach` need that.
pub(crate) fn upload(
    graph: &GraphRef,
    dtype: Dtype,
    shape: &[Dim],
    bytes: Vec<u8>,
) -> Result<Tensor> {
    let t = leaf_buffer_node(graph, dtype, shape)?;
    graph.set_leaf_bytes(t.id, bytes);
    Ok(t)
}

impl Tensor {
    /// Upload dense host bytes as a step-local buffer.
    ///
    /// One copy: the caller's slice goes straight into the transfer staging
    /// buffer.
    pub fn from_slice(
        graph: &GraphRef,
        dtype: Dtype,
        shape: &[Dim],
        data: &[u8],
    ) -> Result<Tensor> {
        let want = byte_len(dtype, shape)?;
        if data.len() as u64 != want {
            return Err(Error::Shape(format!(
                "from_slice: {} bytes for a {shape:?} {dtype:?} tensor that needs {want}",
                data.len()
            )));
        }
        let t = leaf_buffer_node(graph, dtype, shape)?;
        let persistence = graph.facts(t.id).persistence;
        let buf = graph.session().device().upload(data, persistence)?;
        graph.set_device_buf(t.id, buf);
        Ok(t)
    }

    /// Upload a typed host slice.
    pub fn from_elements<D: Element>(
        graph: &GraphRef,
        shape: &[Dim],
        data: &[D],
    ) -> Result<Tensor> {
        Self::from_slice(graph, D::DTYPE, shape, bytemuck::cast_slice(data))
    }

    /// Build from a nested Rust array, slice or `Vec`, inferring the shape
    /// from the nesting.
    pub fn new<A: FromArray>(graph: &GraphRef, data: A) -> Result<Tensor> {
        let (shape, flat) = data.to_parts()?;
        Self::from_elements(graph, &shape, &flat)
    }

    /// A constant fill. One `Leaf(Const)`: no upload, no kernel.
    pub fn splat(graph: &GraphRef, value: Splat, shape: &[Dim]) -> Result<Tensor> {
        Tensor::emit(
            graph,
            Logical::Leaf(LeafKind::Const {
                value,
                shape: shape.iter().copied().collect(),
            }),
        )
    }

    /// Argument-order alias of [`Tensor::splat`].
    pub fn full(graph: &GraphRef, shape: &[Dim], value: Splat) -> Result<Tensor> {
        Self::splat(graph, value, shape)
    }

    /// A zero-filled constant tensor.
    pub fn zeros(graph: &GraphRef, dtype: Dtype, shape: &[Dim]) -> Result<Tensor> {
        Self::splat(graph, splat_zero(dtype), shape)
    }

    /// A one-filled constant tensor.
    pub fn ones(graph: &GraphRef, dtype: Dtype, shape: &[Dim]) -> Result<Tensor> {
        Self::splat(graph, splat_one(dtype), shape)
    }

    /// A zero-filled constant with this value's shape and dtype.
    pub fn zeros_like(&self) -> Result<Tensor> {
        let facts = self.facts();
        Self::zeros(&self.graph, facts.dtype, &facts.shape)
    }

    /// A one-filled constant with this value's shape and dtype.
    pub fn ones_like(&self) -> Result<Tensor> {
        let facts = self.facts();
        Self::ones(&self.graph, facts.dtype, &facts.shape)
    }

    /// An uninitialized device allocation. The only constructor whose
    /// contents are undefined; every kernel that writes one must write all
    /// of it.
    pub fn uninit(graph: &GraphRef, dtype: Dtype, shape: &[Dim]) -> Result<Tensor> {
        Tensor::emit(
            graph,
            Logical::Leaf(LeafKind::Buffer {
                name: graph.fresh_buffer_id(),
                dtype,
                shape: shape.iter().copied().collect(),
            }),
        )
    }

    /// A trainable parameter: `Persistence::Persistent`, so a quantized
    /// repack amortizes against its lifetime and the extractor knows it may
    /// not recompute it.
    pub fn param(graph: &GraphRef, name: &str, dtype: Dtype, shape: &[Dim]) -> Result<Tensor> {
        let _ = name;
        Tensor::emit(
            graph,
            Logical::Leaf(LeafKind::Param {
                name: graph.fresh_buffer_id(),
                dtype,
                shape: shape.iter().copied().collect(),
            }),
        )
    }

    /// A runtime scalar read from binding 0. Not a `[1]` tensor and not a
    /// baked literal. Rank 0.
    pub fn uniform(graph: &GraphRef, dtype: Dtype, sym: SymId) -> Result<Tensor> {
        Tensor::emit(graph, Logical::Leaf(LeafKind::Uniform { sym, dtype }))
    }

    /// `[start, end)` with step 1, built host-side and uploaded.
    pub fn arange(graph: &GraphRef, dtype: Dtype, start: f64, end: f64) -> Result<Tensor> {
        Self::arange_step(graph, dtype, start, end, 1.0)
    }

    /// `[start, end)` with an arbitrary nonzero step; a negative step counts
    /// down.
    ///
    /// # Panics
    /// If `step == 0`.
    pub fn arange_step(
        graph: &GraphRef,
        dtype: Dtype,
        start: f64,
        end: f64,
        step: f64,
    ) -> Result<Tensor> {
        let bytes = arange_bytes(dtype, start, end, step)?;
        let n = bytes.len() as u64 / dtype.byte_size().max(1);
        // Callers read the sequence back through `leaf_bytes` without ever
        // resolving, so this leaf must keep its host bytes.
        upload(graph, dtype, &[Dim::Const(n)], bytes)
    }
}

/// Free-function spelling of [`Tensor::arange`].
pub fn arange(graph: &GraphRef, dtype: Dtype, start: f64, end: f64) -> Result<Tensor> {
    Tensor::arange(graph, dtype, start, end)
}

/// Free-function spelling of [`Tensor::arange_step`].
pub fn arange_step(
    graph: &GraphRef,
    dtype: Dtype,
    start: f64,
    end: f64,
    step: f64,
) -> Result<Tensor> {
    Tensor::arange_step(graph, dtype, start, end, step)
}

/// Element count times element size, or an error under a symbolic extent.
fn byte_len(dtype: Dtype, shape: &[Dim]) -> Result<u64> {
    if dtype.is_quantized() {
        return Err(Error::Dtype(
            "a dense upload cannot carry a quantized dtype".into(),
        ));
    }
    let n = shape
        .iter()
        .try_fold(1u64, |acc, d| acc.checked_mul(d.as_const()?))
        .ok_or_else(|| Error::Shape("cannot upload into a symbolic shape".into()))?;
    Ok(n * dtype.byte_size())
}

/// The host-side bytes of `arange_step`. Split out so the value sequence is
/// testable without a graph.
///
/// # Panics
/// If `step == 0`.
pub(crate) fn arange_bytes(dtype: Dtype, start: f64, end: f64, step: f64) -> Result<Vec<u8>> {
    assert!(step != 0.0, "arange_step needs a nonzero step");
    if dtype.is_quantized() {
        return Err(Error::Dtype("arange has no quantized form".into()));
    }
    let raw = (end - start) / step;
    let count = if raw <= 0.0 {
        0usize
    } else {
        raw.ceil() as usize
    };
    let mut out = Vec::with_capacity(count * dtype.byte_size() as usize);
    for i in 0..count {
        let v = start + step * i as f64;
        push_scalar(&mut out, dtype, v);
    }
    Ok(out)
}

fn push_scalar(out: &mut Vec<u8>, dtype: Dtype, v: f64) {
    match dtype {
        Dtype::F32 => out.extend_from_slice(&(v as f32).to_le_bytes()),
        Dtype::F16 => out.extend_from_slice(&half::f16::from_f64(v).to_bits().to_le_bytes()),
        Dtype::BF16 => out.extend_from_slice(&half::bf16::from_f64(v).to_bits().to_le_bytes()),
        Dtype::U32 => out.extend_from_slice(&(v as u32).to_le_bytes()),
        Dtype::I32 => out.extend_from_slice(&(v as i32).to_le_bytes()),
        Dtype::Q(_) => unreachable!("guarded by arange_bytes"),
    }
}

/// Nested host data whose shape is inferred from its nesting. Implemented for
/// arrays and `Vec`s up to depth 4 plus flat slices.
pub trait FromArray {
    /// Scalar element stored by this nested host value.
    type Elem: Element;
    /// The inferred shape and the row-major flattening.
    fn to_parts(&self) -> Result<(Vec<Dim>, Vec<Self::Elem>)>;
}

impl<D: Element> FromArray for [D] {
    type Elem = D;
    fn to_parts(&self) -> Result<(Vec<Dim>, Vec<D>)> {
        Ok((vec![Dim::Const(self.len() as u64)], self.to_vec()))
    }
}

impl<D: Element, const N: usize> FromArray for [D; N] {
    type Elem = D;
    fn to_parts(&self) -> Result<(Vec<Dim>, Vec<D>)> {
        Ok((vec![Dim::Const(N as u64)], self.to_vec()))
    }
}

impl<T: FromArray + ?Sized> FromArray for &T {
    type Elem = T::Elem;
    fn to_parts(&self) -> Result<(Vec<Dim>, Vec<T::Elem>)> {
        (**self).to_parts()
    }
}

impl<D: Element, const M: usize, const N: usize> FromArray for [[D; M]; N] {
    type Elem = D;
    fn to_parts(&self) -> Result<(Vec<Dim>, Vec<D>)> {
        let mut flat = Vec::with_capacity(N * M);
        for row in self {
            flat.extend_from_slice(row);
        }
        Ok((vec![Dim::Const(N as u64), Dim::Const(M as u64)], flat))
    }
}

impl<D: Element, const K: usize, const M: usize, const N: usize> FromArray for [[[D; K]; M]; N] {
    type Elem = D;
    fn to_parts(&self) -> Result<(Vec<Dim>, Vec<D>)> {
        let mut flat = Vec::with_capacity(N * M * K);
        for a in self {
            for b in a {
                flat.extend_from_slice(b);
            }
        }
        Ok((
            vec![
                Dim::Const(N as u64),
                Dim::Const(M as u64),
                Dim::Const(K as u64),
            ],
            flat,
        ))
    }
}

impl<D: Element, const J: usize, const K: usize, const M: usize, const N: usize> FromArray
    for [[[[D; J]; K]; M]; N]
{
    type Elem = D;
    fn to_parts(&self) -> Result<(Vec<Dim>, Vec<D>)> {
        let mut flat = Vec::with_capacity(N * M * K * J);
        for a in self {
            for b in a {
                for c in b {
                    flat.extend_from_slice(c);
                }
            }
        }
        Ok((
            vec![
                Dim::Const(N as u64),
                Dim::Const(M as u64),
                Dim::Const(K as u64),
                Dim::Const(J as u64),
            ],
            flat,
        ))
    }
}

impl<D: Element> FromArray for Vec<D> {
    type Elem = D;
    fn to_parts(&self) -> Result<(Vec<Dim>, Vec<D>)> {
        Ok((vec![Dim::Const(self.len() as u64)], self.clone()))
    }
}

impl<D: Element> FromArray for Vec<Vec<D>> {
    type Elem = D;
    fn to_parts(&self) -> Result<(Vec<Dim>, Vec<D>)> {
        let inner = self.first().map_or(0, Vec::len);
        let mut flat = Vec::with_capacity(self.len() * inner);
        for row in self {
            if row.len() != inner {
                return Err(ragged());
            }
            flat.extend_from_slice(row);
        }
        Ok((
            vec![Dim::Const(self.len() as u64), Dim::Const(inner as u64)],
            flat,
        ))
    }
}

impl<D: Element> FromArray for Vec<Vec<Vec<D>>> {
    type Elem = D;
    fn to_parts(&self) -> Result<(Vec<Dim>, Vec<D>)> {
        let mid = self.first().map_or(0, Vec::len);
        let inner = self.first().and_then(|r| r.first()).map_or(0, Vec::len);
        let mut flat = Vec::new();
        for a in self {
            if a.len() != mid {
                return Err(ragged());
            }
            for b in a {
                if b.len() != inner {
                    return Err(ragged());
                }
                flat.extend_from_slice(b);
            }
        }
        Ok((
            vec![
                Dim::Const(self.len() as u64),
                Dim::Const(mid as u64),
                Dim::Const(inner as u64),
            ],
            flat,
        ))
    }
}

impl<D: Element> FromArray for Vec<Vec<Vec<Vec<D>>>> {
    type Elem = D;
    fn to_parts(&self) -> Result<(Vec<Dim>, Vec<D>)> {
        let d1 = self.first().map_or(0, Vec::len);
        let d2 = self.first().and_then(|a| a.first()).map_or(0, Vec::len);
        let d3 = self
            .first()
            .and_then(|a| a.first())
            .and_then(|b| b.first())
            .map_or(0, Vec::len);
        let mut flat = Vec::new();
        for a in self {
            if a.len() != d1 {
                return Err(ragged());
            }
            for b in a {
                if b.len() != d2 {
                    return Err(ragged());
                }
                for c in b {
                    if c.len() != d3 {
                        return Err(ragged());
                    }
                    flat.extend_from_slice(c);
                }
            }
        }
        Ok((
            vec![
                Dim::Const(self.len() as u64),
                Dim::Const(d1 as u64),
                Dim::Const(d2 as u64),
                Dim::Const(d3 as u64),
            ],
            flat,
        ))
    }
}

fn ragged() -> Error {
    Error::Shape("nested input is ragged; every sibling must have the same length".into())
}
