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
    Ok(graph.tensor(graph.buffer_leaf(dtype, shape)?))
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
        graph.bind_leaf(t.id, buf, None);
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
        leaf_buffer_node(graph, dtype, shape)
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

/// `values` encoded at `dtype`: narrowed for f16 and bf16, f32 bytes otherwise.
pub(crate) fn encode_f32(dtype: Dtype, values: &[f32]) -> Vec<u8> {
    match dtype {
        Dtype::F16 => values
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
            .collect(),
        Dtype::BF16 => values
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_bits().to_le_bytes())
            .collect(),
        _ => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
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

impl<T: FromArray + ?Sized> FromArray for &T {
    type Elem = T::Elem;
    fn to_parts(&self) -> Result<(Vec<Dim>, Vec<T::Elem>)> {
        (**self).to_parts()
    }
}

impl<D: Element> FromArray for Vec<D> {
    type Elem = D;
    fn to_parts(&self) -> Result<(Vec<Dim>, Vec<D>)> {
        self.as_slice().to_parts()
    }
}

/// Fixed-size arrays: the extents are the const parameters, outermost first,
/// and the data is one flattening.
macro_rules! arrays {
    ($([$($n:ident),*] $ty:ty => |$s:ident| $flat:expr;)*) => {$(
        impl<D: Element, $(const $n: usize),*> FromArray for $ty {
            type Elem = D;
            fn to_parts(&self) -> Result<(Vec<Dim>, Vec<D>)> {
                let $s = self;
                Ok((vec![$(Dim::Const($n as u64)),*], $flat.to_vec()))
            }
        }
    )*};
}

arrays! {
    [N] [D; N] => |a| a;
    [N, M] [[D; M]; N] => |a| a.as_flattened();
    [N, M, K] [[[D; K]; M]; N] => |a| a.as_flattened().as_flattened();
    [N, M, K, J] [[[[D; J]; K]; M]; N] => |a| a.as_flattened().as_flattened().as_flattened();
}

/// Nested `Vec`s: every row must have the same shape, stacked along a new
/// leading axis; no rows at all is `rank` zero extents.
macro_rules! nested_vecs {
    ($($ty:ty => $rank:literal;)*) => {$(
        impl<D: Element> FromArray for Vec<$ty> {
            type Elem = D;
            fn to_parts(&self) -> Result<(Vec<Dim>, Vec<D>)> {
                let Some(first) = self.first() else {
                    return Ok((vec![Dim::Const(0); $rank], Vec::new()));
                };
                let (inner, mut flat) = first.to_parts()?;
                for row in &self[1..] {
                    let (shape, data) = row.to_parts()?;
                    if shape != inner {
                        return Err(ragged());
                    }
                    flat.extend(data);
                }
                let mut shape = vec![Dim::Const(self.len() as u64)];
                shape.extend(inner);
                Ok((shape, flat))
            }
        }
    )*};
}

nested_vecs! {
    Vec<D> => 2;
    Vec<Vec<D>> => 3;
    Vec<Vec<Vec<D>>> => 4;
}

fn ragged() -> Error {
    Error::Shape("nested input is ragged; every sibling must have the same length".into())
}
