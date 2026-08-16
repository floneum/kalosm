//! The rope sin/cos table.
//!
//! `[rows, head_dim / 2]`, the half-width table
//! [`crate::composite::rope`] expands through its `table_expansion` gather.
//! [`base_inverse_frequency`] is shared with the rope op itself, so the table
//! and the kernel cannot drift apart.

use fusor_ir::dtype::Dtype;
use fusor_ir::shape::Dim;

use crate::composite::rope::base_inverse_frequency;
use crate::device::ok;
use crate::graph::Graph;
use crate::tensor::Dyn;
use crate::tensor::typed::Element;
use crate::{Error, Result, Tensor};

/// Precomputed rotary sin/cos, extended lazily as the sequence grows.
///
/// `T` is the table's element type. The rank is fixed at 2 — the table is
/// `[rows, head_dim / 2]` by construction.
pub struct RopeCache<T: Element = f32> {
    sin: Tensor<2, T>,
    cos: Tensor<2, T>,
    head_dim: u32,
    theta: f32,
    /// Rows currently in the table. Growth reuploads, so this is a host
    /// number, not a `Dim` the dispatch binds.
    rows: u64,
    graph: Graph,
}

impl<T: Element> RopeCache<T> {
    /// A table covering `max_len` positions.
    ///
    /// Refuses a symbolic `max_len`: the contents are host-computed sines and
    /// there is no length to compute them for until one is bound.
    pub fn new(graph: &Graph, head_dim: u32, max_len: Dim, theta: f32) -> Result<Self> {
        let Some(rows) = max_len.as_const() else {
            return Err(Error::Shape(
                "a rope table needs a concrete row count; bind the symbol, or narrow a \
                 table built for the model's context length"
                    .into(),
            ));
        };
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(Error::Shape(format!(
                "rope pairs head elements, so head_dim must be even and nonzero, got {head_dim}"
            )));
        }
        let (sin, cos) = build::<T>(graph, head_dim, rows, theta)?;
        Ok(Self {
            sin,
            cos,
            head_dim,
            theta,
            rows,
            graph: graph.clone(),
        })
    }

    /// Rows the table currently covers.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// Extend the table to cover `len`, if it does not already.
    ///
    /// Growth doubles, so a decode loop that runs to `n` reuploads
    /// `O(log n)` times rather than once per token.
    pub fn ensure(&mut self, len: Dim) -> Result<()> {
        let Some(want) = len.as_const() else {
            // A symbolic length is bound at dispatch and cannot exceed the
            // table the model was configured for; nothing to extend.
            return Ok(());
        };
        if want <= self.rows {
            return Ok(());
        }
        let rows = want.next_power_of_two().max(self.rows.saturating_mul(2));
        let (sin, cos) = build::<T>(&self.graph, self.head_dim, rows, self.theta)?;
        self.sin = sin;
        self.cos = cos;
        self.rows = rows;
        Ok(())
    }

    /// The `[len, head_dim / 2]` prefix starting at `offset` — what one
    /// decode step feeds `rope`.
    #[track_caller]
    pub fn slice(&self, offset: u64, len: u64) -> (Tensor<2, T>, Tensor<2, T>) {
        if offset + len > self.rows {
            ok::<()>(
                "RopeCache::slice",
                Err(Error::Shape(format!(
                    "rope rows {offset}..{} exceed the {}-row table; call ensure first",
                    offset + len,
                    self.rows
                ))),
            );
        }
        (
            self.sin.narrow(0usize, offset as usize, len as usize),
            self.cos.narrow(0usize, offset as usize, len as usize),
        )
    }
}

/// `sin`/`cos` of `pos * inv_freq[i]`, `[rows, head_dim / 2]` each.
fn build<T: Element>(
    graph: &Graph,
    head_dim: u32,
    rows: u64,
    theta: f32,
) -> Result<(Tensor<2, T>, Tensor<2, T>)> {
    let freqs = base_inverse_frequency(head_dim, theta);
    let half = freqs.len();
    let mut sin = Vec::with_capacity(rows as usize * half);
    let mut cos = Vec::with_capacity(rows as usize * half);
    for pos in 0..rows {
        for f in &freqs {
            // The angle is accumulated in f64: at position 100k a f32 product
            // has already lost the low bits of the fastest frequency.
            let angle = pos as f64 * *f as f64;
            sin.push((angle.sin() as f32).to_le_bytes());
            cos.push((angle.cos() as f32).to_le_bytes());
        }
    }
    let shape = [Dim::Const(rows), Dim::Const(half as u64)];
    let flat = |v: Vec<[u8; 4]>| -> Vec<u8> { v.into_iter().flatten().collect() };
    // The sines are computed in f64 and stored as f32 regardless of `T`, then
    // cast once per upload.
    let upload = |bytes: Vec<u8>| -> Result<Tensor<2, T>> {
        let dense: Dyn = graph.tensor(Dtype::F32, &shape, &bytes)?;
        let dense = if T::DTYPE == Dtype::F32 {
            dense
        } else {
            dense.cast(T::DTYPE)?
        };
        Tensor::<2, T>::try_from_dyn(dense)
    };
    Ok((upload(flat(sin))?, upload(flat(cos))?))
}
