//! The rope sin/cos table.
//!
//! `[rows, head_dim / 2]`, the half-width table
//! [`crate::composite::rope`] expands through its `table_expansion` gather —
//! not a `[rows, head_dim]` table with every column duplicated.
//! [`base_inverse_frequency`] is shared with the rope op itself, so the table
//! and the kernel cannot drift apart.
//!
//! Owned by W13.

use fusor2_ir::dtype::Dtype;
use fusor2_ir::shape::Dim;

use crate::composite::rope::base_inverse_frequency;
use crate::graph::Graph;
use crate::tensor::Tensor;
use crate::{Error, Result};

/// Precomputed rotary sin/cos, extended lazily as the sequence grows.
pub struct RopeCache {
    pub sin: Tensor,
    pub cos: Tensor,
    pub head_dim: u32,
    pub theta: f32,
    /// Rows currently in the table. Growth reuploads, so this is a host
    /// number, not a `Dim` the dispatch binds.
    rows: u64,
    graph: Graph,
}

impl RopeCache {
    /// A table covering `max_len` positions.
    ///
    /// A symbolic `max_len` is refused rather than guessed: the *use* of the
    /// table is symbolic (a narrow or a position gather, so a decode loop
    /// recompiles nothing), but its contents are host-computed sines and
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
        let (sin, cos) = build(graph, head_dim, rows, theta)?;
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
        let (sin, cos) = build(&self.graph, self.head_dim, rows, self.theta)?;
        self.sin = sin;
        self.cos = cos;
        self.rows = rows;
        Ok(())
    }

    /// The `[len, head_dim / 2]` prefix starting at `offset` — what one
    /// decode step feeds `rope`.
    pub fn slice(&self, offset: u64, len: u64) -> Result<(Tensor, Tensor)> {
        if offset + len > self.rows {
            return Err(Error::Shape(format!(
                "rope rows {offset}..{} exceed the {}-row table; call ensure first",
                offset + len,
                self.rows
            )));
        }
        Ok((
            self.sin.narrow(0, offset as usize, len as usize)?,
            self.cos.narrow(0, offset as usize, len as usize)?,
        ))
    }
}

/// `sin`/`cos` of `pos * inv_freq[i]`, `[rows, head_dim / 2]` each.
fn build(graph: &Graph, head_dim: u32, rows: u64, theta: f32) -> Result<(Tensor, Tensor)> {
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
    Ok((
        graph.tensor(Dtype::F32, &shape, &flat(sin))?,
        graph.tensor(Dtype::F32, &shape, &flat(cos))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Device, Session};

    fn graph() -> Graph {
        Graph::new(&Session::new(Device::cpu().unwrap()).unwrap())
    }

    #[test]
    fn the_table_is_half_width_and_matches_the_closed_form() {
        let g = graph();
        let c = RopeCache::new(&g, 8, Dim::Const(3), 10_000.0).unwrap();
        assert_eq!(&c.sin.shape()[..], &[Dim::Const(3), Dim::Const(4)]);
        assert_eq!(&c.cos.shape()[..], &[Dim::Const(3), Dim::Const(4)]);

        let freqs = base_inverse_frequency(8, 10_000.0);
        assert_eq!(freqs.len(), 4);
        let sin = c.sin.to_vec_f32().unwrap();
        let cos = c.cos.to_vec_f32().unwrap();
        for pos in 0..3usize {
            for (i, f) in freqs.iter().enumerate() {
                let angle = pos as f64 * *f as f64;
                let at = pos * 4 + i;
                assert!(
                    (sin[at] - angle.sin() as f32).abs() < 1e-6,
                    "sin[{pos}, {i}]"
                );
                assert!(
                    (cos[at] - angle.cos() as f32).abs() < 1e-6,
                    "cos[{pos}, {i}]"
                );
            }
        }
        // Row 0 is the identity rotation.
        assert_eq!(&sin[..4], &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(&cos[..4], &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn ensure_grows_only_upwards_and_keeps_the_values() {
        let g = graph();
        let mut c = RopeCache::new(&g, 4, Dim::Const(4), 10_000.0).unwrap();
        let before = c.sin.id();
        c.ensure(Dim::Const(4)).unwrap();
        assert_eq!(c.rows(), 4);
        assert_eq!(c.sin.id(), before, "no growth means no reupload");

        c.ensure(Dim::Const(9)).unwrap();
        assert_eq!(c.rows(), 16, "growth doubles past the request");
        assert_eq!(&c.sin.shape()[..], &[Dim::Const(16), Dim::Const(2)]);

        // The rows the old table covered are unchanged.
        let sin = c.sin.to_vec_f32().unwrap();
        let freqs = base_inverse_frequency(4, 10_000.0);
        for pos in 0..4usize {
            for (i, f) in freqs.iter().enumerate() {
                let want = (pos as f64 * *f as f64).sin() as f32;
                assert!((sin[pos * 2 + i] - want).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn a_slice_is_the_rows_a_decode_step_asks_for() {
        let g = graph();
        let c = RopeCache::new(&g, 4, Dim::Const(8), 10_000.0).unwrap();
        let (sin, cos) = c.slice(5, 2).unwrap();
        // The numbers a narrowed view reads back are `views::narrow`'s
        // obligation, and that case is red for a reason that is not this
        // cache: the emitters index every operand with the flat output index
        // and drop `Operand::layout`'s offset. What is asserted here is the
        // slice this cache asks for.
        assert_eq!(&sin.shape()[..], &[Dim::Const(2), Dim::Const(2)]);
        assert_eq!(&cos.shape()[..], &[Dim::Const(2), Dim::Const(2)]);
        assert!(c.slice(7, 2).is_err(), "a slice past the table is an error");
        assert!(c.slice(0, 8).is_ok(), "the whole table is a legal slice");
    }

    #[test]
    fn an_odd_head_dim_and_a_symbolic_length_are_errors_not_panics() {
        let g = graph();
        assert!(RopeCache::new(&g, 5, Dim::Const(4), 10_000.0).is_err());
        assert!(RopeCache::new(&g, 0, Dim::Const(4), 10_000.0).is_err());
        let s = g.sym("len");
        assert!(RopeCache::new(&g, 4, s, 10_000.0).is_err());
    }
}
