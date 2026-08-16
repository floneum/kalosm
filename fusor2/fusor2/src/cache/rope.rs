//! The rope sin/cos table.
//!
//! `[rows, head_dim / 2]`, the half-width table
//! [`crate::composite::rope`] expands through its `table_expansion` gather.
//! [`base_inverse_frequency`] is shared with the rope op itself, so the table
//! and the kernel cannot drift apart.

use fusor2_ir::dtype::Dtype;
use fusor2_ir::shape::Dim;

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
    pub sin: Tensor<2, T>,
    pub cos: Tensor<2, T>,
    pub head_dim: u32,
    pub theta: f32,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{Backend, Session};

    fn graph() -> Graph {
        Graph::new(&Session::new(Backend::cpu().unwrap()).unwrap())
    }

    #[test]
    fn the_table_is_half_width_and_matches_the_closed_form() {
        let g = graph();
        let c: RopeCache = RopeCache::new(&g, 8, Dim::Const(3), 10_000.0).unwrap();
        assert_eq!(c.sin.shape(), [3, 4]);
        assert_eq!(c.cos.shape(), [3, 4]);

        let freqs = base_inverse_frequency(8, 10_000.0);
        assert_eq!(freqs.len(), 4);
        let sin = c.sin.to_vec_f32();
        let cos = c.cos.to_vec_f32();
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
        let mut c: RopeCache = RopeCache::new(&g, 4, Dim::Const(4), 10_000.0).unwrap();
        let before = c.sin.id();
        c.ensure(Dim::Const(4)).unwrap();
        assert_eq!(c.rows(), 4);
        assert_eq!(c.sin.id(), before, "no growth means no reupload");

        c.ensure(Dim::Const(9)).unwrap();
        assert_eq!(c.rows(), 16, "growth doubles past the request");
        assert_eq!(c.sin.shape(), [16, 2]);

        // The rows the old table covered are unchanged.
        let sin = c.sin.to_vec_f32();
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
        let c: RopeCache = RopeCache::new(&g, 4, Dim::Const(8), 10_000.0).unwrap();
        let (sin, cos) = c.slice(5, 2);
        // The values a narrowed view reads back are `views::narrow`'s
        // obligation; what is asserted here is the slice this cache asks for.
        assert_eq!(sin.shape(), [2, 2]);
        assert_eq!(cos.shape(), [2, 2]);
        // The whole table is a legal slice.
        let (whole, _) = c.slice(0, 8);
        assert_eq!(whole.shape(), [8, 2]);
    }

    /// Past the end is a panic naming both bounds, not a silently short read.
    #[test]
    #[should_panic(expected = "exceed the 8-row table")]
    fn a_slice_past_the_table_names_both_bounds() {
        let g = graph();
        let c: RopeCache = RopeCache::new(&g, 4, Dim::Const(8), 10_000.0).unwrap();
        let _ = c.slice(7, 2);
    }

    #[test]
    fn an_odd_head_dim_and_a_symbolic_length_are_errors_not_panics() {
        let g = graph();
        assert!(RopeCache::<f32>::new(&g, 5, Dim::Const(4), 10_000.0).is_err());
        assert!(RopeCache::<f32>::new(&g, 0, Dim::Const(4), 10_000.0).is_err());
        let s = g.sym("len");
        assert!(RopeCache::<f32>::new(&g, 4, s, 10_000.0).is_err());
    }

    /// The table's element type is a parameter, and an f16 model gets an f16
    /// table rather than a cast at every step.
    #[test]
    fn a_half_precision_table_is_uploaded_at_its_own_width() {
        let g = graph();
        let c: RopeCache<half::f16> = RopeCache::new(&g, 4, Dim::Const(4), 10_000.0).unwrap();
        assert_eq!(c.sin.dtype(), Dtype::F16);
        assert_eq!(c.cos.shape(), [4, 2]);
    }
}
