//! Contractions. `matmul`, `mat_mul_transposed_rhs` and every batched form are
//! one `Logical::Contract` with a different `EinSpec` — transposed-rhs is a
//! spec, not an op. Kernel family, tile geometry, split-K and staging depth
//! are all extraction decisions that do not exist yet at this level.

use fusor2_ir::dtype::Dtype;
use fusor2_ir::ir::logical::{EinSpec, Logical, Label};
use smallvec::SmallVec;

use crate::tensor::Tensor;
use crate::{Error, Result};

/// Build the spec for a batched matmul over `batch` leading axes.
///
/// Labels are allocated in order: `0..batch` are the shared batch axes, then
/// `m`, `k`, `n`. `transposed_rhs` only swaps `b`'s last two labels.
pub fn matmul_spec(batch: usize, transposed_rhs: bool) -> Result<EinSpec> {
    if batch + 3 > u8::MAX as usize {
        return Err(Error::Shape(format!(
            "contraction over {batch} batch axes exceeds the 255-label spec"
        )));
    }
    let b0: SmallVec<[Label; 6]> = (0..batch as u8).map(Label).collect();
    let (m, k, n) = (
        Label(batch as u8),
        Label(batch as u8 + 1),
        Label(batch as u8 + 2),
    );
    let mut a = b0.clone();
    a.push(m);
    a.push(k);
    let mut b = b0.clone();
    if transposed_rhs {
        b.push(n);
        b.push(k);
    } else {
        b.push(k);
        b.push(n);
    }
    let mut out = b0;
    out.push(m);
    out.push(n);
    Ok(EinSpec { a, b, out })
}

impl Tensor {
    /// `[batch.., m, k] @ [batch.., k, n] -> [batch.., m, n]`.
    ///
    /// Batch dims must be pairwise [`fusor2_ir::shape::Dim::known_eq`]; there
    /// is no implicit batch broadcast.
    pub fn matmul(&self, rhs: &Tensor) -> Result<Tensor> {
        self.contract_2d(rhs, false)
    }

    /// `self @ rhs^T`. The **same node** as [`Tensor::matmul`]; only `b`'s two
    /// trailing labels swap.
    pub fn matmul_t(&self, rhs: &Tensor) -> Result<Tensor> {
        self.contract_2d(rhs, true)
    }

    fn contract_2d(&self, rhs: &Tensor, transposed_rhs: bool) -> Result<Tensor> {
        // A block-quantized weight is a legal contraction operand on exactly
        // one side: an ordinary `Contract` decodes the blocks on the way into
        // its staging fill, and the extractor prices that against
        // dequantize-then-contract.
        let (q_lhs, q_rhs) = (self.dtype().is_quantized(), rhs.dtype().is_quantized());
        if q_lhs && q_rhs {
            return Err(Error::Dtype(
                "matmul against two quantized operands is not defined; dequantize one side"
                    .into(),
            ));
        }
        if self.rank() < 2 || rhs.rank() < 2 {
            return Err(Error::Shape(format!(
                "matmul needs rank >= 2 on both sides, got {} and {}",
                self.rank(),
                rhs.rank()
            )));
        }
        if self.rank() != rhs.rank() {
            return Err(Error::Shape(format!(
                "matmul operands must share their batch rank: {} vs {}; broadcast_as first",
                self.rank(),
                rhs.rank()
            )));
        }
        let batch = self.rank() - 2;
        for i in 0..batch {
            if !self.dim(i).known_eq(rhs.dim(i)) {
                return Err(Error::Shape(format!(
                    "matmul batch axis {i} disagrees: {} vs {}; there is no implicit batch \
                     broadcast, broadcast_as first",
                    self.dim(i),
                    rhs.dim(i)
                )));
            }
        }
        if !q_lhs && !q_rhs && self.dtype() != rhs.dtype() {
            return Err(Error::Dtype(format!(
                "matmul operands differ in dtype: {:?} vs {:?}",
                self.dtype(),
                rhs.dtype()
            )));
        }
        // The accumulator is the *dense* side's compute dtype: a quantized
        // format has none of its own.
        let acc = if q_lhs { rhs.dtype() } else { self.dtype() }.compute_dtype();
        let spec = matmul_spec(batch, transposed_rhs)?;

        // A quantized side enters the contraction as its *dequantize class*,
        // not its raw leaf. The class is `Logical::Dequant` unioned with the
        // `Restride` + `Map` definitional expansion (see `QMatrix::dequantize`).
        // A quantized value that `QMatrix::of_tensor` cannot name falls back
        // to the raw operand. Both spellings enter the class so the extractor
        // can price the choice per shape.
        let deq = |t: &Tensor| -> Result<Option<Tensor>> {
            match crate::quantized::QMatrix::of_tensor(t) {
                Some(q) => Ok(Some(q.dequantize()?)),
                None => Ok(None),
            }
        };
        // For repacked quantized formats, also union in a contraction over
        // the word-aligned scales twin.
        let requant = |t: &Tensor| -> Result<Option<Tensor>> {
            Ok(self
                .graph
                .repacked_leaf_of(t.id)?
                .map(|id| self.graph.tensor(id)))
        };
        if q_lhs && let Some(w) = deq(self)? {
            let staged = self.contract(rhs, spec.clone(), acc)?;
            let dense = w.contract(rhs, spec.clone(), acc)?;
            // Stable first-union root; see `composite::macro_op`.
            let root = self.graph.union_stable(staged.id, dense.id)?;
            if let Some(twin) = requant(self)? {
                let restaged = twin.contract(rhs, spec, acc)?;
                self.graph.union_stable(root, restaged.id)?;
            }
            return Ok(self.graph.tensor(root));
        }
        if q_rhs && let Some(w) = deq(rhs)? {
            let staged = self.contract(rhs, spec.clone(), acc)?;
            let dense = self.contract(&w, spec.clone(), acc)?;
            let root = self.graph.union_stable(staged.id, dense.id)?;
            if let Some(twin) = requant(rhs)? {
                let restaged = self.contract(&twin, spec, acc)?;
                self.graph.union_stable(root, restaged.id)?;
            }
            return Ok(self.graph.tensor(root));
        }
        self.contract(rhs, spec, acc)
    }

    /// The general contraction escape hatch with an explicit accumulator.
    pub fn contract(&self, rhs: &Tensor, spec: EinSpec, acc: Dtype) -> Result<Tensor> {
        self.emit_here(Logical::Contract {
            spec,
            acc,
            a: self.id,
            b: rhs.id,
            outs: 1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fusor2_ir::contract_spec::{LabelRole, partition, role};

    #[test]
    fn batched_matmul_spec_labels() {
        let s = matmul_spec(1, false).unwrap();
        // a = [b, m, k], b = [b, k, n], out = [b, m, n]
        assert_eq!(&s.a[..], &[Label(0), Label(1), Label(2)]);
        assert_eq!(&s.b[..], &[Label(0), Label(2), Label(3)]);
        assert_eq!(&s.out[..], &[Label(0), Label(1), Label(3)]);

        assert_eq!(role(&s, Label(0)).unwrap(), LabelRole::Batch);
        assert_eq!(role(&s, Label(1)).unwrap(), LabelRole::M);
        assert_eq!(role(&s, Label(2)).unwrap(), LabelRole::K);
        assert_eq!(role(&s, Label(3)).unwrap(), LabelRole::N);
    }

    #[test]
    fn transposed_rhs_is_only_a_spec() {
        let normal = matmul_spec(0, false).unwrap();
        let t = matmul_spec(0, true).unwrap();
        assert_eq!(normal.a, t.a);
        assert_eq!(normal.out, t.out);
        assert_eq!(&normal.b[..], &[Label(1), Label(2)]);
        assert_eq!(&t.b[..], &[Label(2), Label(1)]);
        // Both are well-formed contractions.
        partition(&normal).unwrap();
        partition(&t).unwrap();
    }

    #[test]
    fn adjoint_specs_round_trip_to_the_operand_shapes() {
        let s = matmul_spec(1, false).unwrap();
        let dl = s.d_lhs();
        // grad x b -> a
        assert_eq!(dl.a, s.out);
        assert_eq!(dl.b, s.b);
        assert_eq!(dl.out, s.a);
        let dr = s.d_rhs();
        assert_eq!(dr.a, s.a);
        assert_eq!(dr.b, s.out);
        assert_eq!(dr.out, s.b);
        partition(&dl).unwrap();
        partition(&dr).unwrap();
    }
}
