//! What the compiler knows about a value, and what an op costs.

use crate::dtype::{Dtype, NumericContract, Persistence};
use crate::shape::{Dim, Dims};

/// Everything inference derives about one value. Rank is runtime data.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ValueFacts {
    pub dtype: Dtype,
    pub shape: Dims,
    pub numeric: NumericContract,
    pub persistence: Persistence,
    /// Result count for tuple-producing ops. `1` for ordinary values; a
    /// value with `outs > 1` is only ever read through `Logical::Project`.
    pub outs: u8,
}

impl ValueFacts {
    pub fn new(dtype: Dtype, shape: impl IntoIterator<Item = Dim>) -> Self {
        Self {
            dtype,
            shape: shape.into_iter().collect(),
            numeric: NumericContract::RELAXED,
            persistence: Persistence::Step,
            outs: 1,
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn elements(&self) -> Option<u64> {
        self.shape
            .iter()
            .try_fold(1u64, |acc, d| acc.checked_mul(d.as_const()?))
    }

    pub fn bytes(&self) -> Option<u64> {
        Some(self.elements()? * self.dtype.byte_size())
    }

}

/// The work one op performs, in units the cost model can price.
/// **`verify_l0` rejects a registration whose `work` is a constant**: the
/// reference's `Attention { work: 1 }` placeholder cannot recur, and
/// `index_ops` is exactly the term view-fold-vs-gather needs.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Work {
    pub macs: u64,
    pub transcendentals: u64,
    pub index_ops: u64,
    pub wg_bytes: u64,
}

impl Work {
    pub const fn add(self, other: Self) -> Self {
        Self {
            macs: self.macs + other.macs,
            transcendentals: self.transcendentals + other.transcendentals,
            index_ops: self.index_ops + other.index_ops,
            wg_bytes: self.wg_bytes + other.wg_bytes,
        }
    }

    /// Scale every term (a node inlined into `n` consumers).
    pub const fn scale(self, n: u64) -> Self {
        Self {
            macs: self.macs * n,
            transcendentals: self.transcendentals * n,
            index_ops: self.index_ops * n,
            wg_bytes: self.wg_bytes * n,
        }
    }

    pub const fn is_zero(self) -> bool {
        self.macs == 0 && self.transcendentals == 0 && self.index_ops == 0 && self.wg_bytes == 0
    }
}
