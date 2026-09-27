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
    /// Result count; `outs > 1` is only read through `Logical::Project`.
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

    /// A step-lived single value whose contract is the meet over `ins`.
    pub fn step(dtype: Dtype, shape: Dims, ins: &[ValueFacts]) -> Self {
        Self {
            dtype,
            shape,
            numeric: Self::meet(ins),
            persistence: Persistence::Step,
            outs: 1,
        }
    }

    /// The meet of every operand's contract; `RELAXED` over none.
    pub fn meet(ins: &[ValueFacts]) -> NumericContract {
        ins.iter()
            .map(|f| f.numeric)
            .reduce(NumericContract::meet)
            .unwrap_or(NumericContract::RELAXED)
    }

    /// This value re-viewed at `shape`: same dtype, contract and lifetime.
    pub fn view(&self, shape: Dims) -> Self {
        Self {
            dtype: self.dtype,
            shape,
            numeric: self.numeric,
            persistence: self.persistence,
            outs: 1,
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn elements(&self) -> Option<u64> {
        crate::shape::const_elements(&self.shape)
    }

    pub fn bytes(&self) -> Option<u64> {
        Some(self.elements()? * self.dtype.byte_size())
    }
}

/// The work one op performs, in units the cost model can price;
/// `verify_l0` rejects a `work` that is constant in shape.
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
