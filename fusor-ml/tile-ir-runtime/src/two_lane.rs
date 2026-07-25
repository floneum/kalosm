//! The canonical structural-identity hash.
//!
//! Every "same structure?" question in the stack — kernel cache keys, kernel
//! plan keys, flush-replay fingerprints, fusion-plan window keys, semantic
//! payload identity — answers with a 128-bit key produced here, in one of
//! two flavors:
//!
//! - [`TwoLaneHasher`]: an accumulating hasher for streaming writes. Lane
//!   `b` is fed a deterministic mix of the same 64-bit words as lane `a`,
//!   so per-write entropy stays 64 bits (one FxHash); the second lane widens
//!   the *accumulator* state to make cross-item cancellation collisions
//!   harder, not the per-item hash.
//! - [`two_lane_salted`]: a one-shot flavor that re-runs the input closure
//!   once per differently-seeded lane, giving the full 128 bits of per-key
//!   entropy when the inputs are cheap to re-hash.
//!
//! Collision contract: consumers trust these keys without byte-exact
//! verification. Each key domain documents at its newtype what a collision
//! would cost and what secondary validation (replay validation, verify
//! flags, recorder poisoning) bounds the damage.

use std::hash::{Hash, Hasher};

use rustc_hash::FxHasher;

/// Two differently-seeded accumulator lanes over the same 64-bit words.
pub struct TwoLaneHasher {
    a: FxHasher,
    b: FxHasher,
}

impl Default for TwoLaneHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl TwoLaneHasher {
    pub fn new() -> Self {
        let mut a = FxHasher::default();
        0u64.hash(&mut a);
        let mut b = FxHasher::default();
        1u64.hash(&mut b);
        Self { a, b }
    }

    pub fn write_u64(&mut self, value: u64) {
        value.hash(&mut self.a);
        (value.rotate_left(32) ^ 0x9E37_79B9_7F4A_7C15).hash(&mut self.b);
    }

    pub fn finish(self) -> [u64; 2] {
        [self.a.finish(), self.b.finish()]
    }
}

/// One-shot two-lane key: the closure's writes are hashed once per lane
/// under distinct seeds, so both lanes carry independent 64-bit digests of
/// the full input stream.
pub fn two_lane_salted(hash_inputs: impl Fn(&mut FxHasher)) -> [u64; 2] {
    std::array::from_fn(|salt| {
        let mut hasher = FxHasher::default();
        (salt as u64).hash(&mut hasher);
        hash_inputs(&mut hasher);
        hasher.finish()
    })
}

/// Single-lane convenience for hashing a sub-structure into one word before
/// feeding it to a [`TwoLaneHasher`].
pub fn single_lane(f: impl FnOnce(&mut FxHasher)) -> u64 {
    let mut hasher = FxHasher::default();
    f(&mut hasher);
    hasher.finish()
}
