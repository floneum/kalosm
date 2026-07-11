//! Payload interning: complete [`ExecutionVariant`]s keyed by a two-lane
//! 128-bit structural hash.
//!
//! The key reuses each operation's `hash_kernel_fields` — the kernel-cache
//! surface, which by definition covers every field that changes generated
//! source — widened with the operation's dependency list (which
//! `hash_kernel_fields` deliberately omits) and the variant tag. Key
//! equality is treated as payload identity, the same trade the flush-replay
//! fingerprint makes ([`FingerprintHasher`]-style two-lane accumulation):
//! a collision could only conflate two *alternatives of the same execution
//! node* (provenance salting already separates nodes), and only if they
//! also agree on every dependency.
//!
//! Interning full `ExecutionVariant` clones means delta application reads
//! complete, builder-produced operations straight from the table — identical
//! shader source, kernel-cache keys, and dispatch names to the destructive
//! optimizer.

use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHasher};

use super::super::ExecutionVariant;
use super::lang::PayloadId;
use crate::compute_graph::NodeIndex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PayloadKey([u64; 2]);

/// Two differently-seeded accumulator lanes over the same 64-bit words; the
/// second lane widens the accumulator state against cancellation collisions
/// (the `flush_replay::FingerprintHasher` recipe).
struct TwoLane {
    a: FxHasher,
    b: FxHasher,
}

impl TwoLane {
    fn new() -> Self {
        let mut a = FxHasher::default();
        0u64.hash(&mut a);
        let mut b = FxHasher::default();
        1u64.hash(&mut b);
        Self { a, b }
    }

    fn write_u64(&mut self, value: u64) {
        value.hash(&mut self.a);
        (value.rotate_left(32) ^ 0x9E37_79B9_7F4A_7C15).hash(&mut self.b);
    }

    fn finish(self) -> PayloadKey {
        PayloadKey([self.a.finish(), self.b.finish()])
    }
}

fn local_hash(f: impl FnOnce(&mut FxHasher)) -> u64 {
    let mut hasher = FxHasher::default();
    f(&mut hasher);
    hasher.finish()
}

/// The payload's dependencies in `visit_dependencies` order — the order the
/// e-node's children mirror.
pub(super) fn variant_dependencies(variant: &ExecutionVariant) -> Vec<NodeIndex> {
    let mut deps = Vec::new();
    let mut push = |dep: NodeIndex| deps.push(dep);
    match variant {
        ExecutionVariant::Tensor(_) => {}
        ExecutionVariant::QMatrix(op) => {
            use crate::mir::operation::Operation;
            op.visit_dependencies(&mut push);
        }
        ExecutionVariant::Elementwise(op) => {
            use crate::mir::operation::Operation;
            op.visit_dependencies(&mut push);
        }
        ExecutionVariant::Reduce(op) => {
            use crate::mir::operation::Operation;
            op.visit_dependencies(&mut push);
        }
        ExecutionVariant::View(op) => {
            use crate::mir::operation::Operation;
            op.visit_dependencies(&mut push);
        }
        ExecutionVariant::Assign(op) => {
            use crate::mir::operation::Operation;
            op.visit_dependencies(&mut push);
        }
        ExecutionVariant::Region(op) => op.visit_dependencies(&mut push),
        ExecutionVariant::MatMul(op) => {
            use crate::mir::operation::Operation;
            op.visit_dependencies(&mut push);
        }
        ExecutionVariant::QMatMul(op) => {
            use crate::mir::operation::Operation;
            op.visit_dependencies(&mut push);
        }
        ExecutionVariant::QEmbedding(op) => {
            use crate::mir::operation::Operation;
            op.visit_dependencies(&mut push);
        }
        ExecutionVariant::GraphOp(op) => op.visit_dependencies(&mut push),
    }
    deps
}

fn variant_tag(variant: &ExecutionVariant) -> u8 {
    match variant {
        ExecutionVariant::Tensor(_) => 0,
        ExecutionVariant::QMatrix(_) => 1,
        ExecutionVariant::Elementwise(_) => 2,
        ExecutionVariant::Reduce(_) => 3,
        ExecutionVariant::View(_) => 4,
        ExecutionVariant::Assign(_) => 5,
        ExecutionVariant::Region(_) => 6,
        ExecutionVariant::MatMul(_) => 7,
        ExecutionVariant::QMatMul(_) => 8,
        ExecutionVariant::QEmbedding(_) => 9,
        ExecutionVariant::GraphOp(_) => 10,
    }
}

fn hash_variant_fields(variant: &ExecutionVariant, hasher: &mut FxHasher) {
    use crate::mir::operation::Operation;
    match variant {
        // Tensor leaves are never interned (identified by provenance alone);
        // hash nothing beyond the tag if one ever reaches here.
        ExecutionVariant::Tensor(_) => {}
        ExecutionVariant::QMatrix(op) => op.hash_kernel_fields(hasher),
        ExecutionVariant::Elementwise(op) => op.hash_kernel_fields(hasher),
        ExecutionVariant::Reduce(op) => op.hash_kernel_fields(hasher),
        ExecutionVariant::View(op) => op.hash_kernel_fields(hasher),
        ExecutionVariant::Assign(op) => op.hash_kernel_fields(hasher),
        ExecutionVariant::Region(op) => op.hash_kernel_fields(hasher),
        ExecutionVariant::MatMul(op) => op.hash_kernel_fields(hasher),
        ExecutionVariant::QMatMul(op) => op.hash_kernel_fields(hasher),
        ExecutionVariant::QEmbedding(op) => op.hash_kernel_fields(hasher),
        ExecutionVariant::GraphOp(op) => op.hash_kernel_fields(hasher),
    }
}

fn payload_key(variant: &ExecutionVariant) -> PayloadKey {
    let mut lanes = TwoLane::new();
    lanes.write_u64(variant_tag(variant) as u64);
    lanes.write_u64(local_hash(|hasher| hash_variant_fields(variant, hasher)));
    let deps = variant_dependencies(variant);
    lanes.write_u64(deps.len() as u64);
    for dep in deps {
        lanes.write_u64(dep.index() as u64);
    }
    lanes.finish()
}

/// Append-only table of interned payloads. Ids are assigned in first-intern
/// order, which is deterministic because ingestion and rule application are.
#[derive(Default)]
pub(super) struct PayloadTable {
    payloads: Vec<ExecutionVariant>,
    by_key: FxHashMap<PayloadKey, PayloadId>,
}

impl PayloadTable {
    pub(super) fn intern(&mut self, variant: ExecutionVariant) -> PayloadId {
        let key = payload_key(&variant);
        *self.by_key.entry(key).or_insert_with(|| {
            let id = PayloadId(self.payloads.len() as u32);
            self.payloads.push(variant);
            id
        })
    }

    /// Append without dedup hashing. For Stage-2 generator mints: each fused
    /// form is produced at most once per switch (generators only fire when
    /// they change the current selection), so the interner's idempotence
    /// dedup — which re-hashes the whole incrementally-grown expression on
    /// every switch, quadratic over fusion chains — buys nothing there.
    /// Stage-1 saturation rules must keep using [`Self::intern`]: their
    /// convergence detection relies on the dedup.
    pub(super) fn push_unique(&mut self, variant: ExecutionVariant) -> PayloadId {
        let id = PayloadId(self.payloads.len() as u32);
        self.payloads.push(variant);
        id
    }

    pub(super) fn get(&self, id: PayloadId) -> &ExecutionVariant {
        &self.payloads[id.0 as usize]
    }
}
