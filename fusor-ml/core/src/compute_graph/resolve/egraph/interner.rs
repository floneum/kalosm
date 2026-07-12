//! Payload interning: complete [`ExecutionVariant`]s keyed by a two-lane
//! 128-bit structural hash.
//!
//! The key reuses each operation's `hash_kernel_fields` — the kernel-cache
//! surface, which by definition covers every field that changes generated
//! source, plus the variant tag. Dependencies are e-node children, not
//! payload identity; keeping physical NodeIndexes here defeats congruence.
//! Hashes select a bucket only; [`semantic_payload_eq`] performs exact
//! equality, so collisions cannot conflate operators.
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

/// Rewrite `variant`'s dependency slots — in `visit_dependencies` order — to
/// `new`. Interned payloads deduplicate with their inputs ignored
/// (`semantic_payload_eq`), so a payload fetched through the table may carry
/// the concrete input indices of a *different* structurally-identical
/// instance (another layer's matmul, say). Every materialization of a
/// payload back into the execution graph must rebind its inputs to the
/// e-node's actual children or it computes with the wrong operands.
pub(super) fn rebind_variant_dependencies(variant: &mut ExecutionVariant, new: &[NodeIndex]) {
    let mut slots = new.iter().copied();
    match variant {
        ExecutionVariant::Tensor(_) | ExecutionVariant::QMatrix(_) => {}
        ExecutionVariant::Elementwise(op) => {
            for input in &mut op.inputs {
                *input = slots.next().expect("elementwise rebind arity");
            }
        }
        ExecutionVariant::Reduce(op) => {
            for input in &mut op.inputs {
                *input = slots.next().expect("reduce rebind arity");
            }
        }
        ExecutionVariant::View(op) => {
            op.input = slots.next().expect("view rebind arity");
        }
        ExecutionVariant::Assign(op) => {
            // Order matches `SliceAssignOperation::visit_dependencies`:
            // value first, then input.
            op.value = slots.next().expect("assign rebind arity");
            op.input = slots.next().expect("assign rebind arity");
        }
        ExecutionVariant::Region(op) => {
            for input in &mut op.inputs {
                *input = slots.next().expect("region rebind arity");
            }
        }
        ExecutionVariant::MatMul(op) => {
            op.first = slots.next().expect("matmul rebind arity");
            op.second = slots.next().expect("matmul rebind arity");
        }
        ExecutionVariant::QMatMul(op) => {
            // Order matches `QMatMulOperation::visit_dependencies`: input,
            // then pre-epilogue extras, then post-epilogue extras.
            op.input = slots.next().expect("qmatmul rebind arity");
            for epilogue in [&mut op.pre_element_wise_expr, &mut op.post_element_wise_expr]
                .into_iter()
                .flatten()
            {
                for extra in &mut epilogue.extras {
                    *extra = slots.next().expect("qmatmul extras rebind arity");
                }
            }
        }
        ExecutionVariant::QEmbedding(op) => {
            op.indexes = slots.next().expect("qembedding rebind arity");
        }
        // GraphOp payloads are identified by `Arc::ptr_eq`, so an interned
        // GraphOp payload is always this exact object and its stored
        // dependencies are already its own — nothing to rebind.
        ExecutionVariant::GraphOp(op) => {
            debug_assert_eq!(
                variant_dependencies(&ExecutionVariant::GraphOp(op.clone())),
                new,
                "GraphOp payloads never conflate, so children must already match"
            );
            let _ = &mut slots;
            return;
        }
    }
    debug_assert!(
        slots.next().is_none(),
        "rebind received more children than the variant has dependency slots"
    );
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
    lanes.finish()
}

/// Append-only table of interned payloads. Ids are assigned in first-intern
/// order, which is deterministic because ingestion and rule application are.
#[derive(Default)]
pub(super) struct PayloadTable {
    payloads: Vec<ExecutionVariant>,
    by_key: FxHashMap<PayloadKey, Vec<PayloadId>>,
}

impl PayloadTable {
    pub(super) fn intern(&mut self, variant: ExecutionVariant) -> PayloadId {
        let key = payload_key(&variant);
        if let Some(id) = self.by_key.get(&key).and_then(|ids| {
            ids.iter()
                .copied()
                .find(|id| semantic_payload_eq(&self.payloads[id.0 as usize], &variant))
        }) {
            return id;
        }
        let id = PayloadId(self.payloads.len() as u32);
        self.payloads.push(variant);
        self.by_key.entry(key).or_default().push(id);
        id
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

fn semantic_payload_eq(a: &ExecutionVariant, b: &ExecutionVariant) -> bool {
    let zero = NodeIndex::new(0);
    match (a, b) {
        (ExecutionVariant::Tensor(a), ExecutionVariant::Tensor(b)) => {
            std::sync::Arc::ptr_eq(a.buffer(), b.buffer())
        }
        (ExecutionVariant::QMatrix(a), ExecutionVariant::QMatrix(b)) => a == b,
        (ExecutionVariant::Elementwise(a), ExecutionVariant::Elementwise(b)) => {
            let mut a = a.clone();
            let mut b = b.clone();
            a.inputs.clear();
            b.inputs.clear();
            a == b
        }
        (ExecutionVariant::Reduce(a), ExecutionVariant::Reduce(b)) => {
            let mut a = a.clone();
            let mut b = b.clone();
            a.inputs.clear();
            b.inputs.clear();
            a == b
        }
        (ExecutionVariant::View(a), ExecutionVariant::View(b)) => {
            let mut a = a.clone();
            let mut b = b.clone();
            a.input = zero;
            b.input = zero;
            a == b
        }
        (ExecutionVariant::MatMul(a), ExecutionVariant::MatMul(b)) => {
            let mut a = a.clone();
            let mut b = b.clone();
            a.first = zero;
            a.second = zero;
            b.first = zero;
            b.second = zero;
            a == b
        }
        (ExecutionVariant::QMatMul(a), ExecutionVariant::QMatMul(b)) => {
            let mut a = a.as_ref().clone();
            let mut b = b.as_ref().clone();
            a.input = zero;
            b.input = zero;
            for epilogue in [&mut a.pre_element_wise_expr, &mut a.post_element_wise_expr]
                .into_iter()
                .flatten()
            {
                epilogue.extras.fill(zero);
            }
            for epilogue in [&mut b.pre_element_wise_expr, &mut b.post_element_wise_expr]
                .into_iter()
                .flatten()
            {
                epilogue.extras.fill(zero);
            }
            a == b
        }
        (ExecutionVariant::QEmbedding(a), ExecutionVariant::QEmbedding(b)) => {
            let mut a = a.clone();
            let mut b = b.clone();
            a.indexes = zero;
            b.indexes = zero;
            a == b
        }
        (ExecutionVariant::GraphOp(a), ExecutionVariant::GraphOp(b)) => {
            std::sync::Arc::ptr_eq(a, b)
        }
        // Effects and multi-output regions are observation-specific.
        (ExecutionVariant::Assign(_), ExecutionVariant::Assign(_))
        | (ExecutionVariant::Region(_), ExecutionVariant::Region(_)) => false,
        _ => false,
    }
}
