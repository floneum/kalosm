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

/// Allocation-independent identity of an operation shape used by the
/// planner.  A payload remains the concrete, executable operation; a spec is
/// the reusable part of that operation (kernel fields, tensor shapes and
/// expression structure) with dependency and buffer identities erased.
///
/// Keeping this separate from [`PayloadId`] is essential: two transformer
/// layers may share a spec while still referring to different weights and
/// activations, and therefore must never become value-equal e-nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct SpecId(pub(super) u32);

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
            for epilogue in [
                &mut op.pre_element_wise_expr,
                &mut op.post_element_wise_expr,
            ]
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
    specs: Vec<ExecutionVariant>,
    specs_by_key: FxHashMap<PayloadKey, Vec<SpecId>>,
    spec_of_payload: Vec<SpecId>,
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
        let spec = self.intern_spec(key, &variant);
        self.payloads.push(variant);
        self.by_key.entry(key).or_default().push(id);
        self.spec_of_payload.push(spec);
        id
    }

    /// Append without semantic payload dedup. Stage-2 generates each concrete
    /// occurrence once, so idempotence lookup buys nothing; the first planning
    /// occurrence still establishes its allocation-independent spec. Repeated
    /// occurrences use [`Self::push_unique_with_spec`] and skip that structural
    /// hash too. Stage-1 saturation must keep using [`Self::intern`], because
    /// its convergence detection relies on semantic dedup.
    pub(super) fn push_unique(&mut self, variant: ExecutionVariant) -> PayloadId {
        let key = payload_key(&variant);
        let spec = self.intern_spec(key, &variant);
        self.push_unique_with_spec(variant, spec)
    }

    /// Append a concrete occurrence whose allocation-independent spec was
    /// already established by a shared planning template. This is the hot
    /// repeated-layer path: it avoids re-hashing a potentially large fused
    /// expression for every layer.
    pub(super) fn push_unique_with_spec(
        &mut self,
        variant: ExecutionVariant,
        spec: SpecId,
    ) -> PayloadId {
        debug_assert!((spec.0 as usize) < self.specs.len());
        debug_assert!(planning_payload_eq(&self.specs[spec.0 as usize], &variant));
        let id = PayloadId(self.payloads.len() as u32);
        self.payloads.push(variant);
        self.spec_of_payload.push(spec);
        id
    }

    pub(super) fn get(&self, id: PayloadId) -> &ExecutionVariant {
        &self.payloads[id.0 as usize]
    }

    pub(super) fn spec_of(&self, id: PayloadId) -> SpecId {
        self.spec_of_payload[id.0 as usize]
    }

    pub(super) fn spec_count(&self) -> usize {
        self.specs.len()
    }

    pub(super) fn payload_count(&self) -> usize {
        self.payloads.len()
    }

    fn intern_spec(&mut self, key: PayloadKey, variant: &ExecutionVariant) -> SpecId {
        if let Some(id) = self.specs_by_key.get(&key).and_then(|ids| {
            ids.iter()
                .copied()
                .find(|id| planning_payload_eq(&self.specs[id.0 as usize], variant))
        }) {
            return id;
        }
        let id = SpecId(self.specs.len() as u32);
        self.specs.push(variant.clone());
        self.specs_by_key.entry(key).or_default().push(id);
        id
    }
}

fn same_qmatrix_spec(a: &crate::quantized::QMatrix, b: &crate::quantized::QMatrix) -> bool {
    a.datatype() == b.datatype()
        && a.storage_layout() == b.storage_layout()
        && a.shape() == b.shape()
}

fn same_epilogue_spec(
    a: &Option<crate::quantized::matmul::ElementwiseEpilogue>,
    b: &Option<crate::quantized::matmul::ElementwiseEpilogue>,
) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.expression == b.expression
                && a.extras.len() == b.extras.len()
                && a.input_datatype == b.input_datatype
                && a.output_datatype == b.output_datatype
        }
        _ => false,
    }
}

/// Exact equality for the allocation-independent planning surface.
/// `payload_key` is only a bucket selector, so this comparison deliberately
/// checks every relevant field and makes hash collisions harmless.
fn planning_payload_eq(a: &ExecutionVariant, b: &ExecutionVariant) -> bool {
    let zero = NodeIndex::new(0);
    match (a, b) {
        (ExecutionVariant::Tensor(a), ExecutionVariant::Tensor(b)) => {
            a.datatype() == b.datatype() && a.layout() == b.layout()
        }
        (ExecutionVariant::QMatrix(a), ExecutionVariant::QMatrix(b)) => {
            same_qmatrix_spec(&a.matrix, &b.matrix)
                && a.datatype == b.datatype
                && a.post_dequantize == b.post_dequantize
        }
        (ExecutionVariant::Elementwise(a), ExecutionVariant::Elementwise(b)) => {
            a.expression == b.expression
                && a.shape == b.shape
                && a.output_datatype == b.output_datatype
                && a.inputs.len() == b.inputs.len()
        }
        (ExecutionVariant::Reduce(a), ExecutionVariant::Reduce(b)) => {
            a.expression == b.expression
                && a.shape == b.shape
                && a.function == b.function
                && a.post_element_wise == b.post_element_wise
                && a.axis == b.axis
                && a.inputs.len() == b.inputs.len()
        }
        (ExecutionVariant::View(a), ExecutionVariant::View(b)) => {
            a.stages == b.stages && a.datatype == b.datatype
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
            a.input_datatype == b.input_datatype
                && same_qmatrix_spec(&a.matrix, &b.matrix)
                && a.in_shape == b.in_shape
                && a.out_shape == b.out_shape
                && same_epilogue_spec(&a.pre_element_wise_expr, &b.pre_element_wise_expr)
                && same_epilogue_spec(&a.post_element_wise_expr, &b.post_element_wise_expr)
                && a.post_accumulator_offsets == b.post_accumulator_offsets
        }
        (ExecutionVariant::QEmbedding(a), ExecutionVariant::QEmbedding(b)) => {
            same_qmatrix_spec(&a.matrix, &b.matrix)
                && a.out_shape == b.out_shape
                && a.datatype == b.datatype
        }
        (ExecutionVariant::GraphOp(a), ExecutionVariant::GraphOp(b)) => {
            // GraphOperation has no structural equality contract. Keep its
            // allocation identity until that contract exists.
            std::sync::Arc::ptr_eq(a, b)
        }
        // Effects and multi-output regions are observation-specific.
        (ExecutionVariant::Assign(_), ExecutionVariant::Assign(_))
        | (ExecutionVariant::Region(_), ExecutionVariant::Region(_)) => false,
        _ => false,
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
