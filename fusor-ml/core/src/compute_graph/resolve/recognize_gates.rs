//! Emitter ↔ recognizer binding gates.
//!
//! Every composite the tensor API emits — attention forward and backward,
//! RMS norm, softmax, RoPE, cat/slice-assign chains, the contractions
//! themselves — is a plain cluster of the 3-op vocabulary that some resolver
//! matcher has to claim. A miss is invisible to every other test in the
//! suite: the composed form computes bit-comparable values through the
//! generic elementwise + reduce kernels, several times slower. These gates
//! emit each composite through the public API, run the recognition phase
//! over it, and pin exactly what survives — an emitter and its matcher can
//! only drift apart by turning one of them red.
//!
//! Generalizes the regenerate-and-compare check `match_slice_assign` already
//! performs against `slice_assign_expression`.

use std::ops::Range;

use super::key_goldens::q4k_weight;
use super::recognize_cat::match_slice_assign;
use crate::composite::attention::{MASKED_SCORE_F16, MASKED_SCORE_F32};
use crate::flash_attention::AttentionKernel;
use crate::{Device, Tensor};

use super::*;

/// One surviving execution node: what it lowers to, and the inner nodes it
/// reads in dependency order.
struct Recognized {
    label: &'static str,
    inputs: Vec<NodeIndex>,
}

fn attention_label(kind: AttentionKernel) -> &'static str {
    match kind {
        AttentionKernel::Output => "attention:output",
        AttentionKernel::LogSumExp => "attention:log_sum_exp",
        AttentionKernel::GradQ => "attention:grad_q",
        AttentionKernel::GradK => "attention:grad_k",
        AttentionKernel::GradV => "attention:grad_v",
        AttentionKernel::GradKV => "attention:grad_kv",
    }
}

fn label(variant: &ExecutionVariant) -> &'static str {
    match variant {
        ExecutionVariant::Tensor(_) => "tensor",
        ExecutionVariant::QMatrix(_) => "qmatrix",
        ExecutionVariant::Elementwise(_) => "elementwise",
        ExecutionVariant::Reduce(_) => "reduce",
        ExecutionVariant::Fold(_) => "fold",
        ExecutionVariant::View(_) => "view",
        ExecutionVariant::Assign(_) => "assign",
        ExecutionVariant::Region(_) => "region",
        ExecutionVariant::MatMul(_) => "matmul",
        ExecutionVariant::QMatMul(_) => "qmatmul",
        ExecutionVariant::QEmbedding(_) => "qembedding",
        ExecutionVariant::RowProgram(_) => "row_program",
        ExecutionVariant::Attention(operation) => attention_label(operation.kind),
    }
}

/// Build the execution graph for `targets` and run the recognition phase
/// over it — the same passes, in the same order, that `optimize_operations`
/// runs before it hands the graph to extraction.
fn recognize(device: &Device, targets: &[&Tensor]) -> Vec<Recognized> {
    let targets: Vec<NodeIndex> = targets.iter().map(|tensor| tensor.data().key).collect();
    device.compute_graph().with_mut(|graph| {
        let mut resolver = Resolver::new_batch(graph, targets.clone());
        for &target in &targets {
            resolver.build_execution_graph(graph, target);
        }
        resolver.recognize_all(graph);
        let mut recognized: Vec<Recognized> = resolver
            .execution_graph
            .node_weights()
            .map(|node| {
                let mut inputs = Vec::new();
                node.variant
                    .visit_dependencies(&mut |input| inputs.push(input));
                Recognized {
                    label: label(&node.variant),
                    inputs,
                }
            })
            .collect();
        recognized.sort_by_key(|node| node.label);
        recognized
    })
}

fn labels(recognized: &[Recognized]) -> Vec<&'static str> {
    recognized.iter().map(|node| node.label).collect()
}

/// The dependency list of the one node carrying `label`.
fn inputs_of<'a>(recognized: &'a [Recognized], label: &str) -> &'a [NodeIndex] {
    let mut matches = recognized.iter().filter(|node| node.label == label);
    let node = matches
        .next()
        .unwrap_or_else(|| panic!("no {label} node in {:?}", labels(recognized)));
    assert!(
        matches.next().is_none(),
        "more than one {label} node in {:?}",
        labels(recognized)
    );
    &node.inputs
}

fn splat(device: &Device, value: f32, shape: &[usize]) -> Tensor {
    Tensor::splat::<f32>(device, value, shape)
}

fn splat_f16(device: &Device, value: f32, shape: &[usize]) -> Tensor {
    Tensor::splat::<half::f16>(device, half::f16::from_f32(value), shape)
}

#[test]
fn contraction_composites_recognize() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        let a = splat(&device, 0.5, &[8, 16]);
        let b = splat(&device, 0.25, &[16, 8]);
        let recognized = recognize(&device, &[&a.mat_mul(&b)]);
        assert_eq!(labels(&recognized), ["matmul", "tensor", "tensor"]);
        assert_eq!(inputs_of(&recognized, "matmul"), [a.key(), b.key()]);

        let a = splat(&device, 0.5, &[2, 8, 16]);
        let b = splat(&device, 0.25, &[2, 16, 8]);
        let recognized = recognize(&device, &[&a.mat_mul(&b)]);
        assert_eq!(labels(&recognized), ["matmul", "tensor", "tensor"]);

        let x = splat(&device, 0.5, &[1, 512]);
        let weight = q4k_weight(&device, 512, 512);
        let recognized = recognize(&device, &[&x.q_mat_mul(&weight)]);
        assert_eq!(labels(&recognized), ["qmatmul", "tensor"]);
        assert_eq!(inputs_of(&recognized, "qmatmul"), [x.key()]);
    });
}

#[test]
fn quantized_row_gather_composite_recognizes() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        let indexes = Tensor::new::<u32, 1, _>(&device, &[0u32, 3, 5]);
        let table = q4k_weight(&device, 512, 256);
        let recognized = recognize(&device, &[&table.index_select_rows(&indexes)]);
        assert_eq!(labels(&recognized), ["qembedding", "tensor"]);
        assert_eq!(inputs_of(&recognized, "qembedding"), [indexes.key()]);
    });
}

#[test]
fn softmax_composite_recognizes() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        let x = splat(&device, 0.5, &[8, 64]);
        let recognized = recognize(&device, &[&x.softmax(1)]);
        assert_eq!(labels(&recognized), ["row_program", "tensor"]);
        assert_eq!(inputs_of(&recognized, "row_program"), [x.key()]);

        let x = splat(&device, 0.5, &[2, 4, 64]);
        let recognized = recognize(&device, &[&x.softmax_last_dim()]);
        assert_eq!(labels(&recognized), ["row_program", "tensor"]);
    });
}

#[test]
fn rms_norm_composite_recognizes() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        let x = splat(&device, 0.5, &[8, 64]);
        let weight = splat(&device, 1.5, &[64]);
        let recognized = recognize(&device, &[&x.rms_norm_fused_no_bias(&weight, 1e-5)]);
        assert_eq!(
            labels(&recognized),
            ["row_program", "tensor", "tensor", "view"]
        );

        let bias = splat(&device, 0.125, &[64]);
        let recognized = recognize(&device, &[&x.rms_norm_fused(&weight, Some(&bias), 1e-5)]);
        assert_eq!(
            labels(&recognized),
            ["row_program", "tensor", "tensor", "tensor", "view", "view"]
        );
    });
}

#[test]
fn attention_forward_composite_recognizes() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        let q = splat(&device, 0.5, &[2, 8, 32, 64]);
        let k = splat(&device, 0.25, &[2, 8, 32, 64]);
        let v = splat(&device, 0.125, &[2, 8, 32, 64]);
        let recognized = recognize(&device, &[&q.attention(&k, &v, 0.125, None)]);
        assert_eq!(
            labels(&recognized),
            ["attention:output", "tensor", "tensor", "tensor"]
        );
        assert_eq!(
            inputs_of(&recognized, "attention:output"),
            [q.key(), k.key(), v.key()]
        );

        // One query row is decode's shape: no cross-row K/V reuse to buy, so
        // the cluster lands on the attention row program instead.
        let q = splat(&device, 0.5, &[2, 8, 1, 64]);
        let recognized = recognize(&device, &[&q.attention(&k, &v, 0.125, None)]);
        assert_eq!(
            labels(&recognized),
            ["row_program", "tensor", "tensor", "tensor"]
        );
        assert_eq!(
            inputs_of(&recognized, "row_program"),
            [q.key(), k.key(), v.key()]
        );
    });
}

#[test]
fn attention_mask_composites_recognize() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        let q = splat(&device, 0.5, &[2, 8, 32, 64]);
        let k = splat(&device, 0.25, &[2, 8, 32, 64]);
        let v = splat(&device, 0.125, &[2, 8, 32, 64]);
        let recognized = recognize(&device, &[&q.attention_causal(&k, &v, 0.125)]);
        assert_eq!(
            labels(&recognized),
            ["attention:output", "tensor", "tensor", "tensor"]
        );

        let mask = splat(&device, -1.0, &[32, 32]);
        let recognized = recognize(&device, &[&q.attention(&k, &v, 0.125, Some(&mask))]);
        assert_eq!(
            labels(&recognized),
            ["attention:output", "tensor", "tensor", "tensor", "tensor"]
        );
        assert_eq!(
            inputs_of(&recognized, "attention:output"),
            [q.key(), k.key(), v.key(), mask.key()]
        );

        if !device.f16_supported() {
            return;
        }
        let q = splat_f16(&device, 0.5, &[2, 8, 32, 64]);
        let k = splat_f16(&device, 0.25, &[2, 8, 32, 64]);
        let v = splat_f16(&device, 0.125, &[2, 8, 32, 64]);
        let recognized = recognize(&device, &[&q.attention_causal(&k, &v, 0.125)]);
        assert_eq!(
            labels(&recognized),
            ["attention:output", "tensor", "tensor", "tensor"]
        );
    });
}

/// The composed causal select writes a finite stand-in for `-inf` that the
/// flash kernels duplicate as their own private `MASKED_SCORE` literal
/// (`tile-ir-kernels/src/kernels/attention.rs`). The recognizer compares
/// against the emitter's constant, so only the kernel copy can drift — pin
/// the exact bytes on this side.
#[test]
fn masked_score_constants_are_the_kernel_literals() {
    assert_eq!(MASKED_SCORE_F32.to_bits(), (-3.0e38f32).to_bits());
    assert_eq!(MASKED_SCORE_F16.to_bits(), half::f16::MIN.to_bits());
    assert_eq!(MASKED_SCORE_F16.to_f32(), -65504.0);
}

#[test]
fn attention_gqa_expand_peels_back_to_the_kv_tensors() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        let q = splat(&device, 0.5, &[2, 8, 32, 64]);
        let k = splat(&device, 0.25, &[2, 2, 32, 64]);
        let v = splat(&device, 0.125, &[2, 2, 32, 64]);
        let recognized = recognize(&device, &[&q.attention(&k, &v, 0.125, None)]);
        // The stride-0 group broadcast and its flat reinterpret are peeled:
        // no view survives, and the kernel reads the unexpanded K/V nodes.
        assert_eq!(
            labels(&recognized),
            ["attention:output", "tensor", "tensor", "tensor"]
        );
        assert_eq!(
            inputs_of(&recognized, "attention:output"),
            [q.key(), k.key(), v.key()]
        );

        let q = splat(&device, 0.5, &[2, 8, 1, 64]);
        let recognized = recognize(&device, &[&q.attention(&k, &v, 0.125, None)]);
        assert_eq!(
            labels(&recognized),
            ["row_program", "tensor", "tensor", "tensor"]
        );
        assert_eq!(
            inputs_of(&recognized, "row_program"),
            [q.key(), k.key(), v.key()]
        );
    });
}

#[test]
fn attention_lse_composite_recognizes() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        let q = splat(&device, 0.5, &[2, 8, 32, 64]);
        let k = splat(&device, 0.25, &[2, 8, 32, 64]);
        let recognized = recognize(&device, &[&q.attention_lse(&k, 0.125, None, true)]);
        assert_eq!(
            labels(&recognized),
            ["attention:log_sum_exp", "tensor", "tensor"]
        );
        assert_eq!(
            inputs_of(&recognized, "attention:log_sum_exp"),
            [q.key(), k.key()]
        );
    });
}

#[test]
fn attention_backward_composites_recognize() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        let shape = [2usize, 8, 32, 64];
        let q = splat(&device, 0.5, &shape);
        let k = splat(&device, 0.25, &shape);
        let v = splat(&device, 0.125, &shape);
        let o = splat(&device, 0.375, &shape);
        let grad_o = splat(&device, 0.0625, &shape);
        let lse = splat(&device, 1.0, &[2, 8, 32]);
        let (dq, dk, dv) = q.attention_grads(&k, &v, &o, &grad_o, &lse, 0.125, None, true);
        let recognized = recognize(&device, &[&dq, &dk, &dv]);
        // `dsum = sum(grad_o ∘ o)` is a genuine input to both kernels, so an
        // elementwise + reduce pair survives beside them; the dk/dv halves
        // read back out of the paired kernel's output as views.
        assert_eq!(
            labels(&recognized),
            [
                "attention:grad_kv",
                "attention:grad_q",
                "elementwise",
                "reduce",
                "tensor",
                "tensor",
                "tensor",
                "tensor",
                "tensor",
                "tensor",
                "view",
                "view",
            ]
        );
        let dsum = *inputs_of(&recognized, "attention:grad_q").last().unwrap();
        assert_eq!(
            inputs_of(&recognized, "attention:grad_q"),
            [q.key(), k.key(), v.key(), grad_o.key(), lse.key(), dsum]
        );
        assert_eq!(
            inputs_of(&recognized, "attention:grad_kv"),
            [q.key(), k.key(), v.key(), grad_o.key(), lse.key(), dsum]
        );

        // The same cluster differentiated against the composed log-sum-exp
        // keeps every pattern: the statistic itself streams too.
        let lse = q.attention_lse(&k, 0.125, None, true);
        let (dq, dk, dv) = q.attention_grads(&k, &v, &o, &grad_o, &lse, 0.125, None, true);
        let recognized = recognize(&device, &[&dq, &dk, &dv]);
        assert_eq!(
            labels(&recognized),
            [
                "attention:grad_kv",
                "attention:grad_q",
                "attention:log_sum_exp",
                "elementwise",
                "reduce",
                "tensor",
                "tensor",
                "tensor",
                "tensor",
                "tensor",
                "view",
                "view",
            ]
        );
    });
}

/// The regenerate-and-compare binding in its original form: every shape the
/// emitter can produce must round-trip back to the exact ranges it was given.
#[test]
fn slice_assign_emitter_round_trips_through_its_matcher() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        let cases: &[(&[usize], &[Range<usize>])] = &[
            (&[4, 8], &[0..2, 0..8]),
            (&[4, 8], &[2..4, 0..8]),
            (&[4, 8], &[1..3, 2..5]),
            (&[2, 3, 4, 5], &[0..2, 1..3, 0..4, 2..5]),
            (&[6], &[2..5]),
        ];
        for (shape, slices) in cases {
            let base = splat(&device, 0.5, shape);
            let value_shape: Vec<usize> = slices.iter().map(|slice| slice.len()).collect();
            let value = splat(&device, 1.5, &value_shape);
            let assigned = base.slice_assign(slices.to_vec(), &value);
            let nary = device.compute_graph().with_mut(|graph| {
                match &graph
                    .nodes
                    .nodes
                    .node_weight(assigned.key())
                    .expect("live node")
                    .variant
                {
                    ComputeGraphNodeVariant::Elementwise(nary) => nary.clone(),
                    other => panic!("slice_assign emitted {other:?}"),
                }
            });
            assert_eq!(
                match_slice_assign(&nary).as_deref(),
                Some(*slices),
                "slice_assign {slices:?} over {shape:?}"
            );
        }
    });
}

#[test]
fn slice_assign_chain_recognizes() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        // `Tensor::cat` in its composed form: a zero base, then one
        // slice-assign per chunk. Recognition lifts every branch into one
        // kernel over the destination index space.
        let base = splat(&device, 0.0, &[4, 8]);
        let left = splat(&device, 0.5, &[2, 8]);
        let right = splat(&device, 1.5, &[2, 8]);
        let cat = base
            .slice_assign([0..2, 0..8], &left.exp())
            .slice_assign([2..4, 0..8], &right.exp());
        let recognized = recognize(&device, &[&cat]);
        assert_eq!(
            labels(&recognized),
            ["elementwise", "tensor", "tensor", "tensor"]
        );
        // Slots follow first appearance in the folded select chain, so the
        // last chunk's operand leads and the base is the final fallthrough.
        assert_eq!(
            inputs_of(&recognized, "elementwise"),
            [right.key(), left.key(), base.key()]
        );
    });
}

/// RoPE has no matcher: the emitter writes the fused kernel's expression
/// directly, and a paired call must stay one kernel writing both halves of a
/// single allocation. Splitting it back into per-tensor ops is the same
/// silent slowdown a recognition miss is.
#[test]
fn rope_composites_stay_single_kernels() {
    pollster::block_on(async {
        let Ok(device) = Device::new().await else {
            return;
        };

        let q = splat(&device, 0.5, &[1, 8, 4, 64]);
        let cos = splat(&device, 0.25, &[4, 32]);
        let sin = splat(&device, 0.125, &[4, 32]);
        let recognized = recognize(&device, &[&q.rope_fused(&cos, &sin)]);
        assert_eq!(
            labels(&recognized),
            ["elementwise", "tensor", "tensor", "tensor"]
        );
        assert_eq!(
            inputs_of(&recognized, "elementwise"),
            [q.key(), cos.key(), sin.key()]
        );

        let k = splat(&device, 0.75, &[1, 2, 4, 64]);
        let (rope_q, rope_k) = q.rope_normal_pair_fused(&k, &cos, &sin);
        let recognized = recognize(&device, &[&rope_q, &rope_k]);
        assert_eq!(
            labels(&recognized),
            [
                "elementwise",
                "tensor",
                "tensor",
                "tensor",
                "tensor",
                "view",
                "view"
            ]
        );
        assert_eq!(
            inputs_of(&recognized, "elementwise"),
            [q.key(), k.key(), cos.key(), sin.key()]
        );
    });
}
