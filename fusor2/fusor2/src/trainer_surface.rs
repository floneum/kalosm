//! The trainer's API surface, spelled exactly as the trainer spells it.
//!
//! `betlang-train` consumes fusor2 through exactly two `use` lines and never
//! handles a `Result` from a tensor op. The shapes it depends on are restated
//! here — the same turbofish arities, rank-generic helpers, and operator
//! expressions — so a regression is a compile error in this crate.
//!
//! Functions named `_compiles` are never called. They exist to be type-checked.

#![allow(dead_code, clippy::let_and_return)]

use crate::autograd::{BackwardTarget, Graph, Tensor};

// Byte-identical to the trainer's second `use` line, with `fusor` spelled as
// `crate`.
use crate::{Device, Tensor as RawTensor, ToVec, cat};

/// The trainer's architecture constants, at the sizes that matter for typing.
const EMBED: usize = 24;
const HASH_COUNT: usize = 3;
const POOLED: usize = 256;
const CLASSES: usize = 48;

struct Params {
    flat: RawTensor<1, f32>,
}

impl Params {
    fn from_slice(device: &Device, values: &[f32]) -> Self {
        Self {
            flat: RawTensor::from_slice(device, [values.len()], values),
        }
    }

    fn slices(&self) -> Vec<RawTensor<1, f32>> {
        vec![self.flat.narrow(0, 0, 1).into_concrete()]
    }
}

struct Model {
    leaves: Vec<Tensor<1>>,
    quantized: Vec<Tensor<1>>,
    graph: Graph,
    shapes: Vec<&'static [usize]>,
}

impl Model {
    fn new(graph: &Graph, params: &Params, trainable: bool) -> Self {
        let leaves: Vec<Tensor<1>> = params
            .slices()
            .into_iter()
            .map(|slice| {
                if trainable {
                    graph.leaf(slice)
                } else {
                    Tensor::constant_from_raw(graph, slice)
                }
            })
            .collect();
        let quantized = leaves.to_vec();
        Self {
            leaves,
            quantized,
            graph: graph.clone(),
            shapes: Vec::new(),
        }
    }

    /// The rank-generic weight accessor. `R` is inferred from the return
    /// position, and `reshape` has to work with it unbound.
    fn weight<const R: usize>(&self, index: usize) -> Tensor<R> {
        let mut shape = [0usize; R];
        shape.copy_from_slice(self.shapes[index]);
        self.quantized[index].reshape(shape)
    }

    /// A weight in model precision; the cast is differentiable.
    fn half_weight<const R: usize>(&self, index: usize) -> Tensor<R, half::f16> {
        self.weight::<R>(index).cast::<half::f16>()
    }

    fn forward_compiles(&self, rows: usize, seq: usize, device: &Device) -> Tensor<2> {
        let positions = rows * seq;
        let bins = RawTensor::<1, u32>::from_slice(device, [positions * HASH_COUNT], &[0u32]);
        let table: Tensor<2> = self.weight(0);
        let embedded = hashed_embedding_compiles(&self.graph, &table, &bins, positions, device);

        let x = embedded.gelu().reshape([rows, seq, EMBED]).transpose(1, 2);
        let x = conv_stack_f16(
            &x.cast::<half::f16>(),
            [
                self.half_weight::<3>(1),
                self.half_weight::<3>(3),
                self.half_weight::<3>(5),
            ],
            [
                self.half_weight::<1>(2),
                self.half_weight::<1>(4),
                self.half_weight::<1>(6),
            ],
        )
        .cast::<f32>();
        let x = conv_stack_f32(
            &x,
            [self.weight::<3>(1), self.weight::<3>(3), self.weight::<3>(5)],
            [self.weight::<1>(2), self.weight::<1>(4), self.weight::<1>(6)],
        );
        let pooled = Tensor::cat(vec![x.max::<2>(2), x.mean::<2>(2)], 1);
        debug_assert_eq!(pooled.shape(), [rows, POOLED]);
        let hidden = pooled
            .matmul(&self.weight::<2>(7))
            .add_::<1, 2>(&self.weight::<1>(8))
            .gelu();
        hidden
            .matmul(&self.weight::<2>(9))
            .add_::<1, 2>(&self.weight::<1>(10))
    }
}

/// The convolution stack, stamped out per model dtype — the trainer writes it
/// as a macro because each op carries its own element bounds.
macro_rules! conv_stack {
    ($name:ident, $elem:ty) => {
        fn $name(
            x: &Tensor<3, $elem>,
            weights: [Tensor<3, $elem>; 3],
            biases: [Tensor<1, $elem>; 3],
        ) -> Tensor<3, $elem> {
            fn conv(
                x: &Tensor<3, $elem>,
                weight: &Tensor<3, $elem>,
                bias: &Tensor<1, $elem>,
            ) -> Tensor<3, $elem> {
                let kernel = weight.shape()[0];
                x.conv::<3, 1, 4>(&weight.permute([2, 1, 0]), Some(bias), [kernel / 2], [1])
            }
            fn pool(x: &Tensor<3, $elem>, window: usize) -> Tensor<3, $elem> {
                let [rows, channels, seq] = x.shape();
                assert!(seq.is_multiple_of(window));
                x.reshape([rows, channels, seq / window, window]).max::<3>(3)
            }
            let x = conv(x, &weights[0], &biases[0]).gelu();
            let x = pool(&x, 4);
            let x = conv(&x, &weights[1], &biases[1]).gelu();
            let x = pool(&x, 2);
            conv(&x, &weights[2], &biases[2]).gelu()
        }
    };
}

conv_stack!(conv_stack_f32, f32);
conv_stack!(conv_stack_f16, half::f16);

/// Straight-through fake quantization, both quant modes.
fn fake_quant_compiles(graph: &Graph, leaf: &Tensor<1>, ternary: bool) -> Tensor<1> {
    let weight = leaf.raw().to_concrete();
    let quantized = if ternary {
        let magnitude = weight.abs().into_concrete();
        let total_sum = magnitude.sum::<0>(0);
        let total = total_sum.reshape([1]);
        let nonzero_sum = magnitude.gt_scalar(0.0).sum::<0>(0);
        let nonzero = nonzero_sum.reshape([1]);
        let scale = total
            .div_::<1, 1, _>(&nonzero.max_scalar(1.0).into_concrete())
            .max_scalar(1e-6)
            .into_concrete();
        let normalized = weight.div_::<1, 1, _>(&scale).into_concrete();
        let ternary = normalized.gte_scalar(0.7) - normalized.lte_scalar(-0.7);
        ternary.mul_::<1, 1, _>(&scale).into_concrete()
    } else {
        let scale = weight
            .abs()
            .max::<0>(0)
            .reshape([1])
            .max_scalar(1e-6)
            .div_scalar(7.0)
            .into_concrete();
        let normalized = weight.div_::<1, 1, _>(&scale).into_concrete();
        round_small(&normalized).mul_::<1, 1, _>(&scale).into_concrete()
    };
    let target = leaf.slot();
    Tensor::constant_from_raw(graph, quantized).with_backwards([leaf.parent()], move |gradient| {
        Ok(vec![BackwardTarget::to(target, gradient)])
    })
}

fn round_small(normalized: &RawTensor<1, f32>) -> RawTensor<1, f32> {
    let mut rounded =
        (normalized.gte_scalar(0.5) - normalized.lte_scalar(-0.5)).into_concrete();
    for level in 2..=7 {
        let threshold = level as f32 - 0.5;
        rounded = (rounded + normalized.gte_scalar(threshold)
            - normalized.lte_scalar(-threshold))
        .into_concrete();
    }
    rounded
}

/// The hashed embedding, whose backward is a three-level sorted scatter.
fn hashed_embedding_compiles(
    graph: &Graph,
    table: &Tensor<2>,
    bins: &RawTensor<1, u32>,
    positions: usize,
    device: &Device,
) -> Tensor<2> {
    let with_zero_row = cat(
        [
            table.raw().to_concrete(),
            RawTensor::zeros(device, [1, EMBED]),
        ],
        0,
    );
    let summed = with_zero_row
        .index_select(0, bins)
        .reshape([positions, HASH_COUNT, EMBED])
        .sum::<2>(1)
        .into_concrete();

    let node = Tensor::constant_from_raw(graph, summed);
    let level1 = RawTensor::<1, u32>::from_slice(device, [1], &[0]);
    let target = table.slot();
    let device = device.clone();
    node.with_backwards([table.parent()], move |gradient: RawTensor<2, f32>| {
        let reduce = |input: RawTensor<2, f32>, order: &RawTensor<1, u32>, rows: usize| {
            let padded = cat([input, RawTensor::zeros(&device, [1, EMBED])], 0);
            padded
                .index_select(0, order)
                .reshape([rows, 8, EMBED])
                .sum::<2>(1)
                .into_concrete()
        };
        let tiles = reduce(gradient.to_concrete(), &level1, 1);
        Ok(vec![BackwardTarget::to(target, tiles)])
    })
}

/// The folded distillation loss, written as one node with an analytic
/// backward.
fn distillation_loss_compiles(
    logits: &Tensor<2>,
    targets: &RawTensor<2, f32>,
    softplus_weight: f32,
    rows: usize,
) -> Tensor<0> {
    let graph = logits.graph();
    let raw = logits.raw().to_concrete();
    let softplus =
        (raw.relu() + (raw.abs() * -1.0).exp().add_scalar(1.0).log()).into_concrete();
    let value = ((softplus * softplus_weight - (&raw * targets).into_concrete())
        .flatten_all()
        .sum::<0>(0)
        * (1.0 / rows as f32))
        .into_concrete();

    let logits_value = raw;
    let targets = targets.clone();
    let input = logits.slot();
    Tensor::constant_from_raw(&graph, value).with_backwards(
        [logits.parent()],
        move |gradient: RawTensor<0, f32>| {
            let scale = (gradient.reshape([1, 1]) * (1.0 / rows as f32)).into_concrete();
            let slope =
                (logits_value.sigmoid() * softplus_weight - targets.clone()).into_concrete();
            Ok(vec![BackwardTarget::to(
                input,
                slope.mul_::<2, 2, _>(&scale).into_concrete(),
            )])
        },
    )
}

/// One optimizer step, including the readbacks and the device drain.
fn train_step_compiles(device: &Device, params: &mut Params, rows: usize) {
    let targets = RawTensor::from_slice(device, [rows, CLASSES], &[0.0f32]);
    {
        let graph = Graph::new();
        let model = Model::new(&graph, params, true);
        let logits = model.forward_compiles(rows, 8, device);
        let loss = distillation_loss_compiles(&logits, &targets, 1.5, rows);
        let _reported = loss.raw().clone();
        let seed = RawTensor::splat(device, 1024.0f32, []);
        let gradients = loss.backward_with(seed).expect("backward failed");
        let flat = cat(
            model
                .leaves
                .iter()
                .map(|leaf| gradients.get(leaf).expect("missing gradient")),
            0,
        );
        adamw_step_compiles(device, params, flat.into_concrete(), 1e-4, 1024.0);
    }
    device.flush();
}

/// The eval readback: `Vec<Vec<f32>>`, indexed as `&[f32]` per row.
fn evaluate_compiles(device: &Device, params: &Params, rows: usize) -> f32 {
    let graph = Graph::new();
    let model = Model::new(&graph, params, false);
    let logits = model.forward_compiles(rows, 8, device).into_raw();
    let values = pollster::block_on(logits.as_slice()).unwrap().to_vec();
    let row_logits: &[f32] = &values[0];
    row_logits[0]
}

fn read_weights_compiles(params: &Params) -> Vec<f32> {
    pollster::block_on(params.flat.clone().as_slice())
        .unwrap()
        .to_vec()
}

fn loss_readback_compiles(loss: &RawTensor<0, f32>) -> f32 {
    pollster::block_on(loss.reshape([1]).as_slice())
        .unwrap()
        .to_vec()[0]
}

fn device_selection_compiles() -> Device {
    if false {
        Device::cpu()
    } else {
        Device::gpu_blocking().unwrap_or_else(|err| {
            println!("GPU unavailable ({err}), training on CPU");
            Device::cpu()
        })
    }
}

fn wait_for_device_compiles(device: &Device) {
    if let Device::Gpu(gpu) = device {
        gpu.poll_wait();
    }
}

fn profile_compiles(device: &Device) {
    if let Device::Gpu(gpu) = device {
        let profiles = gpu.take_kernel_profiles();
        let mut totals: std::collections::BTreeMap<String, (usize, f64, f64)> = Default::default();
        let mut span = 0.0;
        let mut kernels = 0;
        for profile in &profiles {
            span += profile.span_ms.unwrap_or(0.0);
            kernels += profile.kernels;
            for row in &profile.top_names {
                let entry = totals.entry(row.name.clone()).or_default();
                entry.0 += row.count;
                entry.1 += row.total_ms;
                entry.2 = entry.2.max(row.max_us);
            }
        }
        let _ = (span, kernels, profiles.is_empty(), profiles.len());
    }
}

fn adamw_step_compiles(
    device: &Device,
    params: &mut Params,
    gradient: RawTensor<1, f32>,
    learning_rate: f32,
    loss_scale: f32,
) {
    let mut momentum = RawTensor::<1, f32>::zeros(device, [4]);
    let mut variance = RawTensor::<1, f32>::zeros(device, [4]);
    let clip = Some(RawTensor::from_slice(device, [1], &[1.0f32]));
    let (beta1, beta2, epsilon, weight_decay, clip_norm) = (0.9f32, 0.999f32, 1e-7f32, 1e-4f32, 1.0f32);

    let alpha = RawTensor::from_slice(device, [1], &[learning_rate]);
    let decay = RawTensor::from_slice(device, [1], &[learning_rate * weight_decay]);

    let gradient = if loss_scale == 1.0 {
        gradient
    } else {
        (gradient * (1.0 / loss_scale)).into_concrete()
    };
    let gradient = match &clip {
        None => gradient,
        Some(clip) => {
            let norm = gradient
                .sqr()
                .sum::<0>(0)
                .reshape([1])
                .sqrt()
                .max_scalar(clip_norm)
                .into_concrete();
            gradient
                .mul_::<1, 1, _>(&clip.div_::<1, 1, _>(&norm).into_concrete())
                .into_concrete()
        }
    };

    momentum = (momentum.clone() * beta1 + gradient.clone() * (1.0 - beta1)).into_concrete();
    variance =
        (variance.clone() * beta2 + (gradient.clone() * gradient) * (1.0 - beta2)).into_concrete();

    let update = momentum
        .mul_::<1, 1, _>(&alpha)
        .div_(&variance.sqrt().add_scalar(epsilon).into_concrete());
    let decayed = params.flat.clone() - params.flat.mul_::<1, 1, _>(&decay);
    params.flat = (decayed - update).into_concrete();
}

fn remaining_spellings_compile(device: &Device) {
    let a = RawTensor::<2, f32>::zeros(device, [2, 3]);
    let _: RawTensor<2, f32> = a.clamp(-1.0, 1.0);
    let _: RawTensor<2, f32> = a.to_concrete();
    let _: usize = a.elements();
    let _: [usize; 2] = a.shape();
    let _: RawTensor<1, f32> = a.flatten_all();
    let _: RawTensor<2, half::f16> = a.cast::<half::f16>();
    let idx = RawTensor::<1, u32>::from_slice(device, [1], &[0]);
    let _: RawTensor<2, f32> = a.index_select(0, &idx);

    let graph = Graph::new();
    let t: Tensor<2> = Tensor::constant_from_raw(&graph, a.clone());
    let _: Tensor<2> = t.clamp(-1.0, 1.0);
    let _: Tensor<2> = t.max_scalar(0.0);
    let _: Tensor<2> = t.lte_scalar(0.5);
    let _: Tensor<2> = t.gte_scalar(0.5);
    let _: Tensor<2> = t.index_select(0, &idx);
    let _: crate::autograd::GradientSlot = t.slot();
    let _: crate::autograd::Parent = t.parent();
    let _: &RawTensor<2, f32> = t.raw();
    let _: usize = t.elements();
    let _: RawTensor<2, f32> = t.into_raw();

    let zero: RawTensor<0, f32> = RawTensor::splat(device, 0.0, []);
    let _: RawTensor<1, f32> = zero.reshape([1]);
    let _: RawTensor<3, f32> = RawTensor::ones(device, [1, 2, 3]);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `use crate::{Device, Tensor as RawTensor, ToVec, cat};` line at
    /// the top of this file is the compile check; this text pin adds that the
    /// root exports live in `lib.rs` rather than behind a feature.
    #[test]
    fn the_root_names_the_const_rank_tensor_unconditionally() {
        let src = include_str!("lib.rs");
        assert!(
            !src.contains("cfg(feature"),
            "lib.rs gained a cfg. The crate root is one naming; a feature that \
             swaps what `Tensor` or `Device` means is what this crate just \
             removed."
        );
        assert!(
            src.contains("pub use device::Device;")
                && src.contains("pub use tensor::typed::{Axis, Element, Minus1, Minus2, Tensor, cat, stack};"),
            "lib.rs no longer re-exports the const-rank root. betlang-train \
             resolves `use fusor::{{Device, Tensor as RawTensor, ToVec, cat}}` \
             through it."
        );
    }

    /// The two `use` lines the trainer opens with, and the constructors they
    /// have to reach. Runs rather than merely compiles, so the ambient graph
    /// is exercised too.
    #[test]
    fn the_trainer_constructors_resolve_and_run() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();
        let params = Params::from_slice(&device, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(params.flat.shape(), [4]);
        let graph = Graph::new();
        let leaf = graph.leaf(params.flat.narrow(0, 0, 2).into_concrete());
        let quantized = fake_quant_compiles(&graph, &leaf, false);
        assert_eq!(quantized.shape(), [2]);
    }

    /// A rank-generic helper: `R` unbound, inferred from the return position.
    #[test]
    fn a_rank_generic_weight_accessor_type_checks() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();
        let graph = Graph::new();
        let flat = RawTensor::<1, f32>::from_slice(&device, [6], &[0.0; 6]);
        let model = Model {
            leaves: Vec::new(),
            quantized: vec![graph.leaf(flat)],
            graph,
            shapes: vec![&[2, 3]],
        };
        let w: Tensor<2> = model.weight(0);
        assert_eq!(w.shape(), [2, 3]);
        let h: Tensor<2, half::f16> = model.half_weight(0);
        assert_eq!(h.shape(), [2, 3]);
    }

    /// The whole of `optim.rs` in miniature: operators, `mul_`, `div_` with
    /// every generic argument inferred, and the readback.
    #[test]
    fn an_adamw_step_runs_end_to_end() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();
        let mut params = Params {
            flat: RawTensor::from_slice(&device, [4], &[1.0, 2.0, 3.0, 4.0]),
        };
        let gradient = RawTensor::from_slice(&device, [4], &[0.5, 0.5, 0.5, 0.5]);
        adamw_step_compiles(&device, &mut params, gradient, 1e-3, 1.0);
        let updated = read_weights_compiles(&params);
        assert_eq!(updated.len(), 4);
        assert!(updated.iter().all(|v| v.is_finite()));
        // A step moves the weights down, since every gradient is positive.
        assert!(updated[0] < 1.0);
    }

    /// `ToVec` on a rank-2 readback is `Vec<Vec<f32>>`, un-`Result`-ed.
    #[test]
    fn a_rank_two_readback_nests_without_a_result() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();
        let a = RawTensor::<2, f32>::from_slice(&device, [2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let values = pollster::block_on(a.as_slice()).unwrap().to_vec();
        let row: &[f32] = &values[1];
        assert_eq!(row, [3.0, 4.0]);
    }

    // `crate::Tensor`'s folds and contractions accumulate in
    // `Dtype::compute_dtype`, so they return f32 for an f16 operand and leave
    // the narrowing cast to the const-rank facade. These tests run that path.

    /// Every rank-reducing fold, every keepdim fold and both contractions,
    /// on an f16 operand: the result is f16, and it is the right f16.
    ///
    /// The values are exact in f16 (halves and small integers), so this
    /// compares for equality rather than under a tolerance.
    #[test]
    fn every_typed_fold_returns_the_operand_dtype_in_f16() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();
        let h = half::f16::from_f32;
        // [[1, 2, 3, 4], [0.5, 0.5, 2, 1]]
        let a = RawTensor::<2, half::f16>::from_slice(
            &device,
            [2, 4],
            &[h(1.0), h(2.0), h(3.0), h(4.0), h(0.5), h(0.5), h(2.0), h(1.0)],
        );
        let f = |t: &RawTensor<1, half::f16>| -> Vec<f32> {
            t.to_flat().into_iter().map(f32::from).collect()
        };

        assert_eq!(f(&a.sum::<1>(1usize)), [10.0, 4.0]);
        assert_eq!(f(&a.product::<1>(1usize)), [24.0, 0.5]);
        assert_eq!(f(&a.max::<1>(1usize)), [4.0, 2.0]);
        assert_eq!(f(&a.min::<1>(1usize)), [1.0, 0.5]);
        assert_eq!(f(&a.mean::<1>(1usize)), [2.5, 1.0]);
        assert_eq!(
            f(&a.norm::<1>(1usize)),
            [f32::from(h(30f32.sqrt())), f32::from(h(5.5f32.sqrt()))]
        );
        assert_eq!(f(&a.count_nonzero::<1>(1usize)), [4.0, 4.0]);
        assert_eq!(f(&a.any::<1>(1usize)), [1.0, 1.0]);
        assert_eq!(f(&a.all::<1>(1usize)), [1.0, 1.0]);
        // var(1,2,3,4) = 1.25; var(0.5,0.5,2,1) = 0.375
        assert_eq!(f(&a.var::<1>(1usize)), [1.25, 0.375]);

        // Keepdim forms keep the rank and the dtype.
        let keep = a.sum_keepdim(1usize);
        assert_eq!(keep.shape(), [2, 1]);
        assert_eq!(keep.dtype(), crate::Dtype::F16);
        for k in [
            a.product::<1>(1usize).unsqueeze(1),
            a.max_keepdim(1usize),
            a.min::<1>(1usize).unsqueeze(1),
            a.mean_keepdim(1usize),
            a.var_keepdim(1usize),
        ] {
            assert_eq!(k.dtype(), crate::Dtype::F16);
            assert_eq!(k.shape(), [2, 1]);
        }

        // Contractions accumulate wide and narrow back too. a @ a^T.
        let m = a.matmul_t(&a);
        assert_eq!(m.dtype(), crate::Dtype::F16);
        let flat: Vec<f32> = m.to_flat().into_iter().map(f32::from).collect();
        // [1,2,3,4].[1,2,3,4] = 30; .[0.5,0.5,2,1] = 11.5; [.5,.5,2,1]^2 = 5.5
        assert_eq!(flat, [30.0, 11.5, 11.5, 5.5]);
        assert_eq!(a.matmul(&a.t()).dtype(), crate::Dtype::F16);
    }

    /// The same contract for bf16, the other dtype whose compute dtype is
    /// wider than itself.
    #[test]
    fn a_bf16_fold_returns_bf16() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();
        let b = half::bf16::from_f32;
        let a = RawTensor::<2, half::bf16>::from_slice(
            &device,
            [1, 4],
            &[b(1.0), b(2.0), b(3.0), b(4.0)],
        );
        let s = a.sum::<1>(1usize);
        assert_eq!(s.dtype(), crate::Dtype::BF16);
        assert_eq!(f32::from(s.to_flat()[0]), 10.0);
        assert_eq!(a.max::<1>(1usize).dtype(), crate::Dtype::BF16);
    }

    /// A genuine dtype disagreement is still a panic naming both dtypes, not
    /// a silent conversion.
    #[test]
    #[should_panic(expected = "value has dtype")]
    fn a_real_dtype_mismatch_is_still_reported() {
        let device = Device::cpu();
        let a = RawTensor::<1, f32>::from_slice(&device, [2], &[1.0, 2.0]);
        // The value is f32; asserting it is f16 is a bug in the model, and
        // `f32 -> f16` is not the promotion `narrow` undoes.
        let _: RawTensor<1, half::f16> = RawTensor::from_dyn(a.into_inner());
    }

    /// The trainer's own `conv_stack_f16` run on real values. Its `pool` is
    /// `reshape(..).max::<3>(3)` on a `Tensor<3, half::f16>`, which is the
    /// fold under test.
    #[test]
    fn the_trainers_f16_convolution_stack_computes() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();
        let graph = Graph::new();
        let h = half::f16::from_f32;

        // (rows 1, channels 2, seq 8) -> pool 4 -> pool 2 -> seq 1.
        let xs: Vec<f32> = (0..16).map(|i| (i % 5) as f32 * 0.5).collect();
        // Three (kernel 3, in 2, out 2) weights and three length-2 biases.
        let ws: Vec<f32> = (0..12).map(|i| ((i % 3) as f32 - 1.0) * 0.5).collect();
        let bs = [0.5f32, -0.5];

        let raw32 = |v: &[f32], s: [usize; 3]| RawTensor::<3, f32>::from_slice(&device, s, v);
        let raw16 = |v: &[f32], s: [usize; 3]| {
            RawTensor::<3, half::f16>::from_slice(
                &device,
                s,
                &v.iter().copied().map(h).collect::<Vec<_>>(),
            )
        };

        let x32 = graph.leaf(raw32(&xs, [1, 2, 8]));
        let w32: Vec<Tensor<3>> = (0..3).map(|_| graph.leaf(raw32(&ws, [3, 2, 2]))).collect();
        let b32: Vec<Tensor<1>> = (0..3)
            .map(|_| graph.leaf(RawTensor::<1, f32>::from_slice(&device, [2], &bs)))
            .collect();
        let out32 = conv_stack_f32(
            &x32,
            [w32[0].clone(), w32[1].clone(), w32[2].clone()],
            [b32[0].clone(), b32[1].clone(), b32[2].clone()],
        );

        let x16 = graph.leaf(raw16(&xs, [1, 2, 8]));
        let w16: Vec<Tensor<3, half::f16>> =
            (0..3).map(|_| graph.leaf(raw16(&ws, [3, 2, 2]))).collect();
        let b16: Vec<Tensor<1, half::f16>> = (0..3)
            .map(|_| {
                graph.leaf(RawTensor::<1, half::f16>::from_slice(
                    &device,
                    [2],
                    &bs.map(h),
                ))
            })
            .collect();
        let out16 = conv_stack_f16(
            &x16,
            [w16[0].clone(), w16[1].clone(), w16[2].clone()],
            [b16[0].clone(), b16[1].clone(), b16[2].clone()],
        );

        assert_eq!(out16.shape(), [1, 2, 1]);
        assert_eq!(out16.shape(), out32.shape());
        // The whole point: the stack stayed in the dtype its type says it is.
        assert_eq!(out16.dtype(), crate::Dtype::F16);
        assert_eq!(out32.dtype(), crate::Dtype::F32);

        let got: Vec<f32> = out16.raw().to_flat().into_iter().map(f32::from).collect();
        let want = out32.raw().to_flat();
        // f16 carries 11 significant bits, so a value near 1 resolves to about
        // 5e-4. Three convolutions, each a 6-term reduction, plus three gelus
        // compound that; 5e-3 absolute is the band that separates "f16 round
        // off" from "a different computation".
        for (g, w) in got.iter().zip(&want) {
            assert!(
                (g - w).abs() <= 5e-3,
                "f16 stack {got:?} disagrees with f32 stack {want:?}"
            );
        }
        // And it is not trivially zero, which would satisfy the band above.
        assert!(got.iter().any(|v| v.abs() > 0.05), "degenerate: {got:?}");

        // `.cast::<f32>()` back out, the way `forward` rejoins the f32 head.
        let rejoined = out16.cast::<f32>();
        assert_eq!(rejoined.dtype(), crate::Dtype::F32);
        assert_eq!(rejoined.shape(), [1, 2, 1]);
    }

    /// The trainer's `conv_stack_f16` forward and the backward through it,
    /// landing on the f32 masters. The pool is `reshape(..).max::<3>(3)` on
    /// an f16 operand, so this exercises the `Max` fold adjoint. The point is
    /// that the backward reaches every weight and does not come back zero.
    #[test]
    fn a_backward_through_the_f16_convolution_stack_reaches_the_weights() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();
        let graph = Graph::new();
        let h = half::f16::from_f32;

        let xs: Vec<f32> = (0..16).map(|i| (i % 5) as f32 * 0.5).collect();
        let ws: Vec<f32> = (0..12).map(|i| ((i % 3) as f32 - 1.0) * 0.5).collect();
        let bs = [0.5f32, -0.5];

        // One f32 master per weight, cast onto the tape exactly as
        // `half_weight` does, so the gradient has to travel back through the
        // f16 stack and the cast to arrive in f32.
        let masters: Vec<Tensor<3>> = (0..3)
            .map(|_| graph.leaf(RawTensor::<3, f32>::from_slice(&device, [3, 2, 2], &ws)))
            .collect();
        let bias_masters: Vec<Tensor<1>> = (0..3)
            .map(|_| graph.leaf(RawTensor::<1, f32>::from_slice(&device, [2], &bs)))
            .collect();
        let x16 = graph.leaf(RawTensor::<3, half::f16>::from_slice(
            &device,
            [1, 2, 8],
            &xs.iter().copied().map(h).collect::<Vec<_>>(),
        ));
        let w16: Vec<Tensor<3, half::f16>> =
            masters.iter().map(|m| m.cast::<half::f16>()).collect();
        let b16: Vec<Tensor<1, half::f16>> =
            bias_masters.iter().map(|m| m.cast::<half::f16>()).collect();

        let out16 = conv_stack_f16(
            &x16,
            [w16[0].clone(), w16[1].clone(), w16[2].clone()],
            [b16[0].clone(), b16[1].clone(), b16[2].clone()],
        );
        // Rejoin f32 the way `forward` does, then reduce to a scalar loss.
        let loss = out16
            .cast::<f32>()
            .sum::<2>(2usize)
            .sum::<1>(1usize)
            .sum::<0>(0usize);
        let seed = RawTensor::<0, f32>::splat(&device, 1.0, []);
        let grads = loss
            .backward_with(seed)
            .expect("a backward through an f16 max pool must build");

        for (i, m) in masters.iter().enumerate() {
            let g = grads.get(m).unwrap_or_else(|| panic!("weight {i} gradient"));
            assert_eq!(g.dtype(), crate::Dtype::F32, "weight {i} master stays f32");
            assert_eq!(g.shape(), [3, 2, 2]);
            let v = g.to_flat();
            assert!(
                v.iter().all(|x| x.is_finite()),
                "weight {i} gradient has a non-finite entry: {v:?}"
            );
        }
        for (i, m) in bias_masters.iter().enumerate() {
            let g = grads.get(m).unwrap_or_else(|| panic!("bias {i} gradient"));
            assert_eq!(g.dtype(), crate::Dtype::F32);
            assert!(g.to_flat().iter().all(|x| x.is_finite()));
        }
        // A max pool routes the gradient to one slot per window, so the last
        // layer's gradient is sparse but never everywhere zero — an all-zero
        // result is exactly what a silently-dropped adjoint looks like.
        let total: f32 = masters
            .iter()
            .flat_map(|m| grads.get(m).unwrap().to_flat())
            .map(f32::abs)
            .sum();
        assert!(total > 1e-3, "every weight gradient is zero: {total}");
    }

    /// `half_weight::<R>` is `weight::<R>().cast::<f16>()`, and the cast is on
    /// the tape: a gradient taken through the f16 copy lands on the f32
    /// master.
    #[test]
    fn a_gradient_through_a_half_weight_lands_on_the_f32_master() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();
        let graph = Graph::new();
        let master = graph.leaf(RawTensor::<1, f32>::from_slice(
            &device,
            [4],
            &[1.0, 2.0, 3.0, 4.0],
        ));
        // sum(2 * f16(w)) -> d/dw = 2, arriving back in f32.
        let loss = master
            .cast::<half::f16>()
            .mul_scalar(2.0)
            .cast::<f32>()
            .sum::<0>(0usize);
        let seed = RawTensor::<0, f32>::splat(&device, 1.0, []);
        let grads = loss.backward_with(seed).expect("backward");
        let g = grads.get(&master).expect("gradient");
        assert_eq!(g.dtype(), crate::Dtype::F32);
        assert_eq!(g.to_flat(), vec![2.0; 4]);
    }

    /// The f32 head the trainer runs by default, end to end with a backward:
    /// global max ++ global mean, `mat_mul`, `add_::<1, 2>`, `gelu`. Values
    /// and gradients both checked against a hand reference.
    #[test]
    fn the_pooled_head_computes_and_backpropagates() {
        let _serial = crate::device::test_device_lock();
        let device = Device::cpu();
        let graph = Graph::new();
        // (rows 1, channels 2, seq 4)
        let x = graph.leaf(RawTensor::<1, f32>::from_slice(
            &device,
            [8],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        ));
        let x3 = x.reshape([1, 2, 4]);
        // max over seq = [4, 8]; mean over seq = [2.5, 6.5]
        let pooled = Tensor::cat(vec![x3.max::<2>(2usize), x3.mean::<2>(2usize)], 1);
        assert_eq!(pooled.shape(), [1, 4]);
        assert_eq!(pooled.raw().to_flat(), vec![4.0, 8.0, 2.5, 6.5]);

        let w = graph.leaf(RawTensor::<1, f32>::from_slice(&device, [12], &[0.25; 12]));
        let b = graph.leaf(RawTensor::<1, f32>::from_slice(&device, [3], &[1.0, 2.0, 3.0]));
        // (1,4) @ (4,3) = 0.25 * 21 = 5.25, + bias.
        let hidden = pooled.matmul(&w.reshape([4, 3])).add_::<1, 2>(&b.reshape([3]));
        assert_eq!(hidden.shape(), [1, 3]);
        assert_eq!(hidden.raw().to_flat(), vec![6.25, 7.25, 8.25]);

        let loss = hidden.flatten_all().sum::<0>(0usize);
        let seed = RawTensor::<0, f32>::splat(&device, 1.0, []);
        let grads = loss.backward_with(seed).expect("backward");

        // d(loss)/d(bias) = 1 per output.
        assert_eq!(grads.get(&b).expect("bias grad").to_flat(), vec![1.0; 3]);
        // d(loss)/d(pooled_j) = sum_k w[j][k] = 0.75, so each of the four
        // pooled entries pulls 0.75 back. The max slot sends all of it to the
        // argmax (positions 3 and 7); the mean slot spreads 0.75/4 over all
        // four positions of its row.
        let gx = grads.get(&x).expect("x grad").to_flat();
        let m = 0.75 / 4.0;
        for (i, g) in gx.iter().enumerate() {
            let want = if i == 3 || i == 7 { 0.75 + m } else { m };
            assert!((g - want).abs() < 1e-6, "x grad {gx:?} at {i}");
        }
    }
}
