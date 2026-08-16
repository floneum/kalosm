//! The trainer's API surface, spelled exactly as the trainer spells it.
//!
//! `betlang-train` consumes fusor through exactly two `use` lines and never
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
            [
                self.weight::<3>(1),
                self.weight::<3>(3),
                self.weight::<3>(5),
            ],
            [
                self.weight::<1>(2),
                self.weight::<1>(4),
                self.weight::<1>(6),
            ],
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
                x.reshape([rows, channels, seq / window, window])
                    .max::<3>(3)
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
        round_small(&normalized)
            .mul_::<1, 1, _>(&scale)
            .into_concrete()
    };
    let target = leaf.slot();
    Tensor::constant_from_raw(graph, quantized).with_backwards([leaf.parent()], move |gradient| {
        Ok(vec![BackwardTarget::to(target, gradient)])
    })
}

fn round_small(normalized: &RawTensor<1, f32>) -> RawTensor<1, f32> {
    let mut rounded = (normalized.gte_scalar(0.5) - normalized.lte_scalar(-0.5)).into_concrete();
    for level in 2..=7 {
        let threshold = level as f32 - 0.5;
        rounded = (rounded + normalized.gte_scalar(threshold) - normalized.lte_scalar(-threshold))
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
    let softplus = (raw.relu() + (raw.abs() * -1.0).exp().add_scalar(1.0).log()).into_concrete();
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
    let (beta1, beta2, epsilon, weight_decay, clip_norm) =
        (0.9f32, 0.999f32, 1e-7f32, 1e-4f32, 1.0f32);

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
