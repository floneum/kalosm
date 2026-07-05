//! Train a small convolutional network on MNIST with the autograd API.
//!
//! Downloads the MNIST dataset on first run (cached under `examples/data`),
//! then trains conv(1->8) -> pool -> conv(8->16) -> pool -> linear(784->10)
//! with softmax cross-entropy and plain SGD, rebuilding the tape each step.
//!
//! Run with:
//! ```sh
//! cargo run --release --example mnist
//! ```

use std::io::Read;
use std::path::PathBuf;

use fusor::autograd::layers::{ConvNd, ConvNdConfig, Linear};
use fusor::autograd::{Gradients, Graph, Tensor};
use fusor::{Device, Tensor as RawTensor, ToVec1, ToVec2};

const BATCH_SIZE: usize = 64;
const EPOCHS: usize = 1;
const LEARNING_RATE: f32 = 0.05;
const IMAGE_SIZE: usize = 28;
const PIXELS: usize = IMAGE_SIZE * IMAGE_SIZE;
const CLASSES: usize = 10;

const MNIST_MIRROR: &str = "https://ossci-datasets.s3.amazonaws.com/mnist";

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/data")
}

/// Download and gunzip one MNIST idx file, caching the decompressed bytes.
fn fetch(name: &str) -> Vec<u8> {
    let path = data_dir().join(name);
    if let Ok(bytes) = std::fs::read(&path) {
        return bytes;
    }
    println!("downloading {name}.gz");
    let mut response = ureq::get(format!("{MNIST_MIRROR}/{name}.gz"))
        .call()
        .unwrap_or_else(|err| panic!("failed to download {name}: {err}"));
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(response.body_mut().as_reader())
        .read_to_end(&mut bytes)
        .unwrap_or_else(|err| panic!("failed to decompress {name}: {err}"));
    std::fs::create_dir_all(data_dir()).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    bytes
}

fn read_be_u32(bytes: &[u8], offset: usize) -> usize {
    u32::from_be_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize
}

/// Parse an idx image file into pixels normalized to [0, 1].
fn load_images(name: &str) -> Vec<f32> {
    let bytes = fetch(name);
    assert_eq!(read_be_u32(&bytes, 0), 0x803, "bad image file magic");
    assert_eq!(read_be_u32(&bytes, 8), IMAGE_SIZE);
    assert_eq!(read_be_u32(&bytes, 12), IMAGE_SIZE);
    bytes[16..].iter().map(|&pixel| pixel as f32 / 255.0).collect()
}

fn load_labels(name: &str) -> Vec<u32> {
    let bytes = fetch(name);
    assert_eq!(read_be_u32(&bytes, 0), 0x801, "bad label file magic");
    bytes[8..].iter().map(|&label| label as u32).collect()
}

/// Deterministic LCG so runs are reproducible without a rand dependency.
struct Lcg(u64);

impl Lcg {
    /// Uniform in (-bound, bound) with `bound = sqrt(1 / fan_in)`, the
    /// PyTorch default init for conv and linear layers.
    fn kaiming(&mut self, count: usize, fan_in: usize) -> Vec<f32> {
        let bound = (1.0 / fan_in as f32).sqrt();
        (0..count)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let uniform = (self.0 >> 32) as f32 / (1u64 << 32) as f32 - 0.5;
                uniform * 2.0 * bound
            })
            .collect()
    }
}

/// Raw parameter tensors that persist across training steps.
struct Params {
    conv1_weight: RawTensor<4, f32>,
    conv1_bias: RawTensor<1, f32>,
    conv2_weight: RawTensor<4, f32>,
    conv2_bias: RawTensor<1, f32>,
    fc_weight: RawTensor<2, f32>,
    fc_bias: RawTensor<1, f32>,
}

impl Params {
    fn new(device: &Device) -> Self {
        let mut rng = Lcg(42);
        let flat = 16 * (IMAGE_SIZE / 4) * (IMAGE_SIZE / 4);
        Self {
            conv1_weight: RawTensor::from_slice(device, [8, 1, 3, 3], &rng.kaiming(8 * 9, 9)),
            conv1_bias: RawTensor::zeros(device, [8]),
            conv2_weight: RawTensor::from_slice(
                device,
                [16, 8, 3, 3],
                &rng.kaiming(16 * 8 * 9, 8 * 9),
            ),
            conv2_bias: RawTensor::zeros(device, [16]),
            fc_weight: RawTensor::from_slice(
                device,
                [CLASSES, flat],
                &rng.kaiming(CLASSES * flat, flat),
            ),
            fc_bias: RawTensor::zeros(device, [CLASSES]),
        }
    }
}

/// conv(1->8) -> relu -> maxpool -> conv(8->16) -> relu -> maxpool -> linear.
struct Cnn {
    conv1: ConvNd<2, 4>,
    conv2: ConvNd<2, 4>,
    fc: Linear,
}

impl Cnn {
    /// Build the model on `graph`, as trainable leaves (training) or
    /// constants (evaluation).
    fn new(graph: &Graph, params: &Params, trainable: bool) -> Self {
        fn wrap<const R: usize>(
            graph: &Graph,
            value: &RawTensor<R, f32>,
            trainable: bool,
        ) -> Tensor<R> {
            if trainable {
                graph.leaf(value.clone())
            } else {
                Tensor::constant_from_raw(graph, value.clone())
            }
        }
        let config = ConvNdConfig {
            padding: [1, 1],
            stride: [1, 1],
            groups: 1,
        };
        Self {
            conv1: ConvNd::new(
                wrap(graph, &params.conv1_weight, trainable),
                Some(wrap(graph, &params.conv1_bias, trainable)),
                config,
            ),
            conv2: ConvNd::new(
                wrap(graph, &params.conv2_weight, trainable),
                Some(wrap(graph, &params.conv2_bias, trainable)),
                config,
            ),
            fc: Linear::new(
                wrap(graph, &params.fc_weight, trainable),
                Some(wrap(graph, &params.fc_bias, trainable)),
            ),
        }
    }

    /// Input shape: (batch, 1, 28, 28). Output shape: (batch, 10).
    fn forward(&self, images: &Tensor<4>) -> Tensor<2> {
        let x = self
            .conv1
            .forward(images)
            .relu()
            .pool_max::<2, 6, 7, 5>([2, 2]);
        let x = self
            .conv2
            .forward(&x)
            .relu()
            .pool_max::<2, 6, 7, 5>([2, 2]);
        self.fc.forward_2d(&x.flatten_last_n::<2, 2>())
    }
}

/// Numerically stable softmax cross-entropy averaged over the batch:
/// log softmax via log-sum-exp so a saturated class cannot underflow.
fn cross_entropy(logits: &Tensor<2>, targets: &RawTensor<1, u32>) -> Tensor<0> {
    let batch = logits.shape()[0];
    let shifted = logits.sub_::<2, 2>(&logits.max_keepdim::<1>(1));
    let log_sum_exp = shifted.exp().sum_keepdim(1).log();
    let label_log_probs = shifted.sub_::<2, 2>(&log_sum_exp).gather_last(targets);
    label_log_probs.sum().mul_scalar(-1.0 / batch as f32)
}

fn sgd_step<const R: usize>(param: &mut RawTensor<R, f32>, gradients: &Gradients, leaf: &Tensor<R>) {
    let gradient = gradients.get(leaf).expect("missing gradient");
    *param = (param.clone() - gradient * LEARNING_RATE).to_concrete();
}

async fn to_scalar(value: RawTensor<0, f32>) -> f32 {
    value.reshape([1]).as_slice().await.unwrap().to_vec1()[0]
}

fn argmax(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0 as u32
}

#[tokio::main]
async fn main() {
    let train_images = load_images("train-images-idx3-ubyte");
    let train_labels = load_labels("train-labels-idx1-ubyte");
    let test_images = load_images("t10k-images-idx3-ubyte");
    let test_labels = load_labels("t10k-labels-idx1-ubyte");
    assert_eq!(train_images.len(), train_labels.len() * PIXELS);

    let device = match Device::gpu().await {
        Ok(gpu) => gpu,
        Err(_) => {
            println!("GPU unavailable, training on CPU");
            Device::cpu()
        }
    };

    let mut params = Params::new(&device);

    let steps_per_epoch = train_labels.len() / BATCH_SIZE;
    for epoch in 0..EPOCHS {
        for step in 0..steps_per_epoch {
            let start = step * BATCH_SIZE;
            let images = RawTensor::from_slice(
                &device,
                [BATCH_SIZE, 1, IMAGE_SIZE, IMAGE_SIZE],
                &train_images[start * PIXELS..(start + BATCH_SIZE) * PIXELS],
            );
            let targets = RawTensor::from_slice(
                &device,
                [BATCH_SIZE],
                &train_labels[start..start + BATCH_SIZE],
            );

            let graph = Graph::new();
            let model = Cnn::new(&graph, &params, true);
            let logits = model.forward(&Tensor::constant_from_raw(&graph, images));
            let loss = cross_entropy(&logits, &targets);

            let loss_value = to_scalar(loss.raw().clone()).await;
            let gradients = loss.backward().unwrap().into_detached();
            sgd_step(&mut params.conv1_weight, &gradients, model.conv1.weight());
            sgd_step(&mut params.conv1_bias, &gradients, model.conv1.bias().unwrap());
            sgd_step(&mut params.conv2_weight, &gradients, model.conv2.weight());
            sgd_step(&mut params.conv2_bias, &gradients, model.conv2.bias().unwrap());
            sgd_step(&mut params.fc_weight, &gradients, model.fc.weight());
            sgd_step(&mut params.fc_bias, &gradients, model.fc.bias().unwrap());

            if step % 50 == 0 {
                println!("epoch {epoch} step {step}/{steps_per_epoch}: loss {loss_value:.4}");
            }
        }
    }

    const EVAL_BATCH: usize = 500;
    let mut correct = 0;
    for (images, labels) in test_images
        .chunks(EVAL_BATCH * PIXELS)
        .zip(test_labels.chunks(EVAL_BATCH))
    {
        let graph = Graph::new();
        let model = Cnn::new(&graph, &params, false);
        let x = Tensor::constant_from_raw(
            &graph,
            RawTensor::from_slice(&device, [labels.len(), 1, IMAGE_SIZE, IMAGE_SIZE], images),
        );
        let logits = model.forward(&x).raw().clone().as_slice().await.unwrap().to_vec2();
        correct += logits
            .iter()
            .zip(labels)
            .filter(|(row, label)| argmax(row) == **label)
            .count();
    }
    let total = test_labels.len();
    println!(
        "test accuracy: {correct}/{total} ({:.2}%)",
        100.0 * correct as f32 / total as f32
    );
}
