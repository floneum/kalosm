#![allow(unused)]
use std::time::Duration;

use candle_core::backend::BackendDevice;
use candle_nn::Module;
use criterion::BatchSize;
use fusor::layers::Linear;
use fusor::{Device, Tensor, VarBuilder};
use futures::executor::block_on;

use criterion::BenchmarkId;
use criterion::Criterion;
use criterion::{criterion_group, criterion_main};

use criterion::async_executor::FuturesExecutor;
use kalosm_common::Cache;
use kalosm_model_types::FileSource;

fn candle_gpu_device() -> Option<candle_core::Device> {
    candle_core::Device::new_cuda(0)
        .or_else(|_| candle_core::Device::new_metal(0))
        .ok()
}

fn quick_bench() -> bool {
    std::env::args().any(|arg| arg == "--quick")
}

// Benchmark LayerNorm operation
fn layer_norm(c: &mut Criterion) {
    let source = FileSource::HuggingFace {
        model_id: "CompendiumLabs/bge-large-en-v1.5-gguf".to_string(),
        revision: "main".to_string(),
        file: "bge-large-en-v1.5-q4_k_m.gguf".to_string(),
    };
    let bytes = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async move {
            let cache = Cache::default();
            let path = cache.get(&source, |_| {}).await.unwrap();
            tokio::fs::read(&path).await.unwrap()
        });

    let mut group = c.benchmark_group("layer_norm");
    let quick = quick_bench();
    group.sample_size(if quick { 10 } else { 20 });
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    let batch_sizes: &[usize] = if quick { &[1] } else { &[1, 32, 512] };
    let seq_lens: &[usize] = if quick { &[13] } else { &[13, 128, 512] };
    for &batch_size in batch_sizes {
        for &seq_len in seq_lens {
            if batch_size * seq_len >= 512 * 128 {
                continue;
            }
            let hidden_size = 1024;
            let random_data: Vec<Vec<Vec<f32>>> = (0..batch_size)
                .map(|_| {
                    (0..seq_len)
                        .map(|_| (0..hidden_size).map(|_| rand::random()).collect())
                        .collect()
                })
                .collect();

            // Fusor LayerNorm benchmark
            {
                let mut reader = std::io::Cursor::new(&bytes);
                let mut var_builder = VarBuilder::from_gguf(&mut reader).unwrap();
                let device = block_on(async { Device::new().await.unwrap() });

                // Load layer norm weights from the model
                let weight: Tensor<1, f32> = var_builder
                    .pp("blk.0.attn_output_norm")
                    .get("weight", &device)
                    .unwrap()
                    .dequantize();
                let bias: Option<Tensor<1, f32>> = var_builder
                    .pp("blk.0.attn_output_norm")
                    .get("bias", &device)
                    .ok()
                    .map(|b| b.dequantize());

                let device = device.clone();
                let random_data = random_data.clone();
                group.bench_with_input(
                    BenchmarkId::new("fusor-gpu", format!("{batch_size}x{seq_len}")),
                    &(batch_size, seq_len),
                    move |b, &(batch_size, seq_len)| {
                        let device = device.clone();
                        let random_data = random_data.clone();
                        b.to_async(FuturesExecutor).iter_custom(async |iters| {
                            let tensor: Tensor<3, f32> = Tensor::from_slice(
                                &device,
                                [batch_size, seq_len, 1024],
                                &random_data
                                    .iter()
                                    .flat_map(|b| b.iter().flat_map(|s| s.iter().copied()))
                                    .collect::<Vec<_>>(),
                            );
                            tensor.as_gpu().unwrap().materialize().await;
                            let weight_broadcast = weight.broadcast_as([batch_size, seq_len, 1024]);
                            let bias_broadcast = bias
                                .as_ref()
                                .map(|b| b.broadcast_as([batch_size, seq_len, 1024]));
                            let mut sum = Duration::ZERO;
                            while sum.is_zero() {
                                for _ in 0..iters {
                                    let start = std::time::Instant::now();
                                    let normalized = tensor.layer_norm(
                                        &weight_broadcast,
                                        bias_broadcast.as_ref(),
                                        1e-12,
                                        true,
                                    );
                                    normalized.as_gpu().unwrap().materialize().await;
                                    sum += start.elapsed();
                                }
                            }
                            sum
                        });
                    },
                );
            }

            // Candle LayerNorm benchmark
            let layer_norm_shape = BertShape {
                batch_size,
                seq_len,
                hidden_size: 1024,
            };
            {
                let candle_device = candle_core::Device::Cpu;
                bench_candle_layer_norm(
                    &bytes,
                    layer_norm_shape,
                    random_data.clone(),
                    candle_device,
                    "candle-cpu",
                    &mut group,
                );
            }

            // Candle LayerNorm benchmark on GPU.
            if let Some(candle_device) = candle_gpu_device() {
                bench_candle_layer_norm(
                    &bytes,
                    layer_norm_shape,
                    random_data.clone(),
                    candle_device,
                    "candle-gpu",
                    &mut group,
                );
            }
        }
    }
    group.finish();
}

/// Common shape parameters for the BERT block benches; bundled to keep
/// the candle bench helpers under the 7-arg clippy limit.
#[derive(Clone, Copy)]
struct BertShape {
    batch_size: usize,
    seq_len: usize,
    hidden_size: usize,
}

fn bench_candle_layer_norm(
    bytes: &[u8],
    shape: BertShape,
    random_data: Vec<Vec<Vec<f32>>>,
    candle_device: candle_core::Device,
    name: &str,
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    let BertShape {
        batch_size,
        seq_len,
        hidden_size,
    } = shape;
    use candle_nn::LayerNorm;

    let var_builder = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
        bytes,
        &candle_device,
    )
    .unwrap()
    .pp("blk.0.attn_output_norm");

    let weight = var_builder
        .get_no_shape("weight")
        .unwrap()
        .dequantize(&candle_device)
        .unwrap();
    let bias = var_builder
        .get_no_shape("bias")
        .unwrap()
        .dequantize(&candle_device)
        .unwrap();

    let layer_norm = candle_nn::LayerNorm::new(weight, bias, 1e-12);

    let probe = candle_core::Tensor::zeros(
        &[batch_size, seq_len, hidden_size],
        candle_core::DType::F32,
        &candle_device,
    )
    .and_then(|tensor| layer_norm.forward(&tensor))
    .and_then(|_| candle_device.synchronize());
    if let Err(error) = probe {
        eprintln!("Skipping {name} layer_norm {batch_size}x{seq_len}: {error}");
        return;
    }

    group.bench_with_input(
        BenchmarkId::new(name, format!("{batch_size}x{seq_len}")),
        &(batch_size, seq_len),
        move |b, &(batch_size, seq_len)| {
            b.to_async(FuturesExecutor).iter_batched(
                || {
                    let candle_tensor = candle_core::Tensor::from_iter(
                        random_data
                            .iter()
                            .flat_map(|b| b.iter().flat_map(|s| s.iter().copied())),
                        &candle_device,
                    )
                    .unwrap()
                    .reshape(&[batch_size, seq_len, hidden_size])
                    .unwrap();
                    candle_device.synchronize().unwrap();
                    (candle_tensor, layer_norm.clone(), candle_device.clone())
                },
                |(tensor, layer_norm, candle_device)| async move {
                    layer_norm.forward(&tensor).unwrap();
                    candle_device.synchronize().unwrap();
                },
                BatchSize::LargeInput,
            );
        },
    );
}

// Benchmark Self-Attention operation
fn self_attention(c: &mut Criterion) {
    let source = FileSource::HuggingFace {
        model_id: "CompendiumLabs/bge-large-en-v1.5-gguf".to_string(),
        revision: "main".to_string(),
        file: "bge-large-en-v1.5-q4_k_m.gguf".to_string(),
    };
    let bytes = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async move {
            let cache = Cache::default();
            let path = cache.get(&source, |_| {}).await.unwrap();
            tokio::fs::read(&path).await.unwrap()
        });

    let mut group = c.benchmark_group("self_attention");
    let quick = quick_bench();
    group.sample_size(if quick { 10 } else { 20 });
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    let batch_sizes: &[usize] = if quick { &[1] } else { &[1, 32] };
    let seq_lens: &[usize] = if quick { &[13] } else { &[13, 128] };
    for &batch_size in batch_sizes {
        for &seq_len in seq_lens {
            if batch_size * seq_len >= 32 * 128 {
                continue;
            }
            let hidden_size = 1024;
            let num_heads = 16;
            let head_size = hidden_size / num_heads;

            let random_data: Vec<Vec<Vec<f32>>> = (0..batch_size)
                .map(|_| {
                    (0..seq_len)
                        .map(|_| (0..hidden_size).map(|_| rand::random()).collect())
                        .collect()
                })
                .collect();

            // Fusor Self-Attention benchmark
            {
                let mut reader = std::io::Cursor::new(&bytes);
                let mut var_builder = VarBuilder::from_gguf(&mut reader).unwrap();
                let device = block_on(async { Device::new().await.unwrap() });

                let query = Linear::load(&device, &mut var_builder.pp("blk.0.attn_q")).unwrap();
                let key = Linear::load(&device, &mut var_builder.pp("blk.0.attn_k")).unwrap();
                let value = Linear::load(&device, &mut var_builder.pp("blk.0.attn_v")).unwrap();

                let device = device.clone();
                let random_data = random_data.clone();
                group.bench_with_input(
                    BenchmarkId::new("fusor-gpu", format!("{batch_size}x{seq_len}")),
                    &(batch_size, seq_len),
                    move |b, &(batch_size, seq_len)| {
                        let device = device.clone();
                        let random_data = random_data.clone();

                        b.to_async(FuturesExecutor).iter_custom(async |iters| {
                            let tensor: Tensor<3, f32> = Tensor::from_slice(
                                &device,
                                [batch_size, seq_len, hidden_size],
                                &random_data
                                    .iter()
                                    .flat_map(|b| b.iter().flat_map(|s| s.iter().copied()))
                                    .collect::<Vec<_>>(),
                            );
                            tensor.as_gpu().unwrap().materialize().await;
                            let mut sum = Duration::ZERO;
                            while sum.is_zero() {
                                for _ in 0..iters {
                                    let start = std::time::Instant::now();

                                    let q = query.forward(&tensor);
                                    let k = key.forward(&tensor);
                                    let v = value.forward(&tensor);

                                    let q = q.reshape([batch_size, seq_len, num_heads, head_size]);
                                    let q = q.transpose(1, 2).to_concrete();
                                    let k = k.reshape([batch_size, seq_len, num_heads, head_size]);
                                    let k = k.transpose(1, 2).to_concrete();
                                    let v = v.reshape([batch_size, seq_len, num_heads, head_size]);
                                    let v = v.transpose(1, 2).to_concrete();

                                    let context = q.flash_attention(
                                        &k,
                                        &v,
                                        1.0 / (head_size as f32).sqrt(),
                                        None,
                                    );
                                    let context = context.transpose(1, 2);
                                    let output = context.flatten_last_n::<1, _>();

                                    output.as_gpu().unwrap().materialize().await;
                                    sum += start.elapsed();
                                }
                            }
                            sum
                        });
                    },
                );
            }

            // Candle Self-Attention benchmark
            let attn_shape = BertShape {
                batch_size,
                seq_len,
                hidden_size,
            };
            if let Some(candle_device) = candle_gpu_device() {
                bench_candle_self_attention(
                    &bytes,
                    attn_shape,
                    num_heads,
                    random_data.clone(),
                    candle_device,
                    "candle-gpu",
                    &mut group,
                );
            }

            {
                let candle_device = candle_core::Device::Cpu;
                bench_candle_self_attention(
                    &bytes,
                    attn_shape,
                    num_heads,
                    random_data.clone(),
                    candle_device,
                    "candle-cpu",
                    &mut group,
                );
            }
        }
    }
    group.finish();
}

fn bench_candle_self_attention(
    bytes: &[u8],
    shape: BertShape,
    num_heads: usize,
    random_data: Vec<Vec<Vec<f32>>>,
    candle_device: candle_core::Device,
    name: &str,
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    let BertShape {
        batch_size,
        seq_len,
        hidden_size,
    } = shape;
    use candle_transformers::quantized_nn::Linear;

    let var_builder = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
        bytes,
        &candle_device,
    )
    .unwrap();

    let q_weight = var_builder
        .pp("blk.0.attn_q")
        .get_no_shape("weight")
        .unwrap();
    let q_bias = var_builder
        .pp("blk.0.attn_q")
        .get_no_shape("bias")
        .unwrap()
        .dequantize(&candle_device)
        .unwrap();
    let k_weight = var_builder
        .pp("blk.0.attn_k")
        .get_no_shape("weight")
        .unwrap();
    let k_bias = var_builder
        .pp("blk.0.attn_k")
        .get_no_shape("bias")
        .unwrap()
        .dequantize(&candle_device)
        .unwrap();
    let v_weight = var_builder
        .pp("blk.0.attn_v")
        .get_no_shape("weight")
        .unwrap();
    let v_bias = var_builder
        .pp("blk.0.attn_v")
        .get_no_shape("bias")
        .unwrap()
        .dequantize(&candle_device)
        .unwrap();

    let q_linear = Linear::from_arc(q_weight, Some(q_bias)).unwrap();
    let k_linear = Linear::from_arc(k_weight, Some(k_bias)).unwrap();
    let v_linear = Linear::from_arc(v_weight, Some(v_bias)).unwrap();

    let head_size = hidden_size / num_heads;
    let probe = candle_core::Tensor::zeros(
        &[batch_size, seq_len, hidden_size],
        candle_core::DType::F32,
        &candle_device,
    )
    .and_then(|tensor| {
        let q = q_linear.forward(&tensor)?;
        let k = k_linear.forward(&tensor)?;
        let v = v_linear.forward(&tensor)?;
        let q = q
            .reshape(&[batch_size, seq_len, num_heads, head_size])?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape(&[batch_size, seq_len, num_heads, head_size])?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape(&[batch_size, seq_len, num_heads, head_size])?
            .transpose(1, 2)?
            .contiguous()?;

        let k_t = k.transpose(2, 3)?.contiguous()?;
        let scores = q.matmul(&k_t)?;
        let scores = (scores / (head_size as f64).sqrt())?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let context = probs.matmul(&v)?;
        let context = context.transpose(1, 2)?.contiguous()?;
        context.flatten_from(2)
    })
    .and_then(|_| candle_device.synchronize());
    if let Err(error) = probe {
        eprintln!("Skipping {name} self_attention {batch_size}x{seq_len}: {error}");
        return;
    }

    group.bench_with_input(
        BenchmarkId::new(name, format!("{batch_size}x{seq_len}")),
        &(batch_size, seq_len),
        move |b, &(batch_size, seq_len)| {
            b.to_async(FuturesExecutor).iter_batched(
                || {
                    let candle_tensor = candle_core::Tensor::from_iter(
                        random_data
                            .iter()
                            .flat_map(|b| b.iter().flat_map(|s| s.iter().copied())),
                        &candle_device,
                    )
                    .unwrap()
                    .reshape(&[batch_size, seq_len, hidden_size])
                    .unwrap();
                    candle_device.synchronize().unwrap();
                    (
                        candle_tensor,
                        q_linear.clone(),
                        k_linear.clone(),
                        v_linear.clone(),
                        candle_device.clone(),
                    )
                },
                |(tensor, q_linear, k_linear, v_linear, candle_device)| async move {
                    let q = q_linear.forward(&tensor).unwrap();
                    let k = k_linear.forward(&tensor).unwrap();
                    let v = v_linear.forward(&tensor).unwrap();

                    let q = q
                        .reshape(&[batch_size, seq_len, num_heads, head_size])
                        .unwrap()
                        .transpose(1, 2)
                        .unwrap()
                        .contiguous()
                        .unwrap();
                    let k = k
                        .reshape(&[batch_size, seq_len, num_heads, head_size])
                        .unwrap()
                        .transpose(1, 2)
                        .unwrap()
                        .contiguous()
                        .unwrap();
                    let v = v
                        .reshape(&[batch_size, seq_len, num_heads, head_size])
                        .unwrap()
                        .transpose(1, 2)
                        .unwrap()
                        .contiguous()
                        .unwrap();

                    let k_t = k.transpose(2, 3).unwrap().contiguous().unwrap();
                    let scores = q.matmul(&k_t).unwrap();
                    let scores = (scores / (head_size as f64).sqrt()).unwrap();
                    let probs = candle_nn::ops::softmax_last_dim(&scores).unwrap();

                    let context = probs.matmul(&v).unwrap();
                    let context = context.transpose(1, 2).unwrap().contiguous().unwrap();
                    let output = context.flatten_from(2).unwrap();

                    candle_device.synchronize().unwrap();
                },
                BatchSize::LargeInput,
            );
        },
    );
}

// Benchmark FFN (Feed-Forward Network) block
fn ffn_block(c: &mut Criterion) {
    let source = FileSource::HuggingFace {
        model_id: "CompendiumLabs/bge-large-en-v1.5-gguf".to_string(),
        revision: "main".to_string(),
        file: "bge-large-en-v1.5-q4_k_m.gguf".to_string(),
    };
    let bytes = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async move {
            let cache = Cache::default();
            let path = cache.get(&source, |_| {}).await.unwrap();
            tokio::fs::read(&path).await.unwrap()
        });

    let mut group = c.benchmark_group("ffn_block");
    let quick = quick_bench();
    group.sample_size(if quick { 10 } else { 20 });
    group.plot_config(
        criterion::PlotConfiguration::default().summary_scale(criterion::AxisScale::Logarithmic),
    );

    let batch_sizes: &[usize] = if quick { &[1] } else { &[1, 32, 512] };
    let seq_lens: &[usize] = if quick { &[13] } else { &[13, 128] };
    for &batch_size in batch_sizes {
        for &seq_len in seq_lens {
            if batch_size * seq_len >= 512 * 128 {
                continue;
            }
            let hidden_size = 1024;

            let random_data: Vec<Vec<Vec<f32>>> = (0..batch_size)
                .map(|_| {
                    (0..seq_len)
                        .map(|_| (0..hidden_size).map(|_| rand::random()).collect())
                        .collect()
                })
                .collect();

            // Fusor FFN benchmark
            {
                let mut reader = std::io::Cursor::new(&bytes);
                let mut var_builder = VarBuilder::from_gguf(&mut reader).unwrap();
                let device = block_on(async { Device::new().await.unwrap() });

                let ffn_up = Linear::load(&device, &mut var_builder.pp("blk.0.ffn_up")).unwrap();
                let ffn_down =
                    Linear::load(&device, &mut var_builder.pp("blk.0.ffn_down")).unwrap();

                let device = device.clone();
                let random_data = random_data.clone();
                group.bench_with_input(
                    BenchmarkId::new("fusor-gpu", format!("{batch_size}x{seq_len}")),
                    &(batch_size, seq_len),
                    move |b, &(batch_size, seq_len)| {
                        let device = device.clone();
                        let random_data = random_data.clone();

                        b.to_async(FuturesExecutor).iter_custom(async |iters| {
                            let tensor: Tensor<3, f32> = Tensor::from_slice(
                                &device,
                                [batch_size, seq_len, hidden_size],
                                &random_data
                                    .iter()
                                    .flat_map(|b| b.iter().flat_map(|s| s.iter().copied()))
                                    .collect::<Vec<_>>(),
                            );
                            tensor.as_gpu().unwrap().materialize().await;
                            let mut sum = Duration::ZERO;
                            while sum.is_zero() {
                                for _ in 0..iters {
                                    let start = std::time::Instant::now();

                                    let intermediate = ffn_up.forward(&tensor);
                                    let intermediate = intermediate.gelu();

                                    let output = ffn_down.forward(&intermediate);

                                    output.as_gpu().unwrap().materialize().await;
                                    sum += start.elapsed();
                                }
                            }
                            sum
                        });
                    },
                );
            }

            // Candle FFN benchmark
            let ffn_shape = BertShape {
                batch_size,
                seq_len,
                hidden_size,
            };
            if let Some(candle_device) = candle_gpu_device() {
                bench_candle_ffn(
                    &bytes,
                    ffn_shape,
                    random_data.clone(),
                    candle_device,
                    "candle-gpu",
                    &mut group,
                );
            }

            {
                let candle_device = candle_core::Device::Cpu;
                bench_candle_ffn(
                    &bytes,
                    ffn_shape,
                    random_data.clone(),
                    candle_device,
                    "candle-cpu",
                    &mut group,
                );
            }
        }
    }
    group.finish();
}

fn bench_candle_ffn(
    bytes: &[u8],
    shape: BertShape,
    random_data: Vec<Vec<Vec<f32>>>,
    candle_device: candle_core::Device,
    name: &str,
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    let BertShape {
        batch_size,
        seq_len,
        hidden_size,
    } = shape;
    use candle_transformers::quantized_nn::Linear;

    let var_builder = candle_transformers::quantized_var_builder::VarBuilder::from_gguf_buffer(
        bytes,
        &candle_device,
    )
    .unwrap();

    let up_weight = var_builder
        .pp("blk.0.ffn_up")
        .get_no_shape("weight")
        .unwrap();
    let up_bias = var_builder
        .pp("blk.0.ffn_up")
        .get_no_shape("bias")
        .unwrap()
        .dequantize(&candle_device)
        .unwrap();
    let down_weight = var_builder
        .pp("blk.0.ffn_down")
        .get_no_shape("weight")
        .unwrap();
    let down_bias = var_builder
        .pp("blk.0.ffn_down")
        .get_no_shape("bias")
        .unwrap()
        .dequantize(&candle_device)
        .unwrap();

    let up_linear = Linear::from_arc(up_weight, Some(up_bias)).unwrap();
    let down_linear = Linear::from_arc(down_weight, Some(down_bias)).unwrap();
    let probe = candle_core::Tensor::zeros(
        &[batch_size, seq_len, hidden_size],
        candle_core::DType::F32,
        &candle_device,
    )
    .and_then(|tensor| {
        let intermediate = up_linear.forward(&tensor)?;
        let intermediate = intermediate.gelu()?;
        down_linear.forward(&intermediate)
    })
    .and_then(|_| candle_device.synchronize());
    if let Err(error) = probe {
        eprintln!("Skipping {name} ffn_block {batch_size}x{seq_len}: {error}");
        return;
    }

    group.bench_with_input(
        BenchmarkId::new(name, format!("{batch_size}x{seq_len}")),
        &(batch_size, seq_len),
        move |b, &(batch_size, seq_len)| {
            b.to_async(FuturesExecutor).iter_batched(
                || {
                    let candle_tensor = candle_core::Tensor::from_iter(
                        random_data
                            .iter()
                            .flat_map(|b| b.iter().flat_map(|s| s.iter().copied())),
                        &candle_device,
                    )
                    .unwrap()
                    .reshape(&[batch_size, seq_len, hidden_size])
                    .unwrap();
                    candle_device.synchronize().unwrap();
                    (
                        candle_tensor,
                        up_linear.clone(),
                        down_linear.clone(),
                        candle_device.clone(),
                    )
                },
                |(tensor, up_linear, down_linear, candle_device)| async move {
                    let intermediate = up_linear.forward(&tensor).unwrap();
                    let intermediate = intermediate.gelu().unwrap();

                    let output = down_linear.forward(&intermediate).unwrap();

                    candle_device.synchronize().unwrap();
                },
                BatchSize::LargeInput,
            );
        },
    );
}

criterion_group!(benches, layer_norm, self_attention, ffn_block);
criterion_main!(benches);
