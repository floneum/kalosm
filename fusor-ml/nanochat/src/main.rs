mod model;

use fusor::{Device, Tensor, ToVec3};
use fusor_gguf::{
    BlockQ4_0, BlockQ8_0, GgmlType, GgufMetadata, GgufTensorMetadata, GgufValue, GgufVersion,
};
use fusor_nanochat::data::{
    CanvasStateIndexes, CanvasStateSpec, DatasetSplit, SourceDataset, SourceFile, StrokeTokenizer,
    autoregressive_context, canvas_state_indexes, load_dataset_source, position_indexes,
    token_component_indexes, windows_to_token_inputs, windows_to_token_targets,
    write_tokens_to_svg_file,
};
use fusor_nanochat::{
    ComparisonReport, LivePredictor, RuntimeConfig as SharedRuntimeConfig,
    SaveQuantization as SharedSaveQuantization,
    build_comparison_report as build_shared_comparison_report,
    generate_sample as generate_shared_sample,
};
use fusor_train::{AdamW, AdamWSettings};
use half::f16;
use model::{NamedTensor, NanoChatModel};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::{
    env,
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use fusor_nanochat::{RuntimeConfig, SaveQuantization};

#[tokio::main]
async fn main() {
    {
        let args = Args::parse();
        let runtime = RuntimeConfig::load();

        if args.infer {
            run_inference(&runtime, args.force_cpu);
            return;
        }
        if args.compare {
            run_comparison_report(&runtime, args.force_cpu);
            return;
        }

        let dataset_source = load_dataset_source(&runtime);
        let dataset_label = dataset_source.label.clone();
        let tokenizer = dataset_source.tokenizer.clone();

        let device = if args.force_cpu {
            Device::cpu()
        } else {
            Device::gpu().await.unwrap()
        };

        let DatasetSplit {
            train: train_dataset,
            validation: validation_dataset,
            test: test_dataset,
        } = dataset_source.split;
        assert!(train_dataset.num_docs() > 0, "training split is empty");

        let max_tokens_per_example = train_dataset
            .max_tokens_per_example()
            .max(validation_dataset.max_tokens_per_example())
            .max(test_dataset.max_tokens_per_example());
        assert!(
            max_tokens_per_example <= runtime.block_size + 1,
            "block size {} is too small for the current dataset; need at least {}",
            runtime.block_size,
            max_tokens_per_example.saturating_sub(1)
        );

        let evaluation_dataset =
            preferred_eval_dataset(&validation_dataset, &test_dataset, &train_dataset);
        let mut rng = StdRng::seed_from_u64(runtime.seed);
        let mut model = NanoChatModel::new(&device, &mut rng, &tokenizer, &runtime);
        let mut optimizer = AdamW::new(
            &device,
            &model,
            AdamWSettings::new(
                runtime.learning_rate,
                runtime.beta1,
                runtime.beta2,
                runtime.adam_eps,
                runtime.weight_decay,
            ),
        );

        println!(
            "task={} train_examples={} valid_examples={} test_examples={} train_tokens={} valid_tokens={} test_tokens={} train_windows={} valid_windows={} vocab={} params={} max_file_tokens={}",
            dataset_label,
            train_dataset.num_docs(),
            validation_dataset.num_docs(),
            test_dataset.num_docs(),
            train_dataset.num_tokens(),
            validation_dataset.num_tokens(),
            test_dataset.num_tokens(),
            train_dataset.num_training_windows(runtime.block_size),
            validation_dataset.num_training_windows(runtime.block_size),
            tokenizer.vocab_size(),
            model.num_parameters(),
            max_tokens_per_example,
        );

        let steps_per_epoch = train_dataset.steps_per_epoch(runtime.block_size, runtime.batch_size);
        let total_steps = runtime.epochs * steps_per_epoch;
        println!(
            "epochs={} steps_per_epoch={} total_steps={}",
            runtime.epochs, steps_per_epoch, total_steps
        );

        let mut global_step: usize = 0;
        for epoch in 0..runtime.epochs {
            let batches = train_dataset.epoch_batches(&mut rng, tokenizer.eot_token(), &runtime);
            let epoch_batch_count = batches.len().max(1);
            for (batch_index, batch) in batches.into_iter().enumerate() {
                global_step += 1;
                let learning_rate = scheduled_learning_rate(global_step, total_steps, &runtime);
                optimizer.set_learning_rate(learning_rate);

                let input_values = windows_to_token_inputs(&batch.windows);
                let target_values = windows_to_token_targets(&batch.windows);
                let components = token_component_indexes(&tokenizer, &input_values);
                let mode_inputs: Tensor<2, u32> = Tensor::new(&device, &components.mode);
                let direction_inputs: Tensor<2, u32> = Tensor::new(&device, &components.direction);
                let count_inputs: Tensor<2, u32> = Tensor::new(&device, &components.count);
                let (cursor_x_inputs, cursor_y_inputs, pen_state_inputs) = canvas_state_tensors(
                    &device,
                    &tokenizer,
                    &input_values,
                    model.canvas_state_spec(),
                );
                let position_values = position_indexes(batch.windows.len().max(1), batch.seq_len);
                let position_inputs: Tensor<2, u32> = Tensor::new(&device, &position_values);
                let causal_mask = causal_mask_tensor(
                    model.graph(),
                    &device,
                    batch.windows.len().max(1),
                    batch.seq_len,
                );
                let logits = model.forward(
                    &mode_inputs,
                    &direction_inputs,
                    &count_inputs,
                    &position_inputs,
                    &cursor_x_inputs,
                    &cursor_y_inputs,
                    &pen_state_inputs,
                    &causal_mask,
                );
                let loss = structured_action_loss_autograd(
                    model.graph(),
                    &device,
                    &tokenizer,
                    &logits,
                    &target_values,
                    &batch.masks,
                    batch.valid_tokens,
                );

                let is_last = global_step == total_steps;
                let should_log = global_step % runtime.log_every == 0 || is_last;
                // Always materialize the loss before backward(). On GPU, this
                // forces the forward-pass compute graph to resolve, caching all
                // intermediate buffers. The backward pass then only adds its own
                // operations on top of already-resolved tensors, keeping the
                // compute graph small enough for the GPU to handle.
                let loss_value = loss.raw().to_scalar().await.unwrap();
                let gradients = loss.backward().unwrap();
                drop(loss);
                drop(logits);
                drop(causal_mask);
                model = optimizer.step(model, gradients);

                if should_log {
                    let validation_metrics = evaluate_autoregressive_metrics(
                        &model,
                        &device,
                        &tokenizer,
                        tokenizer.eot_token(),
                        evaluation_dataset,
                        &runtime,
                    )
                    .await;
                    println!(
                        "epoch {}/{} batch {:>3}/{} | global {:>6}/{} | lr={learning_rate:.6} | train_loss={loss_value:.6} | eval_loss={:.6} | eval_joint_action_acc={:.2}%",
                        epoch + 1,
                        runtime.epochs,
                        batch_index + 1,
                        epoch_batch_count,
                        global_step,
                        total_steps,
                        validation_metrics.loss,
                        validation_metrics.joint_action_acc * 100.0,
                    );
                }

                if runtime.save_every_steps > 0 && global_step % runtime.save_every_steps == 0 {
                    let checkpoint_path = checkpoint_path(&runtime.gguf_path, global_step);
                    let checkpoint_path = resolve_output_path(&checkpoint_path);
                    save_gguf(&model, &tokenizer, &runtime, &checkpoint_path).await;
                    println!("saved checkpoint: {}", checkpoint_path.display());
                }
            }
        }

        if runtime.save_final_model {
            let gguf_path = resolve_output_path(&runtime.gguf_path);
            save_gguf(&model, &tokenizer, &runtime, &gguf_path).await;
            println!("saved gguf: {}", gguf_path.display());
        } else {
            println!("skipped final gguf save");
        }

        drop(optimizer);

        let (metrics, sample_generated) = if runtime.save_final_model {
            drop(model);
            let predictor = LivePredictor::load(to_shared_runtime_config(&runtime), args.force_cpu)
                .unwrap_or_else(|error| panic!("could not load live predictor: {error}"));
            let metrics = evaluate_interactive_continuation(
                &predictor,
                &tokenizer,
                evaluation_dataset,
                &runtime,
            )
            .await;
            let sample_generated = if let Some(file) = evaluation_dataset.files().first() {
                let prompt_tokens = continuation_prompt_tokens(&tokenizer, file, &runtime);
                Some(
                    predictor
                        .predict_greedy(&prompt_tokens[1..], runtime.sample_tokens, None)
                        .unwrap_or_else(|error| {
                            panic!(
                                "could not generate continuation sample from checkpoint: {error}"
                            )
                        }),
                )
            } else {
                None
            };
            (metrics, sample_generated)
        } else {
            let metrics =
                evaluate_continuation(&model, &device, &tokenizer, evaluation_dataset, &runtime)
                    .await;
            let sample_generated = if let Some(file) = evaluation_dataset.files().first() {
                let prompt_tokens = continuation_prompt_tokens(&tokenizer, file, &runtime);
                Some(
                    generate_completion(
                        &model,
                        &device,
                        &tokenizer,
                        &runtime,
                        &prompt_tokens,
                        tokenizer.eot_token(),
                        &mut rng,
                        SamplingMode::Sample,
                    )
                    .await,
                )
            } else {
                None
            };
            (metrics, sample_generated)
        };

        println!(
            "\ncontinuation token accuracy: {}/{} ({:.2}%)",
            metrics.correct_tokens,
            metrics.total_tokens,
            metrics.token_accuracy() * 100.0,
        );
        println!(
            "continuation exact match: {}/{} ({:.2}%)",
            metrics.exact_matches,
            metrics.exact_match_examples,
            metrics.exact_match_rate() * 100.0,
        );

        if let Some(file) = evaluation_dataset.files().first() {
            let prompt_tokens = continuation_prompt_tokens(&tokenizer, file, &runtime);
            let generated = sample_generated.unwrap_or_default();
            let expected = expected_continuation_tokens(
                file,
                prompt_tokens.len().saturating_sub(1),
                runtime.sample_tokens,
            );
            let sample_output_path = resolve_output_path(&runtime.sample_output_path);
            write_tokens_to_svg_file(
                &tokenizer,
                &prompt_tokens[1..],
                &generated,
                &sample_output_path,
            );
            println!("\n--- sample export ---");
            println!("source: {}", file.path());
            println!(
                "prompt: {}",
                tokenizer.describe_tokens(&prompt_tokens[1..], 48)
            );
            println!("expected: {}", tokenizer.describe_tokens(expected, 48));
            println!("generated: {}", tokenizer.describe_tokens(&generated, 48));
            println!("wrote sample SVG: {}", sample_output_path.display());
        }
    }
}

struct Args {
    force_cpu: bool,
    infer: bool,
    compare: bool,
}

impl Args {
    fn parse() -> Self {
        let mut force_cpu = false;
        let mut infer = false;
        let mut compare = false;
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--cpu" => force_cpu = true,
                "--infer" => infer = true,
                "--compare" => compare = true,
                other => panic!("unknown argument: {other}"),
            }
        }

        Self {
            force_cpu,
            infer,
            compare,
        }
    }
}

async fn save_gguf(
    model: &NanoChatModel,
    tokenizer: &StrokeTokenizer,
    runtime: &RuntimeConfig,
    path: &Path,
) {
    let save_start = Instant::now();
    println!(
        "saving gguf to {} ({})...",
        path.display(),
        runtime.save_quantization.as_str()
    );
    println!("materializing model weights for save...");
    let materialize_start = Instant::now();
    let named_tensors = model.named_tensors().await;
    let materialize_elapsed = materialize_start.elapsed();
    let materialized_bytes = named_tensors
        .iter()
        .map(|tensor| tensor.values.len() * std::mem::size_of::<f32>())
        .sum::<usize>();
    println!(
        "materialized {} tensors in {:.2?} ({:.2} MiB of f32 weights)",
        named_tensors.len(),
        materialize_elapsed,
        materialized_bytes as f64 / (1024.0 * 1024.0),
    );

    println!("serializing tensors for gguf...");
    let serialize_start = Instant::now();
    let mut serialized_tensors = Vec::with_capacity(named_tensors.len());
    let mut serialized_bytes = 0usize;
    for tensor in named_tensors {
        let tensor_start = Instant::now();
        let input_bytes = tensor.values.len() * std::mem::size_of::<f32>();
        let name = tensor.name.clone();
        let shape = tensor.shape.clone();
        let serialized = serialize_tensor(tensor, runtime.save_quantization);
        serialized_bytes += serialized.data.len();
        println!(
            "serialized tensor {name} shape={shape:?} input={:.2} MiB output={:.2} MiB ty={} in {:.2?}",
            input_bytes as f64 / (1024.0 * 1024.0),
            serialized.data.len() as f64 / (1024.0 * 1024.0),
            serialized.ty,
            tensor_start.elapsed(),
        );
        serialized_tensors.push(serialized);
    }
    println!(
        "prepared {} tensors for gguf write in {:.2?} ({:.2} MiB serialized)",
        serialized_tensors.len(),
        serialize_start.elapsed(),
        serialized_bytes as f64 / (1024.0 * 1024.0),
    );

    let tensor_info_start = Instant::now();
    let mut offset = 0u64;
    let tensor_infos = serialized_tensors
        .iter()
        .map(|tensor| {
            let metadata = GgufTensorMetadata {
                ty: tensor.ty,
                shape: tensor.shape.clone().into_boxed_slice(),
                offset,
            };
            offset += tensor.data.len() as u64;
            (tensor.name.clone().into_boxed_str(), metadata)
        })
        .collect();
    println!(
        "built tensor metadata in {:.2?}",
        tensor_info_start.elapsed()
    );

    let metadata_start = Instant::now();
    let mut metadata = vec![
        (
            String::from("general.architecture").into_boxed_str(),
            GgufValue::String("nanochat".into()),
        ),
        (
            String::from("general.name").into_boxed_str(),
            GgufValue::String("fusor-nanochat-strokes".into()),
        ),
        (
            String::from("general.alignment").into_boxed_str(),
            GgufValue::U32(1),
        ),
        (
            String::from("nanochat.block_size").into_boxed_str(),
            GgufValue::U32(model.block_size() as u32),
        ),
        (
            String::from("nanochat.embedding_length").into_boxed_str(),
            GgufValue::U32(model.n_embd() as u32),
        ),
        (
            String::from("nanochat.feed_forward_length").into_boxed_str(),
            GgufValue::U32(model.n_ff() as u32),
        ),
        (
            String::from("nanochat.conv_kernel_size").into_boxed_str(),
            GgufValue::U32(model.conv_kernel_size() as u32),
        ),
        (
            String::from("nanochat.head_count").into_boxed_str(),
            GgufValue::U32(model.n_head() as u32),
        ),
        (
            String::from("nanochat.kv_head_count").into_boxed_str(),
            GgufValue::U32(model.n_kv_head() as u32),
        ),
        (
            String::from("nanochat.block_count").into_boxed_str(),
            GgufValue::U32(model.n_layer() as u32),
        ),
        (
            String::from("nanochat.attention_period").into_boxed_str(),
            GgufValue::U32(model.attention_period() as u32),
        ),
        (
            String::from("nanochat.vocab_size").into_boxed_str(),
            GgufValue::U32(model.vocab_size() as u32),
        ),
        (
            String::from("nanochat.use_rope").into_boxed_str(),
            GgufValue::Bool(model.use_rope()),
        ),
        (
            String::from("nanochat.rope_theta").into_boxed_str(),
            GgufValue::F32(model.rope_theta()),
        ),
        (
            String::from("nanochat.use_extra_norms").into_boxed_str(),
            GgufValue::Bool(model.use_extra_norms()),
        ),
        (
            String::from("nanochat.use_canvas_state_embeddings").into_boxed_str(),
            GgufValue::Bool(model.canvas_state_spec().is_some()),
        ),
        (
            String::from("nanochat.epochs").into_boxed_str(),
            GgufValue::U32(runtime.epochs.min(u32::MAX as usize) as u32),
        ),
        (
            String::from("nanochat.learning_rate").into_boxed_str(),
            GgufValue::F32(runtime.learning_rate),
        ),
        (
            String::from("nanochat.batch_size").into_boxed_str(),
            GgufValue::U32(runtime.batch_size as u32),
        ),
        (
            String::from("nanochat.training_objective").into_boxed_str(),
            GgufValue::String("autoregressive-stroke-autocomplete".into()),
        ),
        (
            String::from("nanochat.eps").into_boxed_str(),
            GgufValue::F32(model.eps()),
        ),
        (
            String::from("nanochat.save_quantization").into_boxed_str(),
            GgufValue::String(runtime.save_quantization.as_str().into()),
        ),
    ];
    if let Some(spec) = model.canvas_state_spec() {
        metadata.push((
            String::from("nanochat.canvas_coordinate_vocab").into_boxed_str(),
            GgufValue::U32(spec.coordinate_vocab_size as u32),
        ));
        metadata.push((
            String::from("nanochat.canvas_coordinate_offset").into_boxed_str(),
            GgufValue::U32(spec.coordinate_offset.max(0) as u32),
        ));
    }
    metadata.extend(
        tokenizer
            .gguf_metadata()
            .into_iter()
            .map(|(key, value)| (key.into_boxed_str(), value)),
    );
    println!(
        "built gguf metadata in {:.2?} ({} metadata entries)",
        metadata_start.elapsed(),
        metadata.len(),
    );

    let gguf_build_start = Instant::now();
    let gguf = GgufMetadata {
        version: GgufVersion::V3,
        metadata: metadata.into_iter().collect(),
        tensor_infos,
        tensor_data_offset: 0,
    };
    println!(
        "assembled gguf structure in {:.2?}",
        gguf_build_start.elapsed()
    );

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }

    let writer = File::create(path).unwrap();
    let mut writer = BufWriter::new(writer);
    println!("writing gguf bytes...");
    gguf.write(
        &mut writer,
        serialized_tensors
            .iter()
            .map(|tensor| (tensor.name.as_str(), tensor.data.as_slice())),
    )
    .unwrap();
    writer.flush().unwrap();
    println!("total save time: {:.2?}", save_start.elapsed());
}

struct SerializedTensor {
    name: String,
    shape: Vec<u32>,
    ty: GgmlType,
    data: Vec<u8>,
}

fn serialize_tensor(
    tensor: NamedTensor,
    preferred_quantization: SaveQuantization,
) -> SerializedTensor {
    let (ty, data) = match preferred_quantization {
        SaveQuantization::F32 => (
            GgmlType::F32,
            bytemuck::cast_slice::<f32, u8>(&tensor.values).to_vec(),
        ),
        SaveQuantization::F16 => (GgmlType::F16, f16_bytes(&tensor.values)),
        SaveQuantization::Q4_0 => serialize_q4_0_or_f16(&tensor.shape, &tensor.values),
        SaveQuantization::Q8_0 => serialize_q8_0_or_f16(&tensor.shape, &tensor.values),
    };

    SerializedTensor {
        name: tensor.name,
        shape: tensor.shape,
        ty,
        data,
    }
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f16>());
    for value in values {
        bytes.extend_from_slice(&f16::from_f32(*value).to_bits().to_le_bytes());
    }
    bytes
}

fn serialize_q8_0_or_f16(shape: &[u32], values: &[f32]) -> (GgmlType, Vec<u8>) {
    let Some(&inner_dim) = shape.last() else {
        return (GgmlType::F16, f16_bytes(values));
    };

    if inner_dim as usize % BlockQ8_0::BLOCK_SIZE != 0
        || !values.len().is_multiple_of(BlockQ8_0::BLOCK_SIZE)
    {
        return (GgmlType::F16, f16_bytes(values));
    }

    let mut bytes =
        Vec::with_capacity(values.len() / BlockQ8_0::BLOCK_SIZE * std::mem::size_of::<BlockQ8_0>());
    for chunk in values.chunks_exact(BlockQ8_0::BLOCK_SIZE) {
        let block = BlockQ8_0::quantize(chunk.try_into().unwrap());
        bytes.extend_from_slice(bytemuck::bytes_of(&block));
    }
    (GgmlType::Q8_0, bytes)
}

fn serialize_q4_0_or_f16(shape: &[u32], values: &[f32]) -> (GgmlType, Vec<u8>) {
    let Some(&inner_dim) = shape.last() else {
        return (GgmlType::F16, f16_bytes(values));
    };

    if inner_dim as usize % BlockQ4_0::BLOCK_SIZE != 0
        || !values.len().is_multiple_of(BlockQ4_0::BLOCK_SIZE)
    {
        return (GgmlType::F16, f16_bytes(values));
    }

    let mut bytes =
        Vec::with_capacity(values.len() / BlockQ4_0::BLOCK_SIZE * std::mem::size_of::<BlockQ4_0>());
    for chunk in values.chunks_exact(BlockQ4_0::BLOCK_SIZE) {
        let block = BlockQ4_0::quantize(chunk.try_into().unwrap());
        bytes.extend_from_slice(bytemuck::bytes_of(&block));
    }
    (GgmlType::Q4_0, bytes)
}

fn checkpoint_path(base_path: &Path, step: usize) -> PathBuf {
    let parent = base_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let stem = base_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("nanochat");
    let extension = base_path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("gguf");
    parent.join(format!("{stem}-step-{step:05}.{extension}"))
}

fn resolve_output_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
    }
}

fn canvas_state_tensors(
    device: &Device,
    tokenizer: &StrokeTokenizer,
    token_windows: &[Vec<u32>],
    spec: Option<CanvasStateSpec>,
) -> (Tensor<2, u32>, Tensor<2, u32>, Tensor<2, u32>) {
    let seq_len = token_windows.first().map_or(0, Vec::len);
    let indexes = spec
        .map(|spec| canvas_state_indexes(tokenizer, token_windows, spec))
        .unwrap_or_else(|| CanvasStateIndexes::zeros(token_windows.len(), seq_len));
    (
        Tensor::new(device, &indexes.cursor_x),
        Tensor::new(device, &indexes.cursor_y),
        Tensor::new(device, &indexes.pen_state),
    )
}

async fn generate_completion(
    model: &NanoChatModel,
    device: &Device,
    tokenizer: &StrokeTokenizer,
    runtime: &RuntimeConfig,
    prompt_tokens: &[u32],
    stop_token: u32,
    rng: &mut StdRng,
    mode: SamplingMode,
) -> Vec<u32> {
    let mut tokens = prompt_tokens.to_vec();

    for _ in 0..runtime.sample_tokens {
        let (context, last_index) = autoregressive_context(&tokens, stop_token, runtime.block_size);
        let components = token_component_indexes(tokenizer, std::slice::from_ref(&context));
        let mode_inputs: Tensor<2, u32> = Tensor::new(device, &components.mode);
        let direction_inputs: Tensor<2, u32> = Tensor::new(device, &components.direction);
        let count_inputs: Tensor<2, u32> = Tensor::new(device, &components.count);
        let position_values = position_indexes(1, context.len().max(1));
        let position_inputs: Tensor<2, u32> = Tensor::new(device, &position_values);
        let causal_mask = causal_mask_tensor(model.graph(), device, 1, context.len().max(1));
        let (cursor_x_inputs, cursor_y_inputs, pen_state_inputs) = canvas_state_tensors(
            device,
            tokenizer,
            std::slice::from_ref(&context),
            model.canvas_state_spec(),
        );
        let logits = model.forward(
            &mode_inputs,
            &direction_inputs,
            &count_inputs,
            &position_inputs,
            &cursor_x_inputs,
            &cursor_y_inputs,
            &pen_state_inputs,
            &causal_mask,
        );
        let mode_logits: Vec<Vec<Vec<f32>>> = logits
            .mode
            .raw()
            .clone()
            .as_slice()
            .await
            .unwrap()
            .to_vec3();
        let direction_logits: Vec<Vec<Vec<f32>>> = logits
            .direction
            .raw()
            .clone()
            .as_slice()
            .await
            .unwrap()
            .to_vec3();
        let count_logits: Vec<Vec<Vec<f32>>> = logits
            .count
            .raw()
            .clone()
            .as_slice()
            .await
            .unwrap()
            .to_vec3();
        let next = decode_next_action_token(
            tokenizer,
            &context[..=last_index],
            &mode_logits[0][last_index],
            &direction_logits[0][last_index],
            &count_logits[0][last_index],
            runtime.sample_temperature,
            runtime.sample_top_k,
            rng,
            mode,
        );
        tokens.push(next);
        if next == stop_token {
            break;
        }
    }

    tokens[prompt_tokens.len()..]
        .iter()
        .copied()
        .take_while(|&token| token != stop_token)
        .collect()
}

fn argmax_from_logits(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index as u32)
        .unwrap()
}

fn sample_from_logits(rng: &mut StdRng, logits: &[f32], temperature: f32, top_k: usize) -> u32 {
    if temperature <= 0.0 || top_k <= 1 {
        return argmax_from_logits(logits);
    }

    let mut candidates = logits.iter().copied().enumerate().collect::<Vec<_>>();
    candidates.sort_unstable_by(|(_, left), (_, right)| right.total_cmp(left));
    candidates.truncate(top_k.min(candidates.len()).max(1));

    let max_logit = candidates[0].1;
    let mut weighted = Vec::with_capacity(candidates.len());
    let mut total = 0.0f32;
    for (index, logit) in candidates {
        let weight = ((logit - max_logit) / temperature).exp();
        total += weight;
        weighted.push((index as u32, weight));
    }

    let mut threshold = rng.random::<f32>() * total.max(f32::MIN_POSITIVE);
    for (index, weight) in weighted {
        threshold -= weight;
        if threshold <= 0.0 {
            return index;
        }
    }

    argmax_from_logits(logits)
}

fn decode_next_action_token(
    tokenizer: &StrokeTokenizer,
    context: &[u32],
    mode_logits: &[f32],
    direction_logits: &[f32],
    count_logits: &[f32],
    temperature: f32,
    top_k: usize,
    rng: &mut StdRng,
    sampling: SamplingMode,
) -> u32 {
    let mode_index = match sampling {
        SamplingMode::Greedy => argmax_from_logits(mode_logits),
        SamplingMode::Sample => sample_from_logits(rng, mode_logits, temperature, top_k),
    };
    if mode_index >= 2 {
        return tokenizer.eot_token();
    }

    let cursor = tokenizer.cursor_after_tokens(context);
    let mut filtered_direction_logits = direction_logits.to_vec();
    for (direction_index, logit) in filtered_direction_logits.iter_mut().enumerate() {
        if tokenizer.legal_count_limit(cursor, direction_index as u32) == 0 {
            *logit = f32::NEG_INFINITY;
        }
    }
    if !filtered_direction_logits
        .iter()
        .any(|logit| logit.is_finite())
    {
        return tokenizer.eot_token();
    }

    let direction_index = match sampling {
        SamplingMode::Greedy => argmax_from_logits(&filtered_direction_logits),
        SamplingMode::Sample => {
            sample_from_logits(rng, &filtered_direction_logits, temperature, top_k)
        }
    };
    let legal_limit = tokenizer.legal_count_limit(cursor, direction_index);
    if legal_limit == 0 {
        return tokenizer.eot_token();
    }

    let mut filtered_count_logits = count_logits.to_vec();
    for (count_index, logit) in filtered_count_logits.iter_mut().enumerate() {
        let count = count_index + 1;
        if count > legal_limit {
            *logit = f32::NEG_INFINITY;
        }
    }
    let count_index = match sampling {
        SamplingMode::Greedy => argmax_from_logits(&filtered_count_logits),
        SamplingMode::Sample => sample_from_logits(rng, &filtered_count_logits, temperature, top_k),
    } as usize;
    let count = (count_index + 1).clamp(1, legal_limit.max(1));

    tokenizer.token_from_components(mode_index, direction_index, count)
}

fn causal_mask_tensor(
    graph: &fusor::autograd::Graph,
    device: &Device,
    batch_size: usize,
    block_size: usize,
) -> fusor::autograd::Tensor<3> {
    let mut causal_mask_values = vec![vec![vec![0.0f32; block_size]; block_size]; batch_size];
    for batch in &mut causal_mask_values {
        for i in 0..block_size {
            for j in (i + 1)..block_size {
                batch[i][j] = f32::NEG_INFINITY;
            }
        }
    }
    fusor::autograd::Tensor::constant_from_raw(
        graph,
        fusor::Tensor::new(device, &causal_mask_values),
    )
}

fn masked_cross_entropy_autograd(
    graph: &fusor::autograd::Graph,
    device: &fusor::Device,
    logits: &fusor::autograd::Tensor<3>,
    targets: &[Vec<u32>],
    mask: &[Vec<f32>],
    valid_tokens: f32,
    vocab_size: usize,
) -> fusor::autograd::Tensor<0> {
    let batch_size = logits.shape()[0];
    let block_size = logits.shape()[1];

    let mut flat_targets = Vec::with_capacity(batch_size * block_size);
    for batch in 0..batch_size {
        for position in 0..block_size {
            flat_targets.push(targets[batch][position]);
        }
    }

    let target_indices =
        fusor::Tensor::from_slice(device, [batch_size * block_size], &flat_targets);
    let mask = fusor::autograd::Tensor::constant_from_raw(graph, fusor::Tensor::new(device, mask));

    let log_norm = logits.exp().sum_keepdim(2).log();
    let log_probs = logits.sub(&log_norm.broadcast_as([batch_size, block_size, vocab_size]));
    let flat_log_probs = log_probs.reshape([batch_size * block_size, vocab_size]);
    let selected = flat_log_probs
        .gather_last(&target_indices)
        .reshape([batch_size, block_size]);
    let token_nll = selected.neg().mul(&mask);
    token_nll.sum(1).sum().div_scalar(valid_tokens.max(1.0))
}

fn structured_action_loss_autograd(
    graph: &fusor::autograd::Graph,
    device: &fusor::Device,
    tokenizer: &StrokeTokenizer,
    logits: &model::ActionLogits,
    targets: &[Vec<u32>],
    mask: &[Vec<f32>],
    valid_tokens: f32,
) -> fusor::autograd::Tensor<0> {
    let batch_size = targets.len();
    let block_size = targets.first().map_or(0, Vec::len);
    let mut mode_targets = vec![vec![0u32; block_size]; batch_size];
    let mut direction_targets = vec![vec![0u32; block_size]; batch_size];
    let mut count_targets = vec![vec![0u32; block_size]; batch_size];
    let mut direction_mask = vec![vec![0.0f32; block_size]; batch_size];
    let mut count_mask = vec![vec![0.0f32; block_size]; batch_size];
    let mut action_tokens = 0.0f32;

    for batch in 0..batch_size {
        for position in 0..block_size {
            let decoded = tokenizer.decode_training_target(targets[batch][position]);
            let valid = mask[batch][position];
            mode_targets[batch][position] = decoded.mode_index;
            if let (Some(direction_index), Some(count)) = (decoded.direction_index, decoded.count) {
                direction_targets[batch][position] = direction_index;
                count_targets[batch][position] = (count - 1) as u32;
                direction_mask[batch][position] = valid;
                count_mask[batch][position] = valid;
                action_tokens += valid;
            }
        }
    }

    let mode_loss = masked_cross_entropy_autograd(
        graph,
        device,
        &logits.mode,
        &mode_targets,
        mask,
        valid_tokens,
        3,
    );
    let direction_loss = masked_cross_entropy_autograd(
        graph,
        device,
        &logits.direction,
        &direction_targets,
        &direction_mask,
        action_tokens,
        tokenizer.direction_count(),
    );
    let count_loss = masked_cross_entropy_autograd(
        graph,
        device,
        &logits.count,
        &count_targets,
        &count_mask,
        action_tokens,
        tokenizer.max_count(),
    );

    mode_loss.add(&direction_loss).add(&count_loss)
}

struct EvalMetrics {
    loss: f32,
    joint_action_acc: f32,
}

async fn evaluate_autoregressive_metrics(
    model: &NanoChatModel,
    device: &Device,
    tokenizer: &StrokeTokenizer,
    pad_token: u32,
    dataset: &SourceDataset,
    runtime: &RuntimeConfig,
) -> EvalMetrics {
    let batches = dataset.evaluation_batches(pad_token, runtime);
    if batches.is_empty() {
        return EvalMetrics {
            loss: 0.0,
            joint_action_acc: 0.0,
        };
    }

    let mut total_nats = 0.0;
    let mut total_valid_tokens = 0.0;
    let mut joint_action_correct = 0usize;
    let mut joint_action_total = 0usize;
    for batch in batches {
        let batch_size = batch.windows.len();
        let batch_causal_mask =
            causal_mask_tensor(model.graph(), device, batch_size, batch.seq_len);
        let position_values = position_indexes(batch_size, batch.seq_len);
        let position_inputs: Tensor<2, u32> = Tensor::new(device, &position_values);
        let input_values = windows_to_token_inputs(&batch.windows);
        let target_values = windows_to_token_targets(&batch.windows);
        let components = token_component_indexes(tokenizer, &input_values);
        let mode_inputs: Tensor<2, u32> = Tensor::new(device, &components.mode);
        let direction_inputs: Tensor<2, u32> = Tensor::new(device, &components.direction);
        let count_inputs: Tensor<2, u32> = Tensor::new(device, &components.count);
        let (cursor_x_inputs, cursor_y_inputs, pen_state_inputs) =
            canvas_state_tensors(device, tokenizer, &input_values, model.canvas_state_spec());
        let logits = model.forward(
            &mode_inputs,
            &direction_inputs,
            &count_inputs,
            &position_inputs,
            &cursor_x_inputs,
            &cursor_y_inputs,
            &pen_state_inputs,
            &batch_causal_mask,
        );
        let loss = structured_action_loss_autograd(
            model.graph(),
            device,
            tokenizer,
            &logits,
            &target_values,
            &batch.masks,
            batch.valid_tokens,
        );
        total_nats += loss.raw().to_scalar().await.unwrap() * batch.valid_tokens.max(1.0);
        total_valid_tokens += batch.valid_tokens;

        let mode_logits: Vec<Vec<Vec<f32>>> = logits
            .mode
            .raw()
            .clone()
            .as_slice()
            .await
            .unwrap()
            .to_vec3();
        let direction_logits: Vec<Vec<Vec<f32>>> = logits
            .direction
            .raw()
            .clone()
            .as_slice()
            .await
            .unwrap()
            .to_vec3();
        let count_logits: Vec<Vec<Vec<f32>>> = logits
            .count
            .raw()
            .clone()
            .as_slice()
            .await
            .unwrap()
            .to_vec3();
        let mut greedy_rng = StdRng::seed_from_u64(0);
        for batch_index in 0..batch_size {
            for position in 0..batch.seq_len {
                if batch.masks[batch_index][position] <= 0.0 {
                    continue;
                }
                joint_action_total += 1;
                let predicted = decode_next_action_token(
                    tokenizer,
                    &input_values[batch_index][..=position],
                    &mode_logits[batch_index][position],
                    &direction_logits[batch_index][position],
                    &count_logits[batch_index][position],
                    0.0,
                    1,
                    &mut greedy_rng,
                    SamplingMode::Greedy,
                );
                if predicted == target_values[batch_index][position] {
                    joint_action_correct += 1;
                }
            }
        }
    }

    EvalMetrics {
        loss: total_nats / total_valid_tokens.max(1.0),
        joint_action_acc: joint_action_correct as f32 / joint_action_total.max(1) as f32,
    }
}

fn scheduled_learning_rate(step: usize, total_steps: usize, runtime: &RuntimeConfig) -> f32 {
    if total_steps <= 1 {
        return runtime.learning_rate;
    }

    let warmup_steps = runtime.warmup_steps.max(1).min(total_steps);
    if step <= warmup_steps {
        return runtime.learning_rate * (step as f32 / warmup_steps as f32);
    }

    let decay_steps = total_steps.saturating_sub(warmup_steps).max(1);
    let progress = (step.saturating_sub(warmup_steps) as f32 / decay_steps as f32).clamp(0.0, 1.0);
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
    runtime.min_learning_rate + cosine * (runtime.learning_rate - runtime.min_learning_rate)
}

struct PrefixMetrics {
    correct_tokens: usize,
    total_tokens: usize,
    exact_matches: usize,
    exact_match_examples: usize,
}

impl PrefixMetrics {
    fn token_accuracy(&self) -> f32 {
        self.correct_tokens as f32 / self.total_tokens.max(1) as f32
    }

    fn exact_match_rate(&self) -> f32 {
        self.exact_matches as f32 / self.exact_match_examples.max(1) as f32
    }
}

async fn evaluate_continuation(
    model: &NanoChatModel,
    device: &Device,
    tokenizer: &StrokeTokenizer,
    dataset: &SourceDataset,
    runtime: &RuntimeConfig,
) -> PrefixMetrics {
    let mut correct_tokens = 0;
    let mut total_tokens = 0;
    let mut exact_matches = 0;
    let files = dataset.files();
    let exact_match_examples = files.len();
    let batch_size = runtime.batch_size.max(1);
    let total_batches = files.len().div_ceil(batch_size).max(1);
    let mut processed_files = 0;
    let mut greedy_rng = StdRng::seed_from_u64(0);

    println!(
        "evaluating continuation token accuracy in {} batches of up to {} files...",
        total_batches, batch_size
    );

    for (batch_index, chunk) in files.chunks(batch_size).enumerate() {
        let batch_start = Instant::now();

        for file in chunk {
            let prompt_tokens = continuation_prompt_tokens(tokenizer, file, runtime);
            let generated = generate_completion(
                model,
                device,
                tokenizer,
                runtime,
                &prompt_tokens,
                tokenizer.eot_token(),
                &mut greedy_rng,
                SamplingMode::Greedy,
            )
            .await;
            let expected = expected_continuation_tokens(
                file,
                prompt_tokens.len().saturating_sub(1),
                runtime.sample_tokens,
            );

            total_tokens += expected.len();
            correct_tokens += expected
                .iter()
                .copied()
                .zip(generated.iter().copied())
                .filter(|(target, predicted)| predicted == target)
                .count();
            if generated == expected {
                exact_matches += 1;
            }
        }

        processed_files += chunk.len();
        println!(
            "continuation eval batch {}/{} | files {}/{} | token_acc={:.2}% | exact_match={:.2}% | elapsed={:.2?}",
            batch_index + 1,
            total_batches,
            processed_files,
            exact_match_examples,
            correct_tokens as f32 / total_tokens.max(1) as f32 * 100.0,
            exact_matches as f32 / processed_files.max(1) as f32 * 100.0,
            batch_start.elapsed(),
        );
    }

    PrefixMetrics {
        correct_tokens,
        total_tokens,
        exact_matches,
        exact_match_examples,
    }
}

async fn evaluate_interactive_continuation(
    predictor: &LivePredictor,
    tokenizer: &StrokeTokenizer,
    dataset: &SourceDataset,
    runtime: &RuntimeConfig,
) -> PrefixMetrics {
    let mut correct_tokens = 0;
    let mut total_tokens = 0;
    let mut exact_matches = 0;
    let files = dataset.files();
    let exact_match_examples = files.len();
    let batch_size = runtime.batch_size.min(8).max(1);
    let total_batches = files.len().div_ceil(batch_size).max(1);
    let mut processed_files = 0;

    println!(
        "evaluating continuation token accuracy in {} interactive batches of up to {} files...",
        total_batches, batch_size
    );

    for (batch_index, chunk) in files.chunks(batch_size).enumerate() {
        let batch_start = Instant::now();

        for file in chunk {
            let prompt_tokens = continuation_prompt_tokens(tokenizer, file, runtime);
            let generated = predictor
                .predict_greedy(&prompt_tokens[1..], runtime.sample_tokens, None)
                .unwrap_or_else(|error| {
                    panic!(
                        "interactive continuation evaluation failed at file {}: {error}",
                        file.path()
                    )
                });
            let expected = expected_continuation_tokens(
                file,
                prompt_tokens.len().saturating_sub(1),
                runtime.sample_tokens,
            );

            total_tokens += expected.len();
            correct_tokens += expected
                .iter()
                .copied()
                .zip(generated.iter().copied())
                .filter(|(target, predicted)| predicted == target)
                .count();
            if generated == expected {
                exact_matches += 1;
            }
        }

        processed_files += chunk.len();
        println!(
            "interactive continuation eval batch {}/{} | files {}/{} | token_acc={:.2}% | exact_match={:.2}% | elapsed={:.2?}",
            batch_index + 1,
            total_batches,
            processed_files,
            exact_match_examples,
            correct_tokens as f32 / total_tokens.max(1) as f32 * 100.0,
            exact_matches as f32 / processed_files.max(1) as f32 * 100.0,
            batch_start.elapsed(),
        );
    }

    PrefixMetrics {
        correct_tokens,
        total_tokens,
        exact_matches,
        exact_match_examples,
    }
}

fn continuation_prompt_tokens(
    tokenizer: &StrokeTokenizer,
    file: &SourceFile,
    runtime: &RuntimeConfig,
) -> Vec<u32> {
    let mut prompt = vec![tokenizer.bos_token()];
    prompt.extend_from_slice(file.completion_prompt_tokens(
        runtime.sample_prefix_tokens,
        runtime.block_size,
        runtime.sample_tokens,
    ));
    prompt
}

fn to_shared_runtime_config(runtime: &RuntimeConfig) -> SharedRuntimeConfig {
    SharedRuntimeConfig {
        epochs: runtime.epochs,
        warmup_steps: runtime.warmup_steps,
        learning_rate: runtime.learning_rate,
        min_learning_rate: runtime.min_learning_rate,
        beta1: runtime.beta1,
        beta2: runtime.beta2,
        adam_eps: runtime.adam_eps,
        weight_decay: runtime.weight_decay,
        log_every: runtime.log_every,
        eval_batches: runtime.eval_batches,
        sample_tokens: runtime.sample_tokens,
        sample_prefix_tokens: runtime.sample_prefix_tokens,
        sample_temperature: runtime.sample_temperature,
        sample_top_k: runtime.sample_top_k,
        block_size: runtime.block_size,
        batch_size: runtime.batch_size,
        n_embd: runtime.n_embd,
        n_head: runtime.n_head,
        n_kv_head: runtime.n_kv_head,
        n_ff: runtime.n_ff,
        n_layer: runtime.n_layer,
        conv_kernel_size: runtime.conv_kernel_size,
        attention_period: runtime.attention_period,
        use_rope: runtime.use_rope,
        rope_theta: runtime.rope_theta,
        use_canvas_state_embeddings: runtime.use_canvas_state_embeddings,
        use_extra_norms: runtime.use_extra_norms,
        eps: runtime.eps,
        init_scale: runtime.init_scale,
        seed: runtime.seed,
        save_every_steps: runtime.save_every_steps,
        save_final_model: runtime.save_final_model,
        save_quantization: to_shared_save_quantization(runtime.save_quantization),
        train_examples: runtime.train_examples,
        validation_examples: runtime.validation_examples,
        test_examples: runtime.test_examples,
        dataset_path: runtime.dataset_path.clone(),
        include_synthetic_data: runtime.include_synthetic_data,
        synthetic_multiplier: runtime.synthetic_multiplier,
        gguf_path: runtime.gguf_path.clone(),
        sample_output_path: runtime.sample_output_path.clone(),
    }
}

fn to_shared_save_quantization(quantization: SaveQuantization) -> SharedSaveQuantization {
    match quantization {
        SaveQuantization::F32 => SharedSaveQuantization::F32,
        SaveQuantization::F16 => SharedSaveQuantization::F16,
        SaveQuantization::Q4_0 => SharedSaveQuantization::Q4_0,
        SaveQuantization::Q8_0 => SharedSaveQuantization::Q8_0,
    }
}

fn expected_continuation_tokens(
    file: &SourceFile,
    prompt_tokens: usize,
    sample_tokens: usize,
) -> &[u32] {
    let start = prompt_tokens.min(file.target_tokens().len());
    let end = (start + sample_tokens).min(file.target_tokens().len());
    &file.target_tokens()[start..end]
}

fn preferred_eval_dataset<'a>(
    validation: &'a SourceDataset,
    test: &'a SourceDataset,
    train: &'a SourceDataset,
) -> &'a SourceDataset {
    if validation.num_docs() > 0 {
        validation
    } else if test.num_docs() > 0 {
        test
    } else {
        train
    }
}

#[derive(Clone, Copy)]
enum SamplingMode {
    Greedy,
    Sample,
}

fn run_inference(runtime: &RuntimeConfig, force_cpu: bool) {
    println!("generating {} tokens...", runtime.sample_tokens);
    let sample = generate_shared_sample(to_shared_runtime_config(runtime), force_cpu)
        .unwrap_or_else(|error| panic!("could not generate sample: {error}"));
    let sample_output_path = resolve_output_path(&runtime.sample_output_path);
    fs::write(&sample_output_path, &sample.svg).unwrap_or_else(|error| {
        panic!(
            "failed to write sample SVG {}: {error}",
            sample_output_path.display()
        )
    });
    println!(
        "generated {} tokens: {}",
        sample.token_count, sample.generated_tokens_description
    );
    println!("wrote sample SVG: {}", sample_output_path.display());
}

fn run_comparison_report(runtime: &RuntimeConfig, force_cpu: bool) {
    let report = build_shared_comparison_report(to_shared_runtime_config(runtime), force_cpu)
        .unwrap_or_else(|error| panic!("could not build comparison report: {error}"));
    let report_dir = resolve_output_path(Path::new("examples/compare"));
    fs::create_dir_all(&report_dir).unwrap_or_else(|error| {
        panic!(
            "failed to create comparison output directory {}: {error}",
            report_dir.display()
        )
    });
    write_comparison_report_assets(&report_dir, &report);
    let report_path = report_dir.join("index.html");
    fs::write(&report_path, comparison_report_html(&report)).unwrap_or_else(|error| {
        panic!(
            "failed to write comparison report {}: {error}",
            report_path.display()
        )
    });

    println!(
        "comparison summary: prompted_examples={} unique_completions={} avg_completion_similarity={:.1}%",
        report.sample_count,
        report.unique_completions,
        report.average_similarity * 100.0
    );
    for count in &report.shape_counts {
        println!("comparison_shape {}: {}", count.shape, count.count);
    }
    println!("wrote comparison report: {}", report_path.display());
}

fn write_comparison_report_assets(report_dir: &Path, report: &ComparisonReport) {
    for (index, sample) in report.samples.iter().enumerate() {
        write_report_svg_asset(
            &report_dir.join(format!("generated-{index:03}.svg")),
            &sample.generated_svg,
        );
        write_report_svg_asset(
            &report_dir.join(format!("expected-{index:03}.svg")),
            &sample.expected_svg,
        );
    }

    for (index, item) in report.dataset_gallery.iter().enumerate() {
        write_report_svg_asset(&report_dir.join(format!("train-{index:03}.svg")), &item.svg);
    }
}

fn write_report_svg_asset(path: &Path, svg: &str) {
    fs::write(path, svg)
        .unwrap_or_else(|error| panic!("failed to write SVG asset {}: {error}", path.display()));
}

fn comparison_report_html(report: &ComparisonReport) -> String {
    let comparison_shapes = report
        .shape_counts
        .iter()
        .map(|count| {
            format!(
                "<li><strong>{}</strong>: {} prompted comparisons</li>",
                html_escape(&count.shape),
                count.count
            )
        })
        .collect::<Vec<_>>()
        .join("");
    let sample_cards = report
        .samples
        .iter()
        .enumerate()
        .map(|(index, sample)| {
            format!(
                "<article class=\"compare-card\"><div class=\"compare-head\"><h2>{label}</h2><p>Prompted from <strong>{shape}</strong> in <code>{path}</code></p><p><strong>Prompt</strong><br><code>{prompt_tokens}</code></p></div><div class=\"pair\"><figure><img src=\"{generated_name}\" alt=\"generated completion for {label}\"><figcaption>Model completion</figcaption></figure><figure><img src=\"{expected_name}\" alt=\"expected continuation for {label}\"><figcaption>Expected completion</figcaption></figure></div><dl class=\"metrics\"><div><dt>Edit distance</dt><dd>{distance}</dd></div><div><dt>Similarity</dt><dd>{similarity:.1}%</dd></div></dl><p><strong>Model continuation</strong><br><code>{generated_tokens}</code></p><p><strong>Expected continuation</strong><br><code>{expected_tokens}</code></p></article>",
                label = html_escape(&sample.example_label),
                shape = html_escape(&sample.shape),
                path = html_escape(&sample.source_path),
                prompt_tokens = html_escape(&sample.prompt_tokens),
                generated_name = format!("generated-{index:03}.svg"),
                expected_name = format!("expected-{index:03}.svg"),
                distance = sample.edit_distance,
                similarity = sample.similarity * 100.0,
                generated_tokens = html_escape(&sample.generated_tokens),
                expected_tokens = html_escape(&sample.expected_tokens),
            )
        })
        .collect::<Vec<_>>()
        .join("");
    let gallery_cards = report
        .dataset_gallery
        .iter()
        .enumerate()
        .map(|(index, item)| {
            format!(
                "<article class=\"gallery-card\"><img src=\"{image_name}\" alt=\"{shape} training sample\"><h3>{shape}</h3><p><code>{path}</code></p><p><code>{tokens}</code></p></article>",
                image_name = format!("train-{index:03}.svg"),
                shape = html_escape(&item.shape),
                path = html_escape(&item.path),
                tokens = html_escape(&item.tokens),
            )
        })
        .collect::<Vec<_>>()
        .join("");

    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>Nanochat Comparison Report</title><style>:root{{--bg:#f5efe3;--panel:#fffaf0;--ink:#2f261c;--muted:#6f6558;--line:#deceb2;--accent:#d96c42;--accent-soft:#f1c9b8;--prompt:#264653;}}body{{margin:0;font-family:\"Iowan Old Style\",\"Palatino Linotype\",serif;background:radial-gradient(circle at top,#fdf7ec 0%,var(--bg) 58%,#eadcc4 100%);color:var(--ink);}}main{{max-width:1320px;margin:0 auto;padding:32px 20px 56px;}}h1,h2,h3,p,ul{{margin:0;}}header{{padding:28px;border:1px solid var(--line);border-radius:24px;background:rgba(255,250,240,0.92);box-shadow:0 20px 60px rgba(83,58,26,0.08);}}header p{{margin-top:10px;color:var(--muted);font-size:18px;line-height:1.45;}}.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-top:18px;}}.stat{{padding:14px 16px;border-radius:16px;background:#fffdf7;border:1px solid var(--line);}}.stat strong{{display:block;font-size:28px;color:var(--accent);}}section{{margin-top:28px;}}.shape-list{{margin-top:14px;padding-left:20px;color:var(--muted);}}.compare-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:18px;}}.compare-card,.gallery-card{{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:16px;box-shadow:0 14px 36px rgba(83,58,26,0.08);}}.compare-head h2{{font-size:24px;}}.compare-head p{{margin-top:6px;color:var(--muted);line-height:1.4;}}.pair{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:16px;}}figure{{margin:0;}}img{{display:block;width:100%;aspect-ratio:1;border-radius:14px;background:white;border:1px solid #eadcc4;}}figcaption{{margin-top:8px;font-size:14px;color:var(--muted);text-align:center;}}.metrics{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px;margin:16px 0;}}.metrics div{{padding:10px 12px;border-radius:14px;background:#fffdf7;border:1px solid var(--line);}}dt{{font-size:13px;color:var(--muted);}}dd{{margin:6px 0 0;font-size:22px;font-weight:700;color:var(--accent);}}code{{font-family:\"SFMono-Regular\",Menlo,Consolas,monospace;font-size:12px;line-height:1.5;white-space:pre-wrap;word-break:break-word;}}.gallery-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:16px;}}.gallery-card h3{{margin-top:10px;font-size:20px;}}.gallery-card p{{margin-top:8px;color:var(--muted);line-height:1.4;}}.legend{{display:flex;gap:14px;flex-wrap:wrap;margin-top:14px;color:var(--muted);font-size:14px;}}.swatch{{display:inline-block;width:14px;height:14px;border-radius:999px;vertical-align:-2px;margin-right:6px;border:1px solid rgba(0,0,0,0.08);}}.swatch.prompt{{background:var(--prompt);}}.swatch.cont{{background:var(--accent);}}@media (max-width: 720px){{.pair{{grid-template-columns:1fr;}}}}</style></head><body><main><header><h1>Nanochat Prompted Completion</h1><p>Each comparison starts from the first two commands of a held-out evaluation example. The teal segment is the prompt, and the orange segment is either the model's completion or the expected continuation from the same test example.</p><div class=\"stats\"><div class=\"stat\"><strong>{train_examples}</strong>train examples</div><div class=\"stat\"><strong>{compare_examples}</strong>eval examples</div><div class=\"stat\"><strong>{sample_count}</strong>prompted comparisons</div><div class=\"stat\"><strong>{average_similarity:.1}%</strong>avg completion similarity</div></div><div class=\"legend\"><span><span class=\"swatch prompt\"></span>Prompt</span><span><span class=\"swatch cont\"></span>Continuation</span></div><ul class=\"shape-list\">{comparison_shapes}</ul></header><section><div class=\"compare-grid\">{sample_cards}</div></section><section><header><h2>Training Gallery</h2><p>Representative examples from the current training split, rendered with the same SVG exporter as the prompted completions.</p></header><div class=\"gallery-grid\">{gallery_cards}</div></section></main></body></html>",
        train_examples = report.train_examples,
        compare_examples = report.compare_examples,
        sample_count = report.sample_count,
        average_similarity = report.average_similarity * 100.0,
        comparison_shapes = comparison_shapes,
        sample_cards = sample_cards,
        gallery_cards = gallery_cards,
    )
}

fn html_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('\"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parity_runtime(gguf_path: PathBuf) -> RuntimeConfig {
        RuntimeConfig {
            epochs: 1,
            warmup_steps: 1,
            learning_rate: 1e-3,
            min_learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.95,
            adam_eps: 1e-8,
            weight_decay: 0.01,
            log_every: 1,
            eval_batches: 1,
            sample_tokens: 8,
            sample_prefix_tokens: 4,
            sample_temperature: 0.7,
            sample_top_k: 4,
            block_size: 8,
            batch_size: 2,
            n_embd: 16,
            n_head: 4,
            n_kv_head: 1,
            n_ff: 32,
            n_layer: 1,
            conv_kernel_size: 3,
            attention_period: 1,
            use_rope: true,
            rope_theta: 10_000.0,
            use_canvas_state_embeddings: true,
            use_extra_norms: false,
            eps: 1e-5,
            init_scale: 0.05,
            seed: 7,
            save_every_steps: 0,
            save_final_model: false,
            save_quantization: SaveQuantization::F32,
            train_examples: 1,
            validation_examples: 1,
            test_examples: 1,
            dataset_path: None,
            include_synthetic_data: false,
            synthetic_multiplier: 1,
            gguf_path,
            sample_output_path: PathBuf::from("sample.svg"),
        }
    }

    #[tokio::test]
    async fn gguf_runtime_matches_training_greedy_generation() {
        let gguf_path = std::env::temp_dir().join(format!(
            "nanochat-parity-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let runtime = parity_runtime(gguf_path.clone());
        let device = Device::cpu();
        let tokenizer = StrokeTokenizer::new();
        let mut rng = StdRng::seed_from_u64(runtime.seed);
        let model = NanoChatModel::new(&device, &mut rng, &tokenizer, &runtime);

        save_gguf(&model, &tokenizer, &runtime, &gguf_path).await;

        let prompt = vec![
            tokenizer.token_from_components(0, 2, 1),
            tokenizer.token_from_components(1, 4, 1),
        ];
        let mut model_prompt = vec![tokenizer.bos_token()];
        model_prompt.extend_from_slice(&prompt);
        let expected = generate_completion(
            &model,
            &device,
            &tokenizer,
            &runtime,
            &model_prompt,
            tokenizer.eot_token(),
            &mut StdRng::seed_from_u64(runtime.seed),
            SamplingMode::Greedy,
        )
        .await;

        let predictor = LivePredictor::load(to_shared_runtime_config(&runtime), true)
            .unwrap_or_else(|error| panic!("failed to load GGUF predictor: {error}"));
        let actual = predictor
            .predict_greedy(&prompt, runtime.sample_tokens, None)
            .unwrap_or_else(|error| panic!("failed to generate GGUF continuation: {error}"));

        assert_eq!(actual, expected);

        let _ = fs::remove_file(gguf_path);
    }
}
