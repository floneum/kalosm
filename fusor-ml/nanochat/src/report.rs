use crate::{
    config::RuntimeConfig,
    data::{
        CanvasStateIndexes, CanvasStateSpec, DatasetSplit, StrokeTokenizer, autoregressive_context,
        canvas_state_indexes, load_dataset_source, position_indexes, token_component_indexes,
        tokens_to_svg_string,
    },
    interactive_model::InteractiveNanoChatModel,
};
use fusor::{Device, Tensor, ToVec3, VarBuilder};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::BufReader,
    panic::{self, AssertUnwindSafe},
    path::{Path, PathBuf},
};

const COMPARISON_SAMPLE_LIMIT: usize = 12;
const DATASET_GALLERY_LIMIT: usize = 24;

#[derive(Clone, Debug, PartialEq)]
pub struct InferenceSample {
    pub token_count: usize,
    pub generated_tokens: Vec<u32>,
    pub generated_tokens_description: String,
    pub svg: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComparisonReport {
    pub dataset_label: String,
    pub train_examples: usize,
    pub compare_examples: usize,
    pub sample_count: usize,
    pub unique_completions: usize,
    pub average_similarity: f32,
    pub shape_counts: Vec<ShapeCount>,
    pub samples: Vec<ComparisonSample>,
    pub dataset_gallery: Vec<DatasetGalleryItem>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShapeCount {
    pub shape: String,
    pub count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComparisonSample {
    pub example_label: String,
    pub source_path: String,
    pub shape: String,
    pub prompt_tokens: String,
    pub generated_tokens: String,
    pub expected_tokens: String,
    pub edit_distance: usize,
    pub similarity: f32,
    pub generated_svg: String,
    pub expected_svg: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DatasetGalleryItem {
    pub path: String,
    pub shape: String,
    pub tokens: String,
    pub svg: String,
}

struct LoadedCheckpoint {
    runtime: RuntimeConfig,
    tokenizer: StrokeTokenizer,
    device: Device,
    model: InteractiveNanoChatModel,
}

pub struct LivePredictor {
    runtime: RuntimeConfig,
    tokenizer: StrokeTokenizer,
    device: Device,
    model: InteractiveNanoChatModel,
}

#[derive(Clone, Copy)]
enum SamplingMode {
    Greedy,
    Sample,
}

/// Constraint applied to the first predicted token during constrained decoding.
/// Used to handle partial-token boundaries: when the user has drawn part of what
/// would be a single RLE token, we pop that partial token from the prompt and
/// constrain the model's first prediction to continue in the same direction.
#[derive(Clone, Debug)]
pub struct FirstTokenConstraint {
    pub mode_index: u32,
    pub direction_index: u32,
    pub min_count: usize,
}

pub fn load_runtime_config() -> Result<RuntimeConfig, String> {
    RuntimeConfig::try_load()
}

pub fn load_tokenizer(runtime: &RuntimeConfig) -> Result<StrokeTokenizer, String> {
    let runtime = runtime.clone();
    catch_panic(move || load_checkpoint_tokenizer(&runtime)).and_then(|result| result)
}

pub fn generate_sample(runtime: RuntimeConfig, force_cpu: bool) -> Result<InferenceSample, String> {
    catch_panic(move || {
        let predictor = LivePredictor::load(runtime, force_cpu)?;
        predictor.generate_sample()
    })
    .and_then(|result| result)
}

pub fn build_comparison_report(
    runtime: RuntimeConfig,
    force_cpu: bool,
) -> Result<ComparisonReport, String> {
    catch_panic(move || {
        let predictor = LivePredictor::load(runtime, force_cpu)?;
        predictor.build_comparison_report()
    })
    .and_then(|result| result)
}

impl LivePredictor {
    pub fn load(runtime: RuntimeConfig, force_cpu: bool) -> Result<Self, String> {
        catch_panic(move || {
            let LoadedCheckpoint {
                runtime,
                tokenizer,
                device,
                model,
            } = load_checkpoint(runtime, force_cpu)?;
            Ok(Self {
                runtime,
                tokenizer,
                device,
                model,
            })
        })
        .and_then(|result| result)
    }

    pub fn tokenizer(&self) -> &StrokeTokenizer {
        &self.tokenizer
    }

    pub fn predict_greedy(
        &self,
        prompt_tokens: &[u32],
        max_tokens: usize,
        first_token_constraint: Option<FirstTokenConstraint>,
    ) -> Result<Vec<u32>, String> {
        let mut model_prompt = vec![self.tokenizer.bos_token()];
        model_prompt.extend_from_slice(prompt_tokens);
        catch_panic(|| {
            Ok(pollster::block_on(generate_interactive_completion(
                &self.model,
                &self.device,
                &self.tokenizer,
                &self.runtime,
                self.runtime.seed,
                &model_prompt,
                max_tokens,
                SamplingMode::Greedy,
                first_token_constraint,
            )))
        })
        .and_then(|result| result)
    }

    pub fn generate_sample(&self) -> Result<InferenceSample, String> {
        catch_panic(|| {
            let generated = pollster::block_on(generate_interactive_completion(
                &self.model,
                &self.device,
                &self.tokenizer,
                &self.runtime,
                self.runtime.seed,
                &[self.tokenizer.bos_token()],
                self.runtime.sample_tokens,
                SamplingMode::Sample,
                None,
            ));
            let svg = tokens_to_svg_string(&self.tokenizer, &[], &generated);

            InferenceSample {
                token_count: generated.len(),
                generated_tokens_description: self.tokenizer.describe_tokens(&generated, 80),
                generated_tokens: generated,
                svg,
            }
        })
    }

    pub fn build_comparison_report(&self) -> Result<ComparisonReport, String> {
        catch_panic(|| self.build_comparison_report_impl()).and_then(|result| result)
    }

    fn build_comparison_report_impl(&self) -> Result<ComparisonReport, String> {
        let dataset_source = load_dataset_source(&self.runtime);
        if !dataset_source.tokenizer.same_vocabulary(&self.tokenizer) {
            return Err(
                "checkpoint tokenizer vocabulary does not match the configured report dataset"
                    .to_string(),
            );
        }
        let dataset_label = dataset_source.label.clone();
        pollster::block_on(build_comparison_report_async(
            &self.model,
            &self.device,
            &self.tokenizer,
            &self.runtime,
            dataset_label,
            dataset_source.split,
        ))
    }
}

async fn build_comparison_report_async(
    model: &InteractiveNanoChatModel,
    device: &Device,
    tokenizer: &StrokeTokenizer,
    runtime: &RuntimeConfig,
    dataset_label: String,
    datasets: DatasetSplit,
) -> Result<ComparisonReport, String> {
    let train_dataset = datasets.train;
    let validation_dataset = datasets.validation;
    let test_dataset = datasets.test;
    assert!(train_dataset.num_docs() > 0, "training split is empty");
    let compare_dataset = if test_dataset.num_docs() > 0 {
        &test_dataset
    } else {
        preferred_eval_dataset(&validation_dataset, &test_dataset, &train_dataset)
    };
    assert!(
        compare_dataset.num_docs() > 0,
        "comparison dataset is empty"
    );

    let mut samples = Vec::new();
    for (index, file) in compare_dataset
        .files()
        .iter()
        .take(COMPARISON_SAMPLE_LIMIT)
        .enumerate()
    {
        let prompt = file
            .target_tokens()
            .iter()
            .copied()
            .take(runtime.sample_prefix_tokens)
            .collect::<Vec<_>>();
        let expected = &file.target_tokens()[prompt.len()..];
        let mut model_prompt = vec![tokenizer.bos_token()];
        model_prompt.extend_from_slice(&prompt);
        let generated = generate_interactive_completion(
            model,
            device,
            tokenizer,
            runtime,
            runtime.seed + index as u64,
            &model_prompt,
            expected.len(),
            SamplingMode::Greedy,
            None,
        )
        .await;
        let edit_distance = token_edit_distance(&generated, expected);

        samples.push(ComparisonSample {
            example_label: format!("Example {:03}", index),
            source_path: file.path().to_string(),
            shape: shape_name_from_path(file.path()).to_string(),
            prompt_tokens: tokenizer.describe_tokens(&prompt, 80),
            generated_tokens: tokenizer.describe_tokens(&generated, 80),
            expected_tokens: tokenizer.describe_tokens(expected, 80),
            edit_distance,
            similarity: similarity_score(&generated, expected, edit_distance),
            generated_svg: tokens_to_svg_string(tokenizer, &prompt, &generated),
            expected_svg: tokens_to_svg_string(tokenizer, &prompt, expected),
        });
    }

    let mut dataset_gallery = Vec::new();
    for file in train_dataset.files().iter().take(DATASET_GALLERY_LIMIT) {
        dataset_gallery.push(DatasetGalleryItem {
            path: file.path().to_string(),
            shape: shape_name_from_path(file.path()).to_string(),
            tokens: tokenizer.describe_tokens(file.target_tokens(), 80),
            svg: tokens_to_svg_string(tokenizer, &[], file.target_tokens()),
        });
    }

    let average_similarity =
        samples.iter().map(|sample| sample.similarity).sum::<f32>() / samples.len().max(1) as f32;
    let unique_completions = samples
        .iter()
        .map(|sample| sample.generated_tokens.clone())
        .collect::<BTreeSet<_>>()
        .len();
    let shape_counts = comparison_shape_counts(&samples)
        .into_iter()
        .map(|(shape, count)| ShapeCount {
            shape: shape.to_string(),
            count,
        })
        .collect();

    Ok(ComparisonReport {
        dataset_label,
        train_examples: train_dataset.num_docs(),
        compare_examples: compare_dataset.num_docs(),
        sample_count: samples.len(),
        unique_completions,
        average_similarity,
        shape_counts,
        samples,
        dataset_gallery,
    })
}

fn load_checkpoint(runtime: RuntimeConfig, force_cpu: bool) -> Result<LoadedCheckpoint, String> {
    let device = init_device(force_cpu)?;
    let gguf_path = resolve_output_path(&runtime.gguf_path);
    let mut reader = BufReader::new(
        File::open(&gguf_path)
            .map_err(|error| format!("could not open {}: {error}", gguf_path.display()))?,
    );
    let mut vb = VarBuilder::from_gguf(&mut reader)
        .map_err(|error| format!("could not parse GGUF {}: {error}", gguf_path.display()))?;
    let tokenizer = StrokeTokenizer::from_var_builder(&vb)
        .map_err(|error| format!("could not load tokenizer metadata: {error}"))?;
    let model = InteractiveNanoChatModel::load(&device, &mut vb)
        .map_err(|error| format!("could not load model: {error}"))?;

    Ok(LoadedCheckpoint {
        runtime,
        tokenizer,
        device,
        model,
    })
}

fn load_checkpoint_tokenizer(runtime: &RuntimeConfig) -> Result<StrokeTokenizer, String> {
    let gguf_path = resolve_output_path(&runtime.gguf_path);
    let mut reader = BufReader::new(
        File::open(&gguf_path)
            .map_err(|error| format!("could not open {}: {error}", gguf_path.display()))?,
    );
    let vb = VarBuilder::from_gguf(&mut reader)
        .map_err(|error| format!("could not parse GGUF {}: {error}", gguf_path.display()))?;
    StrokeTokenizer::from_var_builder(&vb)
        .map_err(|error| format!("could not load tokenizer metadata: {error}"))
}

fn init_device(force_cpu: bool) -> Result<Device, String> {
    if force_cpu {
        Ok(Device::cpu())
    } else {
        pollster::block_on(Device::gpu())
            .map_err(|error| format!("failed to initialize GPU device: {error:?}"))
    }
}

async fn generate_interactive_completion(
    model: &InteractiveNanoChatModel,
    device: &Device,
    tokenizer: &StrokeTokenizer,
    runtime: &RuntimeConfig,
    seed: u64,
    prompt_tokens: &[u32],
    max_tokens: usize,
    mode: SamplingMode,
    first_token_constraint: Option<FirstTokenConstraint>,
) -> Vec<u32> {
    let block_size = model.block_size().min(runtime.block_size);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut tokens = prompt_tokens.to_vec();

    let prompt_names: Vec<&str> = prompt_tokens
        .iter()
        .map(|&t| tokenizer.token_name(t))
        .collect();
    println!(
        "[DEBUG] generate_interactive_completion: block_size={}, prompt_len={}, prompt={:?}",
        block_size,
        prompt_tokens.len(),
        prompt_names
    );

    for step in 0..max_tokens {
        let (context, last_index) =
            autoregressive_context(&tokens, tokenizer.eot_token(), block_size);

        let context_names: Vec<&str> = context.iter().map(|&t| tokenizer.token_name(t)).collect();
        println!(
            "[DEBUG] step={}: context_len={}, last_index={}, context={:?}",
            step,
            context.len(),
            last_index,
            context_names
        );

        if let Some(spec) = model.canvas_state_spec() {
            let indexes = canvas_state_indexes(tokenizer, std::slice::from_ref(&context), spec);
            println!(
                "[DEBUG] step={}: canvas_x={:?}, canvas_y={:?}, pen={:?}, coord_vocab_size={}, coord_offset={}",
                step,
                indexes.cursor_x[0],
                indexes.cursor_y[0],
                indexes.pen_state[0],
                spec.coordinate_vocab_size,
                spec.coordinate_offset
            );
        }

        let components = token_component_indexes(tokenizer, std::slice::from_ref(&context));
        let mode_inputs: Tensor<2, u32> = Tensor::new(device, &components.mode);
        let direction_inputs: Tensor<2, u32> = Tensor::new(device, &components.direction);
        let count_inputs: Tensor<2, u32> = Tensor::new(device, &components.count);
        let position_values = position_indexes(1, context.len().max(1));
        let position_inputs: Tensor<2, u32> = Tensor::new(device, &position_values);
        let causal_mask = fusor::cache::AttentionMask::<f32>::causal(device, context.len().max(1));
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
        let mode_logits: Vec<Vec<Vec<f32>>> = logits.mode.as_slice().await.unwrap().to_vec3();
        let direction_logits: Vec<Vec<Vec<f32>>> =
            logits.direction.as_slice().await.unwrap().to_vec3();
        let count_logits: Vec<Vec<Vec<f32>>> = logits.count.as_slice().await.unwrap().to_vec3();

        let ml = &mode_logits[0][last_index];
        let dl = &direction_logits[0][last_index];
        let cl = &count_logits[0][last_index];
        println!("[DEBUG] step={}: mode_logits={:?}", step, ml);
        let mut dir_indexed: Vec<(usize, f32)> = dl.iter().copied().enumerate().collect();
        dir_indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top3_dir: Vec<(usize, f32)> = dir_indexed.into_iter().take(3).collect();
        println!("[DEBUG] step={}: top3_direction={:?}", step, top3_dir);
        let mut count_indexed: Vec<(usize, f32)> = cl.iter().copied().enumerate().collect();
        count_indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        let top3_count: Vec<(usize, f32)> = count_indexed.into_iter().take(3).collect();
        println!("[DEBUG] step={}: top3_count={:?}", step, top3_count);

        let step_constraint = if step == 0 {
            first_token_constraint.as_ref()
        } else {
            None
        };
        let next = decode_next_action_token(
            tokenizer,
            &context[..=last_index],
            &mode_logits[0][last_index],
            &direction_logits[0][last_index],
            &count_logits[0][last_index],
            runtime.sample_temperature,
            runtime.sample_top_k,
            &mut rng,
            mode,
            step_constraint,
        );
        println!(
            "[DEBUG] step={}: predicted token={} ({})",
            step,
            next,
            tokenizer.token_name(next)
        );
        tokens.push(next);
        if next == tokenizer.eot_token() {
            break;
        }
    }

    tokens[prompt_tokens.len()..]
        .iter()
        .copied()
        .take_while(|&token| token != tokenizer.eot_token())
        .collect()
}

fn resolve_output_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
    }
}

fn preferred_eval_dataset<'a>(
    validation: &'a crate::data::SourceDataset,
    test: &'a crate::data::SourceDataset,
    train: &'a crate::data::SourceDataset,
) -> &'a crate::data::SourceDataset {
    if validation.num_docs() > 0 {
        validation
    } else if test.num_docs() > 0 {
        test
    } else {
        train
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
    constraint: Option<&FirstTokenConstraint>,
) -> u32 {
    let mode_index = if let Some(c) = constraint {
        c.mode_index
    } else {
        match sampling {
            SamplingMode::Greedy => argmax_from_logits(mode_logits),
            SamplingMode::Sample => sample_from_logits(rng, mode_logits, temperature, top_k),
        }
    };
    if mode_index >= 2 {
        return tokenizer.eot_token();
    }

    let cursor = tokenizer.cursor_after_tokens(context);

    let direction_index = if let Some(c) = constraint {
        c.direction_index
    } else {
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
        match sampling {
            SamplingMode::Greedy => argmax_from_logits(&filtered_direction_logits),
            SamplingMode::Sample => {
                sample_from_logits(rng, &filtered_direction_logits, temperature, top_k)
            }
        }
    };

    let legal_limit = tokenizer.legal_count_limit(cursor, direction_index);
    if legal_limit == 0 {
        return tokenizer.eot_token();
    }

    let min_count = constraint.map_or(0, |c| c.min_count);
    let mut filtered_count_logits = count_logits.to_vec();
    for (count_index, logit) in filtered_count_logits.iter_mut().enumerate() {
        let count = count_index + 1;
        if count > legal_limit || count < min_count {
            *logit = f32::NEG_INFINITY;
        }
    }
    if !filtered_count_logits.iter().any(|logit| logit.is_finite()) {
        // min_count exceeds legal_limit — fall back to the legal limit itself
        let count = legal_limit.max(1);
        return tokenizer.token_from_components(mode_index, direction_index, count);
    }
    let count_index = match sampling {
        SamplingMode::Greedy => argmax_from_logits(&filtered_count_logits),
        SamplingMode::Sample => sample_from_logits(rng, &filtered_count_logits, temperature, top_k),
    } as usize;
    let count = (count_index + 1).clamp(1, legal_limit.max(1));

    tokenizer.token_from_components(mode_index, direction_index, count)
}

fn token_edit_distance(left: &[u32], right: &[u32]) -> usize {
    let mut previous: Vec<usize> = (0..=right.len()).collect();
    let mut current = vec![0; right.len() + 1];

    for (i, &left_token) in left.iter().enumerate() {
        current[0] = i + 1;
        for (j, &right_token) in right.iter().enumerate() {
            let substitution_cost = usize::from(left_token != right_token);
            current[j + 1] = (previous[j + 1] + 1)
                .min(current[j] + 1)
                .min(previous[j] + substitution_cost);
        }
        std::mem::swap(&mut previous, &mut current);
    }

    previous[right.len()]
}

fn similarity_score(generated: &[u32], target: &[u32], edit_distance: usize) -> f32 {
    let scale = generated.len().max(target.len()).max(1) as f32;
    1.0 - (edit_distance as f32 / scale)
}

fn shape_name_from_path(path: &str) -> &str {
    let filename = path.rsplit('/').next().unwrap_or(path);
    let stem = filename.strip_suffix(".stroke").unwrap_or(filename);
    match stem.rsplit_once('-') {
        Some((shape, suffix)) if suffix.chars().all(|character| character.is_ascii_digit()) => {
            shape
        }
        _ => stem,
    }
}

fn comparison_shape_counts(samples: &[ComparisonSample]) -> BTreeMap<&str, usize> {
    let mut counts = BTreeMap::new();
    for sample in samples {
        *counts.entry(sample.shape.as_str()).or_insert(0) += 1;
    }
    counts
}

fn catch_panic<T, F>(f: F) -> Result<T, String>
where
    F: FnOnce() -> T,
{
    panic::catch_unwind(AssertUnwindSafe(f)).map_err(|payload| {
        if let Some(message) = payload.downcast_ref::<String>() {
            message.clone()
        } else if let Some(message) = payload.downcast_ref::<&str>() {
            (*message).to_string()
        } else {
            "nanochat task panicked".to_string()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SaveQuantization;
    use fusor_gguf::{GgufMetadata, GgufVersion};
    use std::{fs, io::Cursor, path::PathBuf};

    fn test_runtime(gguf_path: PathBuf) -> RuntimeConfig {
        RuntimeConfig {
            epochs: 1,
            warmup_steps: 1,
            learning_rate: 1e-3,
            min_learning_rate: 1e-4,
            beta1: 0.9,
            beta2: 0.95,
            adam_eps: 1e-8,
            weight_decay: 0.1,
            log_every: 10,
            eval_batches: 1,
            sample_tokens: 8,
            sample_prefix_tokens: 4,
            sample_temperature: 0.7,
            sample_top_k: 4,
            block_size: 32,
            batch_size: 4,
            n_embd: 32,
            n_head: 2,
            n_kv_head: 2,
            n_ff: 64,
            n_layer: 2,
            conv_kernel_size: 3,
            attention_period: 1,
            use_rope: false,
            rope_theta: 10_000.0,
            use_canvas_state_embeddings: true,
            use_extra_norms: false,
            eps: 1e-5,
            init_scale: 0.08,
            seed: 42,
            save_every_steps: 0,
            save_final_model: false,
            save_quantization: SaveQuantization::F32,
            train_examples: 1,
            validation_examples: 1,
            test_examples: 1,
            dataset_path: Some(PathBuf::from("/path/that/does/not/exist.json")),
            include_synthetic_data: false,
            synthetic_multiplier: 1,
            gguf_path,
            sample_output_path: PathBuf::from("sample.svg"),
        }
    }

    #[test]
    fn load_tokenizer_reads_checkpoint_metadata_without_loading_dataset() {
        let tokenizer = StrokeTokenizer::with_grid(9);
        let gguf = GgufMetadata {
            version: GgufVersion::V3,
            metadata: tokenizer
                .gguf_metadata()
                .into_iter()
                .map(|(key, value)| (key.into_boxed_str(), value))
                .collect(),
            tensor_infos: Default::default(),
            tensor_data_offset: 0,
        };
        let path = std::env::temp_dir().join(format!(
            "nanochat-tokenizer-{}-{}.gguf",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let mut bytes = Cursor::new(Vec::new());
        gguf.write(&mut bytes, std::iter::empty::<(&str, &[u8])>())
            .unwrap();
        fs::write(&path, bytes.into_inner()).unwrap();

        let runtime = test_runtime(path.clone());
        let loaded = load_tokenizer(&runtime).unwrap();
        assert!(loaded.same_vocabulary(&tokenizer));

        let _ = fs::remove_file(path);
    }
}
