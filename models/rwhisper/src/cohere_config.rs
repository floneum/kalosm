use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CohereConfig {
    pub batch_size: usize,
    pub sample_rate: usize,
    pub max_audio_clip_s: usize,
    pub overlap_chunk_second: usize,
    #[serde(default = "default_min_energy_window_samples")]
    pub min_energy_window_samples: usize,
    pub supported_languages: Vec<String>,
    pub preprocessor: CoherePreprocessorConfig,
    pub encoder: CohereEncoderConfig,
    pub transf_decoder: CohereDecoderWrapperConfig,
    pub head: CohereHeadConfig,
    pub vocab_size: usize,
}

fn default_min_energy_window_samples() -> usize {
    1600
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CoherePreprocessorConfig {
    pub features: usize,
    pub sample_rate: usize,
    pub window_size: f32,
    pub window_stride: f32,
    pub n_fft: usize,
    pub normalize: String,
    pub dither: f32,
    pub pad_to: usize,
    pub pad_value: f32,
    pub window: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CohereEncoderConfig {
    pub att_context_size: [isize; 2],
    pub conv_kernel_size: usize,
    pub conv_norm_type: String,
    pub d_model: usize,
    pub dropout: f32,
    pub feat_in: usize,
    pub feat_out: isize,
    pub ff_expansion_factor: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub pos_emb_max_len: usize,
    pub self_attention_model: String,
    pub subsampling: String,
    pub subsampling_conv_channels: usize,
    pub subsampling_factor: usize,
    pub untie_biases: bool,
    pub xscaling: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CohereDecoderWrapperConfig {
    pub config_dict: CohereDecoderConfig,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CohereDecoderConfig {
    pub hidden_act: String,
    pub hidden_size: usize,
    pub inner_size: usize,
    pub learn_positional_encodings: bool,
    pub max_sequence_length: usize,
    pub num_attention_heads: usize,
    pub num_layers: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct CohereHeadConfig {
    pub hidden_size: usize,
    pub log_softmax: bool,
    pub num_classes: usize,
}
