use serde::Deserialize;

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MoonshineStreamingConfig {
    pub(crate) encoder_config: MoonshineStreamingEncoderConfig,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) use_cache: bool,
    pub(crate) pad_token_id: Option<u32>,
    pub(crate) bos_token_id: Option<u32>,
    pub(crate) eos_token_id: u32,
    pub(crate) decoder_start_token_id: Option<u32>,
    pub(crate) attention_bias: bool,
    pub(crate) attention_dropout: f32,
    pub(crate) head_dim: Option<usize>,
    pub(crate) pad_head_dim_to_multiple_of: Option<usize>,
    pub(crate) tie_word_embeddings: bool,
    pub(crate) vocab_size: usize,
    #[serde(default)]
    pub(crate) rope_parameters: Option<MoonshineRopeParameters>,
}

impl MoonshineStreamingConfig {
    pub(crate) fn decoder_start_token(&self) -> u32 {
        self.decoder_start_token_id
            .or(self.bos_token_id)
            .unwrap_or(1)
    }

    pub(crate) fn decoder_head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads.max(1))
    }

    pub(crate) fn encoder_hidden_size(&self) -> usize {
        self.encoder_config.hidden_size
    }

    pub(crate) fn rope_theta(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|params| params.rope_theta)
            .unwrap_or(10_000.0)
    }

    pub(crate) fn partial_rotary_factor(&self) -> f32 {
        self.rope_parameters
            .as_ref()
            .and_then(|params| params.partial_rotary_factor)
            .unwrap_or(0.5)
    }

    pub(crate) fn decoder_rotary_dim(&self) -> usize {
        let head_dim = self.decoder_head_dim();
        let factor = self.partial_rotary_factor().clamp(0.0, 1.0);
        let rotary_dim = ((head_dim as f32 * factor).floor() as usize / 2) * 2;
        rotary_dim.max(2).min(head_dim)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MoonshineStreamingEncoderConfig {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) hidden_act: String,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) max_position_embeddings: usize,
    pub(crate) attention_dropout: f32,
    pub(crate) attention_bias: bool,
    pub(crate) sample_rate: usize,
    pub(crate) frame_ms: f32,
    pub(crate) sliding_windows: Vec<[usize; 2]>,
    pub(crate) head_dim: Option<usize>,
}

impl MoonshineStreamingEncoderConfig {
    pub(crate) fn frame_len(&self) -> usize {
        ((self.sample_rate as f32 * self.frame_ms) / 1000.0).round() as usize
    }

    pub(crate) fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads.max(1))
    }

}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct MoonshineRopeParameters {
    #[serde(default)]
    pub(crate) rope_theta: Option<f32>,
    #[serde(default)]
    pub(crate) partial_rotary_factor: Option<f32>,
}
