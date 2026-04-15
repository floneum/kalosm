//! Calculates the embeddings for a given input.
//!
//! Bert embeddings contain word embeddings, embeddings about the token type and position information.

use fusor::layers::{Embedding, LayerNorm};
use fusor::{Device, VarBuilder};
use fusor::{Result, Tensor};

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L180
pub(crate) struct BertEmbeddings {
    word_embeddings: Embedding<f32>,
    position_embeddings: Option<Embedding<f32>>,
    token_type_embeddings: Embedding<f32>,
    layer_norm: LayerNorm<1, f32>,
    span: tracing::Span,
}

impl BertEmbeddings {
    pub(crate) fn load(
        device: &Device,
        vb: &mut VarBuilder,
        config: &super::Config,
    ) -> Result<Self> {
        let word_embeddings = Embedding::load(device, &mut vb.pp("token_embd"))?;
        let position_embeddings = Embedding::load(device, &mut vb.pp("position_embd"))?;
        let token_type_embeddings = Embedding::load(device, &mut vb.pp("token_types"))?;
        let layer_norm = LayerNorm::load(
            device,
            &mut vb.pp("token_embd_norm"),
            config.layer_norm_eps as _,
        )?;
        Ok(Self {
            word_embeddings,
            position_embeddings: Some(position_embeddings),
            token_type_embeddings,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "embeddings"),
        })
    }

    pub(crate) fn forward(
        &self,
        input_ids: &Tensor<2, u32>,
        token_type_ids: &Tensor<2, u32>,
    ) -> Tensor<3, f32> {
        let _enter = self.span.enter();
        let [_bsize, seq_len] = input_ids.shape();
        let input_embeddings = self.word_embeddings.forward(input_ids);
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids);
        let mut embeddings: Tensor<3, f32> =
            (&input_embeddings + &token_type_embeddings).to_concrete();
        if let Some(position_embeddings) = &self.position_embeddings {
            let device = input_ids.device();
            let position_ids = fusor::arange(&device, 0u32, seq_len as u32);
            let pos_emb = position_embeddings.forward(&position_ids);
            embeddings = embeddings.add_(&pos_emb)
        }
        self.layer_norm.forward(&embeddings)
    }

    pub(crate) fn embedding_dim(&self) -> usize {
        self.word_embeddings.embedding_dim()
    }

    pub(crate) fn max_seq_len(&self) -> usize {
        self.position_embeddings
            .as_ref()
            .map(|p| p.num_embeddings())
            .unwrap_or(0)
    }
}
