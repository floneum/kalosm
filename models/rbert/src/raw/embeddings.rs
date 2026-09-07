//! Calculates the embeddings for a given input.
//!
//! Bert embeddings contain word embeddings, embeddings about the token type and position information.

use fusor::layers::{Embedding, LayerNorm};
use fusor::{Device, Result, Tensor, VarBuilder};

// https://github.com/huggingface/transformers/blob/6eedfa6dd15dc1e22a55ae036f681914e5a0d9a1/src/transformers/models/bert/modeling_bert.py#L180
pub(crate) struct BertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Option<Embedding>,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    span: tracing::Span,
}

impl BertEmbeddings {
    pub(crate) fn load(device: &Device, vb: &VarBuilder, config: &super::Config) -> Result<Self> {
        let graph = device.graph().handle();
        let word_embeddings = Embedding::load(&vb.pp("token_embd"), graph)?;
        let position_embeddings = Embedding::load(&vb.pp("position_embd"), graph)?;
        let token_type_embeddings = Embedding::load(&vb.pp("token_types"), graph)?;
        let layer_norm = LayerNorm::load(
            &vb.pp("token_embd_norm"),
            graph,
            config.layer_norm_eps as f32,
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
    ) -> Tensor<3> {
        let _enter = self.span.enter();
        let seq_len = input_ids
            .extent(1)
            .as_const()
            .expect("input ids have a const seq len");
        let input_embeddings: Tensor<3> = self.word_embeddings.forward(input_ids);
        let token_type_embeddings: Tensor<3> = self.token_type_embeddings.forward(token_type_ids);
        let mut embeddings = input_embeddings.add(&token_type_embeddings);
        if let Some(position_embeddings) = &self.position_embeddings {
            let position_ids =
                Tensor::<1, u32>::arange(&input_ids.device(), 0.0f32, seq_len as f32);
            let pos_emb: Tensor<2> = position_embeddings.forward(&position_ids);
            // `[seq, hidden]` broadcasts right-aligned onto `[batch, seq, hidden]`.
            embeddings = embeddings.add_(&pos_emb);
        }
        self.layer_norm.forward(&embeddings)
    }

    pub(crate) fn embedding_dim(&self) -> usize {
        self.word_embeddings
            .embedding_dim()
            .as_const()
            .expect("embedding dim is const") as usize
    }

    pub(crate) fn max_seq_len(&self) -> usize {
        self.position_embeddings
            .as_ref()
            .and_then(|p| p.num_embeddings().as_const())
            .unwrap_or(0) as usize
    }
}
