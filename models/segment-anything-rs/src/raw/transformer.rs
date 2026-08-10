//! TwoWayTransformer for cross-attention between queries and image embeddings.

use fusor2::cache::MaskKind;
use fusor2::layers::{LayerNorm, Linear};
use fusor2::{Device, Tensor};
use fusor2_gguf::VarBuilder;

use super::{linear, Activation, MlpBlock, Result};

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
}

impl Attention {
    /// Load Q/K/V/out projections. `downsample_rate` matches the SAM
    /// constructor signature but is currently unused (the projection layout
    /// encodes the downsample).
    fn load(
        device: &Device,
        vb: &VarBuilder,
        embedding_dim: usize,
        num_heads: usize,
        _downsample_rate: usize,
    ) -> Result<Self> {
        let q_proj = linear(&vb.pp("q_proj"), device)?;
        let k_proj = linear(&vb.pp("k_proj"), device)?;
        let v_proj = linear(&vb.pp("v_proj"), device)?;
        let out_proj = linear(&vb.pp("out_proj"), device)?;
        debug_assert_eq!(
            q_proj.in_features().as_const(),
            Some(embedding_dim as u64),
            "Q proj dim mismatch"
        );
        debug_assert_eq!(
            k_proj.in_features().as_const(),
            Some(embedding_dim as u64),
            "K proj dim mismatch"
        );
        debug_assert_eq!(
            v_proj.in_features().as_const(),
            Some(embedding_dim as u64),
            "V proj dim mismatch"
        );
        debug_assert_eq!(
            out_proj.out_features().as_const(),
            Some(embedding_dim as u64),
            "out proj mismatch"
        );
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
        })
    }

    fn separate_heads(&self, x: &Tensor<3>) -> Tensor<4> {
        let [b, n, c] = x.shape();
        let c_per_head = c / self.num_heads;
        x.reshape([b, n, self.num_heads, c_per_head]).transpose(1, 2)
    }

    fn recombine_heads(&self, x: &Tensor<4>) -> Tensor<3> {
        let [b, n_heads, n_tokens, c_per_head] = x.shape();
        x.transpose(1, 2)
            .reshape([b, n_tokens, n_heads * c_per_head])
    }

    fn forward(&self, q: &Tensor<3>, k: &Tensor<3>, v: &Tensor<3>) -> Tensor<3> {
        let q = self.q_proj.forward(q);
        let k = self.k_proj.forward(k);
        let v = self.v_proj.forward(v);

        let q = self.separate_heads(&q);
        let k = self.separate_heads(&k);
        let v = self.separate_heads(&v);

        let c_per_head = q.shape()[3];
        let scale = 1.0 / (c_per_head as f32).sqrt();

        let out = q.attention(&k, &v, MaskKind::None, Some(scale));
        self.out_proj.forward(&self.recombine_heads(&out))
    }
}

struct TwoWayAttentionBlock {
    self_attn: Attention,
    norm1: LayerNorm,
    cross_attn_token_to_image: Attention,
    norm2: LayerNorm,
    mlp: MlpBlock,
    norm3: LayerNorm,
    norm4: LayerNorm,
    cross_attn_image_to_token: Attention,
    skip_first_layer_pe: bool,
}

impl TwoWayAttentionBlock {
    fn load(
        device: &Device,
        vb: &VarBuilder,
        embedding_dim: usize,
        num_heads: usize,
        mlp_dim: usize,
        skip_first_layer_pe: bool,
    ) -> Result<Self> {
        let graph = device.graph().handle();
        let norm1 = LayerNorm::load(&vb.pp("norm1"), graph, 1e-5)?;
        let norm2 = LayerNorm::load(&vb.pp("norm2"), graph, 1e-5)?;
        let norm3 = LayerNorm::load(&vb.pp("norm3"), graph, 1e-5)?;
        let norm4 = LayerNorm::load(&vb.pp("norm4"), graph, 1e-5)?;
        let self_attn = Attention::load(device, &vb.pp("self_attn"), embedding_dim, num_heads, 1)?;
        let cross_attn_token_to_image = Attention::load(
            device,
            &vb.pp("cross_attn_token_to_image"),
            embedding_dim,
            num_heads,
            2,
        )?;
        let cross_attn_image_to_token = Attention::load(
            device,
            &vb.pp("cross_attn_image_to_token"),
            embedding_dim,
            num_heads,
            2,
        )?;
        let mlp = MlpBlock::load(
            device,
            &vb.pp("mlp"),
            Some(embedding_dim),
            Some(mlp_dim),
            Activation::Relu,
        )?;
        Ok(Self {
            self_attn,
            norm1,
            cross_attn_image_to_token,
            norm2,
            mlp,
            norm3,
            norm4,
            cross_attn_token_to_image,
            skip_first_layer_pe,
        })
    }

    fn forward(
        &self,
        queries: &Tensor<3>,
        keys: &Tensor<3>,
        query_pe: &Tensor<3>,
        key_pe: &Tensor<3>,
    ) -> (Tensor<3>, Tensor<3>) {
        // Self attention block
        let queries = if self.skip_first_layer_pe {
            self.self_attn.forward(queries, queries, queries)
        } else {
            let q = queries.add(query_pe);
            let attn_out = self.self_attn.forward(&q, &q, queries);
            queries.add(&attn_out)
        };
        let queries = self.norm1.forward(&queries);

        // Cross attention block, tokens attending to image embedding
        let q = queries.add(query_pe);
        let k = keys.add(key_pe);
        let attn_out = self.cross_attn_token_to_image.forward(&q, &k, keys);
        let queries = self.norm2.forward(&queries.add(&attn_out));

        // MLP block
        let mlp_out = self.mlp.forward(&queries);
        let queries = self.norm3.forward(&queries.add(&mlp_out));

        // Cross attention block, image embedding attending to tokens
        let q = queries.add(query_pe);
        let k = keys.add(key_pe);
        let attn_out = self.cross_attn_image_to_token.forward(&k, &q, &queries);
        let keys = self.norm4.forward(&keys.add(&attn_out));

        (queries, keys)
    }
}

/// Two-way attention transformer used inside `MaskDecoder`. Alternates
/// token-to-image and image-to-token cross-attention. `forward` takes
/// `(image_embedding: (B, C, H, W), image_pe: (B, C, H, W), point_embedding:
/// (B, N, C))` and returns the updated `(queries, keys)` 3D tensors.
pub struct TwoWayTransformer {
    layers: Vec<TwoWayAttentionBlock>,
    final_attn_token_to_image: Attention,
    norm_final_attn: LayerNorm,
}

impl TwoWayTransformer {
    pub fn load(
        device: &Device,
        vb: &VarBuilder,
        depth: usize,
        embedding_dim: usize,
        num_heads: usize,
        mlp_dim: usize,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(depth);
        for i in 0..depth {
            let layer = TwoWayAttentionBlock::load(
                device,
                &vb.pp(format!("layers.{i}")),
                embedding_dim,
                num_heads,
                mlp_dim,
                i == 0,
            )?;
            layers.push(layer);
        }
        let final_attn_token_to_image = Attention::load(
            device,
            &vb.pp("final_attn_token_to_image"),
            embedding_dim,
            num_heads,
            2,
        )?;
        let norm_final_attn =
            LayerNorm::load(&vb.pp("norm_final_attn"), device.graph().handle(), 1e-5)?;
        Ok(Self {
            layers,
            final_attn_token_to_image,
            norm_final_attn,
        })
    }

    pub fn forward(
        &self,
        image_embedding: &Tensor<4>,
        image_pe: &Tensor<4>,
        point_embedding: &Tensor<3>,
    ) -> (Tensor<3>, Tensor<3>) {
        let [b, c, h, w] = image_embedding.shape();

        // Flatten spatial dims and permute: (B, C, H, W) -> (B, H*W, C)
        let image_embedding = image_embedding.reshape([b, c, h * w]).transpose(1, 2);
        let image_pe = image_pe.reshape([b, c, h * w]).transpose(1, 2);

        let mut queries = point_embedding.clone();
        let mut keys = image_embedding;

        for layer in &self.layers {
            (queries, keys) = layer.forward(&queries, &keys, point_embedding, &image_pe);
        }

        let q = queries.add(point_embedding);
        let k = keys.add(&image_pe);
        let attn_out = self.final_attn_token_to_image.forward(&q, &k, &keys);
        let queries = self.norm_final_attn.forward(&queries.add(&attn_out));

        (queries, keys)
    }
}
