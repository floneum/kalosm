//! TwoWayTransformer for cross-attention between queries and image embeddings.

use fusor::layers::{LayerNormNd, Linear};
use fusor::{Concrete, Device, Fusion, Tensor, VarBuilder};

use super::{Activation, MlpBlock, Result};

struct Attention {
    q_proj: Linear<f32>,
    k_proj: Linear<f32>,
    v_proj: Linear<f32>,
    out_proj: Linear<f32>,
    num_heads: usize,
}

impl Attention {
    /// Load Q/K/V/out projections. `embedding_dim` is asserted against the
    /// loaded shapes; `downsample_rate` matches the SAM constructor signature
    /// but is currently unused (the projection layout encodes the downsample).
    fn load(
        device: &Device,
        vb: &mut VarBuilder,
        embedding_dim: usize,
        num_heads: usize,
        _downsample_rate: usize,
    ) -> Result<Self> {
        let q_proj = Linear::load(device, &mut vb.pp("q_proj"))?;
        let k_proj = Linear::load(device, &mut vb.pp("k_proj"))?;
        let v_proj = Linear::load(device, &mut vb.pp("v_proj"))?;
        let out_proj = Linear::load(device, &mut vb.pp("out_proj"))?;
        debug_assert_eq!(q_proj.in_features(), embedding_dim, "Q proj dim mismatch");
        debug_assert_eq!(k_proj.in_features(), embedding_dim, "K proj dim mismatch");
        debug_assert_eq!(v_proj.in_features(), embedding_dim, "V proj dim mismatch");
        debug_assert_eq!(out_proj.out_features(), embedding_dim, "out proj mismatch");
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
        })
    }

    fn separate_heads(&self, x: &Tensor<3, f32>) -> Tensor<4, f32, Concrete<f32, 4>> {
        let shape = x.shape();
        let b = shape[0];
        let n = shape[1];
        let c = shape[2];
        let c_per_head = c / self.num_heads;
        x.reshape([b, n, self.num_heads, c_per_head])
            .transpose(1, 2)
            .to_concrete()
    }

    fn recombine_heads(&self, x: &Tensor<4, f32>) -> Tensor<3, f32, Concrete<f32, 3>> {
        let shape = x.shape();
        let b = shape[0];
        let n_heads = shape[1];
        let n_tokens = shape[2];
        let c_per_head = shape[3];
        x.transpose(1, 2)
            .reshape([b, n_tokens, n_heads * c_per_head])
            .to_concrete()
    }

    fn forward(
        &self,
        q: &Tensor<3, f32, impl Fusion<3, f32>>,
        k: &Tensor<3, f32, impl Fusion<3, f32>>,
        v: &Tensor<3, f32, impl Fusion<3, f32>>,
    ) -> Tensor<3, f32> {
        let q = self.q_proj.forward(q);
        let k = self.k_proj.forward(k);
        let v = self.v_proj.forward(v);

        let q = self.separate_heads(&q);
        let k = self.separate_heads(&k);
        let v = self.separate_heads(&v);

        let c_per_head = q.shape()[3];
        let scale = 1.0 / (c_per_head as f32).sqrt();

        let out = q.flash_attention(&k, &v, scale, None);
        let out = self.recombine_heads(&out);
        self.out_proj.forward(&out)
    }
}

struct TwoWayAttentionBlock {
    self_attn: Attention,
    norm1: LayerNormNd<f32>,
    cross_attn_token_to_image: Attention,
    norm2: LayerNormNd<f32>,
    mlp: MlpBlock,
    norm3: LayerNormNd<f32>,
    norm4: LayerNormNd<f32>,
    cross_attn_image_to_token: Attention,
    skip_first_layer_pe: bool,
}

impl TwoWayAttentionBlock {
    fn load(
        device: &Device,
        vb: &mut VarBuilder,
        embedding_dim: usize,
        num_heads: usize,
        mlp_dim: usize,
        skip_first_layer_pe: bool,
    ) -> Result<Self> {
        let norm1 = LayerNormNd::load(device, &mut vb.pp("norm1"), 1e-5)?;
        let norm2 = LayerNormNd::load(device, &mut vb.pp("norm2"), 1e-5)?;
        let norm3 = LayerNormNd::load(device, &mut vb.pp("norm3"), 1e-5)?;
        let norm4 = LayerNormNd::load(device, &mut vb.pp("norm4"), 1e-5)?;
        let self_attn =
            Attention::load(device, &mut vb.pp("self_attn"), embedding_dim, num_heads, 1)?;
        let cross_attn_token_to_image = Attention::load(
            device,
            &mut vb.pp("cross_attn_token_to_image"),
            embedding_dim,
            num_heads,
            2,
        )?;
        let cross_attn_image_to_token = Attention::load(
            device,
            &mut vb.pp("cross_attn_image_to_token"),
            embedding_dim,
            num_heads,
            2,
        )?;
        let mlp = MlpBlock::load(
            device,
            &mut vb.pp("mlp"),
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
        queries: &Tensor<3, f32, impl Fusion<3, f32>>,
        keys: &Tensor<3, f32, impl Fusion<3, f32>>,
        query_pe: &Tensor<3, f32, impl Fusion<3, f32>>,
        key_pe: &Tensor<3, f32, impl Fusion<3, f32>>,
    ) -> (Tensor<3, f32>, Tensor<3, f32>) {
        // Self attention block
        let queries: Tensor<3, f32> = if self.skip_first_layer_pe {
            self.self_attn.forward(queries, queries, queries)
        } else {
            let q: Tensor<3, f32> = (queries + query_pe).to_concrete();
            let attn_out = self.self_attn.forward(&q, &q, queries);
            (queries + attn_out).to_concrete()
        };
        let queries = self.norm1.forward(&queries);

        // Cross attention block, tokens attending to image embedding
        let q: Tensor<3, f32> = (&queries + query_pe).to_concrete();
        let k: Tensor<3, f32> = (keys + key_pe).to_concrete();
        let attn_out = self.cross_attn_token_to_image.forward(&q, &k, keys);
        let queries: Tensor<3, f32> = (&queries + attn_out).to_concrete();
        let queries = self.norm2.forward(&queries);

        // MLP block
        let mlp_out = self.mlp.forward(&queries);
        let queries: Tensor<3, f32> = (&queries + mlp_out).to_concrete();
        let queries = self.norm3.forward(&queries);

        // Cross attention block, image embedding attending to tokens
        let q: Tensor<3, f32> = (&queries + query_pe).to_concrete();
        let k: Tensor<3, f32> = (keys + key_pe).to_concrete();
        let attn_out = self.cross_attn_image_to_token.forward(&k, &q, &queries);
        let keys: Tensor<3, f32> = (keys + attn_out).to_concrete();
        let keys = self.norm4.forward(&keys);

        (queries.to_concrete(), keys.to_concrete())
    }
}

/// Two-way attention transformer used inside `MaskDecoder`. Alternates
/// token→image and image→token cross-attention. `forward` takes
/// `(image_embedding: (B, C, H, W), image_pe: (B, C, H, W), point_embedding:
/// (B, N, C))` and returns the updated `(queries, keys)` 3D tensors.
pub struct TwoWayTransformer {
    layers: Vec<TwoWayAttentionBlock>,
    final_attn_token_to_image: Attention,
    norm_final_attn: LayerNormNd<f32>,
}

impl TwoWayTransformer {
    pub fn load(
        device: &Device,
        vb: &mut VarBuilder,
        depth: usize,
        embedding_dim: usize,
        num_heads: usize,
        mlp_dim: usize,
    ) -> Result<Self> {
        let mut layers = Vec::with_capacity(depth);
        for i in 0..depth {
            let layer = TwoWayAttentionBlock::load(
                device,
                &mut vb.pp(format!("layers.{i}")),
                embedding_dim,
                num_heads,
                mlp_dim,
                i == 0,
            )?;
            layers.push(layer);
        }
        let final_attn_token_to_image = Attention::load(
            device,
            &mut vb.pp("final_attn_token_to_image"),
            embedding_dim,
            num_heads,
            2,
        )?;
        let norm_final_attn = LayerNormNd::load(device, &mut vb.pp("norm_final_attn"), 1e-5)?;
        Ok(Self {
            layers,
            final_attn_token_to_image,
            norm_final_attn,
        })
    }

    pub fn forward(
        &self,
        image_embedding: &Tensor<4, f32>,
        image_pe: &Tensor<4, f32>,
        point_embedding: &Tensor<3, f32>,
    ) -> (Tensor<3, f32>, Tensor<3, f32>) {
        let shape = image_embedding.shape();
        let b = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];

        // Flatten spatial dims and permute: (B, C, H, W) -> (B, H*W, C)
        let image_embedding = image_embedding
            .reshape([b, c, h * w])
            .transpose(1, 2)
            .to_concrete();
        let image_pe = image_pe.reshape([b, c, h * w]);
        let image_pe = image_pe.transpose(1, 2);

        let mut queries = point_embedding.clone();
        let mut keys = image_embedding;

        for layer in &self.layers {
            (queries, keys) = layer.forward(&queries, &keys, point_embedding, &image_pe);
        }

        let q: Tensor<3, f32> = (&queries + point_embedding).to_concrete();
        let k: Tensor<3, f32> = (&keys + &image_pe).to_concrete();
        let attn_out = self.final_attn_token_to_image.forward(&q, &k, &keys);
        let queries: Tensor<3, f32> = (queries + attn_out).to_concrete();
        let queries = self.norm_final_attn.forward(&queries);

        (queries.to_concrete(), keys)
    }
}
