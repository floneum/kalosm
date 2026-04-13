//! TwoWayTransformer for cross-attention between queries and image embeddings.

use fusor::layers::{LayerNorm, Linear};
use fusor::{ConcreteTensor, Device, Tensor, TensorBacking, VarBuilder};

use super::{Activation, MlpBlock, Result};

struct Attention {
    q_proj: Linear<f32>,
    k_proj: Linear<f32>,
    v_proj: Linear<f32>,
    out_proj: Linear<f32>,
    num_heads: usize,
}

impl Attention {
    fn load(
        device: &Device,
        vb: &mut VarBuilder,
        _embedding_dim: usize,
        num_heads: usize,
        _downsample_rate: usize,
    ) -> Result<Self> {
        let q_proj = Linear::load(device, &mut vb.pp("q_proj"))?;
        let k_proj = Linear::load(device, &mut vb.pp("k_proj"))?;
        let v_proj = Linear::load(device, &mut vb.pp("v_proj"))?;
        let out_proj = Linear::load(device, &mut vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
        })
    }

    fn separate_heads(&self, x: &Tensor<3, f32>) -> Tensor<4, f32, ConcreteTensor<f32, 4>> {
        let shape = x.shape();
        let b = shape[0];
        let n = shape[1];
        let c = shape[2];
        let c_per_head = c / self.num_heads;
        x.reshape([b, n, self.num_heads, c_per_head])
            .transpose(1, 2)
            .to_concrete()
    }

    fn recombine_heads(&self, x: &Tensor<4, f32>) -> Tensor<3, f32, ConcreteTensor<f32, 3>> {
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
        q: &Tensor<3, f32, impl TensorBacking<3, Elem = f32>>,
        k: &Tensor<3, f32, impl TensorBacking<3, Elem = f32>>,
        v: &Tensor<3, f32, impl TensorBacking<3, Elem = f32>>,
    ) -> Tensor<3, f32> {
        let q = self.q_proj.forward(&q);
        let k = self.k_proj.forward(&k);
        let v = self.v_proj.forward(&v);

        let q = self.separate_heads(&q);
        let k = self.separate_heads(&k);
        let v = self.separate_heads(&v);

        let c_per_head = q.shape()[3];
        let scale = 1.0 / (c_per_head as f32).sqrt();

        // attn = (q * scale) @ k^T
        let q_scaled = q.mul_scalar(scale);
        let attn = q_scaled.mat_mul(&k.transpose(2, 3));
        let attn: Tensor<4, f32> = attn.softmax_last_dim::<3>();

        let out = attn.mat_mul(&v);
        let out = self.recombine_heads(&out);
        self.out_proj.forward(&out)
    }
}

struct TwoWayAttentionBlock {
    self_attn: Attention,
    norm1: LayerNorm<1, f32>,
    cross_attn_token_to_image: Attention,
    norm2: LayerNorm<1, f32>,
    mlp: MlpBlock,
    norm3: LayerNorm<1, f32>,
    norm4: LayerNorm<1, f32>,
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
        let norm1 = LayerNorm::load(device, &mut vb.pp("norm1"), 1e-5)?;
        let norm2 = LayerNorm::load(device, &mut vb.pp("norm2"), 1e-5)?;
        let norm3 = LayerNorm::load(device, &mut vb.pp("norm3"), 1e-5)?;
        let norm4 = LayerNorm::load(device, &mut vb.pp("norm4"), 1e-5)?;
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
            embedding_dim,
            mlp_dim,
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
        queries: &Tensor<3, f32, impl TensorBacking<3, Elem = f32>>,
        keys: &Tensor<3, f32, impl TensorBacking<3, Elem = f32>>,
        query_pe: &Tensor<3, f32, impl TensorBacking<3, Elem = f32>>,
        key_pe: &Tensor<3, f32, impl TensorBacking<3, Elem = f32>>,
    ) -> (Tensor<3, f32>, Tensor<3, f32>) {
        // Self attention block
        let queries: Tensor<3, f32> = if self.skip_first_layer_pe {
            self.self_attn.forward(queries, queries, queries)
        } else {
            let q = (queries + query_pe);
            let attn_out = self.self_attn.forward(&q, &q, queries);
            (queries + attn_out).to_concrete()
        };
        let queries = self.norm1.forward(&queries);

        // Cross attention block, tokens attending to image embedding
        let q = (&queries + query_pe);
        let k = (keys + key_pe);
        let attn_out = self.cross_attn_token_to_image.forward(&q, &k, keys);
        let queries = (&queries + attn_out);
        let queries = self.norm2.forward(&queries);

        // MLP block
        let mlp_out = self.mlp.forward(&queries);
        let queries = (&queries + mlp_out);
        let queries = self.norm3.forward(&queries);

        // Cross attention block, image embedding attending to tokens
        let q = (&queries + query_pe);
        let k = (keys + key_pe);
        let attn_out = self.cross_attn_image_to_token.forward(&k, &q, &queries);
        let keys = (keys + attn_out);
        let keys = self.norm4.forward(&keys);

        (queries.to_concrete(), keys.to_concrete())
    }
}

pub struct TwoWayTransformer {
    layers: Vec<TwoWayAttentionBlock>,
    final_attn_token_to_image: Attention,
    norm_final_attn: LayerNorm<1, f32>,
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
                &mut vb.pp(&format!("layers.{i}")),
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
        let norm_final_attn = LayerNorm::load(device, &mut vb.pp("norm_final_attn"), 1e-5)?;
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

        let q = (&queries + point_embedding);
        let k = (&keys + &image_pe);
        let attn_out = self.final_attn_token_to_image.forward(&q, &k, &keys);
        let queries = (queries + attn_out);
        let queries = self.norm_final_attn.forward(&queries);

        (queries.to_concrete(), keys)
    }
}
