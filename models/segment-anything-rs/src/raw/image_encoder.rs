//! ViT-based image encoder for SAM.

use fusor2::graph::Graph;
use fusor2::layers::{ConvNd, LayerNorm, Linear};
use fusor2::tensor::Tensor;
use fusor2_gguf::VarBuilder;

use super::{channel_layer_norm, dims, linear, load_dense, udim, Activation, MlpBlock, Result};

fn conv2d(
    vb: &VarBuilder,
    graph: &Graph,
    bias: bool,
    stride: u32,
    padding: u32,
) -> Result<ConvNd> {
    let mut conv = ConvNd::load(vb, graph.handle(), bias)?;
    conv.stride = [stride, stride].into_iter().collect();
    conv.padding = [padding, padding].into_iter().collect();
    Ok(conv)
}

struct PatchEmbed {
    proj: ConvNd,
}

impl PatchEmbed {
    fn load(graph: &Graph, vb: &VarBuilder, patch_size: usize, padding: usize) -> Result<Self> {
        let proj = conv2d(
            &vb.pp("proj"),
            graph,
            true,
            patch_size as u32,
            padding as u32,
        )?;
        Ok(Self { proj })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let out = self.proj.forward(xs)?;
        // (B, C, H, W) -> (B, H, W, C)
        out.permute(&[0, 2, 3, 1])
    }
}

struct Attention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    scale: f32,
    rel_pos_h: Option<Tensor>,
    rel_pos_w: Option<Tensor>,
}

impl Attention {
    fn load(
        graph: &Graph,
        vb: &VarBuilder,
        dim: usize,
        num_heads: usize,
        use_rel_pos: bool,
    ) -> Result<Self> {
        let qkv = linear(&vb.pp("qkv"), graph)?;
        let proj = linear(&vb.pp("proj"), graph)?;
        let head_dim = dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let (rel_pos_h, rel_pos_w) = if use_rel_pos {
            let h = load_dense(vb, graph, "rel_pos_h")?;
            let w = load_dense(vb, graph, "rel_pos_w")?;
            (Some(h), Some(w))
        } else {
            (None, None)
        };
        Ok(Self {
            qkv,
            proj,
            num_heads,
            scale,
            rel_pos_h,
            rel_pos_w,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let b = udim(xs, 0);
        let h = udim(xs, 1);
        let w = udim(xs, 2);
        let c = udim(xs, 3);

        // Flatten to (b, h*w, c) for linear
        let flat = xs.reshape_dims(&dims(&[b, h * w, c]))?;
        let qkv = self.qkv.forward(&flat)?;

        // Reshape to (b, h*w, 3, num_heads, c/num_heads)
        let c_per_head = c / self.num_heads;
        let qkv = qkv.reshape_dims(&dims(&[b, h * w, 3, self.num_heads, c_per_head]))?;

        // Permute to (3, b, num_heads, h*w, c/num_heads) then reshape
        // -> (3, b*num_heads, h*w, c/num_heads)
        let qkv = qkv.permute(&[2, 0, 3, 1, 4])?;
        let qkv = qkv.reshape_dims(&dims(&[3, b * self.num_heads, h * w, c_per_head]))?;

        let q = qkv
            .narrow(0, 0, 1)?
            .reshape_dims(&dims(&[b * self.num_heads, h * w, c_per_head]))?;
        let k = qkv
            .narrow(0, 1, 1)?
            .reshape_dims(&dims(&[b * self.num_heads, h * w, c_per_head]))?;
        let v = qkv
            .narrow(0, 2, 1)?
            .reshape_dims(&dims(&[b * self.num_heads, h * w, c_per_head]))?;

        // attn = (q * scale) @ k^T
        let q_scaled = q.mul_scalar(self.scale)?;
        let attn = q_scaled.matmul_t(&k)?;

        // Add relative position bias
        let attn = self.add_decomposed_rel_pos(attn, &q, (h, w), (h, w))?;

        // Softmax
        let attn = attn.softmax_last_dim()?;

        // attn @ v
        let attn = attn.matmul(&v)?;

        // Reshape back to (b, num_heads, h, w, c_per_head), permute to
        // (b, h, w, num_heads, c_per_head), then flatten to (b, h*w, c)
        let attn = attn.reshape_dims(&dims(&[b, self.num_heads, h, w, c_per_head]))?;
        let attn = attn.permute(&[0, 2, 3, 1, 4])?;
        let attn = attn.reshape_dims(&dims(&[b, h * w, c]))?;

        let out = self.proj.forward(&attn)?;
        out.reshape_dims(&dims(&[b, h, w, c]))
    }

    fn add_decomposed_rel_pos(
        &self,
        attn: Tensor,
        q: &Tensor,
        (q_h, q_w): (usize, usize),
        (k_h, k_w): (usize, usize),
    ) -> Result<Tensor> {
        match (&self.rel_pos_h, &self.rel_pos_w) {
            (Some(rel_pos_h), Some(rel_pos_w)) => {
                let r_h = get_rel_pos(q_h, k_h, rel_pos_h)?;
                let r_w = get_rel_pos(q_w, k_w, rel_pos_w)?;

                let b_nh = udim(q, 0); // b * num_heads
                let dim = udim(q, 2);

                let r_q = q.reshape_dims(&dims(&[b_nh, q_h, q_w, dim]))?;

                // rel_h = r_q @ r_h^T: (b_nh, q_h, q_w, dim) @ (q_h, k_h, dim)^T -> (b_nh, q_h, q_w, k_h)
                let r_h_t = r_h.transpose(1, 2)?; // (q_h, dim, k_h)
                let r_h_broadcast = r_h_t
                    .reshape_dims(&dims(&[1, q_h, dim, k_h]))?
                    .broadcast_as(&dims(&[b_nh, q_h, dim, k_h]))?;
                let rel_h = r_q.matmul(&r_h_broadcast)?;

                // rel_w: transpose r_q to (b_nh, q_w, q_h, dim), matmul with r_w^T, transpose back
                let r_w_t = r_w.transpose(1, 2)?; // (q_w, dim, k_w)
                let r_w_broadcast = r_w_t
                    .reshape_dims(&dims(&[1, q_w, dim, k_w]))?
                    .broadcast_as(&dims(&[b_nh, q_w, dim, k_w]))?;
                let r_q_t = r_q.transpose(1, 2)?; // (b_nh, q_w, q_h, dim)
                let rel_w = r_q_t.matmul(&r_w_broadcast)?;
                let rel_w = rel_w.transpose(1, 2)?; // (b_nh, q_h, q_w, k_w)

                // attn = attn.reshape(b_nh, q_h, q_w, k_h, k_w) + rel_h.unsqueeze(4) + rel_w.unsqueeze(3)
                let attn_5d = attn.reshape_dims(&dims(&[b_nh, q_h, q_w, k_h, k_w]))?;
                let rel_h_5d = rel_h
                    .reshape_dims(&dims(&[b_nh, q_h, q_w, k_h, 1]))?
                    .broadcast_as(&dims(&[b_nh, q_h, q_w, k_h, k_w]))?;
                let rel_w_5d = rel_w
                    .reshape_dims(&dims(&[b_nh, q_h, q_w, 1, k_w]))?
                    .broadcast_as(&dims(&[b_nh, q_h, q_w, k_h, k_w]))?;

                let result = attn_5d.add(&rel_h_5d)?.add(&rel_w_5d)?;
                result.reshape_dims(&dims(&[b_nh, q_h * q_w, k_h * k_w]))
            }
            _ => Ok(attn),
        }
    }
}

fn get_rel_pos(q_size: usize, k_size: usize, rel_pos: &Tensor) -> Result<Tensor> {
    // For SAM, q_size == k_size and rel_pos has shape (2*q_size-1, head_dim)
    let q_scale = f32::max(1.0, k_size as f32 / q_size as f32);
    let k_scale = f32::max(1.0, q_size as f32 / k_size as f32);
    let offset = (k_size as f32 - 1.0) * q_scale;

    // Compute relative coordinate indices entirely on the CPU - these are
    // deterministic integer offsets derived from q_size and k_size.
    let rc_data: Vec<u32> = (0..q_size)
        .flat_map(|q| {
            (0..k_size).map(move |k| (q as f32 * q_scale - k as f32 * k_scale + offset) as u32)
        })
        .collect();
    let relative_coords_u32 =
        Tensor::from_elements(rel_pos.graph(), &dims(&[q_size * k_size]), &rc_data)?;

    // index_select from rel_pos
    let selected = rel_pos.index_select(0, &relative_coords_u32)?;
    let head_dim = udim(rel_pos, 1);
    selected.reshape_dims(&dims(&[q_size, k_size, head_dim]))
}

struct Block {
    norm1: LayerNorm,
    attn: Attention,
    norm2: LayerNorm,
    mlp: MlpBlock,
    window_size: usize,
}

impl Block {
    fn load(
        graph: &Graph,
        vb: &VarBuilder,
        dim: usize,
        num_heads: usize,
        use_rel_pos: bool,
        window_size: usize,
    ) -> Result<Self> {
        let norm1 = LayerNorm::load(&vb.pp("norm1"), graph.handle(), 1e-6)?;
        let norm2 = LayerNorm::load(&vb.pp("norm2"), graph.handle(), 1e-6)?;
        let attn = Attention::load(graph, &vb.pp("attn"), dim, num_heads, use_rel_pos)?;
        let mlp = MlpBlock::load(
            graph,
            &vb.pp("mlp"),
            Some(dim),
            Some(dim * 4),
            Activation::Gelu,
        )?;
        Ok(Self {
            norm1,
            attn,
            norm2,
            mlp,
            window_size,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let shortcut = xs;
        let h = udim(xs, 1);
        let w = udim(xs, 2);

        // LayerNorm over the last (channel) dim of BHWC
        let xs = self.norm1.forward(xs)?;

        let (xs, pad_hw) = if self.window_size > 0 {
            window_partition(&xs, self.window_size)?
        } else {
            (xs, (0, 0))
        };
        let xs = self.attn.forward(&xs)?;
        let xs = if self.window_size > 0 {
            window_unpartition(&xs, self.window_size, pad_hw, (h, w))?
        } else {
            xs
        };

        let xs = xs.add(shortcut)?;

        // MLP
        let mlp_in = self.norm2.forward(&xs)?;
        let b = udim(&mlp_in, 0);
        let c = udim(&mlp_in, 3);
        let mlp_flat = mlp_in.reshape_dims(&dims(&[b, h * w, c]))?;
        let mlp_out = self.mlp.forward(&mlp_flat)?;
        let mlp_out = mlp_out.reshape_dims(&dims(&[b, h, w, c]))?;

        xs.add(&mlp_out)
    }
}

fn window_partition(xs: &Tensor, window_size: usize) -> Result<(Tensor, (usize, usize))> {
    let b = udim(xs, 0);
    let h = udim(xs, 1);
    let w = udim(xs, 2);
    let c = udim(xs, 3);

    let pad_h = (window_size - h % window_size) % window_size;
    let pad_w = (window_size - w % window_size) % window_size;

    let xs = if pad_h > 0 {
        xs.pad_with_zeros(1, 0, pad_h)?
    } else {
        xs.clone()
    };
    let xs = if pad_w > 0 {
        xs.pad_with_zeros(2, 0, pad_w)?
    } else {
        xs
    };

    let h_p = h + pad_h;
    let w_p = w + pad_w;

    // (b, h_p/ws, ws, w_p/ws, ws, c) -> transpose(2,3) -> (b, n_h, n_w, ws, ws, c)
    // -> flatten first 3 dims -> (b * n_windows, ws, ws, c)
    let n_h = h_p / window_size;
    let n_w = w_p / window_size;
    let windows = xs
        .reshape_dims(&dims(&[b, n_h, window_size, n_w, window_size, c]))?
        .transpose(2, 3)?
        .reshape_dims(&dims(&[b * n_h * n_w, window_size, window_size, c]))?;

    Ok((windows, (h_p, w_p)))
}

fn window_unpartition(
    windows: &Tensor,
    window_size: usize,
    (h_p, w_p): (usize, usize),
    (h, w): (usize, usize),
) -> Result<Tensor> {
    let total = udim(windows, 0);
    let c = udim(windows, 3);
    let n_h = h_p / window_size;
    let n_w = w_p / window_size;
    let b = total / (n_h * n_w);

    let xs = windows
        .reshape_dims(&dims(&[b, n_h, n_w, window_size, window_size, c]))?
        .transpose(2, 3)? // (b, n_h, ws, n_w, ws, c)
        .reshape_dims(&dims(&[b, h_p, w_p, c]))?;

    let xs = if h_p > h { xs.narrow(1, 0, h)? } else { xs };
    let xs = if w_p > w { xs.narrow(2, 0, w)? } else { xs };
    Ok(xs)
}

/// Standard ViT-B/H/L image encoder used by the upstream SAM checkpoints.
///
/// `forward` takes a `(B, 3, IMAGE_SIZE, IMAGE_SIZE)` preprocessed input and
/// returns `(B, prompt_embed_dim, IMAGE_SIZE/16, IMAGE_SIZE/16)` features.
pub struct ImageEncoderViT {
    patch_embed: PatchEmbed,
    blocks: Vec<Block>,
    neck_conv1: ConvNd,
    neck_ln1: LayerNorm,
    neck_conv2: ConvNd,
    neck_ln2: LayerNorm,
    pos_embed: Option<Tensor>,
}

impl ImageEncoderViT {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        graph: &Graph,
        vb: &VarBuilder,
        _img_size: usize,
        patch_size: usize,
        embed_dim: usize,
        depth: usize,
        num_heads: usize,
        _out_chans: usize,
        use_rel_pos: bool,
        use_abs_pos: bool,
        window_size: usize,
        global_attn_indexes: &[usize],
    ) -> Result<Self> {
        let patch_embed = PatchEmbed::load(graph, &vb.pp("patch_embed"), patch_size, 0)?;

        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let ws = if global_attn_indexes.contains(&i) {
                0
            } else {
                window_size
            };
            let block = Block::load(
                graph,
                &vb.pp(format!("blocks.{i}")),
                embed_dim,
                num_heads,
                use_rel_pos,
                ws,
            )?;
            blocks.push(block);
        }

        let neck_conv1 = conv2d(&vb.pp("neck.0"), graph, false, 1, 0)?;
        let neck_ln1 = LayerNorm::load(&vb.pp("neck.1"), graph.handle(), 1e-6)?;
        let neck_conv2 = conv2d(&vb.pp("neck.2"), graph, false, 1, 1)?;
        let neck_ln2 = LayerNorm::load(&vb.pp("neck.3"), graph.handle(), 1e-6)?;

        let pos_embed = if use_abs_pos {
            Some(load_dense(vb, graph, "pos_embed")?)
        } else {
            None
        };

        Ok(Self {
            patch_embed,
            blocks,
            neck_conv1,
            neck_ln1,
            neck_conv2,
            neck_ln2,
            pos_embed,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.patch_embed.forward(xs)?; // (B, H, W, C)

        let mut xs = match &self.pos_embed {
            Some(pos_embed) => xs.add_(pos_embed)?,
            None => xs,
        };

        for block in &self.blocks {
            xs = block.forward(&xs)?;
        }

        // (B, H, W, C) -> (B, C, H, W)
        let xs = xs.permute(&[0, 3, 1, 2])?;

        // Neck. The neck LayerNorms are Meta's LayerNorm2d: over channels.
        let xs = self.neck_conv1.forward(&xs)?;
        let xs = channel_layer_norm(&self.neck_ln1, &xs)?;
        let xs = self.neck_conv2.forward(&xs)?;
        channel_layer_norm(&self.neck_ln2, &xs)
    }
}
