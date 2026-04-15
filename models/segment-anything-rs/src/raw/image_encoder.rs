//! ViT-based image encoder for SAM.

use fusor::layers::{ConvNd, ConvNdConfig, LayerNormNd, Linear};
use fusor::{ConcreteTensor, Device, Tensor, TensorBacking, VarBuilder};

use super::{Activation, MlpBlock, Result};

struct PatchEmbed {
    proj: ConvNd<2, 4, f32>,
}

impl PatchEmbed {
    fn load(
        device: &Device,
        vb: &mut VarBuilder,
        patch_size: usize,
        padding: usize,
    ) -> Result<Self> {
        let cfg = ConvNdConfig {
            padding: [padding, padding],
            stride: [patch_size, patch_size],
            groups: 1,
        };
        let proj = ConvNd::<2, 4, f32>::load(device, &mut vb.pp("proj"), cfg)?;
        Ok(Self { proj })
    }

    fn forward(&self, xs: &Tensor<4, f32, impl TensorBacking<4, Elem = f32>>) -> Tensor<4, f32> {
        let out = self.proj.forward(xs);
        // (B, C, H, W) -> (B, H, W, C)
        let out = out.transpose(1, 2);
        let out = out.transpose(2, 3);
        out.to_concrete()
    }
}

struct Attention {
    qkv: Linear<f32>,
    proj: Linear<f32>,
    num_heads: usize,
    scale: f32,
    rel_pos_h: Option<Tensor<2, f32, ConcreteTensor<f32, 2>>>,
    rel_pos_w: Option<Tensor<2, f32, ConcreteTensor<f32, 2>>>,
}

impl Attention {
    fn load(
        device: &Device,
        vb: &mut VarBuilder,
        dim: usize,
        num_heads: usize,
        use_rel_pos: bool,
        _input_size: (usize, usize),
    ) -> Result<Self> {
        let qkv = Linear::load(device, &mut vb.pp("qkv"))?;
        let proj = Linear::load(device, &mut vb.pp("proj"))?;
        let head_dim = dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let (rel_pos_h, rel_pos_w) = if use_rel_pos {
            let h: Tensor<2, f32> = vb.get("rel_pos_h", device)?.dequantize();
            let w: Tensor<2, f32> = vb.get("rel_pos_w", device)?.dequantize();
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

    fn forward(&self, xs: &Tensor<4, f32>) -> Tensor<4, f32> {
        let shape = xs.shape();
        let b = shape[0];
        let h = shape[1];
        let w = shape[2];
        let c = shape[3];

        // Flatten to (b, h*w, c) for linear
        let flat = xs.reshape([b, h * w, c]);
        let qkv = self.qkv.forward(&flat);

        // Reshape to (b, h*w, 3, num_heads, c/num_heads)
        let c_per_head = c / self.num_heads;
        let qkv = qkv.reshape([b, h * w, 3, self.num_heads, c_per_head]);

        // Permute to (3, b, num_heads, h*w, c/num_heads) then reshape
        // -> (3, b*num_heads, h*w, c/num_heads)
        let qkv = qkv.transpose(1, 2); // (b, 3, h*w, num_heads, c_per_head)
        let qkv = qkv.transpose(0, 1); // (3, b, h*w, num_heads, c_per_head)
        let qkv = qkv.transpose(2, 3); // (3, b, num_heads, h*w, c_per_head);
        let qkv = qkv.reshape([3, b * self.num_heads, h * w, c_per_head]);

        let q = qkv
            .narrow(0, 0, 1)
            .reshape([b * self.num_heads, h * w, c_per_head])
            .to_concrete();
        let k = qkv.narrow(0, 1, 1);
        let k = k.reshape([b * self.num_heads, h * w, c_per_head]);
        let v = qkv.narrow(0, 2, 1);
        let v = v.reshape([b * self.num_heads, h * w, c_per_head]);

        // attn = (q * scale) @ k^T
        let q_scaled = q.mul_scalar(self.scale);
        let attn: Tensor<3, f32> = q_scaled.mat_mul(&k.transpose(1, 2));

        // Add relative position bias
        let attn = self.add_decomposed_rel_pos(attn, &q, (h, w), (h, w));

        // Softmax
        let attn: Tensor<3, f32> = attn.softmax_last_dim::<2>();

        // attn @ v
        let attn = attn.mat_mul(&v);

        // Reshape back to (b, num_heads, h, w, c_per_head)
        let attn = attn.reshape([b, self.num_heads, h, w, c_per_head]);
        // Permute to (b, h, w, num_heads, c_per_head) then reshape to (b, h*w, c)
        let attn = attn.transpose(1, 2); // (b, h, num_heads, w, c_per_head)
        let attn = attn.transpose(2, 3); // (b, h, w, num_heads, c_per_head)
        let attn = attn.reshape([b, h * w, c]);

        let out = self.proj.forward(&attn);
        out.reshape([b, h, w, c]).to_concrete()
    }

    fn add_decomposed_rel_pos(
        &self,
        attn: Tensor<3, f32>,
        q: &Tensor<3, f32>,
        (q_h, q_w): (usize, usize),
        (k_h, k_w): (usize, usize),
    ) -> Tensor<3, f32> {
        match (&self.rel_pos_h, &self.rel_pos_w) {
            (Some(rel_pos_h), Some(rel_pos_w)) => {
                let r_h = get_rel_pos(q_h, k_h, rel_pos_h);
                let r_w = get_rel_pos(q_w, k_w, rel_pos_w);

                let q_shape = q.shape();
                let b_nh = q_shape[0]; // b * num_heads
                let dim = q_shape[2];

                let r_q = q.reshape([b_nh, q_h, q_w, dim]);

                // rel_h = r_q @ r_h^T: (b_nh, q_h, q_w, dim) @ (q_h, k_h, dim)^T -> (b_nh, q_h, q_w, k_h)
                let r_h_t = r_h.transpose(1, 2); // (q_h, dim, k_h)
                let r_h_broadcast = r_h_t.reshape([1, q_h, dim, k_h]);
                let r_h_broadcast = r_h_broadcast.broadcast_as([b_nh, q_h, dim, k_h]);
                let rel_h = r_q.mat_mul(&r_h_broadcast);

                // rel_w: transpose r_q to (b_nh, q_w, q_h, dim), matmul with r_w^T, transpose back
                let r_w_t = r_w.transpose(1, 2); // (q_w, dim, k_w)
                let r_w_broadcast = r_w_t.reshape([1, q_w, dim, k_w]);
                let r_w_broadcast = r_w_broadcast.broadcast_as([b_nh, q_w, dim, k_w]);
                let r_q_t = r_q.transpose(1, 2); // (b_nh, q_w, q_h, dim)
                let rel_w = r_q_t.mat_mul(&r_w_broadcast);
                let rel_w = rel_w.transpose(1, 2); // (b_nh, q_h, q_w, k_w)

                // attn = attn.reshape(b_nh, q_h, q_w, k_h, k_w) + rel_h.unsqueeze(4) + rel_w.unsqueeze(3)
                let attn_5d = attn.reshape([b_nh, q_h, q_w, k_h, k_w]);
                // rel_h: (b_nh, q_h, q_w, k_h) -> (b_nh, q_h, q_w, k_h, 1)
                let rel_h_5d = rel_h.reshape([b_nh, q_h, q_w, k_h, 1]);
                let rel_h_5d = rel_h_5d.broadcast_as([b_nh, q_h, q_w, k_h, k_w]);
                // rel_w: (b_nh, q_h, q_w, k_w) -> (b_nh, q_h, q_w, 1, k_w)
                let rel_w_5d = rel_w.reshape([b_nh, q_h, q_w, 1, k_w]);
                let rel_w_5d = rel_w_5d.broadcast_as([b_nh, q_h, q_w, k_h, k_w]);

                let result = attn_5d + rel_h_5d + rel_w_5d;
                result.reshape([b_nh, q_h * q_w, k_h * k_w]).to_concrete()
            }
            _ => attn,
        }
    }
}

fn get_rel_pos(
    q_size: usize,
    k_size: usize,
    rel_pos: &Tensor<2, f32, ConcreteTensor<f32, 2>>,
) -> Tensor<3, f32, ConcreteTensor<f32, 3>> {
    // For SAM, q_size == k_size and rel_pos has shape (2*q_size-1, head_dim)
    let device = rel_pos.device();

    let q_scale = f32::max(1.0, k_size as f32 / q_size as f32);
    let k_scale = f32::max(1.0, q_size as f32 / k_size as f32);
    let offset = (k_size as f32 - 1.0) * q_scale;

    // Compute relative coordinate indices entirely on the CPU — these are
    // deterministic integer offsets derived from q_size and k_size.
    let rc_data: Vec<u32> = (0..q_size)
        .flat_map(|q| {
            (0..k_size).map(move |k| (q as f32 * q_scale - k as f32 * k_scale + offset) as u32)
        })
        .collect();
    let relative_coords_u32: Tensor<1, u32> =
        Tensor::from_slice(&device, [q_size * k_size], &rc_data);

    // index_select from rel_pos
    let selected: Tensor<2, f32> = rel_pos.index_select(0, &relative_coords_u32);
    let head_dim = rel_pos.shape()[1];
    selected.reshape([q_size, k_size, head_dim]).to_concrete()
}

struct Block {
    norm1: LayerNormNd<3, f32>,
    attn: Attention,
    norm2: LayerNormNd<3, f32>,
    mlp: MlpBlock,
    window_size: usize,
}

impl Block {
    fn load(
        device: &Device,
        vb: &mut VarBuilder,
        dim: usize,
        num_heads: usize,
        use_rel_pos: bool,
        window_size: usize,
        input_size: (usize, usize),
    ) -> Result<Self> {
        let norm1 = LayerNormNd::load(device, &mut vb.pp("norm1"), 1e-6)?;
        let norm2 = LayerNormNd::load(device, &mut vb.pp("norm2"), 1e-6)?;
        let input_size_attn = if window_size == 0 {
            input_size
        } else {
            (window_size, window_size)
        };
        let attn = Attention::load(
            device,
            &mut vb.pp("attn"),
            dim,
            num_heads,
            use_rel_pos,
            input_size_attn,
        )?;
        let mlp = MlpBlock::load(
            device,
            &mut vb.pp("mlp"),
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

    fn forward(&self, xs: &Tensor<4, f32>) -> Tensor<4, f32> {
        let shortcut = xs;
        let shape = xs.shape();
        let h = shape[1];
        let w = shape[2];

        // LayerNormNd<4, f32> over last dim
        let xs = layer_norm_bhwc(&self.norm1, xs);

        let (xs, pad_hw) = if self.window_size > 0 {
            window_partition(&xs, self.window_size)
        } else {
            (xs, (0, 0))
        };
        let xs = self.attn.forward(&xs);
        let xs = if self.window_size > 0 {
            window_unpartition(&xs, self.window_size, pad_hw, (h, w))
        } else {
            xs
        };

        let xs = xs + shortcut;

        // MLP
        let mlp_in = layer_norm_bhwc(&self.norm2, &xs);
        let mlp_shape = mlp_in.shape();
        let mlp_flat = mlp_in.reshape([mlp_shape[0], mlp_shape[1] * mlp_shape[2], mlp_shape[3]]);
        let mlp_out = self.mlp.forward(&mlp_flat);
        let mlp_out = mlp_out.reshape(mlp_shape);

        (&xs + mlp_out).to_concrete()
    }
}

fn window_partition(
    xs: &Tensor<4, f32>,
    window_size: usize,
) -> (Tensor<4, f32, ConcreteTensor<f32, 4>>, (usize, usize)) {
    let shape = xs.shape();
    let b = shape[0];
    let h = shape[1];
    let w = shape[2];
    let c = shape[3];

    let pad_h = (window_size - h % window_size) % window_size;
    let pad_w = (window_size - w % window_size) % window_size;

    let xs = if pad_h > 0 {
        xs.pad_with_zeros(1, 0, pad_h)
    } else {
        xs.to_concrete()
    };
    let xs = if pad_w > 0 {
        xs.pad_with_zeros(2, 0, pad_w)
    } else {
        xs
    };

    let h_p = h + pad_h;
    let w_p = w + pad_w;

    // (b, h_p/ws, ws, w_p/ws, ws, c) -> transpose(2,3) -> (b, h_p/ws, w_p/ws, ws, ws, c)
    // -> flatten first 3 dims -> (b * n_windows, ws, ws, c)
    let n_h = h_p / window_size;
    let n_w = w_p / window_size;
    let windows: Tensor<4, f32, ConcreteTensor<f32, 4>> = xs
        .reshape([b, n_h, window_size, n_w, window_size, c])
        .transpose(2, 3) // (b, n_h, n_w, ws, ws, c)
        .reshape([b * n_h * n_w, window_size, window_size, c])
        .to_concrete();

    (windows, (h_p, w_p))
}

fn window_unpartition(
    windows: &Tensor<4, f32>,
    window_size: usize,
    (h_p, w_p): (usize, usize),
    (h, w): (usize, usize),
) -> Tensor<4, f32, ConcreteTensor<f32, 4>> {
    let shape = windows.shape();
    let total = shape[0];
    let c = shape[3];
    let n_h = h_p / window_size;
    let n_w = w_p / window_size;
    let b = total / (n_h * n_w);

    let xs: Tensor<4, f32, ConcreteTensor<f32, 4>> = windows
        .reshape([b, n_h, n_w, window_size, window_size, c])
        .to_concrete()
        .transpose(2, 3) // (b, n_h, ws, n_w, ws, c)
        .to_concrete()
        .reshape([b, h_p, w_p, c])
        .to_concrete();

    let xs = if h_p > h {
        xs.narrow(1, 0, h).to_concrete()
    } else {
        xs
    };
    let xs = if w_p > w {
        xs.narrow(2, 0, w).to_concrete()
    } else {
        xs
    };
    xs
}

/// Standard ViT-B/H/L image encoder used by the upstream SAM checkpoints.
///
/// `forward` takes a `(B, 3, IMAGE_SIZE, IMAGE_SIZE)` preprocessed input and
/// returns `(B, prompt_embed_dim, IMAGE_SIZE/16, IMAGE_SIZE/16)` features.
pub struct ImageEncoderViT {
    patch_embed: PatchEmbed,
    blocks: Vec<Block>,
    neck_conv1: ConvNd<2, 4, f32>,
    neck_ln1: LayerNormNd<4, f32>,
    neck_conv2: ConvNd<2, 4, f32>,
    neck_ln2: LayerNormNd<4, f32>,
    pos_embed: Option<Tensor<4, f32, ConcreteTensor<f32, 4>>>,
}

impl ImageEncoderViT {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        device: &Device,
        vb: &mut VarBuilder,
        img_size: usize,
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
        let patch_embed = PatchEmbed::load(device, &mut vb.pp("patch_embed"), patch_size, 0)?;
        let grid_size = img_size / patch_size;

        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let ws = if global_attn_indexes.contains(&i) {
                0
            } else {
                window_size
            };
            let block = Block::load(
                device,
                &mut vb.pp(format!("blocks.{i}")),
                embed_dim,
                num_heads,
                use_rel_pos,
                ws,
                (grid_size, grid_size),
            )?;
            blocks.push(block);
        }

        let neck_conv1 =
            ConvNd::<2, 4, f32>::load_no_bias(device, &mut vb.pp("neck.0"), ConvNdConfig::default())?;
        let neck_ln1 = LayerNormNd::<4, f32>::load_over_axis(device, &mut vb.pp("neck.1"), 1, 1e-6)?;
        let cfg_pad1 = ConvNdConfig {
            padding: [1, 1],
            stride: [1, 1],
            groups: 1,
        };
        let neck_conv2 = ConvNd::<2, 4, f32>::load_no_bias(device, &mut vb.pp("neck.2"), cfg_pad1)?;
        let neck_ln2 = LayerNormNd::<4, f32>::load_over_axis(device, &mut vb.pp("neck.3"), 1, 1e-6)?;

        let pos_embed = if use_abs_pos {
            let p: Tensor<4, f32> = vb.get("pos_embed", device)?.dequantize();
            Some(p.to_concrete())
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

    pub fn forward(
        &self,
        xs: &Tensor<4, f32, impl TensorBacking<4, Elem = f32>>,
    ) -> Tensor<4, f32> {
        let xs = self.patch_embed.forward(xs); // (B, H, W, C)

        let mut xs: Tensor<4, f32> = match &self.pos_embed {
            Some(pos_embed) => (xs + pos_embed).to_concrete(),
            None => xs.to_concrete(),
        };

        for block in &self.blocks {
            xs = block.forward(&xs);
        }

        // (B, H, W, C) -> (B, C, H, W)
        let xs: Tensor<4, f32> = xs
            .transpose(2, 3) // (B, H, C, W)
            .to_concrete()
            .transpose(1, 2) // (B, C, H, W)
            .to_concrete();

        let xs = self.neck_conv1.forward(&xs);
        let xs = self.neck_ln1.forward(&xs);
        let xs = self.neck_conv2.forward(&xs.to_concrete());
        self.neck_ln2.forward(&xs)
    }
}

/// LayerNormNd<4, f32> helper for BHWC tensors (normalizes last dim).
fn layer_norm_bhwc(
    norm: &LayerNormNd<3, f32>,
    input: &Tensor<4, f32, impl TensorBacking<4, Elem = f32>>,
) -> Tensor<4, f32, ConcreteTensor<f32, 4>> {
    let [b, h, w, c] = input.shape();
    // Flatten to 3D (b*h, w, c) for layer_norm, then reshape back
    let flat = input.reshape([b * h, w, c]);
    let normed = norm.forward(&flat);
    normed.reshape([b, h, w, c]).to_concrete()
}
