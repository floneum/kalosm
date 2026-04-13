//! TinyViT image encoder for MobileSAM.
//!
//! BatchNorm is fused into conv weights at GGUF conversion time,
//! so Conv2dBN becomes plain Conv2d here.

use fusor::layers::{Conv2d, Conv2dConfig, LayerNorm, LayerNorm2d, Linear};
use fusor::{ConcreteTensor, Device, Tensor, TensorBacking, VarBuilder};

use super::Result;

const MBCONV_EXPAND_RATIO: usize = 4;
const MLP_RATIO: usize = 4;
const LOCAL_CONV_SIZE: usize = 3;
const IMG_SIZE: usize = 1024;
const IN_CHANNELS: usize = 3;

/// Conv2d with fused BatchNorm (BN fused into weights at conversion time).
/// At runtime, this is just a Conv2d with no bias (bias comes from fused BN).
struct Conv2dBN {
    conv: Conv2d<f32>,
}

impl Conv2dBN {
    fn load(device: &Device, vb: &mut VarBuilder, cfg: Conv2dConfig) -> Result<Self> {
        // BN is fused into the conv at GGUF conversion time, so we load
        // a regular conv from the "c" sub-namespace with fused weights.
        let conv = Conv2d::load(device, &mut vb.pp("c"), cfg)?;
        Ok(Self { conv })
    }

    fn forward(&self, xs: &Tensor<4, f32, impl TensorBacking<4, Elem = f32>>) -> Tensor<4, f32> {
        self.conv.forward(xs)
    }
}

pub(crate) struct PatchEmbed {
    conv1: Conv2dBN,
    conv2: Conv2dBN,
}

impl PatchEmbed {
    fn load(device: &Device, vb: &mut VarBuilder, _embed_dim: usize) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: [1, 1],
            stride: [2, 2],
            groups: 1,
        };
        let conv1 = Conv2dBN::load(device, &mut vb.pp("seq.0"), cfg)?;
        let conv2 = Conv2dBN::load(device, &mut vb.pp("seq.2"), cfg)?;
        Ok(Self { conv1, conv2 })
    }

    pub(crate) fn forward(
        &self,
        xs: &Tensor<4, f32, impl TensorBacking<4, Elem = f32>>,
    ) -> Tensor<4, f32> {
        let xs = self.conv1.forward(xs);
        let xs = xs.gelu();
        self.conv2.forward(&xs)
    }
}

struct MBConv {
    conv1: Conv2dBN,
    conv2: Conv2dBN,
    conv3: Conv2dBN,
}

impl MBConv {
    fn load(
        device: &Device,
        vb: &mut VarBuilder,
        in_: usize,
        _out: usize,
        expand_ratio: usize,
    ) -> Result<Self> {
        let hidden = in_ * expand_ratio;
        let cfg_dw = Conv2dConfig {
            padding: [1, 1],
            stride: [1, 1],
            groups: hidden,
        };
        let conv1 = Conv2dBN::load(device, &mut vb.pp("conv1"), Conv2dConfig::default())?;
        let conv2 = Conv2dBN::load(device, &mut vb.pp("conv2"), cfg_dw)?;
        let conv3 = Conv2dBN::load(device, &mut vb.pp("conv3"), Conv2dConfig::default())?;
        Ok(Self {
            conv1,
            conv2,
            conv3,
        })
    }

    fn forward(
        &self,
        xs: &Tensor<4, f32, ConcreteTensor<f32, 4>>,
    ) -> Tensor<4, f32, ConcreteTensor<f32, 4>> {
        let shortcut = xs;
        let out = self.conv1.forward(xs);
        let out = out.gelu();
        let out = self.conv2.forward(&out);
        let out = out.gelu();
        let out = self.conv3.forward(&out);
        (out + shortcut).to_concrete().gelu().to_concrete()
    }
}

struct PatchMerging {
    conv1: Conv2dBN,
    conv2: Conv2dBN,
    conv3: Conv2dBN,
    input_resolution: (usize, usize),
}

impl PatchMerging {
    fn load(
        device: &Device,
        vb: &mut VarBuilder,
        input_resolution: (usize, usize),
        _dim: usize,
        out: usize,
    ) -> Result<Self> {
        let stride = if [320, 448, 576].contains(&out) { 1 } else { 2 };
        let cfg_dw = Conv2dConfig {
            padding: [1, 1],
            stride: [stride, stride],
            groups: out,
        };
        let conv1 = Conv2dBN::load(device, &mut vb.pp("conv1"), Conv2dConfig::default())?;
        let conv2 = Conv2dBN::load(device, &mut vb.pp("conv2"), cfg_dw)?;
        let conv3 = Conv2dBN::load(device, &mut vb.pp("conv3"), Conv2dConfig::default())?;
        Ok(Self {
            conv1,
            conv2,
            conv3,
            input_resolution,
        })
    }

    fn forward(&self, xs: &Tensor<3, f32, impl TensorBacking<3, Elem = f32>>) -> Tensor<3, f32> {
        let shape = xs.shape();
        let b = shape[0];
        let _l = shape[1];
        let c = shape[2];
        let (h, w) = self.input_resolution;

        // If rank is 3, reshape to (B, H, W, C) then permute to (B, C, H, W)
        let xs = xs.reshape([b, h, w, c]);
        let xs = xs.transpose(2, 3); // (B, H, C, W)
        let xs = xs.transpose(1, 2); // (B, C, H, W)

        let xs = self.conv1.forward(&xs);
        let xs = xs.gelu();
        let xs = self.conv2.forward(&xs);
        let xs = xs.gelu();
        let xs = self.conv3.forward(&xs);

        // Flatten spatial dims and transpose to (B, L, C)
        let out_shape = xs.shape();
        let out_c = out_shape[1];
        let out_h = out_shape[2];
        let out_w = out_shape[3];
        let xs = xs.reshape([b, out_c, out_h * out_w]);
        xs.transpose(1, 2).to_concrete() // (B, L, C)
    }
}

pub(crate) struct ConvLayerConfig {
    pub dim: usize,
    pub out: usize,
    pub input_resolution: (usize, usize),
    pub depth: usize,
    pub downsample: bool,
    pub conv_expand_ratio: usize,
}

pub(crate) struct ConvLayer {
    blocks: Vec<MBConv>,
    downsample: Option<PatchMerging>,
}

impl ConvLayer {
    fn load(device: &Device, vb: &mut VarBuilder, cfg: ConvLayerConfig) -> Result<Self> {
        let ConvLayerConfig {
            dim,
            out,
            input_resolution,
            depth,
            downsample,
            conv_expand_ratio,
        } = cfg;
        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let block = MBConv::load(
                device,
                &mut vb.pp(format!("blocks.{i}")),
                dim,
                dim,
                conv_expand_ratio,
            )?;
            blocks.push(block);
        }
        let downsample = if downsample {
            Some(PatchMerging::load(
                device,
                &mut vb.pp("downsample"),
                input_resolution,
                dim,
                out,
            )?)
        } else {
            None
        };
        Ok(Self { blocks, downsample })
    }

    pub(crate) fn forward(&self, xs: &Tensor<4, f32, ConcreteTensor<f32, 4>>) -> Tensor<3, f32> {
        let mut xs = xs.clone();
        for block in &self.blocks {
            xs = block.forward(&xs);
        }
        // After ConvLayer blocks the output is still BCHW.
        // Downsample expects BLC format (3D), so flatten + transpose.
        let shape = xs.shape();
        let b = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        let flat_reshaped = xs.reshape([b, c, h * w]);
        let flat = flat_reshaped.transpose(1, 2); // (B, L, C)
        match &self.downsample {
            Some(ds) => ds.forward(&flat),
            None => flat.to_concrete(),
        }
    }
}

/// MLP for TinyViTBlock: LayerNorm -> Linear -> GELU -> Linear
struct TinyMlp {
    norm: LayerNorm<1, f32>,
    fc1: Linear<f32>,
    fc2: Linear<f32>,
}

impl TinyMlp {
    fn load(
        device: &Device,
        vb: &mut VarBuilder,
        _in_features: usize,
        _hidden: usize,
    ) -> Result<Self> {
        let norm = LayerNorm::load(device, &mut vb.pp("norm"), 1e-5)?;
        let fc1 = Linear::load(device, &mut vb.pp("fc1"))?;
        let fc2 = Linear::load(device, &mut vb.pp("fc2"))?;
        Ok(Self { norm, fc1, fc2 })
    }

    fn forward(&self, xs: &Tensor<3, f32, impl TensorBacking<3, Elem = f32>>) -> Tensor<3, f32> {
        let xs = self.norm.forward(xs);
        let xs = self.fc1.forward(&xs);
        let xs = xs.gelu();
        self.fc2.forward(&xs)
    }
}

/// Attention module for TinyViTBlock.
/// Uses pre-computed attention biases (indexed at load time).
struct TinyAttention {
    norm: LayerNorm<1, f32>,
    qkv: Linear<f32>,
    proj: Linear<f32>,
    ab: Tensor<3, f32, ConcreteTensor<f32, 3>>, // (num_heads, n_points, n_points)
    key_dim: usize,
    num_heads: usize,
    d: usize,
    dh: usize,
    scale: f32,
}

impl TinyAttention {
    fn load(
        device: &Device,
        vb: &mut VarBuilder,
        _dim: usize,
        key_dim: usize,
        num_heads: usize,
        attn_ratio: usize,
        resolution: (usize, usize),
    ) -> Result<Self> {
        let d = attn_ratio * key_dim;
        let dh = d * num_heads;
        let nh_kd = key_dim * num_heads;
        let _h = dh + nh_kd * 2;

        let norm = LayerNorm::load(device, &mut vb.pp("norm"), 1e-5)?;
        let qkv = Linear::load(device, &mut vb.pp("qkv"))?;
        let proj = Linear::load(device, &mut vb.pp("proj"))?;

        // Build attention bias index table
        let points: Vec<(i64, i64)> = (0..resolution.0)
            .flat_map(|x| (0..resolution.1).map(move |y| (x as i64, y as i64)))
            .collect();
        let mut attention_offsets = std::collections::HashMap::new();
        let mut idxs = Vec::with_capacity(points.len() * points.len());
        for &(x1, y1) in &points {
            for &(x2, y2) in &points {
                let offset = ((x2 - x1).unsigned_abs(), (y2 - y1).unsigned_abs());
                let l = attention_offsets.len();
                let idx = *attention_offsets.entry(offset).or_insert(l);
                idxs.push(idx as u32);
            }
        }

        // Load attention_biases: (num_heads, num_offsets)
        let attention_biases: Tensor<2, f32> = vb.get("attention_biases", device)?.dequantize();

        // index_select along dim 1 to get (num_heads, n_points * n_points)
        let n_points = points.len();
        let idxs_tensor: Tensor<1, u32> = Tensor::from_slice(device, [idxs.len()], &idxs);
        let selected: Tensor<2, f32> = attention_biases.index_select(1, &idxs_tensor);
        // Reshape to (num_heads, n_points, n_points)
        let ab = selected
            .reshape([num_heads, n_points, n_points])
            .to_concrete();

        let scale = 1.0 / (key_dim as f32).sqrt();

        Ok(Self {
            norm,
            qkv,
            proj,
            ab,
            key_dim,
            num_heads,
            d,
            dh,
            scale,
        })
    }

    fn forward(&self, xs: &Tensor<3, f32, impl TensorBacking<3, Elem = f32>>) -> Tensor<3, f32> {
        let shape = xs.shape();
        let b = shape[0];
        let n = shape[1];

        let xs = self.norm.forward(xs);
        let qkv = self.qkv.forward(&xs);

        // (b, n, num_heads, key_dim + key_dim + d) -> split into q, k, v
        let qkv = qkv.reshape([b, n, self.num_heads, self.key_dim * 2 + self.d]);

        // q: (b, n, num_heads, key_dim) -> (b, num_heads, n, key_dim)
        let q_narrow = qkv.narrow(3, 0, self.key_dim);
        let q = q_narrow.transpose(1, 2); // (b, num_heads, n, key_dim)
                                          // k: (b, n, num_heads, key_dim) -> (b, num_heads, n, key_dim)
        let k_narrow = qkv.narrow(3, self.key_dim, self.key_dim);
        let k = k_narrow.transpose(1, 2);
        // v: (b, n, num_heads, d) -> (b, num_heads, n, d)
        let v_narrow = qkv.narrow(3, 2 * self.key_dim, self.d);
        let v = v_narrow.transpose(1, 2);

        // attn = q * scale @ k^T
        let k_t = k.transpose(2, 3);
        let attn = q.mul_scalar(self.scale).mat_mul(&k_t);

        // Add pre-computed attention bias: (num_heads, n, n) broadcast to (b, num_heads, n, n)
        let ab_reshaped = self.ab.reshape([1, self.num_heads, n, n]);
        let ab_broadcast = ab_reshaped.broadcast_as([b, self.num_heads, n, n]);
        let attn = attn + ab_broadcast;

        // Softmax
        let attn = attn.softmax_last_dim::<3>();

        // attn @ v -> (b, num_heads, n, d)
        let out = attn.mat_mul(&v);

        // transpose -> (b, n, num_heads, d) -> reshape to (b, n, dh)
        let out_transposed = out.transpose(1, 2); // (b, n, num_heads, d)
        let out = out_transposed.reshape([b, n, self.dh]);

        self.proj.forward(&out)
    }
}

struct TinyViTBlock {
    attn: TinyAttention,
    local_conv: Conv2dBN,
    mlp: TinyMlp,
    window_size: usize,
    input_resolution: (usize, usize),
}

impl TinyViTBlock {
    fn load(
        device: &Device,
        vb: &mut VarBuilder,
        dim: usize,
        input_resolution: (usize, usize),
        num_heads: usize,
        window_size: usize,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let attn = TinyAttention::load(
            device,
            &mut vb.pp("attn"),
            dim,
            head_dim,
            num_heads,
            1, // attn_ratio
            (window_size, window_size),
        )?;
        let mlp = TinyMlp::load(device, &mut vb.pp("mlp"), dim, dim * MLP_RATIO)?;
        let cfg_local = Conv2dConfig {
            padding: [LOCAL_CONV_SIZE / 2, LOCAL_CONV_SIZE / 2],
            stride: [1, 1],
            groups: dim,
        };
        let local_conv = Conv2dBN::load(device, &mut vb.pp("local_conv"), cfg_local)?;
        Ok(Self {
            attn,
            local_conv,
            mlp,
            window_size,
            input_resolution,
        })
    }

    fn forward(&self, xs: &Tensor<3, f32>) -> Tensor<3, f32> {
        let shape = xs.shape();
        let b = shape[0];
        let l = shape[1];
        let c = shape[2];
        let (h, w) = self.input_resolution;

        let res_x = xs.to_concrete();

        let xs = if h == self.window_size && w == self.window_size {
            self.attn.forward(xs)
        } else {
            // Reshape to (B, H, W, C)
            let xs = xs.reshape([b, h, w, c]);

            let pad_b = (self.window_size - h % self.window_size) % self.window_size;
            let pad_r = (self.window_size - w % self.window_size) % self.window_size;

            let xs = if pad_b > 0 {
                xs.to_concrete().pad_with_zeros(1, 0, pad_b).to_concrete()
            } else {
                xs.to_concrete()
            };
            let xs = if pad_r > 0 {
                xs.pad_with_zeros(2, 0, pad_r).to_concrete()
            } else {
                xs
            };

            let p_h = h + pad_b;
            let p_w = w + pad_r;
            let n_h = p_h / self.window_size;
            let n_w = p_w / self.window_size;

            // Window partition: (B, n_h, ws, n_w, ws, C) -> transpose(2,3) -> reshape
            let xs_r1 = xs.reshape([b, n_h, self.window_size, n_w, self.window_size, c]);
            let xs_t1 = xs_r1.transpose(2, 3); // (B, n_h, n_w, ws, ws, C)
            let xs = xs_t1.reshape([b * n_h * n_w, self.window_size * self.window_size, c]);

            let xs = self.attn.forward(&xs);

            // Window unpartition
            let xs_r2 = xs.reshape([b, n_h, n_w, self.window_size, self.window_size, c]);
            let xs_t2 = xs_r2.transpose(2, 3); // (B, n_h, ws, n_w, ws, C)
            let xs = xs_t2.reshape([b, p_h, p_w, c]);

            // Remove padding
            let xs = if pad_r > 0 {
                xs.narrow(2, 0, w).to_concrete()
            } else {
                xs.to_concrete()
            };
            let xs = if pad_b > 0 {
                xs.narrow(1, 0, h).to_concrete()
            } else {
                xs
            };

            // Flatten back to (B, L, C)
            xs.reshape([b, l, c]).to_concrete()
        };

        // Residual
        let xs: Tensor<3, f32> = (xs + &res_x).to_concrete();

        // Local conv: (B, L, C) -> (B, C, H, W) -> conv -> (B, C, L) -> (B, L, C)
        let xs_t = xs.transpose(1, 2); // (B, C, L)
        let xs_conv = xs_t.reshape([b, c, h, w]);
        let xs_conv = self.local_conv.forward(&xs_conv);
        let xs_conv_shape = xs_conv.shape();
        let xs_r = xs_conv.reshape([b, c, xs_conv_shape[2] * xs_conv_shape[3]]);
        let xs = xs_r.transpose(1, 2); // (B, L, C)

        // MLP residual
        let mlp_out = self.mlp.forward(&xs);
        (&xs + mlp_out).to_concrete()
    }
}

pub(crate) struct BasicLayerConfig {
    pub dim: usize,
    pub input_resolution: (usize, usize),
    pub depth: usize,
    pub num_heads: usize,
    pub window_size: usize,
    pub downsample: bool,
    pub out: usize,
}

pub(crate) struct BasicLayer {
    blocks: Vec<TinyViTBlock>,
    downsample: Option<PatchMerging>,
}

impl BasicLayer {
    fn load(device: &Device, vb: &mut VarBuilder, cfg: BasicLayerConfig) -> Result<Self> {
        let BasicLayerConfig {
            dim,
            input_resolution,
            depth,
            num_heads,
            window_size,
            downsample,
            out,
        } = cfg;
        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let block = TinyViTBlock::load(
                device,
                &mut vb.pp(format!("blocks.{i}")),
                dim,
                input_resolution,
                num_heads,
                window_size,
            )?;
            blocks.push(block);
        }
        let downsample = if downsample {
            Some(PatchMerging::load(
                device,
                &mut vb.pp("downsample"),
                input_resolution,
                dim,
                out,
            )?)
        } else {
            None
        };
        Ok(Self { blocks, downsample })
    }

    pub(crate) fn forward(&self, xs: &Tensor<3, f32>) -> Tensor<3, f32> {
        let mut xs = xs.clone();
        for block in &self.blocks {
            xs = block.forward(&xs).to_concrete();
        }
        match &self.downsample {
            Some(ds) => ds.forward(&xs),
            None => xs.to_concrete(),
        }
    }
}

pub struct TinyViT {
    pub(crate) patch_embed: PatchEmbed,
    pub(crate) layer0: ConvLayer,
    pub(crate) layers: Vec<BasicLayer>,
    neck_conv1: Conv2d<f32>,
    neck_ln1: LayerNorm2d,
    neck_conv2: Conv2d<f32>,
    neck_ln2: LayerNorm2d,
}

impl TinyViT {
    pub fn load(
        device: &Device,
        vb: &mut VarBuilder,
        embed_dims: &[usize],
        depths: &[usize],
        num_heads: &[usize],
        window_sizes: &[usize],
    ) -> Result<Self> {
        let patch_embed = PatchEmbed::load(device, &mut vb.pp("patch_embed"), embed_dims[0])?;
        let patches_resolution = IMG_SIZE / 4;

        let layer0 = ConvLayer::load(
            device,
            &mut vb.pp("layers.0"),
            ConvLayerConfig {
                dim: embed_dims[0],
                out: embed_dims[1],
                input_resolution: (patches_resolution, patches_resolution),
                depth: depths[0],
                downsample: true,
                conv_expand_ratio: MBCONV_EXPAND_RATIO,
            },
        )?;

        let num_layers = embed_dims.len();
        let mut layers = Vec::with_capacity(num_layers - 1);
        for i_layer in 1..num_layers {
            let patches_resolution = patches_resolution / (1 << usize::min(i_layer, 2));
            let layer = BasicLayer::load(
                device,
                &mut vb.pp(format!("layers.{i_layer}")),
                BasicLayerConfig {
                    dim: embed_dims[i_layer],
                    input_resolution: (patches_resolution, patches_resolution),
                    depth: depths[i_layer],
                    num_heads: num_heads[i_layer],
                    window_size: window_sizes[i_layer],
                    downsample: i_layer < num_layers - 1,
                    out: embed_dims[usize::min(i_layer + 1, num_layers - 1)],
                },
            )?;
            layers.push(layer);
        }

        let _last_embed_dim = embed_dims[embed_dims.len() - 1];
        let neck_conv1 =
            Conv2d::load_no_bias(device, &mut vb.pp("neck.0"), Conv2dConfig::default())?;
        let neck_ln1 = LayerNorm2d::load(device, &mut vb.pp("neck.1"), 1e-6)?;
        let cfg_pad1 = Conv2dConfig {
            padding: [1, 1],
            stride: [1, 1],
            groups: 1,
        };
        let neck_conv2 = Conv2d::load_no_bias(device, &mut vb.pp("neck.2"), cfg_pad1)?;
        let neck_ln2 = LayerNorm2d::load(device, &mut vb.pp("neck.3"), 1e-6)?;

        Ok(Self {
            patch_embed,
            layer0,
            layers,
            neck_conv1,
            neck_ln1,
            neck_conv2,
            neck_ln2,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor<4, f32, impl TensorBacking<4, Elem = f32>>,
    ) -> Tensor<4, f32> {
        // PatchEmbed: (B, C, H, W) -> (B, C', H/4, W/4)
        let xs = self.patch_embed.forward(xs);

        // ConvLayer0: still BCHW -> output flattened to BLC
        let mut xs = self.layer0.forward(&xs.to_concrete());

        for layer in self.layers.iter() {
            xs = layer.forward(&xs);
        }

        // Reshape to (B, 64, 64, C) then permute to (B, C, 64, 64)
        let shape = xs.shape();
        let b = shape[0];
        let c = shape[2];
        let xs_reshaped = xs.reshape([b, 64, 64, c]);
        let xs_t1 = xs_reshaped.transpose(2, 3); // (B, 64, C, 64)
        let xs = xs_t1.transpose(1, 2); // (B, C, 64, 64)

        // Neck
        let xs = self.neck_conv1.forward(&xs);
        let xs = self.neck_ln1.forward(&xs);
        let xs = self.neck_conv2.forward(&xs);
        self.neck_ln2.forward(&xs)
    }
}

pub fn tiny_vit_5m(device: &Device, vb: &mut VarBuilder) -> Result<TinyViT> {
    TinyViT::load(
        device,
        vb,
        &[64, 128, 160, 320],
        &[2, 2, 6, 2],
        &[2, 4, 5, 10],
        &[7, 7, 14, 7],
    )
}
