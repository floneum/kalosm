//! TinyViT image encoder for MobileSAM.
//!
//! BatchNorm is fused into conv weights at GGUF conversion time,
//! so Conv2dBN becomes a plain conv here.

use fusor::cache::MaskKind;
use fusor::layers::{ConvNd, LayerNorm, Linear};
use fusor::{Device, Tensor};
use fusor_gguf::VarBuilder;

use super::{linear, load_dense, Result};

const MBCONV_EXPAND_RATIO: usize = 4;
const LOCAL_CONV_SIZE: usize = 3;
const IMG_SIZE: usize = 1024;

/// 2-d conv configuration: `[stride, stride]` / `[padding, padding]` /
/// `groups`, mirroring the reference's `ConvNdConfig<2>`.
#[derive(Clone, Copy)]
struct Conv2dConfig {
    padding: u32,
    stride: u32,
    groups: u32,
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
            groups: 1,
        }
    }
}

fn conv2d(vb: &VarBuilder, device: &Device, bias: bool, cfg: Conv2dConfig) -> Result<ConvNd> {
    let mut conv = ConvNd::load(vb, device.graph().handle(), bias)?;
    conv.stride = [cfg.stride, cfg.stride].into_iter().collect();
    conv.padding = [cfg.padding, cfg.padding].into_iter().collect();
    conv.groups = cfg.groups;
    Ok(conv)
}

/// Conv with fused BatchNorm (BN fused into weights at conversion time).
/// At runtime, this is just a conv whose bias comes from the fused BN.
struct ConvBN {
    conv: ConvNd,
}

impl ConvBN {
    fn load(device: &Device, vb: &VarBuilder, cfg: Conv2dConfig) -> Result<Self> {
        // BN is fused into the conv at GGUF conversion time, so we load
        // a regular conv from the "c" sub-namespace with fused weights.
        let conv = conv2d(&vb.pp("c"), device, true, cfg)?;
        Ok(Self { conv })
    }

    fn forward(&self, xs: &Tensor<4>) -> Tensor<4> {
        self.conv.forward(xs)
    }
}

pub(crate) struct PatchEmbed {
    conv1: ConvBN,
    conv2: ConvBN,
}

impl PatchEmbed {
    fn load(device: &Device, vb: &VarBuilder, _embed_dim: usize) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: 1,
            stride: 2,
            groups: 1,
        };
        let conv1 = ConvBN::load(device, &vb.pp("seq.0"), cfg)?;
        let conv2 = ConvBN::load(device, &vb.pp("seq.2"), cfg)?;
        Ok(Self { conv1, conv2 })
    }

    pub(crate) fn forward(&self, xs: &Tensor<4>) -> Tensor<4> {
        self.conv2.forward(&self.conv1.forward(xs).gelu())
    }
}

struct MBConv {
    conv1: ConvBN,
    conv2: ConvBN,
    conv3: ConvBN,
}

impl MBConv {
    fn load(
        device: &Device,
        vb: &VarBuilder,
        in_: usize,
        _out: usize,
        expand_ratio: usize,
    ) -> Result<Self> {
        let hidden = in_ * expand_ratio;
        let cfg_dw = Conv2dConfig {
            padding: 1,
            stride: 1,
            groups: hidden as u32,
        };
        let conv1 = ConvBN::load(device, &vb.pp("conv1"), Conv2dConfig::default())?;
        let conv2 = ConvBN::load(device, &vb.pp("conv2"), cfg_dw)?;
        let conv3 = ConvBN::load(device, &vb.pp("conv3"), Conv2dConfig::default())?;
        Ok(Self {
            conv1,
            conv2,
            conv3,
        })
    }

    fn forward(&self, xs: &Tensor<4>) -> Tensor<4> {
        let out = self.conv1.forward(xs).gelu();
        let out = self.conv2.forward(&out).gelu();
        let out = self.conv3.forward(&out);
        out.add(xs).gelu()
    }
}

struct PatchMerging {
    conv1: ConvBN,
    conv2: ConvBN,
    conv3: ConvBN,
    input_resolution: (usize, usize),
}

impl PatchMerging {
    /// `spatial_stride` is the stride of the depthwise conv: 2 when this
    /// PatchMerging is meant to halve the spatial resolution, 1 when it should
    /// keep it unchanged (used for the channel-only transition into TinyViT's
    /// final stage).
    fn load(
        device: &Device,
        vb: &VarBuilder,
        input_resolution: (usize, usize),
        out: usize,
        spatial_stride: usize,
    ) -> Result<Self> {
        let cfg_dw = Conv2dConfig {
            padding: 1,
            stride: spatial_stride as u32,
            groups: out as u32,
        };
        let conv1 = ConvBN::load(device, &vb.pp("conv1"), Conv2dConfig::default())?;
        let conv2 = ConvBN::load(device, &vb.pp("conv2"), cfg_dw)?;
        let conv3 = ConvBN::load(device, &vb.pp("conv3"), Conv2dConfig::default())?;
        Ok(Self {
            conv1,
            conv2,
            conv3,
            input_resolution,
        })
    }

    fn forward(&self, xs: &Tensor<3>) -> Tensor<3> {
        let [b, _, c] = xs.shape();
        let (h, w) = self.input_resolution;

        // (B, L, C) -> (B, H, W, C) -> (B, C, H, W)
        let xs = xs.reshape([b, h, w, c]).permute([0, 3, 1, 2]);

        let xs = self.conv1.forward(&xs).gelu();
        let xs = self.conv2.forward(&xs).gelu();
        let xs = self.conv3.forward(&xs);

        // Flatten spatial dims and transpose to (B, L, C)
        let [_, out_c, out_h, out_w] = xs.shape();
        xs.reshape([b, out_c, out_h * out_w]).transpose(1, 2)
    }
}

pub(crate) struct ConvLayerConfig {
    pub dim: usize,
    pub out: usize,
    pub input_resolution: (usize, usize),
    pub depth: usize,
    pub downsample: bool,
    pub conv_expand_ratio: usize,
    /// Spatial stride of the depthwise downsample conv (2 = halve resolution,
    /// 1 = channel-only transition).
    pub downsample_spatial_stride: usize,
}

pub(crate) struct ConvLayer {
    blocks: Vec<MBConv>,
    downsample: Option<PatchMerging>,
}

impl ConvLayer {
    fn load(device: &Device, vb: &VarBuilder, cfg: ConvLayerConfig) -> Result<Self> {
        let ConvLayerConfig {
            dim,
            out,
            input_resolution,
            depth,
            downsample,
            conv_expand_ratio,
            downsample_spatial_stride,
        } = cfg;
        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let block = MBConv::load(
                device,
                &vb.pp(format!("blocks.{i}")),
                dim,
                dim,
                conv_expand_ratio,
            )?;
            blocks.push(block);
        }
        let downsample = if downsample {
            Some(PatchMerging::load(
                device,
                &vb.pp("downsample"),
                input_resolution,
                out,
                downsample_spatial_stride,
            )?)
        } else {
            None
        };
        Ok(Self { blocks, downsample })
    }

    pub(crate) fn forward(&self, xs: &Tensor<4>) -> Tensor<3> {
        let mut xs = xs.clone();
        for block in &self.blocks {
            xs = block.forward(&xs);
        }
        // After ConvLayer blocks the output is still BCHW.
        // Downsample expects BLC format (3D), so flatten + transpose.
        let [b, c, h, w] = xs.shape();
        let flat = xs.reshape([b, c, h * w]).transpose(1, 2); // (B, L, C)
        match &self.downsample {
            Some(ds) => ds.forward(&flat),
            None => flat,
        }
    }
}

/// MLP for TinyViTBlock: LayerNorm -> Linear -> GELU -> Linear
struct TinyMlp {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl TinyMlp {
    fn load(device: &Device, vb: &VarBuilder) -> Result<Self> {
        let norm = LayerNorm::load(&vb.pp("norm"), device.graph().handle(), 1e-5)?;
        let fc1 = linear(&vb.pp("fc1"), device)?;
        let fc2 = linear(&vb.pp("fc2"), device)?;
        Ok(Self { norm, fc1, fc2 })
    }

    fn forward(&self, xs: &Tensor<3>) -> Tensor<3> {
        let xs = self.norm.forward(xs);
        let xs = self.fc1.forward(&xs).gelu();
        self.fc2.forward(&xs)
    }
}

/// Attention module for TinyViTBlock.
/// Uses pre-computed attention biases (indexed at load time).
struct TinyAttention {
    norm: LayerNorm,
    qkv: Linear,
    proj: Linear,
    ab: Tensor<3>, // (num_heads, n_points, n_points)
    key_dim: usize,
    num_heads: usize,
    d: usize,
    dh: usize,
    scale: f32,
}

impl TinyAttention {
    fn load(
        device: &Device,
        vb: &VarBuilder,
        _dim: usize,
        key_dim: usize,
        num_heads: usize,
        attn_ratio: usize,
        resolution: (usize, usize),
    ) -> Result<Self> {
        let d = attn_ratio * key_dim;
        let dh = d * num_heads;

        let norm = LayerNorm::load(&vb.pp("norm"), device.graph().handle(), 1e-5)?;
        let qkv = linear(&vb.pp("qkv"), device)?;
        let proj = linear(&vb.pp("proj"), device)?;

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

        // Load attention_biases: (num_heads, num_offsets), then index_select
        // along dim 1 to get (num_heads, n_points * n_points).
        let attention_biases: Tensor<2> = load_dense(vb, device, "attention_biases")?;
        let n_points = points.len();
        let idxs_tensor = Tensor::<1, u32>::from_slice(device, [idxs.len()], &idxs);
        let ab = attention_biases
            .index_select(1, &idxs_tensor)
            .reshape([num_heads, n_points, n_points]);

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

    fn forward(&self, xs: &Tensor<3>) -> Tensor<3> {
        let [b, n, _] = xs.shape();

        let qkv = self.qkv.forward(&self.norm.forward(xs));

        // (b, n, num_heads, key_dim + key_dim + d) -> split into q, k, v
        let qkv = qkv.reshape([b, n, self.num_heads, self.key_dim * 2 + self.d]);

        // q/k: (b, n, num_heads, key_dim) -> (b, num_heads, n, key_dim)
        let q = qkv.narrow(3, 0, self.key_dim).transpose(1, 2);
        let k = qkv.narrow(3, self.key_dim, self.key_dim).transpose(1, 2);
        // v: (b, n, num_heads, d) -> (b, num_heads, n, d)
        let v = qkv.narrow(3, 2 * self.key_dim, self.d).transpose(1, 2);

        // Scaled dot-product attention with the pre-computed additive bias:
        // (num_heads, n, n) broadcasts right-aligned onto (b, num_heads, n, n).
        let out = q.attention_masked(&k, &v, MaskKind::QkMask, Some(&self.ab), Some(self.scale));

        // (b, num_heads, n, d) -> (b, n, num_heads, d) -> (b, n, dh)
        let out = out.transpose(1, 2).reshape([b, n, self.dh]);

        self.proj.forward(&out)
    }
}

struct TinyViTBlock {
    attn: TinyAttention,
    local_conv: ConvBN,
    mlp: TinyMlp,
    window_size: usize,
    input_resolution: (usize, usize),
}

impl TinyViTBlock {
    fn load(
        device: &Device,
        vb: &VarBuilder,
        dim: usize,
        input_resolution: (usize, usize),
        num_heads: usize,
        window_size: usize,
    ) -> Result<Self> {
        let head_dim = dim / num_heads;
        let attn = TinyAttention::load(
            device,
            &vb.pp("attn"),
            dim,
            head_dim,
            num_heads,
            1, // attn_ratio
            (window_size, window_size),
        )?;
        let mlp = TinyMlp::load(device, &vb.pp("mlp"))?;
        let cfg_local = Conv2dConfig {
            padding: (LOCAL_CONV_SIZE / 2) as u32,
            stride: 1,
            groups: dim as u32,
        };
        let local_conv = ConvBN::load(device, &vb.pp("local_conv"), cfg_local)?;
        Ok(Self {
            attn,
            local_conv,
            mlp,
            window_size,
            input_resolution,
        })
    }

    fn forward(&self, xs: &Tensor<3>) -> Tensor<3> {
        let [b, l, c] = xs.shape();
        let (h, w) = self.input_resolution;

        let res_x = xs;

        let attended = if h == self.window_size && w == self.window_size {
            self.attn.forward(xs)
        } else {
            // Reshape to (B, H, W, C)
            let xs = xs.reshape([b, h, w, c]);

            let pad_b = (self.window_size - h % self.window_size) % self.window_size;
            let pad_r = (self.window_size - w % self.window_size) % self.window_size;

            let xs = if pad_b > 0 {
                xs.pad_with_zeros(1, 0, pad_b)
            } else {
                xs
            };
            let xs = if pad_r > 0 {
                xs.pad_with_zeros(2, 0, pad_r)
            } else {
                xs
            };

            let p_h = h + pad_b;
            let p_w = w + pad_r;
            let n_h = p_h / self.window_size;
            let n_w = p_w / self.window_size;

            // Window partition: (B, n_h, ws, n_w, ws, C) -> transpose(2,3) -> reshape
            let xs = xs
                .reshape([b, n_h, self.window_size, n_w, self.window_size, c])
                .transpose(2, 3) // (B, n_h, n_w, ws, ws, C)
                .reshape([b * n_h * n_w, self.window_size * self.window_size, c]);

            let xs = self.attn.forward(&xs);

            // Window unpartition
            let xs = xs
                .reshape([b, n_h, n_w, self.window_size, self.window_size, c])
                .transpose(2, 3) // (B, n_h, ws, n_w, ws, C)
                .reshape([b, p_h, p_w, c]);

            // Remove padding
            let xs = if pad_r > 0 { xs.narrow(2, 0, w) } else { xs };
            let xs = if pad_b > 0 { xs.narrow(1, 0, h) } else { xs };

            // Flatten back to (B, L, C)
            xs.reshape([b, l, c])
        };

        // Residual
        let xs = attended.add(res_x);

        // Local conv: (B, L, C) -> (B, C, H, W) -> conv -> (B, C, L) -> (B, L, C)
        let xs_conv = xs.transpose(1, 2).reshape([b, c, h, w]);
        let xs_conv = self.local_conv.forward(&xs_conv);
        let [_, _, out_h, out_w] = xs_conv.shape();
        let xs = xs_conv.reshape([b, c, out_h * out_w]).transpose(1, 2); // (B, L, C)

        // MLP residual
        let mlp_out = self.mlp.forward(&xs);
        xs.add(&mlp_out)
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
    /// Spatial stride of the depthwise downsample conv (2 = halve resolution,
    /// 1 = channel-only transition into the final TinyViT stage).
    pub downsample_spatial_stride: usize,
}

pub(crate) struct BasicLayer {
    blocks: Vec<TinyViTBlock>,
    downsample: Option<PatchMerging>,
}

impl BasicLayer {
    fn load(device: &Device, vb: &VarBuilder, cfg: BasicLayerConfig) -> Result<Self> {
        let BasicLayerConfig {
            dim,
            input_resolution,
            depth,
            num_heads,
            window_size,
            downsample,
            out,
            downsample_spatial_stride,
        } = cfg;
        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let block = TinyViTBlock::load(
                device,
                &vb.pp(format!("blocks.{i}")),
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
                &vb.pp("downsample"),
                input_resolution,
                out,
                downsample_spatial_stride,
            )?)
        } else {
            None
        };
        Ok(Self { blocks, downsample })
    }

    pub(crate) fn forward(&self, xs: &Tensor<3>) -> Tensor<3> {
        let mut xs = xs.clone();
        for block in &self.blocks {
            xs = block.forward(&xs);
        }
        match &self.downsample {
            Some(ds) => ds.forward(&xs),
            None => xs,
        }
    }
}

/// TinyViT image encoder used by Mobile-SAM.
///
/// `forward` takes a `(B, 3, IMG_SIZE, IMG_SIZE)` input and returns
/// `(B, neck_dim, IMG_SIZE/16, IMG_SIZE/16)` features - same output shape as
/// the standard `ImageEncoderViT` so both can plug into the prompt encoder.
pub struct TinyViT {
    pub(crate) patch_embed: PatchEmbed,
    pub(crate) layer0: ConvLayer,
    pub(crate) layers: Vec<BasicLayer>,
    neck_conv1: ConvNd,
    neck_ln1: LayerNorm,
    neck_conv2: ConvNd,
    neck_ln2: LayerNorm,
}

impl TinyViT {
    pub fn load(
        device: &Device,
        vb: &VarBuilder,
        embed_dims: &[usize],
        depths: &[usize],
        num_heads: &[usize],
        window_sizes: &[usize],
    ) -> Result<Self> {
        let graph = device.graph().handle();
        let patch_embed = PatchEmbed::load(device, &vb.pp("patch_embed"), embed_dims[0])?;
        let patches_resolution = IMG_SIZE / 4;

        let num_layers = embed_dims.len();

        let layer0 = ConvLayer::load(
            device,
            &vb.pp("layers.0"),
            ConvLayerConfig {
                dim: embed_dims[0],
                out: embed_dims[1],
                input_resolution: (patches_resolution, patches_resolution),
                depth: depths[0],
                downsample: true,
                conv_expand_ratio: MBCONV_EXPAND_RATIO,
                // ConvLayer always feeds a transformer stage that expects half
                // the spatial resolution.
                downsample_spatial_stride: 2,
            },
        )?;

        let mut layers = Vec::with_capacity(num_layers - 1);
        for i_layer in 1..num_layers {
            let patches_resolution = patches_resolution / (1 << usize::min(i_layer, 2));
            // The last PatchMerging in TinyViT is a channel-only transition
            // into the final stage and must keep the spatial resolution.
            let downsample_spatial_stride = if i_layer + 2 < num_layers { 2 } else { 1 };
            let layer = BasicLayer::load(
                device,
                &vb.pp(format!("layers.{i_layer}")),
                BasicLayerConfig {
                    dim: embed_dims[i_layer],
                    input_resolution: (patches_resolution, patches_resolution),
                    depth: depths[i_layer],
                    num_heads: num_heads[i_layer],
                    window_size: window_sizes[i_layer],
                    downsample: i_layer < num_layers - 1,
                    out: embed_dims[usize::min(i_layer + 1, num_layers - 1)],
                    downsample_spatial_stride,
                },
            )?;
            layers.push(layer);
        }

        let neck_conv1 = conv2d(&vb.pp("neck.0"), device, false, Conv2dConfig::default())?;
        let neck_ln1 = LayerNorm::load(&vb.pp("neck.1"), graph, 1e-6)?;
        let cfg_pad1 = Conv2dConfig {
            padding: 1,
            stride: 1,
            groups: 1,
        };
        let neck_conv2 = conv2d(&vb.pp("neck.2"), device, false, cfg_pad1)?;
        let neck_ln2 = LayerNorm::load(&vb.pp("neck.3"), graph, 1e-6)?;

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

    pub fn forward(&self, xs: &Tensor<4>) -> Tensor<4> {
        // PatchEmbed: (B, C, H, W) -> (B, C', H/4, W/4)
        let xs = self.patch_embed.forward(xs);

        // ConvLayer0: still BCHW -> output flattened to BLC
        let mut xs = self.layer0.forward(&xs);

        for layer in self.layers.iter() {
            xs = layer.forward(&xs);
        }

        // Reshape from BLC to BCHW. After all stages, L = (IMG_SIZE / total_stride)^2.
        let [b, l, c] = xs.shape();
        let s = (l as f64).sqrt() as usize;
        assert_eq!(
            s * s,
            l,
            "TinyViT output token count ({l}) must be a perfect square"
        );
        let xs = xs.reshape([b, s, s, c]).permute([0, 3, 1, 2]); // (B, C, s, s)

        // Neck. The neck LayerNorms are Meta's LayerNorm2d: over channels.
        let xs = self.neck_conv1.forward(&xs);
        let xs = super::channel_layer_norm(&self.neck_ln1, &xs);
        let xs = self.neck_conv2.forward(&xs);
        super::channel_layer_norm(&self.neck_ln2, &xs)
    }
}

pub fn tiny_vit_5m(device: &Device, vb: &VarBuilder) -> Result<TinyViT> {
    TinyViT::load(
        device,
        vb,
        &[64, 128, 160, 320],
        &[2, 2, 6, 2],
        &[2, 4, 5, 10],
        &[7, 7, 14, 7],
    )
}
