//! The Qwen2.5-VL vision tower: pixels in, LLM-space embeddings out.
//!
//! Everything that is bookkeeping — resizing, patch extraction, the window
//! permutation, the rope indices — runs on the host in plain loops; the
//! device sees the patch matmul, the transformer blocks and the merger. The
//! `mmproj` GGUF llama.cpp ships carries the weights under `v.*` / `mm.*`.

mod image;
mod merger;
mod rope_index;
mod window;

pub(crate) use rope_index::{rope_index, RopePosition};

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use fusor::cache::MaskKind;
use fusor::{Device, Dim, Minus1, Tensor};
use fusor_gguf::{Gguf, VarBuilder};

use super::dense_1d;
use super::weight::Weight;
use crate::raw::rope::create_inverse_frequency;

pub(crate) const QWEN_EPS: f32 = 1e-6;

/// Loading-time metadata the tower needs from the `mmproj` file.
struct VisionMeta {
    block_count: usize,
    head_count: usize,
    patch_size: usize,
    hidden_size: usize,
    fullatt_block_indexes: Vec<usize>,
    layer_norm_eps: f32,
    image_mean: [f32; 3],
    image_std: [f32; 3],
}

impl VisionMeta {
    fn read(vb: &VarBuilder) -> VisionMeta {
        let u = |key: &str, default: usize| {
            vb.get_metadata(key)
                .and_then(|v| v.to_u32().ok())
                .map_or(default, |v| v as usize)
        };
        let f3 = |key: &str, default: [f32; 3]| {
            vb.get_metadata(key)
                .and_then(|v| v.to_array().ok())
                .and_then(|a| {
                    let v: Vec<f32> = a.iter().filter_map(|x| x.to_f32().ok()).collect();
                    <[f32; 3]>::try_from(v).ok()
                })
                .unwrap_or(default)
        };
        let block_count = u("clip.vision.block_count", 32);
        let fullatt_block_indexes = vb
            .get_metadata("clip.vision.n_wa_pattern")
            .and_then(|v| v.to_u32().ok())
            .filter(|&n| n > 0)
            .map(|n| full_attention_blocks(block_count, n as usize))
            .unwrap_or_else(|| vec![7, 15, 23, 31]);
        VisionMeta {
            block_count,
            head_count: u("clip.vision.attention.head_count", 16),
            patch_size: u("clip.vision.patch_size", 14),
            hidden_size: u("clip.vision.embedding_length", 1280),
            fullatt_block_indexes,
            layer_norm_eps: vb
                .get_metadata("clip.vision.attention.layer_norm_epsilon")
                .and_then(|v| v.to_f32().ok())
                .unwrap_or(QWEN_EPS),
            image_mean: f3(
                "clip.vision.image_mean",
                [0.481_454_67, 0.457_827_5, 0.408_210_72],
            ),
            image_std: f3(
                "clip.vision.image_std",
                [0.268_629_55, 0.261_302_6, 0.275_777_1],
            ),
        }
    }
}

/// Every `n`th block, counting from one, attends over the whole image; the
/// rest attend within windows.
fn full_attention_blocks(block_count: usize, n_wa_pattern: usize) -> Vec<usize> {
    (n_wa_pattern - 1..block_count)
        .step_by(n_wa_pattern)
        .collect()
}

pub(crate) struct QwenVisionTransformer {
    pub(crate) spatial_merge_size: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    window_size: usize,
    fullatt_block_indexes: Vec<usize>,
    /// The patch "convolution" as the `[embed, 3 * 2 * p * p]` matrix it is:
    /// the kernel covers one whole patch at stride one patch.
    patch_embed: Tensor<2>,
    /// Half the rotary dimension's inverse frequencies (`head_dim / 4` of
    /// them): each of the height and width axes gets half the rotary space.
    rotary_inv_freq: Vec<f32>,
    blocks: Vec<VisionBlock>,
    merger: merger::PatchMerger,
    image_mean: [f32; 3],
    image_std: [f32; 3],
    device: Device,
    /// One replayable block graph per image grid; see [`BlockGraph`].
    graphs: Mutex<HashMap<[u32; 3], Arc<BlockGraph>>>,
}

/// The block as one graph over slot leaves, built once per image grid and
/// replayed for every block of every image of that grid.
///
/// Every block of the tower is the same computation over different weights,
/// and a resolve of a fresh graph is a saturation and an extraction of
/// ~40k nodes (seconds on the host) that the session graph then keeps for
/// good. So the graph is built once over leaves — the hidden state, the
/// rope rows, every weight — and a block runs by rebinding those leaves to
/// its own buffers and resolving the same root, which replays. Window and
/// full attention are two roots over the same slots.
struct BlockGraph {
    x: Tensor<3>,
    slots: VisionBlock,
    out_window: Tensor<3>,
    out_full: Tensor<3>,
}

impl QwenVisionTransformer {
    pub(crate) fn from_gguf(bytes: Vec<u8>, device: &Device) -> fusor::Result<Self> {
        let gguf = Gguf::from_bytes(bytes)?;
        let vb = VarBuilder::new(Arc::new(gguf));
        let meta = VisionMeta::read(&vb);
        let spatial_merge_size = 2;
        let temporal_patch_size = 2;
        let in_channels = 3;
        let head_dim = meta.hidden_size / meta.head_count;
        let graph = device.graph();

        let patch_embed = {
            // llama.cpp splits the `[out, in, t, p, p]` kernel into one
            // `[out, in, p, p]` tensor per temporal slice.
            let slice = |name: &str| -> fusor::Result<Tensor<3>> {
                let mut raw = vb.get_raw(name)?;
                let out = raw.shape[0] as usize;
                // `[out, in, p, p]` read as the `[out, in * p * p]` matrix.
                let cols: u64 = raw.shape[1..].iter().product();
                raw.shape = vec![out as u64, cols].into();
                let w = Weight::from_raw(graph, &raw)?;
                let flat = match w {
                    Weight::Quantized(q) => q.to_tensor(),
                    Weight::Dense(d) => {
                        let d = if d.dtype() == fusor::Dtype::F32 {
                            d
                        } else {
                            d.cast(fusor::Dtype::F32)?
                        };
                        Tensor::<2>::from_dyn(d)
                    }
                };
                Ok(flat.reshape([out, in_channels, meta.patch_size * meta.patch_size]))
            };
            let t0 = slice("v.patch_embd.weight")?;
            let t1 = slice("v.patch_embd.weight.1")?;
            let out = t0.dim(0);
            let stacked: Tensor<4> = fusor::stack([t0, t1], 2);
            stacked
                .reshape([
                    out,
                    in_channels * temporal_patch_size * meta.patch_size * meta.patch_size,
                ])
                .detach()
        };

        let blocks = (0..meta.block_count)
            .map(|i| {
                VisionBlock::load(
                    &vb.pp(format!("v.blk.{i}")),
                    device,
                    meta.head_count,
                    head_dim,
                    meta.layer_norm_eps,
                )
            })
            .collect::<fusor::Result<Vec<_>>>()?;
        let merger = merger::PatchMerger::load(
            &vb,
            device,
            meta.hidden_size,
            spatial_merge_size,
            meta.layer_norm_eps,
        )?;

        Ok(Self {
            spatial_merge_size,
            patch_size: meta.patch_size,
            temporal_patch_size,
            window_size: 112,
            fullatt_block_indexes: meta.fullatt_block_indexes,
            patch_embed,
            rotary_inv_freq: create_inverse_frequency(None, None, head_dim / 2, 10_000.0),
            blocks,
            merger,
            image_mean: meta.image_mean,
            image_std: meta.image_std,
            device: device.clone(),
            graphs: Mutex::new(HashMap::new()),
        })
    }

    /// Tokens one image occupies in the language model after merging.
    pub(crate) fn tokens_for(&self, grid: [u32; 3]) -> usize {
        let [t, h, w] = grid;
        (t as usize * h as usize * w as usize) / self.spatial_merge_size.pow(2)
    }

    /// Resize and patchify one image: `[patches, 3 * 2 * p * p]` pixels and
    /// the `[t, h, w]` patch grid.
    pub(crate) fn preprocess_image(
        &self,
        image: &::image::DynamicImage,
        min_pixels: Option<u32>,
        max_pixels: Option<u32>,
    ) -> (Tensor<2>, [u32; 3]) {
        let (data, grid, width) = image::patchify(
            image,
            self.patch_size,
            self.spatial_merge_size,
            self.temporal_patch_size,
            min_pixels,
            max_pixels,
            &self.image_mean,
            &self.image_std,
        );
        let rows = data.len() / width;
        (Tensor::from_slice(&self.device, [rows, width], &data), grid)
    }

    /// The per-patch rotary tables in window order: `[patches, head_dim / 2]`
    /// cos and sin, the first half over the row index and the second over the
    /// column index.
    fn rotary_rows(&self, grid: [u32; 3], window_index: &[u32]) -> (Vec<f32>, Vec<f32>) {
        let [t, h, w] = [grid[0] as usize, grid[1] as usize, grid[2] as usize];
        let m = self.spatial_merge_size;
        // (row, col) per patch in merge-block order: the order the patches
        // arrive in from `patchify`.
        let mut pos: Vec<(usize, usize)> = Vec::with_capacity(t * h * w);
        for _ in 0..t {
            for hb in 0..h / m {
                for wb in 0..w / m {
                    for i in 0..m {
                        for j in 0..m {
                            pos.push((hb * m + i, wb * m + j));
                        }
                    }
                }
            }
        }
        // The window permutation moves whole merge blocks.
        let unit = m * m;
        let mut ordered = Vec::with_capacity(pos.len());
        for &block in window_index {
            let start = block as usize * unit;
            ordered.extend_from_slice(&pos[start..start + unit]);
        }
        let half = self.rotary_inv_freq.len();
        let mut cos = Vec::with_capacity(ordered.len() * 2 * half);
        let mut sin = Vec::with_capacity(ordered.len() * 2 * half);
        for (r, c) in ordered {
            for axis in [r, c] {
                for f in &self.rotary_inv_freq {
                    let angle = axis as f64 * *f as f64;
                    cos.push(angle.cos() as f32);
                    sin.push(angle.sin() as f32);
                }
            }
        }
        (cos, sin)
    }

    /// The block graph for one image grid, built on first use.
    fn block_graph(
        &self,
        grid: [u32; 3],
        seq_len: usize,
        window_index: &[u32],
        cu_window_seqlens: &[u32],
    ) -> fusor::Result<Arc<BlockGraph>> {
        if let Some(g) = self.graphs.lock().unwrap().get(&grid) {
            return Ok(Arc::clone(g));
        }
        let dim = self.patch_embed.dim(0);
        let half = 2 * self.rotary_inv_freq.len();
        let device = &self.device;
        let x = Tensor::<3>::leaf(
            device,
            [
                Dim::Const(1),
                Dim::Const(seq_len as u64),
                Dim::Const(dim as u64),
            ],
        );
        let cos = Tensor::<2>::leaf(
            device,
            [Dim::Const(seq_len as u64), Dim::Const(half as u64)],
        );
        let sin = Tensor::<2>::leaf(
            device,
            [Dim::Const(seq_len as u64), Dim::Const(half as u64)],
        );
        let (cos_rows, sin_rows) = self.rotary_rows(grid, window_index);
        cos.set_elements(&cos_rows);
        sin.set_elements(&sin_rows);
        let slots = self.blocks[0].slot(device)?;
        let full = [0u32, seq_len as u32];
        let out_window = slots.forward(&x, cu_window_seqlens, &cos, &sin);
        let out_full = slots.forward(&x, &full, &cos, &sin);
        let g = Arc::new(BlockGraph {
            x,
            slots,
            out_window,
            out_full,
        });
        self.graphs.lock().unwrap().insert(grid, Arc::clone(&g));
        Ok(g)
    }

    /// One image's patches to `[tokens, llm_dim]` embeddings, materialized.
    pub(crate) fn forward_image(
        &self,
        patches: &Tensor<2>,
        grid: [u32; 3],
    ) -> fusor::Result<Tensor<2>> {
        let seq_len = patches.dim(0);
        let dim = self.patch_embed.dim(0);
        let unit = self.spatial_merge_size * self.spatial_merge_size;
        let (window_index, cu_window_seqlens) = window::window_index(
            grid,
            self.window_size,
            self.spatial_merge_size,
            self.patch_size,
        );
        let g = self.block_graph(grid, seq_len, &window_index, &cu_window_seqlens)?;

        // Patch embedding, permuted into window order, into the slot.
        let index = Tensor::from_slice(&self.device, [window_index.len()], &window_index);
        let x0 = patches
            .matmul_t(&self.patch_embed)
            .reshape([seq_len / unit, unit, dim])
            .index_select(0, &index)
            .reshape([1, seq_len, dim])
            .materialize();
        g.x.as_dyn().adopt_buffer(x0.as_dyn())?;
        x0.as_dyn().clear_device_buf();

        for (i, block) in self.blocks.iter().enumerate() {
            g.slots.bind(block)?;
            let root = if self.fullatt_block_indexes.contains(&i) {
                &g.out_full
            } else {
                &g.out_window
            };
            self.device
                .session()
                .resolve(std::slice::from_ref(root.as_dyn()))?;
            g.x.as_dyn().adopt_buffer(root.as_dyn())?;
            root.as_dyn().clear_device_buf();
        }

        // Merge, then undo the window permutation on the merged tokens.
        let mut reverse = vec![0u32; window_index.len()];
        for (i, &block) in window_index.iter().enumerate() {
            reverse[block as usize] = i as u32;
        }
        let reverse = Tensor::from_slice(&self.device, [reverse.len()], &reverse);
        Ok(self
            .merger
            .forward(&g.x.reshape([seq_len, dim]))
            .index_select(0, &reverse)
            .materialize())
    }
}

struct VisionBlock {
    ln1: Tensor<1>,
    ln2: Tensor<1>,
    eps: f32,
    attn: VisionAttention,
    gate: Weight,
    gate_bias: Tensor<1>,
    up: Weight,
    up_bias: Tensor<1>,
    down: Weight,
    down_bias: Tensor<1>,
}

/// `t` as an external leaf: itself when it already is one, else
/// materialized once. A slot can only adopt the buffer of a leaf, and a
/// weight that went through a cast at load is a computed node.
fn leafed_1d(t: Tensor<1>) -> Tensor<1> {
    if t.as_dyn().is_external_leaf() {
        t
    } else {
        t.materialize()
    }
}

fn leafed_weight(w: Weight) -> Weight {
    match w {
        Weight::Dense(d) if !d.is_external_leaf() => {
            Weight::Dense(d.materialize().expect("materialize a dense weight"))
        }
        other => other,
    }
}

/// An empty `[n]` leaf shaped like `t`.
fn slot_1d(t: &Tensor<1>, device: &Device) -> Tensor<1> {
    Tensor::<1>::leaf(device, [Dim::Const(t.dim(0) as u64)])
}

/// Rebind slot `to` to the device buffer of `from`, uploading it first.
fn bind_1d(to: &Tensor<1>, from: &Tensor<1>) -> fusor::Result<()> {
    from.as_dyn().upload()?;
    to.as_dyn().adopt_buffer(from.as_dyn())
}

fn bind_weight(to: &Weight, from: &Weight) -> fusor::Result<()> {
    from.upload()?;
    to.adopt(from)
}

impl VisionBlock {
    fn load(
        vb: &VarBuilder,
        device: &Device,
        head_count: usize,
        head_dim: usize,
        eps: f32,
    ) -> fusor::Result<Self> {
        let graph = device.graph();
        let weight = |name: &str| Weight::from_raw(graph, &vb.get_raw(name)?).map(leafed_weight);
        let dense = |name: &str| dense_1d(device, &vb.get_raw(name)?).map(leafed_1d);
        Ok(Self {
            ln1: dense("ln1.weight")?,
            ln2: dense("ln2.weight")?,
            eps,
            attn: VisionAttention::load(vb, device, head_count, head_dim)?,
            gate: weight("ffn_gate.weight")?,
            gate_bias: dense("ffn_gate.bias")?,
            up: weight("ffn_up.weight")?,
            up_bias: dense("ffn_up.bias")?,
            down: weight("ffn_down.weight")?,
            down_bias: dense("ffn_down.bias")?,
        })
    }

    /// A block of empty leaves shaped like this one.
    fn slot(&self, device: &Device) -> fusor::Result<Self> {
        let graph = device.graph();
        Ok(Self {
            ln1: slot_1d(&self.ln1, device),
            ln2: slot_1d(&self.ln2, device),
            eps: self.eps,
            attn: self.attn.slot(device)?,
            gate: self.gate.slot(graph)?,
            gate_bias: slot_1d(&self.gate_bias, device),
            up: self.up.slot(graph)?,
            up_bias: slot_1d(&self.up_bias, device),
            down: self.down.slot(graph)?,
            down_bias: slot_1d(&self.down_bias, device),
        })
    }

    /// Rebind every slot of this block to `from`'s buffers.
    fn bind(&self, from: &Self) -> fusor::Result<()> {
        bind_1d(&self.ln1, &from.ln1)?;
        bind_1d(&self.ln2, &from.ln2)?;
        self.attn.bind(&from.attn)?;
        bind_weight(&self.gate, &from.gate)?;
        bind_1d(&self.gate_bias, &from.gate_bias)?;
        bind_weight(&self.up, &from.up)?;
        bind_1d(&self.up_bias, &from.up_bias)?;
        bind_weight(&self.down, &from.down)?;
        bind_1d(&self.down_bias, &from.down_bias)
    }

    /// `[1, seq, dim]` in and out.
    fn forward(
        &self,
        x: &Tensor<3>,
        cu_seqlens: &[u32],
        cos: &Tensor<2>,
        sin: &Tensor<2>,
    ) -> Tensor<3> {
        let attn = self
            .attn
            .forward(&x.rms_norm(&self.ln1, self.eps), cu_seqlens, cos, sin);
        let x = x.add(&attn);
        let h = x.rms_norm(&self.ln2, self.eps);
        let gate: Tensor<3> = self.gate.mat_mul(&h).add_(&self.gate_bias);
        let up: Tensor<3> = self.up.mat_mul(&h).add_(&self.up_bias);
        let mlp: Tensor<3> = self
            .down
            .mat_mul(&gate.silu().mul(&up))
            .add_(&self.down_bias);
        x.add(&mlp)
    }
}

struct VisionAttention {
    qkv: Weight,
    qkv_bias: Tensor<1>,
    proj: Weight,
    proj_bias: Tensor<1>,
    head_count: usize,
    head_dim: usize,
}

impl VisionAttention {
    fn load(
        vb: &VarBuilder,
        device: &Device,
        head_count: usize,
        head_dim: usize,
    ) -> fusor::Result<Self> {
        let graph = device.graph();
        let (q, k, v) = (
            Weight::from_raw(graph, &vb.get_raw("attn_q.weight")?)?,
            Weight::from_raw(graph, &vb.get_raw("attn_k.weight")?)?,
            Weight::from_raw(graph, &vb.get_raw("attn_v.weight")?)?,
        );
        let qkv = match Weight::concat_rows(&[&q, &k, &v]) {
            Some(fused) => fused,
            None => {
                // A dense projection concatenates as a plain tensor.
                let dense = |w: Weight| -> fusor::Result<Tensor<2>> {
                    Ok(match w {
                        Weight::Quantized(q) => q.to_tensor(),
                        Weight::Dense(d) => {
                            Tensor::<2>::from_dyn(if d.dtype() == fusor::Dtype::F32 {
                                d
                            } else {
                                d.cast(fusor::Dtype::F32)?
                            })
                        }
                    })
                };
                let cat = Tensor::cat([dense(q)?, dense(k)?, dense(v)?], 0).materialize();
                Weight::Dense(cat.into_dyn())
            }
        };
        let bias = |name: &str| dense_1d(device, &vb.get_raw(name)?);
        let qkv_bias = Tensor::cat(
            [
                bias("attn_q.bias")?,
                bias("attn_k.bias")?,
                bias("attn_v.bias")?,
            ],
            0,
        )
        .materialize();
        Ok(Self {
            qkv: leafed_weight(qkv),
            qkv_bias,
            proj: leafed_weight(Weight::from_raw(graph, &vb.get_raw("attn_out.weight")?)?),
            proj_bias: leafed_1d(bias("attn_out.bias")?),
            head_count,
            head_dim,
        })
    }

    fn slot(&self, device: &Device) -> fusor::Result<Self> {
        let graph = device.graph();
        Ok(Self {
            qkv: self.qkv.slot(graph)?,
            qkv_bias: slot_1d(&self.qkv_bias, device),
            proj: self.proj.slot(graph)?,
            proj_bias: slot_1d(&self.proj_bias, device),
            head_count: self.head_count,
            head_dim: self.head_dim,
        })
    }

    fn bind(&self, from: &Self) -> fusor::Result<()> {
        bind_weight(&self.qkv, &from.qkv)?;
        bind_1d(&self.qkv_bias, &from.qkv_bias)?;
        bind_weight(&self.proj, &from.proj)?;
        bind_1d(&self.proj_bias, &from.proj_bias)
    }

    /// `[1, seq, dim]` in and out; attention is block-diagonal over the
    /// `cu_seqlens` boundaries.
    fn forward(
        &self,
        x: &Tensor<3>,
        cu_seqlens: &[u32],
        cos: &Tensor<2>,
        sin: &Tensor<2>,
    ) -> Tensor<3> {
        let seq_len = x.dim(1);
        let dim = self.head_count * self.head_dim;
        let qkv = self.qkv.mat_mul(x).add_(&self.qkv_bias);
        let heads = |t: Tensor<3>| -> Tensor<4> {
            t.reshape([seq_len, self.head_count, self.head_dim])
                .transpose(0, 1)
                .unsqueeze(0)
        };
        let q = heads(qkv.narrow(Minus1, 0, dim));
        let k = heads(qkv.narrow(Minus1, dim, dim));
        let v = heads(qkv.narrow(Minus1, 2 * dim, dim));
        let (q, k) = q.rope_pair(&k, cos, sin, 0);
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        let full = cu_seqlens.len() == 2 && cu_seqlens[0] == 0 && cu_seqlens[1] as usize == seq_len;
        let attn: Tensor<4> = if full {
            q.attention(&k, &v, MaskKind::None, Some(scale))
        } else {
            // Windows of equal length batch as one attention call: the run
            // `[1, heads, run * len, hd]` viewed as `[run, heads, len, hd]`.
            let windows: Vec<(usize, usize)> = cu_seqlens
                .windows(2)
                .map(|p| (p[0] as usize, (p[1] - p[0]) as usize))
                .filter(|(_, len)| *len > 0)
                .collect();
            let mut outs = Vec::new();
            let mut i = 0;
            while i < windows.len() {
                let (start, len) = windows[i];
                let mut run = 1;
                while i + run < windows.len() && windows[i + run].1 == len {
                    run += 1;
                }
                let batched = |t: &Tensor<4>| -> Tensor<4> {
                    t.narrow(2, start, run * len)
                        .reshape([self.head_count, run, len, self.head_dim])
                        .transpose(0, 1)
                };
                let out =
                    batched(&q).attention(&batched(&k), &batched(&v), MaskKind::None, Some(scale));
                outs.push(out.transpose(0, 1).reshape([
                    1,
                    self.head_count,
                    run * len,
                    self.head_dim,
                ]));
                i += run;
            }
            Tensor::cat(outs, 2)
        };
        let merged = attn.transpose(1, 2).reshape([1, seq_len, dim]);
        self.proj.mat_mul(&merged).add_(&self.proj_bias)
    }
}
