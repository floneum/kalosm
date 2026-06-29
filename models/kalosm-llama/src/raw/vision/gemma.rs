use fusor::{
    layers::RmsNorm, AddOp, CastTensor, CastTo, Device, DivOp, ExpOp, FloatDataType, FloatOps,
    Fusion, MatmulImpl, MulOp, NegOp, QMatrix, Result, SimdBinaryOp, SimdElement, SimdUnaryOp,
    Tensor, VarBuilder,
};
use fusor_gguf::GgufMetadata;

use crate::raw::rope::create_inverse_frequency;

pub(crate) struct GemmaVisionTransformer<F: FloatDataType + SimdElement = f32> {
    patch_size: usize,
    merge_size: usize,
    patch_embed: GemmaVisionPatchEmbed<F>,
    position_embeddings: Tensor<3, F>,
    blocks: Vec<GemmaVisionBlock<F>>,
    projector_norm: RmsNorm<1, F>,
    projector: GemmaClippedLinear,
    std_bias: Option<Tensor<1, f32>>,
    std_scale: Option<Tensor<1, f32>>,
    image_mean: Vec<f32>,
    image_std: Vec<f32>,
    rope_theta: f32,
    device: Device,
    uses_isolated_device: bool,
}

impl<F> GemmaVisionTransformer<F>
where
    F: FloatDataType
        + SimdElement
        + FloatOps
        + MatmulImpl
        + Default
        + CastTo<f32>
        + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
    fusor::MulOp: fusor::SimdBinaryOp<F>,
    fusor::AddOp: fusor::SimdBinaryOp<F>,
    fusor::SumOp: fusor::SimdReduceOp<F>,
{
    pub(crate) fn from_gguf(
        vision_ct: GgufMetadata,
        vision_bytes: &[u8],
        device: &Device,
    ) -> Result<Self> {
        let uses_isolated_device = cfg!(not(target_arch = "wasm32"))
            && device.is_gpu()
            && std::env::var_os("KALOSM_GEMMA4_VISION_DISABLE_SUBGROUPS").is_some();
        let vision_device = if uses_isolated_device {
            // Debug fallback: use the no-subgroup sibling while sharing the
            // same GPU adapter. The normal path is faster and covered by the
            // split/combine long-attention tests.
            device.without_subgroups()
        } else {
            device.clone()
        };
        let device = &vision_device;
        let block_count = metadata_usize(&vision_ct, "clip.vision.block_count", 16);
        let head_count = metadata_usize(&vision_ct, "clip.vision.attention.head_count", 12);
        let patch_size = metadata_usize(&vision_ct, "clip.vision.patch_size", 16);
        let hidden_size = metadata_usize(&vision_ct, "clip.vision.embedding_length", 768);
        let merge_size = metadata_usize(&vision_ct, "clip.vision.projector.scale_factor", 3);
        let rope_theta = vision_ct
            .metadata
            .get("clip.vision.rope_theta")
            .and_then(|x| x.to_f64().ok())
            .unwrap_or(100.0) as f32;
        let layer_norm_eps = vision_ct
            .metadata
            .get("clip.vision.attention.layer_norm_epsilon")
            .and_then(|x| x.to_f64().ok())
            .unwrap_or(1e-6) as f32;
        let image_mean = metadata_f32_array(&vision_ct, "clip.vision.image_mean")
            .unwrap_or_else(|| vec![0.0, 0.0, 0.0]);
        let image_std = metadata_f32_array(&vision_ct, "clip.vision.image_std")
            .unwrap_or_else(|| vec![1.0, 1.0, 1.0]);

        let mut cursor = std::io::Cursor::new(vision_bytes);
        let mut vb = VarBuilder::from_gguf(&mut cursor)?;
        let patch_embed = GemmaVisionPatchEmbed::new(
            patch_size,
            hidden_size,
            &mut vb.pp("v.patch_embd"),
            device,
        )?;
        let position_embeddings: Tensor<3, F> = vb
            .get("v.position_embd.weight", device)?
            .dequantize()
            .cast();
        let head_dim = hidden_size / head_count;
        let blocks = (0..block_count)
            .map(|i| {
                GemmaVisionBlock::new(
                    &mut vb.pp(format!("v.blk.{i}")),
                    device,
                    head_count,
                    head_dim,
                    hidden_size,
                    layer_norm_eps,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let projector_norm =
            RmsNorm::new(Tensor::ones(device, [hidden_size]), None, layer_norm_eps);
        let projector = clipped_linear(&mut vb, device, "mm.input_projection")?;
        let std_bias: Option<Tensor<1, f32>> =
            vb.get("v.std_bias", device).ok().map(|x| x.dequantize());
        let std_scale: Option<Tensor<1, f32>> =
            vb.get("v.std_scale", device).ok().map(|x| x.dequantize());

        Ok(Self {
            patch_size,
            merge_size,
            patch_embed,
            position_embeddings,
            blocks,
            projector_norm,
            projector,
            std_bias,
            std_scale,
            image_mean,
            image_std,
            rope_theta,
            device: device.clone(),
            uses_isolated_device,
        })
    }

    pub(crate) fn uses_isolated_device(&self) -> bool {
        self.uses_isolated_device
    }

    pub(crate) fn preprocess_image(
        &self,
        image: &image::DynamicImage,
        min_pixels: Option<u32>,
        max_pixels: Option<u32>,
    ) -> Result<(Tensor<2, f32>, [u32; 3])> {
        let (target_width, target_height) = self.target_image_size(image, min_pixels, max_pixels);
        let resized = image.resize_exact(
            target_width as u32,
            target_height as u32,
            image::imageops::FilterType::Triangle,
        );
        let rgb = image_to_rgb(&resized, &self.device)?;
        let mean_tensor = Tensor::new(&self.device, &self.image_mean);
        let mean = mean_tensor.reshape([1, 3, 1, 1]);
        let std_tensor = Tensor::new(&self.device, &self.image_std);
        let std = std_tensor.reshape([1, 3, 1, 1]);
        let rgb = rgb
            .sub_(&mean)
            .div_(&std)
            .mul_scalar(2.0)
            .add_scalar(-1.0)
            .to_concrete();
        let grid_h = target_height / self.patch_size;
        let grid_w = target_width / self.patch_size;
        if std::env::var_os("KALOSM_TRACE_VISION_STATS").is_some() {
            tracing::info!(
                "[vision_stats] preprocess original={}x{} target={}x{} grid=[1,{grid_h},{grid_w}] pooled_tokens={}",
                image.width(),
                image.height(),
                target_width,
                target_height,
                (grid_h / self.merge_size) * (grid_w / self.merge_size)
            );
        }
        let patches = rgb
            .reshape([1, 3, grid_h, self.patch_size, grid_w, self.patch_size])
            .permute([0, 2, 4, 1, 3, 5])
            .reshape([grid_h * grid_w, 3 * self.patch_size * self.patch_size])
            .to_concrete();
        crate::raw::debug_tensor_stats_f32(&patches, "image_patches");

        Ok((patches, [1, grid_h as u32, grid_w as u32]))
    }

    pub(crate) fn image_token_count(&self, grid: [u32; 3]) -> u32 {
        grid[0] * (grid[1] / self.merge_size as u32) * (grid[2] / self.merge_size as u32)
    }

    pub(crate) fn forward_image(
        &self,
        pixels: &Tensor<2, F>,
        grid: [u32; 3],
    ) -> Result<Tensor<2, F>> {
        let [pos_x, pos_y] = self.patch_positions(grid)?;
        let (cos_x, sin_x) = self.rope_sin_cos(&pos_x)?;
        let (cos_y, sin_y) = self.rope_sin_cos(&pos_y)?;
        let mut hidden_states = self.patch_embed.forward(pixels);
        let hidden_f32: Tensor<2, f32> = hidden_states.cast();
        crate::raw::debug_tensor_stats_f32(&hidden_f32, "patch_embeds");
        hidden_states = self.add_position_embeddings(&hidden_states, &pos_x, &pos_y, grid)?;
        let hidden_f32: Tensor<2, f32> = hidden_states.cast();
        crate::raw::debug_tensor_stats_f32(&hidden_f32, "patch_plus_position");
        let mut hidden_states = hidden_states.unsqueeze(0).to_concrete();
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states, &cos_x, &sin_x, &cos_y, &sin_y);
        }
        let hidden_f32: Tensor<3, f32> = hidden_states.cast();
        crate::raw::debug_tensor_stats_f32(&hidden_f32, "vision_blocks_out");
        let mut hidden_states = self.pool(hidden_states, grid)?;
        let hidden_f32: Tensor<3, f32> = hidden_states.cast();
        crate::raw::debug_tensor_stats_f32(&hidden_f32, "vision_pooled");
        if let (Some(std_bias), Some(std_scale)) = (&self.std_bias, &self.std_scale) {
            let hidden_f32: Tensor<3, f32> = hidden_states.cast();
            hidden_states = hidden_f32.sub_(std_bias).mul_(std_scale).cast();
        }
        let hidden_states = self.projector_norm.forward_generic(&hidden_states);
        let hidden_f32: Tensor<3, f32> = hidden_states.cast();
        crate::raw::debug_tensor_stats_f32(&hidden_f32, "vision_projector_norm");
        Ok(self
            .projector
            .forward(&hidden_states)
            .squeeze::<2>(0)
            .to_concrete())
    }

    fn target_image_size(
        &self,
        image: &image::DynamicImage,
        min_pixels: Option<u32>,
        max_pixels: Option<u32>,
    ) -> (usize, usize) {
        let align = self.patch_size * self.merge_size;
        let (min_pixels, max_pixels) = gemma_image_pixel_bounds(align, min_pixels, max_pixels);
        smart_resize(
            image.width() as usize,
            image.height() as usize,
            align,
            min_pixels,
            max_pixels,
        )
    }

    fn add_position_embeddings<B>(
        &self,
        hidden_states: &Tensor<2, F, B>,
        x_ids: &Tensor<1, u32>,
        y_ids: &Tensor<1, u32>,
        grid: [u32; 3],
    ) -> Result<Tensor<2, F>>
    where
        B: Fusion<2, F>,
    {
        let [grid_t, _grid_h, _grid_w] = grid;
        if grid_t != 1 {
            return Err(fusor::Error::msg(
                "Gemma 4 vision currently supports image inputs, not video frames.",
            ));
        }

        let pos_x_table = self.position_embeddings.i((0, .., ..)).to_concrete();
        let pos_y_table = self.position_embeddings.i((1, .., ..)).to_concrete();
        let pos_x = pos_x_table.index_select(0, x_ids);
        let pos_y = pos_y_table.index_select(0, y_ids);
        let position_embeddings = (pos_x + pos_y).to_concrete();
        Ok((hidden_states.to_concrete() + position_embeddings).to_concrete())
    }

    fn patch_positions(&self, grid: [u32; 3]) -> Result<[Tensor<1, u32>; 2]> {
        let [grid_t, grid_h, grid_w] = grid;
        if grid_t != 1 {
            return Err(fusor::Error::msg(
                "Gemma 4 vision currently supports image inputs, not video frames.",
            ));
        }
        let mut y_ids = Vec::with_capacity((grid_h * grid_w) as usize);
        let mut x_ids = Vec::with_capacity((grid_h * grid_w) as usize);
        for y in 0..grid_h {
            for x in 0..grid_w {
                y_ids.push(y);
                x_ids.push(x);
            }
        }
        Ok([
            Tensor::new(&self.device, &x_ids),
            Tensor::new(&self.device, &y_ids),
        ])
    }

    fn rope_sin_cos(&self, positions: &Tensor<1, u32>) -> Result<(Tensor<2, f32>, Tensor<2, f32>)> {
        let half_head_dim = self
            .blocks
            .first()
            .map(|block| block.head_dim() / 2)
            .ok_or_else(|| {
                fusor::Error::msg("Gemma 4 vision transformer must have at least one block")
            })?;
        let positions: Tensor<2, f32> = positions
            .cast::<f32>()
            .reshape([positions.shape()[0], 1])
            .to_concrete();
        let inv_freq: Tensor<2, f32> = create_inverse_frequency::<f32>(
            None,
            None,
            half_head_dim,
            self.rope_theta,
            &self.device,
        );
        let freqs = positions.matmul(&inv_freq);
        Ok((freqs.cos().to_concrete(), freqs.sin().to_concrete()))
    }

    fn pool(&self, hidden_states: Tensor<3, F>, grid: [u32; 3]) -> Result<Tensor<3, F>> {
        let [batch, seq, hidden] = hidden_states.shape();
        let grid_h = grid[1] as usize;
        let grid_w = grid[2] as usize;
        if batch != 1 || seq != grid_h * grid_w {
            return Err(fusor::Error::msg(
                "Gemma 4 vision grid does not match hidden states",
            ));
        }
        if grid_h % self.merge_size != 0 || grid_w % self.merge_size != 0 {
            return Err(fusor::Error::msg(
                "Gemma 4 vision grid must be divisible by the merge size",
            ));
        }

        let out_h = grid_h / self.merge_size;
        let out_w = grid_w / self.merge_size;
        let pooled = hidden_states
            .reshape([
                batch,
                out_h,
                self.merge_size,
                out_w,
                self.merge_size,
                hidden,
            ])
            .sum::<5>(4)
            .mul_scalar(F::from_f32(1.0 / self.merge_size as f32))
            .sum::<4>(2)
            .mul_scalar(F::from_f32(1.0 / self.merge_size as f32))
            .reshape([batch, out_h * out_w, hidden])
            .mul_scalar(F::from_f32((hidden as f32).sqrt()))
            .to_concrete();
        Ok(pooled)
    }
}

fn gemma_image_pixel_bounds(
    align: usize,
    min_pixels: Option<u32>,
    max_pixels: Option<u32>,
) -> (usize, usize) {
    (
        min_pixels
            .map(|pixels| pixels as usize)
            .unwrap_or(40 * align * align),
        max_pixels
            .map(|pixels| pixels as usize)
            .unwrap_or(280 * align * align),
    )
}

struct GemmaVisionPatchEmbed<F: FloatDataType + SimdElement> {
    weight: Tensor<2, F>,
}

impl<F> GemmaVisionPatchEmbed<F>
where
    F: FloatDataType + SimdElement + FloatOps + MatmulImpl + CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    fn new(
        patch_size: usize,
        hidden_size: usize,
        vb: &mut VarBuilder,
        device: &Device,
    ) -> Result<Self> {
        let weight: Tensor<4, F> = vb.get("weight", device)?.dequantize().cast();
        let weight = weight
            .permute([1, 2, 3, 0])
            .reshape([3 * patch_size * patch_size, hidden_size])
            .to_concrete();
        Ok(Self { weight })
    }

    fn forward<B>(&self, pixels: &Tensor<2, F, B>) -> Tensor<2, F>
    where
        B: Fusion<2, F>,
    {
        pixels.matmul(&self.weight)
    }
}

struct GemmaVisionBlock<F: FloatDataType + SimdElement> {
    norm1: RmsNorm<1, F>,
    norm2: RmsNorm<1, F>,
    attn: GemmaVisionAttention<F>,
    attn_post_norm: RmsNorm<1, F>,
    mlp: GemmaVisionFeedForward,
    ffn_post_norm: RmsNorm<1, F>,
}

impl<F> GemmaVisionBlock<F>
where
    F: FloatDataType + SimdElement + Default + CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    fn new(
        vb: &mut VarBuilder,
        device: &Device,
        head_count: usize,
        head_dim: usize,
        hidden_size: usize,
        layer_norm_eps: f32,
    ) -> Result<Self> {
        let norm1 = rms_norm(vb.get("ln1.weight", device)?, layer_norm_eps);
        let norm2 = rms_norm(vb.get("ln2.weight", device)?, layer_norm_eps);
        let attn_post_norm = rms_norm(vb.get("attn_post_norm.weight", device)?, layer_norm_eps);
        let ffn_post_norm = rms_norm(vb.get("ffn_post_norm.weight", device)?, layer_norm_eps);
        let attn = GemmaVisionAttention::new(vb, device, head_count, head_dim, hidden_size)?;
        let mlp = GemmaVisionFeedForward::new(vb, device)?;

        Ok(Self {
            norm1,
            norm2,
            attn,
            attn_post_norm,
            mlp,
            ffn_post_norm,
        })
    }

    fn head_dim(&self) -> usize {
        self.attn.head_dim
    }

    fn forward<B>(
        &self,
        xs: &Tensor<3, F, B>,
        cos_x: &Tensor<2, f32>,
        sin_x: &Tensor<2, f32>,
        cos_y: &Tensor<2, f32>,
        sin_y: &Tensor<2, f32>,
    ) -> Tensor<3, F>
    where
        B: Fusion<3, F>,
    {
        let residual: Tensor<3, f32> = xs.cast();
        let x = self.norm1.forward_generic(xs);
        let attn = self.attn.forward(&x, cos_x, sin_x, cos_y, sin_y);
        let attn = self.attn_post_norm.forward_generic(&attn);
        let attn_f32: Tensor<3, f32> = attn.cast();
        let x = self.norm2.forward_residual_f32(&attn_f32, &residual);
        let ffn = self.mlp.forward(&x);
        let ffn = self.ffn_post_norm.forward_generic(&ffn);
        let ffn_f32: Tensor<3, f32> = ffn.cast();
        (ffn_f32 + attn_f32 + residual).cast()
    }
}

struct GemmaVisionAttention<F: FloatDataType + SimdElement> {
    q: GemmaClippedLinear,
    k: GemmaClippedLinear,
    v: GemmaClippedLinear,
    out: GemmaClippedLinear,
    q_norm: RmsNorm<1, F>,
    k_norm: RmsNorm<1, F>,
    v_norm: RmsNorm<1, F>,
    head_count: usize,
    head_dim: usize,
    hidden_size: usize,
}

impl<F> GemmaVisionAttention<F>
where
    F: FloatDataType + SimdElement + Default + CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    fn new(
        vb: &mut VarBuilder,
        device: &Device,
        head_count: usize,
        head_dim: usize,
        hidden_size: usize,
    ) -> Result<Self> {
        Ok(Self {
            q: clipped_linear(vb, device, "attn_q")?,
            k: clipped_linear(vb, device, "attn_k")?,
            v: clipped_linear(vb, device, "attn_v")?,
            out: clipped_linear(vb, device, "attn_out")?,
            q_norm: rms_norm(vb.get("attn_q_norm.weight", device)?, 1e-6),
            k_norm: rms_norm(vb.get("attn_k_norm.weight", device)?, 1e-6),
            v_norm: RmsNorm::new(Tensor::ones(device, [head_dim]), None, 1e-6),
            head_count,
            head_dim,
            hidden_size,
        })
    }

    fn forward<B>(
        &self,
        xs: &Tensor<3, F, B>,
        cos_x: &Tensor<2, f32>,
        sin_x: &Tensor<2, f32>,
        cos_y: &Tensor<2, f32>,
        sin_y: &Tensor<2, f32>,
    ) -> Tensor<3, F>
    where
        B: Fusion<3, F>,
    {
        let [batch, seq_len, _] = xs.shape();
        let q = self
            .q
            .forward(xs)
            .reshape([batch, seq_len, self.head_count, self.head_dim])
            .transpose(1, 2)
            .to_concrete();
        let k = self
            .k
            .forward(xs)
            .reshape([batch, seq_len, self.head_count, self.head_dim])
            .transpose(1, 2)
            .to_concrete();
        let v = self
            .v
            .forward(xs)
            .reshape([batch, seq_len, self.head_count, self.head_dim])
            .transpose(1, 2)
            .to_concrete();
        let q: Tensor<4, f32> = self.q_norm.forward_generic_4d(&q).cast();
        let k: Tensor<4, f32> = self.k_norm.forward_generic_4d(&k).cast();
        let v: Tensor<4, f32> = self.v_norm.forward_generic_4d(&v).cast();
        let half = self.head_dim / 2;
        let q_x = q.narrow(3, 0, half).to_concrete();
        let k_x = k.narrow(3, 0, half).to_concrete();
        let q_y = q.narrow(3, half, half).to_concrete();
        let k_y = k.narrow(3, half, half).to_concrete();
        let (q_x, k_x) = q_x.rope_normal_pair_fused(&k_x, cos_x, sin_x);
        let (q_y, k_y) = q_y.rope_normal_pair_fused(&k_y, cos_y, sin_y);
        let q = Tensor::cat([q_x, q_y], 3).to_concrete();
        let k = Tensor::cat([k_x, k_y], 3).to_concrete();
        let attn = q.flash_attention(&k, &v, 1.0, None);
        let attn = attn
            .transpose(1, 2)
            .reshape([batch, seq_len, self.hidden_size])
            .cast();
        self.out.forward(&attn)
    }
}

struct GemmaVisionFeedForward {
    gate: GemmaClippedLinear,
    down: GemmaClippedLinear,
    up: GemmaClippedLinear,
}

impl GemmaVisionFeedForward {
    fn new(vb: &mut VarBuilder, device: &Device) -> Result<Self> {
        Ok(Self {
            gate: clipped_linear(vb, device, "ffn_gate")?,
            down: clipped_linear(vb, device, "ffn_down")?,
            up: clipped_linear(vb, device, "ffn_up")?,
        })
    }

    fn forward<F, B>(&self, x: &Tensor<3, F, B>) -> Tensor<3, F>
    where
        F: FloatDataType + SimdElement + Default + CastTo<f32> + CastTensor<f32>,
        f32: CastTo<F> + CastTensor<F>,
        B: Fusion<3, F>,
    {
        let gate = quick_gelu(&self.gate.forward_f32(x));
        let up = self.up.forward_f32(x);
        let hidden = (gate * up).to_concrete();
        self.down.forward_from_f32(&hidden).cast()
    }
}

fn quick_gelu<const R: usize, B>(x: &Tensor<R, f32, B>) -> Tensor<R, f32>
where
    B: Fusion<R, f32>,
    AddOp: SimdBinaryOp<f32>,
    DivOp: SimdBinaryOp<f32>,
    ExpOp: SimdUnaryOp<f32>,
    MulOp: SimdBinaryOp<f32>,
    NegOp: SimdUnaryOp<f32>,
{
    let x = x.to_concrete();
    let scaled = (x.clone() * 1.702).to_concrete();
    let sigmoid = scaled.sigmoid();
    (x * sigmoid).to_concrete()
}

#[derive(Clone, Copy)]
struct ClampInfo {
    input_min: f32,
    input_max: f32,
    output_min: f32,
    output_max: f32,
}

impl ClampInfo {
    fn from_tensors(vb: &mut VarBuilder, device: &Device, prefix: &str) -> Self {
        Self {
            input_min: scalar_tensor(vb, device, &format!("{prefix}.input_min"), -f32::MAX),
            input_max: scalar_tensor(vb, device, &format!("{prefix}.input_max"), f32::MAX),
            output_min: scalar_tensor(vb, device, &format!("{prefix}.output_min"), -f32::MAX),
            output_max: scalar_tensor(vb, device, &format!("{prefix}.output_max"), f32::MAX),
        }
    }

    fn has_input_clamp(self) -> bool {
        self.input_min > -f32::MAX || self.input_max < f32::MAX
    }

    fn has_output_clamp(self) -> bool {
        self.output_min > -f32::MAX || self.output_max < f32::MAX
    }
}

struct GemmaClippedLinear {
    weight: QMatrix,
    clamp: ClampInfo,
}

impl GemmaClippedLinear {
    fn new(weight: QMatrix, clamp: ClampInfo) -> Self {
        Self { weight, clamp }
    }

    fn forward<F, B>(&self, input: &Tensor<3, F, B>) -> Tensor<3, F>
    where
        F: FloatDataType + SimdElement + Default + CastTo<f32> + CastTensor<f32>,
        f32: CastTo<F> + CastTensor<F>,
        B: Fusion<3, F>,
    {
        self.forward_f32(input).cast()
    }

    fn forward_f32<F, B>(&self, input: &Tensor<3, F, B>) -> Tensor<3, f32>
    where
        F: FloatDataType + SimdElement + Default + CastTo<f32> + CastTensor<f32>,
        B: Fusion<3, F>,
    {
        self.forward_from_f32(&input.cast::<f32>())
    }

    fn forward_from_f32<B>(&self, input: &Tensor<3, f32, B>) -> Tensor<3, f32>
    where
        B: Fusion<3, f32>,
    {
        let input = if self.clamp.has_input_clamp() {
            input
                .clamp(self.clamp.input_min, self.clamp.input_max)
                .to_concrete()
        } else {
            input.to_concrete()
        };
        let output = input.q_mat_mul(&self.weight);
        if self.clamp.has_output_clamp() {
            output
                .clamp(self.clamp.output_min, self.clamp.output_max)
                .to_concrete()
        } else {
            output
        }
    }
}

fn clipped_linear(
    vb: &mut VarBuilder,
    device: &Device,
    prefix: &str,
) -> Result<GemmaClippedLinear> {
    let weight = vb.get(&format!("{prefix}.weight"), device)?;
    let clamp = ClampInfo::from_tensors(vb, device, prefix);
    Ok(GemmaClippedLinear::new(weight, clamp))
}

fn scalar_tensor(vb: &mut VarBuilder, device: &Device, name: &str, default: f32) -> f32 {
    let Ok(tensor) = vb.get(name, device) else {
        return default;
    };
    let tensor: Tensor<1, f32> = tensor.dequantize();
    first_f32(&tensor).unwrap_or(default)
}

#[cfg(not(target_arch = "wasm32"))]
fn first_f32(tensor: &Tensor<1, f32>) -> Option<f32> {
    let slice = pollster::block_on(tensor.as_slice()).ok()?;
    slice.as_slice().first().copied()
}

#[cfg(target_arch = "wasm32")]
fn first_f32(_tensor: &Tensor<1, f32>) -> Option<f32> {
    None
}

fn rms_norm<F>(weight: QMatrix, eps: f32) -> RmsNorm<1, F>
where
    F: FloatDataType + SimdElement + Default + CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    let weight: Tensor<1, F> = weight.dequantize().cast();
    RmsNorm::new(weight, None, eps)
}

fn metadata_usize(metadata: &GgufMetadata, key: &str, default: usize) -> usize {
    metadata
        .metadata
        .get(key)
        .and_then(|x| x.to_u64().ok())
        .map(|x| x as usize)
        .unwrap_or(default)
}

fn smart_resize(
    width: usize,
    height: usize,
    align: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> (usize, usize) {
    let round_by_factor = |x: f64| ((x / align as f64).round() as usize).max(1) * align;
    let ceil_by_factor = |x: f64| ((x / align as f64).ceil() as usize).max(1) * align;
    let floor_by_factor = |x: f64| ((x / align as f64).floor() as usize).max(1) * align;

    let mut target_h = round_by_factor(height as f64);
    let mut target_w = round_by_factor(width as f64);
    let pixels = (width * height) as f64;
    if target_h * target_w > max_pixels {
        let beta = (pixels / max_pixels as f64).sqrt();
        target_h = floor_by_factor(height as f64 / beta);
        target_w = floor_by_factor(width as f64 / beta);
    } else if target_h * target_w < min_pixels {
        let beta = (min_pixels as f64 / pixels).sqrt();
        target_h = ceil_by_factor(height as f64 * beta);
        target_w = ceil_by_factor(width as f64 * beta);
    }

    (target_w, target_h)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma_image_pixel_bounds_honor_media_hints() {
        let align = 8;
        let requested_min = 10 * align * align;
        let requested_max = 20 * align * align;

        assert_eq!(
            gemma_image_pixel_bounds(
                align,
                Some(requested_min as u32),
                Some(requested_max as u32)
            ),
            (requested_min, requested_max),
            "Gemma preprocessing must honor caller-provided MediaHints image budgets"
        );
    }
}

fn metadata_f32_array(metadata: &GgufMetadata, key: &str) -> Option<Vec<f32>> {
    metadata.metadata.get(key).and_then(|value| {
        value.to_array().ok().map(|values| {
            values
                .iter()
                .filter_map(|value| value.to_f32().ok())
                .collect()
        })
    })
}

fn image_to_rgb(image: &image::DynamicImage, device: &Device) -> Result<Tensor<4, f32>> {
    let height = image.height() as usize;
    let width = image.width() as usize;
    let rgb = image.to_rgb8();
    let as_u32 = rgb
        .into_raw()
        .into_iter()
        .map(|x| x as u32)
        .collect::<Vec<_>>();
    let data_tensor = Tensor::new(device, &as_u32);
    let data = data_tensor.reshape([height, width, 3]);
    let img = data.permute([2, 0, 1]).cast::<f32>() * (1.0 / 255.0);

    Ok(img.unsqueeze(0).to_concrete())
}
