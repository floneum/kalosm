//! Prompt encoder: encodes points, boxes, and masks into embeddings.

use fusor::layers::{ConvNd, ConvNdConfig, Embedding, LayerNormNd};
use fusor::{ConcreteTensor, Device, Tensor, TensorBacking, VarBuilder};

use super::Result;

pub(crate) struct PositionEmbeddingRandom {
    pub(crate) positional_encoding_gaussian_matrix: Tensor<2, f32, ConcreteTensor<f32, 2>>,
}

impl PositionEmbeddingRandom {
    fn load(device: &Device, vb: &mut VarBuilder) -> Result<Self> {
        let m: Tensor<2, f32> = vb
            .get("positional_encoding_gaussian_matrix", device)?
            .dequantize();
        Ok(Self {
            positional_encoding_gaussian_matrix: m,
        })
    }

    fn pe_encoding(&self, coords: &Tensor<3, f32>) -> Tensor<3, f32> {
        // coords * 2 - 1
        let coords = coords.mul_scalar(2.0) + (-1.0f32);
        let shape = coords.shape();
        let b = shape[0];
        let d = self.positional_encoding_gaussian_matrix.shape()[1];
        let gm = self
            .positional_encoding_gaussian_matrix
            .reshape([1, shape[2], d]);
        let gm = gm.broadcast_as([b, shape[2], d]);
        let coords = coords.mat_mul(&gm);
        // coords * 2 * pi
        let coords = coords.mul_scalar(2.0 * std::f32::consts::PI);
        // cat([sin, cos], last_dim)
        let sin_coords = coords.sin().to_concrete();
        let cos_coords = coords.cos().to_concrete();
        Tensor::cat([sin_coords, cos_coords], 2)
    }

    pub(crate) fn forward(&self, h: usize, w: usize) -> Tensor<3, f32> {
        let device = self.positional_encoding_gaussian_matrix.device();
        // Create grid coordinates
        let x_embed: Tensor<1, f32> = fusor::arange_step::<f32>(&device, 0.5, w as f32 + 0.5, 1.0);
        let y_embed: Tensor<1, f32> = fusor::arange_step::<f32>(&device, 0.5, h as f32 + 0.5, 1.0);

        // Normalize to [0, 1]
        let x_embed = x_embed.div_scalar(w as f32);
        let y_embed = y_embed.div_scalar(h as f32);

        // x_embed: (1, w) -> broadcast to (h, w)
        let x_embed = x_embed.reshape([1, w]);
        let x_embed = x_embed.broadcast_as([h, w]);
        // y_embed: (h, 1) -> broadcast to (h, w)
        let y_embed = y_embed.reshape([h, 1]);
        let y_embed = y_embed.broadcast_as([h, w]);

        // Stack: (h, w, 2)
        let x_unsq = x_embed.reshape([h, w, 1]);
        let y_unsq = y_embed.reshape([h, w, 1]);
        let coords: Tensor<3, f32> = Tensor::cat([x_unsq, y_unsq], 2);

        // pe_encoding -> (h, w, embed_dim), then permute to (embed_dim, h, w)
        let encoded = self.pe_encoding(&coords);
        encoded.transpose(1, 2).transpose(0, 1).to_concrete()
    }

    fn forward_with_coords(
        &self,
        coords_input: &Tensor<3, f32, impl TensorBacking<3, Elem = f32>>,
        image_size: (usize, usize),
    ) -> Tensor<3, f32> {
        // Normalize coordinates by image size
        let shape = coords_input.shape();
        let last = shape[2];
        // coords0 = coords[..., 0:1] / width
        let coords0 = coords_input.narrow(2, 0, 1).div_scalar(image_size.1 as f32);
        // coords1 = coords[..., 1:2] / height
        let coords1 = coords_input.narrow(2, 1, 1).div_scalar(image_size.0 as f32);

        let mut parts = vec![coords0.to_concrete(), coords1.to_concrete()];
        if last > 2 {
            let rest: Tensor<3, f32> = coords_input.narrow(2, 2, last - 2).to_concrete();
            parts.push(rest);
        }
        let coords = Tensor::cat(parts, 2);
        self.pe_encoding(&coords)
    }
}

/// Encodes user prompts (points, boxes, masks) into the sparse and dense
/// embeddings that `MaskDecoder` consumes.
///
/// `forward` returns:
/// - sparse embeddings: `(batch, num_prompts, embed_dim)`
/// - dense embeddings: `(batch, embed_dim, image_embedding_size, image_embedding_size)`
pub struct PromptEncoder {
    pub(crate) pe_layer: PositionEmbeddingRandom,
    point_embeddings: Vec<Embedding<f32>>,
    not_a_point_embed: Embedding<f32>,
    mask_downscaling_conv1: ConvNd<2, 4, f32>,
    mask_downscaling_ln1: LayerNormNd<f32>,
    mask_downscaling_conv2: ConvNd<2, 4, f32>,
    mask_downscaling_ln2: LayerNormNd<f32>,
    mask_downscaling_conv3: ConvNd<2, 4, f32>,
    no_mask_embed: Embedding<f32>,
    image_embedding_size: (usize, usize),
    input_image_size: (usize, usize),
    embed_dim: usize,
}

impl PromptEncoder {
    pub fn load(
        device: &Device,
        vb: &mut VarBuilder,
        embed_dim: usize,
        image_embedding_size: (usize, usize),
        input_image_size: (usize, usize),
    ) -> Result<Self> {
        let pe_layer = PositionEmbeddingRandom::load(device, &mut vb.pp("pe_layer"))?;
        let not_a_point_embed = Embedding::load(device, &mut vb.pp("not_a_point_embed"))?;
        let no_mask_embed = Embedding::load(device, &mut vb.pp("no_mask_embed"))?;

        let cfg_s2 = ConvNdConfig {
            padding: [0, 0],
            stride: [2, 2],
            groups: 1,
        };
        let mask_downscaling_conv1 =
            ConvNd::<2, 4, f32>::load(device, &mut vb.pp("mask_downscaling.0"), cfg_s2)?;
        let mask_downscaling_ln1 =
            LayerNormNd::<f32>::load_over_axis(device, &mut vb.pp("mask_downscaling.1"), 1, 1e-6)?;
        let mask_downscaling_conv2 =
            ConvNd::<2, 4, f32>::load(device, &mut vb.pp("mask_downscaling.3"), cfg_s2)?;
        let mask_downscaling_ln2 =
            LayerNormNd::<f32>::load_over_axis(device, &mut vb.pp("mask_downscaling.4"), 1, 1e-6)?;
        let mask_downscaling_conv3 = ConvNd::<2, 4, f32>::load(
            device,
            &mut vb.pp("mask_downscaling.6"),
            ConvNdConfig::default(),
        )?;

        // SAM's prompt encoder learns four point-type embeddings:
        //   0 = background point, 1 = foreground point,
        //   2 = box top-left,     3 = box bottom-right.
        const NUM_POINT_TYPE_EMBEDDINGS: usize = 4;
        let mut point_embeddings = Vec::with_capacity(NUM_POINT_TYPE_EMBEDDINGS);
        for i in 0..NUM_POINT_TYPE_EMBEDDINGS {
            let emb = Embedding::load(device, &mut vb.pp(format!("point_embeddings.{i}")))?;
            point_embeddings.push(emb);
        }

        Ok(Self {
            pe_layer,
            point_embeddings,
            not_a_point_embed,
            mask_downscaling_conv1,
            mask_downscaling_ln1,
            mask_downscaling_conv2,
            mask_downscaling_ln2,
            mask_downscaling_conv3,
            no_mask_embed,
            image_embedding_size,
            input_image_size,
            embed_dim,
        })
    }

    pub fn get_dense_pe(&self) -> Tensor<4, f32> {
        let pe = self
            .pe_layer
            .forward(self.image_embedding_size.0, self.image_embedding_size.1);
        // (embed_dim, h, w) -> (1, embed_dim, h, w)
        let shape = pe.shape();
        pe.reshape([1, shape[0], shape[1], shape[2]]).to_concrete()
    }

    fn embed_masks(&self, masks: &Tensor<4, f32>) -> Tensor<4, f32> {
        let x = self.mask_downscaling_conv1.forward(masks);
        let x = self.mask_downscaling_ln1.forward(&x);
        let x = x.gelu();
        let x = self.mask_downscaling_conv2.forward(&x.to_concrete());
        let x = self.mask_downscaling_ln2.forward(&x);
        let x = x.gelu();
        self.mask_downscaling_conv3.forward(&x.to_concrete())
    }

    fn embed_points(
        &self,
        points: &Tensor<3, f32>,
        labels: &Tensor<2, f32>,
        pad: bool,
    ) -> Tensor<3, f32> {
        let points = (points + 0.5f32).to_concrete();
        let device = points.device();
        let points_shape = points.shape();
        let batch = points_shape[0];

        let (points, labels) = if pad {
            let padding_point: Tensor<3, f32> = Tensor::zeros(&device, [batch, 1, 2]);
            let padding_label: Tensor<2, f32> =
                (Tensor::zeros(&device, [batch, 1]) + (-1.0f32)).to_concrete();
            let points = Tensor::cat([points, padding_point], 1);
            let labels: Tensor<2, f32> =
                Tensor::cat([labels.to_concrete(), padding_label.to_concrete()], 1);
            (points, labels)
        } else {
            (points, labels.to_concrete())
        };

        let point_embedding = self
            .pe_layer
            .forward_with_coords(&points, self.input_image_size);

        let pe_shape = point_embedding.shape();
        // labels: (batch, n_points) -> (batch, n_points, 1) broadcast to (batch, n_points, embed_dim)
        let labels_broadcast = labels.reshape([pe_shape[0], pe_shape[1], 1]);
        let labels_broadcast = labels_broadcast.broadcast_as(pe_shape);

        let zeros: Tensor<3, f32> = Tensor::zeros(&device, pe_shape);

        // Where labels < 0, use not_a_point embedding; else use point_embedding
        let not_a_point = self.not_a_point_embed.embeddings().broadcast_as(pe_shape);
        let point_embedding = labels_broadcast
            .lt_scalar(0.0f32)
            .where_cond(&not_a_point, &point_embedding);

        // Add point_embeddings[0] where label == 0
        let emb0 = self.point_embeddings[0].embeddings().broadcast_as(pe_shape);
        let labels0 = labels_broadcast.eq_scalar(0.0f32).where_cond(&emb0, &zeros);
        let point_embedding = point_embedding + labels0;

        // Add point_embeddings[1] where label == 1
        let emb1 = self.point_embeddings[1].embeddings().broadcast_as(pe_shape);
        let labels1 = labels_broadcast.eq_scalar(1.0f32).where_cond(&emb1, &zeros);
        let point_embedding: Tensor<3, f32> = (point_embedding + labels1).to_concrete();

        point_embedding
    }

    fn embed_boxes(&self, boxes: &Tensor<3, f32>) -> Tensor<3, f32> {
        let boxes = boxes + 0.5f32;
        let shape = boxes.shape();
        let batch = shape[0];
        // (batch, N, 4) -> (batch, N*2, 2)
        let coords = boxes.reshape([batch, shape[1] * 2, 2]);
        let corner_embedding = self
            .pe_layer
            .forward_with_coords(&coords, self.input_image_size);
        let ce_shape = corner_embedding.shape();

        // ce1 = corner_embedding[:, 0] + point_embeddings[2]
        let ce1 = corner_embedding.narrow(1, 0, 1);
        let ce1 = ce1.reshape([batch, ce_shape[2]]);

        let ce1 = ce1 + self.point_embeddings[2].embeddings();

        // ce2 = corner_embedding[:, 1] + point_embeddings[3]
        let ce2 = corner_embedding.narrow(1, 1, 1);
        let ce2 = ce2.reshape([batch, ce_shape[2]]);
        let ce2 = ce2 + self.point_embeddings[3].embeddings();

        // Stack: (batch, 2, dim)
        let ce1_3d = ce1.reshape([batch, 1, ce_shape[2]]);
        let ce2_3d = ce2.reshape([batch, 1, ce_shape[2]]);
        Tensor::cat([ce1_3d, ce2_3d], 1)
    }

    pub fn forward(
        &self,
        points: Option<(&Tensor<3, f32>, &Tensor<2, f32>)>,
        boxes: Option<&Tensor<3, f32>>,
        masks: Option<&Tensor<4, f32>>,
    ) -> (Tensor<3, f32>, Tensor<4, f32>) {
        let se_points =
            points.map(|(coords, labels)| self.embed_points(coords, labels, boxes.is_none()));
        let se_boxes = boxes.map(|b| self.embed_boxes(b));

        let device = self.no_mask_embed.embeddings().device();

        let sparse_embeddings = match (se_points, se_boxes) {
            (Some(se_points), Some(se_boxes)) => Tensor::cat([se_points, se_boxes], 1),
            (Some(se_points), None) => se_points,
            (None, Some(se_boxes)) => se_boxes,
            (None, None) => Tensor::zeros(&device, [1, 0, self.embed_dim]),
        };

        let dense_embeddings = match masks {
            None => {
                let batch = sparse_embeddings.shape()[0];
                let emb = self.no_mask_embed.embeddings(); // (1, embed_dim)
                let emb_shape = emb.shape();
                let embed_dim = emb_shape[1];
                emb.reshape([1, embed_dim, 1, 1])
                    .broadcast_as([
                        batch,
                        embed_dim,
                        self.image_embedding_size.0,
                        self.image_embedding_size.1,
                    ])
                    .to_concrete()
            }
            Some(masks) => self.embed_masks(masks),
        };

        (sparse_embeddings, dense_embeddings)
    }
}
