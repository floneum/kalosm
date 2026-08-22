//! Prompt encoder: encodes points, boxes, and masks into embeddings.

use fusor::layers::{ConvNd, Embedding, LayerNorm};
use fusor::{Device, Tensor};
use fusor_gguf::VarBuilder;

use super::{channel_layer_norm, load_dense, Result};

pub(crate) struct PositionEmbeddingRandom {
    pub(crate) positional_encoding_gaussian_matrix: Tensor<2>,
}

impl PositionEmbeddingRandom {
    fn load(device: &Device, vb: &VarBuilder) -> Result<Self> {
        let m = load_dense(vb, device, "positional_encoding_gaussian_matrix")?;
        Ok(Self {
            positional_encoding_gaussian_matrix: m,
        })
    }

    fn pe_encoding(&self, coords: &Tensor<3>) -> Tensor<3> {
        // coords * 2 - 1
        let coords = coords.mul_scalar(2.0f32).add_scalar(-1.0f32);
        let [b, _, n] = coords.shape();
        let d = self.positional_encoding_gaussian_matrix.shape()[1];
        let gm = self
            .positional_encoding_gaussian_matrix
            .reshape([1, n, d])
            .broadcast_as([b, n, d]);
        // coords @ gm * 2 * pi
        let coords = coords.matmul(&gm).mul_scalar(2.0 * std::f32::consts::PI);
        // cat([sin, cos], last_dim)
        Tensor::cat([coords.sin(), coords.cos()], 2)
    }

    pub(crate) fn forward(&self, h: usize, w: usize) -> Tensor<3> {
        let device = self.positional_encoding_gaussian_matrix.device();
        // Create grid coordinates, normalized to [0, 1]
        let x_embed =
            Tensor::<1>::arange_step(&device, 0.5, w as f64 + 0.5, 1.0).div_scalar(w as f32);
        let y_embed =
            Tensor::<1>::arange_step(&device, 0.5, h as f64 + 0.5, 1.0).div_scalar(h as f32);

        // x_embed: (1, w) -> broadcast to (h, w); y_embed: (h, 1) -> (h, w)
        let x_embed = x_embed.reshape([1, w]).broadcast_as([h, w]);
        let y_embed = y_embed.reshape([h, 1]).broadcast_as([h, w]);

        // Stack: (h, w, 2)
        let coords = Tensor::cat([x_embed.reshape([h, w, 1]), y_embed.reshape([h, w, 1])], 2);

        // pe_encoding -> (h, w, embed_dim), then permute to (embed_dim, h, w)
        self.pe_encoding(&coords).permute([2, 0, 1])
    }

    fn forward_with_coords(
        &self,
        coords_input: &Tensor<3>,
        image_size: (usize, usize),
    ) -> Tensor<3> {
        // Normalize coordinates by image size
        let last = coords_input.shape()[2];
        // coords0 = coords[..., 0:1] / width
        let coords0 = coords_input.narrow(2, 0, 1).div_scalar(image_size.1 as f32);
        // coords1 = coords[..., 1:2] / height
        let coords1 = coords_input.narrow(2, 1, 1).div_scalar(image_size.0 as f32);

        let mut parts = vec![coords0, coords1];
        if last > 2 {
            parts.push(coords_input.narrow(2, 2, last - 2));
        }
        self.pe_encoding(&Tensor::cat(parts, 2))
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
    point_embeddings: Vec<Embedding>,
    not_a_point_embed: Embedding,
    mask_downscaling_conv1: ConvNd,
    mask_downscaling_ln1: LayerNorm,
    mask_downscaling_conv2: ConvNd,
    mask_downscaling_ln2: LayerNorm,
    mask_downscaling_conv3: ConvNd,
    no_mask_embed: Embedding,
    image_embedding_size: (usize, usize),
    input_image_size: (usize, usize),
    embed_dim: usize,
}

impl PromptEncoder {
    pub fn load(
        device: &Device,
        vb: &VarBuilder,
        embed_dim: usize,
        image_embedding_size: (usize, usize),
        input_image_size: (usize, usize),
    ) -> Result<Self> {
        let graph = device.graph().handle();
        let pe_layer = PositionEmbeddingRandom::load(device, &vb.pp("pe_layer"))?;
        let not_a_point_embed = Embedding::load(&vb.pp("not_a_point_embed"), graph)?;
        let no_mask_embed = Embedding::load(&vb.pp("no_mask_embed"), graph)?;

        let conv_s2 = |vb: &VarBuilder| -> Result<ConvNd> {
            let mut conv = ConvNd::load(vb, graph, true)?;
            conv.stride = [2u32, 2].into_iter().collect();
            conv.padding = [0u32, 0].into_iter().collect();
            Ok(conv)
        };
        let mask_downscaling_conv1 = conv_s2(&vb.pp("mask_downscaling.0"))?;
        let mask_downscaling_ln1 = LayerNorm::load(&vb.pp("mask_downscaling.1"), graph, 1e-6)?;
        let mask_downscaling_conv2 = conv_s2(&vb.pp("mask_downscaling.3"))?;
        let mask_downscaling_ln2 = LayerNorm::load(&vb.pp("mask_downscaling.4"), graph, 1e-6)?;
        let mask_downscaling_conv3 = ConvNd::load(&vb.pp("mask_downscaling.6"), graph, true)?;

        // SAM's prompt encoder learns four point-type embeddings:
        //   0 = background point, 1 = foreground point,
        //   2 = box top-left,     3 = box bottom-right.
        const NUM_POINT_TYPE_EMBEDDINGS: usize = 4;
        let mut point_embeddings = Vec::with_capacity(NUM_POINT_TYPE_EMBEDDINGS);
        for i in 0..NUM_POINT_TYPE_EMBEDDINGS {
            let emb = Embedding::load(&vb.pp(format!("point_embeddings.{i}")), graph)?;
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

    pub fn get_dense_pe(&self) -> Tensor<4> {
        let pe = self
            .pe_layer
            .forward(self.image_embedding_size.0, self.image_embedding_size.1);
        // (embed_dim, h, w) -> (1, embed_dim, h, w)
        pe.unsqueeze(0)
    }

    fn embed_masks(&self, masks: &Tensor<4>) -> Tensor<4> {
        let x = self.mask_downscaling_conv1.forward(masks);
        let x = channel_layer_norm(&self.mask_downscaling_ln1, &x).gelu();
        let x = self.mask_downscaling_conv2.forward(&x);
        let x = channel_layer_norm(&self.mask_downscaling_ln2, &x).gelu();
        self.mask_downscaling_conv3.forward(&x)
    }

    fn embed_points(&self, points: &Tensor<3>, labels: &Tensor<2>, pad: bool) -> Tensor<3> {
        let points = points.add_scalar(0.5f32);
        let device = points.device();
        let batch = points.shape()[0];

        let (points, labels) = if pad {
            let padding_point = Tensor::<3>::zeros(&device, [batch, 1, 2]);
            let padding_label = Tensor::<2>::full(&device, [batch, 1], -1.0);
            (
                Tensor::cat([points, padding_point], 1),
                Tensor::cat([labels.clone(), padding_label], 1),
            )
        } else {
            (points, labels.clone())
        };

        let point_embedding = self
            .pe_layer
            .forward_with_coords(&points, self.input_image_size);

        let pe_shape = point_embedding.shape();
        // labels: (batch, n_points) -> (batch, n_points, 1) broadcast to (batch, n_points, embed_dim)
        let labels_broadcast = labels.unsqueeze::<3>(2).broadcast_as(pe_shape);

        let zeros = Tensor::<3>::zeros(&device, pe_shape);

        // Where labels < 0, use not_a_point embedding; else use point_embedding
        let not_a_point = self.not_a_point_embed.table.broadcast_as(pe_shape);
        let point_embedding = labels_broadcast
            .lt_scalar(0.0f32)
            .where_cond(&not_a_point, &point_embedding);

        // Add point_embeddings[0] where label == 0
        let emb0 = self.point_embeddings[0].table.broadcast_as(pe_shape);
        let labels0 = labels_broadcast.eq_scalar(0.0f32).where_cond(&emb0, &zeros);
        let point_embedding = point_embedding.add(&labels0);

        // Add point_embeddings[1] where label == 1
        let emb1 = self.point_embeddings[1].table.broadcast_as(pe_shape);
        let labels1 = labels_broadcast.eq_scalar(1.0f32).where_cond(&emb1, &zeros);
        point_embedding.add(&labels1)
    }

    fn embed_boxes(&self, boxes: &Tensor<3>) -> Tensor<3> {
        let boxes = boxes.add_scalar(0.5f32);
        let [batch, n, _] = boxes.shape();
        // (batch, N, 4) -> (batch, N*2, 2)
        let coords = boxes.reshape([batch, n * 2, 2]);
        let corner_embedding = self
            .pe_layer
            .forward_with_coords(&coords, self.input_image_size);
        let ce_dim = corner_embedding.shape()[2];

        // ce1 = corner_embedding[:, 0] + point_embeddings[2]
        let ce1: Tensor<2> = corner_embedding
            .narrow(1, 0, 1)
            .reshape([batch, ce_dim])
            .add_(&self.point_embeddings[2].table);

        // ce2 = corner_embedding[:, 1] + point_embeddings[3]
        let ce2: Tensor<2> = corner_embedding
            .narrow(1, 1, 1)
            .reshape([batch, ce_dim])
            .add_(&self.point_embeddings[3].table);

        // Stack: (batch, 2, dim)
        Tensor::cat(
            [
                ce1.reshape([batch, 1, ce_dim]),
                ce2.reshape([batch, 1, ce_dim]),
            ],
            1,
        )
    }

    pub fn forward(
        &self,
        points: Option<(&Tensor<3>, &Tensor<2>)>,
        boxes: Option<&Tensor<3>>,
        masks: Option<&Tensor<4>>,
    ) -> (Tensor<3>, Tensor<4>) {
        let se_points =
            points.map(|(coords, labels)| self.embed_points(coords, labels, boxes.is_none()));
        let se_boxes = boxes.map(|b| self.embed_boxes(b));

        let device = self.no_mask_embed.table.device();

        let sparse_embeddings = match (se_points, se_boxes) {
            (Some(se_points), Some(se_boxes)) => Tensor::cat([se_points, se_boxes], 1),
            (Some(se_points), None) => se_points,
            (None, Some(se_boxes)) => se_boxes,
            (None, None) => Tensor::zeros(&device, [1, 0, self.embed_dim]),
        };

        let dense_embeddings = match masks {
            None => {
                let batch = sparse_embeddings.shape()[0];
                let emb = &self.no_mask_embed.table; // (1, embed_dim)
                let embed_dim = emb.shape()[1];
                emb.reshape([1, embed_dim, 1, 1]).broadcast_as([
                    batch,
                    embed_dim,
                    self.image_embedding_size.0,
                    self.image_embedding_size.1,
                ])
            }
            Some(masks) => self.embed_masks(masks),
        };

        (sparse_embeddings, dense_embeddings)
    }
}
