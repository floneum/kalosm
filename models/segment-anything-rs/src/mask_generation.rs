use crate::raw::sam::IMAGE_SIZE;

const PRED_IOU_THRESH: f32 = 0.78;
const STABILITY_SCORE_OFFSET: f32 = 1.0;
const STABILITY_SCORE_THRESHOLD: f32 = 0.88;
const MODEL_MASK_THRESHOLD: f32 = 0.0;
const CROP_NMS_THRESH: f32 = 0.7;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BoundingBox {
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
}

impl BoundingBox {
    fn area(self) -> usize {
        (self.x1 - self.x0) * (self.y1 - self.y0)
    }
}

#[derive(Clone, Debug)]
struct BinaryMask {
    bits: Vec<u64>,
    len: usize,
}

impl BinaryMask {
    fn from_thresholded(mask: &[f32], width: usize, threshold: f32) -> Option<(Self, BoundingBox)> {
        let mut bits = vec![0u64; mask.len().div_ceil(64)];
        let mut min_x = usize::MAX;
        let mut min_y = usize::MAX;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        let mut has_foreground = false;

        for (idx, &value) in mask.iter().enumerate() {
            if value < threshold {
                continue;
            }

            has_foreground = true;
            bits[idx / 64] |= 1u64 << (idx % 64);

            let x = idx % width;
            let y = idx / width;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }

        has_foreground.then_some((
            Self {
                bits,
                len: mask.len(),
            },
            BoundingBox {
                x0: min_x,
                y0: min_y,
                x1: max_x + 1,
                y1: max_y + 1,
            },
        ))
    }

    fn bit(&self, idx: usize) -> bool {
        let word = self.bits[idx / 64];
        ((word >> (idx % 64)) & 1) != 0
    }

    fn to_rgb_pixels(&self) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(self.len * 3);
        for idx in 0..self.len {
            let value = if self.bit(idx) { 255u8 } else { 0u8 };
            pixels.extend_from_slice(&[value, value, value]);
        }
        pixels
    }
}

#[derive(Clone, Debug)]
pub(crate) struct MaskCandidate {
    mask: BinaryMask,
    bbox: BoundingBox,
    score: f32,
    height: usize,
    width: usize,
}

impl MaskCandidate {
    pub(crate) fn into_image(self) -> Option<image::ImageBuffer<image::Rgb<u8>, Vec<u8>>> {
        image::ImageBuffer::from_raw(
            self.width as u32,
            self.height as u32,
            self.mask.to_rgb_pixels(),
        )
    }
}

pub(crate) struct LowResMaskBatch<'a> {
    pub(crate) masks: &'a [f32],
    pub(crate) iou: &'a [f32],
    pub(crate) batch: usize,
    pub(crate) masks_per_point: usize,
    pub(crate) mask_w: usize,
    pub(crate) crop_h: usize,
    pub(crate) crop_w: usize,
}

pub(crate) fn low_res_crop_extent(original_len: usize, low_res_len: usize) -> usize {
    (original_len * low_res_len)
        .div_ceil(IMAGE_SIZE)
        .min(low_res_len)
}

pub(crate) fn suppress_overlaps(mut candidates: Vec<MaskCandidate>) -> Vec<MaskCandidate> {
    candidates.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<MaskCandidate> = Vec::with_capacity(candidates.len());
    'candidate: for candidate in candidates {
        for kept_candidate in &kept {
            if bbox_iou(candidate.bbox, kept_candidate.bbox) > CROP_NMS_THRESH {
                continue 'candidate;
            }
        }
        kept.push(candidate);
    }

    kept
}

pub(crate) fn collect_mask_candidates(
    batch: LowResMaskBatch<'_>,
    candidates: &mut Vec<MaskCandidate>,
) {
    let mask_pixels = batch.masks.len() / (batch.batch * batch.masks_per_point);
    let hi_thresh = MODEL_MASK_THRESHOLD + STABILITY_SCORE_OFFSET;
    let lo_thresh = MODEL_MASK_THRESHOLD - STABILITY_SCORE_OFFSET;

    for point_idx in 0..batch.batch {
        for mask_idx in 0..batch.masks_per_point {
            let flat_idx = point_idx * batch.masks_per_point + mask_idx;
            let iou = batch.iou[flat_idx];
            if iou < PRED_IOU_THRESH {
                continue;
            }

            let mask_start = flat_idx * mask_pixels;
            let mut cropped_mask = Vec::with_capacity(batch.crop_h * batch.crop_w);
            for y in 0..batch.crop_h {
                let row_start = mask_start + y * batch.mask_w;
                cropped_mask.extend_from_slice(&batch.masks[row_start..row_start + batch.crop_w]);
            }

            if stability_score(&cropped_mask, hi_thresh, lo_thresh) < STABILITY_SCORE_THRESHOLD {
                continue;
            }

            if let Some((mask, bbox)) =
                BinaryMask::from_thresholded(&cropped_mask, batch.crop_w, MODEL_MASK_THRESHOLD)
            {
                candidates.push(MaskCandidate {
                    mask,
                    bbox,
                    score: iou,
                    height: batch.crop_h,
                    width: batch.crop_w,
                });
            }
        }
    }
}

fn stability_score(mask: &[f32], hi_thresh: f32, lo_thresh: f32) -> f32 {
    let intersections = mask.iter().filter(|&&v| v >= hi_thresh).count() as f32;
    let unions = mask.iter().filter(|&&v| v >= lo_thresh).count() as f32;
    if unions > 0.0 {
        intersections / unions
    } else {
        0.0
    }
}

fn bbox_iou(a: BoundingBox, b: BoundingBox) -> f32 {
    let x0 = a.x0.max(b.x0);
    let y0 = a.y0.max(b.y0);
    let x1 = a.x1.min(b.x1);
    let y1 = a.y1.min(b.y1);

    if x0 >= x1 || y0 >= y1 {
        return 0.0;
    }

    let intersection = (x1 - x0) * (y1 - y0);
    let union = a.area() + b.area() - intersection;
    intersection as f32 / union as f32
}
