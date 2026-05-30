use crate::raw::rope::create_inverse_frequency;
use fusor::{arange, CastTensor, CastTo, Device, FloatDataType, SimdElement, Tensor};

pub struct VisionRotaryEmbedding<F: FloatDataType + SimdElement> {
    inv_freq: Tensor<2, F>,
}

impl<F> VisionRotaryEmbedding<F>
where
    F: FloatDataType + SimdElement + CastTo<f32> + CastTensor<f32>,
    f32: CastTo<F> + CastTensor<F>,
{
    pub(crate) fn new(dim: usize, rope_theta: f32, device: &Device) -> fusor::Result<Self> {
        Ok(Self {
            inv_freq: create_inverse_frequency(None, None, dim, rope_theta, device),
        })
    }

    pub(crate) fn make_embeds(&self, sequence_length: u32) -> fusor::Result<Tensor<2, F>> {
        // Work in f32 for mat_mul
        let seq_f32: Tensor<2, f32> = arange(&self.inv_freq.device(), 0f32, sequence_length as f32)
            .reshape([sequence_length as usize, 1])
            .to_concrete();
        let inv_freq_f32: Tensor<2, f32> = self.inv_freq.cast();
        let result_f32 = seq_f32.mat_mul(&inv_freq_f32);
        Ok(result_f32.cast())
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ImageSize {
    width: u32,
    height: u32,
}

impl ImageSize {
    fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    fn patch_size(&self, merge_size: usize) -> Self {
        let width = self.width.div_ceil(merge_size as u32);
        let height = self.height.div_ceil(merge_size as u32);
        Self { width, height }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct VideoSize {
    image_size: ImageSize,
    frames: u32,
}

impl VideoSize {
    fn new(width: u32, height: u32, frames: u32) -> Self {
        Self {
            image_size: ImageSize { width, height },
            frames,
        }
    }

    fn patch_size(&self, merge_size: usize) -> Self {
        let image_size = self.image_size.patch_size(merge_size);
        Self {
            image_size,
            frames: self.frames,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RopeIndex {
    pub(crate) x: u32,
    pub(crate) y: u32,
    pub(crate) time: u32,
}

// Calculate the rope index for each token in the input_ids.
// Images are encoded as vision_start_token_id then a sequence of image_token_id.
// Videos are encoded as vision_start_token_id then a sequence of video_token_id.
//
// Rope indexes are calculated with an monotonic increasing time index.
// The x and y indexes are calculated based on the patch index of the image or video.
//
// Eg. The series [vision_start_token_id, image_token_id, image_token_id, image_token_id, image_token_id]
//     will be encoded as:
//     [RopeIndex { x: 0, y: 0, time: 0 }, RopeIndex { x: 1, y: 1, time: 1 }, RopeIndex { x: 1, y: 2, time: 1 },
//      RopeIndex { x: 2, y: 1, time: 1 }, RopeIndex { x: 2, y: 2, time: 1 }]
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_rope_index(
    spatial_merge_size: usize,
    image_token_id: u32,
    video_token_id: u32,
    vision_start_token_id: u32,
    input_ids: &[u32],
    image_sizes: &[[u32; 3]],
    video_sizes: &[[u32; 3]],
    start_time: u32,
) -> (Vec<RopeIndex>, u32) {
    let mut images = image_sizes.iter().map(|&[_, w, h]| ImageSize::new(w, h));
    let mut videos = video_sizes.iter().map(|&[t, w, h]| VideoSize::new(w, h, t));
    let mut indexes = Vec::new();
    let mut max_time_index = start_time;
    let mut index = 0;

    while let Some(token) = input_ids.get(index).copied() {
        push_text_token(&mut indexes, &mut max_time_index, &mut index);
        if token == vision_start_token_id {
            let Some(token) = input_ids.get(index).copied() else {
                break;
            };
            if token == image_token_id {
                let image = images.next().unwrap();
                let image_size = image.patch_size(spatial_merge_size);
                let image_width = image_size.width;
                let image_height = image_size.height;
                for j in 0..image_width {
                    for k in 0..image_height {
                        indexes.push(RopeIndex {
                            x: max_time_index + j,
                            y: max_time_index + k,
                            time: max_time_index,
                        });
                    }
                }
                max_time_index += 2;
                index += image_width as usize * image_height as usize;
            } else if token == video_token_id {
                let video = videos.next().unwrap();
                let video_size = video.patch_size(spatial_merge_size);
                let image_width = video_size.image_size.width;
                let image_height = video_size.image_size.height;
                for frame in 0..video_size.frames {
                    for j in 0..image_width {
                        for k in 0..image_height {
                            indexes.push(RopeIndex {
                                x: max_time_index + j,
                                y: max_time_index + k,
                                time: max_time_index + frame,
                            });
                        }
                    }
                }
                index += image_width as usize * image_height as usize * video.frames as usize;
                max_time_index = max_time_index.max(max_time_index + video.frames + 1);
            }
        }
    }
    debug_assert_eq!(
        indexes.len(),
        input_ids.len(),
        "The length of indexes should match the length of input_ids"
    );

    (indexes, max_time_index)
}

fn push_text_token(indexes: &mut Vec<RopeIndex>, max_time_index: &mut u32, index: &mut usize) {
    // Push the token sequence with increasing time, x, and y positions
    indexes.push(RopeIndex {
        x: *max_time_index,
        y: *max_time_index,
        time: *max_time_index,
    });
    *max_time_index += 1;
    *index += 1;
}

#[test]
fn test_get_rope_index_text() {
    let spatial_merge_size = 16;
    let image_token_id = 100;
    let video_token_id = 200;
    let vision_start_token_id = 10;
    let input_ids = vec![0, 1, 2, 3, 4, 5];
    let image_sizes = vec![[1, 32, 32], [1, 64, 64]];
    let video_sizes = vec![[5, 32, 32], [20, 64, 64]];

    let (indexes, max_time_index) = get_rope_index(
        spatial_merge_size,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        &input_ids,
        &image_sizes,
        &video_sizes,
        0,
    );
    assert_eq!(max_time_index, 6);

    assert_eq!(
        indexes,
        vec![
            RopeIndex {
                x: 0,
                y: 0,
                time: 0
            },
            RopeIndex {
                x: 1,
                y: 1,
                time: 1
            },
            RopeIndex {
                x: 2,
                y: 2,
                time: 2
            },
            RopeIndex {
                x: 3,
                y: 3,
                time: 3
            },
            RopeIndex {
                x: 4,
                y: 4,
                time: 4
            },
            RopeIndex {
                x: 5,
                y: 5,
                time: 5
            },
        ]
    );
}

#[test]
fn test_get_rope_index() {
    let spatial_merge_size = 16;
    let image_token_id = 1;
    let video_token_id = 2;
    let vision_start_token_id = 0;
    let input_ids = vec![
        0, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5,
    ];
    let image_sizes = vec![[1, 32, 32]];
    let video_sizes = vec![[5, 32, 32]];

    let (indexes, max_time_index) = get_rope_index(
        spatial_merge_size,
        image_token_id,
        video_token_id,
        vision_start_token_id,
        &input_ids,
        &image_sizes,
        &video_sizes,
        0,
    );
    assert_eq!(max_time_index, 13);

    assert_eq!(
        indexes,
        [
            // Start image token
            RopeIndex {
                x: 0,
                y: 0,
                time: 0
            },
            // 4 image tokens for a 2x2 patch
            RopeIndex {
                x: 1,
                y: 1,
                time: 1
            },
            RopeIndex {
                x: 1,
                y: 2,
                time: 1
            },
            RopeIndex {
                x: 2,
                y: 1,
                time: 1
            },
            RopeIndex {
                x: 2,
                y: 2,
                time: 1
            },
            // Start video token
            RopeIndex {
                x: 3,
                y: 3,
                time: 3
            },
            // 20 video tokens for a 2x2x5 patch
            RopeIndex {
                x: 4,
                y: 4,
                time: 4
            },
            RopeIndex {
                x: 4,
                y: 5,
                time: 4
            },
            RopeIndex {
                x: 5,
                y: 4,
                time: 4
            },
            RopeIndex {
                x: 5,
                y: 5,
                time: 4
            },
            RopeIndex {
                x: 4,
                y: 4,
                time: 5
            },
            RopeIndex {
                x: 4,
                y: 5,
                time: 5
            },
            RopeIndex {
                x: 5,
                y: 4,
                time: 5
            },
            RopeIndex {
                x: 5,
                y: 5,
                time: 5
            },
            RopeIndex {
                x: 4,
                y: 4,
                time: 6
            },
            RopeIndex {
                x: 4,
                y: 5,
                time: 6
            },
            RopeIndex {
                x: 5,
                y: 4,
                time: 6
            },
            RopeIndex {
                x: 5,
                y: 5,
                time: 6
            },
            RopeIndex {
                x: 4,
                y: 4,
                time: 7
            },
            RopeIndex {
                x: 4,
                y: 5,
                time: 7
            },
            RopeIndex {
                x: 5,
                y: 4,
                time: 7
            },
            RopeIndex {
                x: 5,
                y: 5,
                time: 7
            },
            RopeIndex {
                x: 4,
                y: 4,
                time: 8
            },
            RopeIndex {
                x: 4,
                y: 5,
                time: 8
            },
            RopeIndex {
                x: 5,
                y: 4,
                time: 8
            },
            RopeIndex {
                x: 5,
                y: 5,
                time: 8
            },
            // Remaining text tokens
            RopeIndex {
                x: 10,
                y: 10,
                time: 10
            },
            RopeIndex {
                x: 11,
                y: 11,
                time: 11
            },
            RopeIndex {
                x: 12,
                y: 12,
                time: 12
            }
        ]
    );
}
