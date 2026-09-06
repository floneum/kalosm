//! Image to patches, on the host.

use image::DynamicImage;

/// Resize `image` to a patch-aligned size within the pixel budget, normalize
/// it, and lay it out as `[patches, 3 * temporal * p * p]` in merge-block
/// order: for every `merge × merge` block of patches, row-major within the
/// block, each patch stored `(channel, temporal, row, col)`.
///
/// Returns the flat data, the `[t, h, w]` patch grid and the row width.
#[allow(clippy::too_many_arguments)]
pub(crate) fn patchify(
    image: &DynamicImage,
    patch_size: usize,
    merge_size: usize,
    temporal_patch_size: usize,
    min_pixels: Option<u32>,
    max_pixels: Option<u32>,
    image_mean: &[f32; 3],
    image_std: &[f32; 3],
) -> (Vec<f32>, [u32; 3], usize) {
    let merge_patch = (patch_size * merge_size) as u32;
    let resized = normalize_image_shape(
        merge_patch,
        min_pixels.unwrap_or(4 * 28 * 28),
        max_pixels.unwrap_or(512 * 28 * 28),
        image,
    );
    let (width, height) = (resized.width() as usize, resized.height() as usize);
    debug_assert!(height.is_multiple_of(merge_patch as usize));
    debug_assert!(width.is_multiple_of(merge_patch as usize));
    let rgb = resized.to_rgb8();
    let pixel = |c: usize, y: usize, x: usize| -> f32 {
        let v = rgb.as_raw()[(y * width + x) * 3 + c] as f32 / 255.0;
        (v - image_mean[c]) / image_std[c]
    };

    let grid_h = height / patch_size;
    let grid_w = width / patch_size;
    let row_width = 3 * temporal_patch_size * patch_size * patch_size;
    let mut data = Vec::with_capacity(grid_h * grid_w * row_width);
    for hb in 0..grid_h / merge_size {
        for wb in 0..grid_w / merge_size {
            for mh in 0..merge_size {
                for mw in 0..merge_size {
                    let py = (hb * merge_size + mh) * patch_size;
                    let px = (wb * merge_size + mw) * patch_size;
                    for c in 0..3 {
                        // A still image is the same frame at every temporal slot.
                        for _ in 0..temporal_patch_size {
                            for dy in 0..patch_size {
                                for dx in 0..patch_size {
                                    data.push(pixel(c, py + dy, px + dx));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    (data, [1, grid_h as u32, grid_w as u32], row_width)
}

/// The size the image is resized to: a multiple of `patch` on both axes,
/// within `[min_pixels, max_pixels]`, aspect ratio kept.
fn normalize_image_shape(
    patch: u32,
    min_pixels: u32,
    max_pixels: u32,
    image: &DynamicImage,
) -> DynamicImage {
    let round_up = |v: f64| (v / patch as f64).ceil() as u32 * patch;
    let round_down = |v: f64| (v / patch as f64).floor() as u32 * patch;
    let mut width = round_up(image.width() as f64);
    let mut height = round_up(image.height() as f64);
    if width * height > max_pixels {
        let by = ((width * height) as f64 / max_pixels as f64).sqrt();
        width = round_down(width as f64 / by);
        height = round_down(height as f64 / by);
    } else if width * height < min_pixels {
        let by = (min_pixels as f64 / (width * height) as f64).sqrt();
        width = round_up(width as f64 * by);
        height = round_up(height as f64 * by);
    }
    if width == 0 || height == 0 {
        width = patch;
        height = patch;
    }
    image.resize_exact(width, height, image::imageops::FilterType::CatmullRom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_lands_on_the_patch_grid() {
        let image = DynamicImage::new_rgb8(1000, 700);
        let resized = normalize_image_shape(28, 256 * 28 * 28, 512 * 28 * 28, &image);
        assert_eq!(resized.width() % 28, 0);
        assert_eq!(resized.height() % 28, 0);
        let pixels = resized.width() * resized.height();
        assert!(pixels <= 512 * 28 * 28);
        assert!(pixels >= 256 * 28 * 28);
    }

    #[test]
    fn patch_layout_is_channel_temporal_row_col() {
        // A 2x2-patch image of 1-pixel patches: every patch is one pixel.
        let mut image = image::RgbImage::new(2, 2);
        for (i, p) in image.pixels_mut().enumerate() {
            *p = image::Rgb([i as u8 * 10, 0, 0]);
        }
        let image = DynamicImage::ImageRgb8(image);
        let (data, grid, width) = patchify(&image, 1, 2, 2, Some(1), Some(4), &[0.0; 3], &[1.0; 3]);
        assert_eq!(grid, [1, 2, 2]);
        assert_eq!(width, 3 * 2);
        assert_eq!(data.len(), 4 * 6);
        // Patch order within the merge block is row-major: pixel 0, 1, 2, 3.
        for (patch, expected) in [0.0f32, 10.0, 20.0, 30.0].into_iter().enumerate() {
            let row = &data[patch * 6..patch * 6 + 6];
            // Red channel at both temporal slots, then zeros for green and blue.
            assert_eq!(row[0] * 255.0, expected);
            assert_eq!(row[1] * 255.0, expected);
            assert_eq!(&row[2..], &[0.0; 4]);
        }
    }
}
